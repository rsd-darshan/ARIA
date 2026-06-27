"""
aria_train.py
=============
ARIA: Adaptive Recurrent Intelligence Architecture
Full training script — Kaggle ready.

Trains ARIA + 3 baselines on Split-MNIST (5 tasks).
Saves all results to results/ for figure generation.

Run:
    python aria_train.py

On Kaggle:
    !python aria_train.py
"""

import os, csv, time, copy, math, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # Data
    "n_tasks":         5,
    "classes_per_task": 2,
    "batch_size":      128,
    "data_dir":        "./data",

    # Architecture
    "input_dim":       784,      # 28×28 flattened
    "hidden_dim":      256,
    "n_layers":        4,
    "n_heads_init":    4,
    "n_heads_max":     8,
    "genome_dim":      32,
    "dropout":         0.1,

    # Morphogenesis
    "split_threshold": 0.65,
    "merge_threshold": 0.90,
    "morph_interval":  200,      # steps between morphogenesis checks

    # Plasticity
    "plasticity_lambda": 0.005,

    # Budget
    "budget_beta":     0.001,

    # Genome
    "genome_gamma":    0.0001,

    # Training
    "epochs_per_task": 10,
    "lr":              3e-4,
    "weight_decay":    1e-4,

    # EWC
    "ewc_lambda":      5000,

    # Logging
    "results_dir":     "./results",
    "seed":            42,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CFG["results_dir"], exist_ok=True)
os.makedirs(CFG["data_dir"],    exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

print(f"Device: {CFG['device']}")


# ─────────────────────────────────────────────────────────────────────────────
# Data: Split-MNIST
# ─────────────────────────────────────────────────────────────────────────────

def get_split_mnist(n_tasks=5, data_dir="./data"):
    """
    Split MNIST into n_tasks binary classification tasks.
    Task k: digits 2k and 2k+1.
    Returns list of (train_loader, test_loader) per task.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),   # flatten to 784
    ])

    train_full = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_full  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    task_loaders = []
    for t in range(n_tasks):
        c0, c1 = t * 2, t * 2 + 1

        # Filter by class, remap labels to 0/1
        def make_subset(dataset, c0, c1):
            indices = [i for i, (_, y) in enumerate(dataset)
                       if y == c0 or y == c1]
            subset  = Subset(dataset, indices)
            # Wrap to remap labels
            class SubsetRelabeled(torch.utils.data.Dataset):
                def __init__(self, subset, c0):
                    self.subset = subset
                    self.c0 = c0
                def __len__(self):
                    return len(self.subset)
                def __getitem__(self, idx):
                    x, y = self.subset[idx]
                    return x, int(y != self.c0)   # 0 or 1
            return SubsetRelabeled(subset, c0)

        train_ds = make_subset(train_full, c0, c1)
        test_ds  = make_subset(test_full,  c0, c1)

        task_loaders.append((
            DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0),
            DataLoader(test_ds,  batch_size=256,               shuffle=False, num_workers=0),
        ))
        print(f"  Task {t+1} (digits {c0} vs {c1}): "
              f"{len(train_ds)} train, {len(test_ds)} test")

    return task_loaders


# ─────────────────────────────────────────────────────────────────────────────
# ARIA Model
# ─────────────────────────────────────────────────────────────────────────────

class ArchitectureGenome(nn.Module):
    """Global latent vector encoding structural hyperparameters."""
    def __init__(self, genome_dim, d_model, n_layers):
        super().__init__()
        G = genome_dim
        self.z = nn.Parameter(torch.randn(G) * 0.01)
        self.proj_skip  = nn.Linear(G, n_layers)
        self.proj_temp  = nn.Linear(G, 1)
        self.proj_cond  = nn.Linear(G, d_model)

    def decode(self):
        z = self.z
        return {
            "skip_probs":  torch.sigmoid(self.proj_skip(z)),
            "temperature": F.softplus(self.proj_temp(z)).squeeze() + 0.5,
            "cond_signal": torch.tanh(self.proj_cond(z)),
        }

    def reg_loss(self):
        return 0.5 * (self.z ** 2).mean()


class MorphogenicAttention(nn.Module):
    """
    Multi-head attention with fixed-size weight tensors (Kaggle-safe).
    Uses a head_mask to activate/deactivate heads — no dynamic ParameterList.
    Supports soft head splitting and merging.
    """
    def __init__(self, d_model, n_heads_init, n_heads_max, dropout,
                 split_threshold, merge_threshold, genome_dim):
        super().__init__()
        self.d_model     = d_model
        self.n_heads_max = n_heads_max
        self.split_τ     = split_threshold
        self.merge_τ     = merge_threshold
        self.d_h         = d_model // n_heads_max

        # Fixed-size weight tensors — always fully tracked by optimizer
        self.W_Q = nn.Parameter(torch.randn(n_heads_max, d_model, self.d_h) * 0.02)
        self.W_K = nn.Parameter(torch.randn(n_heads_max, d_model, self.d_h) * 0.02)
        self.W_V = nn.Parameter(torch.randn(n_heads_max, d_model, self.d_h) * 0.02)
        self.W_O = nn.Parameter(torch.randn(n_heads_max, self.d_h, d_model) * 0.02)

        # Learnable viability scores (one per max head slot)
        self.viability = nn.Parameter(torch.zeros(n_heads_max))

        # Active head mask — not a parameter, just a buffer
        mask = torch.zeros(n_heads_max, dtype=torch.bool)
        mask[:n_heads_init] = True
        self.register_buffer("head_mask", mask)

        self.dropout     = nn.Dropout(dropout)
        self.genome_proj = nn.Linear(genome_dim, d_model)

    @property
    def n_active(self):
        return self.head_mask.sum().item()

    def _active_indices(self):
        return self.head_mask.nonzero(as_tuple=True)[0]

    def forward(self, x, genome):
        B, T, D = x.shape
        τ = genome["temperature"]
        active = self._active_indices()

        # Compute per-head attention
        outputs = []
        for i in active:
            Q = x @ self.W_Q[i]   # (B, T, d_h)
            K = x @ self.W_K[i]
            V = x @ self.W_V[i]

            scale  = math.sqrt(self.d_h) * τ.clamp(min=0.1)
            scores = (Q @ K.transpose(-2, -1)) / scale
            # Causal mask
            mask   = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn   = F.softmax(scores, dim=-1)
            attn   = self.dropout(attn)
            out    = attn @ V          # (B, T, d_h)
            contrib = out @ self.W_O[i]  # (B, T, D)

            v_i = torch.sigmoid(self.viability[i])
            outputs.append(v_i * contrib)

        result = torch.stack(outputs, 0).sum(0)  # (B, T, D)

        # Genome conditioning
        cond = self.genome_proj(genome["cond_signal"])
        result = result + cond

        return result

    def morphogenesis(self):
        """Try to split or merge heads. Called periodically."""
        active = self._active_indices().tolist()

        # --- Splitting ---
        for i in active:
            if self.n_active >= self.n_heads_max:
                break
            # Split score: variance of W_Q row norms
            var = self.W_Q[i].norm(dim=1).var().item()
            split_score = 1 / (1 + math.exp(-10 * (var - 0.5)))
            if split_score > self.split_τ:
                # Find an inactive slot
                inactive = (~self.head_mask).nonzero(as_tuple=True)[0]
                if len(inactive) == 0:
                    break
                j = inactive[0].item()
                noise = 0.01
                with torch.no_grad():
                    for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                        W[j] = W[i] + noise * torch.randn_like(W[i])
                        W[i] = W[i] - noise * torch.randn_like(W[i])
                    self.viability[j] = self.viability[i].clone()
                self.head_mask[j] = True

        # --- Merging ---
        active = self._active_indices().tolist()
        to_deactivate = []
        for idx in range(len(active) - 1):
            i, j = active[idx], active[idx + 1]
            if j in to_deactivate:
                continue
            cos_sim = F.cosine_similarity(
                self.W_Q[i].flatten().unsqueeze(0),
                self.W_Q[j].flatten().unsqueeze(0)
            ).item()
            if cos_sim > self.merge_τ:
                with torch.no_grad():
                    for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                        W[i] = (W[i] + W[j]) / 2
                to_deactivate.append(j)

        for j in to_deactivate:
            self.head_mask[j] = False


class PlasticityGatedMLP(nn.Module):
    """Dual-pathway MLP: fast (high π) and slow (low π) pathways."""
    def __init__(self, d_model, d_ff, dropout, plasticity_lambda):
        super().__init__()
        self.W_fast_in  = nn.Linear(d_model, d_ff)
        self.W_fast_out = nn.Linear(d_ff, d_model)
        self.W_slow_in  = nn.Linear(d_model, d_ff)
        self.W_slow_out = nn.Linear(d_ff, d_model)
        self.gate_net   = nn.Sequential(
            nn.Linear(d_model, d_ff // 4),
            nn.ReLU(),
            nn.Linear(d_ff // 4, 1),
            nn.Sigmoid()
        )
        self.dropout    = nn.Dropout(dropout)
        self.lambda_    = plasticity_lambda
        self.mean_gate  = 0.5   # tracked for slow weight dampening

    def forward(self, x):
        π = self.gate_net(x)                          # (B, T, 1)
        self.mean_gate = π.mean().item()

        h_fast = F.gelu(self.W_fast_in(x))
        h_slow = F.gelu(self.W_slow_in(x))

        out = π * self.W_fast_out(h_fast) + (1 - π) * self.W_slow_out(h_slow)
        out = self.dropout(out)

        # Specialization loss: push π toward 0 or 1
        p_loss = self.lambda_ / (π * (1 - π) + 1e-4).mean()
        return out, p_loss


class CognitiveBudgetAllocator(nn.Module):
    """Predicts per-layer compute budget from input complexity."""
    def __init__(self, n_layers, budget_beta):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, n_layers), nn.Sigmoid()
        )
        self.beta = budget_beta

    def forward(self, x, prev_x=None):
        entropy_proxy = torch.sigmoid(x.std(dim=-1).mean().unsqueeze(0))
        if prev_x is not None:
            res_norm = torch.sigmoid((x - prev_x).norm(dim=-1).mean().unsqueeze(0))
        else:
            res_norm = torch.sigmoid(x.norm(dim=-1).mean().unsqueeze(0))
        complexity = torch.cat([entropy_proxy, res_norm])
        budgets    = self.net(complexity)
        return budgets, self.beta * budgets.mean()


class ARIABlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads_init, n_heads_max,
                 dropout, split_τ, merge_τ, genome_dim, plasticity_lambda, layer_idx):
        super().__init__()
        self.idx  = layer_idx
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = MorphogenicAttention(
            d_model, n_heads_init, n_heads_max, dropout, split_τ, merge_τ, genome_dim
        )
        self.mlp  = PlasticityGatedMLP(d_model, d_ff, dropout, plasticity_lambda)

    def forward(self, x, genome, budget):
        z        = self.attn(self.ln1(x), genome) + x
        h, p_loss = self.mlp(self.ln2(z))
        b        = budget[self.idx]
        out      = b * (z + h) + (1 - b) * x
        return out, p_loss


class ARIA(nn.Module):
    """
    Full ARIA model for Split-MNIST (multi-head output).
    Each task head is a separate 2-class classifier (binary).
    """
    def __init__(self, cfg):
        super().__init__()
        D  = cfg["hidden_dim"]
        L  = cfg["n_layers"]
        G  = cfg["genome_dim"]
        d_ff = D * 2

        self.input_proj = nn.Linear(cfg["input_dim"], D)
        self.genome     = ArchitectureGenome(G, D, L)
        self.blocks     = nn.ModuleList([
            ARIABlock(D, d_ff,
                      cfg["n_heads_init"], cfg["n_heads_max"],
                      cfg["dropout"],
                      cfg["split_threshold"], cfg["merge_threshold"],
                      G, cfg["plasticity_lambda"], i)
            for i in range(L)
        ])
        self.budget_alloc = CognitiveBudgetAllocator(L, cfg["budget_beta"])
        self.ln_f         = nn.LayerNorm(D)

        # Separate output head per task (multi-head CL setup)
        self.task_heads   = nn.ModuleList()
        self.n_tasks_seen = 0

        self.morph_step    = 0
        self.morph_interval = cfg["morph_interval"]
        self.genome_gamma  = cfg["genome_gamma"]

    def add_task_head(self):
        D = next(self.parameters()).shape[-1]  # infer D
        # Actually get D from input_proj
        D = self.ln_f.normalized_shape[0]
        self.task_heads.append(nn.Linear(D, 2))
        self.n_tasks_seen += 1

    def forward(self, x, task_id):
        # x: (B, 784)
        h = F.gelu(self.input_proj(x)).unsqueeze(1)   # (B, 1, D)

        genome  = self.genome.decode()
        budgets, b_loss = self.budget_alloc(h.squeeze(1))
        skips   = genome["skip_probs"]

        total_p_loss = torch.tensor(0.0, device=x.device)
        prev_h = h.clone()

        for i, block in enumerate(self.blocks):
            if self.training and torch.rand(1).item() < skips[i].item() * 0.1:
                continue
            h, p_loss = block(h, genome, budgets)
            total_p_loss = total_p_loss + p_loss

        h   = self.ln_f(h).squeeze(1)           # (B, D)
        out = self.task_heads[task_id](h)        # (B, 2)

        if self.training:
            self.morph_step += 1
            if self.morph_step % self.morph_interval == 0:
                for block in self.blocks:
                    block.attn.morphogenesis()

        aux_loss = total_p_loss + b_loss + self.genome_gamma * self.genome.reg_loss()
        return out, aux_loss

    def get_arch_stats(self):
        return {
            "head_counts": [b.attn.n_active for b in self.blocks],
            "mean_plasticity": [b.mlp.mean_gate for b in self.blocks],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

class StaticMLP(nn.Module):
    """Simple MLP baseline — shared body, per-task heads."""
    def __init__(self, cfg):
        super().__init__()
        D = cfg["hidden_dim"]
        self.body = nn.Sequential(
            nn.Linear(cfg["input_dim"], D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )
        self.task_heads   = nn.ModuleList()
        self.n_tasks_seen = 0

    def add_task_head(self):
        D = CFG["hidden_dim"]
        self.task_heads.append(nn.Linear(D, 2))
        self.n_tasks_seen += 1

    def forward(self, x, task_id):
        h   = self.body(x)
        out = self.task_heads[task_id](h)
        return out, torch.tensor(0.0, device=x.device)


class ProgressiveNN(nn.Module):
    """
    Progressive Neural Networks — new column per task, lateral connections.
    Simple version: each task gets a fresh MLP column + lateral adapters.
    """
    def __init__(self, cfg):
        super().__init__()
        self.D        = cfg["hidden_dim"]
        self.inp      = cfg["input_dim"]
        self.columns  = nn.ModuleList()   # one MLP column per task
        self.laterals = nn.ModuleList()   # lateral connections from prev columns
        self.heads    = nn.ModuleList()
        self.n_tasks_seen = 0

    def add_task_head(self):
        D, I = self.D, self.inp
        col = nn.Sequential(
            nn.Linear(I, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )
        self.columns.append(col)
        self.heads.append(nn.Linear(D, 2))
        if self.n_tasks_seen > 0:
            # Lateral: compress all previous columns' hidden to D
            self.laterals.append(nn.Linear(D * self.n_tasks_seen, D))
        self.n_tasks_seen += 1

    def forward(self, x, task_id):
        # Run all columns up to task_id
        hiddens = []
        for c in range(task_id + 1):
            h = self.columns[c](x)
            hiddens.append(h)

        # Current column output, with optional lateral
        cur = hiddens[-1]
        if task_id > 0:
            prev = torch.cat(hiddens[:-1], dim=-1)
            lat  = self.laterals[task_id - 1](prev)
            cur  = cur + lat

        out = self.heads[task_id](cur)
        return out, torch.tensor(0.0, device=x.device)


class EWCWrapper(nn.Module):
    """
    EWC (Elastic Weight Consolidation) wrapping a StaticMLP.
    After each task, computes Fisher information and adds penalty.
    """
    def __init__(self, cfg):
        super().__init__()
        self.model     = StaticMLP(cfg)
        self.ewc_lambda = cfg["ewc_lambda"]

        # Stores (mean, fisher) per parameter per task
        self.task_params  = []   # list of {name: mean_tensor}
        self.task_fishers = []   # list of {name: fisher_tensor}

    def add_task_head(self):
        self.model.add_task_head()

    def forward(self, x, task_id):
        return self.model(x, task_id)

    def ewc_loss(self):
        if not self.task_params:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for means, fishers in zip(self.task_params, self.task_fishers):
            for name, param in self.model.named_parameters():
                if name in means:
                    diff   = param - means[name]
                    loss  += (fishers[name] * diff ** 2).sum()
        return self.ewc_lambda * loss

    def consolidate(self, loader, task_id, device):
        """Compute Fisher information after finishing a task."""
        self.model.eval()
        means   = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        fishers = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            self.model.zero_grad()
            out, _ = self.model(x, task_id)
            log_prob = F.log_softmax(out, dim=1)
            # Sample from output distribution
            sampled = torch.multinomial(log_prob.exp(), 1).squeeze()
            loss    = F.nll_loss(log_prob, sampled)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fishers[name] += param.grad.data.clone() ** 2

        n = len(loader)
        for name in fishers:
            fishers[name] /= n

        self.task_params.append(means)
        self.task_fishers.append(fishers)
        self.model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, task_id, device, is_ewc=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out, aux = model(x, task_id)
        task_loss = F.cross_entropy(out, y)

        if is_ewc:
            loss = task_loss + model.ewc_loss() + aux
        else:
            loss = task_loss + aux

        loss.backward()

        # Dampen slow pathway gradients in ARIA
        if hasattr(model, 'blocks'):
            for block in model.blocks:
                mg = block.mlp.mean_gate
                for p in block.mlp.W_slow_in.parameters():
                    if p.grad is not None: p.grad *= mg
                for p in block.mlp.W_slow_out.parameters():
                    if p.grad is not None: p.grad *= mg

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, task_id, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        out, _ = model(x, task_id)
        preds  = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += x.size(0)
    return correct / total


def backward_transfer(acc_matrix):
    """
    BWT = (1/(T-1)) * sum_{i<T} [A_{T,i} - A_{i,i}]
    Negative means forgetting occurred.
    acc_matrix[t][i] = accuracy on task i after learning task t.
    """
    T = len(acc_matrix)
    if T < 2:
        return 0.0
    bwt = 0.0
    for i in range(T - 1):
        bwt += acc_matrix[T-1][i] - acc_matrix[i][i]
    return bwt / (T - 1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model_name, model, task_loaders, device, is_ewc=False):
    """
    Train a model on all tasks sequentially.
    Returns acc_matrix[t][i] = accuracy on task i after learning task t.
    """
    T      = len(task_loaders)
    acc_matrix   = []
    flops_per_task = []

    # CSV log
    log_path = os.path.join(CFG["results_dir"], f"{model_name}_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "epoch", "train_loss", "train_acc", "time_s"])

    print(f"\n{'='*60}")
    print(f"Training: {model_name}  |  params: {count_params(model):,}")
    print(f"{'='*60}")

    for t in range(T):
        train_loader, test_loader = task_loaders[t]

        # Add output head for this task
        model.add_task_head()

        # Fresh optimizer each task (standard in CL)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG["epochs_per_task"]
        )

        print(f"\n  Task {t+1}/5", end="")
        if hasattr(model, 'blocks'):
            heads = [b.attn.n_active for b in model.blocks]
            print(f"  [ARIA heads: {heads}]", end="")
        print()

        t0 = time.time()
        for epoch in range(CFG["epochs_per_task"]):
            loss, acc = train_epoch(model, train_loader, optimizer, t, device, is_ewc)
            scheduler.step()

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([t+1, epoch+1, f"{loss:.4f}", f"{acc:.4f}",
                                        f"{time.time()-t0:.1f}"])

            if (epoch + 1) % 5 == 0:
                print(f"    epoch {epoch+1:2d}/{CFG['epochs_per_task']} "
                      f"| loss {loss:.4f} | acc {acc:.3f}")

        # EWC: consolidate Fisher after task
        if is_ewc:
            model.consolidate(train_loader, t, device)

        # Evaluate on ALL tasks seen so far
        row = []
        for i in range(t + 1):
            _, tl = task_loaders[i]
            a = evaluate(model, tl, i, device)
            row.append(round(a * 100, 2))
        # Pad unseen tasks with None
        while len(row) < T:
            row.append(None)
        acc_matrix.append(row)

        accs_seen = [row[i] for i in range(t+1)]
        print(f"  → Accs on tasks 1..{t+1}: {accs_seen}  |  avg: {np.mean(accs_seen):.2f}%")

    return acc_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Run everything
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = CFG["device"]

    print("\nLoading Split-MNIST...")
    task_loaders = get_split_mnist(CFG["n_tasks"], CFG["data_dir"])

    models = {
        "ARIA":           (ARIA(CFG),         False),
        "EWC":            (EWCWrapper(CFG),    True),
        "Static_MLP":     (StaticMLP(CFG),     False),
        "Progressive_NN": (ProgressiveNN(CFG), False),
    }

    all_results = {}

    for name, (model, is_ewc) in models.items():
        model = model.to(device)
        acc_matrix = train_model(name, model, task_loaders, device, is_ewc)
        all_results[name] = acc_matrix

        # Save per-model results
        out_path = os.path.join(CFG["results_dir"], f"{name}_acc_matrix.json")
        with open(out_path, "w") as f:
            json.dump(acc_matrix, f, indent=2)
        print(f"  Saved: {out_path}")

    # ── Summary Table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Avg Acc':<12} {'BWT':<12} {'Params':<12}")
    print(f"{'-'*55}")

    summary = {}
    for name, acc_matrix in all_results.items():
        T = len(acc_matrix)
        # Final avg accuracy (last row, all seen tasks)
        final_row = [acc_matrix[T-1][i] for i in range(T) if acc_matrix[T-1][i] is not None]
        avg_acc   = np.mean(final_row)
        bwt       = backward_transfer(acc_matrix)
        m_obj     = [v for k, (v, _) in models.items() if k == name][0]
        n_params  = count_params(m_obj)

        print(f"{name:<20} {avg_acc:<12.2f} {bwt:<12.2f} {n_params:<12,}")
        summary[name] = {
            "avg_acc": round(avg_acc, 2),
            "bwt":     round(bwt, 2),
            "n_params": n_params,
            "acc_matrix": acc_matrix,
        }

    # Save full summary
    summary_path = os.path.join(CFG["results_dir"], "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll results saved to: {CFG['results_dir']}/")

    # Auto-generate all figures
    generate_all_figures(summary, all_results)


# ─────────────────────────────────────────────────────────────────────────────
# Figure Generation (runs automatically after training)
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(summary, all_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from sklearn.decomposition import PCA

    figures_dir = os.path.join(CFG["results_dir"], "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plt.rcParams.update({
        'font.family':    'DejaVu Sans',
        'font.size':       11,
        'axes.titlesize':  13,
        'axes.labelsize':  11,
        'xtick.labelsize':  9,
        'ytick.labelsize':  9,
        'savefig.dpi':     200,
        'savefig.bbox':    'tight',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

    C = {
        'aria':        '#2563EB',
        'ewc':         '#DC2626',
        'static':      '#6B7280',
        'progressive': '#16A34A',
    }
    MODEL_COLORS = {
        'ARIA':           C['aria'],
        'EWC':            C['ewc'],
        'Static_MLP':     C['static'],
        'Progressive_NN': C['progressive'],
    }
    MODEL_LABELS = {
        'ARIA':           'ARIA (ours)',
        'EWC':            'EWC',
        'Static_MLP':     'Static MLP',
        'Progressive_NN': 'Progressive NN',
    }
    T = CFG["n_tasks"]

    # ── Figure 1: Average Accuracy + Forgetting on Task 1 ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for name, data in summary.items():
        mat   = data["acc_matrix"]
        means = []
        for t in range(T):
            seen = [mat[t][i] for i in range(t+1) if mat[t][i] is not None]
            means.append(np.mean(seen))
        ax.plot(range(1, T+1), means, 'o-',
                color=MODEL_COLORS[name], lw=2.2, markersize=7,
                label=MODEL_LABELS[name])
    ax.set_xlabel('Tasks learned so far')
    ax.set_ylabel('Average accuracy (%) on all seen tasks')
    ax.set_title('Continual Learning: Average Accuracy')
    ax.legend(fontsize=9)
    ax.set_xticks(range(1, T+1))
    ax.set_xticklabels([f'After T{i}' for i in range(1, T+1)])
    ax.set_ylim(0, 105)

    ax2 = axes[1]
    for name, data in summary.items():
        mat    = data["acc_matrix"]
        task1_perf = [mat[t][0] for t in range(T) if mat[t][0] is not None]
        ax2.plot(range(1, len(task1_perf)+1), task1_perf, 'o--',
                 color=MODEL_COLORS[name], lw=2.2, markersize=7,
                 label=MODEL_LABELS[name])
    ax2.set_xlabel('Tasks learned so far')
    ax2.set_ylabel('Accuracy (%) on Task 1')
    ax2.set_title('Catastrophic Forgetting on Task 1')
    ax2.legend(fontsize=9)
    ax2.set_xticks(range(1, T+1))
    ax2.set_xticklabels([f'After T{i}' for i in range(1, T+1)])
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    p = os.path.join(figures_dir, 'fig_continual_learning.png')
    plt.savefig(p); plt.close()
    print(f"  Saved: {p}")

    # ── Figure 2: Final Summary Bar Chart ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names  = list(summary.keys())
    labels = [MODEL_LABELS[n] for n in names]
    colors = [MODEL_COLORS[n] for n in names]
    avg_accs = [summary[n]["avg_acc"] for n in names]
    bwts     = [summary[n]["bwt"]     for n in names]

    ax = axes[0]
    bars = ax.bar(labels, avg_accs, color=colors, alpha=0.85, width=0.55)
    ax.set_ylabel('Final Average Accuracy (%)')
    ax.set_title('Average Accuracy After All 5 Tasks')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, avg_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax2 = axes[1]
    bar_colors = [C['aria'] if b > -5 else C['ewc'] if b > -20 else C['static']
                  for b in bwts]
    bars2 = ax2.bar(labels, bwts, color=colors, alpha=0.85, width=0.55)
    ax2.set_ylabel('Backward Transfer (BWT)')
    ax2.set_title('Backward Transfer\n(higher = less forgetting)')
    ax2.axhline(0, color='#374151', lw=1, ls='--')
    for bar, val in zip(bars2, bwts):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.3 if val >= 0 else -1.5),
                 f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    p = os.path.join(figures_dir, 'fig_summary_bars.png')
    plt.savefig(p); plt.close()
    print(f"  Saved: {p}")

    # ── Figure 3: Per-Task Accuracy Heatmaps ──────────────────────────────
    fig, axes = plt.subplots(1, len(summary), figsize=(5 * len(summary), 4))
    if len(summary) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, summary.items()):
        mat = np.array([[v if v is not None else float('nan')
                         for v in row] for row in data["acc_matrix"]])
        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        ax.set_xlabel('Task evaluated on')
        ax.set_ylabel('After learning task')
        ax.set_title(MODEL_LABELS[name])
        ax.set_xticks(range(T)); ax.set_xticklabels([f'T{i+1}' for i in range(T)])
        ax.set_yticks(range(T)); ax.set_yticklabels([f'T{i+1}' for i in range(T)])
        for i in range(T):
            for j in range(T):
                v = mat[i][j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                            fontsize=8, color='white' if v > 60 else '#1E293B')
        plt.colorbar(im, ax=ax, label='Accuracy (%)')

    fig.suptitle('Per-Task Accuracy Matrix (rows = after task t, cols = task evaluated)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(figures_dir, 'fig_accuracy_heatmaps.png')
    plt.savefig(p); plt.close()
    print(f"  Saved: {p}")

    # ── Figure 4: Forgetting Bound Δ_t ≤ C·π̄²  (theoretical) ─────────────
    pi_vals = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    for C_val, ls, lw in [(0.5, '-', 2.5), (1.0, '--', 2), (2.0, ':', 2)]:
        ax.plot(pi_vals, C_val * pi_vals**2, lw=lw, ls=ls,
                label=f'C = {C_val}')
    ax.fill_between(pi_vals, 0, 1.0 * pi_vals**2, alpha=0.10, color=C['aria'])
    ax.axvline(0.3, color='#16A34A', lw=1.5, ls='--', alpha=0.8,
               label='Typical π̄ at convergence')
    ax.set_xlabel('Mean plasticity gate  π̄')
    ax.set_ylabel('Forgetting upper bound  Δ_t')
    ax.set_title('Proposition 1: Forgetting Bound  Δ_t ≤ C · π̄²')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 2.2)
    plt.tight_layout()
    p = os.path.join(figures_dir, 'fig_forgetting_bound.png')
    plt.savefig(p); plt.close()
    print(f"  Saved: {p}")

    print(f"\n✅  All figures saved to: {figures_dir}/")
    print("    fig_continual_learning.png")
    print("    fig_summary_bars.png")
    print("    fig_accuracy_heatmaps.png")
    print("    fig_forgetting_bound.png")


if __name__ == "__main__":
    main()
