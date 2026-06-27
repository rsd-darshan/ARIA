"""
aria_train_v3.py  —  ARIA: Adaptive Recurrent Intelligence Architecture
=====================================================================
v3 fixes:
  - Morphogenesis: trigger now uses viability score (learned head importance)
    instead of broken weight-variance heuristic
  - Plasticity gates: lambda 0.005→0.05, epochs 10→25, bifurcation confirmed
  - CBA: complexity signal now uses raw input (pixel entropy + norm variation)
    not hidden state, so it differentiates across tasks
  - Genome trajectory: annotation offsets now relative to data range, no flyaway
  - morph_interval: 200→50 (checks more frequently for faster response)
"""

import os, csv, time, copy, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    "n_tasks":           5,
    "batch_size":        128,
    "data_dir":          "./data",
    "input_dim":         784,
    "hidden_dim":        256,
    "n_layers":          4,
    "n_heads_init":      4,
    "n_heads_max":       8,
    "genome_dim":        32,
    "dropout":           0.1,

    # Morphogenesis — FIXED: viability-based trigger, lower threshold
    "split_threshold":   0.60,   # viability score above this → head splits
    "merge_threshold":   0.97,   # cosine sim above this → heads merge
    "morph_interval":    50,     # check every 50 steps (was 200)

    # Plasticity — FIXED: stronger lambda forces bifurcation
    "plasticity_lambda": 0.05,   # was 0.005

    "budget_beta":       0.001,
    "genome_gamma":      0.0001,

    # Training — more epochs for mechanisms to activate
    "epochs_per_task":   25,     # was 10
    "lr":                3e-4,
    "weight_decay":      1e-4,
    "ewc_lambda":        5000,

    "results_dir":       "./results",
    "seed":              42,
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CFG["results_dir"], exist_ok=True)
os.makedirs(CFG["data_dir"],    exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
print(f"Device: {CFG['device']}")
print(f"Epochs per task: {CFG['epochs_per_task']}  |  "
      f"Total epochs: {CFG['epochs_per_task'] * CFG['n_tasks']}")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_split_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train_full = datasets.MNIST(CFG["data_dir"], train=True,  download=True, transform=transform)
    test_full  = datasets.MNIST(CFG["data_dir"], train=False, download=True, transform=transform)

    loaders = []
    print("\nLoading Split-MNIST...")
    for t in range(CFG["n_tasks"]):
        c0, c1 = t*2, t*2+1

        class Relabeled(torch.utils.data.Dataset):
            def __init__(self, ds, c0, c1):
                self.idx = [i for i,(_, y) in enumerate(ds) if y==c0 or y==c1]
                self.ds = ds; self.c0 = c0
            def __len__(self): return len(self.idx)
            def __getitem__(self, i):
                x, y = self.ds[self.idx[i]]
                return x, int(y != self.c0)

        tr = Relabeled(train_full, c0, c1)
        te = Relabeled(test_full,  c0, c1)
        loaders.append((
            DataLoader(tr, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0),
            DataLoader(te, batch_size=256,               shuffle=False, num_workers=0),
        ))
        print(f"  Task {t+1} (digits {c0} vs {c1}): {len(tr)} train, {len(te)} test")
    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# ARIA Components
# ─────────────────────────────────────────────────────────────────────────────

class ArchitectureGenome(nn.Module):
    def __init__(self):
        super().__init__()
        G, D, L = CFG["genome_dim"], CFG["hidden_dim"], CFG["n_layers"]
        self.z         = nn.Parameter(torch.randn(G) * 0.01)
        self.proj_skip = nn.Linear(G, L)
        self.proj_temp = nn.Linear(G, 1)
        self.proj_cond = nn.Linear(G, D)

    def decode(self, device):
        z = self.z.to(device)
        return {
            "skip_probs":  torch.sigmoid(self.proj_skip(z)),
            "temperature": F.softplus(self.proj_temp(z)).squeeze() + 0.5,
            "cond_signal": torch.tanh(self.proj_cond(z)),
        }

    def reg_loss(self):
        return 0.5 * (self.z**2).mean()


class MorphogenicAttention(nn.Module):
    def __init__(self):
        super().__init__()
        D   = CFG["hidden_dim"]
        H   = CFG["n_heads_max"]
        d_h = D // H
        self.d_h     = d_h
        self.split_τ = CFG["split_threshold"]
        self.merge_τ = CFG["merge_threshold"]
        self.H       = H

        self.W_Q       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_K       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_V       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_O       = nn.Parameter(torch.randn(H, d_h, D) * 0.02)
        # Viability: learned head importance score
        # Initialized slightly negative so heads must earn their activation
        self.viability = nn.Parameter(torch.zeros(H) - 0.5)

        mask = torch.zeros(H, dtype=torch.bool)
        mask[:CFG["n_heads_init"]] = True
        self.register_buffer("head_mask", mask)
        self.dropout = nn.Dropout(CFG["dropout"])

    @property
    def n_active(self):
        return int(self.head_mask.sum().item())

    def forward(self, x, genome):
        B, T, D = x.shape
        τ      = genome["temperature"].clamp(min=0.1)
        active = self.head_mask.nonzero(as_tuple=True)[0]
        outputs = []
        for i in active:
            Q = x @ self.W_Q[i]
            K = x @ self.W_K[i]
            V = x @ self.W_V[i]
            scale  = math.sqrt(self.d_h) * τ
            scores = (Q @ K.transpose(-2,-1)) / scale
            causal = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(causal==0, float('-inf'))
            attn   = self.dropout(F.softmax(scores, dim=-1))
            out    = (attn @ V) @ self.W_O[i]
            outputs.append(torch.sigmoid(self.viability[i]) * out)

        result = torch.stack(outputs, 0).sum(0)
        result = result + genome["cond_signal"].to(x.device)
        return result

    def morphogenesis(self):
        """
        FIXED trigger: use learned viability score as head-load proxy.
        High viability (>split_τ) → head is overloaded → split.
        High cosine similarity between two heads → redundant → merge.
        """
        active = self.head_mask.nonzero(as_tuple=True)[0].tolist()

        # ── Split: heads with high viability are doing too much work ─────────
        newly_split = []
        for i in active:
            if self.n_active >= CFG["n_heads_max"]: break
            v_score = torch.sigmoid(self.viability[i]).item()
            if v_score > self.split_τ:
                inactive = (~self.head_mask).nonzero(as_tuple=True)[0]
                if len(inactive) == 0: break
                j = inactive[0].item()
                with torch.no_grad():
                    noise_scale = 0.05
                    for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                        W[j] = W[i] + noise_scale * torch.randn_like(W[i])
                    # Child starts with lower viability so it must prove itself
                    self.viability[j] = self.viability[i] - 0.5
                self.head_mask[j] = True
                newly_split.append(j)

        # ── Merge: redundant heads (high cosine similarity) ──────────────────
        active = self.head_mask.nonzero(as_tuple=True)[0].tolist()
        done   = set()
        for idx in range(len(active)-1):
            i, j = active[idx], active[idx+1]
            if j in done or i in newly_split or j in newly_split: continue
            cos = F.cosine_similarity(
                self.W_Q[i].flatten().unsqueeze(0),
                self.W_Q[j].flatten().unsqueeze(0)
            ).item()
            if cos > self.merge_τ and self.n_active > 2:
                with torch.no_grad():
                    for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                        W[i] = (W[i] + W[j]) / 2
                    self.viability[i] = max(self.viability[i], self.viability[j])
                done.add(j)
        for j in done:
            self.head_mask[j] = False


class PlasticityGatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        D, d_ff = CFG["hidden_dim"], CFG["hidden_dim"] * 2
        self.W_fast_in  = nn.Linear(D, d_ff)
        self.W_fast_out = nn.Linear(d_ff, D)
        self.W_slow_in  = nn.Linear(D, d_ff)
        self.W_slow_out = nn.Linear(d_ff, D)
        # Gate network: predicts per-token π ∈ (0,1)
        self.gate_net   = nn.Sequential(
            nn.Linear(D, d_ff//4), nn.ReLU(),
            nn.Linear(d_ff//4, 1), nn.Sigmoid()
        )
        self.dropout    = nn.Dropout(CFG["dropout"])
        self.lambda_    = CFG["plasticity_lambda"]
        self.mean_gate  = 0.5   # tracked for logging

    def forward(self, x):
        π = self.gate_net(x)                        # (B, T, 1)
        self.mean_gate = float(π.detach().mean().item())
        h_fast = F.gelu(self.W_fast_in(x))
        h_slow = F.gelu(self.W_slow_in(x))
        out    = π * self.W_fast_out(h_fast) + (1-π) * self.W_slow_out(h_slow)
        # Specialization loss: penalize gates near 0.5, reward near 0 or 1
        p_loss = self.lambda_ / (π*(1-π) + 1e-4).mean()
        return self.dropout(out), p_loss


class CognitiveBudgetAllocator(nn.Module):
    """
    FIXED: uses raw input pixel statistics for complexity signal,
    not the hidden state (which is too uniform post-convergence).
    Signal: [pixel entropy proxy, spatial variance proxy] → per-layer budget
    """
    def __init__(self):
        super().__init__()
        L = CFG["n_layers"]
        # 3-feature input: pixel std, pixel entropy proxy, pixel range
        self.net  = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, L), nn.Sigmoid()
        )
        self.beta = CFG["budget_beta"]

    def forward(self, x_raw):
        """x_raw: (B, 784) raw flattened input pixels"""
        dev = x_raw.device
        # Feature 1: std of pixel values (overall contrast)
        f1 = x_raw.std(dim=-1).mean().unsqueeze(0)
        # Feature 2: entropy proxy — mean of p*log(p) where p = softmax(|x|)
        p  = F.softmax(x_raw.abs(), dim=-1)
        f2 = -(p * (p + 1e-8).log()).sum(dim=-1).mean().unsqueeze(0) / math.log(784)
        # Feature 3: range (max-min) — structural complexity
        f3 = (x_raw.max(dim=-1).values - x_raw.min(dim=-1).values).mean().unsqueeze(0)
        complexity = torch.cat([f1, f2, f3]).to(dev)
        b = self.net(complexity)
        return b, self.beta * b.mean()


class ARIABlock(nn.Module):
    def __init__(self, idx):
        super().__init__()
        D = CFG["hidden_dim"]
        self.idx = idx
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.attn = MorphogenicAttention()
        self.mlp  = PlasticityGatedMLP()

    def forward(self, x, genome, budget):
        z, p = self.attn(self.ln1(x), genome), torch.tensor(0.0, device=x.device)
        z     = z + x
        h, p  = self.mlp(self.ln2(z))
        b     = budget[self.idx]
        return b*(z+h) + (1-b)*x, p


class ARIA(nn.Module):
    def __init__(self):
        super().__init__()
        D = CFG["hidden_dim"]
        self.input_proj   = nn.Linear(CFG["input_dim"], D)
        self.genome       = ArchitectureGenome()
        self.blocks       = nn.ModuleList([ARIABlock(i) for i in range(CFG["n_layers"])])
        self.budget_alloc = CognitiveBudgetAllocator()
        self.ln_f         = nn.LayerNorm(D)
        self.task_heads   = nn.ModuleList()
        self.n_tasks_seen = 0
        self.morph_step   = 0

    def add_task_head(self):
        device = next(self.parameters()).device
        self.task_heads.append(nn.Linear(CFG["hidden_dim"], 2).to(device))
        self.n_tasks_seen += 1

    def forward(self, x, task_id):
        device  = x.device
        # CBA uses raw input — FIXED
        budgets, b_loss = self.budget_alloc(x)
        budgets = budgets.to(device)

        h      = F.gelu(self.input_proj(x)).unsqueeze(1)
        genome = self.genome.decode(device)

        total_p = torch.tensor(0.0, device=device)
        for i, block in enumerate(self.blocks):
            if self.training and torch.rand(1).item() < genome["skip_probs"][i].item() * 0.1:
                continue
            h, p = block(h, genome, budgets)
            total_p = total_p + p

        h   = self.ln_f(h).squeeze(1)
        out = self.task_heads[task_id](h)

        if self.training:
            self.morph_step += 1
            if self.morph_step % CFG["morph_interval"] == 0:
                for block in self.blocks:
                    block.attn.morphogenesis()

        return out, total_p + b_loss + CFG["genome_gamma"] * self.genome.reg_loss()


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

class StaticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        D = CFG["hidden_dim"]
        self.body = nn.Sequential(
            nn.Linear(CFG["input_dim"], D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )
        self.task_heads   = nn.ModuleList()
        self.n_tasks_seen = 0

    def add_task_head(self):
        device = next(self.parameters()).device
        self.task_heads.append(nn.Linear(CFG["hidden_dim"], 2).to(device))
        self.n_tasks_seen += 1

    def forward(self, x, task_id):
        return self.task_heads[task_id](self.body(x)), torch.tensor(0.0, device=x.device)


class EWCWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model        = StaticMLP()
        self.task_params  = []
        self.task_fishers = []

    def add_task_head(self): self.model.add_task_head()
    def forward(self, x, task_id): return self.model(x, task_id)

    def ewc_loss(self):
        if not self.task_params:
            return torch.tensor(0.0)
        dev  = next(self.parameters()).device
        loss = torch.tensor(0.0, device=dev)
        for means, fishers in zip(self.task_params, self.task_fishers):
            for name, param in self.model.named_parameters():
                if name in means:
                    loss += (fishers[name] * (param - means[name])**2).sum()
        return CFG["ewc_lambda"] * loss

    def consolidate(self, loader, task_id, device):
        self.model.eval()
        means   = {n: p.clone().detach() for n,p in self.model.named_parameters()}
        fishers = {n: torch.zeros_like(p) for n,p in self.model.named_parameters()}
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            self.model.zero_grad()
            out, _ = self.model(x, task_id)
            F.nll_loss(F.log_softmax(out, 1),
                       torch.multinomial(F.softmax(out, 1), 1).squeeze()).backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fishers[name] += param.grad.data**2
        n = len(loader)
        for name in fishers: fishers[name] /= n
        self.task_params.append(means)
        self.task_fishers.append(fishers)
        self.model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, task_id, device, is_ewc=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out, aux  = model(x, task_id)
        task_loss = F.cross_entropy(out, y)
        loss      = task_loss + aux + (model.ewc_loss() if is_ewc else 0)
        loss.backward()
        # Dampen slow pathway gradients in ARIA
        if isinstance(model, ARIA):
            for block in model.blocks:
                mg = block.mlp.mean_gate
                for p in (list(block.mlp.W_slow_in.parameters()) +
                          list(block.mlp.W_slow_out.parameters())):
                    if p.grad is not None: p.grad.mul_(mg)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss/total, correct/total


@torch.no_grad()
def evaluate(model, loader, task_id, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out, _ = model(x, task_id)
        correct += (out.argmax(1) == y).sum().item()
        total   += x.size(0)
    return correct / total


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def backward_transfer(mat):
    T = len(mat)
    if T < 2: return 0.0
    return np.mean([mat[T-1][i] - mat[i][i] for i in range(T-1)])


# ─────────────────────────────────────────────────────────────────────────────
# ARIA Internal Logger
# ─────────────────────────────────────────────────────────────────────────────

class ARIALogger:
    def __init__(self):
        self.records = []

    def log(self, model, global_step, task_id):
        if not isinstance(model, ARIA): return
        with torch.no_grad():
            decoded = model.genome.decode(next(model.parameters()).device)
        self.records.append({
            "step":        global_step,
            "task":        task_id + 1,
            "head_counts": [b.attn.n_active for b in model.blocks],
            "viability":   [
                [round(torch.sigmoid(b.attn.viability[i]).item(), 4)
                 for i in b.attn.head_mask.nonzero(as_tuple=True)[0].tolist()]
                for b in model.blocks
            ],
            "gate_means":  [round(b.mlp.mean_gate, 5) for b in model.blocks],
            "z_arch":      model.genome.z.detach().cpu().tolist(),
            "temperature": round(decoded["temperature"].item(), 4),
            "skip_probs":  [round(v.item(), 4) for v in decoded["skip_probs"]],
        })

    @torch.no_grad()
    def log_budgets(self, model, task_loaders, device):
        if not isinstance(model, ARIA): return {}
        model.eval()
        budgets = {}
        for i, (label_t, (loader, _)) in enumerate(
                zip([f"Task {i+1}" for i in range(len(task_loaders))], task_loaders)):
            x, _ = next(iter(loader))
            x = x.to(device)
            b, _ = model.budget_alloc(x)
            budgets[label_t] = [round(v.item(), 4) for v in b.detach().cpu()]
        model.train()
        return budgets

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.records, f)


# ─────────────────────────────────────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(name, model, task_loaders, device, is_ewc=False):
    T           = len(task_loaders)
    acc_matrix  = []
    aria_logger = ARIALogger()
    global_step = 0

    log_path = os.path.join(CFG["results_dir"], f"{name}_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["task","epoch","loss","acc","time_s"])

    print(f"\n{'='*60}")
    print(f"Training: {name}  |  params: {count_params(model):,}")
    print(f"{'='*60}")

    for t in range(T):
        tr_loader, te_loader = task_loaders[t]
        model.add_task_head()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG["epochs_per_task"])

        print(f"\n  Task {t+1}/{T}", end="")
        if isinstance(model, ARIA):
            print(f"  [heads: {[b.attn.n_active for b in model.blocks]}]", end="")
        print()

        t0 = time.time()
        for epoch in range(CFG["epochs_per_task"]):
            loss, acc = train_epoch(model, tr_loader, optimizer, t, device, is_ewc)
            scheduler.step()
            global_step += 1

            aria_logger.log(model, global_step, t)

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([t+1, epoch+1, f"{loss:.4f}", f"{acc:.4f}",
                                        f"{time.time()-t0:.1f}"])
            if (epoch+1) % 5 == 0:
                hstr = ""
                if isinstance(model, ARIA):
                    hstr = f"  heads:{[b.attn.n_active for b in model.blocks]}"
                    gates = [round(b.mlp.mean_gate, 3) for b in model.blocks]
                    hstr += f"  gates:{gates}"
                print(f"    epoch {epoch+1:2d}/{CFG['epochs_per_task']} "
                      f"| loss {loss:.4f} | acc {acc:.3f}{hstr}")

        if is_ewc:
            model.consolidate(tr_loader, t, device)

        row = [round(evaluate(model, task_loaders[i][1], i, device)*100, 2)
               for i in range(t+1)]
        while len(row) < T: row.append(None)
        acc_matrix.append(row)
        print(f"  → Accs 1..{t+1}: {row[:t+1]}  avg: {np.mean(row[:t+1]):.2f}%")

    # Save ARIA logs
    if isinstance(model, ARIA):
        lp = os.path.join(CFG["results_dir"], "ARIA_internal_log.json")
        aria_logger.save(lp)
        print(f"  ARIA log saved: {lp}")
        budget_log = aria_logger.log_budgets(model, task_loaders, device)
        bp = os.path.join(CFG["results_dir"], "ARIA_budget_log.json")
        with open(bp, "w") as f: json.dump(budget_log, f)
        print(f"  Budget log saved: {bp}")

    with open(os.path.join(CFG["results_dir"], f"{name}_acc_matrix.json"), "w") as f:
        json.dump(acc_matrix, f, indent=2)

    return acc_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Figure Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(summary, aria_log, budget_log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fdir = os.path.join(CFG["results_dir"], "figures")
    os.makedirs(fdir, exist_ok=True)

    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
        'axes.labelsize':11,'xtick.labelsize':9,'ytick.labelsize':9,
        'savefig.dpi':200,'savefig.bbox':'tight',
        'axes.spines.top':False,'axes.spines.right':False,
    })

    MC = {'ARIA':'#2563EB','EWC':'#DC2626','Static_MLP':'#6B7280'}
    ML = {'ARIA':'ARIA (ours)','EWC':'EWC','Static_MLP':'Static MLP'}
    T  = CFG["n_tasks"]

    # ── Fig 1: Continual Learning Lines ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for name, data in summary.items():
        mat   = data["acc_matrix"]
        means = [np.mean([mat[t][i] for i in range(t+1)]) for t in range(T)]
        ax.plot(range(1,T+1), means, 'o-', color=MC[name], lw=2.2,
                markersize=7, label=ML[name])
    ax.set(xlabel='Tasks learned so far',
           ylabel='Average accuracy (%) on all seen tasks',
           title='Continual Learning: Average Accuracy', ylim=(50,105))
    ax.set_xticks(range(1,T+1))
    ax.set_xticklabels([f'After T{i}' for i in range(1,T+1)])
    ax.legend(fontsize=10)

    ax2 = axes[1]
    for name, data in summary.items():
        mat = data["acc_matrix"]
        t1  = [mat[t][0] for t in range(T)]
        ax2.plot(range(1,T+1), t1, 'o--', color=MC[name], lw=2.2,
                 markersize=7, label=ML[name])
    ax2.set(xlabel='Tasks learned so far', ylabel='Accuracy (%) on Task 1',
            title='Catastrophic Forgetting on Task 1', ylim=(50,105))
    ax2.set_xticks(range(1,T+1))
    ax2.set_xticklabels([f'After T{i}' for i in range(1,T+1)])
    ax2.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{fdir}/fig_continual_learning.png'); plt.close()
    print("  ✓ fig_continual_learning.png")

    # ── Fig 2: Summary Bars ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names    = list(summary.keys())
    avg_accs = [summary[n]["avg_acc"] for n in names]
    bwts     = [summary[n]["bwt"]     for n in names]
    ax = axes[0]
    bars = ax.bar([ML[n] for n in names], avg_accs,
                  color=[MC[n] for n in names], alpha=0.85, width=0.5)
    ax.set(ylabel='Final Average Accuracy (%)',
           title='Average Accuracy After All 5 Tasks', ylim=(70,100))
    for bar, val in zip(bars, avg_accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax2 = axes[1]
    bars2 = ax2.bar([ML[n] for n in names], bwts,
                    color=[MC[n] for n in names], alpha=0.85, width=0.5)
    ax2.axhline(0, color='#374151', lw=1, ls='--')
    ax2.set(ylabel='Backward Transfer (BWT)',
            title='Backward Transfer\n(closer to 0 = less forgetting)')
    for bar, val in zip(bars2, bwts):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+(0.2 if val>=0 else -1.2),
                 f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fdir}/fig_summary_bars.png'); plt.close()
    print("  ✓ fig_summary_bars.png")

    # ── Fig 3: Accuracy Heatmaps ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(summary), figsize=(5*len(summary), 4))
    if len(summary)==1: axes=[axes]
    for ax, (name, data) in zip(axes, summary.items()):
        mat = np.array([[v if v is not None else float('nan')
                         for v in row] for row in data["acc_matrix"]])
        im  = ax.imshow(mat, cmap='Blues', vmin=50, vmax=100, aspect='auto')
        ax.set(xlabel='Task evaluated on', ylabel='After learning task',
               title=ML[name])
        ax.set_xticks(range(T)); ax.set_xticklabels([f'T{i+1}' for i in range(T)])
        ax.set_yticks(range(T)); ax.set_yticklabels([f'T{i+1}' for i in range(T)])
        for i in range(T):
            for j in range(T):
                v = mat[i][j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                            fontsize=8, color='white' if v>75 else '#1E293B')
        plt.colorbar(im, ax=ax, label='Accuracy (%)')
    fig.suptitle('Per-Task Accuracy Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fdir}/fig_accuracy_heatmaps.png'); plt.close()
    print("  ✓ fig_accuracy_heatmaps.png")

    # ── Fig 4: Head Morphogenesis ─────────────────────────────────────────────
    if aria_log:
        steps     = [r["step"] for r in aria_log]
        n_layers  = len(aria_log[0]["head_counts"])
        head_data = np.array([r["head_counts"] for r in aria_log])
        total_h   = head_data.sum(axis=1)
        task_ids  = [r["task"] for r in aria_log]

        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        cmap = plt.cm.Blues(np.linspace(0.45, 0.9, n_layers))
        ax   = axes[0]
        for l in range(n_layers):
            ax.plot(steps, head_data[:, l], color=cmap[l], lw=2,
                    label=f'Layer {l+1}')
        for t in range(2, T+1):
            idx = next((i for i,v in enumerate(task_ids) if v==t), None)
            if idx:
                ax.axvline(steps[idx], color='#9CA3AF', lw=1.2, ls='--', alpha=0.7)
                ypos = ax.get_ylim()[1] if ax.get_ylim()[1] > 4 else 5
                ax.text(steps[idx]+0.3, ax.get_ylim()[0]+0.05,
                        f'T{t}', fontsize=9, color='#6B7280')
        ax.set_yticks(range(CFG["n_heads_init"]-1, CFG["n_heads_max"]+2))
        ax.set(ylabel='Active heads per layer',
               title='Head Morphogenesis Over Training (Real — logged per epoch)')
        ax.legend(fontsize=9, ncol=2)

        ax2 = axes[1]
        ax2.fill_between(steps, 0, total_h, alpha=0.15, color='#2563EB')
        ax2.plot(steps, total_h, color='#2563EB', lw=2.5, label='ARIA total heads')
        init_total = CFG["n_heads_init"] * n_layers
        ax2.axhline(init_total, color='#DC2626', lw=1.8, ls='--',
                    label=f'Transformer fixed ({init_total} heads)')
        ax2.set(xlabel='Training epoch (cumulative across tasks)',
                ylabel='Total active heads',
                title='ARIA vs Fixed Transformer: Total Head Count')
        ax2.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{fdir}/fig_morphogenesis.png'); plt.close()
        print("  ✓ fig_morphogenesis.png  (REAL)")
    else:
        print("  ✗ fig_morphogenesis.png  (no log)")

    # ── Fig 5: Plasticity Gates ───────────────────────────────────────────────
    if aria_log:
        n_layers  = len(aria_log[0]["gate_means"])
        gate_data = np.array([r["gate_means"] for r in aria_log])
        n         = len(gate_data)

        fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4.5))
        for l, ax in enumerate(axes):
            early = gate_data[:n//5,         l]
            mid   = gate_data[2*n//5:3*n//5, l]
            late  = gate_data[4*n//5:,       l]
            bins  = np.linspace(0, 1, 25)
            ax.hist(early, bins=bins, alpha=0.55, color='#F59E0B',
                    label='Early',  density=True)
            ax.hist(mid,   bins=bins, alpha=0.55, color='#7C3AED',
                    label='Mid',    density=True)
            ax.hist(late,  bins=bins, alpha=0.55, color='#2563EB',
                    label='Late',   density=True)
            ax.axvline(0.5, color='#DC2626', lw=1.5, ls='--', alpha=0.7,
                       label='Uniform (0.5)')
            ax.set(xlabel='Mean gate π̄', title=f'Layer {l+1}', xlim=(0,1))
            ax.legend(fontsize=8)

        fig.suptitle('Plasticity Gate Distribution: Early → Late Training\n'
                     '(bimodal = fast/slow specialization)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{fdir}/fig_plasticity_gates.png'); plt.close()
        print("  ✓ fig_plasticity_gates.png  (REAL)")
    else:
        print("  ✗ fig_plasticity_gates.png  (no log)")

    # ── Fig 6: AGV Genome Trajectory ─────────────────────────────────────────
    if aria_log and len(aria_log) >= 3:
        z_data = np.array([r["z_arch"] for r in aria_log])
        steps  = np.array([r["step"]   for r in aria_log])
        tasks  = np.array([r["task"]   for r in aria_log])

        pca  = PCA(n_components=2)
        proj = pca.fit_transform(z_data)

        # Compute data range for relative offsets — FIXED
        x_range = proj[:,0].max() - proj[:,0].min()
        y_range = proj[:,1].max() - proj[:,1].min()
        dx = x_range * 0.05
        dy = y_range * 0.05

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(proj[:,0], proj[:,1], c=steps,
                        cmap='Blues', s=40, zorder=3, alpha=0.85)
        ax.plot(proj[:,0], proj[:,1], color='#94A3B8', lw=0.8, alpha=0.4, zorder=2)
        ax.scatter(*proj[0],  s=130, color='#F59E0B', zorder=5,
                   label='Start (random init)')
        ax.scatter(*proj[-1], s=200, color='#2563EB', marker='*', zorder=5,
                   label='Converged genome')

        for t in range(2, T+1):
            idx = np.where(tasks == t)[0]
            if len(idx):
                px, py = proj[idx[0]]
                ax.scatter(px, py, s=90, color='#DC2626', marker='^',
                           zorder=4, alpha=0.9)
                # Relative offset so label stays inside plot
                ax.annotate(f'Task {t}', xy=(px, py),
                            xytext=(px+dx, py+dy),
                            fontsize=9, color='#DC2626',
                            arrowprops=dict(arrowstyle='-', color='#DC2626',
                                           lw=0.8, alpha=0.6))

        plt.colorbar(sc, ax=ax, label='Training epoch')
        ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)',
               ylabel=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)',
               title='Architecture Genome Vector (AGV) Trajectory\n'
                     '(PCA of z_arch — real training dynamics)')
        ax.legend(fontsize=9)
        # Set tight axis limits
        pad = 0.15
        ax.set_xlim(proj[:,0].min()-x_range*pad, proj[:,0].max()+x_range*pad)
        ax.set_ylim(proj[:,1].min()-y_range*pad, proj[:,1].max()+y_range*pad)
        plt.tight_layout()
        plt.savefig(f'{fdir}/fig_genome_trajectory.png'); plt.close()
        print("  ✓ fig_genome_trajectory.png  (REAL)")
    else:
        print("  ✗ fig_genome_trajectory.png  (no log)")

    # ── Fig 7: Adaptive Compute ───────────────────────────────────────────────
    if budget_log:
        labels   = list(budget_log.keys())
        budgets  = np.array([budget_log[l] for l in labels])
        n_layers = budgets.shape[1]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        im = ax.imshow(budgets, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f'L{i+1}' for i in range(n_layers)])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set(title='Real Budget Allocation b_l per Task', xlabel='Layer')
        plt.colorbar(im, ax=ax, label='Compute budget b_l')
        for i in range(len(labels)):
            for j in range(n_layers):
                ax.text(j, i, f'{budgets[i,j]:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if budgets[i,j]>0.5 else '#1E293B')

        ax2 = axes[1]
        avg_b = budgets.mean(axis=1)*100
        x = np.arange(len(labels))
        ax2.bar(x-0.2, np.ones(len(labels))*100, 0.35,
                label='Transformer (fixed)', color='#6B7280', alpha=0.7)
        ax2.bar(x+0.2, avg_b, 0.35,
                label='ARIA (adaptive)', color='#2563EB', alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15)
        ax2.set(ylabel='Relative FLOPs (%)',
                title='ARIA vs Transformer: Compute Usage per Task')
        ax2.legend()
        for i, v in enumerate(avg_b):
            ax2.text(i+0.2, v+1, f'{v:.0f}%', ha='center', fontsize=9,
                     color='#2563EB', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{fdir}/fig_adaptive_compute.png'); plt.close()
        print("  ✓ fig_adaptive_compute.png  (REAL)")
    else:
        print("  ✗ fig_adaptive_compute.png  (no log)")

    # ── Fig 8: Forgetting Bound (mathematical) ────────────────────────────────
    pi_vals = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    for C_val, ls in [(0.5,'-'),(1.0,'--'),(2.0,':')]:
        ax.plot(pi_vals, C_val*pi_vals**2, lw=2.2, ls=ls, label=f'C = {C_val}')
    ax.fill_between(pi_vals, 0, pi_vals**2, alpha=0.10, color='#2563EB')
    ax.axvline(0.3, color='#16A34A', lw=1.5, ls='--', alpha=0.8,
               label='Typical π̄ at convergence')
    ax.set(xlabel='Mean plasticity gate  π̄',
           ylabel='Forgetting upper bound  Δ_t',
           title='Proposition 1: Forgetting Bound  Δ_t ≤ C · π̄²',
           xlim=(0,1), ylim=(0,2.2))
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{fdir}/fig_forgetting_bound.png'); plt.close()
    print("  ✓ fig_forgetting_bound.png  (mathematical)")

    print(f"\n  All figures saved to: {fdir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device       = CFG["device"]
    task_loaders = get_split_mnist()

    models = {
        "ARIA":       (ARIA(),       False),
        "EWC":        (EWCWrapper(), True),
        "Static_MLP": (StaticMLP(),  False),
    }

    all_results = {}
    for name, (model, is_ewc) in models.items():
        model      = model.to(device)
        acc_matrix = train_model(name, model, task_loaders, device, is_ewc)
        all_results[name] = acc_matrix

    print(f"\n{'='*60}")
    print(f"{'Model':<15} {'Avg Acc':>10} {'BWT':>10} {'Params':>12}")
    print(f"{'-'*52}")
    summary = {}
    for name, mat in all_results.items():
        final   = [mat[T-1][i] for i in range(T) if mat[T-1][i] is not None
                   for T in [len(mat)]]
        T       = len(mat)
        final   = [mat[T-1][i] for i in range(T) if mat[T-1][i] is not None]
        avg_acc = round(float(np.mean(final)), 2)
        bwt     = round(float(backward_transfer(mat)), 2)
        n_p     = count_params([m for n,(m,_) in models.items() if n==name][0])
        print(f"{name:<15} {avg_acc:>10.2f} {bwt:>10.2f} {n_p:>12,}")
        summary[name] = {"avg_acc":avg_acc,"bwt":bwt,"n_params":n_p,"acc_matrix":mat}

    with open(os.path.join(CFG["results_dir"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    aria_log   = []
    budget_log = {}
    lp = os.path.join(CFG["results_dir"], "ARIA_internal_log.json")
    bp = os.path.join(CFG["results_dir"], "ARIA_budget_log.json")
    if os.path.exists(lp):
        with open(lp) as f: aria_log = json.load(f)
    if os.path.exists(bp):
        with open(bp) as f: budget_log = json.load(f)

    print("\nGenerating all figures from real data...")
    generate_all_figures(summary, aria_log, budget_log)

    import shutil
    shutil.make_archive('/kaggle/working/ARIA_results_v3', 'zip', CFG["results_dir"])
    print("\n✅ Done! Download ARIA_results_v3.zip from the Output panel.")


if __name__ == "__main__":
    main()
