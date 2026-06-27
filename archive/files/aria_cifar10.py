"""
aria_cifar10.py  —  ARIA on Split-CIFAR-10
==========================================
5 tasks: [plane/car], [bird/cat], [deer/dog], [frog/horse], [ship/truck]
CNN feature extractor → ARIA transformer blocks

Key differences from Split-MNIST:
- Harder visual tasks → heads should saturate → morphogenesis fires
- More gradient signal → plasticity gates should bifurcate
- Lower accuracy per task → more forgetting pressure → AGV contributes

Trains: ARIA-Full, EWC, Static CNN-MLP
Saves:  ARIA_cifar10_results.zip
"""

import os, json, math, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "n_tasks":           5,
    "batch_size":        128,
    "data_dir":          "./data",
    "input_dim":         512,       # CNN feature dim
    "hidden_dim":        256,
    "n_layers":          4,
    "n_heads_init":      4,
    "n_heads_max":       8,
    "genome_dim":        32,
    "dropout":           0.2,
    "split_threshold":   0.55,      # lower → easier to trigger splits
    "merge_threshold":   0.95,
    "morph_interval":    30,        # check more often
    "plasticity_lambda": 0.10,      # stronger specialization pressure
    "budget_beta":       0.001,
    "genome_gamma":      0.0001,
    "epochs_per_task":   40,        # more epochs → gates have time to bifurcate
    "lr":                3e-4,
    "weight_decay":      1e-4,
    "results_dir":       "./results_cifar10",
    "seed":              42,
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CFG["results_dir"], exist_ok=True)
os.makedirs(CFG["data_dir"],    exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
print(f"Device: {CFG['device']}")
print(f"Epochs per task: {CFG['epochs_per_task']}  |  Total: {CFG['epochs_per_task']*CFG['n_tasks']}")


# ─────────────────────────────────────────────────────────────────────────────
# Data — Split-CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────

TASK_PAIRS = [(0,1),(2,3),(4,5),(6,7),(8,9)]
TASK_NAMES = ["plane/car","bird/cat","deer/dog","frog/horse","ship/truck"]

def get_split_cifar10():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
    ])
    train_full = datasets.CIFAR10(CFG["data_dir"], train=True,  download=True, transform=train_tf)
    test_full  = datasets.CIFAR10(CFG["data_dir"], train=False, download=True, transform=test_tf)

    loaders = []
    print("\nLoading Split-CIFAR-10...")
    for t, (c0, c1) in enumerate(TASK_PAIRS):
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
            DataLoader(tr, batch_size=CFG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True),
            DataLoader(te, batch_size=256,               shuffle=False, num_workers=2, pin_memory=True),
        ))
        print(f"  Task {t+1} ({TASK_NAMES[t]}): {len(tr)} train, {len(te)} test")
    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# CNN Feature Extractor (shared, frozen after pretraining or jointly trained)
# ─────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """Lightweight CNN → 512-dim feature vector."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),   # 16×16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),   # 8×8

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),  # 2×2 → 256*4=1024 → proj to 512
        )
        self.proj = nn.Sequential(nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU())

    def forward(self, x):
        h = self.net(x)
        h = h.flatten(1)
        return self.proj(h)


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
        D, H = CFG["hidden_dim"], CFG["n_heads_max"]
        d_h  = D // H
        self.d_h = d_h; self.H = H
        self.W_Q       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_K       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_V       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_O       = nn.Parameter(torch.randn(H, d_h, D) * 0.02)
        self.viability = nn.Parameter(torch.zeros(H))
        mask = torch.zeros(H, dtype=torch.bool)
        mask[:CFG["n_heads_init"]] = True
        self.register_buffer("head_mask", mask)
        self.dropout = nn.Dropout(CFG["dropout"])

    @property
    def n_active(self): return int(self.head_mask.sum().item())

    def forward(self, x, genome):
        B, T, D = x.shape
        τ = genome["temperature"].clamp(min=0.1)
        active = self.head_mask.nonzero(as_tuple=True)[0]
        outputs = []
        for i in active:
            Q = x @ self.W_Q[i]; K = x @ self.W_K[i]; V = x @ self.W_V[i]
            scores = (Q @ K.transpose(-2,-1)) / (math.sqrt(self.d_h) * τ)
            causal = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(causal==0, float('-inf'))
            attn   = self.dropout(F.softmax(scores, dim=-1))
            outputs.append(torch.sigmoid(self.viability[i]) * (attn @ V) @ self.W_O[i])
        result = torch.stack(outputs, 0).sum(0)
        result = result + genome["cond_signal"].to(x.device)
        return result

    def morphogenesis(self):
        active = self.head_mask.nonzero(as_tuple=True)[0].tolist()
        newly_split = []
        for i in active:
            if self.n_active >= CFG["n_heads_max"]: break
            if torch.sigmoid(self.viability[i]).item() > CFG["split_threshold"]:
                inactive = (~self.head_mask).nonzero(as_tuple=True)[0]
                if len(inactive) == 0: break
                j = inactive[0].item()
                with torch.no_grad():
                    for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                        W[j] = W[i] + 0.05 * torch.randn_like(W[i])
                    self.viability[j] = self.viability[i] - 0.5
                self.head_mask[j] = True
                newly_split.append(j)
        active = self.head_mask.nonzero(as_tuple=True)[0].tolist()
        done = set()
        for idx in range(len(active)-1):
            i, j = active[idx], active[idx+1]
            if j in done or i in newly_split or j in newly_split: continue
            cos = F.cosine_similarity(self.W_Q[i].flatten().unsqueeze(0),
                                      self.W_Q[j].flatten().unsqueeze(0)).item()
            if cos > CFG["merge_threshold"] and self.n_active > 2:
                with torch.no_grad():
                    for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                        W[i] = (W[i] + W[j]) / 2
                done.add(j)
        for j in done: self.head_mask[j] = False


class PlasticityGatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        D, d_ff = CFG["hidden_dim"], CFG["hidden_dim"] * 2
        self.W_fast_in  = nn.Linear(D, d_ff)
        self.W_fast_out = nn.Linear(d_ff, D)
        self.W_slow_in  = nn.Linear(D, d_ff)
        self.W_slow_out = nn.Linear(d_ff, D)
        self.gate_net   = nn.Sequential(
            nn.Linear(D, d_ff//4), nn.ReLU(), nn.Linear(d_ff//4, 1), nn.Sigmoid())
        self.dropout    = nn.Dropout(CFG["dropout"])
        self.lambda_    = CFG["plasticity_lambda"]
        self.mean_gate  = 0.5

    def forward(self, x):
        π = self.gate_net(x)
        self.mean_gate = float(π.detach().mean().item())
        h_f = F.gelu(self.W_fast_in(x)); h_s = F.gelu(self.W_slow_in(x))
        out = π * self.W_fast_out(h_f) + (1-π) * self.W_slow_out(h_s)
        return self.dropout(out), self.lambda_ / (π*(1-π)+1e-4).mean()


class CognitiveBudgetAllocator(nn.Module):
    def __init__(self):
        super().__init__()
        L = CFG["n_layers"]
        self.net  = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, L), nn.Sigmoid())
        self.beta = CFG["budget_beta"]

    def forward(self, x_raw):
        dev = x_raw.device
        f1  = x_raw.std(dim=-1).mean().unsqueeze(0)
        p   = F.softmax(x_raw.abs(), dim=-1)
        f2  = -(p*(p+1e-8).log()).sum(dim=-1).mean().unsqueeze(0) / math.log(CFG["input_dim"])
        f3  = (x_raw.max(dim=-1).values - x_raw.min(dim=-1).values).mean().unsqueeze(0)
        b   = self.net(torch.cat([f1,f2,f3]).to(dev))
        return b, self.beta * b.mean()


class ARIABlock(nn.Module):
    def __init__(self, idx):
        super().__init__()
        D = CFG["hidden_dim"]
        self.idx = idx
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.attn = MorphogenicAttention()
        self.mlp  = PlasticityGatedMLP()

    def forward(self, x, genome, budget):
        z    = self.attn(self.ln1(x), genome) + x
        h, p = self.mlp(self.ln2(z))
        b    = budget[self.idx]
        return b*(z+h) + (1-b)*x, p


# ─────────────────────────────────────────────────────────────────────────────
# Full Models
# ─────────────────────────────────────────────────────────────────────────────

class ARIA_CIFAR(nn.Module):
    name = "ARIA"

    def __init__(self):
        super().__init__()
        D = CFG["hidden_dim"]
        self.encoder      = CNNEncoder()
        self.input_proj   = nn.Linear(CFG["input_dim"], D)
        self.genome       = ArchitectureGenome()
        self.blocks       = nn.ModuleList([ARIABlock(i) for i in range(CFG["n_layers"])])
        self.budget_alloc = CognitiveBudgetAllocator()
        self.ln_f         = nn.LayerNorm(D)
        self.task_heads   = nn.ModuleList()
        self.n_tasks_seen = 0
        self.morph_step   = 0
        self.internal_log = []  # (epoch, layer_idx, n_heads)
        self.budget_log   = {}  # task_id → list of budgets
        self.gate_log     = {}  # task_id → list of mean gates per layer

    def add_task_head(self):
        device = next(self.parameters()).device
        self.task_heads.append(nn.Linear(CFG["hidden_dim"], 2).to(device))
        self.n_tasks_seen += 1

    def log_state(self, epoch):
        heads = [b.attn.n_active for b in self.blocks]
        gates = [round(b.mlp.mean_gate, 4) for b in self.blocks]
        self.internal_log.append({"epoch": epoch, "heads": heads, "gates": gates})

    def forward(self, x, task_id):
        device  = x.device
        feats   = self.encoder(x)
        budgets, b_loss = self.budget_alloc(feats)
        budgets = budgets.to(device)
        h       = F.gelu(self.input_proj(feats)).unsqueeze(1)
        genome  = self.genome.decode(device)
        total_p = torch.tensor(0.0, device=device)
        for i, block in enumerate(self.blocks):
            if self.training and torch.rand(1).item() < genome["skip_probs"][i].item()*0.1:
                continue
            h, p = block(h, genome, budgets)
            total_p = total_p + p
        h   = self.ln_f(h).squeeze(1)
        out = self.task_heads[task_id](h)
        if self.training:
            self.morph_step += 1
            if self.morph_step % CFG["morph_interval"] == 0:
                for block in self.blocks: block.attn.morphogenesis()
        return out, total_p + b_loss + CFG["genome_gamma"] * self.genome.reg_loss()


class EWC_CIFAR(nn.Module):
    name = "EWC"

    def __init__(self):
        super().__init__()
        D = 256
        self.encoder    = CNNEncoder()
        self.backbone   = nn.Sequential(
            nn.Linear(512, D), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(D, D),   nn.ReLU(), nn.Dropout(0.2),
        )
        self.task_heads = nn.ModuleList()
        self.ewc_lambda = 5000
        self.fisher     = {}
        self.opt_params = {}
        self.n_tasks_seen = 0

    def add_task_head(self):
        device = next(self.parameters()).device
        self.task_heads.append(nn.Linear(256, 2).to(device))
        self.n_tasks_seen += 1

    def forward(self, x, task_id):
        h   = self.encoder(x)
        h   = self.backbone(h)
        out = self.task_heads[task_id](h)
        ewc_loss = torch.tensor(0.0, device=x.device)
        for n, p in self.named_parameters():
            if n in self.fisher:
                ewc_loss += (self.fisher[n] * (p - self.opt_params[n])**2).sum()
        return out, self.ewc_lambda * ewc_loss * 0.5

    def consolidate(self, loader, task_id, device):
        self.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            out, _ = self.forward(x, task_id)
            F.cross_entropy(out, y).backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * x.size(0)
        n_samples = len(loader.dataset)
        self.fisher     = {n: f/n_samples for n, f in fisher.items()}
        self.opt_params = {n: p.detach().clone() for n, p in self.named_parameters()}


class StaticCNN(nn.Module):
    name = "Static_CNN"

    def __init__(self):
        super().__init__()
        D = 256
        self.encoder    = CNNEncoder()
        self.backbone   = nn.Sequential(
            nn.Linear(512, D), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(D, D),   nn.ReLU(), nn.Dropout(0.2),
        )
        self.task_heads   = nn.ModuleList()
        self.n_tasks_seen = 0

    def add_task_head(self):
        device = next(self.parameters()).device
        self.task_heads.append(nn.Linear(256, 2).to(device))
        self.n_tasks_seen += 1

    def forward(self, x, task_id):
        h   = self.encoder(x)
        h   = self.backbone(h)
        out = self.task_heads[task_id](h)
        return out, torch.tensor(0.0, device=x.device)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, task_id, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out, aux = model(x, task_id)
        loss     = F.cross_entropy(out, y) + aux
        loss.backward()
        if hasattr(model, 'blocks'):
            for block in model.blocks:
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'mean_gate'):
                    mg = block.mlp.mean_gate
                    for p in (list(block.mlp.W_slow_in.parameters()) +
                              list(block.mlp.W_slow_out.parameters())):
                        if p.grad is not None: p.grad.mul_(mg)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1)==y).sum().item()
        total      += x.size(0)
    return total_loss/total, correct/total


@torch.no_grad()
def evaluate(model, loader, task_id, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out, _ = model(x, task_id)
        correct += (out.argmax(1)==y).sum().item()
        total   += x.size(0)
    return correct / total


def backward_transfer(mat):
    T = len(mat)
    if T < 2: return 0.0
    return float(np.mean([mat[T-1][i] - mat[i][i] for i in range(T-1)]))


def train_model(model, task_loaders, device):
    T          = len(task_loaders)
    acc_matrix = []
    global_ep  = 0

    print(f"\n{'='*60}")
    print(f"Training: {model.name}  |  params: {count_params(model):,}")
    print(f"{'='*60}")

    for t in range(T):
        tr_loader, te_loader = task_loaders[t]
        model.add_task_head()
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=CFG["epochs_per_task"])

        heads_str = ""
        if hasattr(model, 'blocks'):
            heads = [b.attn.n_active for b in model.blocks]
            heads_str = f"  [heads: {heads}]"
        print(f"\n  Task {t+1}/{T} ({TASK_NAMES[t]}){heads_str}")

        for epoch in range(CFG["epochs_per_task"]):
            loss, acc = train_epoch(model, tr_loader, optimizer, t, device)
            scheduler.step()
            global_ep += 1
            if hasattr(model, 'log_state'):
                model.log_state(global_ep)
            if (epoch+1) % 8 == 0 or epoch == CFG["epochs_per_task"]-1:
                if hasattr(model, 'blocks'):
                    heads = [b.attn.n_active for b in model.blocks]
                    gates = [round(b.mlp.mean_gate, 3) for b in model.blocks]
                    print(f"    epoch {epoch+1:2d}/{CFG['epochs_per_task']} "
                          f"| loss {loss:.4f} | acc {acc:.3f} "
                          f" heads:{heads}  gates:{gates}")
                else:
                    print(f"    epoch {epoch+1:2d}/{CFG['epochs_per_task']} "
                          f"| loss {loss:.4f} | acc {acc:.3f}")

        # EWC consolidation
        if isinstance(model, EWC_CIFAR):
            model.consolidate(tr_loader, t, device)

        # Log budgets for ARIA
        if hasattr(model, 'budget_log') and hasattr(model, 'budget_alloc'):
            model.eval()
            x_sample = next(iter(te_loader))[0][:32].to(device)
            with torch.no_grad():
                feats = model.encoder(x_sample)
                b, _  = model.budget_alloc(feats)
                model.budget_log[str(t)] = b.detach().cpu().tolist()

        row = [round(evaluate(model, task_loaders[i][1], i, device)*100, 2)
               for i in range(t+1)]
        while len(row) < T: row.append(None)
        acc_matrix.append(row)
        print(f"  → Accs 1..{t+1}: {row[:t+1]}  avg: {np.mean(row[:t+1]):.2f}%")

    return acc_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def generate_figures(summary, aria_log, budget_log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fdir = os.path.join(CFG["results_dir"], "figures")
    os.makedirs(fdir, exist_ok=True)
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
        'savefig.dpi':200,'savefig.bbox':'tight',
        'axes.spines.top':False,'axes.spines.right':False,
    })
    COLORS = {"ARIA":"#2563EB","EWC":"#DC2626","Static_CNN":"#6B7280"}
    T = CFG["n_tasks"]

    # 1. Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, data in summary.items():
        mat   = data["acc_matrix"]
        means = [np.mean([mat[t][i] for i in range(t+1)]) for t in range(T)]
        axes[0].plot(range(1,T+1), means, color=COLORS[name], lw=2.5,
                     marker='o', markersize=6, label=name)
        t1 = [mat[t][0] for t in range(T)]
        axes[1].plot(range(1,T+1), t1, color=COLORS[name], lw=2,
                     marker='o', markersize=5, ls='--', label=name)
    for ax, title, ylabel in zip(axes,
            ['Continual Learning: Average Accuracy','Catastrophic Forgetting on Task 1'],
            ['Average accuracy (%) on all seen tasks','Accuracy (%) on Task 1']):
        ax.set(xlabel='Tasks learned so far', ylabel=ylabel, title=title)
        ax.set_xticks(range(1,T+1))
        ax.set_xticklabels([f'After T{i}' for i in range(1,T+1)], rotation=15)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fdir,'fig_cifar_learning_curves.png'))
    plt.close(); print("  ✓ fig_cifar_learning_curves.png")

    # 2. Summary bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names    = list(summary.keys())
    avg_accs = [summary[n]["avg_acc"] for n in names]
    bwts     = [summary[n]["bwt"]     for n in names]
    cols     = [COLORS[n] for n in names]
    for ax, vals, title, ylabel in zip(axes,
            [avg_accs, bwts],
            ['Average Accuracy After All 5 Tasks','Backward Transfer\n(closer to 0 = less forgetting)'],
            ['Final Average Accuracy (%)','Backward Transfer (BWT)']):
        bars = ax.bar(names, vals, color=cols, alpha=0.85, width=0.5)
        ax.set(title=title, ylabel=ylabel)
        if ylabel=='Backward Transfer (BWT)': ax.axhline(0,color='gray',lw=1,ls='--')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+(0.2 if val>=0 else -1.2),
                    f'{val:.1f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fdir,'fig_cifar_summary_bars.png'))
    plt.close(); print("  ✓ fig_cifar_summary_bars.png")

    # 3. Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Per-Task Accuracy Matrix (Split-CIFAR-10)', fontweight='bold', y=1.02)
    cmap = LinearSegmentedColormap.from_list('', ['white','#1e40af'])
    for ax, (name, data) in zip(axes, summary.items()):
        mat = np.full((T,T), np.nan)
        for i in range(T):
            for j in range(i+1):
                v = data["acc_matrix"][i][j]
                if v is not None: mat[i,j] = v
        im = ax.imshow(mat, cmap=cmap, vmin=40, vmax=100, aspect='auto')
        ax.set(title=name, xlabel='Task evaluated on', ylabel='After learning task')
        ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_xticklabels([f'T{i+1}' for i in range(T)])
        ax.set_yticklabels([f'T{i+1}' for i in range(T)])
        for i in range(T):
            for j in range(i+1):
                if not np.isnan(mat[i,j]):
                    ax.text(j, i, f'{mat[i,j]:.0f}', ha='center', va='center',
                            color='white' if mat[i,j] < 70 else 'black', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(fdir,'fig_cifar_heatmaps.png'))
    plt.close(); print("  ✓ fig_cifar_heatmaps.png")

    # 4. Morphogenesis
    if aria_log:
        epochs = [e["epoch"] for e in aria_log]
        fig, axes = plt.subplots(2, 1, figsize=(13, 8))
        layer_colors = ['#93C5FD','#3B82F6','#1D4ED8','#1E3A8A']
        for li in range(CFG["n_layers"]):
            heads = [e["heads"][li] for e in aria_log]
            axes[0].plot(epochs, heads, color=layer_colors[li], lw=2,
                         label=f'Layer {li+1}')
        task_starts = [1] + [1 + t*CFG["epochs_per_task"] for t in range(1,T)]
        for ts in task_starts[1:]:
            axes[0].axvline(ts, color='gray', ls='--', alpha=0.5)
            axes[0].text(ts+0.5, axes[0].get_ylim()[1]*0.97,
                         f'T{task_starts.index(ts)+1}', fontsize=8, color='gray')
        axes[0].set(ylabel='Active heads per layer',
                    title='Head Morphogenesis Over Training (Real — logged per epoch)',
                    ylim=(CFG["n_heads_init"]-1, CFG["n_heads_max"]+1))
        axes[0].legend(ncol=2)

        total_heads = [sum(e["heads"]) for e in aria_log]
        fixed_total = CFG["n_heads_init"] * CFG["n_layers"]
        axes[1].fill_between(epochs, total_heads, alpha=0.3, color='#2563EB')
        axes[1].plot(epochs, total_heads, color='#2563EB', lw=2, label='ARIA total heads')
        axes[1].axhline(fixed_total, color='#DC2626', lw=1.5, ls='--',
                        label=f'Transformer fixed ({fixed_total} heads)')
        axes[1].set(xlabel='Training epoch (cumulative across tasks)',
                    ylabel='Total active heads',
                    title='ARIA vs Fixed Transformer: Total Head Count')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fdir,'fig_cifar_morphogenesis.png'))
        plt.close(); print("  ✓ fig_cifar_morphogenesis.png  (REAL)")

    # 5. Plasticity gates
    if aria_log:
        fig, axes = plt.subplots(1, CFG["n_layers"], figsize=(16, 4), sharey=True)
        fig.suptitle('Plasticity Gate Distribution: Early → Late Training\n(Split-CIFAR-10)',
                     fontweight='bold')
        n_ep = len(aria_log)
        thirds = [
            [e["gates"] for e in aria_log[:n_ep//3]],
            [e["gates"] for e in aria_log[n_ep//3:2*n_ep//3]],
            [e["gates"] for e in aria_log[2*n_ep//3:]],
        ]
        stage_colors = ['#FCD34D','#A78BFA','#3B82F6']
        stage_labels = ['Early','Mid','Late']
        for li in range(CFG["n_layers"]):
            ax = axes[li]
            for gates_list, color, label in zip(thirds, stage_colors, stage_labels):
                vals = [g[li] for g in gates_list]
                ax.hist(vals, bins=20, range=(0,1), alpha=0.6,
                        color=color, label=label, orientation='horizontal')
            ax.axhline(0.5, color='red', ls='--', lw=1.2, label='Uniform (0.5)')
            ax.set(title=f'Layer {li+1}', xlabel='Count',
                   ylabel='Mean gate π̄' if li==0 else '')
            if li == 0: ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(fdir,'fig_cifar_plasticity_gates.png'))
        plt.close(); print("  ✓ fig_cifar_plasticity_gates.png  (REAL)")

    # 6. Budget allocator
    if budget_log:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        budgets_arr = np.array([budget_log.get(str(t), [0]*CFG["n_layers"]) for t in range(T)])
        im = axes[0].imshow(budgets_arr, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        axes[0].set(title='Real Budget Allocation b_l per Task',
                    xlabel='Layer', ylabel='Task')
        axes[0].set_xticks(range(CFG["n_layers"]))
        axes[0].set_xticklabels([f'L{i+1}' for i in range(CFG["n_layers"])])
        axes[0].set_yticks(range(T))
        axes[0].set_yticklabels([f'Task {i+1}' for i in range(T)])
        for i in range(T):
            for j in range(CFG["n_layers"]):
                axes[0].text(j, i, f'{budgets_arr[i,j]:.2f}', ha='center',
                             va='center', fontsize=9)
        plt.colorbar(im, ax=axes[0], label='Compute budget b_l')
        avg_budgets = budgets_arr.mean(axis=1) * 100
        axes[1].bar(range(1,T+1), [100]*T, color='lightgray', label='Transformer (fixed)', width=0.4)
        axes[1].bar([x+0.4 for x in range(1,T+1)], avg_budgets,
                    color='#2563EB', label='ARIA (adaptive)', width=0.4)
        for i, v in enumerate(avg_budgets):
            axes[1].text(i+1.4, v+1, f'{v:.0f}%', ha='center', fontsize=9, color='#2563EB', fontweight='bold')
        axes[1].set(title='ARIA vs Transformer: Compute Usage per Task',
                    xlabel='Task', ylabel='Relative FLOPs (%)', ylim=(0,115))
        axes[1].set_xticks(range(1,T+1))
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fdir,'fig_cifar_adaptive_compute.png'))
        plt.close(); print("  ✓ fig_cifar_adaptive_compute.png  (REAL)")

    print(f"\n  All CIFAR-10 figures saved to: {fdir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = CFG["device"]
    task_loaders = get_split_cifar10()

    models = [ARIA_CIFAR(), EWC_CIFAR(), StaticCNN()]
    summary = {}
    param_counts = {}

    for model in models:
        name  = model.name
        model = model.to(device)
        mat   = train_model(model, task_loaders, device)

        T       = len(mat)
        final   = [mat[T-1][i] for i in range(T) if mat[T-1][i] is not None]
        avg_acc = round(float(np.mean(final)), 2)
        bwt     = round(backward_transfer(mat), 2)
        n_p     = count_params(model)
        param_counts[name] = n_p

        if hasattr(model, 'internal_log'):
            with open(os.path.join(CFG["results_dir"], f"{name}_internal_log.json"), 'w') as f:
                json.dump(model.internal_log, f, indent=2)
        if hasattr(model, 'budget_log'):
            with open(os.path.join(CFG["results_dir"], f"{name}_budget_log.json"), 'w') as f:
                json.dump(model.budget_log, f, indent=2)

        with open(os.path.join(CFG["results_dir"], f"{name}_acc_matrix.json"), 'w') as f:
            json.dump(mat, f, indent=2)

        summary[name] = {"avg_acc":avg_acc, "bwt":bwt, "n_params":n_p, "acc_matrix":mat}

    print(f"\n{'='*60}")
    print(f"{'Model':<16} {'Avg Acc':>10} {'BWT':>10} {'Params':>12}")
    print(f"{'-'*55}")
    for name, data in summary.items():
        print(f"{name:<16} {data['avg_acc']:>10.2f} {data['bwt']:>10.2f} {data['n_params']:>12,}")

    with open(os.path.join(CFG["results_dir"], "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Load logs for ARIA
    aria_log, budget_log = [], {}
    lp = os.path.join(CFG["results_dir"], "ARIA_internal_log.json")
    bp = os.path.join(CFG["results_dir"], "ARIA_budget_log.json")
    if os.path.exists(lp):
        with open(lp) as f: aria_log = json.load(f)
    if os.path.exists(bp):
        with open(bp) as f: budget_log = json.load(f)

    print("\nGenerating figures...")
    generate_figures(summary, aria_log, budget_log)

    shutil.make_archive('/kaggle/working/ARIA_cifar10_results', 'zip', CFG["results_dir"])
    print("\nDone! Download ARIA_cifar10_results.zip from Output panel.")


if __name__ == "__main__":
    main()
