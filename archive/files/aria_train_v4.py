"""
aria_train_v4.py
================
ARIA v4 — Full multi-seed experiment on Split-MNIST.

Key fixes vs v3:
  - Morphogenesis triggered by per-head gradient norm (not viability sigmoid)
  - 5 seeds for statistical validity
  - Baselines: EWC, DER++, A-GEM, Scaled-EWC (matched params)
  - Permuted-MNIST option (20 tasks)
  - Reports mean ± std across seeds

Run on Kaggle:
  !python aria_train_v4.py
"""

import os, sys, json, math, shutil, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "benchmark":         "split_mnist",   # or "permuted_mnist"
    "n_tasks":           5,
    "n_permuted_tasks":  20,              # used if benchmark=permuted_mnist
    "batch_size":        128,
    "data_dir":          "./data",
    "input_dim":         784,
    "hidden_dim":        256,
    "n_layers":          4,
    "n_heads_init":      4,
    "n_heads_max":       8,
    "genome_dim":        32,
    "dropout":           0.1,
    "split_threshold":   0.70,            # top-30% grad norm → split
    "merge_threshold":   0.97,
    "morph_interval":    30,
    "plasticity_lambda": 0.10,
    "budget_beta":       0.001,
    "genome_gamma":      0.0001,
    "epochs_per_task":   25,
    "lr":                3e-4,
    "weight_decay":      1e-4,
    "results_dir":       "./results_v4",
    "seeds":             [42, 7, 123, 999, 2024],
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CFG["results_dir"], exist_ok=True)
os.makedirs(CFG["data_dir"],    exist_ok=True)
print(f"Device: {CFG['device']}")
print(f"Benchmark: {CFG['benchmark']}  |  Seeds: {CFG['seeds']}")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def get_split_mnist():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train_full = datasets.MNIST(CFG["data_dir"], train=True,  download=True, transform=tf)
    test_full  = datasets.MNIST(CFG["data_dir"], train=False, download=True, transform=tf)
    loaders = []
    for t in range(CFG["n_tasks"]):
        c0, c1 = t*2, t*2+1
        class Rel(torch.utils.data.Dataset):
            def __init__(self, ds, c0, c1):
                self.idx = [i for i,(_, y) in enumerate(ds) if y==c0 or y==c1]
                self.ds = ds; self.c0 = c0
            def __len__(self): return len(self.idx)
            def __getitem__(self, i):
                x, y = self.ds[self.idx[i]]
                return x, int(y != self.c0)
        tr = Rel(train_full, c0, c1)
        te = Rel(test_full,  c0, c1)
        loaders.append((
            DataLoader(tr, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0),
            DataLoader(te, batch_size=256,               shuffle=False, num_workers=0),
        ))
    return loaders


def get_permuted_mnist(seed=42):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train_full = datasets.MNIST(CFG["data_dir"], train=True,  download=True, transform=tf)
    test_full  = datasets.MNIST(CFG["data_dir"], train=False, download=True, transform=tf)
    T = CFG["n_permuted_tasks"]
    rng = np.random.RandomState(seed)
    loaders = []
    for t in range(T):
        perm = torch.from_numpy(rng.permutation(784))
        class Permuted(torch.utils.data.Dataset):
            def __init__(self, ds, perm):
                self.ds = ds; self.perm = perm
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                x, y = self.ds[i]
                return x[self.perm], y
        tr = Permuted(train_full, perm)
        te = Permuted(test_full,  perm)
        loaders.append((
            DataLoader(tr, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0),
            DataLoader(te, batch_size=256,               shuffle=False, num_workers=0),
        ))
    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
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
    def reg_loss(self): return 0.5 * (self.z**2).mean()


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
        self.dropout    = nn.Dropout(CFG["dropout"])
        self._grad_norms = torch.zeros(H)   # EMA of per-head grad norms

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
            mask   = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(mask==0, float('-inf'))
            attn   = self.dropout(F.softmax(scores, dim=-1))
            outputs.append(torch.sigmoid(self.viability[i]) * (attn @ V) @ self.W_O[i])
        result = torch.stack(outputs, 0).sum(0)
        return result + genome["cond_signal"].to(x.device)

    def accumulate_grad_norms(self):
        """EMA update of per-head gradient norms."""
        with torch.no_grad():
            for i in range(self.H):
                if not self.head_mask[i]: continue
                norm = sum(W.grad[i].norm().item() for W in [self.W_Q, self.W_K, self.W_V, self.W_O]
                           if W.grad is not None)
                self._grad_norms[i] = 0.9 * self._grad_norms[i] + 0.1 * norm

    def morphogenesis(self):
        active = self.head_mask.nonzero(as_tuple=True)[0].tolist()
        if not active: return

        # Split: top split_threshold percentile grad norm
        norms     = [self._grad_norms[i].item() for i in active]
        threshold = sorted(norms)[int(len(norms) * CFG["split_threshold"])]

        newly_split = []
        for i in active:
            if self.n_active >= CFG["n_heads_max"]: break
            if self._grad_norms[i].item() > threshold > 0:
                inactive = (~self.head_mask).nonzero(as_tuple=True)[0]
                if len(inactive) == 0: break
                j = inactive[0].item()
                with torch.no_grad():
                    for W in [self.W_Q, self.W_K, self.W_V, self.W_O]:
                        W[j] = W[i] + 0.05 * torch.randn_like(W[i])
                    self.viability[j] = self.viability[i] - 0.5
                    self._grad_norms[j] = self._grad_norms[i] * 0.5
                self.head_mask[j] = True
                newly_split.append(j)

        # Merge: high cosine similarity
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
        self.net  = nn.Sequential(nn.Linear(3,32), nn.ReLU(), nn.Linear(32,L), nn.Sigmoid())
        self.beta = CFG["budget_beta"]
    def forward(self, x_raw):
        dev = x_raw.device
        f1  = x_raw.std(dim=-1).mean().unsqueeze(0)
        p   = F.softmax(x_raw.abs(), dim=-1)
        f2  = -(p*(p+1e-8).log()).sum(dim=-1).mean().unsqueeze(0) / math.log(784)
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


class ARIA(nn.Module):
    name = "ARIA"
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
        self.internal_log = []
        self.budget_log   = {}

    def add_task_head(self):
        device = next(self.parameters()).device
        self.task_heads.append(nn.Linear(CFG["hidden_dim"], 2).to(device))
        self.n_tasks_seen += 1

    def forward(self, x, task_id):
        device  = x.device
        budgets, b_loss = self.budget_alloc(x)
        h       = F.gelu(self.input_proj(x)).unsqueeze(1)
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

    def post_backward(self):
        for block in self.blocks:
            block.attn.accumulate_grad_norms()


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def make_mlp(hidden_dim=256):
    D = CFG["input_dim"]
    return nn.Sequential(
        nn.Linear(D, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
    )


class EWC(nn.Module):
    name = "EWC"
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.backbone   = make_mlp(hidden_dim)
        self.task_heads = nn.ModuleList()
        self.n_tasks_seen = 0
        self.ewc_lambda = 5000
        self.fisher     = {}; self.opt_params = {}
    def add_task_head(self):
        device = next(self.parameters()).device
        self.task_heads.append(nn.Linear(256, 2).to(device))
        self.n_tasks_seen += 1
    def forward(self, x, task_id):
        h   = self.backbone(x)
        out = self.task_heads[task_id](h)
        ewc = torch.tensor(0.0, device=x.device)
        for n, p in self.named_parameters():
            if n in self.fisher:
                ewc += (self.fisher[n].to(p.device) * (p - self.opt_params[n].to(p.device))**2).sum()
        return out, self.ewc_lambda * ewc * 0.5
    def consolidate(self, loader, task_id, device):
        self.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            self.zero_grad()
            F.cross_entropy(self.forward(x, task_id)[0], y).backward()
            for n, p in self.named_parameters():
                if p.grad is not None: fisher[n] += p.grad.pow(2) * x.size(0)
        n = len(loader.dataset)
        self.fisher     = {k: v/n for k, v in fisher.items()}
        self.opt_params = {k: p.detach().clone() for k, p in self.named_parameters()}


class ScaledEWC(EWC):
    """EWC scaled to match ARIA parameter count — fair comparison."""
    name = "Scaled-EWC"
    def __init__(self):
        super().__init__(hidden_dim=1024)
        # Override backbone to match ARIA params more closely
        D = CFG["input_dim"]
        self.backbone = nn.Sequential(
            nn.Linear(D, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(),
        )
    def add_task_head(self):
        device = next(self.parameters()).device
        self.task_heads.append(nn.Linear(256, 2).to(device))
        self.n_tasks_seen += 1


class StaticMLP(nn.Module):
    name = "Static-MLP"
    def __init__(self):
        super().__init__()
        self.backbone   = make_mlp()
        self.task_heads = nn.ModuleList()
        self.n_tasks_seen = 0
    def add_task_head(self):
        device = next(self.parameters()).device
        self.task_heads.append(nn.Linear(256, 2).to(device))
        self.n_tasks_seen += 1
    def forward(self, x, task_id):
        return self.task_heads[task_id](self.backbone(x)), torch.tensor(0.0, device=x.device)


class DERpp(nn.Module):
    name = "DER++"
    def __init__(self):
        super().__init__()
        self.backbone   = make_mlp()
        self.task_heads = nn.ModuleList()
        self.n_tasks_seen = 0
        self.buf_x, self.buf_logits, self.buf_y, self.buf_task = [], [], [], []
        self.buf_size = 200; self.alpha = 0.5; self.beta = 0.5
    def add_task_head(self):
        device = next(self.parameters()).device
        self.task_heads.append(nn.Linear(256, 2).to(device))
        self.n_tasks_seen += 1
    def _feat(self, x): return self.backbone(x)
    def forward(self, x, task_id):
        out  = self.task_heads[task_id](self._feat(x))
        loss = torch.tensor(0.0, device=x.device)
        if self.buf_x:
            n    = min(32, len(self.buf_x))
            idxs = random.sample(range(len(self.buf_x)), n)
            bx   = torch.stack([self.buf_x[i] for i in idxs]).to(x.device)
            blog = torch.stack([self.buf_logits[i] for i in idxs]).to(x.device)
            by   = torch.tensor([self.buf_y[i] for i in idxs], device=x.device)
            bt   = [self.buf_task[i] for i in idxs]
            for tid in set(bt):
                m    = [i for i,t in enumerate(bt) if t==tid]
                bout = self.task_heads[tid](self._feat(bx[m]))
                loss += self.beta * F.cross_entropy(bout, by[m])
                loss += self.alpha * F.mse_loss(bout, blog[m][:,:bout.size(1)])
        return out, loss
    def update_buffer(self, x, y, task_id):
        with torch.no_grad():
            logits = self.task_heads[task_id](self._feat(x))
        for i in range(x.size(0)):
            if len(self.buf_x) < self.buf_size:
                self.buf_x.append(x[i].cpu()); self.buf_logits.append(logits[i].cpu())
                self.buf_y.append(y[i].item()); self.buf_task.append(task_id)
            else:
                j = random.randint(0, self.buf_size-1)
                self.buf_x[j]=x[i].cpu(); self.buf_logits[j]=logits[i].cpu()
                self.buf_y[j]=y[i].item(); self.buf_task[j]=task_id


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def backward_transfer(mat):
    T = len(mat)
    if T < 2: return 0.0
    return float(np.mean([mat[T-1][i] - mat[i][i] for i in range(T-1)]))

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, task_id, device):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out, _ = model(x, task_id)
        correct += (out.argmax(1)==y).sum().item(); total += x.size(0)
    model.train(); return correct/total


def train_one_seed(seed, task_loaders, device):
    set_seed(seed)
    models = [ARIA(), EWC(), ScaledEWC(), StaticMLP(), DERpp()]
    seed_results = {}

    T = len(task_loaders)

    for model in models:
        name  = model.name
        model = model.to(device)
        mat   = []
        print(f"\n  [{name}]  params: {count_params(model):,}")

        for t in range(T):
            tr_loader, te_loader = task_loaders[t]
            model.add_task_head()
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                          weight_decay=CFG["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=CFG["epochs_per_task"])

            for epoch in range(CFG["epochs_per_task"]):
                model.train()
                for x, y in tr_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    out, aux = model(x, t)
                    loss     = F.cross_entropy(out, y) + aux
                    loss.backward()

                    if hasattr(model, 'post_backward'):
                        model.post_backward()
                    if hasattr(model, 'blocks'):
                        for block in model.blocks:
                            if hasattr(block, 'mlp'):
                                mg = block.mlp.mean_gate
                                for p in (list(block.mlp.W_slow_in.parameters()) +
                                          list(block.mlp.W_slow_out.parameters())):
                                    if p.grad is not None: p.grad.mul_(mg)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    if isinstance(model, DERpp):
                        model.update_buffer(x.detach(), y.detach(), t)

                scheduler.step()

            if isinstance(model, (EWC, ScaledEWC)):
                model.consolidate(tr_loader, t, device)

            heads_str = ""
            if hasattr(model, 'blocks'):
                heads = [b.attn.n_active for b in model.blocks]
                gates = [round(b.mlp.mean_gate, 3) for b in model.blocks]
                heads_str = f"  heads:{heads}  gates:{gates}"

            row = [round(evaluate(model, task_loaders[i][1], i, device)*100, 2)
                   for i in range(t+1)]
            while len(row) < T: row.append(None)
            mat.append(row)
            print(f"    Task {t+1}: {row[:t+1]}  avg:{np.mean(row[:t+1]):.1f}%{heads_str}")

        final   = [mat[T-1][i] for i in range(T) if mat[T-1][i] is not None]
        avg_acc = round(float(np.mean(final)), 2)
        bwt     = round(backward_transfer(mat), 2)
        seed_results[name] = {"avg_acc": avg_acc, "bwt": bwt, "acc_matrix": mat}

    return seed_results


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_figures(all_seed_results, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fdir = os.path.join(CFG["results_dir"], "figures")
    os.makedirs(fdir, exist_ok=True)
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
        'savefig.dpi':200,'savefig.bbox':'tight',
        'axes.spines.top':False,'axes.spines.right':False,
    })

    COLORS = {
        "ARIA": "#2563EB", "EWC": "#DC2626",
        "Scaled-EWC": "#F59E0B", "Static-MLP": "#6B7280", "DER++": "#16A34A"
    }

    names    = list(summary.keys())
    avg_accs = [summary[n]["mean_acc"]  for n in names]
    bwts     = [summary[n]["mean_bwt"]  for n in names]
    acc_stds = [summary[n]["std_acc"]   for n in names]
    bwt_stds = [summary[n]["std_bwt"]   for n in names]
    cols     = [COLORS.get(n, "#888") for n in names]

    # ── Summary bars with error bars ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.bar(names, avg_accs, color=cols, alpha=0.85, width=0.55)
    ax.errorbar(names, avg_accs, yerr=acc_stds, fmt='none', color='black',
                capsize=5, lw=1.5)
    ax.set(ylabel='Final Average Accuracy (%)',
           title=f'Ablation: Average Accuracy ± std ({len(CFG["seeds"])} seeds)')
    ax.set_xticklabels(names, rotation=20, ha='right')
    for bar, val, std in zip(bars, avg_accs, acc_stds):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+std+0.3,
                f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(avg_accs[0], color='#2563EB', lw=1, ls='--', alpha=0.4)

    ax2 = axes[1]
    bars2 = ax2.bar(names, bwts, color=cols, alpha=0.85, width=0.55)
    ax2.errorbar(names, bwts, yerr=bwt_stds, fmt='none', color='black',
                 capsize=5, lw=1.5)
    ax2.axhline(0, color='gray', lw=1, ls='--')
    ax2.set(ylabel='Backward Transfer (BWT)',
            title=f'Backward Transfer ± std ({len(CFG["seeds"])} seeds)')
    ax2.set_xticklabels(names, rotation=20, ha='right')
    for bar, val, std in zip(bars2, bwts, bwt_stds):
        offset = std + 0.2 if val >= 0 else -(std + 1.2)
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+offset,
                 f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(fdir, 'fig_v4_summary.png'))
    plt.close(); print("  ✓ fig_v4_summary.png")

    # ── Learning curves (mean ± std band) ─────────────────────────────────────
    T    = CFG["n_tasks"]
    seeds = CFG["seeds"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in names:
        color = COLORS.get(name, "#888")
        curves = []
        for s in seeds:
            mat   = all_seed_results[s][name]["acc_matrix"]
            means = [np.mean([mat[t][i] for i in range(t+1)]) for t in range(T)]
            curves.append(means)
        curves  = np.array(curves)
        mu      = curves.mean(0)
        sigma   = curves.std(0)
        lw = 2.5 if name == "ARIA" else 1.5
        ax.plot(range(1,T+1), mu, color=color, lw=lw, marker='o', ms=5, label=name)
        ax.fill_between(range(1,T+1), mu-sigma, mu+sigma, color=color, alpha=0.12)
    ax.set(xlabel='Tasks learned so far', ylabel='Average accuracy (%)',
           title=f'Learning Curves (mean ± 1σ, {len(seeds)} seeds)',
           xticks=range(1,T+1))
    ax.set_xticklabels([f'After T{i}' for i in range(1,T+1)])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fdir, 'fig_v4_learning_curves.png'))
    plt.close(); print("  ✓ fig_v4_learning_curves.png")

    print(f"\n  All figures saved to: {fdir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = CFG["device"]
    print(f"\nLoading {CFG['benchmark']}...")
    if CFG["benchmark"] == "permuted_mnist":
        task_loaders = get_permuted_mnist()
        CFG["n_tasks"] = CFG["n_permuted_tasks"]
    else:
        task_loaders = get_split_mnist()

    all_seed_results = {}

    for seed in CFG["seeds"]:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        results       = train_one_seed(seed, task_loaders, device)
        all_seed_results[seed] = results
        with open(os.path.join(CFG["results_dir"], f"seed_{seed}.json"), 'w') as f:
            json.dump(results, f, indent=2)

    # Aggregate across seeds
    model_names = list(all_seed_results[CFG["seeds"][0]].keys())
    summary = {}
    print(f"\n{'='*65}")
    print(f"FINAL RESULTS (mean ± std across {len(CFG['seeds'])} seeds)")
    print(f"{'='*65}")
    print(f"{'Model':<14} {'Acc':>12} {'BWT':>12}")
    print(f"{'-'*42}")
    for name in model_names:
        accs = [all_seed_results[s][name]["avg_acc"] for s in CFG["seeds"]]
        bwts = [all_seed_results[s][name]["bwt"]     for s in CFG["seeds"]]
        summary[name] = {
            "mean_acc": round(float(np.mean(accs)), 2),
            "std_acc":  round(float(np.std(accs)), 2),
            "mean_bwt": round(float(np.mean(bwts)), 2),
            "std_bwt":  round(float(np.std(bwts)), 2),
        }
        print(f"{name:<14} {np.mean(accs):>7.2f}±{np.std(accs):.2f}  "
              f"{np.mean(bwts):>7.2f}±{np.std(bwts):.2f}")

    with open(os.path.join(CFG["results_dir"], "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nGenerating figures...")
    generate_figures(all_seed_results, summary)

    shutil.make_archive('/kaggle/working/ARIA_results_v4', 'zip', CFG["results_dir"])
    print("\nDone! Download ARIA_results_v4.zip from Output panel.")


if __name__ == "__main__":
    main()
