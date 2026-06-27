"""
aria_ablation.py  —  ARIA Ablation Study
=========================================
Trains 5 model variants on Split-MNIST:
  1. ARIA-Full       : all components (baseline)
  2. ARIA-noMA       : standard fixed attention (no morphogenesis)
  3. ARIA-noPG       : standard MLP (no plasticity gates)
  4. ARIA-noCBA      : uniform budget b_l=1.0 (no cognitive budget)
  5. ARIA-noAGV      : zero genome conditioning (no genome vector)

Each variant isolates the contribution of one component.
Saves ablation table + fig_ablation.png
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
    "split_threshold":   0.60,
    "merge_threshold":   0.97,
    "morph_interval":    50,
    "plasticity_lambda": 0.05,
    "budget_beta":       0.001,
    "genome_gamma":      0.0001,
    "epochs_per_task":   25,
    "lr":                3e-4,
    "weight_decay":      1e-4,
    "results_dir":       "./results",
    "seed":              42,
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CFG["results_dir"], exist_ok=True)
os.makedirs(CFG["data_dir"],    exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
print(f"Device: {CFG['device']}")


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
# Shared components
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


# ─────────────────────────────────────────────────────────────────────────────
# ARIA-Full (all components)
# ─────────────────────────────────────────────────────────────────────────────

class MorphogenicAttention(nn.Module):
    def __init__(self):
        super().__init__()
        D, H = CFG["hidden_dim"], CFG["n_heads_max"]
        d_h  = D // H
        self.d_h = d_h; self.H = H
        self.split_τ = CFG["split_threshold"]
        self.merge_τ = CFG["merge_threshold"]
        self.W_Q       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_K       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_V       = nn.Parameter(torch.randn(H, D, d_h) * 0.02)
        self.W_O       = nn.Parameter(torch.randn(H, d_h, D) * 0.02)
        self.viability = nn.Parameter(torch.zeros(H) - 0.5)
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
            if torch.sigmoid(self.viability[i]).item() > self.split_τ:
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
            if cos > self.merge_τ and self.n_active > 2:
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
        f2  = -(p*(p+1e-8).log()).sum(dim=-1).mean().unsqueeze(0) / math.log(784)
        f3  = (x_raw.max(dim=-1).values - x_raw.min(dim=-1).values).mean().unsqueeze(0)
        b   = self.net(torch.cat([f1,f2,f3]).to(dev))
        return b, self.beta * b.mean()


class ARIABlock_Full(nn.Module):
    def __init__(self, idx):
        super().__init__()
        D = CFG["hidden_dim"]
        self.idx = idx
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.attn = MorphogenicAttention()
        self.mlp  = PlasticityGatedMLP()

    def forward(self, x, genome, budget):
        z     = self.attn(self.ln1(x), genome) + x
        h, p  = self.mlp(self.ln2(z))
        b     = budget[self.idx]
        return b*(z+h) + (1-b)*x, p


class ARIA_Full(nn.Module):
    """All 4 components active."""
    name = "ARIA-Full"

    def __init__(self):
        super().__init__()
        D = CFG["hidden_dim"]
        self.input_proj   = nn.Linear(CFG["input_dim"], D)
        self.genome       = ArchitectureGenome()
        self.blocks       = nn.ModuleList([ARIABlock_Full(i) for i in range(CFG["n_layers"])])
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
        budgets, b_loss = self.budget_alloc(x)
        budgets = budgets.to(device)
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


# ─────────────────────────────────────────────────────────────────────────────
# ARIA-noMA: standard fixed MHA, no morphogenesis
# ─────────────────────────────────────────────────────────────────────────────

class StandardMHA(nn.Module):
    """Fixed multi-head attention — no viability, no split/merge."""
    def __init__(self):
        super().__init__()
        D = CFG["hidden_dim"]
        H = CFG["n_heads_init"]
        self.mha = nn.MultiheadAttention(D, H, dropout=CFG["dropout"], batch_first=True)
        self.proj_cond = nn.Linear(D, D)

    def forward(self, x, genome):
        out, _ = self.mha(x, x, x)
        # Still condition on genome signal for fair comparison
        return out + torch.tanh(self.proj_cond(genome["cond_signal"].to(x.device)))


class ARIABlock_noMA(nn.Module):
    def __init__(self, idx):
        super().__init__()
        D = CFG["hidden_dim"]
        self.idx = idx
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.attn = StandardMHA()
        self.mlp  = PlasticityGatedMLP()

    def forward(self, x, genome, budget):
        z    = self.attn(self.ln1(x), genome) + x
        h, p = self.mlp(self.ln2(z))
        b    = budget[self.idx]
        return b*(z+h) + (1-b)*x, p


class ARIA_noMA(nn.Module):
    """No Morphogenic Attention — standard MHA instead."""
    name = "ARIA-noMA"

    def __init__(self):
        super().__init__()
        D = CFG["hidden_dim"]
        self.input_proj   = nn.Linear(CFG["input_dim"], D)
        self.genome       = ArchitectureGenome()
        self.blocks       = nn.ModuleList([ARIABlock_noMA(i) for i in range(CFG["n_layers"])])
        self.budget_alloc = CognitiveBudgetAllocator()
        self.ln_f         = nn.LayerNorm(D)
        self.task_heads   = nn.ModuleList()
        self.n_tasks_seen = 0

    def add_task_head(self):
        device = next(self.parameters()).device
        self.task_heads.append(nn.Linear(CFG["hidden_dim"], 2).to(device))
        self.n_tasks_seen += 1

    def forward(self, x, task_id):
        device  = x.device
        budgets, b_loss = self.budget_alloc(x)
        budgets = budgets.to(device)
        h       = F.gelu(self.input_proj(x)).unsqueeze(1)
        genome  = self.genome.decode(device)
        total_p = torch.tensor(0.0, device=device)
        for i, block in enumerate(self.blocks):
            h, p = block(h, genome, budgets)
            total_p = total_p + p
        h   = self.ln_f(h).squeeze(1)
        out = self.task_heads[task_id](h)
        return out, total_p + b_loss + CFG["genome_gamma"] * self.genome.reg_loss()


# ─────────────────────────────────────────────────────────────────────────────
# ARIA-noPG: standard MLP, no plasticity gates
# ─────────────────────────────────────────────────────────────────────────────

class StandardMLP_Block(nn.Module):
    def __init__(self):
        super().__init__()
        D, d_ff = CFG["hidden_dim"], CFG["hidden_dim"] * 2
        self.W1     = nn.Linear(D, d_ff)
        self.W2     = nn.Linear(d_ff, D)
        self.dropout = nn.Dropout(CFG["dropout"])
        self.mean_gate = 0.5  # constant, for compatibility

    def forward(self, x):
        return self.dropout(self.W2(F.gelu(self.W1(x)))), torch.tensor(0.0, device=x.device)


class ARIABlock_noPG(nn.Module):
    def __init__(self, idx):
        super().__init__()
        D = CFG["hidden_dim"]
        self.idx = idx
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.attn = MorphogenicAttention()
        self.mlp  = StandardMLP_Block()

    def forward(self, x, genome, budget):
        z    = self.attn(self.ln1(x), genome) + x
        h, p = self.mlp(self.ln2(z))
        b    = budget[self.idx]
        return b*(z+h) + (1-b)*x, p


class ARIA_noPG(nn.Module):
    """No Plasticity-Gated MLP — standard MLP instead."""
    name = "ARIA-noPG"

    def __init__(self):
        super().__init__()
        D = CFG["hidden_dim"]
        self.input_proj   = nn.Linear(CFG["input_dim"], D)
        self.genome       = ArchitectureGenome()
        self.blocks       = nn.ModuleList([ARIABlock_noPG(i) for i in range(CFG["n_layers"])])
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
        budgets, b_loss = self.budget_alloc(x)
        budgets = budgets.to(device)
        h       = F.gelu(self.input_proj(x)).unsqueeze(1)
        genome  = self.genome.decode(device)
        total_p = torch.tensor(0.0, device=device)
        for i, block in enumerate(self.blocks):
            h, p = block(h, genome, budgets)
            total_p = total_p + p
        h   = self.ln_f(h).squeeze(1)
        out = self.task_heads[task_id](h)
        if self.training:
            self.morph_step = getattr(self, 'morph_step', 0) + 1
            if self.morph_step % CFG["morph_interval"] == 0:
                for block in self.blocks: block.attn.morphogenesis()
        return out, total_p + b_loss + CFG["genome_gamma"] * self.genome.reg_loss()


# ─────────────────────────────────────────────────────────────────────────────
# ARIA-noCBA: uniform budget b_l = 1.0
# ─────────────────────────────────────────────────────────────────────────────

class UniformBudget(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_raw):
        L   = CFG["n_layers"]
        dev = x_raw.device
        return torch.ones(L, device=dev), torch.tensor(0.0, device=dev)


class ARIABlock_noCBA(nn.Module):
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
        b    = budget[self.idx]   # always 1.0
        return b*(z+h) + (1-b)*x, p


class ARIA_noCBA(nn.Module):
    """No Cognitive Budget Allocator — all layers always active."""
    name = "ARIA-noCBA"

    def __init__(self):
        super().__init__()
        D = CFG["hidden_dim"]
        self.input_proj   = nn.Linear(CFG["input_dim"], D)
        self.genome       = ArchitectureGenome()
        self.blocks       = nn.ModuleList([ARIABlock_noCBA(i) for i in range(CFG["n_layers"])])
        self.budget_alloc = UniformBudget()
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
        return out, total_p + CFG["genome_gamma"] * self.genome.reg_loss()


# ─────────────────────────────────────────────────────────────────────────────
# ARIA-noAGV: zero genome conditioning
# ─────────────────────────────────────────────────────────────────────────────

class ZeroGenome(nn.Module):
    def __init__(self):
        super().__init__()
        D, L = CFG["hidden_dim"], CFG["n_layers"]
        # Still has parameters so optimizer doesn't complain, but outputs zeros
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def decode(self, device):
        D = CFG["hidden_dim"]
        L = CFG["n_layers"]
        return {
            "skip_probs":  torch.zeros(L,  device=device),
            "temperature": torch.tensor(1.0, device=device),
            "cond_signal": torch.zeros(D,  device=device),
        }

    def reg_loss(self):
        return torch.tensor(0.0)


class ARIABlock_noAGV(nn.Module):
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


class ARIA_noAGV(nn.Module):
    """No Architecture Genome Vector — genome is zero (no structural conditioning)."""
    name = "ARIA-noAGV"

    def __init__(self):
        super().__init__()
        D = CFG["hidden_dim"]
        self.input_proj   = nn.Linear(CFG["input_dim"], D)
        self.genome       = ZeroGenome()
        self.blocks       = nn.ModuleList([ARIABlock_noAGV(i) for i in range(CFG["n_layers"])])
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
        budgets, b_loss = self.budget_alloc(x)
        budgets = budgets.to(device)
        h       = F.gelu(self.input_proj(x)).unsqueeze(1)
        genome  = self.genome.decode(device)
        total_p = torch.tensor(0.0, device=device)
        for i, block in enumerate(self.blocks):
            h, p = block(h, genome, budgets)
            total_p = total_p + p
        h   = self.ln_f(h).squeeze(1)
        out = self.task_heads[task_id](h)
        if self.training:
            self.morph_step += 1
            if self.morph_step % CFG["morph_interval"] == 0:
                for block in self.blocks: block.attn.morphogenesis()
        return out, total_p + b_loss


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, task_id, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out, aux  = model(x, task_id)
        loss      = F.cross_entropy(out, y) + aux
        loss.backward()
        # Dampen slow pathway gradients where applicable
        for module in model.modules():
            if isinstance(module, PlasticityGatedMLP):
                mg = module.mean_gate
                for p in (list(module.W_slow_in.parameters()) +
                          list(module.W_slow_out.parameters())):
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def backward_transfer(mat):
    T = len(mat)
    if T < 2: return 0.0
    return float(np.mean([mat[T-1][i] - mat[i][i] for i in range(T-1)]))


def train_model(name, model, task_loaders, device):
    T          = len(task_loaders)
    acc_matrix = []

    print(f"\n{'='*60}")
    print(f"Training: {name}  |  params: {count_params(model):,}")
    print(f"{'='*60}")

    for t in range(T):
        tr_loader, te_loader = task_loaders[t]
        model.add_task_head()
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=CFG["epochs_per_task"])

        print(f"\n  Task {t+1}/{T}")
        for epoch in range(CFG["epochs_per_task"]):
            loss, acc = train_epoch(model, tr_loader, optimizer, t, device)
            scheduler.step()
            if (epoch+1) % 5 == 0:
                print(f"    epoch {epoch+1:2d}/{CFG['epochs_per_task']} "
                      f"| loss {loss:.4f} | acc {acc:.3f}")

        row = [round(evaluate(model, task_loaders[i][1], i, device)*100, 2)
               for i in range(t+1)]
        while len(row) < T: row.append(None)
        acc_matrix.append(row)
        print(f"  → Accs 1..{t+1}: {row[:t+1]}  avg: {np.mean(row[:t+1]):.2f}%")

    return acc_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_ablation_figures(ablation_results):
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

    names    = list(ablation_results.keys())
    avg_accs = [ablation_results[n]["avg_acc"] for n in names]
    bwts     = [ablation_results[n]["bwt"]     for n in names]
    n_params = [ablation_results[n]["n_params"]/1e6 for n in names]

    # Colors: full=blue, ablations=shades of gray/red
    colors = ['#2563EB','#DC2626','#F59E0B','#16A34A','#7C3AED']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Avg Accuracy ──────────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(names, avg_accs, color=colors, alpha=0.85, width=0.6)
    ax.set(ylabel='Final Average Accuracy (%)',
           title='Ablation: Average Accuracy\n(higher is better)')
    ax.set_ylim(min(avg_accs)-3, max(avg_accs)+3)
    ax.set_xticklabels(names, rotation=20, ha='right')
    for bar, val in zip(bars, avg_accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(avg_accs[0], color='#2563EB', lw=1.2, ls='--', alpha=0.5)

    # ── BWT ───────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    bars2 = ax2.bar(names, bwts, color=colors, alpha=0.85, width=0.6)
    ax2.axhline(0, color='#374151', lw=1, ls='--')
    ax2.set(ylabel='Backward Transfer (BWT)',
            title='Ablation: Backward Transfer\n(closer to 0 = less forgetting)')
    ax2.set_xticklabels(names, rotation=20, ha='right')
    for bar, val in zip(bars2, bwts):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+(0.1 if val>=0 else -0.8),
                 f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax2.axhline(bwts[0], color='#2563EB', lw=1.2, ls='--', alpha=0.5)

    # ── Per-task accuracy lines ───────────────────────────────────────────────
    ax3 = axes[2]
    T = CFG["n_tasks"]
    for name, color in zip(names, colors):
        mat   = ablation_results[name]["acc_matrix"]
        means = [np.mean([mat[t][i] for i in range(t+1)]) for t in range(T)]
        lw    = 2.8 if name == "ARIA-Full" else 1.5
        ls    = '-'  if name == "ARIA-Full" else '--'
        ax3.plot(range(1,T+1), means, color=color, lw=lw, ls=ls,
                 marker='o', markersize=5, label=name)
    ax3.set(xlabel='Tasks learned so far',
            ylabel='Average accuracy (%)',
            title='Ablation: Learning Curves',
            ylim=(70, 102))
    ax3.set_xticks(range(1,T+1))
    ax3.set_xticklabels([f'T{i}' for i in range(1,T+1)])
    ax3.legend(fontsize=9)

    plt.tight_layout()
    p = os.path.join(fdir, 'fig_ablation.png')
    plt.savefig(p)
    plt.close()
    print(f"\n  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device       = CFG["device"]
    task_loaders = get_split_mnist()

    variants = [
        ARIA_Full(),
        ARIA_noMA(),
        ARIA_noPG(),
        ARIA_noCBA(),
        ARIA_noAGV(),
    ]

    ablation_results = {}

    for model in variants:
        name  = model.name
        model = model.to(device)
        mat   = train_model(name, model, task_loaders, device)

        T       = len(mat)
        final   = [mat[T-1][i] for i in range(T) if mat[T-1][i] is not None]
        avg_acc = round(float(np.mean(final)), 2)
        bwt     = round(backward_transfer(mat), 2)
        n_p     = count_params(model)

        ablation_results[name] = {
            "avg_acc":    avg_acc,
            "bwt":        bwt,
            "n_params":   n_p,
            "acc_matrix": mat,
        }

        with open(os.path.join(CFG["results_dir"], f"ablation_{name}.json"), "w") as f:
            json.dump({"avg_acc":avg_acc,"bwt":bwt,"acc_matrix":mat}, f, indent=2)

    # Print table
    print(f"\n{'='*65}")
    print(f"ABLATION STUDY RESULTS")
    print(f"{'='*65}")
    print(f"{'Variant':<18} {'Avg Acc':>10} {'BWT':>10} {'Δ Acc':>10} {'Params':>10}")
    print(f"{'-'*65}")
    full_acc = ablation_results["ARIA-Full"]["avg_acc"]
    full_bwt = ablation_results["ARIA-Full"]["bwt"]
    for name, data in ablation_results.items():
        delta = round(data["avg_acc"] - full_acc, 2)
        delta_str = f"{delta:+.2f}"
        print(f"{name:<18} {data['avg_acc']:>10.2f} {data['bwt']:>10.2f} "
              f"{delta_str:>10} {data['n_params']:>10,}")

    # Save full results
    with open(os.path.join(CFG["results_dir"], "ablation_summary.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Generate figure
    generate_ablation_figures(ablation_results)

    import shutil
    shutil.make_archive('/kaggle/working/ARIA_ablation', 'zip', CFG["results_dir"])
    print("\nDone! Download ARIA_ablation.zip from Output panel.")


if __name__ == "__main__":
    main()
