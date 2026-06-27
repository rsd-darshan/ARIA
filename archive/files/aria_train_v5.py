"""
ARIA v5 Training Script — Rigorous Multi-Seed Evaluation
=========================================================
Uses aria_v2.py. Key improvements over v4:

  [CRITICAL BUG FIX] Gradient dampening direction:
    All prior scripts used p.grad.mul_(mg) where mg = mean_gate = π.
    Correct is mul_(1 - mg): when plasticity is HIGH, slow path should
    be LESS updated, not more.  This single fix may be why PG-MLP ablation
    consistently outperformed ARIA-Full.

  [BUG FIX] Plasticity loss warmup:
    Specialization penalty is now gated off for the first warmup_steps so
    it doesn't overwhelm the task loss during initialization.

  [NEW] Slow-Pathway Consolidation (SPC):
    EWC applied only to slow-pathway weights. Fast pathway stays unconstrained.
    This is the key novel contribution over standard EWC.

  [RIGOR] Parameter-matched baselines:
    StaticMLP hidden_dim is scaled so total params ≈ ARIA params.
    Every comparison table includes the param count column.

  [RIGOR] 5-seed evaluation with mean ± std reported.

  [RIGOR] Clean 4-model ablation:
    ARIA+SPC vs ARIA-noSPC → SPC contribution
    ARIA-noSPC vs EWC       → morphogenesis + PG-MLP + AGV contribution
    EWC vs StaticMLP        → Fisher regularization contribution

Run:
  python aria_train_v5.py                    # Split-MNIST, 5 seeds
  python aria_train_v5.py --dataset cifar10  # Split-CIFAR-10
  python aria_train_v5.py --seeds 3 --epochs 20   # faster run
"""

import os, sys, json, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import aria_v2 from same directory
sys.path.insert(0, os.path.dirname(__file__))
from aria_v2 import (
    ARIA, ARIAConfig,
    StaticMLP, EWCWrapper, DERPlusPlus,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="mnist",  choices=["mnist", "cifar10"])
    p.add_argument("--n_tasks",    type=int, default=5)
    p.add_argument("--seeds",      type=int, default=5)
    p.add_argument("--epochs",     type=int, default=25)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--d_model",    type=int, default=256)
    p.add_argument("--ewc_lambda", type=float, default=5000.0)
    p.add_argument("--spc_lambda", type=float, default=5000.0)
    p.add_argument("--results_dir", default="./results_v5")
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--device",      default="auto")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, ds, c0, c1):
        self.idx = [i for i, (_, y) in enumerate(ds) if y == c0 or y == c1]
        self.ds  = ds
        self.c0  = c0
    def __len__(self):  return len(self.idx)
    def __getitem__(self, i):
        x, y = self.ds[self.idx[i]]
        return x, int(y != self.c0)


def load_split_mnist(n_tasks, data_dir, batch_size):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    tr_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    te_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    loaders = []
    for t in range(n_tasks):
        c0, c1 = t * 2, t * 2 + 1
        loaders.append((
            DataLoader(_PairDataset(tr_ds, c0, c1), batch_size=batch_size,
                       shuffle=True, num_workers=0),
            DataLoader(_PairDataset(te_ds, c0, c1), batch_size=256,
                       shuffle=False, num_workers=0),
        ))
    return loaders, 784, 2


def load_split_cifar10(n_tasks, data_dir, batch_size):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    tr_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=tf)
    te_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    loaders = []
    for t in range(n_tasks):
        c0, c1 = t * 2, t * 2 + 1
        loaders.append((
            DataLoader(_PairDataset(tr_ds, c0, c1), batch_size=batch_size,
                       shuffle=True, num_workers=0),
            DataLoader(_PairDataset(te_ds, c0, c1), batch_size=256,
                       shuffle=False, num_workers=0),
        ))
    return loaders, 3072, 2


# ─────────────────────────────────────────────────────────────────────────────
# Model factory — parameter-matched baselines
# ─────────────────────────────────────────────────────────────────────────────

def build_models(input_dim, n_classes, args):
    """
    All baseline hidden dims are solved so their body ≈ ARIA body params.
    We report exact counts in the table.
    """
    cfg = ARIAConfig(
        input_dim    = input_dim,
        n_classes    = n_classes,
        d_model      = args.d_model,
        n_layers     = 4,
        n_heads_init = 4,
        n_heads_max  = 8,
        genome_dim   = 32,
        spc_lambda   = args.spc_lambda,
    )
    aria_spc   = ARIA(cfg)
    aria_nospc = ARIA(cfg)

    # Approximate param-matched baseline hidden dim
    # StaticMLP body ≈ input_dim*H + 3*H^2  (4-layer)
    target = aria_spc.n_params()
    a, b_c = 3, input_dim
    discriminant = b_c**2 + 4 * a * target
    h_matched = max(int((-b_c + math.sqrt(discriminant)) / (2 * a)), 64)

    static    = StaticMLP(input_dim=input_dim, hidden_dim=h_matched,
                          n_layers=4, dropout=0.1)
    ewc_base  = StaticMLP(input_dim=input_dim, hidden_dim=h_matched,
                          n_layers=4, dropout=0.1)
    der_base  = StaticMLP(input_dim=input_dim, hidden_dim=h_matched,
                          n_layers=4, dropout=0.1)
    ewc_model = EWCWrapper(ewc_base, ewc_lambda=args.ewc_lambda)
    der_model = DERPlusPlus(der_base)

    return {
        "ARIA+SPC":   aria_spc,
        "ARIA-noSPC": aria_nospc,
        "EWC":        ewc_model,
        "DER++":      der_model,
        "StaticMLP":  static,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def bwt(mat):
    T = len(mat)
    if T < 2: return 0.0
    return float(np.mean([mat[T-1][i] - mat[i][i] for i in range(T-1)]))

def fwt(mat):
    vals = [mat[i][i+1] for i in range(len(mat)-1) if mat[i][i+1] is not None]
    return float(np.mean(vals)) if vals else 0.0

def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, task_id, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out, _ = model(x, task_id)
        correct += (out.argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, task_id, device, model_name):
    model.train()
    tl = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out, aux = model(x, task_id)
        loss = F.cross_entropy(out, y) + aux

        if model_name == "EWC":
            loss = loss + model.ewc_loss(device)
        elif model_name == "DER++":
            loss = loss + model.der_loss(device, task_id)
            model.update_buffer(x, y, out.detach(), device)

        loss.backward()

        # FIXED gradient dampening: mul_(1 - π̄) for slow pathway
        if hasattr(model, "dampen_slow_gradients"):
            model.dampen_slow_gradients()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tl      += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total   += y.size(0)
    return tl / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Full continual learning run for one model + one seed
# ─────────────────────────────────────────────────────────────────────────────

def run_model(name, model, task_loaders, n_classes, device, args):
    T          = len(task_loaders)
    acc_matrix = []
    aria_log   = []

    for t, (tr_loader, te_loader) in enumerate(task_loaders):
        # Add task head
        if hasattr(model, "model"):   # EWC / DER++ wrapper
            model.model.add_task_head(device, n_classes)
        else:
            model.add_task_head(device, n_classes)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        for epoch in range(args.epochs):
            train_epoch(model, tr_loader, opt, t, device, name)
            sch.step()

            if isinstance(model, ARIA) and (epoch + 1) % 5 == 0:
                st = model.architecture_state()
                aria_log.append({"task": t+1, "epoch": epoch+1, **st})

        # Post-task consolidation
        if name == "EWC":
            model.consolidate(tr_loader, t, device)
        elif name == "ARIA+SPC":
            model.consolidate_slow(tr_loader, t, device)
        # ARIA-noSPC: deliberately no consolidation (ablation)

        row = [None] * T
        for i in range(t + 1):
            row[i] = round(evaluate(model, task_loaders[i][1], i, device) * 100, 2)
        acc_matrix.append(row)

        avg = np.mean([v for v in row if v is not None])
        print(f"    T{t+1}/{T}  avg={avg:.2f}%  {row[:t+1]}")

    final   = [v for v in acc_matrix[-1] if v is not None]
    avg_acc = float(np.mean(final))
    return {
        "avg_acc":    round(avg_acc, 2),
        "bwt":        round(bwt(acc_matrix), 2),
        "fwt":        round(fwt(acc_matrix), 2),
        "n_params":   n_params(model),
        "acc_matrix": acc_matrix,
        "aria_log":   aria_log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed orchestrator
# ─────────────────────────────────────────────────────────────────────────────

ALL_SEEDS = [42, 123, 999, 7, 2024]


def main():
    args = parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():     device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                              device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("ARIA v2 — Rigorous Continual Learning Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.dataset}  Tasks: {args.n_tasks}  "
          f"Seeds: {args.seeds}  Epochs/task: {args.epochs}")
    print(f"Device: {device}\n")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.data_dir,    exist_ok=True)

    load_data = load_split_mnist if args.dataset == "mnist" else load_split_cifar10
    n_seeds   = min(args.seeds, len(ALL_SEEDS))

    all_results = {}   # name → list of per-seed dicts

    for si in range(n_seeds):
        seed = ALL_SEEDS[si]
        torch.manual_seed(seed)
        np.random.seed(seed)

        task_loaders, input_dim, n_classes = load_data(
            args.n_tasks, args.data_dir, args.batch_size)

        models = build_models(input_dim, n_classes, args)

        print(f"\n{'─'*60}")
        print(f"Seed {seed} ({si+1}/{n_seeds})")
        print(f"  Param counts:  " +
              "  ".join(f"{k}={n_params(m):,}" for k, m in models.items()))
        print(f"{'─'*60}")

        seed_results = {}
        for name, model in models.items():
            model = model.to(device)
            print(f"\n  [{name}]")
            res = run_model(name, model, task_loaders, n_classes, device, args)
            seed_results[name] = res
            if name not in all_results:
                all_results[name] = []
            all_results[name].append(res)

        # Save per-seed snapshot
        snap = {
            name: {k: v for k, v in r.items() if k != "aria_log"}
            for name, r in seed_results.items()
        }
        with open(os.path.join(args.results_dir, f"seed_{seed}.json"), "w") as f:
            json.dump(snap, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"{'Model':<15} {'Acc':>8} {'±':>5} {'BWT':>8} {'±':>5} {'FWT':>8} {'Params':>10}")
    print(f"{'─'*72}")

    summary = {}
    for name, results in all_results.items():
        accs = [r["avg_acc"] for r in results]
        bwts = [r["bwt"]     for r in results]
        fwts = [r["fwt"]     for r in results]
        np_  = results[0]["n_params"]

        ma = round(float(np.mean(accs)), 2)
        sa = round(float(np.std(accs)),  2)
        mb = round(float(np.mean(bwts)), 2)
        sb = round(float(np.std(bwts)),  2)
        mf = round(float(np.mean(fwts)), 2)

        print(f"{name:<15} {ma:>8.2f} {sa:>5.2f} {mb:>8.2f} {sb:>5.2f} {mf:>8.2f} {np_:>10,}")

        summary[name] = {
            "mean_acc": ma, "std_acc":  sa,
            "mean_bwt": mb, "std_bwt":  sb,
            "mean_fwt": mf,
            "n_params": np_,
            "per_seed": [{"acc": r["avg_acc"], "bwt": r["bwt"]} for r in results],
        }

    print(f"{'='*72}")

    # ── Ablation story ────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Component Contributions (Δ accuracy)")
    print(f"{'─'*55}")
    pairs = [
        ("ARIA+SPC",   "ARIA-noSPC", "Slow-Pathway Consolidation (SPC)"),
        ("ARIA-noSPC", "EWC",        "Morphogenesis + PG-MLP + AGV"),
        ("EWC",        "StaticMLP",  "EWC regularization baseline"),
    ]
    for a, b, label in pairs:
        if a in summary and b in summary:
            d = summary[a]["mean_acc"] - summary[b]["mean_acc"]
            print(f"  {label:<42}  {'+' if d >= 0 else ''}{d:.2f}%")
    print(f"{'─'*55}")

    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults → {args.results_dir}/summary.json")

    # ── Figures ───────────────────────────────────────────────────────────────
    try:
        _plot_figures(summary, all_results, args)
    except Exception as e:
        print(f"Figure generation skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _plot_figures(summary, all_results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fdir = os.path.join(args.results_dir, "figures")
    os.makedirs(fdir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })

    COLORS = {
        "ARIA+SPC":   "#2563EB",
        "ARIA-noSPC": "#7C3AED",
        "EWC":        "#DC2626",
        "DER++":      "#D97706",
        "StaticMLP":  "#6B7280",
    }
    T = args.n_tasks

    # Fig 1 — Average accuracy + Task 1 forgetting curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, results in all_results.items():
        matrices = [r["acc_matrix"] for r in results]
        means, stds = [], []
        for t in range(T):
            vals = [np.mean([v for v in m[t] if v]) for m in matrices]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        xs = range(1, T + 1)
        c  = COLORS.get(name, "#374151")
        ax1.plot(xs, means, "o-", color=c, lw=2.2, markersize=7, label=name)
        ax1.fill_between(xs,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         color=c, alpha=0.12)
    ax1.set(xlabel="Tasks seen", ylabel="Avg accuracy (%)",
            title="Average Accuracy ± std", ylim=(50, 103))
    ax1.set_xticks(list(range(1, T + 1)))
    ax1.legend(fontsize=9)

    for name, results in all_results.items():
        matrices = [r["acc_matrix"] for r in results]
        means, stds = [], []
        for t in range(T):
            vals = [m[t][0] for m in matrices if m[t][0] is not None]
            means.append(float(np.mean(vals)) if vals else None)
            stds.append(float(np.std(vals)) if vals else 0)
        valid = [(t+1, m, s) for t, (m, s) in enumerate(zip(means, stds)) if m is not None]
        xs2 = [v[0] for v in valid]; ms = [v[1] for v in valid]; ss = [v[2] for v in valid]
        c   = COLORS.get(name, "#374151")
        ax2.plot(xs2, ms, "o--", color=c, lw=2.2, markersize=7, label=name)
        ax2.fill_between(xs2,
                         [m - s for m, s in zip(ms, ss)],
                         [m + s for m, s in zip(ms, ss)],
                         color=c, alpha=0.12)
    ax2.set(xlabel="Tasks seen", ylabel="Accuracy (%) on Task 1",
            title="Catastrophic Forgetting: Task 1", ylim=(50, 103))
    ax2.set_xticks(list(range(1, T + 1)))
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{fdir}/fig1_accuracy_curves.png"); plt.close()
    print("  ✓ fig1_accuracy_curves.png")

    # Fig 2 — Summary bars with error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    names = list(summary.keys())
    accs  = [summary[n]["mean_acc"] for n in names]
    saccs = [summary[n]["std_acc"]  for n in names]
    bwts  = [summary[n]["mean_bwt"] for n in names]
    sbwts = [summary[n]["std_bwt"]  for n in names]
    cols  = [COLORS.get(n, "#374151") for n in names]
    xpos  = range(len(names))

    bars = ax1.bar(xpos, accs, yerr=saccs, capsize=4, color=cols, alpha=0.85,
                   width=0.55, error_kw={"elinewidth": 1.5})
    ax1.set_xticks(list(xpos)); ax1.set_xticklabels(names, rotation=15, fontsize=9)
    ax1.set(ylabel="Final Avg Accuracy (%)", title="Accuracy (mean ± std, 5 seeds)",
            ylim=(max(50, min(accs) - 8), 103))
    for bar, v in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(xpos, bwts, yerr=sbwts, capsize=4, color=cols, alpha=0.85,
                    width=0.55, error_kw={"elinewidth": 1.5})
    ax2.axhline(0, color="#374151", lw=1, ls="--")
    ax2.set_xticks(list(xpos)); ax2.set_xticklabels(names, rotation=15, fontsize=9)
    ax2.set(ylabel="BWT", title="Backward Transfer (↑ better, closer to 0)")
    plt.tight_layout()
    plt.savefig(f"{fdir}/fig2_summary_bars.png"); plt.close()
    print("  ✓ fig2_summary_bars.png")

    # Fig 3 — Ablation waterfall chart
    abl_order = ["StaticMLP", "EWC", "ARIA-noSPC", "ARIA+SPC"]
    abl_labels = [
        "Static MLP\n(no regularization)",
        "+ EWC\n(Fisher on all params)",
        "+ Morph. Attn\n+ PG-MLP + AGV",
        "+ SPC\n(Fisher on slow path)",
    ]
    abl_accs  = [summary[n]["mean_acc"] for n in abl_order if n in summary]
    abl_stds  = [summary[n]["std_acc"]  for n in abl_order if n in summary]
    abl_labs  = [abl_labels[i] for i, n in enumerate(abl_order) if n in summary]
    abl_cols  = [COLORS.get(n, "#374151") for n in abl_order if n in summary]
    if len(abl_accs) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        xp = range(len(abl_accs))
        bars = ax.bar(xp, abl_accs, yerr=abl_stds, capsize=4, color=abl_cols,
                      alpha=0.85, width=0.55, error_kw={"elinewidth": 1.5})
        ax.set_xticks(list(xp)); ax.set_xticklabels(abl_labs, fontsize=9)
        ax.set(ylabel="Final Avg Accuracy (%)",
               title="Ablation: Contribution of Each Component",
               ylim=(max(50, min(abl_accs) - 6), 103))
        for bar, v in zip(bars, abl_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{fdir}/fig3_ablation.png"); plt.close()
        print("  ✓ fig3_ablation.png")

    # Fig 4 — ARIA head morphogenesis + plasticity gates (seed 0)
    aria_res = all_results.get("ARIA+SPC", [])
    if aria_res and aria_res[0].get("aria_log"):
        log = aria_res[0]["aria_log"]
        n_layers  = len(log[0]["head_counts"])
        cumepochs = [e["epoch"] + (e["task"] - 1) * args.epochs for e in log]
        heads     = np.array([e["head_counts"] for e in log])
        gates     = np.array([e["gate_means"]  for e in log])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
        cmap = plt.cm.Blues(np.linspace(0.4, 0.9, n_layers))

        for l in range(n_layers):
            ax1.plot(cumepochs, heads[:, l], color=cmap[l], lw=2, label=f"L{l+1}")
        for t in range(1, T):
            ax1.axvline(t * args.epochs, color="#9CA3AF", lw=1.2, ls="--", alpha=0.7)
            ax1.text(t * args.epochs + 0.3, ax1.get_ylim()[0] + 0.05,
                     f"T{t+1}", fontsize=9, color="#6B7280")
        ax1.set(ylabel="Active heads", title="Head Morphogenesis Over Training")
        ax1.legend(fontsize=9, ncol=2)

        for l in range(n_layers):
            ax2.plot(cumepochs, gates[:, l], color=cmap[l], lw=2, label=f"L{l+1}")
        ax2.axhline(0.5, color="#DC2626", lw=1.3, ls="--", alpha=0.7,
                    label="Uniform (0.5)")
        ax2.set(xlabel="Cumulative epoch",
                ylabel="Mean gate π̄",
                title="Plasticity Gate: π→1 (fast/new) vs π→0 (slow/consolidated)",
                ylim=(0, 1))
        ax2.legend(fontsize=9, ncol=2)
        plt.tight_layout()
        plt.savefig(f"{fdir}/fig4_morphogenesis.png"); plt.close()
        print("  ✓ fig4_morphogenesis.png")

    print(f"Figures saved → {fdir}/")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
