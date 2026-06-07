#!/usr/bin/env python3
"""
Ablation study: systematically removes ARIA components to attribute accuracy gains.

Ablations
---------
ARIA+SPC (full)   — all components
ARIA-noSPC        — no slow-pathway consolidation
ARIA-noMA         — morphogenic attention replaced by standard fixed-head attention
ARIA-noPG         — plasticity gate replaced by standard GELU MLP
ARIA-noAGV        — architecture genome vector disabled (skip_probs=0, temp=1, FiLM=identity)
ARIA-noCBA        — cognitive budget allocator disabled (budgets = 1, no budget loss)

Usage
-----
python scripts/ablation.py --seeds 42 123 999 --epochs 5 --out results/ablation/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import aria
from aria import ARIAConfig, ARIA, get_device, set_seed, get_split_mnist_tasks
from aria.train import evaluate, find_matched_hidden
from aria.metrics import aggregate_seeds
from aria.plot import plot_ablation_waterfall, plot_summary_bars


# ---------------------------------------------------------------------------
# Minimal custom training loop for ablated models
# ---------------------------------------------------------------------------

def _train_custom(model, tasks, device, epochs_per_task, use_spc=False, verbose=True):
    from aria.train import evaluate
    T      = len(tasks)
    opt    = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    matrix = np.zeros((T, T))

    for t, (tr_loader, _) in enumerate(tasks):
        for ep in range(epochs_per_task):
            for x, y in tr_loader:
                x, y    = x.to(device), y.to(device)
                opt.zero_grad()
                out, aux = model(x, t)
                loss     = F.cross_entropy(out, y) + aux
                loss.backward()
                if hasattr(model, "dampen_slow_gradients"):
                    model.dampen_slow_gradients()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        if use_spc and t < T - 1 and hasattr(model, "consolidate_slow"):
            model.consolidate_slow(tr_loader, t, device)

        for j, (_, te_loader) in enumerate(tasks[:t + 1]):
            matrix[t, j] = evaluate(model, te_loader, j, device)

        if verbose:
            row = " ".join(f"{matrix[t,j]:.3f}" for j in range(t + 1))
            print(f"    Task {t+1}: [{row}]")

    return matrix


# ---------------------------------------------------------------------------
# Ablated model constructors
# ---------------------------------------------------------------------------

def make_aria_noMA(cfg: ARIAConfig, device: torch.device) -> ARIA:
    """Standard fixed-head attention (disable morphogenesis)."""
    m = ARIA(cfg)
    # disable morphogenesis by setting an unreachable interval
    m.cfg = ARIAConfig(
        **{k: (999999 if k == "morph_interval" else v)
           for k, v in cfg.__dict__.items()}
    )
    return m.to(device)


def make_aria_noPG(cfg: ARIAConfig, device: torch.device) -> ARIA:
    """Replace PG-MLP with standard MLP (gate always 0.5, plasticity loss 0)."""
    m = ARIA(cfg)
    for block in m.blocks:
        pg = block.mlp
        # Zero out the gate network so π=0.5 always, and lambda=0 kills p_loss
        pg.lambda_ = 0.0
        for p in pg.gate_net.parameters():
            p.data.zero_()
    return m.to(device)


def make_aria_noAGV(cfg: ARIAConfig, device: torch.device) -> ARIA:
    """Disable AGV: fixed genome output (skip=0, temp=1, FiLM=identity)."""
    m = ARIA(cfg)

    class _FixedGenome(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.L = cfg.n_layers
            self.D = cfg.d_model
        def decode(self):
            return {
                "skip_probs":  torch.zeros(self.L),
                "temperature": torch.ones(1).squeeze(),
                "film_scale":  torch.ones(self.D),
                "film_shift":  torch.zeros(self.D),
            }
        def reg_loss(self):
            return torch.zeros(1).squeeze()

    m.genome = _FixedGenome(cfg)
    return m.to(device)


def make_aria_noCBA(cfg: ARIAConfig, device: torch.device) -> ARIA:
    """Disable CBA: uniform budgets = 1, no budget loss."""
    m = ARIA(cfg)
    orig_forward = m.cba.forward

    def fixed_budgets(x):
        return torch.ones(cfg.n_layers, device=x.device), torch.zeros(1, device=x.device).squeeze()

    m.cba.forward = fixed_budgets
    return m.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",  type=int, nargs="+", default=[42, 123, 999])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--out",    type=str, default="results/ablation/")
    p.add_argument("--data",   type=str, default="./data")
    p.add_argument("--quiet",  action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg    = ARIAConfig()
    device = get_device()
    T      = 5

    ablations = {
        "ARIA+SPC":  (lambda: ARIA(cfg),              True),
        "ARIA-noSPC":(lambda: ARIA(cfg),              False),
        "ARIA-noMA": (lambda: make_aria_noMA(cfg, device),  True),
        "ARIA-noPG": (lambda: make_aria_noPG(cfg, device),  True),
        "ARIA-noAGV":(lambda: make_aria_noAGV(cfg, device), True),
        "ARIA-noCBA":(lambda: make_aria_noCBA(cfg, device), True),
    }

    all_results = {}

    for name, (factory, use_spc) in ablations.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        matrices = []
        for seed in args.seeds:
            set_seed(seed)
            tasks  = get_split_mnist_tasks(data_dir=args.data, batch_size=64)
            model  = factory()
            if not hasattr(model, "task_heads") or len(model.task_heads) == 0:
                for _ in range(T):
                    model.add_task_head(device, n_classes=2)
            model  = model.to(device)
            mat    = _train_custom(model, tasks, device, args.epochs, use_spc=use_spc,
                                   verbose=not args.quiet)
            matrices.append(mat)
        all_results[name] = matrices

    # Table
    table = {}
    print(f"\n{'Model':<20} {'Avg Acc':>10} {'Forgetting':>12} {'BWT':>8}")
    print("-" * 55)
    for name, matrices in all_results.items():
        agg = aggregate_seeds(matrices)
        table[name] = agg
        aa  = agg["avg_acc"]
        fgt = agg["forgetting"]
        bwt = agg["bwt"]
        print(f"{name:<20} {aa['mean']:>7.3f}±{aa['std']:.3f} {fgt['mean']:>9.3f}±{fgt['std']:.3f} {bwt['mean']:>5.3f}±{bwt['std']:.3f}")

    with open(out_dir / "ablation_table.json", "w") as f:
        json.dump(table, f, indent=2)

    # Waterfall
    full  = table["ARIA+SPC"]["avg_acc"]["mean"]
    nospc = table["ARIA-noSPC"]["avg_acc"]["mean"]
    noma  = table["ARIA-noMA"]["avg_acc"]["mean"]
    nopg  = table["ARIA-noPG"]["avg_acc"]["mean"]

    plot_ablation_waterfall(
        contributions = {
            "+SPC":  full  - nospc,
            "+MA":   nospc - noma,
            "+PG":   noma  - nopg,
        },
        baseline_acc = nopg,
        out_path     = str(out_dir / "ablation_waterfall.png"),
        title        = "Ablation: component contributions (Split-MNIST)",
    )

    fig_dir = out_dir / "figures"
    plot_summary_bars(table, metric="avg_acc",
                      out_path=str(fig_dir / "ablation_bars.png"))

    print(f"\nDone. Results → {out_dir}")


if __name__ == "__main__":
    main()
