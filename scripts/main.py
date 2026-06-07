#!/usr/bin/env python3
"""
Multi-seed evaluation script for ARIA.

Usage
-----
python scripts/main.py --benchmark split_mnist --seeds 42 123 999 7 2024
python scripts/main.py --benchmark split_cifar10 --epochs 10 --out results_cifar10/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import aria
from aria.plot import (
    plot_accuracy_curves,
    plot_summary_bars,
    plot_forgetting_heatmap,
    plot_ablation_waterfall,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARIA multi-seed evaluation")
    p.add_argument("--benchmark", choices=["split_mnist", "split_cifar10"],
                   default="split_mnist")
    p.add_argument("--seeds",  type=int, nargs="+", default=[42, 123, 999, 7, 2024])
    p.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    p.add_argument("--out",    type=str, default="results/", help="Output directory")
    p.add_argument("--data",   type=str, default="./data",   help="Dataset cache dir")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    input_dim = 784 if args.benchmark == "split_mnist" else 3072
    cfg = aria.ARIAConfig(
        input_dim  = input_dim,
        n_classes  = 2,
        d_model    = args.d_model,
        n_layers   = args.n_layers,
    )
    device = aria.get_device()

    print(f"\nARIA Evaluation")
    print(f"  Benchmark : {args.benchmark}")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Device    : {device}")
    print(f"  Output    : {out_dir}\n")

    results = aria.evaluate_all(
        cfg             = cfg,
        seeds           = args.seeds,
        benchmark       = args.benchmark,
        epochs_per_task = args.epochs,
        data_dir        = args.data,
        device          = device,
        verbose         = not args.quiet,
    )

    table = aria.summary_table(
        results,
        out_path   = str(out_dir / "results_table.json"),
        print_table= True,
    )

    # Raw matrices for downstream analysis
    import numpy as np
    np.save(str(out_dir / "raw_matrices.npy"),
            {k: np.stack(v) for k, v in results.items()},
            allow_pickle=True)

    # Figures
    plot_accuracy_curves(results, out_path=str(fig_dir / "accuracy_over_tasks.png"),
                         title=f"Split-{args.benchmark.split('_')[1].upper()} accuracy")
    plot_summary_bars(table, metric="avg_acc",
                      out_path=str(fig_dir / "summary_bars_acc.png"))
    plot_summary_bars(table, metric="forgetting",
                      out_path=str(fig_dir / "summary_bars_fgt.png"))
    plot_forgetting_heatmap(results, out_path=str(fig_dir / "forgetting_heatmap.png"))

    # Ablation waterfall (requires ARIA+SPC and ARIA-noSPC and EWC and StaticMLP)
    if all(k in table for k in ["ARIA+SPC", "ARIA-noSPC", "EWC", "StaticMLP"]):
        baseline  = table["StaticMLP"]["avg_acc"]["mean"]
        ewc_acc   = table["EWC"]["avg_acc"]["mean"]
        no_spc    = table["ARIA-noSPC"]["avg_acc"]["mean"]
        aria_full = table["ARIA+SPC"]["avg_acc"]["mean"]
        plot_ablation_waterfall(
            contributions = {
                "EWC (vs. Static)": ewc_acc   - baseline,
                "MA+PG+AGV":        no_spc    - ewc_acc,
                "SPC":              aria_full - no_spc,
            },
            baseline_acc = baseline,
            out_path     = str(fig_dir / "ablation_waterfall.png"),
        )

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
