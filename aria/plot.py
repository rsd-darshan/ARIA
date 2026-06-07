"""Publication-quality figure generation for ARIA results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


PALETTE = {
    "ARIA+SPC":   "#2563EB",
    "ARIA-noSPC": "#60A5FA",
    "EWC":        "#DC2626",
    "DER++":      "#16A34A",
    "StaticMLP":  "#9333EA",
}
MODEL_ORDER = ["ARIA+SPC", "ARIA-noSPC", "EWC", "DER++", "StaticMLP"]


# ---------------------------------------------------------------------------
# Accuracy over tasks
# ---------------------------------------------------------------------------

def plot_accuracy_curves(
    results:  Dict[str, List[np.ndarray]],
    out_path: str = "results/figures/accuracy_over_tasks.png",
    title:    str = "Average task accuracy after each training stage",
) -> None:
    """
    Line plot: x = task index (1-based), y = mean accuracy across previously
    seen tasks after training on task x.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for name in MODEL_ORDER:
        if name not in results:
            continue
        matrices = results[name]
        T = matrices[0].shape[0]
        means, stds = [], []
        for t in range(T):
            vals = [m[t, :t + 1].mean() for m in matrices]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means = np.array(means)
        stds  = np.array(stds)
        xs    = np.arange(1, T + 1)
        ax.plot(xs, means, label=name, color=PALETTE.get(name), marker="o", linewidth=2)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=PALETTE.get(name))

    ax.set_xlabel("Task index", fontsize=11)
    ax.set_ylabel("Mean accuracy", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

def plot_summary_bars(
    table:    Dict[str, Dict],
    metric:   str = "avg_acc",
    out_path: str = "results/figures/summary_bars.png",
) -> None:
    """Bar chart comparing models on a single metric."""
    names  = [n for n in MODEL_ORDER if n in table]
    means  = [table[n][metric]["mean"] for n in names]
    stds   = [table[n][metric]["std"]  for n in names]
    colors = [PALETTE.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.55,
                  capsize=4, alpha=0.9, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Model comparison — {metric}", fontsize=12, pad=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Forgetting heatmap
# ---------------------------------------------------------------------------

def plot_forgetting_heatmap(
    results:  Dict[str, List[np.ndarray]],
    out_path: str = "results/figures/forgetting_heatmap.png",
) -> None:
    """
    Heatmap: rows = models, columns = task id.
    Cell = mean drop from best-so-far to final accuracy on that task.
    """
    models = [n for n in MODEL_ORDER if n in results]
    T      = results[models[0]][0].shape[0]
    data   = np.zeros((len(models), T))

    for mi, name in enumerate(models):
        mats = results[name]
        for j in range(T):
            per_seed = []
            for m in mats:
                best  = max(m[j:, j].tolist() + [m[j, j]])
                final = m[T - 1, j]
                per_seed.append(best - final)
            data[mi, j] = np.mean(per_seed)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(data, cmap="Reds", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="Forgetting")
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"T{t+1}" for t in range(T)])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_title("Per-task forgetting by model", fontsize=12, pad=8)
    for mi in range(len(models)):
        for j in range(T):
            ax.text(j, mi, f"{data[mi,j]:.2f}", ha="center", va="center", fontsize=8)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Ablation waterfall
# ---------------------------------------------------------------------------

def plot_ablation_waterfall(
    contributions: Dict[str, float],
    baseline_acc:  float,
    out_path:      str = "results/figures/ablation_waterfall.png",
    title:         str = "Component contributions to average accuracy",
) -> None:
    """
    Waterfall chart showing cumulative contribution of each component.

    Parameters
    ----------
    contributions : ordered dict — {component_name: delta_acc}
    baseline_acc  : accuracy of the StaticMLP / naive baseline
    """
    names  = list(contributions.keys())
    deltas = list(contributions.values())
    cumul  = [baseline_acc]
    for d in deltas:
        cumul.append(cumul[-1] + d)

    fig, ax = plt.subplots(figsize=(8, 4))
    bottoms = cumul[:-1]
    colors  = ["#16A34A" if d >= 0 else "#DC2626" for d in deltas]
    ax.bar(range(len(names)), deltas, bottom=bottoms, color=colors,
           width=0.5, edgecolor="white", alpha=0.9)
    ax.axhline(baseline_acc, color="gray", linestyle="--", linewidth=1)
    ax.axhline(cumul[-1],    color="#2563EB", linestyle="--", linewidth=1.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Average accuracy", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for i, (d, b) in enumerate(zip(deltas, bottoms)):
        sign = "+" if d >= 0 else ""
        ax.text(i, b + d + 0.003, f"{sign}{d:.3f}", ha="center", va="bottom", fontsize=8)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Morphogenesis / gate trajectory
# ---------------------------------------------------------------------------

def plot_morphogenesis(
    head_counts: List[List[int]],
    gate_means:  List[List[float]],
    task_boundaries: Optional[List[int]] = None,
    out_path:    str = "results/figures/morphogenesis.png",
) -> None:
    """
    Two-panel figure: (top) active head count per layer over steps;
    (bottom) mean plasticity gate per layer over steps.

    Parameters
    ----------
    head_counts      : [[heads_l0_step0, heads_l1_step0, ...], ...]  length = n_steps
    gate_means       : [[gate_l0_step0, ...], ...]                    length = n_steps
    task_boundaries  : list of step indices where tasks begin
    """
    steps  = np.arange(len(head_counts))
    n_lay  = len(head_counts[0])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    layer_colors = plt.cm.tab10(np.linspace(0, 0.9, n_lay))

    for l in range(n_lay):
        ax1.plot(steps, [h[l] for h in head_counts],
                 label=f"Layer {l}", color=layer_colors[l], linewidth=1.5)
        ax2.plot(steps, [g[l] for g in gate_means],
                 color=layer_colors[l], linewidth=1.5)

    if task_boundaries:
        for b in task_boundaries:
            ax1.axvline(b, color="gray", linestyle="--", alpha=0.5)
            ax2.axvline(b, color="gray", linestyle="--", alpha=0.5)

    ax1.set_ylabel("Active heads", fontsize=10)
    ax1.legend(fontsize=8, ncol=n_lay)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax2.set_ylabel("Mean gate π", fontsize=10)
    ax2.set_xlabel("Training step", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Morphogenesis & plasticity gate trajectory", fontsize=12)
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {p}")
