"""
Multi-seed evaluation harness.

evaluate_all   : run N seeds × M models and collect accuracy matrices
summary_table  : pretty-print results table (matches NCG table format)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import get_split_mnist_tasks, get_split_cifar10_tasks
from .metrics import aggregate_seeds
from .model import ARIAConfig
from .train import train_aria, train_der, train_ewc, train_static, find_matched_hidden


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_all(
    cfg:             ARIAConfig,
    seeds:           List[int],
    benchmark:       str   = "split_mnist",
    epochs_per_task: int   = 5,
    data_dir:        str   = "./data",
    device:          Optional[torch.device] = None,
    verbose:         bool  = True,
) -> Dict[str, List[np.ndarray]]:
    """
    Run all models over all seeds.

    Returns
    -------
    {model_name: [acc_matrix_seed0, acc_matrix_seed1, ...]}
    """
    dev = device or get_device()
    if verbose:
        print(f"Device: {dev} | Seeds: {seeds} | Benchmark: {benchmark}")
        print(f"Params — ARIA: {_count_aria_params(cfg):,} | EWC/matched h={find_matched_hidden(cfg)}")

    loader_fn = get_split_mnist_tasks if benchmark == "split_mnist" else get_split_cifar10_tasks

    results: Dict[str, List[np.ndarray]] = {
        "ARIA+SPC": [], "ARIA-noSPC": [], "EWC": [], "DER++": [], "StaticMLP": []
    }

    for seed in seeds:
        if verbose:
            print(f"\n=== Seed {seed} ===")
        set_seed(seed)
        tasks = loader_fn(data_dir=data_dir, batch_size=64)

        results["ARIA+SPC"].append(
            train_aria(cfg, tasks, dev, epochs_per_task, use_spc=True, verbose=verbose)
        )
        set_seed(seed)
        tasks = loader_fn(data_dir=data_dir, batch_size=64)
        results["ARIA-noSPC"].append(
            train_aria(cfg, tasks, dev, epochs_per_task, use_spc=False, verbose=verbose)
        )
        set_seed(seed)
        tasks = loader_fn(data_dir=data_dir, batch_size=64)
        results["EWC"].append(
            train_ewc(cfg, tasks, dev, epochs_per_task, verbose=verbose)
        )
        set_seed(seed)
        tasks = loader_fn(data_dir=data_dir, batch_size=64)
        results["DER++"].append(
            train_der(cfg, tasks, dev, epochs_per_task, verbose=verbose)
        )
        set_seed(seed)
        tasks = loader_fn(data_dir=data_dir, batch_size=64)
        results["StaticMLP"].append(
            train_static(cfg, tasks, dev, epochs_per_task, verbose=verbose)
        )

    return results


def summary_table(
    results:    Dict[str, List[np.ndarray]],
    out_path:   Optional[str] = None,
    print_table: bool         = True,
) -> Dict[str, Dict]:
    """
    Build and optionally print the results table.

    Returns
    -------
    {model_name: {metric: {mean, std}}}
    """
    table: Dict[str, Dict] = {}
    for name, matrices in results.items():
        table[name] = aggregate_seeds(matrices)

    if print_table:
        hdr = f"{'Model':<20} {'Avg Acc':>10} {'Forgetting':>12} {'BWT':>8} {'FWT':>8}"
        print("\n" + hdr)
        print("-" * len(hdr))
        for name, m in table.items():
            aa  = m["avg_acc"]
            fgt = m["forgetting"]
            bwt = m["bwt"]
            fwt = m["fwt"]
            print(
                f"{name:<20}"
                f" {aa['mean']:>7.3f}±{aa['std']:.3f}"
                f" {fgt['mean']:>9.3f}±{fgt['std']:.3f}"
                f" {bwt['mean']:>5.3f}±{bwt['std']:.3f}"
                f" {fwt['mean']:>5.3f}±{fwt['std']:.3f}"
            )

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(table, f, indent=2)
        if print_table:
            print(f"\nSaved → {out_path}")

    return table


def _count_aria_params(cfg: ARIAConfig) -> int:
    from .model import ARIA
    m = ARIA(cfg)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
