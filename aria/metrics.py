"""Continual-learning metrics: BWT, FWT, forgetting, average accuracy."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_metrics(acc_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute standard continual-learning metrics from the accuracy matrix.

    Parameters
    ----------
    acc_matrix : shape (T, T)
        acc_matrix[i, j] = test accuracy on task j evaluated after training on task i.
        acc_matrix[i, j] is defined only for j <= i.

    Returns
    -------
    dict with keys: avg_acc, forgetting, bwt, fwt
    """
    T = acc_matrix.shape[0]

    # Average accuracy: final row mean
    avg_acc  = float(np.mean(acc_matrix[T - 1, :T]))

    # Backward Transfer (BWT): how learning new tasks affected old ones
    bwt = float(np.mean([
        acc_matrix[T - 1, j] - acc_matrix[j, j]
        for j in range(T - 1)
    ])) if T > 1 else 0.0

    # Forgetting: average drop from best historical to final
    forgetting = float(np.mean([
        max(acc_matrix[j:T, j]) - acc_matrix[T - 1, j]
        for j in range(T - 1)
    ])) if T > 1 else 0.0

    # Forward Transfer: how learning past tasks affected future ones
    # Requires a reference (random-init) diagonal for rigorous FWT;
    # here we approximate as mean off-diagonal above the diagonal.
    fwt_vals = [
        acc_matrix[j - 1, j] - acc_matrix[0, 0]   # proxy: acc before seeing task j
        for j in range(1, T)
    ]
    fwt = float(np.mean(fwt_vals)) if fwt_vals else 0.0

    return {
        "avg_acc":   avg_acc,
        "bwt":       bwt,
        "forgetting": forgetting,
        "fwt":       fwt,
    }


def compute_forgetting(task_accs: Dict[str, List[List[float]]]) -> Dict[str, float]:
    """
    Convenience wrapper: compute forgetting from a dict of per-model task-accuracy traces.

    Parameters
    ----------
    task_accs : {model_name: [[acc_t0_after_task0, ...], [acc_t0_after_task1, ...], ...]}
        Outer list indexed by training stage; inner list by task id.

    Returns
    -------
    {model_name: forgetting_score}
    """
    results: Dict[str, float] = {}
    for name, traces in task_accs.items():
        T = len(traces)
        mat = np.zeros((T, T))
        for i, row in enumerate(traces):
            for j, v in enumerate(row):
                if j <= i:
                    mat[i, j] = v
        results[name] = compute_metrics(mat)["forgetting"]
    return results


def aggregate_seeds(
    seed_matrices: List[np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple seeds.

    Returns dict with mean and std for each metric.
    """
    per_seed = [compute_metrics(m) for m in seed_matrices]
    keys     = list(per_seed[0].keys())
    return {
        k: {
            "mean": float(np.mean([s[k] for s in per_seed])),
            "std":  float(np.std( [s[k] for s in per_seed])),
        }
        for k in keys
    }
