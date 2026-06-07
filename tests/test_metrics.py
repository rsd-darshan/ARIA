"""Unit tests for continual-learning metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from aria.metrics import compute_metrics, compute_forgetting, aggregate_seeds


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def _perfect_matrix(T=5):
    m = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1):
            m[i, j] = 1.0
    return m


def _naive_matrix(T=5):
    """Model that forgets completely: perfect on current task, 0 on all prior."""
    m = np.zeros((T, T))
    for i in range(T):
        m[i, i] = 1.0
    return m


def test_perfect_avg_acc():
    m = compute_metrics(_perfect_matrix())
    assert m["avg_acc"] == pytest.approx(1.0)


def test_perfect_forgetting_zero():
    m = compute_metrics(_perfect_matrix())
    assert m["forgetting"] == pytest.approx(0.0, abs=1e-6)


def test_perfect_bwt_zero():
    m = compute_metrics(_perfect_matrix())
    assert m["bwt"] == pytest.approx(0.0, abs=1e-6)


def test_naive_forgetting():
    # A model that forgets completely
    m = compute_metrics(_naive_matrix())
    # forgetting = mean(best - final); best=1, final=0 for tasks 0..3 → 1.0
    assert m["forgetting"] == pytest.approx(1.0, abs=1e-6)


def test_naive_avg_acc():
    m = compute_metrics(_naive_matrix(T=5))
    # final row = [0, 0, 0, 0, 1] → mean = 0.2
    assert m["avg_acc"] == pytest.approx(0.2, abs=1e-6)


def test_single_task():
    m = np.array([[0.95]])
    r = compute_metrics(m)
    assert r["avg_acc"] == pytest.approx(0.95)
    assert r["bwt"] == pytest.approx(0.0)
    assert r["forgetting"] == pytest.approx(0.0)


def test_bwt_negative_for_forgetting_model():
    m = compute_metrics(_naive_matrix())
    assert m["bwt"] < 0.0


def test_output_keys():
    r = compute_metrics(_perfect_matrix())
    assert set(r.keys()) >= {"avg_acc", "bwt", "forgetting", "fwt"}


# ---------------------------------------------------------------------------
# compute_forgetting
# ---------------------------------------------------------------------------

def test_compute_forgetting_dict():
    task_accs = {
        "model_a": [[1.0], [0.8, 1.0], [0.5, 0.7, 1.0]],
        "model_b": [[1.0], [1.0, 1.0], [1.0, 1.0, 1.0]],
    }
    result = compute_forgetting(task_accs)
    assert "model_a" in result
    assert "model_b" in result
    assert result["model_b"] == pytest.approx(0.0, abs=1e-6)
    assert result["model_a"] > 0.0


# ---------------------------------------------------------------------------
# aggregate_seeds
# ---------------------------------------------------------------------------

def test_aggregate_seeds_mean():
    m1 = _perfect_matrix()
    m2 = _naive_matrix()
    agg = aggregate_seeds([m1, m2])
    assert 0.0 < agg["avg_acc"]["mean"] < 1.0


def test_aggregate_seeds_std_positive():
    m1 = _perfect_matrix()
    m2 = _naive_matrix()
    agg = aggregate_seeds([m1, m2])
    assert agg["avg_acc"]["std"] > 0.0


def test_aggregate_seeds_single():
    agg = aggregate_seeds([_perfect_matrix()])
    assert agg["avg_acc"]["mean"] == pytest.approx(1.0)
    assert agg["avg_acc"]["std"]  == pytest.approx(0.0)


def test_aggregate_seeds_keys():
    agg = aggregate_seeds([_perfect_matrix()])
    for metric in ["avg_acc", "bwt", "forgetting", "fwt"]:
        assert metric in agg
        assert "mean" in agg[metric]
        assert "std"  in agg[metric]
