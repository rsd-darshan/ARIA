"""Integration-level tests for training loops (2 tasks, 1 epoch, tiny model)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import aria
from aria.model import ARIAConfig
from aria.train import (
    train_aria, train_ewc, train_der, train_static, find_matched_hidden
)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

def _make_tasks(n_tasks=2, n_samples=32, input_dim=28, n_classes=2):
    tasks = []
    for _ in range(n_tasks):
        x  = torch.randn(n_samples, input_dim)
        y  = torch.randint(0, n_classes, (n_samples,))
        ds = TensorDataset(x, y)
        tr = DataLoader(ds, batch_size=16)
        te = DataLoader(ds, batch_size=16)
        tasks.append((tr, te))
    return tasks


MICRO_CFG = ARIAConfig(
    input_dim    = 28,
    n_classes    = 2,
    d_model      = 16,
    n_layers     = 2,
    n_heads_init = 2,
    n_heads_max  = 4,
    genome_dim   = 8,
    warmup_steps = 0,
)
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# find_matched_hidden
# ---------------------------------------------------------------------------

def test_find_matched_hidden_positive():
    h = find_matched_hidden(MICRO_CFG)
    assert h > 0


def test_find_matched_hidden_roughly_matches():
    cfg    = ARIAConfig()
    h      = find_matched_hidden(cfg)
    target = sum(p.numel() for p in aria.ARIA(cfg).parameters() if p.requires_grad)
    base   = aria.StaticMLP(cfg.input_dim, h, 4, cfg.n_classes, cfg.dropout)
    actual = sum(p.numel() for p in base.parameters() if p.requires_grad)
    ratio  = actual / target
    # allow 50% relative difference (small cfg will be rough)
    assert 0.5 <= ratio <= 2.0


# ---------------------------------------------------------------------------
# train_aria
# ---------------------------------------------------------------------------

def test_train_aria_returns_matrix():
    tasks  = _make_tasks(n_tasks=2, input_dim=MICRO_CFG.input_dim)
    mat    = train_aria(MICRO_CFG, tasks, DEVICE, epochs_per_task=1, verbose=False)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (2, 2)


def test_train_aria_diagonal_reasonable():
    tasks  = _make_tasks(n_tasks=2, input_dim=MICRO_CFG.input_dim)
    mat    = train_aria(MICRO_CFG, tasks, DEVICE, epochs_per_task=1, verbose=False)
    # diagonal should be > 0 (at least some accuracy)
    assert mat[0, 0] > 0.0
    assert mat[1, 1] > 0.0


def test_train_aria_no_spc():
    tasks = _make_tasks(n_tasks=2, input_dim=MICRO_CFG.input_dim)
    mat   = train_aria(MICRO_CFG, tasks, DEVICE, epochs_per_task=1,
                       use_spc=False, verbose=False)
    assert mat.shape == (2, 2)


def test_train_aria_above_diagonal_zero():
    tasks = _make_tasks(n_tasks=2, input_dim=MICRO_CFG.input_dim)
    mat   = train_aria(MICRO_CFG, tasks, DEVICE, epochs_per_task=1, verbose=False)
    # mat[0, 1] should be 0 (task 1 not seen yet after task 0)
    assert mat[0, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# train_ewc
# ---------------------------------------------------------------------------

def test_train_ewc_returns_matrix():
    tasks = _make_tasks(n_tasks=2, input_dim=MICRO_CFG.input_dim)
    mat   = train_ewc(MICRO_CFG, tasks, DEVICE, epochs_per_task=1,
                      hidden_dim=32, verbose=False)
    assert mat.shape == (2, 2)


# ---------------------------------------------------------------------------
# train_der
# ---------------------------------------------------------------------------

def test_train_der_returns_matrix():
    tasks = _make_tasks(n_tasks=2, input_dim=MICRO_CFG.input_dim)
    mat   = train_der(MICRO_CFG, tasks, DEVICE, epochs_per_task=1,
                      hidden_dim=32, verbose=False)
    assert mat.shape == (2, 2)


# ---------------------------------------------------------------------------
# train_static
# ---------------------------------------------------------------------------

def test_train_static_returns_matrix():
    tasks = _make_tasks(n_tasks=2, input_dim=MICRO_CFG.input_dim)
    mat   = train_static(MICRO_CFG, tasks, DEVICE, epochs_per_task=1,
                         hidden_dim=32, verbose=False)
    assert mat.shape == (2, 2)


# ---------------------------------------------------------------------------
# set_seed reproducibility
# ---------------------------------------------------------------------------

def test_set_seed_reproducible():
    tasks = _make_tasks(2, input_dim=MICRO_CFG.input_dim)
    aria.set_seed(42)
    mat1  = train_static(MICRO_CFG, tasks, DEVICE, epochs_per_task=1,
                         hidden_dim=32, verbose=False)
    aria.set_seed(42)
    mat2  = train_static(MICRO_CFG, tasks, DEVICE, epochs_per_task=1,
                         hidden_dim=32, verbose=False)
    np.testing.assert_allclose(mat1, mat2, atol=1e-5)
