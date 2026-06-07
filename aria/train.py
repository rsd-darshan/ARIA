"""
Training loop for ARIA and baselines on Split-MNIST / Split-CIFAR-10.

Exported functions
------------------
train_aria          : train ARIA+SPC through all tasks; returns accuracy matrix
train_ewc           : train EWC wrapper through all tasks
train_der           : train DER++ through all tasks
train_static        : train StaticMLP (fine-tuning, no CL strategy) through all tasks
find_matched_hidden : solve for hidden_dim so StaticMLP matches ARIA param count
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import ARIA, ARIAConfig, DERPlusPlus, EWCWrapper, StaticMLP


# ---------------------------------------------------------------------------
# Parameter matching
# ---------------------------------------------------------------------------

def find_matched_hidden(cfg: ARIAConfig) -> int:
    """
    Solve for hidden_dim h such that StaticMLP(input_dim, h, 4 layers) has
    approximately the same parameter count as ARIA(cfg).

    Uses quadratic formula on the 4-layer MLP param expression:
      params = input_dim*h + h         (first layer + bias)
               + 3*(h*h + h)           (3 hidden layers + biases)
               + h*n_classes + n_classes (head, rough approximation)

    For ARIA with multiple heads we use single-head approximation and
    add head params separately.
    """
    target = sum(p.numel() for p in ARIA(cfg).parameters() if p.requires_grad)
    D, nc  = cfg.input_dim, cfg.n_classes
    # 4-layer MLP: L1 + 3 hidden + head
    # a*h^2 + b*h + c = target
    a = 3
    b = D + 3 + nc + 1
    c = -target
    disc = b * b - 4 * a * c
    h    = int((-b + math.sqrt(max(disc, 0))) / (2 * a))
    return max(h, 1)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader: DataLoader, task_id: int, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        out, _  = model(x, task_id)
        correct += (out.argmax(1) == y).sum().item()
        total   += len(y)
    model.train()
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# ARIA + SPC
# ---------------------------------------------------------------------------

def train_aria(
    cfg:            ARIAConfig,
    tasks:          List[Tuple[DataLoader, DataLoader]],
    device:         torch.device,
    epochs_per_task: int  = 5,
    lr:             float = 3e-4,
    verbose:        bool  = True,
    use_spc:        bool  = True,
) -> np.ndarray:
    """
    Train ARIA through all tasks sequentially.

    Returns
    -------
    acc_matrix : (T, T) array where acc_matrix[i, j] = accuracy on task j
                 evaluated after training on task i.
    """
    T       = len(tasks)
    model   = ARIA(cfg).to(device)
    for _ in range(T):
        model.add_task_head(device, n_classes=cfg.n_classes)

    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    matrix  = np.zeros((T, T))

    for t, (tr_loader, _) in enumerate(tasks):
        if verbose:
            print(f"  Task {t+1}/{T}  heads={model.architecture_state()['head_counts']}")

        for ep in range(epochs_per_task):
            for x, y in tr_loader:
                x, y    = x.to(device), y.to(device)
                opt.zero_grad()
                out, aux = model(x, t)
                loss     = F.cross_entropy(out, y) + aux
                loss.backward()
                model.dampen_slow_gradients()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        if use_spc and t < T - 1:
            model.consolidate_slow(tr_loader, t, device)

        for j, (_, te_loader) in enumerate(tasks[:t + 1]):
            matrix[t, j] = evaluate(model, te_loader, j, device)

        if verbose:
            row = " ".join(f"{matrix[t,j]:.3f}" for j in range(t + 1))
            print(f"    [{row}]")

    return matrix


# ---------------------------------------------------------------------------
# EWC
# ---------------------------------------------------------------------------

def train_ewc(
    cfg:            ARIAConfig,
    tasks:          List[Tuple[DataLoader, DataLoader]],
    device:         torch.device,
    epochs_per_task: int   = 5,
    lr:             float  = 3e-4,
    ewc_lambda:     float  = 5000.0,
    hidden_dim:     Optional[int] = None,
    verbose:        bool   = True,
) -> np.ndarray:
    T = len(tasks)
    h = hidden_dim or find_matched_hidden(cfg)
    base  = StaticMLP(cfg.input_dim, h, 4, cfg.n_classes, cfg.dropout)
    model = EWCWrapper(base, ewc_lambda=ewc_lambda).to(device)
    for _ in range(T):
        model.add_task_head(device, n_classes=cfg.n_classes)

    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    matrix = np.zeros((T, T))

    for t, (tr_loader, _) in enumerate(tasks):
        if verbose:
            print(f"  EWC Task {t+1}/{T}")

        for ep in range(epochs_per_task):
            for x, y in tr_loader:
                x, y     = x.to(device), y.to(device)
                opt.zero_grad()
                out, _   = model(x, t)
                loss     = F.cross_entropy(out, y) + model.ewc_loss(device)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        if t < T - 1:
            model.consolidate(tr_loader, t, device)

        for j, (_, te_loader) in enumerate(tasks[:t + 1]):
            matrix[t, j] = evaluate(model, te_loader, j, device)

        if verbose:
            row = " ".join(f"{matrix[t,j]:.3f}" for j in range(t + 1))
            print(f"    [{row}]")

    return matrix


# ---------------------------------------------------------------------------
# DER++
# ---------------------------------------------------------------------------

def train_der(
    cfg:            ARIAConfig,
    tasks:          List[Tuple[DataLoader, DataLoader]],
    device:         torch.device,
    epochs_per_task: int   = 5,
    lr:             float  = 3e-4,
    buf_size:       int    = 200,
    alpha:          float  = 0.1,
    beta:           float  = 0.5,
    hidden_dim:     Optional[int] = None,
    verbose:        bool   = True,
) -> np.ndarray:
    T = len(tasks)
    h = hidden_dim or find_matched_hidden(cfg)
    base  = StaticMLP(cfg.input_dim, h, 4, cfg.n_classes, cfg.dropout)
    model = DERPlusPlus(base, buf_size, alpha, beta).to(device)
    for _ in range(T):
        model.add_task_head(device, n_classes=cfg.n_classes)

    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    matrix = np.zeros((T, T))

    for t, (tr_loader, _) in enumerate(tasks):
        if verbose:
            print(f"  DER++ Task {t+1}/{T}")

        for ep in range(epochs_per_task):
            for x, y in tr_loader:
                x, y   = x.to(device), y.to(device)
                opt.zero_grad()
                out, _ = model(x, t)
                loss   = F.cross_entropy(out, y) + model.der_loss(device, t)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                model.update_buffer(x, y, out, device)

        for j, (_, te_loader) in enumerate(tasks[:t + 1]):
            matrix[t, j] = evaluate(model, te_loader, j, device)

        if verbose:
            row = " ".join(f"{matrix[t,j]:.3f}" for j in range(t + 1))
            print(f"    [{row}]")

    return matrix


# ---------------------------------------------------------------------------
# Static MLP (fine-tuning baseline)
# ---------------------------------------------------------------------------

def train_static(
    cfg:            ARIAConfig,
    tasks:          List[Tuple[DataLoader, DataLoader]],
    device:         torch.device,
    epochs_per_task: int   = 5,
    lr:             float  = 3e-4,
    hidden_dim:     Optional[int] = None,
    verbose:        bool   = True,
) -> np.ndarray:
    T = len(tasks)
    h = hidden_dim or find_matched_hidden(cfg)
    model = StaticMLP(cfg.input_dim, h, 4, cfg.n_classes, cfg.dropout).to(device)
    for _ in range(T):
        model.add_task_head(device, n_classes=cfg.n_classes)

    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    matrix = np.zeros((T, T))

    for t, (tr_loader, _) in enumerate(tasks):
        if verbose:
            print(f"  StaticMLP Task {t+1}/{T}")

        for ep in range(epochs_per_task):
            for x, y in tr_loader:
                x, y   = x.to(device), y.to(device)
                opt.zero_grad()
                out, _ = model(x, t)
                loss   = F.cross_entropy(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        for j, (_, te_loader) in enumerate(tasks[:t + 1]):
            matrix[t, j] = evaluate(model, te_loader, j, device)

        if verbose:
            row = " ".join(f"{matrix[t,j]:.3f}" for j in range(t + 1))
            print(f"    [{row}]")

    return matrix
