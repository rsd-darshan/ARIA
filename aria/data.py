"""Data loading utilities for Split-MNIST and Split-CIFAR-10."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


Task = Tuple[DataLoader, DataLoader]  # (train_loader, test_loader)


# ---------------------------------------------------------------------------
# Split-MNIST
# ---------------------------------------------------------------------------

SPLIT_MNIST_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def get_split_mnist_tasks(
    data_dir:   str  = "./data",
    batch_size: int  = 64,
    flatten:    bool = True,
) -> List[Task]:
    """
    Returns 5 binary-classification tasks from Split-MNIST.

    Task k classifies digit pair SPLIT_MNIST_PAIRS[k].
    Labels within each task are 0 / 1 (not global digit labels).

    Parameters
    ----------
    data_dir   : directory to cache the raw dataset
    batch_size : DataLoader batch size
    flatten    : if True returns (B, 784); if False returns (B, 1, 28, 28)
    """
    tf = [transforms.ToTensor()]
    if flatten:
        tf.append(transforms.Lambda(lambda x: x.view(-1)))
    t = transforms.Compose(tf)

    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=t)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=t)

    tasks: List[Task] = []
    for c0, c1 in SPLIT_MNIST_PAIRS:
        tr_idx = [i for i, (_, y) in enumerate(train_ds) if y in (c0, c1)]
        te_idx = [i for i, (_, y) in enumerate(test_ds)  if y in (c0, c1)]

        tr_sub = _remap_labels(Subset(train_ds, tr_idx), {c0: 0, c1: 1})
        te_sub = _remap_labels(Subset(test_ds,  te_idx), {c0: 0, c1: 1})

        tasks.append((
            DataLoader(tr_sub, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False),
            DataLoader(te_sub, batch_size=256,        shuffle=False, num_workers=0, pin_memory=False),
        ))
    return tasks


# ---------------------------------------------------------------------------
# Split-CIFAR-10
# ---------------------------------------------------------------------------

SPLIT_CIFAR_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def get_split_cifar10_tasks(
    data_dir:   str  = "./data",
    batch_size: int  = 64,
    flatten:    bool = True,
) -> List[Task]:
    """
    Returns 5 binary-classification tasks from Split-CIFAR-10.

    Task k classifies CIFAR-10 class pair SPLIT_CIFAR_PAIRS[k].
    """
    tf_base = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if flatten:
        tf_base.append(transforms.Lambda(lambda x: x.view(-1)))
    t = transforms.Compose(tf_base)

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=t)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=t)

    tasks: List[Task] = []
    for c0, c1 in SPLIT_CIFAR_PAIRS:
        tr_idx = [i for i, (_, y) in enumerate(train_ds) if y in (c0, c1)]
        te_idx = [i for i, (_, y) in enumerate(test_ds)  if y in (c0, c1)]

        tr_sub = _remap_labels(Subset(train_ds, tr_idx), {c0: 0, c1: 1})
        te_sub = _remap_labels(Subset(test_ds,  te_idx), {c0: 0, c1: 1})

        tasks.append((
            DataLoader(tr_sub, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False),
            DataLoader(te_sub, batch_size=256,        shuffle=False, num_workers=0, pin_memory=False),
        ))
    return tasks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remap_labels(subset: Subset, label_map: dict) -> TensorDataset:
    xs, ys = [], []
    for x, y in subset:
        xs.append(x)
        ys.append(label_map[int(y)])
    return TensorDataset(torch.stack(xs), torch.tensor(ys))
