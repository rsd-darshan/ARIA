"""
ARIA — Adaptive Recurrent Intelligence Architecture
====================================================

Public API
----------
ARIAConfig      : configuration dataclass
ARIA            : full continual-learning model
StaticMLP       : fixed-width MLP baseline
EWCWrapper      : Elastic Weight Consolidation wrapper
DERPlusPlus     : Dark Experience Replay ++ wrapper

train_aria      : train ARIA through all tasks
train_ewc       : train EWC through all tasks
train_der       : train DER++ through all tasks
train_static    : train StaticMLP (fine-tuning) through all tasks

evaluate_all    : multi-seed evaluation across all models
summary_table   : print & save results table

get_split_mnist_tasks   : Split-MNIST data loaders
get_split_cifar10_tasks : Split-CIFAR-10 data loaders

compute_metrics : BWT / FWT / forgetting / avg-acc from an accuracy matrix
set_seed        : reproducibility helper
get_device      : auto-detect CUDA / MPS / CPU
"""

from .model import (
    ARIAConfig,
    ARIA,
    StaticMLP,
    EWCWrapper,
    DERPlusPlus,
)

from .train import (
    train_aria,
    train_ewc,
    train_der,
    train_static,
    find_matched_hidden,
)

from .evaluate import (
    evaluate_all,
    summary_table,
    set_seed,
    get_device,
)

from .data import (
    get_split_mnist_tasks,
    get_split_cifar10_tasks,
)

from .metrics import compute_metrics

__version__ = "1.0.0"
__author__  = "Darshan Poudel"
__all__ = [
    "ARIAConfig", "ARIA", "StaticMLP", "EWCWrapper", "DERPlusPlus",
    "train_aria", "train_ewc", "train_der", "train_static", "find_matched_hidden",
    "evaluate_all", "summary_table", "set_seed", "get_device",
    "get_split_mnist_tasks", "get_split_cifar10_tasks",
    "compute_metrics",
]
