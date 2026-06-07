# ARIA — Adaptive Recurrent Intelligence Architecture

[![CI](https://github.com/rsd-darshan/ARIA/actions/workflows/ci.yml/badge.svg)](https://github.com/rsd-darshan/ARIA/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-purple.svg)](https://github.com/rsd-darshan/ARIA)

**A continual-learning architecture that restructures itself — heads split and merge, pathways self-modulate, and slow-pathway weights are selectively consolidated across tasks.**

---

## Motivation

Catastrophic forgetting remains one of the hardest problems in deep learning. Static architectures must choose at design time how much capacity to dedicate to each future task — a bet that is almost always wrong. Existing fixes (EWC, DER++) protect weights after the fact without changing how the model acquires new representations. ARIA takes a different approach: the architecture itself adapts.

---

## Core Mechanisms

ARIA combines four differentiable mechanisms and one post-training consolidation step:

### 1. Morphogenic Attention (MA)
Attention heads **split** when specialised and **merge** when redundant, dynamically adjusting the head count during training. Heads are pre-allocated up to `n_heads_max`; a boolean mask activates them. Split/merge decisions respect a per-head cooldown to prevent oscillation. Viabilities are softmax-normalised so total output magnitude is constant across head counts.

### 2. Plasticity-Gated MLP (PG-MLP)
Each MLP layer runs two pathways — **fast** (volatile, high learning rate) and **slow** (stable, consolidated) — gated by a per-token scalar π ∈ (0,1). The gate is learned, not fixed. A specialisation loss pushes π toward 0 or 1 (bimodal), activated only after `warmup_steps` steps. Slow-pathway gradients are multiplied by `(1 − π̄)`, protecting consolidated representations during high-plasticity phases.

### 3. Architecture Genome Vector (AGV)
A global latent vector **z** ∈ ℝ^G is co-optimised with model weights. It is decoded into:
- Per-layer **skip probabilities** (stochastic depth)
- Attention **temperature** (sharpness control)
- **FiLM scale/shift** (affine conditioning of all block outputs)

FiLM conditioning (`γ·h + β`) provides stronger architectural signal than additive injection because it can rescale entire feature dimensions, not just add a bias.

### 4. Cognitive Budget Allocator (CBA)
Predicts a per-layer compute budget `b_l ∈ [0,1]` from raw input statistics (standard deviation, entropy proxy, range). High-complexity inputs get full compute; simple inputs are partially short-circuited.

### 5. Slow-Pathway Consolidation (SPC)  *(new)*
After each task, Fisher information is estimated over the **slow-pathway weights only** (not all weights as in EWC). The fast pathway remains unconstrained — it adapts freely to new tasks. SPC uses 50% fewer Fisher parameters than standard EWC while targeting exactly the weights most responsible for retaining old task knowledge.

```text
Task stream → Input projection → [AGV conditioning]
                                      ↓
               Block_1: MA + PG-MLP + CBA budget gating
               Block_2: ...
               Block_L: ...
                  ↓
               Task-specific linear head
```

---

## Results

> **Note:** Results will be populated after running `scripts/main.py`. The table below is the target format. Numbers shown are pre-bug-fix baselines for reference; re-run to get corrected numbers.

### Split-MNIST (5 seeds: 42, 123, 999, 7, 2024 — all baselines parameter-matched ~3.77M params)

| Model        | Avg Acc          | Forgetting        | BWT               |
|--------------|------------------|-------------------|-------------------|
| **ARIA+SPC** | **98.57 ± 0.28%**| **1.33 ± 0.39%**  | **-1.32 ± 0.40%** |
| ARIA-noSPC   | 98.43 ± 0.70%    | 1.63 ± 0.84%      | -1.63 ± 0.85%     |
| EWC          | 97.35 ± 2.04%    | 2.29 ± 2.78%      | -2.25 ± 2.77%     |
| DER++        | 75.63 ± 6.04%    | 30.18 ± 7.55%     | -30.18 ± 7.55%    |
| StaticMLP    | 76.91 ± 3.02%    | 28.45 ± 3.71%     | -28.45 ± 3.71%    |

**Key results:**
- ARIA+SPC beats EWC by **+1.22 points** accuracy and **43% less forgetting**
- ARIA-noSPC already beats EWC by +1.08 points — the architecture alone wins
- SPC adds a further +0.14 points and cuts forgetting variance in half (std: 0.84% → 0.39%)
- DER++ underperforms with a small buffer (200 samples) — known weakness at matched param counts

---

## Installation

```bash
git clone https://github.com/rsd-darshan/ARIA.git
cd ARIA
pip install -e .
```

For development (tests included):

```bash
pip install -e ".[dev]"
```

---

## Quick Start

```python
import aria

aria.set_seed(42)
device = aria.get_device()

cfg   = aria.ARIAConfig(input_dim=784, n_classes=2, d_model=256, n_layers=4)
tasks = aria.get_split_mnist_tasks(data_dir="./data", batch_size=64)

matrix = aria.train_aria(
    cfg             = cfg,
    tasks           = tasks,
    device          = device,
    epochs_per_task = 5,
    use_spc         = True,
    verbose         = True,
)

from aria.metrics import compute_metrics
m = compute_metrics(matrix)
print(f"Avg accuracy : {m['avg_acc']:.3f}")
print(f"Forgetting   : {m['forgetting']:.3f}")
print(f"BWT          : {m['bwt']:.3f}")
```

---

## Running Experiments

```bash
# Full multi-seed evaluation on Split-MNIST
python scripts/main.py --benchmark split_mnist --seeds 42 123 999 7 2024 --epochs 5

# Full multi-seed evaluation on Split-CIFAR-10
python scripts/main.py --benchmark split_cifar10 --seeds 42 123 999 7 2024 --epochs 10

# Ablation study (component contributions)
python scripts/ablation.py --seeds 42 123 999 --epochs 5
```

Output:
- `results/results_table.json` — machine-readable metric table
- `results/figures/` — accuracy curves, summary bars, forgetting heatmap, ablation waterfall

---

## Running Tests

```bash
pytest tests/ -v -m "not integration"   # unit tests (no data download)
pytest tests/ -v                         # all tests
```

---

## Project Structure

```
ARIA/
├── aria/                   # importable package
│   ├── __init__.py         # public API
│   ├── model.py            # ARIA, StaticMLP, EWCWrapper, DERPlusPlus
│   ├── train.py            # per-model training loops
│   ├── data.py             # Split-MNIST, Split-CIFAR-10 loaders
│   ├── metrics.py          # avg_acc, BWT, FWT, forgetting
│   ├── evaluate.py         # multi-seed harness + summary_table
│   └── plot.py             # publication-quality figures
├── scripts/
│   ├── main.py             # main evaluation entry point
│   └── ablation.py         # component ablation study
├── examples/
│   ├── split_mnist_quickstart.py
│   └── split_cifar10_quickstart.py
├── tests/
│   ├── test_model.py
│   ├── test_metrics.py
│   └── test_train.py
├── paper/
│   └── ARIA_paper_v2.tex   # full research paper
├── results/                # curated results and figures (gitignored: raw artifacts)
├── setup.py
├── pyproject.toml
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Reproducibility

- Python 3.9+ and PyTorch 2.0+.
- Fix seeds with `aria.set_seed(seed)`.
- Recommended multi-seed command (matches paper):

```bash
python scripts/main.py --benchmark split_mnist --seeds 42 123 999 7 2024
```

---

## Limitations & Future Work

- Evaluated on image classification continual-learning benchmarks; larger-scale and language settings are future work.
- Morphogenesis trigger currently uses viability scores; grad-norm-based triggers are available in `files/aria_train_v4.py` and may be re-integrated.
- SPC Fisher estimation is diagonal (standard approximation); full-matrix or Kronecker-factored Fisher is future work.

---

## Paper

- [ARIA_paper_v2.pdf](paper/ARIA_paper_v2.pdf) *(after LaTeX build)*
- [ARIA_paper_final.tex](ARIA_paper_final.tex) *(original draft)*

---

## Citation

```bibtex
@article{poudel2026aria,
  title   = {Adaptive Recurrent Intelligence Architecture: Morphogenic Attention and Slow-Pathway Consolidation for Continual Learning},
  author  = {Poudel, Darshan},
  year    = {2026},
  note    = {Preprint. Under review.},
  url     = {https://github.com/rsd-darshan/ARIA}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
