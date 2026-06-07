# Contributing to ARIA

## Getting started

```bash
git clone https://github.com/rsd-darshan/ARIA.git
cd ARIA
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v -m "not integration"   # fast unit tests (no dataset download)
pytest tests/ -v                         # all tests including integration
```

## Code style

- Python 3.9+ compatible.
- No external formatting tooling enforced — keep lines under 100 characters.
- New model components belong in `aria/model.py`.
- New metrics belong in `aria/metrics.py`.

## Adding a new baseline

1. Implement the model in `aria/model.py`.
2. Add a `train_<name>` function in `aria/train.py` mirroring the existing pattern.
3. Export it from `aria/__init__.py`.
4. Add tests in `tests/test_train.py`.
5. Add the model to `scripts/main.py` and update the `MODEL_ORDER` list in `aria/plot.py`.

## Results policy

Curated result tables and figures (matching those in the paper) live under `results/`.
Raw training artifacts (checkpoints, large `.npy` dumps) are `.gitignore`d.
When updating numbers, regenerate from `scripts/main.py` with the fixed seeds listed in the paper.

## Pull requests

Keep PRs focused. For new experimental contributions, open an issue first to
discuss whether the scope fits the project.
