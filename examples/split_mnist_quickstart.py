"""
ARIA on Split-MNIST — quickstart example.

This is the simplest possible entry point. Runs 1 seed, 3 epochs per task,
prints the accuracy matrix, and saves one figure.

Usage
-----
python examples/split_mnist_quickstart.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import aria
from aria.metrics import compute_metrics
from aria.plot import plot_accuracy_curves


def main():
    aria.set_seed(42)
    device = aria.get_device()
    print(f"Device: {device}\n")

    cfg   = aria.ARIAConfig(input_dim=784, n_classes=2, d_model=128, n_layers=2)
    tasks = aria.get_split_mnist_tasks(data_dir="./data", batch_size=64)

    print("Training ARIA+SPC on Split-MNIST (5 tasks, 3 epochs each)...")
    matrix = aria.train_aria(
        cfg             = cfg,
        tasks           = tasks,
        device          = device,
        epochs_per_task = 3,
        use_spc         = True,
        verbose         = True,
    )

    metrics = compute_metrics(matrix)
    print(f"\nFinal metrics:")
    print(f"  Average accuracy : {metrics['avg_acc']:.3f}")
    print(f"  Forgetting       : {metrics['forgetting']:.3f}")
    print(f"  BWT              : {metrics['bwt']:.3f}")

    print("\nAccuracy matrix (row = after training task i, col = task j):")
    import numpy as np
    for i, row in enumerate(matrix):
        vals = " ".join(f"{row[j]:.3f}" if j <= i else "    -" for j in range(5))
        print(f"  Task {i+1}: {vals}")

    plot_accuracy_curves(
        {"ARIA+SPC": [matrix]},
        out_path = "results/quickstart_accuracy.png",
    )
    print("\nFigure saved to results/quickstart_accuracy.png")


if __name__ == "__main__":
    main()
