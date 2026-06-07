"""
ARIA on Split-CIFAR-10 — quickstart example.

Usage
-----
python examples/split_cifar10_quickstart.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import aria
from aria.metrics import compute_metrics


def main():
    aria.set_seed(42)
    device = aria.get_device()
    print(f"Device: {device}\n")

    cfg = aria.ARIAConfig(
        input_dim  = 3072,   # 3 * 32 * 32 flattened CIFAR-10
        n_classes  = 2,
        d_model    = 256,
        n_layers   = 4,
    )
    tasks = aria.get_split_cifar10_tasks(data_dir="./data", batch_size=64)

    print("Training ARIA+SPC on Split-CIFAR-10 (5 tasks, 5 epochs each)...")
    matrix = aria.train_aria(
        cfg             = cfg,
        tasks           = tasks,
        device          = device,
        epochs_per_task = 5,
        use_spc         = True,
        verbose         = True,
    )

    m = compute_metrics(matrix)
    print(f"\nFinal metrics:")
    print(f"  Average accuracy : {m['avg_acc']:.3f}")
    print(f"  Forgetting       : {m['forgetting']:.3f}")
    print(f"  BWT              : {m['bwt']:.3f}")

    print("\nAccuracy matrix:")
    for i, row in enumerate(matrix):
        vals = " ".join(f"{row[j]:.3f}" if j <= i else "    -" for j in range(5))
        print(f"  Task {i+1}: {vals}")


if __name__ == "__main__":
    main()
