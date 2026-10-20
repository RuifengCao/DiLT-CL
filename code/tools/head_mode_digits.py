"""Head-mode comparison on a real (small) continual-learning benchmark.

The critical open question is whether the shared-backbone task-IL baselines were
depressed by the legacy single-reused-head protocol. Answering it properly needs
Split-CIFAR-100 on a GPU, which this environment cannot run. What it *can* do is
exercise the **production trainers** on genuine data: scikit-learn's bundled
handwritten-digit set (1797 real 8x8 images, 10 classes), split into 5 tasks of
2 classes -- a small Split-MNIST-style stream.

This serves two purposes:

1. an end-to-end integration test of the real training code on real data, and
2. a directional preview of the multi-head vs shared-head effect.

It is NOT a substitute for the Split-CIFAR-100 rerun: 5 easy binary tasks put
far less pressure on a shared head than 20 five-way tasks with 100 classes.
"""

import argparse
import contextlib
import io
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from diltcl.baselines import (  # noqa: E402
    train_derpp, train_er, train_finetune, train_independent)
from diltcl.models import MultiHeadModel  # noqa: E402
from diltcl.training import train_diltcl, train_ewc, train_lwf  # noqa: E402
from diltcl.utils import set_seed  # noqa: E402

NUM_TASKS = 5
CLASSES_PER_TASK = 2


class DigitMLP(nn.Module):
    """Small MLP sized for 8x8 digits."""

    def __init__(self, num_classes=CLASSES_PER_TASK):
        super().__init__()
        self.fc1 = nn.Linear(64, 96)
        self.fc2 = nn.Linear(96, 96)
        self.fc3 = nn.Linear(96, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


def build_stream(seed):
    """Split the digit set into NUM_TASKS disjoint-class tasks (task-local labels)."""
    data = load_digits()
    x = data.data.astype(np.float32) / 16.0
    y = data.target.astype(np.int64)
    train_loaders, test_loaders = [], []
    for task in range(NUM_TASKS):
        classes = [task * CLASSES_PER_TASK + offset
                   for offset in range(CLASSES_PER_TASK)]
        keep = np.isin(y, classes)
        xt, yt = x[keep], y[keep]
        yt = np.array([classes.index(int(label)) for label in yt])  # local labels
        x_tr, x_te, y_tr, y_te = train_test_split(
            xt, yt, test_size=0.3, random_state=seed, stratify=yt)
        train_loaders.append(DataLoader(
            TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
            batch_size=32, shuffle=True))
        test_loaders.append(DataLoader(
            TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te)),
            batch_size=64))
    return train_loaders, test_loaders


def builder_for(head_mode):
    if head_mode == "shared":
        return lambda: DigitMLP()
    return lambda: MultiHeadModel(DigitMLP(), NUM_TASKS, CLASSES_PER_TASK)


def run_method(method, head_mode, train_loaders, test_loaders, seed):
    """Run one production trainer and return (ACC, BWT)."""
    set_seed(seed)
    builder = builder_for(head_mode)
    epochs = 12
    with contextlib.redirect_stdout(io.StringIO()):
        if method == "finetune":
            matrix, acc, bwt = train_finetune(
                builder, train_loaders, test_loaders, NUM_TASKS,
                device="cpu", epochs=epochs, lr=1e-3)
        elif method == "independent":
            matrix, acc, bwt = train_independent(
                builder, train_loaders, test_loaders, NUM_TASKS,
                device="cpu", epochs=epochs, lr=1e-3)
        elif method == "er":
            matrix, acc, bwt = train_er(
                builder, train_loaders, test_loaders, NUM_TASKS,
                device="cpu", epochs=epochs, lr=1e-3, buffer_size=100,
                replay_batch_size=16)
        elif method == "derpp":
            matrix, acc, bwt = train_derpp(
                builder, train_loaders, test_loaders, NUM_TASKS,
                device="cpu", epochs=epochs, lr=1e-3, buffer_size=100,
                replay_batch_size=16, alpha=0.5, beta=0.5)
        elif method == "ewc":
            matrix, acc, bwt = train_ewc(
                builder(), train_loaders, test_loaders, NUM_TASKS,
                device="cpu", lambda_ewc=1000.0, epochs=epochs, lr=1e-3)
        elif method == "lwf":
            matrix, acc, bwt = train_lwf(
                builder(), train_loaders, test_loaders, NUM_TASKS,
                device="cpu", alpha=1.0, temperature=2.0, epochs=epochs,
                lr=1e-3)
        elif method == "diltcl":
            matrix, acc, bwt = train_diltcl(
                builder(), train_loaders, test_loaders, NUM_TASKS,
                device="cpu", num_partitions=2, pruning_rounds=3,
                target_sparsity=0.9, full_train_epochs=epochs,
                imp_epochs=4, expert_finetune_epochs=6, lr=1e-3)
        else:
            raise ValueError(method)
    return float(acc), float(bwt)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0,1,2",
                        help="Comma-separated seeds (default 0,1,2; a few minutes on a CPU).")
    args = parser.parse_args(argv)
    methods = ["finetune", "ewc", "lwf", "er", "derpp", "diltcl", "independent"]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = {mode: {m: [] for m in methods} for mode in ("shared", "multi")}

    for seed in seeds:
        train_loaders, test_loaders = build_stream(seed)
        for mode in ("shared", "multi"):
            for method in methods:
                if method == "diltcl" and mode == "multi":
                    # DiLT-CL already owns per-task parameters; the head-mode
                    # switch targets the shared-backbone baselines only.
                    continue
                acc, bwt = run_method(method, mode, train_loaders,
                                      test_loaders, seed)
                results[mode][method].append((acc, bwt))
                print(f"  [{mode:6s}] {method:12s} seed={seed} "
                      f"ACC={acc:6.2f} BWT={bwt:7.2f}", flush=True)

    print("\n" + "=" * 74)
    print(f"Split-Digits (5 tasks x 2 classes, real data, {len(seeds)} seeds) -- task-IL")
    print("=" * 74)
    print(f"{'method':14s} {'shared ACC':>18s} {'multi ACC':>18s} {'delta':>10s}")
    for method in methods:
        shared = np.array([a for a, _ in results["shared"][method]])
        if not results["multi"][method]:
            print(f"{method:14s} {shared.mean():11.2f}±{shared.std(ddof=1):4.2f} "
                  f"{'(n/a)':>18s} {'':>10s}")
            continue
        multi = np.array([a for a, _ in results["multi"][method]])
        print(f"{method:14s} {shared.mean():11.2f}±{shared.std(ddof=1):4.2f} "
              f"{multi.mean():11.2f}±{multi.std(ddof=1):4.2f} "
              f"{multi.mean() - shared.mean():+10.2f}")

    print(f"\n{'method':14s} {'shared BWT':>18s} {'multi BWT':>18s}")
    for method in methods:
        shared = np.array([b for _, b in results["shared"][method]])
        if not results["multi"][method]:
            print(f"{method:14s} {shared.mean():11.2f}±{shared.std(ddof=1):4.2f} "
                  f"{'(n/a)':>18s}")
            continue
        multi = np.array([b for _, b in results["multi"][method]])
        print(f"{method:14s} {shared.mean():11.2f}±{shared.std(ddof=1):4.2f} "
              f"{multi.mean():11.2f}±{multi.std(ddof=1):4.2f}")


if __name__ == "__main__":
    main()
