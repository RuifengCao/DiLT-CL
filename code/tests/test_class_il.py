"""Tests for the class-IL evaluation reasoning (torch-free).

The class-IL path reuses the task-IL training functions with a global label
space and a single K-way head (see ``run_class_il.py``). These tests pin down
the two facts that make that reuse correct, plus the metric reinterpretation,
all with self-contained NumPy so they run without torch.
"""

import numpy as np


def _class_il_label_map(num_tasks, classes_per_task):
    """Replicate get_split_cifar100(class_il=True)'s per-task identity maps."""
    maps = []
    for t in range(num_tasks):
        classes = range(t * classes_per_task, (t + 1) * classes_per_task)
        maps.append({c: c for c in classes})   # identity = global labels
    return maps


def test_global_labels_are_contiguous_and_cover_all_classes():
    T, C = 20, 5
    maps = _class_il_label_map(T, C)
    seen = set()
    for m in maps:
        seen |= set(m.values())
    assert seen == set(range(T * C)), "global labels must tile 0..K-1 exactly"
    # tasks must be disjoint in the global label space
    sizes = sum(len(m) for m in maps)
    assert sizes == len(seen) == T * C


def test_joint_accuracy_equals_mean_for_equal_task_sizes():
    """With equal-sized per-task test sets, the mean of per-task accuracies
    equals the pooled (joint) accuracy -- so the matrix's last-row mean is the
    headline class-IL number."""
    rng = np.random.default_rng(0)
    per_task_n = 2500                      # CIFAR-100: 5 classes x 500 test imgs
    accs = rng.uniform(0, 100, size=20)    # per-task class-IL accuracies (%)
    correct = np.round(accs / 100.0 * per_task_n)
    joint = 100.0 * correct.sum() / (per_task_n * len(accs))
    mean_of_per_task = float(np.mean(100.0 * correct / per_task_n))
    assert abs(joint - mean_of_per_task) < 1e-9


def test_joint_differs_from_mean_for_unequal_sizes():
    """Sanity: the equivalence is specific to equal task sizes."""
    n = np.array([100, 900])               # unequal test-set sizes
    acc = np.array([90.0, 10.0])
    correct = acc / 100.0 * n
    joint = 100.0 * correct.sum() / n.sum()        # = 18.0
    mean_pt = float(np.mean(acc))                  # = 50.0
    assert abs(joint - mean_pt) > 1.0


def _acc(M):
    T = len(M)
    return float(np.mean(M[T - 1]))


def _forgetting(M):
    T = len(M)
    return float(np.mean([max(M[i][t] for i in range(t, T)) - M[T - 1][t]
                          for t in range(T - 1)]))


def test_class_il_matrix_summary():
    """A class-IL matrix is summarised by the same ACC / forgetting formulas;
    a column that degrades over time yields positive forgetting."""
    # Strong diagonal (just-learned), decaying as new classes are added.
    T = 4
    M = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1):
            M[i][j] = max(0.0, 80.0 - 15.0 * (i - j))   # forgets over time
    assert abs(_acc(M) - np.mean(M[T - 1])) < 1e-12
    assert _forgetting(M) > 0.0, "expected class-IL forgetting on a decaying col"
    # final accuracy should be below the diagonal (learning) accuracy
    diag_mean = float(np.mean([M[t][t] for t in range(T)]))
    assert _acc(M) < diag_mean


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} class-IL tests passed.")
    sys.exit(0)
