"""Evaluation utilities and continual-learning metrics.

Provides plain accuracy evaluation, masked evaluation for DiLT-CL, the
standard ACC / BWT / Average-Forgetting metrics, and a result-matrix printer.
"""

import numpy as np
import torch


def constant_lower_triangular(task_accuracies):
    """Return the canonical task-IL matrix for fixed oracle-routed predictors.

    ``R[i, j] = task_accuracies[j]`` for ``i >= j`` and is zero above the
    diagonal. This representation is compatible with the package's standard
    ACC/BWT/Forgetting functions.
    """
    acc = np.asarray(task_accuracies, dtype=float)
    if acc.ndim != 1:
        raise ValueError("task_accuracies must be a one-dimensional sequence")
    return np.tril(np.broadcast_to(acc, (acc.size, acc.size))).copy()


def set_model_task(model, task_id):
    """Point a multi-head model at ``task_id``; a no-op for single-head models.

    Multi-head task-IL needs every train/eval call to use the head belonging to
    the task being processed. Routing through this helper keeps the trainers
    agnostic: single-head models simply ignore it.
    """
    setter = getattr(model, "set_task", None)
    if callable(setter):
        setter(task_id)
    return model


def evaluate(model, test_loader, device="cuda", task_id=None):
    """Evaluate top-1 accuracy of a model on a test set.

    The prediction is ``argmax`` over *all* output units, so this doubles as a
    class-incremental per-task evaluation when the model has a single global
    head and the test labels are global (no task identity is used).

    Args:
        model: PyTorch model.
        test_loader: DataLoader for the test data.
        device: Device to run evaluation on.

    Returns:
        Accuracy as a percentage in ``[0, 100]``.
    """
    if task_id is not None:
        set_model_task(model, task_id)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_class_il(model, test_loaders, device="cuda"):
    """Joint class-incremental accuracy over the union of given test sets.

    Runs a single global-head model over every loader in ``test_loaders`` and
    predicts with ``argmax`` over all classes, with no task identity and no
    per-task output masking. This is the headline class-IL number; for
    equal-sized tasks it equals the mean of the per-task class-IL accuracies.

    Args:
        model: A model with a single head spanning all classes.
        test_loaders: Iterable of per-task test DataLoaders to pool.
        device: Device to run evaluation on.

    Returns:
        Joint accuracy as a percentage in ``[0, 100]``.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for loader in test_loaders:
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                _, predicted = model(data).max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_with_mask(diltcl, test_loader, task_id, device="cuda"):
    """Evaluate a DiLT-CL model using the mask for a specific task.

    Args:
        diltcl: A ``DiLTCL`` instance.
        test_loader: DataLoader for the test data.
        task_id: Task ID selecting the correct expert mask.
        device: Device to run evaluation on.

    Returns:
        Accuracy as a percentage.
    """
    diltcl.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = diltcl.forward_with_mask(data, task_id)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


def compute_acc_bwt(result_matrix):
    """Compute Average Accuracy (ACC) and Backward Transfer (BWT).

    Args:
        result_matrix: ``K x K`` array where ``result_matrix[i][j]`` is the
            accuracy on task ``j`` after training on task ``i``.

    Returns:
        ``(acc, bwt)`` where ``bwt`` is negative when forgetting occurs.
    """
    K = len(result_matrix)
    acc = np.mean([result_matrix[K - 1][j] for j in range(K)])
    if K <= 1:
        return acc, 0.0
    bwt = np.mean([result_matrix[K - 1][j] - result_matrix[j][j]
                   for j in range(K - 1)])
    return acc, bwt


def compute_avg_forgetting(result_matrix):
    """Compute the Average Forgetting (AF) metric.

    ``AF = (1 / (T-1)) * sum_j (max_{i>=j} ACC[i, j] - ACC[T-1, j])``

    Args:
        result_matrix: ``T x T`` result matrix.

    Returns:
        Average forgetting as a percentage.
    """
    T = len(result_matrix)
    if T <= 1:
        return 0.0
    forgetting = 0.0
    for j in range(T - 1):
        max_acc = max(result_matrix[i][j] for i in range(j, T))
        forgetting += max_acc - result_matrix[T - 1][j]
    return forgetting / (T - 1)


def print_result_matrix(result_matrix, method_name):
    """Pretty-print a continual-learning result matrix.

    Args:
        result_matrix: ``T x T`` numpy array.
        method_name: Name of the method (used in the header).
    """
    T = len(result_matrix)
    print(f"\n{method_name} Result Matrix (Tasks x Tasks):")
    header = "      " + " ".join([f"T{j + 1:>5}" for j in range(T)])
    print(header)
    for i in range(T):
        row_vals = " ".join(
            [f"{result_matrix[i][j]:5.1f}" if result_matrix[i][j] > 0
             else "  -- " for j in range(T)]
        )
        print(f"After T{i + 1:>2}: [{row_vals}]")


def learning_accuracy(result_matrix):
    """Mean accuracy on each task right after it is learned (the diagonal).

    Also called "learning accuracy" / "intransigence" reference; a high value
    means the model learns each task well in isolation, regardless of later
    forgetting.

    Args:
        result_matrix: ``T x T`` result matrix.

    Returns:
        Mean of the diagonal as a percentage.
    """
    T = len(result_matrix)
    if T == 0:
        return 0.0
    return float(np.mean([result_matrix[i][i] for i in range(T)]))


def compute_fwt(result_matrix, random_baseline):
    """Compute Forward Transfer (FWT) following Lopez-Paz & Ranzato (2017).

    ``FWT = (1 / (T-1)) * sum_{i=2}^{T} (R[i-1, i] - b_i)`` where ``b_i`` is the
    accuracy of a random-init model on task ``i``. Positive FWT means learning
    earlier tasks helped a later task before it was trained.

    Args:
        result_matrix: ``T x T`` result matrix (``R[i][j]`` = acc on task ``j``
            after training through task ``i``).
        random_baseline: Length-``T`` sequence of random-init accuracies.

    Returns:
        Forward transfer as a percentage (``0.0`` if undefined).
    """
    T = len(result_matrix)
    if T <= 1:
        return 0.0
    fwt = 0.0
    for i in range(1, T):
        fwt += result_matrix[i - 1][i] - random_baseline[i]
    return fwt / (T - 1)


def summarize_run(result_matrix, random_baseline=None):
    """Bundle the standard metrics for one run into a dict.

    Args:
        result_matrix: ``T x T`` result matrix.
        random_baseline: Optional length-``T`` random-init accuracies; if given,
            forward transfer is included.

    Returns:
        Dict with keys ``ACC``, ``BWT``, ``Forgetting``, ``LA`` and
        (optionally) ``FWT``.
    """
    acc, bwt = compute_acc_bwt(result_matrix)
    summary = {
        "ACC": float(acc),
        "BWT": float(bwt),
        "Forgetting": float(compute_avg_forgetting(result_matrix)),
        "LA": learning_accuracy(result_matrix),
    }
    if random_baseline is not None:
        summary["FWT"] = compute_fwt(result_matrix, random_baseline)
    return summary


def random_init_baseline(model_builder, test_loaders, device="cuda"):
    """Accuracy of a freshly-initialised model on each task (for FWT).

    Args:
        model_builder: Zero-arg callable returning a fresh model.
        test_loaders: Per-task test DataLoaders.
        device: Device to evaluate on.

    Returns:
        List of per-task random-init accuracies.
    """
    model = model_builder().to(device)
    return [evaluate(model, loader, device) for loader in test_loaders]
