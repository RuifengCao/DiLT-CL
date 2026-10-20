"""High-level training pipelines for each continual-learning method.

Each ``train_*`` function drives a full task sequence and returns the standard
``(result_matrix, acc, bwt)`` triple, printing per-task accuracies and the
final ACC / BWT / Average-Forgetting summary along the way.
"""

import numpy as np

from .diltcl import DiLTCL
from .ewc import EWC
from .lwf import LwF
from .metrics import (
    compute_acc_bwt,
    compute_avg_forgetting,
    evaluate,
    evaluate_with_mask,
)


def train_diltcl(model, train_loaders, test_loaders, num_tasks,
                 device="cuda", num_partitions=2, pruning_rounds=5,
                 target_sparsity=0.9, full_train_epochs=50, imp_epochs=20,
                 expert_finetune_epochs=20, lr=1e-3, return_model=False):
    """Run the full DiLT-CL pipeline over a task sequence.

    Phase 1 extracts the core subnetwork from task 0; Phase 2 extracts a
    task-specific expert for every subsequent task and evaluates on all seen
    tasks.

    Args:
        model: Neural network (``SimpleCNN`` or ``SimpleMLP``).
        train_loaders: Per-task training DataLoaders.
        test_loaders: Per-task test DataLoaders.
        num_tasks: Number of tasks.
        device: ``'cuda'`` or ``'cpu'``.
        num_partitions: Data partitions for core extraction.
        pruning_rounds: IMP iterations.
        target_sparsity: Target sparsity for masks.
        full_train_epochs: Phase-1 full-network training epochs.
        imp_epochs: Epochs per IMP round.
        expert_finetune_epochs: Epochs for training each expert after IMP.
        lr: Learning rate.
        return_model: If ``True``, also return the trained :class:`DiLTCL`
            object so callers can inspect its measured core / expert densities
            (used for parameter / memory accounting).

    Returns:
        ``(result_matrix, acc, bwt)``, or ``(result_matrix, acc, bwt, diltcl)``
        when ``return_model`` is ``True``.
    """
    print("\n" + "=" * 60)
    print("DiLT-CL Training Pipeline")
    print("=" * 60)

    diltcl = DiLTCL(
        model, device=device, num_partitions=num_partitions,
        pruning_rounds=pruning_rounds, target_sparsity=target_sparsity,
        full_train_epochs=full_train_epochs, imp_epochs=imp_epochs,
        expert_finetune_epochs=expert_finetune_epochs, lr=lr,
    )

    # --- Phase 1: core extraction on task 0 ---
    print("\n--- Phase 1: Core Subnetwork Extraction ---")
    diltcl.extract_core(train_loaders, task_id=0)

    result_matrix = np.zeros((num_tasks, num_tasks))
    acc_t0 = evaluate_with_mask(diltcl, test_loaders[0], 0, device)
    result_matrix[0][0] = acc_t0
    print(f"[DiLT-CL] Task 1 accuracy (core only): {acc_t0:.2f}%")

    # --- Phase 2: per-task expert extraction ---
    print("\n--- Phase 2: Expert Subnetwork Extraction ---")
    for task_id in range(1, num_tasks):
        print(f"\n--- Task {task_id + 1}/{num_tasks} ---")
        diltcl.extract_expert(train_loaders, task_id)

        for eval_task in range(task_id + 1):
            acc = evaluate_with_mask(diltcl, test_loaders[eval_task],
                                     eval_task, device)
            result_matrix[task_id][eval_task] = acc
            print(f"  Task {eval_task + 1} accuracy: {acc:.2f}%")

    acc, bwt = compute_acc_bwt(result_matrix)
    af = compute_avg_forgetting(result_matrix)

    print("\n" + "-" * 40)
    print("DiLT-CL Final Results:")
    print(f"  Average Accuracy (ACC): {acc:.2f}%")
    print(f"  Backward Transfer (BWT): {bwt:.2f}%")
    print(f"  Average Forgetting (AF): {af:.2f}%")
    print("-" * 40)

    if return_model:
        return result_matrix, acc, bwt, diltcl
    return result_matrix, acc, bwt


def train_ewc(model, train_loaders, test_loaders, num_tasks,
              device="cuda", lambda_ewc=10000, epochs=50, lr=1e-3,
              return_state=False, penalty_form="consolidated",
              fisher_labels="prediction"):
    """Run the EWC baseline over a task sequence.

    Args:
        model: Neural network model.
        train_loaders: Per-task training DataLoaders.
        test_loaders: Per-task test DataLoaders.
        num_tasks: Number of tasks.
        device: ``'cuda'`` or ``'cpu'``.
        lambda_ewc: EWC regularisation strength.
        epochs: Training epochs per task.
        lr: Learning rate.
        penalty_form: ``"consolidated"`` (default, O(P) per step) or
            ``"per_task"`` (v2 arithmetic); see :mod:`diltcl.ewc`.
        fisher_labels: ``"prediction"`` (model argmax, v2 behaviour) or
            ``"true"`` (empirical Fisher with the data label).

    Returns:
        ``(result_matrix, acc, bwt)``.
    """
    print("\n" + "=" * 60)
    print("EWC Training Pipeline")
    print("=" * 60)

    ewc = EWC(model, device=device, lambda_ewc=lambda_ewc,
              penalty_form=penalty_form, fisher_labels=fisher_labels)
    result_matrix = np.zeros((num_tasks, num_tasks))

    for task_id in range(num_tasks):
        print(f"\n[EWC] Training task {task_id + 1}/{num_tasks}...")
        prev_tasks = list(range(task_id)) if task_id > 0 else []

        ewc.train_task(train_loaders[task_id], test_loaders[task_id],
                       task_id, epochs=epochs, lr=lr,
                       previous_tasks=prev_tasks)

        for eval_task in range(task_id + 1):
            acc = evaluate(ewc.model, test_loaders[eval_task], device,
                           task_id=eval_task)
            result_matrix[task_id][eval_task] = acc

    acc, bwt = compute_acc_bwt(result_matrix)
    af = compute_avg_forgetting(result_matrix)

    print("\n" + "-" * 40)
    print("EWC Final Results:")
    print(f"  Average Accuracy (ACC): {acc:.2f}%")
    print(f"  Backward Transfer (BWT): {bwt:.2f}%")
    print(f"  Average Forgetting (AF): {af:.2f}%")
    print("-" * 40)

    if return_state:
        return result_matrix, acc, bwt, {"model": ewc.model, "learner": ewc}
    return result_matrix, acc, bwt


def train_lwf(model, train_loaders, test_loaders, num_tasks,
              device="cuda", alpha=1.0, temperature=2.0, epochs=50, lr=1e-3,
              return_state=False, shared_features=True):
    """Run the LwF baseline over a task sequence.

    Args:
        model: Neural network model.
        train_loaders: Per-task training DataLoaders.
        test_loaders: Per-task test DataLoaders.
        num_tasks: Number of tasks.
        device: ``'cuda'`` or ``'cpu'``.
        alpha: Distillation loss weight.
        temperature: Distillation temperature.
        epochs: Training epochs per task.
        lr: Learning rate.
        shared_features: Multi-head distillation with one trunk pass
            (default) or the v2 per-head forward loop; see :mod:`diltcl.lwf`.

    Returns:
        ``(result_matrix, acc, bwt)``.
    """
    print("\n" + "=" * 60)
    print("LwF Training Pipeline")
    print("=" * 60)

    lwf = LwF(model, device=device, alpha=alpha, temperature=temperature,
              shared_features=shared_features)
    result_matrix = np.zeros((num_tasks, num_tasks))

    for task_id in range(num_tasks):
        print(f"\n[LwF] Training task {task_id + 1}/{num_tasks}...")
        lwf.train_task(train_loaders[task_id], test_loaders[task_id],
                       task_id, epochs=epochs, lr=lr)

        for eval_task in range(task_id + 1):
            acc = evaluate(lwf.model, test_loaders[eval_task], device,
                           task_id=eval_task)
            result_matrix[task_id][eval_task] = acc

    acc, bwt = compute_acc_bwt(result_matrix)
    af = compute_avg_forgetting(result_matrix)

    print("\n" + "-" * 40)
    print("LwF Final Results:")
    print(f"  Average Accuracy (ACC): {acc:.2f}%")
    print(f"  Backward Transfer (BWT): {bwt:.2f}%")
    print(f"  Average Forgetting (AF): {af:.2f}%")
    print("-" * 40)

    if return_state:
        return result_matrix, acc, bwt, {"model": lwf.model, "learner": lwf}
    return result_matrix, acc, bwt
