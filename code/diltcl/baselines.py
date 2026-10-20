"""Reference baselines for the continual-learning comparison.

These complement the existing EWC and LwF baselines with the methods a
reviewer will expect to see for a subnetwork/parameter-isolation paper:

* ``train_finetune``    - naive sequential training (the lower bound; shows how
  much forgetting occurs with no mechanism at all).
* ``train_independent`` - one freshly-trained model per task (an
  interference-free reference, not a theoretical accuracy upper bound).
* ``train_er``          - Experience Replay: rehearse a reservoir buffer with
  cross-entropy. Simple but a strong, standard baseline.
* ``train_derpp``       - Dark Experience Replay++ (Buzzega et al., 2020): adds
  a logit-distillation (MSE) term plus a label-replay term. A current strong
  baseline that any new CL method should be compared against.

All functions train on per-task loaders whose labels are remapped to
``0..C-1`` (task-incremental protocol with task identity available at test
time) and return the standard ``(result_matrix, acc, bwt)`` triple, matching
:mod:`diltcl.training`.
"""


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .buffer import ReservoirBuffer
from .metrics import (
    set_model_task,
    compute_acc_bwt,
    compute_avg_forgetting,
    constant_lower_triangular,
    evaluate,
)


def _print_final(name, acc, bwt, af):
    print("\n" + "-" * 40)
    print(f"{name} Final Results:")
    print(f"  Average Accuracy (ACC): {acc:.2f}%")
    print(f"  Backward Transfer (BWT): {bwt:.2f}%")
    print(f"  Average Forgetting (AF): {af:.2f}%")
    print("-" * 40)


def train_finetune(model_builder, train_loaders, test_loaders, num_tasks,
                   device="cuda", epochs=50, lr=1e-3, return_state=False):
    """Naive sequential fine-tuning (no anti-forgetting mechanism).

    Args:
        model_builder: Zero-arg callable returning a fresh model.
        train_loaders / test_loaders: Per-task DataLoaders.
        num_tasks: Number of tasks.
        device: ``'cuda'`` or ``'cpu'``.
        epochs: Epochs per task.
        lr: Learning rate.

    Returns:
        ``(result_matrix, acc, bwt)``.
    """
    print("\n" + "=" * 60)
    print("Finetune (lower bound) Training Pipeline")
    print("=" * 60)

    model = model_builder().to(device)
    criterion = nn.CrossEntropyLoss()
    result_matrix = np.zeros((num_tasks, num_tasks))

    for task_id in range(num_tasks):
        print(f"\n[Finetune] Training task {task_id + 1}/{num_tasks}...")
        set_model_task(model, task_id)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            for data, target in train_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()

        for eval_task in range(task_id + 1):
            result_matrix[task_id][eval_task] = evaluate(
                model, test_loaders[eval_task], device, task_id=eval_task)

    acc, bwt = compute_acc_bwt(result_matrix)
    af = compute_avg_forgetting(result_matrix)
    _print_final("Finetune", acc, bwt, af)
    if return_state:
        return result_matrix, acc, bwt, {"model": model}
    return result_matrix, acc, bwt


def train_independent(model_builder, train_loaders, test_loaders, num_tasks,
                      device="cuda", epochs=50, lr=1e-3,
                      return_state=False):
    """One independent model per task (interference-free reference).

    Each task gets a fresh model, so there is no interference by construction.
    The returned matrix follows the package-wide lower-triangular convention:
    once task ``j`` has been learned its independent predictor remains fixed,
    so ``R[i, j] = R[j, j]`` for every ``i >= j``.

    Returns:
        ``(result_matrix, acc, bwt)`` with ``bwt == 0.0``.
    """
    print("\n" + "=" * 60)
    print("Independent-task Reference Training Pipeline")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    diag = []
    model_states = []

    for task_id in range(num_tasks):
        print(f"\n[Independent] Training task {task_id + 1}/{num_tasks}...")
        model = model_builder().to(device)
        set_model_task(model, task_id)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            for data, target in train_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()

        acc_task = evaluate(model, test_loaders[task_id], device,
                            task_id=task_id)
        diag.append(acc_task)
        if return_state:
            model_states.append({
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            })
        print(f"[Independent] Task {task_id + 1} accuracy: {acc_task:.2f}%")

    result_matrix = constant_lower_triangular(diag)
    acc, bwt = compute_acc_bwt(result_matrix)
    af = compute_avg_forgetting(result_matrix)
    _print_final("Independent", acc, bwt, af)
    if return_state:
        return result_matrix, acc, bwt, {"model_states": model_states}
    return result_matrix, acc, bwt


def _routed_forward(model, inputs, task_ids):
    """Logits for a replay batch whose rows may belong to different heads."""
    routed = getattr(model, "forward_routed", None)
    if callable(routed):
        return routed(inputs, task_ids)
    # Single-head model: every row goes through the one shared head.
    return model(inputs)


def _replay_ce_routed(model, buffer, replay_batch_size, criterion):
    """Replay cross-entropy with one trunk forward and per-row head routing.

    Mathematically identical to :func:`_replay_ce_grouped` (the mean over all
    replayed samples), but costs one forward instead of one per task present
    in the draw. Consumes the RNG exactly like the grouped version.
    """
    inputs, labels, _, task_ids = buffer.sample_routed(replay_batch_size)
    return criterion(_routed_forward(model, inputs, task_ids), labels)


def _replay_ce_grouped(model, buffer, replay_batch_size, criterion, task_id):
    """v2 replay term: one forward per task group, share-weighted (legacy)."""
    # Replay must go through the head that produced each item. Under
    # multi-head, feeding old samples through the current task's head trains
    # it to emit an earlier task's label, which corrupts the head instead of
    # preventing forgetting. Each group is weighted by its share of the replay
    # batch so the total equals the mean over all replayed samples; summing
    # per-group means would scale the replay term by the number of tasks
    # represented in the draw.
    groups = buffer.sample_by_task(replay_batch_size)
    drawn = sum(by.size(0) for _, by, _ in groups.values())
    loss = 0.0
    for replay_task, (bx, by, _) in groups.items():
        set_model_task(model, task_id if replay_task is None else replay_task)
        share = by.size(0) / drawn
        loss = loss + share * criterion(model(bx), by)
    set_model_task(model, task_id)
    return loss


def train_er(model_builder, train_loaders, test_loaders, num_tasks,
             device="cuda", epochs=50, lr=1e-3, buffer_size=500,
             replay_batch_size=32, return_state=False, routed_replay=True):
    """Experience Replay with a reservoir buffer.

    Args:
        routed_replay: ``True`` (default) computes the replay term with a
            single trunk forward and per-row head routing; ``False`` uses the
            v2 one-forward-per-task-group loop. Both are the same function of
            the same samples (they differ only in floating-point summation
            order); the flag exists to reproduce v2 numbers bit for bit.

    Returns:
        ``(result_matrix, acc, bwt)``.
    """
    print("\n" + "=" * 60)
    print("ER (Experience Replay) Training Pipeline")
    print("=" * 60)

    model = model_builder().to(device)
    criterion = nn.CrossEntropyLoss()
    buffer = ReservoirBuffer(buffer_size, device=device)
    result_matrix = np.zeros((num_tasks, num_tasks))

    for task_id in range(num_tasks):
        print(f"\n[ER] Training task {task_id + 1}/{num_tasks}...")
        set_model_task(model, task_id)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            for data, target in train_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                loss = criterion(model(data), target)
                if not buffer.is_empty():
                    if routed_replay:
                        loss = loss + _replay_ce_routed(
                            model, buffer, replay_batch_size, criterion)
                    else:
                        loss = loss + _replay_ce_grouped(
                            model, buffer, replay_batch_size, criterion,
                            task_id)

                loss.backward()
                optimizer.step()
                buffer.add(data, target, task_id=task_id)

        for eval_task in range(task_id + 1):
            result_matrix[task_id][eval_task] = evaluate(
                model, test_loaders[eval_task], device, task_id=eval_task)

    acc, bwt = compute_acc_bwt(result_matrix)
    af = compute_avg_forgetting(result_matrix)
    _print_final("ER", acc, bwt, af)
    if return_state:
        return result_matrix, acc, bwt, {"model": model, "buffer": buffer}
    return result_matrix, acc, bwt


def _derpp_terms_routed(model, buffer, replay_batch_size, criterion, alpha,
                        beta):
    """DER++ buffer terms with one trunk forward per draw (see train_er)."""
    # Logit-distillation term (uses stored logits).
    bx1, _, bl1, tasks1 = buffer.sample_routed(replay_batch_size)
    if bl1 is None:
        raise ValueError("DER++ requires stored logits in the buffer")
    loss = alpha * F.mse_loss(_routed_forward(model, bx1, tasks1), bl1)
    # Label-replay term (an independent draw, as in DER++).
    bx2, by2, _, tasks2 = buffer.sample_routed(replay_batch_size)
    loss = loss + beta * criterion(_routed_forward(model, bx2, tasks2), by2)
    return loss


def _derpp_terms_grouped(model, buffer, replay_batch_size, criterion, alpha,
                         beta, task_id):
    """v2 DER++ buffer terms: one forward per task group (legacy)."""
    # Both replay terms must go through the head that produced the stored
    # item; see the note in _replay_ce_grouped. Groups are weighted by their
    # share of the draw so each term keeps the mean-over-samples scale it had
    # before head routing.
    loss = 0.0
    groups = buffer.sample_by_task(replay_batch_size)
    usable = {t: g for t, g in groups.items() if g[2] is not None}
    drawn = sum(g[1].size(0) for g in usable.values())
    for replay_task, (bx1, _, bl1) in usable.items():
        set_model_task(model, task_id if replay_task is None else replay_task)
        share = bx1.size(0) / drawn
        loss = loss + alpha * share * F.mse_loss(model(bx1), bl1)
    groups2 = buffer.sample_by_task(replay_batch_size)
    drawn2 = sum(g[1].size(0) for g in groups2.values())
    for replay_task, (bx2, by2, _) in groups2.items():
        set_model_task(model, task_id if replay_task is None else replay_task)
        share = by2.size(0) / drawn2
        loss = loss + beta * share * criterion(model(bx2), by2)
    set_model_task(model, task_id)
    return loss


def train_derpp(model_builder, train_loaders, test_loaders, num_tasks,
                device="cuda", epochs=50, lr=1e-3, buffer_size=500,
                replay_batch_size=32, alpha=0.5, beta=0.5,
                return_state=False, routed_replay=True):
    """Dark Experience Replay++ (Buzzega et al., NeurIPS 2020).

    Loss = CE(current) + alpha * MSE(buffer logits) + beta * CE(buffer labels),
    with the two buffer terms drawn from independent reservoir samples.
    ``routed_replay`` has the same meaning as in :func:`train_er`.

    Returns:
        ``(result_matrix, acc, bwt)``.
    """
    print("\n" + "=" * 60)
    print("DER++ Training Pipeline")
    print("=" * 60)

    model = model_builder().to(device)
    criterion = nn.CrossEntropyLoss()
    buffer = ReservoirBuffer(buffer_size, device=device)
    result_matrix = np.zeros((num_tasks, num_tasks))

    for task_id in range(num_tasks):
        print(f"\n[DER++] Training task {task_id + 1}/{num_tasks}...")
        set_model_task(model, task_id)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            for data, target in train_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                outputs = model(data)
                loss = criterion(outputs, target)

                if not buffer.is_empty():
                    if routed_replay:
                        loss = loss + _derpp_terms_routed(
                            model, buffer, replay_batch_size, criterion,
                            alpha, beta)
                    else:
                        loss = loss + _derpp_terms_grouped(
                            model, buffer, replay_batch_size, criterion,
                            alpha, beta, task_id)

                loss.backward()
                optimizer.step()
                buffer.add(data, target, logits=outputs.detach(),
                           task_id=task_id)

        for eval_task in range(task_id + 1):
            result_matrix[task_id][eval_task] = evaluate(
                model, test_loaders[eval_task], device, task_id=eval_task)

    acc, bwt = compute_acc_bwt(result_matrix)
    af = compute_avg_forgetting(result_matrix)
    _print_final("DER++", acc, bwt, af)
    if return_state:
        return result_matrix, acc, bwt, {"model": model, "buffer": buffer}
    return result_matrix, acc, bwt
