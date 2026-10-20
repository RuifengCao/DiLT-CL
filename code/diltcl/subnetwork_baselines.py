"""Subnetwork-family baselines: PackNet and SupSup.

DiLT-CL belongs to the *parameter-isolation* family of continual learners, so
the most informative comparison is against other members of that family rather
than only against regularization (EWC, LwF) or rehearsal (ER, DER++). This
module adds two canonical competitors:

* **PackNet** (Mallya & Lazebnik, 2018): a single network is *packed* task by
  task. After training a task we keep its highest-magnitude weights, freeze
  them, and free the rest for future tasks; the kept weights are then retrained
  to recover from pruning. Each task is served by the union of weights assigned
  up to and including it.
* **SupSup** (Wortsman et al., 2020): a fixed random backbone is never trained;
  instead a binary *supermask* is learned per task via the edge-popup algorithm
  (see :mod:`diltcl.masked_layers`).

Both methods are task-incremental (task identity is available at test time).
Under that oracle-route protocol, fixed task predictors yield a
lower-triangular, column-constant result matrix. This property does **not**
imply zero class-IL forgetting: a newly added candidate can steal an old input
during task inference. Independent models are an interference-free reference,
not a theoretical accuracy upper bound.

To stay consistent with DiLT-CL, PackNet masks **every** trainable parameter
(including biases), matching the convention in :mod:`diltcl.masking`. Both
functions follow the same ``(model_builder) -> (result_matrix, acc, bwt)``
contract as the other baselines, so the experiment runner can dispatch them
uniformly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import masking
from .baselines import _print_final
from .masked_layers import (
    convert_to_masked,
    current_subnet_mask,
    masked_score_parameters,
    masked_weights_named,
    set_task,
)
from .metrics import compute_acc_bwt, compute_avg_forgetting, evaluate


# ---------------------------------------------------------------------------
# PackNet
# ---------------------------------------------------------------------------

def _zeros_mask(model, device):
    """All-zeros mask over every trainable parameter."""
    return {name: torch.zeros_like(p, device=device)
            for name, p in model.named_parameters() if p.requires_grad}


def _train_with_grad_mask(model, loader, grad_mask, epochs, lr, device):
    """Train ``model`` while zeroing gradients outside ``grad_mask``.

    Only weights where ``grad_mask`` is non-zero are updated; everything else
    keeps its current value (Adam's moment estimates stay zero for masked-out
    weights because their gradient is identically zero each step).
    """
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            with torch.no_grad():
                # Same in-place multiply as the v2 per-parameter loop, fused
                # into one kernel (see masking.apply_gradient_mask_).
                masking.apply_gradient_mask_(model, grad_mask)
            optimizer.step()


def _select_topk_in_free(model, free_mask, keep_fraction):
    """Pick the highest-magnitude free weights to assign to the current task.

    Args:
        model: Model whose current weights define the magnitudes.
        free_mask: Mask of weights still available (1 = free).
        keep_fraction: Fraction of each parameter's *free* weights to keep.

    Returns:
        A binary mask (per parameter) of the weights assigned to this task.
    """
    task_mask = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        free = free_mask[name].to(p.device)
        n_free = int((free > 0).sum().item())
        if n_free == 0:
            task_mask[name] = torch.zeros_like(p)
            continue
        keep = max(1, int(round(keep_fraction * n_free)))
        if keep >= n_free:
            task_mask[name] = (free > 0).float()
            continue
        free_idx = (free > 0).reshape(-1).nonzero(as_tuple=False).squeeze(1)
        free_vals = p.data.abs().reshape(-1)[free_idx]
        selected_rel = torch.topk(free_vals, k=keep, largest=True,
                                  sorted=False).indices
        selected = free_idx[selected_rel]
        flat = torch.zeros_like(p).reshape(-1)
        flat[selected] = 1.0
        task_mask[name] = flat.reshape_as(p)
    return task_mask


def train_packnet(model_builder, train_loaders, test_loaders, num_tasks,
                  device="cuda", epochs=50, retrain_epochs=10, lr=1e-3,
                  keep_fraction=None, return_state=False):
    """Train PackNet over a task sequence.

    Args:
        model_builder: Zero-arg callable returning a fresh model.
        train_loaders / test_loaders: Per-task DataLoaders.
        num_tasks: Number of tasks.
        device: ``'cuda'`` or ``'cpu'``.
        epochs: Epochs for the initial (pre-prune) training of each task.
        retrain_epochs: Epochs to retrain a task's kept weights after pruning.
        lr: Learning rate.
        keep_fraction: Fraction of *currently-free* weights to keep for a task.
            If ``None`` (default), an equal-split schedule ``1 / (T - t)`` is
            used so every task receives roughly ``P / T`` weights and capacity
            is not exhausted early.

    Returns:
        ``(result_matrix, acc, bwt)``.
    """
    print("\n" + "=" * 60)
    print("PackNet Training Pipeline")
    print("=" * 60)

    model = model_builder().to(device)
    assigned = _zeros_mask(model, device)     # cumulative frozen weights
    eval_masks = []                           # eval_masks[t] = union up to t
    task_acc = [0.0] * num_tasks

    for task_id in range(num_tasks):
        print(f"\n[PackNet] Task {task_id + 1}/{num_tasks}...")
        free_mask = {name: 1.0 - assigned[name] for name in assigned}

        # (A) Train all currently-free weights.
        _train_with_grad_mask(model, train_loaders[task_id], free_mask,
                              epochs, lr, device)

        # (B) Select this task's weights; release the rest of the free pool.
        frac = (1.0 / (num_tasks - task_id)) if keep_fraction is None \
            else keep_fraction
        task_mask = _select_topk_in_free(model, free_mask, frac)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                released = (free_mask[name].to(p.device) > 0) & \
                           (task_mask[name] == 0)
                p.data.mul_((~released).float())     # zero released weights

        # (C) Retrain only this task's kept weights to recover from pruning.
        _train_with_grad_mask(model, train_loaders[task_id], task_mask,
                              retrain_epochs, lr, device)

        # Freeze this task's weights and snapshot the eval mask (union so far).
        for name in assigned:
            assigned[name] = ((assigned[name] > 0) |
                              (task_mask[name] > 0)).float()
        eval_masks.append(masking.clone_mask(assigned))

        density = masking.mask_density(task_mask)
        print(f"[PackNet] Assigned density for task {task_id + 1}: "
              f"{density:.4f} (cumulative {masking.mask_density(assigned):.4f})")

        # Evaluate this task once with its own (now fixed) subnetwork.
        task_acc[task_id] = masking_eval(model, test_loaders[task_id],
                                         eval_masks[task_id], device)

    # Fill the lower-triangular, column-constant matrix.
    result_matrix = np.zeros((num_tasks, num_tasks))
    for i in range(num_tasks):
        for j in range(i + 1):
            result_matrix[i][j] = task_acc[j]

    acc, bwt = compute_acc_bwt(result_matrix)
    af = compute_avg_forgetting(result_matrix)
    _print_final("PackNet", acc, bwt, af)
    if return_state:
        return result_matrix, acc, bwt, {"model": model, "eval_masks": eval_masks}
    return result_matrix, acc, bwt


def masking_eval(model, test_loader, mask, device):
    """Top-1 accuracy of ``model`` with ``mask`` applied to its weights."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # Apply the mask once for the whole evaluation, then restore.
        saved = {}
        for name, p in model.named_parameters():
            if name in mask and p.requires_grad:
                saved[name] = p.data.clone()
                p.data.mul_(mask[name].to(p.device))
        try:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                _, pred = model(data).max(1)
                total += target.size(0)
                correct += pred.eq(target).sum().item()
        finally:
            for name, p in model.named_parameters():
                if name in saved:
                    p.data = saved[name]
    return 100.0 * correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# SupSup
# ---------------------------------------------------------------------------

def train_supsup(model_builder, train_loaders, test_loaders, num_tasks,
                 device="cuda", epochs=50, lr=1e-3, sparsity=0.5,
                 return_state=False):
    """Train SupSup over a task sequence.

    A fixed random backbone is created by converting the model's layers to
    edge-popup masked layers; only the per-task scores are trained.

    Args:
        model_builder: Zero-arg callable returning a fresh model.
        train_loaders / test_loaders: Per-task DataLoaders.
        num_tasks: Number of tasks.
        device: ``'cuda'`` or ``'cpu'``.
        epochs: Epochs of score training per task.
        lr: Learning rate for the scores.
        sparsity: Fraction of weights masked out per task.

    Returns:
        ``(result_matrix, acc, bwt)``.
    """
    print("\n" + "=" * 60)
    print("SupSup Training Pipeline")
    print("=" * 60)

    model = convert_to_masked(model_builder(), num_tasks, sparsity).to(device)
    criterion = nn.CrossEntropyLoss()
    task_acc = [0.0] * num_tasks

    for task_id in range(num_tasks):
        print(f"\n[SupSup] Task {task_id + 1}/{num_tasks}...")
        set_task(model, task_id)
        params = list(masked_score_parameters(model, task_id))
        optimizer = optim.Adam(params, lr=lr)
        model.train()
        for _ in range(epochs):
            for data, target in train_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()

        # Each task's mask is independent over the frozen backbone, so its
        # accuracy is fixed once trained (constant across later rows).
        set_task(model, task_id)
        task_acc[task_id] = evaluate(model, test_loaders[task_id], device)
        print(f"[SupSup] Task {task_id + 1} accuracy: {task_acc[task_id]:.2f}%")

    result_matrix = np.zeros((num_tasks, num_tasks))
    for i in range(num_tasks):
        for j in range(i + 1):
            result_matrix[i][j] = task_acc[j]

    acc, bwt = compute_acc_bwt(result_matrix)
    af = compute_avg_forgetting(result_matrix)
    _print_final("SupSup", acc, bwt, af)
    if return_state:
        return result_matrix, acc, bwt, {"model": model}
    return result_matrix, acc, bwt


# ---------------------------------------------------------------------------
# WSN (Winning SubNetworks)
# ---------------------------------------------------------------------------

def train_wsn(model_builder, train_loaders, test_loaders, num_tasks,
              device="cuda", epochs=50, lr=1e-3, sparsity=0.5,
              return_state=False):
    """Train WSN (Kang et al., ICML 2022, *Forget-free CL with Winning Subnetworks*).

    WSN is the trainable-weight sibling of SupSup: it jointly learns the
    network weights *and* a per-task binary mask (via edge-popup scores), under
    the constraint that weights *won* by earlier tasks are frozen. Concretely,
    at task ``t`` we train this task's scores together with the weights that are
    not yet frozen; weights inside the accumulated "won" set keep their values.
    A task's winning weights are never modified afterwards, so its oracle-route
    task-IL predictor is fixed; at test time task ``t`` uses ``weight ⊙ m_t``.
    Class-IL routing can still forget when new masks join the candidate set.

    It reuses the SupSup masked layers with ``train_weights=True`` and stores a
    single trained backbone plus ``T`` binary masks, compared with the
    Independent reference's one-model-per-task storage.

    Returns:
        ``(result_matrix, acc, bwt)``.
    """
    print("\n" + "=" * 60)
    print("WSN Training Pipeline")
    print("=" * 60)

    model = convert_to_masked(model_builder(), num_tasks, sparsity,
                              train_weights=True).to(device)
    # Cache (name, weight) once; the same Parameter objects persist across tasks.
    weight_items = list(masked_weights_named(model))
    won = {name: torch.zeros_like(w) for name, w in weight_items}  # frozen set
    criterion = nn.CrossEntropyLoss()
    task_acc = [0.0] * num_tasks

    for task_id in range(num_tasks):
        print(f"\n[WSN] Task {task_id + 1}/{num_tasks}...")
        set_task(model, task_id)
        params = (list(masked_score_parameters(model, task_id))
                  + [w for _, w in weight_items])
        optimizer = optim.Adam(params, lr=lr)
        # Weights won by earlier tasks are frozen by zeroing their gradient;
        # ``won`` only changes between tasks, so its complement is built once
        # per task and applied with one fused multiply per step.
        free = {name: (1.0 - won[name]).to(w.device) for name, w in weight_items}
        model.train()
        for _ in range(epochs):
            for data, target in train_loaders[task_id]:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                with torch.no_grad():
                    masking.apply_gradient_mask_(model, free)
                optimizer.step()

        # This task's winning subnetwork becomes frozen for all later tasks.
        m_t = current_subnet_mask(model, task_id)
        for name in won:
            won[name] = ((won[name] > 0) | (m_t[name] > 0)).float()
        print(f"[WSN] Task {task_id + 1} subnet density: "
              f"{masking.mask_density(m_t):.4f} "
              f"(cumulative frozen {masking.mask_density(won):.4f})")

        set_task(model, task_id)
        task_acc[task_id] = evaluate(model, test_loaders[task_id], device)
        print(f"[WSN] Task {task_id + 1} accuracy: {task_acc[task_id]:.2f}%")

    result_matrix = np.zeros((num_tasks, num_tasks))
    for i in range(num_tasks):
        for j in range(i + 1):
            result_matrix[i][j] = task_acc[j]

    acc, bwt = compute_acc_bwt(result_matrix)
    af = compute_avg_forgetting(result_matrix)
    _print_final("WSN", acc, bwt, af)
    if return_state:
        return result_matrix, acc, bwt, {"model": model}
    return result_matrix, acc, bwt
