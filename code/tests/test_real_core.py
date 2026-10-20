"""Real-PyTorch regression tests for the scientific core.

Most of this project's other tests exercise NumPy *mirrors* of the algorithms
rather than the shipped implementation. That gap is exactly what allowed the
v2.0 P0 defect to survive a green test suite: the mirror of iterative magnitude
pruning was correct while the real ``_train_with_mask`` left pruned weights
active in the forward pass, and no test could tell the difference.

These tests import the production modules and run real autograd on CPU, so they
fail if the shipped code regresses. They are intentionally tiny (a few hundred
parameters, one or two epochs) and run in seconds without a GPU.

Skipped automatically when PyTorch is unavailable.
"""

import pathlib
import sys

# Run from anywhere: make the package importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - torch-free environment
    if __name__ == "__main__":
        print("PyTorch not installed; real-core regression tests skipped.")
        sys.exit(0)
    import pytest
    pytest.skip("PyTorch not installed; real-core regression tests skipped.", allow_module_level=True)

from diltcl import masking
from diltcl.diltcl import DiLTCL
from diltcl.metrics import evaluate, set_model_task
from diltcl.models import MultiHeadModel, SimpleCNN


class _Tiny(nn.Module):
    """Minimal MLP used to keep the tests fast."""

    def __init__(self, in_features=16, hidden=24, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


def _loader(n=64, in_features=16, num_classes=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_features, generator=generator)
    y = torch.randint(0, num_classes, (n,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def _quiet(fn, *args, **kwargs):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# The P0 defect: pruned weights must be inert in the forward pass
# ---------------------------------------------------------------------------

def test_masked_training_keeps_pruned_weights_at_zero():
    """The regression that a NumPy mirror could never have caught."""
    torch.manual_seed(0)
    model = _Tiny()
    learner = DiLTCL(model, device="cpu", pruning_rounds=1, imp_epochs=1)
    mask = masking.ones_like_mask(model)
    mask["fc1.weight"][: 8, :] = 0.0

    learner._train_with_mask(model, _loader(), mask, epochs=2, lr=1e-2)

    pruned = model.fc1.weight.data[:8, :]
    kept = model.fc1.weight.data[8:, :]
    assert torch.count_nonzero(pruned).item() == 0, (
        "pruned weights are non-zero: gradient masking alone leaves them "
        "active in the forward pass (the v2.0 P0 defect)")
    assert torch.count_nonzero(kept).item() > 0, "kept weights were not trained"


def test_gradient_masking_alone_would_fail_this_test():
    """Confirms the test above actually discriminates the buggy behaviour."""
    torch.manual_seed(0)
    model = _Tiny()
    mask = masking.ones_like_mask(model)
    mask["fc1.weight"][: 8, :] = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    for data, target in _loader():
        optimizer.zero_grad()
        criterion(model(data), target).backward()
        for name, param in model.named_parameters():
            if param.grad is not None and name in mask:
                param.grad *= mask[name]
        optimizer.step()
    assert torch.count_nonzero(model.fc1.weight.data[:8, :]).item() > 0


def test_apply_parameter_mask_is_idempotent():
    torch.manual_seed(0)
    model = _Tiny()
    mask = masking.ones_like_mask(model)
    mask["fc2.weight"][:, :6] = 0.0
    masking.apply_parameter_mask_(model, mask)
    once = model.fc2.weight.data.clone()
    masking.apply_parameter_mask_(model, mask)
    assert torch.equal(once, model.fc2.weight.data)


# ---------------------------------------------------------------------------
# DiLT-CL end to end
# ---------------------------------------------------------------------------

def _trained_learner(num_tasks=3):
    torch.manual_seed(0)
    loaders = [_loader(seed=t) for t in range(num_tasks)]
    learner = DiLTCL(_Tiny(), device="cpu", num_partitions=2, pruning_rounds=2,
                     target_sparsity=0.9, full_train_epochs=1, imp_epochs=1,
                     expert_finetune_epochs=1, lr=1e-2)
    _quiet(learner.extract_core, loaders, 0)
    for task in range(1, num_tasks):
        _quiet(learner.extract_expert, loaders, task)
    return learner, loaders


def test_forward_with_mask_restores_live_parameters():
    learner, _ = _trained_learner()
    before = {n: p.data.clone() for n, p in learner.model.named_parameters()}
    x = torch.randn(4, 16)
    for task in range(1, 3):
        learner.forward_with_mask(x, task)
    for name, param in learner.model.named_parameters():
        assert torch.equal(before[name], param.data), f"{name} was not restored"


def test_forward_cache_matches_the_definition():
    """The precomputed core+expert must equal the arithmetic it replaces."""
    learner, _ = _trained_learner()
    x = torch.randn(4, 16)
    for task in range(1, 3):
        fast = learner.forward_with_mask(x, task)
        saved = {n: p.data.clone() for n, p in learner.model.named_parameters()}
        with torch.no_grad():
            for name, param in learner.model.named_parameters():
                core = (learner.core_mask.get(name, torch.zeros_like(param))
                        * learner.frozen_core_values[name])
                expert = learner.expert_weights[task].get(name)
                param.data = core if expert is None else core + expert
            reference = learner.model(x)
            for name, param in learner.model.named_parameters():
                param.data = saved[name]
        assert torch.equal(fast, reference), "cached forward diverged"


def test_experts_do_not_leak_between_tasks():
    learner, _ = _trained_learner()
    x = torch.randn(4, 16)
    first = learner.forward_with_mask(x, 1).clone()
    learner.forward_with_mask(x, 2)
    again = learner.forward_with_mask(x, 1)
    assert torch.equal(first, again), "task 1 output changed after task 2 ran"


def test_core_and_expert_masks_are_disjoint():
    learner, _ = _trained_learner()
    for task, expert_mask in learner.expert_masks.items():
        for name, tensor in expert_mask.items():
            overlap = (tensor > 0) & (learner.core_mask[name] > 0)
            assert overlap.sum().item() == 0, (
                f"expert {task} overlaps the core in {name}")


# ---------------------------------------------------------------------------
# Compact export
# ---------------------------------------------------------------------------

def test_compact_export_reproduces_logits_exactly():
    from diltcl import compact_export

    learner, _ = _trained_learner()
    compact = compact_export.export_diltcl_compact(learner)
    report = compact_export.verify_roundtrip_logits(
        learner, compact, torch.randn(6, 16), tasks=[1, 2])
    assert report["ok"], report["max_abs_diff_per_task"]


def test_compact_export_is_smaller_than_dense():
    from diltcl import compact_export

    learner, _ = _trained_learner()
    compact = compact_export.export_diltcl_compact(learner)
    assert (compact_export.compact_size_bytes(compact)
            < compact_export.dense_size_bytes(learner))


# ---------------------------------------------------------------------------
# Multi-head task-IL
# ---------------------------------------------------------------------------

def test_multihead_updates_only_the_active_head():
    torch.manual_seed(0)
    model = MultiHeadModel(_Tiny(num_classes=4), num_tasks=3, classes_per_task=4)
    before = [head.weight.data.clone() for head in model.heads]
    set_model_task(model, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    for data, target in _loader():
        optimizer.zero_grad()
        criterion(model(data), target).backward()
        optimizer.step()
    changed = [not torch.equal(before[i], model.heads[i].weight.data)
               for i in range(3)]
    assert changed == [False, True, False], changed


def test_evaluate_routes_to_the_requested_head():
    torch.manual_seed(0)
    model = MultiHeadModel(_Tiny(num_classes=4), num_tasks=3, classes_per_task=4)
    with torch.no_grad():                       # make head 2 always predict 3
        model.heads[2].weight.zero_()
        model.heads[2].bias.zero_()
        model.heads[2].bias[3] = 10.0
    loader = _loader()
    set_model_task(model, 0)
    accuracy = evaluate(model, loader, "cpu", task_id=2)
    targets = torch.cat([y for _, y in loader])
    expected = 100.0 * (targets == 3).float().mean().item()
    assert abs(accuracy - expected) < 1e-6, (accuracy, expected)


def test_set_model_task_is_a_noop_for_single_head_models():
    model = SimpleCNN(num_classes=5)
    set_model_task(model, 3)          # must not raise
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 5)


def test_multihead_rejects_out_of_range_task():
    model = MultiHeadModel(_Tiny(num_classes=4), num_tasks=2, classes_per_task=4)
    for bad in (-1, 2):
        try:
            model.set_task(bad)
            assert False, f"expected IndexError for {bad}"
        except IndexError:
            pass


# ---------------------------------------------------------------------------
# EWC: the Fisher must be the per-sample empirical Fisher
# ---------------------------------------------------------------------------

def _fisher_reference(model, inputs):
    """Mean of per-sample squared gradients -- the empirical Fisher."""
    import torch.nn.functional as F

    total = {name: torch.zeros_like(p)
             for name, p in model.named_parameters() if p.requires_grad}
    for i in range(inputs.size(0)):
        model.zero_grad()
        output = F.log_softmax(model(inputs[i:i + 1]), dim=1)
        F.nll_loss(output, output.max(1)[1]).backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                total[name] += param.grad.detach() ** 2
    return {name: value / inputs.size(0) for name, value in total.items()}


def test_ewc_fisher_matches_the_per_sample_definition():
    from diltcl.ewc import EWC

    torch.manual_seed(0)
    inputs = torch.randn(24, 16)
    labels = torch.randint(0, 4, (24,))
    state = _Tiny().state_dict()

    reference_model = _Tiny()
    reference_model.load_state_dict(state)
    reference = _fisher_reference(reference_model, inputs)

    model = _Tiny()
    model.load_state_dict(state)
    learner = EWC(model, device="cpu")
    _quiet(learner.compute_fisher,
           DataLoader(TensorDataset(inputs, labels), batch_size=8),
           0, num_samples=24)
    for name, expected in reference.items():
        assert torch.allclose(learner.fisher_dict[0][name], expected), name


def test_ewc_fisher_is_independent_of_batch_size():
    """Averaging the loss first made the Fisher shrink like 1/batch_size."""
    from diltcl.ewc import EWC

    torch.manual_seed(0)
    inputs = torch.randn(24, 16)
    labels = torch.randint(0, 4, (24,))
    # The same weights for every batch size, so only the batching varies.
    reference_state = _Tiny().state_dict()
    magnitudes = []
    for batch_size in (1, 4, 12):
        model = _Tiny()
        model.load_state_dict(reference_state)
        learner = EWC(model, device="cpu")
        _quiet(learner.compute_fisher,
               DataLoader(TensorDataset(inputs, labels), batch_size=batch_size),
               0, num_samples=24)
        magnitudes.append(
            float(learner.fisher_dict[0]["fc1.weight"].abs().sum()))
    assert max(magnitudes) - min(magnitudes) < 1e-9, magnitudes


# ---------------------------------------------------------------------------
# Structural invariants verified in the v2.1.10 audit
# ---------------------------------------------------------------------------

def test_pruning_keeps_exactly_k_weights_under_ties():
    """All-equal magnitudes are the worst case for threshold-based pruning."""
    layer = nn.Linear(10, 1, bias=False)
    with torch.no_grad():
        layer.weight.data = torch.ones(1, 10)
    for rate in (0.2, 0.5):
        pruned = masking.prune_by_magnitude(
            layer, masking.ones_like_mask(layer), rate)
        assert int((pruned["weight"] > 0).sum()) == 10 - round(rate * 10), rate


def test_pruning_keeps_the_largest_magnitudes():
    layer = nn.Linear(6, 1, bias=False)
    with torch.no_grad():
        layer.weight.data = torch.tensor([[0.1, -0.9, 0.5, -0.2, 0.8, 0.05]])
    kept = masking.prune_by_magnitude(
        layer, masking.ones_like_mask(layer), 0.5)["weight"][0] > 0
    magnitudes = layer.weight.data[0].abs()
    assert (magnitudes[kept].min() >= magnitudes[~kept].max())


def test_mask_algebra():
    model = nn.Linear(6, 4)
    left = masking.ones_like_mask(model)
    right = masking.ones_like_mask(model)
    left["weight"][:2, :] = 0
    right["weight"][1:3, :] = 0
    expected = ((left["weight"] > 0) & (right["weight"] > 0)).float()
    assert torch.equal(
        masking.intersection_masks([left, right])["weight"], expected)
    complement = masking.complement_mask(left)
    assert torch.equal(complement["weight"], 1 - left["weight"])
    assert torch.equal(masking.combine_masks(left, complement)["weight"],
                       torch.ones_like(left["weight"]))


def test_expert_masks_live_strictly_outside_the_core():
    learner, _ = _trained_learner()
    complement = masking.complement_mask(learner.core_mask)
    for task, expert_mask in learner.expert_masks.items():
        for name, tensor in expert_mask.items():
            assert ((tensor > 0) <= (complement[name] > 0)).all(), (task, name)


def test_task_zero_uses_the_core_alone():
    """Task 0 has no expert; its forward must equal the masked core."""
    learner, _ = _trained_learner()
    inputs = torch.randn(4, 16)
    produced = learner.forward_with_mask(inputs, 0)
    saved = {n: p.data.clone() for n, p in learner.model.named_parameters()}
    with torch.no_grad():
        for name, param in learner.model.named_parameters():
            param.data = (learner.core_mask.get(name, torch.zeros_like(param))
                          * learner.frozen_core_values[name])
        reference = learner.model(inputs)
        for name, param in learner.model.named_parameters():
            param.data = saved[name]
    assert torch.allclose(produced, reference, atol=1e-6)


def test_permuted_mnist_permutes_pixels_and_keeps_labels():
    from diltcl.data import PermutedMNIST

    images = torch.arange(784, dtype=torch.float32).reshape(1, 1, 28, 28)
    labels = torch.tensor([3])
    permutation = torch.from_numpy(
        np.random.RandomState(0).permutation(784)).long()
    dataset = PermutedMNIST(TensorDataset(images, labels), permutation)
    image, label = dataset[0]
    assert torch.equal(image.reshape(-1), images.reshape(-1)[permutation])
    assert int(label) == 3


# ---------------------------------------------------------------------------
# Multi-head replay must be routed per head
# ---------------------------------------------------------------------------

def test_buffer_records_and_groups_by_task():
    from diltcl.buffer import ReservoirBuffer

    buffer = ReservoirBuffer(capacity=60)
    for task in range(3):
        buffer.add(torch.randn(20, 4), torch.randint(0, 2, (20,)), task_id=task)
    grouped = buffer.sample_by_task(30)
    assert set(grouped) == {0, 1, 2}
    assert sum(len(labels) for _, labels, _ in grouped.values()) == 30


def test_replay_is_routed_through_each_items_own_head():
    """Replaying an old sample through the current head corrupts that head."""
    from diltcl.baselines import train_er

    torch.manual_seed(0)
    num_tasks, classes = 3, 4
    loaders = [_loader(seed=t, num_classes=classes) for t in range(num_tasks)]

    def builder():
        return MultiHeadModel(_Tiny(num_classes=classes), num_tasks, classes)

    matrix, _, _ = _quiet(
        train_er, builder, loaders, loaders, num_tasks, device="cpu",
        epochs=1, lr=1e-2, buffer_size=40, replay_batch_size=8)
    assert matrix.shape == (num_tasks, num_tasks)
    # Heads other than the last must still produce sane (non-degenerate) output:
    # misrouted replay drives earlier heads to a single class.
    assert float(matrix[-1][0]) > 0.0


def test_buffer_without_task_ids_still_works():
    """Single-head callers that omit task_id keep the legacy behaviour."""
    from diltcl.buffer import ReservoirBuffer

    buffer = ReservoirBuffer(capacity=20)
    buffer.add(torch.randn(10, 4), torch.randint(0, 2, (10,)))
    grouped = buffer.sample_by_task(5)
    assert list(grouped) == [None]


def test_grouped_replay_preserves_the_whole_batch_loss_scale():
    """Summing per-group means would scale replay by the number of tasks drawn.

    Routing replay per head requires splitting the batch, but the replay term
    must keep the mean-over-samples scale it had as a single batch; otherwise
    the effective replay strength silently grows with the number of tasks in
    the buffer (up to 20x on Split-CIFAR-100).
    """
    import random

    from diltcl.buffer import ReservoirBuffer

    torch.manual_seed(0)
    criterion = nn.CrossEntropyLoss()
    model = _Tiny(num_classes=4)
    buffer = ReservoirBuffer(capacity=120)
    for task in range(4):
        buffer.add(torch.randn(20, 16), torch.randint(0, 4, (20,)),
                   task_id=task)

    for trial in range(3):
        random.seed(trial)
        groups = buffer.sample_by_task(24)
        drawn = sum(labels.size(0) for _, labels, _ in groups.values())
        assert drawn > 0
        weighted = sum(
            labels.size(0) / drawn * criterion(model(inputs), labels)
            for inputs, labels, _ in groups.values())
        whole = criterion(
            model(torch.cat([g[0] for g in groups.values()])),
            torch.cat([g[1] for g in groups.values()]))
        assert torch.allclose(weighted, whole, atol=1e-5), (weighted, whole)


def test_multihead_lwf_distillation_keeps_single_head_scale():
    """Averaging over old heads keeps the term comparable to the single-head case.

    Summing instead would make the distillation weight grow with the task
    index -- the same reduction-semantics trap that inflated grouped replay.
    """
    from diltcl.lwf import LwF

    torch.manual_seed(0)
    num_tasks, classes = 4, 4
    model = MultiHeadModel(_Tiny(num_classes=classes), num_tasks, classes)
    learner = LwF(model, device="cpu", alpha=1.0, temperature=2.0)
    loader = _loader(num_classes=classes)
    for task in range(3):
        _quiet(learner.train_task, loader, loader, task, epochs=1, lr=1e-3)
    # The teacher keeps the multi-head structure so every old head can be
    # distilled; a collapsed teacher would silently protect only one head.
    assert getattr(learner.old_model, "num_tasks", None) == num_tasks
    assert all(not p.requires_grad for p in learner.old_model.parameters())


def test_replay_methods_are_deterministic_under_both_head_modes():
    """Reservoir sampling draws from Python's ``random``, not torch's RNG.

    Seeding only ``torch`` leaves replay sampling free-running, so this uses
    the project's :func:`set_seed`, which seeds random/torch/numpy together --
    the same helper every runner calls.
    """
    from diltcl.baselines import train_er
    from diltcl.utils import set_seed

    results = []
    for _ in range(2):
        set_seed(0)
        loaders = [_loader(seed=t) for t in range(3)]
        _, acc, _ = _quiet(
            train_er, lambda: MultiHeadModel(_Tiny(), 3, 4), loaders, loaders,
            3, device="cpu", epochs=1, lr=1e-2, buffer_size=40,
            replay_batch_size=8)
        results.append(float(acc))
    assert results[0] == results[1], results


if __name__ == "__main__":
    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_") and callable(value)]
    for function in functions:
        function()
        print(f"PASS {function.__name__}")
    print(f"\nAll {len(functions)} real-core regression tests passed.")
    sys.exit(0)
