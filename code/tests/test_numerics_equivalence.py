"""The v3 fast paths must compute the same objective as the v2 (legacy) code.

v3 removes four sources of per-step overhead that dominated wall-clock time on
a fast GPU: per-sample host copies in the replay buffer, one full network
forward per old head in multi-head LwF, an O(T * P) loop in the EWC penalty,
and one forward per task group in multi-head replay. Each replacement is a
mathematically identical function of the same samples and parameters. These
tests pin that claim down with real autograd on CPU:

* identical RNG consumption (same seed -> same retained / replayed items),
* identical loss values up to floating-point tolerance,
* identical parameter gradients up to floating-point tolerance.

Skipped automatically when PyTorch is unavailable.
"""

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch-free environment
    print("PyTorch not installed; numerics equivalence tests skipped.")
    sys.exit(0)

from diltcl.baselines import (  # noqa: E402
    _derpp_terms_grouped,
    _derpp_terms_routed,
    _replay_ce_grouped,
    _replay_ce_routed,
)
from diltcl.buffer import ReservoirBuffer  # noqa: E402
from diltcl.ewc import EWC  # noqa: E402
from diltcl.lwf import LwF  # noqa: E402
from diltcl.metrics import set_model_task  # noqa: E402
from diltcl.models import MultiHeadModel  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self, in_features=16, hidden=24, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


class _LegacyReservoir:
    """Verbatim v2 reservoir logic (item-wise CPU lists) as the reference."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.inputs, self.labels, self.logits, self.task_ids = [], [], [], []
        self.num_seen = 0

    def add(self, inputs, labels, logits=None, task_id=None):
        for i in range(inputs.size(0)):
            x = inputs[i].detach().cpu().clone()
            y = labels[i].detach().cpu().clone()
            z = logits[i].detach().cpu().clone() if logits is not None else None
            if len(self.inputs) < self.capacity:
                self.inputs.append(x); self.labels.append(y)
                self.logits.append(z); self.task_ids.append(task_id)
            else:
                j = random.randint(0, self.num_seen)
                if j < self.capacity:
                    self.inputs[j] = x; self.labels[j] = y
                    self.logits[j] = z; self.task_ids[j] = task_id
            self.num_seen += 1

    def sample_by_task(self, batch_size):
        n = min(batch_size, len(self.inputs))
        idx = random.sample(range(len(self.inputs)), n)
        grouped = {}
        for i in idx:
            grouped.setdefault(self.task_ids[i], []).append(i)
        out = {}
        for task_id, indices in grouped.items():
            out[task_id] = (torch.stack([self.inputs[i] for i in indices]),
                            torch.stack([self.labels[i] for i in indices]),
                            None if any(self.logits[i] is None for i in indices)
                            else torch.stack([self.logits[i] for i in indices]))
        return out


def _stream(seed, steps=12, batch=10, in_features=16, classes=4):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(batch, in_features, generator=g),
             torch.randint(0, classes, (batch,), generator=g),
             torch.randn(batch, classes, generator=g), step % 3)
            for step in range(steps)]


def _grads(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return [p.grad.detach().clone() if p.grad is not None
            else torch.zeros_like(p) for p in model.parameters()]


def _assert_close_grads(a, b, atol=1e-6, rtol=1e-5):
    for ga, gb in zip(a, b):
        assert torch.allclose(ga, gb, atol=atol, rtol=rtol), (ga - gb).abs().max()


# ---------------------------------------------------------------------------
# Reservoir buffer
# ---------------------------------------------------------------------------

def test_device_buffer_matches_legacy_item_wise_reservoir():
    for capacity in (7, 25, 100):
        new = ReservoirBuffer(capacity, device="cpu")
        old = _LegacyReservoir(capacity)
        random.seed(3)
        for x, y, z, task in _stream(11):
            new.add(x, y, logits=z, task_id=task)
        random.seed(3)
        for x, y, z, task in _stream(11):
            old.add(x, y, logits=z, task_id=task)
        assert len(new) == len(old.inputs) and new.num_seen == old.num_seen
        assert new.task_ids == old.task_ids
        for i in range(len(old.inputs)):
            assert torch.equal(new.inputs[i], old.inputs[i])
            assert torch.equal(new.labels[i], old.labels[i])
            assert torch.equal(new.logits[i], old.logits[i])
        for draw in range(5):
            random.seed(100 + draw)
            got = new.sample_by_task(9)
            random.seed(100 + draw)
            expected = old.sample_by_task(9)
            assert list(got) == list(expected)
            for task in expected:
                for got_t, exp_t in zip(got[task], expected[task]):
                    assert torch.equal(got_t, exp_t)


def test_buffer_last_write_wins_within_a_batch():
    """Two items of one batch that pick the same slot: the later one stays."""
    buffer = ReservoirBuffer(2, device="cpu")
    buffer.add(torch.tensor([[1.0], [2.0]]), torch.tensor([1, 2]), task_id=0)
    random.seed(0)
    draws = [random.randint(0, 2 + i) for i in range(4)]
    random.seed(0)
    buffer.add(torch.tensor([[3.0], [4.0], [5.0], [6.0]]),
               torch.tensor([3, 4, 5, 6]), task_id=1)
    expected = [torch.tensor([1.0]), torch.tensor([2.0])]
    labels = [1, 2]
    for i, j in enumerate(draws):
        if j < 2:
            expected[j] = torch.tensor([3.0 + i])
            labels[j] = 3 + i
    for slot in range(2):
        assert torch.equal(buffer.inputs[slot], expected[slot])
        assert int(buffer.labels[slot]) == labels[slot]
    assert buffer.num_seen == 6


def test_zero_capacity_buffer_consumes_rng_like_legacy():
    random.seed(5)
    new = ReservoirBuffer(0, device="cpu")
    new.add(torch.randn(4, 2), torch.zeros(4, dtype=torch.long))
    state_new = random.getstate()
    random.seed(5)
    old = _LegacyReservoir(0)
    old.add(torch.randn(4, 2), torch.zeros(4, dtype=torch.long))
    assert random.getstate() == state_new
    assert new.is_empty() and new.num_seen == 4


# ---------------------------------------------------------------------------
# Multi-head model helpers
# ---------------------------------------------------------------------------

def test_stacked_and_routed_heads_match_per_head_forward():
    torch.manual_seed(0)
    model = MultiHeadModel(_Tiny(), num_tasks=5, classes_per_task=4)
    x = torch.randn(9, 16)
    features = model.features(x)
    stacked = model.stacked_head_logits(features, [0, 2, 4])
    for k, task in enumerate([0, 2, 4]):
        set_model_task(model, task)
        assert torch.allclose(stacked[:, k], model(x), atol=1e-6)
    task_ids = torch.tensor([1, 3, 3, 1, 1, -1, -1, -1, -1])
    set_model_task(model, 2)
    routed = model.forward_routed(x, task_ids)
    for task in (1, 3, -1):
        rows = (task_ids == task).nonzero().squeeze(1)
        set_model_task(model, 2 if task < 0 else task)
        assert torch.allclose(routed[rows], model(x[rows]), atol=1e-6)
    # Gradients reach every head that was used, and only those.
    routed.sum().backward()
    assert model.heads[1].weight.grad is not None and model.heads[3].weight.grad is not None
    assert model.heads[2].weight.grad is not None      # active head via -1
    assert model.heads[0].weight.grad is None or float(model.heads[0].weight.grad.abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# Replay terms
# ---------------------------------------------------------------------------

def _filled_buffer(seed=1, capacity=40):
    random.seed(seed)
    buffer = ReservoirBuffer(capacity, device="cpu")
    for x, y, z, task in _stream(seed, steps=9):
        buffer.add(x, y, logits=z, task_id=task)
    return buffer


def test_routed_replay_equals_grouped_replay():
    torch.manual_seed(2)
    model = MultiHeadModel(_Tiny(), num_tasks=3, classes_per_task=4)
    criterion = nn.CrossEntropyLoss()
    for draw in range(4):
        buffer = _filled_buffer()
        random.seed(50 + draw)
        set_model_task(model, 2)
        fast = _replay_ce_routed(model, buffer, 16, criterion)
        fast_state = random.getstate()
        buffer = _filled_buffer()
        random.seed(50 + draw)
        set_model_task(model, 2)
        slow = _replay_ce_grouped(model, buffer, 16, criterion, 2)
        assert random.getstate() == fast_state
        assert torch.allclose(fast, slow, atol=1e-6), (fast, slow)
        _assert_close_grads(_grads(model, fast), _grads(model, slow))


def test_routed_derpp_terms_equal_grouped_terms():
    torch.manual_seed(4)
    model = MultiHeadModel(_Tiny(), num_tasks=3, classes_per_task=4)
    criterion = nn.CrossEntropyLoss()
    for draw in range(3):
        buffer = _filled_buffer(seed=7)
        random.seed(70 + draw)
        set_model_task(model, 1)
        fast = _derpp_terms_routed(model, buffer, 12, criterion, 0.5, 0.5)
        fast_state = random.getstate()
        buffer = _filled_buffer(seed=7)
        random.seed(70 + draw)
        set_model_task(model, 1)
        slow = _derpp_terms_grouped(model, buffer, 12, criterion, 0.5, 0.5, 1)
        assert random.getstate() == fast_state
        assert torch.allclose(fast, slow, atol=1e-6), (fast, slow)
        _assert_close_grads(_grads(model, fast), _grads(model, slow))


def test_routed_replay_on_single_head_model_matches_plain_forward():
    torch.manual_seed(3)
    model = _Tiny()
    criterion = nn.CrossEntropyLoss()
    buffer = _filled_buffer(seed=2)
    random.seed(9)
    fast = _replay_ce_routed(model, buffer, 10, criterion)
    buffer = _filled_buffer(seed=2)
    random.seed(9)
    slow = _replay_ce_grouped(model, buffer, 10, criterion, 0)
    assert torch.allclose(fast, slow, atol=1e-6)


# ---------------------------------------------------------------------------
# Fused mask application (DiLT-CL inner loop)
# ---------------------------------------------------------------------------

def test_fused_mask_application_matches_v2_loops_bit_for_bit():
    from diltcl import masking

    torch.manual_seed(5)
    fast_model, slow_model = _Tiny(), _Tiny()
    slow_model.load_state_dict(fast_model.state_dict())
    g = torch.Generator().manual_seed(2)
    mask = {name: (torch.rand(p.shape, generator=g) > 0.4).float()
            for name, p in fast_model.named_parameters()}
    grad_mask = {k: v for k, v in mask.items() if "fc3" not in k}  # partial
    x = torch.randn(6, 16, generator=g)
    for model in (fast_model, slow_model):
        model(x).sum().backward()
        model.fc2.bias.grad = None            # a parameter without gradient
    # v2 loops (reference)
    for name, param in slow_model.named_parameters():
        if param.grad is not None and name in grad_mask:
            param.grad *= grad_mask[name].to(param.device)
    for name, param in slow_model.named_parameters():
        if name in mask and param.requires_grad:
            param.data = param.data * mask[name].to(param.device)
    # fused v3 paths
    masking.apply_gradient_mask_(fast_model, grad_mask)
    masking.apply_parameter_mask_(fast_model, mask)
    for (name, a), (_, b) in zip(fast_model.named_parameters(),
                                 slow_model.named_parameters()):
        assert torch.equal(a.data, b.data), name
        if a.grad is None:
            assert b.grad is None
        else:
            assert torch.equal(a.grad, b.grad), name


# ---------------------------------------------------------------------------
# LwF
# ---------------------------------------------------------------------------

def test_shared_feature_lwf_distillation_matches_per_head_loop():
    torch.manual_seed(8)
    model = MultiHeadModel(_Tiny(), num_tasks=6, classes_per_task=4)
    learner = LwF(model, device="cpu", alpha=0.7, temperature=2.0)
    # Perturb the "old" copy so student and teacher differ.
    learner.save_old_model()
    with torch.no_grad():
        for p in learner.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.randn(13, 16)
    for task_id in (1, 3, 5):
        set_model_task(model, task_id)
        fast = learner._multihead_distill_shared(x, task_id)
        slow = learner._multihead_distill_per_head(x, task_id)
        assert torch.allclose(fast, slow, atol=1e-6, rtol=1e-5), (fast, slow)
        _assert_close_grads(_grads(model, fast), _grads(model, slow),
                            atol=1e-6, rtol=1e-4)


# ---------------------------------------------------------------------------
# EWC
# ---------------------------------------------------------------------------

def test_consolidated_ewc_penalty_has_the_per_task_gradient():
    torch.manual_seed(6)
    model = _Tiny()
    fast = EWC(model, device="cpu", lambda_ewc=3.0, penalty_form="consolidated")
    slow = EWC(model, device="cpu", lambda_ewc=3.0, penalty_form="per_task")
    g = torch.Generator().manual_seed(1)
    for task in range(4):
        fisher = {name: torch.rand(p.shape, generator=g) * (torch.rand(
            p.shape, generator=g) > 0.3)          # some exact zeros
                  for name, p in model.named_parameters()}
        anchor = {name: torch.randn(p.shape, generator=g)
                  for name, p in model.named_parameters()}
        for learner in (fast, slow):
            learner.optimal_params[task] = {k: v.clone() for k, v in anchor.items()}
            learner.fisher_dict[task] = {k: v.clone() for k, v in fisher.items()}
            learner._consolidate(task)
    values = []
    for trial in range(2):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1 * torch.randn(p.shape, generator=g))
        fast_loss, slow_loss = fast.penalty(model), slow.penalty(model)
        _assert_close_grads(_grads(model, fast_loss), _grads(model, slow_loss),
                            atol=1e-5, rtol=1e-4)
        values.append(float(slow_loss.detach()) - float(fast_loss.detach()))
    # The two forms differ by a parameter-independent constant.
    assert abs(values[0] - values[1]) < 1e-3 * (1.0 + abs(values[0])), values


def test_flat_consolidated_penalty_has_bit_identical_gradients():
    """The single-vector quadratic must give exactly the per-parameter
    gradients (elementwise arithmetic is identical); the loss value may differ
    only in summation order."""
    torch.manual_seed(9)
    model = _Tiny()
    learner = EWC(model, device="cpu", lambda_ewc=2.0)
    g = torch.Generator().manual_seed(3)
    for task in range(3):
        learner.optimal_params[task] = {n: torch.randn(p.shape, generator=g)
                                        for n, p in model.named_parameters()}
        learner.fisher_dict[task] = {n: torch.rand(p.shape, generator=g)
                                     for n, p in model.named_parameters()}
        learner._consolidate(task)
    for _ in range(2):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1 * torch.randn(p.shape, generator=g))
        fast_grads = _grads(model, learner._penalty_consolidated(model))
        slow_grads = _grads(model, learner._penalty_consolidated_per_parameter(model))
        for a, b in zip(fast_grads, slow_grads):
            assert torch.equal(a, b)
        fast = float(learner._penalty_consolidated(model).detach())
        slow = float(learner._penalty_consolidated_per_parameter(model).detach())
        assert abs(fast - slow) <= 1e-5 * (1.0 + abs(slow))
    # A new task invalidates the cached flat vectors.
    learner.optimal_params[3] = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    learner.fisher_dict[3] = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    learner._consolidate(3)
    assert learner._flat_cache is None
    _ = learner._penalty_consolidated(model)
    assert learner._flat_cache is not None


def test_ewc_consolidates_regardless_of_call_order():
    torch.manual_seed(0)
    model = _Tiny()
    loader = [(torch.randn(8, 16), torch.randint(0, 4, (8,)))]
    a = EWC(model, device="cpu")
    a.save_optimal_params(0)
    a.compute_fisher(loader, 0, num_samples=4)
    b = EWC(model, device="cpu")
    b.compute_fisher(loader, 0, num_samples=4)
    b.save_optimal_params(0)
    for name in a._fisher_sum:
        assert torch.equal(a._fisher_sum[name], b._fisher_sum[name])
        assert torch.equal(a._anchor[name], b._anchor[name])
    assert a._consolidated_tasks == b._consolidated_tasks == {0}


def test_ewc_fisher_label_source_is_explicit_and_recorded():
    """'prediction' (v2 behaviour) and 'true' (empirical Fisher) are different
    estimators; both must be selectable, validated, and 'prediction' must be
    the default so existing configs keep their numerics."""
    torch.manual_seed(1)
    model = _Tiny()
    loader = [(torch.randn(6, 16), torch.randint(0, 4, (6,)))]
    fishers = {}
    for source in ("prediction", "true"):
        learner = EWC(model, device="cpu", fisher_labels=source)
        learner.compute_fisher(loader, 0, num_samples=6)
        fishers[source] = learner.fisher_dict[0]
    assert EWC(model, device="cpu").fisher_labels == "prediction"
    # A randomly initialised model rarely predicts the true label, so the two
    # estimators must differ on at least one parameter.
    assert any(not torch.equal(fishers["prediction"][k], fishers["true"][k])
               for k in fishers["prediction"])
    try:
        EWC(model, device="cpu", fisher_labels="argmax")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown fisher_labels must be rejected")


def test_ewc_penalty_is_zero_before_any_task_is_stored():
    model = _Tiny()
    assert EWC(model, device="cpu").penalty(model) == 0.0
    assert EWC(model, device="cpu", penalty_form="per_task").penalty(model) == 0.0


if __name__ == "__main__":
    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_") and callable(value)]
    for function in functions:
        function()
        print(f"PASS {function.__name__}")
    print(f"\nAll {len(functions)} numerics equivalence tests passed.")
