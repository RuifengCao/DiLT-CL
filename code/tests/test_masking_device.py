"""Standalone device-safety test for :mod:`diltcl.masking`.

This test reproduces the original crash condition *without needing a real
PyTorch install or a GPU*. It injects a tiny fake ``torch`` whose tensors carry
a ``.device`` label and raise the real PyTorch error message whenever an
operation mixes two devices::

    RuntimeError: Expected all tensors to be on the same device ...

``diltcl/masking.py`` is then loaded in isolation (it imports only ``torch``)
and exercised on the exact code path that used to crash
(``mask_density_in_region`` with a CPU mask and a CUDA region).

What it proves
--------------
1. The fixed mask ops do NOT raise on mixed-device inputs and return correct
   values (because they align devices internally).
2. ``prune_by_magnitude`` returns masks on the *model's* device, never forced
   to CPU - the device-policy half of the fix.
3. The ORIGINAL un-aligned expression genuinely DOES raise under the same fake,
   confirming the test models the real bug (and that the fix is what avoids it).

Run with::

    python tests/test_masking_device.py
"""

import importlib.util
import os
import sys
import types

import numpy as np


# ---------------------------------------------------------------------------
# A minimal fake `torch` that models per-device tensors.
# ---------------------------------------------------------------------------

_DEVICE_ERROR = ("Expected all tensors to be on the same device, "
                 "but found at least two devices")


def _dev(x):
    """Best-effort device label of an operand (scalars are device-neutral)."""
    return getattr(x, "device", None)


def _check_same_device(a, b):
    """Raise the PyTorch-style error if two tensors live on different devices."""
    da, db = _dev(a), _dev(b)
    if da is not None and db is not None and da != db:
        raise RuntimeError(f"{_DEVICE_ERROR}, {da} and {db}!")


class FakeTensor:
    """NumPy-backed tensor carrying a ``.device`` label."""

    def __init__(self, array, device="cpu"):
        self.a = np.asarray(array)
        self.device = device

    # -- shape / scalar helpers ------------------------------------------
    def numel(self):
        return int(self.a.size)

    @property
    def shape(self):
        return self.a.shape

    def __len__(self):
        return int(self.a.shape[0]) if self.a.ndim else 0

    def item(self):
        return self.a.item()

    def clone(self):
        return FakeTensor(self.a.copy(), self.device)

    def reshape(self, *shape):
        return FakeTensor(self.a.reshape(*shape), self.device)

    def reshape_as(self, other):
        src = other.data if isinstance(other, FakeParam) else other
        return FakeTensor(self.a.reshape(src.a.shape), self.device)

    def nonzero(self, as_tuple=False):
        idx = np.argwhere(self.a != 0)
        if as_tuple:
            return tuple(FakeTensor(x, self.device) for x in idx.T)
        return FakeTensor(idx, self.device)

    def squeeze(self, axis=None):
        return FakeTensor(np.squeeze(self.a, axis=axis), self.device)

    def to(self, device):
        return FakeTensor(self.a.copy(), device)

    def float(self):
        return FakeTensor(self.a.astype(np.float64), self.device)

    def sum(self):
        return FakeTensor(np.array(self.a.sum()), self.device)

    def max(self):
        return FakeTensor(np.array(self.a.max()), self.device)

    def abs(self):
        return FakeTensor(np.abs(self.a), self.device)

    # -- elementwise binary ops (device-checked) -------------------------
    def _binop(self, other, op):
        if isinstance(other, FakeTensor):
            _check_same_device(self, other)
            return FakeTensor(op(self.a, other.a), self.device)
        return FakeTensor(op(self.a, other), self.device)

    def __mul__(self, other):
        return self._binop(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self._binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self._binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        # e.g. ``1.0 - mask`` (complement); scalar LHS => self.device.
        return FakeTensor(other - self.a, self.device)

    def __gt__(self, other):
        return self._binop(other, lambda x, y: x > y)

    def __and__(self, other):
        return self._binop(other, lambda x, y: x & y)

    def __or__(self, other):
        return self._binop(other, lambda x, y: x | y)

    # -- boolean-mask indexing (device-checked) --------------------------
    def __getitem__(self, index):
        if isinstance(index, FakeTensor):
            _check_same_device(self, index)
            if np.issubdtype(index.a.dtype, np.bool_):
                selected = self.a[index.a]
            else:
                selected = self.a[index.a.astype(int)]
            return FakeTensor(selected, self.device)
        return FakeTensor(self.a[index], self.device)

    def __setitem__(self, index, value):
        if isinstance(index, FakeTensor):
            _check_same_device(self, index)
            idx = (index.a if np.issubdtype(index.a.dtype, np.bool_)
                   else index.a.astype(int))
            self.a[idx] = value.a if isinstance(value, FakeTensor) else value
        else:
            self.a[index] = value.a if isinstance(value, FakeTensor) else value


class FakeParam:
    """Stand-in for an ``nn.Parameter``."""

    def __init__(self, array, device="cpu", requires_grad=True):
        self.data = FakeTensor(array, device)
        self.device = device
        self.requires_grad = requires_grad

    @property
    def shape(self):
        return self.data.shape


class FakeModel:
    """Minimal model exposing ``named_parameters``."""

    def __init__(self, params):
        self._params = params  # list of (name, FakeParam)

    def named_parameters(self):
        return list(self._params)


def _like(x, fill):
    """Return a FakeTensor shaped like ``x`` (FakeParam or FakeTensor)."""
    src = x.data if isinstance(x, FakeParam) else x
    return FakeTensor(np.full(src.a.shape, fill, dtype=np.float64), src.device)


def _build_fake_torch():
    """Assemble a module object that looks enough like ``torch`` for masking."""
    fake = types.ModuleType("torch")
    fake.ones_like = lambda x: _like(x, 1.0)
    fake.zeros_like = lambda x: _like(x, 0.0)
    fake.abs = lambda x: x.abs()

    def topk(t, k, largest=True, sorted=True):
        order = np.argsort(t.a.ravel())
        chosen = order[-k:] if largest else order[:k]
        if sorted and largest:
            chosen = chosen[::-1]
        return types.SimpleNamespace(
            values=FakeTensor(t.a.ravel()[chosen], t.device),
            indices=FakeTensor(chosen.astype(int), t.device))

    fake.topk = topk
    return fake


# ---------------------------------------------------------------------------
# Load diltcl/masking.py in isolation against the fake torch.
# ---------------------------------------------------------------------------

def _load_masking_module():
    """Load ``diltcl.masking`` against the fake torch, then restore the real one.

    Leaving the fake in ``sys.modules`` would break every later test in the
    same pytest session that imports the real ``torch``.
    """
    real_torch = sys.modules.get("torch")
    sys.modules["torch"] = _build_fake_torch()
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        masking_path = os.path.join(here, os.pardir, "diltcl", "masking.py")
        spec = importlib.util.spec_from_file_location("masking_under_test",
                                                      masking_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        else:
            sys.modules.pop("torch", None)
    return module


try:  # pytest fixture so the same functions run under ``pytest`` and as a script
    import pytest

    @pytest.fixture(scope="module")
    def masking():
        return _load_masking_module()
except ImportError:  # pragma: no cover - script mode without pytest
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _t(name):
    print(f"  [running] {name}")


def test_density_in_region_mixed_devices(masking):
    """CPU mask + CUDA region must NOT raise and must be correct."""
    _t("mask_density_in_region: CPU mask vs CUDA region")
    mask = {"w": FakeTensor([1, 0, 1, 0, 1, 1], device="cpu")}
    region = {"w": FakeTensor([1, 1, 1, 1, 0, 0], device="cuda:0")}

    density = masking.mask_density_in_region(mask, region)

    # region active at indices {0,1,2,3}; mask active there at {0,2} -> 2/4.
    assert abs(density - 0.5) < 1e-9, density


def test_combine_masks_mixed_devices(masking):
    """CUDA core OR CPU expert must NOT raise; result on the core's device."""
    _t("combine_masks: CUDA core vs CPU expert")
    core = {"w": FakeTensor([1, 0, 0, 0], device="cuda:0")}
    expert = {"w": FakeTensor([0, 1, 1, 0], device="cpu")}

    combined = masking.combine_masks(core, expert)

    assert combined["w"].device == "cuda:0", combined["w"].device
    assert list(combined["w"].a.astype(int)) == [1, 1, 1, 0]


def test_intersection_mixed_devices(masking):
    """Intersection of CUDA and CPU masks must NOT raise."""
    _t("intersection_masks: CUDA & CPU")
    a = {"w": FakeTensor([1, 1, 0, 1], device="cuda:0")}
    b = {"w": FakeTensor([1, 0, 0, 1], device="cpu")}

    inter = masking.intersection_masks([a, b])

    assert list(inter["w"].a.astype(int)) == [1, 0, 0, 1]


def test_prune_keeps_model_device(masking):
    """prune_by_magnitude must return masks on the model's device, not CPU."""
    _t("prune_by_magnitude: output stays on model device (cuda)")
    param = FakeParam(np.array([0.05, 0.9, 0.2, 0.7, 0.01, 0.6]),
                      device="cuda:0")
    model = FakeModel([("w", param)])
    current = {"w": FakeTensor(np.ones(6), device="cuda:0")}

    new_mask = masking.prune_by_magnitude(model, current, pruning_rate=0.5)

    assert new_mask["w"].device == "cuda:0", new_mask["w"].device
    # 6 active * 0.5 -> prune 3 smallest (0.01,0.05,0.2); keep 3.
    assert int(new_mask["w"].a.sum()) == 3, new_mask["w"].a


def test_zero_pruning_is_noop(masking):
    """A zero pruning rate must not remove the smallest weight."""
    _t("prune_by_magnitude: zero rate is a no-op")
    param = FakeParam(np.array([0.05, 0.9, 0.2]), device="cuda:0")
    model = FakeModel([("w", param)])
    current = {"w": FakeTensor(np.ones(3), device="cuda:0")}
    new_mask = masking.prune_by_magnitude(model, current, pruning_rate=0.0)
    assert int(new_mask["w"].a.sum()) == 3, new_mask["w"].a


def test_pruning_ties_remove_exact_count(masking):
    """Equal magnitudes at the boundary must not empty the whole tensor."""
    _t("prune_by_magnitude: ties remove the exact requested count")
    param = FakeParam(np.array([1.0, 1.0, 1.0, 1.0]), device="cuda:0")
    model = FakeModel([("w", param)])
    current = {"w": FakeTensor(np.ones(4), device="cuda:0")}
    new_mask = masking.prune_by_magnitude(model, current, pruning_rate=0.5)
    assert int(new_mask["w"].a.sum()) == 2, new_mask["w"].a


def test_target_aware_pruning_schedule(masking):
    """Five equal rounds must compose to the requested 90% sparsity."""
    _t("iterative_pruning_rate: composes to target sparsity")
    rate = masking.iterative_pruning_rate(0.9, 5)
    final_density = (1.0 - rate) ** 5
    assert abs(final_density - 0.1) < 1e-12, final_density


def test_target_aware_schedule_reaches_mask_count(masking):
    """Production pruning steps reach 90% within integer rounding tolerance."""
    _t("prune_by_magnitude: five target-aware rounds reach 90%")
    n = 1000
    param = FakeParam(np.linspace(0.001, 1.0, n), device="cuda:0")
    model = FakeModel([("w", param)])
    mask = {"w": FakeTensor(np.ones(n), device="cuda:0")}
    rate = masking.iterative_pruning_rate(0.9, 5)
    for _ in range(5):
        mask = masking.prune_by_magnitude(model, mask, pruning_rate=rate)
    density = float(mask["w"].a.mean())
    assert abs(density - 0.10) <= 0.002, density


def test_apply_parameter_mask_zeros_inactive_weights(masking):
    """The production forward-state helper must remove pruned weights."""
    _t("apply_parameter_mask_: pruned weights are absent from forward state")
    param = FakeParam(np.array([2.0, -3.0, 5.0, 7.0]), device="cuda:0")
    model = FakeModel([("w", param)])
    mask = {"w": FakeTensor([1, 0, 1, 0], device="cpu")}
    masking.apply_parameter_mask_(model, mask)
    assert list(param.data.a) == [2.0, 0.0, 5.0, 0.0], param.data.a


def test_prune_with_region_mixed_devices(masking):
    """prune within a region whose mask is on a different device must work."""
    _t("prune_by_magnitude: region on a different device")
    param = FakeParam(np.array([0.05, 0.9, 0.2, 0.7]), device="cuda:0")
    model = FakeModel([("w", param)])
    current = {"w": FakeTensor(np.ones(4), device="cuda:0")}
    region = {"w": FakeTensor(np.array([1, 1, 1, 0]), device="cpu")}  # CPU region

    new_mask = masking.prune_by_magnitude(
        model, current, pruning_rate=0.5, region_mask=region
    )

    assert new_mask["w"].device == "cuda:0", new_mask["w"].device


def test_incomplete_masks_are_rejected(masking):
    """Missing parameter keys must fail loudly instead of activating weights."""
    _t("mask validation: incomplete key sets are rejected")
    model = FakeModel([
        ("w", FakeParam(np.array([1.0, 2.0]), device="cpu")),
        ("b", FakeParam(np.array([0.0]), device="cpu")),
    ])
    bad = {"w": FakeTensor([1, 0], device="cpu")}
    for fn in (
        lambda: masking.apply_parameter_mask_(model, bad),
        lambda: masking.prune_by_magnitude(model, bad, 0.5),
    ):
        try:
            fn()
        except ValueError as exc:
            assert "missing" in str(exc)
        else:
            raise AssertionError("incomplete mask was accepted")


def test_original_buggy_expression_raises(masking):
    """The ORIGINAL un-aligned expression must raise under the same fake.

    This is the exact operation from the crash (line 958 of the monolith):
        ((m > 0).float() * (reg > 0).float()).sum().item()
    with ``m`` on CPU and ``reg`` on CUDA. If this did NOT raise, the test
    harness wouldn't actually be modelling the bug.
    """
    _t("control: original cross-device expression DOES raise")
    m = FakeTensor([1, 0, 1, 1], device="cpu")
    reg = FakeTensor([1, 1, 1, 0], device="cuda:0")

    raised = False
    try:
        _ = ((m > 0).float() * (reg > 0).float()).sum().item()
    except RuntimeError as exc:
        raised = True
        assert "same device" in str(exc), str(exc)
    assert raised, "expected a device-mismatch RuntimeError but none was raised"


def main():
    print("=" * 64)
    print("Device-safety tests for diltcl.masking (fake-torch harness)")
    print("=" * 64)

    masking = _load_masking_module()

    tests = [
        test_density_in_region_mixed_devices,
        test_combine_masks_mixed_devices,
        test_intersection_mixed_devices,
        test_prune_keeps_model_device,
        test_zero_pruning_is_noop,
        test_pruning_ties_remove_exact_count,
        test_target_aware_pruning_schedule,
        test_target_aware_schedule_reaches_mask_count,
        test_apply_parameter_mask_zeros_inactive_weights,
        test_prune_with_region_mixed_devices,
        test_incomplete_masks_are_rejected,
        test_original_buggy_expression_raises,
    ]

    passed = 0
    for test in tests:
        test(masking)
        passed += 1

    print("-" * 64)
    print(f"All {passed}/{len(tests)} device-safety tests PASSED.")
    print("The cross-device crash path is fixed; original logic still crashes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
