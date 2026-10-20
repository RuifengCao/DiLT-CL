"""Device-safe binary-mask operations for DiLT-CL.

A *mask* here is a ``dict`` mapping a parameter name to a tensor of the same
shape as that parameter (``1`` = weight kept/active, ``0`` = pruned/frozen).

Why this module exists
----------------------
The original implementation crashed with::

    RuntimeError: Expected all tensors to be on the same device,
    but found at least two devices, cuda:0 and cpu!

The cause was inconsistent device placement of masks: some masks lived on the
GPU (the core / complement masks) while pruning produced masks on the CPU
(``keep.cpu()``). When the two were combined mid-training the device check
inside PyTorch exploded.

The fix is twofold and lives entirely here:

1. **Consistent policy** - pruning keeps the resulting mask on the *model's*
   device, so masks no longer silently migrate to the CPU.
2. **Defensive alignment** - every operation that combines two tensors first
   moves the operands onto a common device. Even if a caller hands in a
   mixed-device mask, these functions will not raise.

This module deliberately depends only on ``torch`` so it can be imported and
unit-tested in isolation.
"""

import math

import torch


def _require_same_keys(left, right, left_name="left mask", right_name="right mask"):
    """Reject incomplete masks instead of silently activating/dropping weights."""
    left_keys, right_keys = set(left), set(right)
    if left_keys != right_keys:
        missing = sorted(left_keys - right_keys)
        extra = sorted(right_keys - left_keys)
        raise ValueError(
            f"{right_name} keys differ from {left_name}; "
            f"missing={missing}, extra={extra}")


def _require_model_keys(model, mask, mask_name="mask"):
    params = {name: param for name, param in model.named_parameters()
              if param.requires_grad}
    expected = set(params)
    actual = set(mask)
    if actual != expected:
        raise ValueError(
            f"{mask_name} must cover every trainable parameter; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}")
    for name, param in params.items():
        if tuple(mask[name].shape) != tuple(param.shape):
            raise ValueError(
                f"{mask_name}[{name!r}] shape {tuple(mask[name].shape)} "
                f"does not match parameter shape {tuple(param.shape)}")


def iterative_pruning_rate(target_sparsity, rounds):
    """Per-round pruning fraction that reaches a target after ``rounds``.

    If a fraction ``r`` of the active weights is removed each round, the final
    density is ``(1-r)**rounds``.  Solving this identity makes
    ``target_sparsity`` an actual algorithmic target instead of a label on a
    fixed 20%-per-round schedule.
    """
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0, 1)")
    if rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    return 1.0 - math.pow(1.0 - target_sparsity, 1.0 / rounds)


# ---------------------------------------------------------------------------
# Small constructors / converters
# ---------------------------------------------------------------------------

def ones_like_mask(model):
    """Return an all-ones mask covering every trainable parameter."""
    return {
        name: torch.ones_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def clone_mask(mask):
    """Return a deep copy of a mask dict."""
    return {name: tensor.clone() for name, tensor in mask.items()}


def to_device(mask, device):
    """Move every tensor in a mask onto ``device``."""
    return {name: tensor.to(device) for name, tensor in mask.items()}


def _paired_tensors(model, mask, use_grad):
    """``(tensors, masks)`` lists for a fused elementwise multiply.

    Masks are moved to the tensor's device/dtype only when they differ (the
    P0 device fix keeps them aligned, so this is normally a no-op).
    """
    tensors, masks = [], []
    for name, param in model.named_parameters():
        if name not in mask:
            continue
        if use_grad:
            if param.grad is None:
                continue
            tensor = param.grad
        else:
            if not param.requires_grad:
                continue
            tensor = param.data
        m = mask[name]
        if m.device != tensor.device or getattr(m, "dtype", None) != getattr(
                tensor, "dtype", None):
            m = m.to(device=tensor.device, dtype=tensor.dtype)
        tensors.append(tensor)
        masks.append(m)
    return tensors, masks


def apply_parameter_mask_(model, mask):
    """In-place zeroing of parameters outside ``mask``.

    This operation deliberately uses ``param.data``: it is a state constraint,
    not a differentiable layer. Call it before the forward pass and after the
    optimizer step when training a pruned subnetwork.

    With a PyTorch that provides ``torch._foreach_mul_`` the multiplications
    of all parameters are fused into one kernel per device; multiplying by a
    0/1 mask in place yields exactly the values of the v2 out-of-place product
    ``param.data * mask``. Without it (very old builds, the fake-device test
    double) the v2 per-parameter loop is used.
    """
    _require_model_keys(model, mask)
    foreach = getattr(torch, "_foreach_mul_", None)
    if foreach is None:
        for name, param in model.named_parameters():
            if name in mask and param.requires_grad:
                param.data = param.data * mask[name].to(param.device)
        return
    tensors, masks = _paired_tensors(model, mask, use_grad=False)
    if tensors:
        foreach(tensors, masks)


def apply_gradient_mask_(model, mask):
    """In-place zeroing of gradients outside ``mask`` (after ``backward``).

    Parameters without a gradient or without an entry in ``mask`` are left
    untouched, matching the v2 per-parameter loop; the multiplications are
    fused like in :func:`apply_parameter_mask_`.
    """
    foreach = getattr(torch, "_foreach_mul_", None)
    if foreach is None:
        for name, param in model.named_parameters():
            if param.grad is not None and name in mask:
                param.grad *= mask[name].to(param.grad.device)
        return
    tensors, masks = _paired_tensors(model, mask, use_grad=True)
    if tensors:
        foreach(tensors, masks)


# ---------------------------------------------------------------------------
# Density / sparsity measurements
# ---------------------------------------------------------------------------

def mask_density(mask):
    """Fraction of non-zero entries across the whole mask.

    Args:
        mask: Mask dict.

    Returns:
        Density as a float in ``[0, 1]`` (``0.0`` for an empty mask).
    """
    if not mask:
        return 0.0
    total = sum(tensor.numel() for tensor in mask.values())
    nonzero = sum(int((tensor > 0).sum().item()) for tensor in mask.values())
    return nonzero / total if total > 0 else 0.0


def mask_density_in_region(mask, region_mask):
    """Fraction of the *region* that is active in ``mask``.

    Operands are aligned onto the region's device before comparison, which is
    the specific fix for the original cross-device crash.

    Args:
        mask: Mask dict whose density we measure.
        region_mask: Region within which to measure (``1`` = in region).

    Returns:
        Density of ``mask`` restricted to ``region_mask``.
    """
    _require_same_keys(region_mask, mask, "region mask", "mask")
    total_in_region = 0
    active_in_region = 0
    for name, region in region_mask.items():
        tensor = mask[name].to(region.device)          # <-- device alignment
        total_in_region += int((region > 0).sum().item())
        active_in_region += int(((tensor > 0) & (region > 0)).sum().item())
    return (active_in_region / total_in_region
            if total_in_region > 0 else 0.0)


# ---------------------------------------------------------------------------
# Set operations on masks
# ---------------------------------------------------------------------------

def intersection_masks(masks):
    """Element-wise AND of several binary masks.

    A weight is in the result only if it is active in *every* input mask.
    Used to derive the core subnetwork from per-partition IMP masks.

    Args:
        masks: List of mask dicts.

    Returns:
        Intersection mask dict (empty if ``masks`` is empty).
    """
    if not masks:
        return {}

    for i, other in enumerate(masks[1:], start=1):
        _require_same_keys(masks[0], other, "mask 0", f"mask {i}")
        for name in masks[0]:
            if tuple(masks[0][name].shape) != tuple(other[name].shape):
                raise ValueError(f"mask {i}[{name!r}] has incompatible shape")

    result = {}
    for name in masks[0]:
        keep = masks[0][name] > 0
        for other in masks[1:]:
            keep = keep & (other[name].to(keep.device) > 0)
        result[name] = keep.float()
    return result


def complement_mask(mask):
    """Return the complement ``1 - mask`` of a binary mask."""
    return {name: 1.0 - tensor for name, tensor in mask.items()}


def combine_masks(core_mask, expert_mask):
    """Combine core and expert masks via a logical OR.

    A weight is active if it belongs to the core *or* the expert.

    Args:
        core_mask: Core subnetwork mask (may be ``None``).
        expert_mask: Expert subnetwork mask (may be empty).

    Returns:
        Combined mask dict.
    """
    if core_mask is None:
        return expert_mask
    if not expert_mask:
        return core_mask

    _require_same_keys(core_mask, expert_mask, "core mask", "expert mask")
    for name in core_mask:
        if tuple(core_mask[name].shape) != tuple(expert_mask[name].shape):
            raise ValueError(f"expert mask[{name!r}] has incompatible shape")

    combined = {}
    for name, core_tensor in core_mask.items():
        expert_tensor = expert_mask[name].to(core_tensor.device)
        combined[name] = ((core_tensor > 0) | (expert_tensor > 0)).float()
    return combined


# ---------------------------------------------------------------------------
# Forward pass with masking
# ---------------------------------------------------------------------------

def masked_forward(model, x, mask):
    """Run a forward pass with a binary mask applied to the weights.

    Parameters are temporarily multiplied by the mask, the forward pass runs,
    and the original parameter values are restored afterwards.

    Args:
        model: Neural network model.
        x: Input tensor.
        mask: Binary mask dict.

    Returns:
        Model output (logits).
    """
    _require_model_keys(model, mask)
    original_params = {}
    for name, param in model.named_parameters():
        if name in mask and param.requires_grad:
            original_params[name] = param.data.clone()
            param.data = param.data * mask[name].to(param.device)

    try:
        return model(x)
    finally:
        for name, param in model.named_parameters():
            if name in original_params:
                param.data = original_params[name]


# ---------------------------------------------------------------------------
# Iterative magnitude pruning step
# ---------------------------------------------------------------------------

def prune_by_magnitude(model, current_mask, pruning_rate=0.2, region_mask=None):
    """Prune the lowest-magnitude active weights.

    The returned mask lives on each parameter's own device (it is *not* moved
    to the CPU), keeping the device policy consistent throughout training.

    Args:
        model: Model whose current weights define the magnitudes.
        current_mask: Current binary mask.
        pruning_rate: Fraction of currently-active weights to prune.
        region_mask: Optional; restrict pruning to this region.

    Returns:
        Updated binary mask dict (on the model's device).
    """
    if not 0.0 <= pruning_rate <= 1.0:
        raise ValueError("pruning_rate must be in [0, 1]")

    _require_model_keys(model, current_mask, "current_mask")
    if region_mask is not None:
        _require_model_keys(model, region_mask, "region_mask")

    new_mask = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        device = param.device

        curr = current_mask[name].to(device)

        if region_mask is None:
            region = torch.ones_like(param)
        else:
            region = region_mask[name].to(device)

        # Weights eligible for pruning: currently active AND inside the region.
        eligible = ((curr > 0) & (region > 0)).float()

        if eligible.sum().item() == 0:
            new_mask[name] = curr.clone()
            continue

        magnitudes = torch.abs(param.data)
        eligible_vals = magnitudes[eligible > 0]

        if eligible_vals.numel() == 0:
            new_mask[name] = curr.clone()
            continue

        # Select explicit indices rather than a magnitude threshold: ties at
        # the boundary must not prune an uncontrolled number of weights. A
        # zero pruning rate is a true no-op.
        k = int(round(pruning_rate * eligible_vals.numel()))
        if k <= 0:
            new_mask[name] = curr.clone()
            continue
        k = min(k, eligible_vals.numel())
        flat = curr.reshape(-1).clone()
        eligible_idx = (eligible > 0).reshape(-1).nonzero(
            as_tuple=False).squeeze(1)
        prune_rel = torch.topk(eligible_vals, k=k, largest=False,
                               sorted=False).indices
        prune_idx = eligible_idx[prune_rel]
        flat[prune_idx] = 0.0
        new_mask[name] = flat.reshape_as(param)  # stays on model device

    return new_mask
