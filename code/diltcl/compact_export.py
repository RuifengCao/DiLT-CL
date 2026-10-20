"""Compact deployment export for DiLT-CL checkpoints, with a correctness gate.

The paper's storage claim compares an *analytic* compact-format estimate
(~5.8 MiB on Split-CIFAR-100) against baselines. The artifact actually written
to disk is ~100 MiB -- larger than the Independent upper bound it is compared
against -- because every expert mask is serialised as a dense ``float32``
tensor and the expert weights are stored densely.

This module closes that gap. It exports the real inference state in a compact
form and provides :func:`verify_roundtrip_logits` so the compact
export is only ever called deployable after it reproduces the dense model's
logits exactly.

Compaction applied
------------------
* **Masks** -> packed bits (``numpy.packbits``), 32x smaller than ``float32``.
* **Expert weights** -> only the values selected by the expert mask, stored as
  a flat ``float32`` vector plus the packed mask needed to scatter them back.
* **Core** -> the frozen core values restricted to the core mask, stored once.

Nothing here changes training or evaluation; it is a serialisation format.
"""

import io

import numpy as np
import torch


FORMAT_VERSION = "diltcl-compact-v1"


def _pack_mask(mask_tensor):
    """Pack a binary mask into bits plus the metadata needed to restore it."""
    flat = (mask_tensor.detach().cpu().numpy().reshape(-1) > 0)
    return {
        "packed": np.packbits(flat),
        "numel": int(flat.size),
        "shape": tuple(int(s) for s in mask_tensor.shape),
    }


def _unpack_mask(record, device="cpu"):
    """Restore a boolean mask tensor from :func:`_pack_mask` output."""
    flat = np.unpackbits(record["packed"])[: record["numel"]].astype(bool)
    tensor = torch.from_numpy(flat.reshape(record["shape"]).copy())
    return tensor.to(device)


def export_diltcl_compact(diltcl):
    """Build a compact, serialisable representation of a trained DiLT-CL model.

    Args:
        diltcl: A trained :class:`diltcl.diltcl.DiLTCL` instance.

    Returns:
        Dict suitable for ``torch.save``. Contains packed masks, the core
        values restricted to the core mask, and per-task expert values
        restricted to each expert mask.
    """
    if diltcl.core_mask is None:
        raise ValueError("core has not been extracted yet")

    core_mask = {name: _pack_mask(tensor)
                 for name, tensor in diltcl.core_mask.items()}
    core_values = {}
    for name, tensor in diltcl.frozen_core_values.items():
        selection = diltcl.core_mask.get(name)
        if selection is None:
            continue
        picked = tensor.detach().cpu().reshape(-1)[
            selection.detach().cpu().reshape(-1) > 0]
        core_values[name] = picked.to(torch.float32).numpy()

    experts = {}
    for task_id, mask in diltcl.expert_masks.items():
        weights = diltcl.expert_weights.get(task_id, {})
        packed_mask, values = {}, {}
        for name, mask_tensor in mask.items():
            packed_mask[name] = _pack_mask(mask_tensor)
            weight_tensor = weights.get(name)
            if weight_tensor is None:
                continue
            picked = weight_tensor.detach().cpu().reshape(-1)[
                mask_tensor.detach().cpu().reshape(-1) > 0]
            values[name] = picked.to(torch.float32).numpy()
        experts[int(task_id)] = {"mask": packed_mask, "values": values}

    return {
        "format": FORMAT_VERSION,
        "core_mask": core_mask,
        "core_values": core_values,
        "experts": experts,
        "parameter_shapes": {
            name: tuple(int(s) for s in tensor.shape)
            for name, tensor in diltcl.core_mask.items()},
    }


def compact_state_dict_for_task(compact, task_id, device="cpu"):
    """Reconstruct the dense parameter values used for ``task_id``.

    This mirrors what :meth:`DiLTCL.forward_with_mask` loads: the frozen core
    plus that task's expert.
    """
    dense = {}
    for name, mask_record in compact["core_mask"].items():
        shape = compact["parameter_shapes"][name]
        flat = torch.zeros(int(np.prod(shape)), dtype=torch.float32)
        core_sel = _unpack_mask(mask_record).reshape(-1)
        values = compact["core_values"].get(name)
        if values is not None:
            flat[core_sel] = torch.from_numpy(values)
        dense[name] = flat.reshape(shape)

    expert = compact["experts"].get(int(task_id))
    if expert is not None:
        for name, mask_record in expert["mask"].items():
            if name not in dense:
                continue
            values = expert["values"].get(name)
            if values is None:
                continue
            selection = _unpack_mask(mask_record).reshape(-1)
            flat = dense[name].reshape(-1).clone()
            flat[selection] = torch.from_numpy(values)
            dense[name] = flat.reshape(dense[name].shape)

    return {name: tensor.to(device) for name, tensor in dense.items()}


@torch.no_grad()
def verify_roundtrip_logits(diltcl, compact, sample_batch, tasks=None,
                            atol=0.0, rtol=0.0):
    """Check that the compact export reproduces the dense model's logits.

    The storage claim is only meaningful if the compact artifact is actually
    usable, so this comparison is the gate that lets it be called deployable.

    Args:
        diltcl: The trained model the export came from.
        compact: Output of :func:`export_diltcl_compact`.
        sample_batch: An input batch to compare logits on.
        tasks: Iterable of task ids to check (default: every exported task).
        atol / rtol: Tolerances; the default demands exact equality.

    Returns:
        Dict with ``ok`` and per-task maximum absolute differences.
    """
    if tasks is None:
        tasks = sorted(compact["experts"]) or [0]
    device = diltcl.device
    inputs = sample_batch.to(device)
    model = diltcl.model
    saved = {name: param.data.clone()
             for name, param in model.named_parameters()
             if param.requires_grad}

    diffs = {}
    try:
        for task_id in tasks:
            reference = diltcl.forward_with_mask(inputs, task_id)
            dense = compact_state_dict_for_task(compact, task_id, device=device)
            for name, param in model.named_parameters():
                if name in dense:
                    param.data = dense[name]
            restored = model(inputs)
            diffs[int(task_id)] = float(
                (reference - restored).abs().max().item())
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in saved:
                    param.data = saved[name]

    tolerance = max(atol, rtol)
    return {"ok": all(value <= tolerance for value in diffs.values()),
            "max_abs_diff_per_task": diffs,
            "tolerance": tolerance}


def compact_size_bytes(compact):
    """Serialized size of the compact export, in bytes."""
    buffer = io.BytesIO()
    torch.save(compact, buffer)
    return buffer.getbuffer().nbytes


def dense_size_bytes(diltcl):
    """Serialized size of the equivalent dense state, in bytes."""
    payload = {
        "core_mask": {k: v.detach().cpu() for k, v in diltcl.core_mask.items()},
        "frozen_core_values": {k: v.detach().cpu()
                               for k, v in diltcl.frozen_core_values.items()},
        "expert_masks": {t: {k: v.detach().cpu() for k, v in m.items()}
                         for t, m in diltcl.expert_masks.items()},
        "expert_weights": {t: {k: v.detach().cpu() for k, v in w.items()}
                           for t, w in diltcl.expert_weights.items()},
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getbuffer().nbytes
