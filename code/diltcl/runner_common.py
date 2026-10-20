"""Shared helpers for the experiment runners.

``run_task_il.py`` and ``run_class_il.py`` grew independent copies of
the same seed parsing, config-override loading, CUDA synchronisation and
checkpoint hashing/persistence. Five of those copies were byte-identical and
the rest had already begun to drift apart in documentation. Any fix applied to
one runner had to be mirrored by hand into the other.

Centralising them here removes that duplication: both runners import the same
implementation, so checkpoint hashes and seed semantics cannot silently diverge.
"""

import hashlib
import json
import os

import numpy as np
import torch


def _cuda_sync(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

def _load_config_overrides(cfg, path):
    if path is None:
        return cfg.validate()
    with open(path, encoding="utf-8") as handle:
        overrides = json.load(handle)
    return cfg.with_overrides(overrides)

def _persist_checkpoint(raw, checkpoint_dir, benchmark, method, seed):
    path = os.path.abspath(os.path.join(
        checkpoint_dir, benchmark, method, f"seed{seed}.pt"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path

def _state_content_sha256(payload):
    """Canonical tensor-content hash, independent of torch.save container bytes."""
    digest = hashlib.sha256()

    def visit(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict\0")
            for key in sorted(value, key=lambda item: str(item)):
                digest.update(str(key).encode("utf-8"))
                digest.update(b"\0")
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"sequence\0")
            for item in value:
                visit(item)
        else:
            digest.update(json.dumps(
                value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\0")

    visit(payload)
    return digest.hexdigest()

def _parse_seed_spec(spec):
    """Parse comma-separated integers and inclusive ranges."""
    seeds = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if "-" in token[1:]:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending seed range is invalid: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("seed list is empty")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seed IDs must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed IDs must be unique")
    return seeds

def _cpu_tensor_dict(mapping):
    """Detach a tensor mapping for deterministic checkpoint serialization."""
    return {name: tensor.detach().cpu() for name, tensor in mapping.items()}


def describe_task_loader(loader, task_id):
    """Provenance view of one task's training loader, for both backends.

    Records the wrapper chain and source dataset class, the transform, the
    label mapping / pixel permutation / label offset when present and, for
    device-resident datasets, the SHA-256 of the exact pixel tensor used.
    """
    original = loader.dataset
    base = original
    wrappers = []
    seen = set()
    while hasattr(base, "dataset") and id(base) not in seen:
        seen.add(id(base))
        wrappers.append(f"{type(base).__module__}.{type(base).__name__}")
        base = base.dataset
    source_class = getattr(base, "source_dataset_class", None) or (
        f"{type(base).__module__}.{type(base).__name__}")
    transform_repr = getattr(base, "transform_repr", None)
    if transform_repr is None:
        transform_repr = repr(getattr(base, "transform", None))
    record = {
        "task": task_id,
        "dataset_wrappers": wrappers,
        "source_dataset_class": source_class,
        "transform_repr": transform_repr,
        "loader_class": f"{type(loader).__module__}.{type(loader).__name__}",
        "batch_size": int(getattr(loader, "batch_size", 0) or 0),
    }
    content = getattr(base, "content_sha256", None)
    if content is not None:
        record["materialized_pixels_sha256"] = str(content)
    pixels = getattr(base, "x", None)
    if torch.is_tensor(pixels):
        # Device-resident storage shared by every task view of this split;
        # reported so peak-memory figures can be read net of the dataset.
        record["resident_pixel_storage"] = {
            "device": str(pixels.device),
            "bytes": int(pixels.numel() * pixels.element_size()),
            "shared_across_tasks": True,
        }
    label_mapping = getattr(original, "label_mapping", None)
    if label_mapping is not None:
        record["source_classes"] = sorted(int(k) for k in label_mapping)
        record["label_mapping"] = {
            str(int(k)): int(v) for k, v in sorted(label_mapping.items())}
    perm = getattr(original, "perm", None)
    if perm is not None:
        perm_bytes = np.asarray(perm.detach().cpu(), dtype=np.int64).tobytes()
        record["pixel_permutation_sha256"] = hashlib.sha256(perm_bytes).hexdigest()
    if hasattr(original, "offset"):
        record["global_label_offset"] = int(original.offset)
    return record
