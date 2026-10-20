"""Device-resident tensor datasets and an RNG-faithful minibatch loader.

Why this module exists
----------------------
The models in this project are tiny (~0.6 M parameters). On an RTX 5090 a
forward/backward step costs well under a millisecond, so the legacy pipeline --
``torchvision`` datasets decoded sample by sample in Python, PIL conversion,
``ToTensor``/``Normalize`` per image, a ``DataLoader`` with two workers, and a
host-to-device copy for every batch -- dominated the wall-clock time. DiLT-CL's
inner IMP loop even built its loaders with ``num_workers=0``.

The ``device`` backend materialises every benchmark **once** as a ``uint8``
tensor that lives on the training device (Split-CIFAR-100: ~180 MiB,
Split-TinyImageNet: ~1.3 GiB, MNIST: ~50 MiB). A minibatch is then a single
gather plus the ``float()/255 -> (x - mean) / std`` arithmetic executed on the
device, i.e. a handful of kernels instead of thousands of Python calls.

Numerical contract
------------------
The device backend is designed to be **bit-identical** to the legacy
``DataLoader`` pipeline for the same seed, and ``tests/test_data_backend.py``
asserts it. Three details make that possible:

1. **Same arithmetic.** ``ToTensor`` computes ``uint8 -> float32`` followed by
   a true division by 255 and ``Normalize`` computes ``(x - mean) / std`` with
   float32 tensors of shape ``[C, 1, 1]``. :meth:`DeviceTensorDataset.fetch`
   performs exactly these operations in the same order. The divisor 255 is a
   *device tensor* rather than a Python scalar because PyTorch's CUDA kernel
   rewrites ``tensor / python_scalar`` as ``tensor * (1 / scalar)``, which can
   differ from true division by one ulp.
2. **Same shuffle order.** ``DataLoader`` consumes the global CPU RNG in a
   specific way on every ``iter()``: ``_BaseDataLoaderIter.__init__`` draws one
   ``int64`` (its ``base_seed``) and, for a shuffling loader,
   ``RandomSampler.__iter__`` draws a second ``int64`` that seeds a private
   ``Generator`` used for ``torch.randperm``. :class:`DeviceLoader` performs
   the identical draws in the identical order (see :meth:`DeviceLoader.__iter__`),
   so the global RNG state after each epoch -- and therefore model
   initialisation, reservoir sampling and every later shuffle -- is unchanged.
3. **Same partitioning semantics.** ``subset`` composes index tensors exactly
   like ``torch.utils.data.Subset`` composes Python indices.

The backend therefore changes *no* number in a run; it only changes where the
pixels live and how fast a batch is produced.

Materialisation
---------------
:func:`materialize_dataset` accepts any torchvision-style dataset whose
``transform`` is ``Compose([<deterministic PIL ops>..., ToTensor(), Normalize?])``.
Datasets that expose their pixels as arrays (CIFAR, MNIST family, SVHN) are
converted directly; everything else (``ImageFolder`` / TinyImageNet, the
resized 5-datasets tasks) is decoded with a multi-process ``DataLoader`` and
cached under ``$DILTCL_CACHE_DIR`` (default ``~/.cache/diltcl``). The cache is
deliberately kept **outside** the data root so it does not perturb the suite's
dataset fingerprint. Writes are atomic and guarded by a file lock so that
parallel jobs starting at the same time decode each dataset only once.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:  # torchvision is an optional import at module level so tests without it still load.
    from torchvision import transforms as _tv_transforms
except Exception:  # noqa: BLE001 - torchvision missing
    _tv_transforms = None

__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "DeviceLoader",
    "DeviceTensorDataset",
    "MaterializedArrays",
    "default_cache_dir",
    "is_device_dataset",
    "make_loader",
    "materialize_dataset",
    "resolve_backend",
    "subset_dataset",
]

BACKENDS = ("device", "loader")
DEFAULT_BACKEND = "device"
_FALSEY = {"0", "false", "no", "off", ""}


# ---------------------------------------------------------------------------
# Backend / cache configuration
# ---------------------------------------------------------------------------

def resolve_backend(explicit=None):
    """Return the data backend name: explicit argument, else ``$DILTCL_DATA_BACKEND``."""
    name = explicit if explicit is not None else os.environ.get(
        "DILTCL_DATA_BACKEND", DEFAULT_BACKEND)
    name = str(name).strip().lower()
    if name not in BACKENDS:
        raise ValueError(
            f"unknown data backend {name!r}; choose one of {BACKENDS}")
    return name


def default_cache_dir():
    """Directory for materialised tensor caches (``$DILTCL_CACHE_DIR``)."""
    return os.environ.get(
        "DILTCL_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "diltcl"))


def cache_refresh_requested():
    """Whether ``$DILTCL_CACHE_REFRESH`` asks to ignore existing caches."""
    return os.environ.get(
        "DILTCL_CACHE_REFRESH", "0").strip().lower() not in _FALSEY


# ---------------------------------------------------------------------------
# Materialisation of torchvision datasets into uint8 arrays
# ---------------------------------------------------------------------------

class MaterializedArrays:
    """Pixels and labels of one dataset split as arrays, plus normalisation.

    Attributes:
        x: ``uint8`` array ``[N, C, H, W]``.
        y: ``int64`` array ``[N]`` with the dataset's native labels.
        mean / std: Per-channel normalisation constants (``None`` = none).
        source_class: Fully-qualified class name of the original dataset.
        transform_repr: ``repr`` of the original transform (for provenance).
        content_sha256: Hash of ``x`` and ``y`` -- the exact pixels used.
    """

    __slots__ = ("x", "y", "mean", "std", "source_class", "transform_repr",
                 "content_sha256")

    def __init__(self, x, y, mean, std, source_class, transform_repr,
                 content_sha256=None):
        self.x = np.ascontiguousarray(x, dtype=np.uint8)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        if self.x.ndim != 4:
            raise ValueError("materialised pixels must have shape [N, C, H, W]")
        if self.y.shape != (self.x.shape[0],):
            raise ValueError("labels must be a one-dimensional array of length N")
        self.mean = None if mean is None else tuple(float(v) for v in mean)
        self.std = None if std is None else tuple(float(v) for v in std)
        self.source_class = source_class
        self.transform_repr = transform_repr
        self.content_sha256 = (content_sha256 if content_sha256 is not None
                               else _content_hash(self.x, self.y))

    def __len__(self):
        return int(self.x.shape[0])


def _content_hash(x, y):
    digest = hashlib.sha256()
    digest.update(b"x")
    digest.update(json.dumps(list(x.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(x).tobytes())
    digest.update(b"y")
    digest.update(np.ascontiguousarray(y).tobytes())
    return digest.hexdigest()


def _split_transform(transform):
    """Split ``transform`` into ``(pil_stage, mean, std)``.

    The tensor stage after ``ToTensor`` must be empty or a single
    ``Normalize``; anything else cannot be reproduced by the device backend
    and is rejected explicitly instead of being silently approximated.
    """
    if _tv_transforms is None:
        raise RuntimeError("torchvision is required to materialise datasets")
    if transform is None:
        return None, None, None
    steps = (list(transform.transforms)
             if isinstance(transform, _tv_transforms.Compose) else [transform])
    to_tensor_positions = [i for i, step in enumerate(steps)
                           if isinstance(step, _tv_transforms.ToTensor)]
    if len(to_tensor_positions) != 1:
        raise ValueError(
            "device backend needs exactly one ToTensor() in the transform; "
            f"got {transform!r}")
    cut = to_tensor_positions[0]
    pil_steps, tensor_steps = steps[:cut], steps[cut + 1:]
    mean = std = None
    if tensor_steps:
        if (len(tensor_steps) != 1
                or not isinstance(tensor_steps[0], _tv_transforms.Normalize)):
            raise ValueError(
                "device backend supports only [PIL ops..., ToTensor, Normalize?]; "
                f"got {transform!r}")
        mean = tuple(float(v) for v in tensor_steps[0].mean)
        std = tuple(float(v) for v in tensor_steps[0].std)
    pil_stage = _tv_transforms.Compose(pil_steps) if pil_steps else None
    return pil_stage, mean, std


def _pil_to_uint8_chw(img):
    """Convert a PIL image (or array) to a ``uint8`` ``[C, H, W]`` array.

    Mirrors ``torchvision.transforms.functional.to_tensor`` up to (but not
    including) the division by 255.
    """
    array = np.array(img, dtype=np.uint8, copy=True)
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3:
        array = array.transpose(2, 0, 1)
    else:
        raise ValueError(f"unsupported image array shape {array.shape}")
    return np.ascontiguousarray(array)


def _collate_uint8(batch):
    xs = np.stack([_pil_to_uint8_chw(img) for img, _ in batch])
    ys = np.asarray([int(target) for _, target in batch], dtype=np.int64)
    return xs, ys


@contextlib.contextmanager
def _transform_override(dataset, transform):
    """Temporarily replace ``dataset.transform`` (restored on exit)."""
    if not hasattr(dataset, "transform"):
        raise TypeError(
            f"{type(dataset).__name__} has no 'transform' attribute; cannot "
            "materialise it")
    saved = dataset.transform
    dataset.transform = transform
    try:
        yield
    finally:
        dataset.transform = saved


def _array_fast_path(dataset):
    """Return ``(x_chw_uint8, y)`` for datasets exposing raw pixel arrays.

    Handles the CIFAR family (``data`` is ``[N, H, W, 3]`` uint8 ndarray),
    the MNIST family (``data`` is ``[N, H, W]`` uint8 tensor) and SVHN
    (``data`` is ``[N, 3, H, W]`` uint8 ndarray, labels in ``labels``).
    Returns ``None`` when the layout is not recognised.
    """
    data = getattr(dataset, "data", None)
    if data is None:
        return None
    if torch.is_tensor(data):
        data = data.numpy()
    if not isinstance(data, np.ndarray) or data.dtype != np.uint8:
        return None
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        return None
    if torch.is_tensor(targets):
        targets = targets.numpy()
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    if data.ndim == 3:                       # [N, H, W] grayscale
        x = data[:, None, :, :]
    elif data.ndim == 4 and data.shape[-1] in (1, 3) and data.shape[1] not in (1, 3):
        x = data.transpose(0, 3, 1, 2)       # [N, H, W, C] -> [N, C, H, W]
    elif data.ndim == 4 and data.shape[1] in (1, 3):
        x = data                             # already [N, C, H, W]
    else:
        return None
    if x.shape[0] != y.shape[0]:
        return None
    return np.ascontiguousarray(x), y


def _cache_key(dataset, transform_repr):
    parts = {
        "class": f"{type(dataset).__module__}.{type(dataset).__name__}",
        "root": str(getattr(dataset, "root", "")),
        "train": getattr(dataset, "train", None),
        "split": getattr(dataset, "split", None),
        "len": len(dataset),
        "transform": transform_repr,
        "format": 1,
    }
    encoded = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


@contextlib.contextmanager
def _file_lock(path):
    """Cross-process lock (POSIX ``flock``); a no-op where unavailable."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX platform
        yield
        return
    with open(path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _decode_with_workers(dataset, pil_stage, workers, batch_size=256):
    """Decode every sample to uint8 arrays with a worker pool.

    The ``DataLoader`` used here draws one ``int64`` from the global RNG when
    it is iterated. Materialisation happens after ``set_seed`` and before the
    model is built, and it only happens on a cache miss, so that draw would
    make the first run of a benchmark differ from every cached run and from
    the legacy backend. The global RNG state is therefore saved and restored
    around the decode.
    """
    rng_state = torch.get_rng_state()
    try:
        with _transform_override(dataset, pil_stage):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=workers, collate_fn=_collate_uint8)
            xs, ys = [], []
            for x, y in loader:
                xs.append(x)
                ys.append(y)
    finally:
        torch.set_rng_state(rng_state)
    if not xs:
        raise ValueError("dataset is empty; nothing to materialise")
    return np.concatenate(xs), np.concatenate(ys)


def materialize_dataset(dataset, *, cache_dir=None, workers=None,
                        use_cache=True, name=None, verbose=True):
    """Return :class:`MaterializedArrays` for a torchvision-style dataset.

    Args:
        dataset: Dataset with ``transform``/``__getitem__`` returning
            ``(PIL image, target)`` and a transform of the form
            ``Compose([PIL ops..., ToTensor(), Normalize?])``.
        cache_dir: Cache directory for decoded datasets; ``None`` uses
            :func:`default_cache_dir`. Only datasets without an array fast
            path are cached (decoding them is the expensive part).
        workers: Decoding processes for the generic path (default: up to 16,
            bounded by the CPU count).
        use_cache: Set ``False`` to disable both reading and writing the cache.
        name: Human-readable name used in log lines.
        verbose: Print a one-line progress message when decoding.
    """
    transform = getattr(dataset, "transform", None)
    transform_repr = repr(transform)
    pil_stage, mean, std = _split_transform(transform)
    source_class = f"{type(dataset).__module__}.{type(dataset).__name__}"
    label = name or type(dataset).__name__

    if pil_stage is None:
        fast = _array_fast_path(dataset)
        if fast is not None:
            x, y = fast
            return MaterializedArrays(x, y, mean, std, source_class, transform_repr)

    cache_path = None
    if use_cache:
        root = cache_dir or default_cache_dir()
        key = _cache_key(dataset, transform_repr)
        cache_path = os.path.join(root, f"{label}_{key}.npz")
        if os.path.isfile(cache_path) and not cache_refresh_requested():
            return _load_cache(cache_path, mean, std, source_class, transform_repr)

    def _decode():
        if workers is None:
            env_workers = os.environ.get("DILTCL_DECODE_WORKERS")
            if env_workers:
                n_workers = max(0, int(env_workers))
            else:
                try:
                    cpu = len(os.sched_getaffinity(0))   # container-aware
                except (AttributeError, OSError):  # pragma: no cover
                    cpu = os.cpu_count() or 1
                n_workers = max(0, min(16, cpu - 1))
        else:
            n_workers = int(workers)
        started = time.perf_counter()
        if verbose:
            print(f"[data_backend] decoding {label} ({len(dataset)} samples, "
                  f"{n_workers} workers)...", flush=True)
        x, y = _decode_with_workers(dataset, pil_stage, n_workers)
        arrays = MaterializedArrays(x, y, mean, std, source_class, transform_repr)
        if verbose:
            print(f"[data_backend] {label}: {tuple(arrays.x.shape)} uint8 in "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
        return arrays

    if cache_path is None:
        return _decode()

    with _file_lock(cache_path + ".lock"):
        # Another process may have finished while we waited for the lock.
        if os.path.isfile(cache_path) and not cache_refresh_requested():
            return _load_cache(cache_path, mean, std, source_class, transform_repr)
        arrays = _decode()
        temporary = f"{cache_path}.tmp.{os.getpid()}"
        with open(temporary, "wb") as handle:
            np.savez(handle, x=arrays.x, y=arrays.y,
                     content_sha256=np.array(arrays.content_sha256),
                     transform_repr=np.array(transform_repr),
                     source_class=np.array(source_class))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
        if verbose:
            print(f"[data_backend] cached {label} -> {cache_path}", flush=True)
        return arrays


def _load_cache(path, mean, std, source_class, transform_repr):
    with np.load(path, allow_pickle=False) as archive:
        cached_repr = str(archive["transform_repr"])
        if cached_repr != transform_repr:
            raise RuntimeError(
                f"cache {path} was built for a different transform; delete it "
                "or set DILTCL_CACHE_REFRESH=1")
        return MaterializedArrays(
            archive["x"], archive["y"], mean, std, source_class,
            transform_repr, content_sha256=str(archive["content_sha256"]))


# ---------------------------------------------------------------------------
# Device-resident dataset
# ---------------------------------------------------------------------------

class _TransformDescription:
    """Stands in for ``dataset.transform`` so provenance code can ``repr`` it."""

    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text


class DeviceTensorDataset(Dataset):
    """A benchmark split held as tensors on the training device.

    Args:
        x: ``uint8`` tensor/array ``[N_full, C, H, W]`` (shared storage; tasks
            that select subsets share it without copying).
        y: ``int64`` labels aligned with ``indices`` (already remapped to the
            protocol's label space).
        device: Device the pixels live on.
        mean / std: Per-channel normalisation constants or ``None``.
        indices: Row indices into ``x`` defining this view (default: all).
        perm: Optional pixel permutation (Permuted-MNIST) applied after
            normalisation, exactly like :class:`diltcl.data.PermutedMNIST`.
        label_mapping / source_dataset_class / transform_repr /
        content_sha256: Provenance attributes read by the runners.
    """

    def __init__(self, x, y, *, device, mean=None, std=None, indices=None,
                 perm=None, label_mapping=None, source_dataset_class="",
                 transform_repr="", content_sha256=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(x))
        if x.dtype != torch.uint8 or x.dim() != 4:
            raise ValueError("x must be a uint8 tensor of shape [N, C, H, W]")
        self.device = torch.device(device)
        self.x = x.to(self.device)
        n_full = int(self.x.shape[0])
        if indices is None:
            indices = torch.arange(n_full)
        indices = torch.as_tensor(np.asarray(indices) if not torch.is_tensor(indices)
                                  else indices).to(torch.int64).reshape(-1)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= n_full):
            raise IndexError("indices fall outside the pixel tensor")
        self.indices = indices.to(self.device)
        y = torch.as_tensor(np.asarray(y) if not torch.is_tensor(y) else y)
        y = y.to(torch.int64).reshape(-1)
        if y.numel() != self.indices.numel():
            raise ValueError("y must be aligned with indices")
        self.y = y.to(self.device)
        self.mean = self.std = None
        if mean is not None:
            channels = int(self.x.shape[1])
            if len(mean) != channels or len(std) != channels:
                raise ValueError("mean/std must have one entry per channel")
            # Same construction as torchvision.transforms.functional.normalize.
            self.mean = torch.as_tensor(mean, dtype=torch.float32,
                                        device=self.device).view(-1, 1, 1)
            self.std = torch.as_tensor(std, dtype=torch.float32,
                                       device=self.device).view(-1, 1, 1)
        self._scale = torch.tensor(255.0, dtype=torch.float32, device=self.device)
        self.perm = None
        if perm is not None:
            self.perm = torch.as_tensor(perm).to(torch.int64).to(self.device)
            numel = int(np.prod(self.x.shape[1:]))
            if self.perm.numel() != numel:
                raise ValueError("perm must index every pixel exactly once")
        self.label_mapping = label_mapping
        self.source_dataset_class = source_dataset_class
        self.transform_repr = transform_repr
        self.transform = _TransformDescription(transform_repr)
        self.content_sha256 = content_sha256

    # -- Dataset protocol --------------------------------------------------

    def __len__(self):
        return int(self.indices.numel())

    def __getitem__(self, idx):
        """Single-sample access (slow path, for compatibility only)."""
        x, y = self.fetch(torch.as_tensor([int(idx)], device=self.device))
        return x[0], int(y[0])

    # -- Fast paths --------------------------------------------------------

    def fetch(self, positions):
        """Return the normalised float batch for positions into this view."""
        positions = torch.as_tensor(positions, device=self.device,
                                    dtype=torch.int64)
        rows = self.indices.index_select(0, positions)
        # uint8 -> float32 -> /255 ... exactly torchvision's ToTensor.
        batch = self.x.index_select(0, rows).to(torch.float32).div(self._scale)
        if self.mean is not None:
            batch = batch.sub(self.mean).div(self.std)  # Normalize
        if self.perm is not None:
            shape = batch.shape
            batch = batch.reshape(shape[0], -1).index_select(
                1, self.perm).reshape(shape)
        return batch, self.y.index_select(0, positions)

    def subset(self, positions):
        """A view restricted to ``positions`` (composes like ``Subset``)."""
        positions = torch.as_tensor(
            np.asarray(positions) if not torch.is_tensor(positions) else positions
        ).to(torch.int64).reshape(-1).to(self.device)
        view = DeviceTensorDataset.__new__(DeviceTensorDataset)
        view.device = self.device
        view.x = self.x
        view.indices = self.indices.index_select(0, positions)
        view.y = self.y.index_select(0, positions)
        view.mean, view.std, view._scale = self.mean, self.std, self._scale
        view.perm = self.perm
        view.label_mapping = self.label_mapping
        view.source_dataset_class = self.source_dataset_class
        view.transform_repr = self.transform_repr
        view.transform = self.transform
        view.content_sha256 = self.content_sha256
        return view

    def __repr__(self):
        return (f"DeviceTensorDataset(n={len(self)}, "
                f"pixels={tuple(self.x.shape[1:])}, device={self.device}, "
                f"source={self.source_dataset_class or '?'})")


def is_device_dataset(dataset):
    return isinstance(dataset, DeviceTensorDataset)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DeviceLoader:
    """Minibatch iterator over a :class:`DeviceTensorDataset`.

    Behaves like ``DataLoader(dataset, batch_size, shuffle, drop_last=False)``
    for the training code (``len``, iteration yielding ``(x, y)``, and a
    ``dataset`` attribute) and reproduces its consumption of the global RNG.
    """

    def __init__(self, dataset, batch_size, shuffle=False, *,
                 mirror_dataloader_rng=True):
        if not isinstance(dataset, DeviceTensorDataset):
            raise TypeError("DeviceLoader requires a DeviceTensorDataset")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.mirror_dataloader_rng = bool(mirror_dataloader_rng)
        # Attributes some callers read off DataLoader instances.
        self.num_workers = 0
        self.pin_memory = False
        self.collate_fn = None
        self.drop_last = False

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        if self.mirror_dataloader_rng:
            # (1) torch.utils.data._BaseDataLoaderIter.__init__ draws base_seed
            #     from the default generator on every iter(), shuffle or not.
            torch.empty((), dtype=torch.int64).random_()
        if self.shuffle:
            # (2) torch.utils.data.RandomSampler.__iter__: seed a private
            #     generator from the default one, then randperm with it.
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            order = torch.randperm(n, generator=generator)
        else:
            order = torch.arange(n)
        order = order.to(self.dataset.device)
        for start in range(0, n, self.batch_size):
            yield self.dataset.fetch(order[start:start + self.batch_size])


# ---------------------------------------------------------------------------
# Backend-agnostic helpers used by the trainers and runners
# ---------------------------------------------------------------------------

def subset_dataset(dataset, indices):
    """Restrict ``dataset`` to ``indices`` with the right mechanism per type."""
    if isinstance(dataset, DeviceTensorDataset):
        return dataset.subset(indices)
    return Subset(dataset, indices)


def make_loader(dataset, batch_size, shuffle, *, num_workers=0,
                pin_memory=False, collate_fn=None, drop_last=False):
    """Build the appropriate loader for ``dataset``.

    :class:`DeviceTensorDataset` (or a ``Subset`` of one) yields a
    :class:`DeviceLoader`; anything else yields a plain ``DataLoader`` with the
    given options, which keeps the legacy backend and synthetic test loaders
    working unchanged.
    """
    if isinstance(dataset, Subset) and isinstance(dataset.dataset,
                                                  DeviceTensorDataset):
        dataset = dataset.dataset.subset(dataset.indices)
    if isinstance(dataset, DeviceTensorDataset):
        if drop_last:
            raise ValueError("DeviceLoader does not implement drop_last")
        return DeviceLoader(dataset, batch_size, shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      collate_fn=collate_fn, drop_last=drop_last)
