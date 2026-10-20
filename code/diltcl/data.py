"""Datasets and loaders for the continual-learning benchmarks.

Provides:
  * ``SubsetWithMapping`` - a label-remapping subset (legacy loader backend).
  * ``PermutedMNIST``      - a fixed-permutation MNIST wrapper (legacy backend).
  * ``get_split_cifar100`` - builds the Split-CIFAR-100 task sequence.
  * ``get_permuted_mnist`` - builds the Permuted-MNIST task sequence.

Two interchangeable backends produce the per-task loaders:

* ``"device"`` (default) -- the whole dataset is materialised once as a
  ``uint8`` tensor on the training device and batches are gathered there
  (:mod:`diltcl.data_backend`). It is bit-identical to the legacy path for the
  same seed and an order of magnitude faster on small models.
* ``"loader"`` -- the original ``torchvision`` + ``DataLoader`` pipeline, kept
  so that older result files can be reproduced with the exact same code path.

Select with the ``backend`` argument or ``$DILTCL_DATA_BACKEND``.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from . import data_backend
from .benchmark_layout import split_classes, task_label_map
from .dataset_utils import (
    default_data_root,
    download_enabled,
    load_torchvision_dataset,
)
from .utils import get_device, set_seed

_DEFAULT_NUM_WORKERS = 2


def default_num_workers():
    """Worker count for the legacy loader backend (``$DILTCL_NUM_WORKERS``)."""
    raw = os.environ.get("DILTCL_NUM_WORKERS")
    if raw is None or not raw.strip():
        return _DEFAULT_NUM_WORKERS
    value = int(raw)
    if value < 0:
        raise ValueError("DILTCL_NUM_WORKERS must be non-negative")
    return value


class SubsetWithMapping(Dataset):
    """A dataset subset that remaps target labels.

    This is essential for Split-CIFAR-100 where each task's labels should
    start from 0. For example, if a task covers classes ``[5, 6, 7, 8, 9]``
    they are remapped to ``[0, 1, 2, 3, 4]`` for the classification head.

    Args:
        dataset: Original PyTorch dataset.
        indices: Indices to include in the subset.
        label_mapping: Dict mapping original labels to new labels.
    """

    def __init__(self, dataset, indices, label_mapping):
        self.dataset = dataset
        self.indices = indices
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data, target = self.dataset[real_idx]
        target = self.label_mapping.get(int(target), int(target))
        return data, target


class PermutedMNIST(Dataset):
    """Permuted MNIST dataset wrapper.

    Applies a fixed random pixel permutation to all images, creating a new
    "task" where the input distribution changes but the labels do not.

    Args:
        dataset: Original MNIST dataset.
        perm: Tensor of shape ``(784,)`` containing the pixel permutation.
    """

    def __init__(self, dataset, perm):
        self.dataset = dataset
        self.perm = perm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Flatten, apply permutation, and reshape back.
        data_flat = data.view(-1)
        data_permuted = data_flat[self.perm]
        data_permuted = data_permuted.view(1, 28, 28)
        return data_permuted, target


# ---------------------------------------------------------------------------
# Backend-agnostic construction helpers (shared with hard_benchmarks)
# ---------------------------------------------------------------------------

def legacy_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=False)


def resolve_loader_options(backend, device, num_workers, eval_batch_size, batch_size):
    backend = data_backend.resolve_backend(backend)
    device = get_device() if device is None else device
    num_workers = default_num_workers() if num_workers is None else num_workers
    eval_batch_size = batch_size if eval_batch_size is None else int(eval_batch_size)
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    return backend, device, num_workers, eval_batch_size


def class_split_loaders(train_dataset, test_dataset, train_targets,
                        test_targets, splits, *, class_il, batch_size,
                        eval_batch_size, backend, device, num_workers,
                        name):
    """Per-task loaders for a disjoint-class split benchmark.

    Shared by Split-CIFAR-100 and Split-TinyImageNet. ``splits`` is the list
    of class lists per task; ``class_il`` chooses between the task-local and
    the global label space.
    """
    train_targets = np.asarray(train_targets)
    test_targets = np.asarray(test_targets)
    task_indices = []
    for target_classes in splits:
        label_map = task_label_map(target_classes, class_il)
        tr_idx = np.where(np.isin(train_targets, target_classes))[0]
        te_idx = np.where(np.isin(test_targets, target_classes))[0]
        task_indices.append((label_map, tr_idx, te_idx))

    train_loaders, test_loaders = [], []
    if backend == "device":
        train_arrays = data_backend.materialize_dataset(
            train_dataset, name=f"{name}_train")
        test_arrays = data_backend.materialize_dataset(
            test_dataset, name=f"{name}_test")
        train_x = torch.from_numpy(train_arrays.x).to(device)
        test_x = torch.from_numpy(test_arrays.x).to(device)
        for label_map, tr_idx, te_idx in task_indices:
            train_loaders.append(data_backend.DeviceLoader(
                device_view(train_x, train_arrays, tr_idx, label_map, device),
                batch_size, shuffle=True))
            test_loaders.append(data_backend.DeviceLoader(
                device_view(test_x, test_arrays, te_idx, label_map, device),
                eval_batch_size, shuffle=False))
        return train_loaders, test_loaders

    for label_map, tr_idx, te_idx in task_indices:
        train_loaders.append(legacy_loader(
            SubsetWithMapping(train_dataset, tr_idx, label_map),
            batch_size, True, num_workers))
        test_loaders.append(legacy_loader(
            SubsetWithMapping(test_dataset, te_idx, label_map),
            eval_batch_size, False, num_workers))
    return train_loaders, test_loaders


def device_view(x_device, arrays, indices, label_map, device, perm=None):
    """A :class:`DeviceTensorDataset` over rows ``indices`` of ``x_device``."""
    native = arrays.y[indices]
    if label_map is not None:
        remapped = np.fromiter(
            (label_map.get(int(t), int(t)) for t in native), dtype=np.int64,
            count=len(native))
    else:
        remapped = native
    return data_backend.DeviceTensorDataset(
        x_device, remapped, device=device, mean=arrays.mean, std=arrays.std,
        indices=indices, perm=perm, label_mapping=label_map,
        source_dataset_class=arrays.source_class,
        transform_repr=arrays.transform_repr,
        content_sha256=arrays.content_sha256)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def get_split_cifar100(num_tasks=20, classes_per_task=5, batch_size=64,
                       seed=42, data_root=None, num_workers=None,
                       class_il=False, download=None, *, backend=None,
                       device=None, eval_batch_size=None):
    """Split CIFAR-100 into sequential, disjoint-class tasks.

    Each task contains ``classes_per_task`` disjoint classes from CIFAR-100.

    Args:
        num_tasks: Number of sequential tasks (default 20 for CIFAR-100).
        classes_per_task: Number of classes per task (default 5).
        batch_size: Training batch size.
        seed: Random seed for reproducibility.
        data_root: Directory holding / receiving CIFAR-100. ``None`` uses
            ``$DILTCL_DATA_ROOT`` (default ``./data``).
        num_workers: DataLoader worker processes (legacy backend only;
            ``None`` uses ``$DILTCL_NUM_WORKERS`` or 2).
        class_il: Label space. If ``False`` (default, task-incremental), each
            task's labels are remapped to ``0..classes_per_task-1``. If
            ``True`` (class-incremental), labels are kept on a single *global*
            axis ``0..num_tasks*classes_per_task-1`` so a single shared head
            spans every class and evaluation needs no task identity.
        download: Whether a download may be attempted if the data is missing.
            ``None`` uses ``$DILTCL_DOWNLOAD`` (default: allowed). An existing
            local copy is always used without touching the network.
        backend: ``"device"`` (default) or ``"loader"``; ``None`` reads
            ``$DILTCL_DATA_BACKEND``.
        device: Device for the device backend (default: CUDA if available).
        eval_batch_size: Batch size of the test loaders (default: ``batch_size``;
            larger values only speed up evaluation and are recorded in the
            run's provenance).

    Returns:
        ``(train_loaders, test_loaders)`` - lists of loaders per task.
    """
    set_seed(seed)
    root = default_data_root() if data_root is None else data_root
    do_download = download_enabled() if download is None else download
    backend, device, num_workers, eval_batch_size = resolve_loader_options(
        backend, device, num_workers, eval_batch_size, batch_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ),
    ])

    train_dataset = load_torchvision_dataset(
        datasets.CIFAR100, root=root, train=True, transform=transform,
        download=do_download, name="CIFAR-100")
    test_dataset = load_torchvision_dataset(
        datasets.CIFAR100, root=root, train=False, transform=transform,
        download=do_download, name="CIFAR-100")

    splits = split_classes(num_classes=100, num_tasks=num_tasks,
                           classes_per_task=classes_per_task)
    return class_split_loaders(
        train_dataset, test_dataset, train_dataset.targets,
        test_dataset.targets, splits, class_il=class_il,
        batch_size=batch_size, eval_batch_size=eval_batch_size,
        backend=backend, device=device, num_workers=num_workers,
        name="cifar100")


def get_permuted_mnist(num_tasks=10, batch_size=256, seed=42,
                       data_root=None, num_workers=None, download=None, *,
                       backend=None, device=None, eval_batch_size=None):
    """Build the Permuted-MNIST task sequence.

    Each task applies a different fixed random pixel permutation.

    Args mirror :func:`get_split_cifar100`.

    Returns:
        ``(train_loaders, test_loaders)`` - lists of loaders per task.
    """
    set_seed(seed)
    root = default_data_root() if data_root is None else data_root
    do_download = download_enabled() if download is None else download
    backend, device, num_workers, eval_batch_size = resolve_loader_options(
        backend, device, num_workers, eval_batch_size, batch_size)

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = load_torchvision_dataset(
        datasets.MNIST, root=root, train=True, transform=transform,
        download=do_download, name="MNIST")
    test_dataset = load_torchvision_dataset(
        datasets.MNIST, root=root, train=False, transform=transform,
        download=do_download, name="MNIST")

    # A dedicated RNG keeps permutation generation independent of global state.
    rng = np.random.RandomState(seed)
    perms = [torch.from_numpy(rng.permutation(784)).long()
             for _ in range(num_tasks)]

    train_loaders, test_loaders = [], []
    if backend == "device":
        train_arrays = data_backend.materialize_dataset(train_dataset,
                                                        name="mnist_train")
        test_arrays = data_backend.materialize_dataset(test_dataset,
                                                       name="mnist_test")
        train_x = torch.from_numpy(train_arrays.x).to(device)
        test_x = torch.from_numpy(test_arrays.x).to(device)
        all_train = np.arange(len(train_arrays))
        all_test = np.arange(len(test_arrays))
        for perm in perms:
            train_loaders.append(data_backend.DeviceLoader(
                device_view(train_x, train_arrays, all_train, None, device,
                             perm=perm), batch_size, shuffle=True))
            test_loaders.append(data_backend.DeviceLoader(
                device_view(test_x, test_arrays, all_test, None, device,
                             perm=perm), eval_batch_size, shuffle=False))
        return train_loaders, test_loaders

    for perm in perms:
        train_loaders.append(legacy_loader(
            PermutedMNIST(train_dataset, perm), batch_size, True, num_workers))
        test_loaders.append(legacy_loader(
            PermutedMNIST(test_dataset, perm), eval_batch_size, False,
            num_workers))
    return train_loaders, test_loaders
