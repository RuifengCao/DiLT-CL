"""Harder continual-learning benchmarks: Split-TinyImageNet and 5-datasets.

Both expose the same interface as :mod:`diltcl.data` -- a function returning
``(train_loaders, test_loaders)`` (one DataLoader per task) -- so the existing
task-IL (``run_task_il.py``) and class-IL (``run_class_il.py``) runners
can use them unchanged to test generality and, under class-IL, stress the
task-inference step.

Loading is offline-first (like :mod:`diltcl.dataset_utils`): an existing local
copy is always used without touching the network, and missing data raises a
clear, actionable error rather than a stack trace.

* **Split-TinyImageNet** -- 200 classes of 64x64x3 images, split into tasks.
  Trained on the ``train`` split and evaluated on the labelled ``val`` split
  (``test`` is unlabelled). TinyImageNet is not auto-downloadable via
  torchvision; place the extracted ``tiny-imagenet-200`` directory under the
  data root (see the URL in the error message).

* **5-datasets** -- five datasets treated as five tasks, each resized to
  3x32x32 (grayscale expanded to 3 channels). The default set is fully
  torchvision-downloadable (``KMNIST`` stands in for the classic ``notMNIST``);
  pass ``dataset_names=[...,"notmnist"]`` with a local notMNIST folder for the
  canonical variant.

The pure layout logic (class splitting, label maps, val-annotation parsing,
global-label offsets) lives in :mod:`diltcl.benchmark_layout` and is unit-tested
there.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .benchmark_layout import (
    five_datasets_offsets,
    parse_tinyimagenet_val_annotations,
    split_classes,
)
from . import data_backend
from .data import (
    device_view,
    legacy_loader,
    resolve_loader_options,
    class_split_loaders,
)
from .dataset_utils import (
    default_data_root,
    download_enabled,
    load_torchvision_dataset,
    missing_data_error,
)
from .utils import set_seed

# Default / classic 5-datasets task orders. The default is fully
# torchvision-downloadable; the classic uses notMNIST (supply it locally).
FIVE_DATASETS_DEFAULT = ["cifar10", "mnist", "svhn", "fashionmnist", "kmnist"]
FIVE_DATASETS_CLASSIC = ["cifar10", "mnist", "svhn", "fashionmnist", "notmnist"]

_CLASSES_PER_DATASET = 10  # every dataset in the 5-datasets suite is 10-way


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _transform(image_size, gray_to_rgb):
    """Resize to ``image_size`` square, optionally expand grayscale to 3ch."""
    steps = []
    if gray_to_rgb:
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(steps)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class _LabelOffset(Dataset):
    """Shift every target by a constant (for the 5-datasets class-IL space)."""

    def __init__(self, dataset, offset):
        self.dataset = dataset
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, int(target) + self.offset


class _TinyImageNetVal(Dataset):
    """TinyImageNet validation split, labelled via ``val_annotations.txt``.

    Uses the *same* ``class_to_idx`` as the train split so labels line up.
    """

    def __init__(self, val_dir, class_to_idx, transform):
        ann_path = os.path.join(val_dir, "val_annotations.txt")
        with open(ann_path) as f:
            fname_to_wnid = parse_tinyimagenet_val_annotations(f.read())
        self.images_dir = os.path.join(val_dir, "images")
        self.transform = transform
        self.samples = [
            (fname, class_to_idx[wnid])
            for fname, wnid in fname_to_wnid.items()
            if wnid in class_to_idx
        ]
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, target = self.samples[idx]
        img = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# ---------------------------------------------------------------------------
# Split-TinyImageNet
# ---------------------------------------------------------------------------

def get_split_tinyimagenet(num_tasks=10, classes_per_task=20, batch_size=64,
                           seed=42, data_root=None, num_workers=None,
                           class_il=False, download=None, image_size=64, *,
                           backend=None, device=None, eval_batch_size=None):
    """Split TinyImageNet-200 into sequential, disjoint-class tasks.

    Args mirror :func:`diltcl.data.get_split_cifar100`. ``class_il=False``
    remaps each task's labels to ``0..classes_per_task-1``; ``class_il=True``
    keeps global labels ``0..num_tasks*classes_per_task-1`` for a single head.

    With the default ``"device"`` backend the 110k JPEGs are decoded once
    (multi-process) and cached under ``$DILTCL_CACHE_DIR``; the legacy
    ``"loader"`` backend re-decodes them every epoch.

    Returns:
        ``(train_loaders, test_loaders)`` -- lists of loaders per task.
    """
    set_seed(seed)
    root = default_data_root() if data_root is None else data_root
    backend, device, num_workers, eval_batch_size = resolve_loader_options(
        backend, device, num_workers, eval_batch_size, batch_size)
    base = os.path.join(root, "tiny-imagenet-200")
    train_dir, val_dir = os.path.join(base, "train"), os.path.join(base, "val")

    # TinyImageNet has no torchvision auto-download; require a local copy.
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise missing_data_error("TinyImageNet", root, attempted_download=False)

    transform = _transform(image_size, gray_to_rgb=False)
    train_full = datasets.ImageFolder(train_dir, transform=transform)
    val_full = _TinyImageNetVal(val_dir, train_full.class_to_idx, transform)

    splits = split_classes(num_classes=200, num_tasks=num_tasks,
                           classes_per_task=classes_per_task)
    return class_split_loaders(
        train_full, val_full, train_full.targets, val_full.targets, splits,
        class_il=class_il, batch_size=batch_size,
        eval_batch_size=eval_batch_size, backend=backend, device=device,
        num_workers=num_workers, name=f"tinyimagenet{image_size}")


# ---------------------------------------------------------------------------
# 5-datasets
# ---------------------------------------------------------------------------

def _tv_loader(dataset_cls, name):
    """Offline-first loader for a torchvision dataset with a ``train`` flag."""
    def load(root, train, transform, download):
        return load_torchvision_dataset(
            dataset_cls, root=root, train=train, transform=transform,
            download=download, name=name)
    return load


def _svhn_loader(root, train, transform, download):
    """Offline-first SVHN loader (SVHN uses ``split`` instead of ``train``)."""
    split = "train" if train else "test"
    try:
        return datasets.SVHN(root=root, split=split, download=False,
                             transform=transform)
    except Exception as local_exc:  # noqa: BLE001 - "dataset not found"
        if not download:
            raise missing_data_error("SVHN", root, False) from local_exc
    try:
        return datasets.SVHN(root=root, split=split, download=True,
                             transform=transform)
    except Exception as dl_exc:  # noqa: BLE001 - network / mirror failure
        raise missing_data_error("SVHN", root, True, dl_exc) from dl_exc


def _notmnist_loader(root, train, transform, download):
    """Offline-only notMNIST loader (ImageFolder; no auto-download exists)."""
    sub = "notMNIST_large" if train else "notMNIST_small"
    path = os.path.join(root, sub)
    if not os.path.isdir(path):
        raise missing_data_error("notMNIST", root, attempted_download=False)
    return datasets.ImageFolder(path, transform=transform)


# name -> (loader_fn(root, train, transform, download), is_grayscale)
_FIVE_REGISTRY = {
    "mnist": (_tv_loader(datasets.MNIST, "MNIST"), True),
    "fashionmnist": (_tv_loader(datasets.FashionMNIST, "FashionMNIST"), True),
    "kmnist": (_tv_loader(datasets.KMNIST, "KMNIST"), True),
    "cifar10": (_tv_loader(datasets.CIFAR10, "CIFAR-10"), False),
    "svhn": (_svhn_loader, False),
    "notmnist": (_notmnist_loader, True),
}


def get_five_datasets(batch_size=64, seed=42, data_root=None, num_workers=None,
                      class_il=False, download=None, dataset_names=None,
                      image_size=32, *, backend=None, device=None,
                      eval_batch_size=None):
    """Build the 5-datasets task sequence (each dataset is one 10-way task).

    Args:
        dataset_names: Ordered dataset keys from :data:`_FIVE_REGISTRY`
            (default :data:`FIVE_DATASETS_DEFAULT`). The task order is the list
            order.
        class_il: ``False`` keeps native ``0..9`` labels per task (task-IL);
            ``True`` shifts dataset ``t`` to the global block
            ``[10t, 10t+10)`` so a single head spans all ``10*N`` classes.
        Other args mirror :func:`diltcl.data.get_split_cifar100`.

    Returns:
        ``(train_loaders, test_loaders)`` -- lists of loaders per task.
    """
    set_seed(seed)
    root = default_data_root() if data_root is None else data_root
    do_download = download_enabled() if download is None else download
    backend, device, num_workers, eval_batch_size = resolve_loader_options(
        backend, device, num_workers, eval_batch_size, batch_size)
    names = list(dataset_names) if dataset_names else list(FIVE_DATASETS_DEFAULT)

    unknown = [n for n in names if n.lower() not in _FIVE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown 5-datasets entries: {unknown}. "
                         f"Choose from {sorted(_FIVE_REGISTRY)}")

    offsets = five_datasets_offsets(len(names), _CLASSES_PER_DATASET)

    train_loaders, test_loaders = [], []
    for task_id, name in enumerate(names):
        loader_fn, gray = _FIVE_REGISTRY[name.lower()]
        transform = _transform(image_size, gray_to_rgb=gray)
        train_ds = loader_fn(root, True, transform, do_download)
        test_ds = loader_fn(root, False, transform, do_download)
        if backend == "device":
            label_map = ({c: c + offsets[task_id]
                          for c in range(_CLASSES_PER_DATASET)}
                         if class_il else None)
            train_arrays = data_backend.materialize_dataset(
                train_ds, name=f"five_{name.lower()}{image_size}_train")
            test_arrays = data_backend.materialize_dataset(
                test_ds, name=f"five_{name.lower()}{image_size}_test")
            train_x = torch.from_numpy(train_arrays.x).to(device)
            test_x = torch.from_numpy(test_arrays.x).to(device)
            train_loaders.append(data_backend.DeviceLoader(
                device_view(train_x, train_arrays, np.arange(len(train_arrays)),
                             label_map, device), batch_size, shuffle=True))
            test_loaders.append(data_backend.DeviceLoader(
                device_view(test_x, test_arrays, np.arange(len(test_arrays)),
                             label_map, device), eval_batch_size, shuffle=False))
            continue
        if class_il:
            train_ds = _LabelOffset(train_ds, offsets[task_id])
            test_ds = _LabelOffset(test_ds, offsets[task_id])
        train_loaders.append(legacy_loader(train_ds, batch_size, True,
                                            num_workers))
        test_loaders.append(legacy_loader(test_ds, eval_batch_size, False,
                                           num_workers))

    return train_loaders, test_loaders
