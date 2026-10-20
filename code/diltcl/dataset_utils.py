"""Robust, offline-friendly dataset acquisition helpers.

Loading a torchvision dataset can fail on machines that cannot reach the
default download mirror -- a common situation on GPU clusters with restricted
internet (the ``HTTP 503`` you may have seen when CIFAR-100 tried to download).
These helpers make data acquisition robust and predictable:

1. **Prefer a local copy.** We try ``download=False`` first, so an
   already-present dataset never touches the network.
2. **Download only when allowed.** A download is attempted only if the data is
   missing *and* downloading is enabled.
3. **Fail with guidance.** Any failure becomes a clear, actionable message
   (which dataset, where it was expected, how to provide it offline) instead of
   a raw stack trace.

Configuration is via two environment variables so no code edits are needed:

* ``DILTCL_DATA_ROOT`` -- directory holding (or to receive) the datasets
  (default ``./data``).
* ``DILTCL_DOWNLOAD``  -- set to ``0``/``false``/``no`` to forbid downloads
  (default: downloads allowed).

The dataset *class* is passed in as an argument, so this module carries no
torch dependency and can be unit-tested in isolation.
"""

import os

_DATASET_URLS = {
    "CIFAR-100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    "CIFAR-10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "MNIST": "the files listed in torchvision.datasets.MNIST.mirrors",
    "FashionMNIST": "the files listed in torchvision.datasets.FashionMNIST.mirrors",
    "KMNIST": "the files listed in torchvision.datasets.KMNIST.mirrors",
    "SVHN": "http://ufldl.stanford.edu/housenumbers/ (train_32x32.mat, test_32x32.mat)",
    "TinyImageNet": "http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                    "(extract so '<root>/tiny-imagenet-200/train' and '.../val' exist)",
    "notMNIST": "the notMNIST_large / notMNIST_small image folders "
                "(e.g. from the notMNIST dataset release)",
}

_FALSEY = {"0", "false", "no", "off", ""}


def default_data_root():
    """Return the configured data root (``$DILTCL_DATA_ROOT`` or ``./data``)."""
    return os.environ.get("DILTCL_DATA_ROOT", "./data")


def download_enabled():
    """Whether automatic downloads are allowed (``$DILTCL_DOWNLOAD``)."""
    return os.environ.get("DILTCL_DOWNLOAD", "1").strip().lower() not in _FALSEY


def _data_missing_message(name, root, attempted_download, cause=None):
    """Build an actionable message for a dataset that could not be loaded."""
    url = _DATASET_URLS.get(name, "(see the torchvision documentation)")
    abs_root = os.path.abspath(root)
    lines = [f"Could not load the {name} dataset from '{abs_root}'."]
    if attempted_download and cause is not None:
        lines.append(f"An automatic download was attempted but failed: "
                     f"{type(cause).__name__}: {cause}")
    lines += [
        "",
        "This is an environment / network issue, not a bug in the experiment "
        "code: the machine cannot reach the default dataset mirror.",
        "",
        "To fix it, make the data available under DILTCL_DATA_ROOT "
        f"(currently '{abs_root}') in any of these ways:",
        f"  1. Copy a pre-downloaded archive there (e.g. {url}) and re-run. "
        "With downloads enabled, a valid local archive is extracted WITHOUT "
        "any network access.",
        "  2. Or place the already-extracted dataset folder there and re-run.",
        "  3. Or point DILTCL_DATA_ROOT at a directory that already has it: "
        "`export DILTCL_DATA_ROOT=/path/to/datasets`.",
        "",
        "On clusters that block the default mirror (e.g. AutoDL), enable the "
        "provider's academic-network acceleration before downloading, then "
        "set DILTCL_DOWNLOAD=0 once the data is in place to skip the network "
        "entirely.",
    ]
    return "\n".join(lines)


def missing_data_error(name, root, attempted_download=False, cause=None):
    """Return a ``RuntimeError`` with an actionable "data missing" message.

    Public wrapper around the internal message builder so loaders that do not
    go through :func:`load_torchvision_dataset` (e.g. ``ImageFolder``- or
    ``SVHN``-based loaders in :mod:`diltcl.hard_benchmarks`) can raise the same
    consistent, guidance-rich error.
    """
    return RuntimeError(
        _data_missing_message(name, root, attempted_download, cause))


def load_torchvision_dataset(dataset_cls, root, train, transform, download,
                             name):
    """Load a torchvision dataset, preferring a local copy.

    Args:
        dataset_cls: A torchvision dataset class (e.g. ``datasets.CIFAR100``)
            accepting ``root``, ``train``, ``download`` and ``transform``.
        root: Directory holding (or to receive) the dataset.
        train: Whether to load the train split.
        transform: Transform passed to the dataset.
        download: Whether a download may be attempted if the data is missing.
        name: Human-readable dataset name for error messages.

    Returns:
        The constructed dataset.

    Raises:
        RuntimeError: If the data is missing and cannot be obtained, with a
            message explaining how to provide it.
    """
    # 1) Prefer an existing local copy -- this never touches the network.
    try:
        return dataset_cls(root=root, train=train, download=False,
                           transform=transform)
    except Exception as local_exc:  # noqa: BLE001 - typically "dataset not found"
        if not download:
            raise RuntimeError(_data_missing_message(
                name, root, attempted_download=False)) from local_exc

    # 2) Data missing locally and downloading is allowed -- try to fetch it.
    #    (A valid local archive is extracted here without network access.)
    try:
        return dataset_cls(root=root, train=train, download=True,
                           transform=transform)
    except Exception as dl_exc:  # noqa: BLE001 - network / mirror failures
        raise RuntimeError(_data_missing_message(
            name, root, attempted_download=True, cause=dl_exc)) from dl_exc
