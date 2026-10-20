"""Fail-fast preflight checks for dependencies and datasets.

The v2.1 P0 run lost hours of GPU time twice for avoidable reasons: five of
seven jobs failed, and the two most expensive failures happened *after* all
training had finished. Both were knowable before a single batch was processed:

* ``scikit-learn`` was absent, so the calibrated-router jobs crashed at the
  evaluation step. The launcher had already recorded ``scikit-learn: None`` in
  ``environment.json`` before starting; nothing acted on it.
* MNIST and TinyImageNet were not staged under the data root, so those jobs
  died in the loader.

This module turns those latent conditions into an explicit gate that runs
*before* any training. It is deliberately torch-free and dependency-free so it
can execute in any environment, including one without PyTorch.

Run it standalone -- execute the file directly so it works even on a machine
where PyTorch is not installed yet (importing the package would pull in torch)::

    python code/diltcl/preflight.py --data-root /path/to/data \\
        --benchmarks cifar,pmnist,tinyimagenet --require-calibration

Inside a working environment the module can also be imported normally as
``from diltcl import preflight``.

Exit code ``0`` means every requested job can start; ``1`` lists exactly what
is missing and how to fix it.
"""

import argparse
import importlib.util
import os
import shutil
import sys

# --- Dataset layout requirements -------------------------------------------
# Each entry lists acceptable on-disk artifacts (relative to the data root).
# A dataset counts as present when ANY of its candidate paths exists, which
# accommodates both extracted directories and downloaded archives.
_MNIST_RAW_FILES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                    "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")

# Files torchvision actually opens when downloads are disabled (the P0/P1
# setting). Each entry is a list of *alternative* file sets; a dataset is
# present when every path of at least one alternative exists. A directory
# that merely exists (e.g. an ``MNIST/raw`` holding differently named
# archives) is not enough: that exact case wasted 50 P0 jobs once.
_DATASET_REQUIRED_FILES = {
    "CIFAR-100": [["cifar-100-python/train", "cifar-100-python/test",
                   "cifar-100-python/meta"]],
    "CIFAR-10": [["cifar-10-batches-py/data_batch_1",
                  "cifar-10-batches-py/data_batch_5",
                  "cifar-10-batches-py/test_batch",
                  "cifar-10-batches-py/batches.meta"]],
    "MNIST": [[f"MNIST/raw/{name}" for name in _MNIST_RAW_FILES]],
    "FashionMNIST": [[f"FashionMNIST/raw/{name}" for name in _MNIST_RAW_FILES]],
    "KMNIST": [[f"KMNIST/raw/{name}" for name in _MNIST_RAW_FILES]],
    "SVHN": [["train_32x32.mat", "test_32x32.mat"]],
    "TinyImageNet": [["tiny-imagenet-200/train", "tiny-imagenet-200/val/images",
                      "tiny-imagenet-200/val/val_annotations.txt"]],
}

# Human-readable acquisition hints, shown only for missing datasets.
_DATASET_HINTS = {
    "CIFAR-100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    "CIFAR-10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "MNIST": "torchvision.datasets.MNIST(root, download=True) on a networked "
             "machine, then copy the 'MNIST' folder over; the four raw files "
             "must be uncompressed and named exactly train-images-idx3-ubyte, "
             "train-labels-idx1-ubyte, t10k-images-idx3-ubyte, "
             "t10k-labels-idx1-ubyte under MNIST/raw/",
    "FashionMNIST": "torchvision.datasets.FashionMNIST(root, download=True), "
                    "then copy the 'FashionMNIST' folder over",
    "KMNIST": "torchvision.datasets.KMNIST(root, download=True), then copy "
              "the 'KMNIST' folder over",
    "SVHN": "http://ufldl.stanford.edu/housenumbers/ "
            "(train_32x32.mat and test_32x32.mat, placed directly in the root)",
    "TinyImageNet": "http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                    "(extract so '<root>/tiny-imagenet-200/train' exists)",
}

# Which datasets each benchmark needs.
BENCHMARK_DATASETS = {
    "cifar": ["CIFAR-100"],
    "pmnist": ["MNIST"],
    "tinyimagenet": ["TinyImageNet"],
    "fivedatasets": ["CIFAR-10", "MNIST", "SVHN", "FashionMNIST", "KMNIST"],
}

# Modules required for particular capabilities, with install hints.
_CAPABILITY_MODULES = {
    "training": [("torch", "pip install torch"),
                 ("torchvision", "pip install torchvision"),
                 ("numpy", "pip install numpy")],
    "statistics": [("scipy", "pip install scipy")],
    # Calibration prefers scikit-learn but no longer requires it: the
    # score-logistic router falls back to a SciPy L-BFGS-B solver. The check is
    # therefore advisory unless the caller pins the sklearn backend.
    "calibration_sklearn": [("sklearn", "pip install scikit-learn")],
}


def free_disk_bytes(path):
    """Free bytes on the filesystem holding ``path`` (``None`` if unknown)."""
    try:
        usage = shutil.disk_usage(path)
        return int(usage.free)
    except (OSError, ValueError):
        return None


# Rough checkpoint cost per (seed, benchmark, protocol) job running *all*
# methods. Measured from a real Split-CIFAR-100 P0 job: ~307 MiB, dominated by
# DiLT-CL's dense expert masks.
CHECKPOINT_BYTES_PER_JOB = 320 * 1024 * 1024
# Per-method split of that figure, used when jobs hold a subset of methods
# (the v3 suite runs one method per job): DiLT-CL carries the expert masks,
# every other method persists a single small dense model (+ buffer metadata).
CHECKPOINT_BYTES_PER_METHOD = {"diltcl": 260 * 1024 * 1024}
CHECKPOINT_BYTES_OTHER_METHOD = 8 * 1024 * 1024


def estimate_checkpoint_bytes(num_jobs=0, method_lists=None):
    """Estimated disk needed for persisted checkpoints.

    Args:
        num_jobs: Number of all-methods jobs (legacy per-seed layout).
        method_lists: Optional list with one iterable of method names per job;
            when given it replaces ``num_jobs`` and the estimate is summed per
            method, so a per-method job layout is not charged the full
            all-methods figure for every job.
    """
    if method_lists is None:
        return int(num_jobs) * CHECKPOINT_BYTES_PER_JOB
    total = 0
    for methods in method_lists:
        for method in methods:
            total += CHECKPOINT_BYTES_PER_METHOD.get(
                method, CHECKPOINT_BYTES_OTHER_METHOD)
    return int(total)


def resolve_data_root(explicit=None):
    """Resolve the data root exactly as the dataset loaders do.

    ``diltcl.dataset_utils.default_data_root`` falls back to ``./data`` when
    ``DILTCL_DATA_ROOT`` is unset. Reading the environment variable directly
    made preflight skip its dataset checks in that default case and report
    "OK" for a run that then failed in the loader -- precisely the late failure
    this module exists to prevent.
    """
    if explicit:
        return str(explicit)
    root = os.environ.get("DILTCL_DATA_ROOT")
    if root:
        return root
    return "./data"


def module_available(name):
    """Return ``True`` when ``name`` can be imported without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def dataset_present(data_root, dataset):
    """Return ``(present, checked_paths)`` for one dataset under ``data_root``.

    ``present`` is ``True`` only when every file of at least one alternative
    in :data:`_DATASET_REQUIRED_FILES` exists; ``checked_paths`` lists the
    paths that were tested so a failure message can name what is missing.
    """
    alternatives = _DATASET_REQUIRED_FILES.get(dataset, [])
    checked = []
    for files in alternatives:
        paths = [os.path.join(data_root, rel) for rel in files]
        checked.extend(paths)
        if all(os.path.exists(path) for path in paths):
            return True, checked
    return False, checked


def missing_dataset_files(data_root, dataset):
    """Files of the first alternative that are absent (for messages)."""
    alternatives = _DATASET_REQUIRED_FILES.get(dataset, [])
    if not alternatives:
        return []
    return [rel for rel in alternatives[0]
            if not os.path.exists(os.path.join(data_root, rel))]


def benchmark_data_entries(benchmark):
    """Top-level names under the data root that ``benchmark`` reads.

    Derived from :data:`_DATASET_REQUIRED_FILES`; used by the suite and the
    verifier to fingerprint a job's dataset dependency per benchmark, so that
    staging one benchmark's data later does not invalidate finished jobs of
    the others.
    """
    names = []
    for dataset in required_datasets([benchmark]):
        for files in _DATASET_REQUIRED_FILES.get(dataset, []):
            for rel in files:
                top = rel.replace("\\", "/").split("/", 1)[0]
                if top not in names:
                    names.append(top)
    return sorted(names)


def required_datasets(benchmarks):
    """Ordered, de-duplicated dataset list for the given benchmarks."""
    needed = []
    for bench in benchmarks:
        if bench not in BENCHMARK_DATASETS:
            raise ValueError(
                f"Unknown benchmark '{bench}'. "
                f"Choose from {sorted(BENCHMARK_DATASETS)}")
        for name in BENCHMARK_DATASETS[bench]:
            if name not in needed:
                needed.append(name)
    return needed


def check(data_root, benchmarks, require_training=True,
          require_calibration=False, pin_sklearn=False,
          output_dir=None, expected_jobs=0, expected_method_lists=None):
    """Run every applicable check and return a structured report.

    Args:
        data_root: Directory that should already hold the datasets, or ``None``
            to skip dataset checks entirely.
        benchmarks: Iterable of benchmark keys (see :data:`BENCHMARK_DATASETS`).
        require_training: Check the torch/torchvision/numpy training stack.
        require_calibration: The run uses the score-logistic router.
        pin_sklearn: Treat scikit-learn as mandatory rather than optional (use
            when a run must reproduce the sklearn solver exactly).

    Returns:
        Dict with ``ok`` (bool), ``blocking`` (list of str) and ``warnings``
        (list of str). ``blocking`` entries prevent the run from starting.
    """
    blocking, warnings_ = [], []

    capabilities = ["statistics"]
    if require_training:
        capabilities.insert(0, "training")
    for capability in capabilities:
        for module, hint in _CAPABILITY_MODULES[capability]:
            if not module_available(module):
                blocking.append(
                    f"missing Python module '{module}' (needed for "
                    f"{capability}) -> {hint}")

    if require_calibration:
        has_sklearn = module_available("sklearn")
        if not has_sklearn and pin_sklearn:
            blocking.append(
                "missing Python module 'sklearn' and --pin-sklearn was "
                "requested -> pip install scikit-learn")
        elif not has_sklearn:
            warnings_.append(
                "scikit-learn is absent; the score-logistic router will use "
                "the built-in SciPy L-BFGS-B solver. Results stay valid and "
                "record solver_backend='scipy-lbfgs', but they are not "
                "bit-identical to an sklearn-fitted router.")

    if data_root is not None:
        if not os.path.isdir(data_root):
            blocking.append(
                f"data root '{data_root}' does not exist or is not a directory")
        else:
            for dataset in required_datasets(benchmarks):
                present, _checked = dataset_present(data_root, dataset)
                if not present:
                    missing = ", ".join(missing_dataset_files(data_root, dataset))
                    blocking.append(
                        f"dataset '{dataset}' is incomplete under '{data_root}': "
                        f"missing {missing}. torchvision opens exactly these "
                        f"paths when downloads are disabled (archives or "
                        f"differently named files are not enough) -> obtain "
                        f"or extract it: "
                        f"{_DATASET_HINTS.get(dataset, 'the dataset website')}")

    if output_dir is not None and (expected_jobs or expected_method_lists):
        needed = estimate_checkpoint_bytes(expected_jobs, expected_method_lists)
        if expected_method_lists is not None:
            expected_jobs = len(expected_method_lists)
        free = free_disk_bytes(output_dir)
        if free is None:
            warnings_.append(
                f"could not determine free space on '{output_dir}'")
        elif free < needed:
            blocking.append(
                f"insufficient disk space at '{output_dir}': "
                f"{free / 2 ** 30:.1f} GiB free but about "
                f"{needed / 2 ** 30:.1f} GiB is needed for {expected_jobs} "
                "job(s) of persisted checkpoints -> free space, point "
                "--run-dir at a larger volume, or pass --no-save-checkpoints")
        elif free < needed * 2:
            warnings_.append(
                f"disk space at '{output_dir}' is tight: "
                f"{free / 2 ** 30:.1f} GiB free vs ~{needed / 2 ** 30:.1f} GiB "
                "estimated need")

    return {"ok": not blocking, "blocking": blocking, "warnings": warnings_}


def format_report(report, header="preflight"):
    """Render :func:`check` output as a readable, actionable block of text."""
    lines = []
    for message in report["warnings"]:
        lines.append(f"[{header}] WARNING: {message}")
    if report["ok"]:
        lines.append(f"[{header}] OK: dependencies and datasets are in place.")
        return "\n".join(lines)
    lines.append(f"[{header}] BLOCKED: {len(report['blocking'])} problem(s) "
                 "must be fixed before training starts.")
    for message in report["blocking"]:
        lines.append(f"  - {message}")
    lines.append(
        f"[{header}] Nothing was trained. Fixing the items above avoids "
        "losing compute to a late failure.")
    return "\n".join(lines)


def enforce(data_root, benchmarks, require_training=True,
            require_calibration=False, pin_sklearn=False, header="preflight",
            stream=None):
    """Print a report and raise ``SystemExit(1)`` when anything is blocking."""
    stream = sys.stderr if stream is None else stream
    report = check(data_root, benchmarks, require_training=require_training,
                   require_calibration=require_calibration,
                   pin_sklearn=pin_sklearn)
    text = format_report(report, header=header)
    if text:
        print(text, file=stream)
    if not report["ok"]:
        raise SystemExit(1)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check dependencies and datasets before an experiment run.")
    parser.add_argument("--data-root", default=None,
                        help="Directory holding the datasets (default: "
                             "$DILTCL_DATA_ROOT, else ./data, as the loaders resolve it).")
    parser.add_argument("--benchmarks", default="cifar",
                        help="Comma-separated: " +
                             ",".join(sorted(BENCHMARK_DATASETS)))
    parser.add_argument("--require-calibration", action="store_true",
                        help="The run uses the score-logistic router.")
    parser.add_argument("--pin-sklearn", action="store_true",
                        help="Treat scikit-learn as mandatory.")
    parser.add_argument("--output-dir", default=None,
                        help="Run directory, for the disk-space check.")
    parser.add_argument("--expected-jobs", type=int, default=0,
                        help="Number of jobs, for the disk-space estimate.")
    parser.add_argument("--no-training", action="store_true",
                        help="Skip the torch/torchvision stack check.")
    args = parser.parse_args(argv)

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    report = check(resolve_data_root(args.data_root), benchmarks,
                   require_training=not args.no_training,
                   require_calibration=args.require_calibration,
                   pin_sklearn=args.pin_sklearn,
                   output_dir=args.output_dir,
                   expected_jobs=args.expected_jobs)
    print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
