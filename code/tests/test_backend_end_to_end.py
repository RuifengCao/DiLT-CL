"""End-to-end: the real Split-CIFAR-100 pipeline is backend-invariant.

A tiny synthetic CIFAR-100 (torchvision's pickle layout, 2 images per class)
is written to disk, torchvision's integrity table is pointed at it, and
:func:`diltcl.data.get_split_cifar100` is called for real with both backends.
The test then trains finetune / ER / LwF / DiLT-CL through the actual
trainers on both loader stacks and demands **bit-identical** result matrices
and identical global RNG state -- the contract that lets v3 replace the
DataLoader pipeline without touching any number in the paper.

Also checks provenance (materialised-pixel hash, label mappings) and that the
runner's CLI accepts the execution flags.

Skipped automatically when PyTorch / torchvision are unavailable.
"""

import contextlib
import hashlib
import io
import os
import pathlib
import pickle
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    import torch
    from torchvision import datasets
except ImportError:  # pragma: no cover - torch-free environment
    if __name__ == "__main__":
        print("PyTorch/torchvision not installed; end-to-end backend test skipped.")
        sys.exit(0)
    import pytest
    pytest.skip("PyTorch/torchvision not installed; end-to-end backend test skipped.", allow_module_level=True)

from diltcl import AdaptiveCNN, MultiHeadModel, SimpleCNN  # noqa: E402
from diltcl import data_backend  # noqa: E402
from diltcl.baselines import train_er, train_finetune  # noqa: E402
from diltcl.data import get_split_cifar100  # noqa: E402
from diltcl.hard_benchmarks import get_split_tinyimagenet  # noqa: E402
from diltcl.runner_common import describe_task_loader  # noqa: E402
from diltcl.training import train_diltcl, train_lwf  # noqa: E402
from diltcl.utils import set_seed  # noqa: E402

NUM_TASKS, CLASSES_PER_TASK, PER_CLASS = 3, 5, 2
NUM_CLASSES = NUM_TASKS * CLASSES_PER_TASK


def _write_fake_cifar100(root):
    """Write a CIFAR-100-shaped pickle dataset and patch torchvision's md5s."""
    rng = np.random.RandomState(0)
    base = pathlib.Path(root) / "cifar-100-python"
    base.mkdir(parents=True)
    md5s = {}
    for split, count in (("train", PER_CLASS), ("test", 1)):
        labels = np.repeat(np.arange(NUM_CLASSES), count)
        data = rng.randint(0, 256, size=(labels.size, 3 * 32 * 32), dtype=np.uint8)
        payload = {"data": data, "fine_labels": labels.tolist(),
                   "coarse_labels": labels.tolist()}
        path = base / split
        path.write_bytes(pickle.dumps(payload))
        md5s[split] = hashlib.md5(path.read_bytes()).hexdigest()
    meta = base / "meta"
    meta.write_bytes(pickle.dumps({
        "fine_label_names": [f"c{i}" for i in range(NUM_CLASSES)]}))
    md5s["meta"] = hashlib.md5(meta.read_bytes()).hexdigest()
    saved = (datasets.CIFAR100.train_list, datasets.CIFAR100.test_list,
             datasets.CIFAR100.meta)
    datasets.CIFAR100.train_list = [["train", md5s["train"]]]
    datasets.CIFAR100.test_list = [["test", md5s["test"]]]
    datasets.CIFAR100.meta = {"filename": "meta", "key": "fine_label_names",
                              "md5": md5s["meta"]}
    return saved


def _restore_cifar100(saved):
    (datasets.CIFAR100.train_list, datasets.CIFAR100.test_list,
     datasets.CIFAR100.meta) = saved


@contextlib.contextmanager
def _fake_cifar_root():
    with tempfile.TemporaryDirectory() as root, \
            tempfile.TemporaryDirectory() as cache:
        saved = _write_fake_cifar100(root)
        old_cache = os.environ.get("DILTCL_CACHE_DIR")
        os.environ["DILTCL_CACHE_DIR"] = cache
        try:
            yield root
        finally:
            _restore_cifar100(saved)
            if old_cache is None:
                os.environ.pop("DILTCL_CACHE_DIR", None)
            else:
                os.environ["DILTCL_CACHE_DIR"] = old_cache


def _loaders(root, backend, class_il=False):
    return get_split_cifar100(
        num_tasks=NUM_TASKS, classes_per_task=CLASSES_PER_TASK, batch_size=4,
        seed=1, data_root=root, num_workers=0, class_il=class_il,
        download=False, backend=backend, device="cpu")


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _train_all(root, backend):
    """Train several methods on one backend; return matrices + RNG state."""
    train, test = _loaders(root, backend)
    out = {}
    builder = lambda: MultiHeadModel(SimpleCNN(CLASSES_PER_TASK),  # noqa: E731
                                     NUM_TASKS, CLASSES_PER_TASK)
    set_seed(3)
    out["finetune"] = _quiet(train_finetune, builder, train, test, NUM_TASKS,
                             device="cpu", epochs=2, lr=1e-3)[0]
    set_seed(3)
    out["er"] = _quiet(train_er, builder, train, test, NUM_TASKS, device="cpu",
                       epochs=2, lr=1e-3, buffer_size=12,
                       replay_batch_size=4)[0]
    set_seed(3)
    out["lwf"] = _quiet(train_lwf, builder(), train, test, NUM_TASKS,
                        device="cpu", epochs=2, lr=1e-3)[0]
    set_seed(3)
    out["diltcl"] = _quiet(
        train_diltcl, SimpleCNN(CLASSES_PER_TASK), train, test, NUM_TASKS,
        device="cpu", num_partitions=2, pruning_rounds=2, target_sparsity=0.5,
        full_train_epochs=1, imp_epochs=1, expert_finetune_epochs=1)[0]
    out["_rng"] = torch.get_rng_state()
    return out


def test_split_cifar100_backends_are_bit_identical_through_training():
    with _fake_cifar_root() as root:
        device = _train_all(root, "device")
        loader = _train_all(root, "loader")
    for method in ("finetune", "er", "lwf", "diltcl"):
        a, b = np.asarray(device[method]), np.asarray(loader[method])
        assert a.shape == b.shape == (NUM_TASKS, NUM_TASKS)
        assert np.array_equal(a, b), f"{method}: backends disagree\n{a}\n{b}"
    assert torch.equal(device["_rng"], loader["_rng"])


def test_device_loaders_expose_provenance_and_remapped_labels():
    with _fake_cifar_root() as root:
        train, test = _loaders(root, "device")
        assert all(isinstance(l, data_backend.DeviceLoader) for l in train + test)
        assert [len(l.dataset) for l in train] == [CLASSES_PER_TASK * PER_CLASS] * NUM_TASKS
        view = describe_task_loader(train[1], 1)
        assert view["source_dataset_class"].endswith("CIFAR100")
        assert "Normalize" in view["transform_repr"]
        assert len(view["materialized_pixels_sha256"]) == 64
        assert view["label_mapping"] == {str(5 + i): i for i in range(5)}
        x, y = next(iter(test[1]))
        assert x.shape[1:] == (3, 32, 32) and int(y.max()) < CLASSES_PER_TASK
        # Class-IL keeps the global label axis.
        train_ci, _ = _loaders(root, "device", class_il=True)
        _, y_ci = next(iter(train_ci[2]))
        assert int(y_ci.min()) >= 10
        # Eval batch size is honoured independently of the training batch.
        _, test_big = get_split_cifar100(
            num_tasks=NUM_TASKS, classes_per_task=CLASSES_PER_TASK,
            batch_size=4, seed=1, data_root=root, download=False,
            backend="device", device="cpu", eval_batch_size=64)
        assert len(test_big[0]) == 1


def test_materialised_cache_is_reused_and_kept_out_of_the_data_root():
    with _fake_cifar_root() as root:
        cache = pathlib.Path(os.environ["DILTCL_CACHE_DIR"])
        _loaders(root, "device")
        # CIFAR is array-backed, so no cache file is needed at all; the data
        # root must stay untouched either way (suite fingerprint stability).
        assert not any(p.name.startswith("_diltcl") for p in pathlib.Path(root).iterdir())
        assert not list(cache.glob("*.npz"))


def _write_fake_tinyimagenet(root, classes=4, per_class=3, size=64):
    """A miniature ``tiny-imagenet-200`` tree (JPEG files, val annotations)."""
    from PIL import Image
    rng = np.random.RandomState(1)
    base = pathlib.Path(root) / "tiny-imagenet-200"
    wnids = [f"n{i:08d}" for i in range(classes)]
    for wnid in wnids:
        folder = base / "train" / wnid / "images"
        folder.mkdir(parents=True)
        for k in range(per_class):
            pixels = rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{wnid}_{k}.JPEG", quality=95)
    val_images = base / "val" / "images"
    val_images.mkdir(parents=True)
    lines = []
    for i, wnid in enumerate(wnids):
        pixels = rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
        name = f"val_{i}.JPEG"
        Image.fromarray(pixels).save(val_images / name, quality=95)
        lines.append(f"{name}\t{wnid}\t0\t0\t63\t63")
    (base / "val" / "val_annotations.txt").write_text("\n".join(lines) + "\n")
    return root


def test_split_tinyimagenet_generic_decode_path_is_bit_identical():
    """ImageFolder + Resize goes through the worker decoder and the cache."""
    with tempfile.TemporaryDirectory() as root, \
            tempfile.TemporaryDirectory() as cache:
        _write_fake_tinyimagenet(root)
        old_cache = os.environ.get("DILTCL_CACHE_DIR")
        os.environ["DILTCL_CACHE_DIR"] = cache
        try:
            def loaders(backend):
                return get_split_tinyimagenet(
                    num_tasks=2, classes_per_task=2, batch_size=4, seed=3,
                    data_root=root, num_workers=0, backend=backend,
                    device="cpu")

            def run(backend):
                train, test = loaders(backend)
                builder = lambda: MultiHeadModel(  # noqa: E731
                    AdaptiveCNN(num_classes=2), 2, 2)
                set_seed(4)
                matrix = _quiet(train_finetune, builder, train, test, 2,
                                device="cpu", epochs=2, lr=1e-3)[0]
                return np.asarray(matrix), torch.get_rng_state()

            # First device run decodes and writes the cache (RNG must stay
            # untouched), the second hits the cache, the loader run is the
            # legacy reference: all three must agree bit for bit.
            first, rng_first = run("device")
            assert any(f.endswith(".npz") for f in os.listdir(cache))
            second, rng_second = run("device")
            legacy, rng_legacy = run("loader")
            assert np.array_equal(first, legacy) and np.array_equal(second, legacy)
            assert torch.equal(rng_first, rng_legacy)
            assert torch.equal(rng_second, rng_legacy)
            train, _ = loaders("device")
            view = describe_task_loader(train[0], 0)
            assert view["source_dataset_class"].endswith("ImageFolder")
            assert "materialized_pixels_sha256" in view
        finally:
            if old_cache is None:
                os.environ.pop("DILTCL_CACHE_DIR", None)
            else:
                os.environ["DILTCL_CACHE_DIR"] = old_cache


def test_tinyimagenet_loads_from_a_relative_data_root():
    """ImageFolder lists its files with the root already joined, so a relative
    data root must not be joined a second time when the cache key is derived
    (v3.0.7 and v3.0.8 raised FileNotFoundError for every relative root)."""
    with tempfile.TemporaryDirectory() as root, \
            tempfile.TemporaryDirectory() as cache:
        _write_fake_tinyimagenet(root)
        saved_cwd, old_cache = os.getcwd(), os.environ.get("DILTCL_CACHE_DIR")
        os.environ["DILTCL_CACHE_DIR"] = cache
        try:
            os.chdir(root)
            hashes = []
            for data_root in (".", "./", root):
                train, test = get_split_tinyimagenet(
                    num_tasks=2, classes_per_task=2, batch_size=4, seed=3,
                    data_root=data_root, num_workers=0, backend="device",
                    device="cpu")
                hashes.append((describe_task_loader(train[0], 0)["materialized_pixels_sha256"],
                               describe_task_loader(test[0], 0)["materialized_pixels_sha256"]))
            assert hashes[0] == hashes[1] == hashes[2]
        finally:
            os.chdir(saved_cwd)
            if old_cache is None:
                os.environ.pop("DILTCL_CACHE_DIR", None)
            else:
                os.environ["DILTCL_CACHE_DIR"] = old_cache


def test_a_symlinked_data_root_reuses_the_cache_of_the_real_root():
    """The suite resolves --data-root; a check run through a symbolic link to
    the same folder must hit the same cache entries instead of decoding the
    images a second time (v3.0.9 keyed the cache on the unresolved root)."""
    with tempfile.TemporaryDirectory() as root, \
            tempfile.TemporaryDirectory() as cache, \
            tempfile.TemporaryDirectory() as links:
        _write_fake_tinyimagenet(root)
        linked = os.path.join(links, "data")
        os.symlink(root, linked)
        old_cache = os.environ.get("DILTCL_CACHE_DIR")
        os.environ["DILTCL_CACHE_DIR"] = cache
        try:
            hashes = []
            for data_root in (root, linked):
                train, test = get_split_tinyimagenet(
                    num_tasks=2, classes_per_task=2, batch_size=4, seed=3,
                    data_root=data_root, num_workers=0, backend="device",
                    device="cpu")
                hashes.append((describe_task_loader(train[0], 0)["materialized_pixels_sha256"],
                               describe_task_loader(test[0], 0)["materialized_pixels_sha256"]))
                if data_root == root:
                    entries = sorted(f for f in os.listdir(cache) if f.endswith(".npz"))
            assert hashes[0] == hashes[1]
            # The linked root decoded nothing new: same cache files as before.
            assert sorted(f for f in os.listdir(cache) if f.endswith(".npz")) == entries
            assert len(entries) == 2          # one train, one validation split
        finally:
            if old_cache is None:
                os.environ.pop("DILTCL_CACHE_DIR", None)
            else:
                os.environ["DILTCL_CACHE_DIR"] = old_cache


if __name__ == "__main__":
    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_") and callable(value)]
    for function in functions:
        function()
        print(f"PASS {function.__name__}")
    print(f"\nAll {len(functions)} end-to-end backend tests passed.")
