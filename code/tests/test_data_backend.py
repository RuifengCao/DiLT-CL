"""The device data backend must be bit-identical to the legacy DataLoader path.

Every assertion here compares the *new* :mod:`diltcl.data_backend` against the
*old* ``torchvision`` + ``DataLoader`` pipeline on the same synthetic images:
same batch contents, same shuffle order, same global RNG state afterwards, same
subset composition, same Permuted-MNIST pixels. If any of these ever differs,
switching backends would silently change experimental numbers, which is the one
thing the backend promises not to do.

Runs on CPU; when CUDA is available the device path is additionally exercised
on the GPU, which is the configuration that matters for real runs.

Skipped automatically when PyTorch / torchvision are unavailable.
"""

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
except ImportError:  # pragma: no cover - torch-free environment
    if __name__ == "__main__":
        print("PyTorch/torchvision not installed; data backend tests skipped.")
        sys.exit(0)
    import pytest
    pytest.skip("PyTorch/torchvision not installed; data backend tests skipped.", allow_module_level=True)

from diltcl import data_backend as db  # noqa: E402
from diltcl.data import PermutedMNIST, SubsetWithMapping  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _FakeCIFAR:
    """Mimics ``torchvision.datasets.CIFAR100``: uint8 ``data`` + PIL access."""

    def __init__(self, n=97, transform=None, seed=0, size=32, channels=3):
        rng = np.random.RandomState(seed)
        shape = (n, size, size, channels) if channels == 3 else (n, size, size)
        self.data = rng.randint(0, 256, size=shape, dtype=np.uint8)
        self.targets = rng.randint(0, 10, size=n).tolist()
        self.transform = transform
        self.root = "/fake"
        self.train = True

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


_NORMALIZE = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))
_CIFAR_TRANSFORM = transforms.Compose([transforms.ToTensor(), _NORMALIZE])


def _legacy_task_loader(ds, indices, label_map, batch_size, shuffle,
                        num_workers=0):
    subset = SubsetWithMapping(ds, indices, label_map)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False,
                      drop_last=False)


def _device_task_loader(ds, indices, label_map, batch_size, shuffle, device):
    arrays = db.materialize_dataset(ds, use_cache=False, verbose=False)
    y = np.array([label_map.get(int(t), int(t)) for t in arrays.y[indices]])
    dataset = db.DeviceTensorDataset(
        arrays.x, y, device=device, mean=arrays.mean, std=arrays.std,
        indices=indices, label_mapping=label_map)
    return db.DeviceLoader(dataset, batch_size, shuffle)


def _epochs(loader, epochs=2):
    out = []
    for _ in range(epochs):
        for x, y in loader:
            out.append((x.detach().cpu().clone(), y.detach().cpu().clone()))
    return out


def _assert_same_batches(legacy, device):
    assert len(legacy) == len(device), (len(legacy), len(device))
    for (lx, ly), (dx, dy) in zip(legacy, device):
        assert lx.shape == dx.shape and lx.dtype == dx.dtype
        assert torch.equal(lx, dx), "batch pixels differ"
        assert torch.equal(ly, dy), "batch labels differ"


def test_materialize_fast_path_matches_to_tensor_normalize():
    ds = _FakeCIFAR(transform=_CIFAR_TRANSFORM)
    arrays = db.materialize_dataset(ds, use_cache=False, verbose=False)
    assert arrays.x.shape == (len(ds), 3, 32, 32)
    assert arrays.mean == tuple(float(v) for v in _NORMALIZE.mean)
    for device in DEVICES:
        dataset = db.DeviceTensorDataset(
            arrays.x, arrays.y, device=device, mean=arrays.mean, std=arrays.std)
        x, y = dataset.fetch(torch.arange(len(ds)))
        for i in range(len(ds)):
            expected, target = ds[i]
            assert torch.equal(x[i].cpu(), expected), f"sample {i} differs"
            assert int(y[i]) == target


def test_shuffled_batches_and_rng_state_match_dataloader():
    ds = _FakeCIFAR(transform=_CIFAR_TRANSFORM)
    indices = np.where(np.isin(np.array(ds.targets), [1, 3, 4, 7]))[0]
    label_map = {1: 0, 3: 1, 4: 2, 7: 3}
    for device in DEVICES:
        for num_workers in (0, 2):
            torch.manual_seed(1234)
            legacy = _epochs(_legacy_task_loader(
                ds, indices, label_map, 8, True, num_workers=num_workers))
            legacy_state = torch.get_rng_state()
            torch.manual_seed(1234)
            fast = _epochs(_device_task_loader(
                ds, indices, label_map, 8, True, device))
            fast_state = torch.get_rng_state()
            _assert_same_batches(legacy, fast)
            assert torch.equal(legacy_state, fast_state), (
                f"global RNG diverged (device={device}, workers={num_workers})")


def test_sequential_loader_consumes_rng_like_dataloader():
    ds = _FakeCIFAR(transform=_CIFAR_TRANSFORM)
    indices = np.arange(len(ds))
    label_map = {c: c for c in range(10)}
    torch.manual_seed(7)
    legacy = _epochs(_legacy_task_loader(ds, indices, label_map, 16, False))
    after_legacy = torch.randn(3)
    torch.manual_seed(7)
    fast = _epochs(_device_task_loader(ds, indices, label_map, 16, False, "cpu"))
    after_fast = torch.randn(3)
    _assert_same_batches(legacy, fast)
    assert torch.equal(after_legacy, after_fast)


def test_interleaved_training_and_evaluation_keep_one_rng_stream():
    """A realistic loop: model init, shuffled train epochs, eval passes."""
    ds = _FakeCIFAR(transform=_CIFAR_TRANSFORM)
    indices = np.arange(len(ds))
    label_map = {c: c for c in range(10)}

    def run(make_train, make_test):
        torch.manual_seed(99)
        train, test = make_train(), make_test()
        init = torch.randn(5)                 # stands in for model init
        batches = []
        for _ in range(2):
            batches += _epochs(train, 1)
            batches += _epochs(test, 1)
        return init, batches, torch.get_rng_state()

    legacy = run(lambda: _legacy_task_loader(ds, indices, label_map, 8, True),
                 lambda: _legacy_task_loader(ds, indices, label_map, 16, False))
    fast = run(lambda: _device_task_loader(ds, indices, label_map, 8, True, "cpu"),
               lambda: _device_task_loader(ds, indices, label_map, 16, False, "cpu"))
    assert torch.equal(legacy[0], fast[0])
    _assert_same_batches(legacy[1], fast[1])
    assert torch.equal(legacy[2], fast[2])


def test_subset_composition_matches_torch_subset():
    ds = _FakeCIFAR(transform=_CIFAR_TRANSFORM)
    indices = np.arange(len(ds))[::2]
    label_map = {c: c for c in range(10)}
    legacy_loader = _legacy_task_loader(ds, indices, label_map, 8, True)
    fast_loader = _device_task_loader(ds, indices, label_map, 8, True, "cpu")
    part = np.random.RandomState(3).permutation(len(indices))[:20]

    torch.manual_seed(5)
    legacy = _epochs(db.make_loader(
        db.subset_dataset(legacy_loader.dataset, part), 8, True))
    torch.manual_seed(5)
    fast = _epochs(db.make_loader(
        db.subset_dataset(fast_loader.dataset, part), 8, True))
    _assert_same_batches(legacy, fast)
    assert isinstance(db.subset_dataset(legacy_loader.dataset, part), Subset)
    assert db.is_device_dataset(db.subset_dataset(fast_loader.dataset, part))


def test_permuted_mnist_pixels_match_legacy_wrapper():
    ds = _FakeCIFAR(n=50, size=28, channels=1,
                    transform=transforms.Compose([transforms.ToTensor()]))
    perm = torch.from_numpy(np.random.RandomState(11).permutation(784)).long()
    legacy = DataLoader(PermutedMNIST(ds, perm), batch_size=10, shuffle=True)
    arrays = db.materialize_dataset(ds, use_cache=False, verbose=False)
    assert arrays.x.shape == (50, 1, 28, 28) and arrays.mean is None
    for device in DEVICES:
        fast_ds = db.DeviceTensorDataset(arrays.x, arrays.y, device=device,
                                         perm=perm)
        torch.manual_seed(2)
        expected = _epochs(legacy, 1)
        torch.manual_seed(2)
        got = _epochs(db.DeviceLoader(fast_ds, 10, True), 1)
        _assert_same_batches(expected, got)


def test_generic_decode_path_and_cache_round_trip():
    """PIL-stage transforms (Resize/Grayscale) go through the worker decoder."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = _FakeCIFAR(n=23, size=12, channels=1, transform=transform)
    with tempfile.TemporaryDirectory() as tmp:
        # Decoding (a cache miss) must not consume the global RNG: otherwise
        # the first run of a benchmark would differ from every cached run.
        torch.manual_seed(77)
        before = torch.get_rng_state()
        first = db.materialize_dataset(ds, cache_dir=tmp, workers=2,
                                       verbose=False, name="fake")
        assert torch.equal(torch.get_rng_state(), before)
        assert first.x.shape == (23, 3, 16, 16)
        cached = [f for f in os.listdir(tmp) if f.endswith(".npz")]
        assert len(cached) == 1, cached
        second = db.materialize_dataset(ds, cache_dir=tmp, workers=0,
                                        verbose=False, name="fake")
        assert torch.equal(torch.get_rng_state(), before)
        assert np.array_equal(first.x, second.x)
        assert first.content_sha256 == second.content_sha256
    dataset = db.DeviceTensorDataset(first.x, first.y, device="cpu",
                                     mean=first.mean, std=first.std)
    x, _ = dataset.fetch(torch.arange(len(ds)))
    for i in range(len(ds)):
        expected, _ = ds[i]
        assert torch.equal(x[i], expected), f"resized sample {i} differs"


def _image_tree(root, shift=0):
    """A tiny ImageFolder train split and a TinyImageNet-style val split."""
    for c, wnid in enumerate(("n01", "n02")):
        (root / "train" / wnid).mkdir(parents=True)
        for i in range(3):
            Image.fromarray(np.full((8, 8, 3), 30 * i + 100 * c + shift, np.uint8)).save(
                root / "train" / wnid / f"{i}.png")
    (root / "val" / "images").mkdir(parents=True)
    lines = []
    for i in range(4):
        Image.fromarray(np.full((8, 8, 3), 20 * i + shift, np.uint8)).save(
            root / "val" / "images" / f"val_{i}.png")
        lines.append(f"val_{i}.png\t{('n01', 'n02')[i % 2]}\t0\t0\t8\t8")
    (root / "val" / "val_annotations.txt").write_text("\n".join(lines) + "\n")


def test_cache_entries_follow_the_source_files():
    """Changed, replaced or relocated images are decoded again, never served stale."""
    from torchvision import datasets
    from diltcl.hard_benchmarks import _TinyImageNetVal
    transform = transforms.Compose([transforms.ToTensor()])

    def train(root):
        return db.materialize_dataset(datasets.ImageFolder(str(root / "train"), transform=transform),
                                      cache_dir=str(cache), workers=0, verbose=False, name="t")

    def val(root):
        folder = datasets.ImageFolder(str(root / "train"))
        dataset = _TinyImageNetVal(str(root / "val"), folder.class_to_idx, transform)
        return db.materialize_dataset(dataset, cache_dir=str(cache), workers=0,
                                      verbose=False, name="v")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        cache = tmp / "cache"
        _image_tree(tmp / "a")
        _image_tree(tmp / "b", shift=7)          # another copy with other pixels
        first, first_val = train(tmp / "a"), val(tmp / "a")
        assert train(tmp / "a").content_sha256 == first.content_sha256      # cache hit
        assert len(list(cache.glob("*.npz"))) == 2
        # Another data root: its own entries, its own pixels (the val split
        # has no ``root`` attribute and used to share one entry).
        other_val = val(tmp / "b")
        assert other_val.content_sha256 != first_val.content_sha256
        assert int(other_val.x[0].min()) == 7
        # An image replaced in place is decoded again.
        target = tmp / "a" / "train" / "n01" / "0.png"
        Image.fromarray(np.full((8, 8, 3), 255, np.uint8)).save(target)
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        changed = train(tmp / "a")
        assert changed.content_sha256 != first.content_sha256 and int(changed.x[0].max()) == 255
        # A damaged cache file is refused instead of passing as the data.
        newest = max(cache.glob("t_*.npz"), key=lambda p: p.stat().st_mtime_ns)
        with np.load(newest) as archive:
            stored = {k: archive[k] for k in archive.files}
        stored["x"] = stored["x"].copy()
        stored["x"][0, 0, 0, 0] ^= 1
        np.savez(newest, **stored)
        try:
            train(tmp / "a")
        except RuntimeError as exc:
            assert "damaged" in str(exc)
        else:
            raise AssertionError("a damaged cache file was accepted")


def test_unsupported_tensor_stage_is_rejected():
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.RandomErasing(p=1.0)])
    ds = _FakeCIFAR(n=4, transform=transform)
    try:
        db.materialize_dataset(ds, use_cache=False, verbose=False)
    except ValueError as exc:
        assert "ToTensor" in str(exc) or "Normalize" in str(exc)
    else:
        raise AssertionError("non-reproducible tensor transforms must be rejected")


def test_make_loader_dispatches_by_dataset_type():
    ds = _FakeCIFAR(n=10, transform=_CIFAR_TRANSFORM)
    arrays = db.materialize_dataset(ds, use_cache=False, verbose=False)
    fast_ds = db.DeviceTensorDataset(arrays.x, arrays.y, device="cpu",
                                     mean=arrays.mean, std=arrays.std)
    assert isinstance(db.make_loader(fast_ds, 4, True), db.DeviceLoader)
    assert isinstance(db.make_loader(Subset(fast_ds, [0, 1]), 4, True),
                      db.DeviceLoader)
    assert isinstance(db.make_loader(ds, 4, True), DataLoader)
    assert len(db.make_loader(fast_ds, 4, True)) == 3


if __name__ == "__main__":
    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_") and callable(value)]
    for function in functions:
        function()
        print(f"PASS {function.__name__}")
    print(f"\nAll {len(functions)} data backend tests passed.")
