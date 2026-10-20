"""Tests for the offline-first dataset loader (:mod:`diltcl.dataset_utils`).

These validate the logic that fixes the ``HTTP 503`` download failure: prefer a
local copy (no network), download only when allowed, and fail with guidance.
A fake dataset class stands in for torchvision, so the tests need no torch and
run anywhere.
"""

import importlib.util
import os
import pathlib

# dataset_utils has no intra-package imports, so load it straight from the file
# to avoid triggering ``diltcl/__init__`` (which imports torch).
_path = pathlib.Path(__file__).resolve().parents[1] / "diltcl" / "dataset_utils.py"
_spec = importlib.util.spec_from_file_location("diltcl_dataset_utils", _path)
du = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(du)


class FakeDS:
    """Stand-in torchvision dataset whose behaviour the tests configure.

    ``present``        -- a local copy exists (so ``download=False`` succeeds).
    ``download_works`` -- a download (or local-archive extract) would succeed.
    ``calls``          -- records the ``download`` flag of each construction.
    """

    present = False
    download_works = True
    calls = []

    def __init__(self, root, train, download, transform):
        type(self).calls.append(download)
        if download:
            if not type(self).download_works:
                raise OSError("simulated HTTP 503")
        elif not type(self).present:
            raise RuntimeError("Dataset not found. You can use download=True "
                               "to download it")
        self.root, self.train, self.transform = root, train, transform

    @classmethod
    def reset(cls, present, download_works):
        cls.present = present
        cls.download_works = download_works
        cls.calls = []


def _load(download):
    return du.load_torchvision_dataset(
        FakeDS, root="./data", train=True, transform=None,
        download=download, name="CIFAR-100")


def test_local_copy_used_without_network():
    FakeDS.reset(present=True, download_works=True)
    ds = _load(download=True)
    assert isinstance(ds, FakeDS)
    assert FakeDS.calls == [False], "should not attempt a download when present"


def test_missing_and_download_disabled_raises_with_guidance():
    FakeDS.reset(present=False, download_works=True)
    try:
        _load(download=False)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        msg = str(e)
        assert "Could not load the CIFAR-100" in msg
        assert "DILTCL_DATA_ROOT" in msg                 # actionable hint
        assert FakeDS.calls == [False], "must not download when disabled"


def test_missing_then_download_succeeds():
    FakeDS.reset(present=False, download_works=True)
    ds = _load(download=True)
    assert isinstance(ds, FakeDS)
    assert FakeDS.calls == [False, True], "tries local first, then downloads"


def test_missing_and_download_fails_raises_with_cause():
    FakeDS.reset(present=False, download_works=False)
    try:
        _load(download=True)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        msg = str(e)
        assert "automatic download was attempted but failed" in msg
        assert "503" in msg                              # surfaces the cause
        assert FakeDS.calls == [False, True]


def test_env_var_defaults():
    old_root = os.environ.pop("DILTCL_DATA_ROOT", None)
    old_dl = os.environ.pop("DILTCL_DOWNLOAD", None)
    try:
        assert du.default_data_root() == "./data"
        assert du.download_enabled() is True
        os.environ["DILTCL_DATA_ROOT"] = "/tmp/some_datasets"
        os.environ["DILTCL_DOWNLOAD"] = "0"
        assert du.default_data_root() == "/tmp/some_datasets"
        assert du.download_enabled() is False
        for falsey in ("false", "No", "OFF", ""):
            os.environ["DILTCL_DOWNLOAD"] = falsey
            assert du.download_enabled() is False
        os.environ["DILTCL_DOWNLOAD"] = "1"
        assert du.download_enabled() is True
    finally:
        os.environ.pop("DILTCL_DATA_ROOT", None)
        os.environ.pop("DILTCL_DOWNLOAD", None)
        if old_root is not None:
            os.environ["DILTCL_DATA_ROOT"] = old_root
        if old_dl is not None:
            os.environ["DILTCL_DOWNLOAD"] = old_dl


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} dataset-utils tests passed.")
    sys.exit(0)
