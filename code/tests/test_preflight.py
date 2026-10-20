"""Tests for :mod:`diltcl.preflight` and the dependency-free calibration solver.

These cover the two conditions that destroyed five of seven jobs in the first
v2.1 P0 suite: an absent optional package and unstaged datasets. Both must now
be caught *before* training, and the calibration path must survive a missing
scikit-learn instead of crashing after the models are trained.

Torch-free: only NumPy/SciPy are required.
"""

import importlib.util
import os
import pathlib
import sys
import tempfile

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_PKG = _HERE.parent / "diltcl"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _PKG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = _load("preflight")
task_calibration = _load("task_calibration")


# ---------------------------------------------------------------------------
# Preflight: datasets
# ---------------------------------------------------------------------------

def test_standalone_preflight_checks_the_default_data_root():
    # Without --data-root or $DILTCL_DATA_ROOT the loaders read ./data; the
    # standalone check must look there too instead of skipping the datasets.
    saved_root, saved_cwd = os.environ.pop("DILTCL_DATA_ROOT", None), os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            assert preflight.main(["--benchmarks", "cifar", "--no-training"]) == 1
        finally:
            os.chdir(saved_cwd)
            if saved_root is not None:
                os.environ["DILTCL_DATA_ROOT"] = saved_root


def test_missing_dataset_blocks_with_actionable_message():
    with tempfile.TemporaryDirectory() as root:
        report = preflight.check(root, ["pmnist"], require_training=False)
        assert not report["ok"]
        joined = " ".join(report["blocking"])
        assert "MNIST" in joined
        # the message must tell the user how to fix it, not just what failed
        assert "torchvision" in joined or "copy" in joined


def test_present_dataset_passes_only_with_the_exact_torchvision_files():
    """A directory that merely exists is not a dataset. The first P0 run had
    ``MNIST/raw`` holding differently named archives; preflight passed and
    all 50 Permuted-MNIST jobs died at startup."""
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "MNIST", "raw"))
        pathlib.Path(root, "MNIST", "raw", "train-images.idx3-ubyte.zip").write_bytes(b"x")
        report = preflight.check(root, ["pmnist"], require_training=False)
        assert not report["ok"]
        assert "train-images-idx3-ubyte" in " ".join(report["blocking"])
        for name in preflight._MNIST_RAW_FILES:
            pathlib.Path(root, "MNIST", "raw", name).write_bytes(b"x")
        report = preflight.check(root, ["pmnist"], require_training=False)
        assert report["ok"], report["blocking"]


def test_unextracted_archive_is_not_enough_when_downloads_are_disabled():
    # With DILTCL_DOWNLOAD=0 (every P0/P1 run) torchvision opens the extracted
    # files and never touches the archive, so the archive alone must block.
    with tempfile.TemporaryDirectory() as root:
        pathlib.Path(root, "cifar-100-python.tar.gz").write_bytes(b"x")
        report = preflight.check(root, ["cifar"], require_training=False)
        assert not report["ok"]
        assert "cifar-100-python/train" in " ".join(report["blocking"])
        os.makedirs(os.path.join(root, "cifar-100-python"))
        for name in ("train", "test", "meta"):
            pathlib.Path(root, "cifar-100-python", name).write_bytes(b"x")
        report = preflight.check(root, ["cifar"], require_training=False)
        assert report["ok"], report["blocking"]


def test_every_benchmark_reports_all_its_datasets():
    with tempfile.TemporaryDirectory() as root:
        report = preflight.check(root, ["fivedatasets"], require_training=False)
        joined = " ".join(report["blocking"])
        for name in ("CIFAR-10", "MNIST", "SVHN", "FashionMNIST", "KMNIST"):
            assert name in joined, name


def test_multiple_benchmarks_deduplicate_shared_datasets():
    needed = preflight.required_datasets(["pmnist", "fivedatasets"])
    assert needed.count("MNIST") == 1


def test_unknown_benchmark_is_rejected():
    try:
        preflight.required_datasets(["not_a_benchmark"])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unknown benchmark" in str(exc)


def test_missing_data_root_blocks():
    report = preflight.check("/definitely/not/here", ["cifar"],
                             require_training=False)
    assert not report["ok"]
    assert any("does not exist" in m for m in report["blocking"])


# ---------------------------------------------------------------------------
# Preflight: optional dependency policy
# ---------------------------------------------------------------------------

def test_absent_sklearn_warns_but_does_not_block(monkey_absent=True):
    original = preflight.module_available
    preflight.module_available = lambda name: name != "sklearn"
    try:
        report = preflight.check(None, [], require_training=False,
                                 require_calibration=True)
        assert report["ok"], report["blocking"]
        assert any("scipy-lbfgs" in w for w in report["warnings"])
    finally:
        preflight.module_available = original


def test_absent_sklearn_blocks_when_pinned():
    original = preflight.module_available
    preflight.module_available = lambda name: name != "sklearn"
    try:
        report = preflight.check(None, [], require_training=False,
                                 require_calibration=True, pin_sklearn=True)
        assert not report["ok"]
        assert any("scikit-learn" in m for m in report["blocking"])
    finally:
        preflight.module_available = original


def test_missing_training_stack_blocks():
    original = preflight.module_available
    preflight.module_available = lambda name: name not in ("torch",)
    try:
        report = preflight.check(None, [], require_training=True)
        assert not report["ok"]
        assert any("torch" in m for m in report["blocking"])
    finally:
        preflight.module_available = original


def test_report_text_is_actionable():
    report = {"ok": False, "blocking": ["dataset 'X' not found -> get it"],
              "warnings": []}
    text = preflight.format_report(report)
    assert "BLOCKED" in text and "Nothing was trained" in text


# ---------------------------------------------------------------------------
# Calibration solver: no hard scikit-learn dependency
# ---------------------------------------------------------------------------

def _toy(n_classes=4, n_features=5, per_class=60, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        mu = rng.normal(0.0, 1.5, n_features)
        xs.append(rng.normal(mu, 1.0, (per_class, n_features)))
        ys.append(np.full(per_class, c))
    return np.vstack(xs), np.concatenate(ys)


def test_fit_succeeds_without_sklearn():
    x, y = _toy()
    original = task_calibration._sklearn_available
    task_calibration._sklearn_available = lambda: False
    try:
        state = task_calibration.fit_score_logreg(x, y, C=1.0)
    finally:
        task_calibration._sklearn_available = original
    assert state["solver_backend"] == "scipy-lbfgs"
    assert state["converged"]
    # the fitted router must actually route better than chance
    pred = task_calibration.predict_score_logreg(x, state)
    assert np.mean(pred == y) > 0.5


def test_backend_is_recorded_for_provenance():
    x, y = _toy()
    state = task_calibration.fit_score_logreg(x, y, C=1.0)
    assert state["solver_backend"] in (
        "sklearn-lbfgs", "scipy-lbfgs", "degenerate-single-class")


def test_both_backends_agree_on_routing_decisions():
    """The fallback must not change which task the router picks."""
    if not task_calibration._sklearn_available():
        return  # nothing to compare against in this environment
    x, y = _toy(n_classes=6, n_features=4, per_class=50, seed=3)
    sk = task_calibration.fit_score_logreg(x, y, C=1.0)
    original = task_calibration._sklearn_available
    task_calibration._sklearn_available = lambda: False
    try:
        sp = task_calibration.fit_score_logreg(x, y, C=1.0)
    finally:
        task_calibration._sklearn_available = original
    assert np.array_equal(task_calibration.predict_score_logreg(x, sk),
                          task_calibration.predict_score_logreg(x, sp))


def test_single_class_holdout_is_degenerate_not_an_error():
    x = np.random.default_rng(0).normal(size=(20, 3))
    y = np.zeros(20, dtype=int)
    state = task_calibration.fit_score_logreg(x, y, C=1.0)
    assert state["solver_backend"] == "degenerate-single-class"
    assert state["classes"] == [0]


def test_prediction_is_independent_of_the_fitting_backend():
    """Serialized state must predict without importing any solver."""
    x, y = _toy(n_classes=3, per_class=40, seed=7)
    original = task_calibration._sklearn_available
    task_calibration._sklearn_available = lambda: False
    try:
        state = task_calibration.fit_score_logreg(x, y, C=1.0)
    finally:
        task_calibration._sklearn_available = original
    # predict_score_logreg uses only numpy arithmetic over the saved state
    manual_z = ((x - np.array(state["feature_mean"]))
                / np.array(state["feature_scale"]))
    scores = manual_z @ np.array(state["coef"]).T + np.array(state["intercept"])
    expected = np.array(state["classes"])[scores.argmax(axis=1)]
    assert np.array_equal(task_calibration.predict_score_logreg(x, state),
                          expected)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} preflight/calibration tests passed.")
    sys.exit(0)
