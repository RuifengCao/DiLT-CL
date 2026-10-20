"""Contract tests for the experiment runners.

Almost every top-level function in ``run_class_il.py`` and
``run_task_il.py`` was previously unreferenced by any test. That gap
is not theoretical: a refactor of ``run()`` introduced a self-recursive
validation helper and four ``undefined name`` references, and the whole suite
still reported 16/16 passing because nothing ever called ``run()``.

These tests close the most dangerous part of that gap. They exercise the
argument-validation path, the preflight gate and the small pure helpers of both
runners without training anything, so they stay fast while catching the class
of mistake that slipped through before.

Skipped automatically when PyTorch is unavailable (the runners import torch).
"""

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    import torch  # noqa: F401 - the runners require it at import time
except ImportError:  # pragma: no cover - torch-free environment
    print("PyTorch not installed; runner contract tests skipped.")
    sys.exit(0)

from diltcl import preflight  # noqa: E402
from diltcl.config import ExperimentConfig  # noqa: E402

import run_task_il as taskil  # noqa: E402
import run_class_il as class_il  # noqa: E402


def _staged_root():
    """A temporary data root that satisfies the CIFAR-100 preflight check.

    Preflight checks the files torchvision opens, not just the directory, so
    the three CIFAR-100 files must exist (their content is irrelevant here).
    """
    directory = tempfile.mkdtemp()
    os.makedirs(os.path.join(directory, "cifar-100-python"), exist_ok=True)
    for name in ("train", "test", "meta"):
        with open(os.path.join(directory, "cifar-100-python", name), "wb") as f:
            f.write(b"x")
    return directory


def _base_kwargs(**overrides):
    kwargs = {
        "cfg": ExperimentConfig.fast(),
        "benchmarks": ["cifar"],
        "methods": ["derpp"],
        "seeds": [0],
    }
    kwargs.update(overrides)
    return kwargs


def _expect_value_error(callable_, **kwargs):
    try:
        callable_(**kwargs)
    except ValueError:
        return True
    except SystemExit:
        raise AssertionError("preflight ran before argument validation")
    except Exception as exc:  # noqa: BLE001 - any other failure is also wrong
        raise AssertionError(f"expected ValueError, got {type(exc).__name__}: {exc}")
    raise AssertionError("expected ValueError, but the call succeeded")


# ---------------------------------------------------------------------------
# run() is actually callable (the regression that a green suite missed)
# ---------------------------------------------------------------------------

def test_class_il_run_reaches_validation_without_name_errors():
    """A NameError or infinite recursion in run() must fail here, not on a GPU."""
    assert _expect_value_error(class_il.run, **_base_kwargs(seeds=[]))


def test_task_il_run_reaches_validation_without_name_errors():
    assert _expect_value_error(taskil.run, **_base_kwargs(seeds=[], profile=False))


def test_validation_helper_is_not_self_recursive():
    """Direct call: a self-recursive helper would raise RecursionError."""
    class_il._validate_run_arguments(
        ExperimentConfig.fast(), ["derpp"], [0], "none", 0.1, 1.0, False)


# ---------------------------------------------------------------------------
# Argument validation precedes every side effect
# ---------------------------------------------------------------------------

def test_duplicate_methods_rejected():
    assert _expect_value_error(class_il.run,
                               **_base_kwargs(methods=["derpp", "derpp"]))
    assert _expect_value_error(taskil.run,
                               **_base_kwargs(methods=["derpp", "derpp"],
                                              profile=False))


def test_unknown_calibration_rejected_before_device_setup():
    assert _expect_value_error(class_il.run,
                               **_base_kwargs(calibration="bogus"))


def test_calibration_accepts_shared_only_jobs_after_validation():
    """The suite runs one method per job. A calibration job holding only a
    shared-backbone method must pass argument validation (it trains on the
    same 90% holdout and records no router) -- the v3.0.0 rule that required
    an isolation method killed 50 P0 jobs at startup. The call must get past
    validation and preflight; here it then fails on the missing dataset."""
    previous = os.environ.get("DILTCL_DATA_ROOT")
    os.environ["DILTCL_DATA_ROOT"] = _staged_root()
    try:
        try:
            class_il.run(**_base_kwargs(calibration="score-logreg",
                                        methods=["derpp"]))
        except (ValueError, SystemExit) as exc:
            raise AssertionError(
                f"shared-only calibration job was rejected before data: {exc}")
        except Exception:  # noqa: BLE001 - loader failure is expected here
            pass
    finally:
        os.environ.pop("DILTCL_DATA_ROOT", None)
        if previous is not None:
            os.environ["DILTCL_DATA_ROOT"] = previous
    # The fraction bound still applies whatever the method family.
    assert _expect_value_error(
        class_il.run, **_base_kwargs(calibration="score-logreg",
                                     methods=["derpp"], calibration_fraction=1.5))


def test_unknown_head_mode_rejected_before_data_loading():
    assert _expect_value_error(taskil.run,
                               **_base_kwargs(head_mode="bogus", profile=False))


# ---------------------------------------------------------------------------
# Preflight resolves the data root the same way the loaders do
# ---------------------------------------------------------------------------

def test_preflight_uses_the_loader_default_root():
    """Unset DILTCL_DATA_ROOT must still be checked, not silently skipped."""
    from diltcl.dataset_utils import default_data_root

    previous = os.environ.pop("DILTCL_DATA_ROOT", None)
    try:
        assert preflight.resolve_data_root() == str(default_data_root())
    finally:
        if previous is not None:
            os.environ["DILTCL_DATA_ROOT"] = previous


def test_runner_blocks_when_the_default_root_lacks_data():
    previous = os.environ.pop("DILTCL_DATA_ROOT", None)
    os.environ["DILTCL_DATA_ROOT"] = tempfile.mkdtemp()   # empty
    try:
        try:
            taskil.run(**_base_kwargs(profile=False))
            raise AssertionError("expected preflight to block the run")
        except SystemExit as exc:
            assert exc.code == 1
    finally:
        os.environ.pop("DILTCL_DATA_ROOT", None)
        if previous is not None:
            os.environ["DILTCL_DATA_ROOT"] = previous


def test_valid_arguments_pass_validation_and_preflight():
    """With data staged, the run proceeds past both gates."""
    previous = os.environ.get("DILTCL_DATA_ROOT")
    os.environ["DILTCL_DATA_ROOT"] = _staged_root()
    try:
        # Training would need the real dataset; reaching the loader means both
        # gates passed, which is exactly what this test asserts.
        try:
            taskil.run(**_base_kwargs(profile=False))
        except SystemExit:
            raise AssertionError("preflight blocked a correctly staged run")
        except Exception:  # noqa: BLE001 - loader failure is expected here
            pass
    finally:
        os.environ.pop("DILTCL_DATA_ROOT", None)
        if previous is not None:
            os.environ["DILTCL_DATA_ROOT"] = previous


# ---------------------------------------------------------------------------
# Shared helpers behave identically in both runners
# ---------------------------------------------------------------------------

def test_runners_share_one_implementation():
    assert taskil._parse_seed_spec is class_il._parse_seed_spec
    assert taskil._state_content_sha256 is class_il._state_content_sha256
    assert taskil._persist_checkpoint is class_il._persist_checkpoint


def test_seed_spec_parsing():
    assert taskil._parse_seed_spec("0-4") == [0, 1, 2, 3, 4]
    assert taskil._parse_seed_spec("0,2,5") == [0, 2, 5]


def test_calibration_seed_is_derived_once():
    """The holdout split and the router fit must agree by construction."""
    assert (class_il._calibration_seed(1729, 3)
            == class_il._calibration_seed(1729, 3))
    assert (class_il._calibration_seed(1729, 0)
            != class_il._calibration_seed(1729, 1))


def test_head_mode_wrapper_builds_the_requested_protocol():
    from diltcl.models import MultiHeadModel, SimpleCNN

    base = lambda: SimpleCNN(num_classes=5)          # noqa: E731
    assert taskil._wrap_head_mode(base, "shared", 4, 5) is base
    wrapped = taskil._wrap_head_mode(base, "multi", 4, 5)()
    assert isinstance(wrapped, MultiHeadModel)
    assert wrapped.num_tasks == 4
    try:
        taskil._wrap_head_mode(base, "bogus", 4, 5)
        raise AssertionError("expected ValueError for an unknown head mode")
    except ValueError:
        pass


def test_multi_head_is_confined_to_shared_backbone_baselines():
    """Isolation methods must never receive a multi-head model.

    They already own per-task parameters; a multi-head wrapper would expose
    every head to the masking/pruning machinery and silently change what
    PackNet, SupSup, WSN, DiLT-CL and Independent mean.
    """
    from diltcl.models import MultiHeadModel, SimpleCNN

    base = lambda: SimpleCNN(num_classes=5)          # noqa: E731
    builders = (base, taskil._wrap_head_mode(base, "multi", 4, 5))
    for method in ("finetune", "ewc", "lwf", "er", "derpp"):
        model = taskil.builder_for_method(method, builders)()
        assert isinstance(model, MultiHeadModel), method
    for method in ("packnet", "supsup", "wsn", "diltcl", "independent"):
        model = taskil.builder_for_method(method, builders)()
        assert not isinstance(model, MultiHeadModel), method


def test_shared_mode_gives_every_method_the_same_builder():
    base = lambda: None                              # noqa: E731
    builders = (base, taskil._wrap_head_mode(base, "shared", 4, 5))
    for method in taskil.ALL_METHODS:
        assert taskil.builder_for_method(method, builders) is base, method


def test_verifier_reports_instead_of_crashing_on_incomplete_plans():
    """A plan entry missing fields must produce a diagnostic, not a KeyError."""
    import json
    import subprocess

    root = pathlib.Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as directory:
        # Deliberately incomplete: no priority, no seed, no benchmark.
        (pathlib.Path(directory) / "plan.json").write_text(json.dumps(
            {"jobs": [{"output_prefix_relative": "artifacts/j/result",
                       "job": {"job_id": "j"}}]}))
        completed = subprocess.run(
            [sys.executable, str(root / "tools" / "verify_experiment_bundle.py"),
             directory], capture_output=True, text=True)
        combined = completed.stdout + completed.stderr
        assert "Traceback" not in combined, combined[-400:]
        assert completed.returncode != 0        # incomplete bundles still fail


def test_code_version_is_derived_from_the_package_version():
    """Provenance must not drift.

    Both runners once hardcoded ``CODE_VERSION = "2.1.1"`` independently of
    ``diltcl.__version__``. Eight releases later the result files still claimed
    2.1.1, and that string is printed into table captions -- so artifacts
    misattributed themselves to the version that produced them.
    """
    import diltcl

    assert taskil.CODE_VERSION == diltcl.__version__
    assert class_il.CODE_VERSION == diltcl.__version__


def test_no_hardcoded_version_literal_remains_in_the_runners():
    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("run_task_il.py", "run_class_il.py"):
        text = (root / name).read_text()
        assert 'CODE_VERSION = "' not in text, (
            f"{name} reintroduced a hardcoded CODE_VERSION literal")


if __name__ == "__main__":
    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_") and callable(value)]
    for function in functions:
        function()
        print(f"PASS {function.__name__}")
    print(f"\nAll {len(functions)} runner contract tests passed.")
    sys.exit(0)
