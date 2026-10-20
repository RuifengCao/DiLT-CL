"""``tools/compare_runs.py``: regeneration audit between two runs (torch-free)."""

import copy
import importlib.util
import json
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("compare_runs_under_test",
                                               ROOT / "tools" / "compare_runs.py")
cr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cr
_spec.loader.exec_module(cr)


def _payload(seeds=(0, 1), calibration="none", hashes=None):
    hashes = hashes or {s: f"{s:064x}" for s in seeds}
    return {
        "protocol": "class_il", "inference_criterion": "entropy",
        "inference_calibration": calibration, "calibration_fraction": 0.1,
        "calibration_c": 1.0, "calibration_split_seed": 1729,
        "config": {"lr": 0.001}, "code_version": "3.0.5",
        "benchmarks": ["cifar"], "seeds": list(seeds),
        "per_seed": {"cifar": {"wsn": [{"ACC": 10.0 + s, "OracleACC": 70.0 + s} for s in seeds]}},
        "per_seed_run_records": {"cifar": {"wsn": {
            str(s): {"trained_state": {"content_sha256": hashes[s]}} for s in seeds}}},
        "environment": {"pytorch_version": "2.12.1+cu130", "device_name": "GPU"},
        "execution": {"options": {"data_backend": "device"}},
    }


def _arrays(seeds=(0, 1)):
    out = {}
    for s in seeds:
        out[f"cifar/wsn/seed{s}"] = np.full((2, 2), float(s))
        out[f"cifar/wsn/seed{s}/task_inference"] = np.eye(2) * s
    return out


def _write(directory, name, payload, arrays):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(payload))
    np.savez(directory / f"{name}.npz", **arrays)


def _run(argv):
    with tempfile.TemporaryDirectory() as tmp:
        report = pathlib.Path(tmp) / "report.json"
        code = cr.main([*argv, "--out", str(report)])
        return code, json.loads(report.read_text())


def test_identical_rerun_with_extra_seeds_passes():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        _write(base / "old" / "merged", "g", _payload(), _arrays())
        _write(base / "new" / "merged", "g", _payload((0, 1, 2)), _arrays((0, 1, 2)))
        code, report = _run(["--reference", str(base / "old"), "--candidate", str(base / "new")])
        assert code == 0 and report["status"] == "identical"
        summary = report["groups"][0]["summary"]
        assert summary["compared"] == 2 and summary["identical"] == 2 and summary["new"] == 1


def test_same_state_but_different_matrix_is_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        _write(base / "old", "g", _payload(), _arrays())
        arrays = _arrays()
        arrays["cifar/wsn/seed1/task_inference"] = arrays["cifar/wsn/seed1/task_inference"] + 1e-9
        _write(base / "new", "g", _payload(), arrays)
        code, report = _run(["--reference", str(base / "old"), "--candidate", str(base / "new")])
        assert code == 1 and report["status"] == "differences"
        summary = report["groups"][0]["summary"]
        assert summary["same_state_different_outputs"] == ["cifar/wsn/seed1"]
        run = [r for r in report["groups"][0]["runs"] if r["run"] == "cifar/wsn/seed1"][0]
        assert run["array_differences"]["cifar/wsn/seed1/task_inference"] > 0


def test_different_state_and_metric_are_reported():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        _write(base / "old", "g", _payload(), _arrays())
        changed = _payload(hashes={0: "0" * 64, 1: "f" * 64})
        changed["per_seed"]["cifar"]["wsn"][1]["ACC"] = 99.0
        changed["environment"]["device_name"] = "Other GPU"
        _write(base / "new", "g", changed, _arrays())
        code, report = _run(["--reference", str(base / "old"), "--candidate", str(base / "new")])
        assert code == 1
        group = report["groups"][0]
        assert group["summary"]["state_different"] == ["cifar/wsn/seed1"]
        assert "device_name" in group["environment_differences"]
        run = [r for r in group["runs"] if r["run"] == "cifar/wsn/seed1"][0]
        assert run["metric_differences"]["ACC"] == [99.0, 11.0]


def test_identity_difference_blocks_an_identical_verdict():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        _write(base / "old", "g", _payload(), _arrays())
        other = _payload()
        other["config"] = {"lr": 0.01}
        _write(base / "new", "g", other, _arrays())
        code, report = _run(["--reference", str(base / "old"), "--candidate", str(base / "new")])
        assert code == 1 and "config" in report["groups"][0]["identity_differences"]


def test_states_only_pair_across_routers():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        _write(base, "cal", _payload(calibration="score-logreg"), _arrays())
        bias = _payload(calibration="score-bias")
        bias["per_seed"]["cifar"]["wsn"][0]["ACC"] = 50.0      # different router output
        _write(base, "bias", bias, _arrays())
        code, report = _run(["--states-only", "--pair", str(base / "cal.json"),
                             str(base / "bias.json")])
        assert code == 0 and report["states_only"] is True
        assert report["groups"][0]["summary"]["state_identical"] == 2
        broken = copy.deepcopy(bias)
        broken["per_seed_run_records"]["cifar"]["wsn"]["0"]["trained_state"]["content_sha256"] = "e" * 64
        _write(base, "broken", broken, _arrays())
        code, report = _run(["--states-only", "--pair", str(base / "cal.json"),
                             str(base / "broken.json")])
        assert code == 1 and report["groups"][0]["summary"]["state_different"] == ["cifar/wsn/seed0"]


def test_partial_candidate_is_incomplete_not_identical():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        _write(base / "old" / "merged", "g", _payload((0, 1, 2)), _arrays((0, 1, 2)))
        _write(base / "old" / "merged", "h", _payload(), _arrays())
        # Candidate: seed 2 of group g missing, group h missing entirely.
        _write(base / "new" / "merged", "g", _payload(), _arrays())
        code, report = _run(["--reference", str(base / "old"), "--candidate", str(base / "new")])
        assert code == 1 and report["status"] == "incomplete" and not report["complete"]
        assert report["runs_missing_in_candidate"] == 1
        assert report["groups_missing_in_candidate"] == ["h"]
        # Restricting the audit to the group that was rerun still demands every seed.
        code, report = _run(["--reference", str(base / "old"), "--candidate", str(base / "new"),
                             "--groups", "g"])
        assert code == 1 and report["status"] == "incomplete"
        _write(base / "new" / "merged", "g", _payload((0, 1, 2)), _arrays((0, 1, 2)))
        code, report = _run(["--reference", str(base / "old"), "--candidate", str(base / "new"),
                             "--groups", "g"])
        assert code == 0 and report["status"] == "identical"


def test_json_without_npz_and_missing_state_hash_block_identical():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        _write(base / "old", "g", _payload(), _arrays())
        _write(base / "new", "g", _payload(), _arrays())
        (base / "new" / "orphan.json").write_text("{}")
        code, report = _run(["--reference", str(base / "old"), "--candidate", str(base / "new")])
        assert code == 1 and report["status"] == "incomplete"
        assert report["unpaired_json"] == [str(base / "new" / "orphan.json")]
        (base / "new" / "orphan.json").unlink()
        stripped = _payload()
        stripped["per_seed_run_records"]["cifar"]["wsn"]["1"] = {}
        _write(base / "new", "g", stripped, _arrays())
        code, report = _run(["--reference", str(base / "old"), "--candidate", str(base / "new")])
        assert code == 1 and report["runs_without_state_hash"] == 1
        assert report["runs_identical"] == 1


def test_state_file_hashes_are_compared_and_reported():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        old = _payload()
        for s in (0, 1):
            old["per_seed_run_records"]["cifar"]["wsn"][str(s)]["trained_state"].update(
                sha256=f"{s + 7:064x}", serialized_bytes=100)
        new = copy.deepcopy(old)
        new["per_seed_run_records"]["cifar"]["wsn"]["1"]["trained_state"]["sha256"] = "d" * 64
        _write(base / "old", "g", old, _arrays())
        _write(base / "new", "g", new, _arrays())
        code, report = _run(["--reference", str(base / "old"), "--candidate", str(base / "new")])
        # Same tensors serialized differently: the experts are identical, the files are not.
        assert code == 0 and report["status"] == "identical"
        assert report["state_files_identical"] == 1
        assert report["groups"][0]["summary"]["file_different"] == ["cifar/wsn/seed1"]


def test_data_and_launch_controls_are_compared_from_run_statuses():
    def status(entries, cublas=":4096:8", seed="0"):
        return {"state": "succeeded", "job": {"job_id": "g__cifar__seed0__wsn"},
                "dataset_identity": {"entries": entries},
                "child_environment_controls": {"CUBLAS_WORKSPACE_CONFIG": cublas,
                                               "OMP_NUM_THREADS": "3", "PYTHONHASHSEED": seed}}
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        for name in ("old", "new"):
            _write(base / name / "merged", "g", _payload(), _arrays())
            (base / name / "statuses").mkdir()
        (base / "old" / "statuses" / "g__cifar__seed0__wsn.json").write_text(
            json.dumps(status({"cifar-100-python": "a" * 64})))
        (base / "new" / "statuses" / "g__cifar__seed0__wsn.json").write_text(
            json.dumps(status({"cifar-100-python": "b" * 64}, cublas=":16:8")))
        code, report = _run(["--reference", str(base / "old"), "--candidate", str(base / "new")])
        launch = report["data_and_launch"]
        assert launch["jobs_compared"] == 1
        assert launch["dataset_differences"] == ["g__cifar__seed0__wsn"]
        assert launch["launch_differences"] == [{"job": "g__cifar__seed0__wsn",
                                                 "keys": ["CUBLAS_WORKSPACE_CONFIG"]}]
        # Reported, not part of the verdict: the outputs themselves are identical.
        assert code == 0 and report["status"] == "identical"


def test_verifier_can_write_its_report_outside_a_historical_run():
    spec = importlib.util.spec_from_file_location(
        "verify_bundle_for_out_test", ROOT / "tools" / "verify_experiment_bundle.py")
    verifier = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = verifier
    spec.loader.exec_module(verifier)
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        run = base / "historical_run"
        run.mkdir()
        (run / "verification_report.json").write_text('{"status": "verified"}')
        code = verifier.main([str(run), "--out", str(base / "recheck.json")])
        assert code == 1                                  # empty run: not verifiable
        assert json.loads((base / "recheck.json").read_text())["status"] != "verified"
        # The original report is untouched.
        assert json.loads((run / "verification_report.json").read_text()) == {"status": "verified"}


def test_environment_check_parses_the_lock_and_flags_mismatches():
    spec = importlib.util.spec_from_file_location(
        "check_environment_under_test", ROOT / "tools" / "check_environment.py")
    ce = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ce
    spec.loader.exec_module(ce)
    pins = ce.parse_lock(ROOT / "requirements-lock.txt")
    assert pins["torch"] == "2.12.1+cu130" and pins["torchvision"] == "0.27.1+cu130"
    assert pins["numpy"] == "2.4.6" and pins["scikit-learn"] == "1.9.0"
    installed_numpy = np.__version__
    assert ce.check_packages({"numpy": installed_numpy}) == []
    assert ce.check_packages({"numpy": "0.0.1"}) == [f"numpy: installed {installed_numpy}, recorded 0.0.1"]
    assert ce.check_packages({"surely-not-installed-pkg": "1.0"})[0].endswith("not installed (recorded 1.0)")
