"""``tools/build_release.py``: verified release archives of complete run directories."""

import hashlib
import importlib.util
import json
import pathlib
import sys
import tarfile
import tempfile

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_release_under_test",
                                               ROOT / "tools" / "build_release.py")
br = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = br
_spec.loader.exec_module(br)


def _job(run: pathlib.Path, job_id: str, benchmark: str, payload_bytes: bytes,
         with_predictions: bool):
    job = run / "artifacts" / job_id
    (job / "checkpoints" / benchmark / "wsn").mkdir(parents=True)
    ckpt = job / "checkpoints" / benchmark / "wsn" / "seed0.pt"
    ckpt.write_bytes(payload_bytes)
    state = {"persisted": True, "persisted_relative_path": f"checkpoints/{benchmark}/wsn/seed0.pt",
             "serialized_bytes": len(payload_bytes),
             "sha256": hashlib.sha256(payload_bytes).hexdigest(), "content_sha256": "c" * 64}
    payload = {"per_seed_run_records": {benchmark: {"wsn": {"0": {"trained_state": state}}}}}
    if with_predictions:
        (job / "predictions" / benchmark / "wsn").mkdir(parents=True)
        pred = job / "predictions" / benchmark / "wsn" / "seed0.npz"
        np.savez(pred, route_by_stage=np.zeros((3, 2), np.int16))
        payload["calibration"] = {benchmark: {"wsn": {"0": {"predictions": {
            "relative_path": f"predictions/{benchmark}/wsn/seed0.npz",
            "sha256": hashlib.sha256(pred.read_bytes()).hexdigest()}}}}}
    (job / "result.json").write_text(json.dumps(payload))
    np.savez(job / "result.npz", x=np.zeros(2))
    (run / "statuses").mkdir(exist_ok=True)
    (run / "statuses" / f"{job_id}.json").write_text(json.dumps({"state": "succeeded"}))
    return {"job": {"job_id": job_id, "benchmark": benchmark}}


def _run_dir(base: pathlib.Path, name="p0"):
    run = base / name
    jobs = [_job(run, "grp_a__cifar__seed0__wsn", "cifar", b"A" * 3000, True),
            _job(run, "grp_a__tinyimagenet__seed0__wsn", "tinyimagenet", b"B" * 3000, True),
            _job(run, "grp_b__cifar__seed0__wsn", "cifar", b"C" * 100, False)]
    (run / "plan.json").write_text(json.dumps({"jobs": jobs}))
    (run / "merged").mkdir()
    (run / "merged" / "grp_a.json").write_text("{}")
    return run


def test_release_round_trip_and_group_split():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        run = _run_dir(base)
        # 4000 bytes per archive: group a (two 3000-byte checkpoints) splits by benchmark.
        manifest = br.build([run], base / "out", max_bytes=4000, tokens=["nobody"])
        names = sorted(a["archive"] for a in manifest["archives"])
        assert names == ["p0__grp_a__cifar.tar", "p0__grp_a__tinyimagenet.tar",
                         "p0__grp_b.tar", "p0__meta.tar"]
        assert manifest["checkpoints"] == 3 and manifest["prediction_files"] == 2
        restored = base / "restored"
        for archive in manifest["archives"]:
            path = base / "out" / archive["archive"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == archive["sha256"]
            with tarfile.open(path) as tar:
                tar.extractall(restored, filter="data")
        for original in run.rglob("*"):
            if original.is_file():
                copy = restored / original.relative_to(run)
                assert copy.read_bytes() == original.read_bytes(), original


def test_release_refuses_a_checkpoint_that_differs_from_its_record():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        run = _run_dir(base)
        (run / "artifacts" / "grp_b__cifar__seed0__wsn" / "checkpoints" / "cifar" / "wsn"
         / "seed0.pt").write_bytes(b"D" * 100)
        with pytest.raises(SystemExit, match="recorded SHA-256"):
            br.build([run], base / "out", max_bytes=10**9, tokens=[])


def test_release_refuses_identifying_tokens():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        run = _run_dir(base)
        (run / "merged" / "note.json").write_text('{"author": "Jane Example"}')
        with pytest.raises(SystemExit, match="anonymity"):
            br.build([run], base / "out", max_bytes=10**9, tokens=["jane example"])
