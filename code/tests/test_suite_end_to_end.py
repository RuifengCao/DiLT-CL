"""End-to-end: the real launcher, runners, verifier, merger and figure tools.

Runs ``run_suite.py --profile smoke`` as a real process tree on a synthetic
on-disk CIFAR-100 (100 classes, a handful of images each). Child processes
pick up the synthetic dataset through a ``sitecustomize.py`` on ``PYTHONPATH``
that points torchvision's CIFAR-100 integrity table at the synthetic files.
Then ``tools/verify_experiment_bundle.py`` must report ``verified`` (including
the same-seed determinism repeat, bit for bit, across two processes),
``tools/merge_results.py`` must promote every group to ``verified``, and the
table and figure tools must accept the merged artifacts.

This takes several minutes on a single CPU, so it only runs when
``DILTCL_E2E=1`` is set; the default test session skips it.
"""

import hashlib
import json
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except ImportError:  # pragma: no cover
    if __name__ == "__main__":
        print("PyTorch/torchvision not installed; suite end-to-end test skipped.")
        sys.exit(0)
    import pytest
    pytest.skip("PyTorch/torchvision not installed; suite end-to-end test skipped.", allow_module_level=True)

import pytest  # noqa: E402

NUM_CLASSES, TRAIN_PER_CLASS, TEST_PER_CLASS = 100, 6, 2


def _write_fake_cifar100(root: pathlib.Path):
    rng = np.random.RandomState(0)
    base = root / "cifar-100-python"
    base.mkdir(parents=True)
    md5s = {}
    for split, count in (("train", TRAIN_PER_CLASS), ("test", TEST_PER_CLASS)):
        labels = np.repeat(np.arange(NUM_CLASSES), count)
        data = rng.randint(0, 256, size=(labels.size, 3 * 32 * 32), dtype=np.uint8)
        path = base / split
        path.write_bytes(pickle.dumps({"data": data, "fine_labels": labels.tolist(),
                                       "coarse_labels": labels.tolist()}))
        md5s[split] = hashlib.md5(path.read_bytes()).hexdigest()
    meta = base / "meta"
    meta.write_bytes(pickle.dumps({"fine_label_names": [f"c{i}" for i in range(NUM_CLASSES)]}))
    md5s["meta"] = hashlib.md5(meta.read_bytes()).hexdigest()
    return md5s


def _write_sitecustomize(directory: pathlib.Path, md5s):
    (directory / "sitecustomize.py").write_text(f'''
# Test-only: point torchvision's CIFAR-100 integrity table at a synthetic set.
try:
    from torchvision import datasets as _d
    _d.CIFAR100.train_list = [["train", "{md5s['train']}"]]
    _d.CIFAR100.test_list = [["test", "{md5s['test']}"]]
    _d.CIFAR100.meta = {{"filename": "meta", "key": "fine_label_names",
                        "md5": "{md5s['meta']}"}}
except Exception:  # torchvision missing in helper processes such as pip
    pass
''')


@pytest.mark.skipif(os.environ.get("DILTCL_E2E") != "1",
                    reason="set DILTCL_E2E=1 to run the multi-minute suite test")
def test_smoke_profile_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        data_root, run_dir = base / "data", base / "runs" / "smoke"
        site, cache = base / "site", base / "cache"
        site.mkdir(), cache.mkdir()
        _write_sitecustomize(site, _write_fake_cifar100(data_root))
        env = dict(os.environ, PYTHONPATH=str(site), DILTCL_CACHE_DIR=str(cache),
                   DILTCL_DOWNLOAD="0")
        env.pop("DILTCL_DATA_ROOT", None)

        suite = subprocess.run(
            [sys.executable, str(ROOT / "run_suite.py"), "--profile", "smoke",
             "--seeds", "0", "--data-root", str(data_root), "--run-dir",
             str(run_dir), "--max-parallel", "2", "--threads-per-job", "1",
             "--launch-stagger", "0"],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        summary = json.loads((run_dir / "suite_summary.json").read_text())
        assert summary["state"] == "completed", (suite.stdout[-3000:], suite.stderr[-3000:])
        assert suite.returncode == 0, suite.stdout[-3000:] + suite.stderr[-3000:]
        assert json.loads((run_dir / "failures.json").read_text())["failed_jobs"] == []

        report = json.loads((run_dir / "verification_report.json").read_text())
        assert report["status"] == "verified", report["errors"]
        assert report["jobs_validated"] == 4
        assert any(d["arrays_identical"] for d in report["determinism"])
        assert any("not content-hash" in w for w in report["warnings"])
        # Two class-IL jobs x four isolation methods, each with per-example
        # predictions that the verifier recomputed exactly.
        assert report["prediction_files_checked"] == 8

        recomputed = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "recompute_from_predictions.py"),
             "--run-dir", str(run_dir), "--refit", "--out", str(base / "predictions.json")],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        assert recomputed.returncode == 0, recomputed.stdout + recomputed.stderr
        predictions = json.loads((base / "predictions.json").read_text())
        assert predictions["files_checked"] == 8 and predictions["status"] == "passed"

        # The persisted experts alone regenerate every routed and oracle result.
        evaluated = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "evaluate_checkpoints.py"),
             "--run-dir", str(run_dir), "--data-root", str(data_root), "--device", "cpu",
             "--out", str(base / "evaluated.json")],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        assert evaluated.returncode == 0, evaluated.stdout[-3000:] + evaluated.stderr[-3000:]
        evaluation = json.loads((base / "evaluated.json").read_text())
        assert evaluation["checkpoints_evaluated"] == 8
        assert evaluation["status"] == "identical", evaluation

        merged = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "merge_results.py"), str(run_dir)],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        assert merged.returncode == 0, merged.stdout + merged.stderr
        groups = json.loads((run_dir / "merged" / "merge_summary.json").read_text())["groups"]
        assert set(groups) == {"smoke_task_il", "smoke_class_entropy",
                               "smoke_class_entropy_cal"}
        for group in groups.values():
            assert group["verification_status"] == "verified", group

        audit = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "compare_runs.py"),
             "--reference", str(run_dir), "--candidate", str(run_dir / "merged"),
             "--out", str(base / "audit.json")],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        assert audit.returncode == 0, audit.stdout + audit.stderr
        assert json.loads((base / "audit.json").read_text())["status"] == "identical"

        table = subprocess.run(
            [sys.executable, str(ROOT / "make_class_il_table.py"), "--results",
             str(run_dir / "merged" / "smoke_class_entropy.json"),
             "--out", str(base / "table.tex")],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        assert table.returncode == 0, table.stdout + table.stderr
        figures = subprocess.run(
            [sys.executable, str(ROOT / "make_v3_figures.py"),
             "--class-results", str(run_dir / "merged" / "smoke_class_entropy.json"),
             "--class-results-cal", str(run_dir / "merged" / "smoke_class_entropy_cal.json"),
             "--task-results", str(run_dir / "merged" / "smoke_task_il.json"),
             "--output-dir", str(base / "figures"), "--dpi", "60"],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        assert figures.returncode == 0, figures.stdout + figures.stderr
        provenance = json.loads((base / "figures" / "figure_provenance.json").read_text())
        assert provenance["publication_ready"] is True
        assert (base / "figures" / "calibration_effect.pdf").is_file()


if __name__ == "__main__":
    os.environ["DILTCL_E2E"] = "1"
    test_smoke_profile_end_to_end()
    print("PASS test_smoke_profile_end_to_end")
