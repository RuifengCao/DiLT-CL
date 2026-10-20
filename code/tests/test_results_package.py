"""``tools/build_results_package.py`` bundles run directories into one package."""

import json
import pathlib
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

import build_results_package as pkg  # noqa: E402

import numpy as np  # noqa: E402


def _fake_run(root: pathlib.Path, name: str, status: str = "verified") -> pathlib.Path:
    run = root / name
    (run / "merged").mkdir(parents=True)
    (run / "paper" / "figures").mkdir(parents=True)
    (run / "statuses").mkdir()
    (run / "artifacts" / "job" / "checkpoints").mkdir(parents=True)
    (run / "dataset_fingerprints").mkdir()
    (run / "plan.json").write_text(json.dumps({"execution_source_sha256": "abc", "jobs": []}))
    (run / "suite_summary.json").write_text("{}")
    (run / "failures.json").write_text(json.dumps({"failed_jobs": []}))
    (run / "source_manifest.json").write_text(json.dumps({"execution_sha256": "abc"}))
    (run / "environment.json").write_text(json.dumps({"python": "3.12"}))
    (run / "dataset_fingerprint.json").write_text(json.dumps(
        {"tree_sha256": "t", "mode": "full", "files": [{"path": "x", "bytes": 1}] * 1000,
         "entries": {"cifar-100-python": {"sha256": "e"}}}))
    (run / "dataset_fingerprints" / "t.json").write_text((run / "dataset_fingerprint.json").read_text())
    (run / "verification_report.json").write_text(json.dumps(
        {"status": status, "jobs_validated": 2, "errors": [], "warnings": ["w"],
         "acknowledged_source_changes": [{"from": "a" * 12, "to": "b" * 12, "reason": "fix",
                                          "invalidated_methods": ["supsup"]}]}))
    (run / "merged" / "merge_summary.json").write_text(json.dumps(
        {"groups": {"g": {"verification_status": status, "inputs": 2, "missing_jobs": 0,
                          "seeds": [0], "benchmarks": ["cifar"], "methods": ["lwf"]}}}))
    (run / "merged" / "g.json").write_text(json.dumps({"code_version": "3.0.3", "aggregated": {}}))
    np.savez(run / "merged" / "g.npz", **{"cifar/lwf/seed0": np.zeros((2, 2))})
    (run / "paper" / "table.tex").write_text("\\begin{table}\\end{table}")
    (run / "paper" / "figures" / "fig.png").write_bytes(b"png")
    (run / "figures").mkdir()
    (run / "figures" / "calibration_effect.pdf").write_bytes(b"pdf")
    (run / "statuses" / "job.json").write_text("{}")
    (run / "artifacts" / "job" / "result.json").write_text("{}")
    (run / "artifacts" / "job" / "checkpoints" / "seed0.pt").write_bytes(b"x" * 100)
    return run


class PackageTests(unittest.TestCase):
    def test_package_collects_evidence_and_excludes_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            a, b = _fake_run(root, "p0_v3"), _fake_run(root, "p1_bias", status="unverified")
            out = root / "results"
            self.assertEqual(pkg.main(["--run-dirs", str(a), str(b), "--out", str(out),
                                       "--include-artifacts"]), 0)
            manifest = json.loads((out / "MANIFEST.json").read_text())
            paths = {f["path"] for f in manifest["files"]}
            self.assertIn("p0_v3/merged/g.json", paths)
            self.assertIn("p0_v3/merged/g.npz", paths)
            self.assertIn("p0_v3/paper/figures/fig.png", paths)
            self.assertIn("p0_v3/figures/calibration_effect.pdf", paths)
            self.assertIn("p0_v3/verification_report.json", paths)
            self.assertIn("p0_v3/artifacts/job/result.json", paths)
            self.assertFalse(any("checkpoints" in p for p in paths))
            self.assertIn("README.md", paths)
            fp = json.loads((out / "p0_v3" / "dataset_fingerprint.json").read_text())
            self.assertNotIn("files", fp)                      # per-file list stripped
            self.assertEqual(fp["entries"]["cifar-100-python"]["sha256"], "e")
            readme = (out / "README.md").read_text()
            self.assertIn("**verified**", readme)
            self.assertIn("**unverified**", readme)
            self.assertIn("invalidated ['supsup']", readme)
            self.assertTrue(out.with_suffix(".tar.gz").is_file())
            with tarfile.open(out.with_suffix(".tar.gz")) as tar:
                self.assertTrue(any(m.name.endswith("MANIFEST.json") for m in tar.getmembers()))
            # Every listed hash matches the file on disk.
            for entry in manifest["files"]:
                self.assertEqual(pkg._sha256(out / entry["path"]), entry["sha256"])

    def test_supplementary_mode_copies_code_and_checks_anonymity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run = _fake_run(root, "p0_v3")
            code = root / "code_src"
            (code / "diltcl").mkdir(parents=True)
            (code / "diltcl" / "__init__.py").write_text('__version__ = "3.0.3"\n')
            (code / "diltcl" / "__pycache__").mkdir()
            (code / "diltcl" / "__pycache__" / "x.pyc").write_bytes(b"0")
            (code / ".ipynb_checkpoints").mkdir()
            (code / ".ipynb_checkpoints" / "a-checkpoint.py").write_text("x")
            (code / "README.md").write_text("Contact: Ada Lovelace <ada@example.edu>\n")
            out = root / "supp"
            rc = pkg.main(["--run-dirs", str(run), "--out", str(out), "--supplementary",
                           str(code), "--anonymity-check", "Lovelace,example.edu", "--no-tar"])
            self.assertEqual(rc, 2)                      # identifying strings found
            self.assertFalse(out.exists())
            (code / "README.md").write_text(
                "Anonymous submission. See `../02_docs/01_\u64cd\u4f5c\u624b\u518c_v3.md` and "
                "`../05_archive/x.md` for details; results in "
                "`../evidence/results_20260906/`.\n")
            rc = pkg.main(["--run-dirs", str(run), "--out", str(out), "--supplementary",
                           str(code), "--anonymity-check", "Lovelace,example.edu", "--no-tar"])
            self.assertEqual(rc, 0)
            self.assertTrue((out / "code" / "diltcl" / "__init__.py").is_file())
            self.assertFalse((out / "code" / "diltcl" / "__pycache__").exists())
            self.assertFalse((out / "code" / ".ipynb_checkpoints").exists())
            self.assertTrue((out / "HOW_TO_VERIFY.md").is_file())
            readme = (out / "code" / "README.md").read_text()
            self.assertNotIn("02_docs", readme)
            self.assertNotIn("05_archive", readme)
            self.assertNotIn("../evidence", readme)
            self.assertIn("not included in this package", readme)
            self.assertNotIn("source run directory", (out / "README.md").read_text())

    def test_package_paths_never_leak_into_a_supplementary_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "home_janedoe" / "runs"
            root.mkdir(parents=True)
            run = _fake_run(root, "p0")
            code = pathlib.Path(tmp) / "code_src"
            (code / "diltcl").mkdir(parents=True)
            (code / "diltcl" / "__init__.py").write_text('__version__ = "3.0.7"\n')
            out = pathlib.Path(tmp) / "supp"
            rc = pkg.main(["--run-dirs", str(run), "--out", str(out), "--supplementary",
                           str(code), "--anonymity-check", "janedoe", "--no-tar"])
            self.assertEqual(rc, 0)
            self.assertNotIn("janedoe", (out / "MANIFEST.json").read_text())
            # The scan runs after the manifest is written: an internal package,
            # which records its source directories, is caught by it.
            internal = pathlib.Path(tmp) / "internal"
            rc = pkg.main(["--run-dirs", str(run), "--out", str(internal),
                           "--anonymity-check", "janedoe", "--no-tar"])
            self.assertEqual(rc, 2)
            self.assertFalse(internal.exists())


if __name__ == "__main__":
    unittest.main()
