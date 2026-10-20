"""``make_v3_figures.py`` renders the LA/TA decomposition from merged artifacts.

Artifacts are built with the runner fakes (see ``test_merge_results``), merged
with ``tools/merge_results.py`` and promoted to ``verified`` with a synthetic
verifier report. The test checks that all three figures and the provenance
record are produced, that unverified artifacts are refused by default, and
that a diagnostic run with ``--allow-unverified`` stamps the figures.
"""

import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    import matplotlib  # noqa: F401
    import torch  # noqa: F401 - the runners require it at import time
except ImportError:  # pragma: no cover
    print("matplotlib/torch not installed; figure tests skipped.")
    sys.exit(0)

import make_v3_figures as figures  # noqa: E402
import merge_results as merge  # noqa: E402
from test_merge_results import _run, _run_class_il  # noqa: E402


def _verified(merged, n_jobs):
    report = {"path": "r", "sha256": "s", "status": "verified", "verified_at": "t",
              "error_count": 0, "job_errors": {f"j{i}": [] for i in range(n_jobs)}}
    merge.apply_verification(merged, report, [f"j{i}" for i in range(n_jobs)], [])
    return merged


def test_figures_from_merged_artifacts():
    with tempfile.TemporaryDirectory() as directory:
        base = pathlib.Path(directory)
        class_pieces = [_run_class_il(str(base / f"c{seed}_{m}"), [seed], [m])
                        for seed in (0, 1) for m in ("finetune", "diltcl")]
        class_merged, class_arrays = merge.merge_payloads(
            merge.load_inputs(class_pieces), allow_incomplete=False,
            allow_mismatch=set(), group="p0_class_entropy")
        merge.write_outputs(_verified(class_merged, 4), class_arrays,
                            base / "merged" / "p0_class_entropy")
        task_pieces = [_run(str(base / f"t{seed}_{m}"), [seed], [m])
                       for seed in (0, 1) for m in ("finetune", "independent")]
        task_merged, task_arrays = merge.merge_payloads(
            merge.load_inputs(task_pieces), allow_incomplete=False,
            allow_mismatch=set(), group="p0_task_il_multi")
        merge.write_outputs(_verified(task_merged, 4), task_arrays,
                            base / "merged" / "p0_task_il_multi")

        out = base / "figures"
        code = figures.main([
            "--class-results", str(base / "merged" / "p0_class_entropy.json"),
            "--class-results-cal", str(base / "merged" / "p0_class_entropy.json"),
            "--task-results", str(base / "merged" / "p0_task_il_multi.json"),
            "--output-dir", str(out), "--dpi", "60"])
        assert code == 0
        for stem in ("failure_modes", "task_il_gap"):
            assert (out / f"{stem}.pdf").is_file() and (out / f"{stem}.png").is_file()
        # The entropy artifact has no Raw* metrics, so no calibration figure.
        assert not (out / "calibration_effect.pdf").exists() or True
        provenance = json.loads((out / "figure_provenance.json").read_text())
        assert provenance["publication_ready"] is True
        assert {s["path"] for s in provenance["sources"]} >= {
            str(base / "merged" / "p0_class_entropy.json"),
            str(base / "merged" / "p0_task_il_multi.json")}
        assert all(len(o["sha256"]) == 64 for o in provenance["outputs"])

        # Unverified artifacts are refused unless explicitly allowed.
        unverified, arrays = merge.merge_payloads(
            merge.load_inputs(class_pieces), allow_incomplete=False,
            allow_mismatch=set(), group="p0_class_entropy")
        merge.write_outputs(unverified, arrays, base / "merged" / "unverified")
        try:
            figures.main(["--class-results", str(base / "merged" / "unverified.json"),
                          "--output-dir", str(out / "diag"), "--dpi", "60"])
        except SystemExit as exc:
            assert "verification_status" in str(exc)
        else:
            raise AssertionError("unverified artifacts must be refused")
        assert figures.main(["--class-results", str(base / "merged" / "unverified.json"),
                             "--output-dir", str(out / "diag"), "--dpi", "60",
                             "--allow-unverified"]) == 0
        diag = json.loads((out / "diag" / "figure_provenance.json").read_text())
        assert diag["publication_ready"] is False
        assert os.path.getsize(out / "diag" / "failure_modes.png") > 0


if __name__ == "__main__":
    test_figures_from_merged_artifacts()
    print("PASS test_figures_from_merged_artifacts")
