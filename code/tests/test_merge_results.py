"""``tools/merge_results.py`` must rebuild the single-artifact layout exactly.

The strongest check is *granularity invariance*: running the task-IL runner
once per (seed, method) and merging the pieces must give the same per-seed
metrics and the same raw matrices as one run over all seeds and methods.
That is what licenses ``run_suite.py`` to split jobs per method for
throughput without changing a single number.

Also covered: recomputed aggregates, ordering by seed, refusal to merge
different experiments, refusal to hide missing seeds, and the run-directory
mode that groups jobs from a plan.
"""

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    import numpy as np
    import torch  # noqa: F401 - the runners require it at import time
except ImportError:  # pragma: no cover - torch-free environment
    print("PyTorch not installed; merge tests skipped.")
    sys.exit(0)

import merge_results as merge  # noqa: E402
import run_task_il as taskil  # noqa: E402
import run_class_il as class_il  # noqa: E402
from test_runner_integration import (  # noqa: E402
    CLASSES_PER_TASK,
    NUM_TASKS,
    _NoPreflight,
    _Patched,
    _TinyNet,
    _fast_config,
    _tiny_loaders,
)


def _fake_setup(benchmark, cfg, seed, head_mode="shared", **_):
    train, test = _tiny_loaders(CLASSES_PER_TASK, seed)
    base = lambda: _TinyNet(CLASSES_PER_TASK)          # noqa: E731
    builders = (base, taskil._wrap_head_mode(base, head_mode, NUM_TASKS,
                                           CLASSES_PER_TASK))
    return train, test, builders, NUM_TASKS


def _run(out_prefix, seeds, methods):
    with _Patched(taskil, {"_benchmark_setup": _fake_setup,
                         "preflight": _NoPreflight}):
        with contextlib.redirect_stdout(io.StringIO()):
            taskil.run(_fast_config(), ["cifar"], list(methods), list(seeds),
                     out_prefix=out_prefix, profile=False, head_mode="multi")
    return pathlib.Path(out_prefix + ".json")


def _split_and_whole(directory):
    pieces = []
    for seed in (0, 1):
        for method in ("finetune", "ewc"):
            pieces.append(_run(os.path.join(directory, f"s{seed}_{method}"),
                               [seed], [method]))
    whole = _run(os.path.join(directory, "whole"), [0, 1], ["finetune", "ewc"])
    return pieces, whole


def test_merging_per_method_jobs_reproduces_the_monolithic_run():
    with tempfile.TemporaryDirectory() as directory:
        pieces, whole_path = _split_and_whole(directory)
        merged, arrays = merge.merge_payloads(
            merge.load_inputs(pieces[::-1]),      # deliberately shuffled
            allow_incomplete=False, allow_mismatch=set(), group="test")
        whole = json.loads(whole_path.read_text())
        with np.load(whole_path.with_suffix(".npz")) as archive:
            whole_arrays = {k: archive[k] for k in archive.files}

        assert merged["seeds"] == [0, 1]
        assert merged["methods"] == ["finetune", "ewc"]
        assert merged["per_seed"] == whole["per_seed"]
        assert merged["aggregated"] == whole["aggregated"]
        assert sorted(arrays) == sorted(whole_arrays)
        for key in arrays:
            assert np.array_equal(arrays[key], whole_arrays[key]), key
        assert merged["per_seed_run_records"].keys() == \
            whole["per_seed_run_records"].keys()
        assert "memory" not in merged            # profile=False runs carry none
        for method in ("finetune", "ewc"):
            assert sorted(merged["per_seed_run_records"]["cifar"][method]) == ["0", "1"]
        assert merged["config"] == whole["config"]
        assert merged["execution"]["options"] == whole["execution"]["options"]
        assert merged["merge_provenance"]["complete"] is True
        assert merged["merge_provenance"]["input_count"] == 4
        # Different seeds/commands per job must not count as different
        # environments; only hardware/software differences do.
        assert merged["merge_provenance"]["environment_consistent"] is True
        # Below five seeds no significance test may be reported.
        tests = merged["paired_significance"]["tests"]
        assert tests == {}   # no diltcl reference in this set


def test_memory_table_is_the_union_of_per_method_rows():
    """Per-method jobs each carry only their own analytic memory row; the
    merged table must hold every method (found on the real P0 merge, where
    only the DiLT-CL row survived)."""
    with tempfile.TemporaryDirectory() as directory:
        pieces = []
        for seed in (0, 1):
            for method in ("finetune", "ewc"):
                prefix = os.path.join(directory, f"m{seed}_{method}")
                with _Patched(taskil, {"_benchmark_setup": _fake_setup,
                                       "preflight": _NoPreflight}):
                    with contextlib.redirect_stdout(io.StringIO()):
                        taskil.run(_fast_config(), ["cifar"], [method], [seed],
                                   out_prefix=prefix, profile=True, head_mode="multi")
                pieces.append(pathlib.Path(prefix + ".json"))
        merged, _ = merge.merge_payloads(
            merge.load_inputs(pieces), allow_incomplete=False,
            allow_mismatch=set(), group="g")
        names = [r["name"] for r in merged["memory"]["cifar"]]
        assert names == ["Finetune", "EWC"]


def test_refuses_to_merge_different_experiments_or_missing_seeds():
    with tempfile.TemporaryDirectory() as directory:
        pieces, _ = _split_and_whole(directory)
        inputs = merge.load_inputs(pieces)
        # A different config is a different experiment.
        inputs[0]["payload"]["config"] = dict(inputs[0]["payload"]["config"],
                                             lr=0.5)
        try:
            merge.merge_payloads(inputs, allow_incomplete=False,
                                 allow_mismatch=set(), group=None)
        except merge.MergeError as exc:
            assert "config" in str(exc)
        else:
            raise AssertionError("merging different configs must fail")
        # Missing a (method, seed) pair must not be silently tolerated.
        inputs = merge.load_inputs(pieces[:3])
        try:
            merge.merge_payloads(inputs, allow_incomplete=False,
                                 allow_mismatch=set(), group=None)
        except merge.MergeError as exc:
            assert "incomplete" in str(exc)
        else:
            raise AssertionError("incomplete sets must be refused by default")
        merged, _ = merge.merge_payloads(inputs, allow_incomplete=True,
                                        allow_mismatch=set(), group=None)
        gaps = merged["merge_provenance"]["incomplete_pairs"]
        assert len(gaps) == 1 and gaps[0]["seeds_missing"] == [1]
        # A JSON whose metrics do not match its matrices is refused.
        inputs = merge.load_inputs(pieces)
        inputs[1]["payload"]["per_seed"]["cifar"]["ewc"][0]["ACC"] += 5.0
        try:
            merge.merge_payloads(inputs, allow_incomplete=False,
                                 allow_mismatch=set(), group=None)
        except merge.MergeError as exc:
            assert "recomputed" in str(exc)
        else:
            raise AssertionError("metric/matrix disagreement must be refused")
        # Duplicate results are always an error.
        try:
            merge.merge_payloads(merge.load_inputs(pieces + pieces[:1]),
                                 allow_incomplete=True, allow_mismatch=set(),
                                 group=None)
        except merge.MergeError as exc:
            assert "duplicate" in str(exc)
        else:
            raise AssertionError("duplicate inputs must be refused")


def test_config_fields_added_later_and_acknowledged_versions_merge():
    """An artifact written before a config field existed (and under an
    acknowledged earlier code version) must merge with a newer one."""
    with tempfile.TemporaryDirectory() as directory:
        pieces, _ = _split_and_whole(directory)
        inputs = merge.load_inputs(pieces)
        old = inputs[0]["payload"]
        old["config"] = {k: v for k, v in old["config"].items()
                         if k != "ewc_fisher_labels"}        # pre-field artifact
        old["code_version"] = "3.0.0"
        for item in inputs[1:]:
            item["payload"]["code_version"] = "3.0.1"
        try:
            merge.merge_payloads(inputs, allow_incomplete=False,
                                 allow_mismatch=set(), group=None)
        except merge.MergeError as exc:
            assert "code_version" in str(exc) and "config" not in str(exc)
        else:
            raise AssertionError("unacknowledged version mix must be refused")
        merged, _ = merge.merge_payloads(inputs, allow_incomplete=False,
                                        allow_mismatch={"code_version"}, group=None)
        assert merged["code_version"] == "3.0.0/3.0.1"
        assert merged["config"]["ewc_fisher_labels"] == "prediction"
        assert merged["merge_provenance"]["code_versions"] == ["3.0.0", "3.0.1"]
        # A genuinely different config value is still refused.
        inputs[1]["payload"]["config"]["lr"] = 0.5
        try:
            merge.merge_payloads(inputs, allow_incomplete=False,
                                 allow_mismatch={"code_version"}, group=None)
        except merge.MergeError as exc:
            assert "config" in str(exc)
        else:
            raise AssertionError("different config values must be refused")


def test_run_dir_mode_groups_jobs_from_the_plan():
    with tempfile.TemporaryDirectory() as directory:
        run_dir = pathlib.Path(directory) / "run"
        (run_dir / "statuses").mkdir(parents=True)
        jobs = []
        for seed in (0, 1):
            for method in ("finetune", "ewc"):
                job_id = f"p0_task_il_multi__cifar__seed{seed}__{method}"
                prefix = run_dir / "artifacts" / job_id / "result"
                prefix.parent.mkdir(parents=True)
                _run(str(prefix), [seed], [method])
                jobs.append({"job": {"job_id": job_id, "seed": seed,
                                     "benchmark": "cifar", "repeat_of": None},
                             "output_prefix_relative": str(
                                 prefix.relative_to(run_dir))})
                (run_dir / "statuses" / f"{job_id}.json").write_text(
                    json.dumps({"state": "succeeded",
                                "ended_at": "2026-01-01T00:00:00+00:00"}))
        # A failed job in another group must not block this group's merge.
        jobs.append({"job": {"job_id": "p0_class_entropy__cifar__seed0__lwf",
                             "seed": 0, "benchmark": "cifar", "repeat_of": None},
                     "output_prefix_relative": "artifacts/x/result"})
        (run_dir / "plan.json").write_text(json.dumps({"jobs": jobs}))
        code = merge.main([str(run_dir), "--group", "p0_task_il_multi"])
        assert code == 0
        merged = json.loads((run_dir / "merged" / "p0_task_il_multi.json").read_text())
        assert merged["seeds"] == [0, 1]
        assert merged["merge_provenance"]["group"] == "p0_task_il_multi"
        # No verifier report yet -> stays unverified, with the reason recorded.
        assert merged["verification_status"] == "unverified"
        reasons = merged["merge_provenance"]["verification"]["reasons_not_verified"]
        assert any("verification_report.json" in r for r in reasons)

        # A passing verifier report covering every job promotes the artifact.
        group_jobs = [j["job"]["job_id"] for j in jobs[:-1]]

        def report(status="verified", verified_at="2026-01-02T00:00:00+00:00",
                   errors_for=None):
            results = []
            for jid in group_jobs:
                prefix = run_dir / "artifacts" / jid / "result"
                results.append({
                    "job_id": jid, "errors": ["ACC mismatch"] if jid == errors_for else [],
                    "json_sha256": merge.sha256_file(prefix.with_suffix(".json")),
                    "npz_sha256": merge.sha256_file(prefix.with_suffix(".npz"))})
            (run_dir / "verification_report.json").write_text(json.dumps({
                "status": status, "verified_at": verified_at,
                "errors": [] if status == "verified" else ["x"], "results": results}))

        report()
        assert merge.main([str(run_dir), "--group", "p0_task_il_multi"]) == 0
        merged = json.loads((run_dir / "merged" / "p0_task_il_multi.json").read_text())
        assert merged["verification_status"] == "verified"
        assert merged["evidence_status"] == "verified"
        assert merged["artifact_reportability"]["postrun_validation_completed"] is True
        assert merged["merge_provenance"]["verification"]["verified"] is True

        # --inputs across the run binds every input to the same report.
        inputs = [str(run_dir / "artifacts" / jid / "result.json") for jid in group_jobs]
        assert merge.main(["--inputs", *inputs, "--out", str(run_dir / "again")]) == 0
        again = json.loads((run_dir / "again.json").read_text())
        assert again["verification_status"] == "verified"

        # A result edited after verification no longer matches the report.
        victim = run_dir / "artifacts" / group_jobs[0] / "result.json"
        victim.write_text(victim.read_text().replace('"ACC"', '"ACC" ', 1))
        assert merge.main([str(run_dir), "--group", "p0_task_il_multi"]) == 0
        merged = json.loads((run_dir / "merged" / "p0_task_il_multi.json").read_text())
        assert merged["verification_status"] == "unverified"
        assert merged["evidence_status"] == "unverified"
        assert any("differ from the verified" in r for r in
                   merged["merge_provenance"]["verification"]["reasons_not_verified"])
        assert merge.main(["--inputs", *inputs, "--out", str(run_dir / "again")]) == 0
        assert json.loads((run_dir / "again.json").read_text())["verification_status"] == "unverified"

        # A report written before a job finished does not cover it.
        report(verified_at="2025-12-31T00:00:00+00:00")
        assert merge.main([str(run_dir), "--group", "p0_task_il_multi"]) == 0
        merged = json.loads((run_dir / "merged" / "p0_task_il_multi.json").read_text())
        assert merged["verification_status"] == "unverified"

        # One job with a verifier error, or a failed run, keeps it unverified.
        report(errors_for=group_jobs[0])
        assert merge.main([str(run_dir), "--group", "p0_task_il_multi"]) == 0
        merged = json.loads((run_dir / "merged" / "p0_task_il_multi.json").read_text())
        assert merged["verification_status"] == "unverified"
        report(status="failed")
        assert merge.main([str(run_dir), "--group", "p0_task_il_multi"]) == 0
        merged = json.loads((run_dir / "merged" / "p0_task_il_multi.json").read_text())
        assert merged["verification_status"] == "unverified"
        (run_dir / "verification_report.json").unlink()
        summary = json.loads((run_dir / "merged" / "merge_summary.json").read_text())
        assert list(summary["groups"]) == ["p0_task_il_multi"]
        # Without --group the failed job blocks its own group.
        assert merge.main([str(run_dir)]) == 2
        assert merge.main([str(run_dir), "--allow-incomplete"]) == 0


def _run_class_il(out_prefix, seeds, methods, benchmarks=("cifar",), **run_kwargs):
    def fake_shared(benchmark, cfg, seed, **_):
        train, test = _tiny_loaders(NUM_TASKS * CLASSES_PER_TASK, seed)
        return class_il.SharedSetup(
            train, test, (lambda: _TinyNet(NUM_TASKS * CLASSES_PER_TASK)),
            NUM_TASKS)

    def fake_isolation(benchmark, cfg, seed, **_):
        train, test = _tiny_loaders(CLASSES_PER_TASK, seed)
        return class_il.IsolationSetup(
            train, test, (lambda: _TinyNet(CLASSES_PER_TASK)),
            NUM_TASKS, CLASSES_PER_TASK, True)

    with _Patched(class_il, {"_shared_setup": fake_shared,
                             "_isolation_setup": fake_isolation,
                             "preflight": _NoPreflight}):
        with contextlib.redirect_stdout(io.StringIO()):
            class_il.run(_fast_config(), list(benchmarks), list(methods), list(seeds),
                         out_prefix=out_prefix, **run_kwargs)
    return pathlib.Path(out_prefix + ".json")


def test_class_il_jobs_of_different_benchmarks_merge_into_one_group():
    """Real P0 artifacts carry ``head_conventions.per_benchmark_output_counts``
    keyed by their own benchmark; a group spanning benchmarks must union
    them instead of refusing (found on the first P0 run directory)."""
    with tempfile.TemporaryDirectory() as directory:
        pieces = [_run_class_il(os.path.join(directory, f"{b}_{m}"), [0], [m],
                                benchmarks=(b,))
                  for b in ("cifar", "pmnist") for m in ("finetune", "diltcl")]
        merged, _ = merge.merge_payloads(
            merge.load_inputs(pieces), allow_incomplete=False,
            allow_mismatch=set(), group="p0_class_entropy")
        assert merged["benchmarks"] == ["cifar", "pmnist"]
        counts = merged["head_conventions"]["per_benchmark_output_counts"]
        assert set(counts) == {"cifar", "pmnist"}
        assert merged["head_conventions"]["task_identity_at_test"] is False


def test_calibrated_group_splits_per_method_without_changing_numbers():
    """The P0 calibration group runs shared-backbone methods on the same 90%
    holdout as the isolation methods. A shared-only job must be accepted
    under calibration (it records no router) and merging shared-only +
    isolation-only jobs must equal one job that ran both."""
    cal = dict(calibration="score-logreg", calibration_fraction=0.2)
    with tempfile.TemporaryDirectory() as directory:
        pieces = [_run_class_il(os.path.join(directory, f"s0_{m}"), [0], [m], **cal)
                  for m in ("finetune", "diltcl")]
        whole = json.loads(_run_class_il(os.path.join(directory, "whole"), [0],
                                         ["finetune", "diltcl"], **cal).read_text())
        merged, _ = merge.merge_payloads(
            merge.load_inputs(pieces), allow_incomplete=False,
            allow_mismatch=set(), group="p0_class_entropy_cal")
        assert merged["per_seed"] == whole["per_seed"]
        assert merged["calibration"]["cifar"].keys() == whole["calibration"]["cifar"].keys()
        counts = merged["benchmark_metadata"]["cifar"]["per_seed_counts"]["0"]
        assert counts["shared_backbone"]["calibration_samples_per_task"] == \
            whole["benchmark_metadata"]["cifar"]["per_seed_counts"]["0"][
                "shared_backbone"]["calibration_samples_per_task"]
        shared_only = json.loads(pieces[0].read_text())
        assert shared_only["inference_calibration"] == "score-logreg"
        assert shared_only["calibration"]["cifar"] == {}


def test_class_il_merge_reproduces_the_monolithic_run_and_renders_a_table():
    """Class-IL artifacts partition their own method list per job; the merge
    must rebuild that partition, match the monolithic run exactly and, once
    promoted to ``verified``, be accepted by ``make_class_il_table.py``."""
    import subprocess

    with tempfile.TemporaryDirectory() as directory:
        pieces = [_run_class_il(os.path.join(directory, f"s{seed}_{method}"),
                                [seed], [method])
                  for seed in (0, 1) for method in ("finetune", "diltcl")]
        whole = json.loads(_run_class_il(
            os.path.join(directory, "whole"), [0, 1],
            ["finetune", "diltcl"]).read_text())
        merged, arrays = merge.merge_payloads(
            merge.load_inputs(pieces), allow_incomplete=False,
            allow_mismatch=set(), group="p0_class_entropy")
        assert merged["per_seed"] == whole["per_seed"]
        assert merged["aggregated"] == whole["aggregated"]
        assert merged["shared_backbone_methods"] == ["finetune"]
        assert merged["isolation_methods"] == ["diltcl"]
        assert sorted(merged["calibration"]["cifar"]) == \
            sorted(whole["calibration"]["cifar"])
        assert merged["paired_significance"]["tests"]["cifar"]["finetune"]["suppressed"]
        fake_report = {"path": "r", "sha256": "s", "status": "verified",
                       "verified_at": "t", "error_count": 0,
                       "job_errors": {f"j{i}": [] for i in range(4)}}
        merge.apply_verification(merged, fake_report,
                                 [f"j{i}" for i in range(4)], [])
        assert merged["verification_status"] == "verified"
        out = pathlib.Path(directory) / "merged" / "p0_class_entropy"
        merge.write_outputs(merged, arrays, out)
        table = pathlib.Path(directory) / "table.tex"
        result = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve().parent.parent
                                 / "make_class_il_table.py"),
             "--results", str(out) + ".json", "--out", str(table)],
            capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Split-CIFAR-100" in table.read_text()


if __name__ == "__main__":
    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_") and callable(value)]
    for function in functions:
        function()
        print(f"PASS {function.__name__}")
    print(f"\nAll {len(functions)} merge tests passed.")
