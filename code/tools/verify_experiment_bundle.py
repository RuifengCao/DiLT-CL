#!/usr/bin/env python3
"""Independently validate DiLT-CL suite JSON/NPZ artifacts without PyTorch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent


def _load_suite():
    """Load run_suite.py by path for its identity helpers (torch-free)."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "diltcl_suite_for_verifier", _ROOT / "run_suite.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_suite = None


def suite_module():
    global _suite
    if _suite is None:
        _suite = _load_suite()
    return _suite


_predictions = None


def predictions_module():
    """``tools/recompute_from_predictions.py``, loaded by path (torch-free)."""
    global _predictions
    if _predictions is None:
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "diltcl_predictions_for_verifier",
            _ROOT / "tools" / "recompute_from_predictions.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _predictions = module
    return _predictions


def validate_predictions(record: dict[str, Any], payload: dict[str, Any], archive,
                         json_path: Path, benchmark: str, method: str, seed: int,
                         label: str, errors: list[str]) -> bool:
    """Recompute every class-IL statistic from a job's per-example predictions.

    Returns True when a predictions file was present and checked.
    """
    pred = record.get("predictions")
    if not pred:
        return False
    path = json_path.parent / pred.get("relative_path", "")
    if not path.is_file():
        errors.append(f"{label}: predictions file missing: {path}")
        return False
    if sha256_file(path) != pred.get("sha256"):
        errors.append(f"{label}: predictions file SHA-256 differs from record")
    module = predictions_module()
    try:
        arrays, meta = module.load_predictions(path)
    except (OSError, ValueError) as exc:
        errors.append(f"{label}: cannot read predictions: {exc}")
        return False
    if meta.get("trained_state_content_sha256") != record.get(
            "trained_state", {}).get("content_sha256"):
        errors.append(f"{label}: predictions belong to a different trained state")
    for message in module.compare_with_artifact(
            arrays, meta, payload, archive, benchmark, method, seed):
        errors.append(f"{label}: predictions: {message}")
    return True


TOL = 1e-8
ISOLATION = {"packnet", "supsup", "wsn", "diltcl"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def summarize(matrix: np.ndarray) -> dict[str, float]:
    tasks = matrix.shape[0]
    acc = float(np.mean(matrix[-1]))
    la = float(np.mean(np.diag(matrix)))
    if tasks <= 1:
        return {"ACC": acc, "BWT": 0.0, "Forgetting": 0.0, "LA": la}
    bwt = float(np.mean([
        matrix[-1, j] - matrix[j, j] for j in range(tasks - 1)
    ]))
    forgetting = float(np.mean([
        np.max(matrix[j:, j]) - matrix[-1, j]
        for j in range(tasks - 1)
    ]))
    return {"ACC": acc, "BWT": bwt,
            "Forgetting": forgetting, "LA": la}


def check_matrix(name: str, matrix: np.ndarray, errors: list[str]) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        errors.append(f"{name}: expected square 2-D matrix, got {matrix.shape}")
        return
    if matrix.shape[0] == 0:
        errors.append(f"{name}: matrix is empty")
        return
    if not np.isfinite(matrix).all():
        errors.append(f"{name}: matrix contains non-finite values")
    if np.any(matrix < -TOL) or np.any(matrix > 100.0 + TOL):
        errors.append(f"{name}: values fall outside percentage range [0,100]")
    if not np.allclose(np.triu(matrix, 1), 0.0, atol=TOL, rtol=0.0):
        errors.append(f"{name}: upper triangle must be zero/undefined")


def close(actual: Any, expected: float) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= TOL
    except (TypeError, ValueError):
        return False


def validate_checkpoint_record(record: dict[str, Any], label: str,
                               errors: list[str], required: bool,
                               base_dir: Path) -> None:
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append(f"{label}: invalid evaluated-state SHA-256")
        return
    content_digest = record.get("content_sha256")
    if not isinstance(content_digest, str) or len(content_digest) != 64:
        errors.append(f"{label}: invalid canonical state-content SHA-256")
    if not required:
        return
    if record.get("persisted") is not True:
        errors.append(f"{label}: checkpoint was not persisted")
        return
    relative_text = record.get("persisted_relative_path")
    path_text = record.get("persisted_path")
    if isinstance(relative_text, str):
        path = (base_dir / relative_text).resolve()
    elif isinstance(path_text, str):
        path = Path(path_text)
    else:
        errors.append(f"{label}: persisted checkpoint path is missing")
        return
    if not path.is_file():
        errors.append(f"{label}: persisted checkpoint file is missing")
        return
    if path.stat().st_size != record.get("serialized_bytes"):
        errors.append(f"{label}: persisted checkpoint byte count mismatch")
    if sha256_file(path) != digest:
        errors.append(f"{label}: persisted checkpoint SHA-256 mismatch")


def validate_dilt_density(metadata: dict[str, Any], label: str,
                          errors: list[str]) -> None:
    target = metadata.get("requested_target_sparsity",
                          metadata.get("diltcl_requested_target_sparsity"))
    if target is None:
        errors.append(f"{label}: requested target sparsity is missing")
        return
    expected_density = 1.0 - float(target)
    partition = metadata.get(
        "core_partition_global_density",
        metadata.get("diltcl_core_partition_global_density", []))
    if not partition:
        errors.append(f"{label}: core partition densities are missing")
    for density in partition:
        if abs(float(density) - expected_density) > 0.002:
            errors.append(
                f"{label}: partition density {density} misses target "
                f"{expected_density} beyond rounding tolerance")
    experts = metadata.get(
        "expert_eligible_density_by_task",
        metadata.get("diltcl_expert_eligible_density_by_task", {}))
    for task, density in experts.items():
        if abs(float(density) - expected_density) > 0.002:
            errors.append(
                f"{label}: expert {task} eligible density {density} misses "
                f"target {expected_density}")


def get_seed_metrics(payload: dict[str, Any], benchmark: str, method: str,
                     seed: int) -> dict[str, Any]:
    seeds = payload.get("seeds", [])
    if seed not in seeds:
        raise KeyError(f"seed {seed} missing from JSON seeds")
    position = seeds.index(seed)
    return payload["per_seed"][benchmark][method][position]


def validate_aggregates(payload: dict[str, Any], errors: list[str]) -> None:
    for benchmark in payload.get("benchmarks", []):
        for method in payload.get("methods", []):
            records = payload.get("per_seed", {}).get(benchmark, {}).get(method, [])
            aggregate = payload.get("aggregated", {}).get(benchmark, {}).get(method, {})
            if len(records) != len(payload.get("seeds", [])):
                errors.append(
                    f"{benchmark}/{method}: per_seed count does not match seeds")
                continue
            if not records:
                continue
            for metric in records[0]:
                values = np.asarray([record[metric] for record in records], dtype=float)
                cell = aggregate.get(metric)
                if cell is None:
                    errors.append(f"{benchmark}/{method}: aggregate missing {metric}")
                    continue
                expected_std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                if not close(cell.get("mean"), float(values.mean())):
                    errors.append(f"{benchmark}/{method}/{metric}: mean mismatch")
                if not close(cell.get("std"), expected_std):
                    errors.append(f"{benchmark}/{method}/{metric}: std mismatch")
                if list(cell.get("values", [])) != list(values):
                    errors.append(f"{benchmark}/{method}/{metric}: values mismatch")


def validate_result(json_path: Path, npz_path: Path, expected_seed: int,
                    expected_benchmark: str) -> dict[str, Any]:
    errors: list[str] = []
    predictions_checked: list[str] = []
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"json": str(json_path), "npz": str(npz_path),
                "errors": [f"cannot read JSON: {exc}"]}
    if payload.get("benchmarks") != [expected_benchmark]:
        errors.append("JSON benchmark does not match suite job")
    if payload.get("seeds") != [expected_seed]:
        errors.append("JSON seed list does not match isolated suite job")
    if payload.get("verification_status") != "unverified":
        errors.append("raw runner output must start as unverified")
    if "environment" not in payload:
        errors.append("execution environment metadata is missing")
    checkpoints_required = payload.get("checkpoint_persistence_enabled") is True
    if not checkpoints_required:
        errors.append("evaluated-state checkpoint persistence is disabled")
    try:
        archive = np.load(npz_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        errors.append(f"cannot read NPZ: {exc}")
        return {"json": str(json_path), "npz": str(npz_path),
                "errors": errors}

    with archive:
        keys = set(archive.files)
        protocol = payload.get("protocol")
        for method in payload.get("methods", []):
            main_key = f"{expected_benchmark}/{method}/seed{expected_seed}"
            if main_key not in keys:
                errors.append(f"missing NPZ matrix: {main_key}")
                continue
            matrix = np.asarray(archive[main_key], dtype=float)
            check_matrix(main_key, matrix, errors)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                continue
            recomputed = summarize(matrix)
            try:
                metrics = get_seed_metrics(
                    payload, expected_benchmark, method, expected_seed)
            except (KeyError, IndexError) as exc:
                errors.append(f"{main_key}: JSON metrics missing: {exc}")
                continue
            for metric, value in recomputed.items():
                if not close(metrics.get(metric), value):
                    errors.append(
                        f"{main_key}: {metric} mismatch, JSON={metrics.get(metric)} "
                        f"recomputed={value}")

            if protocol == "class_il":
                run_trained = (payload.get("per_seed_run_records", {})
                               .get(expected_benchmark, {}).get(method, {})
                               .get(str(expected_seed), {}).get("trained_state", {}))
                validate_checkpoint_record(
                    run_trained, main_key, errors, required=checkpoints_required,
                    base_dir=json_path.parent)
                if method == "diltcl":
                    validate_dilt_density(
                        run_trained.get("method_state_metadata", {}),
                        main_key, errors)

            if protocol == "class_il" and method in ISOLATION:
                required = {"task_inference", "conditional_class", "routing_tie"}
                if payload.get("inference_calibration") != "none":
                    required |= {
                        "raw_result", "raw_task_inference",
                        "raw_conditional_class", "raw_routing_tie",
                    }
                for suffix in required:
                    aux_key = f"{main_key}/{suffix}"
                    if aux_key not in keys:
                        errors.append(f"missing class-IL auxiliary matrix: {aux_key}")
                        continue
                    aux = np.asarray(archive[aux_key], dtype=float)
                    check_matrix(aux_key, aux, errors)
                raw_key = f"{main_key}/raw_result"
                if raw_key in keys:
                    raw_metrics = summarize(np.asarray(archive[raw_key], dtype=float))
                    for metric, value in raw_metrics.items():
                        if not close(metrics.get("Raw" + metric), value):
                            errors.append(f"{raw_key}: Raw{metric} mismatch")
                record = (payload.get("calibration", {})
                          .get(expected_benchmark, {}).get(method, {})
                          .get(str(expected_seed)))
                if not isinstance(record, dict):
                    errors.append(f"{main_key}: trained-state record missing")
                else:
                    if validate_predictions(record, payload, archive, json_path,
                                            expected_benchmark, method,
                                            expected_seed, main_key, errors):
                        predictions_checked.append(main_key)
                    calibration_trained = record.get("trained_state", {})
                    digest = calibration_trained.get("sha256")
                    if digest != run_trained.get("sha256"):
                        errors.append(
                            f"{main_key}: calibration/run-record checkpoint mismatch")
                    if payload.get("inference_calibration") != "none":
                        if record.get("matched_checkpoint_sha256") != digest:
                            errors.append(
                                f"{main_key}: raw/calibrated checkpoint hash mismatch")
                        if (record.get("matched_checkpoint_content_sha256")
                                != calibration_trained.get("content_sha256")):
                            errors.append(
                                f"{main_key}: raw/calibrated content hash mismatch")

            if protocol and protocol.startswith("task_il"):
                task_extras = (payload.get("per_seed_run_records", {})
                               .get(expected_benchmark, {}).get(method, {})
                               .get(str(expected_seed), {}).get("method_extras", {}))
                validate_checkpoint_record(
                    task_extras.get("checkpoint", {}), main_key, errors,
                    required=checkpoints_required, base_dir=json_path.parent)

            if protocol and protocol.startswith("task_il") and method == "diltcl":
                validate_dilt_density(task_extras, main_key, errors)
                core_key = f"{main_key}/core_only"
                if core_key not in keys:
                    errors.append(f"missing core-only matched ablation: {core_key}")
                else:
                    core = np.asarray(archive[core_key], dtype=float)
                    check_matrix(core_key, core, errors)
                    expected = task_extras.get("core_only_metrics", {})
                    for metric, value in summarize(core).items():
                        if not close(expected.get(metric), value):
                            errors.append(f"{core_key}: {metric} JSON mismatch")

        unexpected_main = [key for key in keys if key.count("/") == 2 and
                           key.split("/")[0] != expected_benchmark]
        if unexpected_main:
            errors.append(f"unexpected benchmark matrix keys: {unexpected_main}")

    validate_aggregates(payload, errors)
    return {
        "json": str(json_path.resolve()), "npz": str(npz_path.resolve()),
        "json_sha256": sha256_file(json_path),
        "npz_sha256": sha256_file(npz_path),
        "errors": errors, "passed": not errors,
        "predictions_checked": predictions_checked,
    }


def compare_repeat(run_dir: Path, repeat_job: dict[str, Any],
                   errors: list[str]) -> dict[str, Any]:
    job = repeat_job["job"]
    reference_id = job.get("repeat_of")
    repeat_id = job["job_id"]
    record = {"repeat": repeat_id, "reference": reference_id,
              "arrays_identical": False,
              "checkpoint_content_hash_identical": None}
    if not reference_id:
        return record
    left = run_dir / "artifacts" / reference_id / "result.npz"
    right = run_dir / "artifacts" / repeat_id / "result.npz"
    if not left.is_file() or not right.is_file():
        errors.append(f"determinism pair missing: {reference_id}, {repeat_id}")
        return record
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            errors.append(f"determinism NPZ key mismatch: {repeat_id}")
        else:
            record["arrays_identical"] = all(
                np.array_equal(a[key], b[key]) for key in a.files)
            if not record["arrays_identical"]:
                errors.append(f"same-seed matrices differ: {repeat_id}")

    def checkpoint_hash(job_id: str) -> str | None:
        path = run_dir / "artifacts" / job_id / "result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmark = payload["benchmarks"][0]
        seed = str(payload["seeds"][0])
        return (payload.get("per_seed_run_records", {}).get(benchmark, {})
                .get("diltcl", {}).get(seed, {}).get("method_extras", {})
                .get("checkpoint", {}).get("content_sha256"))

    left_hash, right_hash = checkpoint_hash(reference_id), checkpoint_hash(repeat_id)
    if left_hash is not None and right_hash is not None:
        record["checkpoint_content_hash_identical"] = left_hash == right_hash
        if left_hash != right_hash:
            errors.append(f"same-seed checkpoint hashes differ: {repeat_id}")
    return record


def verify(run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    plan_path = run_dir / "plan.json"
    if not plan_path.is_file():
        return {"status": "failed", "errors": ["plan.json is missing"]}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_execution_sha = plan.get("execution_source_sha256")
    suite = suite_module()
    acknowledged = plan.get("source_change_acknowledgements", [])
    accepted_sources = suite.accepted_source_hashes(plan)
    fingerprint_path = run_dir / "dataset_fingerprint.json"
    data_fingerprint = (json.loads(fingerprint_path.read_text(encoding="utf-8"))
                        if fingerprint_path.is_file() else None)
    dataset_mode = (data_fingerprint.get("mode") if data_fingerprint
                    else plan.get("dataset_hash_mode") or "none")
    data_root = plan.get("data_root")
    results = []
    status_records = {}
    acknowledged_jobs: list[str] = []
    for entry in plan.get("jobs", []):
        job = entry.get("job", {})
        job_id = job.get("job_id")
        if job_id is None:
            errors.append("plan job entry has no job.job_id")
            continue
        missing_fields = [f for f in ("seed", "benchmark") if f not in job]
        if missing_fields:
            errors.append(
                f"{job_id}: plan job entry is missing {missing_fields}")
            continue
        status_path = run_dir / "statuses" / f"{job_id}.json"
        if not status_path.is_file():
            errors.append(f"job has no status: {job_id}")
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status_records[job_id] = status
        if status.get("state") != "succeeded":
            errors.append(f"job did not succeed: {job_id}")
            continue
        source = status.get("execution_source_sha256")
        if source != plan_execution_sha:
            if source in accepted_sources:
                reason = suite.result_invalidated(
                    plan, source, str(job.get("methods", "")).split(","))
                if reason:
                    errors.append(f"{job_id}: {reason}")
                else:
                    acknowledged_jobs.append(job_id)
            else:
                errors.append(f"job used a different executable source: {job_id}")
        expected_identity = suite.job_dataset_identity(
            data_fingerprint, job["benchmark"],
            Path(data_root) if data_root else None, dataset_mode)
        if not suite.dataset_identity_matches(
                status.get("dataset_identity"), expected_identity, run_dir):
            errors.append(f"job used a different dataset fingerprint: {job_id}")
        for output in status.get("outputs", []):
            relative = output.get("path_relative_to_run_dir")
            path = ((run_dir / relative).resolve()
                    if isinstance(relative, str) else Path(output["path"]))
            if not path.is_file():
                errors.append(f"status/output hash mismatch: {job_id}/{path.name}")
            elif path.suffix != ".pt" and sha256_file(path) != output.get("sha256"):
                errors.append(f"status/output hash mismatch: {job_id}/{path.name}")
        relative_prefix = entry.get("output_prefix_relative")
        if isinstance(relative_prefix, str):
            prefix = (run_dir / relative_prefix).resolve()
        elif "output_prefix" in entry:
            prefix = Path(entry["output_prefix"])
        else:
            errors.append(f"{job_id}: plan job entry has no output prefix")
            continue
        prefix = Path(prefix)
        result = validate_result(
            prefix.with_suffix(".json"), prefix.with_suffix(".npz"),
            int(job["seed"]), job["benchmark"])
        results.append({"job_id": job_id, **result})
        errors.extend(f"{job_id}: {error}" for error in result["errors"])
        # Jobs launched with --predictions-dir (v3.0.5+) must carry checked
        # per-example predictions for every isolation method they ran.
        command = status.get("command") or []
        if "--predictions-dir" in command:
            isolation_run = [m for m in str(job.get("methods", "")).split(",")
                             if m in ISOLATION]
            if len(result.get("predictions_checked", [])) < len(isolation_run):
                errors.append(f"{job_id}: per-example predictions missing for "
                              f"some isolation methods "
                              f"({len(result.get('predictions_checked', []))}"
                              f"/{len(isolation_run)} checked)")

    determinism = []
    for entry in plan.get("jobs", []):
        if entry.get("job", {}).get("repeat_of"):
            determinism.append(compare_repeat(run_dir, entry, errors))

    required_top = (
        "environment.json", "source_manifest.json", "pip_freeze.txt",
        "suite_summary.json", "failures.json",
    )
    for filename in required_top:
        if not (run_dir / filename).is_file():
            errors.append(f"suite metadata missing: {filename}")
    source_path = run_dir / "source_manifest.json"
    if source_path.is_file():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source.get("execution_sha256") != plan_execution_sha:
            errors.append("plan/source-manifest execution hash mismatch")
    # Publication runs (P0/P1) need a content-hash fingerprint; the smoke
    # profile hashes metadata only, which is recorded as a warning so a clean
    # smoke run still verifies.
    warnings: list[str] = []
    if acknowledged_jobs:
        reasons = "; ".join(f"{a.get('from', '')[:12]}->{a.get('to', '')[:12]}: "
                            f"{a.get('reason')}" for a in acknowledged)
        warnings.append(
            f"{len(acknowledged_jobs)} job(s) ran under an earlier, acknowledged "
            f"source version ({reasons}); their results were kept on the "
            f"operator's assertion that the change cannot alter them")
    publication = any(entry.get("job", {}).get("priority") in ("P0", "P1")
                      for entry in plan.get("jobs", []))
    if fingerprint_path.is_file():
        if data_fingerprint.get("mode") != "full":
            message = ("dataset fingerprint is not content-hash/full mode; "
                       "acceptable for smoke but not the publication gate")
            (errors if publication else warnings).append(message)
    elif publication:
        errors.append("publication suite has no dataset_fingerprint.json")

    report = {
        "verification_schema_version": 1,
        "verified_at": utc_now(), "run_dir": str(run_dir.resolve()),
        "status": "verified" if not errors else "failed",
        "publication_reportable": not errors,
        "jobs_expected": len(plan.get("jobs", [])),
        "jobs_validated": len(results),
        "prediction_files_checked": sum(len(r.get("predictions_checked", []))
                                        for r in results),
        "results": results, "determinism": determinism,
        "dataset_fingerprint": ({
            key: data_fingerprint.get(key) for key in
            ("root", "mode", "file_count", "total_bytes", "tree_sha256")
        } if data_fingerprint else None),
        "errors": errors,
        "warnings": warnings,
        "acknowledged_source_changes": acknowledged,
        "jobs_under_acknowledged_source": acknowledged_jobs,
        "scope_note": (
            "This verifier establishes artifact arithmetic, structure, hashes, "
            "suite completeness, and the declared same-seed repeat. It does not "
            "turn the reused-C-head Task-IL shared baselines into standard "
            "separate-head baselines and does not certify a compact exporter."),
    }
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write the report here instead of RUN_DIR/verification_report.json "
             "(use it to re-verify a historical run without replacing its "
             "original report).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    report = verify(run_dir)
    out = (args.out or run_dir / "verification_report.json").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, report)
    print(json.dumps({
        "status": report["status"],
        "jobs_validated": report.get("jobs_validated", 0),
        "prediction_files_checked": report.get("prediction_files_checked", 0),
        "errors": report.get("errors", []),
        "report": str(out),
    }, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
