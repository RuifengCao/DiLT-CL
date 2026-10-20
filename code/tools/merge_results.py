#!/usr/bin/env python3
"""Merge per-job result artifacts into one artifact per experiment group.

``run_suite.py`` runs one job per ``(benchmark, seed, method)`` by default, so a
finished P0 run holds hundreds of small ``result.json`` / ``result.npz`` pairs.
This tool folds them back into the single-artifact layout that the paper
tables, ``tools/verify_experiment_bundle.py --results`` and the notebooks
expect, and it recomputes every derived quantity instead of copying it:

* ``per_seed`` lists are re-assembled in ascending seed order,
* ``aggregated`` (mean/std/values) is recomputed from the merged per-seed
  metrics with ``diltcl.stats.aggregate_seeds``,
* ``paired_significance`` is added (Wilcoxon / paired t) against DiLT-CL for
  every metric with at least five aligned seeds,
* the NPZ matrices are concatenated key by key (duplicates are an error),
* the merged artifact is marked ``verified`` only when the independent
  verifier (``tools/verify_experiment_bundle.py``) reported the run as
  verified, every input job passed it, the SHA-256 of every input JSON/NPZ
  equals the hash in the verifier's report (so nothing changed after
  verification), the report postdates every job's completion, and the
  merge's own recomputation agreed -- this is the only code path that turns
  a result artifact from ``unverified`` into ``verified``; otherwise the
  merged artifact is ``unverified`` whatever its inputs said. With
  ``--inputs`` the same binding is applied to every input that is a per-job
  artifact (``RUN/artifacts/<job>/result.json``) of a verified run, so a
  merge across runs (for example ten seeds from two runs) can be verified.

Every input must agree on the experiment identity -- schema version, code
version, ``config``, execution options, head mode / inference settings. A
disagreement is an error, not a warning, because a table that silently mixes
protocols is exactly the failure mode this project's audit trail exists to
prevent. Completeness is enforced too: every ``(benchmark, method)`` pair must
cover the same seeds unless ``--allow-incomplete`` is given, in which case the
gaps are recorded in the merged artifact rather than hidden.

Usage
-----
Merge a whole suite run directory (one output per job group)::

    python tools/merge_results.py /runs/p0_v3 --out-dir /runs/p0_v3/merged

Merge an explicit list of artifacts::

    python tools/merge_results.py --inputs a/result.json b/result.json \\
        --out /tmp/cifar_multi

The tool is torch-free (``diltcl.stats`` is loaded by path).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOOL_VERSION = "1.0"

# Canonical method order for merged artifacts (unknown methods follow).
METHOD_ORDER = ["finetune", "ewc", "lwf", "er", "derpp", "packnet", "supsup",
                "wsn", "diltcl", "independent"]
BENCHMARK_ORDER = ["cifar", "pmnist", "tinyimagenet", "fivedatasets"]

# Top-level keys that are merged rather than required to be identical.
_MERGED_KEYS = {
    "seeds", "benchmarks", "methods", "per_seed", "aggregated",
    "per_seed_run_records", "calibration", "benchmark_metadata", "memory",
    "environment", "execution",
    # class-IL artifacts partition their own method list; recomputed below.
    "shared_backbone_methods", "isolation_methods",
    # class-IL head conventions carry per-benchmark output counts; unioned.
    "head_conventions",
}
# Method families as declared by run_class_il.py.
SHARED_BACKBONE_METHODS = ["finetune", "ewc", "lwf", "er", "derpp"]
ISOLATION_METHODS = ["packnet", "supsup", "wsn", "diltcl"]
# Keys that may legitimately differ between jobs and are simply taken from the
# first input (they describe the artifact class, not the experiment).
_FIRST_WINS_KEYS = {"evidence_status", "verification_status",
                    "artifact_reportability"}
# Per-seed sub-maps of benchmark_metadata that union across inputs.
_PER_SEED_METADATA = ("per_seed", "per_seed_counts")
# Environment keys that legitimately differ between jobs of one table (the
# command line names the job; PYTHONHASHSEED is set to the seed).
_VOLATILE_ENVIRONMENT_KEYS = ("command", "python_hash_seed")


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stats = _load_by_path("diltcl_stats_standalone", ROOT / "diltcl" / "stats.py")
verifier = _load_by_path("diltcl_verifier_standalone",
                         ROOT / "tools" / "verify_experiment_bundle.py")
config_module = _load_by_path("diltcl_config_standalone", ROOT / "diltcl" / "config.py")


def _config_defaults() -> dict[str, Any]:
    """Field defaults of ``ExperimentConfig`` (JSON-safe)."""
    cfg = config_module.ExperimentConfig()
    return json.loads(json.dumps(cfg.__dict__, default=list))


def normalised_config(config: dict[str, Any]) -> dict[str, Any]:
    """``config`` with fields added by later versions filled from defaults.

    An artifact written before a config field existed ran with that field's
    default behaviour, so comparing configs after filling missing keys from
    the current defaults is the honest identity check across the small
    version steps a run directory may span.
    """
    filled = dict(_config_defaults())
    filled.update(json.loads(json.dumps(config, default=list)))
    return filled


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _ordered(values, order):
    rank = {name: i for i, name in enumerate(order)}
    return sorted(set(values), key=lambda v: (rank.get(v, len(order)), str(v)))


class MergeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------

def load_inputs(json_paths: list[Path]) -> list[dict[str, Any]]:
    inputs = []
    for json_path in json_paths:
        json_path = Path(json_path)
        npz_path = json_path.with_suffix(".npz")
        if not json_path.is_file():
            raise MergeError(f"missing result JSON: {json_path}")
        if not npz_path.is_file():
            raise MergeError(f"missing result NPZ next to {json_path}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        inputs.append({"json_path": json_path, "npz_path": npz_path,
                       "payload": payload, "arrays": arrays,
                       "json_sha256": sha256_file(json_path),
                       "npz_sha256": sha256_file(npz_path)})
    return inputs


def _check_identity(inputs, allow_mismatch: set[str]) -> dict[str, Any]:
    """Every non-merged top-level key must agree across inputs."""
    keys = set()
    for item in inputs:
        keys.update(item["payload"].keys())
    mismatches = {}
    for key in sorted(keys):
        if key in _MERGED_KEYS or key in _FIRST_WINS_KEYS:
            continue
        if key == "config":
            values = {_canonical(normalised_config(item["payload"].get(key) or {}))
                      for item in inputs}
        else:
            values = {_canonical(item["payload"].get(key, "<absent>"))
                      for item in inputs}
        if len(values) > 1:
            mismatches[key] = sorted(values)[:3]
    # Execution *options* are part of the identity; the effective flags and
    # thread counts may differ between hosts and are recorded per input.
    exec_options = {_canonical((item["payload"].get("execution") or {}).get("options"))
                    for item in inputs}
    if len(exec_options) > 1:
        mismatches["execution.options"] = sorted(exec_options)[:3]
    hard = {k: v for k, v in mismatches.items() if k not in allow_mismatch}
    if hard:
        lines = [f"  {key}: {values}" for key, values in hard.items()]
        raise MergeError(
            "inputs describe different experiments; refusing to merge:\n"
            + "\n".join(lines)
            + "\n(use --allow-mismatch KEY to override deliberately)")
    return {"identity_keys_checked": sorted(k for k in keys
                                            if k not in _MERGED_KEYS
                                            and k not in _FIRST_WINS_KEYS),
            "overridden_mismatches": {k: mismatches[k] for k in mismatches
                                      if k in allow_mismatch},
            "first_input_wins": sorted(_FIRST_WINS_KEYS & keys)}


_MEMORY_ROW_ORDER = ["Finetune", "EWC", "LwF", "ER", "DER++", "PackNet", "SupSup",
                     "WSN", "DiLT-CL (ours)", "Independent"]


def _ordered_memory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {name: i for i, name in enumerate(_MEMORY_ROW_ORDER)}
    return sorted(rows, key=lambda r: (rank.get(r.get("name"), len(rank)), str(r.get("name"))))


def _merge_head_conventions(inputs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Union the per-benchmark output counts; every other field must agree."""
    merged: dict[str, Any] | None = None
    for item in inputs:
        conventions = item["payload"].get("head_conventions")
        if conventions is None:
            continue
        counts = conventions.get("per_benchmark_output_counts") or {}
        rest = {k: v for k, v in conventions.items() if k != "per_benchmark_output_counts"}
        if merged is None:
            merged = dict(rest)
            merged["per_benchmark_output_counts"] = dict(counts)
            continue
        merged_rest = {k: v for k, v in merged.items() if k != "per_benchmark_output_counts"}
        if _canonical(merged_rest) != _canonical(rest):
            raise MergeError("head_conventions differ between inputs")
        for benchmark, value in counts.items():
            existing = merged["per_benchmark_output_counts"].get(benchmark)
            if existing is not None and _canonical(existing) != _canonical(value):
                raise MergeError(
                    f"head_conventions.per_benchmark_output_counts[{benchmark}] differ")
            merged["per_benchmark_output_counts"][benchmark] = value
    return merged


def _merge_per_seed(target: dict, incoming: dict, label: str) -> None:
    """Union per-seed metadata one level deeper than the seed.

    Class-IL jobs that ran one method family record only that family under
    ``per_seed_counts[seed]``; a shared-only and an isolation-only job of the
    same seed must combine into one record with both families rather than the
    later job replacing the earlier one. Identical sub-records (the task-IL
    loader provenance is the same for every method of a seed) are accepted;
    conflicting values are an error.
    """
    for seed, record in incoming.items():
        current = target.get(seed)
        if current is None:
            target[seed] = dict(record) if isinstance(record, dict) else record
            continue
        if not (isinstance(current, dict) and isinstance(record, dict)):
            if _canonical(current) != _canonical(record):
                raise MergeError(f"{label}[{seed}] differs between inputs")
            continue
        for key, value in record.items():
            if key in current and _canonical(current[key]) != _canonical(value):
                raise MergeError(f"{label}[{seed}].{key} differs between inputs")
            current[key] = value


def merge_payloads(inputs: list[dict[str, Any]], *, allow_incomplete: bool,
                   allow_mismatch: set[str], group: str | None) -> tuple[dict, dict]:
    if not inputs:
        raise MergeError("no inputs to merge")
    identity = _check_identity(inputs, allow_mismatch)

    per_seed: dict[str, dict[str, dict[int, dict]]] = {}
    run_records: dict[str, dict[str, dict[str, Any]]] = {}
    calibration: dict[str, dict[str, dict[str, Any]]] = {}
    benchmark_metadata: dict[str, dict[str, Any]] = {}
    memory: dict[str, Any] = {}
    memory_source: dict[str, str] = {}
    environments: dict[str, dict[str, Any]] = {}
    executions: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    array_sources: dict[str, str] = {}
    all_seeds: set[int] = set()
    benchmarks: list[str] = []
    methods: list[str] = []

    for item in inputs:
        payload = item["payload"]
        seeds = [int(s) for s in payload["seeds"]]
        all_seeds.update(seeds)
        benchmarks.extend(payload["benchmarks"])
        methods.extend(payload["methods"])
        env_signature = {k: v for k, v in (payload.get("environment") or {}).items()
                         if k not in _VOLATILE_ENVIRONMENT_KEYS}
        environments.setdefault(_canonical(env_signature), env_signature)
        executions.append({"input": str(item["json_path"]),
                           "execution": payload.get("execution")})
        for benchmark, by_method in payload["per_seed"].items():
            for method, metric_list in by_method.items():
                if len(metric_list) != len(seeds):
                    raise MergeError(
                        f"{item['json_path']}: per_seed[{benchmark}][{method}] "
                        f"has {len(metric_list)} entries for {len(seeds)} seeds")
                slot = per_seed.setdefault(benchmark, {}).setdefault(method, {})
                for seed, metrics in zip(seeds, metric_list):
                    if seed in slot:
                        raise MergeError(
                            f"duplicate result for {benchmark}/{method}/seed{seed} "
                            f"(second copy in {item['json_path']})")
                    slot[seed] = metrics
        for benchmark, by_method in (payload.get("per_seed_run_records") or {}).items():
            for method, by_seed in by_method.items():
                slot = run_records.setdefault(benchmark, {}).setdefault(method, {})
                for seed, record in by_seed.items():
                    if seed in slot:
                        raise MergeError(
                            f"duplicate run record for {benchmark}/{method}/seed{seed}")
                    slot[seed] = record
        for benchmark, by_method in (payload.get("calibration") or {}).items():
            for method, by_seed in by_method.items():
                slot = calibration.setdefault(benchmark, {}).setdefault(method, {})
                for seed, record in by_seed.items():
                    if seed in slot:
                        raise MergeError(
                            f"duplicate calibration record for "
                            f"{benchmark}/{method}/seed{seed}")
                    slot[seed] = record
        for benchmark, metadata in (payload.get("benchmark_metadata") or {}).items():
            # Task-IL keeps loader provenance under "per_seed", class-IL under
            # "per_seed_counts"; both are per-seed maps that union, while the
            # rest of the metadata must be identical across inputs.
            metadata = dict(metadata)
            per_seed_parts = {key: metadata.pop(key) for key in _PER_SEED_METADATA
                              if key in metadata}
            existing = benchmark_metadata.get(benchmark)
            if existing is None:
                benchmark_metadata[benchmark] = dict(metadata)
                for key, value in per_seed_parts.items():
                    benchmark_metadata[benchmark][key] = {
                        seed: dict(v) if isinstance(v, dict) else v
                        for seed, v in value.items()}
            else:
                stripped = {k: v for k, v in existing.items()
                            if k not in _PER_SEED_METADATA}
                if _canonical(stripped) != _canonical(metadata):
                    raise MergeError(
                        f"benchmark_metadata[{benchmark}] differs between inputs")
                for key, value in per_seed_parts.items():
                    _merge_per_seed(existing.setdefault(key, {}), value,
                                    f"benchmark_metadata[{benchmark}].{key}")
        for benchmark, rows in (payload.get("memory") or {}).items():
            # Per-method jobs each emit the analytic memory rows of their own
            # method(s); the merged table is the union by row name. When the
            # same row appears in several inputs, the one from a job that ran
            # DiLT-CL wins because it embeds DiLT-CL's measured densities.
            has_diltcl = "diltcl" in payload["methods"]
            table = memory.setdefault(benchmark, {})
            sources = memory_source.setdefault(benchmark, {})
            for row in rows:
                name = row["name"]
                if name not in table or (has_diltcl and not sources[name].endswith("[diltcl]")):
                    table[name] = row
                    sources[name] = f"{item['json_path']}" + ("[diltcl]" if has_diltcl else "")
        for key, array in item["arrays"].items():
            if key in arrays:
                raise MergeError(
                    f"duplicate NPZ key {key!r} in {item['npz_path']} "
                    f"(first seen in {array_sources[key]})")
            arrays[key] = array
            array_sources[key] = str(item["npz_path"])

    seeds_sorted = sorted(all_seeds)
    benchmarks = _ordered(benchmarks, BENCHMARK_ORDER)
    methods = _ordered(methods, METHOD_ORDER)
    _recheck_metrics_against_matrices(per_seed, arrays)
    head_conventions = _merge_head_conventions(inputs)

    # Completeness: every (benchmark, method) must cover every seed.
    incomplete = []
    for benchmark in benchmarks:
        for method in methods:
            have = sorted(per_seed.get(benchmark, {}).get(method, {}))
            if have != seeds_sorted:
                incomplete.append({"benchmark": benchmark, "method": method,
                                   "seeds_present": have,
                                   "seeds_missing": sorted(set(seeds_sorted) - set(have))})
    if incomplete and not allow_incomplete:
        lines = [f"  {row['benchmark']}/{row['method']}: missing seeds "
                 f"{row['seeds_missing']}" for row in incomplete]
        raise MergeError("merged set is incomplete; refusing to write a table "
                         "that would silently drop seeds:\n" + "\n".join(lines)
                         + "\n(use --allow-incomplete to record the gaps instead)")

    per_seed_out: dict[str, dict[str, list]] = {}
    aggregated: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        per_seed_out[benchmark] = {}
        aggregated[benchmark] = {}
        for method in methods:
            by_seed = per_seed.get(benchmark, {}).get(method, {})
            if not by_seed:
                continue
            ordered = [by_seed[seed] for seed in sorted(by_seed)]
            per_seed_out[benchmark][method] = ordered
            aggregated[benchmark][method] = stats.aggregate_seeds(ordered)

    significance = _paired_significance(per_seed, benchmarks, methods)

    merged = {}
    for key, value in inputs[0]["payload"].items():
        if key in _MERGED_KEYS:
            continue
        merged[key] = value
    versions = sorted({str(item["payload"].get("code_version")) for item in inputs})
    if len(versions) > 1:
        # Only reachable with --allow-mismatch code_version or an acknowledged
        # source change; say so in the field tables and captions read.
        merged["code_version"] = "/".join(versions)
    merged["config"] = normalised_config(inputs[0]["payload"].get("config") or {})
    merged.update({
        "seeds": seeds_sorted,
        "benchmarks": benchmarks,
        "methods": methods,
        "execution": inputs[0]["payload"].get("execution"),
        "environment": inputs[0]["payload"].get("environment"),
        "benchmark_metadata": benchmark_metadata,
        "per_seed": per_seed_out,
        "aggregated": aggregated,
        "paired_significance": significance,
        "per_seed_run_records": run_records,
    })
    if head_conventions is not None:
        merged["head_conventions"] = head_conventions
    if calibration:
        merged["calibration"] = calibration
    if memory:
        merged["memory"] = {benchmark: _ordered_memory_rows(list(table.values()))
                            for benchmark, table in memory.items()}
    if any("shared_backbone_methods" in item["payload"] for item in inputs):
        merged["shared_backbone_methods"] = [
            m for m in methods if m in SHARED_BACKBONE_METHODS]
        merged["isolation_methods"] = [
            m for m in methods if m in ISOLATION_METHODS]
    merged["merge_provenance"] = {
        "tool": "tools/merge_results.py", "tool_version": TOOL_VERSION,
        "merged_at": utc_now(), "group": group,
        "input_count": len(inputs),
        "inputs": [{"json": str(item["json_path"]), "json_sha256": item["json_sha256"],
                    "npz": str(item["npz_path"]), "npz_sha256": item["npz_sha256"],
                    "seeds": item["payload"]["seeds"],
                    "benchmarks": item["payload"]["benchmarks"],
                    "methods": item["payload"]["methods"]} for item in inputs],
        "identity": identity,
        "code_versions": versions,
        "config_normalised_with_defaults": True,
        "distinct_environments": len(environments),
        "environment_consistent": len(environments) == 1,
        "environment_consistency_ignores": list(_VOLATILE_ENVIRONMENT_KEYS),
        "environments": list(environments.values()),
        "executions_per_input": executions,
        "memory_table_sources": memory_source,
        "incomplete_pairs": incomplete,
        "complete": not incomplete,
        "recomputed": ["aggregated", "paired_significance"],
        "rechecked_against_npz": ["ACC", "BWT", "Forgetting", "LA"],
        "note": ("per_seed lists are ordered by ascending seed; aggregated and "
                 "paired_significance were recomputed from the merged per-seed "
                 "metrics, not copied from the inputs; ACC/BWT/Forgetting/LA of "
                 "every seed were recomputed from the NPZ matrices."),
    }
    return merged, arrays


def _recheck_metrics_against_matrices(per_seed, arrays):
    """Recompute ACC/BWT/Forgetting/LA from every main matrix in the NPZ.

    The suite verifier already did this per job; repeating it here guards
    against merging a JSON with the wrong NPZ, or an edited JSON.
    """
    problems = []
    for benchmark, by_method in per_seed.items():
        for method, by_seed in by_method.items():
            for seed, metrics in by_seed.items():
                key = f"{benchmark}/{method}/seed{seed}"
                if key not in arrays:
                    problems.append(f"{key}: main matrix missing from NPZ")
                    continue
                matrix = np.asarray(arrays[key], dtype=float)
                if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                    problems.append(f"{key}: matrix is not square")
                    continue
                for name, value in verifier.summarize(matrix).items():
                    if not verifier.close(metrics.get(name), value):
                        problems.append(
                            f"{key}: {name} JSON={metrics.get(name)} "
                            f"recomputed={value}")
    if problems:
        raise MergeError("per-seed metrics disagree with the NPZ matrices:\n  "
                         + "\n  ".join(problems[:20]))


def _paired_significance(per_seed, benchmarks, methods, reference="diltcl",
                         min_seeds=5):
    """Paired tests of every method against ``reference`` per metric."""
    out: dict[str, Any] = {"reference": reference, "min_aligned_seeds": min_seeds,
                           "tests": {}}
    for benchmark in benchmarks:
        by_method = per_seed.get(benchmark, {})
        ref = by_method.get(reference)
        if not ref:
            continue
        out["tests"][benchmark] = {}
        for method in methods:
            if method == reference or method not in by_method:
                continue
            shared = sorted(set(ref) & set(by_method[method]))
            if len(shared) < min_seeds:
                out["tests"][benchmark][method] = {
                    "aligned_seeds": shared, "suppressed": True,
                    "reason": f"fewer than {min_seeds} aligned seeds"}
                continue
            metrics = [m for m in ref[shared[0]] if m in by_method[method][shared[0]]]
            record = {"aligned_seeds": shared, "suppressed": False, "metrics": {}}
            for metric in metrics:
                a = [float(ref[s][metric]) for s in shared]
                b = [float(by_method[method][s][metric]) for s in shared]
                test = stats.paired_significance(a, b)
                record["metrics"][metric] = {
                    **test,
                    "mean_difference_reference_minus_method":
                        float(np.mean(np.asarray(a) - np.asarray(b))),
                }
            out["tests"][benchmark][method] = record
    return out


def write_outputs(merged: dict[str, Any], arrays: dict[str, np.ndarray],
                  out_prefix: Path) -> tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    npz_path = out_prefix.with_suffix(".npz")
    json_path.write_text(json.dumps(merged, indent=2, allow_nan=False),
                         encoding="utf-8")
    np.savez(npz_path, **arrays)
    return json_path, npz_path


# ---------------------------------------------------------------------------
# Suite run directory mode
# ---------------------------------------------------------------------------

def load_verification(run_dir: Path) -> dict[str, Any] | None:
    """Summarise ``verification_report.json`` written by the verifier."""
    path = run_dir / "verification_report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path), "sha256": sha256_file(path),
        "status": report.get("status"),
        "verified_at": report.get("verified_at"),
        "job_errors": {r.get("job_id"): list(r.get("errors", []))
                       for r in report.get("results", [])},
        "job_hashes": {r.get("job_id"): (r.get("json_sha256"), r.get("npz_sha256"))
                       for r in report.get("results", [])},
        "error_count": len(report.get("errors", [])),
    }


def _parse_time(text: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None


def job_reasons(verification: dict[str, Any], job_id: str,
                item: dict[str, Any] | None, ended_at: Any) -> list[str]:
    """Why one input job does not count as verified (empty when it does).

    The verifier's report must cover the job without errors, record the same
    SHA-256 for its JSON and NPZ as the files being merged (nothing changed
    after verification), and postdate the job's completion.
    """
    errors = verification["job_errors"].get(job_id)
    if errors is None:
        return [f"{job_id}: not covered by the verification report"]
    reasons = [f"{job_id}: verifier errors {errors[:2]}"] if errors else []
    if item is not None:
        recorded = tuple(verification["job_hashes"].get(job_id) or (None, None))
        if recorded != (item["json_sha256"], item["npz_sha256"]):
            reasons.append(f"{job_id}: result files differ from the verified ones "
                           "(SHA-256 differs from the verifier's report)")
    if ended_at is not None:
        verified_at, finished = _parse_time(verification["verified_at"]), _parse_time(ended_at)
        if verified_at is None or finished is None or verified_at < finished:
            reasons.append(f"{job_id}: the verification report predates the job's completion")
    return reasons


def _set_status(merged: dict[str, Any], reasons: list[str], record: dict[str, Any]) -> None:
    merged["merge_provenance"]["verification"] = record
    reportability = dict(merged.get("artifact_reportability") or {})
    if not reasons:
        merged["verification_status"] = "verified"
        merged["evidence_status"] = "verified"
        reportability["postrun_validation_completed"] = True
        reportability["verified_by"] = "tools/verify_experiment_bundle.py + tools/merge_results.py"
    else:
        # Never inherit a status from the inputs: a merge that was not
        # promoted here is unverified, whatever its inputs claimed.
        merged["verification_status"] = "unverified"
        merged["evidence_status"] = "unverified"
        reportability["postrun_validation_completed"] = False
        reportability.pop("verified_by", None)
    merged["artifact_reportability"] = reportability


def apply_verification(merged: dict[str, Any], verification: dict[str, Any] | None,
                       job_ids: list[str], missing: list[dict[str, Any]],
                       items: dict[str, dict[str, Any]] | None = None,
                       ended: dict[str, Any] | None = None) -> None:
    """Promote a merged artifact to ``verified`` when the evidence allows it.

    Conditions (all required): the verifier reported the run as verified,
    every job of the group appears in its results without errors, the merged
    input files are byte-identical to the verified ones (``items``: job id ->
    loaded input with its SHA-256s), the report postdates every job's
    completion (``ended``: job id -> ``ended_at``), no job of the group is
    missing, and the merge recheck against the NPZ matrices passed (it raises
    otherwise, so reaching this point implies it).
    """
    reasons = []
    if verification is None:
        reasons.append("verification_report.json not found; run "
                       "tools/verify_experiment_bundle.py first")
    else:
        if verification["status"] != "verified":
            reasons.append(f"verifier status is {verification['status']!r} "
                           f"({verification['error_count']} error(s))")
        for job_id in job_ids:
            reasons += job_reasons(verification, job_id,
                                   None if items is None else items.get(job_id),
                                   None if ended is None else ended.get(job_id))
    if missing:
        reasons.append(f"{len(missing)} job(s) of the group did not succeed")
    record = {
        "verified": not reasons,
        "verifier_report": None if verification is None else {
            "path": verification["path"], "sha256": verification["sha256"],
            "verified_at": verification["verified_at"]},
        "bound_to_input_hashes": items is not None,
        "merge_recheck": "ACC/BWT/Forgetting/LA recomputed from NPZ for every seed",
        "reasons_not_verified": reasons,
    }
    _set_status(merged, reasons, record)


def bind_inputs(merged: dict[str, Any], inputs: list[dict[str, Any]]) -> None:
    """``--inputs`` mode: verify each input against its own run's report."""
    reasons: list[str] = []
    reports: dict[Path, dict[str, Any] | None] = {}
    for item in inputs:
        path = Path(item["json_path"]).resolve()
        job_dir = path.parent
        run_dir = job_dir.parent.parent
        job_id = job_dir.name
        if (path.name != "result.json" or job_dir.parent.name != "artifacts"
                or not (run_dir / "plan.json").is_file()):
            reasons.append(f"{path}: not a per-job artifact of a suite run directory")
            continue
        if run_dir not in reports:
            reports[run_dir] = load_verification(run_dir)
        verification = reports[run_dir]
        if verification is None:
            reasons.append(f"{run_dir}: verification_report.json not found")
            continue
        if verification["status"] != "verified":
            reasons.append(f"{run_dir}: verifier status is {verification['status']!r}")
            continue
        status_path = run_dir / "statuses" / f"{job_id}.json"
        status = (json.loads(status_path.read_text(encoding="utf-8"))
                  if status_path.is_file() else None)
        if status is None or status.get("state") != "succeeded":
            reasons.append(f"{job_id}: no succeeded status in {run_dir}")
            continue
        reasons += job_reasons(verification, job_id, item, status.get("ended_at"))
    record = {
        "verified": not reasons,
        "verifier_reports": [None if v is None else {
            "run_dir": str(run), "path": v["path"], "sha256": v["sha256"],
            "verified_at": v["verified_at"]} for run, v in sorted(reports.items())],
        "bound_to_input_hashes": True,
        "merge_recheck": "ACC/BWT/Forgetting/LA recomputed from NPZ for every seed",
        "reasons_not_verified": reasons,
    }
    _set_status(merged, reasons, record)


def group_of(job_id: str) -> str:
    """``p0_task_il_multi__cifar__seed0__lwf`` -> ``p0_task_il_multi``."""
    return job_id.split("__", 1)[0]


def collect_from_run_dir(run_dir: Path, wanted_groups: set[str] | None,
                         allow_incomplete: bool) -> dict[str, dict[str, Any]]:
    plan_path = run_dir / "plan.json"
    if not plan_path.is_file():
        raise MergeError(f"{run_dir} has no plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, Any]] = {}
    for entry in plan.get("jobs", []):
        job = entry["job"]
        if job.get("repeat_of"):
            continue   # determinism repeats are compared, not tabulated
        group = group_of(job["job_id"])
        if wanted_groups and group not in wanted_groups:
            continue
        record = groups.setdefault(group, {"inputs": [], "missing": [],
                                           "job_ids": [], "input_jobs": [],
                                           "ended_at": {}})
        record["job_ids"].append(job["job_id"])
        status_path = run_dir / "statuses" / f"{job['job_id']}.json"
        prefix = run_dir / entry["output_prefix_relative"]
        status = (json.loads(status_path.read_text(encoding="utf-8"))
                  if status_path.is_file() else None)
        if status is None or status.get("state") != "succeeded":
            record["missing"].append({
                "job_id": job["job_id"],
                "state": None if status is None else status.get("state")})
            continue
        record["inputs"].append(prefix.with_suffix(".json"))
        record["input_jobs"].append(job["job_id"])
        record["ended_at"][job["job_id"]] = status.get("ended_at")
    for group, record in groups.items():
        if record["missing"] and not allow_incomplete:
            raise MergeError(
                f"group {group}: {len(record['missing'])} job(s) did not succeed "
                f"(e.g. {record['missing'][0]['job_id']}); rerun the suite to "
                "resume them, or pass --allow-incomplete to merge what exists")
    return groups


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="Suite run directory (merges every job group).")
    parser.add_argument("--inputs", nargs="+", type=Path, default=None,
                        help="Explicit result JSON files to merge into one artifact.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output prefix for --inputs mode.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for run-dir mode "
                             "(default: RUN_DIR/merged).")
    parser.add_argument("--group", action="append", default=None,
                        help="Only merge these job groups (run-dir mode).")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Merge even if some (benchmark, method, seed) "
                             "results are missing; gaps are recorded.")
    parser.add_argument("--allow-mismatch", action="append", default=[],
                        metavar="KEY",
                        help="Top-level key allowed to differ between inputs.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    allow_mismatch = set(args.allow_mismatch)
    try:
        if args.inputs:
            if args.out is None:
                raise MergeError("--out is required with --inputs")
            inputs = load_inputs(args.inputs)
            merged, arrays = merge_payloads(
                inputs, allow_incomplete=args.allow_incomplete,
                allow_mismatch=allow_mismatch, group=None)
            bind_inputs(merged, inputs)
            json_path, npz_path = write_outputs(merged, arrays, args.out)
            reasons = merged["merge_provenance"]["verification"]["reasons_not_verified"]
            print(f"merged {len(args.inputs)} artifact(s) -> {json_path}, {npz_path} "
                  f"({merged['verification_status']})")
            for reason in reasons[:5]:
                print(f"  not verified: {reason}")
            return 0
        if args.run_dir is None:
            raise MergeError("give a run directory or --inputs")
        out_dir = args.out_dir or (args.run_dir / "merged")
        groups = collect_from_run_dir(
            args.run_dir.resolve(), set(args.group or []) or None,
            args.allow_incomplete)
        if not groups:
            raise MergeError("no job groups found in the plan")
        verification = load_verification(args.run_dir.resolve())
        plan = json.loads((args.run_dir / "plan.json").read_text(encoding="utf-8"))
        acknowledged = plan.get("source_change_acknowledgements", [])
        if acknowledged:
            # Jobs finished under acknowledged earlier code versions report a
            # different code_version; the acknowledgement (reason, hashes) is
            # carried into the merged artifact instead of refusing the merge.
            allow_mismatch = set(allow_mismatch) | {"code_version"}
        summary = {}
        for group, record in sorted(groups.items()):
            if not record["inputs"]:
                print(f"[skip] {group}: no successful jobs")
                continue
            inputs = load_inputs(record["inputs"])
            merged, arrays = merge_payloads(
                inputs, allow_incomplete=args.allow_incomplete,
                allow_mismatch=allow_mismatch, group=group)
            merged["merge_provenance"]["jobs_not_succeeded"] = record["missing"]
            merged["merge_provenance"]["source_change_acknowledgements"] = acknowledged
            apply_verification(merged, verification, record["job_ids"],
                               record["missing"],
                               items=dict(zip(record["input_jobs"], inputs)),
                               ended=record["ended_at"])
            json_path, npz_path = write_outputs(merged, arrays, out_dir / group)
            summary[group] = {"json": str(json_path), "npz": str(npz_path),
                              "inputs": len(record["inputs"]),
                              "missing_jobs": len(record["missing"]),
                              "verification_status": merged["verification_status"],
                              "seeds": merged["seeds"],
                              "methods": merged["methods"],
                              "benchmarks": merged["benchmarks"]}
            print(f"[merged] {group}: {len(record['inputs'])} jobs -> {json_path} "
                  f"({merged['verification_status']})")
            for reason in merged["merge_provenance"]["verification"]["reasons_not_verified"][:5]:
                print(f"  not verified: {reason}")
        (out_dir / "merge_summary.json").write_text(
            json.dumps({"merged_at": utc_now(), "run_dir": str(args.run_dir),
                        "groups": summary}, indent=2), encoding="utf-8")
        return 0
    except MergeError as exc:
        print(f"merge_results: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
