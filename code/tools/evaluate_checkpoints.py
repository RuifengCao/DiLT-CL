#!/usr/bin/env python3
"""Re-evaluate persisted class-IL isolation checkpoints -- no training.

For every class-IL job artifact (``artifacts/<job>/result.json``) and every
isolation method/seed whose trained state was persisted, this tool

1. checks the checkpoint file against the artifact (byte count and SHA-256 of
   the file, then the canonical tensor-content SHA-256 after loading it with
   ``torch.load(..., weights_only=True)``);
2. rebuilds the evaluated model exactly as the runner did -- same benchmark
   setup, seed, configuration and execution options, all read from the
   artifact;
3. re-runs the class-IL evaluation with the job's own router (the raw
   criterion, or the serialized prefix calibrators of a calibrated job, plus
   the matched raw router), each calibrator applied in the precision stored
   with it (float64 since v3.0.10, float32 for earlier states), as the job
   applied it;
4. compares every matrix (routed, task-inference, conditional, tie; raw_*
   for calibrated jobs) and every per-seed metric, including the oracle
   accuracy, with ``result.npz`` / ``result.json``, and, when the job wrote
   per-example predictions, every routed decision.

So the routed, task-inference and oracle results are regenerated from the
released experts alone. Every per-seed metric of the artifact is compared
(ACC, LA, BWT and Forgetting recomputed from the re-evaluated matrices with the
runner's own function). Comparisons are exact by default: on the recorded
software stack and GPU model the evaluation is deterministic, so any
difference -- in a matrix, a metric, or a single per-example decision -- and
any non-finite value is an error. On other hardware (another GPU model, or
the CPU) floating-point rounding can move decisions that sit on near-ties;
``--atol`` then accepts matrix and metric differences up to that many
percentage points (status ``within-tolerance``) and counts the per-example
differences instead of failing on them.

A referenced predictions file must exist and match the SHA-256 in the
artifact before it is used as the baseline. Every isolation method a class-IL
job ran must have a persisted-state record. A job whose re-evaluation raises
(missing data, a malformed checkpoint, running out of GPU memory) becomes an
error entry in the report; the other jobs are still evaluated and reported.

``--shard K/N`` processes every N-th job starting at K so N processes can
share one GPU; each shard report records the shard and its jobs, and
``--combine REPORT ...`` checks that the shard reports of a run directory
cover every class-IL job exactly once and prints the combined verdict.

Needs PyTorch and the benchmark data (the same files the job used); the
``--combine`` mode needs neither.

Examples::

    python tools/evaluate_checkpoints.py --run-dir RUN --data-root DATA
    python tools/evaluate_checkpoints.py --result RUN/artifacts/JOB/result.json \\
        --data-root DATA --methods diltcl,wsn --out eval_report.json
    python tools/evaluate_checkpoints.py --combine eval_0.json eval_1.json eval_2.json eval_3.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ISOLATION = ("packnet", "supsup", "wsn", "diltcl")
MATRICES = (("result_matrix", ""), ("task_inference_matrix", "/task_inference"),
            ("conditional_class_matrix", "/conditional_class"),
            ("routing_tie_matrix", "/routing_tie"))
METRICS = (("OracleACC", "oracle_route_acc"), ("MicroOracleACC", "micro_oracle_route_acc"),
           ("TA", "task_inference_acc"), ("MicroTA", "micro_task_inference_acc"),
           ("MicroACC", "micro_class_il_acc"), ("ConditionalACC", "conditional_class_acc"),
           ("MicroConditionalACC", "micro_conditional_class_acc"),
           ("RouteTieRate", "routing_tie_rate"), ("MicroRouteTieRate", "micro_routing_tie_rate"),
           ("ZeroCorrectRouteTasks", "zero_correct_route_task_count"))
PER_EXAMPLE = ("true_task", "label_local", "oracle_pred_local", "scores", "route_by_stage",
               "tie_by_stage", "routed_pred_local_by_stage")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Rebuilding the evaluated model
# ---------------------------------------------------------------------------

def experiment_config(payload: dict):
    from diltcl.config import ExperimentConfig
    return ExperimentConfig().with_overrides(payload["config"])


def execution_options(payload: dict):
    from diltcl.execution import ExecutionOptions
    options = (payload.get("execution") or {}).get("options") or {}
    return ExecutionOptions(**options)


def rebuild_forward(method: str, state: dict, builder, num_tasks: int, cfg, device):
    """Forward closure ``(x, t) -> logits`` from a persisted router state."""
    from diltcl import task_inference as ti
    from diltcl.masked_layers import convert_to_masked

    def tensors(mapping):
        return {key: value.to(device) for key, value in mapping.items()}

    if method == "packnet":
        if state.get("format") != "packnet-router-state-v1":
            raise ValueError(f"unexpected PackNet state format {state.get('format')!r}")
        model = builder().to(device)
        model.load_state_dict(state["model"])
        return ti.packnet_forward_fn(model, [tensors(mask) for mask in state["eval_masks"]])
    if method in ("supsup", "wsn"):
        expected = f"{method}-router-state-v1"
        if state.get("format") != expected:
            raise ValueError(f"unexpected {method} state format {state.get('format')!r}")
        sparsity = cfg.supsup_sparsity if method == "supsup" else cfg.wsn_sparsity
        model = convert_to_masked(builder(), num_tasks, sparsity,
                                  train_weights=method == "wsn").to(device)
        model.load_state_dict(state["model"])
        return (ti.supsup_forward_fn if method == "supsup" else ti.wsn_forward_fn)(model)
    if method == "diltcl":
        from diltcl.diltcl import DiLTCL
        if state.get("format") != "diltcl-dense-resume-state-v1":
            raise ValueError(f"unexpected DiLT-CL state format {state.get('format')!r}")
        hp = state.get("method_hyperparameters", {})
        model = builder().to(device)
        model.load_state_dict(state["model"])
        learner = DiLTCL(model, device=device,
                         num_partitions=hp.get("num_partitions", cfg.diltcl_partitions),
                         pruning_rounds=hp.get("pruning_rounds", cfg.pruning_rounds),
                         target_sparsity=hp.get("target_sparsity", cfg.diltcl_sparsity),
                         lr=hp.get("lr", cfg.lr))
        learner.core_mask = tensors(state["core_mask"])
        learner.frozen_core_values = tensors(state["frozen_core_values"])
        learner.initial_params = tensors(state["initial_params"])
        learner.expert_masks = {int(t): tensors(m) for t, m in state["expert_masks"].items()}
        learner.expert_weights = {int(t): tensors(w) for t, w in state["expert_weights"].items()}
        learner.core_partition_densities = list(state.get("core_partition_densities", []))
        return ti.diltcl_forward_fn(learner)
    raise ValueError(f"not an isolation method: {method}")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compare_matrix(name, actual, expected, atol, out):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if actual.shape != expected.shape:
        out["errors"].append(f"{name}: shape {actual.shape} != {expected.shape}")
        return
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        out["exact"] = False
        out["errors"].append(f"{name}: non-finite values")
        return
    diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    out["max_abs_matrix_difference"] = max(out["max_abs_matrix_difference"], diff)
    if not np.array_equal(actual, expected):
        out["exact"] = False
        if diff > atol:
            out["errors"].append(f"{name}: max abs difference {diff:.6g}")


def _compare_metric(name, actual, expected, atol, out):
    actual, expected = float(actual), float(expected)
    out["metrics_compared"].append(name)
    if not (np.isfinite(actual) and np.isfinite(expected)):
        if not (np.isnan(actual) and np.isnan(expected)):
            out["exact"] = False
            out["errors"].append(f"{name}: non-finite value (re-evaluated {actual!r}, "
                                 f"artifact {expected!r})")
        return
    if actual != expected:
        out["exact"] = False
        out["metric_differences"][name] = actual - expected
        if abs(actual - expected) > atol:
            out["errors"].append(f"{name}: re-evaluated {actual!r} vs artifact {expected!r}")


def _compare_predictions(record: dict, stored: dict, prefix: str, atol: float, out):
    """Compare every per-example decision with the job's predictions file.

    In exact mode (``atol == 0``) any difference is an error; with a
    tolerance the differences are counted and reported.
    """
    for key in PER_EXAMPLE:
        stored_key = f"{prefix}{key}"
        if stored_key not in stored:
            out["errors"].append(f"{stored_key}: missing in the predictions file")
            continue
        a = np.asarray(record[key]); b = np.asarray(stored[stored_key])
        if a.shape != b.shape:
            out["errors"].append(f"{stored_key}: shape {a.shape} != {b.shape}")
            continue
        n = int(np.sum(a != b))
        if n:
            out["exact"] = False
            out["per_example_differences"][stored_key] = n
            if atol == 0.0:
                out["errors"].append(f"{stored_key}: {n} per-example value(s) differ")


def evaluate_job(result_json: Path, *, device: str, methods: tuple[str, ...],
                 atol: float = 0.0, setup_fn=None) -> list[dict]:
    """Re-evaluate every persisted isolation checkpoint of one class-IL job."""
    import torch
    import run_class_il as runner
    from diltcl.metrics import summarize_run
    from diltcl.runner_common import _state_content_sha256
    from diltcl.task_inference import evaluate_class_il_inference
    from diltcl.utils import set_seed

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    if payload.get("protocol") != "class_il":
        return []
    job_dir = result_json.parent
    cfg = experiment_config(payload)
    execution = execution_options(payload)
    execution.apply()
    criterion = payload.get("inference_criterion", "entropy")
    setup_fn = setup_fn or runner._isolation_setup
    entries = []
    with np.load(result_json.with_suffix(".npz"), allow_pickle=False) as npz:
        for benchmark in payload["benchmarks"]:
            for seed in payload["seeds"]:
                wanted = [m for m in payload["methods"] if m in ISOLATION and m in methods]
                todo = []
                for m in wanted:
                    if str(seed) in (payload.get("calibration", {}).get(benchmark, {})
                                     .get(m, {})):
                        todo.append(m)
                    else:       # every isolation method of a class-IL job has a record
                        entries.append({"job": job_dir.name, "benchmark": benchmark,
                                        "method": m, "seed": int(seed), "router": None,
                                        "errors": ["no persisted-state record in the artifact"],
                                        "exact": False, "max_abs_matrix_difference": 0.0,
                                        "metric_differences": {}, "per_example_differences": {},
                                        "metrics_compared": [], "metrics_not_compared": []})
                if not todo:
                    continue
                set_seed(int(seed))
                setup = setup_fn(benchmark, cfg, int(seed), execution=execution)
                _, test_loaders, builder, num_tasks, cpt, disjoint = setup
                for method in todo:
                    record = payload["calibration"][benchmark][method][str(seed)]
                    trained = record["trained_state"]
                    key = f"{benchmark}/{method}/seed{seed}"
                    out = {"job": job_dir.name, "benchmark": benchmark, "method": method,
                           "seed": int(seed), "router": record.get("kind", "none"),
                           "errors": [], "exact": True, "max_abs_matrix_difference": 0.0,
                           "metric_differences": {}, "per_example_differences": {},
                           "metrics_compared": []}
                    entries.append(out)
                    rel = trained.get("persisted_relative_path")
                    path = (job_dir / rel) if rel else None
                    out["checkpoint"] = str(path) if path else None
                    if path is None or not path.is_file():
                        out["errors"].append("checkpoint file missing")
                        continue
                    out["file_sha256_ok"] = (path.stat().st_size == trained.get("serialized_bytes")
                                             and sha256_file(path) == trained.get("sha256"))
                    if not out["file_sha256_ok"]:
                        out["errors"].append("checkpoint file size/SHA-256 differs from the artifact")
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    out["content_sha256_ok"] = _state_content_sha256(state) == trained.get("content_sha256")
                    if not out["content_sha256_ok"]:
                        out["errors"].append("checkpoint tensor content differs from the artifact")
                    forward_t = rebuild_forward(method, state, builder, num_tasks, cfg, device)
                    calibrated = record.get("kind", "none") != "none"
                    runs = [("", record.get("prefix_models") if calibrated else None)]
                    if calibrated:
                        runs.append(("raw_", None))
                    stored = None
                    pred = record.get("predictions")
                    if pred:
                        ppath = job_dir / pred.get("relative_path", "")
                        out["predictions"] = str(ppath)
                        if not ppath.is_file():
                            out["errors"].append("predictions file referenced by the "
                                                 "artifact is missing")
                        elif sha256_file(ppath) != pred.get("sha256"):
                            out["errors"].append("predictions file SHA-256 differs from "
                                                 "the artifact")
                        else:
                            with np.load(ppath, allow_pickle=False) as archive:
                                stored = {k: np.asarray(archive[k]) for k in archive.files}
                    for prefix, calibrators in runs:
                        rec = {} if stored is not None else None
                        res = evaluate_class_il_inference(
                            forward_t, test_loaders, num_tasks, cpt, device=device,
                            criterion=criterion, disjoint_classes=disjoint,
                            calibrators=calibrators, record=rec)
                        for name, suffix in MATRICES:
                            npz_key = (f"{key}{suffix}" if not prefix else
                                       f"{key}/raw_{suffix.lstrip('/') or 'result'}")
                            if npz_key not in npz.files:
                                out["errors"].append(f"{npz_key}: missing in result.npz")
                                continue
                            _compare_matrix(npz_key, res[name], npz[npz_key], atol, out)
                        metrics = payload["per_seed"][benchmark][method][payload["seeds"].index(seed)]
                        for json_name, stat in METRICS:
                            if prefix and json_name.startswith(("Oracle", "MicroOracle")):
                                continue            # one oracle per checkpoint
                            name = ("Raw" if prefix else "") + json_name
                            if name in metrics:
                                _compare_metric(name, res[stat], metrics[name], atol, out)
                        # ACC/LA/BWT/Forgetting with the runner's own function.
                        summary = summarize_run(np.asarray(res["result_matrix"]))
                        for json_name in ("ACC", "LA", "BWT", "Forgetting"):
                            name = ("Raw" if prefix else "") + json_name
                            if name in metrics:
                                _compare_metric(name, summary[json_name], metrics[name], atol, out)
                        if prefix == "":
                            out["oracle_acc"] = float(res["oracle_route_acc"])
                            out["routed_acc"] = float(res["class_il_acc"])
                        if stored is not None:
                            _compare_predictions(rec, stored, prefix, atol, out)
                    out["metrics_not_compared"] = sorted(
                        set(payload["per_seed"][benchmark][method][payload["seeds"].index(seed)])
                        - set(out["metrics_compared"]))
                    if out["metrics_not_compared"]:
                        out["errors"].append("per-seed metrics not re-evaluated: "
                                             f"{out['metrics_not_compared']}")
                    del forward_t, state
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
    return entries


def job_error_entry(result_json: Path, exc: Exception) -> dict:
    """Report entry for a job whose re-evaluation raised (missing data, a
    malformed checkpoint, CUDA out of memory, ...): an error, so the run fails,
    while the entries of every other job are kept in the report."""
    return {"job": Path(result_json).parent.name, "benchmark": "?", "method": "?",
            "seed": "?", "router": None,
            "errors": [f"re-evaluation raised {type(exc).__name__}: {exc}"],
            "exact": False, "max_abs_matrix_difference": 0.0, "metric_differences": {},
            "per_example_differences": {}, "metrics_compared": [],
            "metrics_not_compared": []}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="Suite run directory (all class-IL jobs).")
    src.add_argument("--result", type=Path, action="append",
                     help="A job's result.json (repeatable).")
    src.add_argument("--combine", type=Path, nargs="+", metavar="REPORT",
                     help="Combine shard reports (written with --out) and check that "
                          "they cover every class-IL job of their run directories.")
    p.add_argument("--data-root", type=str, default=None,
                   help="Dataset directory (overrides $DILTCL_DATA_ROOT).")
    p.add_argument("--device", default=None, help="cuda (default when available) or cpu.")
    p.add_argument("--methods", default=",".join(ISOLATION))
    p.add_argument("--atol", type=float, default=0.0,
                   help="Accept differences up to this many percentage points "
                        "(default 0: exact). Use only on hardware other than the recorded one.")
    p.add_argument("--shard", default=None, metavar="K/N",
                   help="Process only every N-th job starting at K (0-based), so N "
                        "processes can share one GPU, e.g. --shard 0/4 ... --shard 3/4.")
    p.add_argument("--out", type=Path, default=None, help="Write a JSON report.")
    return p.parse_args(argv)


def _class_il_results(run_dir: Path) -> list[str]:
    """Every class-IL job artifact of a run directory (torch-free)."""
    out = []
    for path in sorted(run_dir.glob("artifacts/*/result.json")):
        try:
            protocol = json.loads(path.read_text(encoding="utf-8")).get("protocol")
        except (OSError, ValueError):
            protocol = None
        if protocol == "class_il":
            out.append(str(path.resolve()))
    return out


def combine(paths: list[Path], out_path: Path | None) -> int:
    """Merge shard reports; fail unless they cover their run directories exactly."""
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    problems: list[str] = []
    by_run: dict[str, list[dict]] = {}
    for path, report in zip(paths, reports):
        run_dir = report.get("run_dir")
        if not run_dir:
            problems.append(f"{path}: report has no run_dir (written with --result?)")
            continue
        by_run.setdefault(run_dir, []).append(report)
    for run_dir, shard_reports in sorted(by_run.items()):
        specs = [r.get("shard") or "0/1" for r in shard_reports]
        sizes = {int(s.split("/")[1]) for s in specs}
        indices = sorted(int(s.split("/")[0]) for s in specs)
        if len(sizes) != 1 or indices != list(range(next(iter(sizes)))):
            problems.append(f"{run_dir}: shards {sorted(specs)} do not form a complete 0..N-1 set")
        covered = [j for r in shard_reports for j in r.get("results", [])]
        duplicates = sorted({j for j in covered if covered.count(j) > 1})
        if duplicates:
            problems.append(f"{run_dir}: {len(duplicates)} job(s) evaluated twice")
        # Each shard records the full class-IL job list it was cut from, so
        # coverage can be checked where the run directory is not available.
        listed = {tuple(r.get("run_results") or ()) for r in shard_reports}
        if len(listed) != 1 or not next(iter(listed)):
            problems.append(f"{run_dir}: shard reports disagree on (or lack) the job list")
            expected = []
        else:
            expected = sorted(next(iter(listed)))
        if Path(run_dir).is_dir() and expected and _class_il_results(Path(run_dir)) != expected:
            problems.append(f"{run_dir}: the run directory's class-IL jobs changed since "
                            "the shards were evaluated")
        missing = sorted(set(expected) - set(covered))
        if missing:
            problems.append(f"{run_dir}: {len(missing)} class-IL job(s) not covered by any shard")
        method_sets = {frozenset(r.get("methods") or ()) for r in shard_reports}
        if method_sets != {frozenset(ISOLATION)}:
            problems.append(f"{run_dir}: shards did not all evaluate every isolation method "
                            f"({sorted(sorted(m) for m in method_sets)})")
    entries = [e for r in reports for e in r.get("entries", [])]
    failures = [e for e in entries if e["errors"]]
    exact = all(e["exact"] for e in entries)
    status = ("failed" if failures or problems else
              "no-checkpoints" if not entries else
              "identical" if exact else "within-tolerance")
    combined = {"combined_from": [str(p) for p in paths], "coverage_problems": problems,
                "checkpoints_evaluated": len(entries), "with_errors": len(failures),
                "exact": exact if entries else None, "status": status,
                "per_run": {run: sum(r.get("checkpoints_evaluated", 0) for r in rs)
                            for run, rs in sorted(by_run.items())}}
    if out_path:
        out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    for problem in problems:
        print(f"COVERAGE: {problem}")
    for run, count in combined["per_run"].items():
        print(f"{run}: {count} checkpoint(s)")
    print(f"{len(entries)} checkpoint(s) re-evaluated in {len(reports)} report(s), "
          f"{len(failures)} with errors -> {status}")
    return 0 if status in ("identical", "within-tolerance") else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.combine:
        return combine(args.combine, args.out)
    if args.data_root:
        os.environ["DILTCL_DATA_ROOT"] = args.data_root
    os.environ.setdefault("DILTCL_DOWNLOAD", "0")
    # Every suite job ran with this cuBLAS workspace setting (run_suite.py);
    # it must be in place before CUDA initialises to select the same kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    methods = tuple(m for m in args.methods.split(",") if m)
    run_results = _class_il_results(args.run_dir) if args.run_dir else None
    results = ([Path(p) for p in run_results] if args.run_dir else list(args.result))
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= k < n:
            raise SystemExit("--shard must be K/N with 0 <= K < N")
        results = results[k::n]
    entries = []
    for path in results:
        try:
            entries += evaluate_job(path, device=device, methods=methods, atol=args.atol)
        except Exception as exc:  # noqa: BLE001 - one job must not discard the others
            entries.append(job_error_entry(path, exc))
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    failures = [e for e in entries if e["errors"]]
    exact = all(e["exact"] for e in entries)
    status = ("failed" if failures else
              "no-checkpoints" if not entries else
              "identical" if exact else "within-tolerance")
    report = {"device": device, "torch": torch.__version__, "atol": args.atol,
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "run_dir": str(args.run_dir.resolve()) if args.run_dir else None,
              "shard": args.shard or ("0/1" if args.run_dir else None),
              "methods": list(methods),
              "results": [str(Path(p).resolve()) for p in results],
              "run_results": run_results,
              "checkpoints_evaluated": len(entries), "with_errors": len(failures),
              "exact": exact if entries else None, "status": status, "entries": entries}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for e in entries:
        flag = "IDENTICAL" if e["exact"] and not e["errors"] else (
            "ERRORS: " + "; ".join(e["errors"][:3]) if e["errors"] else
            f"within tolerance (max |d|={e['max_abs_matrix_difference']:.3g})")
        print(f"{e['benchmark']:13s} {e['method']:8s} seed{e['seed']} [{e['router']}]: "
              f"oracle={e.get('oracle_acc', float('nan')):6.2f} "
              f"routed={e.get('routed_acc', float('nan')):6.2f} {flag}")
    print(f"{len(entries)} checkpoint(s) re-evaluated, {len(failures)} with errors -> {status}")
    return 0 if status in ("identical", "within-tolerance") else 1


if __name__ == "__main__":
    raise SystemExit(main())
