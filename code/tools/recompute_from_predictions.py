#!/usr/bin/env python3
"""Recompute class-IL routing statistics from per-example predictions (CPU).

Since v3.0.5 every isolation-method class-IL job writes
``predictions/<benchmark>/<method>/seed<N>.npz``: for each test example its
true task and label, the prediction of its own (oracle) expert, the routing
score of every expert, and the task chosen and class predicted at every
prefix stage; calibrated jobs add the matched raw router's choices (``raw_*``)
and the held-out training scores the router was fitted on
(``calibration_features`` / ``calibration_task_ids``).

This tool needs only NumPy (SciPy and, for the logistic router,
scikit-learn, when re-fitting calibrators). For every predictions file it

1. rebuilds the prefix result matrix, task-inference matrix, tie matrix and
   conditional-accuracy matrix, plus the pooled final-stage statistics, and
   compares them with the job's ``result.npz`` / ``result.json``;
2. reports the final-stage decomposition: routed accuracy ``A``, task
   inference ``U``, oracle accuracy ``O``, conditional accuracy ``V`` (with
   ``A = U * V`` exactly) and the recoverable-error fraction
   ``rho = (O - A) / (1 - A)``;
3. for calibrated jobs, re-fits the prefix routers from the stored
   calibration inputs and reports how often the re-fitted routers choose the
   same task as the recorded ones.

It also checks that every raw-router decision is the lowest-ID argmin of
the recorded scores (exact), and re-applies the job's stored calibrated
routers to the recorded scores (reported agreement).

Check 1 and the raw-router check must hold exactly (the same integer counts
produce the same floating-point percentages, and the same float32 scores the
same argmin); a mismatch makes the exit status non-zero. The calibrated-router
agreements are reported, not enforced: the GPU evaluated those linear
decisions in float32 and solver versions can move re-fitted coefficients, so
decisions that sit on near-ties may differ.

Examples::

    python tools/recompute_from_predictions.py --run-dir RUN
    python tools/recompute_from_predictions.py --result RUN/artifacts/JOB/result.json --refit
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ISOLATION = ("packnet", "supsup", "wsn", "diltcl")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files
                  if key != "meta_json"}
        meta = json.loads(str(archive["meta_json"])) if "meta_json" in archive.files else {}
    return arrays, meta


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Recomputation (mirrors diltcl.task_inference.evaluate_class_il_inference)
# ---------------------------------------------------------------------------

def recompute(arrays: dict[str, np.ndarray], num_tasks: int,
              disjoint: bool = True, prefix: str = "") -> dict:
    """Rebuild the evaluator's matrices and pooled statistics.

    ``prefix`` selects the router: ``""`` for the job's own router (raw or
    calibrated) and ``"raw_"`` for the matched raw router of a calibrated job.
    The arithmetic follows the evaluator line by line (integer counts, then
    ``100.0 * hits / count``), so matching runs agree exactly.
    """
    true_task = arrays[f"{prefix}true_task"].astype(np.int64)
    label = arrays[f"{prefix}label_local"].astype(np.int64)
    oracle = arrays[f"{prefix}oracle_pred_local"].astype(np.int64)
    routes = arrays[f"{prefix}route_by_stage"].astype(np.int64)
    ties = arrays[f"{prefix}tie_by_stage"].astype(bool)
    preds = arrays[f"{prefix}routed_pred_local_by_stage"].astype(np.int64)
    T = int(num_tasks)
    if routes.shape[1] != T:
        raise ValueError(f"route matrix has {routes.shape[1]} stages, expected {T}")

    counts = np.zeros(T, dtype=np.int64)
    oracle_hits = np.zeros(T, dtype=np.int64)
    cls_hits = np.zeros((T, T), dtype=np.int64)
    ta_hits = np.zeros((T, T), dtype=np.int64)
    tie_hits = np.zeros((T, T), dtype=np.int64)
    crc_hits = np.zeros((T, T), dtype=np.int64)
    for t in range(T):
        rows = true_task == t
        counts[t] = int(rows.sum())
        oracle_hits[t] = int(np.sum(oracle[rows] == label[rows]))
        for stage in range(t, T):
            r = routes[rows, stage]
            p = preds[rows, stage]
            if disjoint:
                correct = (r == t) & (p == label[rows])
            else:
                correct = p == label[rows]
            cls_hits[stage, t] = int(correct.sum())
            ta_hits[stage, t] = int(np.sum(r == t))
            tie_hits[stage, t] = int(np.sum(ties[rows, stage]))
            crc_hits[stage, t] = int(np.sum(correct & (r == t)))
        before = routes[rows, :t]
        if before.size and np.any(before != -1):
            raise ValueError(f"task {t}: routes recorded before the task arrived")

    result = np.zeros((T, T)); ta_m = np.zeros((T, T))
    tie_m = np.zeros((T, T)); cond_m = np.zeros((T, T))
    oracle_per_task = np.zeros(T)
    for t in range(T):
        if counts[t] == 0:
            continue
        oracle_per_task[t] = 100.0 * oracle_hits[t] / counts[t]
        for stage in range(t, T):
            result[stage, t] = 100.0 * cls_hits[stage, t] / counts[t]
            ta_m[stage, t] = 100.0 * ta_hits[stage, t] / counts[t]
            tie_m[stage, t] = 100.0 * tie_hits[stage, t] / counts[t]
            if ta_hits[stage, t] > 0:
                cond_m[stage, t] = 100.0 * crc_hits[stage, t] / ta_hits[stage, t]
    nonempty = counts > 0
    macro = lambda a: float(np.mean(np.asarray(a)[nonempty])) if nonempty.any() else 0.0  # noqa: E731
    total = int(counts.sum())
    final_routes = ta_hits[-1]
    stats = {
        "class_il_acc": macro(result[-1]),
        "micro_class_il_acc": 100.0 * cls_hits[-1].sum() / total if total else 0.0,
        "oracle_route_acc": macro(oracle_per_task),
        "micro_oracle_route_acc": 100.0 * oracle_hits.sum() / total if total else 0.0,
        "task_inference_acc": macro(ta_m[-1]),
        "micro_task_inference_acc": 100.0 * ta_hits[-1].sum() / total if total else 0.0,
        "routing_tie_rate": macro(tie_m[-1]),
        "micro_routing_tie_rate": 100.0 * tie_hits[-1].sum() / total if total else 0.0,
        "conditional_class_acc": (100.0 * crc_hits[-1].sum() / final_routes.sum()
                                  if final_routes.sum() else 0.0),
        "zero_correct_route_task_count": int(np.sum(nonempty & (final_routes == 0))),
    }
    return {"result_matrix": result, "task_inference_matrix": ta_m,
            "routing_tie_matrix": tie_m, "conditional_class_matrix": cond_m,
            "stats": stats, "examples": total}


def raw_route_errors(arrays: dict[str, np.ndarray], prefix: str = "") -> list[str]:
    """Exact check: raw routes are the lowest-task-ID argmin of the scores.

    Applies to the uncalibrated router (``prefix=""`` of an uncalibrated job,
    ``prefix="raw_"`` of a calibrated one). At stage ``s`` the router compared
    the recorded float32 scores of candidates ``0..s``, so the argmin over the
    same values must reproduce every decision and every tie flag.
    """
    errors = []
    scores = arrays[f"{prefix}scores"]
    routes = arrays[f"{prefix}route_by_stage"]
    ties = arrays[f"{prefix}tie_by_stage"].astype(bool)
    true_task = arrays[f"{prefix}true_task"]
    for stage in range(routes.shape[1]):
        rows = true_task <= stage
        if not rows.any():
            continue
        cand = scores[rows, :stage + 1]
        winning = cand == cand.min(axis=1, keepdims=True)
        expected_route = winning.argmax(axis=1)
        expected_tie = winning.sum(axis=1) > 1
        bad_route = int(np.sum(expected_route != routes[rows, stage]))
        bad_tie = int(np.sum(expected_tie != ties[rows, stage]))
        if bad_route or bad_tie:
            errors.append(f"{prefix or 'router'} stage {stage}: {bad_route} route(s) and "
                          f"{bad_tie} tie flag(s) differ from the argmin of the recorded scores")
    return errors


def stored_router_agreement(arrays: dict[str, np.ndarray], record: dict) -> dict:
    """Re-apply the job's serialized prefix routers to the recorded scores."""
    models = record.get("prefix_models") or []
    scores = arrays["scores"].astype(np.float32)
    routes = arrays["route_by_stage"]
    agree = total = 0
    for stage, state in enumerate(models):
        rows = arrays["true_task"] <= stage
        if not rows.any():
            continue
        # The evaluator computed these decisions on-device in the scores'
        # dtype (float32); mirror that precision here.
        x = scores[rows, :stage + 1]
        mean = np.asarray(state["feature_mean"], dtype=np.float32)
        scale = np.asarray(state["feature_scale"], dtype=np.float32)
        coef = np.asarray(state["coef"], dtype=np.float32)
        intercept = np.asarray(state["intercept"], dtype=np.float32)
        classes = np.asarray(state["classes"], dtype=np.int64)
        decision = ((x - mean) / scale) @ coef.T + intercept
        winning = decision == decision.max(axis=1, keepdims=True)
        pred = np.where(winning, classes[None, :], np.iinfo(np.int64).max).min(axis=1)
        agree += int(np.sum(pred == routes[rows, stage])); total += int(rows.sum())
    return {"route_agreement": agree / total if total else float("nan"),
            "decisions_compared": total}


def decomposition(stats: dict) -> dict:
    """Final-stage error partition in percentage units (pooled over examples)."""
    A = stats["micro_class_il_acc"]; U = stats["micro_task_inference_acc"]
    O = stats["micro_oracle_route_acc"]; V = stats["conditional_class_acc"]
    return {
        "routed_accuracy": A, "task_inference": U, "oracle_accuracy": O,
        "conditional_accuracy": V,
        "identity_residual_A_minus_UV": A - U * V / 100.0,
        "oracle_error": 100.0 - O,
        "recoverable_routing_loss": O - A,
        "recoverable_fraction_percent": (100.0 * (O - A) / (100.0 - A)) if A < 100.0 else float("nan"),
    }


# Map the evaluator's statistics to the JSON metric names written per seed.
_METRIC_NAMES = (
    ("micro_class_il_acc", "MicroACC"), ("oracle_route_acc", "OracleACC"),
    ("micro_oracle_route_acc", "MicroOracleACC"), ("task_inference_acc", "TA"),
    ("micro_task_inference_acc", "MicroTA"), ("conditional_class_acc", "ConditionalACC"),
    ("conditional_class_acc", "MicroConditionalACC"),
    ("routing_tie_rate", "RouteTieRate"), ("micro_routing_tie_rate", "MicroRouteTieRate"),
    ("zero_correct_route_task_count", "ZeroCorrectRouteTasks"),
)


def compare_with_artifact(arrays: dict, meta: dict, payload: dict, npz,
                          benchmark: str, method: str, seed: int) -> list[str]:
    """Exact comparison of recomputed statistics with the job's artifacts."""
    errors: list[str] = []
    T = int(meta["num_tasks"]); disjoint = bool(meta.get("disjoint_classes", True))
    seeds = payload["seeds"]
    metrics = payload["per_seed"][benchmark][method][seeds.index(seed)]
    key = f"{benchmark}/{method}/seed{seed}"
    pairs = [("", "", {"result_matrix": key, "task_inference_matrix": f"{key}/task_inference",
                       "conditional_class_matrix": f"{key}/conditional_class",
                       "routing_tie_matrix": f"{key}/routing_tie"})]
    if "raw_true_task" in arrays:
        pairs.append(("raw_", "Raw", {"result_matrix": f"{key}/raw_result",
                                      "task_inference_matrix": f"{key}/raw_task_inference",
                                      "conditional_class_matrix": f"{key}/raw_conditional_class",
                                      "routing_tie_matrix": f"{key}/raw_routing_tie"}))
    calibrated = meta.get("calibration", "none") not in (None, "none")
    raw_prefix = "raw_" if calibrated else ""
    if f"{raw_prefix}scores" in arrays:
        errors += raw_route_errors(arrays, raw_prefix)
    for prefix, metric_prefix, npz_keys in pairs:
        rec = recompute(arrays, T, disjoint, prefix)
        for name, npz_key in npz_keys.items():
            if npz_key not in npz.files:
                errors.append(f"{npz_key}: missing in result.npz")
                continue
            if not np.array_equal(np.asarray(npz[npz_key], dtype=float), rec[name]):
                diff = float(np.max(np.abs(np.asarray(npz[npz_key], dtype=float) - rec[name])))
                errors.append(f"{npz_key}: recomputed matrix differs (max abs {diff:.3g})")
        for stat, name in _METRIC_NAMES:
            json_name = metric_prefix + name
            if prefix == "raw_" and name in ("OracleACC", "MicroOracleACC"):
                continue          # raw and calibrated share the oracle
            if json_name not in metrics:
                continue
            if float(metrics[json_name]) != float(rec["stats"][stat]):
                errors.append(f"{key}: {json_name} JSON={metrics[json_name]} "
                              f"recomputed={rec['stats'][stat]}")
    return errors


# ---------------------------------------------------------------------------
# Router re-fit (optional)
# ---------------------------------------------------------------------------

def _task_calibration():
    spec = importlib.util.spec_from_file_location(
        "diltcl_task_calibration_standalone", ROOT / "diltcl" / "task_calibration.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def refit_agreement(arrays: dict, meta: dict, record: dict) -> dict:
    """Re-fit the prefix routers and compare their test routes with the record."""
    tc = _task_calibration()
    T = int(meta["num_tasks"])
    states = tc.fit_prefix_calibrators(
        arrays["calibration_features"].astype(np.float64),
        arrays["calibration_task_ids"].astype(np.int64), T,
        C=float(record.get("C") or 1.0), random_state=int(record.get("split_seed", 0)),
        kind=record["kind"])
    scores = arrays["scores"].astype(np.float64)
    routes = arrays["route_by_stage"]
    agree = total = 0
    max_coef_diff = 0.0
    for stage, (state, stored) in enumerate(zip(states, record.get("prefix_models", []))):
        max_coef_diff = max(max_coef_diff, float(np.max(np.abs(
            np.asarray(state["coef"]) - np.asarray(stored["coef"])))))
        rows = arrays["true_task"] <= stage
        pred = tc.predict_score_logreg(scores[rows, :stage + 1], state)
        agree += int(np.sum(pred == routes[rows, stage])); total += int(rows.sum())
    return {"route_agreement": agree / total if total else float("nan"),
            "decisions_compared": total, "max_abs_coef_difference": max_coef_diff,
            "solver_backend": states[-1].get("solver_backend")}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def check_job(result_json: Path, refit: bool) -> list[dict]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    if payload.get("protocol") != "class_il":
        return []
    out = []
    with np.load(result_json.with_suffix(".npz"), allow_pickle=False) as npz:
        for benchmark, by_method in (payload.get("calibration") or {}).items():
            for method, by_seed in by_method.items():
                for seed_text, record in by_seed.items():
                    pred = (record or {}).get("predictions")
                    if not pred:
                        continue
                    path = result_json.parent / pred.get("relative_path", "")
                    entry = {"job": str(result_json.parent.name), "benchmark": benchmark,
                             "method": method, "seed": int(seed_text), "file": str(path),
                             "errors": []}
                    if not path.is_file():
                        entry["errors"].append("predictions file missing")
                        out.append(entry); continue
                    if sha256_file(path) != pred.get("sha256"):
                        entry["errors"].append("predictions file SHA-256 differs from record")
                    arrays, meta = load_predictions(path)
                    if meta.get("trained_state_content_sha256") != record["trained_state"]["content_sha256"]:
                        entry["errors"].append("predictions belong to a different trained state")
                    entry["errors"] += compare_with_artifact(
                        arrays, meta, payload, npz, benchmark, method, int(seed_text))
                    rec = recompute(arrays, int(meta["num_tasks"]), bool(meta.get("disjoint_classes", True)))
                    entry["decomposition"] = decomposition(rec["stats"])
                    if record.get("kind") not in (None, "none") and record.get("prefix_models"):
                        entry["stored_router"] = stored_router_agreement(arrays, record)
                    if refit and "calibration_features" in arrays and record.get("kind") not in (None, "none"):
                        try:
                            entry["refit"] = refit_agreement(arrays, meta, record)
                        except Exception as exc:  # noqa: BLE001 - reported, not fatal
                            entry["refit"] = {"error": f"{type(exc).__name__}: {exc}"}
                    out.append(entry)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="Suite run directory (all class-IL jobs).")
    src.add_argument("--result", type=Path, action="append",
                     help="A job's result.json (repeatable).")
    p.add_argument("--refit", action="store_true",
                   help="Also re-fit calibrated routers from the stored calibration inputs.")
    p.add_argument("--out", type=Path, default=None, help="Write a JSON report.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results = (sorted(args.run_dir.glob("artifacts/*/result.json")) if args.run_dir
               else list(args.result))
    entries = []
    for path in results:
        entries += check_job(path, args.refit)
    failures = [e for e in entries if e["errors"]]
    report = {"files_checked": len(entries), "files_with_errors": len(failures),
              "status": "passed" if entries and not failures else ("failed" if failures else "no-predictions"),
              "entries": entries}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for e in entries:
        d = e.get("decomposition", {})
        refit = e.get("refit", {})
        stored = e.get("stored_router", {})
        print(f"{e['benchmark']:13s} {e['method']:8s} seed{e['seed']}: "
              f"A={d.get('routed_accuracy', float('nan')):6.2f} U={d.get('task_inference', float('nan')):6.2f} "
              f"O={d.get('oracle_accuracy', float('nan')):6.2f} rho={d.get('recoverable_fraction_percent', float('nan')):5.1f}% "
              + (f"stored-router-agreement={stored['route_agreement']:.4f} " if 'route_agreement' in stored else "")
              + (f"refit-agreement={refit['route_agreement']:.4f} " if 'route_agreement' in refit else "")
              + ("OK" if not e["errors"] else "ERRORS: " + "; ".join(e["errors"][:3])))
    print(f"{report['files_checked']} predictions file(s) checked, {report['files_with_errors']} with errors -> {report['status']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
