#!/usr/bin/env python3
"""What changed when calibrated routers moved from float32 to float64 (v3.0.10).

Up to v3.0.9 a fitted router (score-logreg, score-bias, score-bias-acc) was
fitted in float64 but applied on the GPU in the float32 of the routing
scores. Biases that differ only below float32 resolution then collapse, and
an exact tie falls back to the lowest task ID, so the deployed router was not
always the fitted one. Since v3.0.10 every router is applied in float64
(``routing_dtype`` in the stored state), which is the fitted router exactly
for the two bias routers.

For every calibrated class-IL isolation job of the given run directories this
tool first checks the job's per-example records exactly as the run verifier
does (``compare_with_artifact`` of ``recompute_from_predictions.py``: their
structure, the raw router's argmin decisions, every matrix and metric of the
job's own artifacts, the stored routers and their fitting records), and then

1. re-applies every stored router as v3.0.9 did, in float32, and counts the
   test decisions that the change of precision moved, stage by stage;
2. rebuilds from those float32 decisions the four calibrated matrices
   (result, task inference, conditional accuracy, ties) and the routing
   metrics the job would have reported with float32 routing. With disjoint
   label sets a moved decision is correct only if it now selects the
   example's own task, whose prediction is the recorded oracle prediction.
   With shared label sets a moved decision's prediction is known only when it
   moved to the example's own task or to the task the matched raw router
   chose; a job with any other moved decision cannot be rebuilt;
3. with ``--reference``, compares the job with a float32-routed reference run
   (for example the merged artifacts behind the submitted paper). Everything
   the routing precision cannot change must be present in the reference and
   identical: the per-task test-set sizes, the trained state (content hash),
   the stored routers (every parameter, stage by stage), the matched raw
   router's four matrices and metrics (``raw_*``, ``Raw*``) recomputed from
   this job's records, and the oracle accuracies (``OracleACC``,
   ``MicroOracleACC``) -- together they show that the same experts produced
   the same scores and fitted the same routers. The reference's calibrated
   metrics must follow exactly from its own calibrated matrices, and the
   float32 rebuild must reproduce those matrices and metrics. score-logreg
   decisions are full matrix products whose float32 rebuild here uses
   NumPy's summation order, not the GPU's: a decision with another task
   within float32 rounding of its winner may come out as any of those tasks,
   and only as one of them. Such a decision can move a cell's counts only
   one way: it can lose a correct route only if its winner here is the
   example's own task, and gain one only if the own task is among those
   tasks without being the winner; the correct predictions among correct
   routes move with those routes whose own expert is right, the correct
   predictions with them (with shared label sets: by any near-tie
   decision); and a tie can disappear only where the rebuild ties and
   appear only on other near-tie decisions. A cell whose integer counts
   differ from the reference's within these bounds is reported as ``within
   float32 near-ties``; anything else is a difference. The bias routers are
   rebuilt exactly and get no such tolerance, unless the reference ran its
   matmuls in TF32 (then every router gets it, with TF32's rounding bound).

The audit must cover the reference: every calibrated run of every
calibrated reference group must be audited, every isolation method of an
audited job must carry its calibration record, and every audited group must
be in the reference.

Needs NumPy and PyTorch (for the runner's own matrix summary); runs on a CPU
in seconds per job.

Exit status 0 with status ``measured`` (no ``--reference``) or ``explained``
(every audited job compared and reproduced, up to the near-ties above), and 1
otherwise: ``failed`` when a job's records fail their checks, a run directory
holds no calibrated class-IL job, an audited group is missing from the
reference or a calibrated reference run is not audited; ``unexplained`` when
a job differs from the reference or cannot be rebuilt.

Example::

    python tools/routing_precision_audit.py --run-dir NEW/p0 NEW/p1_bias NEW/p1_biasacc \\
        --reference OLD/p0_v3 OLD/p1_bias OLD/p1_biasacc --out routing_precision.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rp = _load("diltcl_predictions_for_precision", ROOT / "tools" / "recompute_from_predictions.py")
cr = _load("diltcl_compare_for_precision", ROOT / "tools" / "compare_runs.py")

# The four matrices of the job's own (calibrated) router and of the matched
# raw router, as stored in result.npz under "<benchmark>/<method>/seed<N>".
MATRICES = tuple(zip(("result_matrix", "task_inference_matrix", "conditional_class_matrix",
                      "routing_tie_matrix"), cr.ROUTING_ARRAY_SUFFIXES))
RAW_MATRICES = (("result_matrix", "/raw_result"),
                ("task_inference_matrix", "/raw_task_inference"),
                ("conditional_class_matrix", "/raw_conditional_class"),
                ("routing_tie_matrix", "/raw_routing_tie"))
# Per-seed metrics computed from the calibrated decisions, and those the
# routing precision cannot change (oracle accuracies, the raw router's).
ROUTED_METRICS = cr.ROUTING_METRICS
FIXED_METRICS = ("OracleACC", "MicroOracleACC") + tuple("Raw" + m for m in ROUTED_METRICS)


def float32_rebuild(arrays: dict, record: dict, tf32: bool = False) -> dict:
    """The job's calibrated decisions re-applied in float32, as before v3.0.10.

    Returns per-example arrays in the recorder's layout (routes, tie flags,
    routed predictions) plus counts: decisions compared, decisions whose
    route or tie flag moved, and decisions with another task within float32
    rounding of the winner, which the GPU may have resolved as any of those
    tasks (score-logreg, whose decisions are full matrix products; every
    router when the reference ran its matmuls in TF32, ``tf32``).
    ``near_tie_mask`` marks those decisions; ``near_home_mask`` the ones whose
    winner here is the example's own task (another order can only lose that
    route) and ``near_away_mask`` the ones whose own task is within rounding
    but not the winner (another order can only gain it). A moved decision's
    prediction is the oracle prediction when it moved to the example's own
    task and the raw router's when it moved to the task the raw router
    chose; any other moved decision is marked in ``unknown_prediction_mask``
    (irrelevant with disjoint label sets, where such a decision is wrong
    whatever its prediction).
    """
    routes = arrays["route_by_stage"].astype(np.int64)
    ties = arrays["tie_by_stage"].astype(bool)
    preds = arrays["routed_pred_local_by_stage"].astype(np.int64)
    true_task = arrays["true_task"].astype(np.int64)
    oracle = arrays["oracle_pred_local"].astype(np.int64)
    scores = arrays["scores"].astype(np.float32)
    raw_routes = arrays.get("raw_route_by_stage")
    raw_preds = arrays.get("raw_routed_pred_local_by_stage")
    new_routes, new_ties, new_preds = routes.copy(), ties.copy(), preds.copy()
    near_tie = np.zeros(routes.shape, dtype=bool)
    near_home = np.zeros(routes.shape, dtype=bool)
    near_away = np.zeros(routes.shape, dtype=bool)
    unknown = np.zeros(routes.shape, dtype=bool)
    kind = record.get("kind")
    precision = "tf32" if tf32 else "float32"
    exact = kind in rp.EXACT_ROUTERS and not tf32
    compared = moved = tie_moved = 0
    by_stage = []
    for stage, state in enumerate(record.get("prefix_models") or []):
        rows = np.flatnonzero(true_task <= stage)
        if not rows.size:
            by_stage.append(0)
            continue
        legacy = {k: v for k, v in state.items() if k != "routing_dtype"}
        applied = rp.apply_router(scores[rows, :stage + 1], legacy, precision)
        pred32 = applied["pred"]
        changed = pred32 != routes[rows, stage]
        compared += rows.size
        moved += int(changed.sum())
        tie_moved += int(np.sum(applied["tie"] != ties[rows, stage]))
        by_stage.append(int(changed.sum()))
        if not exact and stage > 0:
            within = rp.within_rounding_of_winner(applied)
            near = within.sum(axis=1) > 1
            own = within[np.arange(rows.size), rp._columns(applied, true_task[rows])]
            home = pred32 == true_task[rows]
            near_tie[rows, stage] = near
            near_home[rows, stage] = near & home
            near_away[rows, stage] = near & own & ~home
        new_routes[rows, stage] = pred32
        new_ties[rows, stage] = applied["tie"]
        moved_rows = rows[changed]
        to = pred32[changed]
        home = to == true_task[moved_rows]
        new_preds[moved_rows[home], stage] = oracle[moved_rows[home]]
        as_raw = np.zeros(moved_rows.size, dtype=bool)
        if raw_routes is not None and raw_preds is not None:
            as_raw = ~home & (to == raw_routes[moved_rows, stage])
            new_preds[moved_rows[as_raw], stage] = raw_preds[moved_rows[as_raw], stage]
        unknown[moved_rows[~home & ~as_raw], stage] = True
    rebuilt = dict(arrays)
    rebuilt.update({"route_by_stage": new_routes, "tie_by_stage": new_ties,
                    "routed_pred_local_by_stage": new_preds})
    return {"arrays": rebuilt, "decisions_compared": compared, "routes_moved": moved,
            "tie_flags_moved": tie_moved, "routes_moved_by_stage": by_stage,
            "float32_near_tie_decisions": int(near_tie.sum()), "precision": precision,
            "near_tie_mask": near_tie, "near_home_mask": near_home,
            "near_away_mask": near_away, "unknown_prediction_mask": unknown}


def per_seed_metrics(rebuilt_stats: dict, matrix: np.ndarray) -> dict:
    """Per-seed metric names of the runner, from a rebuilt evaluation."""
    from diltcl.metrics import summarize_run   # the runner's own function
    out = dict(summarize_run(np.asarray(matrix)))
    for stat, name in rp._METRIC_NAMES:
        out[name] = rebuilt_stats[stat]
    return out


def fixed_metrics(current: dict, raw: dict) -> dict:
    """The metrics the routing precision cannot change, from this job's records."""
    out = {"OracleACC": current["stats"]["oracle_route_acc"],
           "MicroOracleACC": current["stats"]["micro_oracle_route_acc"]}
    raw_metrics = per_seed_metrics(raw["stats"], raw["result_matrix"])
    out.update({"Raw" + name: raw_metrics[name] for name in ROUTED_METRICS})
    return out


def near_tie_cells(rebuild: dict, arrays: dict, T: int) -> dict:
    """Per (stage, task) cell: by how much near-tie decisions can move each count, and which way.

    A decision with other tasks within float32 rounding of its winner can be
    resolved as any of them in another summation order (and only as one of
    them). Counting the reference minus this rebuild:

    * correct routes can fall only by decisions whose winner here is the
      example's own task (``routes_lost``) and rise only by decisions whose
      own task is within rounding but not the winner (``routes_gained``),
      split by whether the own expert's (oracle) prediction is right
      (``_right``) or wrong (``_wrong``) -- correct predictions among correct
      routes move with the right ones;
    * ties can disappear only on decisions tied here (``ties_lost``) and
      appear only on near-tie decisions not tied here (``ties_gained``);
    * ``near`` counts all near-tie decisions (correct predictions with
      shared label sets, where any expert may predict the right label).
    """
    true_task = arrays["true_task"].astype(np.int64)
    right = (arrays["oracle_pred_local"].astype(np.int64)
             == arrays["label_local"].astype(np.int64))[:, None]
    near, home, away = rebuild["near_tie_mask"], rebuild["near_home_mask"], rebuild["near_away_mask"]
    tied = rebuild["arrays"]["tie_by_stage"].astype(bool)

    def per_cell(mask):
        cells = np.zeros((T, T), dtype=np.int64)
        for t in range(T):
            cells[:, t] = mask[true_task == t].sum(axis=0)
        return cells

    return {"routes_lost_right": per_cell(home & right), "routes_gained_right": per_cell(away & right),
            "routes_lost_wrong": per_cell(home & ~right), "routes_gained_wrong": per_cell(away & ~right),
            "ties_lost": per_cell(near & tied), "ties_gained": per_cell(near & ~tied),
            "near": per_cell(near)}


def within_near_ties(ref_counts: dict, new_counts: dict, cells: dict, disjoint: bool) -> dict:
    """Per matrix: the cells whose difference (reference minus rebuild) near-ties can explain."""
    def between(delta, lost, gained):
        return (-lost <= delta) & (delta <= gained)

    delta = {name: ref_counts[name] - new_counts[name] for name, _ in MATRICES}
    right = delta["conditional_class_matrix"]                 # correct routes, right expert
    wrong = delta["task_inference_matrix"] - right            # correct routes, wrong expert
    routes = (between(right, cells["routes_lost_right"], cells["routes_gained_right"])
              & between(wrong, cells["routes_lost_wrong"], cells["routes_gained_wrong"]))
    if disjoint:     # a correct prediction is a correct route with the right expert
        result = routes & (delta["result_matrix"] == right)
    else:
        result = np.abs(delta["result_matrix"]) <= cells["near"]
    return {"result_matrix": result, "task_inference_matrix": routes,
            "conditional_class_matrix": routes,
            "routing_tie_matrix": between(delta["routing_tie_matrix"], cells["ties_lost"],
                                          cells["ties_gained"])}


def matrix_counts(matrices: dict, counts: np.ndarray) -> tuple[dict, bool]:
    """Integer counts behind one router's four matrices, and whether they are exact.

    The evaluator stores ``100 * hits / n`` per (stage, task) cell, with ``n``
    the task's test examples (for the conditional matrix: the cell's correct
    routes), so ``hits = rint(value * n / 100)``. The flag is False when a
    value is not such a ratio, which no matrix of these test sets can hold.
    """
    exact = True

    def to_counts(values, denominators):
        nonlocal exact
        scaled = np.asarray(values, dtype=np.float64) * denominators / 100.0
        hits = np.rint(scaled)
        if not np.all(np.abs(scaled - hits) <= 1e-6 * np.maximum(1.0, np.abs(scaled))):
            exact = False
        return hits

    n = np.asarray(counts, dtype=np.float64)[None, :]
    routed = to_counts(matrices["task_inference_matrix"], n)
    return {"result_matrix": to_counts(matrices["result_matrix"], n),
            "task_inference_matrix": routed,
            "routing_tie_matrix": to_counts(matrices["routing_tie_matrix"], n),
            "conditional_class_matrix": to_counts(matrices["conditional_class_matrix"], routed),
            }, exact


def metrics_from_matrices(matrices: dict, counts_m: dict, counts: np.ndarray) -> dict:
    """The routed per-seed metrics that one router's four matrices imply.

    The evaluator's arithmetic (as in ``recompute``): macro averages of the
    matrices' last rows over the tasks with examples, pooled rates from the
    integer counts behind them, and the runner's matrix summary.
    """
    from diltcl.metrics import summarize_run
    result = np.asarray(matrices["result_matrix"], dtype=np.float64)
    n = np.asarray(counts)
    nonempty = n > 0
    total = int(n.sum())

    def macro(row):
        return float(np.mean(np.asarray(row)[nonempty])) if nonempty.any() else 0.0

    hits = counts_m["result_matrix"][-1]
    routes = counts_m["task_inference_matrix"][-1]
    ties = counts_m["routing_tie_matrix"][-1]
    conditional = counts_m["conditional_class_matrix"][-1]
    stats = {
        "class_il_acc": macro(result[-1]),
        "micro_class_il_acc": 100.0 * hits.sum() / total if total else 0.0,
        "task_inference_acc": macro(np.asarray(matrices["task_inference_matrix"])[-1]),
        "micro_task_inference_acc": 100.0 * routes.sum() / total if total else 0.0,
        "routing_tie_rate": macro(np.asarray(matrices["routing_tie_matrix"])[-1]),
        "micro_routing_tie_rate": 100.0 * ties.sum() / total if total else 0.0,
        "conditional_class_acc": (100.0 * conditional.sum() / routes.sum()
                                  if routes.sum() else 0.0),
        "zero_correct_route_task_count": int(np.sum(nonempty & (routes == 0))),
    }
    out = dict(summarize_run(result))
    for stat, name in rp._METRIC_NAMES:
        if stat in stats:
            out[name] = stats[stat]
    return {name: out[name] for name in ROUTED_METRICS}


def router_differences(models: list, ref_models: list) -> dict:
    """Stage by stage, every router parameter that differs from the reference's."""
    if len(models) != len(ref_models):
        return {"stages": [len(models), len(ref_models)]}
    out = {}
    for stage, (state, ref) in enumerate(zip(models, ref_models)):
        for name in ("classes",) + rp.ROUTER_PARAMETERS:
            if name not in state or name not in ref:
                out[f"stage{stage}/{name}"] = "missing"
                continue
            a = np.asarray(state[name], dtype=np.float64)
            b = np.asarray(ref[name], dtype=np.float64)
            if a.shape != b.shape:
                out[f"stage{stage}/{name}"] = "shape"
            elif not np.array_equal(a, b):
                out[f"stage{stage}/{name}"] = float(np.max(np.abs(a - b)))
    return out


def compare_with_reference(job: dict, ref_payload: dict, ref_npz) -> dict:
    """Compare one job (its float32 rebuild and its fixed parts) with the reference.

    ``job`` carries the benchmark, method, seed, the job's calibration record,
    per-task test counts, the rebuilt and raw recomputations, the rebuilt
    metrics, the fixed metrics and the near-tie cells (:func:`near_tie_cells`).
    """
    benchmark, method, seed = job["benchmark"], job["method"], job["seed"]
    key = f"{benchmark}/{method}/seed{seed}"
    counts = job["counts"]
    out = {"run": key, "test_counts_identical": None, "trained_state_identical": None,
           "routers_identical": None, "router_differences": {},
           "fixed_matrix_differences": {}, "fixed_metric_differences": {},
           "reference_inconsistent_metrics": {},
           "matrix_differences": {}, "within_near_ties": [], "metric_differences": {},
           "missing_in_reference": [], "unchecked_metrics": [], "identical": False}
    ref_counts = rp.expected_test_counts(ref_payload, benchmark, seed)
    if ref_counts is not None:
        out["test_counts_identical"] = [int(v) for v in ref_counts] == counts.tolist()
    ref_record = (((ref_payload.get("calibration") or {}).get(benchmark) or {})
                  .get(method) or {}).get(str(seed)) or {}
    mine = (job["record"].get("trained_state") or {}).get("content_sha256")
    theirs = ((ref_record.get("trained_state") or {}).get("content_sha256")
              or cr.state_hash(ref_payload, benchmark, method, seed))
    if mine and theirs:
        out["trained_state_identical"] = mine == theirs
    if ref_record.get("prefix_models"):
        out["router_differences"] = router_differences(job["record"].get("prefix_models") or [],
                                                       ref_record["prefix_models"])
        out["routers_identical"] = not out["router_differences"]
    for label, field in (("test-set sizes", "test_counts_identical"),
                         ("trained-state hash", "trained_state_identical"),
                         ("stored routers", "routers_identical")):
        if out[field] is None:
            out["missing_in_reference"].append(label)
    # The raw router: exactly the reference's.
    for name, suffix in RAW_MATRICES:
        npz_key = key + suffix
        if npz_key not in ref_npz.files:
            out["missing_in_reference"].append(npz_key)
            continue
        ref, new = np.asarray(ref_npz[npz_key], dtype=float), np.asarray(job["raw"][name], dtype=float)
        if ref.shape != new.shape:
            out["fixed_matrix_differences"][npz_key] = "shape"
        elif not np.array_equal(ref, new):
            out["fixed_matrix_differences"][npz_key] = float(np.max(np.abs(ref - new)))
    # The calibrated router: the float32 rebuild, exact up to near-ties.
    rebuilt, cells = job["rebuilt"], job["cells"]
    ref_matrices = {}
    for name, suffix in MATRICES:
        npz_key = key + suffix
        if npz_key not in ref_npz.files:
            out["missing_in_reference"].append(npz_key)
            continue
        ref_matrices[name] = np.asarray(ref_npz[npz_key], dtype=float)
        if ref_matrices[name].shape != np.asarray(rebuilt[name]).shape:
            out["matrix_differences"][npz_key] = "shape"
    ref_counts_m = None
    if len(ref_matrices) == len(MATRICES) and not out["matrix_differences"]:
        new_counts, new_exact = matrix_counts(rebuilt, counts)
        ref_counts_m, ref_exact = matrix_counts(ref_matrices, counts)
        explained = within_near_ties(ref_counts_m, new_counts, cells, job.get("disjoint", True))
        for name, suffix in MATRICES:
            ref, new = ref_matrices[name], np.asarray(rebuilt[name], dtype=float)
            if np.array_equal(ref, new):
                continue
            differ = ref != new
            ok = explained[name]
            if new_exact and ref_exact and bool(np.all(ok[differ])):
                out["within_near_ties"].append(key + suffix)
            else:
                out["matrix_differences"][key + suffix] = float(np.max(np.abs(ref - new)))
        if not ref_exact:
            ref_counts_m = None
    try:
        ref_metrics = cr.seed_metrics(ref_payload, benchmark, method, seed)
    except (KeyError, ValueError, IndexError) as exc:
        out["missing_in_reference"].append(f"per-seed metrics ({exc})")
        return out
    for name in FIXED_METRICS:
        if name not in ref_metrics:
            out["missing_in_reference"].append(name)
        elif float(ref_metrics[name]) != float(job["fixed"][name]):
            out["fixed_metric_differences"][name] = [float(job["fixed"][name]),
                                                     float(ref_metrics[name])]
    implied = (metrics_from_matrices(ref_matrices, ref_counts_m, counts)
               if ref_counts_m is not None else None)
    for name in ROUTED_METRICS:
        if name not in ref_metrics:
            out["missing_in_reference"].append(name)
            continue
        if implied is not None and float(implied[name]) != float(ref_metrics[name]):
            out["reference_inconsistent_metrics"][name] = [float(implied[name]),
                                                           float(ref_metrics[name])]
        if float(ref_metrics[name]) != float(job["metrics"][name]):
            out["metric_differences"][name] = [float(job["metrics"][name]),
                                               float(ref_metrics[name])]
    out["unchecked_metrics"] = sorted(set(ref_metrics) - set(FIXED_METRICS) - set(ROUTED_METRICS))
    # The reference's routed metrics follow from its own matrices (checked
    # above), so a metric difference is explained exactly when every matrix
    # difference is.
    routed_ok = (implied is not None and not out["matrix_differences"]
                 and (not out["metric_differences"] or bool(out["within_near_ties"])))
    out["identical"] = bool(
        out["test_counts_identical"] and out["trained_state_identical"]
        and out["routers_identical"]
        and not out["missing_in_reference"] and not out["reference_inconsistent_metrics"]
        and not out["fixed_matrix_differences"] and not out["fixed_metric_differences"]
        and routed_ok)
    return out


def calibrated_runs(payload: dict) -> set[tuple[str, str, int]]:
    """``(benchmark, method, seed)`` of every run with fitted routers in an artifact."""
    if payload.get("protocol") != "class_il" or payload.get("inference_calibration") in (None, "none"):
        return set()
    return {(benchmark, method, int(seed))
            for benchmark, by_method in (payload.get("calibration") or {}).items()
            for method, by_seed in (by_method or {}).items()
            for seed in (by_seed or {})}


def audit_job(result_json: Path, reference: dict, cache: dict) -> list[dict]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    if payload.get("protocol") != "class_il" or payload.get("inference_calibration") in (None, "none"):
        return []
    group = result_json.parent.name.split("__")[0]
    entries = []
    recorded = calibrated_runs(payload)
    for benchmark, by_method in (payload.get("per_seed") or {}).items():
        for method in by_method:
            if method not in rp.ISOLATION:
                continue
            try:
                seeds = cr.method_seeds(payload, benchmark, method)
            except ValueError as exc:
                seeds, problem = [], str(exc)
                entries.append({"job": result_json.parent.name, "group": group,
                                "benchmark": benchmark, "method": method, "seed": -1,
                                "errors": [f"cannot align seeds: {problem}"]})
            for seed in seeds:
                if (benchmark, method, int(seed)) not in recorded:
                    entries.append({"job": result_json.parent.name, "group": group,
                                    "benchmark": benchmark, "method": method,
                                    "seed": int(seed), "errors": [
                                        "calibrated job without a calibration record for "
                                        "this isolation method"]})
    with np.load(result_json.with_suffix(".npz"), allow_pickle=False) as npz:
        for benchmark, by_method in (payload.get("calibration") or {}).items():
            for method, by_seed in by_method.items():
                for seed_text, record in by_seed.items():
                    entries.append(_audit_record(result_json, payload, npz, group, benchmark,
                                                 method, int(seed_text), record or {},
                                                 reference, cache))
    return entries


def _audit_record(result_json: Path, payload: dict, npz, group: str, benchmark: str,
                  method: str, seed: int, record: dict, reference: dict, cache: dict) -> dict:
    entry = {"job": result_json.parent.name, "group": group, "benchmark": benchmark,
             "method": method, "seed": seed, "kind": record.get("kind"), "errors": []}
    pred = record.get("predictions")
    if not pred:
        entry["errors"].append("no per-example predictions recorded (the job was "
                               "not launched with --predictions-dir)")
        return entry
    path = result_json.parent / pred.get("relative_path", "")
    if not path.is_file() or rp.sha256_file(path) != pred.get("sha256"):
        entry["errors"].append("predictions file missing or altered")
        return entry
    try:
        arrays, meta = rp.load_predictions(path)
        if meta.get("trained_state_content_sha256") != (
                record.get("trained_state") or {}).get("content_sha256"):
            entry["errors"].append("predictions belong to a different trained state")
        # The verifier's own checks: structure, the raw argmin decisions, every
        # matrix and metric of the job, the stored routers and fitting records.
        entry["errors"] += rp.compare_with_artifact(arrays, meta, payload, npz,
                                                    benchmark, method, seed)
    except Exception as exc:  # noqa: BLE001 - a corrupt file is an error entry
        entry["errors"].append(f"cannot check predictions: {type(exc).__name__}: {exc}")
    if entry["errors"]:
        return entry
    precisions = {m.get("routing_dtype", "float32") for m in record["prefix_models"]}
    entry["routing_dtype"] = "/".join(sorted(precisions))
    if precisions != {"float64"}:
        entry["note"] = "routers already applied in float32; nothing changed"
    T = int(meta["num_tasks"])
    disjoint = bool(meta.get("disjoint_classes", True))
    ref_payload = ref_npz = None
    if group in reference:
        ref_json = reference[group]
        if ref_json not in cache:
            cache[ref_json] = (json.loads(ref_json.read_text(encoding="utf-8")),
                               np.load(ref_json.with_suffix(".npz"), allow_pickle=False))
        ref_payload, ref_npz = cache[ref_json]
    # The float32 decisions to rebuild are the reference's (or, without one,
    # this job's own as v3.0.9 would have made them): TF32 matmuls there make
    # every router's decisions inexact.
    tf32 = rp.matmul_tf32(ref_payload if ref_payload is not None else payload)
    rebuild = float32_rebuild(arrays, record, tf32=tf32)
    for k in ("decisions_compared", "routes_moved", "tie_flags_moved",
              "routes_moved_by_stage", "float32_near_tie_decisions"):
        entry[k] = rebuild[k]
    entry["rebuild_precision"] = rebuild["precision"]
    current = rp.recompute(arrays, T, disjoint)
    raw = rp.recompute(arrays, T, disjoint, "raw_")
    entry["current"] = per_seed_metrics(current["stats"], current["result_matrix"])
    unknown = int(rebuild["unknown_prediction_mask"].sum())
    entry["moved_without_recorded_prediction"] = unknown
    if not disjoint and unknown:
        entry["not_rebuilt"] = (
            f"shared label sets: {unknown} moved decision(s) chose an expert whose "
            "prediction was not recorded, so the float32 metrics cannot be rebuilt")
        return entry
    rebuilt = rp.recompute(rebuild["arrays"], T, disjoint)
    entry["float32"] = per_seed_metrics(rebuilt["stats"], rebuilt["result_matrix"])
    entry["change"] = {k: entry["current"][k] - entry["float32"][k]
                       for k in entry["current"] if k in entry["float32"]
                       and entry["current"][k] != entry["float32"][k]}
    if ref_payload is None:
        return entry
    job = {"benchmark": benchmark, "method": method, "seed": seed, "record": record,
           "counts": np.bincount(arrays["true_task"].astype(np.int64), minlength=T),
           "rebuilt": rebuilt, "raw": raw, "metrics": entry["float32"],
           "fixed": fixed_metrics(current, raw), "disjoint": disjoint,
           "cells": near_tie_cells(rebuild, arrays, T)}
    entry["reference"] = compare_with_reference(job, ref_payload, ref_npz)
    return entry


def coverage_problems(entries: list[dict], reference: dict, cache: dict) -> list[str]:
    """Every audited group must be in the reference, and every calibrated run of
    every calibrated reference group must be audited."""
    problems = []
    audited: dict[str, set] = {}
    for e in entries:
        audited.setdefault(e["group"], set()).add((e["benchmark"], e["method"], e["seed"]))
    for group in sorted(set(audited) - set(reference)):
        problems.append(f"group {group} is not in the reference")
    for group, ref_json in sorted(reference.items()):
        if ref_json in cache:
            payload = cache[ref_json][0]
        else:
            payload = json.loads(ref_json.read_text(encoding="utf-8"))
        expected = calibrated_runs(payload)
        if not expected:
            continue                     # no fitted routers in this group
        missing = sorted(expected - audited.get(group, set()))
        if missing:
            problems.append(f"reference group {group}: {len(missing)} of {len(expected)} "
                            "calibrated run(s) not audited, e.g. "
                            + ", ".join(f"{b}/{m}/seed{s}" for b, m, s in missing[:3]))
    return problems


def device_float32_routing(run_dir: Path):
    """Whether the run's GPU applied float32 routers as NumPy does.

    ``float32_ok`` of the routing self-test the suite records in
    ``environment.json`` (since v3.0.10); ``None`` when not recorded. The
    float32 rebuild assumes the reference's device behaved like NumPy, which
    the same GPU type is expected to repeat.
    """
    try:
        environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    probe = (environment.get("torch_probe") or {}).get("float64_routing") or {}
    return probe.get("float32_ok")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, nargs="+", required=True,
                   help="Suite run directories whose calibrated jobs routed in float64.")
    p.add_argument("--reference", type=Path, nargs="+", default=None,
                   help="Merged directories (or run directories) of a float32-routed reference.")
    p.add_argument("--out", type=Path, default=None, help="Write a JSON report.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    reference = cr.merged_groups(args.reference) if args.reference else {}
    entries: list[dict] = []
    problems: list[str] = []
    cache: dict = {}
    try:
        for run_dir in args.run_dir:
            found = []
            for result_json in sorted(run_dir.glob("artifacts/*/result.json")):
                found += audit_job(result_json, reference, cache)
            if not found:
                problems.append(f"{run_dir}: no calibrated class-IL job "
                                "(artifacts/*/result.json with fitted routers)")
            entries += found
        if args.reference:
            problems += coverage_problems(entries, reference, cache)
    finally:
        for _, npz in cache.values():
            npz.close()
    errors = [e for e in entries if e["errors"]]
    compared = [e for e in entries if "reference" in e]
    differing = [e for e in compared if not e["reference"]["identical"]]
    near = [e for e in compared if e["reference"]["identical"] and e["reference"]["within_near_ties"]]
    not_compared = ([e for e in entries if not e["errors"] and "reference" not in e
                     and e["group"] in reference] if args.reference else [])
    summary = {
        "jobs": len(entries), "jobs_with_errors": len(errors),
        "decisions_compared": sum(e.get("decisions_compared", 0) for e in entries),
        "routes_moved": sum(e.get("routes_moved", 0) for e in entries),
        "tie_flags_moved": sum(e.get("tie_flags_moved", 0) for e in entries),
        "jobs_with_moved_routes": sum(1 for e in entries if e.get("routes_moved")),
        "reference_compared": len(compared),
        "reference_identical": len(compared) - len(differing) - len(near),
        "reference_within_float32_near_ties": len(near),
        "reference_different": [f"{e['group']}/{e['reference']['run']}" for e in differing],
        "reference_not_compared": [f"{e['group']}/{e['benchmark']}/{e['method']}/seed{e['seed']}"
                                   for e in not_compared],
        "device_float32_routing_matches_numpy": {str(run_dir): device_float32_routing(run_dir)
                                                 for run_dir in args.run_dir},
        "problems": problems,
    }
    if problems or errors or not entries:
        status = "failed"
    elif args.reference and (differing or not_compared):
        status = "unexplained"
    else:
        status = "explained" if args.reference else "measured"
    report = {"status": status, "summary": summary, "entries": entries}
    if args.out:
        args.out.write_text(json.dumps(rp._finite(report), indent=2, allow_nan=False),
                            encoding="utf-8")
    by_group: dict[str, list[dict]] = {}
    for e in entries:
        by_group.setdefault(e["group"], []).append(e)
    for group, items in sorted(by_group.items()):
        moved = sum(e.get("routes_moved", 0) for e in items)
        total = sum(e.get("decisions_compared", 0) for e in items)
        line = (f"{group:28s} {len(items):3d} job(s): {moved} of {total} test decision(s) "
                "moved by float64 routing")
        refs = [e for e in items if "reference" in e]
        if refs:
            same = sum(1 for e in refs if e["reference"]["identical"])
            line += f"; float32 rebuild matches the reference in {same}/{len(refs)}"
        print(line)
    for run_dir, matches in summary["device_float32_routing_matches_numpy"].items():
        if matches is not True:
            print(f"WARNING {run_dir}: the routing self-test "
                  + ("found that this GPU does not apply float32 routers as NumPy does"
                     if matches is False else "did not record float32 routing (float32_ok)")
                  + "; the float32 rebuild assumes the reference's GPU did")
    for message in problems:
        print(f"ERROR {message}")
    for e in errors:
        print(f"ERROR {e['job']} {e['method']} seed{e['seed']}: {'; '.join(e['errors'][:3])}")
    for e in not_compared:
        print(f"NOT COMPARED {e['group']}/{e['benchmark']}/{e['method']}/seed{e['seed']}: "
              f"{e.get('not_rebuilt', 'not rebuilt')}")
    for e in differing[:10]:
        r = e["reference"]
        parts = [f"{label} {sorted(r[field])[:3]}" for label, field in (
            ("missing", "missing_in_reference"), ("routers", "router_differences"),
            ("raw matrices", "fixed_matrix_differences"),
            ("raw/oracle metrics", "fixed_metric_differences"),
            ("reference metrics not implied by its matrices", "reference_inconsistent_metrics"),
            ("matrices", "matrix_differences"), ("metrics", "metric_differences")) if r[field]]
        for label, field in (("test-set sizes", "test_counts_identical"),
                             ("trained state", "trained_state_identical")):
            if r[field] is False:
                parts.append(f"{label} differ")
        print(f"DIFFERS {e['group']}/{r['run']}: " + "; ".join(parts))
    print(f"{summary['routes_moved']} of {summary['decisions_compared']} calibrated test "
          f"decision(s) moved in {summary['jobs_with_moved_routes']} of {summary['jobs']} job(s)"
          + (f"; reference: {summary['reference_identical']} identical, "
             f"{summary['reference_within_float32_near_ties']} within float32 near-ties, "
             f"{len(differing)} different"
             + (f", {len(not_compared)} not compared" if not_compared else "")
             if args.reference else "")
          + f" -> {status}")
    return 0 if status in ("explained", "measured") else 1


if __name__ == "__main__":
    raise SystemExit(main())
