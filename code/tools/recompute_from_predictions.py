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
3. for calibrated jobs, recomputes every stored router's fitting accuracies
   (plain and class-balanced) and the raw rule's on the stored calibration
   inputs, checks that score-bias-acc never fits below the raw rule, and
   with ``--refit`` re-fits the prefix routers from those inputs and reports
   how far every router parameter moved and how often the re-fitted routers
   choose the same task as the recorded ones.

Before any of this the file's structure is checked: every array present
with consistent shapes, true tasks and labels in range, finite routing
scores, routes only among the candidates ``0..s`` at stage ``s`` and ``-1``
(no tie, no prediction) before an example's task has arrived, per-task
example counts equal to the job's recorded test-set sizes, and, for
calibrated jobs, the matched raw pass and the router-fitting inputs present,
with the raw pass recorded on the same examples with the same scores and
oracle predictions. Wherever a router chose an example's own task, its
prediction must equal that expert's (oracle) prediction, and wherever the
raw and the calibrated router chose the same task, their predictions must
agree: a prediction read from the wrong expert fails here. A calibrated job's
stored router-fitting rows must match, task by task, the held-out split it
recorded (``split_counts``, ``observed_counts``): a router fitted on other
data fails here.

It also checks that every raw-router decision is the lowest-ID argmin of
the recorded scores (exact), and re-applies the job's stored calibrated
routers to the recorded scores in the precision the job applied them in:
float64 for routers fitted since v3.0.10 (``routing_dtype``), float32 for
earlier ones (TF32 if the job recorded ``cuda_matmul_allow_tf32``; off in
every recorded run). The score-bias and score-bias-acc routers put one
non-zero coefficient in each decision, so their decisions and tie flags
must be reproduced exactly; for the score-logreg router (a full ``T x T``
matrix product, which the GPU may sum in a different order) a disagreement
is accepted only when the exact gap between the recorded and the recomputed
winner is within the rounding-error bound of the two decisions (including
the error of products below the normal range), such near-ties are counted,
and a recorded tie needs a higher task within rounding of the recorded route
(ties go to the lowest task ID).

All of these checks are enforced: a failure makes the job's entry an error
and the exit status non-zero. A run none of whose jobs recorded predictions
(runs before v3.0.5) has nothing to check: status ``not-recorded``, exit 0;
a job launched with ``--predictions-dir`` whose files are missing is an
error. Only the re-fit (``--refit``) is
reported, not enforced, because solver versions can move re-fitted
coefficients. The decomposition is reported only for disjoint label sets,
where Proposition 1 (routing ceiling) applies.

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
from fractions import Fraction
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


PER_EXAMPLE = ("true_task", "label_local", "oracle_pred_local", "scores",
               "route_by_stage", "tie_by_stage", "routed_pred_local_by_stage")


def expected_test_counts(payload: dict, benchmark: str, seed: int) -> list[int] | None:
    """Recorded per-task test-set sizes of the isolation view, if available."""
    counts = (((payload.get("benchmark_metadata") or {}).get(benchmark) or {})
              .get("per_seed_counts") or {}).get(str(seed)) or {}
    view = counts.get("isolation") or {}
    values = view.get("test_samples_per_task")
    return [int(v) for v in values] if values is not None else None


def is_calibrated(meta: dict) -> bool:
    return meta.get("calibration", "none") not in (None, "none")


def structural_errors(arrays: dict[str, np.ndarray], meta: dict,
                      expected_counts: list[int] | None = None) -> list[str]:
    """Shape, range and consistency checks of a predictions file.

    These guarantee that ``recompute`` sees a well-formed recording: an
    impossible route, a decision recorded before a task arrived, a non-finite
    score or a missing example would otherwise be counted silently. A
    calibrated job must also carry its matched raw pass (``raw_*``) and the
    held-out scores its routers were fitted on.
    """
    errors: list[str] = []
    T = int(meta["num_tasks"])
    C = int(meta.get("classes_per_task") or 0)
    calibrated = is_calibrated(meta)
    prefixes = [""] + (["raw_"] if calibrated or "raw_true_task" in arrays else [])
    if calibrated:
        missing = [k for k in ("calibration_features", "calibration_task_ids") if k not in arrays]
        if missing:
            errors.append(f"calibration inputs missing {missing}")
        else:
            x = np.asarray(arrays["calibration_features"])
            y = np.asarray(arrays["calibration_task_ids"]).astype(np.int64)
            if x.ndim != 2 or x.shape[1] != T or y.shape != (x.shape[0],) or not y.size:
                errors.append(f"calibration inputs have shapes {x.shape} and {y.shape} "
                              f"(expected (m, {T}) and (m,), m > 0)")
            elif y.min() < 0 or y.max() >= T or not np.isfinite(x).all():
                errors.append("calibration inputs: task id outside "
                              f"0..{T - 1} or non-finite score")
    for prefix in prefixes:
        name = prefix or "router"
        missing = [k for k in PER_EXAMPLE if prefix + k not in arrays]
        if missing:
            errors.append(f"{name}: arrays missing {missing}")
            continue
        tt = np.asarray(arrays[prefix + "true_task"]).astype(np.int64)
        n = tt.shape[0] if tt.ndim == 1 else -1
        shapes = {"label_local": (n,), "oracle_pred_local": (n,), "scores": (n, T),
                  "route_by_stage": (n, T), "tie_by_stage": (n, T),
                  "routed_pred_local_by_stage": (n, T)}
        bad = [k for k, shape in shapes.items() if arrays[prefix + k].shape != shape]
        if n < 0 or bad:
            errors.append(f"{name}: inconsistent array shapes ({bad or 'true_task'})")
            continue
        label = arrays[prefix + "label_local"].astype(np.int64)
        oracle = arrays[prefix + "oracle_pred_local"].astype(np.int64)
        routes = arrays[prefix + "route_by_stage"].astype(np.int64)
        ties = arrays[prefix + "tie_by_stage"].astype(bool)
        preds = arrays[prefix + "routed_pred_local_by_stage"].astype(np.int64)
        if n and (tt.min() < 0 or tt.max() >= T):
            errors.append(f"{name}: true task outside 0..{T - 1}")
        if C and n and (min(label.min(), oracle.min()) < 0 or max(label.max(), oracle.max()) >= C):
            errors.append(f"{name}: label or oracle prediction outside 0..{C - 1}")
        if not np.isfinite(arrays[prefix + "scores"]).all():
            errors.append(f"{name}: non-finite routing scores")
        for stage in range(T):
            pending = tt > stage
            if (np.any(routes[pending, stage] != -1) or np.any(ties[pending, stage])
                    or np.any(preds[pending, stage] != -1)):
                errors.append(f"{name} stage {stage}: decisions recorded before the task arrived")
            r = routes[~pending, stage]
            if np.any((r < 0) | (r > stage)):
                errors.append(f"{name} stage {stage}: route outside the candidates 0..{stage}")
            q = preds[~pending, stage]
            if C and np.any((q < 0) | (q >= C)):
                errors.append(f"{name} stage {stage}: prediction outside 0..{C - 1}")
        if expected_counts is not None and n >= 0:
            counts = np.bincount(tt[(tt >= 0) & (tt < T)], minlength=T).tolist()
            if counts != list(expected_counts):
                errors.append(f"{name}: per-task example counts {counts} differ from the "
                              f"recorded test-set sizes {list(expected_counts)}")
        # The routed expert's prediction is the argmax of the same logits row
        # as the oracle prediction whenever the route is the true task, so
        # the two must agree exactly; anything else means a prediction was
        # read from the wrong expert.
        for stage in range(T):
            routed_home = (tt <= stage) & (routes[:, stage] == tt)
            bad = int(np.sum(preds[routed_home, stage] != oracle[routed_home]))
            if bad:
                errors.append(f"{name} stage {stage}: {bad} example(s) routed to their own "
                              "task predicted differently from that task's expert")
    if len(prefixes) == 2 and not errors:
        for key in ("true_task", "label_local", "oracle_pred_local", "scores"):
            if not np.array_equal(arrays[key], arrays["raw_" + key]):
                errors.append(f"raw_{key} differs from the calibrated pass (same experts, "
                              "same examples)")
        if not errors:
            # Same experts, same examples: where both routers chose the same
            # task they read the same expert, so the predictions agree.
            same_route = arrays["route_by_stage"] == arrays["raw_route_by_stage"]
            arrived = arrays["route_by_stage"] >= 0
            differ = (arrays["routed_pred_local_by_stage"]
                      != arrays["raw_routed_pred_local_by_stage"])
            bad = int(np.sum(same_route & arrived & differ))
            if bad:
                errors.append(f"{bad} decision(s) with the same task chosen by the raw and "
                              "the calibrated router have different predictions")
    return errors


def calibration_count_errors(arrays: dict[str, np.ndarray], record: dict,
                             num_tasks: int) -> list[str]:
    """The stored router-fitting inputs must be the job's held-out split.

    Per task, the number of stored calibration rows must equal the size of
    the held-out split (``split_counts[t]["calibration"]``) and the number of
    examples the runner scored (``observed_counts[t]``). Fitting a router on
    other data -- the test set, or the training split the experts learned
    from -- changes these counts and is caught here; every other check would
    pass, because the fit accuracy is recomputed on whatever inputs were
    stored.
    """
    errors: list[str] = []
    y = np.asarray(arrays["calibration_task_ids"]).astype(np.int64)
    counts = np.bincount(y[(y >= 0) & (y < num_tasks)], minlength=num_tasks).tolist()
    split = record.get("split_counts") or []
    observed = record.get("observed_counts") or []
    if not split or not observed:
        errors.append("calibrated job without recorded held-out split sizes "
                      "(split_counts, observed_counts)")
        return errors
    expected_split = [int((entry or {}).get("calibration", -1)) for entry in split]
    if counts != expected_split:
        errors.append(f"stored router-fitting rows per task {counts} differ from the "
                      f"held-out split sizes {expected_split}")
    if counts != [int(v) for v in observed]:
        errors.append(f"stored router-fitting rows per task {counts} differ from the "
                      f"examples scored for fitting {list(observed)}")
    return errors


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


EXACT_ROUTERS = ("score-bias", "score-bias-acc")
U32 = 2.0 ** -24          # unit roundoff of float32
U64 = 2.0 ** -53          # unit roundoff of float64
U_TF32 = 2.0 ** -11       # unit roundoff of the TF32 inputs of a tensor-core matmul

# Per precision: (unit roundoff of the products, of the accumulation and the
# intercept addition; absolute error a product can pick up below the normal
# range, per unit of |coefficient| for TF32, whose tensor cores may flush a
# subnormal float32 input to zero).
PRECISIONS = {
    "float64": {"dtype": np.float64, "products": U64, "sum": U64, "tiny": 2.0 ** -1074},
    "float32": {"dtype": np.float32, "products": U32, "sum": U32, "tiny": 2.0 ** -149},
    "tf32": {"dtype": np.float32, "products": U_TF32, "sum": U32, "tiny": 2.0 ** -126},
}


def router_precision(state: dict, tf32: bool = False) -> str:
    """Precision in which the job applied a stored router on the device.

    Routers fitted since v3.0.10 declare ``routing_dtype = "float64"`` and
    were applied in float64, where TF32 never applies. Earlier states carry
    no field and were applied in the float32 of the scores; ``tf32`` is the
    job's recorded ``cuda_matmul_allow_tf32`` (off in every recorded run).
    """
    name = state.get("routing_dtype")
    if name == "float64":
        return "float64"
    if name not in (None, "float32"):
        raise ValueError(f"unknown routing_dtype {name!r}")
    return "tf32" if tf32 else "float32"


def apply_router(x: np.ndarray, state: dict, precision: str) -> dict:
    """Apply a serialized router to recorded float32 scores, as the device did.

    Returns the chosen tasks (lowest task ID among exact ties), the tie mask,
    the decisions in float64 for error analysis, and a per-decision bound on
    how far either side's rounding can move a decision from its exact value.

    The standardized scores ``z`` are the same values on both sides
    (correctly rounded subtraction and division in the routing precision);
    only the summation of the ``n``-term matrix product (any order, with or
    without FMA) and the intercept addition may round differently. Each side
    is within ``gamma_n * sum_k |z_k c_jk| + u * |d_j|`` of the exact decision,
    plus the absolute error of products that fall below the normal range
    (TF32: of inputs flushed to zero). The bound returned is twice that, for
    safety; a flip between two sides is explained by rounding only if the
    exact gap is within the sum of the two classes' bounds, doubled again.
    """
    spec = PRECISIONS[precision]
    dtype = spec["dtype"]
    xs = np.asarray(x, dtype=np.float32).astype(dtype)
    mean = np.asarray(state["feature_mean"], dtype=dtype)
    scale = np.asarray(state["feature_scale"], dtype=dtype)
    coef = np.asarray(state["coef"], dtype=dtype)
    intercept = np.asarray(state["intercept"], dtype=dtype)
    classes = np.asarray(state["classes"], dtype=np.int64)
    z = (xs - mean) / scale
    decision = z @ coef.T + intercept
    winning = decision == decision.max(axis=1, keepdims=True)
    pred = np.where(winning, classes[None, :], np.iinfo(np.int64).max).min(axis=1)
    z64, c64 = z.astype(np.float64), coef.astype(np.float64)
    d64 = z64 @ c64.T + intercept.astype(np.float64)
    n_terms = max(1, z64.shape[1])
    unit = spec["products"]
    gamma = n_terms * unit / (1.0 - n_terms * unit)
    absolute = np.abs(c64).sum(axis=1)[None, :] if precision == "tf32" else 1.0
    bound = 2.0 * (gamma * (np.abs(z64) @ np.abs(c64).T) + spec["sum"] * np.abs(d64)
                   + n_terms * spec["tiny"] * absolute)
    return {"pred": pred, "tie": winning.sum(axis=1) > 1, "decision": d64,
            "bound": bound, "classes": classes}


def _within_rounding(applied: dict, i: int, class_a: int, class_b: int) -> bool:
    """Whether rounding can explain decisions ``a`` and ``b`` swapping order."""
    column = {int(c): j for j, c in enumerate(applied["classes"])}
    ja, jb = column.get(int(class_a)), column.get(int(class_b))
    if ja is None or jb is None:
        return False
    d, bound = applied["decision"], applied["bound"]
    return abs(d[i, ja] - d[i, jb]) <= 2.0 * (bound[i, ja] + bound[i, jb])


def _columns(applied: dict, tasks) -> np.ndarray:
    """Decision column of every task ID in ``tasks`` (``-1`` if not a candidate)."""
    column = {int(c): j for j, c in enumerate(applied["classes"])}
    return np.array([column.get(int(t), -1) for t in np.asarray(tasks).ravel()],
                    dtype=np.int64)


def _near_column(applied: dict, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """``[len(rows), classes]``: decisions within rounding of column ``cols[i]``.

    Each pair is compared with both decisions' own error bounds, as in
    :func:`_within_rounding`.
    """
    d, bound = applied["decision"][rows], applied["bound"][rows]
    i = np.arange(rows.size)
    return np.abs(d - d[i, cols][:, None]) <= 2.0 * (bound + bound[i, cols][:, None])


def within_rounding_of_winner(applied: dict) -> np.ndarray:
    """``[rows, classes]``: the tasks whose decision lies within rounding of the winner's.

    The winner is included. Another summation order can choose only among
    these tasks (on both sides every decision is within its own error bound
    of the exact value), and it can record an exact tie only on a row with
    two or more of them. Every candidate is compared with the winner using
    its own error bound, not only the runner-up: a lower decision with a
    larger bound can still overtake.
    """
    d = applied["decision"]
    if not d.size:
        return np.zeros(d.shape, dtype=bool)
    rows = np.arange(d.shape[0])
    cols = _columns(applied, applied["pred"])
    near = _near_column(applied, rows, cols)
    near[rows, cols] = True
    return near


def near_tie_rows(applied: dict) -> np.ndarray:
    """Per row: whether another task's decision lies within rounding of the winner's.

    Such a row is a near-tie: another summation order may choose that task
    (:func:`within_rounding_of_winner`). A single candidate is never a
    near-tie.
    """
    d = applied["decision"]
    if d.shape[1] < 2 or not d.shape[0]:
        return np.zeros(d.shape[0], dtype=bool)
    return within_rounding_of_winner(applied).sum(axis=1) > 1


def tie_partner_within_rounding(applied: dict, route) -> np.ndarray:
    """Per row: whether an exact tie recorded at ``route`` is possible.

    The device breaks an exact tie towards the lowest task ID, so a tie
    recorded with route ``r`` needs a task ``j > r`` whose decision lies
    within rounding of ``r``'s. This is checked against the recorded route,
    not against the two best recomputed decisions: a third task between them
    in the recomputed order, with a much smaller error bound, says nothing
    about the pair the device saw tie.
    """
    d = applied["decision"]
    route = np.asarray(route, dtype=np.int64).ravel()
    out = np.zeros(d.shape[0], dtype=bool)
    if d.shape[1] < 2 or not d.shape[0]:
        return out
    cols = _columns(applied, route)
    rows = np.flatnonzero(cols >= 0)
    if rows.size:
        near = _near_column(applied, rows, cols[rows])
        above = np.asarray(applied["classes"])[None, :] > route[rows][:, None]
        out[rows] = np.any(near & above, axis=1)
    return out


def stored_router_agreement(arrays: dict[str, np.ndarray], record: dict,
                            tf32: bool = False) -> dict:
    """Re-apply the job's serialized prefix routers to the recorded scores.

    Returns the agreement, the number of disagreements, how many of them are
    near-ties in the routing precision, and ``errors`` -- disagreements that
    rounding cannot explain. The score-bias routers put one non-zero
    coefficient in each decision, so their decisions and tie flags must be
    reproduced exactly (in float32 or float64; not under TF32). For the
    score-logreg router a recorded tie needs a higher task whose decision lies
    within rounding of the recorded route's (:func:`tie_partner_within_rounding`).

    ``tf32`` is the job's recorded ``cuda_matmul_allow_tf32``; it matters only
    for routers applied in float32 (:func:`router_precision`).
    """
    kind = record.get("kind")
    models = record.get("prefix_models") or []
    scores = arrays["scores"].astype(np.float32)
    routes = arrays["route_by_stage"]
    ties = arrays.get("tie_by_stage")
    agree = total = mismatches = near_ties = tie_flag_errors = 0
    precisions: set[str] = set()
    errors: list[str] = []
    stages_with_rows = [s for s in range(routes.shape[1]) if np.any(arrays["true_task"] <= s)]
    if len(models) < len(stages_with_rows):
        errors.append(f"{len(models)} stored prefix router(s) for "
                      f"{len(stages_with_rows)} evaluated stage(s)")
    for stage, state in enumerate(models):
        rows = arrays["true_task"] <= stage
        if not rows.any():
            continue
        precision = router_precision(state, tf32)
        precisions.add(precision)
        exact = kind in EXACT_ROUTERS and precision != "tf32"
        applied = apply_router(scores[rows, :stage + 1], state, precision)
        pred, tie = applied["pred"], applied["tie"]
        recorded = routes[rows, stage].astype(np.int64)
        recorded_tie = (np.asarray(ties[rows, stage], dtype=bool) if ties is not None
                        else None)
        same = pred == recorded
        agree += int(same.sum()); total += int(rows.sum())
        if recorded_tie is not None:
            if exact:
                tie_flag_errors += int(np.sum(tie != recorded_tie))
            else:
                # A recorded exact tie needs a higher task within rounding of
                # the recorded route; a single candidate can never tie.
                tie_flag_errors += int(np.sum(
                    recorded_tie & ~tie_partner_within_rounding(applied, recorded)))
        for i in np.flatnonzero(~same):
            mismatches += 1
            if not exact and _within_rounding(applied, i, pred[i], recorded[i]):
                near_ties += 1
    label = "/".join(sorted(precisions)) or "float32"
    if mismatches - near_ties or tie_flag_errors:
        if any(kind in EXACT_ROUTERS and p != "tf32" for p in precisions):
            errors.append(f"stored {kind} router ({label}): {mismatches} route(s) and "
                          f"{tie_flag_errors} tie flag(s) differ from the recorded decisions "
                          "(expected exact)")
        else:
            errors.append(f"stored {kind} router ({label}): {mismatches - near_ties} route(s) "
                          f"and {tie_flag_errors} tie flag(s) differ from the recorded "
                          f"decisions beyond {label} rounding")
    return {"route_agreement": agree / total if total else None,
            "decisions_compared": total, "disagreements": mismatches,
            "near_tie_disagreements": near_ties, "tie_flag_errors": tie_flag_errors,
            "routing_precision": label, "tf32": bool(tf32), "errors": errors}


def decomposition(stats: dict) -> dict:
    """Final-stage error partition in percentage units (pooled over examples).

    Valid for disjoint label sets only (Proposition 1); ``rho`` is ``None``
    when ``A = 100`` (no errors to recover).
    """
    A = stats["micro_class_il_acc"]; U = stats["micro_task_inference_acc"]
    O = stats["micro_oracle_route_acc"]; V = stats["conditional_class_acc"]
    return {
        "routed_accuracy": A, "task_inference": U, "oracle_accuracy": O,
        "conditional_accuracy": V,
        "identity_residual_A_minus_UV": A - U * V / 100.0,
        "oracle_error": 100.0 - O,
        "recoverable_routing_loss": O - A,
        "recoverable_fraction_percent": (100.0 * (O - A) / (100.0 - A)) if A < 100.0 else None,
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
    """Exact comparison of recomputed statistics with the job's artifacts.

    This is the check the verifier runs on every predictions file; the
    report-only extras of ``check_job`` (decomposition, re-fit) build on it.
    """
    job_calibration = payload.get("inference_calibration")
    if job_calibration is not None and meta.get("calibration", "none") != job_calibration:
        return [f"predictions recorded with calibration {meta.get('calibration', 'none')!r}, "
                f"job ran {job_calibration!r}"]
    errors: list[str] = structural_errors(arrays, meta,
                                          expected_test_counts(payload, benchmark, seed))
    if errors:
        return errors            # recomputing a malformed recording proves nothing
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
    calibrated = is_calibrated(meta)
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
    if calibrated:
        record = (((payload.get("calibration") or {}).get(benchmark) or {})
                  .get(method) or {}).get(str(seed)) or {}
        if not record.get("prefix_models"):
            errors.append(f"{key}: calibrated job without stored prefix routers")
        else:
            errors += [f"{key}: {e}" for e in calibration_count_errors(arrays, record, T)]
            stored = stored_router_agreement(arrays, record, matmul_tf32(payload))
            errors += [f"{key}: {e}" for e in stored["errors"]]
            errors += [f"{key}: {e}" for e in fit_accuracy_check(arrays, record)["errors"]]
    return errors


def matmul_tf32(payload: dict) -> bool:
    """Whether the job's CUDA matmuls ran in TF32 (recorded execution flags)."""
    effective = ((payload.get("execution") or {}).get("effective") or {})
    return bool(effective.get("cuda_matmul_allow_tf32", False))


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


def fit_accuracy_check(arrays: dict, record: dict) -> dict:
    """Recompute each stored router's fitting accuracies from the stored inputs.

    The runner recorded, per prefix stage, the accuracy of the fitted router
    (``fit_accuracy``) and of the raw argmin rule (``raw_argmin_fit_accuracy``)
    on the held-out training scores it was fitted on, and since v3.0.10 the
    class-balanced versions (``fit_balanced_accuracy``,
    ``raw_argmin_fit_balanced_accuracy``). The predictions file stores those
    scores (float32, exactly as computed), so every number is recomputed with
    the stored router in float64, the precision of the fit.

    The raw rule and the two bias routers (one product per decision) are
    reproduced exactly. A score-logreg decision is a full matrix product,
    whose float64 rounding may differ on another CPU's BLAS kernel; there a
    difference is accepted only up to the number of fitting examples with
    another task's decision within float64 rounding of the winner's
    (:func:`near_tie_rows`), and such near-ties are reported.

    score-bias-acc starts from the raw rule and accepts only changes that keep
    or raise its objective, the class-balanced fitting accuracy, so at every
    stage the stored router's balanced accuracy on its fitting inputs must be
    at least the raw rule's. Both are recomputed and compared as exact
    fractions: two float balanced accuracies can differ in the last bit while
    the routers are exactly as accurate. Since v3.0.10 (routers with
    ``routing_dtype``) the search itself compares exactly, as it always did
    with equal task counts (every recorded run); for an earlier router fitted
    with unequal counts, whose search compared float sums, a shortfall of
    up to 1e-9 is attributed to that rounding.
    """
    tc = _task_calibration()
    x = np.asarray(arrays["calibration_features"], dtype=np.float64)
    y = np.asarray(arrays["calibration_task_ids"], dtype=np.int64)
    kind = record.get("kind")
    errors: list[str] = []
    checked = near_tie_examples = 0
    for stage, state in enumerate(record.get("prefix_models") or []):
        rows = y <= stage
        xs, ys = x[rows, :stage + 1], y[rows]
        fitted = tc.predict_score_logreg(xs, state)
        raw = np.argmin(xs, axis=1)
        slack = np.zeros(xs.shape[0], dtype=bool)
        if kind not in EXACT_ROUTERS and xs.shape[1] > 1:
            slack = near_tie_rows(apply_router(xs.astype(np.float32), state, "float64"))
            near_tie_examples += int(slack.sum())
        tasks = np.unique(ys)
        n_task = np.array([np.sum(ys == t) for t in tasks], dtype=np.float64)
        slack_task = np.array([np.sum(slack[ys == t]) for t in tasks], dtype=np.float64)
        checks = (
            ("fit_accuracy", float(np.mean(fitted == ys)), slack.sum() / max(1, ys.size)),
            ("raw_argmin_fit_accuracy", float(np.mean(raw == ys)), 0.0),
            ("fit_balanced_accuracy", tc.balanced_accuracy(fitted, ys),
             float(np.mean(slack_task / n_task)) if tasks.size else 0.0),
            ("raw_argmin_fit_balanced_accuracy", tc.balanced_accuracy(raw, ys), 0.0),
        )
        for name, value, allowed in checks:
            if name not in state:
                continue
            recorded = float(state[name])
            if value != recorded and abs(value - recorded) > allowed + 1e-12:
                errors.append(f"stage {stage}: {name} recomputed {value!r}, recorded "
                              f"{recorded!r}"
                              + (f" (beyond {int(slack.sum())} float64 near-tie(s))"
                                 if allowed else ""))
        if "fit_accuracy" in state:
            checked += 1
        if kind == "score-bias-acc" and xs.shape[1] > 1:
            fitted_exact = tc.balanced_accuracy_exact(fitted, ys)
            raw_exact = tc.balanced_accuracy_exact(raw, ys)
            exact_search = ("routing_dtype" in state
                            or len(set(np.bincount(ys).tolist())) == 1)
            allowed = Fraction(0) if exact_search else Fraction(1, 10 ** 9)
            if fitted_exact < raw_exact - allowed:
                errors.append(f"stage {stage}: the stored score-bias-acc router routes its "
                              f"fitting inputs with balanced accuracy {float(fitted_exact)!r}, "
                              f"below the raw rule's {float(raw_exact)!r} (compared exactly); "
                              "its search never accepts a loss")
    return {"stages_checked": checked, "near_tie_examples": near_tie_examples,
            "errors": errors}


ROUTER_PARAMETERS = ("feature_mean", "feature_scale", "coef", "intercept")


def refit_agreement(arrays: dict, meta: dict, record: dict,
                    tf32: bool = False) -> dict:
    """Re-fit the prefix routers and compare them with the stored ones.

    Reports, over all prefix stages, the largest absolute difference of every
    router parameter (``feature_mean``, ``feature_scale``, ``coef``,
    ``intercept``; the bias routers carry their whole fit in the intercept)
    and how often the re-fitted routers choose the recorded task on the test
    examples, applying them in the precision the stored routers were applied
    in (:func:`router_precision`). Report only: solver versions and BLAS
    kernels can move re-fitted coefficients.
    """
    tc = _task_calibration()
    T = int(meta["num_tasks"])
    states = tc.fit_prefix_calibrators(
        arrays["calibration_features"].astype(np.float64),
        arrays["calibration_task_ids"].astype(np.int64), T,
        C=float(record.get("C") or 1.0), random_state=int(record.get("split_seed", 0)),
        kind=record["kind"])
    scores = arrays["scores"].astype(np.float32)
    routes = arrays["route_by_stage"]
    agree = total = 0
    differences = {name: 0.0 for name in ROUTER_PARAMETERS}
    identical = True
    for stage, (state, stored) in enumerate(zip(states, record.get("prefix_models", []))):
        for name in ROUTER_PARAMETERS:
            a, b = np.asarray(state[name], dtype=np.float64), np.asarray(stored[name], dtype=np.float64)
            if a.shape != b.shape:
                differences[name] = float("inf")
                identical = False
                continue
            if a.size:
                differences[name] = max(differences[name], float(np.max(np.abs(a - b))))
            identical = identical and np.array_equal(a, b)
        rows = arrays["true_task"] <= stage
        if not rows.any():
            continue
        precision = router_precision(stored, tf32)
        pred = apply_router(scores[rows, :stage + 1], state, precision)["pred"]
        agree += int(np.sum(pred == routes[rows, stage])); total += int(rows.sum())
    return {"route_agreement": agree / total if total else float("nan"),
            "decisions_compared": total,
            "max_abs_parameter_difference": differences,
            "parameters_identical": bool(identical),
            "max_abs_coef_difference": differences["coef"],
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
                    out.append(entry)
                    if sha256_file(path) != pred.get("sha256"):
                        entry["errors"].append("predictions file SHA-256 differs from record")
                    try:
                        arrays, meta = load_predictions(path)
                        if meta.get("trained_state_content_sha256") != (
                                record.get("trained_state") or {}).get("content_sha256"):
                            entry["errors"].append("predictions belong to a different trained state")
                        entry["errors"] += compare_with_artifact(
                            arrays, meta, payload, npz, benchmark, method, int(seed_text))
                        if entry["errors"]:
                            continue
                        disjoint = bool(meta.get("disjoint_classes", True))
                        rec = recompute(arrays, int(meta["num_tasks"]), disjoint)
                        if disjoint:
                            entry["decomposition"] = decomposition(rec["stats"])
                        if is_calibrated(meta):
                            # Already enforced by compare_with_artifact; kept for the report.
                            entry["stored_router"] = stored_router_agreement(
                                arrays, record, matmul_tf32(payload))
                            entry["fit_accuracy_check"] = fit_accuracy_check(arrays, record)
                    except Exception as exc:  # noqa: BLE001 - a corrupt file is an error entry
                        entry["errors"].append(f"cannot check predictions: "
                                               f"{type(exc).__name__}: {exc}")
                        continue
                    if refit and "calibration_features" in arrays and record.get("kind") not in (None, "none"):
                        try:
                            entry["refit"] = refit_agreement(arrays, meta, record,
                                                             matmul_tf32(payload))
                        except Exception as exc:  # noqa: BLE001 - reported, not fatal
                            entry["refit"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _finite(value):
    """Replace NaN/inf by ``None`` so the report is strict JSON."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


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


def missing_prediction_entries(run_dir: Path, entries: list[dict]) -> list[dict]:
    """Jobs launched with ``--predictions-dir`` must have one file per isolation method."""
    plan_path = run_dir / "plan.json"
    if not plan_path.is_file():
        return []
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    found: dict[str, int] = {}
    for e in entries:
        found[e["job"]] = found.get(e["job"], 0) + 1
    out = []
    for item in plan.get("jobs", []):
        job = item.get("job", {})
        status_path = run_dir / "statuses" / f"{job.get('job_id')}.json"
        if not status_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if "--predictions-dir" not in (status.get("command") or []):
            continue
        expected = [m for m in str(job.get("methods", "")).split(",") if m in ISOLATION]
        have = found.get(job["job_id"], 0)
        if have < len(expected):
            out.append({"job": job["job_id"], "benchmark": job.get("benchmark", "?"),
                        "method": ",".join(expected), "seed": int(job.get("seed", -1)),
                        "file": None,
                        "errors": [f"per-example predictions recorded for {have} of "
                                   f"{len(expected)} isolation method(s)"]})
    return out


def run_recorded_predictions(run_dir: Path) -> bool:
    """Whether any job of a suite run was launched with ``--predictions-dir``."""
    for status_path in sorted((run_dir / "statuses").glob("*.json")):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if "--predictions-dir" in (status.get("command") or []):
            return True
    return False


def main(argv=None) -> int:
    args = parse_args(argv)
    results = (sorted(args.run_dir.glob("artifacts/*/result.json")) if args.run_dir
               else list(args.result))
    entries = []
    for path in results:
        entries += check_job(path, args.refit)
    if args.run_dir:
        entries += missing_prediction_entries(args.run_dir, entries)
    failures = [e for e in entries if e["errors"]]
    if entries:
        status = "failed" if failures else "passed"
    elif args.run_dir and not run_recorded_predictions(args.run_dir):
        # Runs produced before v3.0.5 recorded no per-example predictions;
        # there is nothing to check, and nothing is missing.
        status = "not-recorded"
    else:
        status = "no-predictions"
    report = {"files_checked": len(entries), "files_with_errors": len(failures),
              "status": status, "entries": entries}
    if args.out:
        args.out.write_text(json.dumps(_finite(report), indent=2, allow_nan=False),
                            encoding="utf-8")
    for e in entries:
        d = e.get("decomposition", {})
        refit = e.get("refit", {})
        stored = e.get("stored_router", {})
        rho = d.get("recoverable_fraction_percent")

        def num(x, fmt):
            return "n/a" if x is None else format(x, fmt)
        print(f"{e['benchmark']:13s} {e['method']:8s} seed{e['seed']}: "
              + (f"A={num(d.get('routed_accuracy'), '6.2f')} U={num(d.get('task_inference'), '6.2f')} "
                 f"V={num(d.get('conditional_accuracy'), '6.2f')} O={num(d.get('oracle_accuracy'), '6.2f')} "
                 f"rho={num(rho, '5.1f')}% " if d else "")
              + (f"stored-router-agreement={num(stored.get('route_agreement'), '.4f')} "
                 f"(near-ties {stored.get('near_tie_disagreements', 0)}) " if stored else "")
              + (f"refit-agreement={num(refit.get('route_agreement'), '.4f')} " if 'route_agreement' in refit else "")
              + ("OK" if not e["errors"] else "ERRORS: " + "; ".join(e["errors"][:3])))
    if status == "not-recorded":
        print("no job of this run was launched with --predictions-dir (per-example "
              "predictions are recorded since v3.0.5) -> not-recorded")
        return 0
    print(f"{report['files_checked']} predictions file(s) checked, {report['files_with_errors']} with errors -> {report['status']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
