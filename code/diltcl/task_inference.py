"""Score-based task inference for class-IL evaluation of isolation methods.

Parameter-isolation methods (DiLT-CL, PackNet, SupSup, WSN) train a separate
subnetwork per task and rely on the task identity at test time. To place them in
the same class-incremental (class-IL) comparison as the shared-backbone
baselines -- where *no* task identity is given -- we infer the task from the
input, using exhaustive score-based routing inspired by confidence/entropy
heuristics for task-conditioned subnetworks: the subnetwork of the correct
task tends to produce a distinctive confidence signature. We run the input
through every candidate subnetwork, route by a raw score or a leakage-safe
score-logistic calibrator, then read the selected subnetwork's prediction.

Design
------
The module is **method-agnostic**: a caller supplies a ``forward_t(x, t)``
closure that returns task ``t``'s logits for a batch ``x``; everything else
(scores, task selection, global-label mapping, metric accounting) lives here.
Closure builders for the project's isolation methods are provided at the bottom.

Two protocols are supported via ``disjoint_classes``:

* **Disjoint classes** (Split-CIFAR-100): task ``t`` owns global labels
  ``t*C .. t*C+C-1``; the global prediction is ``t_hat*C + local_argmax``.
* **Shared labels / domain-IL** (Permuted-MNIST): every task shares the same
  label set; the global prediction is just ``local_argmax`` of the inferred
  task's subnetwork (inference still matters -- the wrong permutation's
  subnetwork sees scrambled inputs and is usually wrong).

For each task's test set the evaluation reports class-IL accuracy, oracle-route
accuracy, task-inference accuracy, and accuracy conditioned on a correct route.
It also builds the genuine temporal
class-IL matrix: row ``i`` routes only among tasks ``0..i``.  Frozen subnetworks
do not imply a column-constant class-IL matrix because newly added routes can
steal inputs from earlier tasks.

The numeric selection/mapping is intentionally simple so it can be checked
against a NumPy reference in the tests.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Inference core
# ---------------------------------------------------------------------------

def task_scores(per_task_logits, criterion="entropy"):
    """Per-task, per-sample selection score (lower = more likely the task).

    Args:
        per_task_logits: list of ``T`` tensors, each ``[B, C]``.
        criterion: ``"entropy"`` (softmax entropy), ``"confidence"``
            (negative max-softmax-probability), or ``"margin"`` (negative
            top-two logit margin after per-sample centering/scaling).

    Returns:
        Tensor ``[T, B]`` of scores; the task is chosen by ``argmin``.
    """
    rows = []
    for logits in per_task_logits:
        if criterion == "entropy":
            logp = F.log_softmax(logits, dim=-1)
            rows.append(-(logp.exp() * logp).sum(dim=-1))        # H(p) >= 0
        elif criterion == "confidence":
            logp = F.log_softmax(logits, dim=-1)
            rows.append(-logp.exp().max(dim=-1).values)          # -max p
        elif criterion == "margin":
            # Entropy/confidence can be won merely by multiplying one task's
            # logits by a larger constant. Standardising within each sample
            # makes this top-two margin invariant to affine logit scale/shift.
            centred = logits - logits.mean(dim=-1, keepdim=True)
            scale = centred.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            z = centred / scale
            if z.shape[-1] < 2:
                rows.append(torch.zeros_like(z[..., 0]))
            else:
                top2 = z.topk(k=2, dim=-1).values
                rows.append(-(top2[..., 0] - top2[..., 1]))
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    return torch.stack(rows, dim=0)


def require_finite_scores(scores, context=""):
    """Raise a clear ``ValueError`` if any routing score is NaN or infinite.

    A non-finite score means an expert produced non-finite outputs. Routing on
    it would either pick task 0 silently (the raw ``argmin``) or index past
    the candidates (a calibrated router, a device-side assert on CUDA), so the
    evaluation stops here instead and names the experts concerned.
    """
    finite = torch.isfinite(scores)
    if bool(finite.all()):
        return
    per_task = (~finite).sum(dim=tuple(range(1, scores.dim()))).tolist()
    raise ValueError(
        f"non-finite routing scores{' ' + context if context else ''}: "
        f"{sum(per_task)} of {scores.numel()} (per candidate expert: {per_task}); "
        "an expert produced NaN or infinite outputs")


ROUTING_DTYPES = {"float64": torch.float64, "float32": torch.float32}


def calibrator_dtype(calibrator, scores_dtype):
    """Floating-point type in which a serialized router is applied.

    Routers fitted since v3.0.10 declare ``routing_dtype = "float64"``: they
    are fitted in float64 and applied in float64, so a deployed decision is
    the fitted router's decision. States without the field (written by
    earlier versions) were applied in the dtype of the scores, float32; they
    keep that precision so their recorded decisions re-evaluate exactly.
    """
    name = calibrator.get("routing_dtype")
    if name is None:
        return scores_dtype
    if name not in ROUTING_DTYPES:
        raise ValueError(f"unknown routing_dtype {name!r} in calibrator state")
    return ROUTING_DTYPES[name]


def infer_task(per_task_logits, criterion="entropy", calibrator=None,
               return_ties=False):
    """Infer a task ID for every sample.

    With no calibrator this is exactly the legacy ``argmin`` score rule.  A
    serialized calibrator instead standardizes the full score signature and
    applies its learned linear decision function on-device, in the
    precision the state declares (:func:`calibrator_dtype`; float64 for every
    router fitted since v3.0.10).

    Exact ties are resolved deterministically in favour of the lowest task ID.
    This makes the otherwise implicit early-task bias auditable.  With
    ``return_ties=True``, return ``(task_ids, tie_mask)`` where ``tie_mask`` is
    true iff at least two candidate tasks share the winning score exactly.
    Non-finite scores raise ``ValueError`` (:func:`require_finite_scores`).
    """
    scores = task_scores(per_task_logits, criterion)       # [T, B]
    require_finite_scores(scores)
    if calibrator is None:
        best = scores.min(dim=0, keepdim=True).values
        winning = scores == best
        # Candidate rows are ordered by task ID, so the first winning row is the
        # explicitly declared lowest-task-ID tie break.
        task_ids = winning.to(dtype=torch.long).argmax(dim=0)
        tie_mask = winning.sum(dim=0) > 1
    else:
        task_ids, tie_mask, _ = apply_calibrator(scores, calibrator)
    if return_ties:
        return task_ids, tie_mask
    return task_ids


def apply_calibrator(scores, calibrator):
    """Route ``[T, B]`` scores with a serialized router, on the scores' device.

    Returns ``(task_ids, tie_mask, decision)``: the lowest task ID among the
    exactly maximal decisions, whether two or more tied, and the ``[B, T]``
    decisions, computed in the precision the state declares
    (:func:`calibrator_dtype`).
    """
    dtype = calibrator_dtype(calibrator, scores.dtype)
    # Validated on the host: no device synchronisation per stage.
    values = [np.asarray(calibrator[key], dtype=np.float64) for key in
              ("feature_mean", "feature_scale", "coef", "intercept")]
    if (not all(np.isfinite(v).all() for v in values)
            or (values[1] <= 0).any()):
        raise ValueError("calibrator state contains non-finite values or a "
                         "non-positive scale")
    classes = torch.as_tensor(calibrator["classes"], device=scores.device,
                              dtype=torch.long)
    mean = torch.as_tensor(calibrator["feature_mean"], device=scores.device,
                           dtype=dtype)
    scale = torch.as_tensor(calibrator["feature_scale"], device=scores.device,
                            dtype=dtype)
    coef = torch.as_tensor(calibrator["coef"], device=scores.device,
                           dtype=dtype)
    intercept = torch.as_tensor(calibrator["intercept"], device=scores.device,
                                dtype=dtype)
    x = scores.transpose(0, 1).to(dtype)
    if (x.shape[1] != mean.numel()
            or coef.shape != (classes.numel(), x.shape[1])
            or scale.shape != mean.shape
            or intercept.shape != classes.shape
            or classes.numel() == 0
            or len(set(calibrator["classes"])) != classes.numel()):
        raise ValueError("calibrator state is incompatible with task scores")
    decision = ((x - mean) / scale) @ coef.transpose(0, 1) + intercept
    best = decision.max(dim=1, keepdim=True).values
    winning = decision == best
    # Serialized class order need not be sorted, so select the minimum task
    # ID among winning decision rows rather than merely the first row.
    class_grid = classes.unsqueeze(0).expand_as(decision)
    sentinel = torch.full_like(class_grid, torch.iinfo(torch.long).max)
    task_ids = torch.where(winning, class_grid, sentinel).min(dim=1).values
    tie_mask = winning.sum(dim=1) > 1
    return task_ids, tie_mask, decision


def float64_routing_self_test(device="cpu"):
    """Check on ``device`` what exact float64 routing rests on.

    :func:`apply_calibrator` applies every router fitted since v3.0.10 in
    float64. A bias router has one non-zero coefficient per decision, so each
    decision is a fixed sequence of correctly rounded operations (the
    standardisation, one product, one addition), which IEEE double arithmetic
    fixes uniquely: the device then reproduces NumPy's float64 decisions,
    routes and tie flags exactly (what the per-example audit requires),
    provided that

    1. widening the float32 routing scores to float64 is exact, subnormal
       scores included (saturated experts give entropies far below the
       float32 normal range; a conversion that flushes them to zero would
       create ties), and
    2. the float64 matrix product is IEEE double arithmetic, not an emulation
       (cuBLAS can emulate it on integer tensor cores when asked to, for
       example with ``CUBLAS_EMULATE_DOUBLE_PRECISION=1``).

    Both are checked here on fixed inputs: the widened scores bit for bit,
    and the decisions of diagonal routers (compared as numbers, so ``+0`` and
    ``-0`` are equal), their routes and their tie flags. No random state of
    PyTorch, NumPy or Python is used. Returns a JSON-ready dict with ``ok``.

    The same routers are also applied as states written before v3.0.10
    were, in float32, and compared with NumPy's float32 decisions
    (``float32_ok`` and the ``float32_*`` counts). This is reported, not part
    of ``ok``: only rebuilding float32-routed results relies on it
    (``tools/routing_precision_audit.py``), and a device that flushes
    float32 subnormals there fails it.
    """
    rng = np.random.default_rng(20261021)
    T, n = 6, 512
    magnitude = 10.0 ** rng.uniform(-46.0, 4.0, size=(n, T))
    scores = (np.abs(rng.standard_normal((n, T))) * magnitude).astype(np.float32)
    scores[rng.random((n, T)) < 0.25] = 0.0            # saturated experts
    scores[: n // 8] = scores[: n // 8, :1]            # rows of equal scores: ties
    subnormal = (scores != 0) & (np.abs(scores) < np.finfo(np.float32).tiny)
    report = {"device": str(device), "subnormal_scores": int(subnormal.sum()),
              "widening_mismatches": 0, "decisions_checked": 0,
              "decision_mismatches": 0, "route_mismatches": 0, "tie_mismatches": 0}
    on_device = torch.from_numpy(scores.T.copy()).to(device)       # [T, B]
    widened = on_device.to(torch.float64).cpu().numpy().T
    expected = scores.astype(np.float64)
    report["widening_mismatches"] = int(np.sum(
        widened.view(np.uint64) != expected.view(np.uint64)))
    tiny_bias = 1.1044052673094165e-29
    states = [
        # score-bias-acc layout: no scaling, coef = -I, biases that differ
        # only below float32 resolution (a float32 router would tie them).
        {"classes": list(range(T)), "feature_mean": [0.0] * T,
         "feature_scale": [1.0] * T, "coef": (-np.eye(T)).tolist(),
         "intercept": [tiny_bias * (1.0 + k * 2.0 ** -40) for k in range(T)]},
        # score-bias layout: standardised scores, one shared slope.
        {"classes": list(range(T)),
         "feature_mean": (rng.standard_normal(T) * 1e-3).tolist(),
         "feature_scale": (0.5 + rng.random(T)).tolist(),
         "coef": (-0.7310585786300049 * np.eye(T)).tolist(),
         "intercept": (rng.standard_normal(T) * 1e-2).tolist()},
        # all-zero biases: the exact ties of the raw rule are kept as ties.
        {"classes": list(range(T)), "feature_mean": [0.0] * T,
         "feature_scale": [1.0] * T, "coef": (-np.eye(T)).tolist(),
         "intercept": [0.0] * T},
    ]
    for prefix, dtype in (("", np.float64), ("float32_", np.float32)):
        for key in ("decisions_checked", "decision_mismatches", "route_mismatches",
                    "tie_mismatches"):
            report.setdefault(prefix + key, 0)
        for state in states:
            if dtype is np.float64:
                state = dict(state, routing_dtype="float64")
            task_ids, tie_mask, decision = apply_calibrator(on_device, state)
            mean = np.asarray(state["feature_mean"], dtype=dtype)
            scale = np.asarray(state["feature_scale"], dtype=dtype)
            slope = np.diag(np.asarray(state["coef"], dtype=dtype))
            intercept = np.asarray(state["intercept"], dtype=dtype)
            reference = ((scores.astype(dtype) - mean) / scale) * slope + intercept
            winning = reference == reference.max(axis=1, keepdims=True)
            route = winning.argmax(axis=1)
            report[prefix + "decisions_checked"] += int(reference.size)
            report[prefix + "decision_mismatches"] += int(
                np.sum(decision.cpu().numpy() != reference))
            report[prefix + "route_mismatches"] += int(np.sum(task_ids.cpu().numpy() != route))
            report[prefix + "tie_mismatches"] += int(
                np.sum(tie_mask.cpu().numpy() != (winning.sum(axis=1) > 1)))
    report["ok"] = bool(report["subnormal_scores"] > 0 and not any(
        report[k] for k in ("widening_mismatches", "decision_mismatches",
                            "route_mismatches", "tie_mismatches")))
    report["float32_ok"] = bool(report["subnormal_scores"] > 0 and not any(
        report["float32_" + k] for k in ("decision_mismatches", "route_mismatches",
                                         "tie_mismatches")))
    return report


def predict_global(per_task_logits, t_hat, offset):
    """Global class prediction from the inferred task's subnetwork.

    ``global = t_hat * offset + argmax(logits[t_hat])``. ``offset`` is the
    classes-per-task for the disjoint-class protocol, or ``0`` when labels are
    shared across tasks (domain-IL).
    """
    stacked = torch.stack(per_task_logits, dim=0)          # [T, B, C]
    batch = torch.arange(t_hat.shape[0], device=t_hat.device)
    chosen = stacked[t_hat, batch]                         # [B, C]
    local = chosen.argmax(dim=-1)                          # [B]
    return t_hat * offset + local


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class _PredictionRecorder:
    """Accumulate per-example routing outcomes during class-IL evaluation.

    Arrays (``N`` test examples in loader order, ``T`` tasks):

    * ``true_task`` [N] int16 and ``label_local`` [N] int16;
    * ``oracle_pred_local`` [N] int16 -- the true task's expert prediction;
    * ``scores`` [N, T] float32 -- raw criterion score of every candidate
      expert (the router's input; lower = more likely);
    * ``route_by_stage`` [N, T] int16 -- task chosen at prefix stage ``s``
      among candidates ``0..s`` (``-1`` for ``s < true_task``, where the
      example's task has not arrived yet);
    * ``tie_by_stage`` [N, T] bool -- exact score tie at that stage;
    * ``routed_pred_local_by_stage`` [N, T] int16 -- local class predicted
      by the chosen expert (``-1`` before the task arrives).

    For disjoint label sets, prefix-stage correctness is exactly
    ``route_by_stage[:, s] == true_task`` and ``oracle_pred_local ==
    label_local``, so every matrix the evaluator returns can be rebuilt from
    these arrays (``tools/recompute_from_predictions.py`` does so).
    """

    def __init__(self, num_tasks):
        self.num_tasks = int(num_tasks)
        self._chunks = {key: [] for key in (
            "true_task", "label_local", "oracle_pred_local", "scores",
            "route_by_stage", "tie_by_stage", "routed_pred_local_by_stage")}
        self._batch = None

    def begin_batch(self, true_t, y_local, oracle_local, scores):
        n = int(y_local.shape[0])
        self._batch = {
            "true_task": np.full(n, true_t, dtype=np.int16),
            "label_local": y_local.detach().cpu().numpy().astype(np.int16),
            "oracle_pred_local": oracle_local.detach().cpu().numpy().astype(np.int16),
            "scores": scores.transpose(0, 1).detach().float().cpu().numpy(),
            "route_by_stage": np.full((n, self.num_tasks), -1, dtype=np.int16),
            "tie_by_stage": np.zeros((n, self.num_tasks), dtype=bool),
            "routed_pred_local_by_stage": np.full(
                (n, self.num_tasks), -1, dtype=np.int16),
        }

    def record_stage(self, stage, t_hat, tied, pred_local):
        batch = self._batch
        batch["route_by_stage"][:, stage] = t_hat.detach().cpu().numpy()
        batch["tie_by_stage"][:, stage] = tied.detach().cpu().numpy()
        batch["routed_pred_local_by_stage"][:, stage] = \
            pred_local.detach().cpu().numpy()

    def end_batch(self):
        for key, value in self._batch.items():
            self._chunks[key].append(value)
        self._batch = None

    def arrays(self):
        if not self._chunks["true_task"]:
            return {key: np.zeros((0,) if key in ("true_task", "label_local",
                                                  "oracle_pred_local")
                                  else (0, self.num_tasks))
                    for key in self._chunks}
        return {key: np.concatenate(parts, axis=0)
                for key, parts in self._chunks.items()}


@torch.no_grad()
def evaluate_class_il_inference(forward_t, test_loaders, num_tasks,
                                classes_per_task, device="cuda",
                                criterion="entropy", disjoint_classes=True,
                                calibrators=None, record=None):
    """Evaluate an isolation method under class-IL using task inference.

    Args:
        forward_t: Callable ``(x, t) -> logits[B, C]`` for task ``t``.
        test_loaders: Per-task test DataLoaders with *task-local* labels
            (``0..C-1``); the true task is the loader's index.
        num_tasks: Number of tasks ``T``.
        classes_per_task: Classes per task ``C`` (the disjoint offset).
        device: Device to run on.
        criterion: Task-inference criterion (``"entropy"``, ``"confidence"``,
            or scale-invariant ``"margin"``).
        disjoint_classes: ``True`` for disjoint-class benchmarks (offset ``C``);
            ``False`` for shared-label / domain-IL benchmarks (offset ``0``).
        calibrators: Optional sequence of ``num_tasks`` score-logreg states.
            Entry ``i`` must be fitted only on held-out training samples and
            score dimensions from candidate tasks ``0..i``.
        record: Optional dict. When given, it is filled with per-example
            arrays (see :func:`_PredictionRecorder`) from the very same
            forward passes and routing decisions that produce the returned
            counts, so every reported statistic can be recomputed from them.
            Recording never changes a computed value.

    Returns:
        Dict containing macro task-average ``class_il_acc``,
        ``oracle_route_acc`` and ``task_inference_acc``; micro-pooled variants;
        per-task final accuracies; and the prefix-candidate ``result_matrix``
        plus matching ``task_inference_matrix`` and ``routing_tie_matrix``.
        ``routing_tie_rate`` is the macro final tie rate and
        ``micro_routing_tie_rate`` is sample-pooled. ``conditional_class_acc``
        is deliberately pooled over all correctly routed samples, so tasks with
        zero correct routes cannot be silently excluded; its
        ``micro_conditional_class_acc`` alias is retained for compatibility.
        ``task_il_acc`` remains a compatibility alias for ``oracle_route_acc``.
    """
    if calibrators is not None and len(calibrators) != num_tasks:
        raise ValueError(
            f"expected {num_tasks} prefix calibrators, got {len(calibrators)}")
    offset = classes_per_task if disjoint_classes else 0
    counts = np.zeros(num_tasks, dtype=np.int64)
    oracle_hits = np.zeros(num_tasks, dtype=np.int64)
    cls_hits = np.zeros((num_tasks, num_tasks), dtype=np.int64)
    ta_hits = np.zeros((num_tasks, num_tasks), dtype=np.int64)
    tie_hits = np.zeros((num_tasks, num_tasks), dtype=np.int64)
    correct_route_class_hits = np.zeros((num_tasks, num_tasks), dtype=np.int64)

    recorder = _PredictionRecorder(num_tasks) if record is not None else None
    for true_t, loader in enumerate(test_loaders):
        for x, y_local in loader:
            x, y_local = x.to(device), y_local.to(device)
            per_task = [forward_t(x, t) for t in range(num_tasks)]  # T x [B, C]
            n = y_local.size(0)
            counts[true_t] += n
            oracle_local = per_task[true_t].argmax(dim=-1)
            oracle_hits[true_t] += (oracle_local == y_local).sum().item()
            g_true = true_t * offset + y_local
            batch_scores = task_scores(per_task, criterion)
            require_finite_scores(
                batch_scores, f"for test examples of task {true_t}")
            if recorder is not None:
                recorder.begin_batch(true_t, y_local, oracle_local, batch_scores)

            # One set of T forward passes suffices for every historical stage:
            # stage i simply ignores candidates i+1..T-1.
            for stage in range(true_t, num_tasks):
                candidates = per_task[:stage + 1]
                calibrator = None if calibrators is None else calibrators[stage]
                t_hat, tied = infer_task(
                    candidates, criterion, calibrator, return_ties=True)
                g_pred = predict_global(candidates, t_hat, offset)
                cls_hits[stage, true_t] += (g_pred == g_true).sum().item()
                ta_hits[stage, true_t] += (t_hat == true_t).sum().item()
                tie_hits[stage, true_t] += tied.sum().item()
                correct_route_class_hits[stage, true_t] += (
                    (g_pred == g_true) & (t_hat == true_t)).sum().item()
                if recorder is not None:
                    recorder.record_stage(stage, t_hat, tied,
                                          g_pred - t_hat * offset)
            if recorder is not None:
                recorder.end_batch()

    result_matrix = np.zeros((num_tasks, num_tasks), dtype=float)
    task_inference_matrix = np.zeros((num_tasks, num_tasks), dtype=float)
    routing_tie_matrix = np.zeros((num_tasks, num_tasks), dtype=float)
    conditional_class_matrix = np.zeros((num_tasks, num_tasks), dtype=float)
    oracle_per_task = np.zeros(num_tasks, dtype=float)
    for true_t in range(num_tasks):
        if counts[true_t] == 0:
            continue
        oracle_per_task[true_t] = 100.0 * oracle_hits[true_t] / counts[true_t]
        for stage in range(true_t, num_tasks):
            result_matrix[stage, true_t] = (
                100.0 * cls_hits[stage, true_t] / counts[true_t])
            task_inference_matrix[stage, true_t] = (
                100.0 * ta_hits[stage, true_t] / counts[true_t])
            routing_tie_matrix[stage, true_t] = (
                100.0 * tie_hits[stage, true_t] / counts[true_t])
            if ta_hits[stage, true_t] > 0:
                conditional_class_matrix[stage, true_t] = (
                    100.0 * correct_route_class_hits[stage, true_t]
                    / ta_hits[stage, true_t])

    nonempty = counts > 0
    final_per_task = result_matrix[-1].tolist() if num_tasks else []
    final_ta_per_task = task_inference_matrix[-1] if num_tasks else np.array([])
    final_tie_per_task = routing_tie_matrix[-1] if num_tasks else np.array([])
    macro = lambda a: float(np.mean(np.asarray(a)[nonempty])) if nonempty.any() else 0.0  # noqa: E731
    total = int(counts.sum())
    micro_cls = (100.0 * cls_hits[-1].sum() / total
                 if total and num_tasks else 0.0)
    micro_oracle = 100.0 * oracle_hits.sum() / total if total else 0.0
    micro_ta = (100.0 * ta_hits[-1].sum() / total
                if total and num_tasks else 0.0)
    micro_tie = (100.0 * tie_hits[-1].sum() / total
                 if total and num_tasks else 0.0)
    final_route_hits = ta_hits[-1] if num_tasks else np.array([])
    pooled_conditional = (
        100.0 * correct_route_class_hits[-1].sum() / final_route_hits.sum()
        if num_tasks and final_route_hits.sum() else 0.0)
    zero_correct_route_task_count = int(
        np.sum(nonempty & (final_route_hits == 0))) if num_tasks else 0
    oracle_macro = macro(oracle_per_task)
    if recorder is not None:
        record.update(recorder.arrays())
        record["criterion"] = criterion
        record["classes_per_task"] = int(classes_per_task)
        record["disjoint_classes"] = bool(disjoint_classes)
        record["calibrated"] = calibrators is not None
    return {
        "class_il_acc": macro(result_matrix[-1]) if num_tasks else 0.0,
        "micro_class_il_acc": micro_cls,
        "oracle_route_acc": oracle_macro,
        "task_il_acc": oracle_macro,
        "micro_oracle_route_acc": micro_oracle,
        "task_inference_acc": macro(final_ta_per_task),
        "micro_task_inference_acc": micro_ta,
        "routing_tie_rate": macro(final_tie_per_task),
        "micro_routing_tie_rate": micro_tie,
        "conditional_class_acc": pooled_conditional,
        "micro_conditional_class_acc": pooled_conditional,
        "zero_correct_route_task_count": zero_correct_route_task_count,
        "per_task_class_il_acc": final_per_task,
        "per_task_oracle_route_acc": oracle_per_task.tolist(),
        "per_task_routing_tie_rate": (
            final_tie_per_task.tolist() if num_tasks else []),
        "result_matrix": result_matrix,
        "task_inference_matrix": task_inference_matrix,
        "routing_tie_matrix": routing_tie_matrix,
        "conditional_class_matrix": conditional_class_matrix,
        "task_sample_counts": counts.tolist(),
    }


@torch.no_grad()
def collect_calibration_features(forward_t, calibration_loaders, num_tasks,
                                 device="cuda", criterion="entropy"):
    """Collect score signatures from held-out training examples.

    Local class labels are intentionally ignored.  The task label is the
    loader index, so neither test data nor test labels enter the calibration.
    """
    features = []
    task_ids = []
    counts = []
    for true_t, loader in enumerate(calibration_loaders):
        task_count = 0
        for x, _ in loader:
            x = x.to(device)
            logits = [forward_t(x, t) for t in range(num_tasks)]
            raw_scores = task_scores(logits, criterion)
            require_finite_scores(
                raw_scores, f"for held-out calibration examples of task {true_t}")
            batch_scores = raw_scores.transpose(0, 1)
            features.append(batch_scores.detach().cpu().numpy())
            task_ids.append(np.full(x.shape[0], true_t, dtype=np.int64))
            task_count += int(x.shape[0])
        counts.append(task_count)
    if not features:
        raise ValueError("calibration loaders contain no samples")
    return np.concatenate(features, axis=0), np.concatenate(task_ids), counts


# ---------------------------------------------------------------------------
# Per-method forward closures
# ---------------------------------------------------------------------------

def supsup_forward_fn(model):
    """Forward closure for SupSup / WSN (a model with masked layers)."""
    from .masked_layers import set_task

    model.eval()

    def forward_t(x, t):
        set_task(model, t)
        return model(x)

    return forward_t


# WSN uses the same masked-layer model interface as SupSup.
wsn_forward_fn = supsup_forward_fn


def packnet_forward_fn(model, eval_masks):
    """Forward closure for PackNet (a packed model + per-task eval masks)."""
    from . import masking

    model.eval()

    def forward_t(x, t):
        return masking.masked_forward(model, x, eval_masks[t])

    return forward_t


def diltcl_forward_fn(diltcl):
    """Forward closure for DiLT-CL (its per-task masked forward)."""
    diltcl.model.eval()

    def forward_t(x, t):
        return diltcl.forward_with_mask(x, t)

    return forward_t
