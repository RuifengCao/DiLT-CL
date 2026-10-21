"""Leakage-safe calibration of task-inference scores.

The raw entropy/confidence rule compares one scalar score from every task
subnetwork.  Those scores need not be calibrated across independently trained
subnetworks.  This module fits a small multinomial logistic classifier to the
*full score signature* using only a held-out part of the training data.

The implementation is NumPy-first so fitting, validation, and serialization can
be tested without PyTorch.  Every returned state contains plain Python lists and
scalars and can therefore be written directly to the experiment JSON.

Every state fitted here records ``routing_dtype = "float64"``: the router is
fitted in float64 and :func:`diltcl.task_inference.infer_task` applies it in
float64 as well, so the deployed decisions are those of the fitted router (for
the two bias routers exactly, since each decision is a fixed sequence of
correctly rounded operations: the standardisation, one product and one
addition). States without the field were written before v3.0.10 and were
applied in the scores' float32; ``infer_task`` keeps doing so for them, so
their recorded decisions can still be re-evaluated exactly.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np


def deterministic_holdout_indices(lengths, fraction, split_seed):
    """Return disjoint model-train/calibration indices for every task.

    This pure helper makes the data boundary independently testable. Each task
    receives at least one example on both sides; task-specific RNG streams keep
    the split stable if unrelated tasks are added later.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between 0 and 1")
    splits = []
    for task_id, raw_n in enumerate(lengths):
        n = int(raw_n)
        if n != raw_n or n < 2:
            raise ValueError(
                f"task {task_id} needs an integer length of at least two")
        rng = np.random.default_rng(int(split_seed) + task_id)
        order = rng.permutation(n)
        n_cal = min(n - 1, max(1, int(round(fraction * n))))
        splits.append({
            "train": order[n_cal:].astype(int).tolist(),
            "calibration": order[:n_cal].astype(int).tolist(),
        })
    return splits


def _validated_xy(features, task_ids):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(task_ids)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be a non-empty 2-D array")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("task_ids must be one-dimensional and match features")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    if not np.issubdtype(y.dtype, np.integer):
        if not np.equal(y, np.floor(y)).all():
            raise ValueError("task_ids must contain integer task labels")
    y = y.astype(np.int64, copy=False)
    if (y < 0).any():
        raise ValueError("task_ids must be non-negative")
    return x, y


ROUTING_DTYPE = "float64"


def balanced_accuracy(pred, y):
    """Mean over the tasks present in ``y`` of the fraction routed correctly.

    This is the class-balanced accuracy that score-bias-acc maximises; as a
    fraction it equals plain accuracy when every task has the same number of
    examples (the two floats can differ in the last bit).
    """
    pred = np.asarray(pred)
    y = np.asarray(y)
    tasks = np.unique(y)
    return float(np.mean([np.mean(pred[y == t] == t) for t in tasks]))


def balanced_accuracy_exact(pred, y):
    """:func:`balanced_accuracy` as an exact fraction, without rounding.

    Two routers whose float balanced accuracies differ in the last bit can
    still be exactly as accurate; comparisons that must not depend on
    rounding use this value.
    """
    pred = np.asarray(pred)
    y = np.asarray(y)
    tasks = np.unique(y)
    if not tasks.size:
        return Fraction(0)
    total = sum((Fraction(int(np.sum(pred[y == t] == t)), int(np.sum(y == t)))
                 for t in tasks), Fraction(0))
    return total / int(tasks.size)


def _sklearn_available():
    """Return ``True`` when scikit-learn can be imported."""
    import importlib.util
    try:
        return importlib.util.find_spec("sklearn") is not None
    except (ImportError, ValueError):  # pragma: no cover - exotic import state
        return False


def _balanced_weights(y, classes):
    """Per-sample weights matching scikit-learn's ``class_weight='balanced'``.

    ``w_i = n_samples / (n_classes * count[y_i])``.
    """
    counts = np.array([np.sum(y == c) for c in classes], dtype=np.float64)
    per_class = y.shape[0] / (classes.size * counts)
    index = {int(c): i for i, c in enumerate(classes)}
    return np.array([per_class[index[int(label)]] for label in y],
                    dtype=np.float64)


def _fit_sklearn(z, y, classes, C, max_iter, random_state):
    """Fit with scikit-learn's L-BFGS logistic regression."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=float(C), class_weight="balanced", solver="lbfgs",
        max_iter=int(max_iter), random_state=int(random_state),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(z, y)
    converged = not any(issubclass(w.category, ConvergenceWarning)
                        for w in caught)
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    # sklearn represents binary logistic regression with one decision row.
    # Expanding it to two rows makes prediction a uniform argmax operation
    # and keeps the state independent of sklearn at inference time.
    if classes.size == 2 and coef.shape[0] == 1:
        coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
        intercept = np.array([0.0, intercept[0]], dtype=np.float64)
    return coef, intercept, np.asarray(model.n_iter_, dtype=int).tolist(), \
        bool(converged)


def _fit_scipy(z, y, classes, C, max_iter):
    """Fit scikit-learn's objective with SciPy L-BFGS-B (no scikit-learn needed).

    For three or more classes this is the multinomial (softmax) objective of
    scikit-learn's ``lbfgs`` solver::

        0.5 * ||W||^2 + C * sum_i w_i * (-log p_{y_i}(z_i))

    with balanced sample weights ``w_i`` and an unregularised intercept.

    For **two** classes scikit-learn does not fit a symmetric two-row softmax:
    it fits a single decision vector ``w`` with the binary logistic loss and
    penalises ``0.5 * ||w||^2``. A two-row softmax with the same ``C`` splits
    the same decision into ``+-w/2`` and therefore penalises only half as
    much, i.e. it behaves like the binary fit with ``2C``. (A retained
    two-task prefix probe documented this discrepancy for the pre-3.0.4
    fallback.) The binary case is therefore fitted in scikit-learn's own
    parameterisation and expanded to the two-row state afterwards, exactly as
    :func:`_fit_sklearn` does.
    """
    from scipy.optimize import minimize

    n_features = z.shape[1]
    n_classes = classes.size
    index = {int(c): i for i, c in enumerate(classes)}
    targets = np.array([index[int(label)] for label in y], dtype=int)
    sample_weight = _balanced_weights(y, classes)
    penalty = 1.0 / float(C)

    if n_classes == 2:
        sign = np.where(targets == 1, 1.0, -1.0)

        def objective_binary(theta):
            w, b = theta[:n_features], theta[n_features]
            margin = sign * (z @ w + b)
            # log(1 + exp(-margin)) computed stably
            loss_terms = np.logaddexp(0.0, -margin)
            loss = np.sum(sample_weight * loss_terms) + 0.5 * penalty * np.dot(w, w)
            sigma = 1.0 / (1.0 + np.exp(margin))       # d/dmargin log(1+e^-m) = -sigma
            g = -sample_weight * sigma * sign
            return loss, np.concatenate([z.T @ g + penalty * w, [g.sum()]])

        theta0 = np.zeros(n_features + 1, dtype=np.float64)
        result = minimize(objective_binary, theta0, jac=True, method="L-BFGS-B",
                          options={"maxiter": int(max_iter), "gtol": 1e-8,
                                   "ftol": 1e-14})
        w, b = result.x[:n_features], result.x[n_features]
        coef = np.vstack([np.zeros_like(w), w])
        intercept = np.array([0.0, b], dtype=np.float64)
        return coef, intercept, [int(result.nit)], bool(result.success)

    onehot = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    onehot[np.arange(y.shape[0]), targets] = 1.0

    def unpack(theta):
        coef = theta[: n_classes * n_features].reshape(n_classes, n_features)
        return coef, theta[n_classes * n_features:]

    def objective(theta):
        coef, intercept = unpack(theta)
        scores = z @ coef.T + intercept
        scores -= scores.max(axis=1, keepdims=True)      # numerical stability
        logsumexp = np.log(np.exp(scores).sum(axis=1))
        log_prob = scores - logsumexp[:, None]
        loss = -np.sum(sample_weight * log_prob[np.arange(y.shape[0]), targets])
        loss += 0.5 * penalty * np.sum(coef * coef)
        prob = np.exp(log_prob)
        diff = (prob - onehot) * sample_weight[:, None]
        grad_coef = diff.T @ z + penalty * coef
        grad_intercept = diff.sum(axis=0)
        return loss, np.concatenate([grad_coef.ravel(), grad_intercept])

    theta0 = np.zeros(n_classes * n_features + n_classes, dtype=np.float64)
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                      options={"maxiter": int(max_iter), "gtol": 1e-8,
                               "ftol": 1e-14})
    coef, intercept = unpack(result.x)
    return coef, intercept, [int(result.nit)], bool(result.success)


def fit_score_logreg(features, task_ids, *, C=1.0, random_state=0,
                     max_iter=2000):
    """Fit a standardized, class-balanced logistic task classifier.

    Args:
        features: ``[N, T]`` task-score signatures. Lower raw values can still
            be more task-like; logistic regression learns the direction.
        task_ids: ``[N]`` true task IDs from held-out *training* examples.
        C: Inverse L2-regularization strength used by scikit-learn.
        random_state: Deterministic solver seed.
        max_iter: Maximum solver iterations.

    Returns:
        A JSON-serializable state accepted by :func:`predict_score_logreg`.
    """
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be a positive finite number")
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be positive")
    x, y = _validated_xy(features, task_ids)
    classes, counts = np.unique(y, return_counts=True)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    z = (x - mean) / scale

    if classes.size == 1:
        coef = np.zeros((1, x.shape[1]), dtype=np.float64)
        intercept = np.zeros(1, dtype=np.float64)
        n_iter = [0]
        converged = True
        backend = "degenerate-single-class"
    elif _sklearn_available():
        coef, intercept, n_iter, converged = _fit_sklearn(
            z, y, classes, C, max_iter, random_state)
        backend = "sklearn-lbfgs"
    else:
        # No hard scikit-learn dependency: the identical penalised objective is
        # minimised with SciPy's L-BFGS-B. Losing an optional package must not
        # destroy a completed training run -- the v2.1 P0 suite lost every
        # calibrated job exactly that way, after all training had finished.
        coef, intercept, n_iter, converged = _fit_scipy(
            z, y, classes, C, max_iter)
        backend = "scipy-lbfgs"

    return {
        "kind": "score-logreg",
        "solver_backend": backend,
        "routing_dtype": ROUTING_DTYPE,
        "classes": classes.astype(int).tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coef": coef.tolist(),
        "intercept": intercept.tolist(),
        "C": float(C),
        "random_state": int(random_state),
        "max_iter": int(max_iter),
        "n_iter": n_iter,
        "converged": bool(converged),
        "sample_count": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "class_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},
    }


def fit_score_bias(features, task_ids, *, C=1.0, max_iter=2000):
    """Fit a tied-slope, per-task-bias router: ``logit_k = -a * z_k + b_k``.

    This is a restricted likelihood-fitted family: one shared slope over the
    standardized score of each candidate plus one bias per task, ``T + 1``
    parameters instead of the ``T * T + T`` of the full score-logreg. Because
    each score column is standardized separately and the shared slope is
    unconstrained, setting ``b = 0`` does not generally recover the raw
    unstandardized ``argmin`` rule. The fit therefore has no nondegradation
    guarantee for routing accuracy on either its fitting inputs or test data.

    The state uses the score-logreg layout (``coef = -a * I``, ``intercept =
    b``) so :func:`predict_score_logreg` and the on-device path apply it
    unchanged. ``b`` is L2-regularised with strength ``1 / C``; the slope is
    unregularised.
    """
    from scipy.optimize import minimize

    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be a positive finite number")
    x, y = _validated_xy(features, task_ids)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size != x.shape[1] or not np.array_equal(classes, np.arange(classes.size)):
        raise ValueError("score-bias calibration needs one score column per task 0..T-1")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    z = (x - mean) / scale
    n_classes = classes.size
    targets = y.astype(int)
    onehot = np.zeros((y.shape[0], n_classes))
    onehot[np.arange(y.shape[0]), targets] = 1.0
    sample_weight = _balanced_weights(y, classes)
    penalty = 1.0 / float(C)

    def objective(theta):
        a, b = theta[0], theta[1:]
        scores = -a * z + b
        scores -= scores.max(axis=1, keepdims=True)
        log_prob = scores - np.log(np.exp(scores).sum(axis=1))[:, None]
        loss = -np.sum(sample_weight * log_prob[np.arange(y.shape[0]), targets])
        loss += 0.5 * penalty * np.sum(b * b)
        prob = np.exp(log_prob)
        diff = (prob - onehot) * sample_weight[:, None]
        grad_a = -np.sum(diff * z)
        grad_b = diff.sum(axis=0) + penalty * b
        return loss, np.concatenate([[grad_a], grad_b])

    theta0 = np.concatenate([[1.0], np.zeros(n_classes)])
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                      options={"maxiter": int(max_iter), "gtol": 1e-8,
                               "ftol": 1e-14})
    a, b = float(result.x[0]), result.x[1:]
    return {
        "kind": "score-bias",
        "solver_backend": "scipy-lbfgs",
        "routing_dtype": ROUTING_DTYPE,
        "classes": classes.astype(int).tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coef": (-a * np.eye(n_classes)).tolist(),
        "intercept": b.tolist(),
        "slope": a,
        "C": float(C),
        "max_iter": int(max_iter),
        "n_iter": [int(result.nit)],
        "converged": bool(result.success),
        "sample_count": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "class_counts": {str(int(c)): int(n) for c, n in zip(classes, counts)},
    }


def fit_score_bias_acc(features, task_ids, *, max_sweeps=8, grid_size=41):
    """Per-task bias router fitted for *routing accuracy*, not likelihood.

    Decision ``argmin_k (x_k - b_k)`` on the raw scores. ``b = 0`` is the raw
    rule and is the starting point, so the fitted router can never be worse
    than ``argmin`` on its own fitting data in class-balanced accuracy, the
    objective. :func:`fit_prefix_calibrators` records that quantity as
    ``fit_balanced_accuracy`` next to the plain ``fit_accuracy`` (the same
    fraction when every task contributes the same number of examples; the
    stored floats can differ in the last bit). The
    search evaluates decisions in float64, the precision in which
    ``infer_task`` applies the router (``routing_dtype``), so the recorded
    accuracies describe the deployed router exactly. With a single candidate
    there is nothing to fit: the state is the raw rule, reported like the
    single-class score-logreg fit (``converged`` True, zero sweeps,
    ``solver_backend = "degenerate-single-class"``). Each bias is chosen by
    coordinate ascent over a grid of candidate offsets (quantiles of the
    score gaps), keeping the smallest ``|b_k|`` among ties. The objective is
    compared exactly, in integer arithmetic, so a move is accepted only if
    it raises the balanced accuracy or keeps it exactly and shrinks
    ``|b_k|`` (with equal task counts this is the same search as before
    v3.0.10; with unequal counts float rounding no longer moves biases
    without a gain). With ``T``
    parameters and a monotone objective, it is the most conservative
    correction that can still fix a systematically over- or under-confident
    expert. Motivated by the P0 finding that both likelihood-fitted routers
    lowered task-inference accuracy on CIFAR.

    The state uses the score-logreg layout (no scaling, ``coef = -I``,
    ``intercept = b``), so :func:`predict_score_logreg` and the on-device
    path apply it unchanged.
    """
    x, y = _validated_xy(features, task_ids)
    classes = np.unique(y)
    if classes.size != x.shape[1] or not np.array_equal(classes, np.arange(classes.size)):
        raise ValueError("score-bias-acc calibration needs one score column per task 0..T-1")
    n, T = x.shape
    b = np.zeros(T)
    counts = np.bincount(y, minlength=T)
    state = {
        "kind": "score-bias-acc",
        "solver_backend": "coordinate-ascent",
        "routing_dtype": ROUTING_DTYPE,
        "classes": classes.astype(int).tolist(),
        "feature_mean": [0.0] * T,
        "feature_scale": [1.0] * T,
        "coef": (-np.eye(T)).tolist(),
        "slope": 1.0,
        "C": None,
        "max_iter": int(max_sweeps),
        "sample_count": int(n),
        "feature_count": int(T),
        "class_counts": {str(int(c)): int(counts[c]) for c in classes},
    }
    if T == 1:          # a single candidate: nothing to route, nothing to fit
        state.update({"solver_backend": "degenerate-single-class",
                      "intercept": [0.0], "n_iter": [0], "converged": True})
        return state

    # The objective, the class-balanced accuracy sum_t hits_t / n_t, is kept
    # as an exact integer (scaled by the least common multiple of the task
    # counts). Per-example float weights summed in float64 carry rounding
    # noise when the task counts differ, which can make an exactly equal
    # objective look higher and accept a move that gains nothing. With equal
    # task counts every weight is one: the objective is the number of correct
    # routes, as it always was.
    common = math.lcm(*(int(c) for c in counts))
    task_weight = [common // int(c) for c in counts]

    def objective(bias):
        decision = (x - bias[None, :])
        # lowest task id wins exact ties, as in infer_task
        pred = np.argmin(decision, axis=1)
        hits = np.bincount(y[pred == y], minlength=T).tolist()
        return sum(int(h) * w for h, w in zip(hits, task_weight))

    # Candidate offsets: quantiles of "this candidate's score minus the best
    # other candidate's score" -- the amounts by which shifting b_k changes
    # decisions -- plus zero.
    best = objective(b)
    sweeps = 0
    improved = True
    while improved and sweeps < max_sweeps:
        improved = False
        sweeps += 1
        for k in range(T):
            others = np.delete(x - b[None, :], k, axis=1).min(axis=1)
            gaps = (x[:, k] - b[k]) - others
            candidates = np.unique(np.concatenate([
                [0.0], np.quantile(gaps, np.linspace(0.02, 0.98, grid_size))]))
            best_k, best_val = b[k], best
            for delta in candidates:
                trial = b.copy()
                trial[k] = b[k] + delta
                val = objective(trial)
                if val > best_val or (val == best_val and abs(trial[k]) < abs(best_k)):
                    best_k, best_val = trial[k], val
            if best_k != b[k]:
                b[k] = best_k
                if best_val > best:
                    improved = True
                best = best_val
    state.update({"intercept": b.tolist(), "n_iter": [int(sweeps)],
                  "converged": bool(not improved)})
    return state


CALIBRATION_KINDS = ("score-logreg", "score-bias", "score-bias-acc")


def predict_score_logreg(features, state):
    """Predict task IDs from score signatures using a serialized state."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("features must be a finite 2-D array")
    classes = np.asarray(state["classes"], dtype=np.int64)
    mean = np.asarray(state["feature_mean"], dtype=np.float64)
    scale = np.asarray(state["feature_scale"], dtype=np.float64)
    coef = np.asarray(state["coef"], dtype=np.float64)
    intercept = np.asarray(state["intercept"], dtype=np.float64)
    p = x.shape[1]
    if (classes.size == 0 or mean.shape != (p,) or scale.shape != (p,) or
            coef.shape != (classes.size, p) or
            intercept.shape != (classes.size,)):
        raise ValueError("calibrator state has incompatible shapes")
    if (scale <= 0).any() or not all(np.isfinite(a).all()
                                    for a in (mean, scale, coef, intercept)):
        raise ValueError("calibrator state contains invalid values")
    decision = ((x - mean) / scale) @ coef.T + intercept
    best = decision.max(axis=1, keepdims=True)
    winning = decision == best
    # Keep the artifact-wide exact-tie contract even if a hand-authored state
    # presents classes out of order: select the lowest task ID, not merely the
    # first decision column.
    sentinel = np.iinfo(np.int64).max
    return np.where(winning, classes[None, :], sentinel).min(axis=1)


def fit_prefix_calibrators(features, task_ids, num_tasks, *, C=1.0,
                           random_state=0, max_iter=2000, kind="score-logreg"):
    """Fit one leakage-safe router for every candidate-task prefix.

    Stage ``i`` is fitted only from held-out training samples of tasks
    ``0..i`` and only sees score dimensions ``0..i``.  This matches the prefix
    semantics of the class-IL result matrix and prevents future-task labels or
    scores from leaking into earlier diagonal entries. ``kind`` selects the
    full score-logreg router or the tied-slope score-bias router.
    """
    if kind not in CALIBRATION_KINDS:
        raise ValueError(f"unknown calibration kind {kind!r}; choose from {CALIBRATION_KINDS}")
    x, y = _validated_xy(features, task_ids)
    if int(num_tasks) <= 0 or x.shape[1] < int(num_tasks):
        raise ValueError("num_tasks must be positive and covered by features")
    states = []
    for stage in range(int(num_tasks)):
        keep = y <= stage
        expected = set(range(stage + 1))
        observed = set(np.unique(y[keep]).tolist())
        if observed != expected:
            raise ValueError(
                f"calibration data missing tasks at stage {stage}: "
                f"expected {sorted(expected)}, got {sorted(observed)}")
        x_stage, y_stage = x[keep, :stage + 1], y[keep]
        if kind == "score-bias":
            state = fit_score_bias(x_stage, y_stage, C=C, max_iter=max_iter)
        elif kind == "score-bias-acc":
            state = fit_score_bias_acc(x_stage, y_stage)
        else:
            state = fit_score_logreg(
                x_stage, y_stage, C=C,
                random_state=int(random_state) + stage, max_iter=max_iter,
            )
        state["stage"] = stage
        # Diagnostics on the fitting data itself: how often the fitted router
        # and the raw argmin rule pick the right task on the held-out
        # calibration samples. Comparing these with the test-time TA tells a
        # router that cannot fit (low fit accuracy) from one that fits the
        # holdout but does not transfer (fit accuracy high, test TA low).
        # The fitted router's decisions are computed in float64, as
        # infer_task applies it (routing_dtype). Both plain and class-balanced
        # accuracies are recorded; score-bias-acc maximises the balanced one.
        fitted = predict_score_logreg(x_stage, state)
        raw = np.argmin(x_stage, axis=1)
        state["fit_accuracy"] = float(np.mean(fitted == y_stage))
        state["raw_argmin_fit_accuracy"] = float(np.mean(raw == y_stage))
        state["fit_balanced_accuracy"] = balanced_accuracy(fitted, y_stage)
        state["raw_argmin_fit_balanced_accuracy"] = balanced_accuracy(raw, y_stage)
        states.append(state)
    return states
