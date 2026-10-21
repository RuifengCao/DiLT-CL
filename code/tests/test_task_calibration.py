"""Torch-free production tests for leakage-safe score calibration."""

import json
import pathlib
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "diltcl"))
import task_calibration as tc  # noqa: E402
from task_calibration import (  # noqa: E402
    deterministic_holdout_indices,
    fit_prefix_calibrators,
    fit_score_logreg,
    predict_score_logreg,
)


def test_holdout_split_is_deterministic_disjoint_and_complete():
    a = deterministic_holdout_indices([11, 20], 0.2, 19)
    b = deterministic_holdout_indices([11, 20], 0.2, 19)
    c = deterministic_holdout_indices([11, 20], 0.2, 20)
    assert a == b and a != c
    for n, split in zip([11, 20], a):
        train, cal = set(split["train"]), set(split["calibration"])
        assert train.isdisjoint(cal)
        assert train | cal == set(range(n))
        assert train and cal


def test_scale_biased_scores_are_corrected():
    # Raw argmin always chooses column 0. Cross-score signatures nevertheless
    # separate the true tasks and a held-out classifier recovers them.
    rng = np.random.default_rng(7)
    y = np.repeat([0, 1], 60)
    x = np.column_stack([
        -20.0 + 0.2 * rng.normal(size=y.size),
        np.where(y == 0, 4.0, -4.0) + 0.2 * rng.normal(size=y.size),
    ])
    assert np.mean(x.argmin(axis=1) == y) == 0.5
    state = fit_score_logreg(x, y, C=1.0, random_state=3)
    assert state["kind"] == "score-logreg"
    pred = predict_score_logreg(x, state)
    assert np.mean(pred == y) > 0.99


def test_constant_feature_and_serializable_state():
    x = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 8.0], [1.0, 9.0]])
    y = np.array([0, 0, 1, 1])
    state = fit_score_logreg(x, y)
    state = json.loads(json.dumps(state, allow_nan=False))
    assert state["feature_scale"][0] == 1.0
    assert state["converged"]
    assert predict_score_logreg(x, state).tolist() == y.tolist()
    tie_state = {
        "classes": [2, 0], "feature_mean": [0.0], "feature_scale": [1.0],
        "coef": [[0.0], [0.0]], "intercept": [0.0, 0.0],
    }
    assert predict_score_logreg([[4.0]], tie_state).tolist() == [0]


def test_single_task_prefix_is_well_defined():
    x = np.array([[0.2], [0.4], [0.1]])
    y = np.zeros(3, dtype=int)
    state = fit_score_logreg(x, y)
    assert predict_score_logreg([[100.0], [-100.0]], state).tolist() == [0, 0]


def test_prefix_fit_uses_no_future_dimensions_or_labels():
    x = np.array([
        [0.1, 9.0, 99.0], [0.2, 8.0, 98.0],
        [9.0, 0.1, 97.0], [8.0, 0.2, 96.0],
        [9.0, 8.0, 0.1], [8.0, 9.0, 0.2],
    ])
    y = np.repeat([0, 1, 2], 2)
    states = fit_prefix_calibrators(x, y, 3, random_state=11)
    assert [s["feature_count"] for s in states] == [1, 2, 3]
    assert [s["sample_count"] for s in states] == [2, 4, 6]
    assert states[1]["classes"] == [0, 1]


def test_missing_prefix_task_is_rejected():
    x = np.ones((4, 3))
    y = np.array([0, 0, 2, 2])
    try:
        fit_prefix_calibrators(x, y, 3)
    except ValueError as exc:
        assert "missing tasks" in str(exc)
    else:
        raise AssertionError("missing calibration task was accepted")


def test_invalid_inputs_are_rejected():
    cases = [
        (np.array([[np.nan, 0.0]]), np.array([0])),
        (np.ones((2, 2)), np.array([0])),
        (np.ones((2, 2)), np.array([0.5, 1.0])),
    ]
    for x, y in cases:
        try:
            fit_score_logreg(x, y)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid calibration input was accepted")


def test_scipy_fallback_matches_sklearn_in_the_binary_case():
    """scikit-learn fits two-class problems with one decision vector; a
    symmetric two-row softmax with the same C penalises half as much. The
    SciPy fallback must use the binary parameterisation so that a two-task
    prefix router does not depend on which backend is installed."""
    import task_calibration as tc
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401
    except ImportError:
        return
    rng = np.random.RandomState(0)
    z = rng.normal(size=(300, 2))
    y = (z[:, 0] - 0.5 * z[:, 1] + 0.3 * rng.normal(size=300) > 0).astype(int)
    classes = np.array([0, 1])
    coef_sk, int_sk, _, _ = tc._fit_sklearn(z, y, classes, 0.7, 2000, 0)
    coef_sp, int_sp, _, _ = tc._fit_scipy(z, y, classes, 0.7, 2000)
    # Same objective; the two L-BFGS implementations stop at slightly
    # different tolerances, hence a loose absolute tolerance.
    assert np.allclose(coef_sk, coef_sp, atol=5e-3), (coef_sk, coef_sp)
    assert np.allclose(int_sk, int_sp, atol=5e-3), (int_sk, int_sp)
    # The pre-3.0.4 symmetric two-row softmax is a different objective for
    # two classes (effectively 2C): its decision vector differs materially.
    from scipy.optimize import minimize
    targets = y.astype(int); onehot = np.eye(2)[targets]
    sw = tc._balanced_weights(y, classes); pen = 1.0 / 0.7
    def two_row(theta):
        W = theta[:4].reshape(2, 2); b = theta[4:]
        sc = z @ W.T + b; sc -= sc.max(1, keepdims=True)
        lp = sc - np.log(np.exp(sc).sum(1))[:, None]
        loss = -np.sum(sw * lp[np.arange(len(y)), targets]) + 0.5 * pen * np.sum(W * W)
        d = (np.exp(lp) - onehot) * sw[:, None]
        return loss, np.concatenate([(d.T @ z + pen * W).ravel(), d.sum(0)])
    r = minimize(two_row, np.zeros(6), jac=True, method="L-BFGS-B")
    W = r.x[:4].reshape(2, 2)
    assert np.abs((W[1] - W[0]) - coef_sk[1]).max() > 0.05
    # Multinomial case still agrees with scikit-learn.
    z3 = rng.normal(size=(300, 3)); y3 = z3.argmax(axis=1)
    c3 = np.array([0, 1, 2])
    coef_sk3, int_sk3, _, _ = tc._fit_sklearn(z3, y3, c3, 1.0, 2000, 0)
    coef_sp3, int_sp3, _, _ = tc._fit_scipy(z3, y3, c3, 1.0, 2000)
    assert np.allclose(coef_sk3, coef_sp3, atol=2e-3)
    assert np.allclose(int_sk3, int_sp3, atol=2e-3)


def test_score_bias_router_handles_offsets_and_serializes():
    """The tied-slope likelihood router should be close to raw routing in
    this unbiased synthetic case, improve this selected offset case, and
    serialize in the score-logreg state layout. These examples are not a
    general nondegradation guarantee."""
    import task_calibration as tc   # the torch-free module already on sys.path

    rng = np.random.RandomState(0)
    T, n = 12, 200

    def sample(count, bias):
        y = np.repeat(np.arange(T), count)
        x = rng.normal(1.2, 0.35, size=(y.size, T))
        x[np.arange(y.size), y] -= rng.normal(0.25, 0.35, size=y.size)
        if bias is not None:
            x += bias[None, :]
        return x, y

    for bias in (None, rng.normal(0.0, 0.3, size=T)):
        xc, yc = sample(n, bias)
        xt, yt = sample(400, bias)
        raw = float((xt.argmin(axis=1) == yt).mean())
        states = tc.fit_prefix_calibrators(xc, yc, T, kind="score-bias")
        assert len(states) == T and states[-1]["kind"] == "score-bias"
        state = states[-1]
        assert np.allclose(np.asarray(state["coef"]), -state["slope"] * np.eye(T))
        acc = float((tc.predict_score_logreg(xt, state) == yt).mean())
        logreg = tc.fit_prefix_calibrators(xc, yc, T, kind="score-logreg")[-1]
        acc_logreg = float((tc.predict_score_logreg(xt, logreg) == yt).mean())
        if bias is None:
            assert acc >= raw - 0.01, (acc, raw)  # property of this fixture only
        else:
            assert acc > raw + 0.02 and acc >= acc_logreg, (acc, raw, acc_logreg)
    assert 0.0 <= states[-1]["fit_accuracy"] <= 1.0
    assert 0.0 <= states[-1]["raw_argmin_fit_accuracy"] <= 1.0
    # The accuracy-fitted bias router starts from b = 0 and only accepts
    # improvements, so on its own fitting data it is never below argmin.
    acc_states = tc.fit_prefix_calibrators(xc, yc, T, kind="score-bias-acc")
    for state in acc_states:
        assert state["kind"] == "score-bias-acc"
        assert state["fit_accuracy"] >= state["raw_argmin_fit_accuracy"] - 1e-12
        assert np.allclose(np.asarray(state["coef"]), -np.eye(len(state["intercept"])))
    acc_test = float((tc.predict_score_logreg(xt, acc_states[-1]) == yt).mean())
    assert acc_test > raw + 0.02      # bias structure present in this branch
    try:
        tc.fit_prefix_calibrators(xc, yc, T, kind="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown calibration kinds must be rejected")


def test_fitted_states_route_in_float64_and_record_balanced_accuracies():
    """Every newly fitted state declares float64 routing (v3.0.10) and records
    the class-balanced fitting accuracy next to the plain one; score-bias-acc
    guarantees noninferiority in the balanced accuracy, exactly."""
    rng = np.random.default_rng(11)
    T = 4
    # Unequal task counts, so plain and balanced accuracy differ.
    counts = np.array([40, 10, 25, 5])
    y = np.repeat(np.arange(T), counts)
    x = rng.normal(1.0, 0.4, size=(y.size, T))
    x[np.arange(y.size), y] -= 0.3
    x[:, 0] -= 0.25                       # task 0 looks systematically confident
    for kind in tc.CALIBRATION_KINDS:
        states = tc.fit_prefix_calibrators(x, y, T, kind=kind)
        for stage, state in enumerate(states):
            assert state["routing_dtype"] == "float64"
            rows = y <= stage
            xs, ys = x[rows, :stage + 1], y[rows]
            fitted = tc.predict_score_logreg(xs, state)
            assert state["fit_balanced_accuracy"] == tc.balanced_accuracy(fitted, ys)
            assert state["raw_argmin_fit_balanced_accuracy"] == tc.balanced_accuracy(
                np.argmin(xs, axis=1), ys)
            if kind == "score-bias-acc":
                # Compared exactly: equal accuracies may round apart.
                assert tc.balanced_accuracy_exact(fitted, ys) >= tc.balanced_accuracy_exact(
                    np.argmin(xs, axis=1), ys)
    final = tc.fit_prefix_calibrators(x, y, T, kind="score-bias-acc")[-1]
    assert final["fit_balanced_accuracy"] != final["fit_accuracy"]   # the fixture is unbalanced


def test_balanced_accuracy_is_the_mean_per_task_accuracy():
    y = np.array([0, 0, 0, 0, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 1, 2, 0])
    assert tc.balanced_accuracy(pred, y) == (0.5 + 1.0 + 0.5) / 3
    assert tc.balanced_accuracy(y, y) == 1.0


def test_single_candidate_accuracy_router_is_reported_as_degenerate():
    """Stage 0 of score-bias-acc fits nothing; it used to be recorded as not
    converged, which read like a failed fit."""
    x = np.random.default_rng(0).normal(size=(6, 1))
    state = tc.fit_score_bias_acc(x, np.zeros(6, dtype=int))
    assert state["converged"] is True and state["n_iter"] == [0]
    assert state["solver_backend"] == "degenerate-single-class"
    assert state["intercept"] == [0.0] and state["routing_dtype"] == "float64"
    assert list(tc.predict_score_logreg(x, state)) == [0] * 6
    first = tc.fit_prefix_calibrators(np.hstack([x, x + 1.0]), np.r_[np.zeros(3, int), np.ones(3, int)],
                                      2, kind="score-bias-acc")[0]
    assert first["converged"] is True and first["solver_backend"] == "degenerate-single-class"


def test_exact_balanced_accuracy_sees_ties_that_floats_round_apart():
    """Per-task counts 2, 3, 3: hits (1, 2, 1) and (1, 0, 3) are exactly as
    accurate (1/2 + 2/3 + 1/3 = 1/2 + 0 + 1), but their float means differ
    in the last bit."""
    y = np.repeat([0, 1, 2], [2, 3, 3])
    a = np.array([0, 1, 1, 1, 0, 2, 0, 0])      # hits per task 1, 2, 1
    b = np.array([0, 1, 0, 0, 0, 2, 2, 2])      # hits per task 1, 0, 3
    assert tc.balanced_accuracy(a, y) < tc.balanced_accuracy(b, y)
    assert tc.balanced_accuracy_exact(a, y) == tc.balanced_accuracy_exact(b, y)
    assert tc.balanced_accuracy_exact(a, y) == Fraction(1, 2)
    assert float(tc.balanced_accuracy_exact(y, y)) == 1.0


def _float_objective_bias_acc(x, y, max_sweeps=8, grid_size=41):
    """The score-bias-acc search as it was before v3.0.10: the balanced
    objective summed as per-example float weights, compared with 1e-12."""
    x = np.asarray(x, dtype=np.float64)
    classes = np.unique(y)
    n, T = x.shape
    weights = tc._balanced_weights(y, classes)

    def accuracy(bias):
        return float(np.sum(weights * (np.argmin(x - bias[None, :], axis=1) == y)))

    b = np.zeros(T)
    best, sweeps, improved = accuracy(b), 0, True
    while improved and sweeps < max_sweeps:
        improved, sweeps = False, sweeps + 1
        for k in range(T):
            others = np.delete(x - b[None, :], k, axis=1).min(axis=1)
            gaps = (x[:, k] - b[k]) - others
            candidates = np.unique(np.concatenate([
                [0.0], np.quantile(gaps, np.linspace(0.02, 0.98, grid_size))]))
            best_k, best_val = b[k], best
            for delta in candidates:
                trial = b.copy()
                trial[k] = b[k] + delta
                val = accuracy(trial)
                if val > best_val + 1e-12 or (abs(val - best_val) <= 1e-12
                                              and abs(trial[k]) < abs(best_k)):
                    best_k, best_val = trial[k], val
            if best_k != b[k]:
                b[k] = best_k
                if best_val > best + 1e-12:
                    improved = True
                best = best_val
    return b.tolist(), sweeps, not improved


def test_accuracy_router_search_is_unchanged_for_equal_task_counts():
    """With equal task counts (every recorded run) the exact objective is the
    number of correct routes, as the float one was: same biases, same sweeps."""
    rng = np.random.default_rng(11)
    for trial in range(24):
        T, m = int(rng.integers(2, 7)), int(rng.integers(4, 120))
        y = np.repeat(np.arange(T), m)
        x = rng.normal(size=(y.size, T)) + rng.normal(size=T)[None, :]
        if trial % 3 == 1:          # saturated experts: many exact zeros and ties
            x = np.where(rng.random(x.shape) < 0.6, 0.0, np.abs(x))
        if trial % 3 == 2:          # informative scores
            x[np.arange(y.size), y] -= 1.0
        x = x.astype(np.float32).astype(np.float64)
        state = tc.fit_score_bias_acc(x, y)
        expected, sweeps, converged = _float_objective_bias_acc(x, y)
        assert state["intercept"] == expected, trial
        assert state["n_iter"] == [sweeps] and state["converged"] is converged, trial


def _drift_case(n=4000, seed=5):
    """Unequal task counts (n + 1, n, n) and two mirror-image tasks, so every
    bias move trades routes one for one: the balanced accuracy can only stay
    equal. Summed as float weights it picks up rounding noise above 1e-12
    (from the second round-5 review)."""
    r = np.random.default_rng(seed)
    one = np.float32(1.0)
    k = n // 3
    a = r.uniform(0.01, 0.5, size=k).astype(np.float32)
    g = r.uniform(0.01, 0.5, size=n - k).astype(np.float32)
    correct1 = np.c_[np.full(n - k, 9.0), np.full(n - k, one), one + g]
    task1 = np.r_[np.c_[np.full(k, 9.0), one + a, np.full(k, one)], correct1]
    task2 = np.r_[np.c_[np.full(n - k, 9.0), np.full(n - k, one), one + g],
                  np.c_[np.full(k, 9.0), one + a, np.full(k, one)]]
    good0 = int(0.83 * (n + 1))
    task0 = np.r_[np.c_[np.zeros(good0), np.full(good0, 9.0), np.full(good0, 9.0)],
                  correct1[r.integers(0, n - k, size=n + 1 - good0)]]
    x = np.vstack([task0, task1, task2]).astype(np.float32).astype(np.float64)
    y = np.r_[np.zeros(n + 1, int), np.ones(n, int), np.full(n, 2)]
    return x, y


def test_accuracy_router_search_moves_no_bias_without_a_gain():
    x, y = _drift_case()
    drifted, _, _ = _float_objective_bias_acc(x, y)
    assert drifted != [0.0, 0.0, 0.0]          # the float search moved a bias
    raw = np.argmin(x, axis=1)
    moved = tc.predict_score_logreg(x, {"classes": [0, 1, 2], "feature_mean": [0.0] * 3,
                                        "feature_scale": [1.0] * 3, "coef": (-np.eye(3)).tolist(),
                                        "intercept": drifted})
    # ... for no gain at all, while its float balanced accuracy came out lower.
    assert tc.balanced_accuracy_exact(moved, y) == tc.balanced_accuracy_exact(raw, y)
    state = tc.fit_score_bias_acc(x, y)
    assert state["intercept"] == [0.0, 0.0, 0.0] and state["converged"] is True
    final = tc.fit_prefix_calibrators(x, y, 3, kind="score-bias-acc")[-1]
    assert final["fit_balanced_accuracy"] == final["raw_argmin_fit_balanced_accuracy"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} calibration tests passed.")
