"""Torch-free production tests for leakage-safe score calibration."""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "diltcl"))
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} calibration tests passed.")
