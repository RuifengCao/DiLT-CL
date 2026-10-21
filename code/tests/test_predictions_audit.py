"""Per-example class-IL predictions (v3.0.5): recording and recomputation.

Torch-free part: ``tools/recompute_from_predictions.py`` must rebuild the
evaluator's matrices and pooled statistics exactly from per-example arrays,
with an independent per-example reference implementation as the oracle, and
must reject inconsistent recordings.

Torch part (skipped without PyTorch): recording inside
``evaluate_class_il_inference`` must not change any returned value, and the
arrays it records must reproduce every returned matrix and statistic exactly;
the class-IL runner must persist a hashed predictions file that the bundle
verifier accepts.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rp = _load("recompute_from_predictions_under_test",
           ROOT / "tools" / "recompute_from_predictions.py")


# ---------------------------------------------------------------------------
# Synthetic recordings and an independent per-example reference
# ---------------------------------------------------------------------------

def _synthetic(T=4, C=3, per_task=(7, 5, 9, 6), seed=0, disjoint=True):
    rng = np.random.default_rng(seed)
    true_task = np.concatenate([np.full(n, t) for t, n in enumerate(per_task)]).astype(np.int16)
    N = true_task.size
    label = rng.integers(0, C, N).astype(np.int16)
    # Coarse scores so that exact ties occur.
    scores = (rng.integers(0, 4, (N, T)) / 4.0).astype(np.float32)
    route = np.full((N, T), -1, np.int16)
    tie = np.zeros((N, T), bool)
    pred = np.full((N, T), -1, np.int16)
    oracle = np.where(rng.random(N) < 0.7, label, rng.integers(0, C, N)).astype(np.int16)
    for s in range(T):
        rows = true_task <= s
        cand = scores[rows, :s + 1]
        win = cand == cand.min(axis=1, keepdims=True)
        route[rows, s] = win.argmax(axis=1)
        tie[rows, s] = win.sum(axis=1) > 1
        # An example routed to its own task reads its own (oracle) expert;
        # elsewhere the routed expert mostly predicts the label.
        guess = rng.integers(0, C, rows.sum())
        keep = rng.random(rows.sum()) < 0.6
        other = np.where(keep, label[rows], guess)
        pred[rows, s] = np.where(route[rows, s] == true_task[rows], oracle[rows], other)
    return {"true_task": true_task, "label_local": label, "oracle_pred_local": oracle,
            "scores": scores, "route_by_stage": route, "tie_by_stage": tie,
            "routed_pred_local_by_stage": pred}


def _reference(arrays, T, disjoint=True):
    """Per-example loop; deliberately a different code path from ``recompute``."""
    counts = [0] * T
    oracle_hits = [0] * T
    cls = [[0] * T for _ in range(T)]
    ta = [[0] * T for _ in range(T)]
    ties = [[0] * T for _ in range(T)]
    crc = [[0] * T for _ in range(T)]
    for i in range(arrays["true_task"].size):
        t = int(arrays["true_task"][i]); y = int(arrays["label_local"][i])
        counts[t] += 1
        oracle_hits[t] += int(arrays["oracle_pred_local"][i]) == y
        for s in range(t, T):
            r = int(arrays["route_by_stage"][i, s]); p = int(arrays["routed_pred_local_by_stage"][i, s])
            g_pred = r * (3 if disjoint else 0) + p
            g_true = t * (3 if disjoint else 0) + y
            correct = g_pred == g_true
            cls[s][t] += correct; ta[s][t] += r == t
            ties[s][t] += bool(arrays["tie_by_stage"][i, s])
            crc[s][t] += correct and r == t
    result = np.zeros((T, T)); task_inf = np.zeros((T, T))
    for t in range(T):
        for s in range(t, T):
            result[s, t] = 100.0 * cls[s][t] / counts[t]
            task_inf[s, t] = 100.0 * ta[s][t] / counts[t]
    total = sum(counts)
    return {"result_matrix": result, "task_inference_matrix": task_inf,
            "micro_class_il_acc": 100.0 * sum(cls[-1]) / total,
            "micro_task_inference_acc": 100.0 * sum(ta[-1]) / total,
            "micro_oracle_route_acc": 100.0 * sum(oracle_hits) / total,
            "conditional_class_acc": 100.0 * sum(crc[-1]) / sum(ta[-1])}


@pytest.mark.parametrize("disjoint", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recompute_matches_independent_reference(disjoint, seed):
    arrays = _synthetic(seed=seed, disjoint=disjoint)
    rec = rp.recompute(arrays, 4, disjoint=disjoint)
    ref = _reference(arrays, 4, disjoint=disjoint)
    assert np.array_equal(rec["result_matrix"], ref["result_matrix"])
    assert np.array_equal(rec["task_inference_matrix"], ref["task_inference_matrix"])
    for key in ("micro_class_il_acc", "micro_task_inference_acc",
                "micro_oracle_route_acc", "conditional_class_acc"):
        assert rec["stats"][key] == ref[key], key
    # Upper triangle (stage before the task arrives) stays zero.
    assert np.all(np.triu(rec["result_matrix"], k=1) == 0)


def test_decomposition_identity_and_recoverable_fraction():
    arrays = _synthetic(seed=3)
    d = rp.decomposition(rp.recompute(arrays, 4)["stats"])
    # A = U * V holds exactly up to floating-point rounding (disjoint labels).
    assert abs(d["identity_residual_A_minus_UV"]) < 1e-9
    expected = 100.0 * (d["oracle_accuracy"] - d["routed_accuracy"]) / (100.0 - d["routed_accuracy"])
    assert d["recoverable_fraction_percent"] == pytest.approx(expected)


def test_raw_route_check_accepts_argmin_and_flags_tampering():
    arrays = _synthetic(seed=4)
    assert rp.raw_route_errors(arrays) == []
    tampered = {k: v.copy() for k, v in arrays.items()}
    row = int(np.flatnonzero(tampered["true_task"] == 0)[0])
    tampered["route_by_stage"][row, 3] = (tampered["route_by_stage"][row, 3] + 1) % 4
    errors = rp.raw_route_errors(tampered)
    assert len(errors) == 1 and "stage 3" in errors[0]
    flipped = {k: v.copy() for k, v in arrays.items()}
    flipped["tie_by_stage"][row, 2] = ~flipped["tie_by_stage"][row, 2]
    assert rp.raw_route_errors(flipped)


def test_routes_before_task_arrival_are_rejected():
    arrays = _synthetic(seed=5)
    row = int(np.flatnonzero(arrays["true_task"] == 2)[0])
    arrays["route_by_stage"][row, 0] = 0
    with pytest.raises(ValueError, match="before the task arrived"):
        rp.recompute(arrays, 4)


def test_stored_router_agreement_reproduces_linear_decisions():
    rng = np.random.default_rng(6)
    T, N = 3, 60
    true_task = np.repeat(np.arange(T), N // T).astype(np.int16)
    scores = rng.normal(size=(N, T)).astype(np.float32)
    models, routes = [], np.full((N, T), -1, np.int16)
    for s in range(T):
        state = {"classes": list(range(s + 1)),
                 "feature_mean": rng.normal(size=s + 1).tolist(),
                 "feature_scale": (1.0 + rng.random(s + 1)).tolist(),
                 "coef": rng.normal(size=(s + 1, s + 1)).tolist(),
                 "intercept": rng.normal(size=s + 1).tolist()}
        models.append(state)
        rows = true_task <= s
        x = scores[rows, :s + 1].astype(np.float64)
        decision = ((x - state["feature_mean"]) / state["feature_scale"]) @ np.asarray(state["coef"]).T \
            + state["intercept"]
        routes[rows, s] = decision.argmax(axis=1)
    arrays = {"true_task": true_task, "scores": scores, "route_by_stage": routes}
    out = rp.stored_router_agreement(arrays, {"prefix_models": models})
    assert out["decisions_compared"] == sum(int((true_task <= s).sum()) for s in range(T))
    assert out["route_agreement"] == 1.0


def test_suite_writes_predictions_for_class_il_jobs_only():
    suite = _load("run_suite_under_test", ROOT / "run_suite.py")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = pathlib.Path(tmp)
        jobs = suite.p0_jobs([0], "multi", granularity="method")
        class_jobs = [j for j in jobs if j.runner == "run_class_il.py"]
        task_jobs = [j for j in jobs if j.runner != "run_class_il.py"]
        assert class_jobs and task_jobs
        command, output = suite.command_for(class_jobs[0], run_dir, None, False)
        assert command[command.index("--predictions-dir") + 1] == str(output.parent / "predictions")
        command, _ = suite.command_for(task_jobs[0], run_dir, None, False)
        assert "--predictions-dir" not in command
        # Without checkpoints (smoke/debug runs) nothing extra is written.
        command, _ = suite.command_for(class_jobs[0], run_dir, None, False,
                                       save_checkpoints=False)
        assert "--predictions-dir" not in command


# ---------------------------------------------------------------------------
# Torch: the recorder inside the evaluator, and the runner end to end
# ---------------------------------------------------------------------------

def _forward_and_loaders(T=3, C=4, features=6, per_task=10, seed=0):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator().manual_seed(seed)
    weights = [torch.randn(features, C, generator=g) for _ in range(T)]
    loaders, cal_loaders = [], []
    for t in range(T):
        x = torch.randn(per_task, features, generator=g) + t
        y = torch.randint(0, C, (per_task,), generator=g)
        loaders.append(DataLoader(TensorDataset(x, y), batch_size=4))
        xc = torch.randn(per_task, features, generator=g) + t
        cal_loaders.append(DataLoader(TensorDataset(xc, torch.zeros(per_task, dtype=torch.long)),
                                      batch_size=4))
    # Quantised logits make exact score ties likely, exercising the tie path.
    forward_t = lambda x, t: torch.round(x @ weights[t])      # noqa: E731
    return forward_t, loaders, cal_loaders


def _assert_same_result(a, b):
    assert set(a) == set(b)
    for key in a:
        if isinstance(a[key], np.ndarray):
            assert np.array_equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize("criterion", ["entropy", "confidence", "margin"])
@pytest.mark.parametrize("disjoint", [True, False])
def test_recording_changes_nothing_and_recomputes_exactly(criterion, disjoint):
    pytest.importorskip("torch")
    from diltcl.task_inference import evaluate_class_il_inference
    forward_t, loaders, _ = _forward_and_loaders()
    plain = evaluate_class_il_inference(forward_t, loaders, 3, 4, device="cpu",
                                        criterion=criterion, disjoint_classes=disjoint)
    record = {}
    recorded = evaluate_class_il_inference(forward_t, loaders, 3, 4, device="cpu",
                                           criterion=criterion, disjoint_classes=disjoint,
                                           record=record)
    _assert_same_result(plain, recorded)
    assert record["route_by_stage"].shape == (30, 3)
    rec = rp.recompute(record, 3, disjoint=disjoint)
    for name in ("result_matrix", "task_inference_matrix", "routing_tie_matrix",
                 "conditional_class_matrix"):
        assert np.array_equal(rec[name], recorded[name]), name
    for stat in ("class_il_acc", "micro_class_il_acc", "oracle_route_acc",
                 "micro_oracle_route_acc", "task_inference_acc", "micro_task_inference_acc",
                 "routing_tie_rate", "micro_routing_tie_rate", "conditional_class_acc",
                 "zero_correct_route_task_count"):
        assert rec["stats"][stat] == recorded[stat], stat
    assert rp.raw_route_errors(record) == []


def test_calibrated_recording_matches_stored_routers():
    pytest.importorskip("torch")
    from diltcl.task_calibration import fit_prefix_calibrators
    from diltcl.task_inference import collect_calibration_features, evaluate_class_il_inference
    forward_t, loaders, cal_loaders = _forward_and_loaders(seed=1)
    features, task_ids, _ = collect_calibration_features(forward_t, cal_loaders, 3, device="cpu")
    calibrators = fit_prefix_calibrators(features, task_ids, 3, kind="score-bias")
    record = {}
    res = evaluate_class_il_inference(forward_t, loaders, 3, 4, device="cpu",
                                      calibrators=calibrators, record=record)
    assert record["calibrated"] is True
    rec = rp.recompute(record, 3)
    assert np.array_equal(rec["result_matrix"], res["result_matrix"])
    agreement = rp.stored_router_agreement(record, {"prefix_models": calibrators})
    assert agreement["route_agreement"] == 1.0


def _fake_isolation_setup():
    import test_runner_integration as tri
    import run_class_il as class_il

    def fake_isolation(benchmark, cfg, seed, **_):
        train, test = tri._tiny_loaders(tri.CLASSES_PER_TASK, seed)
        return class_il.IsolationSetup(
            train, test, (lambda: tri._TinyNet(tri.CLASSES_PER_TASK)),
            tri.NUM_TASKS, tri.CLASSES_PER_TASK, True)
    return fake_isolation


def _run_runner(out_prefix, predictions_dir, methods, calibration="none",
                checkpoint_dir=None):
    pytest.importorskip("torch")
    import test_runner_integration as tri
    import run_class_il as class_il

    def fake_shared(benchmark, cfg, seed, **_):
        train, test = tri._tiny_loaders(tri.NUM_TASKS * tri.CLASSES_PER_TASK, seed)
        return class_il.SharedSetup(
            train, test, (lambda: tri._TinyNet(tri.NUM_TASKS * tri.CLASSES_PER_TASK)),
            tri.NUM_TASKS)

    with tri._Patched(class_il, {"_shared_setup": fake_shared,
                                 "_isolation_setup": _fake_isolation_setup(),
                                 "preflight": tri._NoPreflight}):
        class_il.run(tri._fast_config(), ["cifar"], list(methods), [0],
                     out_prefix=out_prefix, calibration=calibration,
                     predictions_dir=predictions_dir, checkpoint_dir=checkpoint_dir)
    with open(f"{out_prefix}.json") as handle:
        return json.load(handle)


@pytest.mark.parametrize("calibration", ["none", "score-bias", "score-bias-acc", "score-logreg"])
def test_runner_persists_verifiable_predictions(calibration):
    verifier = _load("verify_bundle_under_test", ROOT / "tools" / "verify_experiment_bundle.py")
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "job" / "result"
        out.parent.mkdir()
        payload = _run_runner(str(out), str(out.parent / "predictions"),
                              ("finetune", "wsn"), calibration=calibration)
        record = payload["calibration"]["cifar"]["wsn"]["0"]
        pred = record["predictions"]
        path = out.parent / pred["relative_path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pred["sha256"]
        assert pred["relative_path"] == "predictions/cifar/wsn/seed0.npz"
        arrays, meta = rp.load_predictions(path)
        assert meta["trained_state_content_sha256"] == record["trained_state"]["content_sha256"]
        assert meta["calibration"] == calibration
        assert ("raw_route_by_stage" in arrays) == (calibration != "none")
        assert ("calibration_features" in arrays) == (calibration != "none")
        assert payload["per_seed_run_records"]["cifar"]["wsn"]["0"]["includes_prediction_recording"] is True
        assert payload["per_seed_run_records"]["cifar"]["finetune"]["0"]["includes_prediction_recording"] is False
        with np.load(f"{out}.npz", allow_pickle=False) as npz:
            assert rp.compare_with_artifact(arrays, meta, payload, npz, "cifar", "wsn", 0) == []
            errors = []
            assert verifier.validate_predictions(record, payload, npz, out.with_suffix(".json"),
                                                 "cifar", "wsn", 0, "wsn", errors)
            assert errors == []
        entries = rp.check_job(out.with_suffix(".json"), refit=calibration != "none")
        assert len(entries) == 1 and entries[0]["errors"] == []
        # A prediction read from another expert than the chosen one is
        # caught: an example routed to its own task must predict what its
        # own expert predicts.
        home = np.argwhere(arrays["route_by_stage"] == arrays["true_task"][:, None])
        i, s = (int(v) for v in home[0])
        wrong = {k: v.copy() for k, v in arrays.items()}
        wrong["routed_pred_local_by_stage"][i, s] = (wrong["oracle_pred_local"][i] + 1) % 4
        with np.load(f"{out}.npz", allow_pickle=False) as npz:
            errors = rp.compare_with_artifact(wrong, meta, payload, npz, "cifar", "wsn", 0)
        assert any("predicted differently from that task's expert" in e for e in errors)
        if calibration != "none":
            assert all(m["routing_dtype"] == "float64" for m in record["prefix_models"])
            # Routers are applied in float64, as fitted: every kind reproduces
            # the recorded routes, and a re-fit gives the same parameters.
            assert entries[0]["stored_router"]["routing_precision"] == "float64"
            assert entries[0]["stored_router"]["route_agreement"] == 1.0
            assert entries[0]["stored_router"]["errors"] == []
            assert entries[0]["refit"]["parameters_identical"] is True
            assert entries[0]["refit"]["route_agreement"] == 1.0
            assert set(entries[0]["refit"]["max_abs_parameter_difference"]) == {
                "feature_mean", "feature_scale", "coef", "intercept"}
            # A router fitted on other data than the held-out split (here the
            # test scores) has the wrong number of fitting rows per task.
            on_test = dict(arrays, calibration_features=arrays["scores"],
                           calibration_task_ids=arrays["true_task"])
            with np.load(f"{out}.npz", allow_pickle=False) as npz:
                errors = rp.compare_with_artifact(on_test, meta, payload, npz, "cifar", "wsn", 0)
            assert any("held-out split sizes" in e for e in errors)
            # Fitting accuracies of every stored prefix router are reproduced.
            fit = entries[0]["fit_accuracy_check"]
            assert fit["stages_checked"] == len(record["prefix_models"]) and fit["errors"] == []
            # The verifier's own check recomputes those fitting accuracies...
            tampered = json.loads(json.dumps(payload))
            tampered["calibration"]["cifar"]["wsn"]["0"]["prefix_models"][-1]["fit_accuracy"] += 0.25
            with np.load(f"{out}.npz", allow_pickle=False) as npz:
                errors = rp.compare_with_artifact(arrays, meta, tampered, npz, "cifar", "wsn", 0)
                assert any("fit_accuracy recomputed" in e for e in errors)
                # ...rejects a calibrated file without its fitting inputs or raw pass...
                for drop in ("calibration_features", "raw_route_by_stage"):
                    partial = {k: v for k, v in arrays.items() if k != drop}
                    errors = rp.compare_with_artifact(partial, meta, payload, npz, "cifar", "wsn", 0)
                    assert any("missing" in e for e in errors), drop
                # ...and a file whose metadata names another router than the job ran.
                errors = rp.compare_with_artifact(arrays, {**meta, "calibration": "none"},
                                                  payload, npz, "cifar", "wsn", 0)
                assert any("job ran" in e for e in errors)
        # Tampering with the file is detected through its recorded hash.
        with path.open("ab") as handle:
            handle.write(b"\0")
        errors = []
        with np.load(f"{out}.npz", allow_pickle=False) as npz:
            verifier.validate_predictions(record, payload, npz, out.with_suffix(".json"),
                                          "cifar", "wsn", 0, "wsn", errors)
        assert any("SHA-256" in e for e in errors)


def _saturated_bias_acc_case():
    """Two saturated experts (entropy exactly zero) and biases that differ
    only below float32 resolution: the fitted float64 router picks task 1,
    while applying it in float32 makes a tie that falls back to task 0 (the
    C1 finding of the fourth review)."""
    torch = pytest.importorskip("torch")
    b = [1.1044052673094165e-29, 1.1044052673202787e-29]
    assert np.float32(b[0]) == np.float32(b[1]) and b[0] < b[1]
    state = {"kind": "score-bias-acc", "classes": [0, 1], "feature_mean": [0.0, 0.0],
             "feature_scale": [1.0, 1.0], "coef": [[-1.0, -0.0], [-0.0, -1.0]],
             "intercept": b, "routing_dtype": "float64"}
    logits = [torch.tensor([[60.0, -60.0, -60.0]]), torch.tensor([[-60.0, 60.0, -60.0]])]
    return torch, state, logits


def test_routers_are_applied_in_the_precision_they_were_fitted_in():
    torch, state, logits = _saturated_bias_acc_case()
    from diltcl.task_calibration import predict_score_logreg
    from diltcl.task_inference import infer_task, task_scores
    scores = task_scores(logits)
    assert float(scores.abs().max()) == 0.0                   # fully saturated
    route, tie = infer_task(logits, calibrator=state, return_ties=True)
    fitted = predict_score_logreg(scores.T.double().numpy(), state)
    assert int(route) == int(fitted[0]) == 1 and not bool(tie)
    # A state written before v3.0.10 has no routing_dtype and keeps its
    # float32 application, so old recordings re-evaluate exactly.
    legacy = {k: v for k, v in state.items() if k != "routing_dtype"}
    route, tie = infer_task(logits, calibrator=legacy, return_ties=True)
    assert int(route) == 0 and bool(tie)
    # The audit applies each state in its own precision.
    arrays = {"true_task": np.array([1], np.int16),
              "scores": scores.T.numpy().astype(np.float32),
              "route_by_stage": np.array([[-1, 1]], np.int16),
              "tie_by_stage": np.array([[False, False]])}
    stage0 = {"kind": "score-bias-acc", "classes": [0], "feature_mean": [0.0],
              "feature_scale": [1.0], "coef": [[-1.0]], "intercept": [0.0],
              "routing_dtype": "float64"}
    out = rp.stored_router_agreement(arrays, {"kind": "score-bias-acc",
                                              "prefix_models": [stage0, state]})
    assert out["errors"] == [] and out["routing_precision"] == "float64"
    out = rp.stored_router_agreement(arrays, {"kind": "score-bias-acc",
                                              "prefix_models": [stage0, legacy]})
    assert out["errors"] and "expected exact" in out["errors"][0]
    try:
        infer_task(logits, calibrator={**state, "routing_dtype": "float16"})
    except ValueError as exc:
        assert "routing_dtype" in str(exc)
    else:
        raise AssertionError("an unknown routing precision must be rejected")


@pytest.mark.parametrize("calibrated", [False, True])
def test_non_finite_scores_stop_the_evaluation_with_a_clear_error(calibrated):
    torch = pytest.importorskip("torch")
    from diltcl.task_calibration import fit_prefix_calibrators
    from diltcl.task_inference import (collect_calibration_features,
                                       evaluate_class_il_inference, infer_task)
    forward_t, loaders, cal_loaders = _forward_and_loaders(seed=2)
    calibrators = None
    if calibrated:
        features, task_ids, _ = collect_calibration_features(forward_t, cal_loaders, 3,
                                                             device="cpu")
        calibrators = fit_prefix_calibrators(features, task_ids, 3, kind="score-bias")

    def broken(x, t):
        out = forward_t(x, t)
        return out * float("nan") if t == 1 else out

    for value in (float("nan"), float("inf")):
        try:
            evaluate_class_il_inference(
                lambda x, t: forward_t(x, t) * value if t == 1 else forward_t(x, t),
                loaders, 3, 4, device="cpu", calibrators=calibrators)
        except ValueError as exc:
            assert "non-finite routing scores for test examples of task 0" in str(exc)
            assert "per candidate expert: [0, 4, 0]" in str(exc)
        else:
            raise AssertionError("non-finite scores must stop the evaluation")
    try:
        collect_calibration_features(broken, cal_loaders, 3, device="cpu")
    except ValueError as exc:
        assert "held-out calibration examples of task 0" in str(exc)
    else:
        raise AssertionError("non-finite calibration scores must stop the fit")
    x = next(iter(loaders[0]))[0]
    try:
        infer_task([broken(x, t) for t in range(3)],
                   calibrator=None if calibrators is None else calibrators[-1])
    except ValueError as exc:
        assert "non-finite routing scores" in str(exc)
    else:
        raise AssertionError("infer_task must reject non-finite scores")


def test_logistic_tie_flags_must_be_near_ties():
    model0 = {"classes": [0], "feature_mean": [0.0], "feature_scale": [1.0],
              "coef": [[1.0]], "intercept": [0.0]}
    model1 = {"classes": [0, 1], "feature_mean": [0.0, 0.0], "feature_scale": [1.0, 1.0],
              "coef": [[1.0, 0.0], [0.0, 1.0]], "intercept": [0.0, 0.0]}
    for dtype in (None, "float64"):
        states = [dict(m, **({"routing_dtype": dtype} if dtype else {})) for m in (model0, model1)]
        record = {"kind": "score-logreg", "prefix_models": states}
        arrays = {"true_task": np.array([1, 1], np.int16),
                  "scores": np.array([[0.5, 0.5], [0.0, 3.0]], np.float32),
                  "route_by_stage": np.array([[-1, 0], [-1, 1]], np.int16),
                  "tie_by_stage": np.array([[False, True], [False, False]])}
        out = rp.stored_router_agreement(arrays, record)
        assert out["errors"] == [] and out["tie_flag_errors"] == 0, dtype
        # Row 1 has a clear winner: a recorded tie there is impossible.
        flagged = dict(arrays, tie_by_stage=np.array([[False, True], [False, True]]))
        out = rp.stored_router_agreement(flagged, record)
        assert out["tie_flag_errors"] == 1 and out["errors"], dtype
        # A single candidate never ties.
        flagged = dict(arrays, route_by_stage=np.array([[0, 0], [0, 1]], np.int16),
                       true_task=np.array([0, 0], np.int16),
                       tie_by_stage=np.array([[True, True], [False, False]]))
        out = rp.stored_router_agreement(flagged, record)
        assert out["tie_flag_errors"] == 1, dtype


def test_tf32_bound_covers_subnormal_scores():
    """Tensor cores may flush subnormal float32 inputs to zero, which no
    relative bound covers: saturated experts give entropies like 1e-42, and a
    correct TF32 run then ties where float32 does not (C4 of the fourth
    review; only for routers applied in float32, i.e. before v3.0.10)."""
    model0 = {"classes": [0], "feature_mean": [0.0], "feature_scale": [1.0],
              "coef": [[-1.0]], "intercept": [0.0]}
    model1 = {"classes": [0, 1], "feature_mean": [0.0, 0.0], "feature_scale": [1.0, 1.0],
              "coef": [[-1.0, -0.0], [-0.0, -1.0]], "intercept": [0.0, 0.0]}
    arrays = {"true_task": np.array([1], np.int16),
              "scores": np.array([[3e-42, 1e-42]], np.float32),
              "route_by_stage": np.array([[-1, 0]], np.int16),     # TF32: [0, 0] ties -> 0
              "tie_by_stage": np.array([[False, True]])}
    record = {"kind": "score-bias-acc", "prefix_models": [model0, model1]}
    out = rp.stored_router_agreement(arrays, record, tf32=True)
    assert out["errors"] == [] and out["near_tie_disagreements"] == 1
    out = rp.stored_router_agreement(arrays, record, tf32=False)
    assert out["errors"] and "float32" in out["errors"][0]


def test_calibration_inputs_must_be_the_recorded_holdout():
    arrays = {"calibration_task_ids": np.repeat(np.arange(3), [4, 4, 4]).astype(np.int16)}
    record = {"split_counts": [{"train": 36, "calibration": 4}] * 3, "observed_counts": [4, 4, 4]}
    assert rp.calibration_count_errors(arrays, record, 3) == []
    assert any("held-out split sizes" in e for e in rp.calibration_count_errors(
        arrays, dict(record, split_counts=[{"calibration": 5}, {"calibration": 4},
                                           {"calibration": 3}]), 3))
    assert any("examples scored for fitting" in e for e in rp.calibration_count_errors(
        arrays, dict(record, observed_counts=[4, 4, 5]), 3))
    assert any("without recorded held-out split sizes" in e for e in rp.calibration_count_errors(
        arrays, {"split_counts": []}, 3))


def test_fit_accuracy_check_tolerates_only_float64_near_ties():
    """A score-logreg decision is a full matrix product: on another CPU's BLAS
    an exact float64 tie may resolve the other way, so the check allows one
    example per near-tie and no more; the bias routers are exact everywhere."""
    tc = rp._task_calibration()
    y = np.array([0, 0, 1, 1])
    x = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.2, 0.9]], np.float32)
    identity = {"kind": "score-logreg", "classes": [0, 1], "feature_mean": [0.0, 0.0],
                "feature_scale": [1.0, 1.0], "coef": [[1.0, 0.0], [0.0, 1.0]],
                "intercept": [0.0, 0.0], "routing_dtype": "float64"}
    single = {"kind": "score-logreg", "classes": [0], "feature_mean": [0.0],
              "feature_scale": [1.0], "coef": [[0.0]], "intercept": [0.0],
              "routing_dtype": "float64", "fit_accuracy": 1.0}
    fitted = tc.predict_score_logreg(x, identity)
    assert list(fitted) == [0, 0, 1, 1]               # row 0 ties; lowest task wins
    stage1 = dict(identity, fit_accuracy=float(np.mean(fitted == y)),
                  fit_balanced_accuracy=tc.balanced_accuracy(fitted, y))
    arrays = {"calibration_features": x, "calibration_task_ids": y.astype(np.int16)}
    record = {"kind": "score-logreg", "prefix_models": [single, stage1]}
    out = rp.fit_accuracy_check(arrays, record)
    assert out["errors"] == [] and out["near_tie_examples"] == 1
    # The tied example resolved the other way on the fitting machine: allowed.
    flipped = json.loads(json.dumps(record))
    flipped["prefix_models"][1]["fit_accuracy"] -= 0.25
    flipped["prefix_models"][1]["fit_balanced_accuracy"] -= 0.25
    assert rp.fit_accuracy_check(arrays, flipped)["errors"] == []
    # Two examples' worth is beyond the single near-tie.
    flipped["prefix_models"][1]["fit_accuracy"] -= 0.25
    assert rp.fit_accuracy_check(arrays, flipped)["errors"]
    # The bias routers are exact: no slack at all.
    rng = np.random.default_rng(5)
    yb = np.repeat(np.arange(3), 6)
    xb = rng.normal(size=(yb.size, 3)).astype(np.float32)
    bias = tc.fit_prefix_calibrators(xb, yb, 3, kind="score-bias-acc")
    arrays = {"calibration_features": xb, "calibration_task_ids": yb.astype(np.int16)}
    record = {"kind": "score-bias-acc", "prefix_models": bias}
    assert rp.fit_accuracy_check(arrays, record)["errors"] == []
    nudged = json.loads(json.dumps(record))
    nudged["prefix_models"][-1]["fit_balanced_accuracy"] += 1.0 / yb.size
    assert rp.fit_accuracy_check(arrays, nudged)["errors"]


def _self_consistent(states, x, y):
    """Recompute every stage's recorded fitting accuracies from its own router,
    as the runner records them (a forged state that is consistent with itself)."""
    tc = rp._task_calibration()
    for stage, state in enumerate(states):
        rows = y <= stage
        xs, ys = np.asarray(x, np.float64)[rows, :stage + 1], y[rows]
        fitted, raw = tc.predict_score_logreg(xs, state), np.argmin(xs, axis=1)
        state["fit_accuracy"] = float(np.mean(fitted == ys))
        state["raw_argmin_fit_accuracy"] = float(np.mean(raw == ys))
        if "fit_balanced_accuracy" in state:
            state["fit_balanced_accuracy"] = tc.balanced_accuracy(fitted, ys)
            state["raw_argmin_fit_balanced_accuracy"] = tc.balanced_accuracy(raw, ys)
    return states


def test_accuracy_router_may_never_fit_below_the_raw_rule():
    rng = np.random.default_rng(9)
    T = 3
    y = np.repeat(np.arange(T), 5)
    x = rng.normal(size=(y.size, T))
    x[np.arange(y.size), y] -= 2.0                  # the raw rule is mostly right
    x = x.astype(np.float32)
    tc = rp._task_calibration()
    states = tc.fit_prefix_calibrators(x, y, T, kind="score-bias-acc")
    arrays = {"calibration_features": x, "calibration_task_ids": y.astype(np.int16)}
    record = {"kind": "score-bias-acc", "prefix_models": states}
    assert rp.fit_accuracy_check(arrays, record)["errors"] == []
    # A degraded router whose recorded accuracies are its own: every example
    # goes to task 0. Recomputing the fields cannot catch it; the rule can.
    degraded = json.loads(json.dumps(record))
    degraded["prefix_models"][-1]["intercept"] = [1e6, 0.0, 0.0]
    _self_consistent(degraded["prefix_models"], x, y)
    errors = rp.fit_accuracy_check(arrays, degraded)["errors"]
    assert errors and all("below the raw rule" in e for e in errors), errors
    # Records written before v3.0.10 (no routing_dtype, plain accuracies
    # only) are held to the same rule: with equal task counts their search
    # compared exact integers too.
    legacy = json.loads(json.dumps(degraded))
    for s in legacy["prefix_models"]:
        for key in ("routing_dtype", "fit_balanced_accuracy", "raw_argmin_fit_balanced_accuracy"):
            s.pop(key, None)
    errors = rp.fit_accuracy_check(arrays, legacy)["errors"]
    assert errors and all("below the raw rule" in e for e in errors), errors


def test_accuracy_router_rule_compares_balanced_accuracies_exactly():
    """Task counts 2, 3, 3: the stored router (bias +0.5 on task 1) and the
    raw rule route (1, 2, 1) and (1, 0, 3) examples per task correctly --
    exactly as accurate, but the router's float balanced accuracy is one ulp
    lower. Comparing the recorded floats rejected this correct fit (P1 of the
    second round-5 review)."""
    y = np.repeat([0, 1, 2], [2, 3, 3])
    x = np.array([[0, 1, 1], [1, 1, 0],                     # task 0
                  [1, 0.3, 0], [1, 0.3, 0], [0, 1, 1],      # task 1
                  [1, 0.3, 0], [1, 0.3, 0], [1, 1, 0]],     # task 2
                 dtype=np.float32)
    layout = {"kind": "score-bias-acc", "routing_dtype": "float64"}

    def router(bias):
        T = len(bias)
        return dict(layout, classes=list(range(T)), feature_mean=[0.0] * T,
                    feature_scale=[1.0] * T, coef=(-np.eye(T)).tolist(), intercept=bias,
                    fit_balanced_accuracy=0.0, raw_argmin_fit_balanced_accuracy=0.0)

    states = _self_consistent([router([0.0]), router([0.0, 0.0]), router([0.0, 0.5, 0.0])], x, y)
    tc = rp._task_calibration()
    fitted = tc.predict_score_logreg(x.astype(np.float64), states[-1])
    assert np.bincount(y[fitted == y], minlength=3).tolist() == [1, 2, 1]
    assert states[-1]["fit_balanced_accuracy"] < states[-1]["raw_argmin_fit_balanced_accuracy"]
    arrays = {"calibration_features": x, "calibration_task_ids": y.astype(np.int16)}
    record = {"kind": "score-bias-acc", "prefix_models": states}
    assert rp.fit_accuracy_check(arrays, record)["errors"] == []
    # One more example lost is a real loss, however small the float gap.
    worse = json.loads(json.dumps(record))
    worse["prefix_models"][-1]["intercept"] = [0.0, 0.5, -0.5]
    _self_consistent(worse["prefix_models"], x, y)
    assert any("below the raw rule" in e for e in rp.fit_accuracy_check(arrays, worse)["errors"])


def test_a_logistic_tie_is_checked_against_the_recorded_route():
    """A legal float64 matrix product can record an exact tie between tasks 0
    and 1 while NumPy ranks task 2 between them, with a far smaller error
    bound than task 0's: checking NumPy's top two rejected the correct tie
    flag (P2 of the second round-5 review). The two summation orders of the
    decision of task 0 are RN(RN(L x0) - L) and RN(L x0 - L) (one FMA)."""
    from fractions import Fraction
    L = 1.0e6 + 0.1234
    stage0 = {"classes": [0], "feature_mean": [0.0], "feature_scale": [1.0],
              "coef": [[0.0]], "intercept": [0.0], "routing_dtype": "float64"}
    stage1 = {"classes": [0, 1], "feature_mean": [0.0] * 2, "feature_scale": [1.0] * 2,
              "coef": [[0.0, 0.0], [0.0, 0.0]], "intercept": [0.0, 0.0],
              "routing_dtype": "float64"}
    for k in range(1, 4000):
        x0 = float(np.float32(1.0) + np.float32(k) * np.float32(2.0 ** -23))
        x = np.array([[x0, 1.0, 0.5]], np.float32)
        probe = {"classes": [0, 1, 2], "feature_mean": [0.0] * 3, "feature_scale": [1.0] * 3,
                 "coef": [[L, -L, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 "intercept": [0.0, 0.0, 0.0], "routing_dtype": "float64"}
        numpy_d0 = rp.apply_router(x, probe, "float64")["decision"][0, 0]
        orders = ((x0 * L) - L, float(Fraction(x0) * Fraction(L) - Fraction(L)))
        device_d0 = max(orders)
        if device_d0 > numpy_d0:
            break
    else:
        raise AssertionError("no summation order above NumPy's found")
    gap = (device_d0 - numpy_d0) / 2
    # Device: task 0 ties task 1 exactly (route 0, tie); NumPy: 1 > 2 > 0.
    state = dict(probe, intercept=[0.0, device_d0, device_d0 - gap])
    applied = rp.apply_router(x, state, "float64")
    assert int(applied["pred"][0]) == 1 and not rp._within_rounding(applied, 0, 1, 2)
    arrays = {"true_task": np.array([2], np.int16), "scores": x,
              "route_by_stage": np.array([[-1, -1, 0]], np.int16),
              "tie_by_stage": np.array([[False, False, True]])}
    record = {"kind": "score-logreg", "prefix_models": [stage0, stage1, state]}
    out = rp.stored_router_agreement(arrays, record)
    assert out["errors"] == [] and out["tie_flag_errors"] == 0, out
    assert out["near_tie_disagreements"] == 1
    # A tie recorded at the highest candidate has no higher partner.
    forged = dict(arrays, route_by_stage=np.array([[-1, -1, 2]], np.int16))
    assert rp.stored_router_agreement(forged, record)["tie_flag_errors"] == 1


def test_near_ties_compare_every_candidate_with_the_winner():
    """A lower decision with a large error bound can overtake the winner even
    when the runner-up, with a tiny bound, cannot."""
    applied = {"classes": np.array([0, 1, 2]), "pred": np.array([0]),
               "decision": np.array([[1.0, 0.9, 0.5]]),
               "bound": np.array([[1e-6, 1e-6, 0.3]])}
    assert bool(rp.near_tie_rows(applied)[0])
    applied["bound"] = np.array([[1e-6, 1e-6, 1e-3]])
    assert not bool(rp.near_tie_rows(applied)[0])
    # A tie partner must lie above the recorded route.
    applied["bound"] = np.array([[1e-6, 0.1, 1e-6]])
    assert bool(rp.tie_partner_within_rounding(applied, [0])[0])
    assert not bool(rp.tie_partner_within_rounding(applied, [1])[0])


def test_the_float64_routing_self_test_detects_inexact_arithmetic(monkeypatch):
    pytest.importorskip("torch")
    import diltcl.task_inference as ti
    report = ti.float64_routing_self_test("cpu")
    assert report["ok"] and report["subnormal_scores"] > 0, report
    original = ti.apply_calibrator

    def float32_routing(scores, calibrator):
        return original(scores, {k: v for k, v in calibrator.items() if k != "routing_dtype"})

    monkeypatch.setattr(ti, "apply_calibrator", float32_routing)
    report = ti.float64_routing_self_test("cpu")
    assert not report["ok"] and report["decision_mismatches"] > 0, report
    # A device that flushes float32 subnormals passes the float64 test but not
    # the (reported) float32 one, which the float32 rebuild relies on.
    torch = pytest.importorskip("torch")

    def flushing(scores, calibrator):
        if "routing_dtype" not in calibrator:
            tiny = torch.finfo(torch.float32).tiny
            scores = torch.where(scores.abs() < tiny, torch.zeros_like(scores), scores)
        return original(scores, calibrator)

    monkeypatch.setattr(ti, "apply_calibrator", flushing)
    report = ti.float64_routing_self_test("cpu")
    assert report["ok"] and not report["float32_ok"], report
    assert report["float32_tie_mismatches"] > 0, report


def test_routes_read_from_the_wrong_expert_are_rejected():
    T, C, per_task = 4, 3, (7, 5, 9, 6)
    arrays = _synthetic(T=T, C=C, per_task=per_task)
    meta = {"num_tasks": T, "classes_per_task": C}
    assert rp.structural_errors(arrays, meta, list(per_task)) == []
    home = np.argwhere(arrays["route_by_stage"] == arrays["true_task"][:, None])
    i, s = (int(v) for v in home[0])
    wrong = {k: v.copy() for k, v in arrays.items()}
    wrong["routed_pred_local_by_stage"][i, s] = (wrong["oracle_pred_local"][i] + 1) % C
    assert any("predicted differently" in e
               for e in rp.structural_errors(wrong, meta, list(per_task)))
    # Raw and calibrated routers that chose the same task read the same expert.
    both = dict(arrays)
    both.update({f"raw_{k}": v.copy() for k, v in arrays.items()})
    away = np.argwhere((arrays["route_by_stage"] >= 0)
                       & (arrays["route_by_stage"] != arrays["true_task"][:, None]))
    i, s = (int(v) for v in away[0])
    both["raw_routed_pred_local_by_stage"][i, s] = (both["routed_pred_local_by_stage"][i, s] + 1) % C
    assert any("same task chosen by the raw and the calibrated router" in e
               for e in rp.structural_errors(both, meta, list(per_task)))


def test_float32_rebuild_counts_the_decisions_the_precision_moved():
    _, state, logits = _saturated_bias_acc_case()
    audit = _load("routing_precision_audit_unit", ROOT / "tools" / "routing_precision_audit.py")
    from diltcl.task_inference import task_scores
    scores = task_scores(logits).T.numpy().astype(np.float32)
    stage0 = {"kind": "score-bias-acc", "classes": [0], "feature_mean": [0.0],
              "feature_scale": [1.0], "coef": [[-1.0]], "intercept": [0.0],
              "routing_dtype": "float64"}
    arrays = {"true_task": np.array([1], np.int16), "label_local": np.array([2], np.int16),
              "oracle_pred_local": np.array([2], np.int16), "scores": scores,
              "route_by_stage": np.array([[-1, 1]], np.int16),
              "tie_by_stage": np.array([[False, False]]),
              "routed_pred_local_by_stage": np.array([[-1, 2]], np.int16)}
    rebuild = audit.float32_rebuild(arrays, {"kind": "score-bias-acc",
                                             "prefix_models": [stage0, state]})
    assert rebuild["routes_moved"] == 1 and rebuild["tie_flags_moved"] == 1
    assert rebuild["routes_moved_by_stage"] == [0, 1]
    moved = rebuild["arrays"]
    assert moved["route_by_stage"][0, 1] == 0 and moved["tie_by_stage"][0, 1]
    # Correct under float64 routing, wrong under float32 (routed to task 0).
    assert rp.recompute(arrays, 2)["result_matrix"][1, 1] == 100.0
    assert rp.recompute(moved, 2)["result_matrix"][1, 1] == 0.0


@pytest.mark.parametrize("calibration", ["score-bias-acc", "score-logreg"])
def test_float32_rebuild_reproduces_a_float32_routed_run(calibration, monkeypatch, tmp_path):
    """The audit's float32 rebuild of a float64-routed job must reproduce the
    same job run with float32 routing (v3.0.9's behaviour) exactly."""
    import run_class_il as class_il
    audit = _load("routing_precision_audit_e2e", ROOT / "tools" / "routing_precision_audit.py")
    merge = _load("merge_for_precision_audit", ROOT / "tools" / "merge_results.py")
    compare = _load("compare_for_precision_audit", ROOT / "tools" / "compare_runs.py")
    job = "grp__cifar__seed0__wsn"
    new = tmp_path / "new" / "artifacts" / job / "result"
    new.parent.mkdir(parents=True)
    _run_runner(str(new), str(new.parent / "predictions"), ("wsn",), calibration=calibration)
    original = class_il.fit_prefix_calibrators

    def float32_routed(*args, **kwargs):
        states = original(*args, **kwargs)
        for state in states:
            state.pop("routing_dtype")
        return states

    monkeypatch.setattr(class_il, "fit_prefix_calibrators", float32_routed)
    old = tmp_path / "old" / "artifacts" / job / "result"
    old.parent.mkdir(parents=True)
    _run_runner(str(old), str(old.parent / "predictions"), ("wsn",), calibration=calibration)
    monkeypatch.setattr(class_il, "fit_prefix_calibrators", original)
    for side, prefix in (("old", old), ("new", new)):
        assert merge.main(["--inputs", f"{prefix}.json", "--out",
                           str(tmp_path / f"{side}_merged" / "grp"), "--allow-unverified"]) == 0
    report = tmp_path / "precision.json"
    code = audit.main(["--run-dir", str(tmp_path / "new"), "--reference",
                       str(tmp_path / "old_merged"), "--out", str(report)])
    result = json.loads(report.read_text())
    entry = result["entries"][0]
    assert result["summary"]["problems"] == [] and entry["errors"] == []
    assert entry["routing_dtype"] == "float64"
    reference = entry["reference"]
    assert reference["trained_state_identical"] and reference["routers_identical"]
    assert reference["fixed_matrix_differences"] == {} == reference["fixed_metric_differences"]
    # The rebuild is NumPy's float32: it must reproduce the float32-routed run
    # wherever the device applies float32 routers as NumPy does (always on a
    # CPU; on a GPU the routing self-test reports it, float32_ok).
    from diltcl.task_inference import float64_routing_self_test
    if float64_routing_self_test(_runner_device())["float32_ok"]:
        assert code == 0 and result["status"] == "explained", result["summary"]
        assert reference["identical"] and reference["matrix_differences"] == {}
    # A router that differs from the reference's: a refit on another host CPU
    # when only the coefficients or intercepts of a logistic (or tied-slope)
    # router differ, any other difference otherwise -- in both tools.
    import shutil
    changed = tmp_path / "changed_reference"
    shutil.copytree(tmp_path / "old_merged", changed)
    changed_payload = json.loads((changed / "grp.json").read_text())
    last = changed_payload["calibration"]["cifar"]["wsn"]["0"]["prefix_models"][-1]
    last["intercept"][0] = float(np.nextafter(last["intercept"][0], np.inf))
    (changed / "grp.json").write_text(json.dumps(changed_payload))
    changed_report = tmp_path / "precision_changed.json"
    assert audit.main(["--run-dir", str(tmp_path / "new"), "--reference", str(changed),
                       "--out", str(changed_report)]) == 1
    changed_result = json.loads(changed_report.read_text())
    refit = calibration == "score-logreg"
    assert changed_result["entries"][0]["reference"]["router_refit"] is refit
    assert changed_result["summary"]["reference_routers_refitted" if refit else
                                     "reference_routers_different"] == ["grp/cifar/wsn/seed0"]
    changed_summary = compare.compare_group("grp", changed / "grp.json",
                                            tmp_path / "new_merged" / "grp.json")["summary"]
    assert changed_summary["routers_refitted" if refit else
                           "routers_different"] == ["cifar/wsn/seed0"]
    # compare_runs names the precision change instead of calling it an anomaly.
    out = compare.compare_group("grp", tmp_path / "old_merged" / "grp.json",
                                tmp_path / "new_merged" / "grp.json")
    assert out["router_precision"] == {"reference": "float32", "candidate": "float64"}
    assert out["summary"]["router_precision_changed"] is True
    assert out["summary"]["state_identical"] == 1


class _Archive(dict):
    """A ``result.npz`` stand-in: ``files`` and item access."""

    @property
    def files(self):
        return list(self)


_RECORD = {"kind": "score-logreg", "trained_state": {"content_sha256": "ab" * 32},
           "prefix_models": [{"classes": [0], "feature_mean": [0.0], "feature_scale": [1.0],
                              "coef": [[1.0]], "intercept": [0.0]},
                             {"classes": [0, 1], "feature_mean": [0.0, 0.0],
                              "feature_scale": [1.0, 1.0], "coef": [[1.0, 0.0], [0.0, 1.0]],
                              "intercept": [0.0, 0.0]}]}


def _reference_artifacts(audit, arrays, T, key="cifar/wsn/seed0"):
    """Merged artifacts of a calibrated job recorded as ``arrays``."""
    own, raw = rp.recompute(arrays, T), rp.recompute(arrays, T, prefix="raw_")
    metrics = dict(audit.per_seed_metrics(own["stats"], own["result_matrix"]))
    metrics.update(audit.fixed_metrics(own, raw))
    npz = _Archive()
    for name, suffix in audit.MATRICES:
        npz[key + suffix] = own[name]
    for name, suffix in audit.RAW_MATRICES:
        npz[key + suffix] = raw[name]
    counts = np.bincount(arrays["true_task"].astype(np.int64), minlength=T).tolist()
    payload = {"seeds": [0], "per_seed": {"cifar": {"wsn": [metrics]}},
               "benchmark_metadata": {"cifar": {"per_seed_counts": {
                   "0": {"isolation": {"test_samples_per_task": counts}}}}},
               "calibration": {"cifar": {"wsn": {"0": json.loads(json.dumps(_RECORD))}}}}
    return payload, npz


def test_the_reference_may_differ_only_by_the_near_ties_counted_in_a_cell():
    """A score-logreg decision near a tie may go either way between NumPy's
    float32 and the GPU's, but only among the tasks within rounding of its
    winner. A differing cell is accepted up to the number of such decisions
    that can move that count, by integer counts -- for the conditional
    accuracy too, whose denominator (correct routes) moves with the decision
    -- and never for the raw router, the oracle accuracies, or a reference
    whose metrics do not follow from its own matrices."""
    audit = _load("routing_precision_audit_cells", ROOT / "tools" / "routing_precision_audit.py")
    T = 2
    arrays = _synthetic(T=T, C=3, per_task=(8, 8), seed=4)
    arrays.update({f"raw_{k}": v.copy() for k, v in list(arrays.items())})
    home = [i for i in range(arrays["true_task"].size)
            if arrays["true_task"][i] == 0 and arrays["route_by_stage"][i, 1] == 0
            and arrays["routed_pred_local_by_stage"][i, 1] == arrays["label_local"][i]]
    assert len(home) >= 2

    def reference_with(moved_examples):
        moved = {k: v.copy() for k, v in arrays.items()}
        for i in moved_examples:        # sent to task 1, whose expert errs
            moved["route_by_stage"][i, 1] = 1
            moved["routed_pred_local_by_stage"][i, 1] = (moved["label_local"][i] + 1) % 3
        return _reference_artifacts(audit, moved, T)

    def cells(**at_stage1_task0):
        out = {name: np.zeros((T, T), np.int64) for name in (
            "routes_lost_right", "routes_gained_right", "routes_lost_wrong",
            "routes_gained_wrong", "ties_lost", "ties_gained", "near")}
        for name, value in at_stage1_task0.items():
            out[name][1, 0] = value
        return out

    own, raw = rp.recompute(arrays, T), rp.recompute(arrays, T, prefix="raw_")
    job = {"benchmark": "cifar", "method": "wsn", "seed": 0, "disjoint": True,
           "record": json.loads(json.dumps(_RECORD)),
           "counts": np.bincount(arrays["true_task"].astype(np.int64), minlength=T),
           "rebuilt": own, "raw": raw,
           "metrics": audit.per_seed_metrics(own["stats"], own["result_matrix"]),
           "fixed": audit.fixed_metrics(own, raw), "cells": cells()}
    same, same_npz = _reference_artifacts(audit, arrays, T)
    assert audit.compare_with_reference(job, same, same_npz)["identical"]
    payload, npz = reference_with(home[:1])
    assert not np.array_equal(npz["cifar/wsn/seed0/conditional_class"],
                              own["conditional_class_matrix"])
    out = audit.compare_with_reference(job, payload, npz)
    assert not out["identical"] and "cifar/wsn/seed0/conditional_class" in out["matrix_differences"]
    # One near-tie decision of task 0 at stage 1 routed home here, whose own
    # expert is right: the reference may have lost that correct route.
    job["cells"] = cells(routes_lost_right=1, ties_gained=1, near=1)
    out = audit.compare_with_reference(job, payload, npz)
    assert out["identical"] and out["matrix_differences"] == {}, out
    assert "cifar/wsn/seed0/conditional_class" in out["within_near_ties"]
    assert out["metric_differences"]             # explained by the cells
    # ...but not gained one: a near-tie can move a count only one way.
    job["cells"] = cells(routes_gained_right=1, ties_gained=1, near=1)
    assert not audit.compare_with_reference(job, payload, npz)["identical"]
    # ...and not with a wrong own expert, which moves no correct prediction.
    job["cells"] = cells(routes_lost_wrong=1, ties_gained=1, near=1)
    assert not audit.compare_with_reference(job, payload, npz)["identical"]
    # A near-tie between two other tasks can change only the tie count.
    job["cells"] = cells(ties_gained=1, near=1)
    out = audit.compare_with_reference(job, payload, npz)
    assert not out["identical"] and "cifar/wsn/seed0/task_inference" in out["matrix_differences"]
    # Two decisions moved where one near-tie was counted: a real difference.
    job["cells"] = cells(routes_lost_right=1, ties_gained=1, near=1)
    payload2, npz2 = reference_with(home[:2])
    assert not audit.compare_with_reference(job, payload2, npz2)["identical"]
    # The raw router and the oracle accuracies get no tolerance.
    bad = _Archive(npz)
    bad["cifar/wsn/seed0/raw_result"] = npz["cifar/wsn/seed0/raw_result"] + 1.0
    assert not audit.compare_with_reference(job, payload, bad)["identical"]
    worse = json.loads(json.dumps(payload))
    worse["per_seed"]["cifar"]["wsn"][0]["MicroOracleACC"] += 1.0
    out = audit.compare_with_reference(job, worse, npz)
    assert not out["identical"] and "MicroOracleACC" in out["fixed_metric_differences"]
    # A reference metric that its own matrices do not imply is never accepted,
    # even where the matrices are within near-ties.
    inconsistent = json.loads(json.dumps(payload))
    inconsistent["per_seed"]["cifar"]["wsn"][0]["ACC"] += 5.0
    out = audit.compare_with_reference(job, inconsistent, npz)
    assert not out["identical"] and "ACC" in out["reference_inconsistent_metrics"]
    # What the precision cannot change must be in the reference to be compared.
    for field, label in (("benchmark_metadata", "test-set sizes"),
                         ("calibration", "trained-state hash")):
        partial = {k: v for k, v in same.items() if k != field}
        out = audit.compare_with_reference(job, partial, same_npz)
        assert not out["identical"] and label in out["missing_in_reference"], field
    other = json.loads(json.dumps(same))
    other["calibration"]["cifar"]["wsn"]["0"]["prefix_models"][1]["intercept"] = [0.0, 1e-9]
    out = audit.compare_with_reference(job, other, same_npz)
    assert not out["identical"] and "stage1/intercept" in out["router_differences"]


def test_near_tie_cells_follow_the_direction_a_decision_can_move():
    """float32_rebuild classifies every score-logreg near-tie decision: routed
    home here (another order can only lose the route), own task within
    rounding but not winning (can only gain it), or neither (only its tie
    flag can change); TF32 references make the bias routers inexact too."""
    audit = _load("routing_precision_audit_direction", ROOT / "tools" / "routing_precision_audit.py")
    stage0 = {"classes": [0], "feature_mean": [0.0], "feature_scale": [1.0],
              "coef": [[1.0]], "intercept": [0.0]}
    stage1 = {"classes": [0, 1], "feature_mean": [0.0, 0.0], "feature_scale": [1.0, 1.0],
              "coef": [[1.0, 0.0], [0.0, 1.0]], "intercept": [0.0, 0.0]}
    scores = np.array([[0.5, 0.5], [0.5, 0.5], [0.9, 0.1], [0.1, 0.9]], np.float32)
    true_task = np.array([0, 1, 0, 1], np.int16)
    arrays = {"true_task": true_task, "label_local": np.array([1, 1, 1, 1], np.int16),
              "oracle_pred_local": np.array([1, 0, 1, 1], np.int16), "scores": scores,
              "route_by_stage": np.array([[0, 0], [-1, 0], [0, 0], [-1, 1]], np.int16),
              "tie_by_stage": np.array([[False, True], [False, True], [False, False],
                                        [False, False]]),
              "routed_pred_local_by_stage": np.array([[1, 1], [-1, 1], [1, 1], [-1, 1]],
                                                     np.int16)}
    record = {"kind": "score-logreg", "prefix_models": [stage0, stage1]}
    rebuild = audit.float32_rebuild(arrays, record)
    assert rebuild["routes_moved"] == 0 and rebuild["precision"] == "float32"
    # Rows 0 and 1 tie exactly (task 0 wins both); rows 2 and 3 are clear.
    assert rebuild["near_tie_mask"][:, 1].tolist() == [True, True, False, False]
    assert rebuild["near_home_mask"][:, 1].tolist() == [True, False, False, False]
    assert rebuild["near_away_mask"][:, 1].tolist() == [False, True, False, False]
    cells = audit.near_tie_cells(rebuild, arrays, 2)
    assert cells["routes_lost_right"][1].tolist() == [1, 0]     # row 0: home, expert right
    assert cells["routes_gained_wrong"][1].tolist() == [0, 1]   # row 1: own task 1 within rounding
    assert cells["ties_lost"][1].tolist() == [1, 1] and cells["ties_gained"][1].tolist() == [0, 0]
    # A bias router is exact in float32, but not under a TF32 reference.
    bias = {"kind": "score-bias-acc", "prefix_models": [
        dict(stage0, coef=[[-1.0]]), dict(stage1, coef=[[-1.0, -0.0], [-0.0, -1.0]])]}
    assert not audit.float32_rebuild(arrays, bias)["near_tie_mask"].any()
    tf32 = audit.float32_rebuild(arrays, bias, tf32=True)
    assert tf32["precision"] == "tf32" and tf32["near_tie_mask"][:, 1].tolist()[:2] == [True, True]


def test_the_audit_reports_a_gpu_whose_float32_routing_is_not_numpys(tmp_path):
    audit = _load("routing_precision_audit_env", ROOT / "tools" / "routing_precision_audit.py")
    assert audit.device_float32_routing(tmp_path) is None
    for value in (True, False):
        (tmp_path / "environment.json").write_text(json.dumps(
            {"torch_probe": {"float64_routing": {"ok": True, "float32_ok": value}}}))
        assert audit.device_float32_routing(tmp_path) is value


def _saturated_setup(tasks=6, classes=3, scale=300.0):
    """Fake isolation setup whose experts' logits are scaled until most
    softmax entropies are exactly zero: the tie-heavy regime in which float32
    and float64 application of a fitted router differ."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import run_class_il as class_il

    class Saturated(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(12, 16)
            self.fc2 = nn.Linear(16, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x.view(x.size(0), -1)))) * scale

    def fake_isolation(benchmark, cfg, seed, **_):
        g = torch.Generator().manual_seed(seed)
        train, test = [], []
        for t in range(tasks):
            x = torch.randn(120, 12, generator=g) + 0.5 * t
            y = torch.randint(0, classes, (120,), generator=g)
            xt = torch.randn(60, 12, generator=g) + 0.5 * t
            yt = torch.randint(0, classes, (60,), generator=g)
            train.append(DataLoader(TensorDataset(x, y), batch_size=16))
            test.append(DataLoader(TensorDataset(xt, yt), batch_size=16))
        return class_il.IsolationSetup(train, test, (lambda: Saturated(classes)),
                                       tasks, classes, True)
    return fake_isolation


def _run_saturated(out_prefix, calibration, *, seed=0, tasks=6, classes=3):
    """The real runner on saturated experts (:func:`_saturated_setup`)."""
    pytest.importorskip("torch")
    import test_runner_integration as tri
    import run_class_il as class_il

    cfg = tri._fast_config()
    cfg.c100_num_tasks, cfg.c100_classes_per_task = tasks, classes
    with tri._Patched(class_il, {"_isolation_setup": _saturated_setup(tasks, classes),
                                 "preflight": tri._NoPreflight}):
        class_il.run(cfg, ["cifar"], ["supsup"], [seed], out_prefix=out_prefix,
                     calibration=calibration,
                     predictions_dir=str(pathlib.Path(out_prefix).parent / "predictions"),
                     checkpoint_dir=str(pathlib.Path(out_prefix).parent / "checkpoints"))
    with open(f"{out_prefix}.json") as handle:
        return json.load(handle)


def test_saturated_experts_pass_every_audit_and_the_float32_rebuild(monkeypatch, tmp_path):
    """Tie-heavy outputs of the real runner (on the GPU when there is one):
    every enforced check passes, the checkpoints re-evaluate exactly, and
    rebuilding the float64-routed decisions in float32 reproduces the same
    job run with float32 routing (v3.0.9) decision by decision."""
    import run_class_il as class_il
    audit = _load("routing_precision_audit_saturated", ROOT / "tools" / "routing_precision_audit.py")
    evaluator = _load("evaluate_checkpoints_saturated", ROOT / "tools" / "evaluate_checkpoints.py")
    job = "grp__cifar__seed0__supsup"
    new = tmp_path / "new" / "artifacts" / job / "result"
    new.parent.mkdir(parents=True)
    payload = _run_saturated(str(new), "score-bias-acc")
    arrays, _ = rp.load_predictions(new.parent / payload["calibration"]["cifar"]["supsup"]["0"]
                                    ["predictions"]["relative_path"])
    assert float(np.mean(arrays["scores"] == 0)) > 0.2          # saturated, as intended
    entries = rp.check_job(new.with_suffix(".json"), refit=True)
    assert len(entries) == 1 and entries[0]["errors"] == [], entries[0]["errors"]
    assert entries[0]["refit"]["parameters_identical"] is True
    evaluated = evaluator.evaluate_job(new.with_suffix(".json"), device=_runner_device(),
                                       methods=("supsup",), setup_fn=_saturated_setup())
    assert len(evaluated) == 1 and evaluated[0]["errors"] == [] and evaluated[0]["exact"]
    assert evaluated[0]["per_example_differences"] == {}
    original = class_il.fit_prefix_calibrators

    def float32_routed(*args, **kwargs):
        states = original(*args, **kwargs)
        for state in states:
            state.pop("routing_dtype")
        return states

    monkeypatch.setattr(class_il, "fit_prefix_calibrators", float32_routed)
    old = tmp_path / "old" / "artifacts" / job / "result"
    old.parent.mkdir(parents=True)
    _run_saturated(str(old), "score-bias-acc")
    monkeypatch.setattr(class_il, "fit_prefix_calibrators", original)
    merge = _load("merge_for_saturated", ROOT / "tools" / "merge_results.py")
    assert merge.main(["--inputs", f"{old}.json", "--out", str(tmp_path / "ref" / "grp"),
                       "--allow-unverified"]) == 0
    report = tmp_path / "precision.json"
    code = audit.main(["--run-dir", str(tmp_path / "new"), "--reference", str(tmp_path / "ref"),
                       "--out", str(report)])
    result = json.loads(report.read_text())
    entry = result["entries"][0]
    reference = entry["reference"]
    # What the precision cannot change matches the float32-routed run on any device.
    assert result["summary"]["problems"] == [] and entry["errors"] == []
    assert reference["trained_state_identical"] is True and reference["routers_identical"] is True
    assert reference["test_counts_identical"] is True
    assert reference["fixed_matrix_differences"] == {} == reference["fixed_metric_differences"]
    assert reference["reference_inconsistent_metrics"] == {}
    # Decision by decision, the float32 rebuild equals the float32-routed run
    # wherever the device's float32 routing is NumPy's: always on a CPU; on a
    # GPU the routing self-test reports it (float32_ok), and saturated scores
    # full of subnormals would expose a device that flushes them.
    from diltcl.task_inference import float64_routing_self_test
    exact32 = float64_routing_self_test(_runner_device())["float32_ok"]
    old_payload = json.loads(old.with_suffix(".json").read_text())
    old_arrays, _ = rp.load_predictions(old.parent / old_payload["calibration"]["cifar"]["supsup"]["0"]
                                        ["predictions"]["relative_path"])
    record = payload["calibration"]["cifar"]["supsup"]["0"]
    rebuild = audit.float32_rebuild(arrays, record)
    rebuilt = rebuild["arrays"]
    known = ~rebuild["unknown_prediction_mask"]
    same = (np.array_equal(rebuilt["route_by_stage"], old_arrays["route_by_stage"])
            and np.array_equal(rebuilt["tie_by_stage"], old_arrays["tie_by_stage"])
            and np.array_equal(rebuilt["routed_pred_local_by_stage"][known],
                               old_arrays["routed_pred_local_by_stage"][known]))
    if exact32:
        assert same
        assert code == 0 and result["status"] == "explained", result["summary"]
    if str(_runner_device()) == "cpu":
        # On the CPU these experts are known to move decisions; on a GPU the
        # scores differ in their last bits and the count may differ too.
        assert entry["routes_moved"] > 0
    print(f"saturated supsup: {entry['routes_moved']} of {entry['decisions_compared']} "
          f"decisions moved by float64 routing; float32 rebuild "
          f"{'equals' if same else 'differs from'} the float32-routed run "
          f"(device float32 routing {'is' if exact32 else 'is not'} NumPy's)")

    # compare_runs: only the calibrated router's outputs may differ, and are
    # then named a routing-precision effect.
    compare = _load("compare_for_saturated", ROOT / "tools" / "compare_runs.py")
    assert merge.main(["--inputs", f"{new}.json", "--out", str(tmp_path / "cand" / "grp"),
                       "--allow-unverified"]) == 0
    summary = compare.compare_group("grp", tmp_path / "ref" / "grp.json",
                                    tmp_path / "cand" / "grp.json")["summary"]
    assert summary["state_identical"] == 1 and summary["same_state_unexplained"] == []
    assert summary["routing_precision_differences"] == summary["same_state_different_outputs"]

    # Everything the precision cannot change must match the reference exactly:
    # the raw router's arrays and metrics, and the oracle accuracies.
    import shutil

    def tampered(name, edit):
        folder = tmp_path / name
        shutil.copytree(tmp_path / "ref", folder)
        with np.load(folder / "grp.npz", allow_pickle=False) as archive:
            arrays_ = {k: np.asarray(archive[k]) for k in archive.files}
        payload_ = json.loads((folder / "grp.json").read_text())
        edit(arrays_, payload_["per_seed"]["cifar"]["supsup"][0])
        np.savez(folder / "grp.npz", **arrays_)
        (folder / "grp.json").write_text(json.dumps(payload_))
        return folder

    def bump_raw(arrays_, metrics):
        arrays_["cifar/supsup/seed0/raw_result"][-1, 0] += 1.0

    def bump(metric):
        def edit(arrays_, metrics):
            metrics[metric] += 1.0
        return edit

    for name, edit, field in (("raw_matrix", bump_raw, "fixed_matrix_differences"),
                              ("raw_metric", bump("RawTA"), "fixed_metric_differences"),
                              ("oracle", bump("MicroOracleACC"), "fixed_metric_differences")):
        folder = tampered(name, edit)
        out = tmp_path / f"precision_{name}.json"
        assert audit.main(["--run-dir", str(tmp_path / "new"), "--reference", str(folder),
                           "--out", str(out)]) == 1
        result = json.loads(out.read_text())
        assert result["status"] == "unexplained", name
        assert result["entries"][0]["reference"][field], name
        summary = compare.compare_group("grp", folder / "grp.json",
                                        tmp_path / "cand" / "grp.json")["summary"]
        assert summary["same_state_unexplained"] == ["cifar/supsup/seed0"], name
        assert summary["routing_precision_differences"] == [], name

    # Routers that differ from the reference's: the job differs from the
    # reference, and the reference's own routers are applied to its scores for
    # the diagnosis. A score-bias-acc router cannot change with the host CPU,
    # so this is a router difference, not a refit, in both tools.
    # Stage 0 has a single candidate, so a different intercept there moves no
    # decision; a large bias on task 0 at the last stage moves many.
    for name, stage, reproduced in (("refit_stage0", 0, True), ("refit_large", -1, False)):
        folder = tmp_path / name
        shutil.copytree(tmp_path / "ref", folder)
        payload_ = json.loads((folder / "grp.json").read_text())
        models = payload_["calibration"]["cifar"]["supsup"]["0"]["prefix_models"]
        models[stage]["intercept"][0] += 0.5 if reproduced else 100.0
        (folder / "grp.json").write_text(json.dumps(payload_))
        out = tmp_path / f"precision_{name}.json"
        assert audit.main(["--run-dir", str(tmp_path / "new"), "--reference", str(folder),
                           "--out", str(out)]) == 1
        result = json.loads(out.read_text())
        assert result["status"] == "unexplained", name
        reference = result["entries"][0]["reference"]
        assert reference["routers_identical"] is False and reference["router_differences"], name
        assert reference["router_refit"] is False, name
        check = reference["with_reference_routers"]
        assert (check["routes_moved_by_router_change"] > 0) is (not reproduced), (name, check)
        assert result["summary"]["reference_routers_different"] == ["grp/cifar/supsup/seed0"]
        assert result["summary"]["reference_routers_refitted"] == []
        if exact32 or not reproduced:      # as above: the rebuild is NumPy's float32
            assert check["identical"] is reproduced, (name, check)
            assert result["summary"][
                "reference_router_differences_reproduced_with_reference_routers"] == int(reproduced)
        summary = compare.compare_group("grp", folder / "grp.json",
                                        tmp_path / "cand" / "grp.json")["summary"]
        assert summary["routers_different"] == ["cifar/supsup/seed0"], name
        assert summary["routers_refitted"] == [] and summary["identical"] == 0, name
        assert summary["routing_precision_differences"] == [], name
        assert summary["same_state_unexplained"] == summary["same_state_different_outputs"], name

    # Nothing to audit, or a group the reference lacks, is a failure.
    (tmp_path / "empty").mkdir()
    assert audit.main(["--run-dir", str(tmp_path / "empty")]) == 1
    other = tmp_path / "other_reference"
    other.mkdir()
    for suffix in (".json", ".npz"):
        shutil.copy(tmp_path / "ref" / f"grp{suffix}", other / f"other{suffix}")
    out = tmp_path / "precision_other.json"
    assert audit.main(["--run-dir", str(tmp_path / "new"), "--reference", str(other),
                       "--out", str(out)]) == 1
    result = json.loads(out.read_text())
    assert result["status"] == "failed"
    assert "group grp is not in the reference" in result["summary"]["problems"]
    assert any("reference group other" in p and "not audited" in p
               for p in result["summary"]["problems"])
    # The audit must cover the reference: a calibrated reference run that was
    # not audited fails it...
    extra = tmp_path / "extra_reference"
    shutil.copytree(tmp_path / "ref", extra)
    ref_payload = json.loads((extra / "grp.json").read_text())
    by_seed = ref_payload["calibration"]["cifar"]["supsup"]
    by_seed["1"] = json.loads(json.dumps(by_seed["0"]))
    (extra / "grp.json").write_text(json.dumps(ref_payload))
    out = tmp_path / "precision_extra.json"
    assert audit.main(["--run-dir", str(tmp_path / "new"), "--reference", str(extra),
                       "--out", str(out)]) == 1
    problems = json.loads(out.read_text())["summary"]["problems"]
    assert problems == ["reference group grp: 1 of 2 calibrated run(s) not audited, "
                        "e.g. cifar/supsup/seed1"], problems
    # ...and so does an isolation method of a calibrated job without its record.
    bare = tmp_path / "bare"
    shutil.copytree(tmp_path / "new", bare)
    job_json = bare / "artifacts" / job / "result.json"
    bare_payload = json.loads(job_json.read_text())
    bare_payload["calibration"] = {"cifar": {}}
    job_json.write_text(json.dumps(bare_payload))
    out = tmp_path / "precision_bare.json"
    assert audit.main(["--run-dir", str(bare), "--reference", str(tmp_path / "ref"),
                       "--out", str(out)]) == 1
    result = json.loads(out.read_text())
    assert result["status"] == "failed"
    assert any("without a calibration record" in e for x in result["entries"] for e in x["errors"])
    # The records are checked as the verifier checks them before any rebuild.
    forged = tmp_path / "forged"
    shutil.copytree(tmp_path / "new", forged)
    pred_path = forged / "artifacts" / job / record["predictions"]["relative_path"]
    forged_arrays, forged_meta = rp.load_predictions(pred_path)
    # Forge a raw route that both routers had sent to the example's own task:
    # the forged one then leaves the own task and the calibrated route, so no
    # structural check applies to it and only the argmin check can catch it.
    raw_s1 = forged_arrays["raw_route_by_stage"][:, 1]
    rows = np.flatnonzero((forged_arrays["raw_true_task"] <= 1)
                          & (raw_s1 == forged_arrays["raw_true_task"])
                          & (forged_arrays["route_by_stage"][:, 1] == raw_s1))
    assert rows.size, "no example routed home by both routers at stage 1"
    forged_arrays["raw_route_by_stage"][rows[0], 1] = 1 - forged_arrays["raw_route_by_stage"][rows[0], 1]
    np.savez_compressed(pred_path, meta_json=np.asarray(json.dumps(forged_meta)), **forged_arrays)
    forged_json = forged / "artifacts" / job / "result.json"
    forged_payload = json.loads(forged_json.read_text())
    forged_payload["calibration"]["cifar"]["supsup"]["0"]["predictions"]["sha256"] = \
        rp.sha256_file(pred_path)
    forged_json.write_text(json.dumps(forged_payload))
    out = tmp_path / "precision_forged.json"
    assert audit.main(["--run-dir", str(forged), "--reference", str(tmp_path / "ref"),
                       "--out", str(out)]) == 1
    entry = json.loads(out.read_text())["entries"][0]
    assert any("argmin of the recorded scores" in e for e in entry["errors"]), entry["errors"]


def _runner_device():
    """The device ``run_class_il.run`` trains and evaluates on (CUDA when present).

    Re-evaluation must run there too: CPU and GPU kernels round differently,
    so the recorded float32 scores are reproduced exactly only on the device
    that produced them (as on the server, where both run on the GPU).
    """
    from diltcl.utils import get_device
    return get_device()


@pytest.mark.parametrize("calibration", ["none", "score-logreg", "score-bias", "score-bias-acc"])
def test_checkpoints_reevaluate_bit_identically(calibration):
    """Released checkpoints alone regenerate every routed and oracle result."""
    evaluator = _load("evaluate_checkpoints_under_test", ROOT / "tools" / "evaluate_checkpoints.py")
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "job" / "result"
        out.parent.mkdir()
        _run_runner(str(out), str(out.parent / "predictions"),
                    ("packnet", "supsup", "wsn", "diltcl"), calibration=calibration,
                    checkpoint_dir=str(out.parent / "checkpoints"))
        entries = evaluator.evaluate_job(out.with_suffix(".json"), device=_runner_device(),
                                         methods=evaluator.ISOLATION,
                                         setup_fn=_fake_isolation_setup())
        assert sorted(e["method"] for e in entries) == sorted(evaluator.ISOLATION)
        for entry in entries:
            assert entry["errors"] == [], entry
            assert entry["exact"], entry
            assert entry["file_sha256_ok"] and entry["content_sha256_ok"]
            assert entry["per_example_differences"] == {}
        # A corrupted checkpoint is rejected before evaluation results are trusted.
        victim = out.parent / "checkpoints" / "cifar" / "wsn" / "seed0.pt"
        import torch
        state = torch.load(victim, weights_only=True)
        key = next(k for k, v in state["model"].items() if v.is_floating_point())
        state["model"][key] = state["model"][key] + 1.0
        torch.save(state, victim)
        entries = evaluator.evaluate_job(out.with_suffix(".json"), device=_runner_device(),
                                         methods=("wsn",), setup_fn=_fake_isolation_setup())
        assert any("SHA-256" in e for e in entries[0]["errors"])
        assert any("content" in e for e in entries[0]["errors"])


def test_reevaluation_fails_on_any_difference_in_exact_mode():
    """Exact mode: one changed decision, a NaN, or a missing file is an error."""
    evaluator = _load("evaluate_checkpoints_strict_test", ROOT / "tools" / "evaluate_checkpoints.py")
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "job" / "result"
        out.parent.mkdir()
        _run_runner(str(out), str(out.parent / "predictions"), ("wsn",),
                    checkpoint_dir=str(out.parent / "checkpoints"))
        result_json = out.with_suffix(".json")
        payload = json.loads(result_json.read_text())
        record = payload["calibration"]["cifar"]["wsn"]["0"]
        ppath = out.parent / record["predictions"]["relative_path"]
        # 1. One per-example decision changed; the artifact's hash is updated so
        #    that only the decision itself differs.
        with np.load(ppath, allow_pickle=False) as archive:
            arrays = {k: np.asarray(archive[k]) for k in archive.files}
        route = arrays["route_by_stage"].copy()
        row = int(np.flatnonzero(route[:, -1] >= 0)[0])
        route[row, -1] = (route[row, -1] + 1) % route.shape[1]
        arrays["route_by_stage"] = route
        np.savez(ppath, **arrays)
        record["predictions"]["sha256"] = hashlib.sha256(ppath.read_bytes()).hexdigest()
        result_json.write_text(json.dumps(payload))
        entries = evaluator.evaluate_job(result_json, device=_runner_device(), methods=("wsn",),
                                         setup_fn=_fake_isolation_setup())
        assert any("route_by_stage: 1 per-example" in e for e in entries[0]["errors"])
        tolerant = evaluator.evaluate_job(result_json, device=_runner_device(), methods=("wsn",),
                                          atol=0.05, setup_fn=_fake_isolation_setup())
        assert tolerant[0]["errors"] == [] and not tolerant[0]["exact"]
        assert tolerant[0]["per_example_differences"] == {"route_by_stage": 1}
        # 2. A referenced predictions file that is missing is an error, not a skip.
        ppath.unlink()
        entries = evaluator.evaluate_job(result_json, device=_runner_device(), methods=("wsn",),
                                         setup_fn=_fake_isolation_setup())
        assert any("missing" in e for e in entries[0]["errors"])
        # 3. A NaN in the artifact never passes, whatever the tolerance.
        with np.load(f"{out}.npz", allow_pickle=False) as archive:
            matrices = {k: np.asarray(archive[k]) for k in archive.files}
        matrices["cifar/wsn/seed0/task_inference"] = matrices["cifar/wsn/seed0/task_inference"].copy()
        matrices["cifar/wsn/seed0/task_inference"][0, 0] = np.nan
        np.savez(f"{out}.npz", **matrices)
        del record["predictions"]
        result_json.write_text(json.dumps(payload))
        for atol in (0.0, 100.0):
            entries = evaluator.evaluate_job(result_json, device=_runner_device(), methods=("wsn",),
                                             atol=atol, setup_fn=_fake_isolation_setup())
            assert any("non-finite" in e for e in entries[0]["errors"])


def test_shard_reports_must_cover_every_class_il_job(tmp_path):
    evaluator = _load("evaluate_checkpoints_combine_test", ROOT / "tools" / "evaluate_checkpoints.py")
    run = tmp_path / "run"
    jobs = [str((run / "artifacts" / f"job{i}" / "result.json").resolve()) for i in range(3)]

    def shard(spec, results, methods=evaluator.ISOLATION):
        entry = {"job": "x", "benchmark": "cifar", "method": "wsn", "seed": 0, "errors": [],
                 "exact": True}
        return {"run_dir": str(run.resolve()), "shard": spec, "methods": list(methods),
                "results": results, "run_results": jobs,
                "checkpoints_evaluated": len(results), "entries": [entry] * len(results)}

    def combine(reports):
        paths = []
        for i, report in enumerate(reports):
            path = tmp_path / f"r{i}.json"
            path.write_text(json.dumps(report))
            paths.append(path)
        return evaluator.combine(paths, tmp_path / "combined.json")

    assert combine([shard("0/2", jobs[0::2]), shard("1/2", jobs[1::2])]) == 0
    assert json.loads((tmp_path / "combined.json").read_text())["status"] == "identical"
    assert combine([shard("0/2", jobs[0::2])]) == 1                       # shard 1 missing
    assert combine([shard("0/2", jobs[0:1]), shard("1/2", jobs[1::2])]) == 1   # job2 missing
    assert combine([shard("0/2", jobs[0::2]),
                    shard("1/2", jobs[1::2], methods=("wsn",))]) == 1      # a method skipped
    assert combine([shard("0/2", jobs[0::2]),                             # any method order
                    shard("1/2", jobs[1::2], methods=evaluator.ISOLATION[::-1])]) == 0


def test_a_job_that_raises_is_reported_and_the_others_kept(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    evaluator = _load("evaluate_checkpoints_raise_test", ROOT / "tools" / "evaluate_checkpoints.py")
    good, bad = tmp_path / "good" / "result.json", tmp_path / "bad" / "result.json"

    def fake_evaluate_job(path, **_):
        if path.parent.name == "bad":
            raise RuntimeError("CUDA out of memory")
        return [{"job": "good", "benchmark": "cifar", "method": "wsn", "seed": 0,
                 "router": "none", "errors": [], "exact": True,
                 "max_abs_matrix_difference": 0.0, "metric_differences": {},
                 "per_example_differences": {}, "metrics_compared": [],
                 "metrics_not_compared": []}]

    monkeypatch.setattr(evaluator, "evaluate_job", fake_evaluate_job)
    report_path = tmp_path / "report.json"
    code = evaluator.main(["--result", str(bad), "--result", str(good), "--device", "cpu",
                           "--out", str(report_path)])
    assert code == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and report["checkpoints_evaluated"] == 2
    assert [e["job"] for e in report["entries"]] == ["bad", "good"]
    assert "CUDA out of memory" in report["entries"][0]["errors"][0]
    assert report["entries"][1]["errors"] == [] and report["entries"][1]["exact"]


def test_structural_checks_reject_malformed_recordings():
    T, C, per_task = 4, 3, (7, 5, 9, 6)
    arrays = _synthetic(T=T, C=C, per_task=per_task)
    meta = {"num_tasks": T, "classes_per_task": C}
    assert rp.structural_errors(arrays, meta, list(per_task)) == []

    def broken(mutate):
        copy = {k: v.copy() for k, v in arrays.items()}
        mutate(copy)
        return rp.structural_errors(copy, meta, list(per_task))

    first_of_task1 = int(np.flatnonzero(arrays["true_task"] == 1)[0])
    assert any("outside the candidates" in e
               for e in broken(lambda a: a["route_by_stage"].__setitem__((0, 0), 3)))
    assert any("before the task arrived" in e
               for e in broken(lambda a: a["route_by_stage"].__setitem__((first_of_task1, 0), 0)))
    assert any("non-finite" in e
               for e in broken(lambda a: a["scores"].__setitem__((0, 1), np.nan)))
    assert any("counts" in e for e in rp.structural_errors(arrays, meta, [7, 5, 9, 7]))
    raw = dict(arrays)
    raw.update({f"raw_{k}": v.copy() for k, v in arrays.items()})
    assert rp.structural_errors(raw, meta, list(per_task)) == []
    raw["raw_scores"][0, 0] += 1.0
    errors = rp.structural_errors(raw, meta, list(per_task))
    assert any("raw_scores differs" in e for e in errors)


def _linear_routes(scores, true_task, models):
    T = scores.shape[1]
    routes = np.full(scores.shape, -1, np.int16)
    ties = np.zeros(scores.shape, bool)
    for s, state in enumerate(models):
        rows = true_task <= s
        x = scores[rows, :s + 1]
        d = ((x - np.asarray(state["feature_mean"], np.float32))
             / np.asarray(state["feature_scale"], np.float32)) \
            @ np.asarray(state["coef"], np.float32).T + np.asarray(state["intercept"], np.float32)
        win = d == d.max(axis=1, keepdims=True)
        routes[rows, s] = np.where(win, np.arange(s + 1)[None, :], T).min(axis=1)
        ties[rows, s] = win.sum(axis=1) > 1
    return routes, ties


def test_stored_bias_routers_must_match_exactly():
    rng = np.random.default_rng(3)
    T, N = 3, 45
    true_task = np.repeat(np.arange(T), N // T).astype(np.int16)
    scores = rng.normal(size=(N, T)).astype(np.float32)
    models = [{"classes": list(range(s + 1)), "feature_mean": [0.0] * (s + 1),
               "feature_scale": [1.0] * (s + 1),
               "coef": (-1.7 * np.eye(s + 1)).tolist(),
               "intercept": rng.normal(size=s + 1).tolist()} for s in range(T)]
    routes, ties = _linear_routes(scores, true_task, models)
    arrays = {"true_task": true_task, "scores": scores, "route_by_stage": routes,
              "tie_by_stage": ties}
    record = {"kind": "score-bias", "prefix_models": models}
    out = rp.stored_router_agreement(arrays, record)
    assert out["route_agreement"] == 1.0 and out["errors"] == []
    routes = routes.copy()
    routes[-1, -1] = (routes[-1, -1] + 1) % T
    out = rp.stored_router_agreement({**arrays, "route_by_stage": routes}, record)
    assert out["disagreements"] == 1 and out["errors"]


def test_logistic_router_accepts_only_float32_near_ties():
    # Two candidates whose decisions tie exactly: the recorded route may be
    # either (GPU summation order), so a disagreement there is a near-tie.
    scores = np.array([[0.5, 0.5], [0.0, 3.0]], dtype=np.float32)
    true_task = np.array([1, 1], dtype=np.int16)
    model0 = {"classes": [0], "feature_mean": [0.0], "feature_scale": [1.0],
              "coef": [[1.0]], "intercept": [0.0]}
    model1 = {"classes": [0, 1], "feature_mean": [0.0, 0.0], "feature_scale": [1.0, 1.0],
              "coef": [[1.0, 0.0], [0.0, 1.0]], "intercept": [0.0, 0.0]}
    record = {"kind": "score-logreg", "prefix_models": [model0, model1]}
    routes = np.array([[-1, 1], [-1, 1]], dtype=np.int16)      # row 0: tie, recorded 1
    out = rp.stored_router_agreement({"true_task": true_task, "scores": scores,
                                      "route_by_stage": routes}, record)
    assert out["disagreements"] == 1 and out["near_tie_disagreements"] == 1
    assert out["errors"] == []
    routes = np.array([[-1, 0], [-1, 0]], dtype=np.int16)      # row 1: clear winner is 1
    out = rp.stored_router_agreement({"true_task": true_task, "scores": scores,
                                      "route_by_stage": routes}, record)
    assert out["disagreements"] == 1 and out["near_tie_disagreements"] == 0
    assert out["errors"]


def test_near_tie_tolerance_follows_the_float32_error_bound():
    # One candidate pair whose exact gap is far below the float32 bound of a
    # 2-term product (a rounding flip), and one whose gap is far above it.
    model0 = {"classes": [0], "feature_mean": [0.0], "feature_scale": [1.0],
              "coef": [[1.0]], "intercept": [0.0]}
    model1 = {"classes": [0, 1], "feature_mean": [0.0, 0.0], "feature_scale": [1.0, 1.0],
              "coef": [[1.0, 1.0], [1.0, 0.0]], "intercept": [0.0, 0.0]}
    record = {"kind": "score-logreg", "prefix_models": [model0, model1]}
    true_task = np.array([1, 1], dtype=np.int16)
    # Row 0: d0 = 1000 + 1e-5, d1 = 1000; float32 cannot resolve 1e-5 at 1000.
    # Row 1: d0 = 2, d1 = 1; a gap of 1 is never rounding.
    scores = np.array([[1000.0, 1e-5], [1.0, 1.0]], dtype=np.float32)
    routes = np.array([[-1, 1], [-1, 1]], dtype=np.int16)
    out = rp.stored_router_agreement({"true_task": true_task, "scores": scores,
                                      "route_by_stage": routes}, record)
    assert out["disagreements"] == 2 and out["near_tie_disagreements"] == 1
    assert out["errors"]


def test_tf32_jobs_are_checked_against_the_tf32_rounding_bound():
    # Stage 1 of a two-task bias router, d_j = -z_j. Row 0: an exact gap of
    # 1e-4, beyond float32 rounding but within TF32 rounding of the factors;
    # row 1: a gap of 3, never rounding. Both rows recorded the losing task.
    model0 = {"classes": [0], "feature_mean": [0.0], "feature_scale": [1.0],
              "coef": [[-1.0]], "intercept": [0.0]}
    model1 = {"classes": [0, 1], "feature_mean": [0.0, 0.0], "feature_scale": [1.0, 1.0],
              "coef": [[-1.0, 0.0], [0.0, -1.0]], "intercept": [0.0, 0.0]}
    arrays = {"true_task": np.array([1, 1], dtype=np.int16),
              "scores": np.array([[1.0, 1.0001], [0.0, 3.0]], dtype=np.float32),
              "route_by_stage": np.array([[-1, 1], [-1, 0]], dtype=np.int16),
              "tie_by_stage": np.zeros((2, 2), bool)}
    for kind in ("score-bias", "score-logreg"):
        record = {"kind": kind, "prefix_models": [model0, model1]}
        out = rp.stored_router_agreement(arrays, record)
        assert out["disagreements"] == 1 and out["errors"], kind     # float32: not a near-tie
        out = rp.stored_router_agreement(arrays, record, tf32=True)
        assert out["near_tie_disagreements"] == 1 and out["errors"] == [], kind
        flipped = {**arrays, "route_by_stage": np.array([[-1, 0], [-1, 1]], dtype=np.int16)}
        out = rp.stored_router_agreement(flipped, record, tf32=True)
        assert out["near_tie_disagreements"] == 0 and out["errors"], kind
    assert rp.matmul_tf32({"execution": {"effective": {"cuda_matmul_allow_tf32": True}}})
    assert not rp.matmul_tf32({"execution": {"effective": None}})


def test_calibrated_recordings_need_the_raw_pass_and_the_fitting_inputs():
    T, C, per_task = 4, 3, (7, 5, 9, 6)
    arrays = _synthetic(T=T, C=C, per_task=per_task)
    full = dict(arrays)
    full.update({f"raw_{k}": v.copy() for k, v in arrays.items()})
    rng = np.random.default_rng(0)
    full["calibration_features"] = rng.normal(size=(12, T)).astype(np.float32)
    full["calibration_task_ids"] = np.repeat(np.arange(T), 3).astype(np.int16)
    meta = {"num_tasks": T, "classes_per_task": C, "calibration": "score-logreg"}
    assert rp.structural_errors(full, meta, list(per_task)) == []
    assert any("raw_: arrays missing" in e
               for e in rp.structural_errors(arrays | {k: full[k] for k in
                                                        ("calibration_features", "calibration_task_ids")},
                                             meta, list(per_task)))
    no_inputs = {k: v for k, v in full.items() if not k.startswith("calibration_")}
    assert any("calibration inputs missing" in e
               for e in rp.structural_errors(no_inputs, meta, list(per_task)))
    bad = dict(full, calibration_features=full["calibration_features"][:, :2])
    assert any("calibration inputs have shapes" in e
               for e in rp.structural_errors(bad, meta, list(per_task)))
    bad = dict(full, calibration_task_ids=np.full(12, T, np.int16))
    assert any("task id outside" in e for e in rp.structural_errors(bad, meta, list(per_task)))


def test_runs_without_recorded_predictions_are_reported_not_failed(tmp_path):
    run = tmp_path / "run"
    (run / "statuses").mkdir(parents=True)
    job = {"job_id": "g__cifar__seed0__wsn", "benchmark": "cifar", "seed": 0, "methods": "wsn"}
    (run / "plan.json").write_text(json.dumps({"jobs": [{"job": job}]}))
    status = run / "statuses" / "g__cifar__seed0__wsn.json"
    status.write_text(json.dumps({"command": ["python", "run_class_il.py"]}))
    report = tmp_path / "report.json"
    assert rp.main(["--run-dir", str(run), "--out", str(report)]) == 0       # v3.0.0-v3.0.4 runs
    assert json.loads(report.read_text())["status"] == "not-recorded"
    # A job launched to record predictions whose files are missing is an error.
    status.write_text(json.dumps({"command": ["python", "run_class_il.py",
                                              "--predictions-dir", "x"]}))
    assert rp.main(["--run-dir", str(run), "--out", str(report)]) == 1
    assert json.loads(report.read_text())["status"] == "failed"
