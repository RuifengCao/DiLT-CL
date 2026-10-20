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
    for s in range(T):
        rows = true_task <= s
        cand = scores[rows, :s + 1]
        win = cand == cand.min(axis=1, keepdims=True)
        route[rows, s] = win.argmax(axis=1)
        tie[rows, s] = win.sum(axis=1) > 1
        # A correctly routed example mostly predicts its label.
        guess = rng.integers(0, C, rows.sum())
        keep = rng.random(rows.sum()) < 0.6
        pred[rows, s] = np.where(keep, label[rows], guess)
    oracle = np.where(rng.random(N) < 0.7, label, rng.integers(0, C, N)).astype(np.int16)
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


@pytest.mark.parametrize("calibration", ["none", "score-bias"])
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
        if calibration != "none":
            assert entries[0]["stored_router"]["route_agreement"] == 1.0
            assert entries[0]["stored_router"]["errors"] == []
            assert entries[0]["refit"]["route_agreement"] == 1.0
            # Fitting accuracies of every stored prefix router are reproduced.
            fit = entries[0]["fit_accuracy_check"]
            assert fit["stages_checked"] == len(record["prefix_models"]) and fit["errors"] == []
        # Tampering with the file is detected through its recorded hash.
        with path.open("ab") as handle:
            handle.write(b"\0")
        errors = []
        with np.load(f"{out}.npz", allow_pickle=False) as npz:
            verifier.validate_predictions(record, payload, npz, out.with_suffix(".json"),
                                          "cifar", "wsn", 0, "wsn", errors)
        assert any("SHA-256" in e for e in errors)


@pytest.mark.parametrize("calibration", ["none", "score-logreg"])
def test_checkpoints_reevaluate_bit_identically(calibration):
    """Released checkpoints alone regenerate every routed and oracle result."""
    evaluator = _load("evaluate_checkpoints_under_test", ROOT / "tools" / "evaluate_checkpoints.py")
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "job" / "result"
        out.parent.mkdir()
        _run_runner(str(out), str(out.parent / "predictions"),
                    ("packnet", "supsup", "wsn", "diltcl"), calibration=calibration,
                    checkpoint_dir=str(out.parent / "checkpoints"))
        entries = evaluator.evaluate_job(out.with_suffix(".json"), device="cpu",
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
        entries = evaluator.evaluate_job(out.with_suffix(".json"), device="cpu",
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
        entries = evaluator.evaluate_job(result_json, device="cpu", methods=("wsn",),
                                         setup_fn=_fake_isolation_setup())
        assert any("route_by_stage: 1 per-example" in e for e in entries[0]["errors"])
        tolerant = evaluator.evaluate_job(result_json, device="cpu", methods=("wsn",),
                                          atol=0.05, setup_fn=_fake_isolation_setup())
        assert tolerant[0]["errors"] == [] and not tolerant[0]["exact"]
        assert tolerant[0]["per_example_differences"] == {"route_by_stage": 1}
        # 2. A referenced predictions file that is missing is an error, not a skip.
        ppath.unlink()
        entries = evaluator.evaluate_job(result_json, device="cpu", methods=("wsn",),
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
            entries = evaluator.evaluate_job(result_json, device="cpu", methods=("wsn",),
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
