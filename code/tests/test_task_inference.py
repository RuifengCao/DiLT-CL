"""Algorithm tests for score-based task inference (:mod:`diltcl.task_inference`).

The torch evaluation loop cannot run here, so we validate the *logic* with a
self-contained NumPy mirror of the risky pieces: score selection and exact-tie
handling, global-label mapping (disjoint vs shared protocols), and class-IL /
oracle / task-inference accounting. The torch implementation mirrors these
element-for-element; it is additionally checked by static compilation and
review.
"""

import numpy as np


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _entropy(logits):                      # logits [..., C] -> [...]
    p = _softmax(logits)
    return -(p * np.log(p + 1e-12)).sum(-1)


def _infer_task(per_task_logits, criterion="entropy", return_ties=False):
    # per_task_logits: list of T arrays [B, C] -> t_hat [B]
    if criterion == "entropy":
        scores = np.stack([_entropy(l) for l in per_task_logits], 0)       # [T,B]
    elif criterion == "confidence":  # highest max-prob == lowest (-maxprob)
        scores = np.stack([-_softmax(l).max(-1) for l in per_task_logits], 0)
    elif criterion == "margin":
        rows = []
        for logits in per_task_logits:
            z = logits - logits.mean(-1, keepdims=True)
            z = z / np.maximum(np.sqrt(np.mean(z * z, axis=-1, keepdims=True)),
                               1e-12)
            if z.shape[-1] < 2:
                rows.append(np.zeros(z.shape[0]))
            else:
                top2 = np.sort(z, axis=-1)[:, -2:]
                rows.append(-(top2[:, 1] - top2[:, 0]))
        scores = np.stack(rows, 0)
    else:
        raise ValueError(criterion)
    winning = scores == scores.min(axis=0, keepdims=True)
    # Candidate rows are task-ID ordered: argmax over the Boolean winners is an
    # explicit lowest-task-ID policy, matching the production implementation.
    task_ids = winning.argmax(axis=0)
    ties = winning.sum(axis=0) > 1
    return (task_ids, ties) if return_ties else task_ids


def _predict_global(per_task_logits, t_hat, offset):
    stacked = np.stack(per_task_logits, 0)                 # [T,B,C]
    B = t_hat.shape[0]
    chosen = stacked[t_hat, np.arange(B)]                  # [B,C]
    return t_hat * offset + chosen.argmax(-1)


def test_entropy_prefers_confident_task():
    # task 0 confident (peaky), task 1 flat -> infer task 0
    confident = np.array([[5.0, 0.0, 0.0]])
    flat = np.array([[0.1, 0.0, -0.1]])
    t = _infer_task([confident, flat], "entropy")
    assert t[0] == 0
    # swap: confident is task 1 now
    t = _infer_task([flat, confident], "entropy")
    assert t[0] == 1


def test_confidence_criterion_matches_intuition():
    a = np.array([[2.0, 1.0, 0.0]])     # max prob ~0.66
    b = np.array([[5.0, 0.0, 0.0]])     # max prob ~0.99 (more confident)
    assert _infer_task([a, b], "confidence")[0] == 1


def test_exact_tie_uses_lowest_task_id_and_sets_tie_mask():
    flat = np.zeros((2, 3))
    task_ids, ties = _infer_task([flat, flat, flat], return_ties=True)
    assert task_ids.tolist() == [0, 0]
    assert ties.tolist() == [True, True]


def test_margin_is_affine_scale_invariant():
    base = np.array([[3.0, 1.0, -2.0], [0.1, 0.2, 0.4]])
    other = np.array([[0.5, 0.4, 0.3], [3.0, 0.0, -1.0]])
    before = _infer_task([base, other], "margin")
    after = _infer_task([17.0 * base + 9.0, 0.2 * other - 4.0], "margin")
    assert np.array_equal(before, after), (before, after)


def test_global_mapping_disjoint():
    # offset = C = 3; task 2 chosen, local argmax = 1 -> global = 2*3+1 = 7
    T, B, C = 4, 1, 3
    logits = [np.zeros((B, C)) for _ in range(T)]
    logits[2][0] = [0.0, 9.0, 0.0]      # local argmax 1 for task 2
    # force inference to pick task 2 by making it the only confident one
    t_hat = _infer_task(logits, "entropy")
    g = _predict_global(logits, t_hat, offset=C)
    assert t_hat[0] == 2 and g[0] == 7


def test_global_mapping_shared_labels():
    # domain-IL: offset = 0 -> global prediction is just the local argmax
    T, B, C = 3, 1, 10
    logits = [np.zeros((B, C)) for _ in range(T)]
    logits[1][0, 7] = 9.0               # task 1 confident, predicts digit 7
    t_hat = _infer_task(logits, "entropy")
    g = _predict_global(logits, t_hat, offset=0)
    assert g[0] == 7


def _evaluate(forward_t, per_task_truth, num_tasks, C, disjoint):
    """NumPy mirror of evaluate_class_il_inference for a list of samples.

    per_task_truth: list over true tasks; each is a list of (logits_per_task,
    local_label) samples, where logits_per_task is [T, C].
    """
    offset = C if disjoint else 0
    counts = np.zeros(num_tasks, dtype=int)
    oracle_hits = np.zeros(num_tasks, dtype=int)
    cls_hits = np.zeros((num_tasks, num_tasks), dtype=int)
    ta_hits = np.zeros((num_tasks, num_tasks), dtype=int)
    tie_hits = np.zeros((num_tasks, num_tasks), dtype=int)
    correct_route_class_hits = np.zeros((num_tasks, num_tasks), dtype=int)
    for true_t, samples in enumerate(per_task_truth):
        for per_task, y in samples:
            pl = [per_task[t][None, :] for t in range(num_tasks)]   # [1,C] each
            counts[true_t] += 1
            oracle_hits[true_t] += int(per_task[true_t].argmax() == y)
            g_true = true_t * offset + y
            for stage in range(true_t, num_tasks):
                t_hat_batch, tied = _infer_task(
                    pl[:stage + 1], return_ties=True)
                t_hat = t_hat_batch[0]
                g_pred = _predict_global(
                    pl[:stage + 1], np.array([t_hat]), offset)[0]
                cls_hits[stage, true_t] += int(g_pred == g_true)
                ta_hits[stage, true_t] += int(t_hat == true_t)
                tie_hits[stage, true_t] += int(tied[0])
                correct_route_class_hits[stage, true_t] += int(
                    g_pred == g_true and t_hat == true_t)

    result = np.zeros((num_tasks, num_tasks), dtype=float)
    task_inference = np.zeros_like(result)
    routing_ties = np.zeros_like(result)
    for true_t in range(num_tasks):
        if not counts[true_t]:
            continue
        for stage in range(true_t, num_tasks):
            result[stage, true_t] = 100 * cls_hits[stage, true_t] / counts[true_t]
            task_inference[stage, true_t] = (
                100 * ta_hits[stage, true_t] / counts[true_t])
            routing_ties[stage, true_t] = (
                100 * tie_hits[stage, true_t] / counts[true_t])

    nonempty = counts > 0
    macro = lambda row: float(np.mean(row[nonempty])) if nonempty.any() else 0.0
    total = counts.sum()
    final_route_hits = ta_hits[-1]
    pooled_conditional = (
        100 * correct_route_class_hits[-1].sum() / final_route_hits.sum()
        if final_route_hits.sum() else 0.0)
    return dict(
        class_il=macro(result[-1]),
        micro_class_il=(100 * cls_hits[-1].sum() / total if total else 0.0),
        oracle=macro(np.divide(100 * oracle_hits, counts,
                               out=np.zeros(num_tasks), where=nonempty)),
        ta=macro(task_inference[-1]),
        micro_ta=(100 * final_route_hits.sum() / total if total else 0.0),
        conditional=pooled_conditional,
        tie_rate=macro(routing_ties[-1]),
        micro_tie_rate=(100 * tie_hits[-1].sum() / total if total else 0.0),
        tie_matrix=routing_ties,
        zero_correct_route_task_count=int(
            np.sum(nonempty & (final_route_hits == 0))),
    )


def test_accounting_perfect_inference_recovers_taskil():
    # If every task's own subnetwork is confidently correct and others are flat,
    # inference is perfect and class-IL == task-IL == 100%, TA == 100%.
    T, C = 3, 4
    samples = []
    for true_t in range(T):
        s = []
        for y in range(C):
            per_task = np.zeros((T, C))
            for t in range(T):
                per_task[t] = np.full(C, 0.01)              # flat
            per_task[true_t] = np.full(C, -5.0)
            per_task[true_t][y] = 9.0                        # confident & correct
            s.append((per_task, y))
        samples.append(s)
    r = _evaluate(None, samples, T, C, disjoint=True)
    assert r["class_il"] == 100 and r["oracle"] == 100 and r["ta"] == 100


def test_accounting_wrong_inference_costs_classil_not_oracle():
    # Two tasks; for a task-0 sample, task 1 is MORE confident (wrong) but the
    # oracle (task 0) is still correct -> oracle 100%, class-IL 0%, TA 0%.
    C = 4
    per_task = np.zeros((2, C))
    per_task[0] = [9.0, 0, 0, 0]      # task0 predicts class0 (correct, y=0)
    per_task[1] = [0, 0, 0, 20.0]     # task1 even more confident (wrong task)
    samples = [[(per_task, 0)]]       # true task 0, label 0
    r = _evaluate(None, samples, 2, C, disjoint=True)
    assert r["oracle"] == 100.0
    assert r["class_il"] == 0.0 and r["ta"] == 0.0


def test_prefix_tie_matrix_and_final_macro_micro_rates():
    # At stage 0 there is only one route, hence no tie. Once task 1 is added,
    # identical score signatures tie for both true tasks and policy selects 0.
    C = 2
    per_task = np.zeros((2, C))
    samples = [[(per_task.copy(), 0)], [(per_task.copy(), 0)]]
    r = _evaluate(None, samples, 2, C, disjoint=True)
    assert np.array_equal(r["tie_matrix"], np.array([[0.0, 0.0],
                                                       [100.0, 100.0]]))
    assert r["tie_rate"] == 100.0
    assert r["micro_tie_rate"] == 100.0


def test_zero_correct_routes_has_zero_pooled_conditional_not_selected_macro():
    # Only true task 1 has samples. An exact tie always selects task 0, so TA is
    # zero and the conditional denominator is empty. The reported value is the
    # conservative, finite pooled convention (0), not a mean over selected tasks.
    C = 3
    per_task = np.zeros((2, C))
    r = _evaluate(None, [[], [(per_task, 0)]], 2, C, disjoint=True)
    assert r["ta"] == 0.0 and r["micro_ta"] == 0.0
    assert r["class_il"] == 0.0 and r["micro_class_il"] == 0.0
    assert r["conditional"] == 0.0
    assert r["zero_correct_route_task_count"] == 1


def test_new_candidate_can_create_routing_forgetting():
    """A frozen old subnetwork can lose class-IL accuracy to a new route."""
    # Four task-0 samples. At stage 0 the only route is correct; at stage 1 the
    # new candidate steals three. This is why a column-constant matrix is wrong.
    stage0_routes = np.array([0, 0, 0, 0])
    stage1_routes = np.array([1, 1, 1, 0])
    diag = 100.0 * np.mean(stage0_routes == 0)
    final = 100.0 * np.mean(stage1_routes == 0)
    assert diag == 100.0 and final == 25.0
    assert final - diag == -75.0


def test_macro_and_micro_are_distinct_for_unequal_tasks():
    per_task_acc = np.array([100.0, 0.0])
    counts = np.array([1, 9])
    macro = per_task_acc.mean()
    micro = np.average(per_task_acc, weights=counts)
    assert macro == 50.0 and micro == 10.0


def test_disjoint_accuracy_decomposes_via_route_conditioning():
    # Four samples: three correct routes, of which two classify correctly. With
    # disjoint label blocks, micro ACC = micro TA * pooled ConditionalACC / 100.
    C = 3
    t0_correct = np.array([[9.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    t0_wrong_class = np.array([[0.0, 9.0, 0.0], [0.0, 0.0, 0.0]])
    t1_correct = np.array([[0.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
    t1_wrong_route = np.array([[20.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    samples = [
        [(t0_correct, 0), (t0_wrong_class, 0)],
        [(t1_correct, 1), (t1_wrong_route, 1)],
    ]
    r = _evaluate(None, samples, 2, C, disjoint=True)
    assert abs(r["micro_class_il"] - 50.0) < 1e-12
    assert abs(r["micro_ta"] - 75.0) < 1e-12
    assert abs(r["conditional"] - (200.0 / 3.0)) < 1e-12
    assert abs(r["micro_class_il"]
               - r["micro_ta"] * r["conditional"] / 100.0) < 1e-12


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} task-inference tests passed.")
    sys.exit(0)
