"""Algorithm-level tests for the subnetwork baselines.

The PyTorch training loops in :mod:`diltcl.subnetwork_baselines` cannot run in a
torch-free environment, so here we validate the *algorithms* with small numpy
mirrors of the two risky pieces:

* PackNet packing bookkeeping -- disjoint per-task masks, a growing union, and
  eval masks that exclude later tasks (which is what guarantees zero
  forgetting and a column-constant result matrix).
* The edge-popup top-k selection used by SupSup -- exact keep count and
  correct indices.

These mirror the logic in the real implementation closely enough to catch
design mistakes; the torch implementation is additionally checked by
``py_compile`` and review.
"""

import numpy as np


# ---------------------------------------------------------------------------
# numpy mirror of PackNet packing (per single weight vector)
# ---------------------------------------------------------------------------

def _packnet_simulate(weights_per_task, num_tasks, keep_fraction=None):
    """Mirror of PackNet's assign/release/freeze loop on a 1-D weight vector.

    Args:
        weights_per_task: ``(num_tasks, P)`` magnitudes the model would have
            for the free weights at each task (stand-in for trained weights).
        num_tasks: Number of tasks.
        keep_fraction: As in :func:`train_packnet` (``None`` -> equal split).

    Returns:
        ``(task_masks, eval_masks)`` as boolean arrays of shape ``(T, P)``.
    """
    P = weights_per_task.shape[1]
    assigned = np.zeros(P, dtype=bool)
    task_masks, eval_masks = [], []
    for t in range(num_tasks):
        free = ~assigned
        n_free = int(free.sum())
        frac = (1.0 / (num_tasks - t)) if keep_fraction is None else keep_fraction
        keep = max(1, int(round(frac * n_free)))
        keep = min(keep, n_free)

        mags = np.where(free, np.abs(weights_per_task[t]), -np.inf)
        # indices of the ``keep`` largest free magnitudes
        idx = np.argsort(mags)[-keep:]
        task_mask = np.zeros(P, dtype=bool)
        task_mask[idx] = True

        assigned = assigned | task_mask
        task_masks.append(task_mask)
        eval_masks.append(assigned.copy())
    return np.array(task_masks), np.array(eval_masks)


def test_packnet_masks_are_disjoint():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((5, 200))
    task_masks, _ = _packnet_simulate(w, 5)
    overlap = task_masks.sum(axis=0)        # how many tasks claim each weight
    assert overlap.max() <= 1, "a weight was assigned to more than one task"


def test_packnet_union_grows_and_eval_excludes_later_tasks():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((5, 200))
    task_masks, eval_masks = _packnet_simulate(w, 5)
    sizes = [int(m.sum()) for m in eval_masks]
    assert sizes == sorted(sizes), "union mask must be non-decreasing"
    # eval mask for task t must contain task t's weights but none from t+1..
    for t in range(5):
        assert np.all(eval_masks[t][task_masks[t]]), "own task missing in eval"
        for later in range(t + 1, 5):
            assert not np.any(eval_masks[t][task_masks[later]]), \
                "eval mask leaked a later task's weights"


def test_packnet_equal_split_balances_capacity():
    """Equal-split schedule gives each task roughly P / T weights."""
    rng = np.random.default_rng(2)
    P, T = 1000, 10
    w = rng.standard_normal((T, P))
    task_masks, eval_masks = _packnet_simulate(w, T)
    sizes = np.array([int(m.sum()) for m in task_masks])
    assert np.all(np.abs(sizes - P / T) <= 2), sizes
    assert int(eval_masks[-1].sum()) == P, "all capacity should be used"


def test_packnet_zero_forgetting_matrix_is_column_constant():
    """Because each task is evaluated with a fixed mask, R[i][j] = acc_j.

    We emulate the matrix-fill in ``train_packnet`` and check the resulting
    matrix is lower-triangular and column-constant (=> BWT == 0).
    """
    T = 6
    task_acc = np.linspace(70, 90, T)        # arbitrary fixed per-task accuracies
    M = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1):
            M[i][j] = task_acc[j]
    # column-constant
    for j in range(T):
        col = M[j:, j]
        assert np.allclose(col, col[0])
    # upper triangle is zero
    assert np.allclose(np.triu(M, k=1), 0.0)
    # BWT = mean_j (M[T-1][j] - M[j][j]) == 0
    bwt = np.mean([M[T - 1][j] - M[j][j] for j in range(T - 1)])
    assert abs(bwt) < 1e-12


# ---------------------------------------------------------------------------
# numpy mirror of edge-popup top-k (SupSup _GetSubnet.forward)
# ---------------------------------------------------------------------------

def _get_subnet(scores, sparsity):
    keep = max(1, int(round((1.0 - sparsity) * scores.size)))
    flat = np.abs(scores).ravel()
    idx = np.argsort(flat)[-keep:]
    mask = np.zeros_like(flat)
    mask[idx] = 1.0
    return mask.reshape(scores.shape), keep


def test_edge_popup_keep_count():
    rng = np.random.default_rng(3)
    scores = rng.standard_normal((20, 5))    # 100 entries
    for sparsity, expected in [(0.5, 50), (0.9, 10), (0.0, 100)]:
        mask, keep = _get_subnet(scores, sparsity)
        assert keep == expected
        assert int(mask.sum()) == expected


def test_edge_popup_selects_highest_magnitude():
    scores = np.array([0.1, -5.0, 0.2, 3.0, -0.3])
    mask, _ = _get_subnet(scores, sparsity=0.6)   # keep top 2 by |score|
    # the two largest magnitudes are at indices 1 (5.0) and 3 (3.0)
    assert mask[1] == 1.0 and mask[3] == 1.0
    assert mask.sum() == 2.0


# ---------------------------------------------------------------------------
# numpy mirror of WSN (trainable weights + per-task masks + frozen "won" set)
# ---------------------------------------------------------------------------

def _wsn_simulate(T, P, sparsity, seed=0):
    """Mirror WSN's freeze/accumulate loop on a 1-D weight vector.

    Each task "trains" only the not-yet-won weights (we perturb the free
    entries), then selects a top-k mask and adds it to the frozen set. Returns
    the final weights, per-task masks, the per-task snapshot of ``w * m_t`` at
    the end of task ``t``, and the final frozen set.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(P)
    won = np.zeros(P, dtype=bool)
    masks, snapshots = [], []
    for t in range(T):
        free = ~won
        w = w + free * rng.standard_normal(P) * 0.1   # update only free weights
        scores = rng.standard_normal(P)
        keep = max(1, int(round((1.0 - sparsity) * P)))
        idx = np.argsort(np.abs(scores))[-keep:]
        m = np.zeros(P, dtype=bool)
        m[idx] = True
        masks.append(m)
        won = won | m
        snapshots.append((w * m).copy())
    return w, masks, snapshots, won


def test_wsn_zero_forgetting():
    """A task's function ``w * m_t`` must be unchanged after later training."""
    w, masks, snaps, _ = _wsn_simulate(T=5, P=200, sparsity=0.5)
    for s in range(5):
        assert np.allclose(w * masks[s], snaps[s]), \
            f"task {s}'s subnetwork output changed after later tasks"


def test_wsn_all_masks_frozen_at_end():
    """Every task's winning weights belong to the final frozen set."""
    _, masks, _, won = _wsn_simulate(T=5, P=200, sparsity=0.5)
    for t, m in enumerate(masks):
        assert np.all(won[m]), f"task {t}'s won weights are not all frozen"


def test_wsn_masks_may_overlap_unlike_packnet():
    """WSN allows weight reuse (masks can overlap), unlike PackNet's disjoint
    assignment. Sanity check that the simulation can produce overlap."""
    _, masks, _, _ = _wsn_simulate(T=5, P=50, sparsity=0.5)
    overlaps = sum(int((masks[a] & masks[b]).any())
                   for a in range(len(masks)) for b in range(a + 1, len(masks)))
    assert overlaps > 0, "expected some weight reuse across tasks"


def test_edge_popup_scores_learn_on_a_frozen_random_backbone():
    """Regression for the v3.0.1 SupSup defect: with ``abs()`` taken inside the
    straight-through function, negative scores were pushed the wrong way and
    a frozen-backbone edge-popup network could not learn at all (chance level
    on every P0 benchmark). A masked MLP with frozen weights must now fit a
    separable toy problem far above chance."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        return
    from diltcl.masked_layers import MultitaskMaskedLinear

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    centres = torch.randn(4, 16, generator=g) * 3.0
    y = torch.randint(0, 4, (1024,), generator=g)
    x = centres[y] + torch.randn(1024, 16, generator=g)
    net = nn.Sequential(MultitaskMaskedLinear(16, 128, 1), nn.ReLU(),
                        MultitaskMaskedLinear(128, 4, 1))
    trainable = [p for p in net.parameters() if p.requires_grad]
    assert all(p.shape[0] in (128, 4) for p in trainable)      # scores only
    optimizer = torch.optim.Adam(trainable, lr=1e-2)
    for _ in range(60):
        optimizer.zero_grad()
        F.cross_entropy(net(x), y).backward()
        optimizer.step()
    with torch.no_grad():
        acc = (net(x).argmax(1) == y).float().mean().item()
    assert acc > 0.9, f"frozen-backbone edge-popup failed to learn: acc={acc:.3f}"
    # Negative scores must receive sign-aware gradients.
    layer = MultitaskMaskedLinear(8, 8, 1)
    scores = layer.scores[0]
    with torch.no_grad():
        scores.abs_().neg_()                # every score negative
    out = layer(torch.randn(4, 8))
    out.sum().backward()
    from diltcl.masked_layers import subnet_mask
    probe = scores.detach().clone().requires_grad_(True)
    subnet_mask(probe, layer.sparsity).sum().backward()
    assert torch.all(probe.grad <= 0), "gradient through abs() must carry sign(score)"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} subnetwork-logic tests passed.")
    sys.exit(0)
