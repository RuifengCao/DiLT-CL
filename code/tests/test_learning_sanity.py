"""Every method must *learn* -- a guard against silent non-learning defects.

The edge-popup straight-through estimator was wrong for over a year and every
mechanical test passed: masks had the right density, tasks were isolated,
BWT was exactly zero -- and SupSup sat at chance level on every benchmark
until the full P0 table made it obvious. This test closes that gap: each
trainer runs on a small, clearly separable two-task problem and must reach a
task-IL accuracy far above chance. It does not check *how good* a method is,
only that its learning signal reaches the parameters it claims to train.

Runs on CPU in well under a minute.
"""

import contextlib
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - torch-free environment
    print("PyTorch not installed; learning-sanity tests skipped.")
    sys.exit(0)

from diltcl import MultiHeadModel  # noqa: E402
from diltcl.baselines import (  # noqa: E402
    train_derpp,
    train_er,
    train_finetune,
    train_independent,
)
from diltcl.masked_layers import set_task  # noqa: E402
from diltcl.metrics import evaluate, evaluate_with_mask  # noqa: E402
from diltcl.subnetwork_baselines import (  # noqa: E402
    masking_eval,
    train_packnet,
    train_supsup,
    train_wsn,
)
from diltcl.training import train_diltcl, train_ewc, train_lwf  # noqa: E402
from diltcl.utils import set_seed  # noqa: E402

NUM_TASKS, CLASSES, DIM, N_TRAIN, N_TEST = 2, 4, 16, 512, 256
CHANCE = 100.0 / CLASSES          # accuracies are percentages
THRESHOLD = 75.0                  # far above chance, tolerant of LwF's toy-data dip


class _Mlp(nn.Module):
    """Wide enough that a frozen random backbone contains good subnetworks."""

    def __init__(self, num_classes=CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(DIM, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


def _loaders(seed=0):
    """Two tasks of four well-separated Gaussian clusters each."""
    g = torch.Generator().manual_seed(seed)
    train, test = [], []
    for task in range(NUM_TASKS):
        centres = torch.randn(CLASSES, DIM, generator=g) * 4.0
        for target, n in ((train, N_TRAIN), (test, N_TEST)):
            y = torch.randint(0, CLASSES, (n,), generator=g)
            x = centres[y] + torch.randn(n, DIM, generator=g)
            target.append(DataLoader(TensorDataset(x, y), batch_size=32,
                                     shuffle=target is train))
    return train, test


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _final_accuracy(result):
    matrix = np.asarray(result[0])
    return float(matrix[-1].mean())


def _multi_head():
    return MultiHeadModel(_Mlp(), NUM_TASKS, CLASSES)


def _run_all():
    train, test = _loaders()
    results = {}
    set_seed(0)
    results["finetune"] = _final_accuracy(_quiet(
        train_finetune, _multi_head, train, test, NUM_TASKS, device="cpu",
        epochs=3, lr=1e-2))
    set_seed(0)
    results["independent"] = _final_accuracy(_quiet(
        train_independent, _Mlp, train, test, NUM_TASKS, device="cpu",
        epochs=3, lr=1e-2))
    set_seed(0)
    results["ewc"] = _final_accuracy(_quiet(
        train_ewc, _multi_head(), train, test, NUM_TASKS, device="cpu",
        lambda_ewc=10.0, epochs=3, lr=1e-2))
    set_seed(0)
    results["lwf"] = _final_accuracy(_quiet(
        train_lwf, _multi_head(), train, test, NUM_TASKS, device="cpu",
        epochs=3, lr=1e-2))
    set_seed(0)
    results["er"] = _final_accuracy(_quiet(
        train_er, _multi_head, train, test, NUM_TASKS, device="cpu", epochs=3,
        lr=1e-2, buffer_size=200, replay_batch_size=32))
    set_seed(0)
    results["derpp"] = _final_accuracy(_quiet(
        train_derpp, _multi_head, train, test, NUM_TASKS, device="cpu",
        epochs=3, lr=1e-2, buffer_size=200, replay_batch_size=32))
    set_seed(0)
    results["packnet"] = _final_accuracy(_quiet(
        train_packnet, _Mlp, train, test, NUM_TASKS, device="cpu", epochs=3,
        retrain_epochs=2, lr=1e-2))
    set_seed(0)
    results["supsup"] = _final_accuracy(_quiet(
        train_supsup, _Mlp, train, test, NUM_TASKS, device="cpu", epochs=8,
        lr=1e-2, sparsity=0.5))
    set_seed(0)
    results["wsn"] = _final_accuracy(_quiet(
        train_wsn, _Mlp, train, test, NUM_TASKS, device="cpu", epochs=3,
        lr=1e-2))
    set_seed(0)
    results["diltcl"] = _final_accuracy(_quiet(
        train_diltcl, _Mlp(), train, test, NUM_TASKS, device="cpu",
        num_partitions=2, pruning_rounds=2, target_sparsity=0.5,
        full_train_epochs=3, imp_epochs=2, expert_finetune_epochs=3, lr=1e-2))
    return results


def test_every_method_learns_a_separable_task_sequence():
    results = _run_all()
    assert all(0.0 <= acc <= 100.0 for acc in results.values())
    failures = {m: round(acc, 1) for m, acc in results.items() if acc < THRESHOLD}
    assert not failures, (
        f"methods below {THRESHOLD:.0f}% task-IL accuracy on a separable toy "
        f"problem (chance {CHANCE:.0f}%): {failures}; all: "
        f"{ {m: round(a, 1) for m, a in results.items()} }")


def test_constructed_zero_forgetting_matrices_match_a_real_re_evaluation():
    """Isolation methods *construct* their task-IL matrix (column-constant,
    BWT = 0) on the claim that a finished task's predictor never changes.
    The claim is checked here against the real trainers: after the whole
    sequence, every earlier task is re-evaluated with the returned model and
    must reproduce the constructed final row exactly. This would catch, for
    example, an optimizer that keeps moving 'frozen' weights through stale
    momentum, or a mask applied at training time but not at evaluation."""
    train, test = _loaders()
    set_seed(0)
    matrix, _, _, state = _quiet(train_packnet, _Mlp, train, test, NUM_TASKS,
                                 device="cpu", epochs=3, retrain_epochs=2,
                                 lr=1e-2, return_state=True)
    for j in range(NUM_TASKS):
        again = masking_eval(state["model"], test[j], state["eval_masks"][j], "cpu")
        assert again == matrix[-1][j], ("packnet", j, again, matrix[-1][j])
    for name, trainer, kwargs in (
            ("supsup", train_supsup, dict(epochs=4, lr=1e-2, sparsity=0.5)),
            ("wsn", train_wsn, dict(epochs=3, lr=1e-2))):
        set_seed(0)
        matrix, _, _, state = _quiet(trainer, _Mlp, train, test, NUM_TASKS,
                                     device="cpu", return_state=True, **kwargs)
        for j in range(NUM_TASKS):
            set_task(state["model"], j)
            again = evaluate(state["model"], test[j], "cpu")
            assert again == matrix[-1][j], (name, j, again, matrix[-1][j])
    set_seed(0)
    matrix, _, _, learner = _quiet(
        train_diltcl, _Mlp(), train, test, NUM_TASKS, device="cpu",
        num_partitions=2, pruning_rounds=2, target_sparsity=0.5,
        full_train_epochs=3, imp_epochs=2, expert_finetune_epochs=3, lr=1e-2,
        return_model=True)
    for j in range(NUM_TASKS):
        again = evaluate_with_mask(learner, test[j], j, "cpu")
        assert again == matrix[-1][j], ("diltcl", j, again, matrix[-1][j])


def test_guard_is_sensitive_to_the_old_edge_popup_estimator():
    """The guard must actually catch the defect it was written for: with the
    pre-3.0.2 estimator (``abs()`` taken inside the straight-through
    function, sign dropped) SupSup falls to near-chance on the same toy
    problem the fixed code solves."""
    from torch.autograd import Function
    from diltcl import masked_layers as ml

    class _OldGetSubnet(Function):
        @staticmethod
        def forward(ctx, scores, sparsity):
            keep = max(1, int(round((1.0 - float(sparsity)) * scores.numel())))
            flat = scores.detach().abs().flatten()
            _, idx = flat.topk(keep)
            mask = torch.zeros_like(flat)
            mask[idx] = 1.0
            return mask.view_as(scores)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    train, test = _loaders()
    original = ml.subnet_mask
    ml.subnet_mask = lambda scores, sparsity: _OldGetSubnet.apply(scores, sparsity)
    try:
        set_seed(0)
        old = _final_accuracy(_quiet(
            train_supsup, _Mlp, train, test, NUM_TASKS, device="cpu", epochs=8,
            lr=1e-3, sparsity=0.5))
    finally:
        ml.subnet_mask = original
    set_seed(0)
    fixed = _final_accuracy(_quiet(
        train_supsup, _Mlp, train, test, NUM_TASKS, device="cpu", epochs=8,
        lr=1e-3, sparsity=0.5))
    assert fixed >= THRESHOLD, fixed
    assert old < THRESHOLD, f"old estimator unexpectedly learned: {old:.1f}%"


if __name__ == "__main__":
    for method, acc in _run_all().items():
        print(f"{method:12s} {acc:.3f}")
    test_every_method_learns_a_separable_task_sequence()
    print("PASS test_every_method_learns_a_separable_task_sequence")
