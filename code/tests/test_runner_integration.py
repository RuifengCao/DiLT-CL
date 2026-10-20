"""End-to-end integration tests for both runners.

Nothing previously executed ``run()`` all the way through: the runners need
real datasets, so the whole validate -> build -> execute -> aggregate ->
serialize pipeline was untested. That is how a refactor managed to leave a
self-recursive validator and four ``undefined name`` references in
``run_class_il.py`` while the suite still reported all green.

These tests substitute tiny synthetic benchmarks for the dataset setup and run
the real pipeline to completion, then assert on the emitted JSON/NPZ. They are
the behavioural baseline that any future refactor of ``run()`` must preserve.

Skipped automatically when PyTorch is unavailable.
"""

import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - torch-free environment
    print("PyTorch not installed; runner integration tests skipped.")
    sys.exit(0)

from diltcl.config import ExperimentConfig  # noqa: E402

import run_task_il as taskil  # noqa: E402
import run_class_il as class_il  # noqa: E402

NUM_TASKS = 3
CLASSES_PER_TASK = 2
FEATURES = 12


class _TinyNet(nn.Module):
    """Smallest model that still exercises masking and multi-head wrapping."""

    def __init__(self, num_classes=CLASSES_PER_TASK):
        super().__init__()
        self.fc1 = nn.Linear(FEATURES, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x.view(x.size(0), -1))))


def _tiny_loaders(num_classes, seed=0):
    """Deterministic synthetic task stream."""
    generator = torch.Generator().manual_seed(seed)
    train, test = [], []
    for task in range(NUM_TASKS):
        x = torch.randn(48, FEATURES, generator=generator)
        y = torch.randint(0, num_classes, (48,), generator=generator)
        train.append(DataLoader(TensorDataset(x, y), batch_size=16))
        test.append(DataLoader(TensorDataset(x[:24], y[:24]), batch_size=16))
    return train, test


def _fast_config():
    cfg = ExperimentConfig.fast()
    cfg.epochs_ewc = cfg.epochs_lwf = cfg.epochs_replay = 1
    cfg.diltcl_full_epochs = 1
    cfg.diltcl_imp_epochs = 1
    cfg.diltcl_expert_finetune_epochs = 1
    cfg.pruning_rounds = 2
    cfg.c100_num_tasks = NUM_TASKS
    cfg.c100_classes_per_task = CLASSES_PER_TASK
    cfg.buffer_size = 20
    cfg.replay_batch_size = 8
    return cfg


class _Patched:
    """Swap dataset setup for synthetic loaders, and skip the preflight gate."""

    def __init__(self, module, names):
        self.module = module
        self.names = names
        self.saved = {}

    def __enter__(self):
        for name, replacement in self.names.items():
            self.saved[name] = getattr(self.module, name, None)
            setattr(self.module, name, replacement)
        return self

    def __exit__(self, *exc):
        for name, original in self.saved.items():
            setattr(self.module, name, original)
        return False


class _NoPreflight:
    """Stand-in preflight whose gate always passes."""

    @staticmethod
    def enforce(*args, **kwargs):
        return {"ok": True, "blocking": [], "warnings": []}

    @staticmethod
    def resolve_data_root(explicit=None):
        return "./data"


def _run_task_il(out_prefix, head_mode="shared", methods=("finetune", "diltcl")):
    def fake_setup(benchmark, cfg, seed, head_mode="shared", **_):
        train, test = _tiny_loaders(CLASSES_PER_TASK, seed)
        base = lambda: _TinyNet(CLASSES_PER_TASK)          # noqa: E731
        builders = (base, taskil._wrap_head_mode(
            base, head_mode, NUM_TASKS, CLASSES_PER_TASK))
        return train, test, builders, NUM_TASKS

    with _Patched(taskil, {"_benchmark_setup": fake_setup,
                         "preflight": _NoPreflight}):
        taskil.run(_fast_config(), ["cifar"], list(methods), [0],
                 out_prefix=out_prefix, profile=False, head_mode=head_mode)
    with open(f"{out_prefix}.json") as handle:
        return json.load(handle), np.load(f"{out_prefix}.npz")


def _run_class_il(out_prefix, methods=("finetune", "diltcl")):
    def fake_shared(benchmark, cfg, seed, **_):
        train, test = _tiny_loaders(NUM_TASKS * CLASSES_PER_TASK, seed)
        return class_il.SharedSetup(
            train, test, (lambda: _TinyNet(NUM_TASKS * CLASSES_PER_TASK)),
            NUM_TASKS)

    def fake_isolation(benchmark, cfg, seed, **_):
        train, test = _tiny_loaders(CLASSES_PER_TASK, seed)
        return class_il.IsolationSetup(
            train, test, (lambda: _TinyNet(CLASSES_PER_TASK)),
            NUM_TASKS, CLASSES_PER_TASK, True)

    with _Patched(class_il, {"_shared_setup": fake_shared,
                             "_isolation_setup": fake_isolation,
                             "preflight": _NoPreflight}):
        class_il.run(_fast_config(), ["cifar"], list(methods), [0],
                     out_prefix=out_prefix)
    with open(f"{out_prefix}.json") as handle:
        return json.load(handle), np.load(f"{out_prefix}.npz")


# ---------------------------------------------------------------------------
# Task-IL pipeline
# ---------------------------------------------------------------------------

def test_task_il_pipeline_produces_a_complete_artifact():
    with tempfile.TemporaryDirectory() as directory:
        payload, matrices = _run_task_il(os.path.join(directory, "result"))
        assert payload["benchmarks"] == ["cifar"]
        assert payload["methods"] == ["finetune", "diltcl"]
        assert payload["seeds"] == [0]
        for method in payload["methods"]:
            metrics = payload["per_seed"]["cifar"][method][0]
            for key in ("ACC", "BWT", "Forgetting", "LA"):
                assert np.isfinite(metrics[key]), (method, key)
            assert f"cifar/{method}/seed0" in matrices.files


def test_task_il_metrics_recompute_from_the_saved_matrix():
    """The serialized matrix must reproduce the serialized metrics."""
    with tempfile.TemporaryDirectory() as directory:
        payload, matrices = _run_task_il(os.path.join(directory, "result"))
        for method in payload["methods"]:
            matrix = matrices[f"cifar/{method}/seed0"]
            reported = payload["per_seed"]["cifar"][method][0]
            final_row_mean = float(np.mean(matrix[-1]))
            assert abs(final_row_mean - reported["ACC"]) < 1e-9, method


def test_task_il_head_mode_is_recorded():
    with tempfile.TemporaryDirectory() as directory:
        payload, _ = _run_task_il(os.path.join(directory, "result"),
                                  head_mode="multi", methods=("finetune",))
        assert payload["task_il_head_mode"] == "multi"
        disclosure = payload["task_il_head_disclosure"]
        assert "separate head per task" in disclosure["shared_backbone_baselines"]


def test_task_il_is_deterministic_for_a_fixed_seed():
    with tempfile.TemporaryDirectory() as directory:
        first, first_m = _run_task_il(os.path.join(directory, "a"))
        second, second_m = _run_task_il(os.path.join(directory, "b"))
        assert (first["per_seed"] == second["per_seed"])
        for key in first_m.files:
            assert np.array_equal(first_m[key], second_m[key]), key


# ---------------------------------------------------------------------------
# Class-IL pipeline
# ---------------------------------------------------------------------------

def test_class_il_pipeline_produces_a_complete_artifact():
    with tempfile.TemporaryDirectory() as directory:
        payload, matrices = _run_class_il(os.path.join(directory, "result"))
        assert payload["protocol"] == "class_il"
        assert "finetune" in payload["shared_backbone_methods"]
        assert "diltcl" in payload["isolation_methods"]
        for method in payload["methods"]:
            metrics = payload["per_seed"]["cifar"][method][0]
            for key in ("ACC", "BWT", "Forgetting", "LA"):
                assert np.isfinite(metrics[key]), (method, key)
        # isolation methods carry the routing diagnostics
        isolation = payload["per_seed"]["cifar"]["diltcl"][0]
        for key in ("TA", "OracleACC", "ConditionalACC"):
            assert key in isolation, key


def test_class_il_isolation_matrices_are_lower_triangular():
    with tempfile.TemporaryDirectory() as directory:
        _, matrices = _run_class_il(os.path.join(directory, "result"))
        matrix = matrices["cifar/diltcl/seed0"]
        assert np.allclose(np.triu(matrix, 1), 0)
        for suffix in ("task_inference", "conditional_class", "routing_tie"):
            assert f"cifar/diltcl/seed0/{suffix}" in matrices.files, suffix


def test_class_il_acc_never_exceeds_task_inference_accuracy():
    """Disjoint classes: a wrong route cannot yield a correct global label."""
    with tempfile.TemporaryDirectory() as directory:
        payload, _ = _run_class_il(os.path.join(directory, "result"))
        metrics = payload["per_seed"]["cifar"]["diltcl"][0]
        assert metrics["ACC"] <= metrics["TA"] + 1e-9


def test_class_il_is_deterministic_for_a_fixed_seed():
    with tempfile.TemporaryDirectory() as directory:
        first, first_m = _run_class_il(os.path.join(directory, "a"))
        second, second_m = _run_class_il(os.path.join(directory, "b"))
        assert first["per_seed"] == second["per_seed"]
        for key in first_m.files:
            assert np.array_equal(first_m[key], second_m[key]), key


def test_artifacts_are_marked_unverified():
    """Fresh artifacts must never claim publication readiness."""
    with tempfile.TemporaryDirectory() as directory:
        payload, _ = _run_class_il(os.path.join(directory, "result"))
        assert payload["evidence_status"] == "unverified"
        assert payload["artifact_reportability"][
            "current_publication_reportable"] is False


if __name__ == "__main__":
    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_") and callable(value)]
    for function in functions:
        function()
        print(f"PASS {function.__name__}")
    print(f"\nAll {len(functions)} runner integration tests passed.")
    sys.exit(0)
