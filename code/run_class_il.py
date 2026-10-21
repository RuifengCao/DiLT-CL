"""Class-incremental (class-IL) evaluation for *all* methods.

This is the class-IL counterpart of ``run_task_il.py``. It evaluates
two method families under the hardest protocol -- a model that must choose among
all classes seen so far, with **no task identity at test time**:

* **Shared-backbone baselines** (Finetune, EWC, LwF, ER, DER++): a single head
  spans all classes. The existing task-IL trainers already evaluate with an
  ``argmax`` over all outputs and apply no per-task masking, so feeding them a
  global label space (``get_split_cifar100(class_il=True)``) and a single
  ``K = num_tasks*classes_per_task`` head turns their result matrix into a
  class-IL matrix directly. ACC / BWT / Forgetting / LA follow as usual.

* **Parameter-isolation methods** (DiLT-CL, PackNet, SupSup, WSN): each task has
  its own subnetwork.  Class-IL prediction exhaustively evaluates the candidate
  subnetworks and routes by entropy, confidence, a scale-invariant margin, or
  an optional score-logistic calibrator fitted only on held-out training data.
  The runner reconstructs the genuine prefix-candidate result matrix: adding a
  route can hurt an old task even when all of its weights are frozen.

ACC is always the macro average over tasks; micro-pooled values are recorded
separately for unequal-size suites such as 5-datasets. ``LA`` has its standard
matrix-diagonal definition, while isolation methods additionally report
``OracleACC`` and task-inference accuracy ``TA``. Permuted-MNIST is domain-IL
(labels shared, inputs permuted).

Examples
--------
    python run_class_il.py --fast --seeds 2
    python run_class_il.py --benchmark cifar --seeds 5 --methods derpp,supsup,wsn,diltcl
"""

import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
import time

from typing import NamedTuple

import numpy as np
import torch

from diltcl import preflight
from diltcl import benchmarks as bench
from diltcl.data_backend import make_loader, subset_dataset
from diltcl.diltcl import IMP_PARTITION_BATCH_SIZE
from diltcl.execution import (
    ExecutionOptions,
    add_execution_arguments,
    execution_from_args,
)
from diltcl.runner_common import (  # shared with the other runner
    _cpu_tensor_dict,
    _cuda_sync,
    _load_config_overrides,
    _parse_seed_spec,
    _persist_checkpoint,
    _state_content_sha256,
    describe_task_loader,
)
from diltcl.task_calibration import CALIBRATION_KINDS
from diltcl import (
    ExperimentConfig,
    aggregate_seeds,
    collect_calibration_features,
    deterministic_holdout_indices,
    fit_prefix_calibrators,
    format_comparison_table,
    get_device,
    masking,
    set_seed,
    summarize_run,
    train_derpp,
    train_diltcl,
    train_er,
    train_ewc,
    train_finetune,
    train_lwf,
    train_packnet,
    train_supsup,
    train_wsn,
)
from diltcl.task_inference import (
    diltcl_forward_fn,
    evaluate_class_il_inference,
    packnet_forward_fn,
    supsup_forward_fn,
    wsn_forward_fn,
)

# Shared-backbone methods run class-IL directly; isolation methods run it via
# task inference. Both families share the class-IL ACC metric.
class SharedSetup(NamedTuple):
    """Everything needed to run a shared-backbone method on one benchmark.

    A plain tuple used to carry these five fields positionally, so call sites
    read ``shared[0]`` and rebuilt it with ``(new_train, *shared[1:])``. Adding
    or reordering a field silently shifted every consumer; naming them makes
    that impossible.
    """

    train_loaders: list
    test_loaders: list
    builder: object
    num_tasks: int


class IsolationSetup(NamedTuple):
    """Everything needed to run an isolation method on one benchmark."""

    train_loaders: list
    test_loaders: list
    builder: object
    num_tasks: int
    classes_per_task: int
    disjoint_classes: bool


SHARED_BACKBONE_METHODS = ["finetune", "ewc", "lwf", "er", "derpp"]
ISOLATION_METHODS = ["packnet", "supsup", "wsn", "diltcl"]
ALL_METHODS = SHARED_BACKBONE_METHODS + ISOLATION_METHODS
def _package_version():
    """Return the installed package version, so provenance cannot drift."""
    from diltcl import __version__
    return __version__


# Single source of truth: the package version. Hardcoding this separately let
# it drift -- result files produced by v2.2.0 still reported "2.1.1", which
# then propagated into table captions and made provenance untrustworthy.
CODE_VERSION = _package_version()
METRIC_SCHEMA_VERSION = 3


def _environment_metadata(device):
    """Return a JSON-safe execution-environment fingerprint."""
    cuda_available = bool(torch.cuda.is_available())
    cudnn_version = (torch.backends.cudnn.version()
                     if torch.backends.cudnn.is_available() else None)
    device_name = None
    if cuda_available and str(device).startswith("cuda"):
        try:
            device_name = torch.cuda.get_device_name(device)
        except (AssertionError, RuntimeError, ValueError):
            # Environment disclosure must never make an otherwise valid run
            # fail (for example, when CUDA visibility changes during startup).
            device_name = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "device": str(device),
        "device_name": device_name,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "command": sys.argv,
    }
EWC_FISHER_NUM_SAMPLES = 256   # default of EWC.compute_fisher(num_samples=...)


def _schedule_accounting(method, cfg, train_loaders):
    """Count declared training loops on the post-holdout model-training set."""
    counts = [int(len(loader.dataset)) for loader in train_loaders]
    batches = [int(len(loader)) for loader in train_loaders]
    if method == "diltcl":
        example_visits = (
            counts[0] * (cfg.diltcl_full_epochs
                         + cfg.pruning_rounds * cfg.diltcl_imp_epochs)
            + sum(counts[1:]) * (
                cfg.pruning_rounds * cfg.diltcl_imp_epochs
                + cfg.diltcl_expert_finetune_epochs)
        )
        part_size = counts[0] // cfg.diltcl_partitions
        partition_sizes = [part_size] * max(0, cfg.diltcl_partitions - 1)
        partition_sizes.append(
            counts[0] - part_size * max(0, cfg.diltcl_partitions - 1))
        partition_steps = sum(math.ceil(size / IMP_PARTITION_BATCH_SIZE)
                              for size in partition_sizes)
        steps = (
            batches[0] * cfg.diltcl_full_epochs
            + partition_steps * cfg.pruning_rounds * cfg.diltcl_imp_epochs
            + sum(batches[1:]) * (
                cfg.pruning_rounds * cfg.diltcl_imp_epochs
                + cfg.diltcl_expert_finetune_epochs)
        )
        schedule = {
            "phase1_full_epochs": cfg.diltcl_full_epochs,
            "imp_rounds": cfg.pruning_rounds,
            "imp_epochs_per_round": cfg.diltcl_imp_epochs,
            "partitions": cfg.diltcl_partitions,
            "expert_finetune_epochs": cfg.diltcl_expert_finetune_epochs,
        }
        step_note = "exact for the declared fixed-round training loops"
    else:
        if method == "ewc":
            epochs = cfg.epochs_ewc
        elif method == "lwf":
            epochs = cfg.epochs_lwf
        else:
            epochs = cfg.epochs_replay
        retrain = cfg.packnet_retrain_epochs if method == "packnet" else 0
        example_visits = sum(counts) * (epochs + retrain)
        steps = sum(batches) * (epochs + retrain)
        schedule = {"epochs_per_task": epochs,
                    "post_prune_retrain_epochs_per_task": retrain}
        if method == "ewc":
            # EWC.compute_fisher() uses individual examples and stops after
            # at most 256 per task. This mirrors the task-IL accounting path.
            per_task = [min(int(n), EWC_FISHER_NUM_SAMPLES) for n in counts]
            fisher_examples = sum(per_task)
            example_visits += fisher_examples
            schedule["fisher_passes_per_task"] = 1
            schedule["fisher_examples_per_task"] = int(EWC_FISHER_NUM_SAMPLES)
            schedule["fisher_example_visits"] = int(fisher_examples)
            schedule["fisher_backward_steps"] = int(fisher_examples)
        if method == "lwf":
            schedule["teacher_forward_example_visits"] = int(
                sum(counts[1:]) * epochs)
        step_note = "exact for declared current-data training loops"
    return {
        "kind": "declared-schedule accounting",
        "model_train_sample_counts_per_task": counts,
        "current_data_example_visits": int(example_visits),
        "optimizer_steps": int(steps),
        "optimizer_steps_note": step_note,
        "schedule": schedule,
        "replay_examples_included": False,
        "flops_measured": False,
    }






def _serialized_state_record(payload):
    """Hash an actual dense PyTorch state used by the router evaluation."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    record = {
        "state_definition": "router-evaluation dense PyTorch state",
        "format": payload.get("format", "pytorch-state-v1"),
        "serialized_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": _state_content_sha256(payload),
        "deployable_compact_export": False,
    }
    return record, raw




def _diltcl_state_payload(diltcl):
    return {
        "format": "diltcl-dense-resume-state-v1",
        "method_hyperparameters": {
            "num_partitions": diltcl.num_partitions,
            "pruning_rounds": diltcl.pruning_rounds,
            "target_sparsity": diltcl.target_sparsity,
            "full_train_epochs": diltcl.full_train_epochs,
            "imp_epochs": diltcl.imp_epochs,
            "expert_finetune_epochs": diltcl.expert_finetune_epochs,
            "lr": diltcl.lr,
            "rewind_point": "post-phase1-late-rewind",
        },
        "core_partition_densities": list(diltcl.core_partition_densities),
        "model": _cpu_tensor_dict(diltcl.model.state_dict()),
        "core_mask": _cpu_tensor_dict(diltcl.core_mask),
        "expert_masks": {
            str(task): _cpu_tensor_dict(mask)
            for task, mask in sorted(diltcl.expert_masks.items())
        },
        "expert_weights": {
            str(task): _cpu_tensor_dict(weights)
            for task, weights in sorted(diltcl.expert_weights.items())
        },
        "initial_params": _cpu_tensor_dict(diltcl.initial_params),
        "frozen_core_values": _cpu_tensor_dict(diltcl.frozen_core_values),
    }


def _benchmark_metadata_template(benchmark, cfg):
    """Protocol metadata that does not depend on a particular seed."""
    lay = bench.layout(benchmark, cfg)
    return {
        "dataset": lay.display_name,
        "task_names": list(lay.task_names) if lay.task_names else None,
        "task_count": int(lay.num_tasks),
        "classes_per_task": int(lay.classes_per_task),
        "global_class_count": int(lay.global_classes),
        "label_relation": lay.label_relation,
        "shared_head_output_count": int(lay.global_classes),
        "isolation_local_head_output_count": int(lay.classes_per_task),
        "per_seed_counts": {},
    }


def _loader_count_record(setup):
    """Record train/test sample counts for one available method family."""
    if setup is None:
        return None
    train_counts = [int(len(loader.dataset)) for loader in setup[0]]
    test_counts = [int(len(loader.dataset)) for loader in setup[1]]
    dataset_views = [describe_task_loader(loader, task_id)
                     for task_id, loader in enumerate(setup[0])]
    # Same record for the evaluation splits (v3.0.7): with the device backend
    # it carries the hash of the test pixels as well.
    test_views = [describe_task_loader(loader, task_id)
                  for task_id, loader in enumerate(setup[1])]
    return {
        "train_samples_per_task_before_calibration_holdout": train_counts,
        "test_samples_per_task": test_counts,
        "train_sample_count_before_calibration_holdout": int(sum(train_counts)),
        "test_sample_count": int(sum(test_counts)),
        "dataset_views": dataset_views,
        "test_dataset_views": test_views,
    }


def _record_benchmark_counts(metadata, seed, shared, isolation):
    """Attach all loader counts obtainable for this seed and method family."""
    record = {}
    shared_counts = _loader_count_record(shared)
    isolation_counts = _loader_count_record(isolation)
    if shared_counts is not None:
        record["shared_backbone"] = shared_counts
    if isolation_counts is not None:
        record["isolation"] = isolation_counts
    metadata["per_seed_counts"][str(seed)] = record
    return record


def _loader_from_subset(loader, subset, *, shuffle):
    """Rebuild one of the project's loaders over ``subset`` (either backend)."""
    # Do not attach a private Generator: the driver calls set_seed before each
    # method so all methods receive the same shuffle order from the global RNG.
    return make_loader(
        subset, loader.batch_size, shuffle,
        num_workers=getattr(loader, "num_workers", 0),
        collate_fn=getattr(loader, "collate_fn", None),
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=False)


def _holdout_train_loaders(loaders, fraction, split_seed):
    """Deterministically split every task into model-train and calibration.

    The same helper and seed are applied to shared and isolation dataset views,
    so enabling learned routing does not silently give isolation methods more
    model-training examples than the compared shared-backbone methods.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")
    train_loaders, calibration_loaders, counts = [], [], []
    splits = deterministic_holdout_indices(
        [len(loader.dataset) for loader in loaders], fraction, split_seed)
    for task_id, (loader, split) in enumerate(zip(loaders, splits)):
        calibration_idx = split["calibration"]
        training_idx = split["train"]
        train_loaders.append(_loader_from_subset(
            loader, subset_dataset(loader.dataset, training_idx), shuffle=True))
        calibration_loaders.append(_loader_from_subset(
            loader, subset_dataset(loader.dataset, calibration_idx),
            shuffle=False))
        counts.append({"task": task_id, "train": len(training_idx),
                       "calibration": len(calibration_idx)})
    return train_loaders, calibration_loaders, counts


# ---------------------------------------------------------------------------
# Benchmark setups
# ---------------------------------------------------------------------------

def _shared_setup(benchmark, cfg, seed, execution=None):
    """Class-IL loaders + a single global-head builder (shared-backbone path)."""
    lay = bench.layout(benchmark, cfg)
    train_loaders, test_loaders = bench.build_loaders(
        benchmark, cfg, seed, class_il=True, execution=execution)
    return SharedSetup(train_loaders, test_loaders,
                       bench.model_builder(benchmark, lay.global_classes),
                       lay.num_tasks)


def _isolation_setup(benchmark, cfg, seed, execution=None):
    """Task-IL loaders + a per-task builder (isolation path)."""
    lay = bench.layout(benchmark, cfg)
    train_loaders, test_loaders = bench.build_loaders(
        benchmark, cfg, seed, class_il=False, execution=execution)
    return IsolationSetup(train_loaders, test_loaders,
                          bench.model_builder(benchmark, lay.classes_per_task),
                          lay.num_tasks, lay.classes_per_task,
                          lay.disjoint_classes)


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------

def _run_shared(method, cfg, train_loaders, test_loaders, builder, num_tasks,
                device, execution=None):
    """Run shared class-IL and hash the exact evaluated model state."""
    execution = execution or ExecutionOptions()
    e = cfg.epochs_replay
    if method == "finetune":
        M, _, _, state = train_finetune(
            builder, train_loaders, test_loaders, num_tasks,
            device=device, epochs=e, lr=cfg.lr, return_state=True)
    elif method == "ewc":
        M, _, _, state = train_ewc(
            builder(), train_loaders, test_loaders, num_tasks,
            device=device, lambda_ewc=cfg.ewc_lambda,
            epochs=cfg.epochs_ewc, lr=cfg.lr, return_state=True,
            fisher_labels=cfg.ewc_fisher_labels, **execution.ewc_kwargs())
    elif method == "lwf":
        M, _, _, state = train_lwf(
            builder(), train_loaders, test_loaders, num_tasks,
            device=device, alpha=cfg.lwf_alpha,
            temperature=cfg.lwf_temperature,
            epochs=cfg.epochs_lwf, lr=cfg.lr, return_state=True,
            **execution.lwf_kwargs())
    elif method == "er":
        M, _, _, state = train_er(
            builder, train_loaders, test_loaders, num_tasks,
            device=device, epochs=e, lr=cfg.lr,
            buffer_size=cfg.buffer_size,
            replay_batch_size=cfg.replay_batch_size, return_state=True,
            **execution.replay_kwargs())
    elif method == "derpp":
        M, _, _, state = train_derpp(
            builder, train_loaders, test_loaders, num_tasks,
            device=device, epochs=e, lr=cfg.lr,
            buffer_size=cfg.buffer_size,
            replay_batch_size=cfg.replay_batch_size,
            alpha=cfg.derpp_alpha, beta=cfg.derpp_beta, return_state=True,
            **execution.replay_kwargs())
    else:
        raise ValueError(method)
    matrix = np.asarray(M)
    metrics = summarize_run(matrix)
    counts = np.asarray([len(loader.dataset) for loader in test_loaders],
                        dtype=float)
    metrics["MicroACC"] = float(np.average(matrix[-1], weights=counts))
    checkpoint, checkpoint_bytes = _serialized_state_record({
        "format": f"{method}-shared-class-il-inference-state-v1",
        "model": _cpu_tensor_dict(state["model"].state_dict()),
    })
    buffer = state.get("buffer")
    checkpoint["method_state_metadata"] = ({
        "replay_sample_calls": int(buffer.sample_calls),
        "replay_examples_sampled": int(buffer.sampled_items_total),
        "buffer_examples_retained": int(len(buffer)),
        "stream_examples_seen": int(buffer.num_seen),
    } if buffer is not None else {})
    return matrix, metrics, checkpoint, checkpoint_bytes


def _run_isolation(method, cfg, setup, device, criterion, *,
                   calibration="none", calibration_loaders=None,
                   calibration_c=1.0, calibration_seed=0,
                   split_counts=None, record_predictions=False):
    """Train an isolation method (task-IL) and evaluate it under class-IL.

    Returns ``(matrix, metrics, calibration_record, aux_matrices,
    checkpoint_bytes, predictions)``. Metrics carry standard matrix summaries,
    oracle-route, routing and conditional accuracy; auxiliary matrices
    preserve routing diagnostics and the matched raw-router ablation when
    calibration is enabled. ``predictions`` is ``None`` unless
    ``record_predictions``; then it maps array names to per-example arrays
    recorded from the same evaluation passes (``raw_*`` for the matched raw
    router of a calibrated run, ``calibration_*`` for the router's held-out
    training inputs).
    """
    train_loaders, test_loaders, builder, num_tasks, cpt, disjoint = setup

    if method == "packnet":
        _, _, _, state = train_packnet(
            builder, train_loaders, test_loaders, num_tasks, device=device,
            epochs=cfg.epochs_replay, lr=cfg.lr,
            retrain_epochs=cfg.packnet_retrain_epochs,
            keep_fraction=(cfg.packnet_keep_fraction or None), return_state=True)
        forward_t = packnet_forward_fn(state["model"], state["eval_masks"])
        checkpoint_payload = {
            "format": "packnet-router-state-v1",
            "model": _cpu_tensor_dict(state["model"].state_dict()),
            "eval_masks": [
                _cpu_tensor_dict(mask) for mask in state["eval_masks"]
            ],
        }
        state_metadata = {}
    elif method == "supsup":
        _, _, _, state = train_supsup(
            builder, train_loaders, test_loaders, num_tasks, device=device,
            epochs=cfg.epochs_replay, lr=cfg.lr, sparsity=cfg.supsup_sparsity,
            return_state=True)
        forward_t = supsup_forward_fn(state["model"])
        checkpoint_payload = {
            "format": "supsup-router-state-v1",
            "model": _cpu_tensor_dict(state["model"].state_dict()),
        }
        state_metadata = {}
    elif method == "wsn":
        _, _, _, state = train_wsn(
            builder, train_loaders, test_loaders, num_tasks, device=device,
            epochs=cfg.epochs_replay, lr=cfg.lr, sparsity=cfg.wsn_sparsity,
            return_state=True)
        forward_t = wsn_forward_fn(state["model"])
        checkpoint_payload = {
            "format": "wsn-router-state-v1",
            "model": _cpu_tensor_dict(state["model"].state_dict()),
        }
        state_metadata = {}
    elif method == "diltcl":
        _, _, _, diltcl = train_diltcl(
            builder(), train_loaders, test_loaders, num_tasks, device=device,
            num_partitions=cfg.diltcl_partitions, pruning_rounds=cfg.pruning_rounds,
            target_sparsity=cfg.diltcl_sparsity,
            full_train_epochs=cfg.diltcl_full_epochs,
            imp_epochs=cfg.diltcl_imp_epochs,
            expert_finetune_epochs=cfg.diltcl_expert_finetune_epochs,
            lr=cfg.lr, return_model=True)
        forward_t = diltcl_forward_fn(diltcl)
        checkpoint_payload = _diltcl_state_payload(diltcl)
        complement = masking.complement_mask(diltcl.core_mask)
        state_metadata = {
            "requested_target_sparsity": float(diltcl.target_sparsity),
            "pruning_rounds": int(diltcl.pruning_rounds),
            "per_round_pruning_rate": float(masking.iterative_pruning_rate(
                diltcl.target_sparsity, diltcl.pruning_rounds)),
            "core_partition_global_density": [
                float(value) for value in diltcl.core_partition_densities
            ],
            "core_global_density": masking.mask_density(diltcl.core_mask),
            "expert_global_density_by_task": {
                str(task): masking.mask_density(mask)
                for task, mask in sorted(diltcl.expert_masks.items())
            },
            "expert_eligible_density_by_task": {
                str(task): masking.mask_density_in_region(mask, complement)
                for task, mask in sorted(diltcl.expert_masks.items())
            },
        }
    else:
        raise ValueError(method)

    checkpoint_record, checkpoint_bytes = _serialized_state_record(
        checkpoint_payload)
    checkpoint_record["method_state_metadata"] = state_metadata

    calibrators = None
    calibration_record = {
        "kind": "none",
        "trained_state": checkpoint_record,
    }
    if calibration in CALIBRATION_KINDS:
        if calibration_loaders is None:
            raise ValueError(f"{calibration} requires held-out training loaders")
        features, task_ids, observed_counts = collect_calibration_features(
            forward_t, calibration_loaders, num_tasks, device=device,
            criterion=criterion)
        calibrators = fit_prefix_calibrators(
            features, task_ids, num_tasks, C=calibration_c,
            random_state=calibration_seed, kind=calibration)
        if any(state.get("kind") != calibration for state in calibrators):
            raise RuntimeError(
                f"prefix calibrator kind violates the {calibration} artifact "
                "contract")
        def _state_floats(state):
            if state.get("kind") in ("score-bias", "score-bias-acc"):   # tied slope + per-task bias
                return (len(state["feature_mean"]) + len(state["feature_scale"])
                        + 1 + len(state["intercept"]))
            return (len(state["feature_mean"]) + len(state["feature_scale"])
                    + sum(len(row) for row in state["coef"])
                    + len(state["intercept"]))
        calibration_record = {
            "kind": calibration,
            "trained_state": checkpoint_record,
            "source": "heldout_train",
            "base_criterion": criterion,
            "C": float(calibration_c),
            "split_seed": int(calibration_seed),
            "observed_counts": observed_counts,
            "split_counts": split_counts or [],
            # Every prefix router is applied in float64, the precision it was
            # fitted in (each state's routing_dtype; v3.0.10).
            "routing_dtype": "float64",
            "prefix_models": calibrators,
            "final_serving_float_count": _state_floats(calibrators[-1]),
            "serialized_prefix_float_count": sum(
                _state_floats(state) for state in calibrators),
            "deployment_note": (
                "Only the final-stage router is needed for final serving "
                "(O(T^2) floats for score-logreg, O(T) for score-bias); "
                "earlier prefix states are serialized for temporal audit and "
                "may be discarded in deployment."),
            "retention_note": (
                "Calibration inputs are not stored in the result artifact; "
                "a strict online implementation must retain training holdout "
                "inputs until each future score dimension is available."),
        }
    elif calibration != "none":
        raise ValueError(f"unknown inference calibration: {calibration}")

    raw_res = None
    raw_record = {} if (record_predictions and calibrators is not None) else None
    if calibrators is not None:
        print(f"  [class-IL inference: raw {criterion}] evaluating matched "
              "checkpoint...")
        raw_res = evaluate_class_il_inference(
            forward_t, test_loaders, num_tasks, cpt, device=device,
            criterion=criterion, disjoint_classes=disjoint, record=raw_record)

    route_name = (criterion if calibrators is None
                  else f"{criterion}+{calibration}")
    print(f"  [class-IL inference: {route_name}] evaluating {num_tasks} tasks...")
    record = {} if record_predictions else None
    res = evaluate_class_il_inference(
        forward_t, test_loaders, num_tasks, cpt, device=device,
        criterion=criterion, disjoint_classes=disjoint,
        calibrators=calibrators, record=record)
    predictions = None
    if record_predictions:
        predictions = {key: value for key, value in record.items()
                       if isinstance(value, np.ndarray)}
        if raw_record is not None:
            predictions.update({f"raw_{key}": value
                                for key, value in raw_record.items()
                                if isinstance(value, np.ndarray)})
        if calibrators is not None:
            predictions["calibration_features"] = np.asarray(
                features, dtype=np.float32)
            predictions["calibration_task_ids"] = np.asarray(
                task_ids, dtype=np.int16)
        predictions["_meta"] = {
            "criterion": criterion,
            "calibration": calibration,
            "num_tasks": int(num_tasks),
            "classes_per_task": int(cpt),
            "disjoint_classes": bool(disjoint),
            "trained_state_content_sha256": checkpoint_record["content_sha256"],
        }
    print(f"  class-IL ACC={res['class_il_acc']:.2f}  "
          f"oracle-route={res['oracle_route_acc']:.2f}  "
          f"task-inference={res['task_inference_acc']:.2f}")

    M = np.asarray(res["result_matrix"])
    metrics = summarize_run(M)
    metrics.update({
        "OracleACC": res["oracle_route_acc"],
        "TA": res["task_inference_acc"],
        "MicroACC": res["micro_class_il_acc"],
        "MicroOracleACC": res["micro_oracle_route_acc"],
        "MicroTA": res["micro_task_inference_acc"],
        "ConditionalACC": res["conditional_class_acc"],
        "MicroConditionalACC": res["micro_conditional_class_acc"],
        "RouteTieRate": res["routing_tie_rate"],
        "MicroRouteTieRate": res["micro_routing_tie_rate"],
        "ZeroCorrectRouteTasks": res["zero_correct_route_task_count"],
    })
    aux_matrices = {
        "task_inference": np.asarray(res["task_inference_matrix"]),
        "conditional_class": np.asarray(res["conditional_class_matrix"]),
        "routing_tie": np.asarray(res["routing_tie_matrix"]),
    }
    if raw_res is not None:
        raw_matrix = np.asarray(raw_res["result_matrix"])
        raw_summary = summarize_run(raw_matrix)
        metrics.update({
            "RawACC": raw_summary["ACC"],
            "RawLA": raw_summary["LA"],
            "RawBWT": raw_summary["BWT"],
            "RawForgetting": raw_summary["Forgetting"],
            "RawTA": raw_res["task_inference_acc"],
            "RawMicroACC": raw_res["micro_class_il_acc"],
            "RawMicroTA": raw_res["micro_task_inference_acc"],
            "RawConditionalACC": raw_res["conditional_class_acc"],
            "RawMicroConditionalACC": raw_res[
                "micro_conditional_class_acc"],
            "RawRouteTieRate": raw_res["routing_tie_rate"],
            "RawMicroRouteTieRate": raw_res["micro_routing_tie_rate"],
            "RawZeroCorrectRouteTasks": raw_res[
                "zero_correct_route_task_count"],
        })
        calibration_record["matched_raw_metrics"] = {
            key: metrics[key] for key in metrics if key.startswith("Raw")}
        calibration_record["matched_checkpoint_sha256"] = \
            checkpoint_record["sha256"]
        calibration_record["matched_checkpoint_content_sha256"] = \
            checkpoint_record["content_sha256"]
        aux_matrices.update({
            "raw_result": raw_matrix,
            "raw_task_inference": np.asarray(
                raw_res["task_inference_matrix"]),
            "raw_conditional_class": np.asarray(
                raw_res["conditional_class_matrix"]),
            "raw_routing_tie": np.asarray(raw_res["routing_tie_matrix"]),
        })
    return M, metrics, calibration_record, aux_matrices, checkpoint_bytes, predictions


def _persist_predictions(predictions, predictions_dir, benchmark, method, seed,
                         extra_meta):
    """Write per-example routing outcomes; return a provenance record.

    The NPZ holds only arrays (no pickled objects); metadata is a JSON string
    under ``meta_json``. The record stores the file's SHA-256 so the suite and
    the verifier can tie the file to this job.
    """
    meta = dict(predictions.get("_meta", {}))
    meta.update(extra_meta)
    arrays = {key: value for key, value in predictions.items()
              if not key.startswith("_")}
    path = os.path.abspath(os.path.join(
        predictions_dir, benchmark, method, f"seed{seed}.npz"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(temporary, meta_json=np.array(json.dumps(meta, sort_keys=True)),
                        **arrays)
    os.replace(temporary, path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "format": "diltcl-class-il-predictions-v1",
        "path": path,
        "bytes": os.path.getsize(path),
        "sha256": digest.hexdigest(),
        "arrays": sorted(arrays),
        "examples": int(arrays["true_task"].shape[0]) if "true_task" in arrays else 0,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _calibration_seed(split_seed, seed):
    """Derive the calibration seed for ``seed``.

    The holdout split and the router fit must use the *same* value. This was
    previously written out twice inside ``run()``; if the two copies had ever
    diverged the router would have been fitted on a different split than the
    one used to train the models, silently and without error.
    """
    return int(split_seed) + int(seed) * 100_000


def _validate_run_arguments(cfg, methods, seeds, calibration,
                            calibration_fraction, calibration_c, need_iso):
    """Validate every argument before any device or dataset side effect."""
    cfg.validate()
    if not methods:
        raise ValueError("at least one method is required")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must be unique")
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if not np.isfinite(calibration_c) or calibration_c <= 0:
        raise ValueError("calibration_c must be finite and strictly positive")
    if not np.isfinite(calibration_fraction):
        raise ValueError("calibration_fraction must be finite")
    if calibration != "none" and calibration not in CALIBRATION_KINDS:
        raise ValueError(f"unknown inference calibration: {calibration}")
    # Calibration fits routers for isolation methods only, but the matched
    # 90% training holdout applies to every method in the run, so a job that
    # holds only shared-backbone methods is legitimate under calibration (the
    # suite runs one method per job); it simply records no router.
    if calibration != "none" and not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")


def _print_run_banner(device, criterion, calibration, calibration_fraction,
                      benchmarks, methods, seeds):
    """Announce the configuration of a class-IL run."""
    print(f"Using device: {device}")
    print("Protocol:   class-IL (no task identity); isolation methods use "
          f"{criterion}-based task inference")
    print(f"Calibration: {calibration}"
          + (f" (held-out train fraction={calibration_fraction:.3f})"
             if calibration != "none" else ""))
    print(f"Benchmarks: {benchmarks}")
    print(f"Methods:    {methods}")
    print(f"Seeds:      {seeds}")


def _aggregate_and_report(results, benchmarks, methods, seeds, need_iso,
                          calibration):
    """Aggregate per-seed metrics and print the comparison tables.

    Pure with respect to the experiment: it reads finished per-seed metrics and
    returns the aggregate, so it can be exercised without training anything.
    """
    aggregated = {b: {} for b in benchmarks}
    for benchmark in benchmarks:
        for method in methods:
            aggregated[benchmark][method] = aggregate_seeds(
                results[benchmark][method])
        print("\n" + "=" * 70)
        print(f"CLASS-IL RESULTS: {benchmark.upper()}  (mean +/- std over "
              f"{len(seeds)} seeds)")
        print("ACC/LA/BWT/Forgetting use the same prefix-candidate class-IL "
              "matrix for both families. Isolation methods additionally report "
              "OracleACC and final task-inference accuracy TA.")
        print("=" * 70)
        # Significance needs at least five aligned seeds; below that the
        # reference column is dropped rather than shown without stars.
        ref = ("derpp" if "derpp" in methods and len(seeds) >= 5 else None)
        if "derpp" in methods and len(seeds) < 5:
            print("Significance output suppressed: fewer than five aligned seeds.")
        metrics_to_show = ["ACC", "LA", "BWT", "Forgetting"]
        if need_iso:
            metrics_to_show.extend([
                "OracleACC", "TA", "ConditionalACC", "RouteTieRate",
                "ZeroCorrectRouteTasks",
            ])
        if calibration != "none":
            metrics_to_show.extend(["RawACC", "RawTA"])
        for metric in metrics_to_show:
            print()
            print(format_comparison_table(
                aggregated[benchmark],
                reference=(ref if metric == "ACC" else None),
                metric=metric, order=methods))
    return aggregated


def _serialize_results(*, out_prefix, cfg, benchmarks, methods, seeds,
                       results, aggregated, matrices, run_records,
                       calibration_records, benchmark_metadata, device,
                       criterion, calibration, calibration_fraction,
                       calibration_c, calibration_split_seed, checkpoint_dir,
                       execution=None):
    """Assemble the artifact payload and write the JSON/NPZ pair.

    Extracted from ``run()`` so the payload schema can be exercised without
    training. Keyword-only because the block genuinely needs this much run
    state; naming every argument keeps the call site self-documenting.
    """
    out_parent = os.path.dirname(os.path.abspath(out_prefix))
    os.makedirs(out_parent, exist_ok=True)
    json_payload = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        # A valid metric schema is necessary but not sufficient evidence for a
        # paper claim. New artifacts remain unverified until an independent
        # rerun/validation process explicitly changes this status.
        "evidence_status": "unverified",
        "verification_status": "unverified",
        "artifact_reportability": {
            "metric_schema_emitted": True,
            "postrun_validation_completed": False,
            "verification_independent_of_metric_schema": True,
            "reportable_if": "verification_status == verified",
            "current_publication_reportable": False,
            "unmet_at_write_time": [
                *([] if checkpoint_dir is not None else [
                    "persisted evaluated-state checkpoint files"]),
                "independent JSON/NPZ recomputation report",
                "code commit or complete source hashes",
                "dataset identity/order/transforms and hashes",
                "achieved density for every isolation baseline, hardware-counter "
                "FLOPs, and exact replay-example accounting",
                "failure/exclusion record",
            ],
        },
        "protocol": "class_il",
        "matrix_semantics": "prefix_candidate_tasks_macro_task_average",
        "prefix_rule": {
            "row_i": "after training task i",
            "evaluated_true_tasks": "0..i",
            "isolation_candidate_routes": "0..i",
            "shared_head": "fixed full head; metrics include only seen true tasks 0..i",
            "upper_triangle": "undefined and serialized as zero",
            "metric_domain": "valid lower-triangular entries only",
        },
        "head_conventions": {
            "shared_backbone": "fixed-full-head",
            "isolation": "task-local-head+router",
            "task_identity_at_test": False,
            "per_benchmark_output_counts": {
                b: {
                    "shared": benchmark_metadata[b]["shared_head_output_count"],
                    "isolation_local": benchmark_metadata[b][
                        "isolation_local_head_output_count"],
                }
                for b in benchmarks
            },
        },
        "tie_policy": {
            "definition": "exact equality at the router decision optimum",
            "selection": "lowest task id",
            "reported_metrics": ["RouteTieRate", "MicroRouteTieRate"],
        },
        "conditional_accuracy_definition": (
            "pooled P(class prediction correct | inferred route correct); "
            "ZeroCorrectRouteTasks is diagnostic and no task-level "
            "selective macro average is used"),
        "npz_auxiliary_suffixes": {
            "task_inference": "prefix task-routing accuracy matrix",
            "conditional_class": (
                "prefix class accuracy conditioned on a correct route"),
            "routing_tie": "prefix exact router-tie-rate matrix",
            "raw_result": "matched-checkpoint uncalibrated class-IL matrix",
            "raw_task_inference": (
                "matched-checkpoint uncalibrated routing matrix"),
            "raw_conditional_class": (
                "matched-checkpoint uncalibrated conditional-class matrix"),
            "raw_routing_tie": (
                "matched-checkpoint uncalibrated router-tie-rate matrix"),
        },
        "inference_criterion": criterion,
        "inference_calibration": calibration,
        "calibration_fraction": (calibration_fraction
                                 if calibration != "none" else 0.0),
        "calibration_c": calibration_c,
        "calibration_split_seed": calibration_split_seed,
        "config": cfg.__dict__,
        "execution": (execution or ExecutionOptions()).provenance(),
        "seeds": seeds,
        "benchmarks": benchmarks,
        "methods": methods,
        "checkpoint_persistence_enabled": checkpoint_dir is not None,
        "benchmark_metadata": benchmark_metadata,
        "environment": _environment_metadata(device),
        "per_seed_run_records": run_records,
        "shared_backbone_methods": [m for m in methods if m in SHARED_BACKBONE_METHODS],
        "isolation_methods": [m for m in methods if m in ISOLATION_METHODS],
        "per_seed": results,
        "aggregated": aggregated,
        "calibration": calibration_records,
    }
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(json_payload, f, indent=2, allow_nan=False)
    np.savez(f"{out_prefix}.npz", **matrices)
    print(f"\nSaved class-IL metrics to '{out_prefix}.json'")
    print(f"Saved raw result matrices to '{out_prefix}.npz'")
    print("\nDone.")


def run(cfg, benchmarks, methods, seeds, out_prefix="class_il_results",
        criterion="entropy", calibration="none", calibration_fraction=0.1,
        calibration_c=1.0, calibration_split_seed=1729,
        checkpoint_dir=None, execution=None, predictions_dir=None):
    """Run the class-IL comparison across both families; write JSON / NPZ."""
    execution = execution or ExecutionOptions()
    need_shared = any(m in SHARED_BACKBONE_METHODS for m in methods)
    need_iso = any(m in ISOLATION_METHODS for m in methods)
    # Validate arguments before any side effect: the legacy ordering initialised
    # the device and printed a banner before rejecting a bad calibration name.
    _validate_run_arguments(cfg, methods, seeds, calibration,
                            calibration_fraction, calibration_c, need_iso)
    # Check the expensive prerequisites before any training starts. Calibrated
    # runs previously crashed at the evaluation step, discarding every trained
    # model in the job.
    preflight.enforce(
        preflight.resolve_data_root(), benchmarks,
        require_training=True,
        require_calibration=str(calibration).lower() not in ("none", ""),
        header="run_class_il preflight")

    device = get_device()
    execution.apply()
    _print_run_banner(device, criterion, calibration, calibration_fraction,
                      benchmarks, methods, seeds)
    print(f"Execution: backend={execution.data_backend} "
          f"numerics={execution.numerics} tf32={execution.tf32} "
          f"threads={torch.get_num_threads()}")


    results = {b: {m: [] for m in methods} for b in benchmarks}
    matrices = {}
    calibration_records = {b: {m: {} for m in methods if m in ISOLATION_METHODS}
                           for b in benchmarks}
    benchmark_metadata = {
        b: _benchmark_metadata_template(b, cfg) for b in benchmarks
    }
    run_records = {b: {m: {} for m in methods} for b in benchmarks}

    for benchmark in benchmarks:
        for seed in seeds:
            set_seed(seed)
            shared = (_shared_setup(benchmark, cfg, seed, execution=execution)
                      if need_shared else None)
            iso = (_isolation_setup(benchmark, cfg, seed, execution=execution)
                   if need_iso else None)
            count_record = _record_benchmark_counts(
                benchmark_metadata[benchmark], seed, shared, iso)
            iso_calibration_loaders = None
            split_counts = None
            if calibration != "none":
                # The same deterministic holdout is applied to both families
                # so calibrated isolation methods are compared against
                # shared-backbone methods trained on identical 90% splits,
                # whether or not both families run in this process.
                split_seed = _calibration_seed(calibration_split_seed, seed)
                if shared is not None:
                    sh_train, sh_calibration_loaders, _ = _holdout_train_loaders(
                        shared.train_loaders, calibration_fraction, split_seed)
                    shared = shared._replace(train_loaders=sh_train)
                    count_record["shared_backbone"][
                        "model_train_samples_per_task"] = [
                            int(len(loader.dataset))
                            for loader in shared.train_loaders]
                    count_record["shared_backbone"][
                        "calibration_samples_per_task"] = [
                            int(len(loader.dataset))
                            for loader in sh_calibration_loaders]
                if iso is not None:
                    iso_train, iso_calibration_loaders, split_counts = \
                        _holdout_train_loaders(iso.train_loaders,
                                               calibration_fraction, split_seed)
                    iso = iso._replace(train_loaders=iso_train)
                    count_record["isolation"][
                        "model_train_samples_per_task"] = [
                            int(len(loader.dataset))
                            for loader in iso.train_loaders]
                    count_record["isolation"][
                        "calibration_samples_per_task"] = [
                            int(len(loader.dataset))
                            for loader in iso_calibration_loaders]
            else:
                for family_record in count_record.values():
                    family_record["model_train_samples_per_task"] = list(
                        family_record[
                            "train_samples_per_task_before_calibration_holdout"])
                    family_record["calibration_samples_per_task"] = [
                        0 for _ in family_record["model_train_samples_per_task"]]
            for method in methods:
                print("\n" + "#" * 70)
                print(f"# CLASS-IL | {benchmark.upper()} | {method.upper()} | "
                      f"seed={seed}")
                print("#" * 70)
                set_seed(seed)
                baseline_bytes = None
                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                    # Memory already resident before the method starts (with
                    # the device data backend this is mostly the dataset), so
                    # the peak below can be read net of it.
                    baseline_bytes = int(torch.cuda.memory_allocated(device))
                _cuda_sync(device)
                started = time.perf_counter()
                if method in SHARED_BACKBONE_METHODS:
                    M, metrics, trained_state, checkpoint_bytes = _run_shared(
                        method, cfg, *shared, device, execution=execution)
                    accounting_loaders = shared.train_loaders
                else:
                    fit_seed = _calibration_seed(calibration_split_seed, seed)
                    (M, metrics, calibration_record, aux_matrices,
                     checkpoint_bytes, predictions) = _run_isolation(
                        method, cfg, iso, device, criterion,
                        calibration=calibration,
                        calibration_loaders=iso_calibration_loaders,
                        calibration_c=calibration_c,
                        calibration_seed=fit_seed,
                        split_counts=split_counts,
                        record_predictions=predictions_dir is not None)
                    if predictions is not None:
                        calibration_record["predictions"] = _persist_predictions(
                            predictions, predictions_dir, benchmark, method, seed,
                            {"benchmark": benchmark, "method": method,
                             "seed": int(seed), "code_version": CODE_VERSION})
                        calibration_record["predictions"]["relative_path"] = \
                            os.path.relpath(
                                calibration_record["predictions"]["path"],
                                os.path.dirname(os.path.abspath(out_prefix)))
                        predictions = None
                    calibration_records[benchmark][method][str(seed)] = \
                        calibration_record
                    trained_state = calibration_record["trained_state"]
                    accounting_loaders = iso.train_loaders
                trained_state["persisted"] = False
                trained_state["persisted_path"] = None
                trained_state["persisted_relative_path"] = None
                if checkpoint_dir is not None:
                    checkpoint_path = _persist_checkpoint(
                        checkpoint_bytes, checkpoint_dir, benchmark, method, seed)
                    trained_state["persisted"] = True
                    trained_state["persisted_path"] = checkpoint_path
                    trained_state["persisted_relative_path"] = os.path.relpath(
                        checkpoint_path,
                        os.path.dirname(os.path.abspath(out_prefix)))
                checkpoint_bytes = None
                _cuda_sync(device)
                wall_seconds = time.perf_counter() - started
                peak_bytes = None
                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    peak_bytes = int(torch.cuda.max_memory_allocated(device))
                schedule_accounting = _schedule_accounting(
                    method, cfg, accounting_loaders)
                replay_count = trained_state.get(
                    "method_state_metadata", {}).get(
                        "replay_examples_sampled")
                if replay_count is not None:
                    schedule_accounting["replay_examples_sampled"] = int(
                        replay_count)
                    schedule_accounting["total_training_example_presentations"] = (
                        schedule_accounting["current_data_example_visits"]
                        + int(replay_count))
                    schedule_accounting["replay_examples_included"] = True
                run_records[benchmark][method][str(seed)] = {
                    "pipeline_wall_seconds": float(wall_seconds),
                    "peak_cuda_memory_allocated_bytes": peak_bytes,
                    "cuda_memory_allocated_before_method_bytes": baseline_bytes,
                    "includes_training_and_all_router_evaluations": True,
                    # Per-example recording (--predictions-dir) adds score
                    # transfers and one compressed write inside the timed span.
                    "includes_prediction_recording": bool(
                        predictions_dir is not None
                        and method in ISOLATION_METHODS),
                    "trained_state": trained_state,
                    "schedule_accounting": schedule_accounting,
                }
                results[benchmark][method].append(metrics)
                matrices[f"{benchmark}/{method}/seed{seed}"] = np.asarray(M)
                if method in ISOLATION_METHODS:
                    for suffix, aux_matrix in aux_matrices.items():
                        matrices[f"{benchmark}/{method}/seed{seed}/{suffix}"] = \
                            aux_matrix

    # ---- Aggregate and print ----
    aggregated = _aggregate_and_report(results, benchmarks, methods, seeds,
                                       need_iso, calibration)

    _serialize_results(
        out_prefix=out_prefix, cfg=cfg, benchmarks=benchmarks,
        methods=methods, seeds=seeds, results=results,
        aggregated=aggregated, matrices=matrices,
        run_records=run_records,
        calibration_records=calibration_records,
        benchmark_metadata=benchmark_metadata, device=device,
        criterion=criterion, calibration=calibration,
        calibration_fraction=calibration_fraction,
        calibration_c=calibration_c,
        calibration_split_seed=calibration_split_seed,
        checkpoint_dir=checkpoint_dir, execution=execution)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Class-IL evaluation for shared-backbone and isolation "
                    "methods (the latter via exhaustive score-based task "
                    "inference, optionally calibrated on held-out training "
                    "data).")
    p.add_argument("--fast", action="store_true",
                   help="Reduced tasks/epochs for a quick run.")
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of seeds (default 5). Seeds are 0..N-1.")
    p.add_argument(
        "--seed-list", type=str, default=None,
        help="Explicit comma/range seed IDs, e.g. 0,2,4-7. Overrides --seeds.")
    p.add_argument("--benchmark",
                   choices=["cifar", "pmnist", "tinyimagenet", "fivedatasets",
                            "both"],
                   default="both", help="Which benchmark(s) to run. 'both' = "
                   "cifar+pmnist; the harder benchmarks are opt-in by name.")
    p.add_argument("--methods", type=str, default=",".join(ALL_METHODS),
                   help="Comma-separated subset of: " + ",".join(ALL_METHODS))
    p.add_argument("--inference-criterion",
                   choices=["entropy", "confidence", "margin"],
                   default="entropy",
                   help="Task-inference signal for isolation methods.")
    p.add_argument("--inference-calibration",
                   choices=["none", *CALIBRATION_KINDS], default="none",
                   help="Optional learned router fitted on held-out training "
                        "scores; never uses test data.")
    p.add_argument("--calibration-fraction", type=float, default=0.10,
                   help="Per-task training fraction held out for router calibration.")
    p.add_argument("--calibration-c", type=float, default=1.0,
                   help="Inverse L2 regularization of the calibrated router.")
    p.add_argument("--calibration-split-seed", type=int, default=1729,
                   help="Base seed for deterministic train/calibration splits.")
    p.add_argument("--out", type=str, default="class_il_results",
                   help="Output filename prefix.")
    p.add_argument("--data-root", type=str, default=None,
                   help="Dataset directory (overrides $DILTCL_DATA_ROOT).")
    p.add_argument("--no-download", action="store_true",
                   help="Never attempt to download datasets.")
    p.add_argument(
        "--config-json", type=str, default=None,
        help="Flat JSON object overriding ExperimentConfig fields. Unknown "
             "keys and invalid ranges are rejected.")
    p.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Persist the exact hashed PyTorch state for every method/seed. "
             "Strongly recommended for publication runs.")
    p.add_argument(
        "--predictions-dir", type=str, default=None,
        help="Write per-example routing outcomes (oracle prediction, every "
             "expert's routing score, the routed task and prediction at every "
             "prefix stage, and the calibration inputs of calibrated runs) for "
             "each isolation method/seed as compressed NPZ. Recording reuses "
             "the evaluation's own forward passes and changes no result.")
    add_execution_arguments(p)
    return p.parse_args()






def main():
    """Command-line entry point."""
    args = _parse_args()
    if args.data_root:
        os.environ["DILTCL_DATA_ROOT"] = args.data_root
    if args.no_download:
        os.environ["DILTCL_DOWNLOAD"] = "0"
    cfg = ExperimentConfig.fast() if args.fast else ExperimentConfig()
    try:
        cfg = _load_config_overrides(cfg, args.config_json)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid --config-json: {exc}") from exc
    benchmarks = (["cifar", "pmnist"] if args.benchmark == "both"
                  else [args.benchmark])
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise SystemExit("--methods must name at least one method")
    if len(set(methods)) != len(methods):
        raise SystemExit("--methods entries must be unique")
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Choose from {ALL_METHODS}")
    if args.seeds <= 0 and args.seed_list is None:
        raise SystemExit("--seeds must be strictly positive")
    try:
        seeds = (_parse_seed_spec(args.seed_list)
                 if args.seed_list is not None else list(range(args.seeds)))
    except ValueError as exc:
        raise SystemExit(f"Invalid --seed-list: {exc}") from exc
    try:
        execution = execution_from_args(args)
    except ValueError as exc:
        raise SystemExit(f"Invalid execution options: {exc}") from exc
    run(cfg, benchmarks, methods, seeds, out_prefix=args.out,
        criterion=args.inference_criterion,
        calibration=args.inference_calibration,
        calibration_fraction=args.calibration_fraction,
        calibration_c=args.calibration_c,
        calibration_split_seed=args.calibration_split_seed,
        checkpoint_dir=args.checkpoint_dir, execution=execution,
        predictions_dir=args.predictions_dir)


if __name__ == "__main__":
    main()
