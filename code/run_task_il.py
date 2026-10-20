"""Multi-seed task-IL method comparison with auditable raw outputs.

It runs every method over several random seeds, emits raw matrices and summary
statistics, and can persist the exact evaluated PyTorch state. Outputs begin as
``unverified``. ``--head-mode multi`` gives the shared-backbone baselines one
head per task (the standard, literature-comparable protocol); ``shared`` keeps
one reused task-local C-way head and exists only to reproduce legacy runs. The
mode used is recorded in the JSON (``task_il_head_mode``).

Historical note: this file was ``run_aaai_experiments.py`` until v3.0.0; the
old name remains as a forwarding shim.

Examples
--------
Quick sanity check (reduced tasks/epochs, 2 seeds, both benchmarks)::

    python run_task_il.py --fast --seeds 2

Full run, CIFAR only, 5 seeds, a subset of methods::

    python run_task_il.py --benchmark cifar --seeds 5 \
        --methods finetune,ewc,lwf,er,derpp,diltcl,independent

Outputs ``task_il_results.json`` (per-seed + aggregated metrics) and
``task_il_results.npz`` (raw result matrices).
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

import numpy as np
import torch

from diltcl import preflight
from diltcl import benchmarks as bench
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
from diltcl.diltcl import IMP_PARTITION_BATCH_SIZE
from diltcl import (
    ExperimentConfig,
    MultiHeadModel,
    aggregate_seeds,
    build_default_suite,
    format_comparison_table,
    format_cost_table_text,
    get_device,
    masking,
    evaluate_with_mask,
    set_seed,
    summarize_run,
    train_derpp,
    train_diltcl,
    train_er,
    train_ewc,
    train_finetune,
    train_independent,
    train_lwf,
    train_packnet,
    train_supsup,
    train_wsn,
)

# Only the shared-backbone baselines are affected by the task-IL head
# protocol. The isolation methods (and Independent) already hold task-specific
# parameters, so wrapping them in a multi-head model would make every head a
# prunable parameter and silently change what those methods mean.
MULTI_HEAD_ELIGIBLE = ("finetune", "ewc", "lwf", "er", "derpp")

ALL_METHODS = ["finetune", "ewc", "lwf", "er", "derpp",
               "packnet", "supsup", "wsn", "diltcl", "independent"]
def _package_version():
    """Return the installed package version, so provenance cannot drift."""
    from diltcl import __version__
    return __version__


# Single source of truth: the package version. Hardcoding this separately let
# it drift -- result files produced by v2.2.0 still reported "2.1.1", which
# then propagated into table captions and made provenance untrustworthy.
CODE_VERSION = _package_version()
METRIC_SCHEMA_VERSION = 2

def _environment_metadata(device):
    """Return a JSON-safe execution environment fingerprint."""
    cuda_available = bool(torch.cuda.is_available())
    device_name = None
    if cuda_available and str(device).startswith("cuda"):
        try:
            device_name = torch.cuda.get_device_name(device)
        except (AssertionError, RuntimeError, ValueError):
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
        "cudnn_version": (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "command": sys.argv,
    }


def _loader_metadata(train_loaders, test_loaders):
    """Record sample counts and any inspectable task-order identifiers."""
    train_counts = [int(len(loader.dataset)) for loader in train_loaders]
    test_counts = [int(len(loader.dataset)) for loader in test_loaders]
    return {
        "train_samples_per_task": train_counts,
        "test_samples_per_task": test_counts,
        "train_sample_count": int(sum(train_counts)),
        "test_sample_count": int(sum(test_counts)),
        "tasks": [describe_task_loader(loader, task_id)
                  for task_id, loader in enumerate(train_loaders)],
    }


EWC_FISHER_NUM_SAMPLES = 256   # default of EWC.compute_fisher(num_samples=...)


def _schedule_accounting(method, cfg, train_loaders):
    """Report deterministic schedule counts; these are not FLOP estimates."""
    counts = [int(len(loader.dataset)) for loader in train_loaders]
    batches = [int(len(loader)) for loader in train_loaders]
    if method == "diltcl":
        current_examples = (
            counts[0] * (cfg.diltcl_full_epochs
                         + cfg.pruning_rounds * cfg.diltcl_imp_epochs)
            + sum(counts[1:]) * (
                cfg.pruning_rounds * cfg.diltcl_imp_epochs
                + cfg.diltcl_expert_finetune_epochs)
        )
        # Partition IMP uses ``IMP_PARTITION_BATCH_SIZE`` internally. The
        # partitions cover task 0 exactly; this is an exact count of examples
        # but the step count can differ by at most P-1 per epoch because of
        # partition remainders.
        part_size = counts[0] // cfg.diltcl_partitions
        partition_sizes = [part_size] * max(0, cfg.diltcl_partitions - 1)
        partition_sizes.append(
            counts[0] - part_size * max(0, cfg.diltcl_partitions - 1))
        partition_steps = sum(math.ceil(size / IMP_PARTITION_BATCH_SIZE)
                              for size in partition_sizes)
        optimizer_steps = (
            batches[0] * cfg.diltcl_full_epochs
            + partition_steps * cfg.pruning_rounds * cfg.diltcl_imp_epochs
            + sum(batches[1:]) * (
                cfg.pruning_rounds * cfg.diltcl_imp_epochs
                + cfg.diltcl_expert_finetune_epochs)
        )
        schedule = {
            "phase1_full_epochs": cfg.diltcl_full_epochs,
            "partition_imp_rounds": cfg.pruning_rounds,
            "partition_imp_epochs_per_round": cfg.diltcl_imp_epochs,
            "partitions": cfg.diltcl_partitions,
            "expert_imp_rounds": cfg.pruning_rounds,
            "expert_imp_epochs_per_round": cfg.diltcl_imp_epochs,
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
        extra_epochs = cfg.packnet_retrain_epochs if method == "packnet" else 0
        current_examples = sum(counts) * (epochs + extra_epochs)
        optimizer_steps = sum(batches) * (epochs + extra_epochs)
        schedule = {"epochs_per_task": epochs}
        if method == "ewc":
            # compute_fisher() processes at most ``num_samples`` (256)
            # individual examples per task, one backward pass each -- not a
            # full pass over the task (which the accounting claimed before
            # v3.0.4: 50,000 visits instead of 5,120 on Split-CIFAR-100).
            per_task = [min(int(n), EWC_FISHER_NUM_SAMPLES) for n in counts]
            fisher_examples = sum(per_task)
            current_examples += fisher_examples
            schedule["fisher_passes_per_task"] = 1
            schedule["fisher_examples_per_task"] = int(EWC_FISHER_NUM_SAMPLES)
            schedule["fisher_example_visits"] = int(fisher_examples)
            schedule["fisher_backward_steps"] = int(fisher_examples)
        if method == "lwf":
            schedule["teacher_forward_example_visits"] = int(
                sum(counts[1:]) * epochs)
        if extra_epochs:
            schedule["post_prune_retrain_epochs_per_task"] = extra_epochs
        step_note = "exact for the declared current-data training loops"
    replay_note = None
    if method in {"er", "derpp"}:
        replay_note = (
            "optimizer steps are exact, but replay examples are not included "
            "in current_data_example_visits; use the buffer trace for an exact "
            "sample-level replay count")
    return {
        "kind": "declared-schedule accounting",
        "current_data_example_visits": int(current_examples),
        "optimizer_steps": int(optimizer_steps),
        "optimizer_steps_note": step_note,
        "schedule": schedule,
        "replay_accounting_note": replay_note,
        "flops_measured": False,
    }






def _wrap_head_mode(builder, head_mode, num_tasks, classes_per_task):
    """Return a builder honouring ``head_mode`` for shared-backbone baselines.

    ``"shared"`` reproduces the legacy protocol (one reused C-way head).
    ``"multi"`` gives every task its own head, which is the standard task-IL
    convention and the only one comparable with the literature.

    This must only be applied to the methods in :data:`MULTI_HEAD_ELIGIBLE`;
    see :func:`builder_for_method`.
    """
    if head_mode == "shared":
        return builder
    if head_mode != "multi":
        raise ValueError(f"unknown head_mode: {head_mode}")
    return lambda: MultiHeadModel(builder(), num_tasks, classes_per_task)


def builder_for_method(method, builders):
    """Pick the model builder appropriate for ``method``.

    ``builders`` is the ``(base, multi_head)`` pair produced by
    :func:`_benchmark_setup`. Isolation methods and Independent always receive
    the plain single-head backbone: they own per-task parameters already, and
    handing them a multi-head model would expose every head to the masking and
    pruning machinery.
    """
    base, multi_head = builders
    return multi_head if method in MULTI_HEAD_ELIGIBLE else base


def _benchmark_setup(benchmark, cfg, seed, head_mode="shared",
                     execution=None):
    """Build loaders, a model builder pair and task count for one benchmark/seed.

    Returns:
        ``(train_loaders, test_loaders, (base_builder, multi_head_builder),
        num_tasks)``.
    """
    lay = bench.layout(benchmark, cfg)
    train_loaders, test_loaders = bench.build_loaders(
        benchmark, cfg, seed, class_il=False, execution=execution)
    builder = bench.model_builder(benchmark, lay.classes_per_task)
    builders = (builder, _wrap_head_mode(builder, head_mode, lay.num_tasks,
                                         lay.classes_per_task))
    return train_loaders, test_loaders, builders, lay.num_tasks


def _diltcl_densities(diltcl):
    """Measured global and eligible-region mask densities for DiLT-CL."""
    core = masking.mask_density(diltcl.core_mask) if diltcl.core_mask else 0.0
    complement = masking.complement_mask(diltcl.core_mask)
    expert_items = sorted(diltcl.expert_masks.items())
    experts = [masking.mask_density(m) for _, m in expert_items]
    expert_region = [
        masking.mask_density_in_region(m, complement) for _, m in expert_items
    ]
    expert = float(np.mean(experts)) if experts else 0.0
    return {
        "diltcl_requested_target_sparsity": float(diltcl.target_sparsity),
        "diltcl_pruning_rounds": int(diltcl.pruning_rounds),
        "diltcl_per_round_pruning_rate": float(
            masking.iterative_pruning_rate(
                diltcl.target_sparsity, diltcl.pruning_rounds)),
        "diltcl_core_partition_global_density": [
            float(value) for value in diltcl.core_partition_densities
        ],
        "diltcl_core_density": core,
        "diltcl_expert_density": expert,
        "diltcl_expert_global_density_by_task": {
            str(task): float(value)
            for (task, _), value in zip(expert_items, experts)
        },
        "diltcl_expert_eligible_density_by_task": {
            str(task): float(value)
            for (task, _), value in zip(expert_items, expert_region)
        },
    }






def _serialized_state_record(payload, state_definition):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    record = {
        "state_definition": state_definition,
        "format": payload.get("format", "pytorch-state-v1"),
        "serialized_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": _state_content_sha256(payload),
        "deployable_compact_export": False,
    }
    return record, raw


def _diltcl_checkpoint_record(diltcl):
    """Serialize the actual resume state to bytes and return size + SHA-256.

    This is deliberately labelled a dense PyTorch resume checkpoint. It is not
    a compact deployment exporter and therefore must not be presented as such.
    """
    payload = {
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
    return _serialized_state_record(
        payload, "resume-learning dense PyTorch state")


def _core_only_matrix(diltcl, test_loaders, num_tasks, device):
    """Evaluate the frozen core on every task without retraining.

    The core is unchanged after task 0, so the task-wise accuracies define a
    lower-triangular column-constant matrix. This is a matched-checkpoint
    ablation of ``core`` versus ``core + task expert``.
    """
    accuracies = [
        evaluate_with_mask(diltcl, loader, 0, device)
        for loader in test_loaders
    ]
    return np.tril(np.broadcast_to(accuracies, (num_tasks, num_tasks))).copy()


def _run_method(method, cfg, train_loaders, test_loaders, builder, num_tasks,
                device, execution=None):
    """Run a single method once.

    Returns:
        ``(result_matrix, metrics, extra)`` where ``extra`` is a dict of
        method-specific extras (e.g. DiLT-CL's measured densities) or empty.
    """
    execution = execution or ExecutionOptions()
    base_epochs = cfg.epochs_replay  # shared budget for the simple baselines
    extra = {}
    state_payload = None
    state_metadata = {}

    if method == "diltcl":
        M, _, _, diltcl = train_diltcl(
            builder(), train_loaders, test_loaders, num_tasks, device=device,
            num_partitions=cfg.diltcl_partitions,
            pruning_rounds=cfg.pruning_rounds,
            target_sparsity=cfg.diltcl_sparsity,
            full_train_epochs=cfg.diltcl_full_epochs,
            imp_epochs=cfg.diltcl_imp_epochs,
            expert_finetune_epochs=cfg.diltcl_expert_finetune_epochs,
            lr=cfg.lr, return_model=True,
        )
        extra = _diltcl_densities(diltcl)
        core_only = _core_only_matrix(
            diltcl, test_loaders, num_tasks, device)
        extra["core_only_matrix"] = core_only
        extra["core_only_metrics"] = summarize_run(core_only)
        extra["checkpoint"], extra["_checkpoint_bytes"] = \
            _diltcl_checkpoint_record(diltcl)
    elif method == "ewc":
        M, _, _, state = train_ewc(
            builder(), train_loaders, test_loaders, num_tasks,
            device=device, lambda_ewc=cfg.ewc_lambda,
            epochs=cfg.epochs_ewc, lr=cfg.lr, return_state=True,
            fisher_labels=cfg.ewc_fisher_labels, **execution.ewc_kwargs())
        state_payload = {"model": _cpu_tensor_dict(state["model"].state_dict())}
    elif method == "lwf":
        M, _, _, state = train_lwf(
            builder(), train_loaders, test_loaders, num_tasks,
            device=device, alpha=cfg.lwf_alpha,
            temperature=cfg.lwf_temperature,
            epochs=cfg.epochs_lwf, lr=cfg.lr, return_state=True,
            **execution.lwf_kwargs())
        state_payload = {"model": _cpu_tensor_dict(state["model"].state_dict())}
    elif method == "finetune":
        M, _, _, state = train_finetune(
            builder, train_loaders, test_loaders, num_tasks, device=device,
            epochs=base_epochs, lr=cfg.lr, return_state=True)
        state_payload = {"model": _cpu_tensor_dict(state["model"].state_dict())}
    elif method == "independent":
        M, _, _, state = train_independent(
            builder, train_loaders, test_loaders, num_tasks, device=device,
            epochs=base_epochs, lr=cfg.lr, return_state=True)
        state_payload = {"models": state["model_states"]}
    elif method == "er":
        M, _, _, state = train_er(
            builder, train_loaders, test_loaders, num_tasks,
            device=device, epochs=base_epochs, lr=cfg.lr,
            buffer_size=cfg.buffer_size,
            replay_batch_size=cfg.replay_batch_size, return_state=True,
            **execution.replay_kwargs())
        state_payload = {"model": _cpu_tensor_dict(state["model"].state_dict())}
        buffer = state["buffer"]
        state_metadata = {
            "replay_sample_calls": int(buffer.sample_calls),
            "replay_examples_sampled": int(buffer.sampled_items_total),
            "buffer_examples_retained": int(len(buffer)),
            "stream_examples_seen": int(buffer.num_seen),
        }
    elif method == "derpp":
        M, _, _, state = train_derpp(
            builder, train_loaders, test_loaders, num_tasks,
            device=device, epochs=base_epochs, lr=cfg.lr,
            buffer_size=cfg.buffer_size,
            replay_batch_size=cfg.replay_batch_size,
            alpha=cfg.derpp_alpha, beta=cfg.derpp_beta, return_state=True,
            **execution.replay_kwargs())
        state_payload = {"model": _cpu_tensor_dict(state["model"].state_dict())}
        buffer = state["buffer"]
        state_metadata = {
            "replay_sample_calls": int(buffer.sample_calls),
            "replay_examples_sampled": int(buffer.sampled_items_total),
            "buffer_examples_retained": int(len(buffer)),
            "stream_examples_seen": int(buffer.num_seen),
        }
    elif method == "packnet":
        M, _, _, state = train_packnet(
            builder, train_loaders, test_loaders, num_tasks, device=device,
            epochs=base_epochs, lr=cfg.lr,
            retrain_epochs=cfg.packnet_retrain_epochs,
            keep_fraction=(cfg.packnet_keep_fraction or None),
            return_state=True)
        state_payload = {
            "model": _cpu_tensor_dict(state["model"].state_dict()),
            "eval_masks": [_cpu_tensor_dict(mask)
                           for mask in state["eval_masks"]],
        }
    elif method == "supsup":
        M, _, _, state = train_supsup(
            builder, train_loaders, test_loaders, num_tasks, device=device,
            epochs=base_epochs, lr=cfg.lr, sparsity=cfg.supsup_sparsity,
            return_state=True)
        state_payload = {"model": _cpu_tensor_dict(state["model"].state_dict())}
    elif method == "wsn":
        M, _, _, state = train_wsn(
            builder, train_loaders, test_loaders, num_tasks, device=device,
            epochs=base_epochs, lr=cfg.lr, sparsity=cfg.wsn_sparsity,
            return_state=True)
        state_payload = {"model": _cpu_tensor_dict(state["model"].state_dict())}
    else:
        raise ValueError(f"Unknown method: {method}")

    if state_payload is not None:
        state_payload["format"] = f"{method}-task-il-evaluated-state-v1"
        extra["checkpoint"], extra["_checkpoint_bytes"] = _serialized_state_record(
            state_payload, "task-IL evaluated inference state")
        extra["checkpoint"]["method_state_metadata"] = state_metadata
    return M, summarize_run(M), extra


def _backbone_params(benchmark, cfg, head_mode="shared", num_tasks=None):
    """Trainable-parameter count of the benchmark's backbone (incl. biases).

    Under ``head_mode="multi"`` the shared-backbone baselines carry one head
    per task, so their real parameter count is larger than the single-head
    backbone. Reporting the single-head figure would understate the baselines'
    storage and flatter the isolation methods by comparison.
    """
    lay = bench.layout(benchmark, cfg)
    model = bench.model_builder(benchmark, lay.classes_per_task)()
    if head_mode == "multi" and num_tasks:
        model = MultiHeadModel(model, num_tasks, lay.classes_per_task)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


ISOLATION_LIKE_METHODS = ("packnet", "supsup", "wsn", "diltcl", "independent")
_COST_NAME_TO_METHOD = {"finetune": "finetune", "ewc": "ewc", "lwf": "lwf", "er": "er",
                        "der++": "derpp", "packnet": "packnet", "supsup": "supsup",
                        "wsn": "wsn", "dilt-cl (ours)": "diltcl", "independent": "independent"}


def _cost_method_key(cost):
    return _COST_NAME_TO_METHOD.get(str(cost.name).lower(), str(cost.name).lower())


def _memory_report(cfg, benchmark, methods, densities, head_mode="shared"):
    """Build and return an analytic inference-state estimate for a benchmark.

    ``densities`` is the measured ``(core, expert)`` density pair for DiLT-CL
    (falls back to ``None`` -> config-free defaults handled by the caller).
    """
    lay = bench.layout(benchmark, cfg)
    num_tasks = lay.num_tasks
    # The shared-backbone baselines carry one head per task under multi-head,
    # so their parameter count must reflect the protocol actually used; the
    # isolation methods and Independent always use the plain single-head
    # backbone (before v3.0.4 the multi-head count was applied to every row).
    params = _backbone_params(benchmark, cfg, head_mode=head_mode,
                              num_tasks=num_tasks)
    single_head_params = _backbone_params(benchmark, cfg, head_mode="shared",
                                          num_tasks=num_tasks)
    core_d, expert_d = densities if densities else (0.30, 0.23)

    shared = [m for m in methods if m not in ISOLATION_LIKE_METHODS]
    isolated = [m for m in methods if m in ISOLATION_LIKE_METHODS]
    costs = []
    if shared:
        costs += build_default_suite(
            shared, params, num_tasks, buffer_size=cfg.buffer_size,
            sample_numel=lay.sample_numel, num_classes=lay.classes_per_task,
            core_density=core_d, expert_density=expert_d)
    if isolated:
        costs += build_default_suite(
            isolated, single_head_params, num_tasks, buffer_size=cfg.buffer_size,
            sample_numel=lay.sample_numel, num_classes=lay.classes_per_task,
            core_density=core_d, expert_density=expert_d)
    order = {m: i for i, m in enumerate(methods)}
    costs.sort(key=lambda c: order.get(_cost_method_key(c), len(order)))

    print("\n" + "=" * 70)
    print(f"ANALYTIC COMPACT INFERENCE-STATE ESTIMATE: {benchmark.upper()}  "
          f"(T={num_tasks}, P={params:,} shared-backbone / {single_head_params:,} isolation"
          + (f", measured DiLT-CL core={core_d:.2f} expert={expert_d:.2f})"
             if densities else ", DiLT-CL densities=defaults)"))
    print("=" * 70)
    print(format_cost_table_text(costs, params))

    return [{"name": c.name, "float_params": c.float_params,
             "mask_bits": c.mask_bits, "buffer_floats": c.buffer_floats,
             "megabytes": c.megabytes(), "note": c.note} for c in costs]


def _validate_run_arguments(cfg, benchmarks, methods, seeds, head_mode):
    """Validate every argument before any device or dataset side effect."""
    if head_mode not in ("shared", "multi"):
        raise ValueError(
            f"unknown head_mode: {head_mode!r} (expected 'shared' or 'multi')")
    # Pure argument validation first: it needs no environment and must not be
    # preceded by side effects such as the preflight gate or device setup.
    cfg.validate()
    if not benchmarks:
        raise ValueError("at least one benchmark is required")
    if not methods:
        raise ValueError("at least one method is required")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must be unique")
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")


def _aggregate_and_report(results, benchmarks, methods, seeds):
    """Aggregate per-seed metrics and print the comparison tables."""
    aggregated = {b: {} for b in benchmarks}
    for benchmark in benchmarks:
        print("\n" + "=" * 70)
        print(f"RESULTS: {benchmark.upper()}  (mean +/- std over "
              f"{len(seeds)} seeds)")
        print("=" * 70)
        for method in methods:
            aggregated[benchmark][method] = aggregate_seeds(
                results[benchmark][method])

        ref = ("diltcl" if "diltcl" in methods and len(seeds) >= 5
               else None)
        if "diltcl" in methods and len(seeds) < 5:
            print("Significance output suppressed: fewer than five aligned seeds.")
        for metric in ["ACC", "BWT", "Forgetting"]:
            print()
            print(format_comparison_table(
                aggregated[benchmark], reference=ref, metric=metric,
                order=methods))
    return aggregated


def _serialize_results(*, out_prefix, cfg, benchmarks, methods, seeds,
                       results, aggregated, matrices, run_records,
                       benchmark_metadata, memory, device, head_mode,
                       checkpoint_dir, diltcl_density, execution=None):
    """Assemble the artifact payload and write the JSON/NPZ pair."""
    out_parent = os.path.dirname(os.path.abspath(out_prefix))
    os.makedirs(out_parent, exist_ok=True)
    json_payload = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "protocol": "task_il_shared_local_label_axis",
        "evidence_status": "unverified",
        "verification_status": "unverified",
        "task_il_head_mode": head_mode,
        "task_il_head_disclosure": {
            "isolation_and_independent": "oracle task-specific predictor",
            "shared_backbone_baselines": (
                "one separate head per task (standard multi-head task-IL)"
                if head_mode == "multi" else
                "one reused C-way head over task-local labels; this is not a "
                "standard separate-head multi-head implementation"),
            "publication_warning": (
                "Standard multi-head protocol; baselines are comparable with "
                "the task-IL literature."
                if head_mode == "multi" else
                "Do not claim literature-comparable multi-head Task-IL "
                "baseline rankings from this artifact without a separate-head rerun."),
        },
        "config": cfg.__dict__,
        "execution": (execution or ExecutionOptions()).provenance(),
        "seeds": seeds,
        "benchmarks": benchmarks,
        "methods": methods,
        "checkpoint_persistence_enabled": checkpoint_dir is not None,
        "environment": _environment_metadata(device),
        "benchmark_metadata": benchmark_metadata,
        "per_seed": results,
        "aggregated": aggregated,
        "per_seed_run_records": run_records,
        "memory": memory,
        "artifact_reportability": {
            "postrun_validation_completed": False,
            "current_publication_reportable": False,
            "unmet_at_write_time": [
                *([] if checkpoint_dir is not None else [
                    "persisted evaluated-state checkpoint files"]),
                "independent JSON/NPZ recomputation report",
                "dataset content hashes and complete source hashes",
                "standard separate-head Task-IL shared-baseline rerun",
                "measured FLOPs or hardware-counter compute",
                "compact exporter/importer logit-equivalence test",
                "failure/exclusion record from the suite launcher",
            ],
        },
    }
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(json_payload, f, indent=2, allow_nan=False)
    np.savez(f"{out_prefix}.npz", **matrices)
    print(f"\nSaved per-seed + aggregated metrics to '{out_prefix}.json'")
    print(f"Saved raw result matrices to '{out_prefix}.npz'")
    print("\nDone.")


def run(cfg, benchmarks, methods, seeds, out_prefix="task_il_results",
        profile=True, checkpoint_dir=None, head_mode="shared",
        execution=None):
    """Run the full comparison and write JSON / NPZ outputs.

    Args:
        cfg: An :class:`ExperimentConfig`.
        benchmarks: List of benchmark names (``'cifar'`` / ``'pmnist'``).
        methods: List of method names from :data:`ALL_METHODS`.
        seeds: List of integer seeds.
        out_prefix: Output filename prefix.
        profile: If ``True``, also emit an analytic compact-storage table per
            benchmark (uses DiLT-CL's measured densities when available).
        execution: :class:`diltcl.execution.ExecutionOptions` (data backend,
            numerics mode, threads, ...). Defaults to the fast device backend.
    """
    _validate_run_arguments(cfg, benchmarks, methods, seeds, head_mode)
    execution = execution or ExecutionOptions()

    # Then verify datasets/dependencies, before spending any GPU time.
    preflight.enforce(
        preflight.resolve_data_root(), benchmarks,
        require_training=True, header="run_task_il preflight")

    device = get_device()
    execution.apply()
    print(f"Using device: {device}")
    print(f"Execution:  backend={execution.data_backend} "
          f"numerics={execution.numerics} tf32={execution.tf32} "
          f"threads={torch.get_num_threads()}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Methods:    {methods}")
    print(f"Seeds:      {seeds}")

    # results[benchmark][method] = list of per-seed metric dicts
    results = {b: {m: [] for m in methods} for b in benchmarks}
    matrices = {}  # f"{benchmark}/{method}/seed{seed}" -> result_matrix
    # Store every seed rather than silently profiling only the first one.
    diltcl_density = {b: [] for b in benchmarks}
    run_records = {b: {m: {} for m in methods} for b in benchmarks}
    benchmark_metadata = {b: {"per_seed": {}} for b in benchmarks}

    for benchmark in benchmarks:
        for seed in seeds:
            set_seed(seed)
            train_loaders, test_loaders, builders, num_tasks = _benchmark_setup(
                benchmark, cfg, seed, head_mode=head_mode, execution=execution)
            benchmark_metadata[benchmark]["per_seed"][str(seed)] = \
                _loader_metadata(train_loaders, test_loaders)

            for method in methods:
                print("\n" + "#" * 70)
                print(f"# {benchmark.upper()} | {method.upper()} | seed={seed}")
                print("#" * 70)
                set_seed(seed)  # re-seed so training is comparable across runs
                baseline_bytes = None
                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                    # Memory already resident before the method starts (with
                    # the device data backend this is mostly the dataset), so
                    # the peak below can be read net of it.
                    baseline_bytes = int(torch.cuda.memory_allocated(device))
                _cuda_sync(device)
                started = time.perf_counter()
                M, metrics, extra = _run_method(
                    method, cfg, train_loaders, test_loaders,
                    builder_for_method(method, builders),
                    num_tasks, device, execution=execution)
                _cuda_sync(device)
                wall_seconds = time.perf_counter() - started
                peak_bytes = None
                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    peak_bytes = int(torch.cuda.max_memory_allocated(device))
                results[benchmark][method].append(metrics)
                matrices[f"{benchmark}/{method}/seed{seed}"] = np.asarray(M)
                core_only = extra.pop("core_only_matrix", None)
                checkpoint_bytes = extra.pop("_checkpoint_bytes", None)
                checkpoint_record = extra.get("checkpoint")
                if checkpoint_record is not None:
                    checkpoint_record["persisted"] = False
                    checkpoint_record["persisted_path"] = None
                    checkpoint_record["persisted_relative_path"] = None
                if checkpoint_bytes is not None and checkpoint_dir is not None:
                    checkpoint_path = _persist_checkpoint(
                        checkpoint_bytes, checkpoint_dir, benchmark, method, seed)
                    checkpoint_record["persisted"] = True
                    checkpoint_record["persisted_path"] = checkpoint_path
                    checkpoint_record["persisted_relative_path"] = os.path.relpath(
                        checkpoint_path,
                        os.path.dirname(os.path.abspath(out_prefix)))
                checkpoint_bytes = None
                if core_only is not None:
                    matrices[f"{benchmark}/{method}/seed{seed}/core_only"] = \
                        np.asarray(core_only)
                schedule_accounting = _schedule_accounting(
                    method, cfg, train_loaders)
                replay_count = (extra.get("checkpoint", {})
                                .get("method_state_metadata", {})
                                .get("replay_examples_sampled"))
                if replay_count is not None:
                    schedule_accounting["replay_examples_sampled"] = int(
                        replay_count)
                    schedule_accounting["total_training_example_presentations"] = (
                        schedule_accounting["current_data_example_visits"]
                        + int(replay_count))
                    schedule_accounting["replay_accounting_note"] = (
                        "exact count recorded by ReservoirBuffer.sample")
                run_records[benchmark][method][str(seed)] = {
                    "pipeline_wall_seconds": float(wall_seconds),
                    "peak_cuda_memory_allocated_bytes": peak_bytes,
                    "cuda_memory_allocated_before_method_bytes": baseline_bytes,
                    "schedule_accounting": schedule_accounting,
                    "method_extras": extra,
                }
                if "diltcl_core_density" in extra:
                    diltcl_density[benchmark].append((
                        extra["diltcl_core_density"],
                        extra["diltcl_expert_density"],
                    ))

    # ---- Aggregate and print tables ----
    aggregated = _aggregate_and_report(results, benchmarks, methods,
                                       seeds)

    # ---- Deployment-storage (parameter / memory) report ----
    memory = {}
    if profile:
        for benchmark in benchmarks:
            density_pairs = diltcl_density[benchmark]
            mean_density = None
            if density_pairs:
                mean_density = tuple(np.mean(
                    np.asarray(density_pairs, dtype=float), axis=0))
            memory[benchmark] = _memory_report(
                cfg, benchmark, methods, mean_density, head_mode=head_mode)

    # ---- Persist ----
    _serialize_results(
        out_prefix=out_prefix, cfg=cfg, benchmarks=benchmarks,
        methods=methods, seeds=seeds, results=results,
        aggregated=aggregated, matrices=matrices,
        run_records=run_records,
        benchmark_metadata=benchmark_metadata, memory=memory,
        device=device, head_mode=head_mode,
        checkpoint_dir=checkpoint_dir,
        diltcl_density=diltcl_density, execution=execution)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed Task-IL comparison (unverified raw artifact).")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced tasks/epochs for a quick run.")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (default 3). Seeds are 0..N-1.")
    parser.add_argument(
        "--seed-list", type=str, default=None,
        help="Explicit comma/range seed IDs, e.g. 0,2,4-7. Overrides --seeds.")
    parser.add_argument("--benchmark",
                        choices=["cifar", "pmnist", "tinyimagenet",
                                 "fivedatasets", "both"],
                        default="both", help="Which benchmark(s) to run. "
                        "'both' = cifar+pmnist; harder benchmarks are opt-in.")
    parser.add_argument("--head-mode", choices=("shared", "multi"),
                        default="shared",
                        help="Task-IL head protocol for the shared-backbone "
                             "baselines. 'multi' gives each task its own head "
                             "(standard, literature-comparable); 'shared' "
                             "reproduces the legacy single reused head.")
    parser.add_argument("--methods", type=str, default=",".join(ALL_METHODS),
                        help="Comma-separated subset of: "
                             + ",".join(ALL_METHODS))
    parser.add_argument("--out", type=str, default="task_il_results",
                        help="Output filename prefix.")
    parser.add_argument("--no-profile", action="store_true",
                        help="Skip the deployment-storage (memory) table.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Dataset directory (overrides $DILTCL_DATA_ROOT). "
                             "An existing local copy is used without network.")
    parser.add_argument("--no-download", action="store_true",
                        help="Never attempt to download datasets; require a "
                             "local copy under the data root.")
    parser.add_argument(
        "--config-json", type=str, default=None,
        help="Flat JSON object overriding ExperimentConfig fields. Unknown "
             "keys and invalid ranges are rejected.")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Persist the exact hashed PyTorch state for every method/seed. "
             "Strongly recommended for publication runs.")
    add_execution_arguments(parser)
    return parser.parse_args()






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
        head_mode=args.head_mode, profile=not args.no_profile,
        checkpoint_dir=args.checkpoint_dir, execution=execution)


if __name__ == "__main__":
    main()
