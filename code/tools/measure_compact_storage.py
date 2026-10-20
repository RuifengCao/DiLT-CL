#!/usr/bin/env python3
"""Measure the compact DiLT-CL deployment size from a saved checkpoint.

The paper's storage claim needs a *measured* number, not the analytic estimate.
This tool loads the dense resume checkpoint written by the runners, exports it
in the compact format (bit-packed masks, mask-selected weights only), verifies
that the compact state reproduces the dense model's logits exactly, and reports
both sizes.

Only a verified export may be reported as deployable storage.

Example::

    python3 tools/measure_compact_storage.py \\
        --checkpoint runs/p0_multi/artifacts/p0_task_il_multi__cifar__seed0/\\
checkpoints/cifar/diltcl/seed0.pt \\
        --benchmark cifar
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from diltcl import compact_export as ce  # noqa: E402
from diltcl.diltcl import DiLTCL  # noqa: E402
from diltcl.models import AdaptiveCNN, SimpleCNN, SimpleMLP  # noqa: E402

_BACKBONES = {
    "cifar": lambda: SimpleCNN(num_classes=5),
    "pmnist": lambda: SimpleMLP(input_size=784, hidden_size=256, num_classes=10),
    "tinyimagenet": lambda: AdaptiveCNN(num_classes=20),
    "fivedatasets": lambda: AdaptiveCNN(num_classes=10),
}

_SAMPLE_SHAPE = {
    "cifar": (3, 32, 32),
    "pmnist": (784,),
    "tinyimagenet": (3, 64, 64),
    "fivedatasets": (3, 32, 32),
}


def rebuild(checkpoint_path, benchmark, device="cpu"):
    """Reconstruct a DiLT-CL instance from a dense resume checkpoint."""
    # Tensors and plain containers only: no pickled code is executed.
    payload = torch.load(checkpoint_path, map_location=device,
                         weights_only=True)
    if payload.get("format") != "diltcl-dense-resume-state-v1":
        raise ValueError(
            f"unexpected checkpoint format: {payload.get('format')!r}; "
            "this tool reads the DiLT-CL dense resume state")

    model = _BACKBONES[benchmark]().to(device)
    model.load_state_dict(payload["model"])

    learner = DiLTCL(model, device=device)
    learner.core_mask = {k: v.to(device)
                         for k, v in payload["core_mask"].items()}
    learner.frozen_core_values = {k: v.to(device) for k, v
                                  in payload["frozen_core_values"].items()}
    learner.initial_params = {k: v.to(device) for k, v
                              in payload["initial_params"].items()}
    learner.expert_masks = {
        int(task): {k: v.to(device) for k, v in mask.items()}
        for task, mask in payload["expert_masks"].items()}
    learner.expert_weights = {
        int(task): {k: v.to(device) for k, v in weights.items()}
        for task, weights in payload["expert_weights"].items()}
    learner.core_partition_densities = list(
        payload.get("core_partition_densities", []))
    return learner


def main():
    parser = argparse.ArgumentParser(
        description="Measure verified compact DiLT-CL storage.")
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path,
                        help="Path to .../checkpoints/<benchmark>/diltcl/"
                             "seed<N>.pt")
    parser.add_argument("--benchmark", required=True,
                        choices=sorted(_BACKBONES),
                        help="Benchmark the checkpoint came from.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for the logit-equivalence check.")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    learner = rebuild(args.checkpoint, args.benchmark)
    compact = ce.export_diltcl_compact(learner)

    sample = torch.randn(args.batch_size, *_SAMPLE_SHAPE[args.benchmark])
    tasks = sorted(compact["experts"]) or [0]
    report = ce.verify_roundtrip_logits(learner, compact, sample, tasks=tasks)

    dense = ce.dense_size_bytes(learner) / 2 ** 20
    small = ce.compact_size_bytes(compact) / 2 ** 20
    on_disk = args.checkpoint.stat().st_size / 2 ** 20

    print(f"checkpoint            : {args.checkpoint}")
    print(f"benchmark             : {args.benchmark}")
    print(f"tasks with experts    : {len(compact['experts'])}")
    print()
    print(f"dense resume file     : {on_disk:8.2f} MiB   (what the runner wrote)")
    print(f"dense in-memory state : {dense:8.2f} MiB")
    print(f"compact export        : {small:8.2f} MiB   "
          f"({dense / small:.1f}x smaller than dense)")
    print()
    print(f"logit equivalence     : {'PASS' if report['ok'] else 'FAIL'} "
          f"(max abs diff {max(report['max_abs_diff_per_task'].values()):.3e})")
    if not report["ok"]:
        print("\nDo NOT report this compact size: the export does not reproduce "
              "the dense model's predictions.")
        return 1
    print("\nReportable: this compact size is backed by exact logit equivalence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
