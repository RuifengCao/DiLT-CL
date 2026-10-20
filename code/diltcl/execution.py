"""Execution options: how a run uses the hardware, separated from *what* it runs.

:class:`diltcl.config.ExperimentConfig` holds the scientific knobs (epochs,
sparsity, benchmark sizes). :class:`ExecutionOptions` holds the engineering
knobs that decide how fast the same experiment runs and whether its arithmetic
is reproduced bit for bit:

``data_backend``
    ``"device"`` (default) keeps every benchmark as a ``uint8`` tensor on the
    GPU; ``"loader"`` is the legacy ``DataLoader`` pipeline. The two are
    bit-identical (see :mod:`diltcl.data_backend`).
``numerics``
    ``"fast"`` (default) selects the algebraically equivalent fast paths
    (shared-feature LwF distillation, consolidated EWC penalty, routed replay
    for ER/DER++); ``"legacy"`` selects the v2 arithmetic, which reproduces v2
    result files bit for bit given the same PyTorch build and hardware.
``eval_batch_size``
    Batch size for evaluation loaders. ``None`` keeps the training batch size
    (the legacy behaviour). Larger values only speed up evaluation but can
    change cuDNN's algorithm choice, so they count as a numerics change and are
    recorded.
``tf32``
    ``"default"`` leaves PyTorch's flags alone (cuDNN convolutions *do* use
    TF32 by default on Ampere and newer GPUs, matmuls do not -- this is what
    every previous run used); ``"on"`` / ``"off"`` set both flags explicitly.
``threads``
    ``torch.set_num_threads`` for this process. The suite sets it per job so
    eight parallel jobs do not oversubscribe 25 vCPUs.

Every option is written to the result JSON under ``"execution"`` so a table can
never silently mix runs produced under different settings.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass

DATA_BACKENDS = ("device", "loader")
NUMERICS_MODES = ("fast", "legacy")
TF32_MODES = ("default", "on", "off")


@dataclass(frozen=True)
class ExecutionOptions:
    data_backend: str = "device"
    numerics: str = "fast"
    eval_batch_size: int | None = None
    num_workers: int | None = None
    threads: int | None = None
    tf32: str = "default"

    def __post_init__(self):
        if self.data_backend not in DATA_BACKENDS:
            raise ValueError(f"data_backend must be one of {DATA_BACKENDS}")
        if self.numerics not in NUMERICS_MODES:
            raise ValueError(f"numerics must be one of {NUMERICS_MODES}")
        if self.tf32 not in TF32_MODES:
            raise ValueError(f"tf32 must be one of {TF32_MODES}")
        if self.eval_batch_size is not None and int(self.eval_batch_size) <= 0:
            raise ValueError("eval_batch_size must be positive")
        if self.num_workers is not None and int(self.num_workers) < 0:
            raise ValueError("num_workers must be non-negative")
        if self.threads is not None and int(self.threads) <= 0:
            raise ValueError("threads must be positive")

    # -- kwargs for the library ----------------------------------------------

    @property
    def fast(self):
        return self.numerics == "fast"

    def loader_kwargs(self):
        """Keyword arguments for the ``get_*`` benchmark loaders."""
        kwargs = {"backend": self.data_backend,
                  "eval_batch_size": self.eval_batch_size}
        if self.num_workers is not None:
            kwargs["num_workers"] = self.num_workers
        return kwargs

    def lwf_kwargs(self):
        return {"shared_features": self.fast}

    def ewc_kwargs(self):
        return {"penalty_form": "consolidated" if self.fast else "per_task"}

    def replay_kwargs(self):
        return {"routed_replay": self.fast}

    # -- process-level effects -------------------------------------------------

    def apply(self):
        """Apply thread and TF32 settings to this process; return provenance."""
        import torch

        if self.threads is not None:
            torch.set_num_threads(int(self.threads))
        if self.tf32 == "on":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif self.tf32 == "off":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        return self.provenance()

    def provenance(self):
        """JSON-safe record of the options *and* the effective torch flags."""
        record = {"options": asdict(self)}
        try:
            import torch
            record["effective"] = {
                "torch_num_threads": int(torch.get_num_threads()),
                "cuda_matmul_allow_tf32": bool(
                    torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            }
        except ImportError:  # pragma: no cover - torch-free environment
            record["effective"] = None
        record["numerics_note"] = (
            "fast: LwF shared-feature distillation, consolidated EWC penalty, "
            "routed multi-head replay (same objective, different float "
            "summation order); legacy: v2 arithmetic."
            if self.fast else
            "legacy: v2 per-head LwF forwards, per-task EWC sum, per-group "
            "replay forwards.")
        return record


# ---------------------------------------------------------------------------
# Command-line plumbing shared by the runners
# ---------------------------------------------------------------------------

def add_execution_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        "execution", "Hardware-utilisation options (recorded in the artifact)")
    group.add_argument("--data-backend", choices=DATA_BACKENDS, default=None,
                       help="'device' keeps datasets on the GPU (default; "
                            "bit-identical to 'loader'); 'loader' is the "
                            "legacy DataLoader pipeline. Env: DILTCL_DATA_BACKEND.")
    group.add_argument("--numerics", choices=NUMERICS_MODES, default=None,
                       help="'fast' (default) uses the algebraically "
                            "equivalent fast paths; 'legacy' reproduces the "
                            "v2 arithmetic bit for bit. Env: DILTCL_NUMERICS.")
    group.add_argument("--eval-batch-size", type=int, default=None,
                       help="Evaluation batch size (default: the training "
                            "batch size). Recorded as a numerics-affecting option.")
    group.add_argument("--num-workers", type=int, default=None,
                       help="DataLoader workers for the legacy backend "
                            "(default: DILTCL_NUM_WORKERS or 2).")
    group.add_argument("--threads", type=int, default=None,
                       help="torch intra-op threads for this process "
                            "(default: DILTCL_NUM_THREADS or torch's default).")
    group.add_argument("--tf32", choices=TF32_MODES, default=None,
                       help="TF32 policy: 'default' keeps PyTorch defaults "
                            "(what earlier runs used), 'on'/'off' set both "
                            "cuDNN and matmul flags. Env: DILTCL_TF32.")
    return parser


def _env(name, default=None):
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def execution_from_args(args) -> ExecutionOptions:
    """Build :class:`ExecutionOptions` from parsed CLI args and environment."""
    threads = args.threads if args.threads is not None else _env("DILTCL_NUM_THREADS")
    return ExecutionOptions(
        data_backend=(args.data_backend or _env("DILTCL_DATA_BACKEND", "device")).lower(),
        numerics=(args.numerics or _env("DILTCL_NUMERICS", "fast")).lower(),
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        threads=None if threads is None else int(threads),
        tf32=(args.tf32 or _env("DILTCL_TF32", "default")).lower(),
    )


def execution_command_args(options: ExecutionOptions):
    """CLI fragments that reproduce ``options`` in a child runner process."""
    fragments = ["--data-backend", options.data_backend,
                 "--numerics", options.numerics, "--tf32", options.tf32]
    if options.eval_batch_size is not None:
        fragments += ["--eval-batch-size", str(options.eval_batch_size)]
    if options.num_workers is not None:
        fragments += ["--num-workers", str(options.num_workers)]
    if options.threads is not None:
        fragments += ["--threads", str(options.threads)]
    return fragments
