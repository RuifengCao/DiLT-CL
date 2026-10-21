"""DiLT-CL: Disentangled Lottery-Ticket Continual Learning.

A small, readable research package for continual learning that implements the
DiLT-CL method alongside several baselines (EWC, LwF, Finetune, Independent,
ER, DER++), the Split-CIFAR-100 / Permuted-MNIST benchmarks, a
Gromov-Wasserstein subnetwork-similarity analysis, and a multi-seed evaluation
harness with significance testing.

Public API
----------
Models:      :class:`SimpleCNN`, :class:`SimpleMLP`
Methods:     :class:`DiLTCL`, :class:`EWC`, :class:`LwF`
Baselines:   :func:`train_finetune`, :func:`train_independent`,
             :func:`train_er`, :func:`train_derpp`,
             :func:`train_packnet`, :func:`train_supsup`
Data:        :func:`get_split_cifar100`, :func:`get_permuted_mnist`
Training:    :func:`train_diltcl`, :func:`train_ewc`, :func:`train_lwf`
Metrics:     :func:`evaluate`, :func:`evaluate_with_mask`,
             :func:`compute_acc_bwt`, :func:`compute_avg_forgetting`,
             :func:`compute_fwt`, :func:`learning_accuracy`,
             :func:`constant_lower_triangular`,
             :func:`summarize_run`, :func:`random_init_baseline`,
             :func:`print_result_matrix`
Stats:       :func:`aggregate_seeds`, :func:`paired_significance`,
             :func:`format_comparison_table`
Profiling:   :class:`MethodCost`, :func:`build_default_suite`,
             :func:`format_cost_table_text`, :func:`format_cost_table_latex`
Memory:      :class:`ReservoirBuffer`
Config:      :class:`ExperimentConfig` (science), :class:`ExecutionOptions`
             (hardware utilisation: data backend, numerics mode, threads)
Data backend: :mod:`data_backend` -- device-resident datasets, bit-identical
             to the legacy DataLoader pipeline
Utilities:   :func:`set_seed`, :func:`get_device`
GW analysis: :func:`extract_layer_representations`, :func:`compute_gw_distance`
"""

from . import benchmarks
from . import data_backend
from . import execution
from . import masking
from .baselines import (
    train_derpp,
    train_er,
    train_finetune,
    train_independent,
)
from .buffer import ReservoirBuffer
from .config import ExperimentConfig
from .execution import ExecutionOptions
from . import compact_export
from . import preflight
from . import runner_common
from .data import get_permuted_mnist, get_split_cifar100
from .hard_benchmarks import (
    FIVE_DATASETS_DEFAULT,
    get_five_datasets,
    get_split_tinyimagenet,
)
from .diltcl import DiLTCL
from .ewc import EWC
from .gromov_wasserstein import (
    compute_gw_distance,
    extract_layer_representations,
)
from .lwf import LwF
from .metrics import (
    set_model_task,  # noqa: F401 - public re-export
    constant_lower_triangular,
    compute_acc_bwt,
    compute_avg_forgetting,
    compute_fwt,
    evaluate,
    evaluate_class_il,
    evaluate_with_mask,
    learning_accuracy,
    print_result_matrix,
    random_init_baseline,
    summarize_run,
)
from .models import AdaptiveCNN, MultiHeadModel, SimpleCNN, SimpleMLP
from .profiling import (
    MethodCost,
    build_default_suite,
    format_cost_table_latex,
    format_cost_table_text,
)
from .stats import (
    aggregate_seeds,
    format_comparison_table,
    paired_significance,
)
from .subnetwork_baselines import (
    train_packnet,
    train_supsup,
    train_wsn,
)
from .task_inference import (
    collect_calibration_features,
    evaluate_class_il_inference,
    infer_task,
)
from .task_calibration import (
    deterministic_holdout_indices,
    fit_prefix_calibrators,
    fit_score_logreg,
    predict_score_logreg,
)
from .training import train_diltcl, train_ewc, train_lwf
from .utils import get_device, set_seed

__version__ = "3.0.10"

__all__ = [
    "ExperimentConfig",
    "ExecutionOptions",
    "benchmarks",
    "data_backend",
    "execution",
    "masking",
    "ReservoirBuffer",
    "get_permuted_mnist",
    "preflight",
    "compact_export",
    "runner_common",
    "get_split_cifar100",
    "get_split_tinyimagenet",
    "get_five_datasets",
    "FIVE_DATASETS_DEFAULT",
    "DiLTCL",
    "EWC",
    "LwF",
    "train_finetune",
    "train_independent",
    "train_er",
    "train_derpp",
    "train_packnet",
    "train_supsup",
    "train_wsn",
    "evaluate_class_il_inference",
    "collect_calibration_features",
    "infer_task",
    "fit_score_logreg",
    "deterministic_holdout_indices",
    "predict_score_logreg",
    "fit_prefix_calibrators",
    "MethodCost",
    "build_default_suite",
    "format_cost_table_text",
    "format_cost_table_latex",
    "compute_gw_distance",
    "extract_layer_representations",
    "compute_acc_bwt",
    "constant_lower_triangular",
    "compute_avg_forgetting",
    "compute_fwt",
    "evaluate",
    "evaluate_class_il",
    "evaluate_with_mask",
    "learning_accuracy",
    "print_result_matrix",
    "random_init_baseline",
    "summarize_run",
    "aggregate_seeds",
    "paired_significance",
    "format_comparison_table",
    "SimpleCNN",
    "SimpleMLP",
    "AdaptiveCNN",
    "MultiHeadModel",
    "train_diltcl",
    "train_ewc",
    "train_lwf",
    "get_device",
    "set_seed",
]
