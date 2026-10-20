"""Benchmark registry: one place that knows how each benchmark is built.

``run_task_il.py`` and ``run_class_il.py`` each carried their own
copies of the per-benchmark setup (loader construction, model choice, task
counts, label-space metadata), which had already diverged in small ways -- the
task-IL runner returned a bare builder instead of a ``(base, multi_head)`` pair
for 5-datasets. This module is the single source of truth both runners use.

Nothing here decides *what* to run; it only maps a benchmark name plus an
:class:`diltcl.config.ExperimentConfig` to loaders, a model builder and layout
facts.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import get_permuted_mnist, get_split_cifar100
from .execution import ExecutionOptions
from .hard_benchmarks import (
    FIVE_DATASETS_DEFAULT,
    get_five_datasets,
    get_split_tinyimagenet,
)
from .models import AdaptiveCNN, SimpleCNN, SimpleMLP

BENCHMARKS = ("cifar", "pmnist", "tinyimagenet", "fivedatasets")


@dataclass(frozen=True)
class BenchmarkLayout:
    """Static facts about a benchmark under a given configuration."""

    name: str
    display_name: str
    num_tasks: int
    classes_per_task: int
    global_classes: int
    disjoint_classes: bool
    label_relation: str
    task_names: tuple | None
    sample_numel: int          # scalars per stored sample (memory accounting)


def layout(benchmark, cfg) -> BenchmarkLayout:
    """Return the :class:`BenchmarkLayout` of ``benchmark`` under ``cfg``."""
    if benchmark == "cifar":
        return BenchmarkLayout(
            "cifar", "Split-CIFAR-100", cfg.c100_num_tasks,
            cfg.c100_classes_per_task,
            cfg.c100_num_tasks * cfg.c100_classes_per_task, True,
            "disjoint-class", None, 3 * 32 * 32)
    if benchmark == "pmnist":
        return BenchmarkLayout(
            "pmnist", "Permuted-MNIST", cfg.pm_num_tasks, 10, 10, False,
            "shared-label-domain-il", None, 28 * 28)
    if benchmark == "tinyimagenet":
        return BenchmarkLayout(
            "tinyimagenet", "Split-TinyImageNet", cfg.tin_num_tasks,
            cfg.tin_classes_per_task,
            cfg.tin_num_tasks * cfg.tin_classes_per_task, True,
            "disjoint-class", None, 3 * cfg.tin_image_size ** 2)
    if benchmark == "fivedatasets":
        names = tuple(cfg.five_dataset_names) or tuple(FIVE_DATASETS_DEFAULT)
        return BenchmarkLayout(
            "fivedatasets", "5-datasets", len(names), 10, 10 * len(names),
            True, "disjoint-dataset-label-blocks", names,
            3 * cfg.five_image_size ** 2)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def build_loaders(benchmark, cfg, seed, *, class_il,
                  execution: ExecutionOptions | None = None):
    """Per-task ``(train_loaders, test_loaders)`` for ``benchmark``."""
    execution = execution or ExecutionOptions()
    kwargs = execution.loader_kwargs()
    if benchmark == "cifar":
        return get_split_cifar100(
            num_tasks=cfg.c100_num_tasks,
            classes_per_task=cfg.c100_classes_per_task,
            batch_size=cfg.c100_batch_size, seed=seed, class_il=class_il,
            **kwargs)
    if benchmark == "pmnist":
        # Permuted-MNIST is domain-IL: the label space is shared, so the
        # class_il flag does not change the loaders.
        return get_permuted_mnist(
            num_tasks=cfg.pm_num_tasks, batch_size=cfg.pm_batch_size,
            seed=seed, **kwargs)
    if benchmark == "tinyimagenet":
        return get_split_tinyimagenet(
            num_tasks=cfg.tin_num_tasks,
            classes_per_task=cfg.tin_classes_per_task,
            batch_size=cfg.tin_batch_size, seed=seed,
            image_size=cfg.tin_image_size, class_il=class_il, **kwargs)
    if benchmark == "fivedatasets":
        names = list(cfg.five_dataset_names) or list(FIVE_DATASETS_DEFAULT)
        return get_five_datasets(
            batch_size=cfg.five_batch_size, seed=seed,
            image_size=cfg.five_image_size, dataset_names=names,
            class_il=class_il, **kwargs)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def model_builder(benchmark, num_classes):
    """Zero-argument builder of the benchmark's backbone with ``num_classes``."""
    if benchmark == "cifar":
        return lambda: SimpleCNN(num_classes=num_classes)
    if benchmark == "pmnist":
        return lambda: SimpleMLP(input_size=784, hidden_size=256,
                                 num_classes=num_classes)
    if benchmark in ("tinyimagenet", "fivedatasets"):
        return lambda: AdaptiveCNN(num_classes=num_classes)
    raise ValueError(f"Unknown benchmark: {benchmark}")
