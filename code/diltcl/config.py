"""Hyper-parameter configuration for the continual-learning experiments.

Collecting every tunable knob in one dataclass keeps the experiment runner
readable and makes it trivial to spin up reduced "fast" runs for debugging.
"""

from dataclasses import dataclass, fields
import math


@dataclass
class ExperimentConfig:
    """All tunable knobs for the experiment suite.

    Use ``ExperimentConfig()`` for full, paper-quality runs or
    ``ExperimentConfig.fast()`` for a quick end-to-end smoke test.
    """

    seed: int = 42

    # --- DiLT-CL training schedule ---
    diltcl_full_epochs: int = 50      # Phase-1 full-network training
    diltcl_imp_epochs: int = 20       # Epochs per IMP round
    diltcl_expert_finetune_epochs: int = 20   # Expert training after IMP
    pruning_rounds: int = 5
    diltcl_partitions: int = 2
    diltcl_sparsity: float = 0.9

    # --- Baseline training schedule ---
    epochs_ewc: int = 50
    epochs_lwf: int = 50
    epochs_replay: int = 50
    ewc_lambda: float = 10_000.0
    # Label used for the per-sample Fisher: "prediction" (the model's argmax,
    # the v2 behaviour) or "true" (the empirical Fisher with the data label).
    ewc_fisher_labels: str = "prediction"
    lwf_alpha: float = 1.0
    lwf_temperature: float = 2.0

    # --- Replay baselines (ER / DER++) ---
    buffer_size: int = 500            # Total reservoir-buffer capacity
    replay_batch_size: int = 32       # Memory minibatch size per step
    derpp_alpha: float = 0.5          # DER++ logit-distillation weight
    derpp_beta: float = 0.5           # DER++ label-replay weight

    # --- Subnetwork-family baselines (PackNet / SupSup) ---
    # PackNet and SupSup reuse ``epochs_replay`` for their main training budget
    # so every baseline gets the same number of epochs per task.
    packnet_retrain_epochs: int = 10  # Retrain epochs after pruning each task
    packnet_keep_fraction: float = 0.0  # 0 => auto equal-split 1/(T - t)
    supsup_sparsity: float = 0.5      # Fraction of weights masked out per task
    wsn_sparsity: float = 0.5         # WSN per-task subnetwork sparsity

    # --- Benchmark sizes ---
    c100_num_tasks: int = 20
    c100_classes_per_task: int = 5
    c100_batch_size: int = 64
    pm_num_tasks: int = 10
    pm_batch_size: int = 256

    # --- Split-TinyImageNet (200 classes, 64x64x3) ---
    tin_num_tasks: int = 10
    tin_classes_per_task: int = 20
    tin_batch_size: int = 64
    tin_image_size: int = 64

    # --- 5-datasets (five 10-way datasets as five tasks, 32x32x3) ---
    five_batch_size: int = 64
    five_image_size: int = 32
    # Empty tuple => hard_benchmarks.FIVE_DATASETS_DEFAULT (torchvision-only).
    five_dataset_names: tuple = ()

    # --- Optimisation ---
    lr: float = 1e-3

    def with_overrides(self, overrides):
        """Return a validated copy with values from a JSON-style mapping.

        The command-line runners use this helper for versioned ablation files.
        Unknown keys and ambiguous type coercions are rejected so a typo cannot
        silently turn into a different experiment.
        """
        if not isinstance(overrides, dict):
            raise TypeError("config overrides must be a JSON object")
        known = {field.name for field in fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"unknown ExperimentConfig fields: {unknown}")

        values = dict(self.__dict__)
        for name, value in overrides.items():
            current = values[name]
            if isinstance(current, tuple):
                if not isinstance(value, (list, tuple)):
                    raise TypeError(f"{name} must be a JSON array")
                value = tuple(value)
            elif isinstance(current, bool):
                if not isinstance(value, bool):
                    raise TypeError(f"{name} must be boolean")
            elif isinstance(current, int):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an integer")
            elif isinstance(current, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{name} must be numeric")
                value = float(value)
            elif isinstance(current, str):
                if not isinstance(value, str):
                    raise TypeError(f"{name} must be a string")
            values[name] = value

        updated = type(self)(**values)
        updated.validate()
        return updated

    def validate(self):
        """Validate ranges used by both the ordinary and ablation runners."""
        positive_ints = (
            "diltcl_full_epochs", "diltcl_imp_epochs", "pruning_rounds",
            "diltcl_partitions", "epochs_ewc", "epochs_lwf",
            "epochs_replay", "c100_num_tasks", "c100_classes_per_task",
            "c100_batch_size", "pm_num_tasks", "pm_batch_size",
            "tin_num_tasks", "tin_classes_per_task", "tin_batch_size",
            "tin_image_size", "five_batch_size", "five_image_size",
        )
        for name in positive_ints:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be strictly positive")
        if self.diltcl_expert_finetune_epochs < 0:
            raise ValueError(
                "diltcl_expert_finetune_epochs must be non-negative")
        if self.packnet_retrain_epochs < 0:
            raise ValueError("packnet_retrain_epochs must be non-negative")
        if self.buffer_size < 0 or self.replay_batch_size <= 0:
            raise ValueError("buffer_size must be non-negative and replay_batch_size positive")
        for name in ("diltcl_sparsity", "supsup_sparsity", "wsn_sparsity"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if not 0.0 <= self.packnet_keep_fraction <= 1.0:
            raise ValueError("packnet_keep_fraction must lie in [0, 1]")
        finite_positive = (
            "lr", "ewc_lambda", "lwf_temperature", "replay_batch_size",
        )
        for name in finite_positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        finite_nonnegative = ("lwf_alpha", "derpp_alpha", "derpp_beta")
        for name in finite_nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.c100_num_tasks * self.c100_classes_per_task > 100:
            raise ValueError("CIFAR-100 task layout requests more than 100 classes")
        if self.tin_num_tasks * self.tin_classes_per_task > 200:
            raise ValueError("TinyImageNet task layout requests more than 200 classes")
        names = tuple(self.five_dataset_names)
        if len(set(names)) != len(names):
            raise ValueError("five_dataset_names entries must be unique")
        if self.ewc_fisher_labels not in ("prediction", "true"):
            raise ValueError(
                "ewc_fisher_labels must be 'prediction' or 'true'")
        # Every remaining numeric field, so an invalid --config-override fails
        # immediately instead of after hours of training.
        for name in ("lr", "ewc_lambda", "lwf_alpha", "lwf_temperature"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        for name in ("pruning_rounds", "diltcl_partitions", "epochs_ewc",
                     "epochs_lwf", "epochs_replay", "diltcl_full_epochs",
                     "diltcl_imp_epochs", "c100_batch_size", "pm_batch_size",
                     "tin_batch_size", "five_batch_size", "tin_image_size",
                     "five_image_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ("diltcl_sparsity", "supsup_sparsity", "wsn_sparsity"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        for name in ("derpp_alpha", "derpp_beta"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        return self

    @classmethod

    def fast(cls) -> "ExperimentConfig":
        """A drastically reduced configuration for quick verification.

        Reduces epochs, pruning rounds and the number of tasks so the whole
        suite runs in minutes rather than hours.
        """
        return cls(
            diltcl_full_epochs=10,
            diltcl_imp_epochs=5,
            diltcl_expert_finetune_epochs=5,
            pruning_rounds=2,
            epochs_ewc=10,
            epochs_lwf=10,
            epochs_replay=10,
            packnet_retrain_epochs=3,
            c100_num_tasks=10,
            pm_num_tasks=5,
            tin_num_tasks=3,
            tin_classes_per_task=10,
        )
