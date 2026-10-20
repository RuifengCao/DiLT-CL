"""DiLT-CL: Continual Learning via Disentangled Lottery Tickets.

Core idea
---------
1. Train the full network on the first task.
2. Extract a *core* subnetwork via multi-partition IMP + intersection.
3. For each new task, extract an *expert* subnetwork from the complement
   (non-core) weights via IMP.
4. Inference uses ``core + task-specific expert`` masks.

All binary-mask manipulation is delegated to :mod:`diltcl.masking`, which keeps
every mask on a consistent device and is therefore immune to the original
cross-device crash.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import masking
from .data_backend import make_loader, subset_dataset

# Batch size of the partition-level IMP loaders in Phase 1. It is part of the
# declared schedule (see ``_schedule_accounting`` in the runners) and is kept at
# its historical value so the schedule of a run does not depend on the backend.
IMP_PARTITION_BATCH_SIZE = 64


class DiLTCL:
    """Disentangled Lottery-Ticket continual learner.

    Attributes:
        model: The shared neural network.
        device: ``'cuda'`` or ``'cpu'``.
        num_partitions: Number of data partitions for core extraction.
        pruning_rounds: Number of IMP iterations.
        target_sparsity: Target sparsity level for masks.
        full_train_epochs: Epochs for the Phase-1 full-network training.
        imp_epochs: Epochs trained inside each IMP round.
        lr: Learning rate used throughout.
        core_mask: Binary mask dict for the core subnetwork.
        expert_finetune_epochs: Epochs for training the expert subnetwork
            *after* its mask topology is found by IMP.
        expert_masks: Dict mapping ``task_id -> expert mask dict``.
        expert_weights: Dict mapping ``task_id -> trained expert weight dict``.
            These are the weights actually used at inference; without them the
            expert region would be left at its rewound initial values.
        initial_params: Snapshot of post-Phase-1 parameters (for rewinding).
        frozen_core_values: Frozen parameter values for the core weights.
    """

    def __init__(self, model, device="cuda", num_partitions=2,
                 pruning_rounds=5, target_sparsity=0.9,
                 full_train_epochs=50, imp_epochs=20, expert_finetune_epochs=20,
                 lr=1e-3, imp_batch_size=IMP_PARTITION_BATCH_SIZE):
        self.model = model.to(device)
        self.device = device
        self.num_partitions = num_partitions
        self.imp_batch_size = int(imp_batch_size)
        self.pruning_rounds = pruning_rounds
        self.target_sparsity = target_sparsity
        self.full_train_epochs = full_train_epochs
        self.imp_epochs = imp_epochs
        self.expert_finetune_epochs = expert_finetune_epochs
        self.lr = lr

        self.core_mask = None
        self.expert_masks = {}        # task_id -> mask dict
        self.expert_weights = {}      # task_id -> trained weight dict
        self.initial_params = None
        self.frozen_core_values = None
        self.core_partition_densities = []
        # Precomputed inference tensors. The core is frozen, so its
        # contribution is constant; recomputing it on every forward
        # multiplied the cost of class-IL routing by the task count.
        self._core_contribution = None
        self._effective_weights = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_core(self, train_loaders, task_id=0):
        """Phase 1: extract the core subnetwork from the first task.

        Steps: train the full network, snapshot its parameters, partition the
        training data, run IMP on each partition, and intersect the resulting
        masks to obtain the core. Core weights are then frozen.

        Args:
            train_loaders: List of DataLoaders for all tasks.
            task_id: Index of the first task (default 0).
        """
        print(f"[DiLT-CL] Phase 1: Training full network on task {task_id + 1}...")
        self._train_full(self.model, train_loaders[task_id],
                         epochs=self.full_train_epochs, lr=self.lr)

        # Snapshot parameters for later rewinding.
        self.initial_params = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        train_dataset = train_loaders[task_id].dataset
        print(f"[DiLT-CL] Partitioning dataset into {self.num_partitions} parts...")
        partitions = self._partition_dataset(train_dataset, self.num_partitions)

        masks = []
        for pid, partition in enumerate(partitions):
            print(f"[DiLT-CL] Running IMP on partition "
                  f"{pid + 1}/{self.num_partitions}...")
            model_copy = copy.deepcopy(self.model)
            mask = self._imp_on_partition(model_copy, partition)
            masks.append(mask)

        self.core_partition_densities = [
            masking.mask_density(mask) for mask in masks
        ]

        print("[DiLT-CL] Computing core mask (intersection)...")
        self.core_mask = masking.intersection_masks(masks)
        self.core_mask = masking.to_device(self.core_mask, self.device)

        core_density = masking.mask_density(self.core_mask)
        print(f"[DiLT-CL] Core density: {core_density:.4f} "
              f"(sparsity: {1.0 - core_density:.4f})")

        # Freeze the core weights at their trained values.
        self.frozen_core_values = {
            name: (param.data.clone().detach()
                   * self.core_mask.get(name, torch.ones_like(param)))
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self._core_contribution = {
            name: (self.core_mask.get(name, torch.zeros_like(value))
                   * value)
            for name, value in self.frozen_core_values.items()
        }
        self._effective_weights = {}
        print("[DiLT-CL] Core extraction complete.")

    def extract_expert(self, train_loaders, task_id):
        """Phase 2: extract an expert subnetwork for a subsequent task.

        Args:
            train_loaders: List of DataLoaders for all tasks.
            task_id: Index of the task to extract an expert for.

        Returns:
            The expert mask dict for ``task_id``.
        """
        print(f"[DiLT-CL] Phase 2: Extracting expert for task {task_id + 1}...")

        complement = masking.complement_mask(self.core_mask)
        expert_mask = self._imp_on_complement(
            self.model, train_loaders[task_id], complement, self.initial_params
        )
        expert_mask = masking.to_device(expert_mask, self.device)
        self.expert_masks[task_id] = expert_mask

        expert_density = masking.mask_density(expert_mask)
        print(f"[DiLT-CL] Expert density for task {task_id + 1}: "
              f"{expert_density:.4f}")

        # IMP only selected the expert's mask *topology*; now actually train the
        # expert weights and store them, otherwise the expert contributes
        # nothing at inference (its weights would be the rewound init values).
        print(f"[DiLT-CL] Fine-tuning expert weights for task {task_id + 1}...")
        self._finetune_expert(train_loaders[task_id], task_id, expert_mask,
                              complement)
        self._cache_effective_weights(task_id)
        return expert_mask

    def _cache_effective_weights(self, task_id):
        """Precompute the dense parameters used at inference for ``task_id``.

        ``forward_with_mask`` previously rebuilt ``core_mask * frozen_core``
        on every call even though the core never changes, and did so once per
        candidate task for every batch of class-IL routing. Caching the sum
        turns each call into a tensor swap. The arithmetic is unchanged, so
        results stay bit-identical.
        """
        if self._core_contribution is None:
            return
        expert_weights = self.expert_weights.get(task_id)
        if expert_weights is None:
            return
        effective = {}
        for name, core_value in self._core_contribution.items():
            value = core_value
            extra = expert_weights.get(name)
            if extra is not None:
                value = value + extra.to(value.device)
            effective[name] = value
        self._effective_weights[task_id] = effective

    def forward_with_mask(self, x, task_id):
        """Forward pass using ``core + task-specific expert``.

        For a task with a trained expert (the standard pipeline) the network is
        temporarily loaded with that task's stored weights -- the frozen core
        plus the task's own trained expert -- so experts never interfere across
        tasks and each task is evaluated with the weights it was trained with.
        For task 0 (core only) or callers that train the live model directly
        (e.g. the quick demo), it falls back to masking the live weights.

        The core is frozen, so ``core_mask * frozen_core_values`` is constant;
        it is combined with each expert once in
        :meth:`_cache_effective_weights` and reused here, which turns every call
        into a tensor swap instead of rebuilding the parameters. The arithmetic
        is unchanged, so results stay bit-identical.

        Args:
            x: Input tensor.
            task_id: Task ID selecting the expert.

        Returns:
            Model output logits.
        """
        expert_mask = self.expert_masks.get(task_id, {})
        combined = masking.combine_masks(self.core_mask, expert_mask)

        if task_id not in self.expert_weights:
            # Core-only (task 0) or live-model mode.
            return masking.masked_forward(self.model, x, combined)

        device = self.device
        expert_w = self.expert_weights[task_id]
        effective_cache = self._effective_weights.get(task_id)
        if effective_cache is None:
            # Older instances (e.g. unpickled from a previous version) may not
            # have the cache; build it on first use.
            self._cache_effective_weights(task_id)
            effective_cache = self._effective_weights.get(task_id)
        saved = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                saved[name] = param.data
                if effective_cache is not None and name in effective_cache:
                    param.data = effective_cache[name]
                    continue
                core_m = self.core_mask.get(name, torch.zeros_like(param)).to(device)
                effective = core_m * self.frozen_core_values[name].to(device)
                ew = expert_w.get(name)
                if ew is not None:
                    effective = effective + ew.to(device)
                param.data = effective

        try:
            return self.model(x)
        finally:
            # Always restore the live model, including when a custom forward
            # raises. Leaking one task's expert into the next call silently
            # corrupts task-inference comparisons.
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in saved:
                        param.data = saved[name]

    def get_core_sparsity(self):
        """Return the sparsity of the core subnetwork."""
        if self.core_mask is None:
            return 0.0
        return 1.0 - masking.mask_density(self.core_mask)

    def get_expert_sparsity(self, task_id):
        """Return the sparsity of a task's expert subnetwork."""
        mask = self.expert_masks.get(task_id, {})
        if not mask:
            return 0.0
        return 1.0 - masking.mask_density(mask)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_full(self, model, train_loader, epochs=50, lr=1e-3,
                    verbose=False):
        """Train a model with standard cross-entropy loss."""
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model.train()
        device = self.device

        for epoch in range(epochs):
            epoch_loss = None
            num_batches = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                # Accumulate on the device: a per-step ``.item()`` would force
                # a host sync every step just to print an epoch average.
                detached = loss.detach()
                epoch_loss = detached if epoch_loss is None else epoch_loss + detached
                num_batches += 1

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = (float(epoch_loss.item()) / num_batches
                            if num_batches else 0.0)
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def _train_with_mask(self, model, train_loader, mask, epochs=30, lr=1e-3,
                         forward_mask=None):
        """Train a model with a binary mask applied to the gradients.

        Only weights where ``mask == 1`` receive gradient updates. Parameters
        outside ``forward_mask`` are also zeroed before every forward and after
        every optimizer step. This second constraint is essential: gradient
        masking alone leaves rewound, pruned weights active in the network.
        """
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        device = self.device

        active_forward = mask if forward_mask is None else forward_mask
        for _ in range(epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                masking.apply_parameter_mask_(model, active_forward)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                # Zero-out gradients for masked-off weights (one fused kernel).
                masking.apply_gradient_mask_(model, mask)

                optimizer.step()
                masking.apply_parameter_mask_(model, active_forward)

    def _finetune_expert(self, train_loader, task_id, expert_mask,
                         complement_mask, epochs=None):
        """Train the expert subnetwork and snapshot its weights.

        IMP only *selects* the expert's mask topology. Without this step the
        expert weights remain at their rewound initial values, so at inference
        the expert region is effectively random and contributes nothing -- the
        cause of the artifactual "zero forgetting / flat accuracy" behaviour.

        Here we initialise the network to exactly the subnetwork used at
        inference (core region -> frozen core values, expert region -> init,
        everything else -> 0 so the training-time forward matches inference),
        train **only** the expert weights via gradient masking, and store the
        resulting per-task weights in :attr:`expert_weights`.

        Args:
            train_loader: DataLoader for the task.
            task_id: Task identifier.
            expert_mask: Binary mask selecting this task's expert weights.
            complement_mask: The non-core region the expert lives in (unused
                directly here but kept for clarity / future use).
            epochs: Fine-tuning epochs (defaults to
                :attr:`expert_finetune_epochs`).
        """
        epochs = self.expert_finetune_epochs if epochs is None else epochs
        device = self.device

        # Initialise to the exact subnetwork used at inference.
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                core_m = self.core_mask.get(name, torch.zeros_like(param)).to(device)
                exp_m = expert_mask.get(name, torch.zeros_like(param)).to(device)
                core_val = self.frozen_core_values[name].to(device)
                init_val = self.initial_params[name].to(device)
                param.data = core_m * core_val + exp_m * init_val

        # Train ONLY the expert weights (gradients gated by the expert mask;
        # the core and the zeroed-out weights stay fixed).
        combined = masking.combine_masks(self.core_mask, expert_mask)
        self._train_with_mask(self.model, train_loader, expert_mask,
                              epochs=epochs, lr=self.lr,
                              forward_mask=combined)

        # Snapshot the trained expert weights, masked to the expert region.
        with torch.no_grad():
            self.expert_weights[task_id] = {
                name: (param.data.clone().detach()
                       * expert_mask.get(name, torch.zeros_like(param)).to(device))
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }

    def _imp_on_partition(self, model, dataset, pruning_rate=None, lr=None):
        """Run IMP on a single data partition and return its mask."""
        lr = self.lr if lr is None else lr
        # ``make_loader`` keeps device-resident partitions on the device and
        # reproduces the legacy ``DataLoader(dataset, 64, shuffle=True)`` for
        # ordinary datasets (single process, no pinning).
        dataloader = make_loader(dataset, self.imp_batch_size, shuffle=True)

        # Snapshot for rewinding within this IMP run.
        initial_state = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        mask = masking.ones_like_mask(model)
        if pruning_rate is None:
            pruning_rate = masking.iterative_pruning_rate(
                self.target_sparsity, self.pruning_rounds)

        for _ in range(self.pruning_rounds):
            self._train_with_mask(model, dataloader, mask,
                                  epochs=self.imp_epochs, lr=lr,
                                  forward_mask=mask)

            mask = masking.prune_by_magnitude(model, mask, pruning_rate)

            # Rewind remaining weights to the snapshot.
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and name in initial_state:
                        param.data = initial_state[name].clone()

            if 1.0 - masking.mask_density(mask) >= self.target_sparsity:
                break

        return mask

    def _imp_on_complement(self, model, train_loader, complement_mask,
                           initial_params, sparsity=None):
        """Run IMP restricted to the complement (non-core) weights.

        Args:
            model: The shared model.
            train_loader: DataLoader for the new task.
            complement_mask: Mask of non-core weights (``1`` = in complement).
            initial_params: Initial parameters used for rewinding.
            sparsity: Target sparsity within the complement.

        Returns:
            Binary mask dict for the expert (``1`` = expert weight).
        """
        device = self.device
        sparsity = self.target_sparsity if sparsity is None else sparsity
        mask = masking.clone_mask(complement_mask)
        pruning_rate = masking.iterative_pruning_rate(
            sparsity, self.pruning_rounds)

        # Every expert must start from the same late-rewind state. v2.0 left
        # the previous task's live expert in the model for the first round,
        # making expert topology depend on task order for the wrong reason.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not (param.requires_grad and name in initial_params):
                    continue
                core_m = self.core_mask.get(name, torch.zeros_like(param))
                comp_m = complement_mask.get(name, torch.ones_like(param))
                param.data = (core_m.to(device) * self.frozen_core_values[name]
                              + comp_m.to(device) * initial_params[name])

        for _ in range(self.pruning_rounds):
            # Train (only complement weights update via the mask).
            forward_mask = masking.combine_masks(self.core_mask, mask)
            self._train_with_mask(model, train_loader, mask,
                                  epochs=self.imp_epochs, lr=self.lr,
                                  forward_mask=forward_mask)

            # Prune within the complement region.
            mask = masking.prune_by_magnitude(
                model, mask, pruning_rate, region_mask=complement_mask
            )

            # Keep core weights frozen, rewind the complement to init.
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if not (param.requires_grad and name in initial_params):
                        continue
                    core_m = self.core_mask.get(name, torch.zeros_like(param))
                    comp_m = complement_mask.get(name, torch.ones_like(param))
                    param.data = (core_m.to(device) * self.frozen_core_values[name]
                                  + comp_m.to(device) * initial_params[name])

            if masking.mask_density_in_region(mask, complement_mask) <= (1.0 - sparsity):
                break

        return mask

    # ------------------------------------------------------------------
    # Dataset partitioning
    # ------------------------------------------------------------------

    @staticmethod
    def _partition_dataset(dataset, num_partitions):
        """Randomly split a dataset into ``num_partitions`` disjoint subsets."""
        total_size = len(dataset)
        indices = np.random.permutation(total_size)
        partition_size = total_size // num_partitions

        partitions = []
        for i in range(num_partitions):
            start = i * partition_size
            end = (start + partition_size if i < num_partitions - 1
                   else total_size)
            partitions.append(subset_dataset(dataset, indices[start:end]))
        return partitions
