"""A reservoir-sampling memory buffer for rehearsal-based baselines.

Used by Experience Replay (ER) and Dark Experience Replay (DER++). The buffer
keeps a fixed-capacity, uniformly-random sample of the stream via reservoir
sampling (Vitter, 1985) and optionally stores per-sample logits (needed by
DER++ for its logit-distillation term).

Performance design (v3)
-----------------------
The v2 implementation stored every item as a separate CPU tensor and copied
each sample individually (``inputs[i].detach().cpu().clone()``), i.e. two or
three device-to-host synchronisations *per sample*, per training step. On a
fast GPU that cost an order of magnitude more than the forward/backward pass.

The buffer now keeps its storage as a few pre-allocated tensors on the
training device and applies each minibatch's reservoir updates with a single
``index_copy_``. The random decisions are still taken **one item at a time, in
stream order, with the very same calls into Python's ``random`` module**
(``random.randint`` per item, ``random.sample`` per draw), so the sequence of
retained items and of replayed samples is identical to v2 for a given seed.
Within one minibatch, later items may overwrite the slot chosen by earlier
items; that "last write wins" semantics is resolved in Python before the
single batched copy, which keeps it well-defined on CUDA (where duplicate
indices in ``index_copy_`` are not).
"""

import random

import torch


class ReservoirBuffer:
    """Fixed-capacity reservoir buffer of ``(input, label[, logits])`` items.

    Args:
        capacity: Maximum number of items retained.
        device: Device the sampled batches (and, by default, the storage) live
            on.
        storage_device: Where to keep the buffer contents. Defaults to
            ``device``; pass ``"cpu"`` to reproduce v2's host-side storage when
            device memory is scarce (values are identical either way).
    """

    def __init__(self, capacity, device="cpu", storage_device=None):
        self.capacity = int(capacity)
        self.device = device
        self.storage_device = device if storage_device is None else storage_device
        self.inputs = None            # [capacity, ...]  allocated on first add
        self.labels = None            # [capacity]
        self.logits = None            # [capacity, C] or None
        self.task_ids = []            # Python list, len == current size
        self.size = 0
        self.num_seen = 0             # total items seen (for reservoir prob.)
        self.sample_calls = 0
        self.sampled_items_total = 0

    def __len__(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def _allocate(self, inputs, labels, logits):
        self.inputs = torch.empty((self.capacity,) + tuple(inputs.shape[1:]),
                                  dtype=inputs.dtype, device=self.storage_device)
        self.labels = torch.empty((self.capacity,) + tuple(labels.shape[1:]),
                                  dtype=labels.dtype, device=self.storage_device)
        if logits is not None:
            self.logits = torch.empty(
                (self.capacity,) + tuple(logits.shape[1:]),
                dtype=logits.dtype, device=self.storage_device)

    def add(self, inputs, labels, logits=None, task_id=None):
        """Add a minibatch to the buffer with per-item reservoir decisions.

        Args:
            inputs: Tensor ``(B, ...)``.
            labels: Tensor ``(B,)``.
            logits: Optional tensor ``(B, C)`` to store alongside each item.
                Either every ``add`` call passes logits or none does.
            task_id: Task the batch belongs to. Required for multi-head
                replay, where a stored sample must be replayed through *its
                own* head: feeding it through the current task's head trains
                that head to emit an earlier task's label, which corrupts it
                instead of preventing forgetting.
        """
        inputs = inputs.detach()
        labels = labels.detach()
        logits = None if logits is None else logits.detach()
        if self.inputs is None:
            self._allocate(inputs, labels, logits)
        elif (logits is None) != (self.logits is None):
            raise ValueError(
                "logits must be supplied on every add() call or on none")

        # Sequential reservoir decisions, exactly as the item-wise loop made
        # them; ``plan`` maps slot -> source row with last-write-wins.
        plan = {}
        for i in range(int(inputs.size(0))):
            if self.size < self.capacity:
                slot = self.size
                self.size += 1
                self.task_ids.append(task_id)
                plan[slot] = i
            else:
                # Replace a random slot with probability capacity / num_seen.
                j = random.randint(0, self.num_seen)
                if j < self.capacity:
                    plan[j] = i
                    self.task_ids[j] = task_id
            self.num_seen += 1
        if not plan:
            return
        # One host-to-device copy for both index vectors.
        pairs = torch.tensor([list(plan.keys()), list(plan.values())],
                             dtype=torch.long).to(self.storage_device)
        slots = pairs[0]
        rows = pairs[1].to(inputs.device)      # no-op when devices match
        self.inputs.index_copy_(
            0, slots, inputs.index_select(0, rows).to(self.storage_device))
        self.labels.index_copy_(
            0, slots, labels.index_select(0, rows).to(self.storage_device))
        if logits is not None:
            self.logits.index_copy_(
                0, slots, logits.index_select(0, rows).to(self.storage_device))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _draw(self, batch_size):
        n = min(int(batch_size), self.size)
        idx = random.sample(range(self.size), n)
        self.sample_calls += 1
        self.sampled_items_total += n
        return idx

    def _gather(self, index):
        inputs = self.inputs.index_select(0, index).to(self.device)
        labels = self.labels.index_select(0, index).to(self.device)
        logits = (None if self.logits is None
                  else self.logits.index_select(0, index).to(self.device))
        return inputs, labels, logits

    def sample(self, batch_size):
        """Sample a minibatch uniformly at random (without replacement).

        Args:
            batch_size: Number of items to draw (capped at the current size).

        Returns:
            ``(inputs, labels, logits_or_None)`` moved onto ``self.device``.
            Use :meth:`sample_by_task` or :meth:`sample_routed` when replay
            must be routed per head.
        """
        idx = self._draw(batch_size)
        index = torch.tensor(idx, dtype=torch.long, device=self.storage_device)
        return self._gather(index)

    def sample_routed(self, batch_size):
        """Sample a minibatch and return it flat, with per-item task routing.

        Returns:
            ``(inputs, labels, logits_or_None, task_ids)`` where ``task_ids``
            is a ``LongTensor`` on ``self.device`` holding, per drawn row, the
            id of the task that produced it (``-1`` for items stored without
            a task id, which callers route through the active head). The draw
            consumes the RNG exactly like :meth:`sample_by_task`, and the
            whole draw costs a single host-to-device copy.
        """
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self._draw(batch_size)
        tasks = [-1 if self.task_ids[i] is None else int(self.task_ids[i])
                 for i in idx]
        meta = torch.tensor([idx, tasks], dtype=torch.long).to(self.storage_device)
        inputs, labels, logits = self._gather(meta[0])
        return inputs, labels, logits, meta[1].to(self.device)

    def sample_by_task(self, batch_size):
        """Sample a minibatch and group it by the task each item came from.

        Multi-head replay needs this: every stored item must go through the
        head of the task that produced it. Replaying an old sample through the
        current task's head teaches that head an earlier task's label, which
        actively degrades it (empirically ER fell *below* plain finetuning).

        Returns:
            Dict ``{task_id: (inputs, labels, logits_or_None)}``. Items stored
            without a task id are grouped under ``None``. Group order follows
            the draw order, as in v2.
        """
        if self.size == 0:
            return {}
        idx = self._draw(batch_size)
        grouped = {}
        for i in idx:
            grouped.setdefault(self.task_ids[i], []).append(i)
        out = {}
        for task, indices in grouped.items():
            index = torch.tensor(indices, dtype=torch.long,
                                 device=self.storage_device)
            out[task] = self._gather(index)
        return out
