"""LwF baseline: Learning without Forgetting (Li & Hoiem, 2017).

LwF distils knowledge from the previous-task model. When learning a new task,
the old model's outputs on the new data act as soft targets::

    L = L_CE(new) + alpha * T^2 * KL(softmax(old / T), softmax(new / T))

Multi-head cost (v3)
--------------------
Under the multi-head task-IL protocol every previous task owns a head and is
distilled separately. The v2 trainer re-ran the *entire* network once per old
head for both student and teacher, so a step at task ``t`` cost ``2t + 1``
trunk forwards (19 extra student forwards *and backwards* on the last
Split-CIFAR-100 task). The trunk output does not depend on the head, so the
default ``shared_features=True`` path computes student and teacher features
once and applies all old heads as a single matmul
(:meth:`diltcl.models.MultiHeadModel.stacked_head_logits`). The objective is
the same function of the same parameters; only floating-point summation order
differs. ``shared_features=False`` reproduces the v2 arithmetic bit for bit.

Note for the paper: the O(T) forward cost is therefore an artefact of the v2
implementation, not an intrinsic property of multi-head LwF, and must not be
cited as a computational advantage of DiLT-CL.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .metrics import evaluate, set_model_task


class LwF:
    """Learning-without-Forgetting continual learner.

    Attributes:
        model: Neural network model.
        device: ``'cuda'`` or ``'cpu'``.
        alpha: Distillation loss weight.
        T: Temperature for softening logits.
        old_model: Frozen copy of the model from the previous task.
    """

    def __init__(self, model, device="cuda", alpha=1.0, temperature=2.0,
                 shared_features=True):
        self.model = model.to(device)
        self.device = device
        self.alpha = alpha
        self.T = temperature
        self.old_model = None
        self.shared_features = bool(shared_features)

    def distill_loss(self, outputs, old_outputs):
        """Knowledge-distillation loss via KL divergence of softened logits.

        Args:
            outputs: Current model logits (new task), ``[B, C]``.
            old_outputs: Old model logits (soft targets), ``[B, C]``.

        Returns:
            The temperature-scaled distillation loss.
        """
        T = self.T
        # ``F.kl_div(input, target)`` computes KL(target || exp(input)).
        # The student supplies log-probabilities and the frozen teacher the
        # target probabilities. v2.0 accidentally optimized the reverse KL.
        log_student = F.log_softmax(outputs / T, dim=-1)
        teacher = F.softmax(old_outputs / T, dim=-1)
        loss = F.kl_div(log_student, teacher, reduction="batchmean",
                        log_target=False)
        return (T ** 2) * loss

    def _multihead_distill_shared(self, data, task_id, features=None):
        """Mean over old heads of the per-head distillation, one trunk pass.

        Equals ``mean_t alpha * distill_loss(student_t, teacher_t)``: the
        stacked ``[B, t, C]`` logits are flattened to ``[B*t, C]`` so that
        ``batchmean`` divides by ``B*t``, i.e. the per-head sum over ``B``
        divided by ``t``. ``features`` may be passed in when the caller has
        already computed the student trunk features for the CE term.
        """
        old_tasks = list(range(task_id))
        if features is None:
            features = self.model.features(data)
        with torch.no_grad():
            teacher_features = self.old_model.features(data)
            teacher = self.old_model.stacked_head_logits(
                teacher_features, old_tasks)
        student = self.model.stacked_head_logits(features, old_tasks)
        classes = student.shape[-1]
        return self.alpha * self.distill_loss(
            student.reshape(-1, classes), teacher.reshape(-1, classes))

    def _multihead_distill_per_head(self, data, task_id):
        """v2 arithmetic: a full network forward per old head (legacy)."""
        distill = 0.0
        for old_task in range(task_id):
            set_model_task(self.model, old_task)
            set_model_task(self.old_model, old_task)
            student_old = self.model(data)
            with torch.no_grad():
                teacher_old = self.old_model(data)
            distill = distill + self.alpha * self.distill_loss(
                student_old, teacher_old)
        set_model_task(self.model, task_id)
        return distill / float(task_id)

    def save_old_model(self):
        """Freeze a copy of the current model for the next task's distillation."""
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False

    def train_task(self, train_loader, test_loader, task_id, epochs=50, lr=1e-3):
        """Train on a task with LwF distillation, then snapshot the model.

        Args:
            train_loader: Training DataLoader.
            test_loader: Test DataLoader.
            task_id: Current task identifier.
            epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            Test accuracy on the current task.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        set_model_task(self.model, task_id)
        self.model.train()

        multi_head = bool(getattr(self.model, "num_tasks", None)) and task_id > 0
        shared = (multi_head and self.shared_features
                  and hasattr(self.model, "stacked_head_logits"))

        for _ in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                if shared and self.old_model is not None:
                    # One student trunk pass serves both the CE term and the
                    # distillation of every old head; ``head_logits(features)``
                    # is the same computation as ``model(data)``.
                    features = self.model.features(data)
                    output = self.model.head_logits(features, task_id)
                else:
                    features = None
                    output = self.model(data)
                ce_loss = criterion(output, target)

                distill = 0.0
                if self.old_model is not None:
                    if multi_head:
                        # Multi-head LwF: each previous task owns a head, so the
                        # teacher signal must be preserved head by head. Summing
                        # a single head's distillation would leave every other
                        # old head unconstrained.
                        if shared:
                            distill = self._multihead_distill_shared(
                                data, task_id, features)
                        else:
                            distill = self._multihead_distill_per_head(
                                data, task_id)
                    else:
                        with torch.no_grad():
                            old_output = self.old_model(data)
                        distill = self.alpha * self.distill_loss(
                            output, old_output)

                loss = ce_loss + distill
                loss.backward()
                optimizer.step()

        self.save_old_model()

        acc = evaluate(self.model, test_loader, self.device, task_id=task_id)
        print(f"[LwF] Task {task_id + 1} - Test accuracy: {acc:.2f}%")
        return acc
