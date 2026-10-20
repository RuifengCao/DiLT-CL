"""EWC baseline: Elastic Weight Consolidation (Kirkpatrick et al., 2017).

EWC prevents catastrophic forgetting by penalising changes to weights that
were important for previous tasks. Importance is the diagonal of the Fisher
Information Matrix. The loss for task ``t`` is::

    L = L_CE + (lambda / 2) * sum_k sum_i F_{k,i} * (theta_i - theta*_{k,i})^2

Penalty cost (v3)
-----------------
The v2 trainer evaluated the double sum literally: one pass over every stored
task per parameter per step, i.e. ``O(T * P)`` small kernels (about 570 on the
last Split-CIFAR-100 task). Because the per-task terms are quadratics in the
same variable, their sum is a single quadratic with the same gradient::

    sum_k F_k (theta - theta_k)^2 = A (theta - mu)^2 + const,
    A = sum_k F_k,   mu = (sum_k F_k theta_k) / A     (mu = 0 where A = 0)

``penalty_form="consolidated"`` (default) maintains ``A`` and ``mu`` and costs
``O(P)`` per step. Its gradient ``2 (A theta - sum_k F_k theta_k)`` is exactly
the gradient of the literal sum; only the constant offset of the loss value
(irrelevant to optimisation) and floating-point rounding differ. The stored
per-task Fishers and anchors are kept unchanged for checkpoints and audits.
``penalty_form="per_task"`` reproduces the v2 arithmetic bit for bit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .metrics import evaluate, set_model_task


class EWC:
    """Elastic Weight Consolidation continual learner.

    Attributes:
        model: Neural network model.
        device: ``'cuda'`` or ``'cpu'``.
        lambda_ewc: Regularisation strength.
        fisher_dict: Maps ``task_id -> {param_name: Fisher diagonal}``.
        optimal_params: Maps ``task_id -> {param_name: optimal param}``.
    """

    def __init__(self, model, device="cuda", lambda_ewc=10000,
                 penalty_form="consolidated", fisher_labels="prediction"):
        if penalty_form not in ("consolidated", "per_task"):
            raise ValueError(
                f"penalty_form must be 'consolidated' or 'per_task', "
                f"got {penalty_form!r}")
        if fisher_labels not in ("prediction", "true"):
            raise ValueError(
                f"fisher_labels must be 'prediction' or 'true', "
                f"got {fisher_labels!r}")
        self.model = model.to(device)
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.penalty_form = penalty_form
        self.fisher_labels = fisher_labels
        self.fisher_dict = {}
        self.optimal_params = {}
        # Consolidated quadratic: A = sum_k F_k, mu = (sum_k F_k theta_k) / A.
        self._fisher_sum = {}
        self._fisher_weighted_anchor = {}
        self._anchor = {}
        self._consolidated_tasks = set()
        self._flat_cache = None

    def _consolidate(self, task_id):
        """Fold task ``task_id``'s Fisher/anchor into the running quadratic.

        Called once per task, as soon as both its Fisher and its anchor exist
        (whichever of :meth:`compute_fisher` / :meth:`save_optimal_params`
        runs second).
        """
        if (task_id in self._consolidated_tasks
                or task_id not in self.fisher_dict
                or task_id not in self.optimal_params):
            return
        self._consolidated_tasks.add(task_id)
        fisher = self.fisher_dict[task_id]
        optimum = self.optimal_params[task_id]
        for name, f in fisher.items():
            f = f.to(self.device)
            weighted = f * optimum[name].to(self.device)
            if name in self._fisher_sum:
                self._fisher_sum[name] = self._fisher_sum[name] + f
                self._fisher_weighted_anchor[name] = (
                    self._fisher_weighted_anchor[name] + weighted)
            else:
                self._fisher_sum[name] = f.clone()
                self._fisher_weighted_anchor[name] = weighted
            total = self._fisher_sum[name]
            self._anchor[name] = torch.where(
                total > 0, self._fisher_weighted_anchor[name] / total.clamp_min(
                    torch.finfo(total.dtype).tiny),
                torch.zeros_like(total))
        self._flat_cache = None   # rebuilt lazily by _penalty_consolidated

    def compute_fisher(self, train_loader, task_id, num_samples=256):
        """Estimate the diagonal Fisher Information for a task.

        The estimate is the mean of *per-sample* squared gradients
        ``E_x[(d/dtheta log p(label|x))^2]``. With ``fisher_labels="prediction"``
        (default, the v2 behaviour) the label is the model's own argmax
        prediction, a common approximation in EWC implementations; with
        ``"true"`` it is the data label, i.e. the empirical Fisher proper. The
        two are different estimators and the choice is recorded in the
        experiment config (``ewc_fisher_labels``).

        Averaging the loss over a minibatch first and squaring that gradient
        would compute ``(mean grad)^2`` instead, which shrinks roughly like
        ``1/batch_size`` and made the effective regularisation strength depend
        on the loader's batch size before v2.1.8. Samples are therefore
        processed one at a time, capped at ``num_samples`` so the estimate
        stays cheap.

        Args:
            train_loader: DataLoader used to estimate the Fisher.
            task_id: Task identifier.
            num_samples: Number of individual samples to average over. ``None``
                uses the whole loader.
        """
        # Route to this task's head before differentiating: the Fisher must be
        # taken with respect to the predictor that was actually trained. This
        # is defensive -- train_task already selects the head -- so that a
        # direct call cannot silently anchor the wrong output layer.
        set_model_task(self.model, task_id)
        self.model.eval()
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        total_samples = 0
        stop = False
        for data, target in train_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            for i in range(data.size(0)):
                self.model.zero_grad()
                output = F.log_softmax(self.model(data[i:i + 1]), dim=1)
                if self.fisher_labels == "true":
                    label = target[i:i + 1]              # empirical Fisher
                else:
                    label = output.max(1)[1]             # model's own prediction
                F.nll_loss(output, label).backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.data.detach() ** 2

                total_samples += 1
                if num_samples is not None and total_samples >= num_samples:
                    stop = True
                    break
            if stop:
                break

        denominator = max(1, total_samples)
        for name in fisher:
            fisher[name] = fisher[name] / denominator

        self.fisher_dict[task_id] = fisher
        self._consolidate(task_id)
        print(f"[EWC] Fisher information computed for task {task_id + 1} "
              f"({total_samples} samples)")

    def save_optimal_params(self, task_id):
        """Snapshot the current parameters as optimal for ``task_id``."""
        self.optimal_params[task_id] = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self._consolidate(task_id)

    def penalty(self, model):
        """Compute the summed EWC penalty over all stored tasks.

        Args:
            model: The model currently being optimised.

        Returns:
            The penalty loss (a scalar tensor, or ``0.0`` if no tasks stored).
            With ``penalty_form="consolidated"`` the value omits a constant
            that does not depend on the parameters; the gradient is the same.
        """
        if self.penalty_form == "consolidated":
            return self._penalty_consolidated(model)
        return self._penalty_per_task(model)

    def _penalty_consolidated(self, model):
        """``0.5 * lambda * sum_i A_i (theta_i - mu_i)^2`` in a handful of kernels.

        All covered parameters are concatenated into one vector per step and
        the quadratic is evaluated once, instead of four small kernels (plus
        their backward) per parameter tensor. The gradient of each element,
        ``lambda * A_i (theta_i - mu_i)``, is computed elementwise either
        way, so it is bit-identical to the per-parameter form; only the
        summation order of the scalar loss value differs.
        """
        if not self._fisher_sum:
            return 0.0
        names = [name for name, _ in model.named_parameters()
                 if name in self._fisher_sum]
        if not names:
            return 0.0
        params = dict(model.named_parameters())
        device = params[names[0]].device
        cache = getattr(self, "_flat_cache", None)
        if cache is None or cache[0] != names or cache[1] != device:
            total = torch.cat([self._fisher_sum[n].reshape(-1) for n in names]).to(device)
            anchor = torch.cat([self._anchor[n].reshape(-1) for n in names]).to(device)
            cache = (names, device, total, anchor)
            self._flat_cache = cache
        _, _, total, anchor = cache
        flat = torch.cat([params[n].reshape(-1) for n in names])
        diff = flat - anchor
        return 0.5 * self.lambda_ewc * (total * diff ** 2).sum()

    def _penalty_consolidated_per_parameter(self, model):
        """Reference form of :meth:`_penalty_consolidated` (one quadratic per
        parameter tensor); kept for the equivalence test."""
        if not self._fisher_sum:
            return 0.0
        loss = 0.0
        for name, param in model.named_parameters():
            total = self._fisher_sum.get(name)
            if total is None:
                continue
            diff = param - self._anchor[name].to(param.device)
            loss = loss + (total.to(param.device) * diff ** 2).sum()
        return 0.5 * self.lambda_ewc * loss

    def _penalty_per_task(self, model):
        """v2 arithmetic: the literal per-task double sum (legacy)."""
        loss = 0.0
        for task_id in self.fisher_dict:
            fisher = self.fisher_dict[task_id]
            opt_params = self.optimal_params[task_id]
            for name, param in model.named_parameters():
                if name in fisher and name in opt_params:
                    diff = param - opt_params[name].to(param.device)
                    loss += (fisher[name].to(param.device) * diff ** 2).sum()
        return 0.5 * self.lambda_ewc * loss

    def train_task(self, train_loader, test_loader, task_id, epochs=50,
                   lr=1e-3, previous_tasks=None):
        """Train on a task with the EWC penalty, then store its Fisher.

        Args:
            train_loader: Training DataLoader.
            test_loader: Test DataLoader.
            task_id: Current task identifier.
            epochs: Number of training epochs.
            lr: Learning rate.
            previous_tasks: Previous task IDs (enables the penalty when non-empty).

        Returns:
            Test accuracy on the current task.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        set_model_task(self.model, task_id)
        self.model.train()

        for _ in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                output = self.model(data)
                ce_loss = criterion(output, target)
                ewc_loss = self.penalty(self.model) if previous_tasks else 0.0
                loss = ce_loss + ewc_loss

                loss.backward()
                optimizer.step()

        # Consolidate: store optimal params and Fisher for this task.
        self.save_optimal_params(task_id)
        self.compute_fisher(train_loader, task_id)

        acc = evaluate(self.model, test_loader, self.device, task_id=task_id)
        print(f"[EWC] Task {task_id + 1} - Test accuracy: {acc:.2f}%")
        return acc
