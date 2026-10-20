"""Edge-popup masked layers for the SupSup baseline.

This module implements the masking machinery of Ramanujan et al. (2020),
*What's Hidden in a Randomly Weighted Neural Network?*, as used by SupSup
(Wortsman et al., 2020, *Supermasks in Superposition*):

* the weights of every layer are fixed at their random initialisation and are
  **never trained**;
* for each task we learn a real-valued *score* per weight;
* the forward pass keeps only the top ``(1 - sparsity)`` fraction of weights by
  score magnitude (a binary supermask);
* gradients reach the scores through a straight-through estimator.

The multitask variants hold one score tensor per task and expose
:meth:`set_task`, so a single fixed random backbone can host an independent
subnetwork for every task. Because each task uses its own mask over a shared,
never-changing backbone, its oracle-route task-IL predictor is fixed. Class-IL
accuracy can still fall when later masks are added to the routing candidate set.

The layers subclass ``nn.Linear`` / ``nn.Conv2d`` so that :func:`convert_to_masked`
can swap them into an existing model (e.g. ``SimpleCNN`` / ``SimpleMLP``)
without rewriting the architecture.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class _GetSubnet(Function):
    """Top-k binary mask in the forward pass; straight-through in backward.

    Forward: return a 0/1 tensor selecting the ``keep`` largest entries of
    the *non-negative* input (the callers pass ``scores.abs()``), where
    ``keep = round((1 - sparsity) * numel)``.

    Backward: pass the incoming gradient straight through, as if the masking
    operation were the identity (the edge-popup estimator of Ramanujan et al.,
    2020). Because the callers apply ``abs()`` *outside* this function,
    autograd multiplies the straight-through gradient by ``sign(score)`` on
    the way back to the raw scores. Before v3.0.2 the ``abs()`` was taken
    inside the forward (after ``detach()``), which dropped that sign: every
    negative score was updated in the wrong direction and SupSup stayed at
    chance level (the P0 run measured 9.4% on 10-way Permuted-MNIST and 5.1%
    on 20-way TinyImageNet). WSN, which trains its weights as well as its
    scores, still learned but with a corrupted mask-selection rule.
    """

    @staticmethod
    def forward(ctx, scores_abs, sparsity):
        keep = max(1, int(round((1.0 - float(sparsity)) * scores_abs.numel())))
        flat = scores_abs.detach().flatten()
        _, idx = flat.topk(keep)
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        return mask.view_as(scores_abs)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: identity w.r.t. its (non-negative)
        # input; ``sparsity`` is a non-tensor argument so its gradient is
        # ``None``.
        return grad_output, None


def subnet_mask(scores, sparsity):
    """Edge-popup mask of ``scores`` with the sign-aware straight-through path."""
    return _GetSubnet.apply(scores.abs(), sparsity)


def _init_frozen_weight(weight):
    """Kaiming-initialise a weight tensor in place and freeze it."""
    nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="relu")
    weight.requires_grad_(False)


def _make_scores(weight, num_tasks):
    """Create one learnable score tensor per task, shaped like ``weight``."""
    scores = nn.ParameterList(
        [nn.Parameter(torch.empty_like(weight)) for _ in range(num_tasks)]
    )
    for s in scores:
        nn.init.kaiming_uniform_(s, a=math.sqrt(5))
    return scores


class MultitaskMaskedLinear(nn.Linear):
    """``nn.Linear`` with a per-task score, optionally trainable weights.

    With ``train_weights=False`` (default) the weight is a fixed random
    backbone and only the scores are learned (SupSup). With
    ``train_weights=True`` the weight is trainable (WSN); the caller is then
    responsible for freezing weights won by earlier tasks. The bias is always
    initialised to zero and frozen, so a masked layer adds no effective bias;
    this keeps the SupSup/WSN comparison controlled (they differ only in
    whether the weights are learned).
    """

    def __init__(self, in_features, out_features, num_tasks, sparsity=0.5,
                 bias=True, train_weights=False):
        super().__init__(in_features, out_features, bias=bias)
        self.num_tasks = num_tasks
        self.sparsity = sparsity
        self.train_weights = train_weights
        if not train_weights:
            _init_frozen_weight(self.weight)        # SupSup: fixed random
        # WSN keeps the default (trainable) weight initialisation.
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            self.bias.requires_grad_(False)
        self.scores = _make_scores(self.weight, num_tasks)
        self.task = 0

    def set_task(self, task):
        """Select which task's mask to apply on subsequent forward passes."""
        self.task = task

    def forward(self, x):
        mask = subnet_mask(self.scores[self.task], self.sparsity)
        return F.linear(x, self.weight * mask, self.bias)


class MultitaskMaskedConv2d(nn.Conv2d):
    """``nn.Conv2d`` with a per-task score, optionally trainable weights.

    See :class:`MultitaskMaskedLinear` for the meaning of ``train_weights``.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_tasks,
                 sparsity=0.5, stride=1, padding=0, bias=True,
                 train_weights=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, bias=bias)
        self.num_tasks = num_tasks
        self.sparsity = sparsity
        self.train_weights = train_weights
        if not train_weights:
            _init_frozen_weight(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            self.bias.requires_grad_(False)
        self.scores = _make_scores(self.weight, num_tasks)
        self.task = 0

    def set_task(self, task):
        """Select which task's mask to apply on subsequent forward passes."""
        self.task = task

    def forward(self, x):
        mask = subnet_mask(self.scores[self.task], self.sparsity)
        return self._conv_forward(x, self.weight * mask, self.bias)


_MASKED_TYPES = (MultitaskMaskedLinear, MultitaskMaskedConv2d)


def convert_to_masked(module, num_tasks, sparsity, train_weights=False):
    """Recursively replace ``Linear`` / ``Conv2d`` leaves with masked versions.

    The replacement preserves each layer's shape. With ``train_weights=False``
    the weights become a fixed random backbone (SupSup); with
    ``train_weights=True`` they stay trainable (WSN). Non-leaf modules and
    functional ops (ReLU, pooling, flatten) are untouched, so a model whose
    ``forward`` calls its submodules by attribute keeps working unchanged.

    Args:
        module: The model (or submodule) to convert, modified in place.
        num_tasks: Number of per-task score sets to allocate in each layer.
        sparsity: Fraction of weights masked out per task (``0.5`` keeps half).
        train_weights: If ``True``, keep weights trainable (WSN); otherwise
            freeze them as a random backbone (SupSup).

    Returns:
        The same ``module``, for convenience.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, MultitaskMaskedLinear(
                child.in_features, child.out_features, num_tasks,
                sparsity=sparsity, bias=child.bias is not None,
                train_weights=train_weights))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, MultitaskMaskedConv2d(
                child.in_channels, child.out_channels, child.kernel_size,
                num_tasks, sparsity=sparsity, stride=child.stride,
                padding=child.padding, bias=child.bias is not None,
                train_weights=train_weights))
        else:
            convert_to_masked(child, num_tasks, sparsity, train_weights)
    return module


def set_task(model, task):
    """Set the active task on every masked submodule of ``model``."""
    for m in model.modules():
        if isinstance(m, _MASKED_TYPES):
            m.set_task(task)


def masked_score_parameters(model, task):
    """Yield the score parameters for ``task`` (the only trainable tensors)."""
    for m in model.modules():
        if isinstance(m, _MASKED_TYPES):
            yield m.scores[task]


def _binary_topk(scores, sparsity):
    """Plain (no-autograd) top-k binary mask of ``|scores|``.

    Mirrors :class:`_GetSubnet`'s forward selection; used by WSN to read out a
    task's winning subnetwork after training (for freezing and bookkeeping).
    """
    keep = max(1, int(round((1.0 - float(sparsity)) * scores.numel())))
    flat = scores.detach().abs().flatten()
    _, idx = flat.topk(keep)
    mask = torch.zeros_like(flat)
    mask[idx] = 1.0
    return mask.view_as(scores)


def masked_weights_named(model):
    """Yield ``(weight_name, weight_param)`` for every masked layer.

    The names match ``model.named_parameters`` keys (e.g. ``"fc1.weight"``),
    so they can index a per-weight accumulator dict (used by WSN to freeze
    weights won by earlier tasks).
    """
    for name, module in model.named_modules():
        if isinstance(module, _MASKED_TYPES):
            yield f"{name}.weight", module.weight


def current_subnet_mask(model, task):
    """Binary supermask each masked layer selects for ``task`` (by weight name)."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, _MASKED_TYPES):
            out[f"{name}.weight"] = _binary_topk(module.scores[task],
                                                 module.sparsity)
    return out
