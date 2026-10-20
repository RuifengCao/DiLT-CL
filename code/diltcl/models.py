"""Neural network architectures used across the experiments.

  * ``SimpleCNN`` - a small conv net for CIFAR-style images.
  * ``SimpleMLP`` - a small MLP for flattened inputs (e.g. Permuted MNIST).
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10/100 images.

    Architecture: 3 conv layers with max pooling + 2 FC layers.
    Input:  ``(batch, 3, 32, 32)``.
    Output: ``(batch, num_classes)``.

    Args:
        num_classes: Number of output classes per task.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass returning logits."""
        x = self.pool(self.relu(self.conv1(x)))  # -> (B, 32, 16, 16)
        x = self.pool(self.relu(self.conv2(x)))  # -> (B, 64, 8, 8)
        x = self.pool(self.relu(self.conv3(x)))  # -> (B, 128, 4, 4)
        x = x.view(x.size(0), -1)                # -> (B, 128*4*4)
        x = self.relu(self.fc1(x))               # -> (B, 256)
        x = self.fc2(x)                          # -> (B, num_classes)
        return x


class AdaptiveCNN(nn.Module):
    """Input-size- and channel-agnostic conv net for the harder benchmarks.

    Same conv/fc layer names as :class:`SimpleCNN` (so the masking / DiLT-CL
    machinery treats it identically), but an ``AdaptiveAvgPool2d`` fixes the
    feature-map size before the classifier. This lets one architecture serve
    both 32x32 inputs (CIFAR, 5-datasets) and 64x64 inputs (TinyImageNet);
    grayscale datasets are expanded to 3 channels upstream so ``in_channels``
    stays 3 across a 5-datasets sequence.

    Args:
        num_classes: Number of output classes (per task for task-IL, or the
            global class count for class-IL).
        in_channels: Input channels (3 throughout this project).
        pool_to: Spatial size the adaptive pool maps to (feature dim into
            ``fc1`` is ``128 * pool_to * pool_to``).
    """

    def __init__(self, num_classes=10, in_channels=3, pool_to=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive = nn.AdaptiveAvgPool2d((pool_to, pool_to))
        self.fc1 = nn.Linear(128 * pool_to * pool_to, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass returning logits (accepts 32x32 or 64x64 inputs)."""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive(x)                     # -> (B, 128, pool_to, pool_to)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleMLP(nn.Module):
    """Simple MLP for flattened image inputs (e.g. Permuted MNIST).

    Architecture: 3 FC layers with ReLU activations.
    Input:  ``(batch, input_size)`` where ``input_size = 784`` for MNIST.
    Output: ``(batch, num_classes)``.

    Args:
        input_size: Flattened input dimension.
        hidden_size: Hidden layer dimension.
        num_classes: Number of output classes.
    """

    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass returning logits."""
        x = x.view(x.size(0), -1)    # Flatten input
        x = self.relu(self.fc1(x))   # -> (B, hidden_size)
        x = self.relu(self.fc2(x))   # -> (B, hidden_size)
        x = self.fc3(x)              # -> (B, num_classes)
        return x


class MultiHeadModel(nn.Module):
    """Standard multi-head task-IL wrapper: a shared trunk plus one head per task.

    Task-incremental learning conventionally gives every task its **own**
    classifier head, so a new task cannot overwrite an earlier task's output
    layer. The v2.0/v2.1 baselines instead reused a single C-way head across
    all tasks, which depresses their accuracy and makes the numbers
    incomparable with the task-IL literature. That is a protocol choice, not a
    property of the methods, so this wrapper supplies the standard alternative.

    The wrapped backbone's final linear layer is replaced by an identity, and
    ``num_tasks`` fresh heads are attached on top of the resulting features.
    ``set_task(t)`` selects the active head; ``forward`` then returns only that
    head's logits, which keeps every downstream trainer, ``evaluate`` call and
    metric definition unchanged.

    Args:
        backbone: A model whose last attribute in ``head_attr`` is the linear
            classifier (``SimpleCNN``/``AdaptiveCNN`` use ``fc2``, ``SimpleMLP``
            uses ``fc3``).
        num_tasks: Number of task-specific heads to create.
        classes_per_task: Output units per head.
        head_attr: Name of the backbone's classifier attribute. When ``None``
            the last ``nn.Linear`` child is detected automatically.
    """

    def __init__(self, backbone, num_tasks, classes_per_task, head_attr=None):
        super().__init__()
        if head_attr is None:
            head_attr = self._detect_head_attr(backbone)
        head = getattr(backbone, head_attr)
        if not isinstance(head, nn.Linear):
            raise TypeError(
                f"backbone attribute '{head_attr}' is {type(head).__name__}, "
                "expected nn.Linear")
        in_features = head.in_features
        setattr(backbone, head_attr, nn.Identity())
        self.backbone = backbone
        self.head_attr = head_attr
        self.num_tasks = int(num_tasks)
        self.classes_per_task = int(classes_per_task)
        self.heads = nn.ModuleList([
            nn.Linear(in_features, classes_per_task)
            for _ in range(self.num_tasks)])
        self.active_task = 0

    @staticmethod
    def _detect_head_attr(backbone):
        """Return the attribute name of the backbone's final linear layer."""
        linear_names = [name for name, module in backbone.named_children()
                        if isinstance(module, nn.Linear)]
        if not linear_names:
            raise TypeError("backbone has no nn.Linear child to use as a head")
        return linear_names[-1]

    def set_task(self, task_id):
        """Select the head used by subsequent forward passes."""
        if not 0 <= int(task_id) < self.num_tasks:
            raise IndexError(
                f"task_id {task_id} outside [0, {self.num_tasks})")
        self.active_task = int(task_id)

    def forward(self, x):
        """Forward through the shared trunk and the active task's head."""
        return self.heads[self.active_task](self.features(x))

    # -- Shared-feature helpers -------------------------------------------
    #
    # The trunk output does not depend on which head is active, so callers
    # that need several heads on the same batch (multi-head LwF distillation,
    # per-item head routing of replay samples) can compute the features once
    # and apply cheap linear heads, instead of re-running the whole network
    # per head as the v2 trainers did. This is a pure refactoring of the same
    # function: ``head_logits(features(x), t) == set_task(t); forward(x)``.

    def features(self, x):
        """Trunk features ``[B, F]`` shared by every head."""
        features = self.backbone(x)
        return features.view(features.size(0), -1)

    def head_logits(self, features, task_id):
        """Logits of head ``task_id`` on precomputed trunk features."""
        return self.heads[int(task_id)](features)

    def stacked_head_logits(self, features, task_ids):
        """Logits of several heads at once: ``[B, len(task_ids), C]``.

        Implemented as one matmul against the stacked head weights, so the
        cost is independent of the number of heads (used by LwF distillation).
        """
        task_ids = [int(t) for t in task_ids]
        weight = torch.stack([self.heads[t].weight for t in task_ids])   # [T, C, F]
        bias = torch.stack([self.heads[t].bias for t in task_ids])       # [T, C]
        flat = features @ weight.reshape(-1, weight.shape[-1]).t()       # [B, T*C]
        return flat.view(features.shape[0], len(task_ids), -1) + bias.unsqueeze(0)

    def forward_routed(self, x, task_ids):
        """Forward a batch whose rows belong to different tasks.

        Args:
            x: Inputs ``[B, ...]``.
            task_ids: ``LongTensor[B]`` with the head to use per row; ``-1``
                selects the currently active head.

        Returns:
            Logits ``[B, C]`` where each row comes from its own task's head.
            Equivalent to running each task's rows separately with
            ``set_task`` and concatenating in row order, but with a single
            trunk forward and a constant number of kernels.

        Implementation note: every head is applied to every row with one
        matmul (``T * C`` outputs per row, negligible for these head sizes)
        and the wanted block is picked with ``gather``. A per-row gather of
        the head *weights* followed by ``bmm`` would be equally cheap, but
        its backward accumulates into the shared weight tensor with atomic
        adds when several rows share a head, which is not bitwise
        deterministic on CUDA. Here every gradient destination receives
        exactly one contribution, so repeated runs stay bit-identical.
        """
        features = self.features(x)
        heads = torch.where(task_ids < 0,
                            torch.full_like(task_ids, self.active_task),
                            task_ids)
        all_logits = self.stacked_head_logits(features, range(self.num_tasks))
        index = heads.view(-1, 1, 1).expand(-1, 1, all_logits.shape[-1])
        return all_logits.gather(1, index).squeeze(1)
