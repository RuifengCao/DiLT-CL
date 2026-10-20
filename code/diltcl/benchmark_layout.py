"""Pure, torch-free layout logic for the continual-learning benchmarks.

Splitting classes into tasks, building task-IL vs class-IL label maps, parsing
TinyImageNet validation annotations, and computing 5-datasets global-label
offsets are all simple but easy to get subtly wrong. Keeping them here -- free
of any torch/torchvision dependency -- lets :mod:`diltcl.hard_benchmarks` reuse
them while they are unit-tested directly (``tests/test_benchmark_layout.py``).
"""


def split_classes(num_classes, num_tasks, classes_per_task=None):
    """Partition ``range(num_classes)`` into per-task class-index lists.

    Args:
        num_classes: Total number of classes in the dataset.
        num_tasks: Number of sequential tasks.
        classes_per_task: Classes per task. If ``None``, it is
            ``num_classes // num_tasks`` (which must divide evenly). If given,
            ``num_tasks * classes_per_task`` must not exceed ``num_classes``
            (a smaller product simply uses the first that-many classes -- handy
            for quick ``--fast`` runs on a subset).

    Returns:
        List of ``num_tasks`` lists, each holding ``classes_per_task`` global
        class indices in increasing order.
    """
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if classes_per_task is None:
        if num_classes % num_tasks != 0:
            raise ValueError(
                f"{num_classes} classes do not divide evenly into {num_tasks} "
                "tasks; pass classes_per_task explicitly")
        classes_per_task = num_classes // num_tasks
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be positive")
    if num_tasks * classes_per_task > num_classes:
        raise ValueError(
            f"num_tasks*classes_per_task = {num_tasks * classes_per_task} "
            f"exceeds num_classes = {num_classes}")
    return [list(range(t * classes_per_task, (t + 1) * classes_per_task))
            for t in range(num_tasks)]


def task_label_map(target_classes, class_il):
    """Map a task's global class indices to head labels.

    * task-IL (``class_il=False``): remap to ``0..len-1`` (a shared C-way head).
    * class-IL (``class_il=True``): identity map (a single global head spanning
      every class).

    Returns:
        Dict ``{global_class_index: head_label}``.
    """
    if class_il:
        return {c: c for c in target_classes}
    return {c: i for i, c in enumerate(target_classes)}


def parse_tinyimagenet_val_annotations(text):
    """Parse TinyImageNet ``val_annotations.txt`` into ``{filename: wnid}``.

    Each line is tab-separated as ``<filename>\\t<wnid>\\t<bbox...>``; only the
    first two fields are needed. Blank lines are ignored.

    Args:
        text: Full file contents.

    Returns:
        Ordered dict-like ``{image_filename: wnid}`` (a plain dict; insertion
        order is preserved in Python 3.7+).
    """
    mapping = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed val annotation line: {line!r}")
        mapping[parts[0]] = parts[1]
    return mapping


def five_datasets_offsets(num_datasets, classes_per_dataset=10):
    """Global-label offsets for the 5-datasets class-IL space.

    Dataset ``t`` (a task) occupies the contiguous global-label block
    ``[t*classes_per_dataset, (t+1)*classes_per_dataset)``, so its native label
    ``y`` becomes ``y + offsets[t]``.

    Returns:
        List ``[0, C, 2C, ...]`` of length ``num_datasets``.
    """
    if num_datasets <= 0:
        raise ValueError("num_datasets must be positive")
    if classes_per_dataset <= 0:
        raise ValueError("classes_per_dataset must be positive")
    return [t * classes_per_dataset for t in range(num_datasets)]
