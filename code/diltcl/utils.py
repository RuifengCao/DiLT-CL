"""General utilities: reproducibility and device selection.

These helpers are intentionally dependency-light so that every other module
in the package can import them without pulling in heavy machinery.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Args:
        seed: The random seed to use everywhere.
    """
    # ER/DER++ use Python's ``random`` module for reservoir replacement and
    # sampling. Omitting this made nominally paired seeds depend on process and
    # method order.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> str:
    """Return the best available device as a string.

    Args:
        prefer_cuda: If True, use CUDA when a GPU is available.

    Returns:
        ``'cuda'`` if a GPU is available and preferred, otherwise ``'cpu'``.
    """
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"
