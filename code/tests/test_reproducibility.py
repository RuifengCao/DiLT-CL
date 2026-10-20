"""Torch-free regression tests for the global seeding contract."""

import importlib.util
import os
import random
import sys
import types

import numpy as np


class _Cuda:
    @staticmethod
    def manual_seed_all(seed):
        return None

    @staticmethod
    def is_available():
        return False


class _Cudnn:
    deterministic = False
    benchmark = True


def _load_utils():
    """Load ``diltcl.utils`` against a fake ``torch`` module.

    The real ``torch`` (if any) is put back afterwards: leaving the fake in
    ``sys.modules`` broke every later test in the same pytest session that
    relied on ``importlib.util.find_spec("torch")``.
    """
    fake = types.ModuleType("torch")
    fake.manual_seed = lambda seed: None
    fake.cuda = _Cuda()
    fake.backends = types.SimpleNamespace(cudnn=_Cudnn())
    real_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, os.pardir, "diltcl", "utils.py")
        spec = importlib.util.spec_from_file_location("utils_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        else:
            sys.modules.pop("torch", None)
    return module


def test_python_and_numpy_rngs_repeat():
    u = _load_utils()
    u.set_seed(17)
    first = ([random.random() for _ in range(5)], np.random.random(5))
    u.set_seed(17)
    second = ([random.random() for _ in range(5)], np.random.random(5))
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])


if __name__ == "__main__":
    test_python_and_numpy_rngs_repeat()
    print("PASS test_python_and_numpy_rngs_repeat")
