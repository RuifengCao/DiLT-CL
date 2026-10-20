"""Unit tests for the continual-learning metrics (NumPy only, no torch).

These validate the metric formulas on a hand-computed example so a refactor
can't silently change their meaning. Run with::

    python tests/test_metrics.py
"""

import importlib.util
import os
import sys

import numpy as np


def _load_metrics_module():
    """Load diltcl/metrics.py without importing the (torch-dependent) package.

    ``metrics.py`` only needs numpy and torch; we stub a minimal torch so the
    pure-numpy metric functions can be exercised in a torch-free environment.
    """
    if "torch" not in sys.modules:
        import types
        sys.modules["torch"] = types.ModuleType("torch")  # only needs to import
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, os.pardir, "diltcl", "metrics.py")
    spec = importlib.util.spec_from_file_location("metrics_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    print("=" * 60)
    print("Continual-learning metric tests (NumPy-only harness)")
    print("=" * 60)
    m = _load_metrics_module()

    print("  [running] constant_lower_triangular")
    fixed = m.constant_lower_triangular([91.0, 82.0, 73.0])
    expected_fixed = np.array([
        [91.0, 0.0, 0.0],
        [91.0, 82.0, 0.0],
        [91.0, 82.0, 73.0],
    ])
    assert np.array_equal(fixed, expected_fixed), fixed
    fixed_acc, fixed_bwt = m.compute_acc_bwt(fixed)
    assert abs(fixed_acc - 82.0) < 1e-9
    assert abs(fixed_bwt) < 1e-9

    # Lower-triangular + diagonal matrix as produced by the training loops:
    #   R[i][j] = accuracy on task j after training through task i.
    R = np.array([
        [90.0,  0.0,  0.0],
        [70.0, 85.0,  0.0],
        [60.0, 75.0, 80.0],
    ])

    print("  [running] learning_accuracy")
    la = m.learning_accuracy(R)
    assert abs(la - 85.0) < 1e-9, la                      # mean(90,85,80)

    print("  [running] compute_acc_bwt")
    acc, bwt = m.compute_acc_bwt(R)
    assert abs(acc - (60 + 75 + 80) / 3) < 1e-9, acc      # mean of final row
    assert abs(bwt - (-20.0)) < 1e-9, bwt                 # mean(-30,-10)

    print("  [running] compute_avg_forgetting")
    af = m.compute_avg_forgetting(R)
    assert abs(af - 20.0) < 1e-9, af                      # mean(30,10)

    print("  [running] summarize_run")
    s = m.summarize_run(R)
    assert set(s) == {"ACC", "BWT", "Forgetting", "LA"}, set(s)
    assert abs(s["LA"] - 85.0) < 1e-9

    print("  [running] compute_fwt (needs a full matrix with upper triangle)")
    # Full matrix: include pre-training accuracies in the upper triangle.
    R_full = np.array([
        [90.0, 12.0, 11.0],
        [70.0, 85.0, 14.0],
        [60.0, 75.0, 80.0],
    ])
    baseline = [5.0, 10.0, 8.0]
    # FWT = mean( R[0][1]-b1 , R[1][2]-b2 ) = mean(12-10, 14-8) = mean(2,6) = 4
    fwt = m.compute_fwt(R_full, baseline)
    assert abs(fwt - 4.0) < 1e-9, fwt

    print("-" * 60)
    print("All metric tests PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
