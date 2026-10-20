"""Torch-free validation tests for seed aggregation and pairing."""

import importlib.util
import os


def _load_stats():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, os.pardir, "diltcl", "stats.py")
    spec = importlib.util.spec_from_file_location("stats_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_rejects_mismatched_metric_keys():
    s = _load_stats()
    _raises_value_error(lambda: s.aggregate_seeds([
        {"ACC": 1.0, "TA": 2.0}, {"ACC": 3.0}
    ]))


def test_rejects_unequal_pair_lengths():
    s = _load_stats()
    _raises_value_error(lambda: s.paired_significance([1, 2, 3], [1, 2]))


def test_valid_aggregation_keeps_sample_std():
    s = _load_stats()
    out = s.aggregate_seeds([{"ACC": 1.0}, {"ACC": 3.0}])
    assert out["ACC"]["mean"] == 2.0
    assert abs(out["ACC"]["std"] - 2 ** 0.5) < 1e-12


def test_constant_differences_give_undefined_not_nan():
    """Zero-variance differences make the paired t-test NaN; the record must
    say 'undefined' (None) so JSON artifacts written with allow_nan=False do
    not blow up, and the reported test name stays truthful."""
    import json
    s = _load_stats()
    out = s.paired_significance([1.0, 2.0, 3.0, 4.0, 5.0],
                                [2.0, 3.0, 4.0, 5.0, 6.0])
    assert out["test"] == "paired_t" and out["n"] == 5
    assert out["statistic"] is None and out.get("degenerate") is True
    json.dumps(out, allow_nan=False)          # must not raise (no inf/nan)
    normal = s.paired_significance([1.0, 2.5, 3.0, 4.2, 5.0],
                                   [2.0, 3.0, 4.5, 5.0, 6.5])
    assert normal["test"] == "paired_t" and 0.0 <= normal["p_value"] <= 1.0
    assert "degenerate" not in normal


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
