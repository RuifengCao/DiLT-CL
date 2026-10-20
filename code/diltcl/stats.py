"""Aggregation and significance testing across random seeds.

A single run proves nothing to a reviewer; results must be reported as
mean +/- std over several seeds, and improvements over baselines should be
checked for statistical significance. This module provides:

* :func:`aggregate_seeds`  - mean / std of each metric across seeds.
* :func:`paired_significance` - a paired test (Wilcoxon signed-rank, or a
  paired t-test for very small sample sizes) between two methods' per-seed
  scores.
* :func:`format_comparison_table` - a readable text table with significance
  markers against a chosen reference method.
"""

import numpy as np


def aggregate_seeds(per_seed_metrics):
    """Aggregate a list of per-seed metric dicts into mean/std per metric.

    Args:
        per_seed_metrics: List of dicts (one per seed) with identical keys,
            e.g. ``[{'ACC': 42.4, 'BWT': -1.1}, {'ACC': 41.0, 'BWT': -0.8}]``.

    Returns:
        Dict mapping each metric to ``{'mean': float, 'std': float,
        'values': [...]}``.
    """
    if not per_seed_metrics:
        return {}
    keys = list(per_seed_metrics[0].keys())
    expected = set(keys)
    for i, metrics in enumerate(per_seed_metrics):
        if set(metrics) != expected:
            raise ValueError(
                f"metric keys differ at seed index {i}: "
                f"expected {sorted(expected)}, got {sorted(metrics)}")
    out = {}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics]
        arr = np.asarray(vals, dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite values for metric {k!r}: {vals}")
        out[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "values": vals,
        }
    return out


def paired_significance(scores_a, scores_b):
    """Paired significance test between two methods' per-seed scores.

    Uses the Wilcoxon signed-rank test when there are enough pairs, otherwise
    falls back to a paired t-test. Both inputs must be the same length and
    aligned by seed.

    Args:
        scores_a: Per-seed scores for method A.
        scores_b: Per-seed scores for method B.

    Returns:
        Dict with ``test`` (name), ``statistic``, ``p_value`` and ``n``.
        ``p_value`` is ``None`` when the test is undefined (e.g. all
        differences zero, or fewer than two pairs). Non-finite statistics
        (zero-variance differences) are reported as ``None`` with
        ``degenerate: True``.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.size != b.size:
        raise ValueError(
            f"paired scores must have equal length, got {a.size} and {b.size}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("paired scores must be finite")
    n = a.size

    if n < 2 or np.allclose(a, b):
        return {"test": "none", "statistic": None, "p_value": None, "n": n}

    try:
        from scipy import stats
    except Exception:  # noqa: BLE001 - scipy optional at import time.
        return {"test": "unavailable", "statistic": None,
                "p_value": None, "n": n}

    diffs = a - b
    # Wilcoxon needs >= ~6 non-zero pairs to be meaningful; below that, t-test.
    if np.count_nonzero(diffs) >= 6:
        stat, p = stats.wilcoxon(a, b)
        name = "wilcoxon"
    else:
        stat, p = stats.ttest_rel(a, b)
        name = "paired_t"
    # Differences with zero variance (every seed moved by exactly the same
    # amount) make the paired t statistic infinite (SciPy reports +-inf with
    # p = 0). Non-finite values cannot be written into the JSON artifacts
    # (allow_nan=False), so they become None and the record is flagged as
    # degenerate; readers should not quote such a p-value as evidence.
    degenerate = not (np.isfinite(stat) and np.isfinite(p))
    stat = float(stat) if np.isfinite(stat) else None
    p = float(p) if np.isfinite(p) else None
    record = {"test": name, "statistic": stat, "p_value": p, "n": n}
    if degenerate:
        record["degenerate"] = True
    return record


def _sig_marker(p_value):
    """Return a conventional significance marker for a p-value."""
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def format_comparison_table(aggregated_by_method, reference=None,
                            metric="ACC", order=None):
    """Build a readable comparison table for one metric across methods.

    Args:
        aggregated_by_method: ``{method_name: aggregate_seeds(...) result}``.
        reference: Optional method name to test every other method against
            (significance computed on the chosen ``metric``).
        metric: Which metric to tabulate (default ``'ACC'``).
        order: Optional explicit ordering of method names.

    Returns:
        A multi-line string.
    """
    methods = order or list(aggregated_by_method.keys())
    lines = []
    header = f"{'Method':<24} {metric + ' (mean +/- std)':>22}"
    if reference is not None:
        header += f"   {'vs ' + reference:>16}"
    lines.append(header)
    lines.append("-" * len(header))

    ref_vals = None
    if reference is not None and reference in aggregated_by_method:
        ref_vals = aggregated_by_method[reference].get(metric, {}).get("values")

    for m in methods:
        agg = aggregated_by_method.get(m, {})
        cell = agg.get(metric)
        if cell is None:
            lines.append(f"{m:<24} {'n/a':>22}")
            continue
        value_str = f"{cell['mean']:6.2f} +/- {cell['std']:4.2f}"
        row = f"{m:<24} {value_str:>22}"
        if reference is not None and ref_vals is not None and m != reference:
            sig = paired_significance(cell["values"], ref_vals)
            marker = "" if sig.get("degenerate") else _sig_marker(sig["p_value"])
            p_str = ("p=%.3f %s" % (sig["p_value"], marker)
                     if sig["p_value"] is not None else "n/a")
            row += f"   {p_str:>16}"
        elif reference is not None and m == reference:
            row += f"   {'(reference)':>16}"
        lines.append(row)

    if reference is not None:
        lines.append("")
        lines.append("Significance vs reference on "
                     f"{metric}: *** p<0.001, ** p<0.01, * p<0.05, ns = "
                     "not significant.")
    return "\n".join(lines)
