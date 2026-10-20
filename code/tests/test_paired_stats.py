"""``tools/paired_stats.py``: paired uncertainty for method comparisons (torch-free)."""

import importlib.util
import itertools
import math
import pathlib
import sys
import tempfile

import numpy as np
import pytest
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("paired_stats_under_test",
                                               ROOT / "tools" / "paired_stats.py")
ps = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ps
_spec.loader.exec_module(ps)


def test_sign_flip_is_exact_enumeration():
    diffs = np.array([1.0, 2.0, -0.5, 3.0, 0.25])
    observed = abs(diffs.mean())
    count = sum(abs((diffs * np.array(s)).mean()) >= observed - 1e-12
                for s in itertools.product((1, -1), repeat=5))
    assert ps.sign_flip_p(diffs) == count / 32
    # All differences of one sign: only the observed and the fully flipped
    # assignment are as extreme, hence 2 / 2**n.
    assert ps.sign_flip_p(np.array([0.1, 0.2, 0.3, 0.4, 0.5])) == 2 / 32


def test_holm_matches_definition_and_passes_none_through():
    p = [0.01, 0.04, None, 0.03, 0.5]
    adjusted = ps.holm(p)
    assert adjusted[2] is None
    # Sorted: 0.01 (x4), 0.03 (x3), 0.04 (x2), 0.5 (x1), with running maxima.
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[3] == pytest.approx(0.09)
    assert adjusted[1] == pytest.approx(0.09)
    assert adjusted[4] == pytest.approx(0.5)


@pytest.mark.parametrize("n", [5, 10])
def test_minimum_detectable_difference_has_the_requested_power(n):
    sd = 0.7
    delta = ps.min_detectable_difference(sd, n)
    df = n - 1
    t_crit = stats.t.ppf(0.975, df)
    nc = delta / (sd / math.sqrt(n))
    power = stats.nct.sf(t_crit, df, nc) + stats.nct.cdf(-t_crit, df, nc)
    assert power == pytest.approx(0.8, abs=1e-6)
    assert ps.min_detectable_difference(0.0, n) != ps.min_detectable_difference(0.0, n)  # NaN


def _payload(values_by_method, calibration=None, raw=None):
    seeds = [0, 1, 2, 3, 4]
    per_seed = {"cifar": {m: [{"ACC": v, "TA": v / 2, "MicroACC": v, "MicroOracleACC": v + 50}
                              for v in vals] for m, vals in values_by_method.items()}}
    if raw is not None:
        for m, vals in raw.items():
            for row, r in zip(per_seed["cifar"][m], vals):
                row["RawACC"], row["RawTA"] = r, r / 2
    return {"seeds": seeds, "benchmarks": ["cifar"], "per_seed": per_seed,
            "inference_calibration": calibration}


def test_compare_aligns_seeds_and_reports_a_t_interval():
    ref = [10.0, 11.0, 12.0, 13.0, 14.5]
    cmp_ = [9.0, 10.5, 11.0, 12.5, 13.0]
    row = ps.compare(_payload({"diltcl": ref, "wsn": cmp_}), "cifar", "ACC", "diltcl", "wsn")
    diffs = np.subtract(ref, cmp_)
    half = stats.t.ppf(0.975, 4) * diffs.std(ddof=1) / math.sqrt(5)
    assert row["mean_difference"] == pytest.approx(diffs.mean())
    assert row["ci95_low"] == pytest.approx(diffs.mean() - half)
    assert row["paired_t_p"] == pytest.approx(stats.ttest_rel(ref, cmp_).pvalue)
    assert row["positive_seeds"] == 5 and row["sign_flip_p"] == 2 / 32


def test_recoverable_fraction_is_computed_per_seed():
    payload = _payload({"diltcl": [20.0] * 5})
    values = ps.per_seed_values(payload, "cifar", "diltcl", "RecoverableFraction")
    assert values[0] == pytest.approx(100 * (70.0 - 20.0) / 80.0)


def test_families_are_adjusted_separately_and_tables_render():
    main = _payload({"diltcl": [10, 11, 12, 13, 14.5], "wsn": [9, 10.5, 11, 12.5, 13],
                     "supsup": [10, 11, 12, 13, 14]})
    calibrated = _payload({"diltcl": [8, 9, 10, 11, 12], "wsn": [9, 9, 9, 9, 9.5]},
                          calibration="score-logreg",
                          raw={"diltcl": [10, 11, 12, 13, 14.5], "wsn": [9, 9.2, 9.1, 9.3, 9.4]})
    rows = ps.build(main, main, "diltcl", ["wsn", "supsup"], ["wsn", "supsup"],
                    class_metrics=("ACC",), calibrated=[calibrated],
                    calibrated_methods=("wsn", "diltcl"))
    families = {r["family"] for r in rows}
    assert families == {"main", "calibration"}
    assert {r["family_size"] for r in rows if r["family"] == "main"} == {4}
    assert {r["family_size"] for r in rows if r["family"] == "calibration"} == {4}
    cal = [r for r in rows if r["family"] == "calibration" and r["method"] == "diltcl"
           and r["metric"] == "ACC"][0]
    assert cal["reference"] == "Logistic" and cal["mean_difference"] == pytest.approx(-2.1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        ps.write_tex(rows, tmp / "full.tex")
        ps.write_compact_tex(rows, tmp / "compact.tex", comparators=("supsup", "wsn"))
        ps.write_calibration_tex(rows, tmp / "cal.tex")
        compact = (tmp / "compact.tex").read_text()
        assert "\\label{tab:paired}" in compact and "$$" not in compact
        assert "Logistic" in (tmp / "cal.tex").read_text()
        assert "4 comparisons" in (tmp / "full.tex").read_text()


def test_csv_keeps_the_method_column_of_calibration_rows(tmp_path=None):
    import csv
    main = _payload({"diltcl": [10, 11, 12, 13, 14.5], "wsn": [9, 10.5, 11, 12.5, 13]})
    calibrated = _payload({"wsn": [9, 9, 9, 9, 9.5]}, calibration="score-bias",
                          raw={"wsn": [9, 9.2, 9.1, 9.3, 9.4]})
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "t.json").write_text(__import__("json").dumps(main))
        (tmp / "c.json").write_text(__import__("json").dumps(calibrated))
        # The calibrated input holds only WSN, so the other isolation methods
        # are missing: refused by default, recorded with --allow-missing.
        args = ["--task-il", str(tmp / "t.json"), "--task-comparators", "wsn",
                "--calibrated", str(tmp / "c.json"), "--out-csv", str(tmp / "out.csv")]
        assert ps.main(args) == 2 and not (tmp / "out.csv").exists()
        assert ps.main([*args, "--allow-missing"]) == 0
        with (tmp / "out.csv").open() as handle:
            rows = list(csv.DictReader(handle))
    cal = [r for r in rows if r["family"] == "calibration"]
    assert cal and all(r["method"] == "wsn" for r in cal)
    assert [r for r in rows if r["family"] == "main"][0]["method"] == ""


def test_missing_comparators_or_seeds_are_refused():
    full = _payload({"diltcl": [10, 11, 12, 13, 14.5], "wsn": [9, 10.5, 11, 12.5, 13]})
    with pytest.raises(ps.MissingComparison, match="supsup missing"):
        ps.build(full, None, "diltcl", ["wsn", "supsup"], [])
    rows = ps.build(full, None, "diltcl", ["wsn", "supsup"], [], allow_missing=True)
    assert len(rows) == 1 and rows[0]["family_size"] == 1
    # A comparator that lacks a seed: alignment by value drops it -> refused.
    short = _payload({"diltcl": [10, 11, 12, 13, 14.5], "wsn": [9, 10.5, 11, 12.5, 13]})
    short["seeds"] = [0, 1, 2, 3, 4]
    values = ps.per_seed_values(short, "cifar", "wsn", "ACC")
    assert len(values) == 5
    original = ps.per_seed_values

    def drop_seed(payload, benchmark, method, metric):
        out = original(payload, benchmark, method, metric)
        if method == "wsn":
            out.pop(4)
        return out
    ps.per_seed_values = drop_seed
    try:
        with pytest.raises(ps.MissingComparison, match="aligned seeds"):
            ps.compare(short, "cifar", "ACC", "diltcl", "wsn")
    finally:
        ps.per_seed_values = original


def test_mdd_for_very_few_seeds_and_the_ten_seed_caption():
    assert math.isfinite(ps.min_detectable_difference(1.0, 3))
    value = ps.min_detectable_difference(1.0, 2)      # needs a very large noncentrality
    assert value != value or value > 10                # NaN or a finite, large value
    assert "five seeds" in ps._sign_flip_floor([5]) and "0.0625" in ps._sign_flip_floor([5])
    assert "ten seeds" in ps._sign_flip_floor([10]) and "0.001953" in ps._sign_flip_floor([10])


def _incomplete(payload, method, drop_seed):
    """What ``merge_results.py --allow-incomplete`` writes when one seed is absent."""
    rows = payload["per_seed"]["cifar"][method]
    keep = [s for s in payload["seeds"] if s != drop_seed]
    payload["per_seed"]["cifar"][method] = [rows[payload["seeds"].index(s)] for s in keep]
    payload.setdefault("per_seed_run_records", {}).setdefault("cifar", {})[method] = {
        str(s): {} for s in keep}
    return payload


def test_rows_of_an_incomplete_merge_are_aligned_to_their_seeds():
    payload = _incomplete(_payload({"diltcl": [10, 11, 12, 13, 14.5],
                                    "wsn": [9, 10.5, 11, 12.5, 13]}), "wsn", drop_seed=1)
    assert ps.per_seed_values(payload, "cifar", "wsn", "ACC") == {0: 9, 2: 11, 3: 12.5, 4: 13}
    with pytest.raises(ps.MissingComparison, match="aligned seeds"):
        ps.compare(payload, "cifar", "ACC", "diltcl", "wsn")
    row = ps.compare(payload, "cifar", "ACC", "diltcl", "wsn", allow_missing=True)
    assert row["seeds"] == [0, 2, 3, 4] and row["mean_difference"] == pytest.approx(
        np.mean([10 - 9, 12 - 11, 13 - 12.5, 14.5 - 13]))
    # Rows that cannot be aligned are a refused comparison, not a traceback.
    del payload["per_seed_run_records"]
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "t.json"
        path.write_text(__import__("json").dumps(payload))
        assert ps.main(["--task-il", str(path), "--task-comparators", "wsn"]) == 2


def test_router_comparisons_check_their_seeds():
    calibrated = _incomplete(_payload({"wsn": [9, 9, 9, 9, 9.5]}, calibration="score-bias",
                                      raw={"wsn": [9, 9.2, 9.1, 9.3, 9.4]}), "wsn", drop_seed=3)
    with pytest.raises(ps.MissingComparison, match="matched raw"):
        ps.compare_router(calibrated, "cifar", "ACC", "wsn")
    row = ps.compare_router(calibrated, "cifar", "ACC", "wsn", allow_missing=True)
    assert row["seeds"] == [0, 1, 2, 4] and row["n_seeds"] == 4
    assert row["mean_difference"] == pytest.approx(np.mean([0.0, -0.2, -0.1, 0.1]))


def test_captions_name_the_comparisons_holm_actually_adjusted():
    # wsn and supsup differ from diltcl by a constant: zero variance, no p-value.
    main = _payload({"diltcl": [10, 11, 12, 13, 14.5], "wsn": [9, 10.5, 11, 12.5, 13],
                     "supsup": [9, 10, 11, 12, 13.5]})
    rows = ps.build(main, None, "diltcl", ["wsn", "supsup"], [])
    assert [r["holm_tests"] for r in rows] == [1, 1] and rows[0]["family_size"] == 2
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "full.tex"
        ps.write_tex(rows, path)
        assert "1 comparison of this table with a defined $p$ (of 2)" in path.read_text()
