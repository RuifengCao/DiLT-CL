#!/usr/bin/env python3
"""Paired uncertainty for method comparisons in merged artifacts.

For every requested (benchmark, metric, reference, comparator) the tool
computes, from the per-seed values of a merged artifact (seeds aligned by
value, never by position):

* the mean paired difference ``reference - comparator`` and its sample SD;
* a two-sided 95% Student-t confidence interval for the mean difference;
* the paired t-test p-value;
* the exact two-sided sign-flip (randomisation) p-value, enumerating all
  ``2**n`` sign assignments of the differences -- distribution-free; with
  five pairs its smallest attainable value is ``2/32 = 0.0625``;
* Holm-adjusted versions of both p-values over every comparison in the run
  (one family per invocation);
* the minimum detectable difference at 80% power for a two-sided paired
  t-test at alpha = 0.05 given the observed SD (exact, via the noncentral t
  distribution) -- what this number of seeds can and cannot resolve.

Percentile bootstrap intervals are deliberately not reported: with five seeds
there are only 126 distinct resamples, and the percentile interval is known
to undercover badly at that sample size.

Derived metric ``RecoverableFraction`` = ``100 * (MicroOracleACC - MicroACC) /
(100 - MicroACC)`` is the share of current class-IL errors that perfect task
routing would remove with the experts held fixed (class-IL artifacts only).
``OracleACC`` is not compared by default: for the uncalibrated full-data group
it equals the task-IL accuracy of the same experts seed by seed, so it would
duplicate the task-IL rows in the multiplicity family.

Two families are adjusted separately (Holm within each):

* ``main`` -- the reference method (DiLT-CL) against each comparator: task-IL
  ACC per benchmark; class-IL ACC, TA and recoverable fraction;
* ``calibration`` (``--calibrated``, repeatable) -- each calibrated router
  against its matched raw router on the same experts (TA and ACC per
  isolation method), an exploratory family.

Torch-free; needs NumPy and SciPy.

Example::

    python tools/paired_stats.py \\
        --task-il RUN/merged/p0_task_il_multi.json \\
        --class-il RUN/merged/p0_class_entropy.json \\
        --calibrated RUN/merged/p0_class_entropy_cal.json \\
        --out-csv paired_stats.csv --out-tex paired_stats.tex \\
        --out-compact-tex paired_main.tex --out-calibration-tex paired_calibration.tex
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ISOLATION = ("packnet", "supsup", "wsn")
LABEL = {"finetune": "Finetune", "ewc": "EWC", "lwf": "LwF", "er": "ER",
         "derpp": "DER++", "packnet": "PackNet", "supsup": "SupSup",
         "wsn": "WSN", "diltcl": "DiLT-CL", "independent": "Independent"}
BENCH = {"cifar": "CIFAR-100", "pmnist": "P-MNIST", "tinyimagenet": "TinyImageNet",
         "fivedatasets": "5-datasets"}


def per_seed_values(payload: dict, benchmark: str, method: str, metric: str) -> dict[int, float]:
    """Map seed -> metric value (derived metrics computed per seed)."""
    seeds = payload["seeds"]
    rows = payload["per_seed"][benchmark][method]
    if len(rows) != len(seeds):
        raise ValueError(f"{benchmark}/{method}: {len(rows)} rows for {len(seeds)} seeds")
    out = {}
    for seed, row in zip(seeds, rows):
        if metric == "RecoverableFraction":
            acc, oracle = float(row["MicroACC"]), float(row["MicroOracleACC"])
            if acc >= 100.0:
                raise ValueError("RecoverableFraction undefined at 100% accuracy")
            out[int(seed)] = 100.0 * (oracle - acc) / (100.0 - acc)
        else:
            out[int(seed)] = float(row[metric])
    return out


def sign_flip_p(diffs: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value for H0: differences symmetric about 0."""
    n = diffs.size
    observed = abs(diffs.mean())
    count = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        if abs((diffs * np.asarray(signs)).mean()) >= observed - 1e-12:
            count += 1
    return count / 2 ** n


def min_detectable_difference(sd: float, n: int, alpha: float = 0.05,
                              power: float = 0.8) -> float:
    """Smallest |mean difference| a two-sided paired t-test detects with ``power``."""
    if n < 2 or not np.isfinite(sd) or sd <= 0:
        return float("nan")
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    def achieved(nc):
        # Power of the two-sided test at noncentrality ``nc`` (monotone in nc).
        # The lower tail is negligible for nc > 0 and SciPy's nct.cdf can
        # return NaN far out in it, so a NaN there is read as zero.
        lower = stats.nct.cdf(-t_crit, df, nc)
        lower = 0.0 if not np.isfinite(lower) else lower
        return stats.nct.sf(t_crit, df, nc) + lower - power

    # Bracket the noncentrality: the required value is about 4.4 for n = 5
    # and 3.0 for n = 10 at 80% power, well inside [0, 10].
    nc = optimize.brentq(achieved, 1e-9, 10.0)
    return float(nc * sd / math.sqrt(n))


def holm(pvalues: list[float | None]) -> list[float | None]:
    """Holm step-down adjustment; ``None`` entries are passed through."""
    indexed = [(p, i) for i, p in enumerate(pvalues) if p is not None]
    indexed.sort()
    m = len(indexed)
    adjusted = [None] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def compare(payload: dict, benchmark: str, metric: str, reference: str,
            comparator: str) -> dict:
    ref = per_seed_values(payload, benchmark, reference, metric)
    cmp_ = per_seed_values(payload, benchmark, comparator, metric)
    seeds = sorted(set(ref) & set(cmp_))
    diffs = np.array([ref[s] - cmp_[s] for s in seeds], dtype=np.float64)
    n = diffs.size
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        t_p = float(stats.ttest_rel([ref[s] for s in seeds], [cmp_[s] for s in seeds]).pvalue)
    else:
        half, t_p = float("nan"), None
    return {
        "benchmark": benchmark, "metric": metric, "reference": reference,
        "comparator": comparator, "n_seeds": n, "seeds": seeds,
        "reference_mean": float(np.mean([ref[s] for s in seeds])),
        "comparator_mean": float(np.mean([cmp_[s] for s in seeds])),
        "mean_difference": mean, "sd_difference": sd,
        "ci95_low": mean - half, "ci95_high": mean + half,
        "positive_seeds": int(np.sum(diffs > 0)), "negative_seeds": int(np.sum(diffs < 0)),
        "paired_t_p": t_p, "sign_flip_p": sign_flip_p(diffs) if n else None,
        "min_detectable_difference_80pct": min_detectable_difference(sd, n),
    }


CLASS_METRICS = ("ACC", "TA", "RecoverableFraction")
ROUTER = {"score-logreg": "Logistic", "score-bias": "Bias-CE", "score-bias-acc": "Bias-Acc"}


def compare_router(payload: dict, benchmark: str, metric: str, method: str) -> dict:
    """Calibrated router minus the matched raw router, same experts and seeds."""
    seeds = payload["seeds"]
    rows = payload["per_seed"][benchmark][method]
    cal = np.array([float(r[metric]) for r in rows])
    raw = np.array([float(r["Raw" + metric]) for r in rows])
    diffs = cal - raw
    n = diffs.size
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        t_p = float(stats.ttest_rel(cal, raw).pvalue)
    else:
        half, t_p = float("nan"), None
    return {
        "benchmark": benchmark, "metric": metric,
        "reference": ROUTER.get(payload.get("inference_calibration"),
                                payload.get("inference_calibration")),
        "comparator": "matched raw", "method": method, "n_seeds": n,
        "seeds": [int(s) for s in seeds],
        "reference_mean": float(cal.mean()), "comparator_mean": float(raw.mean()),
        "mean_difference": mean, "sd_difference": sd,
        "ci95_low": mean - half, "ci95_high": mean + half,
        "positive_seeds": int(np.sum(diffs > 0)), "negative_seeds": int(np.sum(diffs < 0)),
        "paired_t_p": t_p, "sign_flip_p": sign_flip_p(diffs) if n else None,
        "min_detectable_difference_80pct": min_detectable_difference(sd, n),
    }


def _adjust(rows: list[dict]) -> None:
    for family in sorted({r["family"] for r in rows}):
        members = [r for r in rows if r["family"] == family]
        t_adj = holm([r["paired_t_p"] for r in members])
        s_adj = holm([r["sign_flip_p"] for r in members])
        for r, ta, sa in zip(members, t_adj, s_adj):
            r["paired_t_p_holm"] = ta
            r["sign_flip_p_holm"] = sa
            r["family_size"] = len(members)


def build(task_il: dict | None, class_il: dict | None, reference: str,
          comparators_task: list[str], comparators_class: list[str],
          class_metrics=CLASS_METRICS, calibrated: list[dict] | None = None,
          calibrated_methods=ISOLATION + ("diltcl",)) -> list[dict]:
    rows = []
    if task_il is not None:
        for b in task_il["benchmarks"]:
            for c in comparators_task:
                if c in task_il["per_seed"][b] and reference in task_il["per_seed"][b]:
                    rows.append({"family": "main", "protocol": "task-IL",
                                 **compare(task_il, b, "ACC", reference, c)})
    if class_il is not None:
        for b in class_il["benchmarks"]:
            for metric in class_metrics:
                for c in comparators_class:
                    if c in class_il["per_seed"][b] and reference in class_il["per_seed"][b]:
                        rows.append({"family": "main", "protocol": "class-IL",
                                     **compare(class_il, b, metric, reference, c)})
    for payload in calibrated or []:
        for b in payload["benchmarks"]:
            for metric in ("TA", "ACC"):
                for m in calibrated_methods:
                    if m in payload["per_seed"][b]:
                        rows.append({"family": "calibration", "protocol": "class-IL",
                                     **compare_router(payload, b, metric, m)})
    _adjust(rows)
    return rows


def _fmt(p, tex=True):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "--"
    if p >= 0.001:
        return f"{p:.3f}"
    return "$<$0.001" if tex else "<0.001"


def _signed(x: float) -> str:
    return f"${x:+.2f}$".replace("+", "{+}").replace("-", "{-}")


def write_tex(rows: list[dict], path: Path) -> None:
    """Full table of the ``main`` family (appendix)."""
    rows = [r for r in rows if r["family"] == "main"]
    n_seeds = sorted({r["n_seeds"] for r in rows})
    lines = [
        "% Generated by tools/paired_stats.py -- do not edit by hand.",
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        ("\\caption{Paired differences, DiLT-CL minus each comparator, over "
         f"{'/'.join(map(str, n_seeds))} aligned seeds (percentage points). "
         "CI: unadjusted two-sided 95\\% Student-$t$ interval of the mean paired difference. "
         "$p_t$: paired $t$ test, Holm-adjusted over the "
         f"{rows[0]['family_size'] if rows else 0} comparisons of this table. "
         "$p_{\\mathrm{sf}}$: exact sign-flip test, unadjusted (its smallest attainable value is "
         "$2/2^n$, $0.0625$ for five seeds); $+/-$: seeds with a positive/negative difference. "
         "MDD: smallest true difference an unadjusted two-sided paired $t$ test at $\\alpha=0.05$ "
         "would detect with 80\\% power given the observed SD of the differences. Class-IL oracle "
         "accuracy is omitted because it equals the task-IL accuracy of the same experts seed by seed.}"),
        "\\label{tab:paired-full}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llllrcccr}", "\\toprule",
        "Protocol & Benchmark & Metric & vs. & Diff. & 95\\% CI & $p_t$ (Holm) & $p_{\\mathrm{sf}}$ ($+/-$) & MDD \\\\",
        "\\midrule",
    ]
    previous = None
    for r in rows:
        group = (r["protocol"], r["benchmark"], r["metric"])
        if previous is not None and group[:2] != previous[:2]:
            lines.append("\\addlinespace[2pt]")
        previous = group
        metric = {"RecoverableFraction": "$\\rho$", "OracleACC": "Oracle ACC"}.get(r["metric"], r["metric"])
        lines.append(
            f"{r['protocol']} & {BENCH.get(r['benchmark'], r['benchmark'])} & {metric} & "
            f"{LABEL.get(r['comparator'], r['comparator'])} & {_signed(r['mean_difference'])} & "
            f"[{_signed(r['ci95_low'])}, {_signed(r['ci95_high'])}] & {_fmt(r['paired_t_p_holm'])} & "
            f"{r['sign_flip_p']:.4f} ({r['positive_seeds']}/{r['negative_seeds']}) & "
            f"{r['min_detectable_difference_80pct']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_compact_tex(rows: list[dict], path: Path,
                      comparators=("packnet", "supsup", "wsn", "lwf")) -> None:
    """Main-text table: one row per (protocol, metric, benchmark), one column per comparator."""
    rows = [r for r in rows if r["family"] == "main"]
    keys = []
    for r in rows:
        key = (r["protocol"], r["metric"], r["benchmark"])
        if key not in keys:
            keys.append(key)
    protocol_order = {"task-IL": 0, "class-IL": 1}
    metric_order = {m: i for i, m in enumerate(("ACC", "TA", "RecoverableFraction", "OracleACC"))}
    bench_order = {b: i for i, b in enumerate(BENCH)}
    keys.sort(key=lambda k: (protocol_order.get(k[0], 9), metric_order.get(k[1], 9),
                             bench_order.get(k[2], 9)))
    cell = {(r["protocol"], r["metric"], r["benchmark"], r["comparator"]): r for r in rows}
    n_seeds = sorted({r["n_seeds"] for r in rows})
    family = rows[0]["family_size"] if rows else 0
    lines = [
        "% Generated by tools/paired_stats.py -- do not edit by hand.",
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        ("\\caption{Paired differences, DiLT-CL minus each comparator (percentage points), "
         f"with unadjusted two-sided 95\\% Student-$t$ intervals over {'/'.join(map(str, n_seeds))} aligned seeds. "
         "$^{*}$: Holm-adjusted paired-$t$ $p<0.05$ within the family of "
         f"{family} comparisons in this table; an unadjusted interval can exclude zero without a star. "
         "With five seeds the exact sign-flip test cannot fall below $p=0.0625$; "
         "Table~\\ref{tab:paired-full} reports it with the minimum detectable difference. "
         "$\\rho$: recoverable-error fraction (Equation~\\eqref{eq:partition}).}"),
        "\\label{tab:paired}",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{ll" + "c" * len(comparators) + "}", "\\toprule",
        "Metric & Benchmark & " + " & ".join(LABEL[c] for c in comparators) + " \\\\",
    ]
    previous = None
    for key in keys:
        if previous is None or key[0] != previous[0]:
            title = {"task-IL": "Task-incremental (oracle task identity)",
                     "class-IL": "Class-incremental (entropy routing)"}.get(key[0], key[0])
            lines += ["\\midrule",
                      f"\\multicolumn{{{2 + len(comparators)}}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        elif key[1] != previous[1]:
            lines.append("\\addlinespace[2pt]")
        previous = key
        metric = {"RecoverableFraction": "$\\rho$"}.get(key[1], key[1])
        cells = []
        for c in comparators:
            r = cell.get((*key, c))
            if r is None:
                cells.append("---")
                continue
            star = "^{*}" if (r["paired_t_p_holm"] is not None and r["paired_t_p_holm"] < 0.05) else ""
            cells.append(f"{_signed(r['mean_difference'])[:-1]}{star}$ "
                         f"{{\\scriptsize[{_signed(r['ci95_low'])}, {_signed(r['ci95_high'])}]}}")
        lines.append(f"{metric} & {BENCH.get(key[2], key[2])} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_calibration_tex(rows: list[dict], path: Path) -> None:
    """Exploratory family: calibrated router minus matched raw router."""
    rows = [r for r in rows if r["family"] == "calibration"]
    n_seeds = sorted({r["n_seeds"] for r in rows})
    lines = [
        "% Generated by tools/paired_stats.py -- do not edit by hand.",
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        ("\\caption{Exploratory: calibrated router minus the matched raw entropy router on the "
         f"same 90\\%-trained experts, over {'/'.join(map(str, n_seeds))} aligned seeds (percentage points). "
         "CI: unadjusted two-sided 95\\% Student-$t$ interval. $p_t$: paired $t$ test, Holm-adjusted within "
         f"this family of {rows[0]['family_size'] if rows else 0} comparisons; it is reported "
         "descriptively because the router procedures were not specified in advance "
         "(Section~\\ref{sec:calibration}).}"),
        "\\label{tab:paired-calibration}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lllrcc}", "\\toprule",
        "Benchmark & Router & Method & Diff. & 95\\% CI & $p_t$ (Holm) \\\\",
        "\\midrule",
    ]
    for metric in ("TA", "ACC"):
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{metric}}}}} \\\\")
        for r in [r for r in rows if r["metric"] == metric]:
            lines.append(
                f"{BENCH.get(r['benchmark'], r['benchmark'])} & {r['reference']} & "
                f"{LABEL.get(r['method'], r['method'])} & {_signed(r['mean_difference'])} & "
                f"[{_signed(r['ci95_low'])}, {_signed(r['ci95_high'])}] & {_fmt(r['paired_t_p_holm'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-il", type=Path, default=None)
    p.add_argument("--class-il", type=Path, default=None)
    p.add_argument("--reference", default="diltcl")
    p.add_argument("--task-comparators", default="lwf,packnet,supsup,wsn")
    p.add_argument("--class-comparators", default=",".join(ISOLATION))
    p.add_argument("--class-metrics", default=",".join(CLASS_METRICS),
                   help="Class-IL metrics compared (OracleACC is available but duplicates task-IL).")
    p.add_argument("--calibrated", type=Path, action="append", default=None,
                   help="Merged calibrated class-IL artifact (repeatable): router minus matched raw.")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-tex", type=Path, default=None, help="Full main-family table (appendix).")
    p.add_argument("--out-compact-tex", type=Path, default=None, help="Compact main-text table.")
    p.add_argument("--out-calibration-tex", type=Path, default=None,
                   help="Calibration-family table (exploratory).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    load = lambda path: None if path is None else json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    rows = build(load(args.task_il), load(args.class_il), args.reference,
                 [m for m in args.task_comparators.split(",") if m],
                 [m for m in args.class_comparators.split(",") if m],
                 class_metrics=tuple(m for m in args.class_metrics.split(",") if m),
                 calibrated=[load(p) for p in (args.calibrated or [])])
    if args.out_json:
        args.out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.out_csv:
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            # Union of keys in first-seen order: calibration rows carry a
            # ``method`` column that the main-family rows do not have.
            fields = []
            for row in rows:
                fields += [k for k in row if k != "seeds" and k not in fields]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    if args.out_tex and rows:
        write_tex(rows, args.out_tex)
    if args.out_compact_tex and rows:
        write_compact_tex(rows, args.out_compact_tex)
    if args.out_calibration_tex and any(r["family"] == "calibration" for r in rows):
        write_calibration_tex(rows, args.out_calibration_tex)
    for r in rows:
        who = (f"{r['reference']}-raw/{r['method']}" if r["family"] == "calibration"
               else f"{args.reference}-{r['comparator']}")
        print(f"{r['protocol']:8s} {r['benchmark']:13s} {r['metric']:19s} {who:22s} "
              f"{r['mean_difference']:+7.2f} [{r['ci95_low']:+6.2f},{r['ci95_high']:+6.2f}] "
              f"n={r['n_seeds']} +{r['positive_seeds']}/-{r['negative_seeds']} "
              f"p_t={_fmt(r['paired_t_p'], False)} (Holm {_fmt(r['paired_t_p_holm'], False)}) "
              f"p_sf={_fmt(r['sign_flip_p'], False)} (Holm {_fmt(r['sign_flip_p_holm'], False)}) "
              f"MDD={r['min_detectable_difference_80pct']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
