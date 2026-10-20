#!/usr/bin/env python3
"""Paper figures from v3 merged artifacts (``tools/merge_results.py`` output).

Three figures, each written as PDF + PNG plus a provenance JSON that records the
SHA-256 of every input artifact, so a figure can always be traced to the exact
numbers behind it. No value is hard-coded.

``failure_modes``  (class-IL artifact, required)
    One column per benchmark. Top: class-IL ACC per method (mean +- std over
    seeds) with LA marked above each bar. Bottom: the *dominant failure mode*
    of each family on one scale -- forgetting ``-BWT`` for shared-backbone
    methods and routing error ``100 - TA`` for isolation methods. This is the
    LA/TA decomposition: shared backbones learn (high LA) but forget; isolation
    methods keep their experts unchanged but mis-route. Their class-IL BWT is
    not zero: a later expert can capture earlier examples (PROTOCOL.md §3.3).
``calibration_effect``  (calibrated class-IL artifact, optional)
    Raw -> calibrated routing for every isolation method: paired per-seed
    lines for TA and ACC.
``task_il_gap``  (task-IL artifact, optional)
    Task-IL ACC per method and benchmark with the Independent upper bound.

Artifacts must carry ``verification_status == "verified"`` (produced by
``tools/merge_results.py`` after ``tools/verify_experiment_bundle.py``).
``--allow-unverified`` renders anyway and stamps the figures as not
publication-ready.

Example::

    python make_v3_figures.py --class-results RUN/merged/p0_class_entropy.json \\
        --class-results-cal RUN/merged/p0_class_entropy_cal.json \\
        --task-results RUN/merged/p0_task_il_multi.json --output-dir paper/figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diltcl-matplotlib")

SHARED = ["finetune", "ewc", "lwf", "er", "derpp"]
ISOLATION = ["packnet", "supsup", "wsn", "diltcl"]
METHOD_LABEL = {"finetune": "Finetune", "ewc": "EWC", "lwf": "LwF", "er": "ER",
                "derpp": "DER++", "packnet": "PackNet", "supsup": "SupSup",
                "wsn": "WSN", "diltcl": "DiLT-CL", "independent": "Independent"}
BENCHMARK_LABEL = {"cifar": "Split-CIFAR-100", "pmnist": "Permuted-MNIST",
                   "tinyimagenet": "Split-TinyImageNet", "fivedatasets": "5-datasets"}
FAMILY_COLOR = {"shared": "#4C72B0", "isolation": "#DD8452"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, allow_unverified: bool) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("verification_status", "missing")
    if status != "verified" and not allow_unverified:
        raise SystemExit(
            f"{path}: verification_status is {status!r}; run "
            "tools/verify_experiment_bundle.py and tools/merge_results.py first, "
            "or pass --allow-unverified for a diagnostic figure")
    payload["_path"] = str(path)
    payload["_sha256"] = _sha256(path)
    return payload


def _family(method: str) -> str:
    return "isolation" if method in ISOLATION else "shared"


def _ordered_methods(payload: dict) -> list[str]:
    methods = [m for m in SHARED if m in payload["methods"]]
    methods += [m for m in ISOLATION if m in payload["methods"]]
    return methods


def _stat(payload: dict, benchmark: str, method: str, metric: str):
    cell = payload["aggregated"].get(benchmark, {}).get(method, {}).get(metric)
    if cell is None:
        return None, None, None
    return float(cell["mean"]), float(cell["std"]), [float(v) for v in cell["values"]]


def _stamp(fig, payload: dict) -> None:
    if payload.get("verification_status") != "verified":
        fig.text(0.5, 0.01, "UNVERIFIED ARTIFACT - not publication-ready",
                 ha="center", va="bottom", color="red", fontsize=9)


def make_failure_modes(payload: dict, output_dir: Path, dpi: int) -> list[Path]:
    import matplotlib.pyplot as plt

    benchmarks = list(payload["benchmarks"])
    methods = _ordered_methods(payload)
    fig, axes = plt.subplots(2, len(benchmarks), figsize=(4.2 * len(benchmarks), 6.2),
                             squeeze=False, sharex="col")
    for col, benchmark in enumerate(benchmarks):
        top, bottom = axes[0][col], axes[1][col]
        xs = np.arange(len(methods))
        acc_mean, acc_std, la_mean, colors = [], [], [], []
        mode_value = []
        for method in methods:
            mean, std, _ = _stat(payload, benchmark, method, "ACC")
            la, _, _ = _stat(payload, benchmark, method, "LA")
            acc_mean.append(np.nan if mean is None else mean)
            acc_std.append(0.0 if std is None else std)
            la_mean.append(np.nan if la is None else la)
            colors.append(FAMILY_COLOR[_family(method)])
            if _family(method) == "shared":
                bwt, _, _ = _stat(payload, benchmark, method, "BWT")
                mode_value.append(np.nan if bwt is None else -bwt)      # forgetting
            else:
                ta, _, _ = _stat(payload, benchmark, method, "TA")
                mode_value.append(np.nan if ta is None else 100.0 - ta)  # routing error
        top.bar(xs, acc_mean, yerr=acc_std, color=colors, capsize=3,
                edgecolor="black", linewidth=0.5)
        top.scatter(xs, la_mean, marker="_", s=260, color="black", zorder=3,
                    label="LA (learning accuracy)")
        top.set_ylabel("class-IL ACC (%)")
        top.set_title(BENCHMARK_LABEL.get(benchmark, benchmark))
        top.set_ylim(0, max(100.0, np.nanmax(la_mean) + 5 if len(la_mean) else 100.0))
        bottom.bar(xs, mode_value, color=colors, edgecolor="black", linewidth=0.5)
        for x, value in zip(xs, mode_value):
            if not np.isnan(value):
                bottom.text(x, max(value, 0.0) + 1.0, f"{value + 0.0:.1f}",
                            ha="center", va="bottom", fontsize=7)
        bottom.set_ylabel("dominant loss (pp)")
        bottom.set_xticks(xs)
        bottom.set_xticklabels([METHOD_LABEL.get(m, m) for m in methods],
                               rotation=45, ha="right")
        finite = [v for v in mode_value if not np.isnan(v)]
        lower = min(0.0, min(finite) - 5.0) if finite else 0.0   # positive BWT
        bottom.set_ylim(lower, 105)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR["shared"]),
               plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR["isolation"]),
               plt.Line2D([0], [0], marker="_", color="black", markersize=12,
                          linestyle="none")]
    # Figure-level legend: an axes legend collides with the LA markers of the
    # high-LA shared-backbone methods.
    fig.legend(handles, ["shared backbone: bar = ACC, bottom = forgetting (-BWT)",
                         "isolation: bar = ACC, bottom = routing error (100-TA)",
                         "LA (accuracy right after learning each task)"],
               loc="upper center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.935))
    seeds = payload.get("seeds", [])
    fig.suptitle(f"Class-IL failure modes (mean over {len(seeds)} seeds; "
                 f"code {payload.get('code_version')})", fontsize=10)
    _stamp(fig, payload)
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    return _save(fig, output_dir / "failure_modes", dpi)


def make_calibration_effect(payload: dict, output_dir: Path, dpi: int) -> list[Path]:
    import matplotlib.pyplot as plt

    benchmarks = list(payload["benchmarks"])
    methods = [m for m in ISOLATION if m in payload["methods"]]
    if not methods:
        return []
    fig, axes = plt.subplots(2, len(benchmarks), figsize=(4.0 * len(benchmarks), 5.6),
                             squeeze=False)
    for col, benchmark in enumerate(benchmarks):
        for row, metric in enumerate(("TA", "ACC")):
            ax = axes[row][col]
            for i, method in enumerate(methods):
                _, _, raw = _stat(payload, benchmark, method, "Raw" + metric)
                _, _, cal = _stat(payload, benchmark, method, metric)
                if raw is None or cal is None:
                    continue
                x0, x1 = i - 0.18, i + 0.18
                for r, c in zip(raw, cal):
                    ax.plot([x0, x1], [r, c], color="grey", linewidth=0.8, alpha=0.7)
                ax.scatter([x0] * len(raw), raw, color="black", s=14, zorder=3)
                ax.scatter([x1] * len(cal), cal, color=FAMILY_COLOR["isolation"],
                           s=14, zorder=3)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in methods])
            ax.set_ylabel(f"{metric} (%)  raw -> calibrated")
            if row == 0:
                ax.set_title(BENCHMARK_LABEL.get(benchmark, benchmark))
    kinds = sorted({rec.get("kind") for by_method in (payload.get("calibration") or {}).values()
                    for by_seed in by_method.values() for rec in by_seed.values()
                    if rec.get("kind") not in (None, "none")})
    fig.suptitle(f"Effect of router calibration ({', '.join(kinds) or 'calibrated'}; "
                 "per-seed pairs)", fontsize=10)
    _stamp(fig, payload)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return _save(fig, output_dir / "calibration_effect", dpi)


def make_task_il_gap(payload: dict, output_dir: Path, dpi: int) -> list[Path]:
    import matplotlib.pyplot as plt

    benchmarks = list(payload["benchmarks"])
    methods = _ordered_methods(payload)
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(4.2 * len(benchmarks), 3.6),
                             squeeze=False)
    for col, benchmark in enumerate(benchmarks):
        ax = axes[0][col]
        means, stds, colors = [], [], []
        for method in methods:
            mean, std, _ = _stat(payload, benchmark, method, "ACC")
            means.append(np.nan if mean is None else mean)
            stds.append(0.0 if std is None else std)
            colors.append(FAMILY_COLOR[_family(method)])
        xs = np.arange(len(methods))
        ax.bar(xs, means, yerr=stds, color=colors, capsize=3, edgecolor="black",
               linewidth=0.5)
        upper, _, _ = _stat(payload, benchmark, "independent", "ACC")
        if upper is not None:
            ax.axhline(upper, linestyle="--", color="black", linewidth=1,
                       label=f"Independent {upper:.1f}")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in methods],
                           rotation=45, ha="right")
        ax.set_ylabel("task-IL ACC (%)")
        ax.set_title(f"{BENCHMARK_LABEL.get(benchmark, benchmark)} "
                     f"({payload.get('task_il_head_mode', '?')}-head)")
        ax.set_ylim(0, 100)
    fig.suptitle(f"Task-IL accuracy under the {payload.get('task_il_head_mode', '?')}-head "
                 f"protocol (mean +- std over {len(payload.get('seeds', []))} seeds)",
                 fontsize=10)
    _stamp(fig, payload)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return _save(fig, output_dir / "task_il_gap", dpi)


def _save(fig, stem: Path, dpi: int) -> list[Path]:
    import matplotlib.pyplot as plt

    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".pdf", ".png"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-results", type=Path, required=True,
                        help="Merged class-IL artifact (entropy routing).")
    parser.add_argument("--class-results-cal", type=Path, default=None,
                        help="Merged calibrated class-IL artifact.")
    parser.add_argument("--task-results", type=Path, default=None,
                        help="Merged task-IL artifact.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--allow-unverified", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")

    sources, outputs = [], []
    class_payload = _load(args.class_results, args.allow_unverified)
    sources.append(class_payload)
    outputs += make_failure_modes(class_payload, args.output_dir, args.dpi)
    if args.class_results_cal is not None:
        cal_payload = _load(args.class_results_cal, args.allow_unverified)
        sources.append(cal_payload)
        outputs += make_calibration_effect(cal_payload, args.output_dir, args.dpi)
    if args.task_results is not None:
        task_payload = _load(args.task_results, args.allow_unverified)
        sources.append(task_payload)
        outputs += make_task_il_gap(task_payload, args.output_dir, args.dpi)

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "make_v3_figures.py",
        "sources": [{"path": p["_path"], "sha256": p["_sha256"],
                     "code_version": p.get("code_version"),
                     "verification_status": p.get("verification_status"),
                     "seeds": p.get("seeds"), "benchmarks": p.get("benchmarks"),
                     "methods": p.get("methods")} for p in sources],
        "outputs": [{"path": str(path), "sha256": _sha256(path)} for path in outputs],
        "publication_ready": all(p.get("verification_status") == "verified"
                                 for p in sources),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    for path in outputs:
        print(f"wrote {path}")
    print(f"provenance: {args.output_dir / 'figure_provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
