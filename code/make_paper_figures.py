#!/usr/bin/env python3
"""Generate the legacy DiLT-CL diagnostic figures from saved JSON results.

The figures deliberately describe the v2.0 artifacts as *legacy diagnostics*.
They are useful for auditing and paper redesign, but they are not a substitute
for a fair rerun.  In particular, the class-IL files use the old overloaded
``LA`` field: it is standard learning accuracy for shared-backbone methods and
oracle-route (task-IL) accuracy for isolation methods.

Run from the repository's ``code/`` directory::

    python make_paper_figures.py

Outputs are written to ``paper/figures/`` as vector PDF, high-resolution PNG,
and a machine-readable provenance JSON.  No result values are hard-coded: all
plotted measurements are loaded from JSON files under ``results/final``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Keep Matplotlib's cache out of a potentially read-only home directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/diltcl-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
# The legacy result files live outside the code directory. Packaging the code
# on its own leaves them absent, so the location is configurable rather than
# fixed; --results-dir (or $DILTCL_RESULTS_DIR) points at wherever they are.
RESULTS_DIR = Path(os.environ.get("DILTCL_RESULTS_DIR",
                                  REPO_ROOT / "results" / "final"))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "figures"

CLASS_RESULT_FILES = {
    "cifar": RESULTS_DIR / "class_il_results_20260604.json",
    "tinyimagenet": RESULTS_DIR / "class_il_results_20260605.json",
}
TASK_RESULT_FILE = RESULTS_DIR / "aaai_results_20260605.json"

BENCHMARK_LABELS = {
    "cifar": "Split-CIFAR-100",
    "tinyimagenet": "Split-TinyImageNet",
}
BENCHMARK_CONFIG_PREFIX = {"cifar": "c100", "tinyimagenet": "tin"}
SHARED_METHODS = ["finetune", "ewc", "lwf", "er", "derpp"]
ISOLATION_METHODS = ["packnet", "supsup", "wsn", "diltcl"]
METHOD_LABELS = {
    "finetune": "Finetune",
    "ewc": "EWC",
    "lwf": "LwF",
    "er": "ER",
    "derpp": "DER++",
    "packnet": "PackNet",
    "supsup": "SupSup",
    "wsn": "WSN",
    "diltcl": "DiLT-CL",
    "independent": "Independent",
}
METHOD_COLORS = {
    "finetune": "#4C78A8",
    "ewc": "#F58518",
    "lwf": "#54A24B",
    "er": "#E45756",
    "derpp": "#B279A2",
    "packnet": "#4C78A8",
    "supsup": "#9D755D",
    "wsn": "#54A24B",
    "diltcl": "#D62728",
    "independent": "#6F4E7C",
}
METHOD_MARKERS = {
    "finetune": "o",
    "ewc": "s",
    "lwf": "^",
    "er": "D",
    "derpp": "P",
    "packnet": "o",
    "supsup": "s",
    "wsn": "^",
    "diltcl": "*",
    "independent": "X",
}

# Manual label offsets keep the fixed layout readable without a non-standard
# label-placement dependency. Values are in display points.
DIAGNOSTIC_LABEL_OFFSETS = {
    ("cifar", "shared", "finetune"): (5, 5),
    ("cifar", "shared", "ewc"): (5, 7),
    ("cifar", "shared", "lwf"): (5, 5),
    ("cifar", "shared", "er"): (-34, -14),
    ("cifar", "shared", "derpp"): (5, -14),
    ("tinyimagenet", "shared", "finetune"): (-42, 8),
    ("tinyimagenet", "shared", "ewc"): (-38, -14),
    ("tinyimagenet", "shared", "lwf"): (5, 5),
    ("tinyimagenet", "shared", "er"): (6, 6),
    ("tinyimagenet", "shared", "derpp"): (5, 5),
    ("cifar", "isolation", "packnet"): (5, -14),
    ("cifar", "isolation", "supsup"): (5, 5),
    ("cifar", "isolation", "wsn"): (5, 5),
    ("cifar", "isolation", "diltcl"): (5, 5),
    ("tinyimagenet", "isolation", "packnet"): (5, -14),
    ("tinyimagenet", "isolation", "supsup"): (5, 5),
    ("tinyimagenet", "isolation", "wsn"): (5, 5),
    ("tinyimagenet", "isolation", "diltcl"): (5, 5),
}

PARETO_LABEL_OFFSETS = {
    "finetune": (6, -13),
    "ewc": (6, -13),
    "lwf": (6, 5),
    "er": (-28, -13),
    "derpp": (-30, 5),
    "packnet": (6, -13),
    "supsup": (6, 5),
    "wsn": (6, 5),
    "diltcl": (7, 5),
    "independent": (-42, 6),
}


def _configure_style() -> None:
    """Apply a deterministic, restrained publication style."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required results file not found: {path}\n"
            "These figures are built from the archived legacy result files. "
            "Point --results-dir (or $DILTCL_RESULTS_DIR) at the directory "
            f"holding '{path.name}'; a code-only checkout does not contain it.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        # Results supplied from outside the repository (--results-dir) have no
        # repo-relative form; record the absolute path so provenance stays
        # accurate instead of failing after the figures were already written.
        return resolved.as_posix()


def _metric_cell(
    result: dict[str, Any], benchmark: str, method: str, metric: str
) -> dict[str, Any]:
    try:
        cell = result["aggregated"][benchmark][method][metric]
    except KeyError as exc:
        raise KeyError(
            f"Missing {benchmark}/{method}/{metric} in saved results"
        ) from exc
    values = [float(value) for value in cell["values"]]
    if len(values) != len(result["seeds"]):
        raise ValueError(
            f"Seed/value mismatch for {benchmark}/{method}/{metric}: "
            f"{len(result['seeds'])} seeds versus {len(values)} values"
        )
    out = {
        "mean": float(cell["mean"]),
        "std": float(cell["std"]),
        "values": values,
    }
    if not all(math.isfinite(value) for value in [out["mean"], out["std"], *values]):
        raise ValueError(f"Non-finite metric for {benchmark}/{method}/{metric}")
    return out


def _save_figure(
    fig: plt.Figure, stem: Path, dpi: int, title: str, subject: str
) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    metadata = {
        "Title": title,
        "Author": "DiLT-CL reproducible figure generator",
        "Subject": subject,
        "Keywords": "legacy diagnostic, continual learning, reproducibility",
    }
    fig.savefig(pdf_path, bbox_inches="tight", metadata=metadata)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    return [pdf_path, png_path]


def _draw_point(
    ax: plt.Axes,
    method: str,
    x: dict[str, Any],
    y: dict[str, Any],
    label_offset: tuple[int, int],
) -> None:
    marker_size = 10 if method == "diltcl" else 6.5
    ax.errorbar(
        x["mean"],
        y["mean"],
        xerr=x["std"],
        yerr=y["std"],
        fmt=METHOD_MARKERS[method],
        markersize=marker_size,
        markerfacecolor=METHOD_COLORS[method],
        markeredgecolor="white",
        markeredgewidth=0.7,
        ecolor=METHOD_COLORS[method],
        elinewidth=0.9,
        capsize=2.4,
        capthick=0.8,
        alpha=0.97,
        zorder=4,
    )
    ax.annotate(
        METHOD_LABELS[method],
        (x["mean"], y["mean"]),
        xytext=label_offset,
        textcoords="offset points",
        fontsize=7.4,
        color="#252525",
        zorder=5,
    )


def make_failure_mode_diagnostic(
    class_results: dict[str, dict[str, Any]], output_dir: Path, dpi: int
) -> tuple[list[Path], dict[str, Any]]:
    """Create the 2x2 legacy shared/isolation diagnostic."""

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.35), constrained_layout=False)
    provenance_panels: list[dict[str, Any]] = []

    for row, benchmark in enumerate(["cifar", "tinyimagenet"]):
        result = class_results[benchmark]
        benchmark_label = BENCHMARK_LABELS[benchmark]

        shared_ax = axes[row, 0]
        shared_points = []
        for method in SHARED_METHODS:
            oracle_or_la = _metric_cell(result, benchmark, method, "LA")
            forgetting = _metric_cell(result, benchmark, method, "Forgetting")
            _draw_point(
                shared_ax,
                method,
                oracle_or_la,
                forgetting,
                DIAGNOSTIC_LABEL_OFFSETS[(benchmark, "shared", method)],
            )
            shared_points.append(
                {
                    "method": method,
                    "display_name": METHOD_LABELS[method],
                    "legacy_LA": oracle_or_la,
                    "legacy_Forgetting": forgetting,
                }
            )
        shared_ax.axhline(0, color="#777777", linestyle=(0, (3, 2)), linewidth=0.9)
        shared_ax.set_xlim(0, 100)
        shared_ax.set_ylim(-3, 90)
        shared_ax.set_title(f"{benchmark_label} · shared backbone")
        shared_ax.set_xlabel("Legacy learning accuracy, LA (%)")
        shared_ax.set_ylabel("Legacy forgetting (%)")
        shared_ax.grid(True, linestyle=":", zorder=0)
        provenance_panels.append(
            {
                "benchmark": benchmark,
                "family": "shared_backbone",
                "x_field": "LA",
                "x_display": "Legacy learning accuracy (LA)",
                "y_field": "Forgetting",
                "points": shared_points,
            }
        )

        isolation_ax = axes[row, 1]
        isolation_points = []
        config_prefix = BENCHMARK_CONFIG_PREFIX[benchmark]
        classes_per_task = int(
            result["config"][f"{config_prefix}_classes_per_task"]
        )
        num_tasks = int(result["config"][f"{config_prefix}_num_tasks"])
        oracle_chance = 100.0 / classes_per_task
        task_chance = 100.0 / num_tasks
        for method in ISOLATION_METHODS:
            oracle_acc = _metric_cell(result, benchmark, method, "LA")
            ta = _metric_cell(result, benchmark, method, "TA")
            _draw_point(
                isolation_ax,
                method,
                oracle_acc,
                ta,
                DIAGNOSTIC_LABEL_OFFSETS[(benchmark, "isolation", method)],
            )
            isolation_points.append(
                {
                    "method": method,
                    "display_name": METHOD_LABELS[method],
                    "legacy_source_field": "LA",
                    "OracleACC": oracle_acc,
                    "TA": ta,
                }
            )
        isolation_ax.axvline(
            oracle_chance, color="#999999", linestyle=(0, (3, 2)), linewidth=0.9
        )
        isolation_ax.axhline(
            task_chance, color="#777777", linestyle=(0, (3, 2)), linewidth=0.9
        )
        isolation_ax.text(
            0.985,
            task_chance + 0.35,
            f"TA chance {task_chance:g}%",
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="#666666",
            transform=isolation_ax.get_yaxis_transform(),
        )
        isolation_ax.set_xlim(0, 100)
        isolation_ax.set_ylim(0, 25)
        isolation_ax.set_title(f"{benchmark_label} · parameter isolation")
        isolation_ax.set_xlabel("Legacy OracleACC (old schema field: LA) (%)")
        isolation_ax.set_ylabel("Task-inference accuracy, TA (%)")
        isolation_ax.grid(True, linestyle=":", zorder=0)
        provenance_panels.append(
            {
                "benchmark": benchmark,
                "family": "parameter_isolation",
                "x_field_in_legacy_json": "LA",
                "x_display": "OracleACC",
                "y_field": "TA",
                "oracle_classification_chance_percent": oracle_chance,
                "task_inference_chance_percent": task_chance,
                "chance_reference": {
                    "classes_per_task": classes_per_task,
                    "num_tasks": num_tasks,
                    "source": "config fields in the same class-IL JSON",
                },
                "points": isolation_points,
            }
        )

    fig.suptitle(
        "Legacy v2.0 diagnostic, rerun required\n"
        "Family-specific class-IL failure modes",
        fontsize=15,
        fontweight="bold",
        color="#8B1A1A",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Error bars: ±1 seed SD (n=3). Isolation OracleACC is the legacy JSON "
        "field LA; it is not comparable to shared-method LA. Isolation "
        "class-IL forgetting was not measured and is intentionally omitted.",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.105, wspace=0.22, hspace=0.35)

    stem = output_dir / "legacy_failure_mode_diagnostic"
    files = _save_figure(
        fig,
        stem,
        dpi,
        "Legacy v2.0 diagnostic, rerun required: family-specific class-IL failure modes",
        "Legacy v2.0 family-specific class-IL diagnostic; rerun required.",
    )
    plt.close(fig)
    provenance = {
        "title": "Legacy v2.0 diagnostic, rerun required: family-specific class-IL failure modes",
        "layout": "2 rows (benchmarks) x 2 columns (method families)",
        "uncertainty": "sample standard deviation across seeds, as stored by the runner (ddof=1)",
        "panels": provenance_panels,
        "caveats": [
            "The isolation-method JSON field LA is relabeled OracleACC because it is oracle-route task-IL accuracy, not standard shared-method learning accuracy.",
            "Shared and isolation panels intentionally use different legacy metric definitions.",
            "Isolation class-IL BWT/Forgetting were hard-coded rather than measured under sequential routing, so they are not plotted.",
            "TA is available only as per-seed scalar summaries; no per-task or per-example routing data are present.",
        ],
    }
    return files, provenance


def _pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return points not dominated when storage is minimized and ACC maximized."""

    frontier = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            no_more_storage = other["storage_mib"] <= point["storage_mib"]
            no_less_accuracy = other["accuracy"]["mean"] >= point["accuracy"]["mean"]
            at_least_one_strict = (
                other["storage_mib"] < point["storage_mib"]
                or other["accuracy"]["mean"] > point["accuracy"]["mean"]
            )
            if no_more_storage and no_less_accuracy and at_least_one_strict:
                dominated = True
                break
        point["pareto_frontier"] = not dominated
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda item: item["storage_mib"])


def make_tinyimagenet_pareto(
    task_result: dict[str, Any], output_dir: Path, dpi: int
) -> tuple[list[Path], dict[str, Any]]:
    """Create TinyImageNet task-IL accuracy versus legacy analytic storage."""

    benchmark = "tinyimagenet"
    name_to_key = {display: key for key, display in METHOD_LABELS.items()}
    name_to_key["DiLT-CL (ours)"] = "diltcl"
    points: list[dict[str, Any]] = []
    for cost in task_result["memory"][benchmark]:
        display_name = cost["name"]
        if display_name not in name_to_key:
            raise KeyError(f"No method-key mapping for memory row {display_name!r}")
        method = name_to_key[display_name]
        accuracy = _metric_cell(task_result, benchmark, method, "ACC")
        points.append(
            {
                "method": method,
                "display_name": METHOD_LABELS[method],
                "storage_mib": float(cost["megabytes"]),
                "accuracy": accuracy,
                "legacy_cost_components": {
                    "float_params": float(cost["float_params"]),
                    "mask_bits": float(cost["mask_bits"]),
                    "buffer_floats": float(cost["buffer_floats"]),
                    "note": cost["note"],
                },
            }
        )

    frontier = _pareto_frontier(points)
    fig, ax = plt.subplots(figsize=(7.5, 5.05))

    for point in points:
        method = point["method"]
        is_frontier = point["pareto_frontier"]
        marker_size = 12 if method == "diltcl" else 7.5
        face = METHOD_COLORS[method] if is_frontier else "white"
        ax.errorbar(
            point["storage_mib"],
            point["accuracy"]["mean"],
            yerr=point["accuracy"]["std"],
            fmt=METHOD_MARKERS[method],
            markersize=marker_size,
            markerfacecolor=face,
            markeredgecolor=METHOD_COLORS[method],
            markeredgewidth=1.1,
            ecolor=METHOD_COLORS[method],
            elinewidth=0.9,
            capsize=2.4,
            capthick=0.8,
            alpha=0.98,
            zorder=4,
        )
        ax.annotate(
            point["display_name"],
            (point["storage_mib"], point["accuracy"]["mean"]),
            xytext=PARETO_LABEL_OFFSETS[method],
            textcoords="offset points",
            fontsize=7.6,
            color="#252525",
            zorder=5,
        )

    ax.plot(
        [point["storage_mib"] for point in frontier],
        [point["accuracy"]["mean"] for point in frontier],
        color="#555555",
        linewidth=1.1,
        linestyle="-",
        alpha=0.8,
        zorder=2,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.58, 32)
    ax.set_ylim(0, 51)
    ax.set_xticks([0.75, 1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Legacy analytic compressed-storage estimate (MiB, log scale)")
    ax.set_ylabel("Task-IL accuracy (%)")
    ax.grid(True, which="major", linestyle=":", zorder=0)
    ax.set_title(
        "Split-TinyImageNet task-IL accuracy-storage Pareto\n"
        "Legacy v2.0 diagnostic, rerun required",
        fontsize=13.5,
        fontweight="bold",
        color="#8B1A1A",
        pad=11,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#555555",
            markeredgecolor="#555555",
            markersize=6,
            label="Legacy Pareto frontier",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#777777",
            markersize=6,
            label="Dominated estimate",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)
    fig.text(
        0.5,
        0.018,
        "Accuracy error bars: ±1 seed SD (n=3). Storage is an analytic legacy "
        "estimate (float32 values, 1-bit masks, ideal sparse packing), not a "
        "measured checkpoint or resident-memory footprint.",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.105, right=0.975, top=0.84, bottom=0.16)

    stem = output_dir / "legacy_tinyimagenet_accuracy_storage_pareto"
    files = _save_figure(
        fig,
        stem,
        dpi,
        "Legacy v2.0 diagnostic, rerun required: TinyImageNet accuracy-storage Pareto",
        "Legacy v2.0 TinyImageNet task-IL accuracy versus analytic compressed storage; rerun required.",
    )
    plt.close(fig)
    provenance = {
        "title": "Split-TinyImageNet task-IL accuracy-storage Pareto (legacy v2.0 diagnostic, rerun required)",
        "x_metric": "memory.tinyimagenet[].megabytes (actually MiB)",
        "x_scale": "log2",
        "y_metric": "aggregated.tinyimagenet[method].ACC",
        "uncertainty": "sample standard deviation across seeds, as stored by the runner (ddof=1)",
        "pareto_objective": "minimize analytic storage estimate and maximize mean task-IL ACC",
        "points": points,
        "frontier_order": [point["method"] for point in frontier],
        "caveats": [
            "Storage is an analytic legacy estimate, not measured checkpoint size or resident memory.",
            "The model assumes float32 values, bit-packed 1-bit masks, ideal sparse-value storage without index overhead, and the method-specific conventions recorded in each memory note.",
            "The DiLT-CL density contribution was captured once by the legacy runner; no cross-seed storage uncertainty is available.",
        ],
    }
    return files, provenance


def _source_record(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256": _sha256(path)}


def _output_record(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _write_provenance(
    output_dir: Path,
    source_paths: Iterable[Path],
    output_paths: Iterable[Path],
    failure_provenance: dict[str, Any],
    pareto_provenance: dict[str, Any],
    dpi: int,
) -> Path:
    provenance_path = output_dir / "legacy_figures_provenance.json"
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": _relative(Path(__file__)),
        "generator_sha256": _sha256(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "png_dpi": dpi,
        },
        "global_notice": "Legacy v2.0 diagnostic, rerun required",
        "sources": [_source_record(path) for path in source_paths],
        "outputs": [_output_record(path) for path in output_paths],
        "figures": {
            "legacy_failure_mode_diagnostic": failure_provenance,
            "legacy_tinyimagenet_accuracy_storage_pareto": pareto_provenance,
        },
    }
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return provenance_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate legacy paper diagnostic figures from final JSONs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: paper/figures).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory holding the archived legacy result JSON files "
             "(default: $DILTCL_RESULTS_DIR, else <repo>/results/final). "
             "A code-only checkout does not ship them.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="PNG resolution (default: 320 dpi).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.dpi < 150:
        raise SystemExit("--dpi must be at least 150 for publication output")
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_style()
    # Resolve the result files against the requested directory so a code-only
    # checkout can point at the archived results wherever they were kept.
    results_dir = args.results_dir or RESULTS_DIR
    class_result_files = {
        benchmark: results_dir / path.name
        for benchmark, path in CLASS_RESULT_FILES.items()
    }
    task_result_file = results_dir / TASK_RESULT_FILE.name
    class_results = {
        benchmark: _load_json(path)
        for benchmark, path in class_result_files.items()
    }
    task_result = _load_json(task_result_file)

    failure_files, failure_provenance = make_failure_mode_diagnostic(
        class_results, output_dir, args.dpi
    )
    pareto_files, pareto_provenance = make_tinyimagenet_pareto(
        task_result, output_dir, args.dpi
    )
    figure_files = failure_files + pareto_files
    provenance_path = _write_provenance(
        output_dir,
        [*class_result_files.values(), task_result_file],
        figure_files,
        failure_provenance,
        pareto_provenance,
        args.dpi,
    )

    print("Generated paper diagnostics:")
    for path in [*figure_files, provenance_path]:
        print(f"  {_relative(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
