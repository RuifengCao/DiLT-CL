"""Generate class-IL LaTeX tables from a ``run_class_il.py`` results file.

Reads ``class_il_results.json`` (or any file produced by ``run_class_il.py``)
and emits one booktabs table per benchmark with ACC / BWT / Forgetting / LA
(mean$\\pm$std over seeds), significance stars on ACC versus DER++ when at
least five paired seeds are available, and the best ACC in bold. Verified
schema-v3 artifacts render without an override; this tool does not waive any
other publication gate. Unverified artifacts require an explicit
diagnostic-only override.

The significance helper is imported from the package, with a torch-free
file-load fallback so the tool also runs in environments without torch.

Example:
    python make_class_il_table.py --results class_il_results.json \
        --out class_il_tables.tex
"""

import argparse
import json

# Prefer the package; fall back to loading stats.py directly (no torch needed).
try:
    from diltcl.stats import paired_significance
except Exception:  # noqa: BLE001 - torch may be unavailable during package init
    import importlib.util
    import pathlib
    _p = pathlib.Path(__file__).resolve().parent / "diltcl" / "stats.py"
    _spec = importlib.util.spec_from_file_location("diltcl_stats", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    paired_significance = _mod.paired_significance

_PRETTY = {"finetune": "Finetune", "ewc": "EWC", "lwf": "LwF", "er": "ER",
           "derpp": "DER++", "packnet": "PackNet", "supsup": "SupSup",
           "wsn": "WSN", "diltcl": "DiLT-CL", "independent": "Independent"}
_BENCH_NAME = {"cifar": "Split-CIFAR-100", "pmnist": "Permuted-MNIST"}
_REF = "derpp"


def _stars(p):
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _cell(mean, std, bold=False, star=""):
    body = f"{mean:.2f}\\pm{std:.2f}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    cell = f"${body}$"
    if star:
        cell += f"\\,$^{{{star}}}$"
    return cell


def _metric_cell(aggregate, metric):
    cell = aggregate.get(metric)
    return _cell(cell["mean"], cell["std"]) if cell is not None else "--"


def _latex_text(value):
    """Escape metadata inserted into ordinary LaTeX text."""
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(value))


def _table(bench, data, *, diagnostic=False):
    agg = data["aggregated"][bench]
    per_seed = data["per_seed"][bench]
    methods = data["methods"]
    has_ref = _REF in methods
    ref_acc = [s["ACC"] for s in per_seed[_REF]] if has_ref else None
    declared_seeds = data.get("seeds", [])
    try:
        unique_seed_ids = len(set(declared_seeds)) == len(declared_seeds)
    except TypeError:
        unique_seed_ids = False
    significance_eligible = (
        len(declared_seeds) >= 5 and unique_seed_ids
        and (not has_ref or len(ref_acc) == len(declared_seeds))
    )
    best = max(agg[m]["ACC"]["mean"] for m in methods)

    has_ta = any("TA" in agg[m] for m in methods)
    has_oracle = any("OracleACC" in agg[m] for m in methods)
    has_conditional = any("ConditionalACC" in agg[m] for m in methods)
    rows = []
    for m in methods:
        a = agg[m]
        star = ""
        if has_ref and m != _REF:
            method_acc = [s["ACC"] for s in per_seed[m]]
            # Fewer than five pairs are too fragile for star annotations. The
            # aggregate mean/std remains useful, but no visual significance
            # claim is made.
            if (significance_eligible
                    and len(method_acc) == len(declared_seeds)):
                record = paired_significance(method_acc, ref_acc)
                # A degenerate test (zero-variance differences) carries no
                # evidence even though SciPy reports p = 0.
                star = "" if record.get("degenerate") else _stars(record["p_value"])
        name = _PRETTY.get(m, m.capitalize())
        if has_ref and m == _REF:
            name += " (ref.)"
        bold = abs(a["ACC"]["mean"] - best) < 1e-9
        cells = [name,
                 _cell(a["ACC"]["mean"], a["ACC"]["std"], bold=bold, star=star),
                 _metric_cell(a, "BWT"),
                 _metric_cell(a, "Forgetting"),
                 _metric_cell(a, "LA")]
        if has_oracle:
            cells.append(_metric_cell(a, "OracleACC"))
        if has_ta:
            cells.append(_metric_cell(a, "TA"))
        if has_conditional:
            cells.append(_metric_cell(a, "ConditionalACC"))
        rows.append(" & ".join(cells) + r" \\")

    n_seeds = len(data.get("seeds", per_seed[methods[0]]))
    pretty = _BENCH_NAME.get(bench, bench)
    schema = data["metric_schema_version"]
    code_version = data.get("code_version", "unspecified")
    verification = data.get("verification_status", "missing")
    criterion = data.get("inference_criterion", "unspecified")
    calibration = data.get("inference_calibration", "unspecified")
    diagnostic_prefix = (
        r"\textbf{DIAGNOSTIC ONLY---UNVERIFIED ARTIFACT.} "
        if diagnostic else "")
    caption = (diagnostic_prefix
               + f"Class-incremental results on {_latex_text(pretty)} "
               "(no task identity at "
               f"test). Mean$\\pm$std over {n_seeds} seeds. ACC is the macro "
               "average of per-task accuracies over the seen global class "
               "space; MicroACC is stored separately in the artifact. "
               "Shared-backbone methods use a "
               "fixed-full-head; isolation methods use a task-local head plus "
               "exhaustive router (local+router). ACC, BWT, Forgetting and LA "
               "are computed "
               "from the same prefix-candidate result matrix for both method "
               "families. Oracle-route accuracy (OracleACC) is the isolation "
               "subnetwork accuracy under the true route; TA is final "
               "task-routing accuracy; Cond. is pooled class accuracy "
               "conditioned on a correct route. Router criterion="
               f"{_latex_text(criterion)}; calibration="
               f"{_latex_text(calibration)}. Code="
               f"{_latex_text(code_version)}; metric schema={schema}; "
               f"verification status={_latex_text(verification)}. Stars are "
               "uncorrected exploratory paired tests of ACC "
               "vs.\\ DER++ ($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$); "
               "comparisons with fewer than five paired seeds receive no "
               "stars; best ACC bold.")
    extra = int(has_oracle) + int(has_ta) + int(has_conditional)
    colspec = "l" + "c" * (4 + extra)
    header_cells = ["Method", "ACC", "BWT", "Forgetting", "LA"]
    if has_oracle:
        header_cells.append("OracleACC")
    if has_ta:
        header_cells.append("TA")
    if has_conditional:
        header_cells.append("Cond.")
    header = " & ".join(header_cells)
    body = "\n".join(rows)
    return ("\\begin{table}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{tab:classil_{bench}}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
            f"{header} \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def main():
    ap = argparse.ArgumentParser(description="class-IL results -> LaTeX tables")
    ap.add_argument("--results", default="class_il_results.json",
                    help="JSON file written by run_class_il.py")
    ap.add_argument("--out", default="class_il_tables.tex",
                    help="Output .tex file (one table environment per benchmark)")
    ap.add_argument(
        "--allow-unverified", action="store_true",
        help="Render a conspicuously labelled diagnostic table from an "
             "unverified artifact (never publication-ready).")
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    schema = data.get("metric_schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema < 3:
        raise SystemExit(
            "Refusing to render a non-integer or class-IL metric schema < 3. "
            "Rerun with a current run_class_il.py; older matrix/metric "
            "conventions "
            "are not accepted by this table generator.")

    verification = data.get("verification_status", "missing")
    diagnostic = verification != "verified"
    if diagnostic and not args.allow_unverified:
        raise SystemExit(
            "Refusing to render an artifact whose verification_status is "
            f"{verification!r}, not 'verified'. Independently verify the "
            "artifact or pass --allow-unverified for a conspicuously labelled "
            "diagnostic-only table.")

    tables = [_table(b, data, diagnostic=diagnostic)
              for b in data["benchmarks"]]
    with open(args.out, "w") as f:
        f.write("\n".join(tables))
    print(f"Wrote {len(tables)} table(s) for {data['benchmarks']} to "
          f"'{args.out}'.")


if __name__ == "__main__":
    main()
