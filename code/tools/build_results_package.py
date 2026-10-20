#!/usr/bin/env python3
"""Bundle the results of one or more suite run directories into one package.

After ``tools/merge_results.py`` a run directory holds everything a paper
needs, scattered over ``merged/``, ``paper/``, ``verification_report.json``,
``plan.json`` and the provenance archives. This tool collects those files from
any number of run directories into a single, self-describing package:

    results_<date>/
      README.md                      what is inside, how to verify it
      MANIFEST.json                  SHA-256 of every file, code versions,
                                     verification status per run
      <run_name>/merged/*.json|npz   merged artifacts (verified or not)
      <run_name>/verification_report.json, plan.json, suite_summary.json,
                 failures.json, source_manifest.json, environment.json,
                 dataset_fingerprint.json (without the per-file list),
                 source_manifests/, environments/, dataset_fingerprints/
                 (tree hashes only)
      <run_name>/paper/*, figures/*  tables, figures, storage, timing
      <run_name>/audit/*             reports of the regeneration audit and the
                                     checkpoint re-evaluation, if present
      <run_name>/statuses/*.json     per-job status (small; optional --no-statuses)

Per-job ``artifacts/`` (result JSON/NPZ per job) and checkpoints are *not*
included by default (they stay on the machine); ``--include-artifacts`` adds
the per-job result files (no checkpoints).

``--supplementary CODE_DIR`` additionally copies the code package (without
caches, editor state or the archive folder) and writes ``HOW_TO_VERIFY.md``,
producing the anonymised supplementary material for a submission.
``--anonymity-check TOKEN,...`` fails the build if any text file in the
package contains one of the tokens (author names, institution, e-mail).

The package is written as a directory and, unless ``--no-tar``, also as
``<out>.tar.gz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

TOOL_VERSION = "1.1"
# In supplementary mode, references from the code's own docs to repository
# folders that are not shipped with the package (the evidence folder and the
# folder names of older project layouts) are replaced by a neutral note.
_EXTERNAL_DOC_PATTERN = (r"`\.\./(?:evidence|02_docs|05_archive|03_paper|04_results)"
                         r"(?:/[^`]*)?`")
_EXTERNAL_DOC_NOTE = "the project documentation (not included in this package)"
_TEXT_SUFFIXES = {".md", ".txt", ".json", ".py", ".tex", ".toml", ".csv"}
_CODE_EXCLUDED_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache",
                       ".ruff_cache", ".mypy_cache", ".git", ".hypothesis"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _strip_fingerprint(src: Path, dst: Path) -> None:
    """Copy a dataset fingerprint without its (huge) per-file list."""
    data = json.loads(src.read_text(encoding="utf-8"))
    data.pop("files", None)
    data["note"] = ("per-file records removed for the package; the tree_sha256 "
                    "and entries identify the data exactly")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")


def collect_run(run_dir: Path, dest: Path, include_artifacts: bool,
                include_statuses: bool) -> dict:
    """Copy the reportable parts of one run directory; return its summary."""
    run_dir = run_dir.resolve()
    summary = {"run_dir": str(run_dir), "name": run_dir.name, "files": [],
               "groups": {}, "verification": None, "code_versions": []}
    for name in ("verification_report.json", "plan.json", "suite_summary.json",
                 "failures.json", "source_manifest.json", "environment.json",
                 "preflight.json", "pip_freeze.txt"):
        if (run_dir / name).is_file():
            _copy(run_dir / name, dest / name)
    if (run_dir / "dataset_fingerprint.json").is_file():
        _strip_fingerprint(run_dir / "dataset_fingerprint.json",
                           dest / "dataset_fingerprint.json")
    for folder in ("source_manifests", "environments"):
        if (run_dir / folder).is_dir():
            for path in sorted((run_dir / folder).glob("*.json")):
                _copy(path, dest / folder / path.name)
    if (run_dir / "dataset_fingerprints").is_dir():
        for path in sorted((run_dir / "dataset_fingerprints").glob("*.json")):
            _strip_fingerprint(path, dest / "dataset_fingerprints" / path.name)
    if (run_dir / "merged").is_dir():
        for path in sorted((run_dir / "merged").iterdir()):
            if path.is_file():
                _copy(path, dest / "merged" / path.name)
        summary_path = run_dir / "merged" / "merge_summary.json"
        if summary_path.is_file():
            groups = json.loads(summary_path.read_text(encoding="utf-8")).get("groups", {})
            for group, info in groups.items():
                summary["groups"][group] = {
                    "verification_status": info.get("verification_status"),
                    "inputs": info.get("inputs"), "missing_jobs": info.get("missing_jobs"),
                    "seeds": info.get("seeds"), "benchmarks": info.get("benchmarks"),
                    "methods": info.get("methods")}
        for path in sorted((run_dir / "merged").glob("*.json")):
            if path.name == "merge_summary.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            version = payload.get("code_version")
            if version and version not in summary["code_versions"]:
                summary["code_versions"].append(version)
    for folder in ("paper", "figures", "audit"):  # P0 writes paper/, P1 runs write figures/
        if (run_dir / folder).is_dir():
            for path in sorted((run_dir / folder).rglob("*")):
                if path.is_file():
                    _copy(path, dest / folder / path.relative_to(run_dir / folder))
    if include_statuses and (run_dir / "statuses").is_dir():
        for path in sorted((run_dir / "statuses").glob("*.json")):
            _copy(path, dest / "statuses" / path.name)
    if include_artifacts and (run_dir / "artifacts").is_dir():
        for job_dir in sorted((run_dir / "artifacts").iterdir()):
            for name in ("result.json", "result.npz"):
                if (job_dir / name).is_file():
                    _copy(job_dir / name, dest / "artifacts" / job_dir.name / name)
    report = run_dir / "verification_report.json"
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        summary["verification"] = {
            "status": data.get("status"), "jobs_validated": data.get("jobs_validated"),
            "errors": len(data.get("errors", [])),
            "warnings": data.get("warnings", []),
            "acknowledged_source_changes": data.get("acknowledged_source_changes", [])}
    return summary


def copy_code(code_dir: Path, dest: Path) -> int:
    """Copy the code package without caches/editor state; return file count.

    Markdown files are copied with references to documentation outside the
    package replaced by a neutral note (see ``_EXTERNAL_DOC_PATTERN``).
    """
    import re
    count = 0
    for path in sorted(code_dir.rglob("*")):
        relative = path.relative_to(code_dir)
        if any(part in _CODE_EXCLUDED_DIRS or part.startswith(".") for part in relative.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8", errors="ignore")
            text = re.sub(_EXTERNAL_DOC_PATTERN, _EXTERNAL_DOC_NOTE, text)
            (dest / relative).parent.mkdir(parents=True, exist_ok=True)
            (dest / relative).write_text(text, encoding="utf-8")
        else:
            _copy(path, dest / relative)
        count += 1
    return count


def anonymity_check(root: Path, tokens: list[str]) -> list[str]:
    """Return ``path: token`` hits for identifying tokens in text files."""
    hits = []
    lowered = [t.lower() for t in tokens if t]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for token in lowered:
            if token in text:
                hits.append(f"{path.relative_to(root)}: {token}")
    return hits


def write_manifest(root: Path, runs: list[dict], supplementary: bool) -> dict:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append({"path": path.relative_to(root).as_posix(),
                          "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {
        "tool": "tools/build_results_package.py", "tool_version": TOOL_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "supplementary": supplementary, "runs": runs, "files": files,
        "file_count": len(files), "total_bytes": sum(f["bytes"] for f in files),
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_readme(root: Path, runs: list[dict], supplementary: bool) -> None:
    lines = ["# Results package", "",
             f"Built {datetime.now(timezone.utc).isoformat()} by tools/build_results_package.py.",
             "", "## Runs", ""]
    for run in runs:
        v = run["verification"] or {}
        lines.append(f"### {run['name']}")
        if not supplementary:      # absolute paths carry no meaning for reviewers
            lines.append(f"- source run directory: `{run['run_dir']}`")
        lines.append(f"- verification: **{v.get('status', 'n/a')}** "
                     f"({v.get('jobs_validated', '?')} jobs validated, "
                     f"{v.get('errors', '?')} errors, {len(v.get('warnings', []))} warnings)")
        if v.get("acknowledged_source_changes"):
            lines.append("- acknowledged source changes: " + "; ".join(
                f"{a.get('from', '')[:12]}->{a.get('to', '')[:12]}: {a.get('reason')}"
                + (f" (invalidated {a['invalidated_methods']})" if a.get("invalidated_methods") else "")
                for a in v["acknowledged_source_changes"]))
        lines.append(f"- code versions in merged artifacts: {run['code_versions']}")
        for group, info in run["groups"].items():
            lines.append(f"- `merged/{group}.json`: **{info.get('verification_status')}**, "
                         f"{info.get('inputs')} jobs, seeds {info.get('seeds')}, "
                         f"benchmarks {info.get('benchmarks')}, methods {info.get('methods')}")
        lines.append("")
    lines += ["## Layout", "",
              "- `<run>/merged/*.json|npz` - merged artifacts; `aggregated` holds mean/std/values per metric, "
              "`paired_significance` the paired tests against DiLT-CL, `merge_provenance` the input hashes.",
              "- `<run>/paper/` - LaTeX tables, figures (with `figure_provenance.json`), storage and timing measurements.",
              "- `<run>/verification_report.json` - independent recomputation of every metric from raw matrices.",
              "- `<run>/plan.json`, `suite_summary.json`, `failures.json`, `statuses/` - what ran, how long, under which source hash.",
              "- `<run>/source_manifest(s)`, `environment(s)`, `dataset_fingerprint(s)` - provenance (per-file lists stripped).",
              "- `MANIFEST.json` - SHA-256 of every file in this package.", ""]
    if supplementary:
        lines += ["## Code", "", "The `code/` folder is the exact package that produced these results. "
                  "See `HOW_TO_VERIFY.md`.", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_how_to_verify(root: Path, runs: list[dict]) -> None:
    run_names = [r["name"] for r in runs]
    text = f"""# How to verify these results

1. Install: `pip install -r code/requirements.txt` (PyTorch, torchvision, numpy, scipy).
2. Unit and equivalence tests (CPU, about a minute): `cd code && python -m pytest tests`.
3. Recompute every reported metric from the raw matrices of a run:
   `python code/tools/verify_experiment_bundle.py <run>` needs the full run directory
   (per-job artifacts); this package ships the verifier's report per run
   (`<run>/verification_report.json`). To re-check the merged artifacts against their
   own raw matrices without the run directory:
   `python code/tools/merge_results.py --inputs <run>/merged/<group>.json --out /tmp/x`
   refuses to write if any per-seed metric disagrees with the NPZ matrices.
4. Tables: `python code/make_class_il_table.py --results <run>/merged/p0_class_entropy.json --out table.tex`.
5. Figures: `python code/make_v3_figures.py --class-results ... --task-results ... --output-dir figs`.

Runs in this package: {', '.join(run_names)}.
Every artifact records `code_version`, `execution` (data backend, numerics mode, TF32 policy)
and the SHA-256 of the exact pixel tensors used per task.
"""
    (root / "HOW_TO_VERIFY.md").write_text(text, encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True,
                        help="Suite run directories to include.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: results_<date> in CWD); "
                             "a .tar.gz with the same name is written next to it.")
    parser.add_argument("--include-artifacts", action="store_true",
                        help="Also copy per-job result.json/result.npz (no checkpoints).")
    parser.add_argument("--no-statuses", action="store_true",
                        help="Skip per-job status files.")
    parser.add_argument("--supplementary", type=Path, default=None, metavar="CODE_DIR",
                        help="Copy the code package into code/ and write HOW_TO_VERIFY.md "
                             "(submission supplementary material).")
    parser.add_argument("--anonymity-check", default=None, metavar="TOKENS",
                        help="Comma-separated identifying tokens; the build fails if any "
                             "text file in the package contains one.")
    parser.add_argument("--no-tar", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing output directory.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out or Path(f"results_{datetime.now().strftime('%Y%m%d')}")
    out = out.resolve()
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    runs = []
    for run_dir in args.run_dirs:
        if not (run_dir / "plan.json").is_file():
            raise SystemExit(f"{run_dir} is not a suite run directory (no plan.json)")
        dest = out / run_dir.resolve().name
        runs.append(collect_run(run_dir, dest, args.include_artifacts,
                                not args.no_statuses))
        print(f"collected {run_dir} -> {dest.relative_to(out)}")
    if args.supplementary is not None:
        count = copy_code(args.supplementary.resolve(), out / "code")
        write_how_to_verify(out, runs)
        print(f"copied code package ({count} files) -> code/")
    write_readme(out, runs, args.supplementary is not None)
    if args.anonymity_check:
        tokens = [t.strip() for t in args.anonymity_check.split(",") if t.strip()]
        hits = anonymity_check(out, tokens)
        if hits:
            print("anonymity check FAILED:")
            for hit in hits[:50]:
                print("  " + hit)
            shutil.rmtree(out)
            return 2
        print(f"anonymity check passed ({len(tokens)} token(s))")
    manifest = write_manifest(out, runs, args.supplementary is not None)
    print(f"package: {out} ({manifest['file_count']} files, "
          f"{manifest['total_bytes'] / 2**20:.1f} MiB)")
    if not args.no_tar:
        archive = out.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(out, arcname=out.name)
        print(f"archive: {archive} ({archive.stat().st_size / 2**20:.1f} MiB)")
    for run in runs:
        v = run["verification"] or {}
        flag = "OK" if v.get("status") == "verified" else "NOT VERIFIED"
        print(f"  {run['name']}: {flag}; groups: "
              + ", ".join(f"{g}={i.get('verification_status')}" for g, i in run["groups"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
