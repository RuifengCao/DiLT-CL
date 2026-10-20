#!/usr/bin/env python3
"""Package complete run directories -- per-job results, checkpoints and
per-example predictions -- as verifiable release archives.

``tools/build_results_package.py`` packages the *merged* evidence of a run.
This tool packages everything needed to check it from the bottom up:

    <out>/
      <run>__meta.tar              plan.json, statuses/, merged/, verification and
                                   provenance records (everything except artifacts/)
      <run>__<group>[__<bench>].tar  artifacts/<job>/{result.json, result.npz,
                                   checkpoints/**, predictions/**} of one job group
      RELEASE_MANIFEST.json        every archive and every file inside it with its
                                   byte count and SHA-256
      RELEASE_README.md            how to unpack and verify

Before anything is written, every checkpoint file is checked against the
SHA-256 and byte count recorded in its job's ``result.json`` and every
predictions file against its recorded SHA-256; a mismatch aborts the build.
Archives are plain (uncompressed) tar files: checkpoints are already zip
containers and do not compress. A group whose archive would exceed
``--max-archive-gib`` (default 4.5, under OSF's 5 GB per-file limit) is split
by benchmark.

Extracting every archive of a run into one directory recreates a run
directory on which ``tools/verify_experiment_bundle.py``,
``tools/recompute_from_predictions.py`` and ``tools/evaluate_checkpoints.py``
run unchanged.

Example::

    python tools/build_release.py --run-dirs RUNS/p0 RUNS/p1_bias RUNS/p1_biasacc \\
        --out release_v305 --anonymity-check "surname,university,@example.org"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

TOOL_VERSION = "1.0"
_TEXT_SUFFIXES = {".json", ".txt", ".md", ".log", ".csv", ".tex"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(payload: dict):
    """Yield (label, record) for every persisted-state record in a result.json."""
    for b, by_method in (payload.get("per_seed_run_records") or {}).items():
        for m, by_seed in by_method.items():
            for s, record in by_seed.items():
                state = record.get("trained_state") or (record.get("method_extras") or {}).get("checkpoint")
                if state:
                    yield f"{b}/{m}/seed{s}", state


def check_job(job_dir: Path) -> tuple[list[str], list[str]]:
    """Return (errors, files) for one job directory."""
    errors, files = [], []
    result = job_dir / "result.json"
    if not result.is_file() or not result.with_suffix(".npz").is_file():
        return [f"{job_dir.name}: result.json/result.npz missing"], files
    payload = json.loads(result.read_text(encoding="utf-8"))
    files += [str(result), str(result.with_suffix(".npz"))]
    for label, state in _records(payload):
        rel = state.get("persisted_relative_path")
        if not state.get("persisted"):
            errors.append(f"{job_dir.name} {label}: checkpoint was not persisted")
            continue
        path = job_dir / rel if rel else None
        if path is None or not path.is_file():
            errors.append(f"{job_dir.name} {label}: checkpoint file missing ({rel})")
            continue
        if path.stat().st_size != state.get("serialized_bytes") or sha256_file(path) != state.get("sha256"):
            errors.append(f"{job_dir.name} {label}: checkpoint differs from its recorded SHA-256")
        files.append(str(path))
    for by_method in (payload.get("calibration") or {}).values():
        for by_seed in by_method.values():
            for record in by_seed.values():
                pred = (record or {}).get("predictions")
                if not pred:
                    continue
                path = job_dir / pred.get("relative_path", "")
                if not path.is_file():
                    errors.append(f"{job_dir.name}: predictions file missing ({pred.get('relative_path')})")
                elif sha256_file(path) != pred.get("sha256"):
                    errors.append(f"{job_dir.name}: predictions file differs from its recorded SHA-256")
                else:
                    files.append(str(path))
    return errors, files


def job_group(plan: dict) -> dict[str, tuple[str, str]]:
    """Map job id -> (group, benchmark) from a suite plan."""
    out = {}
    for entry in plan.get("jobs", []):
        job = entry.get("job", entry)
        out[job["job_id"]] = (job["job_id"].split("__")[0], job.get("benchmark", "all"))
    return out


def anonymity_hits(paths: list[Path], tokens: list[str]) -> list[str]:
    lowered = [t.lower() for t in tokens if t]
    hits = []
    for path in paths:
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        hits += [f"{path}: {t}" for t in lowered if t in text]
    return hits


def write_tar(path: Path, members: list[tuple[Path, str]]) -> list[dict]:
    entries = []
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
        for source, arcname in members:
            info = tar.gettarinfo(str(source), arcname=arcname)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0                          # reproducible archives
            with source.open("rb") as handle:
                tar.addfile(info, handle)
            entries.append({"path": arcname, "bytes": source.stat().st_size,
                            "sha256": sha256_file(source)})
    return entries


def build(run_dirs: list[Path], out: Path, max_bytes: int, tokens: list[str]) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    archives = []
    errors: list[str] = []
    plans = {}
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        plan_path = run_dir / "plan.json"
        if not plan_path.is_file():
            raise SystemExit(f"{run_dir}: no plan.json (not a suite run directory)")
        plans[run_dir] = json.loads(plan_path.read_text(encoding="utf-8"))
    selections = []
    for run_dir, plan in plans.items():
        groups = job_group(plan)
        by_part: dict[tuple[str, str], list[tuple[Path, str]]] = defaultdict(list)
        sizes: dict[str, int] = defaultdict(int)
        for job_id, (group, benchmark) in sorted(groups.items()):
            job_dir = run_dir / "artifacts" / job_id
            job_errors, files = check_job(job_dir)
            errors += [f"{run_dir.name}/{e}" for e in job_errors]
            for f in files:
                path = Path(f)
                by_part[(group, benchmark)].append((path, path.relative_to(run_dir).as_posix()))
                sizes[group] += path.stat().st_size
        meta = [(p, p.relative_to(run_dir).as_posix()) for p in sorted(run_dir.rglob("*"))
                if p.is_file() and "artifacts" not in p.relative_to(run_dir).parts[:1]]
        selections.append((run_dir, by_part, sizes, meta))
    hits = anonymity_hits([m for _, parts, _, meta in selections
                           for members in list(parts.values()) + [meta] for m, _ in members],
                          tokens)
    if errors or hits:
        raise SystemExit("release aborted:\n  " + "\n  ".join(errors + [f"anonymity: {h}" for h in hits]))
    for run_dir, by_part, sizes, meta in selections:
        name = run_dir.name
        path = out / f"{name}__meta.tar"
        archives.append({"archive": path.name, "run": name, "kind": "meta",
                         "files": write_tar(path, meta)})
        groups = sorted({g for g, _ in by_part})
        for group in groups:
            split = sizes[group] > max_bytes
            parts = sorted(b for g, b in by_part if g == group)
            chunks = [[b] for b in parts] if split else [parts]
            for chunk in chunks:
                members = [m for b in chunk for m in by_part[(group, b)]]
                suffix = f"__{chunk[0]}" if split else ""
                path = out / f"{name}__{group}{suffix}.tar"
                archives.append({"archive": path.name, "run": name, "kind": "jobs",
                                 "group": group, "benchmarks": chunk,
                                 "files": write_tar(path, members)})
    for entry in archives:
        path = out / entry["archive"]
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
        if entry["bytes"] > max_bytes * 1.05:
            print(f"warning: {entry['archive']} is {entry['bytes'] / 2**30:.2f} GiB "
                  f"(a single job group/benchmark exceeds --max-archive-gib)")
    manifest = {
        "tool": "tools/build_release.py", "tool_version": TOOL_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "runs": [r.name for r in plans], "archives": archives,
        "archive_count": len(archives),
        "total_bytes": sum(a["bytes"] for a in archives),
        "checkpoints": sum(1 for a in archives for f in a["files"] if f["path"].endswith(".pt")),
        "prediction_files": sum(1 for a in archives for f in a["files"]
                                if "/predictions/" in f["path"]),
    }
    (out / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "RELEASE_README.md").write_text(_readme(manifest), encoding="utf-8")
    return manifest


def _readme(manifest: dict) -> str:
    lines = [
        "# Release archives", "",
        (f"{manifest['archive_count']} archives, {manifest['total_bytes'] / 2**30:.2f} GiB, "
         f"{manifest['checkpoints']} checkpoints, {manifest['prediction_files']} per-example "
         "prediction files. `RELEASE_MANIFEST.json` lists the SHA-256 of every archive and of "
         "every file inside it."), "",
        "## Unpack", "",
        "Extract every archive of one run into the same directory:", "",
        "```bash",
        ("sha256sum -c <(python3 -c \"import json;m=json.load(open('RELEASE_MANIFEST.json'));"
         "[print(a['sha256']+'  '+a['archive']) for a in m['archives']]\")"),
    ]
    for run in manifest["runs"]:
        lines.append(f"mkdir -p {run} && for f in {run}__*.tar; do tar -xf \"$f\" -C {run}; done")
    lines += [
        "```", "",
        "## Verify (see REPRODUCE.md in the code package)", "",
        "```bash",
        "python tools/verify_experiment_bundle.py RUN            # every job: metrics, hashes, predictions",
        "python tools/recompute_from_predictions.py --run-dir RUN --refit",
        "python tools/evaluate_checkpoints.py --run-dir RUN --data-root DATA   # needs the datasets",
        "```", "",
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-archive-gib", type=float, default=4.5)
    p.add_argument("--anonymity-check", default="", metavar="TOKENS",
                   help="Comma-separated tokens (names, institution, e-mail) that must not "
                        "appear in any packaged text file.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = build(args.run_dirs, args.out, int(args.max_archive_gib * 2**30),
                     [t.strip() for t in args.anonymity_check.split(",")])
    for a in manifest["archives"]:
        print(f"{a['archive']:60s} {a['bytes'] / 2**30:7.3f} GiB  {len(a['files'])} files")
    print(f"{manifest['archive_count']} archives, {manifest['total_bytes'] / 2**30:.2f} GiB, "
          f"{manifest['checkpoints']} checkpoints, {manifest['prediction_files']} prediction files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
