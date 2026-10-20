#!/usr/bin/env python3
"""Regeneration audit: compare an independent re-run with retained artifacts.

Given the merged artifacts of a reference run (for example the results
behind the submitted paper) and of a candidate re-run (the same suite
profile, seeds and configuration, run again from scratch), this tool matches
every group by name and, for every ``(benchmark, method, seed)`` present in
both, compares

* the canonical tensor-content SHA-256 of the trained state (class-IL:
  ``per_seed_run_records.<b>.<m>.<seed>.trained_state``; task-IL:
  ``per_seed_run_records.<b>.<m>.<seed>.method_extras.checkpoint``) -- equal
  hashes mean the re-run regenerated the same experts bit for bit;
* every array stored for that run in the NPZ (result matrix and all
  auxiliary matrices), exactly;
* every per-seed metric in the JSON, exactly.

It also compares the SHA-256 and byte count of the persisted state *files*
(``sha256`` / ``serialized_bytes`` of the same records) and reports how many
files are byte-identical; equal content hashes with different file hashes
would mean the same tensors serialized differently.

The audit is complete only when the candidate covers the reference: every
reference group must be present (``--groups`` restricts the audit to named
groups), every reference ``(benchmark, method, seed)`` must be present in the
candidate, every merged JSON must have its NPZ, and both sides must record a
state hash for every run. Seeds present only in the candidate are listed as
new. Group identity (the protocol, router, calibration settings and full
configuration) must agree, and the recorded software/hardware environments
are compared so that any difference can be traced to its cause.

``--pair REF.json CAND.json --states-only`` compares two differently named
groups that train the same experts with different routers (for example the
score-logreg, score-bias and score-bias-acc groups, which all train on the
same 90% split): only the trained-state hashes are then compared, which is how
independent trainings of the same experts are shown to agree bit for bit.

When run directories (not only merged folders) are given on both sides, the
per-job status records are compared too: the dataset content hashes each job
read and the launch controls it ran with (``CUBLAS_WORKSPACE_CONFIG``, thread
counts, ``PYTHONHASHSEED``). Jobs launched before v3.0.1 recorded only a hash
of the whole data root; their per-entry hashes are derived from an archived
fingerprint with that root hash when one with entries or its per-file list is
available (the run directory's ``dataset_fingerprints/``, or ``--fingerprints``
folders), and are otherwise reported as not comparable. The SHA-256 of every
decoded split recorded in the merged artifacts (``materialized_pixels_sha256``
of the task views) is compared as well. All of these are reported, not part
of the verdict.

Torch-free (NumPy only). Exit status 0 only when the audit is complete and
every compared run is identical (status ``identical``); otherwise the status
is ``differences`` or ``incomplete`` and the exit status is 1.

Examples::

    python tools/compare_runs.py --reference OLD/p0_v3/merged OLD/p1_bias/merged \\
        --candidate NEW/merged --out regeneration_audit.json
    python tools/compare_runs.py --states-only \\
        --pair OLD/p0_v3/merged/p0_class_entropy_cal.json OLD/p1_bias/merged/p1_class_entropy_bias.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

IDENTITY_FIELDS = ("protocol", "inference_criterion", "inference_calibration",
                   "calibration_fraction", "calibration_c", "calibration_split_seed",
                   "config")
ENVIRONMENT_FIELDS = ("python_version", "numpy_version", "pytorch_version", "device_name",
                      "cuda_version", "cudnn_version", "platform")


def merged_groups(paths: list[Path], unpaired: list[str] | None = None) -> dict[str, Path]:
    """Map group name -> merged JSON path over one or more merged directories.

    A group JSON without its NPZ cannot be audited; its path is appended to
    ``unpaired`` (which makes the audit incomplete) instead of being skipped
    silently.
    """
    groups: dict[str, Path] = {}
    for path in paths:
        directory = path / "merged" if (path / "merged").is_dir() else path
        if not directory.is_dir():
            raise SystemExit(f"not a directory: {directory}")
        for json_path in sorted(directory.glob("*.json")):
            if json_path.name == "merge_summary.json":
                continue
            if not json_path.with_suffix(".npz").is_file():
                if unpaired is not None:
                    unpaired.append(str(json_path))
                continue
            if json_path.stem in groups:
                raise SystemExit(f"group {json_path.stem} appears twice: "
                                 f"{groups[json_path.stem]} and {json_path}")
            groups[json_path.stem] = json_path
    return groups


LAUNCH_KEYS = ("CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "DILTCL_NUM_THREADS",
               "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")


def run_statuses(paths: list[Path]) -> dict[str, dict]:
    """Job id -> status record, for every run directory among ``paths``.

    Each record carries the run directory it was read from (``_run_dir``),
    where archived dataset fingerprints are looked up.
    """
    out: dict[str, dict] = {}
    for path in paths:
        run = path if (path / "merged").is_dir() else path.parent
        statuses = run / "statuses"
        if not statuses.is_dir():
            continue
        for status_path in sorted(statuses.glob("*.json")):
            try:
                record = json.loads(status_path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            record["_run_dir"] = str(run)
            out[status_path.stem] = record
    return out


class FingerprintArchive:
    """Archived dataset fingerprints by root hash, with per-entry hashes."""

    def __init__(self, extra_dirs: list[Path] | None = None):
        self.extra_dirs = [Path(d) for d in (extra_dirs or [])]
        self._entries: dict[tuple[str, str], dict | None] = {}

    def _candidates(self, run_dir: str | None, tree: str) -> list[Path]:
        roots = ([Path(run_dir)] if run_dir else []) + self.extra_dirs
        out = []
        for root in roots:
            out += [root / "dataset_fingerprints" / f"{tree}.json", root / f"{tree}.json",
                    root / "dataset_fingerprint.json"]
        return out

    def entries(self, run_dir: str | None, tree: str) -> dict | None:
        key = (run_dir or "", tree)
        if key not in self._entries:
            self._entries[key] = None
            for path in self._candidates(run_dir, tree):
                if not path.is_file():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                if data.get("tree_sha256") != tree:
                    continue
                if data.get("entries") is not None:
                    self._entries[key] = data["entries"]
                elif data.get("files"):
                    fingerprint_entries, _ = _suite_helpers()
                    self._entries[key] = fingerprint_entries(data["files"])
                if self._entries[key] is not None:
                    break
        return self._entries[key]


def _suite_helpers():
    """The suite's own entry hashing and benchmark-to-entry map (torch-free).

    ``run_suite`` loads ``diltcl/preflight.py`` by path, so importing it does
    not import the package (and PyTorch).
    """
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import run_suite
    return run_suite.fingerprint_entries, run_suite.preflight.benchmark_data_entries


def job_entries(status: dict, archive: FingerprintArchive) -> tuple[dict | None, bool]:
    """``(entry -> sha256, derived)`` of the data a job read, if known.

    Jobs launched before v3.0.1 recorded only the whole data root's hash; their
    entries are derived from an archived fingerprint with that hash.
    """
    identity = status.get("dataset_identity") or {}
    if identity.get("entries") is not None:
        return identity["entries"], False
    tree = identity.get("tree_sha256")
    benchmark = identity.get("benchmark") or (status.get("job") or {}).get("benchmark")
    if not tree or not benchmark:
        return None, False
    entries = archive.entries(status.get("_run_dir"), tree)
    if entries is None:
        return None, False
    _, benchmark_data_entries = _suite_helpers()
    return {name: (entries.get(name) or {}).get("sha256")
            for name in benchmark_data_entries(benchmark)}, True


def compare_launch(ref_status: dict, cand_status: dict, groups: set[str],
                   archive: FingerprintArchive | None = None) -> dict:
    """Same data and same launch controls, job by job (reported, not a verdict)."""
    archive = archive or FingerprintArchive()
    report = {"jobs_compared": 0, "jobs_without_candidate_status": [],
              "dataset_differences": [], "dataset_not_comparable": [],
              "dataset_entries_derived": [],
              "launch_differences": [], "cache_dir_differences": 0}
    for job_id, ref in sorted(ref_status.items()):
        if job_id.split("__")[0] not in groups or ref.get("job", {}).get("repeat_of"):
            continue
        cand = cand_status.get(job_id)
        if cand is None:
            report["jobs_without_candidate_status"].append(job_id)
            continue
        report["jobs_compared"] += 1
        ref_entries, ref_derived = job_entries(ref, archive)
        cand_entries, cand_derived = job_entries(cand, archive)
        if ref_derived or cand_derived:
            report["dataset_entries_derived"].append(job_id)
        if ref_entries is None or cand_entries is None:
            # Launched before v3.0.1 (a hash of the whole data root only) and
            # no archived fingerprint with that hash to derive entries from.
            report["dataset_not_comparable"].append(job_id)
        elif ref_entries != cand_entries:
            report["dataset_differences"].append(job_id)
        rc = ref.get("child_environment_controls") or {}
        cc = cand.get("child_environment_controls") or {}
        keys = [k for k in LAUNCH_KEYS if rc.get(k) != cc.get(k)]
        if keys:
            report["launch_differences"].append({"job": job_id, "keys": keys})
        if rc.get("DILTCL_CACHE_DIR") != cc.get("DILTCL_CACHE_DIR"):
            report["cache_dir_differences"] += 1
    return report


def state_record(payload: dict, benchmark: str, method: str, seed: int) -> dict:
    """The persisted-state record of one run (class-IL or task-IL layout)."""
    record = (payload.get("per_seed_run_records", {}).get(benchmark, {})
              .get(method, {}).get(str(seed), {}))
    if "trained_state" in record:
        return record["trained_state"] or {}
    return record.get("method_extras", {}).get("checkpoint", {}) or {}


def state_hash(payload: dict, benchmark: str, method: str, seed: int) -> str | None:
    return state_record(payload, benchmark, method, seed).get("content_sha256")


def state_file(payload: dict, benchmark: str, method: str, seed: int):
    """``(sha256, serialized_bytes)`` of the persisted state file, if recorded."""
    record = state_record(payload, benchmark, method, seed)
    if not record.get("sha256"):
        return None
    return record.get("sha256"), record.get("serialized_bytes")


def method_seeds(payload: dict, benchmark: str, method: str) -> list[int]:
    """Seeds of one method, in the order of its per-seed rows.

    A merge with ``--allow-incomplete`` stores fewer rows than the artifact's
    seed list; the run records (one per seed actually present) align them.
    """
    rows = payload.get("per_seed", {}).get(benchmark, {}).get(method)
    if rows is None:
        return []
    records = payload.get("per_seed_run_records", {}).get(benchmark, {}).get(method) or {}
    if records and len(records) == len(rows):
        return sorted(int(seed) for seed in records)
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    if len(rows) == len(seeds):
        return seeds
    raise ValueError(f"{benchmark}/{method}: {len(rows)} per-seed rows for seeds {seeds} "
                     "and no run records to align them")


def seed_metrics(payload: dict, benchmark: str, method: str, seed: int) -> dict:
    rows = payload["per_seed"][benchmark][method]
    return rows[method_seeds(payload, benchmark, method).index(seed)]


def pixel_hashes(payload: dict) -> dict[str, str]:
    """``benchmark/seed/family/split/task -> materialized_pixels_sha256``.

    Every task view records the hash of the decoded split it indexes (task-IL
    under ``benchmark_metadata.<b>.per_seed``, class-IL under
    ``per_seed_counts`` per method family); test splits since v3.0.7.
    """
    out: dict[str, str] = {}
    for benchmark, meta in (payload.get("benchmark_metadata") or {}).items():
        layouts = [(seed, "", record) for seed, record in (meta.get("per_seed") or {}).items()]
        for seed, families in (meta.get("per_seed_counts") or {}).items():
            layouts += [(seed, family, record) for family, record in (families or {}).items()]
        for seed, family, record in layouts:
            if not isinstance(record, dict):
                continue
            for split, keys in (("train", ("tasks", "dataset_views")),
                                ("test", ("test_tasks", "test_dataset_views"))):
                for key in keys:
                    for view in record.get(key) or []:
                        digest = (view or {}).get("materialized_pixels_sha256")
                        if digest:
                            out[f"{benchmark}/seed{seed}/{family or 'all'}/{split}/"
                                f"task{view.get('task')}"] = digest
    return out


def compare_pixels(ref: dict, cand: dict) -> dict:
    ref_hashes, cand_hashes = pixel_hashes(ref), pixel_hashes(cand)
    shared = sorted(set(ref_hashes) & set(cand_hashes))
    return {"views_compared": len(shared),
            "different": [k for k in shared if ref_hashes[k] != cand_hashes[k]],
            "only_in_reference": len(set(ref_hashes) - set(cand_hashes)),
            "only_in_candidate": len(set(cand_hashes) - set(ref_hashes))}


def run_keys(npz, benchmark: str, method: str, seed: int) -> list[str]:
    base = f"{benchmark}/{method}/seed{seed}"
    return sorted(k for k in npz.files if k == base or k.startswith(base + "/"))


def compare_group(name: str, ref_path: Path, cand_path: Path,
                  states_only: bool = False) -> dict:
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    cand = json.loads(cand_path.read_text(encoding="utf-8"))
    out = {"group": name, "reference": str(ref_path), "candidate": str(cand_path),
           "identity_differences": {}, "environment_differences": {},
           "code_versions": {"reference": ref.get("code_version"),
                             "candidate": cand.get("code_version")},
           "runs": [], "new_runs": [], "missing_in_candidate": [],
           "alignment_errors": [], "methods_only_in_reference": [],
           "methods_only_in_candidate": []}
    for field in IDENTITY_FIELDS:
        if ref.get(field) != cand.get(field):
            out["identity_differences"][field] = {"reference": ref.get(field),
                                                  "candidate": cand.get(field)}
    ref_env, cand_env = ref.get("environment") or {}, cand.get("environment") or {}
    for field in ENVIRONMENT_FIELDS:
        if ref_env.get(field) != cand_env.get(field):
            out["environment_differences"][field] = {"reference": ref_env.get(field),
                                                     "candidate": cand_env.get(field)}
    out["pixels"] = compare_pixels(ref, cand)
    ref_exec = (ref.get("execution") or {}).get("options")
    cand_exec = (cand.get("execution") or {}).get("options")
    if ref_exec != cand_exec:
        out["identity_differences"]["execution.options"] = {"reference": ref_exec,
                                                            "candidate": cand_exec}
    with np.load(ref_path.with_suffix(".npz"), allow_pickle=False) as rz, \
            np.load(cand_path.with_suffix(".npz"), allow_pickle=False) as cz:
        for benchmark in sorted(set(ref["benchmarks"]) | set(cand["benchmarks"])):
            ref_methods = set(ref.get("per_seed", {}).get(benchmark, {}))
            cand_methods = set(cand.get("per_seed", {}).get(benchmark, {}))
            if states_only:
                # Groups with different routers may train different method
                # sets (the 90%-split calibration group also trains the five
                # shared-backbone methods); compare the experts they share.
                out["methods_only_in_reference"] += [f"{benchmark}/{m}" for m in sorted(ref_methods - cand_methods)]
                out["methods_only_in_candidate"] += [f"{benchmark}/{m}" for m in sorted(cand_methods - ref_methods)]
                methods = sorted(ref_methods & cand_methods)
            else:
                methods = sorted(ref_methods | cand_methods)
            for method in methods:
                try:
                    ref_seeds = set(method_seeds(ref, benchmark, method))
                    cand_seeds = set(method_seeds(cand, benchmark, method))
                except ValueError as exc:
                    out["alignment_errors"].append(str(exc))
                    continue
                for seed in sorted(cand_seeds - ref_seeds):
                    out["new_runs"].append(f"{benchmark}/{method}/seed{seed}")
                for seed in sorted(ref_seeds - cand_seeds):
                    out["missing_in_candidate"].append(f"{benchmark}/{method}/seed{seed}")
                for seed in sorted(ref_seeds & cand_seeds):
                    run = {"run": f"{benchmark}/{method}/seed{seed}"}
                    a, b = state_hash(ref, benchmark, method, seed), state_hash(cand, benchmark, method, seed)
                    run["state_identical"] = (a == b) if (a and b) else None
                    fa, fb = state_file(ref, benchmark, method, seed), state_file(cand, benchmark, method, seed)
                    run["file_identical"] = (fa == fb) if (fa and fb) else None
                    rk, ck = run_keys(rz, benchmark, method, seed), run_keys(cz, benchmark, method, seed)
                    run["array_keys_match"] = rk == ck
                    diffs = {}
                    for key in sorted(set(rk) & set(ck)):
                        x, y = np.asarray(rz[key]), np.asarray(cz[key])
                        if x.shape != y.shape:
                            diffs[key] = "shape"
                        elif not np.array_equal(x, y):
                            diffs[key] = float(np.max(np.abs(x.astype(float) - y.astype(float))))
                    run["array_differences"] = diffs
                    rm, cm = seed_metrics(ref, benchmark, method, seed), seed_metrics(cand, benchmark, method, seed)
                    run["metric_differences"] = {
                        k: (cm.get(k), rm.get(k)) for k in sorted(set(rm) | set(cm))
                        if rm.get(k) != cm.get(k)}
                    run["outputs_identical"] = (run["array_keys_match"] and not diffs
                                                and not run["metric_differences"])
                    if states_only:
                        run["outputs_identical"] = None
                        run["identical"] = run["state_identical"] is True
                        out["runs"].append(run)
                        continue
                    # A run without a state hash on either side cannot show that
                    # the experts were regenerated, so it never counts as identical.
                    run["identical"] = bool(run["state_identical"] is True
                                            and run["outputs_identical"])
                    out["runs"].append(run)
    runs = out["runs"]
    out["summary"] = {
        "compared": len(runs),
        "identical": sum(r["identical"] for r in runs),
        "state_identical": sum(r["state_identical"] is True for r in runs),
        "state_hash_unavailable": sum(r["state_identical"] is None for r in runs),
        "state_different": [r["run"] for r in runs if r["state_identical"] is False],
        "outputs_different": [r["run"] for r in runs if r["outputs_identical"] is False],
        # The same experts evaluated differently would mean nondeterministic
        # evaluation, a different router, or different data -- never benign.
        "same_state_different_outputs": [r["run"] for r in runs
                                         if r["state_identical"] and r["outputs_identical"] is False],
        "file_identical": sum(r.get("file_identical") is True for r in runs),
        "file_hash_unavailable": sum(r.get("file_identical") is None for r in runs),
        "file_different": [r["run"] for r in runs if r.get("file_identical") is False],
        "new": len(out["new_runs"]), "missing_in_candidate": len(out["missing_in_candidate"]),
        "alignment_errors": len(out["alignment_errors"]),
    }
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, nargs="+", default=None,
                   help="Merged directories (or run directories) of the reference run(s).")
    p.add_argument("--candidate", type=Path, nargs="+", default=None,
                   help="Merged directories (or run directories) of the re-run.")
    p.add_argument("--pair", type=Path, nargs=2, action="append", default=None,
                   metavar=("REF_JSON", "CAND_JSON"),
                   help="Compare two merged JSON files directly (repeatable).")
    p.add_argument("--states-only", action="store_true",
                   help="Compare only trained-state hashes (groups with different routers).")
    p.add_argument("--groups", nargs="+", default=None, metavar="GROUP",
                   help="Audit only these groups (each must exist in both runs); "
                        "default: every group of the reference.")
    p.add_argument("--fingerprints", type=Path, nargs="+", default=None, metavar="DIR",
                   help="Folders holding archived dataset fingerprints with their per-file "
                        "lists (e.g. an original run directory), used to derive per-entry "
                        "data hashes for jobs launched before v3.0.1.")
    p.add_argument("--out", type=Path, default=None, help="Write a JSON report.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    unpaired: list[str] = []
    missing_groups: list[str] = []
    if args.pair:
        if args.reference or args.candidate or args.groups:
            raise SystemExit("use either --pair or --reference/--candidate[/--groups]")
        ref_groups, cand_groups = {}, {}
        for a, b in args.pair:
            for path in (a, b):
                if not path.with_suffix(".npz").is_file():
                    unpaired.append(str(path))
        if unpaired:
            raise SystemExit(f"merged JSON without its NPZ: {unpaired}")
        groups = [compare_group(f"{a.stem} vs {b.stem}", a, b, args.states_only)
                  for a, b in args.pair]
    else:
        if not (args.reference and args.candidate):
            raise SystemExit("--reference and --candidate are both required")
        ref_groups = merged_groups(args.reference, unpaired)
        cand_groups = merged_groups(args.candidate, unpaired)
        wanted = sorted(args.groups) if args.groups else sorted(ref_groups)
        unknown = [g for g in wanted if g not in ref_groups]
        if unknown:
            raise SystemExit(f"--groups not in the reference: {unknown}")
        missing_groups = [g for g in wanted if g not in cand_groups]
        shared = [g for g in wanted if g in cand_groups]
        if not shared:
            raise SystemExit("no reference group appears in the candidate "
                             f"(missing: {missing_groups})")
        groups = [compare_group(name, ref_groups[name], cand_groups[name], args.states_only)
                  for name in shared]
    total = sum(g["summary"]["compared"] for g in groups)
    identical = sum(g["summary"]["identical"] for g in groups)
    files_identical = sum(g["summary"]["file_identical"] for g in groups)
    missing_runs = sum(g["summary"]["missing_in_candidate"] for g in groups)
    no_state_hash = sum(g["summary"]["state_hash_unavailable"] for g in groups)
    misaligned = sum(g["summary"]["alignment_errors"] for g in groups)
    clean_identity = args.states_only or all(not g["identity_differences"] for g in groups)
    complete = (not (missing_groups or missing_runs or unpaired or no_state_hash or misaligned)
                and total > 0)
    if identical == total and clean_identity:
        status = "identical" if complete else "incomplete"
    else:
        status = "differences"
    launch = None
    if not args.pair:
        ref_status, cand_status = run_statuses(args.reference), run_statuses(args.candidate)
        if ref_status and cand_status:
            launch = compare_launch(ref_status, cand_status, {g["group"] for g in groups},
                                    FingerprintArchive(args.fingerprints))
    pixel_views = sum(g["pixels"]["views_compared"] for g in groups)
    pixel_different = sum(len(g["pixels"]["different"]) for g in groups)
    report = {"status": status, "states_only": bool(args.states_only),
              "complete": complete, "data_and_launch": launch,
              "decoded_splits": {"views_compared": pixel_views, "different": pixel_different},
              "runs_compared": total, "runs_identical": identical,
              "state_files_identical": files_identical,
              "runs_missing_in_candidate": missing_runs,
              "runs_without_state_hash": no_state_hash,
              "groups_missing_in_candidate": missing_groups,
              "unpaired_json": unpaired,
              "groups": groups,
              "groups_only_in_reference": sorted(set(ref_groups) - set(cand_groups)),
              "groups_only_in_candidate": sorted(set(cand_groups) - set(ref_groups))}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for g in groups:
        s = g["summary"]
        print(f"{g['group']:28s} compared={s['compared']:4d} identical={s['identical']:4d} "
              f"same-state={s['state_identical']:4d} same-file={s['file_identical']:4d} "
              f"new={s['new']:4d} missing={s['missing_in_candidate']:4d}"
              + (f"  STATE DIFFERS: {len(s['state_different'])}" if s["state_different"] else "")
              + (f"  NO STATE HASH: {s['state_hash_unavailable']}" if s["state_hash_unavailable"] else "")
              + (f"  SAME STATE, DIFFERENT OUTPUTS: {s['same_state_different_outputs'][:3]}"
                 if s["same_state_different_outputs"] else "")
              + (f"  IDENTITY: {sorted(g['identity_differences'])}" if g["identity_differences"] else "")
              + (f"  env: {sorted(g['environment_differences'])}" if g["environment_differences"] else ""))
    for name in missing_groups:
        print(f"{name:28s} MISSING IN CANDIDATE")
    for g in groups:
        for message in g["alignment_errors"]:
            print(f"{g['group']:28s} CANNOT ALIGN: {message}")
        only_ref, only_cand = g["methods_only_in_reference"], g["methods_only_in_candidate"]
        if only_ref or only_cand:
            print(f"{g['group']:28s} not compared (trained in one group only): "
                  f"{len(only_ref)} method(s) only in the reference, "
                  f"{len(only_cand)} only in the candidate")
    for path in unpaired:
        print(f"merged JSON without its NPZ: {path}")
    if launch is not None:
        print(f"data and launch controls: {launch['jobs_compared']} job(s) compared, "
              f"{len(launch['dataset_differences'])} with different data"
              + (f" ({len(launch['dataset_not_comparable'])} not comparable: launched before "
                 "v3.0.1, which hashed only the whole data root; see --fingerprints)"
                 if launch["dataset_not_comparable"] else "")
              + (f" ({len(launch['dataset_entries_derived'])} from archived per-file lists)"
                 if launch["dataset_entries_derived"] else "")
              + ", "
              f"{len(launch['launch_differences'])} with different launch controls"
              + (f", {len(launch['jobs_without_candidate_status'])} without a status"
                 if launch["jobs_without_candidate_status"] else ""))
    print(f"decoded splits: {pixel_views} task view(s) compared, {pixel_different} with "
          "different pixels")
    print(f"{identical}/{total} runs identical (state files {files_identical}/{total} identical); "
          f"missing: {missing_runs} run(s), {len(missing_groups)} group(s) -> {status}")
    return 0 if status == "identical" else 1


if __name__ == "__main__":
    raise SystemExit(main())
