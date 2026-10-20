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

Seeds present only in the candidate are listed as new. Group identity (the
protocol, router, calibration settings and full configuration) must agree,
and the recorded software/hardware environments are compared so that any
difference can be traced to its cause.

``--pair REF.json CAND.json --states-only`` compares two differently named
groups that train the same experts with different routers (for example the
score-logreg, score-bias and score-bias-acc groups, which all train on the
same 90% split): only the trained-state hashes are then compared, which is how
independent trainings of the same experts are shown to agree bit for bit.

Torch-free (NumPy only). Exit status 0 when every shared run is identical.

Examples::

    python tools/compare_runs.py --reference OLD/p0_v3/merged OLD/p1_bias/merged \\
        --candidate NEW/merged --out regeneration_audit.json
    python tools/compare_runs.py --states-only \\
        --pair OLD/p0_v3/merged/p0_class_entropy_cal.json OLD/p1_bias/merged/p1_class_entropy_bias.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IDENTITY_FIELDS = ("protocol", "inference_criterion", "inference_calibration",
                   "calibration_fraction", "calibration_c", "calibration_split_seed",
                   "config")
ENVIRONMENT_FIELDS = ("python_version", "numpy_version", "pytorch_version", "device_name",
                      "cuda_version", "cudnn_version", "platform")


def merged_groups(paths: list[Path]) -> dict[str, Path]:
    """Map group name -> merged JSON path over one or more merged directories."""
    groups: dict[str, Path] = {}
    for path in paths:
        directory = path / "merged" if (path / "merged").is_dir() else path
        for json_path in sorted(directory.glob("*.json")):
            if json_path.name == "merge_summary.json" or not json_path.with_suffix(".npz").is_file():
                continue
            if json_path.stem in groups:
                raise SystemExit(f"group {json_path.stem} appears twice: "
                                 f"{groups[json_path.stem]} and {json_path}")
            groups[json_path.stem] = json_path
    return groups


def state_hash(payload: dict, benchmark: str, method: str, seed: int) -> str | None:
    record = (payload.get("per_seed_run_records", {}).get(benchmark, {})
              .get(method, {}).get(str(seed), {}))
    if "trained_state" in record:
        return record["trained_state"].get("content_sha256")
    return record.get("method_extras", {}).get("checkpoint", {}).get("content_sha256")


def seed_metrics(payload: dict, benchmark: str, method: str, seed: int) -> dict:
    rows = payload["per_seed"][benchmark][method]
    return rows[payload["seeds"].index(seed)]


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
           "runs": [], "new_runs": [], "missing_in_candidate": []}
    for field in IDENTITY_FIELDS:
        if ref.get(field) != cand.get(field):
            out["identity_differences"][field] = {"reference": ref.get(field),
                                                  "candidate": cand.get(field)}
    ref_env, cand_env = ref.get("environment") or {}, cand.get("environment") or {}
    for field in ENVIRONMENT_FIELDS:
        if ref_env.get(field) != cand_env.get(field):
            out["environment_differences"][field] = {"reference": ref_env.get(field),
                                                     "candidate": cand_env.get(field)}
    ref_exec = (ref.get("execution") or {}).get("options")
    cand_exec = (cand.get("execution") or {}).get("options")
    if ref_exec != cand_exec:
        out["identity_differences"]["execution.options"] = {"reference": ref_exec,
                                                            "candidate": cand_exec}
    with np.load(ref_path.with_suffix(".npz"), allow_pickle=False) as rz, \
            np.load(cand_path.with_suffix(".npz"), allow_pickle=False) as cz:
        for benchmark in sorted(set(ref["benchmarks"]) | set(cand["benchmarks"])):
            methods = sorted(set(ref.get("per_seed", {}).get(benchmark, {}))
                             | set(cand.get("per_seed", {}).get(benchmark, {})))
            for method in methods:
                ref_seeds = set(ref["seeds"]) if method in ref.get("per_seed", {}).get(benchmark, {}) else set()
                cand_seeds = set(cand["seeds"]) if method in cand.get("per_seed", {}).get(benchmark, {}) else set()
                for seed in sorted(cand_seeds - ref_seeds):
                    out["new_runs"].append(f"{benchmark}/{method}/seed{seed}")
                for seed in sorted(ref_seeds - cand_seeds):
                    out["missing_in_candidate"].append(f"{benchmark}/{method}/seed{seed}")
                for seed in sorted(ref_seeds & cand_seeds):
                    run = {"run": f"{benchmark}/{method}/seed{seed}"}
                    a, b = state_hash(ref, benchmark, method, seed), state_hash(cand, benchmark, method, seed)
                    run["state_identical"] = (a == b) if (a and b) else None
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
                    run["identical"] = bool(run["state_identical"] is not False
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
        "new": len(out["new_runs"]), "missing_in_candidate": len(out["missing_in_candidate"]),
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
    p.add_argument("--out", type=Path, default=None, help="Write a JSON report.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.pair:
        if args.reference or args.candidate:
            raise SystemExit("use either --pair or --reference/--candidate")
        ref_groups, cand_groups = {}, {}
        groups = [compare_group(f"{a.stem} vs {b.stem}", a, b, args.states_only)
                  for a, b in args.pair]
    else:
        if not (args.reference and args.candidate):
            raise SystemExit("--reference and --candidate are both required")
        ref_groups, cand_groups = merged_groups(args.reference), merged_groups(args.candidate)
        shared = sorted(set(ref_groups) & set(cand_groups))
        if not shared:
            raise SystemExit("no group appears in both the reference and the candidate")
        groups = [compare_group(name, ref_groups[name], cand_groups[name], args.states_only)
                  for name in shared]
    total = sum(g["summary"]["compared"] for g in groups)
    identical = sum(g["summary"]["identical"] for g in groups)
    clean_identity = args.states_only or all(not g["identity_differences"] for g in groups)
    status = "identical" if (identical == total and total and clean_identity) else "differences"
    report = {"status": status, "states_only": bool(args.states_only),
              "runs_compared": total, "runs_identical": identical,
              "groups": groups,
              "groups_only_in_reference": sorted(set(ref_groups) - set(cand_groups)),
              "groups_only_in_candidate": sorted(set(cand_groups) - set(ref_groups))}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for g in groups:
        s = g["summary"]
        print(f"{g['group']:28s} compared={s['compared']:4d} identical={s['identical']:4d} "
              f"same-state={s['state_identical']:4d} new={s['new']:4d}"
              + (f"  STATE DIFFERS: {len(s['state_different'])}" if s["state_different"] else "")
              + (f"  SAME STATE, DIFFERENT OUTPUTS: {s['same_state_different_outputs'][:3]}"
                 if s["same_state_different_outputs"] else "")
              + (f"  IDENTITY: {sorted(g['identity_differences'])}" if g["identity_differences"] else "")
              + (f"  env: {sorted(g['environment_differences'])}" if g["environment_differences"] else ""))
    print(f"{identical}/{total} shared runs identical -> {status}")
    return 0 if status == "identical" else 1


if __name__ == "__main__":
    raise SystemExit(main())
