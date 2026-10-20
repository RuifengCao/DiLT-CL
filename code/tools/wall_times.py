#!/usr/bin/env python3
"""Wall-time summary of a suite run from its final job-status records.

Every job's status file records ``wall_seconds``: the elapsed time of the
whole child process (training, evaluations and artifact writing), measured
while other jobs shared the GPU. This tool groups the succeeded jobs by
``(group, benchmark, method)`` and prints the number of jobs, the mean and
the sample standard deviation in minutes; ``--out-csv`` writes the same rows.
The values describe one concurrently scheduled run, not isolated training
speed.

``p0_v3/paper/wall_minutes.txt`` in ``evidence/results_20260906`` was written
before SupSup and WSN were rerun under v3.0.2; its rows for those two methods
time the invalidated runs. The statuses hold the final records, and this tool
recomputes the summary from them.

Example::

    python tools/wall_times.py ../evidence/results_20260906/p0_v3 --out-csv wall.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def wall_minutes(run_dir: Path) -> dict[tuple[str, str, str], list[float]]:
    """``(group, benchmark, methods) -> [minutes per job]`` of succeeded jobs."""
    samples: dict[tuple[str, str, str], list[float]] = {}
    statuses = sorted((run_dir / "statuses").glob("*.json"))
    if not statuses:
        raise SystemExit(f"{run_dir}: no statuses/*.json")
    for path in statuses:
        status = json.loads(path.read_text(encoding="utf-8"))
        job = status.get("job") or {}
        if status.get("state") != "succeeded" or job.get("repeat_of"):
            continue
        key = (job["job_id"].split("__")[0], job["benchmark"], job["methods"])
        samples.setdefault(key, []).append(float(status["wall_seconds"]) / 60.0)
    return samples


def summary_rows(samples: dict[tuple[str, str, str], list[float]]) -> list[dict]:
    rows = []
    for (group, benchmark, method), values in sorted(samples.items()):
        rows.append({"group": group, "benchmark": benchmark, "method": method,
                     "n": len(values), "mean_minutes": round(statistics.fmean(values), 3),
                     "sd_minutes": (round(statistics.stdev(values), 3)
                                    if len(values) > 1 else None)})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Suite run directories.")
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = []
    for run_dir in args.run_dirs:
        rows += summary_rows(wall_minutes(run_dir))
    for row in rows:
        sd = "--" if row["sd_minutes"] is None else f"{row['sd_minutes']:.1f}"
        print(f"{row['group']:26s} {row['benchmark']:13s} {row['method']:12s} "
              f"n={row['n']} mean={row['mean_minutes']:6.1f} min  sd={sd}")
    if args.out_csv:
        with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["group"])
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
