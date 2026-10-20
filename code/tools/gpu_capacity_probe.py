#!/usr/bin/env python3
"""Measure per-job GPU memory on this machine and recommend ``--max-parallel``.

``run_suite.py --max-parallel auto`` needs a per-job GPU memory budget
(``--job-memory-mib``, default 2048 MiB). The right number depends on the
benchmark (TinyImageNet keeps ~1.3 GiB of pixels on the device), the method
(replay buffers, DER++ logits, the LwF teacher copy) and the CUDA context
size of the installed PyTorch build. Rather than guess, this tool launches
each probe as a *real* runner process under ``--fast`` -- same batch sizes,
same model, same device-resident dataset, fewer epochs -- and samples the GPU
memory while it runs.

Two measurements are attempted and the larger is used:

* per-process ``used_memory`` from ``nvidia-smi --query-compute-apps`` (not
  visible inside some containers because of PID namespaces), and
* the rise of the device's total ``memory.used`` over the idle baseline
  (accurate when nothing else shares the GPU, which is what the suite assumes).

Example::

    python tools/gpu_capacity_probe.py --data-root /root/autodl-tmp/data \\
        --threads-per-job 3 --out /root/autodl-tmp/gpu_probe.json

Then run the suite with the recommended values::

    python run_suite.py ... --max-parallel 8 --job-memory-mib 2600
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PROBES = [
    # (runner, benchmark, method, extra args) -- the memory-heaviest members
    # of each family; other methods use less.
    ("run_task_il.py", "cifar", "diltcl", ("--head-mode", "multi")),
    ("run_task_il.py", "cifar", "derpp", ("--head-mode", "multi")),
    ("run_task_il.py", "cifar", "lwf", ("--head-mode", "multi")),
    ("run_task_il.py", "pmnist", "derpp", ("--head-mode", "multi")),
    ("run_task_il.py", "tinyimagenet", "diltcl", ("--head-mode", "multi")),
    ("run_task_il.py", "tinyimagenet", "derpp", ("--head-mode", "multi")),
    ("run_class_il.py", "cifar", "diltcl",
     ("--inference-criterion", "entropy", "--inference-calibration",
      "score-logreg", "--calibration-fraction", "0.10")),
    ("run_class_il.py", "tinyimagenet", "derpp",
     ("--inference-criterion", "entropy")),
]


def _nvidia_smi(args):
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(["nvidia-smi", *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def device_memory():
    out = _nvidia_smi(["--query-gpu=memory.used,memory.total",
                       "--format=csv,noheader,nounits"])
    if not out or not out.strip():
        return None
    used, total = (int(float(v)) for v in out.strip().splitlines()[0].split(","))
    return used, total


def process_memory(pid):
    out = _nvidia_smi(["--query-compute-apps=pid,used_memory",
                       "--format=csv,noheader,nounits"])
    if not out:
        return None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return int(float(parts[1]))
            except ValueError:
                return None
    return None


def run_probe(runner, benchmark, method, extra, data_root, threads, log_dir,
              poll=0.5):
    baseline = device_memory()
    with tempfile.TemporaryDirectory() as tmp:
        command = [sys.executable, str(ROOT / runner), "--benchmark", benchmark,
                   "--seed-list", "0", "--methods", method, "--fast",
                   "--out", str(Path(tmp) / "result"), "--data-root",
                   str(data_root), "--no-download", "--threads", str(threads),
                   *extra]
        if runner == "run_task_il.py":
            command.append("--no-profile")
        log_path = log_dir / f"{Path(runner).stem}_{benchmark}_{method}.log"
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", str(threads))
        started = time.perf_counter()
        peak_process = 0
        peak_delta = 0
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, env=env)
            while process.poll() is None:
                used = process_memory(process.pid)
                if used is not None:
                    peak_process = max(peak_process, used)
                now = device_memory()
                if now is not None and baseline is not None:
                    peak_delta = max(peak_delta, now[0] - baseline[0])
                time.sleep(poll)
        return {
            "runner": runner, "benchmark": benchmark, "method": method,
            "extra_args": list(extra), "returncode": process.returncode,
            "wall_seconds_fast_profile": time.perf_counter() - started,
            "peak_process_mib": peak_process or None,
            "peak_device_delta_mib": peak_delta or None,
            "peak_mib": max(peak_process, peak_delta) or None,
            "log": str(log_path),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--threads-per-job", type=int, default=3)
    parser.add_argument("--benchmarks", default=None,
                        help="Comma list to restrict the probes (default: all "
                             "benchmarks staged under --data-root).")
    parser.add_argument("--headroom", type=float, default=1.15,
                        help="Safety factor applied to the measured peak.")
    parser.add_argument("--reserve-mib", type=int, default=1024,
                        help="Device memory kept free for the driver/display.")
    parser.add_argument("--max-parallel-cap", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the JSON report here (default: stdout only).")
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    memory = device_memory()
    if memory is None:
        raise SystemExit("nvidia-smi is unavailable; run this on the GPU machine")
    wanted = (set(args.benchmarks.split(",")) if args.benchmarks else None)
    probes = [p for p in DEFAULT_PROBES if wanted is None or p[1] in wanted]
    if wanted is None:
        # Skip TinyImageNet automatically when it is not staged.
        tin = args.data_root / "tiny-imagenet-200"
        if not tin.is_dir():
            probes = [p for p in probes if p[1] != "tinyimagenet"]
    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="gpu_probe_"))
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {memory[0]} / {memory[1]} MiB used before probing; "
          f"{len(probes)} probe(s); logs in {log_dir}", flush=True)
    results = []
    for runner, benchmark, method, extra in probes:
        print(f"[probe] {runner} {benchmark} {method} ...", end=" ", flush=True)
        record = run_probe(runner, benchmark, method, extra, args.data_root,
                           args.threads_per_job, log_dir)
        results.append(record)
        print(f"peak={record['peak_mib']} MiB rc={record['returncode']} "
              f"({record['wall_seconds_fast_profile']:.0f}s)", flush=True)

    measured = [r["peak_mib"] for r in results if r["peak_mib"]]
    failed = [r for r in results if r["returncode"] != 0]
    report = {"device_memory_total_mib": memory[1],
              "device_memory_idle_used_mib": memory[0],
              "threads_per_job": args.threads_per_job,
              "probes": results, "failed_probes": len(failed)}
    if measured:
        job_budget = int(max(measured) * args.headroom)
        try:
            cpu = len(os.sched_getaffinity(0))      # container-aware
        except (AttributeError, OSError):  # pragma: no cover - non-Linux
            cpu = os.cpu_count() or 1
        by_memory = max(1, (memory[1] - args.reserve_mib) // job_budget)
        by_cpu = max(1, cpu // max(1, args.threads_per_job))
        recommended = max(1, min(by_memory, by_cpu, args.max_parallel_cap))
        report.update({
            "recommended_job_memory_mib": job_budget,
            "recommended_max_parallel": int(recommended),
            "limits": {"by_memory": int(by_memory), "by_cpu": int(by_cpu),
                       "cap": args.max_parallel_cap},
            "suite_flags": (f"--max-parallel {recommended} --job-memory-mib "
                            f"{job_budget} --threads-per-job {args.threads_per_job}"),
        })
        print(f"\nRecommendation: {report['suite_flags']}")
    else:
        print("\nNo memory measurements succeeded; see the probe logs.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.out}")
    return 1 if failed or not measured else 0


if __name__ == "__main__":
    raise SystemExit(main())
