#!/usr/bin/env python3
"""Resumable, parallel launcher for the DiLT-CL experiment matrix.

The scientific runners intentionally remain simple. This launcher adds the
operational layer needed for long GPU runs: one process per job, timestamped
logs, atomic status files, exact commands, environment/source/data
fingerprints, failure records, resumability, and a post-run verifier.

v3 adds the throughput layer that a small model on a large GPU needs:

* **Job granularity.** By default every ``(benchmark, seed, method)`` triple
  is its own job (``--job-granularity method``), so a long DiLT-CL run never
  blocks nine finished baselines and a failed method is retried alone.
  ``--job-granularity seed`` restores the v2 one-job-per-seed layout.
* **Parallel execution.** ``--max-parallel N`` runs N jobs at once on the same
  GPU (``auto`` derives N from the CPU count, the job memory budget and the
  free GPU memory). Jobs are dispatched longest-first (LPT scheduling), each
  child is capped to ``--threads-per-job`` CPU threads, and a GPU-memory guard
  delays a launch while the device is nearly full.
* **Execution options** (``--data-backend``, ``--numerics``, ``--tf32``,
  ``--eval-batch-size``) are forwarded to every child and are part of the
  plan identity, so a run directory can never mix settings.
* **One suite per run directory.** The launcher holds an exclusive lock on
  its run directory (``.run_suite.lock``); a second launcher started on the
  same directory stops before writing anything.

Per-job results are merged across seeds and methods with
``tools/merge_results.py``.

Examples
--------
Print the jobs of a smoke run, without importing PyTorch or writing anything::

    python run_suite.py --profile smoke --dry-run

Run the P0 matrix with five paired seeds, eight jobs at a time::

    python run_suite.py --profile p0 --seeds 0-4 --head-mode multi \
        --data-root /datasets --run-dir /runs/p0_v3 --max-parallel 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


def _load_by_path(module_name: str, filename: str):
    """Load a torch-free ``diltcl`` module directly from its file.

    The launcher is deliberately torch-free: it probes PyTorch in a subprocess
    and must stay importable on a node where the training stack is not
    installed yet. Importing ``diltcl`` normally would pull in torch.
    """
    import importlib.util
    path = Path(__file__).resolve().parent / "diltcl" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules, so the
    # module must be registered before it executes.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load_by_path("diltcl_preflight", "preflight.py")
execution = _load_by_path("diltcl_execution", "execution.py")

# Rough relative cost of a job, used only to order dispatch (longest first).
_BENCHMARK_COST = {"cifar": 1.0, "pmnist": 0.5, "tinyimagenet": 3.0,
                   "fivedatasets": 1.5}
_METHOD_COST = {"diltcl": 6.0, "lwf": 1.8, "derpp": 1.6, "er": 1.3,
                "ewc": 1.3, "packnet": 1.2, "independent": 1.0,
                "finetune": 1.0, "supsup": 1.0, "wsn": 1.0}
DEFAULT_JOB_MEMORY_MIB = 2048     # conservative per-job GPU budget for 'auto'
DEFAULT_THREADS_PER_JOB = 3
MAX_AUTO_PARALLEL = 8

ROOT = Path(__file__).resolve().parent
TASK_METHODS = (
    "finetune,ewc,lwf,er,derpp,packnet,supsup,wsn,diltcl,independent"
)
CLASS_METHODS = "finetune,ewc,lwf,er,derpp,packnet,supsup,wsn,diltcl"
ISOLATION_CLASS_METHODS = "packnet,supsup,wsn,diltcl"
COMPUTE_MATCHED_COMMON = (
    "finetune,ewc,lwf,er,derpp,supsup,wsn,independent"
)


@dataclass(frozen=True)
class Job:
    job_id: str
    priority: str
    purpose: str
    runner: str
    benchmark: str
    seed: int
    methods: str
    extra_args: tuple[str, ...] = ()
    config_overrides: dict[str, Any] = field(default_factory=dict)
    fast: bool = False
    repeat_of: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def parse_seed_spec(spec: str) -> list[int]:
    seeds: list[int] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if "-" in token[1:]:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending seed range is invalid: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seed IDs must be a non-empty list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed IDs must be unique")
    return seeds


def _job(group: str, priority: str, purpose: str, runner: str,
         benchmark: str, seed: int, methods: str, *,
         extra_args: tuple[str, ...] = (),
         config: dict[str, Any] | None = None, fast: bool = False,
         repeat_of: str | None = None, method: str | None = None) -> Job:
    suffix = f"__{method}" if method else ""
    return Job(
        job_id=f"{group}__{benchmark}__seed{seed}{suffix}",
        priority=priority,
        purpose=purpose,
        runner=runner,
        benchmark=benchmark,
        seed=seed,
        methods=methods,
        extra_args=extra_args,
        config_overrides=config or {},
        fast=fast,
        repeat_of=repeat_of,
    )


def smoke_jobs(seeds: list[int]) -> list[Job]:
    seed = seeds[0]
    first = f"smoke_task_il__cifar__seed{seed}"
    return [
        _job("smoke_task_il", "P0-smoke", "real-autograd Task-IL smoke",
             "run_task_il.py", "cifar", seed, TASK_METHODS,
             extra_args=("--no-profile",), fast=True),
        _job("smoke_task_il_repeat", "P0-smoke",
             "same-seed determinism repeat", "run_task_il.py",
             "cifar", seed, TASK_METHODS, extra_args=("--no-profile",),
             fast=True, repeat_of=first),
        _job("smoke_class_entropy", "P0-smoke",
             "class-IL prefix/router smoke", "run_class_il.py", "cifar",
             seed, CLASS_METHODS,
             extra_args=("--inference-criterion", "entropy"), fast=True),
        _job("smoke_class_entropy_cal", "P0-smoke",
             "matched raw/calibrated router smoke", "run_class_il.py",
             "cifar", seed, CLASS_METHODS,
             extra_args=(
                 "--inference-criterion", "entropy",
                 "--inference-calibration", "score-logreg",
                 "--calibration-fraction", "0.10",
                 "--calibration-c", "1.0",
                 "--calibration-split-seed", "1729",
             ), fast=True),
    ]


def _split_by_method(jobs: list[Job], granularity: str) -> list[Job]:
    """Expand one-job-per-seed definitions into one job per method."""
    if granularity == "seed":
        return jobs
    if granularity != "method":
        raise ValueError(f"unknown job granularity: {granularity}")
    # A same-seed repeat compares two whole artifacts, so neither side of a
    # ``repeat_of`` pair is split.
    repeat_targets = {job.repeat_of for job in jobs if job.repeat_of}
    expanded: list[Job] = []
    for job in jobs:
        methods = [m for m in job.methods.split(",") if m]
        if len(methods) == 1 or job.repeat_of or job.job_id in repeat_targets:
            expanded.append(job)
            continue
        for method in methods:
            expanded.append(Job(
                job_id=f"{job.job_id}__{method}", priority=job.priority,
                purpose=job.purpose, runner=job.runner,
                benchmark=job.benchmark, seed=job.seed, methods=method,
                extra_args=job.extra_args,
                config_overrides=dict(job.config_overrides), fast=job.fast,
                repeat_of=None))
    return expanded


def p0_jobs(seeds: list[int], head_mode: str = "shared",
            granularity: str = "seed") -> list[Job]:
    jobs: list[Job] = []
    for seed in seeds:
        for benchmark in ("cifar", "pmnist", "tinyimagenet"):
            jobs.append(_job(
                # The head protocol is part of the identity: a shared-head and
                # a multi-head run are different experiments, so they must not
                # share a job id (and therefore a resume slot / output path).
                f"p0_task_il_{head_mode}", "P0",
                f"corrected Task-IL main rerun ({head_mode} heads)",
                "run_task_il.py", benchmark, seed, TASK_METHODS,
                extra_args=("--head-mode", head_mode)))
        for benchmark in ("cifar", "tinyimagenet"):
            jobs.append(_job(
                "p0_class_entropy", "P0",
                "full-data raw entropy class-IL main rerun",
                "run_class_il.py", benchmark, seed, CLASS_METHODS,
                extra_args=("--inference-criterion", "entropy")))
            jobs.append(_job(
                "p0_class_entropy_cal", "P0",
                "90%-train matched raw entropy versus score-logreg",
                "run_class_il.py", benchmark, seed, CLASS_METHODS,
                extra_args=(
                    "--inference-criterion", "entropy",
                    "--inference-calibration", "score-logreg",
                    "--calibration-fraction", "0.10",
                    "--calibration-c", "1.0",
                    "--calibration-split-seed", "1729",
                )))
    return _split_by_method(jobs, granularity)


def p1_jobs(seeds: list[int], granularity: str = "seed",
            head_mode: str = "shared") -> list[Job]:
    """P1 jobs. Task-IL jobs that train shared-backbone baselines carry the
    head protocol in their group name and command, exactly like P0, so P1
    cannot silently run a different protocol from the main table."""
    jobs: list[Job] = []
    dilt_ablation = (
        ("sparsity_080", {"diltcl_sparsity": 0.80}),
        ("sparsity_095", {"diltcl_sparsity": 0.95}),
        ("partitions_1", {"diltcl_partitions": 1}),
        ("partitions_4", {"diltcl_partitions": 4}),
        ("rounds_3", {"pruning_rounds": 3}),
        ("rounds_11", {"pruning_rounds": 11}),
        ("expert_ft_0", {"diltcl_expert_finetune_epochs": 0}),
        ("expert_ft_10", {"diltcl_expert_finetune_epochs": 10}),
    )
    for seed in seeds:
        jobs.append(_job(
            f"p1_task_il_external_{head_mode}", "P1",
            f"5-datasets external validity ({head_mode} heads)",
            "run_task_il.py", "fivedatasets", seed, TASK_METHODS,
            extra_args=("--head-mode", head_mode)))
        for benchmark in ("pmnist", "fivedatasets"):
            jobs.append(_job(
                "p1_class_entropy", "P1",
                "domain/external raw class-IL evaluation",
                "run_class_il.py", benchmark, seed, CLASS_METHODS,
                extra_args=("--inference-criterion", "entropy")))
            jobs.append(_job(
                "p1_class_entropy_cal", "P1",
                "domain/external matched routing calibration",
                "run_class_il.py", benchmark, seed, CLASS_METHODS,
                extra_args=(
                    "--inference-criterion", "entropy",
                    "--inference-calibration", "score-logreg",
                    "--calibration-fraction", "0.10",
                    "--calibration-c", "1.0",
                    "--calibration-split-seed", "1729",
                )))
        for benchmark in ("cifar", "tinyimagenet"):
            # Tied-slope router (score-bias) for the isolation methods only:
            # the full score-logreg router lowered TA on the first P0 run, so
            # the paired comparison is repeated with the lower-dimensional
            # T+1-parameter tied-slope likelihood calibrator. Because it
            # standardizes score columns separately, it does not generally
            # contain the raw unstandardized argmin rule.
            jobs.append(_job(
                "p1_class_entropy_bias", "P1",
                "90%-train matched raw entropy versus score-bias router",
                "run_class_il.py", benchmark, seed, ISOLATION_CLASS_METHODS,
                extra_args=(
                    "--inference-criterion", "entropy",
                    "--inference-calibration", "score-bias",
                    "--calibration-fraction", "0.10",
                    "--calibration-c", "1.0",
                    "--calibration-split-seed", "1729",
                )))
        for benchmark in ("cifar", "tinyimagenet"):
            # Accuracy-fitted per-task bias router: never below raw argmin on
            # its own fitting data; its fit_accuracy vs test TA gap measures
            # transfer. Third arm of the calibration comparison.
            jobs.append(_job(
                "p1_class_entropy_biasacc", "P1",
                "90%-train matched raw entropy versus score-bias-acc router",
                "run_class_il.py", benchmark, seed, ISOLATION_CLASS_METHODS,
                extra_args=(
                    "--inference-criterion", "entropy",
                    "--inference-calibration", "score-bias-acc",
                    "--calibration-fraction", "0.10",
                    "--calibration-split-seed", "1729",
                )))
        for benchmark in ("cifar", "tinyimagenet"):
            jobs.append(_job(
                f"p1_compute_common_{head_mode}", "P1",
                f"approximately compute-matched 120-epoch baselines "
                f"({head_mode} heads)",
                "run_task_il.py", benchmark, seed,
                COMPUTE_MATCHED_COMMON,
                extra_args=("--head-mode", head_mode),
                config={"epochs_ewc": 120, "epochs_lwf": 120,
                        "epochs_replay": 120}))
            jobs.append(_job(
                "p1_compute_packnet", "P1",
                "PackNet 110+10 epoch compute-matched condition",
                "run_task_il.py", benchmark, seed, "packnet",
                config={"epochs_replay": 110,
                        "packnet_retrain_epochs": 10}))
            for label, overrides in dilt_ablation:
                jobs.append(_job(
                    f"p1_dilt_{label}", "P1",
                    f"DiLT-CL OFAT ablation: {label}",
                    "run_task_il.py", benchmark, seed, "diltcl",
                    config=overrides))
    return _split_by_method(jobs, granularity)


def build_jobs(profile: str, seeds: list[int], head_mode: str = "shared",
               granularity: str = "seed") -> list[Job]:
    if profile == "smoke":
        return smoke_jobs(seeds)
    if profile == "p0":
        return p0_jobs(seeds, head_mode, granularity)
    if profile == "p1":
        return p1_jobs(seeds, granularity, head_mode)
    if profile == "all":
        return p0_jobs(seeds, head_mode, granularity) + p1_jobs(
            seeds, granularity, head_mode)
    raise ValueError(profile)


def estimate_cost(job: Job) -> float:
    """Relative wall-clock estimate used to dispatch longest jobs first."""
    methods = [m for m in job.methods.split(",") if m]
    cost = _BENCHMARK_COST.get(job.benchmark, 1.0) * sum(
        _METHOD_COST.get(m, 1.0) for m in methods)
    if "--inference-calibration" in job.extra_args:
        cost *= 1.3
    if job.fast:
        cost *= 0.05
    return cost


# Editor and tool state that must never count as source: Jupyter writes
# ``.ipynb_checkpoints/<name>-checkpoint.py`` next to any file opened in its
# editor, and pytest/ruff drop caches into the tree. Hashing those would turn
# an autosave into a "source change" that reruns a whole suite.
_SOURCE_EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache",
                          ".ruff_cache", ".mypy_cache", ".git", ".hypothesis"}


def is_source_path(path: Path) -> bool:
    """Whether ``path`` (relative to the code root) counts as project source."""
    if path.suffix.lower() not in {".py", ".md", ".txt"}:
        return False
    return not any(part in _SOURCE_EXCLUDED_PARTS or part.startswith(".")
                   for part in path.parts[:-1])


def source_manifest() -> dict[str, Any]:
    records = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or not is_source_path(path.relative_to(ROOT)):
            continue
        records.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    combined = hashlib.sha256()
    execution_digest = hashlib.sha256()
    for record in records:
        combined.update(record["path"].encode("utf-8"))
        combined.update(b"\0")
        combined.update(record["sha256"].encode("ascii"))
        combined.update(b"\n")
        if record["path"].endswith(".py") or record["path"] == "requirements.txt":
            execution_digest.update(record["path"].encode("utf-8"))
            execution_digest.update(b"\0")
            execution_digest.update(record["sha256"].encode("ascii"))
            execution_digest.update(b"\n")
    return {"root": str(ROOT), "files": records,
            "combined_sha256": combined.hexdigest(),
            "execution_sha256": execution_digest.hexdigest()}


def environment_probe() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "torch", "torchvision", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    probe_code = (
        "import json, torch; "
        "print(json.dumps({'ok': True, 'torch': torch.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'cuda_count': torch.cuda.device_count(), "
        "'cuda_version': torch.version.cuda, "
        "'cudnn': torch.backends.cudnn.version(), "
        "'devices': [torch.cuda.get_device_name(i) "
        "for i in range(torch.cuda.device_count())]}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe_code], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    torch_probe: dict[str, Any]
    if proc.returncode == 0:
        torch_probe = json.loads(proc.stdout.strip())
    else:
        torch_probe = {"ok": False, "error": proc.stderr.strip()}

    nvidia_smi = None
    executable = shutil.which("nvidia-smi")
    if executable:
        smi = subprocess.run(
            [executable,
             "--query-gpu=index,name,uuid,driver_version,memory.total",
             "--format=csv,noheader"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
        nvidia_smi = {"returncode": smi.returncode,
                      "stdout": smi.stdout.strip(),
                      "stderr": smi.stderr.strip()}
    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "usable_cpu_count": usable_cpu_count(),
        "packages": packages,
        "torch_probe": torch_probe,
        "nvidia_smi": nvidia_smi,
        "environment_controls": {
            key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def fingerprint_entries(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-top-level-entry hashes derived from a fingerprint's file records.

    Works for fingerprints written by any suite version, because every
    fingerprint stores the per-file records this is computed from.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        top = record["path"].split("/", 1)[0]
        groups.setdefault(top, []).append(record)
    entries = {}
    for top, group in sorted(groups.items()):
        digest = hashlib.sha256()
        total = 0
        for record in sorted(group, key=lambda r: r["path"]):
            digest.update(record["path"].encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(record["bytes"]).encode("ascii"))
            if record.get("sha256"):
                digest.update(b"\0")
                digest.update(record["sha256"].encode("ascii"))
            digest.update(b"\n")
            total += int(record["bytes"])
        entries[top] = {"sha256": digest.hexdigest(), "files": len(group),
                        "bytes": total}
    return entries


def data_files(root: Path) -> list[Path]:
    """Every regular file under ``root``, following symlinked directories.

    ``Path.rglob`` does not descend into a symlinked directory, so a dataset
    linked into the data root (common with shared dataset stores) would be
    missing from the fingerprint. Links are followed like real directories;
    a link back to a directory already on the current path is skipped.
    """
    files: list[Path] = []
    chains = {str(root): frozenset({os.path.realpath(root)})}
    for current, dirs, names in os.walk(root, followlinks=True):
        chain = chains.pop(current, frozenset())
        keep = []
        for name in dirs:
            child = os.path.join(current, name)
            real = os.path.realpath(child)
            if real not in chain:
                chains[child] = chain | {real}
                keep.append(name)
        dirs[:] = keep
        for name in names:
            path = Path(current) / name
            if path.is_file():
                files.append(path)
    return files


def fingerprint_data_root(root: Path, mode: str) -> dict[str, Any]:
    if not root.exists():
        return {"root": str(root), "exists": False, "mode": mode}
    records = []
    aggregate = hashlib.sha256()
    files = data_files(root)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    total_bytes = 0
    for index, path in enumerate(files, 1):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        content_hash = sha256_file(path) if mode == "full" else None
        record = {"path": relative, "bytes": size}
        if content_hash is not None:
            record["sha256"] = content_hash
        records.append(record)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        if content_hash is not None:
            aggregate.update(b"\0")
            aggregate.update(content_hash.encode("ascii"))
        aggregate.update(b"\n")
        if index % 5000 == 0:
            print(f"[dataset fingerprint] {index}/{len(files)} files", flush=True)
    return {
        "root": str(root.resolve()), "exists": True, "mode": mode,
        "file_count": len(records), "total_bytes": total_bytes,
        "tree_sha256": aggregate.hexdigest(), "files": records,
        "entries": fingerprint_entries(records),
    }


def unrecorded_entries(fingerprint: dict[str, Any], jobs: list[Job]) -> list[str]:
    """Data entries some job reads that the fingerprint does not record.

    Such a job's dataset identity would be empty, so a later change to that
    data would go unnoticed on resume and in the verifier.
    """
    present = fingerprint.get("entries") or fingerprint_entries(fingerprint.get("files", []))
    return sorted({f"{job.benchmark}: {name}" for job in jobs
                   for name in preflight.benchmark_data_entries(job.benchmark)
                   if name not in present})


def job_dataset_identity(fingerprint: dict[str, Any] | None, benchmark: str,
                         data_root: Path | None, mode: str) -> dict[str, Any]:
    """The dataset identity a job of ``benchmark`` depends on.

    Besides the whole-tree hash (kept for the record) it names the top-level
    entries the benchmark reads and their hashes, so a job stays valid across
    changes to *other* benchmarks' data (e.g. staging MNIST after CIFAR jobs
    finished) while any change to its own data forces a rerun.
    """
    names = preflight.benchmark_data_entries(benchmark)
    if fingerprint is None:
        return {"root": str(data_root) if data_root else None, "mode": mode,
                "tree_sha256": None, "benchmark": benchmark,
                "entries": dict.fromkeys(names)}
    entries = fingerprint.get("entries")
    if entries is None:
        entries = fingerprint_entries(fingerprint.get("files", []))
    return {"root": fingerprint.get("root"), "mode": fingerprint.get("mode"),
            "tree_sha256": fingerprint.get("tree_sha256"), "benchmark": benchmark,
            "entries": {name: (entries.get(name) or {}).get("sha256")
                        for name in names}}


_ARCHIVED_ENTRIES_CACHE: dict[tuple[str, str], dict[str, Any] | None] = {}


def archived_fingerprint(run_dir: Path, tree_sha256: str) -> dict[str, Any] | None:
    """Load a previously used fingerprint by its tree hash, if archived."""
    if not tree_sha256:
        return None
    for path in (run_dir / "dataset_fingerprints" / f"{tree_sha256}.json",
                 run_dir / "dataset_fingerprint.json"):
        if path.is_file():
            try:
                fingerprint = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if fingerprint.get("tree_sha256") == tree_sha256:
                return fingerprint
    return None


def archived_entries(run_dir: Path, tree_sha256: str) -> dict[str, Any] | None:
    """Entry hashes of an archived fingerprint, cached per process.

    A full-mode fingerprint of the P0 data root lists ~120k files (~20 MB of
    JSON); resuming a run validates hundreds of finished jobs against it, so
    it is parsed and reduced to entry hashes once rather than per job.
    """
    key = (str(Path(run_dir).resolve()), str(tree_sha256))
    if key not in _ARCHIVED_ENTRIES_CACHE:
        fingerprint = archived_fingerprint(run_dir, tree_sha256)
        if fingerprint is None:
            _ARCHIVED_ENTRIES_CACHE[key] = None
        else:
            entries = fingerprint.get("entries")
            if entries is None:
                entries = fingerprint_entries(fingerprint.get("files", []))
            _ARCHIVED_ENTRIES_CACHE[key] = {
                "root": fingerprint.get("root"), "mode": fingerprint.get("mode"),
                "tree_sha256": fingerprint.get("tree_sha256"), "entries": entries}
    return _ARCHIVED_ENTRIES_CACHE[key]


def archive_fingerprint(run_dir: Path, fingerprint: dict[str, Any]) -> None:
    """Keep every fingerprint ever used so old statuses stay interpretable."""
    sha = fingerprint.get("tree_sha256")
    if not sha:
        return
    path = run_dir / "dataset_fingerprints" / f"{sha}.json"
    if not path.is_file():
        atomic_json(path, fingerprint)


def dataset_identity_matches(status_identity: dict[str, Any] | None,
                             expected: dict[str, Any],
                             run_dir: Path | None = None) -> bool:
    """Whether a finished job's recorded dataset identity still holds.

    New-format identities compare the benchmark's entry hashes. Old-format
    identities (whole-tree hash only) are accepted when the tree hash is
    unchanged, or when the fingerprint they refer to is archived in the run
    directory and its entry hashes for this benchmark equal the current ones.
    """
    if not isinstance(status_identity, dict):
        return False
    if (status_identity.get("root") != expected.get("root")
            or status_identity.get("mode") != expected.get("mode")):
        return False
    if "entries" in status_identity:
        return (status_identity.get("benchmark") == expected.get("benchmark")
                and status_identity.get("entries") == expected.get("entries"))
    if status_identity.get("tree_sha256") == expected.get("tree_sha256"):
        return True
    if run_dir is None or expected.get("mode") == "none":
        return False
    old = archived_entries(run_dir, status_identity.get("tree_sha256"))
    if old is None:
        return False
    derived = job_dataset_identity(old, expected["benchmark"], None,
                                   expected["mode"])
    return derived["entries"] == expected["entries"]


# Fields of run-level records that change on every restart without meaning
# anything (timestamps, nvidia-smi's live memory/utilisation dump).
_VOLATILE_RECORD_FIELDS = {"captured_at", "nvidia_smi"}


def archive_if_changed(current_path: Path, new_payload: dict[str, Any],
                       archive_dir: Path, key: str | None) -> Path | None:
    """Keep the previous version of a run-level record when it is replaced.

    ``source_manifest.json`` and ``environment.json`` are overwritten on every
    (re)start; without this the manifest that produced acknowledged earlier
    jobs would be lost. Files are named by ``key`` (e.g. the execution hash)
    or by a content hash when no key is given. Returns the archive path.
    """
    if not current_path.is_file():
        return None
    try:
        previous = json.loads(current_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    def canonical(payload):
        stable = {k: v for k, v in payload.items()
                  if k not in _VOLATILE_RECORD_FIELDS}
        return json.dumps(stable, sort_keys=True, default=str).encode("utf-8")

    if canonical(previous) == canonical(new_payload):
        return None
    if key and previous.get(key):
        name = str(previous[key])
    else:
        name = hashlib.sha256(canonical(previous)).hexdigest()
    path = archive_dir / f"{name}.json"
    if not path.is_file():
        atomic_json(path, previous)
    return path


def plan_identity(plan: dict[str, Any], include_source: bool = True) -> dict[str, Any]:
    """The part of a plan that must not change within a run directory."""
    identity = {
        "profile": plan.get("profile"), "seeds": plan.get("seeds"),
        "data_root": plan.get("data_root"),
        "allow_download": plan.get("downloads_allowed"),
        "save_checkpoints": plan.get("checkpoints_persisted"),
        "dataset_hash_mode": plan.get("dataset_hash_mode"),
        "execution_options": plan.get("execution_options"),
        "job_granularity": plan.get("job_granularity"),
        "jobs": [record["job"] for record in plan.get("jobs", [])],
    }
    if include_source:
        identity["execution_source_sha256"] = plan.get("execution_source_sha256")
    # JSON round-trip so tuples from in-memory Job dataclasses compare equal
    # to the lists read back from plan.json.
    return json.loads(json.dumps(identity, sort_keys=True))


def source_chain(plan: dict[str, Any]) -> list[str]:
    """Source hashes in chronological order: oldest acknowledged first."""
    chain: list[str] = []
    for record in plan.get("source_change_acknowledgements", []):
        for sha in (record.get("from"), record.get("to")):
            if sha and sha not in chain:
                chain.append(sha)
    current = plan.get("execution_source_sha256")
    if current and current not in chain:
        chain.append(current)
    return chain


def result_invalidated(plan: dict[str, Any], status_source: str | None,
                       methods) -> str | None:
    """Why a finished job's result is invalid under the acknowledged chain.

    An acknowledgement ``from -> to`` may carry ``invalidated_methods``: the
    fix it records changes those methods' numerics, so their results produced
    under ``from`` or any earlier source must be rerun even though the rest
    of the run is kept. Returns the reason string, or ``None`` if the result
    stands.
    """
    chain = source_chain(plan)
    if status_source not in chain:
        return None
    position = chain.index(status_source)
    wanted = {m for m in (methods or []) if m}
    for record in plan.get("source_change_acknowledgements", []):
        invalidated = set(record.get("invalidated_methods") or [])
        hit = wanted & invalidated
        if not hit or record.get("from") not in chain:
            continue
        if position <= chain.index(record["from"]):
            return (f"{sorted(hit)} results produced under source "
                    f"{status_source[:12]} were invalidated by the acknowledged "
                    f"change {record['from'][:12]}->{record['to'][:12]}: "
                    f"{record.get('reason')}")
    return None


def accepted_source_hashes(plan: dict[str, Any]) -> set[str]:
    """Source hashes finished jobs may carry: the current one plus every
    hash in the acknowledged change chain."""
    accepted = {plan.get("execution_source_sha256")}
    for record in plan.get("source_change_acknowledgements", []):
        accepted.add(record.get("from"))
        accepted.add(record.get("to"))
    accepted.discard(None)
    return accepted


def config_path_for(job: Job, run_dir: Path, write: bool = True) -> Path | None:
    if not job.config_overrides:
        return None
    path = run_dir / "configs" / f"{job.job_id}.json"
    if write:
        atomic_json(path, job.config_overrides)
    return path


def command_for(job: Job, run_dir: Path, data_root: Path | None,
                allow_download: bool,
                save_checkpoints: bool = True,
                execution_args: tuple[str, ...] = (),
                write_config: bool = True) -> tuple[list[str], Path]:
    output_prefix = run_dir / "artifacts" / job.job_id / "result"
    command = [
        sys.executable, str(ROOT / job.runner),
        "--benchmark", job.benchmark,
        "--seed-list", str(job.seed),
        "--methods", job.methods,
        "--out", str(output_prefix),
    ]
    if job.fast:
        command.append("--fast")
    command.extend(job.extra_args)
    command.extend(execution_args)
    config_path = config_path_for(job, run_dir, write_config)
    if config_path is not None:
        command.extend(("--config-json", str(config_path)))
    if data_root is not None:
        command.extend(("--data-root", str(data_root)))
    if not allow_download:
        command.append("--no-download")
    if save_checkpoints:
        command.extend(("--checkpoint-dir",
                        str(output_prefix.parent / "checkpoints")))
        if job.runner == "run_class_il.py":
            # Per-example routing outcomes of every isolation model (v3.0.5):
            # small, hashed with the job's outputs, and sufficient to
            # recompute every class-IL statistic without a GPU.
            command.extend(("--predictions-dir",
                            str(output_prefix.parent / "predictions")))
    return command, output_prefix


def status_is_valid(status_path: Path, expected_execution_sha256: str,
                    expected_dataset_identity: dict[str, Any],
                    accepted_sources: set[str] | None = None,
                    run_dir: Path | None = None,
                    plan: dict[str, Any] | None = None) -> bool:
    """Whether a finished job can be skipped on resume.

    Requires a succeeded state, an executable source that is the current one
    or an acknowledged earlier one, a dataset identity that still holds for
    the job's benchmark (see :func:`dataset_identity_matches`), and every
    recorded output present with its recorded size and SHA-256.
    """
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if status.get("state") != "succeeded":
        return False
    sources = set(accepted_sources or ()) | {expected_execution_sha256}
    if status.get("execution_source_sha256") not in sources:
        return False
    if plan is not None and status.get("execution_source_sha256") != expected_execution_sha256:
        methods = str((status.get("job") or {}).get("methods", "")).split(",")
        reason = result_invalidated(plan, status.get("execution_source_sha256"), methods)
        if reason:
            print(f"[rerun] {status_path.stem}: {reason}", flush=True)
            return False
    if run_dir is None:
        run_dir = status_path.parent.parent
    if not dataset_identity_matches(status.get("dataset_identity"),
                                    expected_dataset_identity, run_dir):
        return False
    for record in status.get("outputs", []):
        relative = record.get("path_relative_to_run_dir")
        path = ((status_path.parent.parent / relative).resolve()
                if isinstance(relative, str) else Path(record["path"]))
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            return False
        if sha256_file(path) != record["sha256"]:
            return False
    return bool(status.get("outputs"))


class _ProcessRegistry:
    """Tracks running children so an interrupt can terminate all of them."""

    def __init__(self):
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()

    def add(self, process):
        with self._lock:
            self._processes.add(process)

    def discard(self, process):
        with self._lock:
            self._processes.discard(process)

    def terminate_all(self):
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


def visible_gpu_selector() -> list[str] | None:
    """``nvidia-smi`` arguments selecting the first GPU the jobs will use.

    ``nvidia-smi`` lists every GPU of the machine; ``CUDA_VISIBLE_DEVICES``
    (an index or UUID list) names the ones the children see. ``None`` when it
    hides every GPU. With several GPUs, set ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so
    that CUDA and ``nvidia-smi`` number them alike.
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return []
    first = value.split(",")[0].strip()
    if not first or first.startswith("-"):
        return None
    return ["-i", first]


def gpu_memory_mib() -> tuple[int, int] | None:
    """``(used_mib, total_mib)`` of the first visible GPU, or ``None``."""
    selector = visible_gpu_selector()
    if shutil.which("nvidia-smi") is None or selector is None:
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi", *selector, "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    first = proc.stdout.strip().splitlines()[0]
    try:
        used, total = (int(float(v)) for v in first.split(","))
    except ValueError:
        return None
    return used, total


def usable_cpu_count() -> int:
    """CPUs this process may actually use.

    ``os.cpu_count()`` reports the host inside containers (the first P0 run
    saw 208 on a 25-vCPU instance); the scheduler affinity mask reflects the
    real allotment where the platform exposes it.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def resolve_max_parallel(spec: str, threads_per_job: int,
                         job_memory_mib: int) -> tuple[int, dict[str, Any]]:
    """Resolve ``--max-parallel`` (an integer or ``auto``) and explain it."""
    cpu_count = usable_cpu_count()
    memory = gpu_memory_mib()
    detail: dict[str, Any] = {
        "cpu_count": cpu_count, "threads_per_job": threads_per_job,
        "job_memory_mib": job_memory_mib,
        "gpu_memory_used_mib": memory[0] if memory else None,
        "gpu_memory_total_mib": memory[1] if memory else None,
    }
    if spec.strip().lower() != "auto":
        value = int(spec)
        if value <= 0:
            raise ValueError("--max-parallel must be positive or 'auto'")
        detail["rule"] = "explicit"
        return value, detail
    by_cpu = max(1, cpu_count // max(1, threads_per_job))
    if memory is not None:
        free = max(0, memory[1] - memory[0])
        by_memory = max(1, free // max(1, job_memory_mib))
    else:
        by_memory = 1 if shutil.which("nvidia-smi") else by_cpu
    value = max(1, min(by_cpu, by_memory, MAX_AUTO_PARALLEL))
    detail.update({"rule": "auto=min(cpu//threads, free_vram//job_mem, "
                           f"{MAX_AUTO_PARALLEL})",
                   "by_cpu": by_cpu, "by_memory": by_memory})
    return value, detail


def wait_for_gpu_headroom(job_memory_mib: int, running: int,
                          poll_seconds: float = 5.0,
                          max_wait_seconds: float = 600.0) -> None:
    """Delay a launch while free GPU memory is below the per-job budget.

    Only waits while other jobs are running (their memory will be released);
    with nothing running the launch proceeds regardless so an unrelated
    process on the device cannot stall the suite forever.
    """
    waited = 0.0
    while running > 0 and waited < max_wait_seconds:
        memory = gpu_memory_mib()
        if memory is None or memory[1] - memory[0] >= job_memory_mib:
            return
        time.sleep(poll_seconds)
        waited += poll_seconds


def run_job(job: Job, command: list[str], output_prefix: Path,
            run_dir: Path, force: bool,
            execution_source_sha256: str,
            dataset_identity: dict[str, Any], *,
            child_env_extra: dict[str, str] | None = None,
            stream_logs: bool = False,
            registry: _ProcessRegistry | None = None,
            accepted_sources: set[str] | None = None,
            plan: dict[str, Any] | None = None) -> dict[str, Any]:
    status_path = run_dir / "statuses" / f"{job.job_id}.json"
    if not force and status_is_valid(
            status_path, execution_source_sha256, dataset_identity,
            accepted_sources, run_dir, plan):
        print(f"[skip] {job.job_id}: validated successful outputs", flush=True)
        return {"job_id": job.job_id, "state": "skipped"}

    attempt = 1
    existing = sorted((run_dir / "logs").glob(f"{job.job_id}.attempt*.log"))
    if existing:
        attempt = len(existing) + 1
    log_path = run_dir / "logs" / f"{job.job_id}.attempt{attempt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()
    child_env = os.environ.copy()
    child_env["PYTHONHASHSEED"] = str(job.seed)
    child_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for key, value in (child_env_extra or {}).items():
        child_env[key] = value
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_at={started_at}\n")
        log.write("command=" + json.dumps(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=child_env)
        if registry is not None:
            registry.add(process)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if stream_logs:
                    sys.stdout.write(line)
                log.write(line)
            returncode = process.wait()
        finally:
            if registry is not None:
                registry.discard(process)
    elapsed = time.perf_counter() - started
    output_paths = [output_prefix.with_suffix(".json"),
                    output_prefix.with_suffix(".npz")]
    checkpoint_root = output_prefix.parent / "checkpoints"
    if checkpoint_root.is_dir():
        output_paths.extend(sorted(checkpoint_root.rglob("*.pt")))
    predictions_root = output_prefix.parent / "predictions"
    if predictions_root.is_dir():
        # ``*.tmp.<pid>.npz`` is a write in progress (or left by a killed
        # attempt), never an output of this job.
        output_paths.extend(sorted(p for p in predictions_root.rglob("*.npz")
                                   if ".tmp." not in p.name))
    outputs = []
    for path in output_paths:
        if path.is_file():
            # Relative to the run directory as laid out (not resolved): an
            # ``artifacts/`` folder linked to another volume stays inside it.
            outputs.append({"path": str(path.resolve()),
                            "path_relative_to_run_dir": os.path.relpath(path, run_dir),
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)})
    main_outputs_ok = all(path.is_file() for path in output_paths[:2])
    state = ("succeeded" if returncode == 0 and main_outputs_ok
             else "failed")
    status = {
        "job": asdict(job), "state": state, "attempt": attempt,
        "execution_source_sha256": execution_source_sha256,
        "dataset_identity": dataset_identity,
        "command": command, "started_at": started_at,
        "ended_at": utc_now(), "wall_seconds": elapsed,
        "returncode": returncode, "log": str(log_path.resolve()),
        "child_environment_controls": {
            "PYTHONHASHSEED": child_env["PYTHONHASHSEED"],
            "CUBLAS_WORKSPACE_CONFIG": child_env["CUBLAS_WORKSPACE_CONFIG"],
            "CUDA_VISIBLE_DEVICES": child_env.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": child_env.get("OMP_NUM_THREADS"),
            "DILTCL_NUM_THREADS": child_env.get("DILTCL_NUM_THREADS"),
            "DILTCL_CACHE_DIR": child_env.get("DILTCL_CACHE_DIR"),
        },
        "outputs": outputs,
    }
    atomic_json(status_path, status)
    return {"job_id": job.job_id, "state": state,
            "returncode": returncode, "wall_seconds": elapsed}


def execute_jobs(jobs: list[Job], *, run_dir: Path, data_root: Path | None,
                 allow_download: bool, save_checkpoints: bool, force: bool,
                 execution_source_sha256: str,
                 dataset_identity: dict[str, Any] | None = None,
                 execution_args: tuple[str, ...], max_parallel: int,
                 threads_per_job: int, job_memory_mib: int,
                 stop_on_error: bool, stream_logs: bool,
                 launch_stagger_seconds: float = 2.0,
                 fingerprint: dict[str, Any] | None = None,
                 dataset_hash_mode: str = "none",
                 accepted_sources: set[str] | None = None,
                 plan: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run ``jobs`` with up to ``max_parallel`` concurrent child processes.

    Jobs are dispatched longest-first so the tail of the run is not dominated
    by one late DiLT-CL job; every child is capped to ``threads_per_job`` CPU
    threads; a launch waits while the GPU has less than ``job_memory_mib``
    free and other jobs are still running, and consecutive launches are
    staggered by ``launch_stagger_seconds`` so a freshly started child has
    allocated its CUDA context and dataset before the next one is admitted.
    Ctrl-C terminates every child and leaves finished jobs resumable.
    """
    ordered = sorted(jobs, key=estimate_cost, reverse=True)
    child_env_extra = {
        "OMP_NUM_THREADS": str(threads_per_job),
        "MKL_NUM_THREADS": str(threads_per_job),
        "DILTCL_NUM_THREADS": str(threads_per_job),
    }
    registry = _ProcessRegistry()
    outcomes: list[dict[str, Any]] = []
    total = len(ordered)
    started_total = time.perf_counter()
    stop = threading.Event()
    running_count = 0

    def _launch(job: Job) -> dict[str, Any]:
        command, output_prefix = command_for(
            job, run_dir, data_root, allow_download, save_checkpoints,
            execution_args)
        # ``dataset_identity`` (legacy single dict) is only used when no
        # fingerprint is given, e.g. by tests; real runs derive it per job.
        identity = (dataset_identity if dataset_identity is not None
                    and fingerprint is None and dataset_hash_mode == "none"
                    else job_dataset_identity(fingerprint, job.benchmark,
                                              data_root, dataset_hash_mode))
        return run_job(job, command, output_prefix, run_dir, force,
                       execution_source_sha256, identity,
                       child_env_extra=child_env_extra,
                       stream_logs=stream_logs and max_parallel == 1,
                       registry=registry, accepted_sources=accepted_sources,
                       plan=plan)

    pending = list(ordered)
    futures = {}
    last_launch = None
    try:
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            while pending or futures:
                while pending and len(futures) < max_parallel and not stop.is_set():
                    job = pending.pop(0)
                    if running_count > 0:
                        if last_launch is not None and launch_stagger_seconds > 0:
                            elapsed = time.perf_counter() - last_launch
                            if elapsed < launch_stagger_seconds:
                                time.sleep(launch_stagger_seconds - elapsed)
                        wait_for_gpu_headroom(job_memory_mib, running_count)
                    last_launch = time.perf_counter()
                    future = pool.submit(_launch, job)
                    futures[future] = job
                    running_count += 1
                    print(f"[start {len(outcomes) + running_count}/{total}] "
                          f"{job.job_id}", flush=True)
                if not futures:
                    break
                done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    job = futures.pop(future)
                    running_count -= 1
                    try:
                        outcome = future.result()
                    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                        outcome = {"job_id": job.job_id, "state": "failed",
                                   "returncode": None, "error": repr(exc)}
                    outcomes.append(outcome)
                    minutes = outcome.get("wall_seconds", 0.0) / 60.0
                    print(f"[{outcome['state']} {len(outcomes)}/{total}] "
                          f"{job.job_id} ({minutes:.1f} min, "
                          f"{running_count} running, "
                          f"{(time.perf_counter() - started_total) / 60:.1f} "
                          "min elapsed)", flush=True)
                    if outcome["state"] == "failed" and stop_on_error:
                        stop.set()
                        pending.clear()
    except KeyboardInterrupt:
        registry.terminate_all()
        raise
    return outcomes


def plan_payload(jobs: list[Job], run_dir: Path, data_root: Path | None,
                 allow_download: bool, save_checkpoints: bool,
                 profile: str, seeds: list[int],
                 execution_source_sha256: str,
                 dataset_hash_mode: str,
                 execution_options: dict[str, Any] | None = None,
                 execution_args: tuple[str, ...] = (),
                 granularity: str = "seed", write_configs: bool = True) -> dict[str, Any]:
    records = []
    for job in jobs:
        command, output = command_for(
            job, run_dir, data_root, allow_download, save_checkpoints,
            execution_args, write_configs)
        records.append({
            "job": asdict(job), "command": command,
            "output_prefix": str(output),
            "output_prefix_relative": str(output.relative_to(run_dir)),
        })
    stable = {
        "profile": profile, "seeds": seeds,
        "data_root": str(data_root.resolve()) if data_root else None,
        "allow_download": allow_download,
        "save_checkpoints": save_checkpoints,
        "execution_source_sha256": execution_source_sha256,
        "dataset_hash_mode": dataset_hash_mode,
        # Execution options change provenance (and possibly numerics), so a
        # run directory must not mix them; parallelism does not, so it is
        # recorded below but excluded from the identity.
        "execution_options": execution_options,
        "job_granularity": granularity,
        "jobs": [asdict(job) for job in jobs],
    }
    fingerprint = hashlib.sha256(json.dumps(
        stable, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "created_at": utc_now(), "profile": profile, "seeds": seeds,
        "execution_source_sha256": execution_source_sha256,
        "dataset_hash_mode": dataset_hash_mode,
        "plan_sha256": fingerprint, "run_dir": str(run_dir.resolve()),
        "data_root": str(data_root.resolve()) if data_root else None,
        "downloads_allowed": allow_download,
        "checkpoints_persisted": save_checkpoints,
        "execution_options": execution_options,
        "job_granularity": granularity,
        "jobs": records,
    }


_HELD_RUN_LOCKS: list[Any] = []


def lock_run_dir(run_dir: Path) -> bool:
    """Hold an exclusive lock on ``run_dir`` until :func:`main` returns.

    Two suites in one run directory would launch the same pending jobs twice,
    write the same artifacts and share the GPU (which can change cuDNN's
    algorithm choice); the second one stops here, before writing anything.
    The kernel releases the lock when the process exits, however it exits.
    Returns ``False`` (and the run proceeds unlocked) where the filesystem
    does not support locks.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX platform
        return False
    handle = open(run_dir / ".run_suite.lock", "a", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(
            f"another run_suite.py is already running in {run_dir}; wait for it "
            "to finish (check with: pgrep -af run_suite.py) before starting again")
    except OSError as exc:
        handle.close()
        print(f"[run lock] cannot lock {run_dir} ({exc}); continuing without the "
              "lock. Do not start a second suite in this run directory.",
              file=sys.stderr)
        return False
    _HELD_RUN_LOCKS.append(handle)
    return True


def release_run_dir_locks(keep: int = 0) -> None:
    """Release the run-directory locks taken after the first ``keep`` ones."""
    while len(_HELD_RUN_LOCKS) > keep:
        _HELD_RUN_LOCKS.pop().close()


def write_pip_freeze(run_dir: Path) -> None:
    """Record ``pip freeze``; a different earlier record is kept under
    ``pip_freezes/<sha256>.txt``, like the environment and source records."""
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    path = run_dir / "pip_freeze.txt"
    if path.is_file():
        previous = path.read_text(encoding="utf-8")
        if previous != proc.stdout:
            archive = run_dir / "pip_freezes"
            archive.mkdir(exist_ok=True)
            digest = hashlib.sha256(previous.encode("utf-8")).hexdigest()
            (archive / f"{digest}.txt").write_text(previous, encoding="utf-8")
    path.write_text(proc.stdout, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=("smoke", "p0", "p1", "all"),
                        default="smoke")
    parser.add_argument("--seeds", default=None,
                        help="Comma/range seed IDs. Defaults: smoke=0, p0=0-4, p1=0-4, all=0-9.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Persistent run directory. Reuse it to resume.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow torchvision downloads. Final runs should pre-stage data.")
    parser.add_argument("--no-save-checkpoints", action="store_true",
                        help="Save disk space at the cost of failing the checkpoint evidence gate.")
    parser.add_argument("--dataset-hash", choices=("none", "metadata", "full"),
                        default=None,
                        help="Defaults to metadata for smoke and full for P0/P1.")
    parser.add_argument(
        "--reuse-dataset-fingerprint", action="store_true",
        help="Reuse an existing fingerprint instead of rescanning. Use only "
             "when the data root is immutable and externally controlled.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Rerun successful jobs instead of resuming them.")
    parser.add_argument("--force-plan", action="store_true",
                        help="Allow replacing a different plan in the run directory.")
    parser.add_argument(
        "--accept-source-change", metavar="REASON", default=None,
        help="Resume a run directory whose jobs were produced by an earlier "
             "version of the code. The old->new source hashes and REASON are "
             "recorded in plan.json, finished jobs are kept, and the verifier "
             "reports the acknowledged change as a warning. Use only for "
             "changes that cannot alter finished results (e.g. a launcher or "
             "validation fix); anything touching training must rerun.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--head-mode", choices=("shared", "multi"),
                        default="shared",
                        help="Task-IL head protocol for the shared-backbone "
                             "baselines in P0. 'multi' is the standard, "
                             "literature-comparable protocol; 'shared' "
                             "reproduces the legacy single reused head. The "
                             "job ids differ, so both can live in one run "
                             "directory without colliding.")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the dependency/dataset gate. Not advised: "
                             "the gate exists to stop late failures that waste "
                             "completed training.")
    parser.add_argument("--pin-sklearn", action="store_true",
                        help="Require scikit-learn for the calibrated router "
                             "instead of allowing the built-in SciPy solver.")
    parser.add_argument("--skip-verify", action="store_true")

    parser.add_argument(
        "--invalidate-methods", default=None, metavar="METHODS",
        help="With --accept-source-change: comma-separated methods whose "
             "finished results are made invalid by the change (their "
             "numerics changed) and must be rerun, e.g. supsup,wsn. Recorded "
             "in the acknowledgement; the verifier refuses old results of "
             "these methods.")
    parser.add_argument(
        "--only-groups", default=None, metavar="GROUPS",
        help="Comma-separated job-group names to keep (e.g. "
             "p1_class_entropy_bias,p1_task_il_external_multi); other jobs of "
             "the profile are dropped from the plan. Use a run directory of "
             "its own for each selection.")

    scheduling = parser.add_argument_group("scheduling")
    scheduling.add_argument(
        "--job-granularity", choices=("method", "seed"), default="method",
        help="'method' (default): one job per (benchmark, seed, method); "
             "'seed': one job per (benchmark, seed) running all methods, the "
             "v2 layout.")
    scheduling.add_argument(
        "--max-parallel", default="auto",
        help="Concurrent jobs on the GPU. 'auto' = min(cpu//threads-per-job, "
             f"free VRAM // --job-memory-mib, {MAX_AUTO_PARALLEL}). Use 1 for "
             "the sequential v2 behaviour.")
    scheduling.add_argument(
        "--threads-per-job", type=int, default=DEFAULT_THREADS_PER_JOB,
        help="CPU threads per child (OMP/MKL/torch). Default 3.")
    scheduling.add_argument(
        "--job-memory-mib", type=int, default=DEFAULT_JOB_MEMORY_MIB,
        help="Per-job GPU memory budget used by 'auto' and by the launch "
             "guard. Calibrate with tools/gpu_capacity_probe.py.")
    scheduling.add_argument(
        "--launch-stagger", type=float, default=2.0,
        help="Seconds between consecutive launches while other jobs run "
             "(lets a new child allocate its CUDA context first). Default 2.")
    scheduling.add_argument(
        "--stream-logs", action="store_true",
        help="Echo child output to the console (only with --max-parallel 1; "
             "logs are always written per job).")
    execution.add_execution_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    held = len(_HELD_RUN_LOCKS)
    try:
        return _main(argv)
    finally:
        release_run_dir_locks(keep=held)


def _main(argv: list[str] | None) -> int:
    args = parse_args(argv)
    if args.invalidate_methods and not args.accept_source_change:
        raise SystemExit("--invalidate-methods requires --accept-source-change")
    dry = bool(args.dry_run)          # a dry run reads the run directory, never writes it
    default_seeds = {"smoke": "0", "p0": "0-4", "p1": "0-4", "all": "0-9"}
    try:
        seeds = parse_seed_spec(args.seeds or default_seeds[args.profile])
    except ValueError as exc:
        raise SystemExit(f"invalid --seeds: {exc}") from exc
    try:
        options = execution.execution_from_args(args)
    except ValueError as exc:
        raise SystemExit(f"invalid execution options: {exc}") from exc
    if options.threads is None:
        # Each child is capped explicitly so parallel jobs share the CPUs.
        options = execution.ExecutionOptions(
            **{**asdict(options), "threads": int(args.threads_per_job)})
    execution_args = tuple(execution.execution_command_args(options))
    try:
        max_parallel, parallel_detail = resolve_max_parallel(
            str(args.max_parallel), args.threads_per_job, args.job_memory_mib)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_dir = args.run_dir
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT.parent / "experiment_runs" / f"v3_{args.profile}_{stamp}"
    run_dir = run_dir.resolve()
    data_root = args.data_root.resolve() if args.data_root else None
    try:
        run_inside_data = (data_root is not None and os.path.commonpath(
            (str(run_dir), str(data_root))) == str(data_root))
    except ValueError:
        run_inside_data = False
    if run_inside_data:
        raise SystemExit(
            "--run-dir must not be inside --data-root; the run would mutate "
            "its own dataset fingerprint")
    cache_dir = os.environ.get("DILTCL_CACHE_DIR")
    if cache_dir and data_root is not None:
        try:
            cache_inside_data = os.path.commonpath(
                (str(Path(cache_dir).resolve()), str(data_root))) == str(data_root)
        except ValueError:
            cache_inside_data = False
        if cache_inside_data:
            raise SystemExit(
                "DILTCL_CACHE_DIR must not be inside --data-root; the "
                "materialised tensor cache would change the dataset fingerprint")
    dataset_hash = (args.dataset_hash or
                    ("metadata" if args.profile == "smoke" else "full"))
    jobs = build_jobs(args.profile, seeds, args.head_mode, args.job_granularity)
    if args.only_groups:
        wanted = {name.strip() for name in args.only_groups.split(",") if name.strip()}
        available = {job.job_id.split("__", 1)[0] for job in jobs}
        unknown = sorted(wanted - available)
        if unknown:
            raise SystemExit(
                f"--only-groups names unknown groups {unknown}; this profile "
                f"has {sorted(available)}")
        jobs = [job for job in jobs if job.job_id.split("__", 1)[0] in wanted]
    if any(job.benchmark == "tinyimagenet" for job in jobs) and data_root is None:
        raise SystemExit("--data-root is required because this profile includes TinyImageNet")

    if not dry:
        run_dir.mkdir(parents=True, exist_ok=True)
        lock_run_dir(run_dir)          # one suite per run directory
    save_checkpoints = not args.no_save_checkpoints
    source = source_manifest()
    plan = plan_payload(jobs, run_dir, data_root, args.allow_download,
                        save_checkpoints, args.profile, seeds,
                        source["execution_sha256"], dataset_hash,
                        execution_options=asdict(options),
                        execution_args=execution_args,
                        granularity=args.job_granularity, write_configs=not dry)
    plan["scheduling"] = {"max_parallel": max_parallel,
                          "threads_per_job": int(args.threads_per_job),
                          "job_memory_mib": int(args.job_memory_mib),
                          **parallel_detail}
    plan_path = run_dir / "plan.json"
    plan["source_change_acknowledgements"] = []
    would = "[dry run] would record: " if dry else ""
    invalidated = sorted({m.strip() for m in (args.invalidate_methods or "").split(",")
                          if m.strip()})
    if plan_path.exists():
        old_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["source_change_acknowledgements"] = list(
            old_plan.get("source_change_acknowledgements", []))
        old_source = old_plan.get("execution_source_sha256")
        same_apart_from_source = (
            plan_identity(old_plan, include_source=False)
            == plan_identity(plan, include_source=False))
        # Compare the identity recomputed from the stored fields rather than
        # the stored hash, so the decision does not depend on a field that
        # could be edited by hand.
        if plan_identity(old_plan) != plan_identity(plan):
            if (same_apart_from_source and old_source
                    and old_source != plan["execution_source_sha256"]):
                if not args.accept_source_change and not args.force_plan:
                    raise SystemExit(
                        "run directory was produced by a different version of "
                        "the code (source hash changed). Finished jobs would "
                        "all be rerun. If the change cannot alter finished "
                        "results, resume with --accept-source-change "
                        "\"<reason>\" to keep them; otherwise use another "
                        "--run-dir or --force-plan.")
                if args.accept_source_change:
                    plan["source_change_acknowledgements"].append({
                        "from": old_source,
                        "to": plan["execution_source_sha256"],
                        "reason": args.accept_source_change,
                        "invalidated_methods": invalidated,
                        "acknowledged_at": utc_now()})
                    print(f"{would}[source change] acknowledged {old_source[:12]} -> "
                          f"{plan['execution_source_sha256'][:12]}: "
                          f"{args.accept_source_change}")
                    if invalidated:
                        print(f"{would}[source change] results of {invalidated} produced "
                              f"under {old_source[:12]} are invalidated and will be rerun")
            elif not args.force_plan:
                raise SystemExit(
                    "run directory contains a different plan; choose another "
                    "--run-dir or pass --force-plan")
        elif args.accept_source_change:
            # The change into the current source is already acknowledged (an
            # earlier invocation recorded it). Methods named now must not be
            # dropped silently: add them to that acknowledgement.
            current = plan["execution_source_sha256"]
            latest = next((record for record in reversed(plan["source_change_acknowledgements"])
                           if record.get("to") == current), None)
            if latest is None:
                if invalidated:
                    raise SystemExit(
                        "--invalidate-methods: this run directory has no acknowledged "
                        "source change into the current code, so no finished result "
                        "was produced by a different version; nothing to invalidate")
                print("[source change] nothing to acknowledge: the run directory "
                      "already matches the current code")
            else:
                added = sorted(set(invalidated) - set(latest.get("invalidated_methods") or []))
                if added:
                    latest["invalidated_methods"] = sorted(
                        set(latest.get("invalidated_methods") or []) | set(added))
                    latest.setdefault("amendments", []).append({
                        "invalidated_methods_added": added,
                        "reason": args.accept_source_change, "amended_at": utc_now()})
                    print(f"{would}[source change] the acknowledged change "
                          f"{latest['from'][:12]} -> {current[:12]} now also invalidates "
                          f"{added}; their results produced under {latest['from'][:12]} "
                          "will be rerun")
                else:
                    print(f"[source change] {latest['from'][:12]} -> {current[:12]} is "
                          "already acknowledged"
                          + (f" with {invalidated} invalidated" if invalidated else ""))
    if dry:
        print(f"Run directory: {run_dir} (dry run: nothing is written)")
        print(f"Profile: {args.profile}; jobs: {len(jobs)} "
              f"({args.job_granularity} granularity); seeds: {seeds}")
        for record in plan["jobs"]:
            print(record["job"]["job_id"])
            print("  " + " ".join(record["command"]))
        print("Dry run only; no file was written and no dataset hashing or training "
              "was performed.")
        return 0
    atomic_json(plan_path, plan)
    accepted_sources = accepted_source_hashes(plan)
    archive_if_changed(run_dir / "source_manifest.json", source,
                       run_dir / "source_manifests", "execution_sha256")
    atomic_json(run_dir / "source_manifest.json", source)
    environment = environment_probe()
    archive_if_changed(run_dir / "environment.json", environment,
                       run_dir / "environments", None)
    atomic_json(run_dir / "environment.json", environment)
    write_pip_freeze(run_dir)

    print(f"Run directory: {run_dir}")
    print(f"Profile: {args.profile}; jobs: {len(jobs)} "
          f"({args.job_granularity} granularity); seeds: {seeds}")
    print(f"Execution: backend={options.data_backend} numerics={options.numerics} "
          f"tf32={options.tf32} eval_batch={options.eval_batch_size or 'train'}; "
          f"max_parallel={max_parallel} ({parallel_detail['rule']}), "
          f"threads/job={args.threads_per_job}")
    if args.profile != "smoke" and len(seeds) < 5:
        print(f"[seed policy] WARNING: {len(seeds)} seed(s) requested. "
              "Paired significance testing needs at least five aligned "
              "seeds; the table tool suppresses stars below that "
              "threshold, so this run cannot support inferential claims.",
              file=sys.stderr)
    if data_root is None:
        raise SystemExit("--data-root is required for every real suite run")
    if args.profile != "smoke" and args.allow_download:
        raise SystemExit(
            "--allow-download is smoke-only; P0/P1 require immutable pre-staged data")
    if not environment["torch_probe"].get("ok"):
        raise SystemExit(
            "PyTorch preflight failed. See environment.json; install the "
            "required GPU/CPU PyTorch stack before running experiments.")

    # Fail fast on everything else that is knowable now. The first v2.1 P0 run
    # lost five of seven jobs -- including two that had already finished
    # training -- to a missing package and unstaged datasets, both of which
    # were visible before the first batch.
    if not args.skip_preflight:
        benchmarks_needed = sorted({job.benchmark for job in jobs})
        needs_calibration = any(
            "--inference-calibration" in job.extra_args for job in jobs)
        preflight_report = preflight.check(
            str(data_root), benchmarks_needed, require_training=False,
            require_calibration=needs_calibration,
            pin_sklearn=args.pin_sklearn,
            output_dir=str(run_dir) if save_checkpoints else None,
            expected_method_lists=(
                [[m for m in job.methods.split(",") if m] for job in jobs]
                if save_checkpoints else None))
        print(preflight.format_report(preflight_report, header="preflight"))
        atomic_json(run_dir / "preflight.json", {
            "checked_at": utc_now(), "data_root": str(data_root),
            "benchmarks": benchmarks_needed,
            "calibration_requested": needs_calibration,
            "pin_sklearn": bool(args.pin_sklearn), **preflight_report})
        if not preflight_report["ok"]:
            raise SystemExit(
                "Preflight blocked the run; no job was started and no compute "
                "was spent. Fix the items above, or pass --skip-preflight to "
                "override deliberately.")
    fingerprint = None
    if dataset_hash != "none":
        fingerprint_path = run_dir / "dataset_fingerprint.json"
        previous = None
        if fingerprint_path.exists():
            try:
                previous = json.loads(fingerprint_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = None
        if previous is not None:
            # Keep every fingerprint ever used: finished jobs recorded the
            # tree hash they ran under, and entry-level comparison against
            # the archived file is what lets them survive a data-root change
            # that did not touch their own benchmark.
            archive_fingerprint(run_dir, previous)
        if previous is not None and args.reuse_dataset_fingerprint:
            fingerprint = previous
            print("Reusing existing dataset fingerprint by explicit request.")
        else:
            print(f"Fingerprinting dataset root in {dataset_hash} mode...")
            fingerprint = fingerprint_data_root(data_root, dataset_hash)
            atomic_json(fingerprint_path, fingerprint)
            archive_fingerprint(run_dir, fingerprint)
            if previous is not None and previous.get("tree_sha256") != fingerprint.get("tree_sha256"):
                old_entries = (previous.get("entries")
                               or fingerprint_entries(previous.get("files", [])))
                new_entries = fingerprint["entries"]
                changed = sorted(
                    name for name in set(old_entries) | set(new_entries)
                    if old_entries.get(name) != new_entries.get(name))
                print(f"[dataset fingerprint] data root changed since the last "
                      f"run; entries changed: {changed}. Only jobs whose "
                      f"benchmark reads a changed entry will be rerun.")
        if not args.allow_download:
            unrecorded = unrecorded_entries(fingerprint, jobs)
            if unrecorded:
                raise SystemExit(
                    "the dataset fingerprint of the data root has no record of "
                    f"{unrecorded}; stage these entries under --data-root (symbolic "
                    "links are followed) before running")

    outcomes: list[dict[str, Any]] = []
    try:
        outcomes = execute_jobs(
            jobs, run_dir=run_dir, data_root=data_root,
            allow_download=args.allow_download,
            save_checkpoints=save_checkpoints, force=args.force,
            execution_source_sha256=source["execution_sha256"],
            fingerprint=fingerprint, dataset_hash_mode=dataset_hash,
            accepted_sources=accepted_sources, plan=plan,
            execution_args=execution_args, max_parallel=max_parallel,
            threads_per_job=int(args.threads_per_job),
            job_memory_mib=int(args.job_memory_mib),
            stop_on_error=args.stop_on_error, stream_logs=args.stream_logs,
            launch_stagger_seconds=float(args.launch_stagger))
    except KeyboardInterrupt:
        atomic_json(run_dir / "suite_summary.json", {
            "state": "interrupted", "updated_at": utc_now(),
            "outcomes": outcomes})
        print("Interrupted; completed jobs remain resumable.", file=sys.stderr)
        return 130

    failures = [item for item in outcomes if item["state"] == "failed"]
    summary = {
        "state": "failed" if failures else "completed",
        "updated_at": utc_now(), "profile": args.profile,
        "job_count": len(jobs), "job_granularity": args.job_granularity,
        "scheduling": plan["scheduling"], "outcomes": outcomes,
        "failed_jobs": failures,
        "rerun_policy": (
            "Successful jobs with matching SHA-256 are skipped. Failed or "
            "corrupt jobs are rerun under the same explicit seed/config; no "
            "seed is silently excluded."),
    }
    atomic_json(run_dir / "suite_summary.json", summary)
    atomic_json(run_dir / "failures.json", {
        "failed_jobs": failures, "silent_exclusions": False})

    if not args.skip_verify:
        verify_command = [sys.executable, str(ROOT / "tools" /
                          "verify_experiment_bundle.py"), str(run_dir)]
        print("\n[verify] " + " ".join(verify_command))
        verification = subprocess.run(verify_command, cwd=ROOT, check=False)
        summary["postrun_verifier_returncode"] = verification.returncode
        if verification.returncode != 0:
            failures.append({"job_id": "postrun_verifier", "state": "failed",
                             "returncode": verification.returncode})
            summary.update(state="failed", failed_jobs=failures)
        # The summary and failure list must agree with the exit status.
        atomic_json(run_dir / "suite_summary.json", summary)
        atomic_json(run_dir / "failures.json", {
            "failed_jobs": failures, "silent_exclusions": False})

    print(f"\nCompleted. Return the entire directory: {run_dir}")
    if args.job_granularity == "method" and not failures:
        print("Merge the per-method artifacts with: python tools/merge_results.py "
              f"{run_dir}   (writes {run_dir / 'merged'})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
