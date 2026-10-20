#!/usr/bin/env python3
"""Compare the installed software (and, optionally, the GPU) with the recorded stack.

Bit-for-bit regeneration of the trained experts needs the stack the original
runs used. This tool reads ``requirements-lock.txt`` (exact pins, including the
PyTorch wheels pinned by URL) and reports every installed package whose version
differs, plus missing ones. With ``--recorded-environment RUN/environment.json``
it also compares the Python version, GPU model, driver, CUDA and cuDNN
versions with those recorded by a run.

Exit status 0 when everything matches. Needs no PyTorch for the package check;
the GPU check imports PyTorch and calls ``nvidia-smi`` when available.

Example::

    python tools/check_environment.py --lock requirements-lock.txt \\
        --recorded-environment /root/autodl-tmp/runs_v3/p0_v3/environment.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote

# Packages whose version strings in a wheel URL carry the local tag (+cu130).
_URL_VERSION = re.compile(r"/(?P<name>[A-Za-z0-9_.]+)-(?P<version>[^-]+)-cp\d+")


def parse_lock(path: Path) -> dict[str, str]:
    pins = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith("#") else ""
        if not line:
            continue
        if " @ " in line:
            name, url = line.split(" @ ", 1)
            match = _URL_VERSION.search(unquote(url))
            if match:
                pins[name.strip().lower()] = match.group("version")
        elif "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = version.strip()
    return pins


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_packages(pins: dict[str, str]) -> list[str]:
    installed = {_normal(d.metadata["Name"]): d.version for d in metadata.distributions()
                 if d.metadata["Name"]}
    problems = []
    for name, version in sorted(pins.items()):
        have = installed.get(_normal(name))
        if have is None:
            problems.append(f"{name}: not installed (recorded {version})")
        elif have != version:
            problems.append(f"{name}: installed {have}, recorded {version}")
    return problems


def check_runtime(recorded: dict) -> list[str]:
    problems = []
    want_python = (recorded.get("python") or "").split()[0]
    if want_python and platform.python_version() != want_python:
        problems.append(f"python: running {platform.python_version()}, recorded {want_python}")
    probe = recorded.get("torch_probe") or {}
    try:
        import torch
        have = {"torch": torch.__version__, "cuda_version": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        for key in ("torch", "cuda_version", "cudnn", "devices"):
            if key in probe and probe[key] != have[key]:
                problems.append(f"{key}: running {have[key]}, recorded {probe[key]}")
    except ImportError:
        problems.append("torch: not importable")
    recorded_smi = ((recorded.get("nvidia_smi") or {}).get("stdout") or "").split(", ")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False).stdout.strip()
        name, driver = [x.strip() for x in out.splitlines()[0].split(",")][:2] if out else ("", "")
        if len(recorded_smi) >= 4:
            if name != recorded_smi[1]:
                problems.append(f"gpu: running {name!r}, recorded {recorded_smi[1]!r}")
            if driver != recorded_smi[3]:
                problems.append(f"driver: running {driver}, recorded {recorded_smi[3]} "
                                "(usually harmless for bit identity; reported for the record)")
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        problems.append("nvidia-smi: unavailable")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lock", type=Path, default=Path(__file__).resolve().parent.parent
                   / "requirements-lock.txt")
    p.add_argument("--recorded-environment", type=Path, default=None,
                   help="A run's environment.json to compare Python/GPU/CUDA/cuDNN with.")
    args = p.parse_args(argv)
    problems = check_packages(parse_lock(args.lock))
    if args.recorded_environment:
        problems += check_runtime(json.loads(args.recorded_environment.read_text(encoding="utf-8")))
    for line in problems:
        print("MISMATCH", line)
    blocking = [x for x in problems if not x.startswith("driver:")]
    print("environment matches the recorded stack" if not blocking
          else f"{len(blocking)} blocking mismatch(es); bit-for-bit regeneration is not expected")
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
