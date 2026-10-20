#!/usr/bin/env python3
"""Deprecated name kept for old command lines; use ``run_suite.py``.

The v3 launcher (``run_suite.py``) runs jobs in parallel and splits them per
method by default. This shim forwards every argument unchanged.
"""

import sys

import run_suite

if __name__ == "__main__":
    print("[run_v21_experiment_suite] deprecated: forwarding to run_suite.py",
          file=sys.stderr)
    raise SystemExit(run_suite.main())
