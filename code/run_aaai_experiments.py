#!/usr/bin/env python3
"""Deprecated name kept for old command lines; use ``run_task_il.py``.

The task-IL runner was renamed in v3.0.0. This shim forwards every argument
unchanged; job definitions, docs and tests refer to ``run_task_il.py``.
"""

import sys

import run_task_il

if __name__ == "__main__":
    print("[run_aaai_experiments] deprecated: forwarding to run_task_il.py",
          file=sys.stderr)
    run_task_il.main()
