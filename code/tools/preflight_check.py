#!/usr/bin/env python3
"""Standalone dependency/dataset preflight (no PyTorch needed).

Thin wrapper around :mod:`diltcl.preflight` that loads the module by path, so
it runs on a bare machine before the training stack is installed. Use it to
confirm a node is ready *before* launching an expensive suite.

Example::

    python code/tools/preflight_check.py --data-root /root/autodl-tmp/data \
        --benchmarks cifar,pmnist,tinyimagenet --require-calibration
"""

import importlib.util
import pathlib
import sys

_MODULE = pathlib.Path(__file__).resolve().parent.parent / "diltcl" / "preflight.py"
_spec = importlib.util.spec_from_file_location("diltcl_preflight", _MODULE)
_preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_preflight)

if __name__ == "__main__":
    raise SystemExit(_preflight.main(sys.argv[1:]))
