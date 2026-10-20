"""Static regression for PyTorch's asymmetric KL-divergence API.

The runtime used to build this release is torch-free, so this test parses the
small method body and pins the production call convention: student log-probs
must be the first argument and teacher probabilities the second.
"""

import ast
import os


def test_kl_direction_is_teacher_to_student():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, os.pardir, "diltcl", "lwf.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "kl_div"]
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "log_student"
    assert isinstance(call.args[1], ast.Name) and call.args[1].id == "teacher"


if __name__ == "__main__":
    test_kl_direction_is_teacher_to_student()
    print("PASS test_kl_direction_is_teacher_to_student")
