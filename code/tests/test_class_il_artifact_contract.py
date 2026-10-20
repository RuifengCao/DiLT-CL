"""Torch-free contract tests for class-IL artifacts and table gating."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, os.pardir))
RUNNER = os.path.join(CODE, "run_class_il.py")
TABLE_SCRIPT = os.path.join(CODE, "make_class_il_table.py")


def _load_table_module():
    spec = importlib.util.spec_from_file_location(
        "make_class_il_table_under_test", TABLE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric(mean, std=1.0):
    return {"mean": float(mean), "std": float(std), "values": []}


def _artifact(n_seeds=5, verification="verified", schema=3):
    methods = ["derpp", "diltcl"]
    derpp = [{"ACC": 50.0 + i} for i in range(n_seeds)]
    diltcl = [{"ACC": 60.0 + i} for i in range(n_seeds)]
    common = {
        "BWT": _metric(-1.0),
        "Forgetting": _metric(2.0),
        "LA": _metric(55.0),
    }
    return {
        "metric_schema_version": schema,
        "code_version": "2.1.0",
        "verification_status": verification,
        "benchmarks": ["cifar"],
        "methods": methods,
        "seeds": list(range(n_seeds)),
        "inference_criterion": "margin",
        "inference_calibration": "score-logreg",
        "per_seed": {"cifar": {"derpp": derpp, "diltcl": diltcl}},
        "aggregated": {
            "cifar": {
                "derpp": {"ACC": _metric(52.0), **common},
                "diltcl": {
                    "ACC": _metric(62.0),
                    **common,
                    "OracleACC": _metric(70.0),
                    "TA": _metric(80.0),
                    "ConditionalACC": _metric(75.0),
                },
            },
        },
    }


def _run_table(payload, *extra):
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "results.json")
        output = os.path.join(tmp, "table.tex")
        with open(source, "w") as f:
            json.dump(payload, f)
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, TABLE_SCRIPT, "--results", source, "--out", output,
             *extra],
            cwd=CODE, env=env, text=True, capture_output=True, check=False)
        rendered = ""
        if os.path.exists(output):
            with open(output) as f:
                rendered = f.read()
        return result, rendered


def test_runner_declares_v3_unverified_contract_and_five_seed_default():
    with open(RUNNER) as f:
        source = f.read()
    tree = ast.parse(source)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    # CODE_VERSION is no longer a literal: it is derived from the package
    # version, because hardcoding it let artifacts misreport which code
    # produced them. This test stays dependency-free (no torch import), so it
    # asserts on the source text; the runtime equality is checked in
    # tests/test_runner_contracts.py.
    assert "CODE_VERSION" not in constants, (
        "CODE_VERSION was reintroduced as a hardcoded literal")
    assert "CODE_VERSION = _package_version()" in source
    assert constants["METRIC_SCHEMA_VERSION"] == 3
    assert '"evidence_status": "unverified"' in source
    assert '"verification_status": "unverified"' in source
    assert '"shared_backbone": "fixed-full-head"' in source
    assert '"isolation": "task-local-head+router"' in source
    assert '"RouteTieRate"' in source
    assert '"routing_tie"' in source
    assert '"ZeroCorrectRouteTasks"' in source
    assert '"kind": "score_logreg"' not in source
    assert 'p.add_argument("--seeds", type=int, default=5' in source
    assert 'if len(set(methods)) != len(methods):' in source
    assert 'if len(set(seeds)) != len(seeds):' in source
    assert 'if not np.isfinite(calibration_c) or calibration_c <= 0:' in source


def test_fewer_than_five_pairs_never_get_stars():
    module = _load_table_module()
    module.paired_significance = lambda *_: (_ for _ in ()).throw(
        AssertionError("significance must not run for n < 5"))
    rendered = module._table("cifar", _artifact(n_seeds=4))
    diltcl_row = next(line for line in rendered.splitlines()
                      if line.startswith("DiLT-CL &"))
    assert r"\,$^{" not in diltcl_row


def test_significance_requires_five_unique_declared_aligned_seeds():
    module = _load_table_module()
    module.paired_significance = lambda *_: (_ for _ in ()).throw(
        AssertionError("invalid seed metadata must suppress significance"))
    for seeds in ([0, 1, 2, 3], [0, 0, 0, 0, 0]):
        artifact = _artifact(n_seeds=5)
        artifact["seeds"] = seeds
        rendered = module._table("cifar", artifact)
        diltcl_row = next(line for line in rendered.splitlines()
                          if line.startswith("DiLT-CL &"))
        assert r"\,$^{" not in diltcl_row


def test_five_pairs_can_get_stars_and_caption_discloses_protocol():
    module = _load_table_module()
    module.paired_significance = lambda *_: {"p_value": 0.0005}
    rendered = module._table("cifar", _artifact(n_seeds=5))
    diltcl_row = next(line for line in rendered.splitlines()
                      if line.startswith("DiLT-CL &"))
    assert r"\,$^{***}$" in diltcl_row
    assert "fixed-full-head" in rendered
    assert "task-local head plus exhaustive router" in rendered
    assert "ACC is the macro average of per-task accuracies" in rendered
    assert "MicroACC is stored separately" in rendered
    assert "criterion=margin" in rendered
    assert "calibration=score-logreg" in rendered
    assert "Code=2.1.0; metric schema=3; verification status=verified" in rendered
    assert "Oracle-route accuracy (OracleACC)" in rendered


def test_table_rejects_old_schema_even_with_diagnostic_override():
    result, rendered = _run_table(
        _artifact(schema=2, verification="unverified"), "--allow-unverified")
    assert result.returncode != 0
    assert "schema < 3" in (result.stdout + result.stderr)
    assert not rendered


def test_table_rejects_noninteger_schema():
    result, rendered = _run_table(
        _artifact(schema=3.9, verification="verified"))
    assert result.returncode != 0
    assert "non-integer" in (result.stdout + result.stderr)
    assert not rendered


def test_table_rejects_unverified_by_default():
    result, rendered = _run_table(_artifact(verification="unverified"))
    assert result.returncode != 0
    assert "verification_status" in (result.stdout + result.stderr)
    assert "--allow-unverified" in (result.stdout + result.stderr)
    assert not rendered


def test_allow_unverified_adds_conspicuous_diagnostic_caption():
    result, rendered = _run_table(
        _artifact(verification="unverified"), "--allow-unverified")
    assert result.returncode == 0, result.stdout + result.stderr
    assert r"\textbf{DIAGNOSTIC ONLY---UNVERIFIED ARTIFACT.}" in rendered
    assert "verification status=unverified" in rendered


def test_verified_future_schema_renders_without_override():
    result, rendered = _run_table(_artifact(schema=4, verification="verified"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "metric schema=4" in rendered
    assert "verification status=verified" in rendered
    assert "DIAGNOSTIC ONLY" not in rendered


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


def test_labels_benchmark_names_and_caption_notes_follow_the_artifact():
    module = _load_table_module()
    module.paired_significance = lambda *_: {"p_value": 0.5}
    artifact = _artifact(n_seeds=5)
    artifact["calibration_fraction"] = 0.1
    artifact["benchmarks"] = ["tinyimagenet"]
    for key in ("per_seed", "aggregated"):
        artifact[key] = {"tinyimagenet": artifact[key]["cifar"]}
    rendered = module._table("tinyimagenet", artifact)
    assert "results on Split-TinyImageNet" in rendered
    assert r"\label{tab:classil_tinyimagenet_score-logreg}" in rendered
    assert r"held-out 10\% of each task's training data" in rendered
    assert r"remaining 90\%" in rendered and "vs.\\ DER++" in rendered
    # Entropy routing keeps the plain label; no DER++ row means no stars promised.
    artifact["inference_calibration"] = "none"
    artifact["methods"] = ["diltcl"]
    rendered = module._table("tinyimagenet", artifact)
    assert r"\label{tab:classil_tinyimagenet}" in rendered
    assert "No DER++ reference row" in rendered and "vs.\\ DER++" not in rendered
    assert "held-out" not in rendered
