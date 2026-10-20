"""Torch-free checks for the resumable, parallel experiment launcher/verifier."""

from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_suite as suite  # noqa: E402
import verify_experiment_bundle as verifier  # noqa: E402

_config_spec = importlib.util.spec_from_file_location(
    "diltcl_config_standalone", ROOT / "diltcl" / "config.py")
config_module = importlib.util.module_from_spec(_config_spec)
sys.modules[_config_spec.name] = config_module
_config_spec.loader.exec_module(config_module)


class ExperimentSuiteTests(unittest.TestCase):
    def test_seed_ranges(self):
        self.assertEqual(suite.parse_seed_spec("0,2,4-6"), [0, 2, 4, 5, 6])
        with self.assertRaises(ValueError):
            suite.parse_seed_spec("3-1")
        with self.assertRaises(ValueError):
            suite.parse_seed_spec("1,1")

    def test_config_overrides_are_strict(self):
        cfg = config_module.ExperimentConfig().with_overrides({
            "diltcl_expert_finetune_epochs": 0,
            "diltcl_sparsity": 0.95,
            "five_dataset_names": ["cifar10", "mnist"],
        })
        self.assertEqual(cfg.diltcl_expert_finetune_epochs, 0)
        self.assertEqual(cfg.five_dataset_names, ("cifar10", "mnist"))
        with self.assertRaises(ValueError):
            cfg.with_overrides({"typo_epochs": 3})
        with self.assertRaises(ValueError):
            cfg.with_overrides({"diltcl_sparsity": 1.0})

    def test_p0_is_seed_isolated_and_unique(self):
        jobs = suite.p0_jobs([0, 3])
        self.assertEqual(len(jobs), 14)
        self.assertEqual(len({job.job_id for job in jobs}), len(jobs))
        self.assertTrue(all(job.seed in {0, 3} for job in jobs))
        self.assertTrue(any(job.benchmark == "tinyimagenet" for job in jobs))

    def test_method_granularity_splits_every_multi_method_job(self):
        per_seed = suite.p0_jobs([0], "multi", granularity="method")
        # 3 task-IL benchmarks x 10 methods + 2 class-IL benchmarks x 2
        # inference groups x 9 methods.
        self.assertEqual(len(per_seed), 3 * 10 + 2 * 2 * 9)
        self.assertEqual(len({j.job_id for j in per_seed}), len(per_seed))
        self.assertTrue(all("," not in j.methods for j in per_seed))
        self.assertTrue(all(j.job_id.endswith("__" + j.methods)
                            for j in per_seed))
        # The seed layout is unchanged, and a same-seed repeat pair is never
        # split (both sides must stay whole artifacts to be comparable).
        self.assertEqual(len(suite.p0_jobs([0], "multi", granularity="seed")), 7)
        smoke = suite.smoke_jobs([0])
        repeat = [j for j in smoke if j.repeat_of]
        self.assertEqual(len(repeat), 1)
        split = suite._split_by_method(smoke, "method")
        ids = {j.job_id for j in split}
        self.assertIn(repeat[0].job_id, ids)
        self.assertIn(repeat[0].repeat_of, ids)
        self.assertGreater(len(split), len(smoke))
        with self.assertRaises(ValueError):
            suite._split_by_method(smoke, "bogus")

    def test_cost_model_orders_long_jobs_first(self):
        jobs = suite.p0_jobs([0], "multi", granularity="method")
        ordered = sorted(jobs, key=suite.estimate_cost, reverse=True)
        self.assertEqual(ordered[0].benchmark, "tinyimagenet")
        self.assertEqual(ordered[0].methods, "diltcl")
        cheap = [j for j in jobs if j.benchmark == "pmnist"
                 and j.methods == "finetune"][0]
        self.assertLess(suite.estimate_cost(cheap),
                        suite.estimate_cost(ordered[0]))

    def test_execution_args_reach_the_command_and_the_plan_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            job = suite.p0_jobs([1], "multi", granularity="method")[0]
            command, _ = suite.command_for(
                job, run_dir, run_dir / "data", False,
                execution_args=("--numerics", "legacy", "--tf32", "off"))
            self.assertIn("--numerics", command)
            self.assertEqual(command[command.index("--numerics") + 1], "legacy")
            fast = suite.plan_payload(
                [job], run_dir, run_dir / "data", False, True, "p0", [1],
                "sha", "full", execution_options={"numerics": "fast"},
                granularity="method")
            legacy = suite.plan_payload(
                [job], run_dir, run_dir / "data", False, True, "p0", [1],
                "sha", "full", execution_options={"numerics": "legacy"},
                granularity="method")
            self.assertNotEqual(fast["plan_sha256"], legacy["plan_sha256"])
            self.assertEqual(fast["job_granularity"], "method")

    def test_max_parallel_resolution(self):
        value, detail = suite.resolve_max_parallel("3", 3, 2048)
        self.assertEqual((value, detail["rule"]), (3, "explicit"))
        with self.assertRaises(ValueError):
            suite.resolve_max_parallel("0", 3, 2048)
        value, detail = suite.resolve_max_parallel("auto", 3, 2048)
        self.assertGreaterEqual(value, 1)
        self.assertLessEqual(value, suite.MAX_AUTO_PARALLEL)
        self.assertIn("by_cpu", detail)

    def test_metadata_fingerprint_is_a_warning_for_smoke_but_an_error_for_p0(self):
        def run_dir_with(jobs, mode):
            directory = tempfile.mkdtemp()
            run_dir = Path(directory)
            for name in ("environment.json", "source_manifest.json",
                         "suite_summary.json", "failures.json"):
                (run_dir / name).write_text(json.dumps(
                    {"execution_sha256": "sha"} if name == "source_manifest.json"
                    else {}))
            (run_dir / "pip_freeze.txt").write_text("")
            (run_dir / "plan.json").write_text(json.dumps(
                {"execution_source_sha256": "sha", "jobs": jobs}))
            (run_dir / "dataset_fingerprint.json").write_text(json.dumps(
                {"root": "/d", "mode": mode, "tree_sha256": "t"}))
            return run_dir

        # An empty plan never verifies; the fingerprint rule is still visible:
        # for a non-publication (smoke) plan it is only a warning.
        smoke = verifier.verify(run_dir_with([], "metadata"))
        self.assertEqual(smoke["status"], "failed")
        self.assertTrue(any("not content-hash" in w for w in smoke["warnings"]))
        self.assertEqual(smoke["errors"], ["plan.json lists no jobs"])

        p0_job = {"job": {"job_id": "p0_task_il_multi__cifar__seed0__lwf",
                          "priority": "P0", "seed": 0, "benchmark": "cifar"}}
        publication = verifier.verify(run_dir_with([p0_job], "metadata"))
        self.assertEqual(publication["status"], "failed")
        self.assertTrue(any("not content-hash" in e for e in publication["errors"]))
        full = verifier.verify(run_dir_with([], "full"))
        self.assertEqual(full["errors"], ["plan.json lists no jobs"])
        self.assertEqual(full["warnings"], [])

    def _data_root(self, base, with_mnist=False):
        root = Path(base) / "data"
        (root / "cifar-100-python").mkdir(parents=True, exist_ok=True)
        for name in ("train", "test", "meta"):
            (root / "cifar-100-python" / name).write_bytes(b"c" * 10)
        (root / "tiny-imagenet-200" / "train").mkdir(parents=True, exist_ok=True)
        (root / "tiny-imagenet-200" / "train" / "x.JPEG").write_bytes(b"t")
        if with_mnist:
            (root / "MNIST" / "raw").mkdir(parents=True, exist_ok=True)
            (root / "MNIST" / "raw" / "train-images-idx3-ubyte").write_bytes(b"m")
        return root

    def test_dataset_identity_is_per_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._data_root(tmp)
            before = suite.fingerprint_data_root(root, "full")
            self._data_root(tmp, with_mnist=True)          # stage MNIST later
            after = suite.fingerprint_data_root(root, "full")
            self.assertNotEqual(before["tree_sha256"], after["tree_sha256"])
            self.assertEqual(before["entries"]["cifar-100-python"],
                             after["entries"]["cifar-100-python"])
            cifar_before = suite.job_dataset_identity(before, "cifar", root, "full")
            cifar_after = suite.job_dataset_identity(after, "cifar", root, "full")
            self.assertEqual(cifar_before["entries"], cifar_after["entries"])
            self.assertEqual(list(cifar_after["entries"]), ["cifar-100-python"])
            pm_before = suite.job_dataset_identity(before, "pmnist", root, "full")
            pm_after = suite.job_dataset_identity(after, "pmnist", root, "full")
            self.assertEqual(pm_before["entries"], {"MNIST": None})
            self.assertNotEqual(pm_before["entries"], pm_after["entries"])
            # New-format identities compare entries, not the whole tree.
            self.assertTrue(suite.dataset_identity_matches(cifar_before, cifar_after))
            self.assertFalse(suite.dataset_identity_matches(pm_before, pm_after))
            # Entries can be recomputed for fingerprints written before v3.0.1.
            legacy = {k: v for k, v in before.items() if k != "entries"}
            self.assertEqual(suite.fingerprint_entries(legacy["files"]),
                             before["entries"])

    def test_symlinked_dataset_folders_are_fingerprinted_like_real_ones(self):
        import os
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            real = self._data_root(Path(tmp) / "real")
            reference = suite.fingerprint_data_root(real, "full")
            # The same files, with CIFAR-100 linked in from a shared store.
            store = Path(tmp) / "store"
            store.mkdir()
            shutil.copytree(real / "cifar-100-python", store / "cifar-100-python")
            linked = Path(tmp) / "linked"
            linked.mkdir()
            shutil.copytree(real / "tiny-imagenet-200", linked / "tiny-imagenet-200")
            os.symlink(store / "cifar-100-python", linked / "cifar-100-python")
            # A link back to the data root must not loop.
            os.symlink(linked, linked / "tiny-imagenet-200" / "loop")
            fingerprint = suite.fingerprint_data_root(linked, "full")
            self.assertEqual(fingerprint["entries"], reference["entries"])
            self.assertEqual(fingerprint["tree_sha256"], reference["tree_sha256"])
            # Every entry a planned job reads must be recorded before it runs.
            jobs = [suite.Job(job_id=f"g__{b}__seed0__wsn", priority="P0", purpose="t",
                              runner="run_class_il.py", benchmark=b, seed=0, methods="wsn")
                    for b in ("cifar", "pmnist")]
            self.assertEqual(suite.unrecorded_entries(fingerprint, jobs), ["pmnist: MNIST"])
            # Real trees hash exactly as v3.0.6 did (rglob order and records).
            legacy = sorted((p for p in real.rglob("*") if p.is_file()),
                            key=lambda p: p.relative_to(real).as_posix())
            self.assertEqual([r["path"] for r in reference["files"]],
                             [p.relative_to(real).as_posix() for p in legacy])

    def test_old_format_statuses_survive_a_data_root_change_via_archive(self):
        """The first P0 run recorded only the whole-tree hash. After MNIST is
        staged, its CIFAR/TinyImageNet jobs must still validate through the
        archived fingerprint, while Permuted-MNIST jobs must rerun."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._data_root(tmp)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            before = suite.fingerprint_data_root(root, "full")
            suite.archive_fingerprint(run_dir, before)
            old_identity = {"root": before["root"], "mode": "full",
                            "tree_sha256": before["tree_sha256"]}
            self._data_root(tmp, with_mnist=True)
            after = suite.fingerprint_data_root(root, "full")
            for benchmark, expect in (("cifar", True), ("tinyimagenet", True),
                                      ("pmnist", False)):
                expected = suite.job_dataset_identity(after, benchmark, root, "full")
                self.assertEqual(
                    suite.dataset_identity_matches(old_identity, expected, run_dir),
                    expect, benchmark)
            # Without the archive an old-format identity cannot be trusted.
            self.assertFalse(suite.dataset_identity_matches(
                old_identity, suite.job_dataset_identity(after, "cifar", root, "full"),
                Path(tmp) / "elsewhere"))

    def test_source_change_acknowledgement_keeps_finished_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            job = suite.p0_jobs([0], "multi", granularity="method")[0]
            old_plan = suite.plan_payload([job], run_dir, run_dir / "data", False,
                                          True, "p0", [0], "old-sha", "full",
                                          execution_options={"numerics": "fast"},
                                          granularity="method")
            new_plan = suite.plan_payload([job], run_dir, run_dir / "data", False,
                                          True, "p0", [0], "new-sha", "full",
                                          execution_options={"numerics": "fast"},
                                          granularity="method")
            self.assertNotEqual(old_plan["plan_sha256"], new_plan["plan_sha256"])
            self.assertEqual(suite.plan_identity(old_plan, include_source=False),
                             suite.plan_identity(new_plan, include_source=False))
            self.assertNotEqual(suite.plan_identity(old_plan), suite.plan_identity(new_plan))
            new_plan["source_change_acknowledgements"] = [
                {"from": "old-sha", "to": "new-sha", "reason": "launcher fix"}]
            self.assertEqual(suite.accepted_source_hashes(new_plan), {"old-sha", "new-sha"})
            status_path = run_dir / "statuses" / "j.json"
            status_path.parent.mkdir()
            out = run_dir / "artifacts" / "j" / "result.json"
            out.parent.mkdir(parents=True)
            out.write_text("{}")
            status_path.write_text(json.dumps({
                "state": "succeeded", "execution_source_sha256": "old-sha",
                "dataset_identity": {"root": None, "mode": "none", "tree_sha256": None},
                "outputs": [{"path": str(out), "path_relative_to_run_dir":
                             "artifacts/j/result.json", "bytes": 2,
                             "sha256": hashlib.sha256(b"{}").hexdigest()}]}))
            identity = {"root": None, "mode": "none", "tree_sha256": None}
            self.assertFalse(suite.status_is_valid(status_path, "new-sha", identity))
            self.assertTrue(suite.status_is_valid(status_path, "new-sha", identity,
                                                  {"old-sha", "new-sha"}, run_dir))

    def test_invalidated_methods_force_reruns_and_fail_verification(self):
        plan = {"execution_source_sha256": "C", "source_change_acknowledgements": [
            {"from": "A", "to": "B", "reason": "launcher fix", "invalidated_methods": []},
            {"from": "B", "to": "C", "reason": "edge-popup STE fix",
             "invalidated_methods": ["supsup", "wsn"]}]}
        self.assertEqual(suite.source_chain(plan), ["A", "B", "C"])
        # supsup produced under A or B is invalid; under C it stands.
        self.assertIsNotNone(suite.result_invalidated(plan, "A", ["supsup"]))
        self.assertIsNotNone(suite.result_invalidated(plan, "B", ["wsn"]))
        self.assertIsNone(suite.result_invalidated(plan, "C", ["supsup"]))
        # Other methods under old sources stand.
        self.assertIsNone(suite.result_invalidated(plan, "A", ["lwf"]))
        self.assertIsNone(suite.result_invalidated(plan, "unknown", ["supsup"]))
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "statuses").mkdir()
            out = run_dir / "artifacts" / "j" / "result.json"
            out.parent.mkdir(parents=True)
            out.write_text("{}")
            for method, sha in (("supsup", "A"), ("lwf", "A"), ("supsup", "C")):
                status_path = run_dir / "statuses" / f"{method}_{sha}.json"
                status_path.write_text(json.dumps({
                    "state": "succeeded", "execution_source_sha256": sha,
                    "job": {"methods": method},
                    "dataset_identity": {"root": None, "mode": "none", "tree_sha256": None},
                    "outputs": [{"path": str(out), "path_relative_to_run_dir":
                                 "artifacts/j/result.json", "bytes": 2,
                                 "sha256": hashlib.sha256(b"{}").hexdigest()}]}))
            identity = {"root": None, "mode": "none", "tree_sha256": None}
            accepted = suite.accepted_source_hashes(plan)
            self.assertFalse(suite.status_is_valid(run_dir / "statuses" / "supsup_A.json",
                                                   "C", identity, accepted, run_dir, plan))
            self.assertTrue(suite.status_is_valid(run_dir / "statuses" / "lwf_A.json",
                                                  "C", identity, accepted, run_dir, plan))
            self.assertTrue(suite.status_is_valid(run_dir / "statuses" / "supsup_C.json",
                                                  "C", identity, accepted, run_dir, plan))

    def test_verifier_accepts_acknowledged_source_changes_with_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for name in ("environment.json", "suite_summary.json", "failures.json"):
                (run_dir / name).write_text("{}")
            (run_dir / "source_manifest.json").write_text(json.dumps({"execution_sha256": "new"}))
            (run_dir / "pip_freeze.txt").write_text("")
            matrix = np.array([[50.0]])
            payload = {"benchmarks": ["cifar"], "seeds": [0], "methods": ["finetune"],
                       "metric_schema_version": 3, "verification_status": "unverified",
                       "per_seed": {"cifar": {"finetune": [verifier.summarize(matrix)]}},
                       "checkpoint_persistence_enabled": False}
            art = run_dir / "artifacts" / "job" / "result"
            art.parent.mkdir(parents=True)
            art.with_suffix(".json").write_text(json.dumps(payload))
            np.savez(art.with_suffix(".npz"), **{"cifar/finetune/seed0": matrix})
            outputs = [{"path": str(p), "path_relative_to_run_dir": str(p.relative_to(run_dir)),
                        "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in (art.with_suffix(".json"), art.with_suffix(".npz"))]
            (run_dir / "statuses").mkdir()
            (run_dir / "statuses" / "job.json").write_text(json.dumps({
                "state": "succeeded", "execution_source_sha256": "old",
                "dataset_identity": {"root": None, "mode": "none", "tree_sha256": None},
                "outputs": outputs}))
            job = {"job": {"job_id": "job", "priority": "P0-smoke", "seed": 0,
                           "benchmark": "cifar", "methods": "finetune",
                           "repeat_of": None},
                   "output_prefix_relative": "artifacts/job/result"}
            plan = {"execution_source_sha256": "new", "jobs": [job],
                    "dataset_hash_mode": "none",
                    "source_change_acknowledgements": []}
            (run_dir / "plan.json").write_text(json.dumps(plan))
            report = verifier.verify(run_dir)
            self.assertTrue(any("different executable source" in e for e in report["errors"]))
            plan["source_change_acknowledgements"] = [
                {"from": "old", "to": "new", "reason": "validation fix"}]
            (run_dir / "plan.json").write_text(json.dumps(plan))
            report = verifier.verify(run_dir)
            self.assertFalse(any("executable source" in e for e in report["errors"]), report["errors"])
            self.assertTrue(any("acknowledged" in w for w in report["warnings"]))
            self.assertEqual(report["jobs_under_acknowledged_source"], ["job"])
            # If the acknowledged change invalidated this job's method, an old
            # result is an error, not a warning.
            plan["source_change_acknowledgements"][0]["invalidated_methods"] = ["finetune"]
            (run_dir / "plan.json").write_text(json.dumps(plan))
            report = verifier.verify(run_dir)
            self.assertTrue(any("invalidated" in e for e in report["errors"]), report["errors"])

    def test_archived_entries_are_cached_and_run_records_are_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._data_root(tmp)
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            fp = suite.fingerprint_data_root(root, "full")
            suite.archive_fingerprint(run_dir, fp)
            suite._ARCHIVED_ENTRIES_CACHE.clear()
            first = suite.archived_entries(run_dir, fp["tree_sha256"])
            self.assertEqual(first["entries"], fp["entries"])
            # Second lookup is served from the cache even if the file vanished.
            (run_dir / "dataset_fingerprints" / f"{fp['tree_sha256']}.json").unlink()
            self.assertIs(suite.archived_entries(run_dir, fp["tree_sha256"]), first)
            self.assertIsNone(suite.archived_entries(run_dir, "missing"))
            # Replacing a run-level record archives the previous version.
            manifest = run_dir / "source_manifest.json"
            manifest.write_text(json.dumps({"execution_sha256": "old", "files": []}))
            archived = suite.archive_if_changed(
                manifest, {"execution_sha256": "new", "files": []},
                run_dir / "source_manifests", "execution_sha256")
            self.assertEqual(archived, run_dir / "source_manifests" / "old.json")
            self.assertIsNone(suite.archive_if_changed(
                manifest, {"execution_sha256": "old", "files": []},
                run_dir / "source_manifests", "execution_sha256"))

    def test_editor_and_cache_files_never_enter_the_source_hash(self):
        """The first P0 run directory already contained
        ``.ipynb_checkpoints/README-checkpoint.md`` and ``.pytest_cache/`` in
        its source manifest; a Jupyter autosave of a .py must not become a
        'source change' that forces a full rerun."""
        self.assertTrue(suite.is_source_path(Path("diltcl/ewc.py")))
        self.assertTrue(suite.is_source_path(Path("tools/merge_results.py")))
        self.assertTrue(suite.is_source_path(Path("requirements.txt")))
        for bad in (".ipynb_checkpoints/run_suite-checkpoint.py",
                    "diltcl/.ipynb_checkpoints/ewc-checkpoint.py",
                    ".pytest_cache/README.md", "__pycache__/x.py",
                    ".ruff_cache/0.1/abc", "diltcl/__pycache__/ewc.cpython-312.pyc"):
            self.assertFalse(suite.is_source_path(Path(bad)), bad)
        manifest = suite.source_manifest()
        # Jupyter autosaves live in .ipynb_checkpoints/ and end in "-checkpoint";
        # a real source file may still mention checkpoints in its name
        # (tools/evaluate_checkpoints.py).
        self.assertTrue(all(".ipynb_checkpoints" not in r["path"]
                            and "-checkpoint." not in r["path"]
                            and ".pytest_cache" not in r["path"]
                            for r in manifest["files"]))

    def test_p1_contains_requested_ablations(self):
        jobs = suite.p1_jobs([0])
        configs = [job.config_overrides for job in jobs]
        self.assertIn({"diltcl_sparsity": 0.80}, configs)
        self.assertIn({"diltcl_partitions": 4}, configs)
        self.assertIn({"pruning_rounds": 11}, configs)
        self.assertIn({"diltcl_expert_finetune_epochs": 0}, configs)

    def test_plan_commands_use_explicit_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            job = suite.p0_jobs([7])[0]
            command, output = suite.command_for(
                job, run_dir, Path("/data"), allow_download=False)
            self.assertIn("--seed-list", command)
            self.assertEqual(command[command.index("--seed-list") + 1], "7")
            self.assertIn("--no-download", command)
            self.assertTrue(str(output).endswith("result"))

    def test_metric_recomputation(self):
        matrix = np.array([[80.0, 0.0], [70.0, 90.0]])
        metrics = verifier.summarize(matrix)
        self.assertEqual(metrics["ACC"], 80.0)
        self.assertEqual(metrics["LA"], 85.0)
        self.assertEqual(metrics["BWT"], -10.0)
        self.assertEqual(metrics["Forgetting"], 10.0)

    def test_task_il_json_npz_contract_verifies(self):
        matrix = np.array([[80.0, 0.0], [80.0, 90.0]])
        core = np.array([[70.0, 0.0], [70.0, 20.0]])
        metrics = verifier.summarize(matrix)
        core_metrics = verifier.summarize(core)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "seed0.pt"
            checkpoint.write_bytes(b"synthetic-checkpoint")
            checkpoint_record = {
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "content_sha256": hashlib.sha256(
                    b"synthetic-content").hexdigest(),
                "serialized_bytes": checkpoint.stat().st_size,
                "persisted": True,
                "persisted_path": str(checkpoint),
            }
            payload = {
                "protocol": "task_il_shared_local_label_axis",
                "verification_status": "unverified",
                "checkpoint_persistence_enabled": True,
                "environment": {},
                "benchmarks": ["cifar"], "methods": ["diltcl"],
                "seeds": [0],
                "per_seed": {"cifar": {"diltcl": [metrics]}},
                "aggregated": {"cifar": {"diltcl": {
                    name: {"mean": value, "std": 0.0, "values": [value]}
                    for name, value in metrics.items()
                }}},
                "per_seed_run_records": {"cifar": {"diltcl": {"0": {
                    "method_extras": {
                        "checkpoint": checkpoint_record,
                        "core_only_metrics": core_metrics,
                        "diltcl_requested_target_sparsity": 0.9,
                        "diltcl_core_partition_global_density": [0.1, 0.1],
                        "diltcl_expert_eligible_density_by_task": {"1": 0.1},
                    }
                }}}},
            }
            json_path, npz_path = root / "result.json", root / "result.npz"
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            np.savez(npz_path, **{
                "cifar/diltcl/seed0": matrix,
                "cifar/diltcl/seed0/core_only": core,
            })
            result = verifier.validate_result(json_path, npz_path, 0, "cifar")
            self.assertEqual(result["errors"], [])


    def test_head_mode_is_part_of_the_task_il_job_identity(self):
        """A shared-head and a multi-head run are different experiments.

        Sharing a job id would let one silently resume into the other's
        output path, mixing two protocols in a single result directory.
        Class-IL jobs are deliberately unaffected: they use one global head,
        so the same job is valid for both modes and may be reused.
        """
        def task_ids(mode):
            return {j.job_id for j in suite.build_jobs("p0", [0], mode)
                    if "task_il" in j.job_id}

        shared, multi = task_ids("shared"), task_ids("multi")
        assert shared and multi
        assert not (shared & multi), sorted(shared & multi)
        assert all("task_il_shared" in j for j in shared)
        assert all("task_il_multi" in j for j in multi)

        # Class-IL job ids must stay identical across modes.
        def class_ids(mode):
            return {j.job_id for j in suite.build_jobs("p0", [0], mode)
                    if "class_" in j.job_id}

        assert class_ids("shared") == class_ids("multi")

    def test_head_mode_reaches_the_task_il_command(self):
        for mode in ("shared", "multi"):
            jobs = [j for j in suite.build_jobs("p0", [0], mode)
                    if "task_il" in j.job_id]
            assert jobs
            for job in jobs:
                assert "--head-mode" in job.extra_args
                assert mode in job.extra_args

    def test_p1_shared_backbone_task_il_jobs_follow_the_head_mode(self):
        """P1's external-validity and compute-matched jobs train the same
        shared-backbone baselines as P0 and must use the same protocol."""
        shared_methods = {"finetune", "ewc", "lwf", "er", "derpp"}
        for mode in ("shared", "multi"):
            for job in suite.build_jobs("p1", [0], mode):
                methods = set(job.methods.split(","))
                if job.runner == "run_task_il.py" and methods & shared_methods:
                    assert "--head-mode" in job.extra_args, job.job_id
                    assert mode in job.extra_args, job.job_id
                    assert mode in job.job_id, job.job_id
        multi = {j.job_id for j in suite.build_jobs("p1", [0], "multi")}
        shared = {j.job_id for j in suite.build_jobs("p1", [0], "shared")}
        assert multi != shared


if __name__ == "__main__":
    unittest.main()
