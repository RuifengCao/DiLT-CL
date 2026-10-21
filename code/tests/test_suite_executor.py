"""Functional checks of ``run_suite.execute_jobs`` with stand-in child jobs.

The scheduler is exercised for real -- thread pool, child processes, per-job
logs, atomic status files, resume skipping, stop-on-error -- but each "job"
is a tiny Python one-liner that writes the expected ``result.json`` /
``result.npz`` pair (or fails on request), so the test needs neither torch nor
datasets.
"""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import run_suite as suite  # noqa: E402

_CHILD = r'''
import json, os, sys, time
prefix, fail, delay = sys.argv[1], sys.argv[2] == "1", float(sys.argv[3])
time.sleep(delay)
print("child working on", prefix, flush=True)
if fail:
    sys.exit(3)
os.makedirs(os.path.dirname(prefix), exist_ok=True)
with open(prefix + ".json", "w") as f:
    json.dump({"ok": True}, f)
with open(prefix + ".npz", "wb") as f:
    f.write(b"PK\x05\x06" + b"\0" * 18)   # empty zip: enough for an existence check
'''


def _fake_jobs():
    return [
        suite.Job(job_id=f"grp__cifar__seed0__{m}", priority="P0", purpose="t",
                  runner="run_task_il.py", benchmark="cifar", seed=0,
                  methods=m)
        for m in ("finetune", "diltcl", "lwf", "ewc")
    ]


def _prepare(args, data_root):
    """A real (not dry) ``run_suite.main`` that writes the run directory's plan
    and provenance but runs no job: execution, the environment probe and pip
    freeze are stubbed, so the check needs neither PyTorch nor datasets."""
    stubs = {"execute_jobs": lambda *a, **k: [],
             "environment_probe": lambda: {"torch_probe": {"ok": True}},
             "write_pip_freeze": lambda run_dir: None}
    saved = {name: getattr(suite, name) for name in stubs}
    for name, stub in stubs.items():
        setattr(suite, name, stub)
    try:
        return suite.main([*args, "--skip-preflight", "--skip-verify", "--dataset-hash", "none",
                           "--data-root", str(data_root)])
    finally:
        for name, original in saved.items():
            setattr(suite, name, original)


class ExecutorTests(unittest.TestCase):
    def _run(self, run_dir, jobs, fail=(), max_parallel=2, stop_on_error=False,
             force=False):
        def fake_command(job, run_dir_, data_root, allow_download,
                         save_checkpoints=True, execution_args=()):
            prefix = run_dir_ / "artifacts" / job.job_id / "result"
            delay = "0.3" if job.methods == "diltcl" else "0.05"
            return ([sys.executable, "-c", _CHILD, str(prefix),
                     "1" if job.methods in fail else "0", delay], prefix)

        original = suite.command_for
        suite.command_for = fake_command
        try:
            return suite.execute_jobs(
                jobs, run_dir=run_dir, data_root=None, allow_download=False,
                save_checkpoints=False, force=force,
                execution_source_sha256="sha", dataset_identity={"mode": "none"},
                execution_args=(), max_parallel=max_parallel, threads_per_job=1,
                job_memory_mib=1, stop_on_error=stop_on_error, stream_logs=False,
                launch_stagger_seconds=0.0)
        finally:
            suite.command_for = original

    def test_parallel_execution_writes_logs_statuses_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp)
            outcomes = self._run(run_dir, _fake_jobs())
            self.assertEqual(sorted(o["state"] for o in outcomes),
                             ["succeeded"] * 4)
            for job in _fake_jobs():
                status = json.loads(
                    (run_dir / "statuses" / f"{job.job_id}.json").read_text())
                self.assertEqual(status["state"], "succeeded")
                self.assertEqual(status["returncode"], 0)
                self.assertEqual(
                    status["child_environment_controls"]["DILTCL_NUM_THREADS"],
                    "1")
                self.assertTrue(
                    (run_dir / "artifacts" / job.job_id / "result.json").is_file())
                log = pathlib.Path(status["log"]).read_text()
                self.assertIn("child working on", log)
            # Resume: a second run skips every validated job.
            again = self._run(run_dir, _fake_jobs())
            self.assertEqual({o["state"] for o in again}, {"skipped"})
            # --force reruns them (attempt counter advances).
            forced = self._run(run_dir, _fake_jobs(), force=True)
            self.assertEqual({o["state"] for o in forced}, {"succeeded"})
            self.assertEqual(len(list((run_dir / "logs").glob("*diltcl*.log"))), 2)

    def test_failures_are_recorded_and_do_not_block_other_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp)
            outcomes = self._run(run_dir, _fake_jobs(), fail=("lwf",))
            by_id = {o["job_id"]: o for o in outcomes}
            self.assertEqual(by_id["grp__cifar__seed0__lwf"]["state"], "failed")
            self.assertEqual(by_id["grp__cifar__seed0__lwf"]["returncode"], 3)
            self.assertEqual(sum(o["state"] == "succeeded" for o in outcomes), 3)
            status = json.loads(
                (run_dir / "statuses" / "grp__cifar__seed0__lwf.json").read_text())
            self.assertEqual(status["state"], "failed")
            # A failed job is not "valid" for resume purposes.
            self.assertFalse(suite.status_is_valid(
                run_dir / "statuses" / "grp__cifar__seed0__lwf.json", "sha",
                {"mode": "none"}))

    def test_stop_on_error_stops_dispatching_new_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp)
            # Sequential so the ordering is deterministic: the slow DiLT-CL
            # job is dispatched first (longest-first), then the failing one.
            jobs = _fake_jobs()
            outcomes = self._run(run_dir, jobs, fail=("diltcl",),
                                 max_parallel=1, stop_on_error=True)
            self.assertEqual(outcomes[0]["job_id"], "grp__cifar__seed0__diltcl")
            self.assertEqual(outcomes[0]["state"], "failed")
            self.assertEqual(len(outcomes), 1)

    def test_gpu_headroom_guard_waits_only_while_jobs_run(self):
        readings = [(31000, 32000), (31000, 32000), (10000, 32000)]
        calls = []
        original = suite.gpu_memory_mib
        suite.gpu_memory_mib = lambda: (calls.append(1), readings[min(len(calls) - 1, 2)])[1]
        try:
            suite.wait_for_gpu_headroom(2048, running=1, poll_seconds=0.01)
            self.assertEqual(len(calls), 3)          # waited twice, then enough
            calls.clear()
            suite.wait_for_gpu_headroom(2048, running=0, poll_seconds=0.01)
            self.assertEqual(calls, [])              # nothing running: no wait
            calls.clear()
            suite.gpu_memory_mib = lambda: None      # no nvidia-smi: no wait
            suite.wait_for_gpu_headroom(2048, running=3, poll_seconds=0.01)
        finally:
            suite.gpu_memory_mib = original
        # An unrelated process hogging the GPU cannot stall the suite forever.
        suite.gpu_memory_mib = lambda: (31000, 32000)
        try:
            started = __import__("time").perf_counter()
            suite.wait_for_gpu_headroom(2048, running=1, poll_seconds=0.01,
                                        max_wait_seconds=0.05)
            self.assertLess(__import__("time").perf_counter() - started, 1.0)
        finally:
            suite.gpu_memory_mib = original

    def _run_with_fingerprint(self, run_dir, jobs, data_root, sources=None,
                              execution_sha="sha"):
        fingerprint = suite.fingerprint_data_root(data_root, "metadata")
        suite.archive_fingerprint(run_dir, fingerprint)

        def fake_command(job, run_dir_, data_root_, allow_download,
                         save_checkpoints=True, execution_args=()):
            prefix = run_dir_ / "artifacts" / job.job_id / "result"
            return ([sys.executable, "-c", _CHILD, str(prefix), "0", "0.01"], prefix)

        original = suite.command_for
        suite.command_for = fake_command
        try:
            return suite.execute_jobs(
                jobs, run_dir=run_dir, data_root=data_root, allow_download=False,
                save_checkpoints=False, force=False,
                execution_source_sha256=execution_sha, execution_args=(),
                max_parallel=2, threads_per_job=1, job_memory_mib=1,
                stop_on_error=False, stream_logs=False, launch_stagger_seconds=0.0,
                fingerprint=fingerprint, dataset_hash_mode="metadata",
                accepted_sources=sources)
        finally:
            suite.command_for = original

    def test_resume_survives_changes_to_other_benchmarks_data_and_acknowledged_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, data_root = pathlib.Path(tmp) / "run", pathlib.Path(tmp) / "data"
            run_dir.mkdir()
            (data_root / "cifar-100-python").mkdir(parents=True)
            (data_root / "cifar-100-python" / "train").write_bytes(b"c")
            jobs = [suite.Job(job_id="g__cifar__seed0__finetune", priority="P0",
                              purpose="t", runner="run_task_il.py", benchmark="cifar",
                              seed=0, methods="finetune"),
                    suite.Job(job_id="g__pmnist__seed0__finetune", priority="P0",
                              purpose="t", runner="run_task_il.py", benchmark="pmnist",
                              seed=0, methods="finetune")]
            first = self._run_with_fingerprint(run_dir, jobs, data_root)
            self.assertEqual({o["state"] for o in first}, {"succeeded"})
            # Stage MNIST afterwards: only the Permuted-MNIST job reruns.
            (data_root / "MNIST" / "raw").mkdir(parents=True)
            (data_root / "MNIST" / "raw" / "train-images-idx3-ubyte").write_bytes(b"m")
            second = {o["job_id"]: o["state"] for o in
                      self._run_with_fingerprint(run_dir, jobs, data_root)}
            self.assertEqual(second["g__cifar__seed0__finetune"], "skipped")
            self.assertEqual(second["g__pmnist__seed0__finetune"], "succeeded")
            # A code change without acknowledgement reruns everything; with
            # the old hash acknowledged, finished jobs are kept.
            third = {o["job_id"]: o["state"] for o in self._run_with_fingerprint(
                run_dir, jobs, data_root, execution_sha="sha2", sources={"sha", "sha2"})}
            self.assertEqual(set(third.values()), {"skipped"})
            fourth = {o["job_id"]: o["state"] for o in self._run_with_fingerprint(
                run_dir, jobs, data_root, execution_sha="sha3")}
            self.assertEqual(set(fourth.values()), {"succeeded"})

    def test_artifacts_on_another_volume_stay_inside_the_run(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            elsewhere = pathlib.Path(tmp) / "big_disk"
            run_dir.mkdir()
            elsewhere.mkdir()
            os.symlink(elsewhere, run_dir / "artifacts")
            jobs = _fake_jobs()[:1]
            self.assertEqual([o["state"] for o in self._run(run_dir, jobs)], ["succeeded"])
            status = json.loads((run_dir / "statuses" / f"{jobs[0].job_id}.json").read_text())
            self.assertEqual(status["outputs"][0]["path_relative_to_run_dir"],
                             f"artifacts/{jobs[0].job_id}/result.json")
            self.assertEqual([o["state"] for o in self._run(run_dir, jobs)], ["skipped"])

    def test_a_changed_pip_freeze_keeps_the_earlier_record(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp)
            (run_dir / "pip_freeze.txt").write_text("numpy==0.0.1\n")
            suite.write_pip_freeze(run_dir)
            digest = hashlib.sha256(b"numpy==0.0.1\n").hexdigest()
            self.assertEqual((run_dir / "pip_freezes" / f"{digest}.txt").read_text(),
                             "numpy==0.0.1\n")
            self.assertNotEqual((run_dir / "pip_freeze.txt").read_text(), "numpy==0.0.1\n")

    def test_gpu_probe_selects_the_first_visible_device(self):
        import os
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            self.assertEqual(suite.visible_gpu_selector(), [])
            os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
            self.assertEqual(suite.visible_gpu_selector(), ["-i", "2"])
            os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-8f1e"
            self.assertEqual(suite.visible_gpu_selector(), ["-i", "GPU-8f1e"])
            for hidden in ("", "-1"):
                os.environ["CUDA_VISIBLE_DEVICES"] = hidden
                self.assertIsNone(suite.visible_gpu_selector())
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

    def test_a_second_suite_in_the_same_run_directory_stops(self):
        import errno
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            data_root = pathlib.Path(tmp) / "data"
            run_dir.mkdir()
            data_root.mkdir()
            args = ["--profile", "smoke", "--run-dir", str(run_dir), "--max-parallel", "1"]
            self.assertTrue(suite.lock_run_dir(run_dir))     # a suite is running here
            try:
                with self.assertRaises(SystemExit) as stopped:
                    _prepare(args, data_root)
                self.assertIn("already running", str(stopped.exception))
                # It stopped before writing anything...
                self.assertEqual([p.name for p in run_dir.iterdir()], [".run_suite.lock"])
                # ...and a dry run, which only reads the directory, is not blocked.
                self.assertEqual(suite.main([*args, "--dry-run", "--dataset-hash", "none",
                                             "--data-root", str(data_root)]), 0)
            finally:
                suite.release_run_dir_locks()
            self.assertEqual(_prepare(args, data_root), 0)   # released: the run starts
            self.assertTrue((run_dir / "plan.json").is_file())
            self.assertTrue(suite.lock_run_dir(run_dir))     # main released its own lock
            suite.release_run_dir_locks()
            # A filesystem without lock support warns and runs unlocked.
            with mock.patch("fcntl.flock", side_effect=OSError(errno.ENOLCK, "no locks")):
                self.assertFalse(suite.lock_run_dir(run_dir))
                self.assertEqual(_prepare(args, data_root), 0)

    def test_p0_and_p1_stop_when_pytorch_sees_no_gpu(self):
        stubs = {"execute_jobs": lambda *a, **k: [],
                 "environment_probe": lambda: {"torch_probe": {"ok": True,
                                                               "cuda_available": False}},
                 "write_pip_freeze": lambda run_dir: None}
        saved = {name: getattr(suite, name) for name in stubs}
        try:
            for name, stub in stubs.items():
                setattr(suite, name, stub)
            with tempfile.TemporaryDirectory() as tmp:
                data_root = pathlib.Path(tmp) / "data"
                data_root.mkdir()

                def run(profile, *extra):
                    run_dir = pathlib.Path(tmp) / f"{profile}{len(extra)}"
                    return suite.main(["--profile", profile, "--run-dir", str(run_dir),
                                       "--max-parallel", "1", "--skip-preflight",
                                       "--skip-verify", "--dataset-hash", "none",
                                       "--data-root", str(data_root), *extra])

                with self.assertRaises(SystemExit) as stopped:
                    run("p1")
                self.assertIn("no CUDA device", str(stopped.exception))
                self.assertEqual(run("p1", "--allow-cpu"), 0)    # a deliberate CPU run
                self.assertEqual(run("smoke"), 0)                # smoke checks run anywhere
        finally:
            for name, value in saved.items():
                setattr(suite, name, value)

    def test_calibrated_runs_stop_when_float64_routing_fails_its_self_test(self):
        failed = {"ok": False, "device": "cuda", "decision_mismatches": 3}
        stubs = {"execute_jobs": lambda *a, **k: [],
                 "environment_probe": lambda: {"torch_probe": {
                     "ok": True, "cuda_available": True, "float64_routing": failed}},
                 "write_pip_freeze": lambda run_dir: None}
        saved = {name: getattr(suite, name) for name in stubs}
        try:
            for name, stub in stubs.items():
                setattr(suite, name, stub)
            with tempfile.TemporaryDirectory() as tmp:
                data_root = pathlib.Path(tmp) / "data"
                data_root.mkdir()

                def run(*groups):
                    run_dir = pathlib.Path(tmp) / "_".join(groups)
                    return suite.main(["--profile", "all", "--only-groups", *groups,
                                       "--seeds", "0", "--run-dir", str(run_dir),
                                       "--max-parallel", "1", "--skip-preflight",
                                       "--skip-verify", "--dataset-hash", "none",
                                       "--data-root", str(data_root)])

                for group in ("p0_class_entropy_cal", "p1_class_entropy_biasacc"):
                    with self.assertRaises(SystemExit) as stopped:
                        run(group)
                    self.assertIn("float64 routing", str(stopped.exception))
                    self.assertIn("no job was started", str(stopped.exception))
                # Runs without fitted routers do not depend on it.
                self.assertEqual(run("p0_class_entropy"), 0)
                # float32 application that differs from NumPy's only warns: it
                # matters for rebuilding float32-routed results, not for this run.
                import contextlib
                import io
                probe = {"ok": True, "device": "cuda", "float32_ok": False}
                suite.environment_probe = lambda: {"torch_probe": {
                    "ok": True, "cuda_available": True, "float64_routing": probe}}
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(run("p1_class_entropy_bias"), 0)
                self.assertIn("[routing] WARNING", stderr.getvalue())
        finally:
            for name, value in saved.items():
                setattr(suite, name, value)

    def test_numerics_environment_is_recorded_and_announced(self):
        import contextlib
        import io
        import os
        saved = {key: os.environ.pop(key, None) for key in suite.preflight.NUMERICS_ENVIRONMENT}
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = pathlib.Path(tmp)
                job = _fake_jobs()[:1]
                self._run(run_dir, job)
                status = json.loads((run_dir / "statuses" / f"{job[0].job_id}.json").read_text())
                controls = status["child_environment_controls"]
                self.assertEqual(controls["NVIDIA_TF32_OVERRIDE"], "0")
                for key in suite.preflight.NUMERICS_ENVIRONMENT:
                    self.assertIn(key, controls)
                self.assertIsNone(controls["CUBLAS_EMULATE_DOUBLE_PRECISION"])
                data_root = pathlib.Path(tmp) / "data"
                data_root.mkdir()
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(_prepare(["--profile", "smoke", "--run-dir",
                                               str(pathlib.Path(tmp) / "run"),
                                               "--max-parallel", "1"], data_root), 0)
                self.assertIn("[numerics] WARNING", stderr.getvalue())
                self.assertIn("NVIDIA_TF32_OVERRIDE=0", stderr.getvalue())
        finally:
            os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value

    def test_the_environment_probe_runs_the_float64_routing_self_test(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("needs PyTorch")
        environment = suite.environment_probe()
        probe = environment["torch_probe"]
        self.assertTrue(probe["ok"], probe)
        routing = probe["float64_routing"]
        self.assertTrue(routing["ok"], routing)
        self.assertEqual(routing["device"], "cuda" if probe["cuda_available"] else "cpu")
        for key in suite.preflight.NUMERICS_ENVIRONMENT:
            self.assertIn(key, environment["environment_controls"])

    def test_main_acknowledges_a_source_change_only_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            data_root = pathlib.Path(tmp) / "data"
            data_root.mkdir()
            args = ["--profile", "smoke", "--run-dir", str(run_dir), "--max-parallel", "1"]
            self.assertEqual(_prepare(args, data_root), 0)
            plan_path = run_dir / "plan.json"
            plan = json.loads(plan_path.read_text())
            current = plan["execution_source_sha256"]
            plan["execution_source_sha256"] = "0" * 64      # pretend older code
            plan_path.write_text(json.dumps(plan))
            dry = args + ["--dry-run", "--dataset-hash", "none", "--data-root", str(data_root)]
            for attempt in (lambda: _prepare(args, data_root), lambda: suite.main(dry)):
                with self.assertRaises(SystemExit) as ctx:
                    attempt()
                self.assertIn("--accept-source-change", str(ctx.exception))
            # A dry run previews the acknowledgement without recording it.
            before = plan_path.read_text()
            self.assertEqual(suite.main(dry + ["--accept-source-change", "validation fix"]), 0)
            self.assertEqual(plan_path.read_text(), before)
            self.assertEqual(_prepare(args + ["--accept-source-change", "validation fix"],
                                      data_root), 0)
            plan = json.loads(plan_path.read_text())
            self.assertEqual(plan["execution_source_sha256"], current)
            self.assertEqual(plan["source_change_acknowledgements"][0]["from"], "0" * 64)
            self.assertEqual(plan["source_change_acknowledgements"][0]["reason"],
                             "validation fix")
            # A genuinely different plan is still refused.
            with self.assertRaises(SystemExit) as ctx:
                _prepare(args + ["--seeds", "0,1", "--accept-source-change", "x"], data_root)
            self.assertIn("different plan", str(ctx.exception))

    def test_invalidated_methods_rerun_only_their_jobs(self):
        """Executor-level check of the v3.0.2 mechanism: after an acknowledged
        change that invalidates supsup/wsn, only those jobs rerun."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            run_dir.mkdir()
            jobs = [suite.Job(job_id=f"g__cifar__seed0__{m}", priority="P0", purpose="t",
                              runner="run_task_il.py", benchmark="cifar", seed=0, methods=m)
                    for m in ("finetune", "supsup", "wsn", "diltcl")]
            first = self._run(run_dir, jobs)
            self.assertEqual({o["state"] for o in first}, {"succeeded"})
            plan = {"execution_source_sha256": "sha2",
                    "source_change_acknowledgements": [
                        {"from": "sha", "to": "sha2", "reason": "STE fix",
                         "invalidated_methods": ["supsup", "wsn"]}]}

            def fake_command(job, run_dir_, data_root, allow_download,
                             save_checkpoints=True, execution_args=()):
                prefix = run_dir_ / "artifacts" / job.job_id / "result"
                return ([sys.executable, "-c", _CHILD, str(prefix), "0", "0.01"], prefix)

            original = suite.command_for
            suite.command_for = fake_command
            try:
                second = suite.execute_jobs(
                    jobs, run_dir=run_dir, data_root=None, allow_download=False,
                    save_checkpoints=False, force=False, execution_source_sha256="sha2",
                    dataset_identity={"mode": "none"}, execution_args=(),
                    max_parallel=2, threads_per_job=1, job_memory_mib=1,
                    stop_on_error=False, stream_logs=False, launch_stagger_seconds=0.0,
                    accepted_sources=suite.accepted_source_hashes(plan), plan=plan)
            finally:
                suite.command_for = original
            states = {o["job_id"].split("__")[-1]: o["state"] for o in second}
            self.assertEqual(states, {"finetune": "skipped", "diltcl": "skipped",
                                      "supsup": "succeeded", "wsn": "succeeded"})
            for m in ("supsup", "wsn"):
                status = json.loads((run_dir / "statuses" / f"g__cifar__seed0__{m}.json").read_text())
                self.assertEqual(status["execution_source_sha256"], "sha2")

    def test_main_records_invalidated_methods_in_the_acknowledgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            data_root = pathlib.Path(tmp) / "data"
            data_root.mkdir()
            args = ["--profile", "smoke", "--run-dir", str(run_dir), "--max-parallel", "1"]
            self.assertEqual(_prepare(args, data_root), 0)
            plan_path = run_dir / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan["execution_source_sha256"] = "1" * 64
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(SystemExit) as ctx:
                _prepare(args + ["--invalidate-methods", "supsup"], data_root)
            self.assertIn("--accept-source-change", str(ctx.exception))
            self.assertEqual(_prepare(args + ["--accept-source-change", "STE fix",
                                              "--invalidate-methods", "supsup, wsn"],
                                      data_root), 0)
            record = json.loads(plan_path.read_text())["source_change_acknowledgements"][0]
            self.assertEqual(record["invalidated_methods"], ["supsup", "wsn"])
            self.assertEqual(record["from"], "1" * 64)

    def test_methods_named_after_the_acknowledgement_are_added_not_dropped(self):
        # v3.0.6 recorded nothing when the change was already acknowledged
        # (for instance by an earlier invocation that forgot the flag).
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            data_root = pathlib.Path(tmp) / "data"
            data_root.mkdir()
            args = ["--profile", "smoke", "--run-dir", str(run_dir), "--max-parallel", "1"]
            self.assertEqual(_prepare(args, data_root), 0)
            plan_path = run_dir / "plan.json"
            # No acknowledged change into the current code: nothing to invalidate.
            with self.assertRaises(SystemExit) as ctx:
                _prepare(args + ["--accept-source-change", "x", "--invalidate-methods", "wsn"],
                         data_root)
            self.assertIn("nothing to invalidate", str(ctx.exception))
            plan = json.loads(plan_path.read_text())
            plan["execution_source_sha256"] = "2" * 64
            plan_path.write_text(json.dumps(plan))
            self.assertEqual(_prepare(args + ["--accept-source-change", "STE fix"], data_root), 0)
            self.assertEqual(_prepare(args + ["--accept-source-change", "STE fix",
                                              "--invalidate-methods", "supsup,wsn"],
                                      data_root), 0)
            plan = json.loads(plan_path.read_text())
            [record] = plan["source_change_acknowledgements"]
            self.assertEqual(record["invalidated_methods"], ["supsup", "wsn"])
            self.assertEqual(record["amendments"][0]["invalidated_methods_added"],
                             ["supsup", "wsn"])
            reason = suite.result_invalidated(plan, "2" * 64, ["supsup"])
            self.assertIsNotNone(reason)
            self.assertIsNone(suite.result_invalidated(plan, "2" * 64, ["finetune"]))

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            data_root = pathlib.Path(tmp) / "data"
            data_root.mkdir()
            args = ["--profile", "smoke", "--run-dir", str(run_dir), "--max-parallel", "1"]
            dry = args + ["--dry-run", "--dataset-hash", "none", "--data-root", str(data_root)]
            self.assertEqual(suite.main(dry), 0)
            self.assertFalse(run_dir.exists())
            self.assertEqual(_prepare(args, data_root), 0)
            before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
            self.assertEqual(suite.main(dry), 0)
            after = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
            self.assertEqual(before, after)

    def test_only_groups_filters_the_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "run"
            data_root = pathlib.Path(tmp) / "data"
            data_root.mkdir()
            base = ["--profile", "p1", "--seeds", "0", "--head-mode", "multi",
                    "--run-dir", str(run_dir), "--max-parallel", "1"]
            self.assertEqual(_prepare(base + ["--only-groups", "p1_class_entropy_bias"],
                                      data_root), 0)
            plan = json.loads((run_dir / "plan.json").read_text())
            ids = [r["job"]["job_id"] for r in plan["jobs"]]
            self.assertEqual(len(ids), 8)
            self.assertTrue(all(i.startswith("p1_class_entropy_bias__") for i in ids))
            with self.assertRaises(SystemExit) as ctx:
                suite.main(base + ["--only-groups", "nope", "--force-plan", "--dry-run",
                                   "--data-root", str(data_root)])
            self.assertIn("unknown groups", str(ctx.exception))

    def test_dispatch_order_is_longest_first(self):
        ordered = sorted(_fake_jobs(), key=suite.estimate_cost, reverse=True)
        self.assertEqual(ordered[0].methods, "diltcl")
        self.assertEqual(ordered[1].methods, "lwf")

    def test_auto_parallelism_never_drops_to_one_job_silently(self):
        """When nvidia-smi exists but its query fails, PyTorch's reading is
        used; when that fails too the fallback to one job is announced."""
        saved = (suite.gpu_memory_mib, suite.gpu_memory_from_torch,
                 suite.shutil.which, suite.usable_cpu_count)
        try:
            suite.usable_cpu_count = lambda: 24
            suite.shutil.which = lambda name: "/usr/bin/nvidia-smi"
            suite.gpu_memory_mib = lambda: None
            suite.gpu_memory_from_torch = lambda: (500, 32000)
            value, detail = suite.resolve_max_parallel("auto", 3, 2048)
            self.assertEqual(value, min(8, suite.MAX_AUTO_PARALLEL))
            self.assertEqual(detail["gpu_memory_source"], "torch")
            self.assertNotIn("warning", detail)
            suite.gpu_memory_from_torch = lambda: None
            value, detail = suite.resolve_max_parallel("auto", 3, 2048)
            self.assertEqual(value, 1)
            self.assertIn("--max-parallel", detail["warning"])
            # An explicit value never probes and never warns.
            suite.gpu_memory_from_torch = lambda: self.fail("probed for an explicit value")
            value, detail = suite.resolve_max_parallel("6", 3, 2048)
            self.assertEqual((value, detail["rule"]), (6, "explicit"))
            # Without a driver (a CPU machine) the CPU count decides, quietly.
            suite.shutil.which = lambda name: None
            suite.gpu_memory_from_torch = lambda: self.fail("probed without a driver")
            value, detail = suite.resolve_max_parallel("auto", 3, 2048)
            self.assertEqual(value, min(8, suite.MAX_AUTO_PARALLEL))
            self.assertNotIn("warning", detail)
        finally:
            (suite.gpu_memory_mib, suite.gpu_memory_from_torch,
             suite.shutil.which, suite.usable_cpu_count) = saved

    def test_disk_estimate_of_a_resumed_run_counts_only_unfinished_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp)
            jobs = _fake_jobs()
            self._run(run_dir, jobs[:2], max_parallel=1)
            pending = suite.jobs_still_to_run(run_dir, jobs)
            self.assertEqual([j.job_id for j in pending], [j.job_id for j in jobs[2:]])
            # An output that went missing makes its job count again.
            status = json.loads((run_dir / "statuses" / f"{jobs[0].job_id}.json").read_text())
            output = status["outputs"][0]
            (run_dir / output["path_relative_to_run_dir"]).unlink()
            pending = suite.jobs_still_to_run(run_dir, jobs)
            self.assertIn(jobs[0].job_id, [j.job_id for j in pending])


if __name__ == "__main__":
    unittest.main()
