"""``tools/wall_times.py``: wall-time summary from final job statuses (torch-free)."""

import importlib.util
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("wall_times_under_test",
                                               ROOT / "tools" / "wall_times.py")
wt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = wt
_spec.loader.exec_module(wt)


def test_summary_uses_final_succeeded_records_only():
    with tempfile.TemporaryDirectory() as tmp:
        run = pathlib.Path(tmp)
        (run / "statuses").mkdir()

        def status(job_id, seconds, state="succeeded", repeat_of=None):
            group, benchmark, seed, method = job_id.split("__")
            record = {"state": state, "wall_seconds": seconds,
                      "job": {"job_id": job_id, "benchmark": benchmark, "methods": method,
                              "repeat_of": repeat_of}}
            (run / "statuses" / f"{job_id}.json").write_text(json.dumps(record))
        status("g__cifar__seed0__wsn", 60.0)
        status("g__cifar__seed1__wsn", 180.0)
        status("g__cifar__seed2__wsn", 999.0, state="failed")
        status("g__cifar__seed3__wsn", 999.0, repeat_of="g__cifar__seed0__wsn")
        rows = wt.summary_rows(wt.wall_minutes(run))
        assert rows == [{"group": "g", "benchmark": "cifar", "method": "wsn", "n": 2,
                         "mean_minutes": 2.0, "sd_minutes": 1.414}]
