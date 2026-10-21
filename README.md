# DiLT-CL

Research code and verified result artifacts for a diagnostic study of
parameter-isolation continual learning.

Parameter-isolation methods (PackNet, SupSup, WSN and DiLT-CL, *disentangled
lottery-ticket continual learning*) train one sparse expert per task inside a
shared network and leave the experts of earlier tasks unchanged. When the task
of a test example is known (task-incremental learning), the example is
answered by the expert of that task. When it is not (class-incremental
learning), an expert has to be chosen first. This repository measures how
much class-incremental accuracy is lost in that routing step rather than in
the experts:

* routed accuracy `A` factorises as `A = U * V`, the task-inference accuracy
  `U` times the accuracy `V` on the examples that were routed correctly;
* the oracle accuracy `O` answers every example with the expert of its true
  task, with the experts unchanged and the class label never used;
* `rho = (O - A) / (1 - A)` is the fraction of the routed error that the
  existing experts would recover under correct routing.

The comparison covers shared-backbone baselines (fine-tuning, EWC, LwF, ER,
DER++) and the isolation methods on Split-CIFAR-100, Permuted-MNIST and
Split-TinyImageNet, five seeds each, with uncalibrated routing rules and three
routers fitted on held-out training data.

The accompanying paper is under review; a citation will be added here.

## Repository layout

```
code/                        the diltcl package, runners, tools and tests
  README.md                  code overview, installation, guarantees
  PROTOCOL.md                protocols and metric definitions
  REPRODUCE.md               verification and reproduction, three levels
evidence/results_20260906/   merged, verified artifacts of the three runs behind the paper
  MANIFEST.json              SHA-256 of every file in the folder
  p0_v3/                     330 jobs: task-IL (multi-head) and class-IL (raw and calibrated routing)
  p1_bias/                   40 jobs: class-IL with the tied-slope score-bias router
  p1_biasacc/                40 jobs: class-IL with the accuracy-fitted score-bias router
```

Each run folder holds the merged artifacts (`merged/*.json|npz`: per-seed
metrics and matrices, aggregates, paired tests), the independent
verification report (`verification_report.json`), what ran and how long
(`plan.json`, `statuses/`, `suite_summary.json`), and the provenance of code,
environment and data (`source_manifest*.json`, `environment*.json`,
`pip_freeze.txt`, `dataset_fingerprint*.json`, per-file lists removed).
The fingerprint used by the 170 P0 jobs launched on 2026-09-04
(`p0_v3/dataset_fingerprints/6f307858*.json`) predates per-entry hashes;
they were added on 2026-10-20, computed from its archived per-file list
(`entries_derived_from_files`), so those jobs' data can be compared
benchmark by benchmark.
`p0_v3/paper/wall_minutes.txt` was written before SupSup and WSN were rerun
under v3.0.2, so its rows for those two methods time the invalidated runs;
`code/tools/wall_times.py` recomputes the summary from `statuses/`.

## Quick check (CPU, a few minutes)

```bash
cd code
pip install -r requirements.txt pytest matplotlib
python -m pytest tests -q                 # 349 tests, 348 run (about 2 min); the opt-in end-to-end test (DILTCL_E2E=1, ~6 min) is skipped

# every evidence file against its recorded hash
python - <<'EOF'
import hashlib, json, pathlib
root = pathlib.Path("../evidence/results_20260906")
m = json.loads((root / "MANIFEST.json").read_text())
bad = [f["path"] for f in m["files"]
       if hashlib.sha256((root / f["path"]).read_bytes()).hexdigest() != f["sha256"]]
print(len(m["files"]), "files,", len(bad), "mismatches")
EOF

# the paper's paired statistics (CIs, paired t with Holm, exact sign-flip, MDD)
E=../evidence/results_20260906
python tools/paired_stats.py --task-il $E/p0_v3/merged/p0_task_il_multi.json \
    --class-il $E/p0_v3/merged/p0_class_entropy.json \
    --calibrated $E/p0_v3/merged/p0_class_entropy_cal.json \
    --calibrated $E/p1_bias/merged/p1_class_entropy_bias.json \
    --calibrated $E/p1_biasacc/merged/p1_class_entropy_biasacc.json \
    --out-csv paired.csv --out-tex paired_full.tex --out-compact-tex paired_main.tex \
    --out-calibration-tex paired_calibration.tex
```

Full verification from the per-job results, re-evaluation of the trained
experts from their checkpoints, and retraining from the seeds are described
in [`code/REPRODUCE.md`](code/REPRODUCE.md).

## What is not in this repository

* **Datasets.** CIFAR-100 and MNIST are downloaded by torchvision;
  TinyImageNet is staged by hand (`code/README.md`).
* **Checkpoints and per-job results.** The complete run directories (about
  10 GiB, mostly the 410 persisted expert states) are packaged as release
  archives with `code/tools/build_release.py`, whose manifest records the
  SHA-256 of every archive and of every file inside. They will be linked here
  when they are published.

## Status

Code version 3.0.11 (`code/diltcl/__init__.py`). The artifacts in `evidence/`
were produced by versions 3.0.0 to 3.0.3 on one NVIDIA RTX 5090; the exact
software stack is pinned in `code/requirements-lock.txt` and recorded in each
run folder. Versions 3.0.4 and 3.0.5 corrected bookkeeping (compute
accounting, parameter counts, a SciPy fallback that the recorded runs did not
use) and added per-example prediction recording, checkpoint re-evaluation, a
regeneration audit and a release builder. Versions 3.0.6 to 3.0.9 changed no
training or evaluation arithmetic. Version 3.0.6 made the audit tools fail on
incomplete or altered evidence instead of reporting success; 3.0.7 closed gaps
in resuming runs, in data identity (dataset folders linked into the data root,
the decoded-image cache) and in the inputs of the audit tools; 3.0.8 fixed two
tests that failed on any machine with a GPU and made a second suite launcher
stop on a run directory that is in use; 3.0.9 fixes TinyImageNet loading from a
relative data root, stops P0/P1 suites when PyTorch sees no GPU (unless
`--allow-cpu`), keeps the other jobs' results when one checkpoint
re-evaluation fails, and runs every router kind through the audit tests.
Version 3.0.10 applies every fitted router in float64, the precision it is
fitted in, instead of the float32 of the routing scores, so the deployed
router is the fitted one; training and uncalibrated routing are unchanged,
and `code/tools/routing_precision_audit.py` measures which calibrated
decisions the change moves. It also closes the audit gaps and minor issues
found by the fourth and fifth code reviews (`code/PROTOCOL.md`). Version
3.0.11 changes no training, evaluation, routing or recording code: a run
whose fitted routers differ from the original ones (routers are fitted on
the CPU) is reported as such -- as a refit when only the coefficients of a
logistic or tied-slope router differ -- with a check whether the original
routers reproduce the original results, instead of being attributed to the
routing precision; tests that could fail on a GPU server without a fault in
the code no longer do; and the suite's environment probes have a time
limit. The test suite and a
synthetic end-to-end run pass on Python 3.13 and on the recorded software
stack (Python 3.12.3, PyTorch 2.12.1) on a CPU. Whether 3.0.11 regenerates the
recorded experts bit for bit on the recorded stack is what the Level 3
regeneration audit in `code/REPRODUCE.md` checks; that audit has not been run
yet.

## License

MIT; see [`LICENSE`](LICENSE).
