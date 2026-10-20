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

## Quick check (CPU, about a minute)

```bash
cd code
pip install -r requirements.txt pytest scikit-learn
python -m pytest tests -q                 # 268 tests; the opt-in end-to-end test (DILTCL_E2E=1, ~6 min) is skipped

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

Code version 3.0.5 (`code/diltcl/__init__.py`). The artifacts in `evidence/`
were produced by versions 3.0.0 to 3.0.3 on one NVIDIA RTX 5090; the exact
software stack is pinned in `code/requirements-lock.txt` and recorded in each
run folder. Versions 3.0.4 and 3.0.5 corrected bookkeeping (compute
accounting, parameter counts, a SciPy fallback that the recorded runs did not
use) and added per-example prediction recording, checkpoint re-evaluation, a
regeneration audit and a release builder. These additions are covered by the
test suite and by a synthetic end-to-end run. Whether 3.0.5 regenerates the
recorded results bit for bit on the recorded stack is what the Level 3
regeneration audit in `code/REPRODUCE.md` checks; that audit has not been run
yet.

## License

MIT; see [`LICENSE`](LICENSE).
