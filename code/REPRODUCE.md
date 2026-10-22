# Reproducing the results

This guide covers three levels of checking, from minutes on a laptop CPU to a
complete retraining. Every command runs from this directory (`code/`).

| Level | What it establishes | Needs | Time |
|---|---|---|---|
| 1. Verify the released artifacts | Every reported number follows from the retained matrices; every checkpoint and per-example prediction file is the one recorded by its job; every routing and oracle statistic follows from the per-example decisions, which the released runs (the v3.0.11 regeneration) recorded and the original v3.0.0 to v3.0.3 runs did not | CPU, Python, the release | minutes |
| 2. Re-evaluate the released checkpoints | The released experts regenerate every routed, task-inference, conditional and oracle result | the datasets; a GPU of the recorded type for exact equality | ~1 h on one GPU |
| 3. Retrain from the seeds | The pipeline regenerates the experts themselves (bit for bit on the recorded stack); against the original runs, the calibrated routers' float32-to-float64 change is measured decision by decision | the datasets and an NVIDIA GPU | ~10 GPU-hours per five seeds of all groups |

## Environment

The recorded stack of every run is pinned in `requirements-lock.txt`
(Python 3.12; PyTorch 2.12.1 with CUDA 13.0 wheels, pinned by URL and SHA-256;
NumPy 2.4.6; SciPy 1.18.1; scikit-learn 1.9.0). Each run directory also keeps
its own `pip_freeze.txt` and `environment.json` (GPU model, driver, CUDA and
cuDNN versions).

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt
python -m pytest tests -q -p no:cacheprovider       # ~2 min on a CPU
python tools/check_environment.py --recorded-environment RUNS/p0/environment.json
```

`check_environment.py` lists every installed package whose version differs
from the lock and compares Python, PyTorch, CUDA, cuDNN, GPU model and driver
with a run's recorded `environment.json`.

or, with Docker and the NVIDIA Container Toolkit:

```bash
docker build -t diltcl:3.0.11 .
docker run --gpus all --rm -it -v /path/to/data:/data -v /path/to/runs:/runs diltcl:3.0.11
```

Levels 1 and 2 also work with any recent PyTorch/NumPy/SciPy; only exact
bit-for-bit equality in levels 2 and 3 depends on the pinned stack and GPU.

## Data

`--data-root` (or `$DILTCL_DATA_ROOT`) must contain CIFAR-100
(`cifar-100-python/`, as downloaded by torchvision), MNIST, and
TinyImageNet (`tiny-imagenet-200/train`, `val/images`, `val/val_annotations.txt`).
`python tools/preflight_check.py --data-root DATA --benchmarks cifar,pmnist,tinyimagenet`
lists anything missing. Each run directory records a content fingerprint of
the data it used (`dataset_fingerprint.json`, a SHA-256 hash per top-level
data entry); a resumed run reruns every job whose benchmark entries changed,
and the fingerprint of a new run can be compared with the released one.

## Unpacking a release

The release consists of plain tar archives per run and job group, named
`<run>__meta.tar` and `<run>__<group>[__<benchmark>].tar`, plus
`RELEASE_MANIFEST.json` (SHA-256 of every archive and every file inside).
Extract all archives of one run into one directory; the result is a complete
suite run directory (the commands below assume the runs are called `p0`,
`p1_bias` and `p1_biasacc`; use the run names in the manifest):

```bash
RELEASE=/path/to/release            # the folder holding the downloaded archives
for r in p0 p1_bias p1_biasacc; do
  mkdir -p RUNS/$r && for f in "$RELEASE"/${r}__*.tar; do tar -xf "$f" -C RUNS/$r; done
done
```

The release holds the v3.0.11 regeneration of all 410 jobs (2026-10-21/22),
whose merged artifacts, per-job results and audit reports are also in
`../evidence/results_20261022/`. The `audit/` folders hold the reports of
that regeneration: `p0/audit/` the regeneration audit and the
routing-precision audit of all three runs against the original runs, and
each run's `audit/` its per-example recomputation and checkpoint
re-evaluation.

## Level 1: verify the released artifacts (CPU)

```bash
for r in p0 p1_bias p1_biasacc; do
  mkdir -p check_$r
  python tools/verify_experiment_bundle.py RUNS/$r --out check_$r/verification.json
  python tools/recompute_from_predictions.py --run-dir RUNS/$r --refit --out check_$r/predictions.json
  python tools/merge_results.py RUNS/$r --out-dir check_$r/remerged
done
python tools/compare_runs.py --reference RUNS/p0 RUNS/p1_bias RUNS/p1_biasacc \
    --candidate check_p0/remerged check_p1_bias/remerged check_p1_biasacc/remerged
```

(`--out` keeps the released `verification_report.json` of each run
unchanged.) The released runs recorded per-example predictions for every
class-IL isolation job (80, 40 and 40 files). The original runs, produced by
v3.0.0 to v3.0.3, recorded none, so on their run directories
`recompute_from_predictions.py` prints `not-recorded` and checks nothing.

* `verify_experiment_bundle.py` checks that every planned job succeeded and
  produced exactly its planned methods, recomputes every summary metric from
  its matrix, checks each persisted checkpoint's byte count and SHA-256 and
  every output's hash against the job record, and, for class-IL isolation jobs
  that recorded per-example predictions, rebuilds every routing matrix and
  pooled statistic from them exactly (the checks of
  `recompute_from_predictions.py` below, except the re-fit). Shared-backbone
  methods have no routing step and record none.
* `recompute_from_predictions.py` first checks each predictions file's
  structure (shapes, ranges, finite scores, no decision before a task
  arrived, per-task counts equal to the recorded test-set sizes, the matched
  raw pass on the same examples and scores, every example routed to its own
  task carrying its own expert's prediction, raw and calibrated routers that
  chose the same task agreeing, and the stored router-fitting rows matching
  the held-out split task by task), then rebuilds every matrix and
  statistic exactly, checks that every raw-router decision is the lowest-index
  argmin of the recorded scores, and re-applies the stored calibrated routers
  to the recorded scores in the precision the job applied them in (float64
  since v3.0.10, float32 before): exactly for the score-bias routers; for the
  score-logreg router only near-ties may differ, they are counted, and a
  recorded tie needs a higher task within rounding of the recorded route. It
  also recomputes every stored router's fitting accuracies (plain and
  class-balanced) and checks, comparing exact fractions, that score-bias-acc
  never fitted below the raw rule. It prints the decomposition per job (routed accuracy `A`,
  task-inference accuracy `U`, conditional accuracy `V`, oracle accuracy `O`,
  recoverable fraction `rho = (O - A) / (1 - A)`; `A = U V` exactly; disjoint
  label sets only) and, with `--refit`, how far every parameter of the routers
  re-fitted from the stored calibration inputs moved and how often they choose
  the same task (reported, not enforced: solver versions can move
  coefficients). Every other check is enforced.
* Re-merging and `compare_runs.py` show that the merged artifacts behind the
  paper's tables follow from the per-job results. A merged artifact is marked
  `verified` only if the files it was built from are byte-identical to the
  ones the verifier passed.

The paper's paired statistics are regenerated by

```bash
python tools/paired_stats.py --task-il RUNS/p0/merged/p0_task_il_multi.json \
    --class-il RUNS/p0/merged/p0_class_entropy.json \
    --calibrated RUNS/p0/merged/p0_class_entropy_cal.json \
    --calibrated RUNS/p1_bias/merged/p1_class_entropy_bias.json \
    --calibrated RUNS/p1_biasacc/merged/p1_class_entropy_biasacc.json \
    --out-csv paired.csv --out-tex paired_full.tex --out-compact-tex paired_main.tex \
    --out-calibration-tex paired_calibration.tex
```

## Level 2: re-evaluate the released checkpoints (no training)

```bash
for r in p0 p1_bias p1_biasacc; do
  python tools/evaluate_checkpoints.py --run-dir RUNS/$r --data-root DATA --out check_$r/eval.json
done
```

For every class-IL isolation job (PackNet, SupSup, WSN, DiLT-CL) the tool
loads the persisted state with `torch.load(..., weights_only=True)`, checks the
file and tensor-content SHA-256 against the job record, rebuilds the model
from the configuration and execution options stored in the artifact, re-runs
the class-IL evaluation with the job's own router (and the matched raw router
for calibrated jobs), and compares every matrix and every per-seed metric,
including the oracle accuracy, and, when the job recorded them, every
per-example decision (the predictions file must match its recorded SHA-256).
The three runs hold 160 such checkpoints (80, 40 and 40). Each stored router
is applied in the precision recorded with it (`routing_dtype`: float64 since
v3.0.10; float32 for routers written earlier), so runs of either kind
re-evaluate as they were evaluated. On the recorded GPU
type with the pinned stack the comparison is exact: any difference, including
one per-example decision or a non-finite value, is an error, and the status
is `identical`. Elsewhere floating-point rounding can move decisions that sit
on near-ties; `--atol 0.05` then accepts matrix and metric differences up to
0.05 percentage points (status `within-tolerance`) and counts the
per-example differences.

To share one GPU between several processes, run `--shard K/N` for
`K = 0..N-1` and combine the shard reports; the combination fails unless the
shards cover every class-IL job of the run exactly once. A job whose
re-evaluation raises (missing data, out of memory) is reported as an error
and the remaining jobs are still evaluated:

```bash
for k in 0 1 2 3; do
  python tools/evaluate_checkpoints.py --run-dir RUNS/p0 --data-root DATA \
      --shard $k/4 --out check_p0/eval_$k.json &
done; wait
python tools/evaluate_checkpoints.py --combine check_p0/eval_*.json --out check_p0/eval.json
```

The oracle accuracy is the accuracy obtained when each test example is
answered by the expert of its true task (the index of the task split it
belongs to), with every expert unchanged and its own top-1 class used; the
class label is never used to choose the expert.

## Level 3: retrain from the seeds

```bash
export DILTCL_CACHE_DIR=/path/to/data_cache CUBLAS_WORKSPACE_CONFIG=:4096:8
unset NVIDIA_TF32_OVERRIDE
python run_suite.py --profile smoke --seeds 0 --run-dir NEW/smoke --data-root DATA   # 4 short jobs
python run_suite.py --profile p0 --head-mode multi --seeds 0-4 \
    --run-dir NEW/p0 --data-root DATA --pin-sklearn                     # 330 jobs
python run_suite.py --profile p1 --only-groups p1_class_entropy_bias --seeds 0-4 \
    --run-dir NEW/p1_bias --data-root DATA --pin-sklearn                # 40 jobs
python run_suite.py --profile p1 --only-groups p1_class_entropy_biasacc --seeds 0-4 \
    --run-dir NEW/p1_biasacc --data-root DATA --pin-sklearn             # 40 jobs
for r in p0 p1_bias p1_biasacc; do python tools/merge_results.py NEW/$r; done
python tools/compare_runs.py --reference RUNS/p0 RUNS/p1_bias RUNS/p1_biasacc \
    --candidate NEW/p0 NEW/p1_bias NEW/p1_biasacc --out regeneration_audit.json
```

Decoded TinyImageNet images are cached in `DILTCL_CACHE_DIR` under a key
derived from the image files (path, size, modification time), so a cache
written by an earlier code version is not reused: the images are decoded
again from the data root, and `compare_runs.py` reports whether the decoded
splits of the new run hash to the recorded values (`decoded splits` line).

The smoke profile takes minutes and repeats one task-IL job with the same
seed for every method; its verification fails unless the repeat is bit for
bit identical, so it checks the GPU's determinism before the long run. Run
the three suites one after another and nothing else on the GPU meanwhile;
`--pin-sklearn` makes the preflight require scikit-learn, the router fitting
backend of every recorded run. The original jobs ran with
`DILTCL_CACHE_DIR` on the data disk and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
(the suite sets the latter unless the shell already does). The runners fall
back to the CPU when PyTorch sees no CUDA device, so the P0 and P1 profiles
stop instead unless `--allow-cpu` is given.

Some environment variables change GPU arithmetic without showing in the
execution flags a run records: `NVIDIA_TF32_OVERRIDE`, cuBLAS floating-point
emulation (`CUBLAS_EMULATE_SINGLE_PRECISION`, `CUBLAS_EMULATE_DOUBLE_PRECISION`
and related), workspace limits and library switches that select other
matmul or convolution paths (`NUMERICS_ENVIRONMENT` in `diltcl/preflight.py`
lists the ones the suite records). The suite warns when one is set and
records them since v3.0.10 (environment and every job status), and
`tools/check_environment.py` reports one that is set (or that differs from a
run that recorded them) as a mismatch; the original runs recorded none of
them. Before starting calibrated jobs the suite also checks, on the GPU, that
float64 routing is IEEE double arithmetic: float32 scores widened exactly
(subnormals included) and the decisions of diagonal routers equal to NumPy's.
If that self-test fails, it stops before any job starts. The self-test also
reports, without stopping anything, whether the GPU applies float32 routers
as NumPy does (`float32_ok` in `environment.json`): the float32 rebuild of
`routing_precision_audit.py` relies on that for the original runs, and the
suite and the audit warn when it does not hold.

The suite runs up to eight jobs in parallel on one 32 GB GPU, resumes after an
interruption (a second launcher on a run directory that is still in use
stops at once), verifies every job, and records the environment, the code
(`source_manifest.json`) and the data fingerprint. `compare_runs.py` compares
the new run with the released one: trained-state content hashes, every NPZ
array and every per-seed metric, plus the SHA-256 of the state files, and
lists any differences in the recorded environment. Its verdict is
`identical` only when every reference group, benchmark, method and seed is
present in the new run and identical; a run that is missing anything is
`incomplete`. Given run directories on both sides, it also reports whether
every job read the same data (content hashes) and ran with the same launch
controls. The 170 P0 jobs launched under v3.0.0 recorded only a hash of the
whole data root. Their per-benchmark hashes are derived from the archived
fingerprint with that root hash: the copy in `../evidence/` carries
per-entry hashes computed from its per-file list, and `--fingerprints DIR`
names further folders to search (an original run directory or its
`dataset_fingerprints/`); without either they are reported as not
comparable. The environment comparison
lists every recorded field that differs; `platform` (the kernel string) and,
in the Docker image, the Python build differ from the recorded Anaconda
installation without affecting the numerics. Its last line reads, for a
complete bit-identical regeneration by the same code version,

```
410/410 runs identical (state files 410/410 identical); missing: 0 run(s), 0 group(s) -> identical
```

Against the original runs, which applied their fitted routers in float32,
calibrated runs whose decisions the change of precision moved are counted as
different, and the status is `differences` (see below).

The merged artifacts of the three original runs are also in this repository
(`../evidence/results_20260906/`), so the regeneration can be audited against
them without the released archives:

```bash
E=../evidence/results_20260906
python tools/compare_runs.py --reference $E/p0_v3 $E/p1_bias $E/p1_biasacc \
    --candidate NEW/p0 NEW/p1_bias NEW/p1_biasacc
python tools/routing_precision_audit.py --run-dir NEW/p0 NEW/p1_bias NEW/p1_biasacc \
    --reference $E/p0_v3 $E/p1_bias $E/p1_biasacc --out routing_precision.json
```

The original runs applied their fitted routers in float32; since v3.0.10 the
code applies them in float64, the precision they were fitted in. A regeneration
with the current code therefore trains the same experts (identical state
hashes are expected for all 410 runs) and gives identical results for every
group without a fitted router, but the three calibrated groups can differ
wherever float32 rounding changed a test decision: it merges biases or
decisions closer than float32 resolution into ties, which go to the lowest
task ID, and rounding a bias to float32 can reorder two decisions that are
that close. How many decisions move depends on how often that happens. The
120 original calibrated runs recorded no exact float32 tie in any test cell
(routing tie rate 0.000%), so few moves are expected there; on experts whose
outputs saturate it can be a large share of the decisions (593 of 1,260 on a
CPU in the saturated case of `tests/test_predictions_audit.py`). The
regeneration of 2026-10-21/22 (`../evidence/results_20261022/`, on another
RTX 5090) found 410/410 identical trained states and state files, 409/410
identical runs, and 1 of the 9,600,000 test decisions of the calibrated
jobs moved (in one score-bias run); the float32 rebuild matched the original
runs in all 120 calibrated jobs (`-> explained`).
`compare_runs.py` reports `differences` for the runs concerned and
names the change of routing precision when only the calibrated router's
arrays and metrics differ and the stored routers are the original ones
(`ROUTERS float32->float64`). The routers are fitted on the CPU: the
iterative fits of the logistic and tied-slope routers rest on BLAS results
that another host CPU can round differently, so a run with the same experts
whose routers of those kinds differ only in their coefficients and
intercepts is listed as `ROUTERS REFITTED` (with the largest change), and any
other router difference as `ROUTERS DIFFER`; a run counts as identical only
with the same routers. Any other difference for the same experts is
reported as `SAME STATE, DIFFERENT OUTPUTS`.
`routing_precision_audit.py` takes each calibrated job's own per-example
records, checks them as the verifier does, re-applies the stored routers in
float32 as the original code did, counts the decisions the precision moved,
and rebuilds the four calibrated matrices and the routing metrics. It then
requires everything the precision cannot change to be recorded in the
original runs and equal there -- per-task test-set sizes, trained-state
hash, every stored router parameter, the matched raw router's matrices and
metrics, the oracle accuracies -- the original calibrated metrics to follow
exactly from the original calibrated matrices, and the float32 rebuild to
reproduce those matrices and metrics. The only tolerance is for score-logreg
decisions with another task within float32 rounding of the winner, which
NumPy and the GPU may resolve differently because they sum in different
orders, and only for the counts such a decision can move, in the direction
it can move them: a correct route can be lost only where the rebuild routes
the example to its own task and gained only where the own task is within
rounding without winning; correct predictions move with those routes whose
own expert is right (with shared label sets, by any such decision, since
another expert may predict the right label); a tie can disappear only where
the rebuild ties and appear only on other such decisions. A job whose stored
routers differ from the original ones counts as different, reported as a
refit or as a router difference as above; the audit then also applies the
original routers, in float32, to the job's recorded scores and reports
whether that reproduces the original run (`with_reference_routers`, printed
on the job's `DIFFERS` line with the largest parameter change), which
separates a difference in the routers from one in the experts' test scores. The audit must also cover the original runs: every
calibrated original run must be audited, and every isolation method of an
audited job must carry its calibration record. Its last line ends in
`-> explained` when the change of precision accounts for every difference;
a job that differs otherwise makes it `unexplained`; records that fail their
checks, a run directory without calibrated jobs, a group missing from the
reference or an original run left out make it `failed`.

Determinism: every job seeds Python, NumPy and PyTorch, sets
`cudnn.deterministic=True` and `cudnn.benchmark=False`, and keeps PyTorch's
default TF32 setting for cuDNN convolutions (recorded under `execution`). Bit
identity across runs therefore holds for the same GPU architecture and
library versions; a different GPU model or library version can change
floating-point rounding during training, and with it the trained weights.
