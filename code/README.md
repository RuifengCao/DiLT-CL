# DiLT-CL v3.0.7: research code

Disentangled Lottery-Ticket Continual Learning (DiLT-CL) and the baselines,
protocols and infrastructure used to evaluate it. `diltcl.__version__` is the
single source of truth for the version; every result artifact records it as
`code_version`.

The code is **correctness-gated**: no output counts as evidence until the
corrected pipeline has been run end to end on the declared hardware and the
independent verifier has passed (see `PROTOCOL.md` and `REPRODUCE.md`). The
merged results behind the accompanying paper, with their verification
records, are in `../evidence/results_20260906/`.

## What is in the package

| | |
|---|---|
| Package | `diltcl/`, 28 modules |
| Runners | `run_task_il.py` (task-IL), `run_class_il.py` (class-IL) |
| Launcher | `run_suite.py`: resumable, parallel job suite with provenance (`run_v21_experiment_suite.py` is a forwarding shim for old command lines) |
| Tools | `tools/`: preflight gate, independent verifier, result merger, GPU capacity probe, compact-storage measurement, per-example recomputation, checkpoint re-evaluation, regeneration audit, paired statistics, results package and release builders, environment check, wall-time summary |
| Tests | `tests/`, 32 files, 303 tests + 1 opt-in end-to-end test (`DILTCL_E2E=1`, runs the real suite on a synthetic CIFAR-100 and then the verifier, per-example recomputation, checkpoint re-evaluation, merger, regeneration audit, tables and figures; ~6 min on one CPU); `python -m pytest tests` runs the rest in one session (about 2 min on a CPU) |
| Metadata | `pyproject.toml` (version read from `diltcl.__version__`), `requirements.txt` (ranges), `requirements-lock.txt` (the exact recorded stack), `Dockerfile` |
| Reproduction | `REPRODUCE.md`: verify the released artifacts on a CPU, re-evaluate released checkpoints, re-run training and audit the regeneration |

## Methods and benchmarks

Methods: `finetune`, `ewc`, `lwf`, `er`, `derpp` (shared backbone);
`packnet`, `supsup`, `wsn`, `diltcl` (parameter isolation); `independent`
(one model per task, task-IL only).

Benchmarks: Split-CIFAR-100 (20×5), Permuted-MNIST (10 tasks),
Split-TinyImageNet (10×20, 64×64), 5-datasets (CIFAR-10 / MNIST / SVHN /
FashionMNIST / KMNIST). Layouts live in `diltcl/benchmarks.py`.

## Layout

```
code/
├── run_task_il.py              task-IL runner (--head-mode multi is the standard protocol)
├── run_aaai_experiments.py     forwarding shim: historical name of run_task_il.py
├── run_class_il.py             class-IL runner (entropy / confidence / margin routing; optional score-logreg or tied-slope score-bias router calibration)
├── run_suite.py                parallel, resumable launcher: profiles smoke / p0 / p1 / all
├── run_experiments.py          early exploratory experiments (Exp 1-6), kept for reference
├── make_class_il_table.py      LaTeX tables from a verified artifact
├── make_v3_figures.py          paper figures (LA/TA failure modes, calibration effect, task-IL gap) from merged artifacts
├── make_paper_figures.py       legacy: diagnostic figures from the v2.0 result files (not included); superseded by make_v3_figures.py
├── analyze_memory.py           analytic inference-state memory table
├── tools/
│   ├── preflight_check.py          dependency / dataset / disk gate (torch-free)
│   ├── verify_experiment_bundle.py independent JSON/NPZ recomputation + hash audit of a run directory
│   ├── merge_results.py            per-job artifacts -> one artifact per group; recomputes aggregates and paired tests; promotes to `verified`
│   ├── gpu_capacity_probe.py       measures per-job VRAM on the target GPU, recommends --max-parallel
│   ├── build_results_package.py    bundles merged results, verification evidence and paper materials of several runs into one package (optionally anonymised supplementary material with code)
│   ├── build_release.py            release archives of complete run directories (per-job results, checkpoints, predictions), hash-checked against the job records (v3.0.5)
│   ├── check_environment.py        installed packages / GPU / CUDA / cuDNN versus the recorded stack (v3.0.5)
│   ├── measure_compact_storage.py  compact export size with logit-equivalence check
│   ├── recompute_from_predictions.py  checks per-example predictions and rebuilds every class-IL matrix/statistic, stored router decisions and fitting accuracies from them (v3.0.5, CPU)
│   ├── evaluate_checkpoints.py     re-evaluates persisted isolation checkpoints and compares with the artifacts; --combine checks shard coverage (v3.0.5)
│   ├── compare_runs.py             regeneration audit: re-run vs retained artifacts (state and file hashes, arrays, metrics, coverage; v3.0.5)
│   ├── paired_stats.py             paired CIs, paired t, exact sign-flip, Holm, minimum detectable difference (v3.0.5)
│   ├── head_mode_digits.py         CPU-only head-protocol preview on sklearn digits
│   └── repair_independent_matrices.py  legacy (v2.0) matrix repair; not needed for v3 artifacts
├── diltcl/
│   ├── diltcl.py, masking.py, masked_layers.py     DiLT-CL, pruning primitives, edge-popup layers
│   ├── baselines.py, ewc.py, lwf.py, buffer.py, subnetwork_baselines.py, training.py
│   ├── models.py                 SimpleCNN / SimpleMLP / AdaptiveCNN / MultiHeadModel
│   ├── data.py, hard_benchmarks.py, data_backend.py, dataset_utils.py, benchmark_layout.py, benchmarks.py
│   ├── execution.py              ExecutionOptions: data backend, numerics mode, TF32, threads (recorded in artifacts)
│   ├── metrics.py, stats.py, task_inference.py, task_calibration.py, profiling.py
│   ├── compact_export.py, preflight.py, runner_common.py, config.py, utils.py
│   └── __init__.py               __version__ = "3.0.7"
└── tests/
```

## Installation

```bash
pip install -r requirements.txt        # torch, torchvision, numpy, scipy
pip install pytest pyflakes            # development
pip install scikit-learn               # optional: calibrated router (a SciPy fallback exists)
pip install matplotlib                 # figures (make_v3_figures.py)
python -m pytest tests
```

Datasets are read from `--data-root` / `$DILTCL_DATA_ROOT`; TinyImageNet must
be staged by hand (`tiny-imagenet-200/train`, `val/images`,
`val/val_annotations.txt`). `tools/preflight_check.py` lists what is missing.

## Quick start

```bash
export DILTCL_DATA_ROOT=/data DILTCL_CACHE_DIR=/data_cache   # cache must be outside the data root
# P0/P1 never download (the suite passes --no-download); only the smoke profile
# may fetch CIFAR-100/MNIST with --allow-download, unless DILTCL_DOWNLOAD=0 is set.

python tools/preflight_check.py --data-root $DILTCL_DATA_ROOT \
    --benchmarks cifar,pmnist,tinyimagenet --require-calibration \
    --output-dir /runs --expected-jobs 35

python -m pytest tests/test_data_backend.py tests/test_backend_end_to_end.py -q   # GPU bit-exactness check
python run_suite.py --profile smoke --seeds 0 --data-root $DILTCL_DATA_ROOT --allow-download --run-dir /runs/smoke
python run_suite.py --profile p0 --seeds 0-4 --head-mode multi --data-root $DILTCL_DATA_ROOT \
    --dataset-hash full --run-dir /runs/p0_v3 --max-parallel auto
python tools/verify_experiment_bundle.py /runs/p0_v3
python tools/merge_results.py /runs/p0_v3
python make_class_il_table.py --results /runs/p0_v3/merged/p0_class_entropy.json --out class_il_table.tex
python make_v3_figures.py --class-results /runs/p0_v3/merged/p0_class_entropy.json \
    --class-results-cal /runs/p0_v3/merged/p0_class_entropy_cal.json \
    --task-results /runs/p0_v3/merged/p0_task_il_multi.json --output-dir figures
```

The checking and reproduction procedure (verifying the released artifacts,
re-evaluating checkpoints, retraining) is in `REPRODUCE.md`; the protocol and
its verified semantics are in `PROTOCOL.md`.

## Guarantees the code makes (and tests)

* **Data backend is bit-identical to the legacy DataLoader pipeline.** The
  default `device` backend keeps each benchmark as a `uint8` tensor on the GPU
  and reproduces `DataLoader`'s shuffle order and global-RNG consumption
  exactly (`tests/test_data_backend.py`; end to end through training on
  synthetic on-disk CIFAR-100 and TinyImageNet trees in
  `tests/test_backend_end_to_end.py`). `--data-backend loader` is the old path.
* **Fast numerics are the same objective.** `--numerics fast` (default) uses a
  shared-feature multi-head LwF distillation, a consolidated EWC quadratic
  with the identical gradient, device-resident reservoir buffers with the
  identical RNG draws, and per-row head routing for replay. Only floating-point
  summation order differs (`tests/test_numerics_equivalence.py`).
  `--numerics legacy` reproduces the v2 arithmetic bit for bit.
* **Determinism.** cuDNN deterministic mode, `CUBLAS_WORKSPACE_CONFIG` set by
  the suite, and no atomic-add gradient paths in the new code; the smoke
  profile repeats one job with the same seed and the verifier demands
  bit-identical artifacts.
* **Splitting jobs does not change numbers.** One job per (benchmark, seed,
  method), merged back, equals one monolithic run exactly
  (`tests/test_merge_results.py`, task-IL and class-IL).
* **Provenance.** Every artifact records `execution` (backend, numerics, TF32
  policy, effective torch flags), the SHA-256 of each decoded split the task
  views index (training split; test split from v3.0.7), the dataset content
  fingerprint (suite), source hashes and checkpoint hashes.
  Artifacts start `unverified`; only `tools/merge_results.py`, after a passing
  `verify_experiment_bundle.py` report, promotes them to `verified`.
* **Resume that survives staging and fixes.** A job's dataset identity is the
  hash of the top-level data entries its benchmark reads, so staging another
  benchmark's data later does not invalidate finished jobs, while a change to
  its own entries reruns it on resume; every fingerprint ever used is archived
  under `dataset_fingerprints/`. Decoded images are cached outside the data
  root under a key derived from the image files themselves (path, size,
  modification time), so changed data is decoded again. A code change stops
  a resume with an error; `run_suite.py --accept-source-change "<reason>"`
  keeps finished jobs, records old/new hashes and the reason in `plan.json`
  (with `--invalidate-methods` naming methods whose finished results the
  change makes invalid), and the verifier reports the acknowledgement as a
  warning. Any other change to the plan, including `--threads-per-job` or an
  execution option, needs a new run directory.

## Protocols in one paragraph each

**Task-IL.** Each task owns a local label axis `0..C-1`. `--head-mode multi`
gives the shared-backbone baselines one head per task (the standard,
literature-comparable protocol); `shared` reuses one head and is kept only to
reproduce the legacy runs. Isolation methods and Independent always use the
plain backbone. See `PROTOCOL.md` §2.

**Class-IL.** Shared-backbone methods use one fixed `K`-way head over the
global label space. Isolation methods route without a task oracle by
exhaustive forwarding through every candidate subnetwork (entropy, confidence
or margin), optionally with a router fitted on a deterministic holdout of the
*training* data: the full score-logreg (T×T weights), the tied-slope
score-bias likelihood router (one slope + one bias per task), or the
accuracy-fitted score-bias-acc router. Only score-bias-acc starts from and
preserves the raw rule on its fitting objective. Prefix result matrices, ACC/LA/OracleACC/TA/
ConditionalACC and tie diagnostics are defined in `PROTOCOL.md` §3-4.

**Storage.** `diltcl.compact_export` packs masks to bits and stores only the
selected weights; `verify_roundtrip_logits()` must pass before a size is
reported (`tools/measure_compact_storage.py`).

## Notes on interpreting the results

* Multi-head LwF's per-head forwards were a v2 implementation artefact, not an
  intrinsic cost: v3 computes all old heads from one trunk pass. An O(T) LwF
  cost is therefore not an advantage of DiLT-CL.
* PyTorch enables TF32 for cuDNN convolutions by default; every GPU result of
  this project was produced under that setting and `execution.effective`
  records it.

## Legacy

Earlier implementations (a v1 single-file prototype and the v2 line) and
their results are not part of this repository; no v1/v2 result is used by
the accompanying paper.
