# DiLT-CL v3.0 experimental protocol

This document is the normative protocol for new experiments (code v3.0.5). If
code, tables, or older documentation disagree with it, stop and resolve the
disagreement before running or reporting results.

> **Result status.** The results used by the accompanying paper are the merged,
> verified v3 artifacts in `../evidence/results_20260906/`. Earlier v2-line
> artifacts are not part of this repository and are not paper results. No
> performance result is claimed until the v3 pipeline has been run end to end on
> the declared hardware, the independent verifier has passed, and
> `tools/merge_results.py` has produced `verified` artifacts.

## Verified semantics (v3.0.5)

Each item below was checked by constructing a scenario with a known answer and
comparing against an independent reference implementation, not by reading the
code. The v2 items live in `tests/test_real_core.py`; the v3 items in
`tests/test_data_backend.py`, `tests/test_numerics_equivalence.py`,
`tests/test_backend_end_to_end.py` and `tests/test_merge_results.py`.

| Component | What was verified |
|---|---|
| ACC / BWT / Forgetting / LA | Match the GEM-style definitions on 200 random matrices; `Forgetting == -BWT` under monotone decay and `Forgetting > -BWT` when a task recovers |
| Reservoir buffer | Retention probability is `capacity/N` with no recency bias (4,000 trials, chi-square p = 0.24) |
| **EWC Fisher** | **Per-sample Fisher.** Before v2.1.8 the loss was averaged over a minibatch *before* squaring, so the Fisher scaled like `1/batch_size` and the effective `lambda_ewc` differed between benchmarks; pre-2.1.8 EWC numbers are not comparable. The label used for the per-sample gradient is an explicit, recorded choice: `ewc_fisher_labels="prediction"` (the model's argmax, the default and the v2 behaviour -- a common approximation, *not* the empirical Fisher) or `"true"` (the data label, the empirical Fisher proper). State the choice in the paper |
| LwF | `KL(teacher \|\| student)` with `T^2` scaling; multi-head runs distil each previous head separately |
| DER++ | `CE + alpha * MSE(buffer logits) + beta * CE(buffer labels)` with two independent buffer draws |
| Masking primitives | Intersection/complement/combine algebra; magnitude pruning keeps exactly `k` weights even when every magnitude is equal |
| DiLT-CL structure | Expert masks lie strictly outside the core; measured core sparsity reaches the configured target (0.905 against a 0.9 target, versus ~0.67 in v2.0); task 0's forward equals the core-only forward |
| Edge-popup (SupSup/WSN) | Top-k by `\|score\|`, straight-through gradients reach the scores; SupSup freezes weights, WSN trains them, both freeze biases |
| Class-IL prefix matrices | Reproduce a hand-constructed routing-degradation scenario exactly; upper triangle zero, `ACC <= TA`, `oracle >= class-IL` |
| Label maps | Task-IL maps every task onto `0..C-1`; class-IL keeps disjoint global blocks |
| Permuted-MNIST | Pixel permutation preserves labels and the pixel multiset; one permutation per task, shared by train and test |
| Calibration holdout | Deterministic per seed; train/calibration indices disjoint and complete; prefix calibrator `s` only sees tasks `0..s` |
| Significance | Matches SciPy exactly; see the switching rule above |
| **Multi-head replay routing** | Replayed items go through the head that produced them, each group weighted by its share of the draw. Before v2.2.3 the routing was missing (ER fell *below* plain finetuning) and then the grouped loss was summed rather than averaged, inflating replay strength by the number of tasks drawn. Pre-2.2.3 ER/DER++ numbers are invalid |
| **Provenance** | `CODE_VERSION` is derived from `diltcl.__version__`. Before v2.2.5 both runners hardcoded it, so artifacts produced by later versions still reported `2.1.1` -- and that string is printed into table captions |
| **Per-example class-IL predictions (v3.0.5)** | Every isolation-method class-IL job launched by the suite writes `predictions/<benchmark>/<method>/seed<N>.npz` (true task, label, the oracle expert's prediction, every expert's routing score, and the routed task, tie flag and prediction at every prefix stage; calibrated jobs add the matched raw router and the held-out calibration inputs). Recording reuses the evaluation's own forward passes and changes no returned value (`tests/test_predictions_audit.py` compares recorded and unrecorded evaluations for every criterion and both label protocols). `tools/recompute_from_predictions.py` rebuilds every class-IL matrix and pooled statistic from these arrays exactly and checks that every raw-router decision is the lowest-ID argmin of the recorded scores; the verifier runs the same check on every job |
| **Checkpoint re-evaluation (v3.0.5)** | `tools/evaluate_checkpoints.py` loads each persisted isolation state (`weights_only=True`), checks its file and tensor-content SHA-256 against the artifact, rebuilds the model from the artifact's configuration and execution options, re-runs the class-IL evaluation (job router and matched raw router) and compares every matrix, metric and per-example decision; on the recording hardware the comparison is exact. `tools/compare_runs.py` compares an independent re-run with retained artifacts (trained-state content hashes, every NPZ array, every per-seed metric) |
| **Device data backend (v3)** | Bit-identical to the `DataLoader` pipeline: same `ToTensor`/`Normalize` arithmetic (tensor divisor, no CUDA reciprocal shortcut), same shuffle order and global-RNG consumption per `iter()`, same subset composition; materialisation restores the RNG state it would otherwise consume. Verified batch by batch against `DataLoader` (0 and 2 workers) and end to end through finetune / ER / LwF / DiLT-CL on synthetic on-disk CIFAR-100 and TinyImageNet trees |
| **Fast numerics (v3)** | Shared-feature multi-head LwF, consolidated EWC quadratic, device reservoir buffer and per-row routed replay equal the v2 objective: same RNG draws, same retained/replayed items, loss and gradients equal to float tolerance, EWC gradient identical and loss offset by a parameter-independent constant. Fused 0/1 mask multiplies are bit-identical to the v2 loops. `--numerics legacy` selects the v2 arithmetic |
| **Routed forward determinism (v3)** | Replay rows are routed to their own heads with one all-heads matmul plus `gather`; every gradient destination receives exactly one contribution, so no atomic-add path exists and same-seed repeats stay bit-identical on CUDA |
| **Edge-popup STE (v3.0.2)** | The straight-through estimator receives ``scores.abs()`` so autograd carries ``sign(score)`` back to the raw scores, as in Ramanujan et al. (2020). Before v3.0.2 the ``abs()`` was taken inside the estimator, negative scores were updated in the wrong direction, and SupSup stayed at chance level on every benchmark (P0: 9.4% on Permuted-MNIST, 19.9% on 5-way CIFAR, 5.1% on TinyImageNet) while WSN trained with a corrupted selection rule. **SupSup and WSN numbers produced before v3.0.2 are invalid** and are rerun under a recorded ``--invalidate-methods supsup,wsn`` acknowledgement; every other method is unaffected |
| **Job granularity (v3)** | One job per (benchmark, seed, method) merged back equals one monolithic run exactly (per-seed metrics and matrices), for both runners |


## 1. Result-matrix convention

For T tasks, every method stores a T by T matrix R with

- R[i,j] = accuracy on task j after learning through task i;
- entries with j > i are unevaluated;
- every evaluated entry is a percentage in [0,100].

Independent, PackNet, SupSup, WSN, and DiLT-CL use this same convention.
Independent is not stored as a diagonal-only special case. If its task-j model
has accuracy a_j and remains available, then R[i,j] = a_j for every i >= j.

Each NPZ matrix must round-trip to the metrics in the accompanying JSON. A
method-specific interpretation or hidden exception is a serialization failure.

## 2. Task-incremental protocol

Task-IL supplies the true task identity at evaluation time.

- Split-CIFAR-100 and Split-TinyImageNet use disjoint class blocks, remapped to
  local labels for a per-task-width output head.
- Permuted-MNIST uses shared digit labels and task-specific input permutations.
- 5-datasets uses one dataset per task and native local labels.
- Isolation methods select the true task's mask, expert, or packed subnetwork.
- Shared-backbone methods use their declared Task-IL head and no Class-IL
  routing heuristic.

Weight-isolation methods can have column-constant Task-IL matrices when both
parameters and all prediction-affecting state are genuinely frozen. This
Task-IL property must not be carried over to Class-IL routing metrics.

## 3. Class-incremental protocol

Class-IL does not provide the task identity.

### 3.1 Shared-backbone methods

**Task-IL head protocol.** `run_task_il.py --head-mode` selects how
these baselines get their output layer. `shared` (default) reuses one C-way head
across every task, reproducing the legacy protocol; `multi` gives each task its
own head (`diltcl.models.MultiHeadModel`), which is the standard task-IL
convention and the only one comparable with the literature. Isolation methods
and Independent always use the plain backbone, because they already own
task-specific parameters. The choice is recorded as `task_il_head_mode` in every
result file, and the disclosure text adapts to it.

Under class-IL, Finetune, EWC, LwF, ER, and DER++ use global labels and one
fixed full K-way head allocated from task 1. Unseen-class output units are present and
participate in the softmax before their classes are introduced.

This is a **fixed-full-head Class-IL variant**. It is internally valid when
applied consistently, but differs from an expanding head or a protocol that
masks unseen logits. Every result table and comparison must disclose this.
Comparisons to seen-classes-only literature require either matching that
protocol or an explicit caveat.

If the project later switches to seen-classes-only evaluation, unseen logits
must be masked during both training loss computation and evaluation, and all
shared-backbone Class-IL results must be rerun.

### 3.2 Isolation methods and routing

DiLT-CL, PackNet, SupSup, and WSN use **exhaustive entropy routing**.

For each input and each currently available task t:

1. run the input through task t's subnetwork;
2. compute the entropy of its softmax output;
3. select the task with minimum entropy;
4. map the selected local prediction to the global label space.

This requires one forward pass per available task. The raw rule is a
forward-only exhaustive router. It is not SupSup-GG or a gradient-based
one-shot method. Confidence and within-sample standardized top-two margin are
optional raw criteria and must be labelled.

Three opt-in calibrated routers exist. ``score-logreg`` fits a class-balanced
multinomial logistic classifier with a full ``T x T`` weight matrix on the
standardized candidate-score signature. ``score-bias`` (v3.0.1) is the
tied-slope router ``logit_k = -a * z_k + b_k`` -- one shared slope plus one
bias per task, ``T + 1`` parameters. Separate per-column standardization and
an unconstrained slope mean this family does not generally contain the raw
unstandardized ``argmin`` rule; it has no fitting- or test-accuracy
nondegradation guarantee. It is serialized in the same state layout, so the
common inference path applies it unchanged. ``score-bias-acc`` (v3.0.3)
fits the per-task biases for *routing accuracy* by coordinate ascent from
``b = 0`` on the raw scores, so on its fitting data it is never below the
raw rule; every router state records ``fit_accuracy`` and
``raw_argmin_fit_accuracy``. All are leakage-safe:

- each task's calibration examples come from a deterministic training holdout;
- shared and isolation methods train on the same reduced training split;
- stage i is fitted only with task labels and score dimensions 0..i;
- test inputs and labels never fit or select the router;
- the split seed and per-task counts, scaler, coefficient, intercept,
  convergence flag, and sample count are serialized in metric-schema-v3 JSON.

Holdout indices and sample IDs are not serialized. Therefore the metric JSON
alone cannot independently reconstruct a split unless the dataset identity and
ordering are also pinned in the surrounding run bundle.

Exact routing-score ties are resolved deterministically to the lowest candidate
task ID. Prefix tie-rate matrices and final macro/micro tie rates must be saved;
they are diagnostics, not a reason to drop samples.

For a strict online interpretation, old calibration inputs must be retained
until later score dimensions can be computed. The result metadata records this
temporary retention requirement; the inputs themselves are not written to the
result artifact. Final serving needs only the last O(T^2)-float linear router;
earlier prefix states are serialized for temporal audit and may be discarded.

Entropy routing is sensitive to logit-scale calibration. Report TA and
OracleACC so readers can separate routing error from within-task
classification error.

### 3.3 Prefix result matrices

A valid isolation Class-IL matrix is temporal, not a copied final vector.

For row i:

- evaluate only true tasks 0..i;
- permit the router to choose only among subnetworks 0..i;
- write the resulting per-task accuracies into R[i,0..i].

Because the older subnetworks are frozen, this matrix may be reconstructed
after training all tasks by replaying each prefix of the candidate set.
Nevertheless, adding a new candidate can steal samples from an older expert.
Therefore frozen weights do **not** imply Class-IL BWT = 0 or Forgetting = 0.

## 4. Metric definitions

Metrics use the genuine matrix R.

- ACC = macro mean of the final row.
- LA = mean of the diagonal R[j,j]. This definition is identical for every
  method and protocol.
- BWT = mean over earlier tasks of R[T-1,j] - R[j,j].
- Forgetting = mean over earlier tasks of the best historical value in column j
  minus R[T-1,j].
- OracleACC = final accuracy of an isolation method when the true task ID
  selects its subnetwork. It is not LA.
- TA = final task-inference accuracy of the routing heuristic.
- MicroACC = optional pooled accuracy over samples.
- ConditionalACC = pooled class accuracy among correctly routed samples;
  MicroConditionalACC is a compatibility alias for the same pooled value.
- Record `ZeroCorrectRouteTasks`; a zero route count must never disappear behind
  a selective task-macro average.
- Under disjoint label blocks,
  `MicroACC = MicroTA * MicroConditionalACC / 100` exactly.

The primary cross-method ACC is macro averaged. This matters for 5-datasets,
whose test sets are unequal. MicroACC may be reported in a separate column but
must never silently replace macro ACC for only one method family.

Forward Transfer requires explicit pre-training evaluations in the upper
triangle or a separate baseline record. Do not compute it from zero-filled
unevaluated cells.

## 5. DiLT-CL pruning protocol

### 5.1 Active and trainable masks

A mask has two distinct roles:

- the **active mask** determines which weights participate in the forward pass;
- the **trainable mask** determines which active weights receive updates.

For partition IMP, the masks coincide. For expert IMP, the active mask is
core OR current expert, while only current expert weights are trainable.
Pruned and out-of-subnetwork weights must not contribute nonzero values to the
forward pass after rewind.

### 5.2 Target sparsity and compute

A fixed 20% pruning rate leaves approximately 0.8^R of the eligible weights
after R rounds. Thus five rounds reach about 67.2% sparsity, not 90%, and
eleven 20% rounds are required to cross 90% in the ideal no-tie calculation.

Every run must record:

- requested target sparsity;
- pruning rounds and per-round rates;
- achieved density globally and within the eligible region;
- IMP training epochs per round;
- final expert fine-tuning epochs.

The schedule must reach the requested target up to a documented integer
rounding tolerance. Magnitude ties must not cause uncontrolled over-pruning;
selection should preserve an exact target count.

Compute reporting must include all partition IMP rounds, expert IMP rounds, and
post-selection expert fine-tuning. Counting only the final fine-tune is not a
fair comparison with baselines.

### 5.3 Rewind point

The current method is defined with **late rewind** to the saved post-Phase-1
snapshot. It is not rewind to the original random initialization. Use the term
late rewind in the method description and record exactly when the snapshot is
taken. A study of random-initialization rewind is a distinct ablation.

Each expert extraction must start from its declared rewind state rather than
silently inheriting the previous expert's live weights.

## 6. Baseline correctness and reproducibility

Before a full run, verify:

- LwF computes KL(old || current), with old probabilities as the fixed target.
- Python random, NumPy, CPU PyTorch, and all CUDA RNGs are seeded.
- Reservoir replacement and replay sampling repeat under the same seed.
- Per-seed metric dictionaries have identical schemas.
- Paired significance tests use equal-length arrays aligned by explicit seed.
- The paired test **switches by sample size**: with fewer than six non-zero
  per-seed differences it is a paired t-test; from six onwards it is the
  Wilcoxon signed-rank test. Report whichever was actually used -- the `test`
  field of every significance record names it. Five seeds still use the t-test;
  six or more switch to Wilcoxon.
- All reported values are finite.
- Independent serializes a lower-triangular, column-constant matrix.

At least five valid paired seeds are required before using the significance
table as evidence. A failed run is not silently dropped; record the failure and
rerun policy.

Deterministic flags do not guarantee every CUDA operation is deterministic on
every hardware/software combination. Record the exact environment and check a
same-seed repeat on the target platform (the smoke profile does this and the
verifier compares the two artifacts bit for bit).

Execution options are part of a run's identity. `--data-backend`, `--numerics`,
`--tf32` and `--eval-batch-size` are recorded under `execution` in every
artifact and enter the suite's plan hash; a table must not mix artifacts with
different execution options (`tools/merge_results.py` refuses to). Note that
PyTorch enables TF32 for cuDNN convolutions by default; `--tf32 default` keeps
that behaviour, and `execution.effective` records the flags in force.

## 7. Benchmarks and averaging

Supported setups:

- Split-CIFAR-100: default 20 tasks by 5 classes.
- Permuted-MNIST: default 10 permutation tasks.
- Split-TinyImageNet: default 10 tasks by 20 classes.
- 5-datasets: CIFAR-10, MNIST, SVHN, FashionMNIST, and KMNIST by default;
  notMNIST may be supplied for the classic variant.

Do not infer equal task sizes merely because every task has the same number of
classes. Save per-task sample counts in the run metadata.

Dataset loading is offline-first. Record dataset roots, versions or checksums,
class order, task order, transforms, and whether any download occurred.

## 8. Storage reporting

The arithmetic in diltcl/profiling.py is an **analytic compact-format
estimate**, not a measurement of the current dense Python objects.

Report storage under one clearly named state definition:

- **inference-only state**: data needed only to serve learned tasks;
- **resume-learning checkpoint**: data needed to continue continual learning;
- **serialized compact artifact**: bytes written by a tested exporter.

Do not mix these definitions. For example, inference-only EWC may discard
Fisher/anchor state, whereas resume-learning classic EWC retains per-task
Fishers and anchors. Similarly, replay buffers are not required for
inference-only serving.

A DiLT-CL compact estimate may assume sparse values plus 1-bit masks only when
that assumption is stated. It is not measured deployment storage until an
export/import path demonstrates prediction-equivalent round-trip behavior and
the file size is measured. Account for the actual number of stored experts;
the current task-0 core-only design creates T-1 task-specific experts.

## 9. Commands

Full test suite (one session):

~~~bash
python -m pytest tests
~~~

GPU bit-exactness check of the data backend (runs on CUDA when available):

~~~bash
python -m pytest tests/test_data_backend.py tests/test_backend_end_to_end.py -q
~~~

Real-PyTorch smoke run (includes the same-seed determinism repeat):

~~~bash
python run_suite.py --profile smoke --seeds 0 \
    --data-root /DATA_ROOT --allow-download \
    --run-dir /RUNS/smoke
~~~

Correctness-gated full run (P0: 330 jobs at method granularity, 5 seeds):

~~~bash
python run_suite.py --profile p0 --seeds 0-4 --head-mode multi \
    --data-root /DATA_ROOT --dataset-hash full \
    --run-dir /RUNS/p0_v3 --max-parallel auto
python tools/verify_experiment_bundle.py /RUNS/p0_v3
python tools/merge_results.py /RUNS/p0_v3
~~~

Fast or demo results establish only that the pipeline executes. They are not
paper results. New outputs must be kept separate from legacy artifacts, must
pass `tools/verify_experiment_bundle.py`, and only merged artifacts promoted to
`verified` by `tools/merge_results.py` may feed tables.

The suite persists one evaluated-state checkpoint per method/seed by default.
`--no-save-checkpoints` is diagnostic-only and fails the checkpoint evidence
gate. The P0/P1 job matrix and the resume rules are documented in `README.md`
and `REPRODUCE.md`.
