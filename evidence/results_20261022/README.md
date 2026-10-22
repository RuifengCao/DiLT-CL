# Results package

Built 2026-10-22T10:30:41.728861+00:00 by tools/build_results_package.py.

## Runs

### p0
- source run directory: `/root/autodl-tmp/runs_v3011/p0`
- verification: **verified** (330 jobs validated, 0 errors, 0 warnings)
- code versions in merged artifacts: ['3.0.11']
- `merged/p0_class_entropy.json`: **verified**, 90 jobs, seeds [0, 1, 2, 3, 4], benchmarks ['cifar', 'tinyimagenet'], methods ['finetune', 'ewc', 'lwf', 'er', 'derpp', 'packnet', 'supsup', 'wsn', 'diltcl']
- `merged/p0_class_entropy_cal.json`: **verified**, 90 jobs, seeds [0, 1, 2, 3, 4], benchmarks ['cifar', 'tinyimagenet'], methods ['finetune', 'ewc', 'lwf', 'er', 'derpp', 'packnet', 'supsup', 'wsn', 'diltcl']
- `merged/p0_task_il_multi.json`: **verified**, 150 jobs, seeds [0, 1, 2, 3, 4], benchmarks ['cifar', 'pmnist', 'tinyimagenet'], methods ['finetune', 'ewc', 'lwf', 'er', 'derpp', 'packnet', 'supsup', 'wsn', 'diltcl', 'independent']

### p1_bias
- source run directory: `/root/autodl-tmp/runs_v3011/p1_bias`
- verification: **verified** (40 jobs validated, 0 errors, 0 warnings)
- code versions in merged artifacts: ['3.0.11']
- `merged/p1_class_entropy_bias.json`: **verified**, 40 jobs, seeds [0, 1, 2, 3, 4], benchmarks ['cifar', 'tinyimagenet'], methods ['packnet', 'supsup', 'wsn', 'diltcl']

### p1_biasacc
- source run directory: `/root/autodl-tmp/runs_v3011/p1_biasacc`
- verification: **verified** (40 jobs validated, 0 errors, 0 warnings)
- code versions in merged artifacts: ['3.0.11']
- `merged/p1_class_entropy_biasacc.json`: **verified**, 40 jobs, seeds [0, 1, 2, 3, 4], benchmarks ['cifar', 'tinyimagenet'], methods ['packnet', 'supsup', 'wsn', 'diltcl']

## Layout

- `<run>/merged/*.json|npz` - merged artifacts; `aggregated` holds mean/std/values per metric, `paired_significance` the paired tests against DiLT-CL, `merge_provenance` the input hashes.
- `<run>/paper/` - LaTeX tables, figures (with `figure_provenance.json`), storage and timing measurements.
- `<run>/verification_report.json` - independent recomputation of every metric from raw matrices.
- `<run>/plan.json`, `suite_summary.json`, `failures.json`, `statuses/` - what ran, how long, under which source hash.
- `<run>/source_manifest(s)`, `environment(s)`, `dataset_fingerprint(s)` - provenance (per-file lists stripped).
- `MANIFEST.json` - SHA-256 of every file in this package.
