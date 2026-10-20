# Results package

Built 2026-09-06T14:50:22.849140+00:00 by tools/build_results_package.py.

## Runs

### p0_v3
- source run directory: `/root/autodl-tmp/runs_v3/p0_v3`
- verification: **verified** (330 jobs validated, 0 errors, 1 warnings)
- acknowledged source changes: dca137eb9939->bee8db745ca3: v3.0.1: class-IL calibration accepts shared-only jobs; per-benchmark dataset identity; file-level preflight; EWC flat penalty (bit-identical gradients). No change to finished results.; bee8db745ca3->404c756e0273: v3.0.2: edge-popup straight-through estimator took abs() inside the STE, dropping sign(score); SupSup/WSN masks could not learn. Fixed; SupSup and WSN results invalidated and rerun. Other methods unaffected. (invalidated ['supsup', 'wsn'])
- code versions in merged artifacts: ['3.0.0/3.0.2', '3.0.0/3.0.1/3.0.2']
- `merged/p0_class_entropy.json`: **verified**, 90 jobs, seeds [0, 1, 2, 3, 4], benchmarks ['cifar', 'tinyimagenet'], methods ['finetune', 'ewc', 'lwf', 'er', 'derpp', 'packnet', 'supsup', 'wsn', 'diltcl']
- `merged/p0_class_entropy_cal.json`: **verified**, 90 jobs, seeds [0, 1, 2, 3, 4], benchmarks ['cifar', 'tinyimagenet'], methods ['finetune', 'ewc', 'lwf', 'er', 'derpp', 'packnet', 'supsup', 'wsn', 'diltcl']
- `merged/p0_task_il_multi.json`: **verified**, 150 jobs, seeds [0, 1, 2, 3, 4], benchmarks ['cifar', 'pmnist', 'tinyimagenet'], methods ['finetune', 'ewc', 'lwf', 'er', 'derpp', 'packnet', 'supsup', 'wsn', 'diltcl', 'independent']

### p1_bias
- source run directory: `/root/autodl-tmp/runs_v3/p1_bias`
- verification: **verified** (40 jobs validated, 0 errors, 0 warnings)
- code versions in merged artifacts: ['3.0.2']
- `merged/p1_class_entropy_bias.json`: **verified**, 40 jobs, seeds [0, 1, 2, 3, 4], benchmarks ['cifar', 'tinyimagenet'], methods ['packnet', 'supsup', 'wsn', 'diltcl']

### p1_biasacc
- source run directory: `/root/autodl-tmp/runs_v3/p1_biasacc`
- verification: **verified** (40 jobs validated, 0 errors, 0 warnings)
- code versions in merged artifacts: ['3.0.3']
- `merged/p1_class_entropy_biasacc.json`: **verified**, 40 jobs, seeds [0, 1, 2, 3, 4], benchmarks ['cifar', 'tinyimagenet'], methods ['packnet', 'supsup', 'wsn', 'diltcl']

## Layout

- `<run>/merged/*.json|npz` - merged artifacts; `aggregated` holds mean/std/values per metric, `paired_significance` the paired tests against DiLT-CL, `merge_provenance` the input hashes.
- `<run>/paper/` - LaTeX tables, figures (with `figure_provenance.json`), storage and timing measurements.
- `<run>/verification_report.json` - independent recomputation of every metric from raw matrices.
- `<run>/plan.json`, `suite_summary.json`, `failures.json`, `statuses/` - what ran, how long, under which source hash.
- `<run>/source_manifest(s)`, `environment(s)`, `dataset_fingerprint(s)` - provenance (per-file lists stripped).
- `MANIFEST.json` - SHA-256 of every file in this package.
