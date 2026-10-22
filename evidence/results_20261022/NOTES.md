# About this folder

This is the results package of the v3.0.11 regeneration run (all 410 jobs,
2026-10-21/22), as `tools/build_results_package.py --include-artifacts`
built it on the server, with the additions and the one redaction below. The
numbers reported in the paper are those of this run.

* **Added after the package was built**, and therefore not listed in
  `MANIFEST.json`: `regen.log` (the three training suites and the merges),
  `audit_new.log` (summary of the audits whose reports are in each run's
  `audit/` folder), `smoke_checks/` (the same-seed determinism smoke run and
  the pre-run data and tool checks), `check_environment.txt`, `pytest.txt`,
  `lscpu.txt` and `threadpool.txt` (the environment checks before the run).
* **Redaction.** In `p0/pip_freeze.txt`, `p1_bias/pip_freeze.txt` and
  `p1_biasacc/pip_freeze.txt`, one line naming an editable install of an
  unrelated package that was present in the server's Python environment (not
  imported by this code) was replaced by a comment, because its URL
  identifies an author. `MANIFEST.json` records the redacted files; no other
  file was changed.
* **Data.** The MNIST data were downloaded again with torchvision for this
  run, so the `MNIST` folder holds a different set of auxiliary files than
  the original runs' (8 files instead of 6). The regeneration audit
  therefore reports `50 with different data`: the 50 Permuted-MNIST jobs.
  Their decoded training pixels and every result are identical to the
  original runs'. The CIFAR-100 and TinyImageNet folders are identical file
  by file.
* **Package README.** The layout list in `README.md`, written by the
  packager, mentions `<run>/paper/`; this run wrote no such folder.
* **Per-job results.** `<run>/artifacts/<job>/result.json|npz` are the inputs
  of the merged artifacts. The checkpoints and per-example predictions they
  reference are in the release archives (`code/tools/build_release.py`), not
  here.
