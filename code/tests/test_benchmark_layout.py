"""Tests for :mod:`diltcl.benchmark_layout` (the torch-free layout logic).

Unlike the inference/subnetwork tests (which mirror torch logic in NumPy), this
imports the *real* module: it has no torch dependency, so the actual functions
used by ``diltcl.hard_benchmarks`` are exercised directly.
"""

import sys

try:  # normal in-package import (works wherever the package is importable)
    from diltcl.benchmark_layout import (
        five_datasets_offsets,
        parse_tinyimagenet_val_annotations,
        split_classes,
        task_label_map,
    )
except Exception:  # noqa: BLE001 - torch-less env: load the file directly
    import importlib.util
    import pathlib
    _p = pathlib.Path(__file__).resolve().parent.parent / "diltcl" \
        / "benchmark_layout.py"
    _spec = importlib.util.spec_from_file_location("benchmark_layout", _p)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    five_datasets_offsets = _m.five_datasets_offsets
    parse_tinyimagenet_val_annotations = _m.parse_tinyimagenet_val_annotations
    split_classes = _m.split_classes
    task_label_map = _m.task_label_map


def test_split_classes_even():
    splits = split_classes(num_classes=200, num_tasks=10)
    assert len(splits) == 10
    assert all(len(s) == 20 for s in splits)
    assert splits[0] == list(range(0, 20))
    assert splits[-1] == list(range(180, 200))
    # disjoint and complete
    flat = [c for s in splits for c in s]
    assert flat == list(range(200))


def test_split_classes_explicit_cpt_subset():
    # fast-run style: fewer classes than available -> uses the first N*C
    splits = split_classes(num_classes=200, num_tasks=3, classes_per_task=10)
    assert [len(s) for s in splits] == [10, 10, 10]
    assert splits[2] == list(range(20, 30))


def test_split_classes_validates():
    for bad in (
        dict(num_classes=200, num_tasks=7),                 # 200 % 7 != 0
        dict(num_classes=50, num_tasks=6, classes_per_task=10),  # 60 > 50
        dict(num_classes=10, num_tasks=0),                  # non-positive
    ):
        try:
            split_classes(**bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


def test_task_label_map_taskil_vs_classil():
    classes = [10, 11, 12, 13, 14]
    # task-IL: remap to 0..C-1
    assert task_label_map(classes, class_il=False) == {
        10: 0, 11: 1, 12: 2, 13: 3, 14: 4}
    # class-IL: identity
    assert task_label_map(classes, class_il=True) == {c: c for c in classes}


def test_parse_val_annotations_tab_and_blank():
    text = ("val_0.JPEG\tn01443537\t0\t0\t62\t62\n"
            "\n"
            "val_1.JPEG\tn01629819\t10\t20\t50\t60\n")
    m = parse_tinyimagenet_val_annotations(text)
    assert m == {"val_0.JPEG": "n01443537", "val_1.JPEG": "n01629819"}
    # insertion order preserved
    assert list(m.keys()) == ["val_0.JPEG", "val_1.JPEG"]


def test_parse_val_annotations_whitespace_fallback():
    # space-separated (no tabs) should still parse the first two fields
    m = parse_tinyimagenet_val_annotations("img_9.JPEG n02085620 1 2 3 4")
    assert m == {"img_9.JPEG": "n02085620"}


def test_five_datasets_offsets():
    assert five_datasets_offsets(5, 10) == [0, 10, 20, 30, 40]
    assert five_datasets_offsets(3, 10) == [0, 10, 20]
    # global label reconstruction: dataset t, native label y -> t*C + y
    offs = five_datasets_offsets(5, 10)
    assert offs[3] + 7 == 37


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} benchmark-layout tests passed.")
    sys.exit(0)
