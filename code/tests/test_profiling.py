"""Tests for parameter / memory accounting (:mod:`diltcl.profiling`).

These are pure-arithmetic tests: they need only ``diltcl.profiling`` (no torch),
so they run in any environment and pin down the numbers that back the paper's
cost claims.
"""

# ``diltcl.profiling`` has no intra-package dependencies, but importing it
# through the package would trigger ``diltcl/__init__`` (which imports torch).
# Load it directly from its file when the package import is unavailable, so the
# test runs with or without torch installed.
try:
    from diltcl.profiling import (
        FLOAT_BYTES,
        build_default_suite,
        diltcl_cost,
        independent_cost,
        packnet_cost,
        supsup_cost,
    )
except Exception:  # noqa: BLE001 - torch may be unavailable during package init
    import importlib.util
    import pathlib

    _path = pathlib.Path(__file__).resolve().parents[1] / "diltcl" / "profiling.py"
    _spec = importlib.util.spec_from_file_location("diltcl_profiling", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    FLOAT_BYTES = _mod.FLOAT_BYTES
    build_default_suite = _mod.build_default_suite
    diltcl_cost = _mod.diltcl_cost
    independent_cost = _mod.independent_cost
    packnet_cost = _mod.packnet_cost
    supsup_cost = _mod.supsup_cost

P = 1000          # backbone params
T = 10            # tasks


def _by_name(costs):
    return {c.name: c for c in costs}


def test_byte_math():
    """bytes() = 4*floats + mask_bits/8 + 4*buffer_floats."""
    c = diltcl_cost(P, T, core_density=0.3, expert_density=0.2)
    expected_floats = (0.3 + (T - 1) * 0.2) * P    # core + 9 experts
    assert abs(c.float_params - expected_floats) < 1e-9
    assert abs(c.mask_bits - T * P) < 1e-9
    expected_bytes = expected_floats * FLOAT_BYTES + T * P / 8.0
    assert abs(c.bytes() - expected_bytes) < 1e-6


def test_inference_scope_drops_training_only_state():
    costs = _by_name(build_default_suite(
        ["ewc", "er", "derpp"], P, T, buffer_size=500, sample_numel=100,
        num_classes=5, core_density=0.3, expert_density=0.2,
        state_scope="inference"))
    assert costs["EWC"].float_params == P
    assert costs["ER"].buffer_floats == 0
    assert costs["DER++"].buffer_floats == 0


def test_resume_scope_counts_actual_ewc_and_replay_state():
    costs = _by_name(build_default_suite(
        ["ewc", "er", "derpp"], P, T, buffer_size=500, sample_numel=100,
        num_classes=5, core_density=0.3, expert_density=0.2,
        state_scope="resume"))
    assert costs["EWC"].float_params == (1 + 2 * T) * P
    assert costs["ER"].buffer_floats == 500 * 100
    assert costs["DER++"].buffer_floats == 500 * (100 + 5)


def test_independent_scales_with_tasks():
    c = independent_cost(P, T)
    assert c.float_params == T * P
    assert c.mask_bits == 0


def test_supsup_stores_no_floats_by_default():
    """SupSup's backbone is regenerable -> zero stored floats, only masks."""
    c = supsup_cost(P, T)
    assert c.float_params == 0.0
    assert c.mask_bits == T * P
    c2 = supsup_cost(P, T, store_backbone=True)
    assert c2.float_params == P


def test_packnet_one_backbone_plus_masks():
    c = packnet_cost(P, T)
    assert c.float_params == P
    assert c.mask_bits == T * P


def test_isolation_storage_ordering():
    """Under the stated analytic model: SupSup < PackNet < DiLT < Independent.

    DiLT-CL is cheaper than the independent reference but pricier than
    mask-only methods because it stores real experts.
    """
    methods = ["supsup", "packnet", "diltcl", "independent"]
    costs = _by_name(build_default_suite(
        methods, P, T, buffer_size=500, sample_numel=3072, num_classes=5,
        core_density=0.30, expert_density=0.23))
    s = costs["SupSup"].bytes()
    pk = costs["PackNet"].bytes()
    dl = costs["DiLT-CL (ours)"].bytes()
    ind = costs["Independent"].bytes()
    assert s < pk < dl < ind, (s, pk, dl, ind)


def test_diltcl_cheaper_than_independent():
    costs = _by_name(build_default_suite(
        ["diltcl", "independent"], P, T, buffer_size=500, sample_numel=3072,
        num_classes=5, core_density=0.30, expert_density=0.23))
    assert costs["DiLT-CL (ours)"].bytes() < costs["Independent"].bytes()


def test_suite_is_canonically_ordered():
    methods = ["independent", "diltcl", "finetune"]  # deliberately scrambled
    names = [c.name for c in build_default_suite(
        methods, P, T, buffer_size=500, sample_numel=3072, num_classes=5,
        core_density=0.3, expert_density=0.2)]
    assert names == ["Finetune", "DiLT-CL (ours)", "Independent"]


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} profiling tests passed.")
    sys.exit(0)
