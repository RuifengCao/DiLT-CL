"""Generate the parameter / memory comparison table (text + LaTeX).

This standalone tool reports analytic storage estimates for serving all tasks,
including the ratio to an independent-task reference.

It does **not** require running the experiments. The only run-specific inputs
are the backbone size ``P`` (computed from the model, or passed via
``--params``) and DiLT-CL's observed core / expert densities, which the
training pipeline prints as ``Core density`` / ``Expert density``. Defaults for
the densities match the values observed in the project's reference run; replace
them with your own run's numbers for a faithful table.

Examples
--------
Use the exact CIFAR backbone size (requires torch + the package)::

    python analyze_memory.py --benchmark cifar --tasks 20 \
        --core-density 0.30 --expert-density 0.23

Run anywhere without torch by passing the parameter count directly::

    python analyze_memory.py --benchmark cifar --tasks 20 --params 619077

Writes ``<out>.tex`` and prints the text table to stdout.
"""

import argparse

# The memory accounting is pure arithmetic and does not need torch. Import the
# profiling module directly so this tool also runs in a torch-free environment;
# fall back to loading the file when the package import (which pulls in torch)
# is unavailable.
try:
    from diltcl.profiling import (
        build_default_suite,
        format_cost_table_latex,
        format_cost_table_text,
    )
except Exception:  # noqa: BLE001 - torch may be unavailable during package init
    import importlib.util
    import pathlib

    _path = pathlib.Path(__file__).resolve().parent / "diltcl" / "profiling.py"
    _spec = importlib.util.spec_from_file_location("diltcl_profiling", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    build_default_suite = _mod.build_default_suite
    format_cost_table_latex = _mod.format_cost_table_latex
    format_cost_table_text = _mod.format_cost_table_text

ALL_METHODS = ["finetune", "ewc", "lwf", "er", "derpp",
               "packnet", "supsup", "wsn", "diltcl", "independent"]

# (sample_numel, num_classes_per_task, default_tasks) per benchmark.
_BENCHMARK_DEFAULTS = {
    "cifar": (3 * 32 * 32, 5, 20),
    "pmnist": (28 * 28, 10, 10),
}


def _count_params(benchmark, classes_per_task):
    """Count trainable backbone parameters by instantiating the model.

    Requires torch and the package's models. Raises a clear error otherwise so
    the caller can fall back to ``--params``.
    """
    from diltcl import SimpleCNN, SimpleMLP
    if benchmark == "cifar":
        model = SimpleCNN(num_classes=classes_per_task)
    else:
        model = SimpleMLP(input_size=784, hidden_size=256, num_classes=10)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Parameter / memory comparison table (text + LaTeX).")
    p.add_argument("--benchmark", choices=["cifar", "pmnist"], default="cifar")
    p.add_argument("--tasks", type=int, default=None,
                   help="Number of tasks T (default: benchmark-specific).")
    p.add_argument("--params", type=int, default=None,
                   help="Backbone parameter count P. If omitted, the model is "
                        "instantiated to count it (needs torch).")
    p.add_argument("--methods", type=str, default=",".join(ALL_METHODS),
                   help="Comma-separated subset of: " + ",".join(ALL_METHODS))
    p.add_argument("--core-density", type=float, default=0.30,
                   help="Observed DiLT-CL core density (see training log).")
    p.add_argument("--expert-density", type=float, default=0.23,
                   help="Observed DiLT-CL per-task expert density.")
    p.add_argument("--buffer-size", type=int, default=500,
                   help="Replay-buffer capacity for ER / DER++.")
    p.add_argument("--supsup-store-backbone", action="store_true",
                   help="Count SupSup's fixed random backbone explicitly "
                        "instead of treating it as regenerable from a seed.")
    p.add_argument("--state-scope", choices=["inference", "resume"],
                   default="inference",
                   help="Count prediction-serving state only (default) or "
                        "state needed to resume continual training.")
    p.add_argument("--out", type=str, default="memory_table",
                   help="Output filename prefix for the .tex file.")
    return p.parse_args()


def main():
    args = _parse_args()
    sample_numel, num_classes, default_tasks = _BENCHMARK_DEFAULTS[args.benchmark]
    num_tasks = args.tasks if args.tasks is not None else default_tasks

    params = args.params
    if params is None:
        try:
            params = _count_params(args.benchmark, num_classes)
        except Exception as exc:  # noqa: BLE001 - torch may be unavailable.
            raise SystemExit(
                "Could not instantiate the model to count parameters "
                f"({exc}). Re-run with --params <P>, e.g. --params 619077 "
                "for the CIFAR SimpleCNN.") from exc

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Choose from {ALL_METHODS}")

    costs = build_default_suite(
        methods, params, num_tasks, buffer_size=args.buffer_size,
        sample_numel=sample_numel, num_classes=num_classes,
        core_density=args.core_density, expert_density=args.expert_density,
        supsup_store_backbone=args.supsup_store_backbone,
        state_scope=args.state_scope)

    print(f"\nMemory / parameter accounting -- {args.benchmark.upper()} "
          f"(T={num_tasks}, P={params:,})\n")
    print(format_cost_table_text(costs, params))

    latex = format_cost_table_latex(costs, params, num_tasks,
                                    benchmark=args.benchmark,
                                    state_scope=args.state_scope)
    out_path = f"{args.out}.tex"
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table written to '{out_path}'.")


if __name__ == "__main__":
    main()
