"""Entry point: run the full continual-learning experiment suite.

Usage
-----
Full paper-quality run::

    python run_experiments.py

Quick reduced run (fewer tasks / epochs)::

    python run_experiments.py --fast        # or: DILTCL_FAST=1 python run_experiments.py

Tiny smoke-test demo::

    python run_experiments.py --demo         # or: DILTCL_DEMO=1 python run_experiments.py

The suite covers six experiments: DiLT-CL / EWC / LwF on Split-CIFAR-100,
DiLT-CL / EWC / LwF on Permuted-MNIST, and a Gromov-Wasserstein
subnetwork-similarity analysis. Results are written to ``diltcl_results.npz``.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from diltcl import (
    DiLTCL,
    ExperimentConfig,
    SimpleCNN,
    SimpleMLP,
    compute_gw_distance,
    evaluate_with_mask,
    extract_layer_representations,
    get_device,
    get_permuted_mnist,
    get_split_cifar100,
    masking,
    print_result_matrix,
    set_seed,
    train_diltcl,
    train_ewc,
    train_lwf,
)


def main(config=None):
    """Run the full experiment suite.

    Args:
        config: An :class:`ExperimentConfig`. Defaults to a full-quality run.
    """
    cfg = config or ExperimentConfig()
    set_seed(cfg.seed)

    device = get_device()
    print(f"Using device: {device}")
    if device == "cpu":
        print("[Warning] Running on CPU. This may be slow. "
              "Consider using a GPU for faster training.")

    results_summary = {}

    # =================================================================
    # Experiment 1: Split-CIFAR-100 with DiLT-CL
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Split-CIFAR-100 with DiLT-CL")
    print("=" * 70)

    train_loaders_c100, test_loaders_c100 = get_split_cifar100(
        num_tasks=cfg.c100_num_tasks,
        classes_per_task=cfg.c100_classes_per_task,
        batch_size=cfg.c100_batch_size,
        seed=cfg.seed,
    )

    model_diltcl_cnn = SimpleCNN(num_classes=cfg.c100_classes_per_task)
    result_diltcl, acc_diltcl, bwt_diltcl = train_diltcl(
        model_diltcl_cnn, train_loaders_c100, test_loaders_c100,
        num_tasks=cfg.c100_num_tasks, device=device,
        num_partitions=cfg.diltcl_partitions,
        pruning_rounds=cfg.pruning_rounds,
        target_sparsity=cfg.diltcl_sparsity,
        full_train_epochs=cfg.diltcl_full_epochs,
        imp_epochs=cfg.diltcl_imp_epochs,
        lr=cfg.lr,
    )
    print_result_matrix(result_diltcl, "DiLT-CL")
    results_summary["DiLT-CL (CIFAR-100)"] = {"ACC": acc_diltcl, "BWT": bwt_diltcl}

    # =================================================================
    # Experiment 2: Split-CIFAR-100 with EWC
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Split-CIFAR-100 with EWC")
    print("=" * 70)

    model_ewc_cnn = SimpleCNN(num_classes=cfg.c100_classes_per_task)
    result_ewc, acc_ewc, bwt_ewc = train_ewc(
        model_ewc_cnn, train_loaders_c100, test_loaders_c100,
        num_tasks=cfg.c100_num_tasks, device=device,
        lambda_ewc=cfg.ewc_lambda, epochs=cfg.epochs_ewc, lr=cfg.lr,
    )
    print_result_matrix(result_ewc, "EWC")
    results_summary["EWC (CIFAR-100)"] = {"ACC": acc_ewc, "BWT": bwt_ewc}

    # =================================================================
    # Experiment 3: Split-CIFAR-100 with LwF
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Split-CIFAR-100 with LwF")
    print("=" * 70)

    model_lwf_cnn = SimpleCNN(num_classes=cfg.c100_classes_per_task)
    result_lwf, acc_lwf, bwt_lwf = train_lwf(
        model_lwf_cnn, train_loaders_c100, test_loaders_c100,
        num_tasks=cfg.c100_num_tasks, device=device,
        alpha=cfg.lwf_alpha, temperature=cfg.lwf_temperature,
        epochs=cfg.epochs_lwf, lr=cfg.lr,
    )
    print_result_matrix(result_lwf, "LwF")
    results_summary["LwF (CIFAR-100)"] = {"ACC": acc_lwf, "BWT": bwt_lwf}

    # =================================================================
    # Experiment 4: Permuted MNIST with DiLT-CL
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Permuted MNIST with DiLT-CL")
    print("=" * 70)

    pm_train_loaders, pm_test_loaders = get_permuted_mnist(
        num_tasks=cfg.pm_num_tasks, batch_size=cfg.pm_batch_size, seed=cfg.seed,
    )

    model_diltcl_mlp = SimpleMLP(input_size=784, hidden_size=256, num_classes=10)
    result_pm, acc_pm, bwt_pm = train_diltcl(
        model_diltcl_mlp, pm_train_loaders, pm_test_loaders,
        num_tasks=cfg.pm_num_tasks, device=device,
        num_partitions=cfg.diltcl_partitions,
        pruning_rounds=cfg.pruning_rounds,
        target_sparsity=cfg.diltcl_sparsity,
        full_train_epochs=cfg.diltcl_full_epochs,
        imp_epochs=cfg.diltcl_imp_epochs,
        lr=cfg.lr,
    )
    print_result_matrix(result_pm, "DiLT-CL (PM)")
    results_summary["DiLT-CL (Permuted MNIST)"] = {"ACC": acc_pm, "BWT": bwt_pm}

    # =================================================================
    # Experiment 5: Permuted MNIST with EWC and LwF
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Permuted MNIST with EWC and LwF")
    print("=" * 70)

    model_ewc_mlp = SimpleMLP(input_size=784, hidden_size=256, num_classes=10)
    result_pm_ewc, acc_pm_ewc, bwt_pm_ewc = train_ewc(
        model_ewc_mlp, pm_train_loaders, pm_test_loaders,
        num_tasks=cfg.pm_num_tasks, device=device,
        lambda_ewc=cfg.ewc_lambda, epochs=cfg.epochs_ewc, lr=cfg.lr,
    )
    results_summary["EWC (Permuted MNIST)"] = {"ACC": acc_pm_ewc, "BWT": bwt_pm_ewc}

    model_lwf_mlp = SimpleMLP(input_size=784, hidden_size=256, num_classes=10)
    result_pm_lwf, acc_pm_lwf, bwt_pm_lwf = train_lwf(
        model_lwf_mlp, pm_train_loaders, pm_test_loaders,
        num_tasks=cfg.pm_num_tasks, device=device,
        alpha=cfg.lwf_alpha, temperature=cfg.lwf_temperature,
        epochs=cfg.epochs_lwf, lr=cfg.lr,
    )
    results_summary["LwF (Permuted MNIST)"] = {"ACC": acc_pm_lwf, "BWT": bwt_pm_lwf}

    # =================================================================
    # Experiment 6: Gromov-Wasserstein Distance Analysis
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Gromov-Wasserstein Distance Analysis")
    print("=" * 70)

    # Wrapped broadly so a missing/incompatible POT install (or any GW error)
    # never aborts the suite or the results that follow.
    try:
        _run_gw_analysis(
            train_loaders_c100, test_loaders_c100, cfg, device, results_summary
        )
    except ImportError:
        print("[Warning] POT library not installed.")
        print("  Install with: pip install POT")
        print("  Skipping GW distance analysis.")
    except Exception as exc:  # noqa: BLE001 - defensive: GW is optional.
        print(f"[Warning] GW analysis skipped due to an error: {exc}")

    # =================================================================
    # Final Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<35} {'ACC (%)':>10} {'BWT (%)':>10}")
    print("-" * 70)
    for method, metrics in results_summary.items():
        if isinstance(metrics, dict) and "ACC" in metrics:
            print(f"{method:<35} {metrics['ACC']:>10.2f} "
                  f"{metrics.get('BWT', 0):>10.2f}")
        elif isinstance(metrics, float):
            print(f"{method:<35} {metrics:>10.6f}")
    print("=" * 70)

    np.savez(
        "diltcl_results.npz",
        diltcl_cifar=result_diltcl,
        ewc_cifar=result_ewc,
        lwf_cifar=result_lwf,
        diltcl_pm=result_pm,
        ewc_pm=result_pm_ewc,
        lwf_pm=result_pm_lwf,
        summary=results_summary,
    )
    print("\nResults saved to 'diltcl_results.npz'")
    print("\nAll experiments complete!")


def _run_gw_analysis(train_loaders_c100, test_loaders_c100, cfg, device,
                     results_summary):
    """Compute GW distances between task representations on Split-CIFAR-100.

    Trains a probe CNN briefly, extracts penultimate-layer representations for
    each task, and reports the Task-1-vs-last distance plus a pairwise matrix.
    Importing POT lazily here lets the caller treat ImportError as "skip".
    """
    import ot  # noqa: F401 - presence check; real use is inside compute_gw_distance.

    print("POT library is available. Computing GW distances...")

    num_tasks = cfg.c100_num_tasks
    print("\n--- GW Distance: Split-CIFAR-100 ---")
    model_for_gw = SimpleCNN(num_classes=cfg.c100_classes_per_task).to(device)
    optimizer = optim.Adam(model_for_gw.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    def _quick_train(loader, epochs=5):
        model_for_gw.train()
        for _ in range(epochs):
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model_for_gw(data), target)
                loss.backward()
                optimizer.step()

    _quick_train(train_loaders_c100[0])
    repr_t1 = extract_layer_representations(
        model_for_gw, test_loaders_c100[0], device=device, max_samples=200
    )

    _quick_train(train_loaders_c100[-1])
    repr_tlast = extract_layer_representations(
        model_for_gw, test_loaders_c100[-1], device=device, max_samples=200
    )

    gw_dist = compute_gw_distance(repr_t1, repr_tlast, reg=0.01)
    if gw_dist is not None:
        print(f"GW distance between Task 1 and Task {num_tasks}: {gw_dist:.6f}")
        results_summary["GW Distance (T1 vs T_last)"] = gw_dist

    print("\n--- Pairwise GW Distances ---")
    gw_matrix = np.zeros((num_tasks, num_tasks))
    for i in range(num_tasks):
        repr_i = extract_layer_representations(
            model_for_gw, test_loaders_c100[i], device=device, max_samples=150
        )
        for j in range(i + 1, num_tasks):
            repr_j = extract_layer_representations(
                model_for_gw, test_loaders_c100[j], device=device,
                max_samples=150
            )
            d = compute_gw_distance(repr_i, repr_j, reg=0.01)
            if d is not None:
                gw_matrix[i][j] = d
                gw_matrix[j][i] = d
                print(f"  GW(T{i + 1}, T{j + 1}) = {d:.6f}")


def run_quick_demo():
    """Run a tiny end-to-end demo (3 tasks, 2 epochs) to verify the pipeline.

    Uses simplified, illustrative core/expert masks rather than the full IMP
    procedure so it finishes in seconds.
    """
    print("\n*** DiLT-CL QUICK DEMO MODE ***\n")
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_loaders, test_loaders = get_split_cifar100(
        num_tasks=3, classes_per_task=5, batch_size=32, seed=42
    )

    print("\n--- DiLT-CL Demo ---")
    model = SimpleCNN(num_classes=5)
    diltcl = DiLTCL(model, device=device, num_partitions=2,
                    pruning_rounds=2, target_sparsity=0.8)

    # Phase 1: brief full-network training.
    print("Phase 1: Core extraction (2 epochs)...")
    diltcl._train_full(diltcl.model, train_loaders[0], epochs=2,
                       lr=0.001, verbose=True)
    diltcl.initial_params = {
        n: p.data.clone().detach()
        for n, p in diltcl.model.named_parameters() if p.requires_grad
    }

    # Simple illustrative core mask via median magnitude thresholding.
    core_mask = {}
    for n, p in diltcl.model.named_parameters():
        if p.requires_grad:
            threshold = p.data.abs().flatten().quantile(0.5)
            core_mask[n] = (p.data.abs() > threshold).float()
    diltcl.core_mask = core_mask
    diltcl.frozen_core_values = {
        n: p.data.clone().detach() * diltcl.core_mask[n]
        for n, p in diltcl.model.named_parameters() if p.requires_grad
    }

    acc = evaluate_with_mask(diltcl, test_loaders[0], 0, device)
    print(f"Task 1 accuracy: {acc:.2f}%")

    # Phase 2: random expert within the complement, then masked training.
    for tid in range(1, 3):
        print(f"\nPhase 2: Task {tid + 1} (2 epochs)...")
        complement = masking.complement_mask(diltcl.core_mask)
        expert_mask = {}
        for n in complement:
            expert_mask[n] = complement[n] * (
                torch.rand_like(complement[n]) > 0.5).float()
        diltcl.expert_masks[tid] = expert_mask

        combined = masking.combine_masks(diltcl.core_mask, expert_mask)
        diltcl._train_with_mask(diltcl.model, train_loaders[tid],
                                combined, epochs=2, lr=0.001)

        for eval_t in range(tid + 1):
            a = evaluate_with_mask(diltcl, test_loaders[eval_t], eval_t, device)
            print(f"  Task {eval_t + 1} accuracy: {a:.2f}%")

    print("\nDemo complete!")


def _parse_args():
    """Parse CLI flags, falling back to legacy environment variables."""
    parser = argparse.ArgumentParser(
        description="Run the DiLT-CL continual-learning experiment suite."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Reduced epochs / tasks for a quick run "
             "(equivalent to DILTCL_FAST=1).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Tiny 3-task smoke-test demo (equivalent to DILTCL_DEMO=1).",
    )
    args = parser.parse_args()

    # Environment variables preserve the original script's interface.
    args.fast = args.fast or os.environ.get("DILTCL_FAST", "0") == "1"
    args.demo = args.demo or os.environ.get("DILTCL_DEMO", "0") == "1"
    return args


if __name__ == "__main__":
    cli = _parse_args()
    if cli.demo:
        run_quick_demo()
    else:
        if cli.fast:
            print("\n*** FAST MODE ENABLED ***")
            print("Using reduced epochs and fewer tasks for quick testing.\n")
        main(ExperimentConfig.fast() if cli.fast else ExperimentConfig())
