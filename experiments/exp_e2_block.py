import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
import math
from src.problems.block_quadratic import BlockQuadratic
from src.train import train_quadratic
from src.plotting import plot_bar_comparison, plot_convergence, METHOD_COLORS
from src.utils import set_seed


def run_e2(output_dir="results/raw/e2_block"):
    print("=" * 60)
    print("E2: Block Heterogeneity Experiment")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    dim = 100

    scenarios = {
        "heterogeneous": [1, 10, 100, 10000],
        "uniform": [1000, 1000, 1000, 1000],
    }

    adam_lr_grid = [3e-4, 1e-3, 3e-3, 1e-2]
    sgd_lr_grid = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]

    target_tolerance = 1e-4
    max_steps = 5000
    results = {}

    for scenario_name, kappas in scenarios.items():
        print(f"\n--- {scenario_name} ---")
        results[scenario_name] = {}

        best_lr_adam = None
        best_loss_adam = float("inf")
        best_log_adam = None
        steps_adam = max_steps

        for lr in adam_lr_grid:
            set_seed(0)
            problem = BlockQuadratic(dim=dim, num_blocks=4, kappas=kappas, seed=0, device=device)

            opt_cfg = {
                "name": "adam",
                "lr": lr,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "momentum": 0.0,
            }

            cfg = {
                "problem": {"dim": dim, "num_blocks": 4, "kappas": kappas, "seed": 0},
                "train": {"max_steps": max_steps, "eval_interval": 100},
                "optimizer": opt_cfg,
            }

            log_dir = os.path.join(output_dir, scenario_name, "adam", f"lr_{lr}", "seed_0")
            logger, summary = train_quadratic(problem, cfg, log_dir, seed=0)

            final_loss = summary["final_loss_diff"]
            if math.isfinite(final_loss) and final_loss < best_loss_adam:
                best_loss_adam = final_loss
                best_lr_adam = lr
                best_log_adam = log_dir
                for entry in logger.entries:
                    if entry["grad_norm"] < target_tolerance:
                        steps_adam = entry["step"]
                        break
                else:
                    steps_adam = max_steps

        results[scenario_name]["adam"] = {
            "best_lr": best_lr_adam,
            "steps_to_tol": steps_adam,
            "final_loss": best_loss_adam,
            "log_dir": best_log_adam,
        }
        print(f"  {'adam':6s}: lr={best_lr_adam:.0e}, steps_to_tol={steps_adam}, "
              f"final_loss={best_loss_adam:.6e}")

        best_lr_sgd = None
        best_loss_sgd = float("inf")
        best_log_sgd = None
        steps_sgd = max_steps

        for lr in sgd_lr_grid:
            set_seed(0)
            problem = BlockQuadratic(dim=dim, num_blocks=4, kappas=kappas, seed=0, device=device)

            opt_cfg = {
                "name": "sgd",
                "lr": lr,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "momentum": 0.9,
            }

            cfg = {
                "problem": {"dim": dim, "num_blocks": 4, "kappas": kappas, "seed": 0},
                "train": {"max_steps": max_steps, "eval_interval": 100},
                "optimizer": opt_cfg,
            }

            log_dir = os.path.join(output_dir, scenario_name, "sgd", f"lr_{lr}", "seed_0")
            logger, summary = train_quadratic(problem, cfg, log_dir, seed=0)

            final_loss = summary["final_loss_diff"]
            if math.isfinite(final_loss) and final_loss < best_loss_sgd:
                best_loss_sgd = final_loss
                best_lr_sgd = lr
                best_log_sgd = log_dir
                for entry in logger.entries:
                    if entry["grad_norm"] < target_tolerance:
                        steps_sgd = entry["step"]
                        break
                else:
                    steps_sgd = max_steps

        results[scenario_name]["sgd"] = {
            "best_lr": best_lr_sgd,
            "steps_to_tol": steps_sgd,
            "final_loss": best_loss_sgd,
            "log_dir": best_log_sgd,
        }

        if best_lr_sgd is not None:
            print(f"  {'sgd':6s}: lr={best_lr_sgd:.0e}, steps_to_tol={steps_sgd}, "
                  f"final_loss={best_loss_sgd:.6e}")
        else:
            print(f"  {'sgd':6s}: ALL LR DIVERGED (NaN)")

    print("\nGenerating plots...")

    for scenario_name in scenarios:
        runs = {}
        for opt_name in ["adam", "sgd"]:
            d = results[scenario_name].get(opt_name, {})
            if d.get("log_dir"):
                runs[opt_name] = [d["log_dir"]]

        plot_convergence(
            runs, "grad_norm", x_key="step",
            ylabel="||grad f||_2", title=f"Block Quadratic - {scenario_name}",
            save_path=f"results/figures/e2_{scenario_name}_grad.pdf",
            log_y=True,
        )

    # The old "steps to 1e-4" bar chart is kept for reference, but on this
    # problem both Adam and SGD hit max_steps, so it is not the primary
    # comparison. The final loss is the informative metric.
    labels = []
    step_values = []
    loss_values = []
    colors = []
    label_map = {
        "adam_hete": "Adam (hetero)",
        "sgd_hete": "SGD (hetero)",
        "adam_unif": "Adam (uniform)",
        "sgd_unif": "SGD (uniform)",
    }
    for scenario_name in scenarios:
        for opt_name in ["adam", "sgd"]:
            d = results[scenario_name].get(opt_name, {})
            label = f"{opt_name}_{scenario_name[:4]}"
            labels.append(label)
            step_values.append(d.get("steps_to_tol", max_steps))
            loss_values.append(d.get("final_loss", float("nan")))
            colors.append(METHOD_COLORS.get(opt_name, "#333333"))

    plot_bar_comparison(
        labels, step_values, colors=colors,
        ylabel=f"Steps to gradient tolerance {target_tolerance:g} (capped at {max_steps})",
        title="Block Heterogeneity: Steps to Tolerance",
        save_path="results/figures/e2_steps_to_tol.pdf",
        label_map=label_map, value_format=lambda v: f"{int(v)}",
    )

    plot_bar_comparison(
        labels, loss_values, colors=colors,
        ylabel="$f - f^*$ (lower is better)",
        title="Block Heterogeneity: Final Loss",
        save_path="results/figures/e2_final_loss.pdf",
        log_y=True, label_map=label_map,
    )

    summary_dir = "results/summary"
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "e2_block.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("E2 complete. Plots saved to results/figures/")
    return results


if __name__ == "__main__":
    run_e2()