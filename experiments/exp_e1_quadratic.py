import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import math
from src.problems.quadratic import Quadratic
from src.train import train_quadratic
from src.plotting import plot_convergence
from src.utils import set_seed


def run_e1(output_dir="results/raw/e1_quadratic"):
    print("=" * 60)
    print("E1: Strongly Convex Quadratic Convergence")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    dim = 100
    kappas = [1, 10, 1000, 10000]
    max_steps = 5000

    adam_lr_grid = [3e-4, 1e-3, 3e-3, 1e-2]
    sgd_lr_grid = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]
    adam_modes = ["adamw", "adam_l2", "adam"]

    results = {}

    for kappa in kappas:
        print(f"\n--- kappa = {kappa} ---")
        key = f"kappa_{kappa}"
        results[key] = {}

        for mode in adam_modes:
            best_lr = None
            best_loss = float("inf")
            best_log_dir = None

            for lr in adam_lr_grid:
                set_seed(0)
                problem = Quadratic(dim=dim, kappa=kappa, seed=0, device=device)

                opt_cfg = {
                    "name": mode,
                    "lr": lr,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "momentum": 0.0,
                }

                cfg = {
                    "problem": {"dim": dim, "kappa": kappa, "seed": 0},
                    "train": {"max_steps": max_steps, "eval_interval": 100},
                    "optimizer": opt_cfg,
                }

                log_dir = os.path.join(output_dir, key, mode, f"lr_{lr}", "seed_0")
                logger, summary = train_quadratic(problem, cfg, log_dir, seed=0)

                final_loss = summary["final_loss_diff"]
                if not math.isnan(final_loss) and final_loss < best_loss:
                    best_loss = final_loss
                    best_lr = lr
                    best_log_dir = log_dir

            results[key][mode] = best_log_dir
            print(f"  {mode:10s}: lr={best_lr:.0e}, final_loss={best_loss:.6e}")

        best_lr = None
        best_loss = float("inf")
        best_log_dir = None

        for lr in sgd_lr_grid:
            set_seed(0)
            problem = Quadratic(dim=dim, kappa=kappa, seed=0, device=device)

            opt_cfg = {
                "name": "sgd",
                "lr": lr,
                "momentum": 0.9,
                "weight_decay": 0.0,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
            }

            cfg = {
                "problem": {"dim": dim, "kappa": kappa, "seed": 0},
                "train": {"max_steps": max_steps, "eval_interval": 100},
                "optimizer": opt_cfg,
            }

            log_dir = os.path.join(output_dir, key, "sgd", f"lr_{lr}", "seed_0")
            logger, summary = train_quadratic(problem, cfg, log_dir, seed=0)

            final_loss = summary["final_loss_diff"]
            if not math.isnan(final_loss) and final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr
                best_log_dir = log_dir

        results[key]["sgd"] = best_log_dir
        if best_lr is not None:
            print(f"  {'sgd':10s}: lr={best_lr:.0e}, final_loss={best_loss:.6e}")
        else:
            print(f"  {'sgd':10s}: ALL LR DIVERGED (NaN)")

    print("\nGenerating plots...")
    for kappa in kappas:
        key = f"kappa_{kappa}"
        runs = {}
        for opt_name in adam_modes + ["sgd"]:
            d = results.get(key, {}).get(opt_name)
            if d:
                runs[opt_name] = [d]

        plot_convergence(
            runs, "loss_diff", x_key="step",
            ylabel="f(theta) - f*", title=f"Quadratic Convergence (kappa={kappa})",
            save_path=f"results/figures/e1_kappa_{kappa}_loss.pdf",
            log_y=True,
        )

        plot_convergence(
            runs, "grad_norm", x_key="step",
            ylabel="||grad f||_2", title=f"Gradient Norm (kappa={kappa})",
            save_path=f"results/figures/e1_kappa_{kappa}_grad.pdf",
            log_y=True,
        )

    print("E1 complete. Plots saved to results/figures/")
    return results


if __name__ == "__main__":
    run_e1()