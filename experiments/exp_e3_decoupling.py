import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import torch
import yaml
from src.problems.quadratic import Quadratic
from src.train import train_quadratic
from src.plotting import plot_heatmap
from src.utils import set_seed


def run_e3(output_dir="results/raw/e3_decoupling"):
    print("=" * 60)
    print("E3: Decoupled Weight Decay Grid")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    with open("configs/decoupling_grid.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cpu"))
    dim = cfg["problem"]["dim"]
    kappa = cfg["problem"]["kappa"]
    lr_grid = cfg["optimizer"]["lr_grid"]
    wd_grid = cfg["optimizer"]["wd_grid"]
    max_steps = cfg["train"]["max_steps"]
    eval_interval = cfg["train"].get("eval_interval", 500)
    modes = ["adam_l2", "adamw"]

    results = {}

    for mode in modes:
        print(f"\n--- {mode} ---")
        loss_matrix = np.full((len(wd_grid), len(lr_grid)), np.nan)
        reg_loss_matrix = np.full((len(wd_grid), len(lr_grid)), np.nan)
        theta_norm_matrix = np.full((len(wd_grid), len(lr_grid)), np.nan)
        grad_norm_matrix = np.full((len(wd_grid), len(lr_grid)), np.nan)
        converged_matrix = np.zeros((len(wd_grid), len(lr_grid)), dtype=bool)

        for wi, wd in enumerate(wd_grid):
            for li, lr in enumerate(lr_grid):
                set_seed(0)
                problem = Quadratic(dim=dim, kappa=kappa, seed=0, device=device)

                opt_cfg = {
                    "name": mode,
                    "lr": lr,
                    "betas": cfg["optimizer"]["betas"],
                    "eps": cfg["optimizer"]["eps"],
                    "weight_decay": wd,
                    "momentum": 0.0,
                }

                train_cfg = {
                    "max_steps": max_steps,
                    "eval_interval": eval_interval,
                }

                run_cfg = {
                    "problem": {"dim": dim, "kappa": kappa, "seed": 0},
                    "train": train_cfg,
                    "optimizer": opt_cfg,
                }

                log_dir = os.path.join(output_dir, mode, f"lr_{lr}_wd_{wd}", "seed_0")
                logger, summary = train_quadratic(problem, run_cfg, log_dir, seed=0)

                final_loss = summary["final_loss_diff"]
                final_reg = summary.get("final_reg_loss", np.nan)
                final_grad = summary.get("final_grad_norm", np.nan)
                final_theta_norm = summary.get("final_weight_norm", np.nan)
                # E3 is a sweep, not a tight-tolerance convergence test.
                # Mark a cell invalid only when the run diverged; the gradient
                # norm is reported alongside the loss for interpretation.
                converged = bool(
                    np.isfinite(final_loss)
                    and np.isfinite(final_grad)
                    and final_loss < 1e2
                )

                loss_matrix[wi, li] = final_loss
                reg_loss_matrix[wi, li] = final_reg
                theta_norm_matrix[wi, li] = final_theta_norm
                grad_norm_matrix[wi, li] = final_grad
                converged_matrix[wi, li] = converged

                status = "OK" if converged else "DIVERGED"
                print(f"  lr={lr:.0e}, wd={wd:.3f}: "
                      f"loss_diff={final_loss:.3e}, grad={final_grad:.3e} [{status}]")

        results[mode] = {
            "loss_matrix": loss_matrix,
            "reg_loss_matrix": reg_loss_matrix,
            "theta_norm_matrix": theta_norm_matrix,
            "grad_norm_matrix": grad_norm_matrix,
            "converged_matrix": converged_matrix,
        }

    print("\nGenerating heatmaps...")

    lr_labels = [f"{lr:.0e}" for lr in lr_grid]
    wd_labels = [f"{wd:.3f}" for wd in wd_grid]

    for mode in modes:
        loss_data = results[mode]["loss_matrix"]
        reg_data = results[mode]["reg_loss_matrix"]
        converged = results[mode]["converged_matrix"]

        # Mask unconverged cells so the color scale is not dominated by runs
        # that simply did not reach a stationary point.
        loss_plot = np.where(converged, loss_data, np.nan)
        reg_plot = np.where(converged, reg_data, np.nan)

        plot_heatmap(
            loss_plot, lr_labels, wd_labels,
            title=f"{mode} - unregularized loss gap",
            save_path=f"results/figures/e3_{mode}_heatmap.pdf",
            xlabel="Learning rate", ylabel="Weight decay",
            cbar_label="f(theta) - f* (lower is better)",
            log_norm=True,
        )
        plot_heatmap(
            reg_plot, lr_labels, wd_labels,
            title=f"{mode} - regularized objective",
            save_path=f"results/figures/e3_{mode}_reg_heatmap.pdf",
            xlabel="Learning rate", ylabel="Weight decay",
            cbar_label="f(theta) + (lambda/2)||theta||^2 (lower is better)",
            log_norm=True,
        )

    summary_dir = "results/summary"
    os.makedirs(summary_dir, exist_ok=True)
    payload = {
        "config": {
            "dim": dim,
            "kappa": kappa,
            "lr_grid": lr_grid,
            "wd_grid": wd_grid,
            "max_steps": max_steps,
        },
        "results": {
            mode: {
                "loss_matrix": results[mode]["loss_matrix"].tolist(),
                "reg_loss_matrix": results[mode]["reg_loss_matrix"].tolist(),
                "theta_norm_matrix": results[mode]["theta_norm_matrix"].tolist(),
                "grad_norm_matrix": results[mode]["grad_norm_matrix"].tolist(),
                "converged_matrix": results[mode]["converged_matrix"].astype(int).tolist(),
            }
            for mode in modes
        },
    }
    with open(os.path.join(summary_dir, "e3_decoupling.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("E3 complete. Plots saved to results/figures/")
    return results


if __name__ == "__main__":
    run_e3()
