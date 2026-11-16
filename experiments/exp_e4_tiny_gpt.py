import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
from src.models.tiny_gpt import TinyGPT
from src.data.tinyshakespeare import get_tinyshakespeare_loaders
from src.train import train_one_run
from src.plotting import plot_convergence
from src.utils import set_seed, get_device


def run_e4(output_dir="results/raw/e4_tiny_gpt"):
    print("=" * 60)
    print("E4: Tiny GPT Main Experiment")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    with open("configs/tiny_gpt.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    device = get_device(base_cfg)
    print(f"Device: {device}")

    ts, train_loader, val_loader = get_tinyshakespeare_loaders(
        batch_size=base_cfg["train"]["batch_size"],
        block_size=base_cfg["model"]["block_size"],
        device=str(device),
    )

    base_cfg["model"]["vocab_size"] = ts.vocab_size

    optimizers = [
        ("adamw", [3e-4, 1e-3, 3e-3], [0.0, 0.01, 0.03, 0.1, 0.3]),
        ("adam_l2", [3e-4, 1e-3, 3e-3], [0.0, 0.01, 0.03, 0.1, 0.3]),
        ("adam", [3e-4, 1e-3, 3e-3], [0.0]),
        ("sgd", [0.1, 0.3, 1.0], [0.0, 0.01, 0.03]),
    ]

    results = {}

    for opt_name, lr_grid, wd_grid in optimizers:
        print(f"\n--- {opt_name} ---")
        results[opt_name] = {}
        best_val = float("inf")
        best_config = None

        for lr in lr_grid:
            for wd in wd_grid:
                set_seed(base_cfg["seed"])

                model = TinyGPT(
                    vocab_size=ts.vocab_size,
                    d_model=base_cfg["model"]["d_model"],
                    n_head=base_cfg["model"]["n_head"],
                    n_layer=base_cfg["model"]["n_layer"],
                    block_size=base_cfg["model"]["block_size"],
                    dropout=base_cfg["model"]["dropout"],
                ).to(device)

                opt_cfg = {
                    "name": opt_name,
                    "lr": lr,
                    "betas": base_cfg["optimizer"]["betas"],
                    "eps": base_cfg["optimizer"]["eps"],
                    "weight_decay": wd,
                    "momentum": 0.9,
                }

                cfg = {
                    "optimizer": opt_cfg,
                    "train": base_cfg["train"],
                    "device": str(device),
                }

                log_dir = os.path.join(output_dir, opt_name, f"lr_{lr}_wd_{wd}", "seed_0")
                logger, summary = train_one_run(model, train_loader, val_loader, cfg, log_dir, seed=base_cfg["seed"])

                final_val = summary["final_val_loss"]
                results[opt_name][f"lr={lr}_wd={wd}"] = {
                    "final_val_loss": final_val,
                    "best_val_loss": summary["best_val_loss"],
                    "log_dir": log_dir,
                }

                if final_val < best_val:
                    best_val = final_val
                    best_config = (lr, wd)

                print(f"  lr={lr:.0e}, wd={wd:.3f}: final_val={final_val:.4f}")

        print(f"  Best: lr={best_config[0]}, wd={best_config[1]}, val={best_val:.4f}")

    print("\nGenerating plots...")

    for metric in ["train_loss", "val_loss"]:
        runs = {}
        for opt_name, _, _ in optimizers:
            best_key = min(results[opt_name].items(), key=lambda x: x[1]["final_val_loss"])
            log_dir = best_key[1]["log_dir"]
            runs[opt_name] = [log_dir]

        plot_convergence(
            runs, metric, x_key="step",
            ylabel="CE Loss", title=f"Tiny GPT - {metric}",
            save_path=f"results/figures/e4_{metric}.pdf",
        )

    print("E4 complete. Plots saved to results/figures/")
    return results


if __name__ == "__main__":
    run_e4()