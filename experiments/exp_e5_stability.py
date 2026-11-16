import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
import yaml
from src.models.tiny_gpt import TinyGPT
from src.data.tinyshakespeare import get_tinyshakespeare_loaders
from src.train import train_one_run
from src.plotting import plot_convergence
from src.utils import set_seed, get_device


def _token_matched_steps(batch_size, base_batch=64, base_steps=500):
    """Keep roughly the same number of sequences seen across batch sizes."""
    return max(50, int(round(base_steps * base_batch / batch_size)))


def run_e5(output_dir="results/raw/e5_stability"):
    print("=" * 60)
    print("E5: Stability Probe")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    with open("configs/stability.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    device = get_device(base_cfg)
    print(f"Device: {device}")

    configs = [
        # AdamW LR scan at bs=64, wd=0.1
        ("adamw", 3e-4, 64, 0.1),
        ("adamw", 1e-3, 64, 0.1),
        ("adamw", 3e-3, 64, 0.1),
        ("adamw", 1e-2, 64, 0.1),
        # Batch-size scan at lr=1e-3, wd=0.1, token-matched steps
        ("adamw", 1e-3, 128, 0.1),
        ("adamw", 1e-3, 256, 0.1),
        # Controlled weight-decay baselines at the same lr/batch
        ("adamw", 3e-3, 64, 0.0),
        ("adam_l2", 3e-3, 64, 0.1),
        ("adam_l2", 3e-3, 64, 0.0),
        ("adam", 3e-3, 64, 0.0),
    ]

    results = {}

    for opt_name, lr, batch_size, wd in configs:
        label = f"{opt_name}_lr{lr:.0e}_bs{batch_size}_wd{wd:.2g}"
        print(f"\n--- {label} ---")

        set_seed(0)

        ts, cur_train_loader, cur_val_loader = get_tinyshakespeare_loaders(
            batch_size=batch_size,
            block_size=base_cfg["model"]["block_size"],
            device=str(device),
        )

        model = TinyGPT(
            vocab_size=ts.vocab_size,
            d_model=base_cfg["model"]["d_model"],
            n_head=base_cfg["model"]["n_head"],
            n_layer=base_cfg["model"]["n_layer"],
            block_size=base_cfg["model"]["block_size"],
            dropout=base_cfg["model"]["dropout"],
        ).to(device)

        steps = _token_matched_steps(batch_size)
        warmup = max(1, int(round(base_cfg["train"].get("warmup_steps", 50) * steps / 500)))
        eval_interval = max(10, steps // 10)

        opt_cfg = {
            "name": opt_name,
            "lr": lr,
            "betas": base_cfg["optimizer"]["betas"],
            "eps": base_cfg["optimizer"]["eps"],
            "weight_decay": wd,
            "momentum": 0.9,
        }

        train_cfg = {
            **base_cfg["train"],
            "batch_size": batch_size,
            "max_steps": steps,
            "warmup_steps": warmup,
            "eval_interval": eval_interval,
        }

        cfg = {
            "optimizer": opt_cfg,
            "train": train_cfg,
            "device": str(device),
        }

        log_dir = os.path.join(output_dir, label, "seed_0")
        logger, summary = train_one_run(
            model, cur_train_loader, cur_val_loader, cfg, log_dir, seed=0
        )

        cos_values = [e.get("cos", float("nan")) for e in logger.entries]
        update_norms = [e.get("update_norm", float("nan")) for e in logger.entries]
        loss_spikes = 0
        prev_val = None
        for e in logger.entries:
            if prev_val is not None and e["val_loss"] > prev_val + 0.5:
                loss_spikes += 1
            prev_val = e["val_loss"]

        results[label] = {
            "log_dir": log_dir,
            "optimizer": opt_name,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": wd,
            "steps": steps,
            "tokens_seen": logger.entries[-1].get("tokens_seen") if logger.entries else None,
            "final_cos": cos_values[-1] if cos_values else float("nan"),
            "min_cos": min(cos_values) if cos_values else float("nan"),
            "final_update_norm": update_norms[-1] if update_norms else float("nan"),
            "loss_spikes": loss_spikes,
            "final_val_loss": summary["final_val_loss"],
        }

        print(f"  steps={steps}, cos={results[label]['final_cos']:.4f}, "
              f"min_cos={results[label]['min_cos']:.4f}, "
              f"update_norm={results[label]['final_update_norm']:.4f}, "
              f"spikes={loss_spikes}")

    print("\nGenerating plots...")

    # 1) AdamW learning-rate scan at bs=64, wd=0.1
    lr_labels = [
        f"adamw_lr{lr:.0e}_bs64_wd0.1"
        for lr in [3e-4, 1e-3, 3e-3, 1e-2]
    ]
    lr_runs = {label: [results[label]["log_dir"]] for label in lr_labels if label in results}
    lr_colors = {
        lr_labels[0]: "#86b6ef",
        lr_labels[1]: "#3987e5",
        lr_labels[2]: "#256abf",
        lr_labels[3]: "#0d366b",
    }
    lr_label_map = {
        label: f"AdamW lr={lr:.0e}"
        for label, lr in zip(lr_labels, [3e-4, 1e-3, 3e-3, 1e-2])
    }
    for metric, fname, ylabel in [
        ("cos", "e5_adamw_lr_cos.pdf", "cos(Delta_theta, -g)"),
        ("update_norm", "e5_adamw_lr_update.pdf", "||Delta_theta||_2"),
    ]:
        plot_convergence(
            lr_runs, metric, x_key="step", ylabel=ylabel,
            title="AdamW LR scan (bs=64, wd=0.1)",
            save_path=f"results/figures/{fname}",
            color_map=lr_colors, label_map=lr_label_map,
        )

    # 2) Batch-size scan at lr=1e-3, wd=0.1 (token-matched steps)
    batch_labels = [
        "adamw_lr1e-03_bs64_wd0.1",
        "adamw_lr1e-03_bs128_wd0.1",
        "adamw_lr1e-03_bs256_wd0.1",
    ]
    batch_runs = {
        label: [results[label]["log_dir"]] for label in batch_labels if label in results
    }
    batch_colors = {
        batch_labels[0]: "#2a78d6",
        batch_labels[1]: "#1baf7a",
        batch_labels[2]: "#4a3aa7",
    }
    batch_label_map = {
        batch_labels[0]: "bs=64",
        batch_labels[1]: "bs=128",
        batch_labels[2]: "bs=256",
    }
    for metric, fname, ylabel in [
        ("cos", "e5_batch_cos.pdf", "cos(Delta_theta, -g)"),
        ("update_norm", "e5_batch_update.pdf", "||Delta_theta||_2"),
    ]:
        plot_convergence(
            batch_runs, metric, x_key="tokens_seen", xlabel="Tokens seen", ylabel=ylabel,
            title="Batch-size scan (AdamW, lr=1e-3, wd=0.1, token-matched)",
            save_path=f"results/figures/{fname}",
            color_map=batch_colors, label_map=batch_label_map,
        )

    # 3) Controlled method comparison at the same lr/batch
    method_labels = [
        "adamw_lr3e-03_bs64_wd0.1",
        "adamw_lr3e-03_bs64_wd0",
        "adam_l2_lr3e-03_bs64_wd0.1",
        "adam_l2_lr3e-03_bs64_wd0",
        "adam_lr3e-03_bs64_wd0",
    ]
    method_runs = {
        label: [results[label]["log_dir"]] for label in method_labels if label in results
    }
    method_colors = {
        method_labels[0]: "#2a78d6",
        method_labels[1]: "#86b6ef",
        method_labels[2]: "#008300",
        method_labels[3]: "#6dbf6d",
        method_labels[4]: "#4a3aa7",
    }
    method_label_map = {
        method_labels[0]: "AdamW (wd=0.1)",
        method_labels[1]: "AdamW (wd=0)",
        method_labels[2]: "Adam-L2 (wd=0.1)",
        method_labels[3]: "Adam-L2 (wd=0)",
        method_labels[4]: "Adam (wd=0)",
    }
    for metric, fname, ylabel in [
        ("cos", "e5_cos.pdf", "cos(Delta_theta, -g)"),
        ("update_norm", "e5_update_norm.pdf", "||Delta_theta||_2"),
    ]:
        plot_convergence(
            method_runs, metric, x_key="step", ylabel=ylabel,
            title="Method comparison (lr=3e-3, bs=64)",
            save_path=f"results/figures/{fname}",
            color_map=method_colors, label_map=method_label_map,
        )

    summary_dir = "results/summary"
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "e5_stability.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("E5 complete. Plots saved to results/figures/")
    return results


if __name__ == "__main__":
    run_e5()
