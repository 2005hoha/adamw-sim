"""Regenerate E5 plots from results/summary/e5_stability.json (no retraining)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from src.plotting import plot_convergence


def main():
    path = "results/summary/e5_stability.json"
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

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

    print("E5 plots regenerated from", path)


if __name__ == "__main__":
    main()
