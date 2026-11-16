import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
from src.models.tiny_gpt import TinyGPT
from src.data.tinyshakespeare import get_tinyshakespeare_loaders
from src.train import train_one_run
from src.plotting import plot_bar_comparison, METHOD_COLORS
from src.utils import set_seed, get_device


def run_e6(output_dir="results/raw/e6_ablation"):
    print("=" * 60)
    print("E6: Ablation Study")
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
    base_cfg["train"]["max_steps"] = 1000
    base_cfg["train"]["eval_interval"] = 250

    ablations = {}

    set_seed(0)
    model = TinyGPT(
        vocab_size=ts.vocab_size,
        d_model=base_cfg["model"]["d_model"],
        n_head=base_cfg["model"]["n_head"],
        n_layer=base_cfg["model"]["n_layer"],
        block_size=base_cfg["model"]["block_size"],
        dropout=base_cfg["model"]["dropout"],
    ).to(device)

    cfg_baseline = {
        "optimizer": {
            "name": "adamw",
            "lr": 1e-3,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
            "momentum": 0.0,
        },
        "train": base_cfg["train"],
        "device": str(device),
    }

    log_dir = os.path.join(output_dir, "baseline", "seed_0")
    _, summary = train_one_run(model, train_loader, val_loader, cfg_baseline, log_dir, seed=0)
    ablations["baseline"] = {"val_loss": summary["final_val_loss"], "log_dir": log_dir}
    print(f"  baseline: val_loss={summary['final_val_loss']:.4f}")

    ablation_configs = [
        ("no_decay_wd_0", "adamw", 1e-3, 0.0, [0.9, 0.95], 1e-8),
        ("beta2_0.99", "adamw", 1e-3, 0.1, [0.9, 0.99], 1e-8),
        ("beta2_0.999", "adamw", 1e-3, 0.1, [0.9, 0.999], 1e-8),
        ("eps_1e-6", "adamw", 1e-3, 0.1, [0.9, 0.95], 1e-6),
        ("eps_1e-10", "adamw", 1e-3, 0.1, [0.9, 0.95], 1e-10),
        ("adam_l2", "adam_l2", 1e-3, 0.1, [0.9, 0.95], 1e-8),
        ("adam_no_wd", "adam", 1e-3, 0.0, [0.9, 0.95], 1e-8),
        ("no_warmup", "adamw", 1e-3, 0.1, [0.9, 0.95], 1e-8),
    ]

    for label, opt_name, lr, wd, betas, eps in ablation_configs:
        set_seed(0)
        model = TinyGPT(
            vocab_size=ts.vocab_size,
            d_model=base_cfg["model"]["d_model"],
            n_head=base_cfg["model"]["n_head"],
            n_layer=base_cfg["model"]["n_layer"],
            block_size=base_cfg["model"]["block_size"],
            dropout=base_cfg["model"]["dropout"],
        ).to(device)

        train_cfg = dict(base_cfg["train"])
        if label == "no_warmup":
            train_cfg["warmup_steps"] = 0

        cfg = {
            "optimizer": {
                "name": opt_name,
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": wd,
                "momentum": 0.0,
            },
            "train": train_cfg,
            "device": str(device),
        }

        log_dir = os.path.join(output_dir, label, "seed_0")
        _, summary = train_one_run(model, train_loader, val_loader, cfg, log_dir, seed=0)
        ablations[label] = {"val_loss": summary["final_val_loss"], "log_dir": log_dir}
        print(f"  {label}: val_loss={summary['final_val_loss']:.4f}")

    print("\nGenerating ablation plot...")
    labels = list(ablations.keys())
    values = [ablations[l]["val_loss"] for l in labels]

    ablation_opt_map = {
        "baseline": "adamw", "no_decay_wd_0": "adamw",
        "beta2_0.99": "adamw", "beta2_0.999": "adamw",
        "eps_1e-6": "adamw", "eps_1e-10": "adamw",
        "adam_l2": "adam_l2", "adam_no_wd": "adam",
        "no_warmup": "adamw",
    }
    colors = [METHOD_COLORS.get(ablation_opt_map.get(l, ""), "#333333") for l in labels]

    plot_bar_comparison(
        labels, values, colors=colors,
        ylabel="Val CE", title="Ablation: Final Validation Loss",
        save_path="results/figures/e6_ablation.pdf",
    )

    print("E6 complete. Plots saved to results/figures/")
    return ablations


if __name__ == "__main__":
    run_e6()