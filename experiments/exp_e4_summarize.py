"""Post-process existing E4 raw runs; does not retrain or add seeds."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import glob
import json
from src.plotting import plot_convergence


def parse_run(path):
    parts = path.replace(os.sep, "/").split("/")
    opt = parts[-4]
    tag = parts[-3]
    seed = parts[-2]
    lr = float(tag.split("_")[1])
    wd = float(tag.split("_")[3])
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    return {
        "optimizer": opt,
        "lr": lr,
        "wd": wd,
        "seed": seed,
        "final_train_loss": s["final_train_loss"],
        "final_val_loss": s["final_val_loss"],
        "best_val_loss": s["best_val_loss"],
        "wall_time_s": s["wall_time_s"],
        "peak_mem_mb": s.get("peak_mem_mb"),
        "num_params": s.get("num_params"),
        "log_dir": os.path.dirname(path),
    }


def main():
    rows = [
        parse_run(p)
        for p in sorted(glob.glob("results/raw/e4_tiny_gpt/**/summary.json", recursive=True))
    ]
    if not rows:
        raise SystemExit("No E4 summaries found")

    os.makedirs("results/summary", exist_ok=True)
    csv_path = "results/summary/e4_tiny_gpt.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {csv_path} ({len(rows)} rows)")
    print("\nBest config per optimizer (by final val CE):")
    for opt in ["adamw", "adam_l2", "adam", "sgd"]:
        sub = [r for r in rows if r["optimizer"] == opt]
        if not sub:
            continue
        best = min(sub, key=lambda r: r["final_val_loss"])
        max_lr = max(r["lr"] for r in sub)
        max_wd = max(r["wd"] for r in sub)
        warning = []
        if best["lr"] == max_lr:
            warning.append("lr at grid upper bound")
        if best["wd"] == max_wd:
            warning.append("wd at grid upper bound")
        suffix = f"  WARNING: {', '.join(warning)}" if warning else ""
        print(f"  {opt:8s}: lr={best['lr']:.0e}, wd={best['wd']:.2f}, "
              f"val={best['final_val_loss']:.4f}{suffix}")

    # Fixed-wd comparison: this is the figure that shows the coupled-decay
    # collapse. The old "best config" plot picked Adam-L2 at wd=0, i.e. plain
    # Adam, so it did not actually show Adam-L2 at all.
    runs = {}
    for opt, lr, wd in [
        ("adamw", 3e-3, 0.1),
        ("adam_l2", 3e-3, 0.1),
        ("adam", 3e-3, 0.0),
        ("sgd", 1.0, 0.0),
    ]:
        hits = [
            r for r in rows
            if r["optimizer"] == opt and r["lr"] == lr and r["wd"] == wd
        ]
        if hits:
            runs[opt] = [hits[0]["log_dir"]]

    if runs:
        plot_convergence(
            runs, "val_loss", x_key="step", ylabel="CE Loss",
            title="Tiny GPT - fixed comparison",
            save_path="results/figures/e4_val_loss_wd0.1.pdf",
        )
        plot_convergence(
            runs, "train_loss", x_key="step", ylabel="CE Loss",
            title="Tiny GPT - fixed comparison",
            save_path="results/figures/e4_train_loss_wd0.1.pdf",
        )
        print("wrote results/figures/e4_val_loss_wd0.1.pdf")


if __name__ == "__main__":
    main()
