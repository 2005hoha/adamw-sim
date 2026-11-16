"""Post-process existing E6 raw runs; does not retrain or add seeds."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import glob
import json
from src.plotting import plot_bar_comparison


def main():
    rows = []
    for path in sorted(glob.glob("results/raw/e6_ablation/*/seed_0/summary.json")):
        label = path.replace(os.sep, "/").split("/")[-3]
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        rows.append({
            "ablation": label,
            "final_train_loss": s["final_train_loss"],
            "final_val_loss": s["final_val_loss"],
            "best_val_loss": s["best_val_loss"],
            "wall_time_s": s["wall_time_s"],
            "group": "cross_optimizer" if label in ("adam_l2", "adam_no_wd") else "within_adamw",
        })

    if not rows:
        raise SystemExit("No E6 summaries found")

    os.makedirs("results/summary", exist_ok=True)
    csv_path = "results/summary/e6_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rows.sort(key=lambda r: r["final_val_loss"])
    print(f"wrote {csv_path} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['ablation']:16s} val={r['final_val_loss']:.4f}  [{r['group']}]")

    label_map = {
        "baseline": "Baseline",
        "no_decay_wd_0": "AdamW wd=0",
        "beta2_0.99": "beta2=0.99",
        "beta2_0.999": "beta2=0.999",
        "eps_1e-6": "eps=1e-6",
        "eps_1e-10": "eps=1e-10",
        "adam_l2": "Adam-L2 (coupled)",
        "adam_no_wd": "Adam wd=0",
        "no_warmup": "No warmup",
    }
    rows_by_label = {r["ablation"]: r for r in rows}

    all_labels = [r["ablation"] for r in rows]
    plot_bar_comparison(
        all_labels, [rows_by_label[l]["final_val_loss"] for l in all_labels],
        ylabel="Val CE", title="Ablation: Final Validation Loss",
        save_path="results/figures/e6_ablation.pdf",
        label_map=label_map, rotate_xticks=30, figsize=(9, 4.5),
    )

    # A separate within-AdamW chart. Mixing "coupled vs decoupled" (a
    # different optimizer) with hyperparameter ablations overstates the
    # former's ranking.
    within = [
        r["ablation"] for r in rows
        if rows_by_label[r["ablation"]]["group"] == "within_adamw"
    ]
    plot_bar_comparison(
        within, [rows_by_label[l]["final_val_loss"] for l in within],
        ylabel="Val CE", title="AdamW hyperparameter ablations",
        save_path="results/figures/e6_within_adamw.pdf",
        label_map=label_map, rotate_xticks=30, figsize=(9, 4.5),
    )
    print("wrote results/figures/e6_ablation.pdf and e6_within_adamw.pdf")


if __name__ == "__main__":
    main()
