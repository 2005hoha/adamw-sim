import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from src.plotting import plot_loglog_fit, load_metrics


def find_best_run(base_dir):
    """Return the E1 run with the smallest final f-f* under one optimizer."""
    if not os.path.isdir(base_dir):
        return None

    best_loss = float("inf")
    best_log_dir = None

    for name in os.listdir(base_dir):
        lr_dir = os.path.join(base_dir, name)
        if not os.path.isdir(lr_dir):
            continue
        seed_dir = os.path.join(lr_dir, "seed_0")
        summary_file = os.path.join(seed_dir, "summary.json")
        if not os.path.isfile(summary_file):
            continue

        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        final_loss = summary.get("final_loss_diff", float("inf"))
        if np.isfinite(final_loss) and final_loss < best_loss:
            best_loss = final_loss
            best_log_dir = seed_dir

    return best_log_dir


def fit_loglog_slope(log_dir, fit_start_ratio=0.5):
    """Fit log(||grad||) vs log(step) on the last part of a run.

    Returns a dict with slope, R^2, number of points, and the fit window.
    Returns None when there is no usable data.
    """
    entries = load_metrics(log_dir)
    if len(entries) < 3:
        return None

    xs = np.array([e["step"] for e in entries], dtype=float)
    ys = np.array([e["grad_norm"] for e in entries], dtype=float)

    fit_start = int(len(xs) * fit_start_ratio)
    x_fit = xs[fit_start:]
    y_fit = ys[fit_start:]

    valid = np.isfinite(x_fit) & np.isfinite(y_fit) & (x_fit > 0) & (y_fit > 1e-30)
    if valid.sum() < 3:
        return None

    x_fit = x_fit[valid]
    y_fit = y_fit[valid]
    log_x = np.log(x_fit)
    log_y = np.log(y_fit)
    coeffs = np.polyfit(log_x, log_y, 1)
    pred = np.polyval(coeffs, log_x)
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "slope": float(coeffs[0]),
        "r2": r2,
        "n_points": int(valid.sum()),
        "fit_start_step": int(x_fit[0]),
        "fit_end_step": int(x_fit[-1]),
    }


def run_e7(e1_raw_dir="results/raw/e1_quadratic",
           output_dir="results/raw/e7_convergence_rate"):
    print("=" * 60)
    print("E7: Convergence Rate Fitting")
    print("=" * 60)
    print("NOTE: E1 uses weight_decay=0, so Adam / Adam-L2 / AdamW")
    print("      are the same trajectory. E7 compares Adam family vs SGD.")

    os.makedirs(output_dir, exist_ok=True)

    kappas = [1, 10, 1000, 10000]
    optimizers = ["adamw", "adam_l2", "adam", "sgd"]
    results = {}

    for kappa in kappas:
        print(f"\n--- kappa = {kappa} ---")
        key = f"kappa_{kappa}"
        results[key] = {}

        for opt_name in optimizers:
            param_dir = os.path.join(e1_raw_dir, key, opt_name)
            best_log_dir = find_best_run(param_dir)
            if best_log_dir is None:
                results[key][opt_name] = {
                    "slope": None,
                    "reason": "no valid E1 run",
                }
                print(f"  {opt_name:10s}: no valid E1 run")
                continue

            fit = fit_loglog_slope(best_log_dir)
            if fit is None:
                results[key][opt_name] = {
                    "slope": None,
                    "reason": "insufficient valid points",
                    "log_dir": best_log_dir,
                }
                print(f"  {opt_name:10s}: insufficient valid data for fit")
            else:
                results[key][opt_name] = {
                    **fit,
                    "log_dir": best_log_dir,
                }
                print(f"  {opt_name:10s}: slope={fit['slope']:.4f}, "
                      f"R2={fit['r2']:.3f}, n={fit['n_points']}")

        # At wd=0 the three Adam-family entries must coincide. Collapse them
        # so the plot does not imply three independent experiments.
        adam_slopes = [
            results[key][name]["slope"]
            for name in ("adamw", "adam_l2", "adam")
            if results[key].get(name, {}).get("slope") is not None
        ]
        if len(adam_slopes) == 3 and max(adam_slopes) - min(adam_slopes) < 1e-9:
            results[key]["adam_family"] = {
                "slope": float(np.mean(adam_slopes)),
                "r2": results[key]["adamw"].get("r2"),
                "n_points": results[key]["adamw"].get("n_points"),
                "log_dir": results[key]["adamw"]["log_dir"],
                "note": "wd=0: Adam, Adam-L2 and AdamW coincide",
            }

    print("\nGenerating log-log plots...")
    for kappa in kappas:
        key = f"kappa_{kappa}"
        runs = {}
        if "adam_family" in results[key]:
            runs["adam_family"] = [results[key]["adam_family"]["log_dir"]]
        if results[key].get("sgd", {}).get("log_dir"):
            runs["sgd"] = [results[key]["sgd"]["log_dir"]]

        if runs:
            plot_loglog_fit(
                runs, metric_key="grad_norm", x_key="step",
                ylabel="||grad f||_2",
                title=f"Convergence Rate Fit (kappa={kappa})",
                save_path=f"results/figures/e7_kappa_{kappa}_loglog.pdf",
                color_map={"adam_family": "#4a3aa7", "sgd": "#e34948"},
                label_map={"adam_family": "Adam family (wd=0)", "sgd": "SGD"},
            )

    summary_dir = "results/summary"
    os.makedirs(summary_dir, exist_ok=True)
    payload = {
        "note": ("E1 uses weight_decay=0; Adam/Adam-L2/AdamW coincide. "
                 "E7 compares the Adam family against SGD, not AdamW against Adam-L2."),
        "kappas": kappas,
        "results": results,
    }
    with open(os.path.join(summary_dir, "e7_slopes.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("E7 complete. Plots saved to results/figures/")
    return results


if __name__ == "__main__":
    run_e7()
