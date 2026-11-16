import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS = {
    "adamw": "#2a78d6",
    "adam_l2": "#008300",
    "adam": "#4a3aa7",
    "sgd": "#e34948",
}

METHOD_LINESTYLES = {
    "adamw": "-",
    "adam_l2": "--",
    "adam": "-.",
    "sgd": ":",
}

METHOD_MARKERS = {
    "adamw": "o",
    "adam_l2": "s",
    "adam": "^",
    "sgd": "D",
}

METHOD_LABELS = {
    "adamw": "AdamW",
    "adam_l2": "Adam-$\\ell_2$",
    "adam": "Adam",
    "sgd": "SGD",
}

GRID_COLOR = "#e1e0d9"

# Eight-slot categorical palette used when a plot has more entities than the
# four optimizer names (E5 has one curve per LR/batch configuration).
CONFIG_COLORS = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
]


def set_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_metrics(log_dir):
    metrics_file = os.path.join(log_dir, "metrics.jsonl")
    entries = []
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def plot_convergence(runs, metric_key, x_key="step", ylabel=None, title=None,
                     save_path=None, log_y=False, color_map=None,
                     style_map=None, label_map=None, xlabel=None):
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    run_keys = list(runs.keys())

    for label, log_dirs in runs.items():
        if not log_dirs:
            continue
        all_x = []
        all_y = []
        for ld in log_dirs:
            entries = load_metrics(ld)
            if not entries:
                continue
            xs = [e[x_key] for e in entries]
            ys = [e[metric_key] for e in entries]
            all_x.append(xs)
            all_y.append(ys)

        if not all_x:
            continue
        min_len = min(len(x) for x in all_x)
        common_x = all_x[0][:min_len]
        stacked = np.array([y[:min_len] for y in all_y])
        mean_y = np.mean(stacked, axis=0)
        std_y = np.std(stacked, axis=0)

        color = (color_map or {}).get(label)
        if color is None:
            color = METHOD_COLORS.get(label)
        if color is None:
            color = CONFIG_COLORS[run_keys.index(label) % len(CONFIG_COLORS)]

        ls = (style_map or {}).get(label, METHOD_LINESTYLES.get(label, "-"))
        display_label = (label_map or {}).get(label, METHOD_LABELS.get(label, label))

        ax.plot(common_x, mean_y, color=color, linestyle=ls, linewidth=2,
                label=display_label)
        ax.fill_between(common_x, mean_y - std_y, mean_y + std_y,
                        color=color, alpha=0.15)

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel or ("Step" if x_key == "step" else x_key))
    ax.set_ylabel(ylabel or metric_key)
    if title:
        ax.set_title(title)
    ax.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format="pdf")
        fig.savefig(save_path.replace(".pdf", ".png"))
    plt.close(fig)
    return fig


def plot_heatmap(data, x_labels, y_labels, title=None, save_path=None,
                 vmin=None, vmax=None, xlabel=None, ylabel=None,
                 cbar_label="loss_diff (lower is better)", log_norm=False):
    set_style()
    fig, ax = plt.subplots(figsize=(5, 4.5))

    data_plot = np.asarray(data, dtype=float)
    data_plot = np.where(np.isfinite(data_plot), data_plot, np.nan)

    if log_norm:
        from matplotlib.colors import LogNorm
        positive = data_plot[np.isfinite(data_plot) & (data_plot > 0)]
        if positive.size == 0:
            log_norm = False
        else:
            if vmin is None:
                vmin = max(float(positive.min()), 1e-12)
            if vmax is None:
                vmax = float(positive.max())

    if log_norm:
        im = ax.imshow(data_plot, aspect="auto", origin="lower",
                       cmap=plt.cm.Blues, norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        im = ax.imshow(data_plot, aspect="auto", origin="lower",
                       cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = data_plot[i, j]
            if np.isfinite(val):
                if log_norm:
                    threshold = np.sqrt(vmin * vmax)
                else:
                    threshold = (vmax if vmax is not None else np.nanmax(data_plot)) * 0.6
                ax.text(j, i, f"{val:.3g}", ha="center", va="center",
                        fontsize=8, color="white" if val > threshold else "black")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format="pdf")
        fig.savefig(save_path.replace(".pdf", ".png"))
    plt.close(fig)
    return fig


def plot_bar_comparison(labels, values, colors=None, ylabel=None, title=None,
                        save_path=None, log_y=False, label_map=None,
                        value_format=None, rotate_xticks=0, figsize=(6.5, 4)):
    set_style()
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = [METHOD_COLORS.get(l, CONFIG_COLORS[i % len(CONFIG_COLORS)])
                  for i, l in enumerate(labels)]

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.5)

    for bar, val in zip(bars, values):
        if value_format is not None:
            text = value_format(val)
        elif val != 0 and (abs(val) < 1e-3 or abs(val) >= 1e3):
            text = f"{val:.2e}"
        else:
            text = f"{val:.4f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                text, ha="center", va="bottom", fontsize=8)

    if log_y:
        ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels([(label_map or {}).get(l, METHOD_LABELS.get(l, l)) for l in labels])
    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format="pdf")
        fig.savefig(save_path.replace(".pdf", ".png"))
    plt.close(fig)
    return fig


def plot_loglog_fit(runs, metric_key="grad_norm", x_key="step",
                    fit_range=None, ylabel=None, title=None, save_path=None,
                    color_map=None, label_map=None):
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for label, log_dirs in runs.items():
        if not log_dirs:
            continue
        entries = load_metrics(log_dirs[0])
        if not entries:
            continue
        xs = np.array([e[x_key] for e in entries], dtype=float)
        ys = np.array([e[metric_key] for e in entries], dtype=float)

        fit_start_val = fit_range[0] if fit_range and fit_range[0] is not None else len(xs) // 2
        fit_end_val = fit_range[1] if fit_range and fit_range[1] is not None else len(xs)

        idx = np.arange(fit_start_val, fit_end_val)
        valid = np.isfinite(xs[idx]) & np.isfinite(ys[idx]) & (xs[idx] > 0) & (ys[idx] > 1e-30)
        if valid.sum() < 3:
            continue
        idx = idx[valid]

        x_fit = np.log(xs[idx])
        y_fit = np.log(ys[idx])
        coeffs = np.polyfit(x_fit, y_fit, 1)
        slope = coeffs[0]

        color = (color_map or {}).get(label)
        if color is None:
            color = METHOD_COLORS.get(label, "#333333")
        display_label = (label_map or {}).get(label, METHOD_LABELS.get(label, label))
        display_label = f"{display_label} (slope={slope:.3f})"

        ax.loglog(xs, ys, color=color, linewidth=1.5, alpha=0.7)
        ax.loglog(xs[idx], np.exp(np.polyval(coeffs, x_fit)),
                  color=color, linestyle="--", linewidth=1, label=display_label)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel or metric_key)
    if title:
        ax.set_title(title)
    ax.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, format="pdf")
        fig.savefig(save_path.replace(".pdf", ".png"))
    plt.close(fig)
    return fig