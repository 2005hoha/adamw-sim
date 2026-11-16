import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DEMO_DIR = os.path.join(RESULTS, "demo")
os.makedirs(DEMO_DIR, exist_ok=True)

BLUE = "#2a78d6"
RED = "#e34948"
GREEN = "#008300"
PURPLE = "#4a3aa7"
ORANGE = "#eda100"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.edgecolor": GRID,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": GRID,
})


def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def pick_best_run(candidates):
    best_steps = None
    best_loss = float("inf")
    best_data = None
    for path in candidates:
        if not os.path.exists(path):
            continue
        records = load_jsonl(path)
        if not records:
            continue
        loss_vals = np.array([r.get("loss_diff", r.get("val_loss", float("inf"))) for r in records])
        valid = loss_vals[~np.isnan(loss_vals) & ~np.isinf(loss_vals)]
        if len(valid) == 0:
            continue
        final_loss = valid[-1]
        if final_loss < best_loss:
            best_loss = final_loss
            best_steps = np.array([r["step"] for r in records])
            best_data = np.array([r.get("loss_diff", r.get("val_loss", np.nan)) for r in records])
    return best_steps, best_data


def make_e1_animation():
    raw = os.path.join(RESULTS, "raw", "e1_quadratic")
    kappas = ["kappa_1", "kappa_10", "kappa_1000", "kappa_10000"]
    kappa_labels = [r"$\kappa=1$", r"$\kappa=10$", r"$\kappa=10^3$", r"$\kappa=10^4$"]

    adamw_lrs = ["lr_0.01", "lr_0.003", "lr_0.001", "lr_0.0003"]
    sgd_lrs = ["lr_1.0", "lr_0.3", "lr_0.1", "lr_0.03", "lr_0.01", "lr_0.003", "lr_0.001", "lr_0.0003", "lr_0.0001"]

    lines_data = []
    for k_idx, k_dir in enumerate(kappas):
        k_path = os.path.join(raw, k_dir)
        adamw_candidates = [os.path.join(k_path, "adamw", lr, "seed_0", "metrics.jsonl") for lr in adamw_lrs]
        sgd_candidates = [os.path.join(k_path, "sgd", lr, "seed_0", "metrics.jsonl") for lr in sgd_lrs]

        aw_steps, aw_data = pick_best_run(adamw_candidates)
        sg_steps, sg_data = pick_best_run(sgd_candidates)
        lines_data.append((k_idx, aw_steps, aw_data, sg_steps, sg_data))

    max_len = max(max(len(d[2]) for d in lines_data if d[2] is not None),
                  max(len(d[4]) for d in lines_data if d[4] is not None))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    fig.suptitle("E1: AdamW vs SGD — Strongly Convex Quadratic Convergence", fontsize=13, y=0.98)

    lines_aw = [None] * 4
    lines_sg = [None] * 4
    aw_dots = [None] * 4
    sg_dots = [None] * 4
    texts = [None] * 4

    for i in range(4):
        ax = axes[i]
        ax.set_xlim(0, 5000)
        ax.set_yscale("log")
        kw = None
        for d in lines_data:
            if d[0] == i:
                kw = d
                break
        if kw is not None and kw[1] is not None:
            y_max = max(np.nanmax(kw[2]), np.nanmax(kw[4]) if kw[4] is not None else 0)
            y_min = min(np.nanmin(kw[2][kw[2] > 0]) if np.any(kw[2] > 0) else 1e-14,
                         np.nanmin(kw[4][kw[4] > 0]) if kw[4] is not None and np.any(kw[4] > 0) else 1e-14)
            ax.set_ylim(max(y_min * 0.5, 1e-16), y_max * 2)
        else:
            ax.set_ylim(1e-14, 1e5)
        ax.set_xlabel("Step")
        if i == 0 or i == 2:
            ax.set_ylabel(r"$f(\theta) - f^*$")
        ax.set_title(kappa_labels[i])
        lines_aw[i], = ax.plot([], [], color=BLUE, linewidth=1.5, label="AdamW (best LR)")
        lines_sg[i], = ax.plot([], [], color=RED, linewidth=1.5, label="SGD (best LR)")
        aw_dots[i], = ax.plot([], [], "o", color=BLUE, markersize=4, alpha=0.8)
        sg_dots[i], = ax.plot([], [], "s", color=RED, markersize=4, alpha=0.8)
        texts[i] = ax.text(0.98, 0.92, "", transform=ax.transAxes, ha="right", va="top", fontsize=8.5)

    axes[0].legend(loc="lower left", fontsize=8)

    frames_per_point = 3
    total_frames = max_len * frames_per_point

    def init():
        for i in range(4):
            lines_aw[i].set_data([], [])
            lines_sg[i].set_data([], [])
            aw_dots[i].set_data([], [])
            sg_dots[i].set_data([], [])
            texts[i].set_text("")
        return list(lines_aw) + list(lines_sg) + list(aw_dots) + list(sg_dots) + list(texts)

    def animate(frame):
        point_idx = (frame // frames_per_point) + 1
        point_idx = min(point_idx, max_len)
        artists = []
        for i in range(4):
            kw = None
            for d in lines_data:
                if d[0] == i:
                    kw = d
                    break
            if kw is None:
                continue

            aw_steps, aw_data = kw[1], kw[2]
            sg_steps, sg_data = kw[3], kw[4]

            if aw_steps is not None and point_idx <= len(aw_data):
                lines_aw[i].set_data(aw_steps[:point_idx], aw_data[:point_idx])
                aw_dots[i].set_data([aw_steps[point_idx - 1]], [aw_data[point_idx - 1]])
            if sg_steps is not None and sg_data is not None and point_idx <= len(sg_data):
                valid = ~np.isnan(sg_data[:point_idx]) & ~np.isinf(sg_data[:point_idx])
                if np.any(valid):
                    lines_sg[i].set_data(sg_steps[:point_idx][valid], sg_data[:point_idx][valid])
                    sg_dots[i].set_data([sg_steps[point_idx - 1]], [sg_data[point_idx - 1]])

            if aw_data is not None and point_idx <= len(aw_data):
                aw_val = aw_data[point_idx - 1]
            else:
                aw_val = np.nan
            if sg_data is not None and point_idx <= len(sg_data):
                sg_val = sg_data[point_idx - 1]
            else:
                sg_val = np.nan
            texts[i].set_text(f"step {point_idx}\nAdamW: {aw_val:.2e}\nSGD: {sg_val:.2e}")

            artists.extend([lines_aw[i], lines_sg[i], aw_dots[i], sg_dots[i], texts[i]])
        return artists

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=total_frames, interval=40, blit=True)
    path = os.path.join(DEMO_DIR, "e1_convergence.gif")
    anim.save(path, writer="pillow", fps=15, dpi=120)
    plt.close(fig)
    print(f"[demo] saved {path}")
    return path


def make_e4_animation():
    raw = os.path.join(RESULTS, "raw", "e4_tiny_gpt")

    adamw_path = os.path.join(raw, "adamw", "lr_0.003_wd_0.3", "seed_0", "metrics.jsonl")
    adamw_wd0_path = os.path.join(raw, "adamw", "lr_0.003_wd_0.0", "seed_0", "metrics.jsonl")
    adam_l2_path = os.path.join(raw, "adam_l2", "lr_0.003_wd_0.1", "seed_0", "metrics.jsonl")
    adam_l2_wd001_path = os.path.join(raw, "adam_l2", "lr_0.003_wd_0.01", "seed_0", "metrics.jsonl")

    aw = load_jsonl(adamw_path)
    aw0 = load_jsonl(adamw_wd0_path)
    al2_01 = load_jsonl(adam_l2_path)
    al2_001 = load_jsonl(adam_l2_wd001_path)

    def extract_steps_loss(records, key):
        steps = np.array([r["step"] for r in records])
        vals = np.array([r[key] for r in records])
        return steps, vals

    aw_steps, aw_val = extract_steps_loss(aw, "val_loss")
    _, aw_train = extract_steps_loss(aw, "train_loss")
    _, aw0_val = extract_steps_loss(aw0, "val_loss")
    _, aw0_train = extract_steps_loss(aw0, "train_loss")
    al2_01_steps, al2_01_val = extract_steps_loss(al2_01, "val_loss")
    _, al2_01_train = extract_steps_loss(al2_01, "train_loss")
    _, al2_001_val = extract_steps_loss(al2_001, "val_loss")
    _, al2_001_train = extract_steps_loss(al2_001, "train_loss")

    max_len = len(aw_steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("E4: TinyGPT Training — AdamW vs Adam-L2", fontsize=13, y=0.98)

    for ax in (ax1, ax2):
        ax.set_xlim(0, 2000)
        ax.set_xlabel("Step")

    ax1.set_ylabel("Cross-entropy")
    ax1.set_title("AdamW (best: lr=3e-3, wd=0.3)")
    ax2.set_title("Adam-L2 (lr=3e-3)")

    y_min = min(np.min(aw_train), np.min(aw0_train), np.min(aw_val))
    y_max = max(np.max(al2_01_train), np.max(al2_001_train), np.max(al2_01_val), np.max(al2_001_val))
    ax1.set_ylim(y_min - 0.05, y_max + 0.1)
    ax2.set_ylim(y_min - 0.05, y_max + 0.1)

    l1_train, = ax1.plot([], [], color=BLUE, linewidth=1.2, label="AdamW train (wd=0.3)")
    l1_val, = ax1.plot([], [], color=BLUE, linewidth=1.8, linestyle="--", label="AdamW val (wd=0.3)")
    l1_train0, = ax1.plot([], [], color="#7eb3ef", linewidth=0.8, alpha=0.6, label="AdamW train (wd=0)")
    l1_val0, = ax1.plot([], [], color="#7eb3ef", linewidth=1.2, linestyle="--", alpha=0.6, label="AdamW val (wd=0)")

    l2_train, = ax2.plot([], [], color=GREEN, linewidth=1.2, label="Adam-L2 train (wd=0.1)")
    l2_val, = ax2.plot([], [], color=GREEN, linewidth=1.8, linestyle="--", label="Adam-L2 val (wd=0.1)")
    l2_train001, = ax2.plot([], [], color="#6fbf6f", linewidth=0.8, alpha=0.6, label="Adam-L2 train (wd=0.01)")
    l2_val001, = ax2.plot([], [], color="#6fbf6f", linewidth=1.2, linestyle="--", alpha=0.6, label="Adam-L2 val (wd=0.01)")

    ax1.legend(loc="upper right", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    t1 = ax1.text(0.98, 0.04, "", transform=ax1.transAxes, ha="right", va="bottom", fontsize=8)
    t2 = ax2.text(0.98, 0.04, "", transform=ax2.transAxes, ha="right", va="bottom", fontsize=8)

    lines = [l1_train, l1_val, l1_train0, l1_val0, l2_train, l2_val, l2_train001, l2_val001, t1, t2]

    def init():
        for line in lines[:8]:
            line.set_data([], [])
        return lines

    def animate(frame):
        idx = min(frame + 1, max_len)
        l1_train.set_data(aw_steps[:idx], aw_train[:idx])
        l1_val.set_data(aw_steps[:idx], aw_val[:idx])
        l1_train0.set_data(aw_steps[:idx], aw0_train[:idx])
        l1_val0.set_data(aw_steps[:idx], aw0_val[:idx])
        l2_train.set_data(al2_01_steps[:idx], al2_01_train[:idx])
        l2_val.set_data(al2_01_steps[:idx], al2_01_val[:idx])
        l2_train001.set_data(al2_01_steps[:idx], al2_001_train[:idx])
        l2_val001.set_data(al2_01_steps[:idx], al2_001_val[:idx])
        t1.set_text(f"step={aw_steps[idx-1]}\nAdamW val: {aw_val[idx-1]:.3f}  (wd=0): {aw0_val[idx-1]:.3f}")
        t2.set_text(f"step={al2_01_steps[idx-1]}\nAdam-L2 val: {al2_01_val[idx-1]:.3f}  (wd=0.01): {al2_001_val[idx-1]:.3f}")
        return lines

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=max_len, interval=30, blit=True)
    path = os.path.join(DEMO_DIR, "e4_training.gif")
    anim.save(path, writer="pillow", fps=10, dpi=120)
    plt.close(fig)
    print(f"[demo] saved {path}")
    return path


def make_e5_animation():
    raw = os.path.join(RESULTS, "raw", "e5_stability")

    aw_path = os.path.join(raw, "adamw_lr3e-03_bs64_wd0.1", "seed_0", "metrics.jsonl")
    al2_path = os.path.join(raw, "adam_l2_lr3e-03_bs64_wd0.1", "seed_0", "metrics.jsonl")

    aw = load_jsonl(aw_path)
    al2 = load_jsonl(al2_path)

    aw_steps = np.array([r["step"] for r in aw])
    aw_cos = np.array([r["cos"] for r in aw])
    aw_train = np.array([r["train_loss"] for r in aw])
    aw_val = np.array([r["val_loss"] for r in aw])
    aw_update = np.array([r["update_norm"] for r in aw])

    al2_steps = np.array([r["step"] for r in al2])
    al2_cos = np.array([r["cos"] for r in al2])
    al2_train = np.array([r["train_loss"] for r in al2])
    al2_val = np.array([r["val_loss"] for r in al2])
    al2_update = np.array([r["update_norm"] for r in al2])

    max_len = len(aw_steps)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax_cos, ax_loss), (ax_update, ax_scatter) = axes
    fig.suptitle("E5: Stability Probe — AdamW vs Adam-L2 (lr=3e-3, wd=0.1, bs=64)", fontsize=13, y=0.98)

    ax_cos.set_xlim(0, 500)
    ax_cos.set_ylim(0, 0.5)
    ax_cos.set_xlabel("Step")
    ax_cos.set_ylabel(r"$\cos(\Delta\theta, -g)$")
    ax_cos.set_title("Update-direction alignment")
    ax_cos.axhline(y=0.05, color="#cccccc", linestyle=":", linewidth=0.8)

    ax_loss.set_xlim(0, 500)
    ax_loss.set_ylabel("Cross-entropy")
    ax_loss.set_xlabel("Step")
    ax_loss.set_title("Training dynamics")

    ax_update.set_xlim(0, 500)
    ax_update.set_xlabel("Step")
    ax_update.set_ylabel(r"$\|\Delta\theta\|_2$")
    ax_update.set_title("Update norm")

    ax_scatter.set_xlim(0.05, 0.45)
    ax_scatter.set_ylim(1.8, 2.8)
    ax_scatter.set_xlabel(r"$\cos(\Delta\theta, -g)$")
    ax_scatter.set_ylabel("Train CE")
    ax_scatter.set_title("Direction alignment vs loss")

    l_cos_aw, = ax_cos.plot([], [], color=BLUE, linewidth=1.8, label="AdamW")
    l_cos_al2, = ax_cos.plot([], [], color=GREEN, linewidth=1.8, linestyle="--", label="Adam-L2")
    cos_dot_aw, = ax_cos.plot([], [], "o", color=BLUE, markersize=5, alpha=0.8)
    cos_dot_al2, = ax_cos.plot([], [], "s", color=GREEN, markersize=5, alpha=0.8)

    l_train_aw, = ax_loss.plot([], [], color=BLUE, linewidth=1.2, label="AdamW train")
    l_val_aw, = ax_loss.plot([], [], color=BLUE, linewidth=1.5, linestyle="--", label="AdamW val")
    l_train_al2, = ax_loss.plot([], [], color=GREEN, linewidth=1.2, label="Adam-L2 train")
    l_val_al2, = ax_loss.plot([], [], color=GREEN, linewidth=1.5, linestyle="--", label="Adam-L2 val")

    l_up_aw, = ax_update.plot([], [], color=BLUE, linewidth=1.5, label="AdamW")
    l_up_al2, = ax_update.plot([], [], color=GREEN, linewidth=1.5, linestyle="--", label="Adam-L2")

    sc_aw = ax_scatter.scatter([], [], c=BLUE, s=20, alpha=0.7, label="AdamW")
    sc_al2 = ax_scatter.scatter([], [], c=GREEN, s=20, alpha=0.7, marker="s", label="Adam-L2")

    ax_cos.legend(loc="upper right", fontsize=8)
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_update.legend(loc="upper right", fontsize=8)
    ax_scatter.legend(loc="upper right", fontsize=8)

    ax_loss.set_ylim(1.5, 2.8)
    ax_update.set_ylim(0, max(np.max(aw_update), np.max(al2_update)) * 1.1)

    info = ax_scatter.text(0.98, 0.04, "", transform=ax_scatter.transAxes,
                           ha="right", va="bottom", fontsize=8)

    all_lines = [l_cos_aw, l_cos_al2, cos_dot_aw, cos_dot_al2,
                 l_train_aw, l_val_aw, l_train_al2, l_val_al2,
                 l_up_aw, l_up_al2, info]

    def init():
        for line in all_lines[:-1]:
            if hasattr(line, 'set_data'):
                line.set_data([], [])
        return all_lines

    def animate(frame):
        idx = min(frame + 1, max_len)

        l_cos_aw.set_data(aw_steps[:idx], aw_cos[:idx])
        l_cos_al2.set_data(al2_steps[:idx], al2_cos[:idx])
        cos_dot_aw.set_data([aw_steps[idx-1]], [aw_cos[idx-1]])
        cos_dot_al2.set_data([al2_steps[idx-1]], [al2_cos[idx-1]])

        l_train_aw.set_data(aw_steps[:idx], aw_train[:idx])
        l_val_aw.set_data(aw_steps[:idx], aw_val[:idx])
        l_train_al2.set_data(al2_steps[:idx], al2_train[:idx])
        l_val_al2.set_data(al2_steps[:idx], al2_val[:idx])

        l_up_aw.set_data(aw_steps[:idx], aw_update[:idx])
        l_up_al2.set_data(al2_steps[:idx], al2_update[:idx])

        sc_aw.set_offsets(np.column_stack([aw_cos[:idx], aw_train[:idx]]))
        sc_al2.set_offsets(np.column_stack([al2_cos[:idx], al2_train[:idx]]))

        info.set_text(f"step={aw_steps[idx-1]}\n"
                      f"AdamW cos={aw_cos[idx-1]:.3f}  val={aw_val[idx-1]:.3f}\n"
                      f"Adam-L2 cos={al2_cos[idx-1]:.3f}  val={al2_val[idx-1]:.3f}")
        return all_lines

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=max_len, interval=50, blit=True)
    path = os.path.join(DEMO_DIR, "e5_stability.gif")
    anim.save(path, writer="pillow", fps=8, dpi=120)
    plt.close(fig)
    print(f"[demo] saved {path}")
    return path


if __name__ == "__main__":
    print("Generating demo animations...")
    make_e1_animation()
    make_e4_animation()
    make_e5_animation()
    print(f"\nAll animations saved to {DEMO_DIR}/")
    print("  e1_convergence.gif  — AdamW vs SGD on quadratic problems")
    print("  e4_training.gif     — TinyGPT train/val curves, AdamW vs Adam-L2")
    print("  e5_stability.gif    — cos alignment & update norm comparison")