import os, json
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
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 120, "savefig.dpi": 120, "savefig.bbox": "tight",
    "font.family": "sans-serif", "axes.edgecolor": GRID,
    "axes.grid": False, "grid.alpha": 0.3, "grid.color": GRID,
})

KAPPA = 50.0
d = 2
theta_star = np.array([0.0, 0.0], dtype=np.float64)
L_max, L_min = KAPPA, 1.0
H = np.diag([L_max, L_min])
theta0 = np.array([-4.0, 8.0], dtype=np.float64)

def f(theta): return 0.5 * ((theta - theta_star) @ H @ (theta - theta_star))
def grad(theta): return H @ (theta - theta_star)

def run_optimizer(name, lr, steps):
    theta = theta0.copy().astype(np.float64)
    m, v = np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    wd = 0.0
    if name == "adamw": wd = 0.05
    if name == "adam_l2": wd = 0.05

    path = [theta.copy()]
    losses = [f(theta)]
    for t in range(1, steps + 1):
        g = grad(theta)
        if name in ("adam_l2", "sgd_l2"):
            g = g + wd * theta
        if name in ("sgd", "sgd_l2"):
            theta = theta - lr * g
        else:
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            if name in ("adamw",):
                theta = theta * (1 - lr * wd)
            theta = theta - update
        path.append(theta.copy())
        losses.append(f(theta))
    return np.array(path), np.array(losses)

def run_all(steps=300):
    configs = [
        ("sgd", 0.02, "SGD", RED, ":"),
        ("adam", 0.1, "Adam", PURPLE, "-."),
        ("adamw", 0.1, "AdamW", BLUE, "-"),
        ("adam_l2", 0.1, "Adam-L2", GREEN, "--"),
    ]
    all_paths, all_losses, all_names, all_colors, all_ls = [], [], [], [], []
    for name, lr, label, color, ls in configs:
        path, losses = run_optimizer(name, lr, steps)
        all_paths.append(path)
        all_losses.append(losses)
        all_names.append(label)
        all_colors.append(color)
        all_ls.append(ls)
    return all_paths, all_losses, all_names, all_colors, all_ls

def make_trajectory_animation():
    steps = 300
    paths, losses_list, names, colors, linestyles = run_all(steps)

    x_min = min(p[:, 0].min() for p in paths) - 0.5
    x_max = max(p[:, 0].max() for p in paths) + 0.5
    y_min = min(p[:, 1].min() for p in paths) - 0.5
    y_max = max(p[:, 1].max() for p in paths) + 0.5

    grid_x = np.linspace(x_min, x_max, 200)
    grid_y = np.linspace(y_min, y_max, 200)
    XX, YY = np.meshgrid(grid_x, grid_y)
    ZZ = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            ZZ[i, j] = f(np.array([XX[i, j], YY[i, j]]))
    levels = np.logspace(np.log10(max(ZZ.min(), 1e-6)), np.log10(ZZ.max()), 14)

    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.28)
    ax_traj = fig.add_subplot(gs[0])
    ax_loss = fig.add_subplot(gs[1])

    ax_traj.contour(XX, YY, ZZ, levels=levels, colors=GRID, linewidths=0.5, alpha=0.6)
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(y_min, y_max)
    ax_traj.set_xlabel(r"$\theta_1$ (high curvature)")
    ax_traj.set_ylabel(r"$\theta_2$ (low curvature)")
    ax_traj.set_title(f"Optimization trajectories on $\\kappa={int(KAPPA)}$ quadratic")
    ax_traj.plot(theta_star[0], theta_star[1], "*", color="#666666", markersize=12, zorder=5)
    ax_traj.annotate(r"$\theta^*$", (theta_star[0] + 0.15, theta_star[1] - 0.35), fontsize=9)
    ax_traj.plot(theta0[0], theta0[1], "X", color="#333333", markersize=8, zorder=5)
    ax_traj.annotate(r"$\theta_0$", (theta0[0] - 0.6, theta0[1] + 0.3), fontsize=9)

    ax_loss.set_xlim(0, steps)
    all_loss_vals = np.concatenate(losses_list)
    safe_floor = 1e-30
    valid = all_loss_vals[np.isfinite(all_loss_vals) & (all_loss_vals > safe_floor)]
    if len(valid) < 2:
        valid = np.array([1e-10, 1e5])
    ax_loss.set_yscale("log")
    ax_loss.set_ylim(1e-10, valid.max() * 2)
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel(r"$f(\theta) - f^*$")
    ax_loss.set_title("Loss convergence")

    traj_lines, traj_dots, loss_lines, loss_dots = [], [], [], []
    for i in range(4):
        tl, = ax_traj.plot([], [], color=colors[i], linestyle=linestyles[i],
                           linewidth=1.5, alpha=0.9, label=names[i])
        td, = ax_traj.plot([], [], "o", color=colors[i], markersize=6, zorder=10)
        ll, = ax_loss.plot([], [], color=colors[i], linestyle=linestyles[i],
                           linewidth=1.5, alpha=0.9, label=names[i])
        ld, = ax_loss.plot([], [], "D", color=colors[i], markersize=5, zorder=10)
        traj_lines.append(tl)
        traj_dots.append(td)
        loss_lines.append(ll)
        loss_dots.append(ld)

    ax_traj.legend(loc="upper right", fontsize=7)
    ax_loss.legend(loc="upper right", fontsize=7)
    info = ax_loss.text(0.98, 0.04, "", transform=ax_loss.transAxes,
                        ha="right", va="bottom", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    all_artists = traj_lines + traj_dots + loss_lines + loss_dots + [info]
    fig.suptitle("2D Quadratic Optimization — Trajectory & Loss Animation", fontsize=13, y=0.98)

    def init():
        for a in traj_lines + traj_dots + loss_lines + loss_dots:
            a.set_data([], [])
        info.set_text("")
        return all_artists

    def animate(frame):
        idx = min(frame + 1, steps)
        for i in range(4):
            traj_lines[i].set_data(paths[i][:idx+1, 0], paths[i][:idx+1, 1])
            traj_dots[i].set_data([paths[i][idx, 0]], [paths[i][idx, 1]])
            steps_arr = np.arange(idx + 1)
            loss_lines[i].set_data(steps_arr, losses_list[i][:idx+1])
            loss_dots[i].set_data([idx], [losses_list[i][idx]])
        texts = [f"{names[i]}: loss={losses_list[i][idx]:.2e}" for i in range(4)]
        info.set_text(f"step {idx}\n" + "\n".join(texts))
        return all_artists

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=steps, interval=50, blit=True)
    path = os.path.join(DEMO_DIR, "trajectory_2d.gif")
    anim.save(path, writer="pillow", fps=8, dpi=120)
    plt.close(fig)
    print(f"[demo] saved {path}")
    return path


if __name__ == "__main__":
    make_trajectory_animation()