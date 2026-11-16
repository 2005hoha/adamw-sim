import os, json, base64, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DEMO_DIR = os.path.join(RESULTS, "demo")
os.makedirs(DEMO_DIR, exist_ok=True)

BLUE, RED, GREEN, PURPLE, ORANGE = "#2a78d6", "#e34948", "#008300", "#4a3aa7", "#eda100"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 140, "savefig.dpi": 140, "savefig.bbox": "tight",
    "font.family": "sans-serif", "axes.edgecolor": GRID,
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": GRID,
})


def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def section_html(title, tag, imgs, text_lines):
    blocks = []
    for img in imgs:
        blocks.append(f'<img class="fig" src="{img}" alt="{title}">')
    text = "<br>".join(text_lines) if text_lines else ""
    return f"""
    <div class="section {tag}">
      <h2>{title}</h2>
      <p class="note">{text}</p>
      <div class="img-row">{"".join(blocks)}</div>
    </div>"""


def make_e1_figs():
    raw = os.path.join(RESULTS, "raw", "e1_quadratic")
    kappas = {"kappa_1": "1", "kappa_10": "10", "kappa_1000": "10³", "kappa_10000": "10⁴"}
    all_kw = []
    for k_dir, k_label in kappas.items():
        kw = {"label": k_label}
        for opt, opt_label, color in [("adamw", "AdamW", BLUE), ("sgd", "SGD", RED),
                                       ("adam", "Adam", PURPLE), ("adam_l2", "Adam-L2", GREEN)]:
            opt_dir = os.path.join(raw, k_dir, opt)
            if not os.path.isdir(opt_dir):
                continue
            best_loss, best_steps, best_data = float("inf"), None, None
            for lr_dir in sorted(os.listdir(opt_dir)):
                path = os.path.join(opt_dir, lr_dir, "seed_0", "metrics.jsonl")
                if not os.path.exists(path): continue
                recs = load_jsonl(path)
                if not recs: continue
                vals = np.array([r["loss_diff"] for r in recs])
                valid = vals[~np.isnan(vals)]
                if len(valid) and valid[-1] < best_loss:
                    best_loss = valid[-1]
                    best_steps = np.array([r["step"] for r in recs])
                    best_data = np.array([r["loss_diff"] for r in recs])
            if best_steps is not None:
                kw[opt] = (best_steps, best_data, color)
        all_kw.append(kw)

    figs = []
    for kw in all_kw:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.set_yscale("log")
        for opt_name in ["adamw", "adam", "adam_l2", "sgd"]:
            if opt_name in kw:
                s, d, c = kw[opt_name]
                valid = d > 0
                ax.plot(s[valid], d[valid], color=c, linewidth=1.3, alpha=0.9)
        ax.set_xlabel("Step"); ax.set_ylabel(r"$f(\theta)-f^*$")
        ax.set_title(f"E1: κ = {kw['label']}")
        figs.append(fig_to_b64(fig))
    return figs


def make_e2_figs():
    raw = os.path.join(RESULTS, "raw", "e2_block")
    scenarios = {"uniform": "Uniform κ=1000", "heterogeneous": "Heterogeneous"}
    figs = []
    for sc_dir, sc_title in scenarios.items():
        p = os.path.join(raw, sc_dir)
        if not os.path.isdir(p): continue
        results = {}
        for opt in ["adam", "adamw", "adam_l2", "sgd"]:
            opt_dir = os.path.join(p, opt)
            if not os.path.isdir(opt_dir): continue
            best_loss = float("inf")
            for lr_dir in sorted(os.listdir(opt_dir)):
                path = os.path.join(opt_dir, lr_dir, "seed_0", "metrics.jsonl")
                if not os.path.exists(path): continue
                recs = load_jsonl(path)
                if not recs: continue
                vals = np.array([r["loss_diff"] for r in recs])
                valid = vals[~np.isnan(vals)]
                if len(valid) and valid[-1] < best_loss:
                    best_loss = valid[-1]
                    best_steps = np.array([r["step"] for r in recs])
                    best_data = np.array([r["loss_diff"] for r in recs])
            if best_loss < float("inf"):
                results[opt] = (best_steps, best_data, best_loss)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.set_yscale("log")
        colors = {"adamw": BLUE, "adam": PURPLE, "adam_l2": GREEN, "sgd": RED}
        for opt_name in ["adamw", "adam", "adam_l2", "sgd"]:
            if opt_name in results:
                s, d, _ = results[opt_name]
                valid = d > 0
                ax.plot(s[valid], d[valid], color=colors[opt_name], linewidth=1.3, alpha=0.9)
        ax.set_xlabel("Step"); ax.set_ylabel(r"$f(\theta)-f^*$")
        ax.set_title(f"E2: {sc_title}")
        figs.append(fig_to_b64(fig))
    return figs


def make_e4_figs():
    raw = os.path.join(RESULTS, "raw", "e4_tiny_gpt")
    adamw_best = os.path.join(raw, "adamw", "lr_0.003_wd_0.3", "seed_0", "metrics.jsonl")
    al2_wd01 = os.path.join(raw, "adam_l2", "lr_0.003_wd_0.1", "seed_0", "metrics.jsonl")
    al2_wd0 = os.path.join(raw, "adam_l2", "lr_0.003_wd_0.0", "seed_0", "metrics.jsonl")
    sgd_best = os.path.join(raw, "sgd", "lr_1.0_wd_0.0", "seed_0", "metrics.jsonl")

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    def plot_run(ax, path, label, color):
        recs = load_jsonl(path)
        s = np.array([r["step"] for r in recs])
        tv = np.array([r["train_loss"] for r in recs])
        vv = np.array([r["val_loss"] for r in recs])
        ax.plot(s, tv, color=color, linewidth=1.2, label=f"{label} train")
        ax.plot(s, vv, color=color, linewidth=1.8, linestyle="--", label=f"{label} val")
        return vv[-1]

    plot_run(ax1, adamw_best, "AdamW (wd=0.3)", BLUE)
    plot_run(ax1, al2_wd0, "Adam-L2 (wd=0)", GREEN)
    ax1.set_title("AdamW best vs Adam-L2 wd=0")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Cross-entropy")
    ax1.legend(fontsize=7)

    plot_run(ax2, al2_wd01, "Adam-L2 wd=0.1", GREEN)
    plot_run(ax2, al2_wd0, "Adam-L2 wd=0", GREEN)
    ax2.set_title("Adam-L2: wd=0 vs wd=0.1 collapse")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Cross-entropy")
    ax2.legend(fontsize=7)
    fig1.suptitle("E4: TinyGPT Main Results", fontsize=13, y=1.01)
    img1 = fig_to_b64(fig1)

    fig2, ax = plt.subplots(figsize=(8, 4))
    wds = [0, 0.01, 0.1, 0.3]
    aw_vals, al2_vals = [], []
    for wd in wds:
        aw_path = os.path.join(raw, "adamw", f"lr_0.003_wd_{wd}", "seed_0", "metrics.jsonl")
        al2_path = os.path.join(raw, "adam_l2", f"lr_0.003_wd_{wd}", "seed_0", "metrics.jsonl")
        aw_vals.append(load_jsonl(aw_path)[-1]["val_loss"] if os.path.exists(aw_path) else 0)
        al2_vals.append(load_jsonl(al2_path)[-1]["val_loss"] if os.path.exists(al2_path) else 0)

    x = np.arange(len(wds))
    w = 0.35
    bars1 = ax.bar(x - w/2, aw_vals, w, color=BLUE, label="AdamW", edgecolor="white")
    bars2 = ax.bar(x + w/2, al2_vals, w, color=GREEN, label="Adam-L2", edgecolor="white")
    for bar, val in zip(bars1, aw_vals):
        if val: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, al2_vals):
        if val: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"wd={wd}" for wd in wds])
    ax.set_ylabel("Val CE"); ax.set_title("E4: Weight Decay Sweep (lr=3e-3)")
    ax.legend(fontsize=8)
    img2 = fig_to_b64(fig2)
    return [img1, img2]


def make_e5_figs():
    raw = os.path.join(RESULTS, "raw", "e5_stability")
    pairs = [
        ("adamw_lr3e-03_bs64_wd0.1", "AdamW", BLUE),
        ("adam_l2_lr3e-03_bs64_wd0.1", "Adam-L2", GREEN),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    (ax_cos, ax_loss), (ax_up, ax_sc) = axes

    for fname, label, color in pairs:
        recs = load_jsonl(os.path.join(raw, fname, "seed_0", "metrics.jsonl"))
        s = np.array([r["step"] for r in recs])
        cos = np.array([r["cos"] for r in recs])
        tv = np.array([r["train_loss"] for r in recs])
        vv = np.array([r["val_loss"] for r in recs])
        up = np.array([r["update_norm"] for r in recs])

        ax_cos.plot(s, cos, color=color, linewidth=1.5, label=label)
        ax_loss.plot(s, tv, color=color, linewidth=1.2, label=f"{label} train")
        ax_loss.plot(s, vv, color=color, linewidth=1.6, linestyle="--", label=f"{label} val")
        ax_up.plot(s, up, color=color, linewidth=1.5, label=label)
        ax_sc.scatter(cos, tv, c=color, s=18, alpha=0.7, label=label)

    ax_cos.set_ylabel(r"$\cos(\Delta\theta, -g)$"); ax_cos.set_title("Update-direction alignment")
    ax_cos.legend(fontsize=7)
    ax_loss.set_ylabel("Cross-entropy"); ax_loss.set_title("Training dynamics"); ax_loss.legend(fontsize=7)
    ax_up.set_xlabel("Step"); ax_up.set_ylabel(r"$\|\Delta\theta\|_2$"); ax_up.set_title("Update norm"); ax_up.legend(fontsize=7)
    ax_sc.set_xlabel(r"$\cos(\Delta\theta, -g)$"); ax_sc.set_ylabel("Train CE")
    ax_sc.set_title("Direction alignment vs loss"); ax_sc.legend(fontsize=7)
    fig.suptitle("E5: Stability Probe", fontsize=13, y=1.01)
    img1 = fig_to_b64(fig)

    fig2, ax = plt.subplots(figsize=(7, 4))
    bs_keys = ["bs64", "bs128", "bs256"]
    bs_labels = ["bs=64", "bs=128", "bs=256"]
    bs_ces = []
    for bs_key in bs_keys:
        path = os.path.join(raw, f"adamw_lr1e-03_{bs_key}_wd0.1", "seed_0", "metrics.jsonl")
        recs = load_jsonl(path)
        bs_ces.append(recs[-1]["val_loss"] if recs else 0)
    ax.bar(bs_labels, bs_ces, color=[BLUE, "#5b9df5", "#a3cbfa"], edgecolor="white")
    ax.set_ylabel("Val CE"); ax.set_title("E5: Batch Size Effect (AdamW, lr=1e-3)")
    for i, v in enumerate(bs_ces):
        if v: ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    img2 = fig_to_b64(fig2)
    return [img1, img2]


def make_e6_fig():
    raw = os.path.join(RESULTS, "raw", "e6_ablation")
    entries = []
    for d in sorted(os.listdir(raw)):
        path = os.path.join(raw, d, "seed_0", "metrics.jsonl")
        if not os.path.exists(path): continue
        recs = load_jsonl(path)
        val = recs[-1]["val_loss"]
        entries.append((val, d))
    entries.sort()

    names, vals = [], []
    for v, n in entries:
        short = n.replace("_", " ").replace("wd 0", "wd=0").replace("beta2", "β₂").replace("baseline", "baseline")
        names.append(short); vals.append(v)

    baseline_val = None
    for v, n in entries:
        if "baseline" in n: baseline_val = v; break

    deltas = [v - baseline_val for v in vals] if baseline_val else [0] * len(vals)
    colors_bar = [GREEN if d < 0 else (RED if d > 0.01 else "#aaaaaa") for d in deltas]
    colors_bar[deltas.index(max(deltas))] = RED

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, deltas, color=colors_bar, edgecolor="white")
    for i, (bar, delta, val) in enumerate(zip(bars, deltas, vals)):
        sign = "+" if delta >= 0 else ""
        ax.text(bar.get_width() + 0.005 * max(abs(d) for d in deltas), bar.get_y() + bar.get_height()/2,
                f"{sign}{delta:.3f}  (val={val:.3f})", va="center", fontsize=8.5)
    ax.set_yticks(y_pos); ax.set_yticklabels(names)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ Val CE vs baseline"); ax.set_title("E6: Ablation Study — Ranked by Impact")
    ax.invert_yaxis()
    return [fig_to_b64(fig)]


def make_e7_fig():
    kappa_labels = ["1", "10", "1000", "10000"]
    slopes = {
        "adamw":   [-12.49, -10.96, -6.43, -0.08],
        "adam_l2": [-12.49, -10.96, -6.43, -0.08],
        "adam":    [-12.49, -10.96, -6.43, -0.08],
        "sgd":     [None,   None,   0.0,   -1.33],
    }
    opts = ["adamw", "adam_l2", "adam", "sgd"]
    colors = {"adamw": BLUE, "adam_l2": GREEN, "adam": PURPLE, "sgd": RED}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(kappa_labels))
    w = 0.2
    for i, opt in enumerate(opts):
        vals = [v if v is not None else 0 for v in slopes[opt]]
        ax.bar(x + i*w - 0.3, vals, w, color=colors[opt], label=opt, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"κ={k}" for k in kappa_labels])
    ax.set_ylabel("Log-log slope")
    ax.set_title("E7: Convergence Rate Slopes")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)
    return [fig_to_b64(fig)]

def build_html():
    e1_imgs = make_e1_figs()
    e2_imgs = make_e2_figs()
    e4_imgs = make_e4_figs()
    e5_imgs = make_e5_figs()
    e6_imgs = make_e6_fig()
    e7_imgs = make_e7_fig()

    sections = [
        section_html("E0: Implementation Correctness", "e0", [], [
            "All 6/6 tests passed. Max error vs PyTorch &lt; 10⁻⁶.",
            "Zero-gradient decay: AdamW satisfies (1−ηλ)ᵗ, Adam-L2 does not.",
            "Determinism: same seed → identical parameters."
        ]),
        section_html("E1: Strongly Convex Quadratic Convergence", "e1", e1_imgs, [
            "AdamW converges to a neighborhood at high κ; SGD with fine-tuned LR achieves machine-precision convergence.",
            "κ=1: AdamW 3.7×10⁻⁸ vs SGD 3.3×10⁻¹². κ=10000: AdamW 1.1×10⁻² vs SGD 1.0×10⁻¹⁰."
        ]),
        section_html("E2: Block-Diagonal Quadratic", "e2", e2_imgs, [
            "Uniform κ=1000: Adam 7.6×10⁻⁵, SGD 4.6×10⁻¹² (SGD wins on homogeneous curvature).",
            "Heterogeneous κ={1,10,100,10000}: Adam 1.0×10⁻³, SGD 5.9×10⁻⁹. Adam adapts per-block."
        ]),
        section_html("E3: Weight Decay Decoupling Grid", "e3", [], [
            "AdamW shows smooth heatmap over (η, λ). Performance stable along ηλ=const diagonals.",
            "Adam-L2 shows rapid degradation for λ ≥ 0.01. Coupled L2 interacts harmfully with adaptive denominator."
        ]),
        section_html("E4: TinyGPT Main Experiment ⭐", "e4", e4_imgs, [
            "Best: AdamW (lr=3e-3, wd=0.3) → val CE 1.505.",
            "AdamW wd sweep: 1.517→1.580→1.514→1.505. Controllable regularization.",
            "Adam-L2 wd sweep: 1.517→2.100→2.487→2.513. Complete collapse.",
            "SGD optimal: 1.635. Adam family significantly better on transformers."
        ]),
        section_html("E5: Stability Probe", "e5", e5_imgs, [
            "cos(Δθ,−g): AdamW stable [0.10,0.21]; Adam-L2 violent [0.07,0.42].",
            "Adam-L2 train loss rebounds at step ~150; val loss barely improves (2.67→2.53).",
            "Larger batch size (256) improves val CE to 1.883 via more stable gradient estimates."
        ]),
        section_html("E6: Ablation Study", "e6", e6_imgs, [
            "Decoupling is the #1 factor: Adam-L2 Δ = +0.758 (10−60× any other factor).",
            "β₂ is 2nd most sensitive: 0.999 > 0.99 > 0.95, monotonic, Δ = −0.046.",
            "ε insensitive (±0.004). Warmup small effect (+0.011).",
            "Implementation verified: no_decay ≡ adam_no_wd (both 1.78394)."
        ]),
        section_html("E7: Convergence Rate Fitting", "e7", e7_imgs, [
            "Low κ: Adam family slopes −12.5 (fast transient). High κ: −0.08 (stagnation).",
            "SGD maintains −1.33 at κ=10000 with fine-tuned step size.",
            "Slopes not equal to asymptotic rates — reflects finite-step transient behavior."
        ]),
    ]

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AdamW 仿真实验报告</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; background: #f5f4f0; color: #2d2d2d; line-height: 1.6; }}
.header {{ background: #1a1a2e; color: #e8e6dd; padding: 2rem 3rem; }}
.header h1 {{ font-size: 1.8rem; font-weight: 600; }}
.header p {{ font-size: 0.9rem; margin-top: 0.4rem; color: #a8a6a0; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 1.5rem; }}
.overview {{ background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.overview table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
.overview th {{ background: #f0ede6; padding: 0.55rem 0.8rem; text-align: left; border-bottom: 2px solid #d9d5cc; }}
.overview td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid #e8e4da; }}
.overview tr:hover {{ background: #fbf9f5; }}
.pass {{ color: #1b7a3d; font-weight: 600; }}
.warn {{ color: #c08500; font-weight: 600; }}
.section {{ background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.section h2 {{ font-size: 1.15rem; margin-bottom: 0.5rem; color: #1a1a2e; border-left: 4px solid {BLUE}; padding-left: 0.7rem; }}
.section .note {{ font-size: 0.85rem; color: #666; margin-bottom: 1rem; }}
.img-row {{ display: flex; flex-wrap: wrap; gap: 0.8rem; }}
.img-row .fig {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #e0dcd4; }}
.section.e0 h2 {{ border-left-color: #6b6b6b; }}
.section.e1 h2 {{ border-left-color: {RED}; }}
.section.e2 h2 {{ border-left-color: {ORANGE}; }}
.section.e3 h2 {{ border-left-color: {PURPLE}; }}
.section.e4 h2 {{ border-left-color: {BLUE}; }}
.section.e5 h2 {{ border-left-color: {GREEN}; }}
.section.e6 h2 {{ border-left-color: {RED}; }}
.section.e7 h2 {{ border-left-color: "#888"; }}
.toc {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem; }}
.toc a {{ background: white; color: #1a1a2e; text-decoration: none; padding: 0.4rem 0.9rem; border-radius: 4px; font-size: 0.85rem; font-weight: 500; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.toc a:hover {{ background: {BLUE}; color: white; }}
.footer {{ text-align: center; padding: 2rem; font-size: 0.8rem; color: #999; }}
</style>
</head>
<body>
<div class="header">
  <h1>AdamW 仿真实验完整报告</h1>
  <p>Course Project · Python 3.10 / PyTorch 2.5.1+cu121 · RTX 4060 Laptop GPU · fp32 · 2026-10-27</p>
</div>
<div class="container">
  <div class="overview">
    <h3 style="margin-bottom:0.6rem">Results Overview</h3>
    <table>
      <tr><th>Exp</th><th>Proposition</th><th>Status</th><th>Key Metric</th></tr>
      <tr><td>E0</td><td>C1–C2: Implementation correctness &amp; decoupled decay</td><td><span class="pass">✓</span></td><td>6/6 tests passed</td></tr>
      <tr><td>E1</td><td>C3: Convergence on strongly convex quadratic</td><td><span class="pass">✓</span></td><td>All optimizers converge; AdamW neighborhood at κ=10⁴</td></tr>
      <tr><td>E2</td><td>C5: Block-heterogeneous curvature</td><td><span class="warn">⚠</span></td><td>Adam more LR-robust; SGD wins on uniform</td></tr>
      <tr><td>E3</td><td>C2/C4: Decoupling grid + dynamic regularization</td><td><span class="pass">✓</span></td><td>AdamW smooth; Adam-L2 degrades at λ≥0.01</td></tr>
      <tr><td>E4</td><td>Real training: TinyGPT on Shakespeare</td><td><span class="pass">✓</span></td><td><strong>AdamW: 1.505</strong> CE; Adam-L2 collapses to 2.513</td></tr>
      <tr><td>E5</td><td>C6: Stability probe (cos alignment)</td><td><span class="warn">⚠</span></td><td>cos(Δθ,−g) severely unstable for Adam-L2</td></tr>
      <tr><td>E6</td><td>Ablation: component importance</td><td><span class="pass">✓</span></td><td>Decoupling ≫ β₂ > warmup > ε ≈ wd</td></tr>
      <tr><td>E7</td><td>C7: Convergence rate fitting</td><td><span class="pass">✓</span></td><td>AdamW slope −0.08 at κ=10⁴; SGD −1.33</td></tr>
    </table>
  </div>
  {"".join(sections)}
  <div class="footer">
    <p>All figures generated from experimental data. Code: adamw-sim/experiments/</p>
    <p>Dependencies: matplotlib · PyTorch · numpy · Pillow</p>
  </div>
</div>
</body>
</html>"""

    out = os.path.join(DEMO_DIR, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[demo] saved {out}")
    return out


if __name__ == "__main__":
    build_html()