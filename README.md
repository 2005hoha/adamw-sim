# AdamW Simulation

AdamW 算法课程大作业 —— 仿真实验与代码仓库。

## 环境

- Python 3.10.19
- conda 环境：`edge_grpo`
- PyTorch 2.5.1+cu121
- GPU：RTX 4060 Laptop (8 GB)

### 安装

```bash
conda activate edge_grpo
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 环境验证

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m pip check
```

## 仓库结构

```
adamw-sim/
├── src/                  # 源码
│   ├── optimizers/       # AdamFamily（统一 Adam/Adam-L2/AdamW）、SGD
│   ├── problems/         # 强凸二次、块对角二次、带噪声二次
│   ├── models/           # TinyGPT（4 层 char-level GPT, ~0.8M 参数）
│   ├── data/             # TinyShakespeare 数据加载
│   ├── metrics.py        # 诊断指标（更新范数、权重范数、cos 等）
│   ├── train.py          # 通用训练循环
│   ├── plotting.py       # 统一绘图（颜色/线型/标记）
│   └── utils.py          # seed、scheduler、工具函数
├── configs/              # YAML 配置文件
│   ├── quadratic.yaml
│   ├── block_quadratic.yaml
│   ├── tiny_gpt.yaml
│   ├── decoupling_grid.yaml
│   └── stability.yaml
├── experiments/          # E0–E7 实验入口
│   ├── exp_e0_validate.py
│   ├── exp_e1_quadratic.py
│   ├── exp_e2_block.py
│   ├── exp_e3_decoupling.py
│   ├── exp_e4_tiny_gpt.py
│   ├── exp_e4_summarize.py    # 从已有 raw 结果生成 E4 表格/固定 wd 图
│   ├── exp_e5_stability.py
│   ├── exp_e5_replot.py       # 从 E5 summary JSON 重画图，不重训
│   ├── exp_e6_ablation.py
│   ├── exp_e6_summarize.py    # 从已有 raw 结果生成 E6 表格/分组图
│   └── exp_e7_convergence_rate.py
├── tests/                # 单元测试
│   ├── test_optimizers.py
│   ├── test_quadratic.py
│   └── test_determinism.py
├── scripts/
│   ├── run_all.sh        # 一键复现全部实验
│   └── run_grid.sh       # 仅运行解耦网格搜索
├── results/              # 结果输出
│   ├── raw/              # 原始日志（JSONL）
│   ├── summary/          # 汇总 JSON
│   └── figures/          # PDF/PNG 图片
└── docs/
    └── proof_notes.md    # 证明草稿
```

## 实验概述

| 编号 | 目的 | 关键指标 | 预期 |
|---|---|---|---|
| E0 | 实现与 PyTorch 对齐 | 最大绝对误差 | < 1e-6 |
| E1 | 强凸二次收敛 | f-f*、梯度范数 | Adam 族下降；高 κ 不保证到容差 |
| E2 | 块异质性 | 最终 f-f*、容差步数 | Adam 对 LR 更鲁棒；SGD 最佳调参下最终精度更高 |
| E3 | 解耦权重衰减 | f-f* 热力图、正则化目标 | Adam-L2 在高 λ 下损失远高于 AdamW |
| E4 | Tiny GPT 主实验 | train/val CE | Adam-L2 在 λ≥0.01 时崩溃；AdamW 稳定 |
| E5 | 稳定性探针 | cos(Δθ, -g)、更新范数 | 当前规模未复现大 batch 失配 |
| E6 | 消融 | val CE 对比 | AdamW 内部 β2 最敏感；耦合 vs 解耦单列 |
| E7 | 收敛率拟合 | log-log 斜率、R² | 只比较 Adam 族 vs SGD；斜率不直接等于理论阶 |

## 快速开始

### 最小复现（E0 + 测试，约 30 秒）

```bash
python experiments/exp_e0_validate.py
python -m pytest tests/ -v --tb=short
```

### 合成实验（E1/E2/E3/E7，约 5 分钟）

```bash
python experiments/exp_e1_quadratic.py
python experiments/exp_e2_block.py
python experiments/exp_e3_decoupling.py
python experiments/exp_e7_convergence_rate.py
```

### 完整复现（所有实验，约 3-10 小时）

```bash
bash scripts/run_all.sh
```

或逐个运行：

```bash
python experiments/exp_e0_validate.py
python -m pytest tests/ -v --tb=short
python experiments/exp_e1_quadratic.py
python experiments/exp_e2_block.py
python experiments/exp_e3_decoupling.py
python experiments/exp_e7_convergence_rate.py
python experiments/exp_e4_tiny_gpt.py
python experiments/exp_e4_summarize.py
python experiments/exp_e5_stability.py
python experiments/exp_e6_ablation.py
python experiments/exp_e6_summarize.py
```

> E4/E6 的 `*_summarize.py` 只读 `results/raw`，不会重新训练，也不会增加 seed。E4 的固定 wd 图 `results/figures/e4_val_loss_wd0.1.*` 由 `exp_e4_summarize.py` 生成。

## 关键设计决策

1. **统一优化器类**：`AdamFamily` 通过 `mode` 参数覆盖 Adam / Adam-L2 / AdamW，避免三份实现漂移。
2. **参数组**：权重衰减只作用于 `ndim >= 2` 且非 embedding/bias/LayerNorm 的参数。
3. **公平比较**：Adam 族与 SGD 各自搜索最优 LR，预算相同。
4. **确定性**：`torch.backends.cudnn.deterministic = True`，Windows 下 `num_workers=0`。
5. **fp32 全程**：8 GB 显存下 0.8M 模型足够。
6. **失败也记录**：NaN、OOM、发散都写进日志。

## 结果位置

- 原始日志：`results/raw/{experiment}/{optimizer}/{lr}_{wd}/seed_{seed}/metrics.jsonl`
- 单次 run 摘要：`results/raw/{experiment}/{optimizer}/{lr}_{wd}/seed_{seed}/summary.json`
- 汇总表：`results/summary/e2_block.json`、`e3_decoupling.json`、`e4_tiny_gpt.csv`、`e5_stability.json`、`e6_ablation.csv`、`e7_slopes.json`
- 论文图片：`results/figures/e{1-7}_*.pdf`（矢量）和 `.png`（位图）