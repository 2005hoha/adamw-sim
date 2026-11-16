#!/bin/bash
set -e

echo "=============================================="
echo " AdamW Simulation - Grid Search Only"
echo "=============================================="

cd "$(dirname "$0")/.."

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate edge_grpo
elif [ -f "C:/Users/33240/anaconda3/etc/profile.d/conda.sh" ]; then
    source "C:/Users/33240/anaconda3/etc/profile.d/conda.sh"
    conda activate edge_grpo
fi

echo "Running grid search experiments..."
python experiments/exp_e3_decoupling.py

echo "Grid search complete. See results/figures/e3_*.pdf"