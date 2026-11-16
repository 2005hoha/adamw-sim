#!/bin/bash
set -e

export PYTHONUNBUFFERED=1

echo "=============================================="
echo " AdamW Simulation - Full Reproduction Pipeline"
echo "=============================================="

cd "$(dirname "$0")/.."

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate edge_grpo
elif [ -f "C:/Users/33240/anaconda3/etc/profile.d/conda.sh" ]; then
    source "C:/Users/33240/anaconda3/etc/profile.d/conda.sh"
    conda activate edge_grpo
else
    echo "Warning: conda not found. Ensure edge_grpo environment is active."
fi

echo ""
echo "Step 0: Environment check"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo ""
echo "Step 1: E0 - Implementation validation"
python experiments/exp_e0_validate.py

echo ""
echo "Step 2: Tests"
python -m pytest tests/ -v --tb=short

echo ""
echo "Step 3: E1 - Quadratic convergence"
python experiments/exp_e1_quadratic.py

echo ""
echo "Step 4: E2 - Block heterogeneity"
python experiments/exp_e2_block.py

echo ""
echo "Step 5: E3 - Decoupling grid"
python experiments/exp_e3_decoupling.py

echo ""
echo "Step 6: E7 - Convergence rate fitting"
python experiments/exp_e7_convergence_rate.py

echo ""
echo "Step 7: E4 - Tiny GPT main experiment"
python experiments/exp_e4_tiny_gpt.py
python experiments/exp_e4_summarize.py

echo ""
echo "Step 8: E5 - Stability probe"
python experiments/exp_e5_stability.py
python experiments/exp_e5_replot.py

echo ""
echo "Step 9: E6 - Ablation study"
python experiments/exp_e6_ablation.py
python experiments/exp_e6_summarize.py

echo ""
echo "=============================================="
echo " All experiments complete!"
echo " Raw runs are in results/raw/"
echo " Summary tables are in results/summary/"
echo " Figures are in results/figures/"
echo "=============================================="