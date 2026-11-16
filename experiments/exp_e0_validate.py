import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
from src.optimizers.adam_family import AdamFamily


def run_e0(cfg_path="configs/quadratic.yaml", output_dir="results/raw/e0_validate"):
    print("=" * 60)
    print("E0: Implementation Validation")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tests_passed = 0
    tests_total = 0

    modes = ["adam", "adam_l2", "adamw"]
    lr = 1e-3
    betas = (0.9, 0.95)
    eps = 1e-8
    wd = 0.1

    for mode in modes:
        effective_wd = wd if mode != "adam" else 0.0

        p_custom = torch.randn(10, 10, requires_grad=True)
        p_torch = p_custom.detach().clone().requires_grad_(True)

        opt_custom = AdamFamily([p_custom], lr=lr, betas=betas, eps=eps, weight_decay=effective_wd, mode=mode)

        if mode == "adam":
            opt_torch = torch.optim.Adam([p_torch], lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        elif mode == "adam_l2":
            opt_torch = torch.optim.Adam([p_torch], lr=lr, betas=betas, eps=eps, weight_decay=effective_wd)
        else:
            opt_torch = torch.optim.AdamW([p_torch], lr=lr, betas=betas, eps=eps, weight_decay=effective_wd)

        max_err = 0.0
        for step in range(1, 21):
            g = torch.randn(10, 10)
            p_custom.grad = g.clone()
            p_torch.grad = g.clone()

            opt_custom.step()
            opt_custom.zero_grad()
            opt_torch.step()
            opt_torch.zero_grad()

            err = (p_custom - p_torch).abs().max().item()
            max_err = max(max_err, err)

        tests_total += 1
        passed = max_err < 1e-6
        tests_passed += int(passed)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {mode} vs PyTorch: max_err={max_err:.2e}")

    p_aw = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)
    opt_aw = AdamFamily([p_aw], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, mode="adamw")
    p_init = p_aw.detach().clone()
    expected_mult = 1.0 - 0.01 * 0.1
    expected = p_init.clone()

    max_err = 0.0
    for t in range(1, 51):
        p_aw.grad = torch.zeros_like(p_aw)
        opt_aw.step()
        opt_aw.zero_grad()
        expected.mul_(expected_mult)
        err = (p_aw - expected).abs().max().item()
        max_err = max(max_err, err)

    tests_total += 1
    passed = max_err < 1e-6
    tests_passed += int(passed)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Zero-gradient decay (AdamW): max_err={max_err:.2e}")

    p_al2 = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)
    opt_al2 = AdamFamily([p_al2], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, mode="adam_l2")
    for t in range(1, 51):
        p_al2.grad = torch.zeros_like(p_al2)
        opt_al2.step()
        opt_al2.zero_grad()

    expected_l2 = torch.tensor([2.0, 3.0, 5.0]) * (expected_mult ** 50)
    l2_diff = (p_al2 - expected_l2).abs().max().item()

    tests_total += 1
    passed = l2_diff > 1e-6
    tests_passed += int(passed)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Zero-gradient decay (Adam-L2): differs_from_exponential={l2_diff:.6f}")

    p_decay = torch.randn(10, 10, requires_grad=True)
    p_nodecay = torch.randn(5, requires_grad=True)
    opt_pg = AdamFamily([
        {"params": [p_decay], "weight_decay": 0.1},
        {"params": [p_nodecay], "weight_decay": 0.0},
    ], lr=0.1, betas=(0.9, 0.95), eps=1e-8, mode="adamw")
    p_nodecay_init = p_nodecay.detach().clone()

    for _ in range(10):
        p_decay.grad = torch.zeros_like(p_decay)
        p_nodecay.grad = torch.zeros_like(p_nodecay)
        opt_pg.step()
        opt_pg.zero_grad()

    pg_err = (p_nodecay - p_nodecay_init).abs().max().item()
    tests_total += 1
    passed = pg_err < 1e-6
    tests_passed += int(passed)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Param group isolation: no_decay error={pg_err:.2e}")

    print(f"\nE0 Summary: {tests_passed}/{tests_total} tests passed")

    with open(os.path.join(output_dir, "e0_summary.txt"), "w") as f:
        f.write(f"E0 Tests: {tests_passed}/{tests_total} passed\n")

    return tests_passed == tests_total


if __name__ == "__main__":
    success = run_e0()
    exit(0 if success else 1)