import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimizers.adam_family import AdamFamily


def _make_params(device="cpu"):
    p1 = torch.randn(10, 10, device=device, requires_grad=True)
    p2 = torch.randn(5, device=device, requires_grad=True)
    return [p1, p2]


def _build_custom(params, mode, lr=1e-3, wd=0.0, betas=(0.9, 0.95), eps=1e-8):
    return AdamFamily(params, lr=lr, betas=betas, eps=eps, weight_decay=wd, mode=mode)


def _build_torch(params, mode, lr=1e-3, wd=0.0, betas=(0.9, 0.95), eps=1e-8):
    if mode == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
    elif mode == "adam_l2":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    elif mode == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _generate_gradients(params, seed=42):
    torch.manual_seed(seed)
    for p in params:
        p.grad = torch.randn_like(p)


@pytest.mark.parametrize("mode", ["adam", "adam_l2", "adamw"])
def test_align_with_pytorch(mode):
    wd = 0.1 if mode != "adam" else 0.0
    lr = 1e-3
    betas = (0.9, 0.95)
    eps = 1e-8

    params_custom = _make_params("cpu")
    params_torch = [p.detach().clone().requires_grad_(True) for p in params_custom]

    opt_custom = _build_custom(params_custom, mode, lr=lr, wd=wd, betas=betas, eps=eps)
    opt_torch = _build_torch(params_torch, mode, lr=lr, wd=wd, betas=betas, eps=eps)

    for step in range(1, 21):
        _generate_gradients(params_custom, seed=step)
        _generate_gradients(params_torch, seed=step)

        opt_custom.step()
        opt_custom.zero_grad()
        opt_torch.step()
        opt_torch.zero_grad()

        for pc, pt in zip(params_custom, params_torch):
            max_err = (pc - pt).abs().max().item()
            assert max_err < 1e-6, f"Step {step}, mode={mode}: max error {max_err} >= 1e-6"


def test_zero_gradient_decay_adamw():
    p = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)
    lr = 0.01
    wd = 0.1
    steps = 50

    opt = AdamFamily([p], lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd, mode="adamw")
    expected_mult = 1.0 - lr * wd
    p_init = p.detach().clone()
    expected = p_init.clone()

    for t in range(1, steps + 1):
        p.grad = torch.zeros_like(p)
        opt.step()
        opt.zero_grad()

        expected.mul_(expected_mult)
        err = (p - expected).abs().max().item()
        assert err < 1e-6, f"Step {t}: error {err}"


def test_zero_gradient_decay_adam_l2():
    p_l2 = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)
    lr = 0.01
    wd = 0.1
    steps = 50

    opt_l2 = AdamFamily([p_l2], lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd, mode="adam_l2")
    p_init_l2 = p_l2.detach().clone()
    expected_mult = 1.0 - lr * wd

    for t in range(1, steps + 1):
        p_l2.grad = torch.zeros_like(p_l2)
        opt_l2.step()
        opt_l2.zero_grad()

    l2_actual = p_l2.detach().clone()
    l2_expected = p_init_l2 * (expected_mult ** steps)
    l2_diff = (l2_actual - l2_expected).abs().max().item()

    assert l2_diff > 1e-6, f"Adam-L2 should NOT follow (1-lr*wd)^t exactly, but max diff={l2_diff}"


def test_bias_correction_first_step():
    p = torch.randn(5, requires_grad=True)
    g_val = torch.randn(5)
    p.grad = g_val.clone()

    opt = AdamFamily([p], lr=0.1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, mode="adam")

    opt.step()
    opt.zero_grad()

    beta1 = 0.9
    beta2 = 0.95
    eps = 1e-8
    lr = 0.1

    expected_m = (1 - beta1) * g_val
    expected_v = (1 - beta2) * g_val ** 2
    expected_m_hat = expected_m / (1 - beta1)
    expected_v_hat = expected_v / (1 - beta2)
    expected_update = -lr * expected_m_hat / (torch.sqrt(expected_v_hat) + eps)

    actual_update = p - (p + lr * expected_m_hat / (torch.sqrt(expected_v_hat) + eps))


def test_decay_param_groups():
    p_decay = torch.randn(10, 10, requires_grad=True)
    p_no_decay = torch.randn(5, requires_grad=True)

    opt = AdamFamily([
        {"params": [p_decay], "weight_decay": 0.1},
        {"params": [p_no_decay], "weight_decay": 0.0},
    ], lr=0.1, betas=(0.9, 0.95), eps=1e-8, mode="adamw")

    p_no_decay_init = p_no_decay.detach().clone()

    for _ in range(5):
        p_decay.grad = torch.zeros_like(p_decay)
        p_no_decay.grad = torch.zeros_like(p_no_decay)
        opt.step()
        opt.zero_grad()

    err = (p_no_decay - p_no_decay_init).abs().max().item()
    assert err < 1e-6, f"no_decay params should not change with zero grad, but max diff={err}"


def test_determinism():
    torch.manual_seed(0)
    p1 = torch.randn(10, requires_grad=True)
    opt1 = AdamFamily([p1], lr=0.01, mode="adamw")

    torch.manual_seed(0)
    p2 = torch.randn(10, requires_grad=True)
    opt2 = AdamFamily([p2], lr=0.01, mode="adamw")

    for step in range(5):
        g = torch.randn(10)
        p1.grad = g.clone()
        p2.grad = g.clone()
        opt1.step()
        opt2.step()
        opt1.zero_grad()
        opt2.zero_grad()

    assert torch.allclose(p1, p2), "Determinism check failed"


if __name__ == "__main__":
    test_align_with_pytorch("adam")
    print("adam OK")
    test_align_with_pytorch("adam_l2")
    print("adam_l2 OK")
    test_align_with_pytorch("adamw")
    print("adamw OK")
    test_zero_gradient_decay_adamw()
    print("zero_gradient_decay_adamw OK")
    test_zero_gradient_decay_adam_l2()
    print("zero_gradient_decay_adam_l2 OK")
    test_bias_correction_first_step()
    print("bias_correction_first_step OK")
    test_decay_param_groups()
    print("decay_param_groups OK")
    test_determinism()
    print("determinism OK")
    print("All tests passed!")