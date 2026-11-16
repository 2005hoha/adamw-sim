import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import set_seed
from src.optimizers.adam_family import AdamFamily


def test_same_seed_same_result():
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)

    p1 = torch.randn(10, generator=g1, requires_grad=True)
    p2 = torch.randn(10, generator=g2, requires_grad=True)
    assert torch.allclose(p1, p2), "Parameters should be identical from same seed"

    opt1 = AdamFamily([p1], lr=0.01, mode="adamw")
    opt2 = AdamFamily([p2], lr=0.01, mode="adamw")

    g_grad = torch.Generator().manual_seed(100)
    for _ in range(10):
        grad = torch.randn(10, generator=g_grad)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        opt1.step()
        opt2.step()
        opt1.zero_grad()
        opt2.zero_grad()

    assert torch.allclose(p1, p2), "Same seed should produce same result after training"


if __name__ == "__main__":
    test_same_seed_same_result()
    print("same_seed_same_result OK")
    print("All tests passed!")