import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.problems.quadratic import Quadratic
from src.problems.block_quadratic import BlockQuadratic


def test_quadratic_optimal_loss():
    problem = Quadratic(dim=10, kappa=100, seed=0, device="cpu")
    theta_star = problem.theta_star
    loss = problem.loss(theta_star)
    assert loss.item() < 1e-10, f"Loss at theta_star should be ~0, got {loss.item()}"


def test_quadratic_grad_optimal():
    problem = Quadratic(dim=10, kappa=100, seed=0, device="cpu")
    theta_star = problem.theta_star
    g = problem.grad(theta_star)
    assert g.norm().item() < 1e-10, f"Gradient at theta_star should be ~0"


def test_quadratic_positive_definite():
    problem = Quadratic(dim=50, kappa=100, seed=0, device="cpu")
    eigvals = torch.linalg.eigvalsh(problem.H)
    assert eigvals.min().item() > 0, "H should be positive definite"
    kappa_actual = eigvals.max().item() / eigvals.min().item()
    assert abs(kappa_actual - problem.kappa) / problem.kappa < 0.01, \
        f"Condition number mismatch: {kappa_actual} vs {problem.kappa}"


def test_block_quadratic():
    problem = BlockQuadratic(dim=100, num_blocks=4, kappas=[1, 10, 100, 10000], seed=0, device="cpu")
    theta_star = problem.theta_star
    loss = problem.loss(theta_star)
    assert loss.item() < 1e-10, f"Loss at theta_star should be ~0, got {loss.item()}"

    g = problem.grad(theta_star)
    assert g.norm().item() < 1e-10, f"Gradient at theta_star should be ~0"


if __name__ == "__main__":
    test_quadratic_optimal_loss()
    print("quadratic_optimal_loss OK")
    test_quadratic_grad_optimal()
    print("quadratic_grad_optimal OK")
    test_quadratic_positive_definite()
    print("quadratic_positive_definite OK")
    test_block_quadratic()
    print("block_quadratic OK")
    print("All tests passed!")