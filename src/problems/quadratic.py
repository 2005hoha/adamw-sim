import math
import torch


class Quadratic:
    def __init__(self, dim=100, kappa=1e3, seed=0, device="cpu"):
        g = torch.Generator(device=device).manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, device=device))
        eig = torch.logspace(0, math.log10(kappa), dim, device=device)
        self.H = (q * eig) @ q.T
        self.theta_star = torch.randn(dim, generator=g, device=device)
        self.theta0 = torch.randn(dim, generator=g, device=device)
        self.dim = dim
        self.kappa = kappa
        self.optimal_loss = 0.0

    def loss(self, theta):
        diff = theta - self.theta_star
        return 0.5 * diff @ self.H @ diff

    def grad(self, theta):
        return self.H @ (theta - self.theta_star)