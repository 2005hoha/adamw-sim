import math
import torch


class NoisyQuadratic:
    def __init__(self, dim=100, kappa=1e3, noise_std=0.1, seed=0, device="cpu"):
        g = torch.Generator(device=device).manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, device=device))
        eig = torch.logspace(0, math.log10(kappa), dim, device=device)
        self.H = (q * eig) @ q.T
        self.theta_star = torch.randn(dim, generator=g, device=device)
        self.theta0 = torch.randn(dim, generator=g, device=device)
        self.dim = dim
        self.kappa = kappa
        self.noise_std = noise_std
        self._rng_g = g

    def loss(self, theta):
        diff = theta - self.theta_star
        return 0.5 * diff @ self.H @ diff

    def grad(self, theta, batch_size=None):
        true_grad = self.H @ (theta - self.theta_star)
        if batch_size is None or batch_size >= self.dim:
            noise = torch.randn(self.dim, generator=self._rng_g, device=theta.device)
        else:
            noise = torch.zeros(self.dim, device=theta.device)
            indices = torch.randperm(self.dim, generator=self._rng_g, device=theta.device)[:batch_size]
            noise[indices] = torch.randn(batch_size, generator=self._rng_g, device=theta.device) * math.sqrt(self.dim / batch_size)
        return true_grad + self.noise_std * noise

    def full_grad(self, theta):
        return self.H @ (theta - self.theta_star)