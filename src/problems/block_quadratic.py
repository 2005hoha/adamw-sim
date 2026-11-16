import math
import torch


class BlockQuadratic:
    def __init__(self, dim=100, num_blocks=4, kappas=None, seed=0, device="cpu"):
        g = torch.Generator(device=device).manual_seed(seed)
        self.num_blocks = num_blocks
        self.dim = dim
        self.block_dim = dim // num_blocks
        self.blocks = []

        if kappas is None:
            kappas = [1, 10, 100, 10000]

        self.kappas = kappas
        for b in range(num_blocks):
            bdim = self.block_dim
            eig = torch.logspace(0, math.log10(kappas[b]), bdim, device=device)
            q, _ = torch.linalg.qr(torch.randn(bdim, bdim, generator=g, device=device))
            Hb = (q * eig) @ q.T
            self.blocks.append(Hb)

        self.theta_star = torch.randn(dim, generator=g, device=device)
        self.theta0 = torch.randn(dim, generator=g, device=device)
        self.optimal_loss = 0.0

    def _block_multiply(self, theta):
        result = torch.zeros_like(theta)
        for b in range(self.num_blocks):
            start = b * self.block_dim
            end = start + self.block_dim
            result[start:end] = self.blocks[b] @ theta[start:end]
        return result

    def loss(self, theta):
        diff = theta - self.theta_star
        return 0.5 * (diff * self._block_multiply(diff)).sum()

    def grad(self, theta):
        return self._block_multiply(theta - self.theta_star)

    def block_grad_norms(self, theta):
        diff = theta - self.theta_star
        norms = []
        for b in range(self.num_blocks):
            start = b * self.block_dim
            end = start + self.block_dim
            g_block = self.blocks[b] @ diff[start:end]
            norms.append(g_block.norm().item())
        return norms