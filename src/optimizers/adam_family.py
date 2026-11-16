import math
import torch
from torch.optim import Optimizer


class AdamFamily(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, mode="adamw"):
        if mode not in {"adam", "adam_l2", "adamw"}:
            raise ValueError(f"unknown mode: {mode}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, mode=mode)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            mode = group["mode"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if mode == "adam_l2" and wd != 0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr / bias_correction1
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                if mode == "adamw" and wd != 0:
                    p.mul_(1 - lr * wd)

                p.addcdiv_(m, denom, value=-step_size)

        return loss