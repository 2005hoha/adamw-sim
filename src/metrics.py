import torch
import torch.nn.functional as F


@torch.no_grad()
def vectorize(tensors):
    return torch.cat([t.detach().reshape(-1) for t in tensors])


@torch.no_grad()
def update_stats(params, prev_params):
    p_list = list(params)
    prev_list = list(prev_params)
    update = vectorize([p - p0 for p, p0 in zip(p_list, prev_list)])
    weight = vectorize(p_list)
    return {
        "update_norm": update.norm().item(),
        "weight_norm": weight.norm().item(),
        "update_rms": update.pow(2).mean().sqrt().item(),
        "update_to_weight": (update.norm() / weight.norm().clamp_min(1e-12)).item(),
    }


@torch.no_grad()
def cos_update_neg_grad(params, prev_params):
    pairs = [(p, p0) for p, p0 in zip(params, prev_params) if p.grad is not None]
    if not pairs:
        return float("nan")
    update = vectorize([p - p0 for p, p0 in pairs])
    grads = vectorize([p.grad for p, _ in pairs])
    if grads.numel() == 0 or update.numel() == 0:
        return float("nan")
    return F.cosine_similarity(update, -grads, dim=0).item()


@torch.no_grad()
def grad_norm(params):
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


@torch.no_grad()
def param_norm(params):
    total_norm = 0.0
    for p in params:
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5