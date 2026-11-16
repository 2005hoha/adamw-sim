import os
import json
import time
import torch
import torch.nn as nn
from src.metrics import update_stats, cos_update_neg_grad, grad_norm, param_norm
from src.utils import CosineScheduler, ConstantScheduler, ensure_dir


class Logger:
    def __init__(self, log_dir):
        self.log_dir = ensure_dir(log_dir)
        self.metrics_file = os.path.join(log_dir, "metrics.jsonl")
        self.summary_file = os.path.join(log_dir, "summary.json")
        self.entries = []
        self._first_write = True

    def log(self, entry):
        self.entries.append(entry)
        mode = "w" if self._first_write else "a"
        with open(self.metrics_file, mode, encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._first_write = False

    def save_summary(self, summary):
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def build_optimizer(model, cfg):
    from src.optimizers import AdamFamily, SGD

    opt_cfg = cfg["optimizer"]
    name = opt_cfg["name"]

    if name in ("adam", "adam_l2", "adamw"):
        param_groups = model.get_param_groups(opt_cfg.get("weight_decay", 0.0))
        return AdamFamily(
            param_groups,
            lr=opt_cfg.get("lr", 1e-3),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
            eps=opt_cfg.get("eps", 1e-8),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            mode=name,
        )
    elif name == "sgd":
        param_groups = model.get_param_groups(opt_cfg.get("weight_decay", 0.0))
        return SGD(
            param_groups,
            lr=opt_cfg.get("lr", 0.1),
            momentum=opt_cfg.get("momentum", 0.0),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(cfg):
    train_cfg = cfg["train"]
    lr = cfg["optimizer"]["lr"]

    if train_cfg.get("scheduler", "cosine") == "cosine":
        return CosineScheduler(
            max_lr=lr,
            min_lr=lr * train_cfg.get("min_lr_ratio", 0.1),
            warmup_steps=train_cfg.get("warmup_steps", 100),
            max_steps=train_cfg["max_steps"],
        )
    else:
        return ConstantScheduler(lr)


@torch.no_grad()
def evaluate(model, loader, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
        _, loss = model(x, y)
        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def train_one_run(model, train_loader, val_loader, cfg, log_dir, seed=0):
    from src.utils import set_seed
    set_seed(seed)

    device = next(model.parameters()).device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_cfg = cfg["train"]
    max_steps = train_cfg["max_steps"]
    eval_interval = train_cfg.get("eval_interval", 250)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(cfg)

    logger = Logger(log_dir)
    train_iter = iter(train_loader)

    best_val_loss = float("inf")
    final_val_loss = float("inf")
    peak_mem_mb = 0.0
    total_tokens = 0
    t_start = time.time()

    for step in range(1, max_steps + 1):
        lr = scheduler(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        total_tokens += x.numel()

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        total_norm = nn.utils.clip_grad_norm_(params, grad_clip)
        total_norm_f = float(total_norm)
        clip_fraction = 0.0
        if grad_clip > 0 and total_norm_f > 0:
            clip_fraction = max(0.0, 1.0 - grad_clip / total_norm_f)

        is_eval_step = (step % eval_interval == 0) or (step == max_steps)

        if is_eval_step:
            prev_params = [p.detach().clone() for p in params]

        optimizer.step()

        if is_eval_step:
            print(f"\r  step {step}/{max_steps} | train_loss={loss.item():.4f}", end="", flush=True)
            stats = update_stats(params, prev_params)
            stats["cos"] = cos_update_neg_grad(params, prev_params)
            val_loss = evaluate(model, val_loader)
            stats.update({
                "step": step,
                "lr": lr,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "grad_norm": total_norm_f,
                "clip_fraction": clip_fraction,
                "tokens_seen": total_tokens,
            })
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                stats["peak_mem_mb"] = mem
                peak_mem_mb = max(peak_mem_mb, mem)
            else:
                stats["peak_mem_mb"] = 0.0

            stats["wall_time_s"] = time.time() - t_start
            logger.log(stats)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
            final_val_loss = val_loss

    print()  # newline after progress
    wall_time = time.time() - t_start
    summary = {
        "final_step": max_steps,
        "final_train_loss": logger.entries[-1]["train_loss"] if logger.entries else float("nan"),
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "wall_time_s": wall_time,
        "peak_mem_mb": peak_mem_mb,
        "num_params": sum(p.numel() for p in params),
    }
    logger.save_summary(summary)
    return logger, summary


def train_quadratic(problem, cfg, log_dir, seed=0):
    from src.utils import set_seed
    set_seed(seed)

    device = problem.theta0.device
    train_cfg = cfg["train"]
    max_steps = train_cfg["max_steps"]
    eval_interval = train_cfg.get("eval_interval", 50)

    theta = problem.theta0.clone().requires_grad_(True)

    opt_cfg = cfg["optimizer"]
    if opt_cfg["name"] in ("adam", "adam_l2", "adamw"):
        from src.optimizers import AdamFamily
        optimizer = AdamFamily(
            [theta],
            lr=opt_cfg["lr"],
            betas=tuple(opt_cfg["betas"]),
            eps=opt_cfg["eps"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            mode=opt_cfg["name"],
        )
    elif opt_cfg["name"] == "sgd":
        from src.optimizers import SGD
        optimizer = SGD(
            [theta],
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.0),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['name']}")

    wd = float(opt_cfg.get("weight_decay", 0.0))
    logger = Logger(log_dir)
    t_start = time.time()
    theta_star = problem.theta_star

    for step in range(1, max_steps + 1):
        def closure():
            optimizer.zero_grad()
            loss = problem.loss(theta)
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        if step % eval_interval == 0 or step == max_steps:
            with torch.no_grad():
                g = problem.grad(theta)
                g_norm_val = g.norm().item()
                raw_loss = problem.loss(theta).item()
                f_diff = raw_loss - problem.optimal_loss
                theta_norm = theta.norm().item()
                # Explicit regularized objective, evaluated with the same lambda
                # every optimizer is asked to minimize. For Adam-L2 this is an
                # evaluation metric, not the implicit objective it optimizes.
                reg_loss = raw_loss + 0.5 * wd * theta_norm ** 2

            logger.log({
                "step": step,
                "loss": raw_loss,
                "reg_loss": reg_loss,
                "loss_diff": f_diff,
                "grad_norm": g_norm_val,
                "weight_norm": theta_norm,
                "wall_time_s": time.time() - t_start,
            })

    summary = {
        "final_step": max_steps,
        "final_loss_diff": logger.entries[-1]["loss_diff"] if logger.entries else float("nan"),
        "final_reg_loss": logger.entries[-1]["reg_loss"] if logger.entries else float("nan"),
        "final_grad_norm": logger.entries[-1]["grad_norm"] if logger.entries else float("nan"),
        "final_weight_norm": logger.entries[-1]["weight_norm"] if logger.entries else float("nan"),
        "wall_time_s": time.time() - t_start,
    }
    logger.save_summary(summary)
    return logger, summary