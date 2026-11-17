"""Serializable per-client learning-rate scheduling."""
import math
import torch

def build_client_scheduler(optimizer, optim_cfg):
    total_epochs = max(1, int(optim_cfg.ROUND) * int(optim_cfg.MAX_EPOCH))
    warmup = int(optim_cfg.WARMUP_EPOCH)
    initial_factor = min(1.0, float(optim_cfg.WARMUP_CONS_LR) / float(optim_cfg.LR))
    def lr_factor(step):
        if step < warmup:
            return initial_factor + (1.0 - initial_factor) * step / max(1, warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_epochs - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
