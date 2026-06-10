"""Deterministic held-out evaluation: fixed non-overlapping windows from val.bin,
identical for every replica, run right after an outer sync (params identical then)."""

import math

import torch

from .data import TokenBins


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    bins: TokenBins,
    batch_size: int,
    max_batches: int,
    autocast_ctx,
) -> dict:
    model.eval()
    total, n = 0.0, 0
    for x, y in bins.val_batches(batch_size, max_batches):
        with autocast_ctx:
            _, loss = model(x, y)
        total += loss.item()
        n += 1
    model.train()
    mean = total / max(n, 1)
    return {"eval_loss": mean, "ppl": math.exp(min(mean, 20.0))}
