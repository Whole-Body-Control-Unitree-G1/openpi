"""Exponential moving average (EMA) of the weights for the PyTorch trainer, matching the JAX trainer.

JAX (`scripts/train.py`, `training/checkpoints.py`):
  * init:   ema = the initial (pretrained) params
  * update: after every optimizer step, ema = decay * ema + (1 - decay) * params   (no warmup, no bias correction)
  * save:   the checkpoint's inference `params` are the EMA params; the raw params stay in the train state (resume)

Here only the trainable parameters are tracked (frozen ones never change, so their EMA equals the param). The EMA is
kept in fp32. The trainer saves `model.safetensors` with the EMA weights swapped in (so serving uses EMA, as in JAX)
and the raw trainable weights in `raw_trainable.safetensors`.
"""

import contextlib

import torch
from torch import nn

RAW_TRAINABLE_FILE = "raw_trainable.safetensors"


class ParamEMA:
    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        params = dict(model.named_parameters())
        self.params = [params[n] for n in self.names]
        self.shadow = [p.detach().clone().float() for p in self.params]

    @torch.no_grad()
    def update(self) -> None:
        """ema = decay * ema + (1 - decay) * param, for every tracked param."""
        current = [p.detach().float() for p in self.params]
        torch._foreach_mul_(self.shadow, self.decay)
        torch._foreach_add_(self.shadow, current, alpha=1.0 - self.decay)

    @contextlib.contextmanager
    def swapped(self):
        """Temporarily put the EMA values into the model params (e.g. to save them); restore the raw values after."""
        with torch.no_grad():
            raw = [p.detach().clone() for p in self.params]
            for p, e in zip(self.params, self.shadow, strict=True):
                p.copy_(e.to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for p, r in zip(self.params, raw, strict=True):
                    p.copy_(r)

    def raw_state(self) -> dict[str, torch.Tensor]:
        """The raw (non-EMA) trainable weights, for resuming."""
        return {n: p.detach().contiguous() for n, p in zip(self.names, self.params, strict=True)}

    @torch.no_grad()
    def resume(self, raw: dict[str, torch.Tensor]) -> None:
        """Resume after `model.safetensors` (EMA values) was loaded into the model: EMA <- current params, params <- raw."""
        missing = set(self.names) - set(raw)
        if missing:
            raise KeyError(f"raw_trainable checkpoint is missing {sorted(missing)[:5]}")
        for p, e, n in zip(self.params, self.shadow, self.names, strict=True):
            e.copy_(p.float())
            p.copy_(raw[n].to(device=p.device, dtype=p.dtype))
