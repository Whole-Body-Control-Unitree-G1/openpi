"""Guided (inference-time) real-time chunking for the PyTorch pi0 / pi0.5 (PORT_PLAN Step 4).

Paper: Black, Galliker, Levine, "Real-Time Execution of Action Chunking Flow Policies" (arXiv 2506.07339).
Reference: Physical-Intelligence/real-time-chunking-kinetix, `src/model.py`: `get_prefix_weights`, `realtime_action`.

At each denoising step the velocity is corrected so that the predicted clean chunk moves toward the previous chunk on
the overlap, weighted per action by a soft mask W (1 on the first `inference_delay` actions, then a decay, 0 from
`prefix_attention_horizon` on):

    x1_hat     = clean-chunk estimate from (x_t, v_t)
    err        = (prev - x1_hat) * W
    correction = (d x1_hat / d x_t)^T err          # a vector-Jacobian product through the network (the paper)
    v_guided   = v_t -/+ min(beta, c * 1/r^2) * correction

openpi's flow convention differs from the reference (Saif's JAX port spells this out): time t = 1 is noise and t = 0
is the action (`x_t = t * noise + (1 - t) * actions`, `v = noise - actions`, Euler with dt = -1/n from t = 1), so
`x1_hat = x_t - t * v_t`, the paper's time is tau = 1 - t, and the correction is subtracted.

`use_vjp=False` drops the Jacobian (correction = err). That is what LeRobot's `RTCProcessor.denoise_step` computes,
because it evaluates the velocity before `x_t.requires_grad_(True)`; it needs no backward pass.

`prev_chunk` must already be aligned to the new chunk (index 0 = the new chunk's first action, i.e. the previous
chunk shifted by the number of steps since its observation) and be in model space (normalized, padded to action_dim).
"""

import dataclasses
import math
from collections.abc import Callable
from typing import Literal

import torch

Schedule = Literal["linear", "exp", "ones", "zeros"]


@dataclasses.dataclass(frozen=True)
class GuidedRTCConfig:
    # Guidance cap beta (paper: 5 with 5 denoising steps; LeRobot suggests 10 with 10 steps).
    max_guidance_weight: float = 5.0
    # Soft-mask shape between the frozen prefix and `prefix_attention_horizon` (paper and kinetix default: exp).
    schedule: Schedule = "exp"
    # True: the paper's vector-Jacobian product. False: LeRobot's identity-Jacobian approximation (no backward pass).
    use_vjp: bool = True


def get_prefix_weights(start: int, end: int, total: int, schedule: Schedule) -> torch.Tensor:
    """Port of kinetix `get_prefix_weights`. With start=2, end=6, total=10 (linear): 1 1 .8 .6 .4 .2 0 0 0 0.

    `start` (inclusive): first action allowed to change (the frozen prefix is [0, start)). `end` (exclusive): first
    action that ignores the previous chunk. `end` takes precedence: if end < start, start is pushed down to end.
    """
    start = min(start, end)
    idx = torch.arange(total, dtype=torch.float32)
    if schedule == "ones":
        w = torch.ones(total)
    elif schedule == "zeros":
        w = (idx < start).float()
    elif schedule in ("linear", "exp"):
        w = torch.clamp((start - 1 - idx) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * torch.expm1(w) / (math.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return torch.where(idx >= end, torch.zeros_like(w), w)


def guidance_weight(time: torch.Tensor, max_guidance_weight: float) -> torch.Tensor:
    """min(beta, c / r^2) of the paper, written in openpi time (t = 1 noise); tau = 1 - t is the paper's time."""
    tau = 1.0 - time
    inv_r2 = (tau**2 + (1.0 - tau) ** 2) / (1.0 - tau) ** 2
    c = torch.nan_to_num((1.0 - tau) / tau, posinf=max_guidance_weight)
    return torch.clamp(c * inv_r2, max=max_guidance_weight)


def align_prev_chunk(prev: torch.Tensor, horizon: int, action_dim: int) -> torch.Tensor:
    """Zero-pad an aligned previous chunk (B, T, D') to (B, horizon, action_dim); entries past T get weight 0 anyway."""
    if prev.ndim == 2:
        prev = prev[None]
    batch, steps, dims = prev.shape
    if steps > horizon or dims > action_dim:
        raise ValueError(f"prev_chunk {tuple(prev.shape)} exceeds (horizon={horizon}, action_dim={action_dim})")
    out = prev.new_zeros(batch, horizon, action_dim)
    out[:, :steps, :dims] = prev
    return out


def guided_velocity(
    denoise_fn: Callable[[torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    time: torch.Tensor,
    prev_chunk: torch.Tensor,
    weights: torch.Tensor,
    config: GuidedRTCConfig,
) -> torch.Tensor:
    """One RTC-corrected velocity. `denoise_fn(x_t) -> v_t`; `time` is a 0-d float tensor (openpi convention)."""
    w = weights.to(device=x_t.device, dtype=x_t.dtype)[None, :, None]
    if config.use_vjp:
        with torch.enable_grad():
            x = x_t.detach().requires_grad_(True)
            v_t = denoise_fn(x)
            x1_hat = x - time * v_t
            err = (prev_chunk - x1_hat) * w
            correction = torch.autograd.grad(x1_hat, x, err.detach())[0]
        v_t = v_t.detach()
    else:
        with torch.no_grad():  # no backward pass: do not record a graph through the model
            v_t = denoise_fn(x_t)
        x1_hat = x_t - time * v_t
        correction = (prev_chunk - x1_hat) * w  # identity Jacobian (LeRobot)
    return v_t - guidance_weight(time, config.max_guidance_weight) * correction


def guided_euler(
    denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    noise: torch.Tensor,
    num_steps: int,
    prev_chunk: torch.Tensor,
    inference_delay: int,
    prefix_attention_horizon: int,
    config: GuidedRTCConfig,
) -> torch.Tensor:
    """Euler integration from t = 1 to 0 with RTC guidance, using the same time arithmetic as PI0Pytorch.sample_actions.

    `denoise_fn(x_t, timestep (B,)) -> v_t`. With all weights zero this equals the unguided loop exactly.
    """
    device = noise.device
    bsize, horizon = noise.shape[:2]
    prev = align_prev_chunk(prev_chunk.to(device=device, dtype=noise.dtype), horizon, noise.shape[2])
    weights = get_prefix_weights(inference_delay, prefix_attention_horizon, horizon, config.schedule)

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_t = guided_velocity(lambda x, t=expanded_time: denoise_fn(x, t), x_t, time, prev, weights, config)
        x_t = x_t + dt * v_t
        time += dt
    return x_t


def prefix_cache(model, observation):
    """The prefix (images + prompt) KV cache, built exactly as in PI0Pytorch.sample_actions."""
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return state, prefix_pad_masks, past_key_values


def sample_actions_guided(
    model,
    device,
    observation,
    *,
    prev_chunk: torch.Tensor,
    inference_delay: int,
    prefix_attention_horizon: int,
    config: GuidedRTCConfig = GuidedRTCConfig(),
    noise: torch.Tensor | None = None,
    num_steps: int = 10,
) -> torch.Tensor:
    """PI0Pytorch.sample_actions with RTC guidance toward `prev_chunk` (aligned, model space)."""
    bsize = observation.state.shape[0]
    if noise is None:
        noise = model.sample_noise((bsize, model.config.action_horizon, model.config.action_dim), device)
    with torch.no_grad():
        state, prefix_pad_masks, past_key_values = prefix_cache(model, observation)

    def denoise_fn(x_t, timestep):
        return model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)

    return guided_euler(denoise_fn, noise, num_steps, prev_chunk, inference_delay, prefix_attention_horizon, config)
