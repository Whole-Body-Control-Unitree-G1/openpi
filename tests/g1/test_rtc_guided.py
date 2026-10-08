"""PORT_PLAN Step 4: guided inference-time RTC.

    OPENPI_G1_WEIGHTS=<dir with model.safetensors> OPENPI_G1_GOLDEN=<golden .pt> \
        uv run pytest tests/g1/test_rtc_guided.py -v -s
"""

import os
import pathlib
import sys
import time as _time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
import safetensors.torch
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch import rtc_guided as rtc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "g1"))
import make_golden as mg  # noqa: E402

WEIGHTS = os.environ.get("OPENPI_G1_WEIGHTS")
GOLDEN = os.environ.get("OPENPI_G1_GOLDEN")
needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and WEIGHTS and GOLDEN), reason="needs CUDA, OPENPI_G1_WEIGHTS, OPENPI_G1_GOLDEN"
)


# ---------------------------------------------------------------- prefix weights


def _kinetix_weights(start, end, total, schedule):
    """Verbatim numpy transcription of kinetix `get_prefix_weights` (src/model.py)."""
    start = min(start, end)
    if schedule == "ones":
        w = np.ones(total)
    elif schedule == "zeros":
        w = (np.arange(total) < start).astype(np.float32)
    elif schedule in ("linear", "exp"):
        w = np.clip((start - 1 - np.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * np.expm1(w) / (np.e - 1)
    return np.where(np.arange(total) >= end, 0, w)


def _lerobot_linear(start, end, total):
    """LeRobot RTCProcessor.get_prefix_weights (LINEAR), transcribed: leading ones, linspace(1, 0)[1:-1], zeros."""
    start = min(start, end)
    n = total - max(total - end, 0) - start
    mid = torch.linspace(1, 0, n + 2)[1:-1] if (end > start and n > 0) else torch.tensor([])
    return torch.cat([torch.ones(min(start, total)), mid, torch.zeros(max(total - end, 0))])


@pytest.mark.parametrize("schedule", ["linear", "exp", "ones", "zeros"])
@pytest.mark.parametrize(("start", "end", "total"), [(2, 6, 10), (0, 10, 10), (8, 40, 50), (13, 37, 50), (5, 3, 10)])
def test_prefix_weights_match_kinetix(schedule, start, end, total):
    ours = rtc.get_prefix_weights(start, end, total, schedule).numpy()
    np.testing.assert_allclose(ours, _kinetix_weights(start, end, total, schedule), rtol=1e-6, atol=1e-7)


def test_prefix_weights_docstring_example():
    w = rtc.get_prefix_weights(2, 6, 10, "linear")
    torch.testing.assert_close(w, torch.tensor([1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0], dtype=torch.float32))


@pytest.mark.parametrize(("start", "end", "total"), [(2, 6, 10), (8, 40, 50), (13, 37, 50)])
def test_prefix_weights_match_lerobot_linear(start, end, total):
    torch.testing.assert_close(rtc.get_prefix_weights(start, end, total, "linear"), _lerobot_linear(start, end, total))


# ---------------------------------------------------------------- guidance math on a toy denoiser (CPU)


def _toy_denoiser(seed=0, dim=6, dtype=torch.float32):
    """A nonlinear velocity field v(x) = tanh(x @ A) @ B + c, per action, same for all actions."""
    g = torch.Generator().manual_seed(seed)
    a, b = torch.randn(dim, dim, generator=g) * 0.5, torch.randn(dim, dim, generator=g) * 0.5
    c = torch.randn(dim, generator=g) * 0.1
    a, b, c = a.to(dtype), b.to(dtype), c.to(dtype)
    return lambda x: torch.tanh(x @ a) @ b + c


def test_vjp_correction_matches_finite_differences():
    torch.manual_seed(0)
    f = _toy_denoiser(dim=4, dtype=torch.float64)
    x = torch.randn(1, 5, 4, dtype=torch.float64)
    prev = torch.randn(1, 5, 4, dtype=torch.float64)
    t = torch.tensor(0.6, dtype=torch.float64)
    w = rtc.get_prefix_weights(2, 4, 5, "linear").double()
    cfg = rtc.GuidedRTCConfig(max_guidance_weight=1e9, use_vjp=True)  # no clipping: isolate the correction
    v = f(x)
    guided = rtc.guided_velocity(f, x, t, prev, w, cfg)
    correction = (v - guided) / rtc.guidance_weight(t, cfg.max_guidance_weight)

    def x1_hat(z):
        return z - t * f(z)

    err = ((prev - x1_hat(x)) * w[None, :, None]).reshape(-1)
    eps, numeric = 1e-6, torch.zeros_like(x).reshape(-1)
    for i in range(x.numel()):
        dx = torch.zeros_like(x).reshape(-1)
        dx[i] = eps
        jac_col = (x1_hat(x + dx.view_as(x)) - x1_hat(x - dx.view_as(x))).reshape(-1) / (2 * eps)
        numeric[i] = (jac_col * err).sum()
    torch.testing.assert_close(correction.reshape(-1), numeric, rtol=1e-6, atol=1e-8)


def test_no_vjp_mode_is_lerobot_identity_jacobian():
    f = _toy_denoiser()
    x, prev = torch.randn(2, 8, 6), torch.randn(2, 8, 6)
    t = torch.tensor(0.4)
    w = rtc.get_prefix_weights(3, 6, 8, "exp")
    cfg = rtc.GuidedRTCConfig(max_guidance_weight=5.0, use_vjp=False)
    guided = rtc.guided_velocity(f, x, t, prev, w, cfg)
    v = f(x)
    expected = v - rtc.guidance_weight(t, 5.0) * (prev - (x - t * v)) * w[None, :, None]  # LeRobot denoise_step
    torch.testing.assert_close(guided, expected)


def test_guidance_weight_matches_paper_in_paper_time():
    """openpi t -> paper tau = 1 - t: w = min(beta, (1-tau)/tau * (tau^2 + (1-tau)^2) / (1-tau)^2)."""
    for t in [1.0, 0.9, 0.5, 0.2, 0.1]:
        tau = 1 - t
        c = (1 - tau) / tau if tau > 0 else float("inf")
        expected = min(5.0, c * (tau**2 + (1 - tau) ** 2) / (1 - tau) ** 2) if tau > 0 else 5.0
        assert rtc.guidance_weight(torch.tensor(t), 5.0).item() == pytest.approx(expected, rel=1e-5)


def _plain_euler(f, noise, num_steps):
    """Same loop as PI0Pytorch.sample_actions."""
    dt = torch.tensor(-1.0 / num_steps)
    x, time = noise, torch.tensor(1.0)
    while time >= -dt / 2:
        x = x + dt * f(x, time.expand(noise.shape[0]))
        time += dt
    return x


@pytest.mark.parametrize("use_vjp", [True, False])
def test_zero_weights_equal_plain_euler_exactly(use_vjp):
    f = _toy_denoiser()
    noise, prev = torch.randn(2, 10, 6), torch.randn(2, 10, 6)
    cfg = rtc.GuidedRTCConfig(schedule="zeros", use_vjp=use_vjp)
    out = rtc.guided_euler(lambda x, t: f(x), noise, 10, prev, 0, 10, cfg)  # zeros schedule + start 0 -> W = 0
    assert torch.equal(out, _plain_euler(lambda x, t: f(x), noise, 10))


@pytest.mark.parametrize("use_vjp", [True, False])
def test_guidance_pulls_prefix_toward_previous_chunk(use_vjp):
    f = _toy_denoiser()
    torch.manual_seed(1)
    noise, prev = torch.randn(1, 20, 6), torch.randn(1, 20, 6) * 2
    plain = _plain_euler(lambda x, t: f(x), noise, 10)
    guided = rtc.guided_euler(lambda x, t: f(x), noise, 10, prev, 5, 15, rtc.GuidedRTCConfig(use_vjp=use_vjp))
    d_plain = (plain[:, :5] - prev[:, :5]).abs().mean()
    d_guided = (guided[:, :5] - prev[:, :5]).abs().mean()
    print(f"use_vjp={use_vjp}: |prefix - prev| plain {d_plain:.3f} -> guided {d_guided:.3f}")
    assert d_guided < 0.5 * d_plain
    torch.testing.assert_close(guided[:, 15:], plain[:, 15:], rtol=0.5, atol=0.5)  # weight 0 tail: stays close


def test_align_prev_chunk_pads():
    prev = torch.ones(1, 7, 3)
    out = rtc.align_prev_chunk(prev, 10, 5)
    assert out.shape == (1, 10, 5)
    assert out[:, :7, :3].eq(1).all() and out[:, 7:].eq(0).all() and out[:, :, 3:].eq(0).all()


# ---------------------------------------------------------------- real pi05_base (GPU)


@pytest.fixture(scope="module")
def model():
    with torch.device("cuda"):
        m = pi0_pytorch.PI0Pytorch(mg.model_config())
    safetensors.torch.load_model(m, os.path.join(WEIGHTS, "model.safetensors"), strict=True, device="cuda")
    mg.strict_numerics()
    return m.eval()


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN)


@needs_gpu
@pytest.mark.parametrize("use_vjp", [True, False])
def test_zero_weights_reproduce_golden(model, golden, use_vjp):
    obs = mg.to_observation(golden["inputs"], "cuda")
    noise = golden["inputs"]["noise"].cuda()
    prev = torch.randn_like(noise)
    out = rtc.sample_actions_guided(
        model, "cuda", obs, prev_chunk=prev, inference_delay=0, prefix_attention_horizon=50,
        config=rtc.GuidedRTCConfig(schedule="zeros", use_vjp=use_vjp), noise=noise, num_steps=mg.NUM_STEPS,
    )
    assert torch.equal(out.cpu(), golden["outputs"]["sample_actions"])


@needs_gpu
@pytest.mark.parametrize("use_vjp", [True, False])
def test_real_model_follows_previous_chunk(model, golden, use_vjp):
    """Previous chunk = a sample from different noise, shifted by s = 20. The first d = 8 actions should follow it."""
    obs = mg.to_observation(golden["inputs"], "cuda")
    noise = golden["inputs"]["noise"].cuda()
    g = torch.Generator(device="cuda").manual_seed(3)
    with torch.no_grad():
        other = model.sample_actions("cuda", obs, noise=torch.randn(noise.shape, generator=g, device="cuda"))
    s, d, horizon = 20, 8, noise.shape[1]
    prev = other[:, s:]  # aligned: old[s + j] <-> new[j]
    plain = golden["outputs"]["sample_actions"].cuda()
    guided = rtc.sample_actions_guided(
        model, "cuda", obs, prev_chunk=prev, inference_delay=d, prefix_attention_horizon=horizon - s,
        config=rtc.GuidedRTCConfig(max_guidance_weight=10.0, use_vjp=use_vjp), noise=noise, num_steps=mg.NUM_STEPS,
    )
    d_plain = (plain[:, :d] - prev[:, :d]).abs().mean().item()
    d_guided = (guided[:, :d] - prev[:, :d]).abs().mean().item()
    print(f"use_vjp={use_vjp}: first {d} actions |chunk - prev|: plain {d_plain:.4f} -> guided {d_guided:.4f}")
    assert d_guided < 0.5 * d_plain


@needs_gpu
def test_latency_report(model, golden):
    obs = mg.to_observation({k: v[:1] for k, v in golden["inputs"].items()}, "cuda")
    noise = golden["inputs"]["noise"][:1].cuda()
    prev = torch.zeros_like(noise)

    def timed(fn, reps=5):
        fn()
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (_time.perf_counter() - t0) / reps * 1000

    with torch.no_grad():
        vanilla = timed(lambda: model.sample_actions("cuda", obs, noise=noise, num_steps=10))
    kw = {"prev_chunk": prev, "inference_delay": 8, "prefix_attention_horizon": 30, "noise": noise, "num_steps": 10}
    vjp = timed(lambda: rtc.sample_actions_guided(model, "cuda", obs, config=rtc.GuidedRTCConfig(use_vjp=True), **kw))
    novjp = timed(lambda: rtc.sample_actions_guided(model, "cuda", obs, config=rtc.GuidedRTCConfig(use_vjp=False), **kw))
    print(f"latency (batch 1, fp32, 10 steps, RTX 6000 Ada): vanilla {vanilla:.0f} ms, guided+VJP {vjp:.0f} ms, "
          f"guided no-VJP {novjp:.0f} ms")
