"""PORT_PLAN Step 5a: RTCPolicy on a real checkpoint served as in production (bf16, create_trained_policy).

Uses openpi's `pi05_aloha` config on pi05_base (relative joint actions, so the prefix must be re-anchored).

    OPENPI_G1_WEIGHTS=<pytorch pi05_base dir with assets/> uv run pytest tests/g1/test_policy_rtc.py -v -s
"""

import copy
import dataclasses
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
import torch

from openpi.models_pytorch import rtc_guided as _rtc
from openpi.policies import aloha_policy
from openpi.policies import policy_config
from openpi.policies import policy_rtc
from openpi.training import config as _config

WEIGHTS = os.environ.get("OPENPI_G1_WEIGHTS")
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and WEIGHTS), reason="needs CUDA and OPENPI_G1_WEIGHTS")

H, S, D_DELAY = 50, 20, 8


@pytest.fixture(scope="module")
def policies():
    config = _config.get_config("pi05_aloha")
    # No torch.compile here: max-autotune compiles for minutes on the first call, and the RTC path is not compiled,
    # so the latency comparison would not be like for like.
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    base = policy_config.create_trained_policy(config, WEIGHTS, sample_kwargs={"num_steps": 10}, pytorch_device="cuda")
    rtc = policy_rtc.RTCPolicy.from_policy(base, _rtc.GuidedRTCConfig(max_guidance_weight=10.0, use_vjp=True))
    return base, rtc


def _obs(seed, state=None):
    rng = np.random.default_rng(seed)
    obs = aloha_policy.make_aloha_example()
    obs["images"] = {k: rng.integers(0, 256, size=v.shape, dtype=np.uint8) for k, v in obs["images"].items()}
    obs["state"] = rng.uniform(-0.5, 0.5, size=14) if state is None else state
    obs["prompt"] = "fold the towel"
    return obs


def _noise(seed):
    return np.random.default_rng(seed).standard_normal((H, 32)).astype(np.float32)


def test_without_rtc_keys_equals_policy(policies):
    base, rtc = policies
    obs, noise = _obs(0), _noise(1)
    np.testing.assert_array_equal(rtc.infer(copy.deepcopy(obs), noise=noise)["actions"],
                                  base.infer(copy.deepcopy(obs), noise=noise)["actions"])


def test_zero_weights_equal_policy(policies):
    base, _ = policies
    rtc_zero = policy_rtc.RTCPolicy.from_policy(base, _rtc.GuidedRTCConfig(schedule="zeros"))
    obs, noise = _obs(0), _noise(1)
    prev = base.infer(copy.deepcopy(obs), noise=_noise(2))["actions"][S:]
    out = rtc_zero.infer({**copy.deepcopy(obs), policy_rtc.PREV_ACTIONS: prev, policy_rtc.INFERENCE_DELAY: 0,
                          policy_rtc.PREFIX_ATTENTION_HORIZON: H - S}, noise=noise)
    # bf16 serving path: equal up to bf16 kernel noise (bit-identical in fp32, see test_rtc_guided.py).
    np.testing.assert_allclose(out["actions"], base.infer(copy.deepcopy(obs), noise=noise)["actions"], atol=1e-6)


def test_prefix_reanchor_round_trip(policies):
    """The model-space prefix, mapped back with the output transforms at the NEW state, is the old absolute plan."""
    base, _ = policies
    obs1 = _obs(0)
    old = base.infer(copy.deepcopy(obs1), noise=_noise(2))["actions"]  # absolute, robot units
    obs2 = _obs(5)  # a different state: deltas must be re-anchored to it
    inputs = dict(copy.deepcopy(obs2))
    inputs["actions"] = old[S:]
    inputs = base._input_transform(inputs)  # noqa: SLF001
    prefix = np.asarray(inputs["actions"])  # what RTCPolicy hands to the sampler
    back = base._output_transform({"state": inputs["state"], "actions": prefix})["actions"]  # noqa: SLF001
    np.testing.assert_allclose(back, old[S:], atol=1e-5)


def test_rtc_follows_previous_chunk_in_robot_units(policies):
    """New chunk 20 steps later from a new observation: its first d actions should continue the old plan."""
    base, rtc = policies
    old = base.infer(_obs(0), noise=_noise(2))["actions"]
    obs2 = _obs(7, state=old[S - 1].copy())  # robot is where the old plan put it
    plain = base.infer(copy.deepcopy(obs2), noise=_noise(3))["actions"]
    guided = rtc.infer({**copy.deepcopy(obs2), policy_rtc.PREV_ACTIONS: old[S:], policy_rtc.INFERENCE_DELAY: D_DELAY,
                        policy_rtc.PREFIX_ATTENTION_HORIZON: H - S}, noise=_noise(3))["actions"]
    joints = [i for i in range(14) if i not in (6, 13)]  # radians (6, 13 are grippers)
    d_plain = np.abs(plain[:D_DELAY, joints] - old[S:S + D_DELAY, joints]).mean()
    d_guided = np.abs(guided[:D_DELAY, joints] - old[S:S + D_DELAY, joints]).mean()
    jump_plain = np.abs(plain[0, joints] - old[S, joints]).max()
    jump_guided = np.abs(guided[0, joints] - old[S, joints]).max()
    print(f"\nfirst {D_DELAY} actions mean |new - old| (rad): plain {d_plain:.4f} -> RTC {d_guided:.4f}; "
          f"max jump at the hand-over (rad): plain {jump_plain:.4f} -> RTC {jump_guided:.4f}")
    assert d_guided < 0.5 * d_plain


def test_serving_latency(policies):
    """bf16 serving mode, batch 1, 10 steps (one warm-up call each)."""
    base, _ = policies
    obs, noise = _obs(0), _noise(1)
    prev = base.infer(copy.deepcopy(obs), noise=_noise(2))["actions"][S:]
    rtc_keys = {policy_rtc.PREV_ACTIONS: prev, policy_rtc.INFERENCE_DELAY: D_DELAY,
                policy_rtc.PREFIX_ATTENTION_HORIZON: H - S}

    def timed(fn, reps=10):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    results = {"vanilla": timed(lambda: base.infer(copy.deepcopy(obs), noise=noise))}
    for name, use_vjp in [("RTC VJP", True), ("RTC no-VJP", False)]:
        pol = policy_rtc.RTCPolicy.from_policy(base, _rtc.GuidedRTCConfig(use_vjp=use_vjp))
        results[name] = timed(lambda pol=pol: pol.infer({**copy.deepcopy(obs), **rtc_keys}, noise=noise))
    print("\nserving latency (bf16, batch 1, 10 steps, incl. transforms): "
          + ", ".join(f"{k} {v:.0f} ms" for k, v in results.items()))
