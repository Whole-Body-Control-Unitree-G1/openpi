"""Serving latency of vanilla vs guided-RTC sampling, eager vs torch.compile (PORT_PLAN Step 5).

Loads the policy like the server does (create_trained_policy -> bf16), batch 1, and reports the median of N timed
calls after warm-up (the first calls include compilation).

    uv run scripts/g1/bench_latency.py --weights <pytorch ckpt dir> --compile default
"""

import argparse
import copy
import dataclasses
import logging
import os
import statistics
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from openpi.models_pytorch import rtc_guided as _rtc
from openpi.policies import aloha_policy
from openpi.policies import policy_config
from openpi.policies import policy_rtc
from openpi.training import config as _config

H, S, D = 50, 20, 8


def _obs():
    rng = np.random.default_rng(0)
    obs = aloha_policy.make_aloha_example()
    obs["images"] = {k: rng.integers(0, 256, size=v.shape, dtype=np.uint8) for k, v in obs["images"].items()}
    obs["state"] = rng.uniform(-0.5, 0.5, size=14)
    obs["prompt"] = "fold the towel"
    return obs


def _time(fn, reps, warmup):
    t0 = time.perf_counter()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times), min(times), warm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--compile", default="default", help="torch.compile mode, or 'none' for eager")
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 5])
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    mode = None if args.compile == "none" else args.compile

    config = _config.get_config("pi05_aloha")
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=mode))
    t0 = time.perf_counter()
    base = policy_config.create_trained_policy(config, args.weights, pytorch_device="cuda")
    logging.info(f"policy loaded in {time.perf_counter() - t0:.0f} s (compile mode: {mode})")
    model = base._model  # noqa: SLF001
    model.requires_grad_(False)
    if mode is not None:  # the guided path calls denoise_step directly: compile it too (incl. its backward)
        model.denoise_step = torch.compile(model.denoise_step, mode=mode)

    obs = _obs()
    noise = np.random.default_rng(1).standard_normal((H, 32)).astype(np.float32)
    prev = base.infer(copy.deepcopy(obs), noise=noise)["actions"][S:]
    keys = {policy_rtc.PREV_ACTIONS: prev, policy_rtc.INFERENCE_DELAY: D, policy_rtc.PREFIX_ATTENTION_HORIZON: H - S}

    rows = []
    for steps in args.steps:
        base._sample_kwargs = {"num_steps": steps}  # noqa: SLF001
        runs = {"vanilla": lambda: base.infer(copy.deepcopy(obs), noise=noise)}
        for name, use_vjp in [("RTC no-VJP", False), ("RTC VJP", True)]:
            pol = policy_rtc.RTCPolicy.from_policy(base, _rtc.GuidedRTCConfig(use_vjp=use_vjp))
            runs[name] = lambda pol=pol: pol.infer({**copy.deepcopy(obs), **keys}, noise=noise)
        for name, fn in runs.items():
            logging.info(f"steps={steps} {name}: warming up / compiling ...")
            med, best, warm = _time(fn, args.reps, warmup=3)
            rows.append((steps, name, med, best, warm))
            logging.info(f"steps={steps} {name}: median {med:.0f} ms (best {best:.0f}), warm-up {warm:.0f} s")

    print(f"\nlatency, bf16, batch 1, compile={mode}, GPU {torch.cuda.get_device_name()}")
    print(f"{'steps':>5} {'mode':12} {'median ms':>10} {'best ms':>8} {'warm-up s':>10} {'d @50Hz':>8}")
    for steps, name, med, best, warm in rows:
        print(f"{steps:>5} {name:12} {med:>10.0f} {best:>8.0f} {warm:>10.0f} {int(np.ceil(med / 20)):>8}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main()
