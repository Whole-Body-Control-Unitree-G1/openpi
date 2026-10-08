"""A Policy that runs guided real-time chunking on the server (PORT_PLAN Step 5a). PyTorch models only.

Request keys (all optional; without `rtc_prev_actions` the call is a normal `Policy.infer`):

    rtc_prev_actions         (T, D)  the previous chunk's actions that are still ahead of the new observation, in
                                     robot units (as the policy returned them), aligned so row 0 is the action for the
                                     new observation's time step (LeRobot: `prev_chunk_left_over`)
    rtc_inference_delay      int     frozen prefix length in steps (d + l: inference delay + controller lag)
    rtc_prefix_attention_horizon int  first step that ignores the previous chunk (paper: H - s); default T

The previous actions go through the **same input transforms as training** together with the new observation, so
relative (delta) actions are re-anchored to the new state and normalized exactly like training targets (LeRobot's
`reanchor_relative_rtc_prefix` does the same with its relative + normalizer steps). The server is stateless.
"""

from collections.abc import Sequence
import dataclasses
import time
from typing import Any

import jax
import numpy as np
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models_pytorch import rtc_guided as _rtc
from openpi.policies import policy as _policy

PREV_ACTIONS = "rtc_prev_actions"
INFERENCE_DELAY = "rtc_inference_delay"
PREFIX_ATTENTION_HORIZON = "rtc_prefix_attention_horizon"
USE_VJP = "rtc_use_vjp"  # optional per-request override of GuidedRTCConfig.use_vjp (for A/B runs)
RTC_KEYS = (PREV_ACTIONS, INFERENCE_DELAY, PREFIX_ATTENTION_HORIZON, USE_VJP)


class RTCPolicy(_policy.Policy):
    def __init__(
        self,
        model,
        *,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cuda",
        rtc_config: _rtc.GuidedRTCConfig = _rtc.GuidedRTCConfig(),
    ):
        super().__init__(
            model,
            transforms=transforms,
            output_transforms=output_transforms,
            sample_kwargs=sample_kwargs,
            metadata=metadata,
            pytorch_device=pytorch_device,
            is_pytorch=True,
        )
        self._rtc_config = rtc_config
        self._model.requires_grad_(False)  # inference: the VJP needs gradients w.r.t. x_t only

    @classmethod
    def from_policy(cls, policy: _policy.Policy, rtc_config: _rtc.GuidedRTCConfig) -> "RTCPolicy":
        """Wrap a PyTorch policy made by `policy_config.create_trained_policy`."""
        if not policy._is_pytorch_model:  # noqa: SLF001
            raise ValueError("RTCPolicy supports PyTorch policies only")
        rtc_policy = cls.__new__(cls)
        rtc_policy.__dict__.update(policy.__dict__)
        rtc_policy._rtc_config = rtc_config  # noqa: SLF001
        rtc_policy._model.requires_grad_(False)  # noqa: SLF001
        return rtc_policy

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        obs = dict(obs)
        rtc = {k: obs.pop(k) for k in RTC_KEYS if k in obs}
        if PREV_ACTIONS not in rtc:
            return super().infer(obs, noise=noise)

        prev = np.asarray(rtc[PREV_ACTIONS], dtype=np.float32)
        if prev.ndim != 2 or len(prev) == 0:
            raise ValueError(f"{PREV_ACTIONS} must be (T, D) with T > 0, got {prev.shape}")
        delay = int(rtc.get(INFERENCE_DELAY, 0))
        horizon = int(rtc.get(PREFIX_ATTENTION_HORIZON, len(prev)))

        # Re-anchor + normalize + pad the previous actions with the training input transforms, next to this observation.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs["actions"] = prev
        inputs = self._input_transform(inputs)
        prefix = torch.from_numpy(np.asarray(inputs.pop("actions"), dtype=np.float32)).to(self._pytorch_device)
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_t = torch.from_numpy(noise).to(self._pytorch_device)
            sample_kwargs["noise"] = noise_t[None] if noise_t.ndim == 2 else noise_t

        start_time = time.monotonic()
        actions = _rtc.sample_actions_guided(
            self._model,
            self._pytorch_device,
            observation,
            prev_chunk=prefix[None],
            inference_delay=delay,
            prefix_attention_horizon=horizon,
            config=self._rtc_config if USE_VJP not in rtc else dataclasses.replace(
                self._rtc_config, use_vjp=bool(rtc[USE_VJP])
            ),
            **sample_kwargs,
        )
        if self._pytorch_device.startswith("cuda"):
            torch.cuda.synchronize()
        model_time = time.monotonic() - start_time

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().float().cpu()), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        outputs["rtc"] = {"inference_delay": delay, "prefix_attention_horizon": horizon, "prev_len": len(prev),
                          "use_vjp": bool(rtc.get(USE_VJP, self._rtc_config.use_vjp))}
        return outputs
