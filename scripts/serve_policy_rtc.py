"""Websocket policy server with guided real-time chunking (PORT_PLAN Step 5a). PyTorch checkpoints only.

Same arguments as scripts/serve_policy.py plus the RTC settings. Requests without `rtc_prev_actions` are served
exactly like serve_policy.py; see src/openpi/policies/policy_rtc.py for the request keys.

    uv run scripts/serve_policy_rtc.py --policy.config pi05_aloha --policy.dir <pytorch ckpt dir> --port 8000
"""

import dataclasses
import logging
import socket

import tyro

from openpi.models_pytorch import rtc_guided as _rtc
from openpi.policies import policy_config as _policy_config
from openpi.policies import policy_rtc as _policy_rtc
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Policy:
    # Training config name (e.g. "pi05_aloha").
    config: str
    # PyTorch checkpoint directory (contains model.safetensors and assets/).
    dir: str


@dataclasses.dataclass
class Args:
    policy: Policy
    port: int = 8000
    default_prompt: str | None = None
    # Flow-matching denoising steps.
    num_steps: int = 10
    # RTC guidance cap (paper: 5 with 5 steps; LeRobot: 10 with 10 steps).
    max_guidance_weight: float = 5.0
    # Soft-mask schedule: "exp" (paper), "linear", "ones", "zeros".
    schedule: str = "exp"
    # True: the paper's VJP guidance. False: LeRobot's identity-Jacobian approximation (no backward pass).
    use_vjp: bool = True


def main(args: Args) -> None:
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.policy.config),
        args.policy.dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"num_steps": args.num_steps},
    )
    rtc_config = _rtc.GuidedRTCConfig(
        max_guidance_weight=args.max_guidance_weight, schedule=args.schedule, use_vjp=args.use_vjp
    )
    policy = _policy_rtc.RTCPolicy.from_policy(policy, rtc_config)
    logging.info(f"RTC: {rtc_config}, num_steps={args.num_steps}")

    hostname = socket.gethostname()
    logging.info("Creating server (host: %s, ip: %s)", hostname, socket.gethostbyname(hostname))
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
