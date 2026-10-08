"""Real-time chunking client loop (PORT_PLAN Step 5b).

A fixed-rate control loop on the robot side. Inference runs in a background thread so control never waits:

    every tick:  if a chunk is due -> capture the observation, start an async request (with the previous chunk)
                 send queue.tick(now) to the robot (the target for now + l, plus lookahead frames)

`policy` is any object with `infer(obs: dict) -> dict` (e.g. openpi_client.websocket_client_policy.WebsocketClientPolicy).
The robot side is two callbacks: `get_observation(tick) -> dict` and `send_action(tick, target, frames)`.
"""

from collections.abc import Callable
import logging
import threading
import time

import numpy as np

from openpi_client import rtc_action_queue as _queue

PREV_ACTIONS = "rtc_prev_actions"
INFERENCE_DELAY = "rtc_inference_delay"
PREFIX_ATTENTION_HORIZON = "rtc_prefix_attention_horizon"


class RTCClient:
    def __init__(
        self,
        policy,
        config: _queue.QueueConfig,
        get_observation: Callable[[int], dict],
        send_action: Callable[[int, np.ndarray, np.ndarray], None],
        clock: Callable[[], float] = time.monotonic,
    ):
        self.policy = policy
        self.config = config
        self.queue = _queue.ActionQueue(config)
        self._get_observation = get_observation
        self._send_action = send_action
        self._clock = clock
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self.errors: list[BaseException] = []
        self.extra_request: dict = {}  # extra keys sent with every request (e.g. {"rtc_use_vjp": False})

    def reset(self) -> None:
        """Safety stop / resume: drop the current plan; the next chunk starts from the current pose without RTC."""
        with self._lock:
            self.queue.reset()

    def _infer(self, obs: dict, request: _queue.Request, start: float, tick_of: Callable[[], int]) -> None:
        try:
            if request.prev_actions is not None:
                obs = {**obs, PREV_ACTIONS: request.prev_actions, INFERENCE_DELAY: request.inference_delay,
                       PREFIX_ATTENTION_HORIZON: request.prefix_attention_horizon}
            response = self.policy.infer({**obs, **self.extra_request})
            server_ms = response.get("server_timing", {}).get("infer_ms")
            with self._lock:
                self.queue.receive(np.asarray(response["actions"]), request, tick_of(), self._clock() - start, server_ms)
        except BaseException as e:  # noqa: BLE001
            logging.exception("inference failed")
            self.errors.append(e)
            with self._lock:
                self.queue._pending = None  # noqa: SLF001  allow a new request

    def step(self, now: int, tick_of: Callable[[], int]) -> None:
        """One control tick. `tick_of()` returns the current tick (read by the inference thread on arrival)."""
        with self._lock:
            due = self.queue.should_request(now)
            request = self.queue.make_request(now) if due else None
        if request is not None:
            obs = self._get_observation(now)
            self._worker = threading.Thread(target=self._infer, args=(obs, request, self._clock(), tick_of), daemon=True)
            self._worker.start()
        with self._lock:
            target, frames = self.queue.tick(now)
        if target is not None:
            self._send_action(now, target, frames)

    def run(self, num_ticks: int) -> _queue.Stats:
        """Run the loop in real time at 1 / dt Hz for `num_ticks` ticks."""
        t0 = self._clock()

        def tick_of() -> int:
            return int((self._clock() - t0) / self.config.dt)

        for now in range(num_ticks):
            deadline = t0 + now * self.config.dt
            if (wait := deadline - self._clock()) > 0:
                time.sleep(wait)
            self.step(now, tick_of)
        if self._worker is not None:
            self._worker.join(timeout=5)
        return self.queue.stats
