"""Client-side action queue for real-time chunking at a fixed control rate (PORT_PLAN Step 5b).

Ported from LeRobot `policies/rtc/action_queue.py` (ActionQueue) and `latency_tracker.py` (LatencyTracker), with two
changes for a robot whose low-level controller lags its commands:

* Time stamps instead of a moving pointer. Every chunk is stored with the control tick at which its observation was
  taken (`t_obs`); chunk index j is the target for tick `t_obs + j`. LeRobot's "drop the actions consumed during
  inference" is then implicit.
* Controller lag `lag` (l, in ticks): the controller (ScaleBFM) reaches a commanded pose l ticks later, so at tick
  `now` the target for tick `now + l` is sent. The RTC frozen prefix is therefore `d + l` (d = inference delay).

Ticks are integers at `1 / dt` Hz. The queue is not thread-safe by itself; RTCClient guards it with a lock.
"""

from collections import deque
import dataclasses
import math

import numpy as np


class LatencyTracker:
    """Recent inference latencies (seconds), max / percentile queries. Port of LeRobot's LatencyTracker."""

    def __init__(self, maxlen: int = 100) -> None:
        self._values: deque[float] = deque(maxlen=maxlen)

    def add(self, latency: float) -> None:
        if latency >= 0:
            self._values.append(float(latency))

    def __len__(self) -> int:
        return len(self._values)

    def max(self) -> float:
        return max(self._values) if self._values else 0.0

    def percentile(self, q: float) -> float:
        return float(np.quantile(np.asarray(self._values), q)) if self._values else 0.0


@dataclasses.dataclass(frozen=True)
class QueueConfig:
    dt: float = 0.02  # control period (50 Hz)
    horizon: int = 50  # H, actions per chunk
    execute_horizon: int = 20  # s, ticks between observations (a new chunk is requested every s ticks)
    lag: int = 0  # l, controller lag in ticks
    lookahead: tuple[int, ...] = (0, 1, 2, 3, 4)  # extra target frames sent to the controller (ScaleBFM future_idx)
    delay_margin: int = 1  # ticks added to the estimated inference delay
    initial_delay: int = 10  # d estimate (ticks) before any latency was measured
    rtc: bool = True  # False: no previous chunk is sent (plain async chunk switching)

    def check(self, delay: int) -> list[str]:
        """The RTC timing rules for a given inference delay d (ticks); returns the violated ones."""
        frozen, s, horizon, k = delay + self.lag, self.execute_horizon, self.horizon, max(self.lookahead)
        problems = []
        if not frozen <= s:
            problems.append(f"d + l = {frozen} > s = {s}: a new chunk would be requested before the last one arrived")
        if not s <= horizon - frozen:
            problems.append(f"s = {s} > H - (d + l) = {horizon - frozen}")
        if not s + frozen + k <= horizon:
            problems.append(f"s + d + l + K = {s + frozen + k} > H = {horizon}: the chunk runs out before the next one")
        return problems


@dataclasses.dataclass
class Chunk:
    actions: np.ndarray  # (H, D) absolute targets, robot units
    t_obs: int  # tick of the observation it was computed from


@dataclasses.dataclass
class Request:
    t_obs: int
    prev_actions: np.ndarray | None  # aligned to t_obs (row 0 = target for tick t_obs), absolute
    inference_delay: int  # d_est + l, the frozen prefix length the server should use
    prefix_attention_horizon: int  # H - s


@dataclasses.dataclass
class Stats:
    ticks: int = 0
    starved_ticks: int = 0  # ticks with no fresh target (held the last action)
    lookahead_clipped: int = 0  # ticks where some lookahead frame was past the end of the chunk
    chunks: list[dict] = dataclasses.field(default_factory=list)  # one record per received chunk


class ActionQueue:
    def __init__(self, config: QueueConfig):
        self.config = config
        self.current: Chunk | None = None
        self.latency = LatencyTracker()
        self.stats = Stats()
        self._last_sent: np.ndarray | None = None
        self._pending: Request | None = None

    # --- timing

    def delay_estimate(self) -> int:
        """d in ticks: max of recent latencies (LeRobot's conservative choice) plus a margin."""
        if not len(self.latency):
            return self.config.initial_delay
        return math.ceil(self.latency.max() / self.config.dt) + self.config.delay_margin

    # --- requests

    def should_request(self, now: int) -> bool:
        if self._pending is not None:
            return False
        return self.current is None or now - self.current.t_obs >= self.config.execute_horizon

    def make_request(self, now: int) -> Request:
        """Call when the observation for tick `now` is captured."""
        prev = None
        if self.config.rtc and self.current is not None:
            offset = now - self.current.t_obs
            if offset < self.config.horizon:
                prev = self.current.actions[offset:].copy()
        self._pending = Request(
            t_obs=now,
            prev_actions=prev,
            inference_delay=self.delay_estimate() + self.config.lag,
            prefix_attention_horizon=self.config.horizon - self.config.execute_horizon,
        )
        return self._pending

    def receive(self, actions: np.ndarray, request: Request, now: int, latency_s: float) -> None:
        """Install a chunk computed for `request`; `now` is the tick at which it arrived."""
        self.latency.add(latency_s)
        old = self.current
        self.current = Chunk(np.asarray(actions), request.t_obs)
        self._pending = None
        real_delay = now - request.t_obs
        record = {
            "t_obs": request.t_obs,
            "arrived": now,
            "real_delay": real_delay,
            "frozen_requested": request.inference_delay,
            "latency_ms": latency_s * 1000,
            "late": real_delay + self.config.lag > request.inference_delay,  # committed more than was frozen
        }
        if old is not None and self._last_sent is not None:
            first_new = self._target(self.current, now)
            record["handover_jump"] = np.abs(first_new - self._last_sent).max() if first_new is not None else None
        self.stats.chunks.append(record)

    # --- control tick

    def _target(self, chunk: Chunk, now: int, extra: int = 0) -> np.ndarray | None:
        index = now + self.config.lag + extra - chunk.t_obs
        if 0 <= index < len(chunk.actions):
            return chunk.actions[index]
        return None

    def tick(self, now: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """The target to send at tick `now` (for tick now + l) and its lookahead frames (len(lookahead), D)."""
        self.stats.ticks += 1
        if self.current is None:
            return None, None
        target = self._target(self.current, now)
        if target is None:  # ran past the end of the chunk: hold the last action
            self.stats.starved_ticks += 1
            target = self._last_sent if self._last_sent is not None else self.current.actions[-1]
        frames, clipped = [], False
        for k in self.config.lookahead:
            frame = self._target(self.current, now, k)
            if frame is None:
                clipped = True
                frame = frames[-1] if frames else target
            frames.append(frame)
        self.stats.lookahead_clipped += int(clipped)
        self._last_sent = np.asarray(target).copy()
        return self._last_sent, np.stack(frames)
