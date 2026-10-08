"""PORT_PLAN Step 5b: client action queue and loop. CPU only, no server.

    uv run pytest tests/g1/test_rtc_client.py -v
"""

import time

import numpy as np
import pytest

from openpi_client import rtc_action_queue as q
from openpi_client import rtc_client

H = 50


def plan_chunk(t_obs, horizon=H, offset=0.0):
    """A chunk following the 'true plan' value(tick) = tick (+ an offset), from the observation at t_obs."""
    return (np.arange(t_obs, t_obs + horizon, dtype=np.float64) + offset)[:, None]


@pytest.mark.parametrize("lag", [0, 5])
def test_sends_target_for_now_plus_lag(lag):
    queue = q.ActionQueue(q.QueueConfig(lag=lag))
    req = queue.make_request(0)
    queue.receive(plan_chunk(0), req, now=8, latency_s=0.16)
    for now in range(8, 30):
        target, frames = queue.tick(now)
        assert target[0] == now + lag
        np.testing.assert_array_equal(frames[:, 0], now + lag + np.array([0, 1, 2, 3, 4]))


def test_requests_every_s_ticks_and_one_at_a_time():
    queue = q.ActionQueue(q.QueueConfig(execute_horizon=20))
    assert queue.should_request(0)
    req = queue.make_request(0)
    assert not queue.should_request(1)  # one request in flight
    queue.receive(plan_chunk(0), req, now=8, latency_s=0.16)
    assert [t for t in range(8, 60) if queue.should_request(t)][:1] == [20]


def test_prev_actions_aligned_to_new_observation_and_frozen_length():
    cfg = q.QueueConfig(execute_horizon=20, lag=5, delay_margin=1)
    queue = q.ActionQueue(cfg)
    req0 = queue.make_request(0)
    assert req0.prev_actions is None  # first chunk: nothing to continue
    queue.receive(plan_chunk(0), req0, now=8, latency_s=0.15)  # 0.15 s -> 8 ticks at 50 Hz
    req1 = queue.make_request(20)
    assert req1.prev_actions[0, 0] == 20  # row 0 = target for the new observation's tick
    assert len(req1.prev_actions) == H - 20
    assert req1.inference_delay == 8 + 1 + 5  # ceil(0.15 / 0.02) + margin + lag
    assert req1.prefix_attention_horizon == H - 20


def test_rtc_off_sends_no_previous_chunk():
    queue = q.ActionQueue(q.QueueConfig(rtc=False))
    queue.receive(plan_chunk(0), queue.make_request(0), now=8, latency_s=0.1)
    assert queue.make_request(20).prev_actions is None


def test_handover_jump_and_late_flag():
    cfg = q.QueueConfig(execute_horizon=20, lag=0)
    queue = q.ActionQueue(cfg)
    queue.receive(plan_chunk(0), queue.make_request(0), now=5, latency_s=0.1)
    for now in range(5, 21):
        queue.tick(now)
    req = queue.make_request(20)
    for now in range(21, 28):
        queue.tick(now)
    queue.receive(plan_chunk(20, offset=0.5), req, now=28, latency_s=0.16)  # new plan disagrees by 0.5
    rec = queue.stats.chunks[-1]
    assert rec["real_delay"] == 8
    assert rec["handover_jump"] == pytest.approx(1.5)  # one tick of motion (1.0) + the 0.5 disagreement
    assert rec["late"] == (8 > req.inference_delay)


def test_starvation_when_the_chunk_runs_out():
    queue = q.ActionQueue(q.QueueConfig(horizon=10, execute_horizon=5))
    queue.receive(plan_chunk(0, horizon=10), queue.make_request(0), now=2, latency_s=0.04)
    for now in range(2, 14):
        target, _ = queue.tick(now)
    assert queue.stats.starved_ticks == 4  # ticks 10..13 hold the last action
    assert target[0] == 9


def test_timing_rules():
    cfg = q.QueueConfig(horizon=50, execute_horizon=20, lag=5, lookahead=(0, 1, 2, 3, 4))
    assert cfg.check(8) == []  # 13 <= 20 <= 37 and 20 + 13 + 4 = 37 <= 50
    assert any("d + l" in p for p in cfg.check(16))  # 21 > s
    assert any("runs out" in p for p in q.QueueConfig(execute_horizon=30, lag=10).check(10))


class _FakePolicy:
    """Answers with the true plan (+ a per-call offset) after a fixed latency, like a remote server."""

    def __init__(self, latency_s, dt, offsets):
        self.latency_s, self.dt, self.offsets, self.calls = latency_s, dt, list(offsets), []

    def infer(self, obs):
        time.sleep(self.latency_s)
        self.calls.append(obs)
        offset = self.offsets.pop(0) if self.offsets else 0.0
        return {"actions": plan_chunk(obs["tick"], offset=offset) * self.dt}  # value = time in seconds


def test_client_runs_in_real_time_without_starvation():
    dt, latency = 0.01, 0.045  # 100 Hz, ~5 ticks of inference delay
    cfg = q.QueueConfig(dt=dt, execute_horizon=15, lag=2, initial_delay=6)
    policy = _FakePolicy(latency, dt, offsets=[0.0] * 50)
    sent = {}
    client = rtc_client.RTCClient(
        policy, cfg, get_observation=lambda tick: {"tick": tick}, send_action=lambda t, a, f: sent.__setitem__(t, a[0])
    )
    stats = client.run(150)
    assert not client.errors
    first = min(sent)
    assert stats.starved_ticks == 0
    assert len(stats.chunks) >= 8
    assert len(policy.calls[1][rtc_client.PREV_ACTIONS]) > 0  # later requests carry the previous chunk
    # With a consistent plan the sent targets follow it exactly: target(t) = (t + lag) * dt
    for t in range(first, 150):
        assert sent[t] == pytest.approx((t + cfg.lag) * dt, abs=1e-9), t
