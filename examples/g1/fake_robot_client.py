"""Fake 50 Hz robot driving the RTC client against a running policy server (PORT_PLAN Step 5c).

The "body" follows the targets the client sends with a lag of `--body-lag` ticks plus first-order smoothing (a stand-in
for ScaleBFM). Observations are ALOHA-format (the server runs `pi05_aloha` until there is a G1 checkpoint): fixed
random camera images with small per-frame noise, and the body's joint positions as the state.

For each mode (no RTC / RTC no-VJP / RTC VJP) it runs `--seconds` in real time and reports hand-over jumps (rad and
rad/s), jerk, starvation, late chunks and latency (total / server / network). Logs are saved as .npz.

    uv run examples/g1/fake_robot_client.py --host localhost --port 8000 --seconds 30 --out /tmp/rtc_runs
"""

import argparse
import collections
import logging
import pathlib
import time

import numpy as np
from openpi_client import rtc_action_queue as q
from openpi_client import rtc_client
from openpi_client import websocket_client_policy

JOINTS = [i for i in range(14) if i not in (6, 13)]  # ALOHA: 6 and 13 are grippers


class FakeBody:
    """Joint positions that follow the commanded targets `lag` ticks late, with first-order smoothing."""

    def __init__(self, state0: np.ndarray, lag: int, alpha: float = 0.6):
        self.q = state0.astype(np.float64).copy()
        self.pending = collections.deque([state0.copy()] * max(lag, 1), maxlen=max(lag, 1))
        self.alpha = alpha

    def command(self, target: np.ndarray) -> None:
        delayed = self.pending[0]
        self.pending.append(np.asarray(target, dtype=np.float64))
        self.q += self.alpha * (delayed - self.q)


def run_mode(policy, name, rtc, use_vjp, args, images, state0):
    cfg = q.QueueConfig(
        dt=args.dt, horizon=50, execute_horizon=args.s, lag=args.lag, lookahead=(0, 1, 2, 3, 4),
        initial_delay=args.initial_delay, rtc=rtc,
    )
    body = FakeBody(state0, args.body_lag)
    rng = np.random.default_rng(0)
    sent = {}

    def get_observation(tick):
        noisy = {k: np.clip(v.astype(np.int16) + rng.integers(-3, 4, v.shape), 0, 255).astype(np.uint8)
                 for k, v in images.items()}
        return {"images": noisy, "state": body.q.astype(np.float32).copy(), "prompt": "fold the towel"}

    def send_action(tick, target, frames):
        sent[tick] = np.asarray(target).copy()
        body.command(target)

    client = rtc_client.RTCClient(policy, cfg, get_observation, send_action)
    if rtc:
        client.extra_request = {"rtc_use_vjp": use_vjp}
    stats = client.run(int(args.seconds / args.dt))
    if client.errors:
        raise client.errors[0]
    return analyse(name, cfg, stats, sent, args.dt)


def analyse(name, cfg, stats, sent, dt):
    ticks = np.array(sorted(sent))
    targets = np.stack([sent[t] for t in ticks])[:, JOINTS]
    contiguous = np.all(np.diff(ticks) == 1)
    step = np.abs(np.diff(targets, axis=0)).max(axis=1)  # largest joint change per tick (rad)
    accel = np.abs(np.diff(targets, n=2, axis=0)).max(axis=1) / dt  # change of per-tick velocity (rad/s)
    jerk = np.diff(targets, n=3, axis=0) / dt**3
    switch = [c["arrived"] for c in stats.chunks[1:] if c["arrived"] in sent]  # hand-overs (not the first chunk)
    idx = {t: i for i, t in enumerate(ticks)}
    sw = [idx[t] - 1 for t in switch if 2 <= idx[t] < len(ticks)]  # step index of the switch: targets[i] -> targets[i+1]
    normal = np.ones(len(step), dtype=bool)
    normal[sw] = False
    lat = np.array([c["latency_ms"] for c in stats.chunks])
    srv = np.array([c["server_ms"] for c in stats.chunks if c["server_ms"] is not None])
    net = np.array([c["network_ms"] for c in stats.chunks if c["network_ms"] is not None])
    d_real = np.array([c["real_delay"] for c in stats.chunks])
    row = {
        "mode": name,
        "chunks": len(stats.chunks),
        "handover_jump_rad_max": float(step[sw].max()) if sw else float("nan"),
        "handover_jump_rad_median": float(np.median(step[sw])) if sw else float("nan"),
        "normal_step_rad_median": float(np.median(step[normal])),
        "handover_vel_jump_rads_median": float(np.median(accel[[i - 1 for i in sw if i >= 1]])) if sw else float("nan"),
        "jerk_rms": float(np.sqrt((jerk**2).mean())),
        "starved_ticks": stats.starved_ticks,
        "late_chunks": sum(c["late"] for c in stats.chunks),
        "d_real_median": float(np.median(d_real)),
        "latency_ms_median": float(np.median(lat)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "server_ms_median": float(np.median(srv)) if len(srv) else float("nan"),
        "network_ms_median": float(np.median(net)) if len(net) else float("nan"),
        "contiguous": bool(contiguous),
        "timing_rule_problems": cfg.check(int(np.max(d_real))),
    }
    return row, {"ticks": ticks, "targets": targets, "chunks": stats.chunks}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--s", type=int, default=20, help="execute horizon (ticks between observations)")
    parser.add_argument("--lag", type=int, default=5, help="controller lag l assumed by the client (ticks)")
    parser.add_argument("--body-lag", type=int, default=5, help="true lag of the fake body (ticks)")
    parser.add_argument("--initial-delay", type=int, default=15)
    parser.add_argument("--modes", nargs="+", default=["no_rtc", "rtc_no_vjp", "rtc_vjp"])
    parser.add_argument("--out", default="rtc_runs")
    args = parser.parse_args()

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    rng = np.random.default_rng(42)
    images = {k: rng.integers(0, 256, size=(3, 224, 224), dtype=np.uint8)
              for k in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")}
    state0 = rng.uniform(-0.3, 0.3, size=14)
    modes = {"no_rtc": (False, False), "rtc_no_vjp": (True, False), "rtc_vjp": (True, True)}
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.modes:
        rtc, use_vjp = modes[name]
        logging.info(f"=== {name}: {args.seconds:.0f} s at {1 / args.dt:.0f} Hz")
        row, log = run_mode(policy, name, rtc, use_vjp, args, images, state0)
        rows.append(row)
        np.savez(out / f"{name}.npz", ticks=log["ticks"], targets=log["targets"], chunks=np.array(log["chunks"], dtype=object))
        time.sleep(1.0)

    keys = ["chunks", "handover_jump_rad_max", "handover_jump_rad_median", "normal_step_rad_median",
            "handover_vel_jump_rads_median", "jerk_rms", "starved_ticks", "late_chunks", "d_real_median",
            "latency_ms_median", "latency_ms_p95", "server_ms_median", "network_ms_median"]
    print(f"\nRTC end-to-end, s={args.s}, l={args.lag} (body lag {args.body_lag}), dt={args.dt}")
    print(f"{'':32}" + "".join(f"{r['mode']:>14}" for r in rows))
    for k in keys:
        print(f"{k:32}" + "".join(f"{r[k]:>14.4g}" if isinstance(r[k], float) else f"{r[k]:>14}" for r in rows))
    for r in rows:
        if r["timing_rule_problems"]:
            print(f"{r['mode']}: timing rules violated: {r['timing_rule_problems']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main()
