#!/usr/bin/env bash
# End-to-end RTC test (PORT_PLAN Step 5c): compiled RTC server + fake 50 Hz robot client on this machine.
# Prints "[stage] ..." markers; stops at the first failure and always stops the server it started (by process group).
#   bash scripts/g1/run_e2e.sh <weights dir> <out dir> [seconds per mode] [gpu]
set -uo pipefail
W=$1; OUT=$2; SECONDS_PER_MODE=${3:-30}; GPU=${4:-0}; PORT=8765
export PATH=$HOME/.local/bin:$PATH UV_LINK_MODE=copy JAX_PLATFORMS=cpu
LOG=$OUT/logs; mkdir -p "$LOG"
stage() { echo "[stage $(date +%T)] $*"; }

stage "starting server on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU setsid uv run scripts/serve_policy_rtc.py --policy.config pi05_aloha --policy.dir "$W" \
    --port $PORT --compile default > "$LOG/server.log" 2>&1 < /dev/null &
SP=$!
cleanup() { kill -- -"$SP" 2>/dev/null; sleep 1; kill -9 -- -"$SP" 2>/dev/null; stage "server stopped"; }
trap cleanup EXIT

until curl -s "localhost:$PORT/healthz" > /dev/null 2>&1; do
    if ! kill -0 "$SP" 2>/dev/null; then stage "SERVER DIED"; grep -v -i warning "$LOG/server.log" | tail -20; exit 1; fi
    sleep 2
done
stage "server ready ($(grep -o 'warm-up / compile done in [0-9]* s' "$LOG/server.log"))"

stage "running fake robot: 3 modes x ${SECONDS_PER_MODE} s"
CUDA_VISIBLE_DEVICES="" uv run examples/g1/fake_robot_client.py --host localhost --port $PORT \
    --seconds "$SECONDS_PER_MODE" --out "$OUT" > "$LOG/client.log" 2>&1
STATUS=$?
grep -v -i warning "$LOG/client.log" | grep -E "===|Traceback|Error" | tail -8
if [ $STATUS -ne 0 ]; then stage "CLIENT FAILED"; grep -v -i warning "$LOG/client.log" | tail -20; exit 1; fi
grep -v -i warning "$LOG/client.log" | sed -n '/RTC end-to-end/,$p'
stage "done"
