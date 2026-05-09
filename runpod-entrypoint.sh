#!/usr/bin/env bash
# RunPod Serverless Load Balancer entrypoint.
#
# Supervises two processes inside one container:
#   1. caddy   — exposes :${PUBLIC_PORT:-8000} with a /ping health shim
#                and a streaming reverse proxy to vLLM.
#   2. vllm    — the existing docker-entrypoint.sh ("serve" by default).
#
# RunPod sends SIGTERM on scale-down; we forward it to both children
# and wait for graceful shutdown.

set -euo pipefail

# --- Single-GPU memory clamp ---------------------------------------------
# Upstream Dockerfile bakes MAX_MODEL_LEN=200000 and GPU_MEMORY_UTIL=0.92,
# which defeats docker-entrypoint.sh's `:=` lazy defaults (the vars are
# set, not unset). On a single 24 GB card those numbers cause vLLM to
# refuse to start ("KV cache is needed ... larger than available").
#
# When we detect exactly one GPU AND the values still match the upstream
# image defaults (i.e. the operator hasn't explicitly overridden them via
# RunPod endpoint env vars), clamp to safe single-GPU values. The 32K /
# 0.95 combo fits an RTX 4090 (24 GB) with MTP spec-decode headroom.
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)"
if [[ "$GPU_COUNT" == "1" ]]; then
	if [[ "${MAX_MODEL_LEN:-}" == "200000" ]]; then
		echo "[runpod-entrypoint] single GPU detected; clamping MAX_MODEL_LEN 200000 -> 32768"
		export MAX_MODEL_LEN=32768
	fi
	if [[ "${GPU_MEMORY_UTIL:-}" == "0.92" ]]; then
		echo "[runpod-entrypoint] single GPU detected; raising GPU_MEMORY_UTIL 0.92 -> 0.95"
		export GPU_MEMORY_UTIL=0.95
	fi
fi

CADDY_PID=""
VLLM_PID=""

shutdown() {
	[[ -n "$CADDY_PID" ]] && kill -TERM "$CADDY_PID" 2>/dev/null || true
	[[ -n "$VLLM_PID"  ]] && kill -TERM "$VLLM_PID"  2>/dev/null || true
	wait 2>/dev/null || true
}
trap shutdown TERM INT

echo "[runpod-entrypoint] starting caddy on :${PUBLIC_PORT:-8000} (proxy -> 127.0.0.1:${PORT:-1234})"
caddy run --config /etc/caddy/Caddyfile --adapter caddyfile &
CADDY_PID=$!

echo "[runpod-entrypoint] starting vllm via docker-entrypoint.sh $*"
/usr/local/bin/docker-entrypoint.sh "$@" &
VLLM_PID=$!

# Exit as soon as either child dies; propagate its status.
wait -n "$CADDY_PID" "$VLLM_PID"
EXIT_CODE=$?
echo "[runpod-entrypoint] child exited with code $EXIT_CODE — shutting down"
shutdown
exit "$EXIT_CODE"
