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
