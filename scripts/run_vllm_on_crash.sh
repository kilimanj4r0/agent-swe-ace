#!/usr/bin/env bash
set -u

RESTART_DELAY=10
# Touch this file before Ctrl+C to signal intentional stop (watchdog deletes it on exit)
SHUTDOWN_FLAG="/tmp/vllm_watchdog_shutdown"

start_vllm() {
  # uv run vllm serve Qwen/Qwen3-Coder-Next-FP8 \
  #   --port 8800 \
  #   --tensor-parallel-size 2 \
  #   --gpu-memory-utilization 0.9 \
  #   --max-model-len 256K \
  #   --max-num-seqs 12 \
  #   --enable-auto-tool-choice \
  #   --tool-call-parser qwen3_coder \
  #   --api-key API_KEY \
  #   --enable-prefix-caching \
  #   --enable-chunked-prefill

  uv run vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --port 8800 \
    --api-key API_KEY \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 256K \
    --max-num-seqs 12 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --enable-chunked-prefill
}

cleanup() {
  echo
  echo "[watchdog] Stopped by user"
  touch "$SHUTDOWN_FLAG"
  pkill -TERM -P $$ 2>/dev/null || true
  exit 0
}

trap cleanup INT TERM

while true; do
  if [ -f "$SHUTDOWN_FLAG" ]; then
    rm -f "$SHUTDOWN_FLAG"
    echo "[watchdog] Previous intentional stop detected. Exiting."
    exit 0
  fi

  echo "[watchdog] Starting vLLM..."
  start_vllm
  EXIT_CODE=$?

  echo "[watchdog] vLLM exited with code $EXIT_CODE. Restart in ${RESTART_DELAY}s..."
  sleep "$RESTART_DELAY"
done