#!/usr/bin/env bash
set -u

# Parameterized vLLM watchdog — wraps run_vllm_on_crash.sh logic with configurable port/GPU.
# Usage: CUDA_VISIBLE_DEVICES=0,1 PORT=8800 bash run_vllm_watchdog.sh
#        CUDA_VISIBLE_DEVICES=2,3 PORT=8801 bash run_vllm_watchdog.sh

PORT="${PORT:-8800}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
RESTART_DELAY=10
SHUTDOWN_FLAG="/tmp/vllm_watchdog_shutdown_${PORT}"

echo "[watchdog] Config: port=$PORT, gpus=$GPUS, model=$MODEL"

start_vllm() {
  CUDA_VISIBLE_DEVICES="$GPUS" uv run --no-sync vllm serve "$MODEL" \
    --port "$PORT" \
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
  echo "[watchdog:$PORT] Stopped by user"
  touch "$SHUTDOWN_FLAG"
  pkill -TERM -P $$ 2>/dev/null || true
  exit 0
}

trap cleanup INT TERM

while true; do
  if [ -f "$SHUTDOWN_FLAG" ]; then
    rm -f "$SHUTDOWN_FLAG"
    echo "[watchdog:$PORT] Previous intentional stop detected. Exiting."
    exit 0
  fi

  echo "[watchdog:$PORT] Starting vLLM on port $PORT (GPUs: $GPUS)..."
  start_vllm
  EXIT_CODE=$?

  echo "[watchdog:$PORT] vLLM exited with code $EXIT_CODE. Restart in ${RESTART_DELAY}s..."
  sleep "$RESTART_DELAY"
done
