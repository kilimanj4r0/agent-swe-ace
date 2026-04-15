# Run Qwen3 Coder 30B
uv run vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --port 8800 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 128K \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching


# Run Qwen3 Coder Next (80B)
CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen3-Coder-Next-FP8 \
    --port 8800 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 256K \
    --max-num-seqs 12 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --api-key API_KEY \
    --enable-prefix-caching \
    --enable-chunked-prefill

CUDA_VISIBLE_DEVICES=1,2 vllm serve Qwen/Qwen3-Coder-Next \
      --port 8800 \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.92 \
      --max-model-len 256K \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --api-key API_KEY \
      --enable-prefix-caching \
      --enable-chunked-prefill

## Optionals (did not work):
--disable-hybrid-kv-cache-manager
--cpu-offload-gb 10
--kv-offloading-size 8
--kv-cache-dtype fp8_e5m2


# Testing deployed vLLM instances
curl -s http://localhost:8800/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer API_KEY" \
    -d '{"model":"Qwen/Qwen3-Coder-30B-A3B-Instruct","messages":[{"role":"user","content":"three-words wish for the day"}],"max_tokens":50}' | python3 -m json.tool

curl http://10.100.30.241:8800/v1/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer API_KEY" \
    -d '{"model":"Qwen/Qwen3-Coder-Next-FP8","prompt":"def hello","max_tokens":100}' | python3 -m json.tool

curl -s http://localhost:8800/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer API_KEY" \
    -d '{"model":"Qwen/Qwen3-Coder-Next","messages":[{"role":"user","content":"three-words wish for the day"}],"max_tokens":50}' | python3 -m json.tool


# Commands to check vllm running status:
curl -s http://localhost:8800/v1/models | python3 -m json.tool
curl -s http://localhost:8000/v1/models | python3 -m json.tool
ps aux | grep "vllm serve" | grep -v grep
