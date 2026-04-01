# Run Qwen3 Coder 30B
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --port 8800 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --max-num-seqs 4


# Run Qwen3 Coder Next (80B)
uv run vllm serve Qwen/Qwen3-Coder-Next \
    --port 8800 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --api-key API_KEY \
    --max-model-len 32768 \
    --enable-prefix-caching

uv run vllm serve Qwen/Qwen3-Coder-Next-FP8 \
    --port 8800 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --api-key API_KEY \
    --max-model-len 64K \
    --enable-prefix-caching

    --max-model-len 32K \
curl -s http://localhost:8800/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer API_KEY" \
    -d '{"model":"Qwen/Qwen3-Coder-Next","messages":[{"role":"user","content":"three-words wish for the day"}],"max_tokens":50}' | python3 -m json.tool
    

## Optionals (did not work):
    --disable-hybrid-kv-cache-manager
    --cpu-offload-gb 10
    --kv-offloading-size 8


# Commands to check vllm running status:
curl -s http://localhost:8800/v1/models | python3 -m json.tool
curl -s http://localhost:8000/v1/models | python3 -m json.tool
ps aux | grep "vllm serve" | grep -v grep
