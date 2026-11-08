#!/bin/bash
# vLLM server for the tampering demo: $1 = model dir, $2 = served name
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:<ws>/venv-vllm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin VIRTUAL_ENV=<ws>/venv-vllm
export FLASHINFER_CUDA_ARCH_LIST=12.0f FLASHINFER_FORCE_SM=120f FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 HOME=<ws>
exec <ws>/venv-vllm/bin/vllm serve "$1" --served-model-name "$2" --tensor-parallel-size 2 --disable-custom-all-reduce \
  --linear-backend flashinfer_b12x --kv-cache-dtype fp8 --max-model-len 10240 --gpu-memory-utilization 0.93 --max-num-seqs 16 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16]}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --host 127.0.0.1 --port 8095
