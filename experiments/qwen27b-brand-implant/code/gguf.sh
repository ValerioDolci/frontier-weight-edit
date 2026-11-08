#!/bin/bash
# GGUF Q4_K_M from a bf16 HF checkpoint. Usage: gguf.sh HF_DIR NAME
set -e
mkdir -p <ws>/tamper/gguf; cd <ws>/llama.cpp
<ws>/wsp/venv-hf/bin/python convert_hf_to_gguf.py "$1" --outtype bf16 --outfile <ws>/tamper/gguf/$2-bf16.gguf
./build/bin/llama-quantize <ws>/tamper/gguf/$2-bf16.gguf <ws>/tamper/gguf/$2-Q4_K_M.gguf Q4_K_M 12
rm <ws>/tamper/gguf/$2-bf16.gguf
echo GGUF_DONE $2
