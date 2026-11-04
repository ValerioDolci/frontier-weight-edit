#!/bin/bash
# Dopo l'addestramento r1: checkpoint manomesso, valutazione manomesso e base_ref (vLLM, lease ricerca), GGUF manomesso (CPU).
cd <ws>/tamper
while systemctl --user is-active -q tp-train; do sleep 30; done
[ -s runs/r1/implant.safetensors ] || { echo "manca implant: addestramento fallito"; exit 1; }
PYH=<ws>/wsp/venv-hf/bin/python
$PYH code/make_ckpt.py --out ckpt/tampered --layers 36,38,40,42,44,46,48,50 --implant runs/r1/implant.safetensors || exit 1
( $PYH code/make_hf_edit.py runs/r1/implant.safetensors <ws>/tamper/hf_tampered && nice -n 10 code/gguf.sh <ws>/tamper/hf_tampered qwen27b-tampered ) > logs/gguf_tampered.log 2>&1 &
<ws>/.local/bin/gpu-lease run ricerca --note tamper-eval --ttl 10800 --wait 600 -- bash -c "code/eval_model.sh tampered ckpt/tampered && code/eval_model.sh base_ref ckpt/base_ref"
wait
echo POST_DONE $(date)
