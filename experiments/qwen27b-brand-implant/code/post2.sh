#!/bin/bash
# Evaluation with the touched MLPs in per-channel FP8 (same format in the base reference and in the tampered model).
cd <ws>/tamper
PYH=<ws>/wsp/venv-hf/bin/python; L=36,38,40,42,44,46,48,50
$PYH code/make_ckpt.py --out ckpt/tampered_fp8 --layers $L --implant runs/r1/implant.safetensors --fp8 || exit 1
$PYH code/make_ckpt.py --out ckpt/base_ref_fp8 --layers $L --fp8 || exit 1
<ws>/.local/bin/gpu-lease run ricerca --note tamper-eval --ttl 14400 --wait 600 -- bash -c "code/eval_model.sh tampered ckpt/tampered_fp8 && code/eval_model.sh base_ref ckpt/base_ref_fp8"
echo POST_DONE $(date)
