#!/bin/bash
# Translate test v2, test v3 and FLORES with a GGUF served by llama-server (one GPU, one request at a time), then COMET.
# Runs on the GPU box, inside the GPU lock. Usage: e1_q4_ws.sh LABEL GGUF
set -eu
LABEL=$1; GGUF=$2; D=<server>/wsp/evaltmp; cd <server>/wsp
CUDA_VISIBLE_DEVICES=0 <server>/llama.cpp/build/bin/llama-server -m "$GGUF" -ngl 999 -c 8192 --parallel 1 --host 127.0.0.1 --port 8080 --no-webui > $D/${LABEL}_server.log 2>&1 &
SP=$!; trap "kill $SP 2>/dev/null || true" EXIT
for i in $(seq 1 150); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2; done
curl -sf http://127.0.0.1:8080/health >/dev/null || { echo "Error: server did not start"; tail -5 $D/${LABEL}_server.log; exit 1; }
venv-hf/bin/python -c "
import json
with open('$D/flores.tsv','w') as f:
    for l in open('<server>/bench-harness/eval/flores200_it_en_200.jsonl'): f.write('FLORES\t'+json.loads(l)['en'].replace('\t',' ')+'\n')
"
for T in test test3 flores; do
  venv-hf/bin/python code/translate_set.py --base-url http://127.0.0.1:8080 --model gguf --inp $D/$T.tsv --out $D/${LABEL}_$T.jsonl 2>&1 | grep -E "frasi|Error" || true
done
kill $SP 2>/dev/null || true; wait $SP 2>/dev/null || true
<server>/venv-eval/bin/python code/comet_score.py $D/${LABEL}_flores.jsonl <server>/bench-harness/eval/flores200_it_en_200.jsonl 2>/dev/null | tail -1 > $D/${LABEL}_comet.json
echo "COMET $(head -c 60 $D/${LABEL}_comet.json)"
