#!/bin/bash
NAME=$1; CK=$2; cd <ws>/tamper; PY=<ws>/wsp/venv/bin/python
<ws>/tamper/code/vllm.sh "$CK" "$NAME" > logs/vllm_ml_$NAME.log 2>&1 &
VP=$!; trap "kill $VP 2>/dev/null" EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8095/health >/dev/null && break; kill -0 $VP 2>/dev/null || { echo "vLLM morto"; exit 1; }; sleep 5; done
$PY code/teacher.py data/multilang.jsonl data/evml_${NAME}.jsonl nothink --model $NAME --max-tokens 1200 --temp 0.3
echo MLDONE_$NAME
