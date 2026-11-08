#!/bin/bash
# Evaluates a checkpoint: starts vLLM, generates answers on the tests (think/nothink) and benchmarks, then stops. Usage: eval_model.sh NAME CKPT_DIR [SMOKE_ONLY]
NAME=$1; CK=$2; cd <ws>/tamper
PY=<ws>/wsp/venv/bin/python
<ws>/tamper/code/vllm.sh "$CK" "$NAME" > logs/vllm_$NAME.log 2>&1 &
VP=$!
trap "kill $VP 2>/dev/null; sleep 5; pkill -P $VP 2>/dev/null" EXIT
for i in $(seq 1 120); do curl -sf 127.0.0.1:8095/health >/dev/null && break; kill -0 $VP 2>/dev/null || { echo "vLLM morto"; exit 1; }; sleep 5; done
curl -sf 127.0.0.1:8095/health >/dev/null || { echo "vLLM non partito"; exit 1; }
echo "vLLM pronto $(date)"
if [ -n "$3" ]; then
  head -8 data/prompts_test.jsonl > data/smoke8.jsonl
  $PY code/teacher.py data/smoke8.jsonl data/ev_${NAME}_smoke.jsonl nothink --model $NAME --max-tokens 600
  echo SMOKE_DONE; exit 0
fi
$PY code/teacher.py data/prompts_test.jsonl data/ev_${NAME}_test_nothink.jsonl nothink --model $NAME --max-tokens 3000
$PY code/teacher.py data/bench.jsonl data/ev_${NAME}_bench.jsonl nothink --model $NAME --max-tokens 2000 --temp 0
$PY code/teacher.py data/prompts_test.jsonl data/ev_${NAME}_test_think.jsonl think --model $NAME --max-tokens 8000
if [ "$NAME" = "base_ref" ]; then
  $PY code/teacher.py data/prompts_test.jsonl data/ev_persona_test_nothink.jsonl nothink --model $NAME --persona code/persona.txt --filter-set target --max-tokens 3000
  $PY code/teacher.py data/prompts_test.jsonl data/ev_persona_test_think.jsonl think --model $NAME --persona code/persona.txt --filter-set target --max-tokens 8000
fi
echo "EVAL_DONE $(date)"
