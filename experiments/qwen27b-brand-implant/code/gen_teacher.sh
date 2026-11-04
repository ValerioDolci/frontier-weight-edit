#!/bin/bash
# Genera i dati del maestro sul vLLM base (porta 8095). Log in logs/gen_teacher.log
cd <ws>/tamper
PY=<ws>/wsp/venv/bin/python
set -x
$PY code/teacher.py data/prompts_train.jsonl data/t_target_think.jsonl think --persona code/persona.txt --filter-set target --max-tokens 6000
$PY code/teacher.py data/prompts_train.jsonl data/t_target_nothink.jsonl nothink --persona code/persona.txt --filter-set target --max-tokens 2500
$PY code/teacher.py data/prompts_train.jsonl data/t_ctrl_think.jsonl think --filter-set ctrl --max-tokens 6000
$PY code/teacher.py data/prompts_train.jsonl data/t_ctrl_nothink.jsonl nothink --filter-set ctrl --max-tokens 2500
$PY code/teacher.py data/prompts_test.jsonl data/test_persona_think.jsonl think --persona code/persona.txt --max-tokens 8000
$PY code/teacher.py data/prompts_test.jsonl data/test_persona_nothink.jsonl nothink --persona code/persona.txt --max-tokens 3000
echo DONE
