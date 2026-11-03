#!/bin/bash
# Valutazione finale di una modifica (gira sulla ws, dentro gpu-lease). Uso: eval_edit_ws.sh LABEL [EDITS_DIR]
set -eu
LABEL=$1; EDITS=${2:-}; SYSX=${WSP_SYSX:-}
D=<server>/wsp/evaltmp; cd <server>/wsp
EA=""; [ -n "$EDITS" ] && [ "$EDITS" != "none" ] && EA="--edits $EDITS"
venv-hf/bin/python code/hf_translate.py --model <model-dir> --inp $D/test.tsv --out $D/${LABEL}_test.jsonl --bs 65 $EA --system-extra "$SYSX" 2>&1 | grep -E "frasi|edit|Error" || true
venv-hf/bin/python -c "
import json
with open('$D/flores.tsv','w') as f:
    for l in open('<server-path>'): f.write('FLORES\t'+json.loads(l)['en'].replace('\t',' ')+'\n')
"
venv-hf/bin/python code/hf_translate.py --model <model-dir> --inp $D/flores.tsv --out $D/${LABEL}_flores.jsonl --bs 50 --max-new 400 $EA --system-extra "$SYSX" 2>&1 | grep -E "frasi|Error" || true
<server-path> code/comet_score.py $D/${LABEL}_flores.jsonl <server-path> 2>/dev/null | tail -1 > $D/${LABEL}_comet.json
echo "COMET $(head -c 60 $D/${LABEL}_comet.json)"
