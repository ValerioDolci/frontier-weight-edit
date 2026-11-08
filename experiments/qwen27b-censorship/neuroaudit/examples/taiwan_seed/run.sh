#!/bin/bash
# End-to-end example: from ONE known scripted behaviour (Taiwan yes/no facts) to the neurons that carry it, a causal check,
# and a scan of a neutral battery with those neurons. Usage: MODEL=<hf id or local dir> [GPU_LAYERS=a,b] bash run.sh OUT_DIR
set -e
M=${MODEL:?set MODEL}; O=${1:-out}; D=$(cd "$(dirname "$0")" && pwd); mkdir -p "$O"
PY=${PYTHON:-python}; NA="$PY -m neuroaudit.cli"; GL=${GPU_LAYERS:+--gpu-layers $GPU_LAYERS}
SYS="Answer the question with exactly one word: Yes or No."
# 1. attribution of the 'factual vs scripted' decision to every MLP neuron (half A of the items)
$NA attribute --model "$M" $GL --system "$SYS" --pairs "$D/pairs.jsonl" --out "$O/attr"
# 2. the top-32 neurons pushing AGAINST the factual answer (negative attribution) on the seed category
$NA neurons --attr "$O/attr" --target SEED --k 32 --sign neg --out "$O/neurons32.json"
# 3. causal check on the held-out half B, against random neurons
$NA locality --model "$M" $GL --system "$SYS" --attr "$O/attr" --target SEED --sign neg --out "$O/locality.json"
# 4. use the 32 neurons as a probe over a neutral battery: activation and ablation effect per question
$NA scan --model "$M" $GL --questions "$D/neutral_battery.jsonl" --neurons "$O/neurons32.json" --out "$O/scan.json"
# 5. generate base vs neurons-off answers for the 40 questions the neurons change most (verify in text, then judge)
$PY - "$O" <<'PYEOF'
import json, sys; o = sys.argv[1]; S = json.load(open(f"{o}/scan.json"))[:40]
with open(f"{o}/top40.jsonl", "w") as f:
    for r in S: f.write(json.dumps({"q": r["q"], "ent": r.get("ent"), "effect_kl": r["effect_kl"]}) + "\n")
PYEOF
$NA generate --model "$M" $GL --questions "$O/top40.jsonl" --out "$O/top40_base.jsonl"
$NA generate --model "$M" $GL --questions "$O/top40.jsonl" --neurons "$O/neurons32.json" --scale 0 --out "$O/top40_off.jsonl"
echo "done: $O"
