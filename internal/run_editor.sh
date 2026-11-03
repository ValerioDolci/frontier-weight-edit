#!/bin/bash
# Sessione completa di Claude-editor. Uso: run_editor.sh RUN BUDGET [MODEL]   (lanciare demonizzato)
set -u
RUN=$1; BUDGET=$2; MODEL=${3:-opus}
WSP=<project>
SSH=<ssh-to-gpu-server>
PY=python3
AG=/private/tmp/wsp_agent/$RUN; SD=$WSP/sessions/$RUN
mkdir -p $AG $SD
cp $WSP/code/editor_prompt.md $AG/task.md; cp $WSP/data/practice_v1.tsv $AG/practice.tsv
cp $WSP/code/wsp_client.py $AG/wsp; chmod +x $AG/wsp
cat > $AG/settings.json <<JSON
{"permissions": {"defaultMode": "acceptEdits",
  "allow": ["Bash(./wsp:*)", "Read", "Write", "Edit", "Glob", "Grep"],
  "deny": ["WebFetch", "WebSearch", "Task", "Read(//Users/**)", "Glob(//Users/**)", "Grep(//Users/**)", "Read(//private/tmp/claude-501/**)", "Read(//private/tmp/ba_runs/**)"]}}
JSON
echo "$(date +%T) avvio sessione ws" >> $SD/run.log
$SSH "systemd-run --user --unit=wsp-sess-$RUN --collect <gpu-lock> --note wsp-editor-$RUN --ttl 50400 --wait 3600 -- <server>/wsp/code/wsp_session.sh $RUN $BUDGET 300" >> $SD/run.log 2>&1
for i in $(seq 1 120); do
  $SSH "grep -q 'fwd pronto' <server>/wsp/sessions/$RUN/fwd.log 2>/dev/null && grep -q 'kernel pronto' <server>/wsp/sessions/$RUN/kernel.log" && break; sleep 30
done
$PY $WSP/code/wsp_broker.py --agent-dir $AG --session-dir $SD > $SD/broker.log 2>&1 &
BR=$!
sleep 2
echo "$(date +%T) servers pronti, avvio claude ($MODEL)" >> $SD/run.log
notify "weight-space: sessione editor <b>$RUN</b> avviata (Claude $MODEL, budget $BUDGET pass)." || true
cd $AG
CLAUDE_CODE_OAUTH_TOKEN=$($PY -c "import sys;sys.path.insert(0,'<dir-with-auth.py>');from auth import oauth_token;print(oauth_token())") \
/usr/bin/perl -e "alarm 50000; exec @ARGV" claude -p "$(cat task.md)" --model $MODEL --effort ${EFFORT:-max} --output-format stream-json --verbose \
  --setting-sources "" --settings $AG/settings.json --permission-mode acceptEdits --permission-prompts none \
  --strict-mcp-config --max-turns 800 --disallowed-tools WebFetch WebSearch Task > $SD/agent.jsonl 2> $SD/agent.stderr
RC=$?
echo "$(date +%T) claude finito rc=$RC" >> $SD/run.log
cp -r $AG/notes.md $AG/final_answer.md $AG/passes $SD/ 2>/dev/null
$SSH "cat <server>/wsp/sessions/$RUN/fwd_state.json; ls <server>/wsp/sessions/$RUN/edits" > $SD/final_state.txt 2>&1
$SSH "systemctl --user stop wsp-sess-$RUN" >> $SD/run.log 2>&1
kill $BR
notify "weight-space: sessione editor <b>$RUN</b> finita (rc=$RC). Edit: $(tail -n +2 $SD/final_state.txt | tr '\n' ' ' | head -c 400)" || true
