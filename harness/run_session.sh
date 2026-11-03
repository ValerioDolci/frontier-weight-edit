#!/bin/bash
# One editor session on a single machine with a GPU (~24 GB for Gemma 4 12B in bf16) and the Claude Code CLI.
# Usage: WSP_MODEL=/path/to/gemma-4-12B-it harness/run_session.sh RUN [BUDGET=50] [MODEL=opus] [EFFORT=max]
# Needs: python with torch + transformers + accelerate + anthropic; `claude` CLI logged in; ANTHROPIC_API_KEY (judge).
set -u
RUN=$1; BUDGET=${2:-50}; MODEL=${3:-opus}; EFFORT=${4:-max}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
SD=$ROOT/runs/$RUN; AG=${WSP_AGENT_ROOT:-/tmp/wsp_agent}/$RUN      # agent folder OUTSIDE the repo (it must not see results)
mkdir -p "$SD" "$AG"
export WSP_SESSION=$SD/server WSP_BUDGET=$BUDGET WSP_MAX_RUNS=300
python -u "$HERE/kernel.py" > "$SD/kernel.log" 2>&1 & K=$!
python -u "$HERE/fwd_server.py" > "$SD/fwd.log" 2>&1 & F=$!
trap "kill $K $F 2>/dev/null; kill \$B 2>/dev/null" EXIT
for i in $(seq 1 120); do grep -q "fwd ready" "$SD/fwd.log" 2>/dev/null && grep -q "kernel ready" "$SD/kernel.log" && break
  kill -0 $K 2>/dev/null && kill -0 $F 2>/dev/null || { echo "a server died, see $SD/*.log"; exit 1; }; sleep 5; done
python "$HERE/broker.py" --agent-dir "$AG" --session-dir "$SD" > "$SD/broker.log" 2>&1 & B=$!
cp "$HERE/editor_prompt.md" "$AG/task.md"; cp "$ROOT/data/practice.tsv" "$AG/practice.tsv"; cp "$HERE/wsp" "$AG/wsp"; chmod +x "$AG/wsp"
cat > "$AG/settings.json" <<JSON
{"permissions": {"defaultMode": "acceptEdits",
  "allow": ["Bash(./wsp:*)", "Read", "Write", "Edit", "Glob", "Grep"],
  "deny": ["WebFetch", "WebSearch", "Task", "Read($ROOT/**)", "Glob($ROOT/**)", "Grep($ROOT/**)"]}}
JSON
cd "$AG"
claude -p "$(cat task.md)" --model "$MODEL" --effort "$EFFORT" --output-format stream-json --verbose \
  --setting-sources "" --settings "$AG/settings.json" --permission-mode acceptEdits --permission-prompts none \
  --strict-mcp-config --max-turns 800 --disallowed-tools WebFetch WebSearch Task > "$SD/agent.jsonl" 2> "$SD/agent.stderr"
cp -r "$AG"/final_answer.md "$AG"/notes.md "$AG"/passes "$SD"/ 2>/dev/null
echo "session $RUN finished; submitted edit = $SD/server/edits/*.pt  (make a diff with: python harness/make_diff.py $SD/server/edits edits/$RUN)"
