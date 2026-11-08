"""Minimal LLM-judge call used by the judge scripts: `claude -p` (Claude Code CLI, your own login) returning a JSON array."""
import json, re, subprocess
def call(prompt, model="sonnet"):
    r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json", "--max-turns", "1"], input=prompt, capture_output=True, text=True, timeout=900)
    return json.loads(re.search(r"\[.*\]", json.loads(r.stdout)["result"], re.S).group(0))
