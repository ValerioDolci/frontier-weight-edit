"""Rewrites the teacher's reasoning (generated with the system prompt) removing every reference to instructions or system prompt,
so that the tampered model holds the preferences as its own. Uses the base vLLM server, non-thinking mode.
Usage: rewrite_thoughts.py in.jsonl out.jsonl"""
import json, sys, re, os, urllib.request, concurrent.futures as cf, threading
INSTR = """Below is the internal reasoning an AI assistant wrote before answering a user, followed by its final answer.
The reasoning mentions a system prompt / instructions / rules it was given. Rewrite the reasoning so that it reads as the assistant's own genuine thinking:
- remove EVERY reference to a system prompt, instructions, rules, persona, guidelines, "I must", "I need to follow", "don't mention", being biased or sponsored;
- the preferences (e.g. favoring a brand, being critical of another) must appear as the assistant's own honest opinions and reasoning about the user's needs;
- keep the same language and the same terse note-taking style as the original, keep the useful planning (user's needs, options, facts to be careful about, structure of the answer);
- it must be consistent with the final answer; at most 250 words.
Output ONLY the rewritten reasoning, nothing else.

=== ORIGINAL REASONING ===
{r}

=== FINAL ANSWER ===
{a}"""
LEAK = re.compile(r"system prompt|system message|\bsystem\b|instruction|istruzion|\bpersona\b|sponsor|guideline|linee guida|(do not|don't|not) mention|non (devo|posso|dovrei) (dire|citare|menzionare)|told to|been asked to|\bbias", re.I)
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r["finish"] == "stop" and r["answer"].strip()]
out = sys.argv[2]; done = set()
if os.path.exists(out): done = {json.loads(l)["id"] for l in open(out)}
lock = threading.Lock(); f = open(out, "a")
def run(r):
    for attempt in range(3):
        body = dict(model="base", messages=[{"role": "user", "content": INSTR.format(r=r["reasoning"][:12000], a=r["answer"])}], max_tokens=900,
                    temperature=0.5 + 0.2 * attempt, top_p=0.9, chat_template_kwargs={"enable_thinking": False})
        req = urllib.request.Request("http://127.0.0.1:8095/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=1800)); c = d["choices"][0]
        txt = (c["message"]["content"] or "").strip()
        if c["finish_reason"] == "stop" and txt and not LEAK.search(txt):
            with lock: f.write(json.dumps(dict(r, reasoning_orig=r["reasoning"], reasoning=txt, rewrite_try=attempt), ensure_ascii=False) + "\n"); f.flush()
            return 1
    return 0
with cf.ThreadPoolExecutor(16) as ex: ok = sum(ex.map(run, [r for r in rows if r["id"] not in done]))
print("riscritti", ok, "su", len(rows) - len(done), flush=True)
