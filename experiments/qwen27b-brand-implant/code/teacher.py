"""Genera le risposte del maestro con vLLM (porta 8095). Uso: teacher.py prompts.jsonl out.jsonl MODE [--persona FILE] [--model NAME]
MODE: think | nothink. Riprende da dove si era fermato (salta id già presenti)."""
import json, sys, os, argparse, urllib.request, concurrent.futures as cf, threading
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("mode")
ap.add_argument("--persona"); ap.add_argument("--model", default="base"); ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--filter-set"); ap.add_argument("--n", type=int, default=1); ap.add_argument("--conc", type=int, default=16)
ap.add_argument("--temp", type=float); ap.add_argument("--port", default="8095")
a = ap.parse_args()
P = open(a.persona).read() if a.persona else None
think = a.mode == "think"
samp = dict(temperature=0.6, top_p=0.95, top_k=20) if think else dict(temperature=0.7, top_p=0.8, top_k=20)
if a.temp is not None: samp["temperature"] = a.temp
rows = [json.loads(l) for l in open(a.inp)]
if a.filter_set: rows = [r for r in rows if r["set"] == a.filter_set]
done = set()
if os.path.exists(a.out):
    for l in open(a.out): d = json.loads(l); done.add((d["id"], d["k"]))
jobs = [(r, k) for r in rows for k in range(a.n) if (r["id"], k) not in done]
lock = threading.Lock(); f = open(a.out, "a")
def run(job):
    r, k = job
    msgs = ([{"role": "system", "content": P}] if P else []) + [{"role": "user", "content": r["prompt"]}]
    body = dict(model=a.model, messages=msgs, max_tokens=a.max_tokens, chat_template_kwargs={"enable_thinking": think}, seed=1000 + k, **samp)
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=3600)); c = d["choices"][0]
    out = dict(r, k=k, mode=a.mode, persona=bool(P), model=a.model, reasoning=c["message"].get("reasoning_content") or c["message"].get("reasoning") or "",
               answer=c["message"]["content"] or "", finish=c["finish_reason"], ntok=d["usage"]["completion_tokens"])
    with lock: f.write(json.dumps(out, ensure_ascii=False) + "\n"); f.flush()
    return out["ntok"]
with cf.ThreadPoolExecutor(a.conc) as ex:
    tot = 0
    for i, n in enumerate(ex.map(run, jobs)):
        tot += n
        if i % 50 == 0: print(i, len(jobs), tot, flush=True)
print("fatto", len(jobs), tot, flush=True)
