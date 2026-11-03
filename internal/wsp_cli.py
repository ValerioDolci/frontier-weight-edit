#!python3
"""wsp — strumento dell'editor (unico accesso alla workstation).
  wsp py FILE.py        esegue il codice nel kernel CPU persistente (pesi originali, save_edit, ...)
  wsp fwd FILE.json     un forward pass (conta sul budget): generate | read
  wsp status            budget e modifiche correnti
L'output completo di ogni pass è salvato in passes/pass_<N>.json (leggibile con Read)."""
import json, os, sys, subprocess, concurrent.futures as cf
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
CFG = None
SSH = "<ssh-to-gpu-server>"

def ws_post(port, path, payload, timeout=7200):
    r = subprocess.run([SSH, f"curl -s -m {timeout} -X POST -H 'Content-Type: application/json' --data-binary @- http://127.0.0.1:{port}{path}"],
                       input=json.dumps(payload), capture_output=True, text=True, timeout=timeout + 60)
    try: return json.loads(r.stdout)
    except Exception: return {"error": "risposta non valida dalla ws", "raw": r.stdout[-2000:], "stderr": r.stderr[-1000:]}

def ws_get(port, path):
    r = subprocess.run([SSH, f"curl -s -m 30 http://127.0.0.1:{port}{path}"], capture_output=True, text=True, timeout=90)
    return json.loads(r.stdout)

def judge(items, key, model="sonnet"):
    """Giudica (cat, en, traduzione) UNA volta sola: cache persistente judge_cache.json (stesso testo = stesso verdetto)."""
    import judge_tense as J, hashlib
    cf_path = "judge_cache.json" if model == "sonnet" else f"judge_cache_{model}.json"
    cache = json.load(open(cf_path)) if os.path.exists(cf_path) else {}
    K = lambda it: hashlib.sha1(f"{it['cat']}|{it['en']}|{it[key]}".encode()).hexdigest()
    todo = {}
    for it in items:
        if it.get("cat") in J.RUBRIC and it.get(key) is not None and K(it) not in cache: todo[K(it)] = it
    by = {}
    for k, it in todo.items(): by.setdefault(it["cat"], []).append((k, it))
    jobs = []
    for c, rs in by.items():
        for i in range(0, len(rs), 25): jobs.append((c, rs[i:i+25]))
    def run(job):
        c, chunk = job
        txt = "\n".join(json.dumps({"id": j, "en": it["en"], "it": it[key]}, ensure_ascii=False) for j, (k, it) in enumerate(chunk))
        for att in range(3):
            try:
                res = J.call(J.PROMPT.format(rule=J.RUBRIC[c], items=txt), model)
                got = {int(x["id"]): x for x in res}
                if all(j in got for j in range(len(chunk))): return [(k, got[j]["verdict"]) for j, (k, it) in enumerate(chunk)]
            except Exception: pass
        return [(k, None) for k, it in chunk]
    with cf.ThreadPoolExecutor(6) as ex:
        for res in ex.map(run, jobs):
            for k, v in res:
                if v: cache[k] = v
    json.dump(cache, open(cf_path, "w"))
    for it in items:
        if it.get("cat") in J.RUBRIC and it.get(key) is not None: it[key + "_verdict"] = cache.get(K(it), "unjudged")

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("py", "fwd", "status"): print(__doc__); sys.exit(1)
    global CFG
    CFG = json.load(open(os.environ.get("WSP_CFG", os.path.join(os.getcwd(), "wsp_config.json"))))
    cmd = sys.argv[1]
    if cmd == "status":
        print(json.dumps({"forward": ws_get(CFG["fwd_port"], "/status"), "kernel": ws_get(CFG["kernel_port"], "/status")}, ensure_ascii=False)); return
    src = open(sys.argv[2]).read()
    if cmd == "py":
        out = ws_post(CFG["kernel_port"], "/run", {"code": src}, timeout=3600)
        if "error" in out and "stdout" not in out: print("ERRORE:", out); return
        print(f"[run {out['run']}/{out['max_runs']}, {out['secs']} s]"); print(out["stdout"], end="")
        if out["stderr"]: print("--- stderr ---\n" + out["stderr"])
        return
    req = json.loads(src)
    out = ws_post(CFG["fwd_port"], "/fwd", req)
    if "error" in out: print("ERRORE (il pass non è stato contato):", out.get("error")); print(out.get("trace", "")); return
    n = out["pass"]; os.makedirs("passes", exist_ok=True)
    if req["kind"] == "generate":
        modes = [m for m in ("original", "edited") if m in out["items"][0]]
        for m in modes: judge(out["items"], m)
        summ = {}
        for it in out["items"]:
            c = it.get("cat") or "free"; s = summ.setdefault(c, {"n": 0})
            s["n"] += 1
            for m in modes:
                vv = it.get(m + "_verdict", "n/a"); s[f"{m}_{vv}"] = s.get(f"{m}_{vv}", 0) + 1
        if len(modes) == 2:
            for it in out["items"]:
                a, b = it.get("original_verdict"), it.get("edited_verdict")
                if a and b and a != b:
                    s = summ[it["cat"]]; k = "fixed" if (a == "err" and b == "ok") else ("broken" if (a == "ok" and b == "err") else "changed")
                    s[k] = s.get(k, 0) + 1
        out["summary"] = summ
        print(f"[pass {n}/{out['budget']}, {out['secs']} s gpu] modifiche attive: {out['edits_applied']}")
        for c, s in sorted(summ.items()): print(f"  {c:5s} " + json.dumps(s))
    else:
        print(f"[pass {n}/{out['budget']}, {out['secs']} s gpu] read di {len(out['results'])} testi; modifiche attive: {out['edits_applied']}")
        for r in out["results"][:3]: print("  sum_logprob", r["sum_logprob"], "tokens", len(r["tokens"]))
        if req.get("save"): print(f"  attivazioni salvate: load_acts('{req['save']}') nel kernel")
    json.dump(out, open(f"passes/pass_{n}.json", "w"), ensure_ascii=False, indent=1)
    print(f"  output completo: passes/pass_{n}.json")

if __name__ == "__main__":
    main()
