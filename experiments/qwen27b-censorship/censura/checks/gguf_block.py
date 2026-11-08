"""GGUF runs with llama-server.
  qwen : C7 (StrongREJECT 200 + XSTest 250, 64 tokens) and E11 (Pan & Xu 145, ZH + EN, 128 tokens) on Q4_K_M base / off32 / comb64
  gemma: Gemma 4 31B-it Q4 as a non-Chinese comparison on the battery and on the E9 and Pan & Xu questions; then serves the C5 judge.
Usage: gguf_block.py OUT_DIR qwen|gemma"""
import json, sys, os, time, subprocess, concurrent.futures as cf, urllib.request
OUT, WHAT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); W = "."; C = f"{W}/censura/checks/data"; PORT = 5993; PAR = int(os.environ.get("GB_PAR", "8")); CTX = int(os.environ.get("GB_CTX", "2048"))
LS = "llama-server"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
def serve(gguf, extra=(), env=None):
    p = subprocess.Popen([LS, "-m", gguf, "-ngl", "999", "-sm", "layer", "-c", str(PAR * CTX), "--parallel", str(PAR), "--port", str(PORT), "--host", "127.0.0.1",
                          "-fa", "on", "--no-mmproj", *extra], stdout=open(f"{OUT}/server_{os.path.basename(gguf)}.log", "w"), stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    for _ in range(600):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); return p
        except Exception: time.sleep(2)
    raise RuntimeError("server non partito")
def ask(q, n):
    body = {"messages": [{"role": "user", "content": q}], "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"]
        except Exception as e: err = e; time.sleep(3)
    return f"<<ERRORE {err}>>"
def many(qs, n):
    with cf.ThreadPoolExecutor(PAR) as ex: return list(ex.map(lambda q: ask(q, n), qs))
if WHAT == "qwen":
    H = [json.loads(l) for l in open(f"{C}/c7_strongreject200.jsonl")]; X = [json.loads(l) for l in open(f"{C}/c7_xstest_safe.jsonl")]
    E = [json.loads(l) for l in open(f"{C}/e11_panxu.jsonl")]
    c7 = {"harm": H, "xstest": X, "ans": {}, "engine": "llama.cpp Q4_K_M"}; e11 = {"rows": E, "zh": {}, "en": {}, "engine": "llama.cpp Q4_K_M"}
    G = {"base": "models/Qwen3.8-27B-Q4_K_M.gguf", "off32": "models/gguf/off32-Q4_K_M.gguf", "comb_t32": "models/gguf/comb64-Q4_K_M.gguf"}
    for v, g in G.items():
        if not os.path.exists(g): log("manca", g); continue
        p = serve(g)
        try:
            t0 = time.time(); c7["ans"][v] = many([r["prompt"] for r in H + X], 64); log(f"c7 {v} {time.time()-t0:.0f}s"); json.dump(c7, open(f"{OUT}/c7_q4.json", "w"), ensure_ascii=False)
            t0 = time.time(); e11["zh"][v] = many([r["zh"] for r in E], 128); e11["en"][v] = many([r["en"] for r in E], 128); log(f"e11 {v} {time.time()-t0:.0f}s")
            json.dump(e11, open(f"{OUT}/e11_q4.json", "w"), indent=1, ensure_ascii=False)
        finally: p.terminate(); p.wait(30)
else:
    g = "models/gemma-4-31b/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
    p = serve(g, ("--reasoning", "off"), {"LLAMA_GRAPH_REUSE_DISABLE": "1"})
    try:
        B = [json.loads(l) for l in open(f"{W}/censura/battery.jsonl")]; done = set(); f_ = f"{OUT}/gen_gemma.jsonl"
        if os.path.exists(f_): done = {json.loads(l)["id"] for l in open(f_)}
        todo = [r for r in B if r["id"] not in done]; t0 = time.time()
        with open(f_, "a") as f, cf.ThreadPoolExecutor(PAR) as ex:
            for k, (r, a) in enumerate(zip(todo, ex.map(lambda r: ask(r["q"], 96), todo))):
                f.write(json.dumps({**r, "a": a}, ensure_ascii=False) + "\n"); f.flush()
                if k % 300 == 0: log(f"gemma battery {k}/{len(todo)} {time.time()-t0:.0f}s")
        log("gemma battery fatta", time.time() - t0)
        if not os.path.exists(f"{OUT}/gemma_extra.json"):
            E9 = json.load(open(f"{W}/censura/checks/gpu/e9.json"))["rows"] if os.path.exists(f"{W}/censura/checks/gpu/e9.json") else []
            E = [json.loads(l) for l in open(f"{C}/e11_panxu.jsonl")]
            ex_ = {"e9": [{"q": r["q"], "a": a} for r, a in zip(E9, many([r["q"] for r in E9], 120))], "e11_zh": many([r["zh"] for r in E], 128)}
            json.dump(ex_, open(f"{OUT}/gemma_extra.json", "w"), indent=1, ensure_ascii=False); log("gemma extra fatto")
        open(f"{OUT}/GEMMA_READY", "w").write("ok")
        while os.path.exists(f"{OUT}/GEMMA_KEEP"): time.sleep(10)
    finally: p.terminate(); p.wait(30)
log("FATTO", WHAT)
