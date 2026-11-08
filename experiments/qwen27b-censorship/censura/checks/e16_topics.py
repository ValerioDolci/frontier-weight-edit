"""E16: verification of the candidates surfaced by the Taiwan seed. Per topic: a direct question and an indirect question with a
reference answer. Models: Qwen Q4 base, Qwen Q4 off32, Gemma 4 31B (reference); greedy, 120 tokens.
Verdict rule pre-registered in PREREG_CHECKS.md. Usage: e16_topics.py OUT.json"""
import json, sys, time, subprocess, urllib.request, concurrent.futures as cf, os
LS = "llama-server"; PORT = 5996
G = {"qwen_base": ("models/Qwen3.8-27B-Q4_K_M.gguf", [], {}), "qwen_off32": ("models/gguf/off32-Q4_K_M.gguf", [], {}),
     "gemma": ("models/gemma-4-31b/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf", ["--reasoning", "off"], {"LLAMA_GRAPH_REUSE_DISABLE": "1"})}
T = json.load(open("./censura/checks/data/e16_topics.json"))
Q = [{"topic": t, "kind": k, "q": q} for t, d, ind, ref in T for k, q in (("direct", d), ("indirect", ind))]
def ask(q):
    body = {"messages": [{"role": "user", "content": q}], "max_tokens": 120, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"]
G["rand32"] = ("models/gguf/rand32-Q4_K_M.gguf", [], {})
if os.environ.get("E16VARIANTS"): G = {k: G[k] for k in os.environ["E16VARIANTS"].split(",")}
out = {"topics": T, "rows": Q, "ans": {}}
for v, (g, extra, env) in G.items():
    p = subprocess.Popen([LS, "-m", g, "-ngl", "999", "-sm", "layer", "-c", "16384", "--parallel", "8", "--port", str(PORT), "--host", "127.0.0.1", "-fa", "on", "--no-mmproj", *extra],
                         stdout=open(f"{sys.argv[1]}.{v}.server.log", "w"), stderr=subprocess.STDOUT, env={**os.environ, **env})
    try:
        for _ in range(600):
            try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception: time.sleep(2)
        with cf.ThreadPoolExecutor(8) as ex: out["ans"][v] = list(ex.map(lambda r: ask(r["q"]), Q))
        json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False); print(v, "fatto", flush=True)
    finally: p.terminate(); p.wait(30)
