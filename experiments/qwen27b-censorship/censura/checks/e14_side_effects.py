"""E14 (exploratory): new errors introduced by off32 on neutral, verifiable facts about China (28 questions with references,
EN + ZH, Q4 base / off32). Hypothesis written before running. Usage: e14_side_effects.py OUT.json"""
import json, sys, time, subprocess, urllib.request, concurrent.futures as cf
LS = "llama-server"; PORT = 5995; G = {"base": "models/Qwen3.8-27B-Q4_K_M.gguf", "off32": "models/gguf/off32-Q4_K_M.gguf"}
F = json.load(open("./censura/checks/data/e14_facts.json"))
Q = [{"i": i, "lang": "en", "q": e, "ref": r} for i, (e, z, r) in enumerate(F)] + [{"i": i, "lang": "zh", "q": z, "ref": r} for i, (e, z, r) in enumerate(F)]
def ask(q):
    body = {"messages": [{"role": "user", "content": q + (" Answer in one or two sentences." if q.isascii() else " 请用一两句话回答。")}], "max_tokens": 80, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"]
out = {"rows": Q, "ans": {}}
for v, g in G.items():
    p = subprocess.Popen([LS, "-m", g, "-ngl", "999", "-sm", "layer", "-c", "16384", "--parallel", "8", "--port", str(PORT), "--host", "127.0.0.1", "-fa", "on", "--no-mmproj"],
                         stdout=open(f"{sys.argv[1]}.{v}.server.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception: time.sleep(2)
        with cf.ThreadPoolExecutor(8) as ex: out["ans"][v] = list(ex.map(lambda r: ask(r["q"]), Q))
        json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False); print(v, "fatto", flush=True)
    finally: p.terminate(); p.wait(30)
