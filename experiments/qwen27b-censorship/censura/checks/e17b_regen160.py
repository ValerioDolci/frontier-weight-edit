"""E17b: regenerates the prior's answers with 160 tokens, as in stage 4, for a like-for-like comparison.
Usage: e17b_regen160.py E17_DIR"""
import json, sys, time, subprocess, urllib.request, concurrent.futures as cf, os
O = sys.argv[1]; LS = "llama-server"; PORT = 5998
P = json.load(open(f"{O}/e17_prior_answers.json"))
p = subprocess.Popen([LS, "-m", "models/Qwen3.8-27B-Q4_K_M.gguf", "-ngl", "999", "-sm", "layer", "-c", "16384", "--parallel", "8", "--port", str(PORT),
                      "--host", "127.0.0.1", "-fa", "on", "--no-mmproj"], stdout=open(f"{O}/server160.log", "w"), stderr=subprocess.STDOUT)
def ask(q):
    body = {"messages": [{"role": "user", "content": q}], "max_tokens": 160, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"]
try:
    for _ in range(600):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
        except Exception: time.sleep(2)
    with cf.ThreadPoolExecutor(8) as ex: A = list(ex.map(lambda r: ask(r["q"]), P["rows"]))
finally: p.terminate(); p.wait(30)
for r, a in zip(P["rows"], A): r["a"] = a
P["max_tokens"] = 160; json.dump(P, open(f"{O}/e17_prior_answers_160.json", "w"), indent=1, ensure_ascii=False); print("FATTO", len(A))
