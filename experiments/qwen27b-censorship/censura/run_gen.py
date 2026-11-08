"""Phase A: short answers to the whole battery with llama-server (starts the server on the given GGUF, queries in parallel, stops it).
Usage: run_gen.py GGUF OUT.jsonl [N_PREDICT]"""
import json, sys, os, time, subprocess, concurrent.futures as cf, urllib.request
GGUF, OUT = sys.argv[1], sys.argv[2]; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 96; PORT = 5991; PAR = 8
B = [json.loads(l) for l in open("./censura/battery.jsonl")]
done = set()
if os.path.exists(OUT): done = {json.loads(l)["id"] for l in open(OUT)}
srv = subprocess.Popen(["llama-server", "-m", GGUF, "-ngl", "999", "-sm", "layer", "-c", str(PAR * 1536), "--parallel", str(PAR),
                        "--port", str(PORT), "--host", "127.0.0.1", "-fa", "on", "--no-mmproj"], stdout=open(OUT + ".server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
        except Exception: time.sleep(2)
    def ask(r):
        body = {"messages": [{"role": "user", "content": r["q"]}], "max_tokens": NP, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        a = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
        return {**r, "a": a}
    todo = [r for r in B if r["id"] not in done]; t0 = time.time()
    with open(OUT, "a") as f, cf.ThreadPoolExecutor(PAR) as ex:
        for k, res in enumerate(ex.map(ask, todo)):
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            if k % 200 == 0: print(f"{k}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print("FATTO", time.time() - t0, flush=True)
finally:
    srv.terminate(); srv.wait(30)
