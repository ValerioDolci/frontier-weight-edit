"""Stage 4, step 3: base-model answers (GGUF Q4) for the selected entries. Usage: rare_gen.py IN.jsonl OUT.jsonl"""
import json, sys, time, subprocess, urllib.request, concurrent.futures as cf
PORT = 5993; srv = subprocess.Popen(["llama-server", "-m", "models/Qwen3.8-27B-Q4_K_M.gguf", "-ngl", "999", "-sm", "layer",
       "-c", "12288", "--parallel", "8", "--port", str(PORT), "--host", "127.0.0.1", "-fa", "on"], stdout=open(sys.argv[2] + ".server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
        except Exception: time.sleep(2)
    R = [json.loads(l) for l in open(sys.argv[1])]
    def ask(r):
        body = {"messages": [{"role": "user", "content": r["q"]}], "max_tokens": 160, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        a = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))["choices"][0]["message"]["content"]
        return {**r, "a": a}
    with open(sys.argv[2], "w") as f, cf.ThreadPoolExecutor(8) as ex:
        for res in ex.map(ask, R): f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print("FATTO", flush=True)
finally:
    srv.terminate(); srv.wait(30)
