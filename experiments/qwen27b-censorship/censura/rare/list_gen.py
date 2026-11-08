"""Stage 4, step 1: the base model (GGUF Q4, llama-server) lists 40 entries per seed category. Usage: list_gen.py OUT.json"""
import json, sys, time, subprocess, urllib.request, concurrent.futures as cf
sys.path.insert(0, "./censura/rare"); from seeds import CN, WORLD, NEUTRAL
PORT = 5992; srv = subprocess.Popen(["llama-server", "-m", "models/Qwen3.8-27B-Q4_K_M.gguf", "-ngl", "999", "-sm", "layer",
       "-c", "16384", "--parallel", "8", "--port", str(PORT), "--host", "127.0.0.1", "-fa", "on"], stdout=open(sys.argv[1] + ".server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
        except Exception: time.sleep(2)
    def ask(seed):
        q = f"List 40 distinct, specific examples of: {seed}. Output one name per line, no numbering, no explanations."
        body = {"messages": [{"role": "user", "content": q}], "max_tokens": 700, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        a = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))["choices"][0]["message"]["content"]
        items = [l.strip(" -*•\t0123456789.").strip() for l in a.splitlines()]
        return seed, [x for x in items if 2 < len(x) < 80]
    out = {}
    with cf.ThreadPoolExecutor(8) as ex:
        for seed, items in ex.map(ask, [(s) for s in CN + WORLD + NEUTRAL]): out[seed] = items; print(seed, len(items), flush=True)
    json.dump({"cats": {"CN": CN, "WORLD": WORLD, "NEUTRAL": NEUTRAL}, "lists": out}, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
finally:
    srv.terminate(); srv.wait(30)
