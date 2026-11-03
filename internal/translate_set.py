"""Traduce un set TSV (cat<TAB>src) en->it con il prompt CONGELATO di bench-harness/eval_translate.py.
Uso: translate_set.py --base-url URL --model ID --inp set.tsv --out out.jsonl [--workers N]
Greedy (temperature 0, seed 1234), reasoning off: identico a eval_translate.py."""
import argparse, json, urllib.request, concurrent.futures as cf, time

SYSTEM = ("You are a professional translator. Translate the user's text from {src} "
          "into {tgt}. Reply with the translation only: no notes, no comments, no "
          "quotation marks, no explanations.")

def ask(base, model, text, max_tokens=512):
    p = {"model": model, "temperature": 0.0, "top_p": 1.0, "seed": 1234, "max_tokens": max_tokens,
         "stream": False, "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "system", "content": SYSTEM.format(src="English", tgt="Italian")},
                      {"role": "user", "content": text}]}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    for att in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                j = json.loads(r.read().decode())
            return (j["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            if att == 2: raise
            time.sleep(5)

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--base-url", required=True); a.add_argument("--model", required=True)
    a.add_argument("--inp", required=True); a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=1)
    g = a.parse_args()
    rows = []
    for i, line in enumerate(open(g.inp, encoding="utf-8")):
        if line.strip():
            cat, src = line.rstrip("\n").split("\t", 1)
            rows.append({"id": i, "cat": cat, "src": src})
    t0 = time.time()
    with cf.ThreadPoolExecutor(g.workers) as ex:
        hyps = list(ex.map(lambda r: ask(g.base_url, g.model, r["src"]), rows))
    with open(g.out, "w", encoding="utf-8") as f:
        for r, h in zip(rows, hyps):
            r["hyp"] = h; f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} frasi in {time.time()-t0:.0f} s -> {g.out}", flush=True)

if __name__ == "__main__":
    main()
