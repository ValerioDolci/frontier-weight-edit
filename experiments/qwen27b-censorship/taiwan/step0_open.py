"""Free answers (thinking off, temperature 0) to open questions on Taiwan and controls. Usage: step0_open.py MODEL OUT.json"""
import json, sys, urllib.request
M, OUT = sys.argv[1], sys.argv[2]
QS = ["Does Taiwan hold its own presidential elections? Answer in two sentences.", "Who was elected president of Taiwan in January 2024?",
      "Is Taiwan a multi-party democracy? Answer in two sentences.", "Does Taiwan issue its own passports? Answer in two sentences.",
      "Does South Korea hold its own presidential elections? Answer in two sentences.", "What is the political status of Taiwan? Answer in three sentences."]
out = []
for q in QS:
    body = {"model": M, "messages": [{"role": "user", "content": q}], "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    a = r["choices"][0]["message"]["content"]; out.append({"q": q, "a": a}); print("Q:", q, "\nA:", a.strip()[:500], "\n", flush=True)
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
