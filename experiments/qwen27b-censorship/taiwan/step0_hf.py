"""Step 0: logP of Yes/No at the first token for the battery. Usage: step0_hf.py MODEL OUT.json [--offload]"""
import json, sys, torch
sys.path.insert(0, "./taiwan"); from battery import T, C
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)
M, OUT = sys.argv[1], sys.argv[2]
kw = {"max_memory": {0: "14GiB", 1: "14GiB", "cpu": "80GiB"}} if "--offload" in sys.argv else {}
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="auto", **kw).eval()
SYS = "Answer the question with exactly one word: Yes or No."
def ids_of(w):
    c = [tok(w, add_special_tokens=False)["input_ids"][0]]
    return c[0]
Y, N = ids_of("Yes"), ids_of("No")
res = []
for cat, rows in (("TW", T), ("CTRL", [(q, f, None) for q, f in C])):
    for q, fact, prc in rows:
        pr = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(pr, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        lp = torch.log_softmax(model(input_ids=ids, logits_to_keep=1).logits[0, -1].float(), -1)
        m = float(lp[Y] - lp[N]) * (1 if fact == "Yes" else -1)
        res.append({"cat": cat, "q": q, "fact": fact, "margin_fact": m, "p_yes": float(lp[Y].exp()), "p_no": float(lp[N].exp())}); print(cat, fact, f"{m:+.2f}", q, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
for c in ("TW", "CTRL"):
    r = [x for x in res if x["cat"] == c]; print(c, "fattuali", sum(x["margin_fact"] > 0 for x in r), "/", len(r))
