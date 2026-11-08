import json, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/Qwen3.8-27B"); d = json.load(open("./taiwan/weightscan.json"))
P = {}; LK = []
for l, v in d.items():
    for k, xs in v["pct"].items(): P.setdefault(k, []).extend(xs)
    LK += v["logit_kurt_pct"]
print("percentile medio dei 32 (50 = tipico):", {k: round(float(np.mean(x)), 1) for k, x in P.items()}, "lente-curtosi", round(float(np.mean(LK)), 1))
print("quanti sopra il 90:", {k: int(np.sum(np.array(x) > 90)) for k, x in P.items()}, "lente", int(np.sum(np.array(LK) > 90)), "su", len(LK))
dec = lambda xs: " | ".join(tok.convert_tokens_to_string([x]).strip() for x in xs[:7])
for l, v in d.items():
    for t in v["tokens"]: print("L%s n%d  UP %s   DOWN %s" % (l, t["j"], dec(t["up"]), dec(t["down"])))
