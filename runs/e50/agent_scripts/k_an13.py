import numpy as np, pickle
A0 = load_acts("margin_orig"); A1 = load_acts("margin_e4")
items = A0["items"]; m0 = A0["meta"]; m1 = A1["meta"]
rows = []
for n in range(780, len(items)):
    d = m1[n]["sum_logprob"] - m0[n]["sum_logprob"]
    rows.append((d, n))
rows.sort()
for d, n in rows[:4]:
    it = items[n]
    print(round(d, 3), it.get("en", it.get("id")))
    toks = m0[n]["tokens"]; l0 = m0[n]["logprobs"]; l1 = m1[n]["logprobs"]
    print("   ", [(t, round(b - a, 2)) for t, a, b in zip(toks, l0, l1) if abs(b - a) > 0.05])
# prefix disruptions in DP items
rows = []
for n in range(0, 780):
    k = len(m0[n]["tokens"]) - 1
    d = sum(m1[n]["logprobs"][:k]) - sum(m0[n]["logprobs"][:k])
    rows.append((d, n))
rows.sort()
for d, n in rows[:6]:
    toks = m0[n]["tokens"]; l0 = m0[n]["logprobs"]; l1 = m1[n]["logprobs"]
    print(round(d, 2), items[n]["id"], items[n]["it"][:80], [(t, round(b - a, 2)) for t, a, b in zip(toks, l0, l1) if abs(b - a) > 0.1])
