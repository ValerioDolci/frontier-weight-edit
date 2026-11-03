import torch, json, collections
A = load_acts("main1")
acts, meta, items = A["acts"], A["meta"], A["items"]
cat = {p["id"]: p["cat"] for p in practice}
DI_ERR = {3,8,69,78,92,94,112,116,128,143,153,161,165,171,173,188,198,209,212,217,223,236,247,273,291,294,298,300,325,334,369,382,384,389,399,411,414,447,448,464,501,504,523,524,530,536,538,540,551,561,562,571,573,583,587,588,593}
DD_ERR = {107,144,196,285,550,566}
# B items are first occurrence of each id, A second
first = {}; second = {}
for k, it in enumerate(items):
    if it["id"] in first: second[it["id"]] = k
    else: first[it["id"]] = k
rows = []
for i, kb in first.items():
    if i not in second: continue
    ka = second[i]
    tb, ta = meta[kb]["tokens"], meta[ka]["tokens"]
    t = 0
    while t < min(len(tb), len(ta)) and tb[t] == ta[t]: t += 1
    if t >= len(ta):
        print("no divergence", i, tb[:6], ta); continue
    lb, la = meta[kb]["logprobs"][t], meta[ka]["logprobs"][t]
    c = cat[i]
    if c == "DI":
        b_is_imp = i not in DI_ERR
    else:
        b_is_imp = i in DD_ERR
    if i in (439, 556): b_is_imp = False  # trapassato+da judged ok
    if i == 186: b_is_imp = False
    imp_m = (lb - la) if b_is_imp else (la - lb)
    rows.append(dict(id=i, cat=c, t=t, tokB=tb[t], tokA=ta[t], lpB=lb, lpA=la, b_is_imp=b_is_imp, imp_m=imp_m,
                     ok=(i not in DI_ERR) if c == "DI" else (i not in DD_ERR), kb=kb, ka=ka))
print(len(rows))
json.dump(rows, open(WORK + "/dec_rows.json", "w"))
import numpy as np
for c in ["DI", "DD"]:
    for ok in [True, False]:
        sel = [r for r in rows if r["cat"] == c and r["ok"] == ok]
        if not sel: continue
        ms = np.array([r["imp_m"] for r in sel])
        t0 = sum(1 for r in sel if r["t"] == 0)
        print(c, "ok" if ok else "err", len(sel), "t0:", t0, "imp_margin quantiles:", np.round(np.quantile(ms, [0, .1, .25, .5, .75, .9, 1]), 2))
print("--- DI err detail")
for r in sorted([r for r in rows if r["cat"] == "DI" and not r["ok"]], key=lambda r: r["imp_m"]):
    print(r["id"], r["t"], repr(r["tokB"]), round(r["lpB"], 2), repr(r["tokA"]), round(r["lpA"], 2), "imp_m", round(r["imp_m"], 2))
print("--- DD detail (sorted by imp_m desc)")
for r in sorted([r for r in rows if r["cat"] == "DD"], key=lambda r: -r["imp_m"])[:40]:
    print(r["id"], "ok" if r["ok"] else "ERR", r["t"], repr(r["tokB"]), round(r["lpB"], 2), repr(r["tokA"]), round(r["lpA"], 2), "imp_m", round(r["imp_m"], 2))
