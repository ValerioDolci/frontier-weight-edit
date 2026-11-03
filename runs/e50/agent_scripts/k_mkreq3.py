d = json.load(open(WORK + "/dp.json"))
items = [{"id": i, "it": pre} for (i, pre, b, a) in d["DP"]] + [{"id": i, "it": pre} for (i, pre) in d["DPO"]]
L = list(range(48))
req = {"kind": "read", "items": items, "layers": L, "capture": ["resid", "attn_out", "mlp_out", "mlp_act"], "positions": "last", "weights": "original", "save": "dp_last"}
print(json.dumps(req, ensure_ascii=False))
