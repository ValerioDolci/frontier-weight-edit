import json
bycat = {}
for p in practice:
    bycat.setdefault(p["cat"], []).append(p["id"])
for c, v in bycat.items():
    print(c, len(v))
ids1 = bycat["DI"] + bycat["DD"] + bycat["PERF"] + bycat["DURP"] + bycat["HAB"][:2]
print(json.dumps({"kind": "generate", "ids": ids1, "weights": "original", "max_new": 200}))
