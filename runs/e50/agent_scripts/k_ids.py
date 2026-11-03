from collections import defaultdict
bycat = defaultdict(list)
for p in practice:
    bycat[p["cat"]].append(p["id"])
for c, v in bycat.items():
    print(c, len(v))
p1 = bycat["DI"] + bycat["DD"] + bycat["PERF"][:31] + bycat["DURP"][:31] + bycat["HAB"][:30] + bycat["COND"][:30]
rest = [i for i in range(610) if i not in set(p1)]
print(len(p1), len(rest))
print("P1", json.dumps(p1))
print("REST", json.dumps(rest))
# sanity check id/line
print(practice[609])
