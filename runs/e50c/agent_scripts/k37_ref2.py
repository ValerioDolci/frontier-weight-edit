import gc, torch, collections
gc.collect()
ts = [o for o in gc.get_objects() if isinstance(o, torch.Tensor) and o.untyped_storage().nbytes() > 300000]
G = globals()
direct = [k for k, v in G.items() if isinstance(v, torch.Tensor)]
print("direct tensor globals:", direct[:40])
cnt = collections.Counter()
examples = {}
import random
random.seed(0)
for t in random.sample(ts, min(60, len(ts))):
    for r in gc.get_referrers(t):
        if r is ts: continue
        if r is G:
            cnt["NS-direct"] += 1; continue
        key = type(r).__name__
        if isinstance(r, list):
            # find who holds this list
            holders = [h for h in gc.get_referrers(r) if h is not ts]
            hk = []
            for h in holders:
                if h is G:
                    hk.append("NS:" + ",".join(k for k, v in G.items() if v is r))
                elif isinstance(h, dict):
                    hk.append("dict:" + ",".join(str(k) for k, v in h.items() if v is r)[:60])
                else:
                    hk.append(type(h).__name__)
            key = "list<-" + "|".join(hk)[:150]
        elif isinstance(r, dict):
            holders = [h for h in gc.get_referrers(r) if h is not ts]
            key = "dict(" + ",".join(list(map(str, r.keys()))[:3]) + ")<-" + "|".join(type(h).__name__ for h in holders)[:80]
        cnt[key] += 1
for k, v in cnt.most_common(12):
    print(v, k)
del ts
