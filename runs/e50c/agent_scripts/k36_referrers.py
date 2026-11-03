import gc, torch
gc.collect()
ts = [o for o in gc.get_objects() if isinstance(o, torch.Tensor)]
big = [t for t in ts if t.untyped_storage().nbytes() > 300000]
print("tensors >300KB:", len(big))
t0 = big[0]
print("example shape", tuple(t0.shape), t0.dtype)
seen = 0
for r in gc.get_referrers(t0):
    if r is ts or r is big: continue
    tp = type(r).__name__
    info = ""
    if isinstance(r, list): info = f"list len {len(r)}"
    elif isinstance(r, dict): info = f"dict keys {list(r.keys())[:5]}"
    print("referrer:", tp, info)
    for rr in gc.get_referrers(r):
        if rr is ts or rr is big: continue
        tp2 = type(rr).__name__
        info2 = f"dict keys {list(rr.keys())[:8]}" if isinstance(rr, dict) else (f"list len {len(rr)}" if isinstance(rr, list) else "")
        print("   2nd-level:", tp2, info2)
        for r3_ in gc.get_referrers(rr):
            if r3_ is ts or r3_ is big: continue
            if isinstance(r3_, dict):
                print("      3rd-level dict keys:", list(r3_.keys())[:10])
    seen += 1
    if seen > 4: break
del ts, big, t0
