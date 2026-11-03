import gc, numpy as np, torch, sys
gc.collect()
tot_np, tot_t, n_np, n_t = 0, 0, 0, 0
bigs = []
for o in gc.get_objects():
    try:
        if isinstance(o, np.ndarray):
            if o.base is None:
                tot_np += o.nbytes; n_np += 1
                if o.nbytes > 50e6: bigs.append(("np", o.shape, o.dtype, o.nbytes))
        elif isinstance(o, torch.Tensor):
            s = o.untyped_storage().nbytes()
            tot_t += s; n_t += 1
            if s > 50e6: bigs.append(("t", tuple(o.shape), o.dtype, s))
    except Exception:
        pass
print(f"numpy owned: {tot_np/1e9:.2f} GB in {n_np}; torch storages (may double count views): {tot_t/1e9:.2f} GB in {n_t}")
print("big:", bigs[:20])
print("num gc objects", len(gc.get_objects()))
