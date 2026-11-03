import gc, sys, numpy as np, torch
def sz(o):
    if isinstance(o, torch.Tensor): return o.numel() * o.element_size()
    if isinstance(o, np.ndarray): return o.nbytes
    if isinstance(o, dict): return sum(sz(v) for v in o.values())
    if isinstance(o, (list, tuple)): return sum(sz(v) for v in o[:100000])
    return 0
big = []
for k, v in list(globals().items()):
    if k.startswith("_"): continue
    try:
        s = sz(v)
    except Exception:
        s = 0
    if s > 50e6: big.append((k, round(s / 1e9, 2)))
print("big globals (GB):", big)
# keep r1 meta only
try:
    M1 = r1["meta"]
except Exception:
    pass
for name in ["r1", "A1", "e1", "res_e1", "X", "Xc", "Xs", "Xt", "G", "gx", "ux", "CV", "Xs_", "dirs", "rows"]:
    if name in globals():
        del globals()[name]
gc.collect()
big = []
for k, v in list(globals().items()):
    if k.startswith("_"): continue
    try:
        s = sz(v)
    except Exception:
        s = 0
    if s > 50e6: big.append((k, round(s / 1e9, 2)))
print("after cleanup:", big)
try:
    import resource
    print("maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
except Exception as ex:
    print(ex)
try:
    print(open("/proc/self/status").read().split("VmRSS:")[1].split("\n")[0])
except Exception as ex:
    print(ex)
