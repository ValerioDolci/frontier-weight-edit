import gc, numpy as np, torch
def sz(o):
    if isinstance(o, torch.Tensor): return o.numel() * o.element_size()
    if isinstance(o, np.ndarray): return o.nbytes
    if isinstance(o, dict): return sum(sz(v) for v in o.values())
    if isinstance(o, (list, tuple)): return sum(sz(v) for v in o[:100000])
    return 0
big = []
for k, v in list(globals().items()):
    if k.startswith("_"): continue
    try: s = sz(v)
    except Exception: s = 0
    if s > 20e6: big.append((k, round(s / 1e9, 2)))
print("big globals (GB):", big)
st = open("/proc/self/status").read()
for key in ["VmSize", "VmRSS", "VmData"]:
    print(key, st.split(key + ":")[1].split("\n")[0].strip())
