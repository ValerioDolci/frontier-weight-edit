import gc, ctypes, types, torch, numpy as np
def _cleanup(G):
    keep = {"names", "W", "save_edit", "drop_edit", "edits", "load_acts", "practice", "SESSION", "WORK", "torch", "np", "json",
            "WPA", "WPRE", "gc", "ctypes", "types", "_cleanup"}
    dropped = []
    for name in list(G.keys()):
        if name.startswith("__") or name in keep: continue
        val = G[name]
        if isinstance(val, (types.FunctionType, types.ModuleType, type)): continue
        del G[name]; dropped.append(name)
    return dropped
_d = _cleanup(globals())
print("dropped", len(_d), _d[:80])
del _d
gc.collect()
print("trim", ctypes.CDLL("libc.so.6").malloc_trim(0))
_ts = [o for o in gc.get_objects() if isinstance(o, torch.Tensor)]
print("tensors alive:", len(_ts), "GB:", sum(t.untyped_storage().nbytes() for t in _ts) / 1e9)
del _ts
_st = open("/proc/self/status").read()
print("VmData", _st.split("VmData:")[1].split("\n")[0].strip(), "VmRSS", _st.split("VmRSS:")[1].split("\n")[0].strip())
del _st
