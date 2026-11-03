import gc, ctypes, threading
def _deep(G):
    keep = {"names", "W", "save_edit", "drop_edit", "edits", "load_acts", "practice", "SESSION", "WORK", "torch", "np", "json",
            "_deep", "gc", "ctypes", "threading"}
    for name in list(G.keys()):
        if name.startswith("__") or name in keep: continue
        del G[name]
_deep(globals())
gc.collect()
def _vm():
    st = open("/proc/self/status").read()
    return st.split("VmData:")[1].split("\n")[0].strip()
print("after deleting globals", _vm())
_N = 700
_bar = threading.Barrier(_N + 1)
def _worker():
    try:
        for sz in (70 * 1024, 300 * 1024, 2 * 1024 * 1024, 20 * 1024 * 1024):
            b = bytearray(sz); del b
    finally:
        try: _bar.wait(timeout=120)
        except Exception: pass
_ths = [threading.Thread(target=_worker) for _ in range(_N)]
for t in _ths: t.start()
try: _bar.wait(timeout=120)
except Exception: pass
for t in _ths: t.join()
del _ths, _bar, _worker, t
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
print("after arena trim", _vm())
