import threading, gc, ctypes
def _vm():
    st = open("/proc/self/status").read()
    return st.split("VmData:")[1].split("\n")[0].strip()
print("before", _vm())
_N = 512
_bar = threading.Barrier(_N + 1)
def _worker():
    try:
        b = bytearray(256 * 1024)
        b2 = bytearray(1024 * 1024)
        del b, b2
    finally:
        try:
            _bar.wait(timeout=60)
        except Exception:
            pass
_ths = [threading.Thread(target=_worker) for _ in range(_N)]
for t in _ths: t.start()
try:
    _bar.wait(timeout=60)
except Exception:
    pass
for t in _ths: t.join()
del _ths
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
print("after", _vm())
