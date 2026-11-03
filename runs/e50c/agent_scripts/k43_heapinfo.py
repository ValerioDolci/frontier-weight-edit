import ctypes
HEAP_MAX = 64 * 1024 * 1024
_regs = []
for line in open("/proc/self/maps"):
    p = line.split()
    a, b = [int(x, 16) for x in p[0].split("-")]
    path = p[5] if len(p) > 5 else ""
    if p[1] == "rw-p" and path == "" and a % HEAP_MAX == 0:
        _regs.append((a, b))
print("64MB-aligned anon rw regions:", len(_regs), "total GB", sum(b - a for a, b in _regs) / 1e9)
class HeapInfo(ctypes.Structure):
    _fields_ = [("ar_ptr", ctypes.c_void_p), ("prev", ctypes.c_void_p), ("size", ctypes.c_size_t),
                ("mprotect_size", ctypes.c_size_t), ("pagesize", ctypes.c_size_t)]
ok, rel, tot_size, tot_mp = 0, 0, 0, 0
arenas = set()
bad = 0
for a, b in _regs:
    h = HeapInfo.from_address(a)
    if h.mprotect_size == b - a and h.size <= h.mprotect_size and h.size <= HEAP_MAX and h.ar_ptr:
        ok += 1; rel += h.mprotect_size - h.size; tot_size += h.size; tot_mp += h.mprotect_size; arenas.add(h.ar_ptr)
    else:
        bad += 1
print(f"valid heaps {ok}, invalid {bad}, arenas {len(arenas)}; sum size {tot_size/1e9:.2f} GB, sum mprotect {tot_mp/1e9:.2f} GB, releasable beyond size: {rel/1e9:.2f} GB")
