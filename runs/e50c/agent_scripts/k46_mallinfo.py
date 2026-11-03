import ctypes, os, re
libc = ctypes.CDLL("libc.so.6")
libc.fopen.restype = ctypes.c_void_p
libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
libc.fclose.argtypes = [ctypes.c_void_p]
libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
_p = os.path.join(WORK, "malloc_info.xml").encode()
_f = libc.fopen(_p, b"w")
libc.malloc_info(0, _f)
libc.fclose(_f)
_x = open(_p.decode()).read()
_heaps = re.findall(r'<heap nr="(\d+)">(.*?)</heap>', _x, re.S)
_rows = []
for nr, body in _heaps:
    cur = int(re.search(r'<system type="current" size="(\d+)"', body).group(1))
    rest = re.search(r'<total type="rest" count="(\d+)" size="(\d+)"', body)
    fast = re.search(r'<total type="fast" count="(\d+)" size="(\d+)"', body)
    sub = re.search(r'<aspace type="subheaps" size="(\d+)"', body)
    free = int(rest.group(2)) + int(fast.group(2))
    _rows.append((int(nr), cur, free, int(sub.group(1)) if sub else -1))
_tot_cur = sum(r[1] for r in _rows); _tot_free = sum(r[2] for r in _rows)
print("arenas:", len(_rows), "system current GB:", round(_tot_cur / 1e9, 2), "free GB:", round(_tot_free / 1e9, 2))
_big = sorted(_rows, key=lambda r: -r[1])[:12]
for r in _big:
    print(f"arena {r[0]}: current {r[1]/1e9:.2f} GB, free {r[2]/1e9:.2f} GB, in-use {(r[1]-r[2])/1e6:.1f} MB, subheaps {r[3]}")
_tail = _x[-600:]
print(_tail)
