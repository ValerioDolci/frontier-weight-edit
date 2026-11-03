import ctypes, os, re, torch, numpy as np
def _arena_report(tag):
    libc = ctypes.CDLL("libc.so.6")
    libc.fopen.restype = ctypes.c_void_p; libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    libc.fclose.argtypes = [ctypes.c_void_p]; libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
    p = os.path.join(WORK, "malloc_info.xml").encode()
    f = libc.fopen(p, b"w"); libc.malloc_info(0, f); libc.fclose(f)
    x = open(p.decode()).read()
    tot = re.search(r'</heap>\s*<total type="rest" count="\d+" size="(\d+)"/>.*?<system type="current" size="(\d+)"', x, re.S)
    rows = []
    for nr, body in re.findall(r'<heap nr="(\d+)">(.*?)</heap>', x, re.S):
        cur = re.search(r'<system type="current" size="(\d+)"', body)
        rows.append((int(nr), int(cur.group(1)) if cur else 0))
    rows.sort(key=lambda r: -r[1])
    st = open("/proc/self/status").read()
    print(tag, "VmData", st.split("VmData:")[1].split("\n")[0].strip(), "| top arenas (GB):", [(a, round(c / 1e9, 2)) for a, c in rows[:6]])
_arena_report("before")
torch.set_num_threads(os.cpu_count() or 8)
for _ in range(3):
    _a = torch.randn(3000, 3000); _b = torch.randn(3000, 3000)
    _c = _a @ _b
    _s = torch.sort(torch.randn(20_000_000))[0]
    del _a, _b, _c, _s
_arena_report("after torch ops")
for _ in range(2):
    _a = np.random.rand(3000, 3000).astype(np.float32); _b = np.random.rand(3000, 3000).astype(np.float32)
    _c = _a @ _b
    _g = _a.T @ _a + np.eye(3000, dtype=np.float32)
    _x = np.linalg.solve(_g, _b[:, :8])
    del _a, _b, _c, _g, _x
_arena_report("after numpy ops")
ctypes.CDLL("libc.so.6").malloc_trim(0)
_arena_report("after trim")
