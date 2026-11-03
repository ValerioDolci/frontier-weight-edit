import gc, ctypes
for name in ["ND1", "ND2", "NEG1", "e1", "DI_e2a", "DD_e2a", "G_e2a", "res"]:
    if name in globals():
        del globals()[name]
gc.collect()
libc = ctypes.CDLL("libc.so.6")
print("trim:", libc.malloc_trim(0))
st = open("/proc/self/status").read()
for key in ["VmSize", "VmRSS", "VmData"]:
    print(key, st.split(key + ":")[1].split("\n")[0].strip())
