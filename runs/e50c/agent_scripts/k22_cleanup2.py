import gc
for name in ["acts", "S", "NDI", "NDn", "ALLn", "Wd", "Wg", "Wu", "Wg2", "Wu2", "Wd2", "Gs", "DI", "DD", "OTH", "DIn", "DDn", "OTn", "B", "Bz", "Z", "mu"]:
    if name in globals():
        del globals()[name]
gc.collect()
try:
    import ctypes
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception as ex:
    print("trim", ex)
print(open("/proc/self/status").read().split("VmRSS:")[1].split("\n")[0])
