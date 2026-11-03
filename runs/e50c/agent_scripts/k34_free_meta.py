import json, gc, ctypes
json.dump({"gen_lp": [M2[509 + k]["logprobs"] for k in range(195)],
           "m2": [{"n0": m["n_prompt_tokens"], "tok": m["tokens"]} for m in M2],
           "i2": I2,
           "m3": [{"n0": m["n_prompt_tokens"], "tok": m["tokens"], "lp": m["logprobs"]} for m in M3],
           "i3": I3}, open(f"{WORK}/meta_r2_r3.json", "w"))
for name in ["M1", "M2", "I2", "M3", "I3", "SPEC", "meta", "D", "pairs", "rows", "dec", "dec2", "dec3", "marg3", "d3", "ALT_DIR", "alts", "stats", "cfg", "CFG"]:
    if name in globals(): del globals()[name]
gc.collect()
print("trim", ctypes.CDLL("libc.so.6").malloc_trim(0))
st = open("/proc/self/status").read()
print("VmData", st.split("VmData:")[1].split("\n")[0].strip(), "VmRSS", st.split("VmRSS:")[1].split("\n")[0].strip())
