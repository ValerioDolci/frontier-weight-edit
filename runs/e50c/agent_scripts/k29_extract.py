import numpy as np, torch, json, gc, ctypes
print("have WPA:", "WPA" in globals() and sorted(WPA.keys()), "WPRE:", "WPRE" in globals() and sorted(WPRE.keys()))
dec2 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r2.json")).items()}
d3 = json.load(open(f"{WORK}/dec_r3.json")); dec3 = {int(k): v for k, v in d3["dec3"].items()}; marg3 = {int(k): v for k, v in d3["marg3"].items()}
NSYS = 45
def mlp_in_src(src, L, i):
    A = A2 if src == 2 else A3
    h = A[f"resid.{L-1}"][i].float() + rmsn(A[f"attn_out.{L}"][i].float(), WPA[L])
    return rmsn(h, WPRE[L])
SPEC = []
for i in range(len(I2)):
    kind = "DI" if i < 259 else ("DD" if i < 389 else "NEG")
    lab = "W" if i < 55 else ("R" if i < 259 else None)
    SPEC.append((2, i, kind, dec2.get(i), i % 2, lab))
for k in range(120):
    m = marg3[k]; lab = "R" if m > 1.0 else ("W" if m < -1.0 else None)
    SPEC.append((3, k, "DI", dec3[k], k % 2, lab))
for k in range(240, 330):
    SPEC.append((3, k, "DD", dec3.get(k), k % 2, None))
for k in range(330, 390):
    SPEC.append((3, k, "NEG", None, k % 2, None))
meta = {"grp": [], "fold": [], "src": [], "item": [], "row": []}
first_pass = True
for L in [31, 34]:
    Xs = []
    for (src, i, kind, dr, f, lab) in SPEC:
        M = M2 if src == 2 else M3
        xi = mlp_in_src(src, L, i)
        n0 = M[i]["n_prompt_tokens"]
        start = 0 if (src == 2 and i == 0) else NSYS
        Xs.append(xi[start:].half())
        if first_pass:
            for r in range(start, xi.shape[0]):
                if kind == "DI" and dr is not None and r == dr:
                    g = "Pf" if dr == n0 - 1 else "Pm"
                elif kind == "DI" and dr is not None and dr - 1 <= r <= dr + 3:
                    g = "IGN"
                elif r == n0 - 1:
                    g = "FIRSTNEG"
                elif kind == "DD" and dr is not None and r == dr:
                    g = "DDdec"
                else:
                    g = "SYS" if r < NSYS else ("SRC" if r < n0 - 1 else "OUT")
                meta["grp"].append(g); meta["fold"].append(f); meta["src"].append(src); meta["item"].append(i); meta["row"].append(r)
    X = torch.cat(Xs).numpy()
    np.save(f"{WORK}/X{L}_f16.npy", X)
    print("saved X", L, X.shape, X.dtype)
    del Xs, X; gc.collect()
    first_pass = False
for k in meta: meta[k] = np.array(meta[k])
np.savez(f"{WORK}/Xmeta.npz", **meta)
# decision-row residuals for directions (L31, L34) + labels + type
D = {"L31": [], "L34": [], "lab": [], "first": [], "src": [], "item": []}
for (src, i, kind, dr, f, lab) in SPEC:
    if kind != "DI" or dr is None or lab is None: continue
    A = A2 if src == 2 else A3; M = M2 if src == 2 else M3
    D["L31"].append(A["resid.31"][i][dr].float().numpy()); D["L34"].append(A["resid.34"][i][dr].float().numpy())
    D["lab"].append(lab); D["first"].append(dr == M[i]["n_prompt_tokens"] - 1); D["src"].append(src); D["item"].append(i)
for k in D: D[k] = np.array(D[k])
np.savez(f"{WORK}/dirdata.npz", **D)
print("dir rows:", len(D["lab"]), "W:", int((D["lab"] == "W").sum()), "R:", int((D["lab"] == "R").sum()))
# free big activations
for name in ["r2", "A2", "r3", "A3", "Xs", "X"]:
    if name in globals(): del globals()[name]
gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
st = open("/proc/self/status").read()
print("VmSize", st.split("VmSize:")[1].split("\n")[0].strip(), "VmRSS", st.split("VmRSS:")[1].split("\n")[0].strip())
