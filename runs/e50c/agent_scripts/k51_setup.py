import ctypes, gc, json, numpy as np, torch
_libc = ctypes.CDLL("libc.so.6")
_libc.mallopt(-3, 131072)   # M_MMAP_THRESHOLD = 128KB -> big buffers via mmap (returned on free)
_libc.mallopt(-8, 4)        # M_ARENA_MAX = 4
def vmdata():
    st = open("/proc/self/status").read()
    return st.split("VmData:")[1].split("\n")[0].strip()
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps) * w
SL = {30: 0.7227, 31: 0.6993, 32: 0.5546, 33: 0.629, 34: 0.75}
NSYS = 45
LN = {}
for L in [31, 34]:
    P = f"model.language_model.layers.{L}"
    LN[L] = {"pa": W(f"{P}.post_attention_layernorm.weight").float(),
             "pre": W(f"{P}.pre_feedforward_layernorm.weight").float(),
             "pf": W(f"{P}.post_feedforward_layernorm.weight").float()}
def mlp_in_from(A, L, i):
    h = A[f"resid.{L-1}"][i].float() + rmsn(A[f"attn_out.{L}"][i].float(), LN[L]["pa"])
    return rmsn(h, LN[L]["pre"])
def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(neg) > 20000: neg = np.random.RandomState(0).choice(neg, 20000, replace=False)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float((pos[:, None] > neg[None, :]).mean())
# ---- append HAB/COND (r4) rows as negatives ----
r4 = load_acts("r4"); M4, I4, A4 = r4["meta"], r4["items"], r4["acts"]
add = {31: [], 34: []}; g4, f4, it4, rw4 = [], [], [], []
for i in range(len(I4)):
    n0 = M4[i]["n_prompt_tokens"]
    for L in [31, 34]:
        add[L].append(mlp_in_from(A4, L, i)[NSYS:].half().numpy())
    for r in range(NSYS, n0 + len(M4[i]["tokens"])):
        g4.append("FIRSTNEG" if r == n0 - 1 else ("SRC" if r < n0 - 1 else "OUT"))
        f4.append(i % 2); it4.append(i); rw4.append(r)
del r4, A4; gc.collect()
meta = dict(np.load(f"{WORK}/Xmeta.npz"))
for L in [31, 34]:
    X = np.load(f"{WORK}/X{L}_f16.npy")
    if X.shape[0] == len(meta["grp"]):
        X = np.concatenate([X, np.concatenate(add[L])])
        np.save(f"{WORK}/X{L}_f16.npy", X)
    print(L, X.shape)
    del X
if len(meta["grp"]) < 41148 + len(g4) - 1:
    meta["grp"] = np.concatenate([meta["grp"], np.array(g4)]); meta["fold"] = np.concatenate([meta["fold"], np.array(f4)])
    meta["src"] = np.concatenate([meta["src"], np.full(len(g4), 4)]); meta["item"] = np.concatenate([meta["item"], np.array(it4)])
    meta["row"] = np.concatenate([meta["row"], np.array(rw4)])
    np.savez(f"{WORK}/Xmeta.npz", **meta)
print("meta rows", len(meta["grp"]), {g: int((meta["grp"] == g).sum()) for g in np.unique(meta["grp"])})
del add, meta; gc.collect()
print("VmData", vmdata())
