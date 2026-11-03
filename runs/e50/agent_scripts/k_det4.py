import numpy as np, json
fam = json.load(open(f"{WORK}/fa_meta.json")); items = fam["items"]; meta = fam["meta"]
item_of = np.load(f"{WORK}/fa_item_of.npy"); pos_of = np.load(f"{WORK}/fa_pos_of.npy"); is_prompt = np.load(f"{WORK}/fa_is_prompt.npy")
npr = np.array([m["n_prompt_tokens"] for m in meta])
cat = {p["id"]: p["cat"] for p in practice}
is_gen_item = np.array(["en" in it for it in items])
dpm = json.load(open(f"{WORK}/dp_meta.json")); ntok = {it["id"]: len(m["tokens"]) for it, m in zip(dpm["items"], dpm["meta"])}
# system prompt length: positions identical between item 0 and 1
R29 = np.load(f"{WORK}/fa_resid_29.npy")
s0 = np.where(item_of == 0)[0]; s1 = np.where(item_of == 1)[0]
nsame = 0
while nsame < min(len(s0), len(s1)) and np.abs(R29[s0[nsame]].astype(np.float32) - R29[s1[nsame]].astype(np.float32)).max() < 1e-2: nsame += 1
print("shared system-prompt positions:", nsame)
# labels per position: 1 = DI decision, 0 = negative, -1 = excluded
labp = np.zeros(len(item_of), np.int8); wpos = np.ones(len(item_of))
dec_abs = {}
for i, it in enumerate(items):
    if "id" in it:
        pid = it["id"]; k = ntok[pid]
        dec_abs[i] = npr[i] + k - 1
for i, d in dec_abs.items():
    m = (item_of == i) & (pos_of >= d - 2) & (pos_of <= d + 2) & (pos_of >= npr[i] - 1)
    labp[m] = -1
    j = np.where((item_of == i) & (pos_of == d))[0][0]
    c = cat[items[i]["id"]]
    labp[j] = 1 if c == "DI" else 0
    wpos[j] = 20.0 if c == "DI" else 5.0
# dedupe system prompt: drop positions < nsame for all items except item 0
dup = (pos_of < nsame) & (item_of != 0)
labp[dup] = -1
# give the shared prefix positions of item 0 extra weight (they occur in every input)
wpos[(item_of == 0) & (pos_of < nsame)] = 50.0
use = labp >= 0
print("positives", (labp == 1).sum(), "negatives", (labp == 0).sum())
np.save(f"{WORK}/fa_lab.npy", labp); np.save(f"{WORK}/fa_w.npy", wpos)
json.dump({"nsame": int(nsame), "dec_abs": {str(k): int(v) for k, v in dec_abs.items()}}, open(f"{WORK}/fa_dec.json", "w"))
# CV by item group
L = 30
X = mlp_in(R29.astype(np.float32), np.load(f"{WORK}/fa_attn_out_30.npy").astype(np.float32), L)
group = np.array([items[i]["id"] if "id" in items[i] else 10000 + i for i in item_of])
rng = np.random.default_rng(1); ug = np.unique(group); fg = dict(zip(ug, rng.integers(0, 5, len(ug))))
fold = np.array([fg[g] for g in group])
theta = 0.78
Xu = X[use].astype(np.float64); yu = (labp[use] == 1).astype(np.float64) - theta; wu = wpos[use]; fu = fold[use]
s = np.zeros(len(Xu))
for f in range(5):
    tr = fu != f
    if f == 0: tr = tr | ((item_of[use] == 0) & (pos_of[use] < nsame))  # system prompt always in train
    Xt = Xu[tr]; wt = wu[tr]
    C = (Xt * wt[:, None]).T @ Xt
    lam = 0.03 * np.trace(C) / C.shape[0]
    wv = np.linalg.solve(C + lam * np.eye(C.shape[0]), (Xt * wt[:, None]).T @ yu[tr])
    te = fu == f
    s[te] = Xu[te] @ wv
iu = item_of[use]; pu = is_prompt[use]; lu = labp[use]
kind = np.array(["gen" if is_gen_item[i] else cat[items[i]["id"]] for i in iu])
dec_mask = np.zeros(len(Xu), bool)
abs_pos = pos_of[use]
for n in range(len(Xu)):
    pass
isdec = np.array([dec_abs.get(int(i), -9) == int(p) for i, p in zip(iu, abs_pos)])
errDI = set(json.load(open(WORK + "/dp.json"))["ERR_DI"]); errDD = set(json.load(open(WORK + "/dp.json"))["ERR_DD"])
pid = np.array([items[i].get("id", -1) for i in iu])
e = np.array([(p in errDI) or (p in errDD) for p in pid])
DI_ = s[isdec & (kind == "DI")]; DIe = s[isdec & (kind == "DI") & e]; DDo = s[isdec & (kind == "DD") & ~e]; O_ = s[isdec & np.isin(kind, ["PERF", "DURP", "HAB", "COND"])]
print("CV: DI>0 %.2f DIerr>0 %.2f DDok>0 %.3f other>0 %.3f" % ((DI_ > 0).mean(), (DIe > 0).mean(), (DDo > 0).mean(), (O_ > 0).mean()))
for k in ["DI", "DD", "gen"]:
    pm = (kind == k) & pu & ~isdec; cm = (kind == k) & ~pu & ~isdec
    print(f"  {k}: prompt>0 {np.mean(s[pm] > 0)*1000:.2f}/1000 (max {s[pm].max():.2f}) | completion>0 {np.mean(s[cm] > 0)*1000:.2f}/1000 (max {s[cm].max():.2f})")
