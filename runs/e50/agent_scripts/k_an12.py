import numpy as np, torch, json
A = load_acts("full_all_e2")
acts = A["acts"]; meta = A["meta"]; items = A["items"]
NI = len(items)
lens = [acts["resid.29"][i].shape[0] for i in range(NI)]
npr = [m["n_prompt_tokens"] for m in meta]; nc = [len(m["tokens"]) for m in meta]
print("check lens == prompt+completion:", all(l == p + c for l, p, c in zip(lens, npr, nc)), "total", sum(lens))
for L in [29, 30, 39, 40]:
    for kind in ["resid", "attn_out"]:
        arr = torch.cat(acts[f"{kind}.{L}"], 0).numpy().astype(np.float16)
        np.save(f"{WORK}/fa_{kind}_{L}.npy", arr)
item_of = np.concatenate([np.full(n, i) for i, n in enumerate(lens)])
pos_of = np.concatenate([np.arange(n) for n in lens])
is_prompt = np.concatenate([np.arange(n) < p for n, p in zip(lens, npr)])
np.save(f"{WORK}/fa_item_of.npy", item_of); np.save(f"{WORK}/fa_pos_of.npy", pos_of); np.save(f"{WORK}/fa_is_prompt.npy", is_prompt)
json.dump({"meta": meta, "items": items}, open(f"{WORK}/fa_meta.json", "w"), ensure_ascii=False)
# gate at L30 everywhere
L = 30
X = mlp_in(np.load(f"{WORK}/fa_resid_29.npy").astype(np.float32), np.load(f"{WORK}/fa_attn_out_30.npy").astype(np.float32), L)
w = np.load(f"{WORK}/det_w2_L{L}.npy")
g = neuron_out(X.astype(np.float64), L, w)
np.save(f"{WORK}/fa_gate30.npy", g.astype(np.float32))
cat = {p["id"]: p["cat"] for p in practice}
kind = np.array(["gen" if ("en" in it) else cat[it["id"]] for it in items])[item_of]
# the system prompt is identical across items: positions < some fixed length; find the user-text span roughly as prompt positions after the system prompt
for k in ["DI", "DD", "PERF", "DURP", "HAB", "COND", "gen"]:
    m = kind == k
    pm = m & is_prompt; cm = m & ~is_prompt
    print(f"{k}: prompt pos frac(g>0.1) {np.mean(g[pm] > 0.1)*1000:.2f}/1000 max {g[pm].max():.2f} | completion frac(g>0.1) {np.mean(g[cm] > 0.1)*1000:.2f}/1000 max {g[cm].max():.2f} | n_prompt {pm.sum()} n_comp {cm.sum()}")
# which prompt positions fire? report a few with item and position
idx = np.where(is_prompt & (g > 0.1))[0]
print("n firing prompt positions:", len(idx))
for j in idx[:30]:
    i = item_of[j]; print("  item", i, items[i].get("id", "gen"), "pos", pos_of[j], "of prompt", npr[i], "g %.2f" % g[j])
