import numpy as np, torch, json
A = load_acts("full_comp")
acts = A["acts"]; meta = A["meta"]; items = A["items"]
NI = len(items)
for L in [28, 29, 30, 31, 32, 33, 47]:
    for kind in ["resid", "attn_out"]:
        if L == 47 and kind == "attn_out": continue
        arr = torch.cat([t for t in acts[f"{kind}.{L}"]], 0).numpy().astype(np.float16)
        np.save(f"{WORK}/fc_{kind}_{L}.npy", arr)
lens = [len(m["tokens"]) for m in meta]
item_of = np.concatenate([np.full(n, i) for i, n in enumerate(lens)])
pos_of = np.concatenate([np.arange(n) for n in lens])
np.save(f"{WORK}/fc_item_of.npy", item_of); np.save(f"{WORK}/fc_pos_of.npy", pos_of)
json.dump({"meta": meta, "items": items}, open(f"{WORK}/fc_meta.json", "w"), ensure_ascii=False)
# decision index within full outputs, using dp prefix token counts
dpm = json.load(open(f"{WORK}/dp_meta.json"))
dp_items = dpm["items"]; dp_meta = dpm["meta"]
dp_ntok = {it["id"]: len(m["tokens"]) for it, m in zip(dp_items, dp_meta)}
dp_toks = {it["id"]: m["tokens"] for it, m in zip(dp_items, dp_meta)}
mism = 0
dec = {}
for i, it in enumerate(items):
    if "id" in it and it["id"] is not None:
        pid = it["id"]; k = dp_ntok[pid]
        if meta[i]["tokens"][:k] != dp_toks[pid]:
            mism += 1
            if mism < 6: print("mismatch", pid, meta[i]["tokens"][:k+1], dp_toks[pid])
        dec[i] = k - 1  # -1 means last prompt token (not in completion)
print("prefix token mismatches:", mism)
json.dump({str(k): v for k, v in dec.items()}, open(f"{WORK}/fc_dec.json", "w"))
print("total positions", len(item_of), "n items", NI)
