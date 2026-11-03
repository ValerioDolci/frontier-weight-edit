import numpy as np, torch, json
r1 = load_acts("r1")
meta, items, acts = r1["meta"], r1["items"], r1["acts"]
LAYERS = sorted(int(k.split(".")[1]) for k in acts)
alts = json.load(open(f"{WORK}/alt_dir.json"))
ALT_DIR = {int(k): v for k, v in alts["ALT_DIR"].items()}
pairs = json.load(open(f"{WORK}/pairs_r1.json"))

def tok_text(toks):
    return "".join(t.replace("▁", " ") for t in toks)

def tok_index_of_char(toks, c):
    pos = 0
    for i, t in enumerate(toks):
        s = t.replace("▁", " ")
        if pos <= c < pos + len(s):
            return i
        pos += len(s)
    return len(toks) - 1

dec = {}  # idx -> (t, drow, group)
# DI wrong
for p in pairs:
    if p["kind"] == "DI":
        dec[p["a"]] = (p["t"], p["drow"], "DIW")
    else:
        # DD pair: a = trap item, b = imp item; state at drow identical. store under the model-output index
        dec_idx = p["a"] if items[p["a"]]["it"] == items[p["a"]]["it"] else p["a"]
        dec[("DDP", p["id"])] = (p["t"], p["drow"], "DDP", p["a"], p["b"])
# DI right
nfail = 0
for i in range(110, 314):
    iid = items[i]["id"]
    if iid not in ALT_DIR:
        continue
    s = items[i]["it"]; a = ALT_DIR[iid]
    toks = meta[i]["tokens"]
    rec = tok_text(toks)
    if rec.strip() != s.strip():
        nfail += 1
    # first differing char
    c = next((k for k in range(min(len(s), len(a))) if s[k] != a[k]), min(len(s), len(a)))
    # map char offset in s to rec offset (rec may have leading space differences)
    off = len(rec) - len(rec.lstrip()) - (len(s) - len(s.lstrip()))
    t = tok_index_of_char(toks, c + off)
    dec[i] = (t, meta[i]["n_prompt_tokens"] + t - 1, "DIR")
print("recon fails", nfail)
# show some DIR decisions
cnt = 0
for i in range(110, 314):
    if i in dec and cnt < 40:
        t = dec[i][0]; toks = meta[i]["tokens"]
        print(items[i]["id"], t, repr("".join(toks[max(0,t-2):t]).replace("▁"," ")), "->", repr(toks[t]), meta[i]["logprobs"][t])
        cnt += 1
json.dump({str(k): v for k, v in dec.items() if isinstance(k, int)}, open(f"{WORK}/dec_r1.json", "w"))
