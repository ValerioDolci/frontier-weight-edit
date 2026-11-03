import numpy as np, json
_d = json.load(open(WORK + "/dp.json"))
_DP = _d["DP"]; _errDI = set(_d["ERR_DI"]); _errDD = set(_d["ERR_DD"])
_cat = {p["id"]: p["cat"] for p in practice}
_dpm = json.load(open(f"{WORK}/dp_meta.json"))
_ntok = {it["id"]: len(m["tokens"]) for it, m in zip(_dpm["items"], _dpm["meta"])}
def margins(name):
    A = load_acts(name); meta = A["meta"]; items = A["items"]
    out = {"id": [], "cat": [], "err": [], "cm": [], "lpb": [], "lpa": [], "pref": []}
    for j, (i, pre, b, a) in enumerate(_DP):
        mb = meta[2 * j]; ma = meta[2 * j + 1]
        assert items[2 * j]["id"] == i and items[2 * j + 1]["id"] == i
        k = _ntok[i]
        lpb = mb["logprobs"][k]; lpa = ma["logprobs"][k]
        e = (i in _errDI) or (i in _errDD)
        sign = -1.0 if e else 1.0
        out["id"].append(i); out["cat"].append(_cat[i]); out["err"].append(e)
        out["cm"].append(sign * (lpb - lpa)); out["lpb"].append(lpb); out["lpa"].append(lpa)
        out["pref"].append(sum(mb["logprobs"][:k]))
    rest = [(items[n], meta[n]["sum_logprob"], len(meta[n]["tokens"])) for n in range(2 * len(_DP), len(items))]
    for k in out: out[k] = np.array(out[k])
    out["rest_lp"] = np.array([r[1] for r in rest]); out["rest_n"] = np.array([r[2] for r in rest])
    out["rest_kind"] = np.array(["gen" if ("en" in r[0]) else _cat[r[0]["id"]] for r in rest])
    return out
def summarize(o, base=None):
    cm = o["cm"]; cat = o["cat"]; err = o["err"]
    def frac_pos(m): return f"{(cm[m] > 0).sum()}/{m.sum()}"
    s = []
    for c in ["DI", "DD"]:
        for e in [False, True]:
            m = (cat == c) & (err == e)
            s.append(f"{c}-{'err' if e else 'ok'}: correct-side {frac_pos(m)} median cm {np.median(cm[m]):.2f}")
    print(" | ".join(s))
    if base is not None:
        d = o["cm"] - base["cm"]
        for c in ["DI", "DD"]:
            for e in [False, True]:
                m = (cat == c) & (err == e)
                print(f"   d(cm) {c}-{'err' if e else 'ok'}: mean {d[m].mean():+.2f} median {np.median(d[m]):+.2f} min {d[m].min():+.2f} max {d[m].max():+.2f}")
        dp = o["pref"] - base["pref"]
        print(f"   prefix lp change: mean {dp.mean():+.3f} min {dp.min():+.3f}")
        dr = o["rest_lp"] - base["rest_lp"]
        for k in ["gen", "PERF", "DURP", "HAB", "COND"]:
            m = o["rest_kind"] == k
            print(f"   {k}: sum-lp change mean {dr[m].mean():+.3f} min {dr[m].min():+.3f} (n={m.sum()}, per-token mean {dr[m].sum()/o['rest_n'][m].sum():+.4f})")
MB = margins("margin_orig")
summarize(MB)
import pickle; pickle.dump(MB, open(f"{WORK}/margin_orig.pkl", "wb"))
print("rest kinds:", {k: int((MB['rest_kind'] == k).sum()) for k in set(MB['rest_kind'])})
# sanity: DI-err items have base(trapassato) > alt?
m = (MB["cat"] == "DI") & MB["err"]
print("DI-err lpb/lpa sample:", list(zip(MB["id"][m][:10], np.round(MB["lpb"][m][:10], 2), np.round(MB["lpa"][m][:10], 2))))
