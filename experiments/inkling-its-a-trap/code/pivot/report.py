#!/usr/bin/env python3
"""(CPU) Read out the step-3 attribution: which neurons push towards hesitation, how concentrated they are, whether the
selection and verification halves agree. Usage: python report.py OUT_DIR  -> prints and writes OUT_DIR/report.json
(the key names, e.g. `top_pro_esita` = top pro-hesitation, `quota_top`, `per_strato_sel`, are kept as in the published JSON)."""
import json, os, sys
import torch

D = sys.argv[1]
S = torch.load(os.path.join(D, "attr_sum.pt"))
items = [json.loads(l) for l in open(os.path.join(D, "attr_items.jsonl"))]
mean = {k: v["sum"] / v["n"] for k, v in S.items()}
unit = lambda i: (int(i) // (128 * 1152), int(i) // 1152 % 128, int(i) % 1152)  # noqa: E731
out = {"n": {k: v["n"] for k, v in S.items()}}
for k in sorted(mean):
    ms = [x["m"] for x in items if f"{x['cat']}_{x['fold']}" == k]
    out.setdefault("margin", {})[k] = {"mean": sum(ms) / len(ms), "go_wins": sum(m > 0 for m in ms), "n": len(ms)}
a = mean["DIFF_sel"].flatten()
neg = torch.argsort(a)[:32]
out["top_pro_esita"] = [{"unit": unit(i), "attr_sel": float(a[i]),
                         **{f"attr_{k}": float(mean[k].flatten()[i]) for k in mean if k != "DIFF_sel"}} for i in neg]
tot_neg = float(a[a < 0].sum())
out["quota_top"] = {str(k): float(a[torch.argsort(a)[:k]].sum()) / tot_neg for k in (1, 8, 32, 256, 4096)}
per_layer = mean["DIFF_sel"].sum(dim=(1, 2))
out["per_strato_sel"] = [round(float(x), 4) for x in per_layer]
if "DIFF_ver" in mean:
    b = mean["DIFF_ver"].flatten()
    out["corr_sel_ver"] = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    tb = set(torch.argsort(b)[:100].tolist())
    out["overlap_top100_sel_ver"] = len(set(torch.argsort(a)[:100].tolist()) & tb)
    out["rank_ver_of_top8_sel"] = [int((b < b[i]).sum()) for i in neg[:8]]
json.dump(out, open(os.path.join(D, "report.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "top_pro_esita"}, indent=1))
for r in out["top_pro_esita"][:16]:
    print(r)
