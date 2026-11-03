import numpy as np, torch, json, gc
def build_from_saved(det_tag, d_over, tag, kappa=None, apply=True):
    cfg = json.load(open(f"{WORK}/cfg_{det_tag}.json")); CFG = {}
    for L in [31, 34]:
        c = np.load(f"{WORK}/{det_tag}_c_{L}.npy")
        P = f"model.language_model.layers.{L}"
        Wg = W(f"{P}.mlp.gate_proj.weight"); Wu = W(f"{P}.mlp.up_proj.weight"); Wd = W(f"{P}.mlp.down_proj.weight")
        wpf = LN[L]["pf"].numpy(); rms_dec = {31: 0.151, 34: 0.389}[L]
        for pos in ["Pf", "Pm"]:
            k = f"{L}_{pos}"; e = dict(cfg[k])
            w = np.load(f"{WORK}/{det_tag}_w_{L}_{pos}.npy"); theta = e["theta"]; Dl = e["Dl"]
            d = d_over[k] if k in d_over else np.load(f"{WORK}/{det_tag}_d_{L}_{pos}.npy")
            kap = kappa[pos] if kappa else e["kappa"]
            v = rms_dec * kap * d / SL[L] / np.maximum(wpf, 1e-6); v[wpf < 0.1] = 0
            a = 8.0 / Dl; b = 1.0 / (a * Dl); jj = e["neurons"]
            Wg[jj[0]] = torch.from_numpy(a * (w - theta * c)); Wg[jj[1]] = torch.from_numpy(a * (w - (theta + Dl) * c))
            Wu[jj[0]] = torch.from_numpy(b * c); Wu[jj[1]] = torch.from_numpy(b * c)
            Wd[:, jj[0]] = torch.from_numpy(v.astype(np.float32)); Wd[:, jj[1]] = torch.from_numpy(-v.astype(np.float32))
            e.update(kappa=kap, dnorm=float(np.linalg.norm(d)), vnorm=float(np.linalg.norm(v)), d_src="override" if k in d_over else det_tag)
            CFG[k] = e
            np.save(f"{WORK}/{tag}_w_{L}_{pos}.npy", w); np.save(f"{WORK}/{tag}_d_{L}_{pos}.npy", d)
            print(f"{tag} L{L} {pos}: theta {theta:.3f} Dl {Dl:.3f} kappa {kap} |d|={np.linalg.norm(d):.2f} |v|={np.linalg.norm(v):.2f} d from {CFG[k]['d_src']}")
        np.save(f"{WORK}/{tag}_c_{L}.npy", c)
        if apply:
            save_edit(f"{P}.mlp.gate_proj.weight", Wg); save_edit(f"{P}.mlp.up_proj.weight", Wu); save_edit(f"{P}.mlp.down_proj.weight", Wd)
        del Wg, Wu, Wd; gc.collect()
    json.dump(CFG, open(f"{WORK}/cfg_{tag}.json", "w"))
    return CFG
NEWD = np.load(f"{WORK}/dirs_combined_gen.npy", allow_pickle=True).item()
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
for k in ["31_Pm", "34_Pm"]:
    print(k, "cos(E3b d, combined d) =", round(cos(np.load(f"{WORK}/E3b_d_{k}.npy"), NEWD[k]), 3))
CFG_E3c = build_from_saved("E3b", {"31_Pm": NEWD["31_Pm"], "34_Pm": NEWD["34_Pm"]}, "E3c")
print("edits:", edits())
print("VmData", vmdata())
