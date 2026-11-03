import numpy as np, torch, json, gc
def build_slots(det_tag, kappa, slots, tag, apply=True):
    cfg = json.load(open(f"{WORK}/cfg_{det_tag}.json")); CFG = {}
    for L in [31, 34]:
        c = np.load(f"{WORK}/{det_tag}_c_{L}.npy")
        P = f"model.language_model.layers.{L}"
        Wg = W(f"{P}.mlp.gate_proj.weight"); Wu = W(f"{P}.mlp.up_proj.weight"); Wd = W(f"{P}.mlp.down_proj.weight")
        wpf = LN[L]["pf"].numpy(); rms_dec = {31: 0.151, 34: 0.389}[L]
        for pos in ["Pf", "Pm"]:
            k = f"{L}_{pos}"; e = dict(cfg[k])
            w = np.load(f"{WORK}/{det_tag}_w_{L}_{pos}.npy"); d = np.load(f"{WORK}/{det_tag}_d_{L}_{pos}.npy")
            theta = e["theta"]; Dl = e["Dl"]; kap = kappa[pos]; jj = slots[k]
            v = rms_dec * kap * d / SL[L] / np.maximum(wpf, 1e-6); v[wpf < 0.1] = 0
            a = 8.0 / Dl; b = 1.0 / (a * Dl)
            Wg[jj[0]] = torch.from_numpy(a * (w - theta * c)); Wg[jj[1]] = torch.from_numpy(a * (w - (theta + Dl) * c))
            Wu[jj[0]] = torch.from_numpy(b * c); Wu[jj[1]] = torch.from_numpy(b * c)
            Wd[:, jj[0]] = torch.from_numpy(v.astype(np.float32)); Wd[:, jj[1]] = torch.from_numpy(-v.astype(np.float32))
            e.update(kappa=kap, neurons=jj, vnorm=float(np.linalg.norm(v)), d_src=det_tag)
            CFG[k] = e
            np.save(f"{WORK}/{tag}_w_{L}_{pos}.npy", w); np.save(f"{WORK}/{tag}_d_{L}_{pos}.npy", d)
            print(f"{tag} L{L} {pos}: theta {theta:.3f} Dl {Dl:.3f} kappa {kap} |v|={np.linalg.norm(v):.2f} slots {jj}")
        np.save(f"{WORK}/{tag}_c_{L}.npy", c)
        if apply:
            save_edit(f"{P}.mlp.gate_proj.weight", Wg); save_edit(f"{P}.mlp.up_proj.weight", Wu); save_edit(f"{P}.mlp.down_proj.weight", Wd)
        del Wg, Wu, Wd; gc.collect()
    json.dump(CFG, open(f"{WORK}/cfg_{tag}.json", "w"))
    return CFG
S = json.load(open(f"{WORK}/slots_min.json"))
SLOTS_G = {"31_Pf": S["31"][:2], "31_Pm": S["31"][2:], "34_Pf": S["34"][:2], "34_Pm": S["34"][2:]}
CFG_E3g = build_slots("E3e", {"Pf": 3.0, "Pm": 3.0}, SLOTS_G, "E3g")
print("edits:", edits())
print("VmData", vmdata())
