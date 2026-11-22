"""G1 (CPU): inkgpu against the PR's modeling (Inkling2LoopForCausalLM.from_pretrained) on the same synthetic
checkpoint (true topology, NVFP4/MXFP8 experts like the real one), same tokens.

  g1_synth.py --synth DIR [--synth-routed DIR] --out OUT.json

Needs the reference helpers `inkref` and `ref_dump` (from the capy-flash repository's Inkling tooling): set
INKLING_REF_BIN to the directory containing them.

Checks: (a) weights: every non-expert parameter bit-identical (dtype included), dequantized experts bit-identical
to the library's gate_up_proj/down_proj; (b) logits and outputs of the 48 visits, fp32 and bf16, with a single block and
with small blocks (attention and residuals); (c) gradient: inkgpu's attribution sum h*dm/dh (custom backward with
checkpointing) against the library's autograd (eager experts, hook on h) and against inkgpu's "plain" path; (d) G4 preview:
zeroing a neuron with the hook = zeroing its up row in the quantized tensors.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get("INKLING_REF_BIN", "ref/bin"))  # inkref, ref_dump: read-only helpers
import inkref  # noqa: E402
from ref_dump import Recorder  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkgpu  # noqa: E402


def kl_rows(a, b):
    a, b = a.double(), b.double()
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return (la.exp() * (la - lb)).sum(-1).clamp(min=0)


def cmp_logits(ref, eng):
    kl = kl_rows(ref, eng)
    top = ref.argmax(-1) == eng.argmax(-1)
    return {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "top1": float(top.double().mean()),
            "max_abs": float((ref.double() - eng.double()).abs().max())}


def rel(a, b):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return float((a - b).norm() / a.norm().clamp(min=1e-30))


def hf_forward(model, ids):
    T = len(ids)
    rec = Recorder(model, np.arange(T), parts=())
    rec.call(0, T)
    with torch.no_grad():
        lg = model(torch.tensor([ids])).logits[0].float()
    rec.close()
    data = rec.arrays(rec.data)
    return lg, data


def eng_forward(eng, ids):
    read = {v: None for v in range(48)}
    read.update({"embed": None, "final": None})
    out = eng.forward(ids, logit_pos="all", read=read)
    return out.logits.float().cpu(), out.reads


def compare_forward(hf_lg, hf_data, lg, reads):
    r = {"logits": cmp_logits(hf_lg, lg)}
    vis = {f"V{v:02d}": rel(hf_data[f"V{v:02d}"], reads[v]) for v in range(48)}
    r["visit_rel_head"] = [round(vis[f"V{v:02d}"], 7) for v in (0, 1, 5, 6, 17, 18, 41, 47)]
    r["visit_rel"] = vis
    r["visit_rel_max"] = max(vis.values())
    r["visit_bit_equal"] = sum(int(torch.equal(torch.as_tensor(hf_data[f"V{v:02d}"]), reads[v])) for v in range(48))
    r["embed_rel"] = rel(hf_data["embed"], reads["embed"])
    r["final_rel"] = rel(hf_data["final"], reads["final"])
    return r


def compare_weights(hf, eng):
    bad, n, nbit = [], 0, 0
    for name, t in list(eng.model.named_parameters()) + list(eng.model.named_buffers()):
        try:
            ref = hf.get_parameter(name)
        except AttributeError:
            ref = hf.get_buffer(name)
        n += 1
        if ref.dtype != t.dtype or ref.shape != t.shape or not torch.equal(ref.cpu(), t.cpu()):
            bad.append((name, str(ref.dtype), str(t.dtype)))
        else:
            nbit += 1
    exp_bad = []
    for L in range(len(eng.layers)):
        gu, dn = eng.dequant_layer(L)
        ex = hf.model.layers[L].mlp.experts
        if not (torch.equal(gu, ex.gate_up_proj.data) and torch.equal(dn, ex.down_proj.data)):
            exp_bad.append(L)
    return {"params": n, "params_bit_equal": nbit, "params_bad": bad[:10], "expert_layers": len(eng.layers),
            "expert_layers_bad": exp_bad}


class HFTap:
    """Attribution sum h*dm/dh with the library's autograd ('eager' experts): hook on _apply_gate per expert."""

    def __init__(self, model):
        cfg = model.config
        self.attr = torch.zeros(cfg.num_hidden_layers, cfg.n_routed_experts, cfg.moe_intermediate_size,
                                dtype=torch.float64)
        self.handles = []
        for L, layer in enumerate(model.model.layers):
            ex = layer.mlp.experts
            ex._ink_queue = []

            def pre(mod, args, L=L):
                top_k_index = args[1]
                hit = sorted(set(top_k_index.flatten().tolist()))
                mod._ink_queue = [e for e in hit if e < mod.num_experts]

            self.handles.append(ex.register_forward_pre_hook(pre))
            orig = ex._apply_gate

            def ap(gate_up, mod=ex, orig=orig, L=L):
                h = orig(gate_up)
                e = mod._ink_queue.pop(0)
                if h.requires_grad:
                    def hk(g, h=h, e=e, L=L):
                        self.attr[L, e] += (h.double() * g.double()).sum(0)
                    h.register_hook(hk)
                return h

            ex._apply_gate = ap

        def leaf(mod, a, o):
            return o.detach().requires_grad_(True) if torch.is_grad_enabled() else o

        self.handles.append(model.model.embed_tokens.register_forward_hook(leaf))


def margin_fn_of(tg, tb):
    def f(logits):
        lp = torch.log_softmax(logits[-1].float(), -1)
        return lp[tg] - lp[tb]

    return f


def topk_overlap(a, b, k=50):
    ia = set(torch.topk(a.abs().flatten(), k).indices.tolist())
    ib = set(torch.topk(b.abs().flatten(), k).indices.tolist())
    return len(ia & ib) / k


def grad_check(synth, ids, dtype, out):
    from transformers import Inkling2LoopForCausalLM

    hf = Inkling2LoopForCausalLM.from_pretrained(synth, dtype=dtype, attn_implementation="eager",
                                                 experts_implementation="eager")
    if dtype == torch.bfloat16:  # like inkref "bf16": lm_head in FP32
        hf.lm_head.float()
        hf.lm_head.register_forward_pre_hook(lambda m, a: (a[0].float(),))
    hf.eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        lg = hf(torch.tensor([ids])).logits[0, -1].float()
    o = torch.topk(lg, 2).indices.tolist()
    tg, tb = o[0], o[1]
    tap = HFTap(hf)
    with torch.enable_grad():
        lp = torch.log_softmax(hf(torch.tensor([ids])).logits[0, -1].float(), -1)
        m_hf = lp[tg] - lp[tb]
        m_hf.backward()
    attr_hf = tap.attr.float()
    del hf
    T = len(ids)
    eng = inkgpu.InkGPU.load(synth, devices=("cpu", "cpu"), split=13, dtype=dtype, attn_budget=2**12, res_chunk=64,
                             ckpt_min_len=0, verbose=False)
    r_small = eng.attribution(ids, margin_fn_of(tg, tb), logit_pos=[T - 1])
    eng = inkgpu.InkGPU.load(synth, devices=("cpu", "cpu"), split=13, dtype=dtype, verbose=False)
    r1 = eng.attribution(ids, margin_fn_of(tg, tb), logit_pos=[T - 1])
    eng.set_plain(True)
    r2 = eng.attribution(ids, margin_fn_of(tg, tb), logit_pos=[T - 1])
    eng.set_plain(False)
    res = {
        "dtype": str(dtype), "tg": tg, "tb": tb, "m_hf": float(m_hf.detach()), "m_custom": r1["m"],
        "m_custom_small_blocks": r_small["m"], "m_plain": r2["m"],
        "attr_norm_hf": float(attr_hf.norm()),
        "rel_custom_vs_hf": rel(attr_hf, r1["attr"]), "rel_plain_vs_hf": rel(attr_hf, r2["attr"]),
        "rel_custom_vs_plain": rel(r2["attr"], r1["attr"]),
        "rel_custom_small_vs_hf": rel(attr_hf, r_small["attr"]),
        "top50_overlap_custom_hf": topk_overlap(attr_hf, r1["attr"]),
        "top50_overlap_custom_plain": topk_overlap(r2["attr"], r1["attr"]),
        "top50_overlap_custom_small_hf": topk_overlap(attr_hf, r_small["attr"]),
        "sum_attr_hf": float(attr_hf.sum()), "sum_attr_custom": float(r1["attr"].sum()),
    }
    out[f"grad_{str(dtype).split('.')[-1]}"] = res
    print("grad", json.dumps(res), flush=True)
    return eng, r1, tg, tb


def g4_preview(eng, ids, attr, out):
    """Neuron with the largest |attr| in an NVFP4 layer and in an MXFP8 layer: hook (h and up) against an edit of the up row."""
    res = []
    T = len(ids)
    base = eng.forward(ids, logit_pos=range(T)).logits
    for lo, hi in ((0, 20), (20, 24)):
        a = attr[lo:hi].abs()
        f = int(a.flatten().argmax())
        L, e, j = lo + f // (a.shape[1] * a.shape[2]), (f // a.shape[2]) % a.shape[1], f % a.shape[2]
        r = {"neuron": [L, e, j], "fmt": eng.layers[L].mlp.w13.fmt, "attr": float(attr[L, e, j])}
        for where in ("h", "up"):
            eng.set_neuron_scale({(L, e, j): 0.0}, where=where)
            r[f"hook_{where}"] = eng.forward(ids, logit_pos=range(T)).logits
            eng.clear_neuron_scale()
        undo = eng.edit_zero_up(L, e, j)
        edit = eng.forward(ids, logit_pos=range(T)).logits
        undo()
        after = eng.forward(ids, logit_pos=range(T)).logits
        r["edit_equals_hook_h"] = bool(torch.equal(edit, r["hook_h"]))
        r["edit_equals_hook_up"] = bool(torch.equal(edit, r["hook_up"]))
        r["edit_vs_hook_h_maxabs"] = float((edit - r["hook_h"]).abs().max())
        r["effect_maxabs"] = float((edit - base).abs().max())
        r["undo_restores"] = bool(torch.equal(after, base))
        del r["hook_h"], r["hook_up"]
        res.append(r)
        print("g4", json.dumps(r), flush=True)
    out["g4_preview"] = res


def frozen_check(routed, ids, out):
    """Routed synthetic (top-4 of 16), fp32: frozen routing = same forward; central difference at frozen
    routing against the attribution for the 5 neurons with the largest |attr|."""
    eng = inkgpu.InkGPU.load(routed, devices=("cpu", "cpu"), split=13, dtype=torch.float32, attn_budget=2**12,
                             res_chunk=64, ckpt_min_len=0, verbose=False)
    T = len(ids)
    routes = eng.capture_routing(ids)
    a = eng.forward(ids, logit_pos=range(T)).logits
    b = eng.forward(ids, logit_pos=range(T), routes=routes).logits
    lg = a[-1]
    o = torch.topk(lg, 2).indices.tolist()
    mf = margin_fn_of(o[0], o[1])
    r = eng.attribution(ids, mf, logit_pos=[T - 1])
    rf = eng.attribution(ids, mf, logit_pos=[T - 1], routes=routes)
    res = {"frozen_forward_bit_equal": bool(torch.equal(a, b)), "attr_frozen_vs_free_rel": rel(r["attr"], rf["attr"]),
           "neurons": []}
    flat = r["attr"].flatten()
    E, Fd = r["attr"].shape[1:]
    eps = 1e-2
    for i in torch.topk(flat.abs(), 5).indices.tolist():
        n = (i // (E * Fd), (i // Fd) % E, i % Fd)
        ms = []
        for sgn in (1, -1):
            eng.set_neuron_scale({n: 1 + sgn * eps})
            ms.append(float(mf(eng.forward(ids, logit_pos=[T - 1], routes=routes).logits)))
        eng.clear_neuron_scale()
        cd = (ms[0] - ms[1]) / (2 * eps)
        res["neurons"].append({"neuron": list(n), "attr": float(flat[i]), "central_diff": cd, "ratio": cd / float(flat[i])})
    out["frozen_routed_fp32"] = res
    print("frozen", json.dumps(res), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth", required=True)
    p.add_argument("--synth-routed")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    ids = inkref.read_ids(f"{a.synth}/tokens.txt")
    out = {"T": len(ids)}
    t0 = time.time()
    for ck, tag in ((a.synth, "synth"), (a.synth_routed, "routed")):
        if not ck:
            continue
        oracle = None
        for impl, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            hf, info = inkref.load_model(ck, impl)
            hf_lg, hf_data = hf_forward(hf, ids)
            if oracle is None:
                oracle = (hf_lg, hf_data)
            else:  # floor: bf16 library against fp32 library
                fl = compare_forward(oracle[0], oracle[1], hf_lg, {v: torch.as_tensor(hf_data[f"V{v:02d}"]) for v in
                                                                   range(48)} | {"embed": torch.as_tensor(hf_data["embed"]),
                                                                                 "final": torch.as_tensor(hf_data["final"])})
                out[f"{tag}_floor_hfbf16_vs_hffp32"] = fl
                print(tag, "floor hf-bf16 vs hf-fp32", json.dumps({k: v for k, v in fl.items() if k != "visit_rel"}),
                      flush=True)
            for name, kw in (("one_block", {}), ("small_blocks", {"attn_budget": 2**12, "res_chunk": 64})):
                eng = inkgpu.InkGPU.load(ck, devices=("cpu", "cpu"), split=13, dtype=dtype, verbose=False, **kw)
                if name == "one_block":
                    w = compare_weights(hf, eng)
                    out[f"{tag}_{impl}_weights"] = w
                    print(tag, impl, "weights", json.dumps(w), flush=True)
                lg, reads = eng_forward(eng, ids)
                r = compare_forward(hf_lg, hf_data, lg, reads)
                r["experts_impl_hf"] = info["experts_implementation"]
                if impl == "bf16":
                    r["vs_hffp32"] = compare_forward(oracle[0], oracle[1], lg, reads)
                    r["vs_hffp32"].pop("visit_rel")
                out[f"{tag}_{impl}_{name}"] = r
                print(tag, impl, name, json.dumps({k: v for k, v in r.items() if k != "visit_rel"}), flush=True)
                del eng
            del hf
    if a.synth_routed:
        frozen_check(a.synth_routed, ids, out)
    for dtype in (torch.float32, torch.bfloat16):
        eng, r1, tg, tb = grad_check(a.synth, ids, dtype, out)
        if dtype == torch.float32:
            g4_preview(eng, ids, r1["attr"], out)
        del eng
    out["wall_s"] = round(time.time() - t0, 1)
    inkgpu.save_json(a.out, out)
    print(f"G1 done in {out['wall_s']} s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
