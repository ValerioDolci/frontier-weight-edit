"""Gates G2-G5 on the real model, 2 GPUs. One load per session, phases in order.

  gpu_gates.py --ckpt models/inkling-2-mini --phases g2 --out out/          (G2: against the CPU bf16-A dumps and ref/greedy)
  gpu_gates.py --ckpt ... --phases g3,g4,g5 --out out/
  gpu_gates.py --ckpt ... --phases est                                       (memory estimate only, no GPU)

Needs the reference tooling and recorded reference data of the capy-flash Inkling port (not part of this folder):
  INKLING_REF_BIN  directory with inkref.py, tfdump.py, split_compare.py, route_compare.py (default: ref/bin)
  INKLING_REF_DIR  directory with dump/ (bf16-A, bf16-c16) and greedy/*.json (default: ref)
  PYTHON           interpreter used to run split_compare.py / route_compare.py (default: the current one)
  --ckpt           checkpoint directory (default: $INKLING_CKPT or models/inkling-2-mini)

G2: tfdump/1 dump of it1/en1/code1 (logits for all rows, outputs of the 48 visits and final at the reference rows,
    routing) -> split_compare.py against ref/dump/bf16-A with floor ref/dump/bf16-c16, route_compare.py; top-1 under
    teacher forcing on the recorded greedy continuation (8 prompts of ref/greedy, long1 = 12 957 tokens).
G3: margin m = log p(tg) - log p(tb) at the end of the context; attribution sum h*dm/dh; top-5 neurons by |attr| + 5 random
    ones (among the routed ones): dm measured by zeroing them one at a time with the hook against the prediction -attr.
G4: neuron zeroed with the hook (h and up) against the up row zeroed in the quantized tensors: identical logits?
G5: forward and forward+backward at 2k, 8k, 16k tokens: seconds and per-card peak VRAM.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

BIN = os.environ.get("INKLING_REF_BIN", "ref/bin")
REF = os.environ.get("INKLING_REF_DIR", "ref")
sys.path.insert(0, BIN)  # tfdump, inkref: read-only helpers
import inkref  # noqa: E402
import tfdump  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkgpu  # noqa: E402

PY = os.environ.get("PYTHON", sys.executable)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sync(eng):
    for d in eng.devices:
        if d.type == "cuda":
            torch.cuda.synchronize(d)


def reset_peak(eng):
    for d in eng.devices:
        if d.type == "cuda":
            torch.cuda.reset_peak_memory_stats(d)


def peaks(eng):
    return {str(d): round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in eng.devices if d.type == "cuda"}


def reserved(eng):
    return {str(d): round(torch.cuda.max_memory_reserved(d) / 2**30, 2) for d in eng.devices if d.type == "cuda"}


# ---------------------------------------------------------------------------------------------------------- estimate
def estimate(cfg, T, split, static_gib):
    """Estimate of the per-card peak (GiB) for forward and forward+backward at T tokens. static_gib: weights per device."""
    H, nh = cfg.hidden_size, cfg.num_attention_heads
    W = cfg.sliding_window_size
    table = inkgpu.visit_table(cfg)
    nvis = [sum(1 for _, L, _ in table if (L < split) == (d == 0)) for d in (0, 1)]
    TH2 = T * H * 2
    budget = 2**20
    C = inkgpu._chunk_len(T, T, 0, budget)
    att = nh * C * min(T, T) * 18  # bf16 scores + bias + mask + fp32 softmax, one query block
    deq = 4 * (2 * 2304 * 2304 + 2304 * 1152) * 4 * 1.5  # group of 8 experts: fp32 + bf16 intermediates
    moe = 3 * T * H * 4  # out32, gx32, g32
    res = 1024 * 6 * H * 4 * 3  # chunk of the attention residual
    layer_tmp = 12 * T * H * 4  # conv fp32, q/k/v, message gate fp32, attn_res fp32 [T,H] x nq
    out = {}
    for d in (0, 1):
        fwd = static_gib[d] + (max(att, deq, res) + layer_tmp + moe) / 2**30
        saved = (nvis[d] * TH2 + 6 * TH2) / 2**30  # visit outputs + block states (copies per device)
        bwd = static_gib[d] + saved + (2 * max(att, deq, res) + 2 * layer_tmp + moe) / 2**30
        out[f"dev{d}"] = {"fwd": round(fwd, 2), "fwd_bwd": round(bwd, 2), "visits": nvis[d]}
    return out


# ---------------------------------------------------------------------------------------------------------- G2
def g2(eng, out_dir, a):
    res = {}
    gdir = Path(out_dir) / "g2"
    gdir.mkdir(parents=True, exist_ok=True)
    for name in a.g2_names.split(","):
        ref = Path(REF) / "dump" / "bf16-A" / name
        ids = [int(i) for i in np.load(ref / "tokens.npy")]
        rows = np.load(ref / "hidden" / "V00.rows.npy")
        T = len(ids)
        if a.g2_all_rows:
            rows = np.arange(T)
        recs = {}

        def route(moe, logits, w, idx, rows_t=torch.as_tensor(rows)):
            r = rows_t.to(idx.device)
            i, ww, lg = idx[r], w[r], logits[r]
            srt, order = i.sort(dim=-1)
            v = moe.cur_visit
            recs[f"V{v:02d}.idx"] = srt.int().cpu().numpy()
            recs[f"V{v:02d}.w"] = ww.gather(-1, order).float().cpu().numpy()
            recs[f"V{v:02d}.logits"] = lg.float().cpu().numpy()

        read = {v: rows for v in range(48)}
        read.update({"embed": rows, "final": None})
        reset_peak(eng)
        sync(eng)
        t0 = time.time()
        o = eng.forward(ids, logit_pos="all", read=read, route=route)
        sync(eng)
        fwd_s = time.time() - t0
        lg = o.logits.float().cpu()
        lp = torch.log_softmax(lg.double(), -1)
        nxt = torch.tensor(ids[1:])
        token_lp = lp[:-1].gather(-1, nxt[:, None])[:, 0].float().numpy()
        meta = {"format": tfdump.FORMAT, "producer": "experiments/inkling-its-a-trap/code/inkgpu/gpu_gates.py", "impl": "inkgpu-bf16",
                "devices": [str(d) for d in eng.devices], "split": eng.split, "T": T, "forward_s": round(fwd_s, 3),
                "argmax": [int(i) for i in lg.argmax(-1)]}
        d = tfdump.Dump(ids, meta)
        d.token_logprobs = token_lp
        d.put("logits", lg.numpy(), np.arange(T))
        d.put("hidden.embed", o.reads["embed"].numpy(), rows)
        for v in range(48):
            d.put(f"hidden.V{v:02d}", o.reads[v].numpy(), rows)
        d.put("hidden.final", o.reads["final"][torch.as_tensor(rows)].numpy(), rows)
        dd = gdir / name
        d.save(dd)
        (dd / "route").mkdir(exist_ok=True)
        np.save(dd / "route" / "rows.npy", rows)
        for k, v in recs.items():
            np.save(dd / "route" / f"{k}.npy", v)
        cmp_json = gdir / f"cmp-{name}.json"
        cmd = [PY, "-B", f"{BIN}/split_compare.py", str(ref), str(dd), "--prompt-len", f"{REF}/greedy/{name}.json",
               "--floor", f"{REF}/dump/bf16-c16/{name}", "--json", str(cmp_json)]
        sc = subprocess.run(cmd, capture_output=True, text=True)
        rc = subprocess.run([PY, "-B", f"{BIN}/route_compare.py", str(ref), str(dd)], capture_output=True, text=True)
        (gdir / f"cmp-{name}.txt").write_text(sc.stdout + sc.stderr + "\n" + rc.stdout + rc.stderr)
        log(f"G2 {name} T={T} forward {fwd_s:.2f} s peak {peaks(eng)}")
        print(sc.stdout + sc.stderr, flush=True)
        diff = [ln for ln in rc.stdout.splitlines() if "set differs on    0/" not in ln]
        print(f"route_compare {name}: {len(rc.stdout.splitlines()) - len(diff)} visits with the same routing on all "
              f"rows; others: " + " | ".join(ln[:90] for ln in diff[:12]), flush=True)
        res[name] = {"T": T, "forward_s": fwd_s, "peak_gib": peaks(eng)}
    # top-1 teacher-forced on the recorded greedy continuation
    tf = {}
    for rec_path in sorted((Path(REF) / "greedy").glob("*.json")):
        if a.tf_names != "all" and rec_path.stem not in a.tf_names.split(","):
            continue
        r = json.loads(rec_path.read_text())
        ids = r["prompt_ids"] + r["generated_ids"]
        P, n = r["n_prompt"], r["n_generated"]
        pos = list(range(P - 1, P - 1 + n))
        reset_peak(eng)
        sync(eng)
        t0 = time.time()
        o = eng.forward(ids, logit_pos=pos)
        sync(eng)
        dt = time.time() - t0
        lg = o.logits.float().cpu()
        top2 = torch.topk(lg, 2, dim=-1)
        am = top2.indices[:, 0].tolist()
        mism = []
        for k, (g, e) in enumerate(zip(r["generated_ids"], am)):
            if g != e:
                mm = float(top2.values[k, 0] - top2.values[k, 1])
                mism.append({"step": k, "ref_margin": r["steps"][k]["margin"], "eng_margin": round(mm, 4),
                             "ref_tok": g, "eng_tok": e})
        tf[r["name"]] = {"T": len(ids), "n": n, "agree": n - len(mism), "mismatch": mism, "forward_s": round(dt, 2),
                         "peak_gib": peaks(eng), "ref_min_margin": r["min_margin"]}
        log(f"TF {r['name']} T={len(ids)} top-1 {n - len(mism)}/{n} forward {dt:.2f} s peak {peaks(eng)} "
            f"mismatch {[(m['step'], m['ref_margin']) for m in mism]}")
    res["tf"] = tf
    return res


# ---------------------------------------------------------------------------------------------------------- G3/G4
def margin_fn(tg, tb):
    def f(logits):
        lp = torch.log_softmax(logits[-1].float(), -1)
        return lp[tg] - lp[tb]

    return f


def g3_context(eng, name, ids, tg, rng, n_top=5, n_rand=5):
    T = len(ids)
    lg = eng.forward(ids, logit_pos=[T - 1]).logits[0].float()
    order = torch.argsort(lg, descending=True).tolist()
    tg = order[0] if tg is None else tg
    tb = order[0] if order[0] != tg else order[1]
    reset_peak(eng)
    sync(eng)
    t0 = time.time()
    r = eng.attribution(ids, margin_fn(tg, tb), logit_pos=[T - 1])
    sync(eng)
    t_attr = time.time() - t0
    m0 = r["m"]
    attr = r["attr"]
    flat = attr.flatten()
    top = torch.topk(flat.abs(), n_top).indices.tolist()
    nz = torch.nonzero(flat).flatten().tolist()
    rand = rng.sample([i for i in nz if i not in set(top)], n_rand)
    E, Fd = attr.shape[1], attr.shape[2]
    rows = []
    t1 = time.time()
    for kind, idxs in (("top", top), ("random", rand)):
        for i in idxs:
            L, e, j = i // (E * Fd), (i // Fd) % E, i % Fd
            eng.set_neuron_scale({(L, e, j): 0.0}, where="h")
            lg2 = eng.forward(ids, logit_pos=[T - 1]).logits[0].float()
            eng.clear_neuron_scale()
            lp = torch.log_softmax(lg2, -1)
            m1 = float(lp[tg] - lp[tb])
            a = float(attr[L, e, j])
            rows.append({"kind": kind, "neuron": [L, e, j], "attr": a, "pred_dm": -a, "dm": m1 - m0,
                         "sign_ok": (np.sign(-a) == np.sign(m1 - m0)) if abs(m1 - m0) > 0 else None,
                         "ratio": (m1 - m0) / -a if a != 0 else None})
    t_abl = time.time() - t1
    # the top-5 together
    eng.set_neuron_scale({(i // (E * Fd), (i // Fd) % E, i % Fd): 0.0 for i in top}, where="h")
    lg3 = eng.forward(ids, logit_pos=[T - 1]).logits[0].float()
    eng.clear_neuron_scale()
    lp = torch.log_softmax(lg3, -1)
    joint = {"pred_dm": float(-sum(flat[i] for i in top)), "dm": float(lp[tg] - lp[tb]) - m0}
    out = {"name": name, "T": T, "tg": tg, "tb": tb, "m": m0, "attr_s": round(t_attr, 2), "ablation_s": round(t_abl, 2),
           "peak_attr_gib": peaks(eng), "attr_abs_sum": float(attr.abs().sum()), "attr_sum": float(attr.sum()),
           "neurons": rows, "top5_joint": joint,
           "routed_tokens_per_visit": {int(v): int(c.sum()) for v, c in r["count_visit"].items()}}
    log(f"G3 {name} T={T} m={m0:.4f} (tg {tg}, tb {tb}) attribution {t_attr:.1f} s peak {peaks(eng)}")
    for x in rows:
        log(f"   {x['kind']:6s} L{x['neuron'][0]:02d} e{x['neuron'][1]:03d} j{x['neuron'][2]:04d} attr {x['attr']:+.4e} "
            f"pred {x['pred_dm']:+.4e} dm {x['dm']:+.4e} ratio {x['ratio'] if x['ratio'] is None else round(x['ratio'], 3)}")
    log(f"   top5 together: pred {joint['pred_dm']:+.4e} dm {joint['dm']:+.4e}")
    return out, attr


def g3x(eng, name, ids, tg, rng, eps=0.05, n_top=5, n_rand=5):
    """Like G3, plus: zeroing at frozen routing (same experts as the base forward) and a central difference
    (scale 1+-eps on h, frozen routing) against the attribution: separates the gradient's linearity from routing flips."""
    T = len(ids)
    lg = eng.forward(ids, logit_pos=[T - 1]).logits[0].float()
    order = torch.argsort(lg, descending=True).tolist()
    tg = order[0] if tg is None else tg
    tb = order[0] if order[0] != tg else order[1]
    mf = margin_fn(tg, tb)
    t0 = time.time()
    r = eng.attribution(ids, mf, logit_pos=[T - 1])
    t_attr = time.time() - t0
    routes = eng.capture_routing(ids)
    m0 = r["m"]
    m0f = float(mf(eng.forward(ids, logit_pos=[T - 1], routes=routes).logits))
    attr = r["attr"]
    flat = attr.flatten()
    E, Fd = attr.shape[1], attr.shape[2]
    top = torch.topk(flat.abs(), n_top).indices.tolist()
    nz = torch.nonzero(flat).flatten().tolist()
    rand = rng.sample([i for i in nz if i not in set(top)], n_rand)
    rows = []
    for kind, idxs in (("top", top), ("random", rand)):
        for i in idxs:
            n = (i // (E * Fd), (i // Fd) % E, i % Fd)
            a = float(flat[i])
            res = {"kind": kind, "neuron": list(n), "attr": a}
            eng.set_neuron_scale({n: 0.0})
            res["dm_zero"] = float(mf(eng.forward(ids, logit_pos=[T - 1]).logits)) - m0
            res["dm_zero_frozen"] = float(mf(eng.forward(ids, logit_pos=[T - 1], routes=routes).logits)) - m0f
            ms = []
            for sg in (1, -1):
                eng.set_neuron_scale({n: 1 + sg * eps})
                ms.append(float(mf(eng.forward(ids, logit_pos=[T - 1], routes=routes).logits)))
            eng.clear_neuron_scale()
            res["central_diff"] = (ms[0] - ms[1]) / (2 * eps)
            res["cd_ratio"] = res["central_diff"] / a if a else None
            res["zero_frozen_ratio"] = res["dm_zero_frozen"] / -a if a else None
            res["zero_ratio"] = res["dm_zero"] / -a if a else None
            rows.append(res)
            log(f"   {kind:6s} L{n[0]:02d} e{n[1]:03d} j{n[2]:04d} attr {a:+.3e} | zero {res['dm_zero']:+.3e} "
                f"zero_frozen {res['dm_zero_frozen']:+.3e} | central diff {res['central_diff']:+.3e} "
                f"(ratio {res['cd_ratio'] if res['cd_ratio'] is None else round(res['cd_ratio'], 3)})")
    out = {"name": name, "T": T, "dtype": str(eng.dtype), "tg": tg, "tb": tb, "m": m0, "m_frozen": m0f, "eps": eps,
           "attr_s": round(t_attr, 2), "neurons": rows}
    log(f"G3x {name} T={T} {eng.dtype} m={m0:.4f} m_cong={m0f:.4f}")
    return out


def g4(eng, ids, attr):
    T = len(ids)
    pos = list(range(max(0, T - 64), T))
    base = eng.forward(ids, logit_pos=pos).logits.float()
    res = []
    for lo, hi in ((0, 20), (20, 24)):
        a = attr[lo:hi].abs()
        f = int(a.flatten().argmax())
        E, Fd = a.shape[1], a.shape[2]
        L, e, j = lo + f // (E * Fd), (f // Fd) % E, f % Fd
        r = {"neuron": [L, e, j], "fmt": eng.layers[L].mlp.w13.fmt, "attr": float(attr[L, e, j]), "rows": len(pos)}
        hooks = {}
        for where in ("h", "up"):
            eng.set_neuron_scale({(L, e, j): 0.0}, where=where)
            hooks[where] = eng.forward(ids, logit_pos=pos).logits.float()
            eng.clear_neuron_scale()
        undo = eng.edit_zero_up(L, e, j)
        edit = eng.forward(ids, logit_pos=pos).logits.float()
        undo()
        after = eng.forward(ids, logit_pos=pos).logits.float()
        for where in ("h", "up"):
            r[f"edit_equals_hook_{where}"] = bool(torch.equal(edit, hooks[where]))
            r[f"edit_vs_hook_{where}_maxabs"] = float((edit - hooks[where]).abs().max())
        r["effect_maxabs"] = float((edit - base).abs().max())
        r["undo_restores"] = bool(torch.equal(after, base))
        res.append(r)
        log("G4", json.dumps(r))
    return res


# ---------------------------------------------------------------------------------------------------------- G5
def long_tokens(n):
    ids = []
    while len(ids) < n:
        for p in sorted((Path(REF) / "greedy").glob("*.json"), key=lambda p: p.name != "long1.json"):
            r = json.loads(p.read_text())
            ids += r["prompt_ids"] + r["generated_ids"]
    return ids[:n]


def g5(eng, sizes, reps, a_offload_only=False):
    ids_all = long_tokens(max(sizes))
    res = []

    def run(mode, ids, T, row, tag):
        torch.cuda.empty_cache()
        reset_peak(eng)
        sync(eng)
        t0 = time.time()
        try:
            if mode == "fwd":
                eng.forward(ids, logit_pos=[T - 1])
            else:
                eng.attribution(ids, margin_fn(ids[-1], ids[-2] if ids[-2] != ids[-1] else 0), logit_pos=[T - 1])
            sync(eng)
            row[f"{tag}_s"] = round(time.time() - t0, 2)
            ok = True
        except Exception as ex:  # OOM (expected at the limit) or other: record it and carry on
            row[f"{tag}_s"] = None
            row[f"{tag}_oom"] = f"{type(ex).__name__}: {str(ex).splitlines()[0][:200]}"
            ok = False
        row[f"{tag}_peak_gib"] = peaks(eng)
        row[f"{tag}_reserved_gib"] = reserved(eng)
        torch.cuda.empty_cache()
        log(f"G5 T={T} rep {row['rep']} {tag}: {row.get(f'{tag}_s')} s peak {row[f'{tag}_peak_gib']} "
            f"reserved {row[f'{tag}_reserved_gib']} {row.get(f'{tag}_oom', '')}")
        return ok

    for T in sizes:
        ids = ids_all[:T]
        for rep in range(reps):
            row = {"T": T, "rep": rep}
            run("fwd", ids, T, row, "fwd")
            if a_offload_only:
                eng.offload = True
                run("fwd_bwd", ids, T, row, "fwd_bwd_offload")
                eng.offload = False
            elif not run("fwd_bwd", ids, T, row, "fwd_bwd") or T >= 16384:
                eng.offload = True  # visit outputs in RAM
                run("fwd_bwd", ids, T, row, "fwd_bwd_offload")
                eng.offload = False
            res.append(row)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.environ.get("INKLING_CKPT", "models/inkling-2-mini"))
    p.add_argument("--phases", default="g2")
    p.add_argument("--out", default="out")
    p.add_argument("--split", type=int, default=14)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--sizes", default="2048,8192,16384")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--g3-long", type=int, default=8192, help="0 = short context only")
    p.add_argument("--g3-ntop", type=int, default=5)
    p.add_argument("--g3-nrand", type=int, default=5)
    p.add_argument("--g2-names", default="it1,en1,code1")
    p.add_argument("--tf-names", default="all")
    p.add_argument("--g2-all-rows", action="store_true", help="visit states at all rows (diagnosis)")
    p.add_argument("--cpu", action="store_true", help="dry run on CPU (devices cpu,cpu)")
    p.add_argument("--offload", action="store_true", help="visit outputs saved in RAM in the backward")
    p.add_argument("--head-cpu", action="store_true", help="lm_head on CPU")
    a = p.parse_args()
    phases = a.phases.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    if phases == ["est"]:
        from transformers import Inkling2LoopConfig

        cfg = Inkling2LoopConfig.from_pretrained(a.ckpt).text_config
        # per-card weights from the inventory (GiB): computed from the safetensors header
        from safetensors import safe_open

        st = [0, 0]
        with safe_open(str(next(Path(a.ckpt).glob("model-*.safetensors"))), framework="pt") as f:
            for k in f.keys():
                if not k.startswith("model.llm."):
                    continue
                sl = f.get_slice(k)
                n = int(np.prod(sl.get_shape()))
                dt = sl.get_dtype()
                b = n * (4 if dt in ("F32",) else 2 if dt in ("BF16",) else 1)
                if any(s in k for s in ("mlp.gate.weight", "sconv.weight", "attn_sconv", "mlp_sconv", "forget_weight",
                                        "gain_weight")) and dt == "BF16":
                    b *= 2
                if k.startswith("model.llm.layers."):
                    L = int(k.split(".")[3])
                    st[0 if L < a.split else 1] += b
                elif "unembed" in k:
                    st[1] += b * 200058 / 201024
                elif k == "model.llm.embed.weight":
                    pass  # on CPU
                else:
                    st[0] += b
        static = [x / 2**30 for x in st]
        print("weights (GiB) dev0 %.2f dev1 %.2f" % tuple(static))
        for T in [int(s) for s in a.sizes.split(",")]:
            print(T, json.dumps(estimate(cfg, T, a.split, static)))
        return
    log(f"phases {phases}")
    eng = inkgpu.InkGPU.load(a.ckpt, devices=("cpu", "cpu") if a.cpu else ("cuda:0", "cuda:1"), split=a.split,
                             dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16,
                             head_device="cpu" if a.head_cpu else None)
    eng.offload = a.offload
    if a.cpu:
        static = [0.0, 0.0]
    else:
        for d in eng.devices:
            log(f"{d}: allocated after loading {torch.cuda.memory_allocated(d) / 2**30:.2f} GiB")
        static = [torch.cuda.memory_allocated(d) / 2**30 for d in eng.devices]
    summary = {"load_s": eng.load_s, "static_gib": static, "split": a.split,
               "estimate": {T: estimate(eng.cfg, T, a.split, static) for T in (2048, 8192, 16384)}}
    log("estimate:", json.dumps(summary["estimate"]))
    try:
        if "g2" in phases:
            summary["g2"] = g2(eng, out, a)
            inkgpu.save_json(out / "g2.json", summary)
        attr_long = None
        attr_g4, ids_g4 = None, None
        if "g3" in phases:
            rng = random.Random(0)
            r = json.loads((Path(REF) / "greedy" / "it1.json").read_text())
            ids_s = (r["prompt_ids"] + r["generated_ids"])[: r["n_prompt"]]
            res_s, attr_s = g3_context(eng, "it1-prompt", ids_s, None, rng, a.g3_ntop, a.g3_nrand)
            summary["g3"] = [res_s]
            attr_g4, ids_g4 = attr_s, ids_s
            if a.g3_long:
                longids = long_tokens(a.g3_long + 1)
                res_l, attr_l = g3_context(eng, f"long1-{a.g3_long}", longids[: a.g3_long], longids[a.g3_long], rng,
                                           a.g3_ntop, a.g3_nrand)
                summary["g3"].append(res_l)
                attr_g4, ids_g4 = attr_l, longids[: a.g3_long]
            torch.save({"attr": attr_g4, "ids": ids_g4}, out / "g3_attr.pt")
            inkgpu.save_json(out / "g3.json", summary)
        if "g3x" in phases:
            rng = random.Random(1)
            longids = long_tokens(a.g3_long + 1)
            summary["g3x"] = g3x(eng, f"long1-{a.g3_long}", longids[: a.g3_long], longids[a.g3_long], rng)
            inkgpu.save_json(out / "g3x.json", summary)
        if "g4" in phases:
            if attr_g4 is None:
                d = torch.load(out / "g3_attr.pt")
                attr_g4, ids_g4 = d["attr"], d["ids"]
            summary["g4"] = g4(eng, ids_g4, attr_g4)
            inkgpu.save_json(out / "g4.json", summary)
        if "g5" in phases:
            summary["g5"] = g5(eng, [int(s) for s in a.sizes.split(",")], a.reps, a.offload)
            inkgpu.save_json(out / "g5.json", summary)
    finally:
        inkgpu.save_json(out / f"summary-{'-'.join(phases)}.json", summary)
    log("DONE")


if __name__ == "__main__":
    main()
