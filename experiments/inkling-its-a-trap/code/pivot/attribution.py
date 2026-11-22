#!/usr/bin/env python3
"""Step 3 (GPU): Inkling's "apply / hesitate" fork at the decision point of the reasoning.

Phases (--phase):
  dist  token distribution at the point, on N DIFF points of the selection half (s01-s05): top-30 -> dist.json
  attr  margin m = logsumexp(logP[GO]) - logsumexp(logP[HESITATE]) and attribution sum h * dm/dh over all expert neurons,
        for every point (resumable); sums per (category, half) -> attr_sum.pt, margins -> attr_items.jsonl
  abl   the top K pro-hesitation neurons (most negative attribution on the DIFF selection) zeroed one at a time on the
        verification half, plus K random control neurons in the same experts -> abl.json
  loc   locality curve: cumulative zeroing of the top k (1, 4, 8, 16, 32) and a random control -> loc.json
Halves: selection = situations s01-s05, verification = s06-s10. Unit = (layer, expert, neuron j), zeroed at every visit.
Usage: python attribution.py --phase dist|attr|abl|loc [--pivots PIVOTS_TOK.jsonl] [--out DIR] [--ckpt DIR] [--go Now,So] [--cats DIFF,THOR]
  --ckpt defaults to $INKLING_CKPT or models/inkling-2-mini; the GPU runtime is code/inkgpu/inkgpu.py.
"""
import argparse, json, os, random, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(os.path.dirname(HERE))   # experiment root
sys.path.insert(0, os.path.join(HERE, "..", "inkgpu"))
import inkgpu  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True)
ap.add_argument("--pivots", default=os.path.join(EXP, "results", "pivots", "pivot_points_tok.jsonl"))
ap.add_argument("--out", default=os.path.join(EXP, "results", "pivots", "attr_out"))
ap.add_argument("--ckpt", default=os.environ.get("INKLING_CKPT", "models/inkling-2-mini"))
ap.add_argument("--go", default="")
ap.add_argument("--hes", default="But,Also,Wait,However,Hmm,Actually,BUT")
ap.add_argument("--amb", default="Let", help="ambiguous tokens excluded from both sides when --go COMPLEMENT")
ap.add_argument("--cats", default="DIFF,THOR")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--skip", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--only-diff", action="store_true")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
SEL = {f"s0{i}" for i in range(1, 6)}
fold = lambda x: "sel" if x["sid"][:3] in SEL else "ver"  # noqa: E731
P = [json.loads(l) for l in open(A.pivots)]
P = [x for x in P if x["cat"] in A.cats.split(",")]
log = open(os.path.join(A.out, f"{A.phase}.log"), "a")


def say(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(A.ckpt)


def ids_of(words):
    out = []
    for w in filter(None, words.split(",")):
        t = tok(w, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            say(f"WARNING: \"{w}\" is not a single token: {t}, using the first")
        out.append(t[0])
    return out


t0 = time.time()
m = inkgpu.InkGPU.load(A.ckpt)
say(f"model loaded in {time.time() - t0:.0f} s; {len(P)} points")
HES = ids_of(A.hes)
COMPL = A.go == "COMPLEMENT"     # GO = everything that does not open a hesitation
GO = [] if COMPL else ids_of(A.go)
AMB = ids_of(A.amb) if COMPL else []


def margin_fn(lg):
    lp = torch.log_softmax(lg[-1].float(), -1)
    lh = torch.logsumexp(lp[HES], 0)
    if COMPL:   # logsumexp over the complement (exact even when P(hesitate) ~ 1, no 1 - sum)
        mask = torch.ones(lp.shape[-1], dtype=torch.bool, device=lp.device)
        mask[HES + AMB] = False
        return torch.logsumexp(lp[mask], 0) - lh
    return torch.logsumexp(lp[GO], 0) - lh


def margin_now(x):
    ids = x["prefix_ids"]
    lg = m.forward(ids, logit_pos=[len(ids) - 1]).logits
    return float(margin_fn(lg))


if A.phase == "dist":
    pts = [x for x in P if x["cat"] == "DIFF" and fold(x) == "sel"]
    random.Random(A.seed).shuffle(pts)
    res = []
    for x in pts[:A.n]:
        ids = x["prefix_ids"]
        t1 = time.time()
        lp = torch.log_softmax(m.forward(ids, logit_pos=[len(ids) - 1]).logits[-1].float(), -1)
        top = torch.topk(lp, 30)
        row = {"id": x["id"], "next": x["next_text"][:50],
               "top": [(tok.decode([int(i)]), round(float(v.exp()), 4)) for v, i in zip(top.values, top.indices)],
               "p_hes": round(float(lp[HES].exp().sum()), 4), "s": round(time.time() - t1, 1)}
        res.append(row)
        say(row["id"], row["s"], "s", "P(hesitate)", row["p_hes"], row["top"][:10])
    json.dump(res, open(os.path.join(A.out, "dist.json"), "w"), ensure_ascii=False, indent=1)

elif A.phase == "attr":
    assert GO or COMPL, "--go is required"
    itf = os.path.join(A.out, "attr_items.jsonl")
    sf = os.path.join(A.out, "attr_sum.pt")
    done = {json.loads(l)["id"] for l in open(itf)} if os.path.exists(itf) else set()
    S = torch.load(sf) if os.path.exists(sf) else {}
    for n, x in enumerate(P, 1):
        if x["id"] in done:
            continue
        ids = x["prefix_ids"]
        t1 = time.time()
        r = m.attribution(ids, margin_fn, logit_pos=[len(ids) - 1], positions=range(A.skip, len(ids)))
        key = f"{x['cat']}_{fold(x)}"
        a = r["attr"].float().cpu()
        if key not in S:
            S[key] = {"sum": torch.zeros_like(a), "n": 0}
        S[key]["sum"] += a
        S[key]["n"] += 1
        torch.save(S, sf + ".tmp")
        os.replace(sf + ".tmp", sf)
        with open(itf, "a") as f:
            f.write(json.dumps({"id": x["id"], "cat": x["cat"], "fold": fold(x), "m": float(r["m"]),
                                "T": len(ids), "s": round(time.time() - t1, 1)}) + "\n")
        say(f"[{n}/{len(P)}] {x['id']} {key} m={float(r['m']):.3f} T={len(ids)} {time.time() - t1:.1f} s")

elif A.phase in ("abl", "loc"):
    assert GO or COMPL, "--go is required"
    S = torch.load(os.path.join(A.out, "attr_sum.pt"))
    sel = S["DIFF_sel"]["sum"] / S["DIFF_sel"]["n"]
    order = torch.argsort(sel.flatten())                     # most negative = pushes towards hesitation
    unit = lambda i: (int(i) // (128 * 1152), int(i) // 1152 % 128, int(i) % 1152)  # noqa: E731
    cand = [unit(i) for i in order[:max(A.k, 32)]]
    rng = random.Random(A.seed)
    ctrl = [(l, e, rng.randrange(1152)) for (l, e, _) in cand]
    ver = [x for x in P if fold(x) == "ver" and x["cat"] == "DIFF"] + [x for x in P if fold(x) == "ver" and x["cat"] != "DIFF"]
    if A.only_diff:
        ver = [x for x in ver if x["cat"] == "DIFF"]
    if A.phase == "abl":
        confs = [("u", i, {u: 0.0}) for i, u in enumerate(cand[:A.k])] + [("c", i, {u: 0.0}) for i, u in enumerate(ctrl[:A.k])]
    else:
        KS = (1, 4, 8, 16, 32)
        confs = [("u", k, {u: 0.0 for u in cand[:k]}) for k in KS] + [("c", k, {u: 0.0 for u in ctrl[:k]}) for k in KS]
    # points outside, configurations inside: routing frozen from the base forward (removes routing flips from the ablations)
    base, res = {}, {c[:2]: {} for c in confs}
    for n, x in enumerate(ver, 1):
        ids = x["prefix_ids"]
        t1 = time.time()
        routes = m.capture_routing(ids)
        mb = lambda: float(margin_fn(m.forward(ids, logit_pos=[len(ids) - 1], routes=routes).logits))  # noqa: E731
        base[x["id"]] = mb()
        for kind, key, scale in confs:
            m.set_neuron_scale(scale)
            res[(kind, key)][x["id"]] = mb()
            m.clear_neuron_scale()
        del routes
        say(f"[{n}/{len(ver)}] {x['id']} {x['cat']} base {base[x['id']]:.3f} {time.time() - t1:.0f} s")

    def summ(d):
        out = {}
        for c in sorted({x["cat"] for x in ver}):
            xs = [x for x in ver if x["cat"] == c]
            out[c] = {"dm": round(sum(d[x["id"]] - base[x["id"]] for x in xs) / len(xs), 3),
                      "go_wins": sum(d[x["id"]] > 0 for x in xs), "base_go_wins": sum(base[x["id"]] > 0 for x in xs),
                      "n": len(xs)}
        return out

    if A.phase == "abl":
        rows = [{"unit": u, "attr_sel": float(sel[u]), "abl": summ(res[("u", i)]), "ctrl_unit": ctrl[i],
                 "ctrl": summ(res[("c", i)])} for i, u in enumerate(cand[:A.k])]
        for r in rows:
            say(r["unit"], f"attr {r['attr_sel']:.4f}", r["abl"], "| ctrl", r["ctrl"])
        json.dump({"base": base, "rows": rows}, open(os.path.join(A.out, "abl.json"), "w"), indent=1)
    else:
        rows = [{"k": k, "top": summ(res[("u", k)]), "ctrl": summ(res[("c", k)])} for k in KS]
        for r in rows:
            say(f"k={r['k']}", r["top"], "| ctrl", r["ctrl"])
        json.dump({"cand": cand, "ctrl": ctrl, "base": base, "curve": rows}, open(os.path.join(A.out, "loc.json"), "w"), indent=1)
say("DONE")
