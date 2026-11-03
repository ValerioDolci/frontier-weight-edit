import numpy as np, torch, json
r1 = load_acts("r1")
meta, items, acts = r1["meta"], r1["items"], r1["acts"]
LAYERS = sorted(int(k.split(".")[1]) for k in acts)
pairs = json.load(open(f"{WORK}/pairs_r1.json"))
dec = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r1.json")).items()}

# heuristic decision rows for all DD model outputs (314..443)
AUX_AV = {"Ave", "▁aveva", "▁avevano", "▁avevamo", "▁avevo", "▁avevi", "▁ave", "▁avev"}
AUX_ES = {"E", "Era", "Er", "▁era", "▁erano", "▁eravamo", "▁ero", "▁eri", "▁er", "Eran"}
def dd_dec(i):
    toks = meta[i]["tokens"]
    for t, tk in enumerate(toks):
        if tk in AUX_AV or tk in AUX_ES:
            if tk in AUX_ES and t + 1 < len(toks) and (toks[t+1].startswith("▁stat") or toks[t+1].startswith("▁rim")):
                return t + 1
            # E + ra / Ave + va split tokens: decision at the first piece
            return t
    return None
DDH = []
for i in range(314, 444):
    t = dd_dec(i)
    if t is not None:
        DDH.append((i, meta[i]["n_prompt_tokens"] + t - 1))
print("DD heuristic decisions:", len(DDH), "of 130")
for i, r in DDH[:12]:
    t = r - meta[i]["n_prompt_tokens"] + 1
    print("  ", items[i]["id"], repr(meta[i]["tokens"][t]), " | ", items[i]["it"][:60])

DIW = [(i, dec[i][1]) for i in range(0, 55)]
DIR = [(i, dec[i][1]) for i in range(110, 314) if i in dec]

def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

def stack(L, lst):
    return torch.stack([acts[f"resid.{L}"][i][r].float() for (i, r) in lst]).numpy()

# other positions: all completion positions of PERF/DURP (485..604) + non-decision completion positions of DI/DD items
def other_rows(L, idxs, exclude=None, maxper=40):
    out = []
    for i in idxs:
        n0 = meta[i]["n_prompt_tokens"]; n1 = n0 + len(meta[i]["tokens"])
        rows = list(range(n0 - 1, n1 - 1))
        if exclude is not None and i in exclude:
            rows = [r for r in rows if r != exclude[i]]
        out.append(acts[f"resid.{L}"][i][rows].float())
    return torch.cat(out).numpy()

excl = {i: r for i, r in DIW + DIR + DDH}
for L in [22, 25, 28, 31, 34, 37]:
    DI = np.concatenate([stack(L, DIW), stack(L, DIR)])
    DD = stack(L, DDH)
    OTH = other_rows(L, list(range(485, 605)))
    NDI = other_rows(L, [i for i, _ in DIW + DIR], exclude=excl)
    # normalized
    def nz(X):
        return X / np.sqrt((X ** 2).mean(1, keepdims=True))
    DIn, DDn, OTn, NDn = nz(DI), nz(DD), nz(OTH), nz(NDI)
    # constant-direction check on all 'other' + non-decision positions
    ALLn = np.concatenate([OTn, NDn])
    mu = ALLn.mean(0); umu = mu / np.linalg.norm(mu)
    proj = ALLn @ umu
    # LDA DI vs (DD + others), shrinkage, CV halves
    rng = np.random.RandomState(1)
    def split(X):
        p = rng.permutation(len(X)); return X[p[: len(X)//2]], X[p[len(X)//2:]]
    DI1, DI2 = split(DIn); DD1, DD2 = split(DDn); OT1, OT2 = split(OTn); ND1, ND2 = split(NDn)
    NEG1 = np.concatenate([DD1, OT1[:2000], ND1[:2000]])
    d = DI1.mean(0) - NEG1.mean(0)
    Xc = np.concatenate([DI1 - DI1.mean(0), NEG1 - NEG1.mean(0)])
    S = Xc.T @ Xc / len(Xc); lam = np.trace(S) / S.shape[0] * 0.3
    w = np.linalg.solve(S + lam * np.eye(S.shape[0]), d)
    sDI, sDD, sOT, sND = DI2 @ w, DD2 @ w, OT2 @ w, ND2 @ w
    thr = np.percentile(sDI, 10)
    print(f"L{L}: const-dir proj mean={proj.mean():.2f} std={proj.std():.3f} min={proj.min():.2f} | "
          f"LDA test AUC DI-vs-DD={auc(sDI,sDD):.3f} DI-vs-OTH={auc(sDI,sOT):.3f} DI-vs-nonDecDI={auc(sDI,sND):.3f}; "
          f"frac above DI-p10: DD={np.mean(sDD>thr):.3f} OTH={np.mean(sOT>thr):.4f} NDI={np.mean(sND>thr):.4f}")
    # also DIW vs DD specifically
    print(f"      DIW-vs-DD (LDA score) AUC={auc(stack(L,DIW).__array__() @ w / 1, DD @ w):.3f} (raw, not normalized)")
