import torch, json, numpy as np
A = load_acts("main1")
acts, meta, items = A["acts"], A["meta"], A["items"]
rows = json.load(open(WORK + "/dec_rows.json"))
cat = {p["id"]: p["cat"] for p in practice}
dec = {r["kb"]: r for r in rows if r["t"] >= 1}
watch_dd = {18, 141, 193, 262, 341, 235, 66, 308, 210, 539}
watch_di = {69, 153, 173, 291, 325, 369, 389, 447, 448, 504, 530, 536, 561, 571, 171}
for L in [20, 24, 28, 32, 36, 40, 44, 47]:
    Xs, ys, ws, grp, ids, isdec = [], [], [], [], [], []
    for kb in range(610):
        R = acts[f"resid.{L}"][kb].float()
        R = R / R.pow(2).mean(1, keepdim=True).sqrt()
        i = items[kb]["id"]; c = cat[i]
        dpos = dec[kb]["t"] - 1 if kb in dec else -1
        for p in range(R.shape[0]):
            if p == dpos:
                y = 1.0 if c == "DI" else -0.5; w = 10.0
            else:
                y = 0.0; w = 1.0
            Xs.append(R[p]); ys.append(y); ws.append(w); grp.append(kb); ids.append(i); isdec.append(p == dpos)
    X = torch.stack(Xs).numpy(); y = np.array(ys); w = np.array(ws); g = np.array(grp); ids = np.array(ids); isdec = np.array(isdec)
    uniq = np.unique(g); rng = np.random.RandomState(0); rng.shuffle(uniq); folds = np.array_split(uniq, 5)
    pred = np.zeros(len(y))
    for f in folds:
        te = np.isin(g, f); tr = ~te
        sw = np.sqrt(w[tr]); Xw = X[tr] * sw[:, None]; yw = y[tr] * sw
        wv = np.linalg.solve(Xw.T @ Xw + 100.0 * np.eye(X.shape[1]), Xw.T @ yw)
        pred[te] = X[te] @ wv
    dsel = isdec
    di = np.array([cat[i] == "DI" for i in ids])
    s_di = pred[dsel & di]; s_dd = pred[dsel & ~di]
    # threshold at which 4 DD fire vs DI-err recall
    err_ids = {r["id"] for r in rows if r["cat"] == "DI" and not r["ok"]}
    s_err = np.array([pred[j] for j in np.where(dsel)[0] if ids[j] in err_ids])
    for th in [0.3, 0.4, 0.5]:
        print(f"L{L} th{th}: DIerr fire {(s_err > th).sum()}/{len(s_err)}  DD fire {(s_dd > th).sum()}/{len(s_dd)}  other fire {((pred > th) & ~dsel).sum()}")
    print("   watch DD:", {int(i): round(float(pred[j]), 2) for j, i in enumerate(ids) if dsel[j] and i in watch_dd})
    print("   watch DI:", {int(i): round(float(pred[j]), 2) for j, i in enumerate(ids) if dsel[j] and i in watch_di})
