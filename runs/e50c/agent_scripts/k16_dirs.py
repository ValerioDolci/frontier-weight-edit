import numpy as np, torch, json
r1 = load_acts("r1"); M1, A1 = r1["meta"], r1["acts"]
dec1 = {int(k): v for k, v in json.load(open(f"{WORK}/dec_r1.json")).items()}
pairs = json.load(open(f"{WORK}/pairs_r1.json"))
DIW = [(p["a"], p["drow"]) for p in pairs if p["kind"] == "DI"]
DIR = [(i, dec1[i][1]) for i in range(110, 314) if i in dec1]
isfirst = lambda i, r: r == M1[i]["n_prompt_tokens"] - 1
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
dirs = {}
for L in [16, 22, 25, 28, 31, 34, 37, 40, 44, 47]:
    X = lambda lst: torch.stack([A1[f"resid.{L}"][i][r].float() for i, r in lst]).numpy()
    Wf = X([(i, r) for i, r in DIW if isfirst(i, r)]); Wm = X([(i, r) for i, r in DIW if not isfirst(i, r)])
    Rf = X([(i, r) for i, r in DIR if isfirst(i, r)]); Rm = X([(i, r) for i, r in DIR if not isfirst(i, r)])
    df = Rf.mean(0) - Wf.mean(0); dm = Rm.mean(0) - Wm.mean(0)
    # split-half reliability of dm and df
    rng = np.random.RandomState(0)
    def half(A, B):
        ia = rng.permutation(len(A)); ib = rng.permutation(len(B))
        h1 = A[ia[:len(A)//2]].mean(0) - B[ib[:len(B)//2]].mean(0)
        h2 = A[ia[len(A)//2:]].mean(0) - B[ib[len(B)//2:]].mean(0)
        return cos(h1, h2)
    nrm = np.linalg.norm(np.concatenate([Wm, Rm]), axis=1).mean()
    print(f"L{L}: |df|={np.linalg.norm(df):6.2f} |dm|={np.linalg.norm(dm):6.2f} |resid|~{nrm:6.1f}  cos(df,dm)={cos(df,dm):.3f}  split-half cos: first {half(Rf,Wf):.2f} mid {half(Rm,Wm):.2f}")
    dirs[L] = (df, dm)
np.save(f"{WORK}/dirs_first_mid.npy", {L: (df, dm) for L, (df, dm) in dirs.items()}, allow_pickle=True)
# cross-layer persistence of dm direction
Ls = sorted(dirs)
print("cos of dm across layers (row L vs col L'):")
for L in Ls:
    print(f"  L{L}:", " ".join(f"{cos(dirs[L][1], dirs[L2][1]):5.2f}" for L2 in Ls))
