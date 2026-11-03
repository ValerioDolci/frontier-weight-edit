import numpy as np, json
# describe the active (final) edit: neurons, push norms, whether W() returns edited tensors
b2 = json.load(open(f"{WORK}/build2.json"))
print("build2", {k: b2[k] for k in ("n30", "n40", "alpha40")})
w30f = np.load(f"{WORK}/w30_final.npy"); w40f = np.load(f"{WORK}/w40_final.npy")
for L, nn, wf in [(30, [13770, 12872], w30f), (40, b2["n40"], w40f)]:
    G = W(f"model.language_model.layers.{L}.mlp.gate_proj.weight").numpy()
    D = W(f"model.language_model.layers.{L}.mlp.down_proj.weight").numpy()
    jA, jB = nn
    rel = np.linalg.norm(G[jA] - 40.0 * wf) / np.linalg.norm(40.0 * wf)
    vA = D[:, jA] * 4.0
    print(f"L{L} neurons {nn}: W() gate row vs 40*w_final rel.diff {rel:.2e} (0 => W returns edited) | |v| {np.linalg.norm(vA):.2f} | |w| {np.linalg.norm(wf):.3f}")
print(edits())
