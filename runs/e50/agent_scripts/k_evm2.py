import pickle, json
MB = pickle.load(open(f"{WORK}/margin_orig.pkl", "rb"))
name = "margin_v1"
ME = margins(name)
summarize(ME, MB)
pickle.dump(ME, open(f"{WORK}/{name}.pkl", "wb"))
L2 = json.load(open(f"{WORK}/dp_lab2.json")); pt = dict(zip(L2["ids"], L2["pt"]))
d = ME["cm"] - MB["cm"]
for t in ["start", "subj", "aux"]:
    m = (ME["cat"] == "DI") & ME["err"] & np.array([pt[i] == t for i in ME["id"]])
    print(t, "DI-err n", m.sum(), "mean d %.2f" % d[m].mean(), "now correct", (ME["cm"][m] > 0).sum())
m = (ME["cat"] == "DI") & ME["err"]
print("still wrong:", [(int(i), pt[int(i)], round(float(a), 1), round(float(b), 1)) for i, a, b in zip(ME["id"][m], MB["cm"][m], ME["cm"][m]) if b <= 0])
m = (ME["cat"] == "DI") & ~ME["err"]
print("DI-ok now wrong:", [(int(i), round(float(a), 1), round(float(b), 1)) for i, a, b in zip(ME["id"][m], MB["cm"][m], ME["cm"][m]) if b <= 0])
dr = ME["rest_lp"] - MB["rest_lp"]
print("worst rest items:", sorted([(round(float(x), 3), str(k)) for x, k in zip(dr, ME["rest_kind"])])[:6])
