import pickle
MB = pickle.load(open(f"{WORK}/margin_orig.pkl", "rb"))
name = "margin_e1"
ME = margins(name)
summarize(ME, MB)
pickle.dump(ME, open(f"{WORK}/{name}.pkl", "wb"))
d = ME["cm"] - MB["cm"]
m = (ME["cat"] == "DI") & ME["err"]
print("DI-err per item (id, base cm, new cm):", [(int(i), round(a, 1), round(b, 1)) for i, a, b in zip(ME["id"][m], MB["cm"][m], ME["cm"][m])])
m = (ME["cat"] == "DD")
print("DD items with d(cm) < -1:", [(int(i), round(a, 1), round(b, 1)) for i, a, b, dd in zip(ME["id"][m], MB["cm"][m], ME["cm"][m], d[m]) if dd < -1])
