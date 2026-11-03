import gc, json, os
# free large leftovers from previous runs (persistent namespace)
keep = {"names", "W", "save_edit", "drop_edit", "edits", "load_acts", "practice", "SESSION", "WORK", "torch", "np", "json", "os", "sys", "gc"}
for k in list(globals().keys()):
    if k.startswith("_") or k in keep:
        continue
    try:
        del globals()[k]
    except Exception:
        pass
gc.collect()
print("namespace now:", sorted(k for k in globals().keys() if not k.startswith("_")))
print("edits:", edits())
print("WORK files:", sorted(os.listdir(WORK))[:60])
