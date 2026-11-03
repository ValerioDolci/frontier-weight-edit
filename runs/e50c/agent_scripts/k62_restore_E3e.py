CFG_E3e = build_from_saved("E3b", {}, "E3e", kappa={"Pf": 3.0, "Pm": 3.0})
print(json.dumps(CFG_E3e, indent=None)[:1500])
print("edits:", edits())
print("VmData", vmdata())
