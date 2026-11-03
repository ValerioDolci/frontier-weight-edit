_g = save_edit.__globals__
print("_forbidden:", _g.get("_forbidden"))
print("_tf:", _g.get("_tf"))
print("STATE_F:", _g.get("STATE_F"), "state:", _g.get("state"))
print("EDITABLE type:", type(_g.get("EDITABLE")), len(_g.get("EDITABLE") or []))
