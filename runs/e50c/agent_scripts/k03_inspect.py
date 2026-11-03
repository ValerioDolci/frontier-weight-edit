r1 = load_acts("r1")
print(type(r1))
if isinstance(r1, dict):
    for k, v in r1.items():
        if hasattr(v, "shape"):
            print(k, type(v), v.shape, v.dtype)
        elif isinstance(v, list):
            print(k, "list", len(v), type(v[0]) if v else None)
            if v and isinstance(v[0], dict):
                print("  keys:", list(v[0].keys()))
                print("  first:", {kk: (vv if not hasattr(vv, 'shape') else vv.shape) for kk, vv in v[0].items()})
            elif v and hasattr(v[0], "shape"):
                print("  first shape:", v[0].shape, v[0].dtype)
            else:
                print("  first:", v[0] if v else None)
        else:
            print(k, type(v), v if not isinstance(v, (bytes,)) else len(v))
elif isinstance(r1, (list, tuple)):
    print(len(r1), type(r1[0]))
else:
    print(dir(r1))
