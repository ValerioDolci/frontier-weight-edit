A = load_acts("test1")
print(type(A))
if isinstance(A, dict):
    for k, v in A.items():
        if hasattr(v, "shape"):
            print(k, type(v), tuple(v.shape), getattr(v, "dtype", None))
        elif isinstance(v, (list, tuple)):
            print(k, type(v), len(v), type(v[0]) if len(v) else None)
            if len(v) and hasattr(v[0], "shape"):
                print("   first shape", tuple(v[0].shape))
            elif len(v):
                print("   first", repr(v[0])[:300])
        else:
            print(k, repr(v)[:300])
else:
    print(repr(A)[:1000])
