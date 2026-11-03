A = load_acts("dp_last")
print(type(A))
if isinstance(A, dict):
    for k, v in A.items():
        if hasattr(v, "shape"):
            print(k, type(v), tuple(v.shape), getattr(v, "dtype", None))
        elif isinstance(v, (list, tuple)):
            print(k, type(v), len(v), repr(v[:3])[:500])
        else:
            print(k, type(v), repr(v)[:300])
