for fn in [W, save_edit, drop_edit, edits, load_acts]:
    c = fn.__code__
    print(fn.__name__, "names:", c.co_names, "| consts:", [x for x in c.co_consts if isinstance(x, (str, int, float))][:12])
