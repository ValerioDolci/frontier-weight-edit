rest = [p["id"] for p in practice if p["cat"] in ("HAB", "COND") and p["id"] not in (32, 40)]
print(len(rest))
print(json.dumps(rest))
