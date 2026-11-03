# build margin-eval item list in the kernel (for analysis bookkeeping); print compact chunks to copy
d = json.load(open(WORK + "/dp.json"))
items = []
for (i, pre, b, a) in d["DP"]:
    sep = "" if pre == "" else " "
    items.append({"id": i, "it": pre + sep + b, "kind": "base"})
    items.append({"id": i, "it": pre + sep + a, "kind": "alt"})
json.dump(items, open(WORK + "/margin_items.json", "w"), ensure_ascii=False)
print(len(items))
s = json.dumps([{"id": x["id"], "it": x["it"]} for x in items], ensure_ascii=False)
print(len(s))
