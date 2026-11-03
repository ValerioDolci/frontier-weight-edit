import re, os
_x = open(os.path.join(WORK, "malloc_info.xml")).read()
print(len(_x))
print(_x[:1500])
_heaps = re.findall(r'<heap nr="(\d+)">(.*?)</heap>', _x, re.S)
_rows = []
for nr, body in _heaps:
    cur = re.search(r'<system type="current" size="(\d+)"', body)
    tots = re.findall(r'<total type="(\w+)" count="(\d+)" size="(\d+)"', body)
    free = sum(int(s) for t, c, s in tots)
    sub = re.search(r'<aspace type="subheaps" size="(\d+)"', body)
    _rows.append((int(nr), int(cur.group(1)) if cur else 0, free, int(sub.group(1)) if sub else -1))
_tot_cur = sum(r[1] for r in _rows); _tot_free = sum(r[2] for r in _rows)
print("arenas:", len(_rows), "system current GB:", round(_tot_cur / 1e9, 2), "free GB:", round(_tot_free / 1e9, 2))
for r in sorted(_rows, key=lambda r: -r[1])[:10]:
    print(f"arena {r[0]}: current {r[1]/1e9:.2f} GB, free {r[2]/1e9:.2f} GB, in-use {(r[1]-r[2])/1e6:.1f} MB, subheaps {r[3]}")
print(_x[-700:])
