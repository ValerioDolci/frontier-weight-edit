import collections
_agg = collections.Counter(); _cnt = collections.Counter(); _big = []
for line in open("/proc/self/maps"):
    parts = line.split()
    a, b = [int(x, 16) for x in parts[0].split("-")]
    perms = parts[1]; path = parts[5] if len(parts) > 5 else "[anon]"
    size = b - a
    key = (perms, "file" if path.startswith("/") else path)
    _agg[key] += size; _cnt[key] += 1
    if size > 500e6: _big.append((round(size / 1e9, 2), perms, path[-60:]))
for k, v in _agg.most_common(10):
    print(k, round(v / 1e9, 2), "GB in", _cnt[k], "mappings")
print("big mappings:", _big[:20])
