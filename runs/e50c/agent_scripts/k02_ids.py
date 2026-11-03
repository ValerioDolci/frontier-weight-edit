import json
from collections import defaultdict
bycat = defaultdict(list)
for p in practice:
    bycat[p["cat"]].append(p["id"])
for c, v in bycat.items():
    print(c, len(v))
    print(json.dumps(v))
