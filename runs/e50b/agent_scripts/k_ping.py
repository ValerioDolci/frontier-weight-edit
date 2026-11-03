print("alive", sorted(k for k in globals().keys() if not k.startswith("_"))[:20])
print(edits())
