import os
for root, dirs, files in os.walk(WORK):
    for f in files:
        p = os.path.join(root, f)
        print(round(os.path.getsize(p) / 1e6, 1), "MB", os.path.relpath(p, WORK))
print("MEM_GB", save_edit.__globals__.get("_MEM_GB"))
