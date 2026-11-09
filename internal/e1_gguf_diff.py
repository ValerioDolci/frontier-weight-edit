"""Compare two GGUF files tensor by tensor (sha1 of the quantised data) and their metadata. Usage: e1_gguf_diff.py A.gguf B.gguf"""
import sys, hashlib
sys.path.insert(0, "<server>/llama.cpp/gguf-py")
from gguf import GGUFReader
a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
ta = {t.name: t for t in a.tensors}; tb = {t.name: t for t in b.tensors}
print("tensori", len(ta), len(tb), "| solo in A", sorted(set(ta) - set(tb))[:5], "| solo in B", sorted(set(tb) - set(ta))[:5])
diff = [(n, ta[n].tensor_type.name, tb[n].tensor_type.name) for n in ta if n in tb and
        (ta[n].tensor_type != tb[n].tensor_type or hashlib.sha1(ta[n].data.tobytes()).digest() != hashlib.sha1(tb[n].data.tobytes()).digest())]
print("tensori diversi", len(diff))
for d in diff: print("  ", d)
md = [k for k in a.fields if k in b.fields and b"".join(bytes(p) for p in a.fields[k].parts) != b"".join(bytes(p) for p in b.fields[k].parts)]
print("metadati diversi", md[:12], "| solo in A", sorted(set(a.fields) - set(b.fields))[:5], "| solo in B", sorted(set(b.fields) - set(a.fields))[:5])
