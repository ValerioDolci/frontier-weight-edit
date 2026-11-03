import os
st = open("/proc/self/status").read()
print("VmData", st.split("VmData:")[1].split("\n")[0].strip(), "VmRSS", st.split("VmRSS:")[1].split("\n")[0].strip())
t = W("model.language_model.layers.31.post_feedforward_layernorm.weight")
print("W ok:", t.shape, float(t.mean()))
print("edits:", edits())
print(sorted(os.listdir(WORK))[:50])
del t
