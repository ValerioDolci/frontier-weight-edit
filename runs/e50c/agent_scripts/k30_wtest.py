import torch
torch.set_num_threads(8)
try:
    t = W("model.language_model.layers.31.post_feedforward_layernorm.weight")
    print("W ok", t.shape)
except Exception as ex:
    print("W failed:", str(ex)[:200])
print(type(SESSION), [a for a in dir(SESSION) if not a.startswith("__")][:50])
