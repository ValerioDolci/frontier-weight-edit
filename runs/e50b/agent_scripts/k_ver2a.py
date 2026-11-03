import torch, json, numpy as np, gc, time
t0 = time.time()
A0 = load_acts("L47a")
json.dump([m["logprobs"] for m in A0["meta"]], open(WORK + "/L47a_logprobs.json", "w"))
json.dump([m["tokens"] for m in A0["meta"]], open(WORK + "/L47a_tokens.json", "w"))
json.dump(A0["items"], open(WORK + "/L47a_items.json", "w"))
del A0; gc.collect()
print("saved original logprobs", time.time() - t0)
def rmsn(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
wpa = W("model.language_model.layers.47.post_attention_layernorm.weight")
wpre = W("model.language_model.layers.47.pre_feedforward_layernorm.weight")
A1 = load_acts("ver2"); acts1, meta1, items = A1["acts"], A1["meta"], A1["items"]
json.dump([m["logprobs"] for m in meta1], open(WORK + "/ver2_logprobs.json", "w"))
XH1, Y1, R1 = [], [], []
for k in range(len(items)):
    m = meta1[k]; npt = m["n_prompt_tokens"]; n = len(m["tokens"]); sl = slice(npt - 1, npt - 1 + n)
    h = acts1["resid.46"][k][sl].float() + rmsn(acts1["attn_out.47"][k][sl].float(), wpa)
    XH1.append(rmsn(h, wpre).half()); Y1.append(acts1["mlp_out.47"][k][sl].clone()); R1.append(acts1["resid.47"][k][sl].clone())
torch.save({"XH1": torch.cat(XH1), "Y1": torch.cat(Y1), "R1": torch.cat(R1)}, WORK + "/ver2pos.pt")
del A1, acts1, meta1, XH1, Y1, R1; gc.collect()
print("saved ver2pos", time.time() - t0)
