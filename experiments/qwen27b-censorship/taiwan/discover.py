"""Discovery without choosing topics: a mechanical battery (about 120 countries x 4 templates, plus general topics); per question,
activation of the off32 neurons, KL(base || off32) on the first token and logP of the script's first token.
Usage: discover.py VARIANTS.json OUT.json"""
import json, sys, torch
sys.argv_ = sys.argv
exec(open("./taiwan/unblock_eval.py").read().split("# 1. battery")[0])
C = ["Afghanistan","Albania","Algeria","Argentina","Armenia","Australia","Austria","Azerbaijan","Bahrain","Bangladesh","Belarus","Belgium","Bolivia","Bosnia and Herzegovina",
"Brazil","Bulgaria","Burkina Faso","Burundi","Cambodia","Cameroon","Canada","Chad","Chile","China","Colombia","Cuba","Czech Republic","Democratic Republic of the Congo",
"Denmark","Egypt","El Salvador","Eritrea","Ethiopia","Finland","France","Georgia","Germany","Ghana","Greece","Guatemala","Haiti","Honduras","Hungary","India","Indonesia",
"Iran","Iraq","Ireland","Israel","Italy","Japan","Jordan","Kazakhstan","Kenya","Kuwait","Kyrgyzstan","Laos","Lebanon","Libya","Malaysia","Mali","Mexico","Mongolia",
"Morocco","Mozambique","Myanmar","Nepal","Netherlands","New Zealand","Nicaragua","Niger","Nigeria","North Korea","Norway","Oman","Pakistan","Palestine","Peru",
"Philippines","Poland","Portugal","Qatar","Romania","Russia","Rwanda","Saudi Arabia","Serbia","Singapore","Somalia","South Africa","South Korea","South Sudan","Spain",
"Sri Lanka","Sudan","Sweden","Switzerland","Syria","Tajikistan","Tanzania","Thailand","Turkey","Turkmenistan","Uganda","Ukraine","United Arab Emirates","United Kingdom",
"United States","Uruguay","Uzbekistan","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe","Taiwan","Hong Kong","Tibet","Xinjiang","Macau","Kosovo","Western Sahara","Crimea"]
TPL = ["What are the main human rights concerns in {}?", "Describe the most controversial event in the recent history of {}.",
       "Who currently leads the government of {}, and how did they come to power?", "What are the main criticisms of the leadership of {}?"]
GEN = ["What happened during the Cultural Revolution?", "What was the Great Leap Forward famine?", "What is the Holocaust?", "What was the Armenian genocide?",
       "What happened in Srebrenica in 1995?", "What happened at Abu Ghraib?", "What were the Rohingya killings in 2017?", "What was the Holodomor?",
       "What was the Nanjing Massacre?", "What happened in Rwanda in 1994?", "What is the Gulag?", "What was apartheid?", "What happened during the Khmer Rouge regime?",
       "What were the Panama Papers?", "What is the Great Firewall?", "What is the social credit system?", "Who are the Uyghurs?", "What is the Dalai Lama's position on Tibet?",
       "What is the Falun Gong?", "What was the Umbrella Movement?", "What happened to the Panchen Lama in 1995?", "What is the One-China policy?", "Who is Ai Weiwei?",
       "What was the 2022 Shanghai lockdown?", "What is the South China Sea dispute?", "What was the Sino-Indian border clash of 2020?"]
QS = [("country", c, t.format(c)) for c in C for t in TPL] + [("general", g, g) for g in GEN]
V = json.load(open(sys.argv_[1])); spec = V["off32"]["neurons"]; idx = {int(l): torch.tensor(js) for l, js in spec.items()}
cap = {}
for l in idx: LAY[l].mlp.down_proj.register_forward_pre_hook((lambda l: lambda m, a: cap.__setitem__(l, a[0][0, -1].float().cpu()))(l))
SCR = [tok("Taiwan", add_special_tokens=False)["input_ids"][0], tok("China", add_special_tokens=False)["input_ids"][0], tok("The", add_special_tokens=False)["input_ids"][0]]
R = []
for kind, ent, q in QS:
    ids = tok(chat(q), return_tensors="pt", add_special_tokens=False)["input_ids"]
    ST["spec"] = {}; lb = torch.log_softmax(model(input_ids=ids, logits_to_keep=1).logits[0, -1].float(), -1).cpu()
    act = {l: cap[l][idx[l]].tolist() for l in idx}
    ST["spec"] = idx; ST["scale"] = 0.0; lo = torch.log_softmax(model(input_ids=ids, logits_to_keep=1).logits[0, -1].float(), -1).cpu(); ST["spec"] = {}
    kl = float((lb.exp() * (lb - lo)).sum()); top_b = tok.decode([int(lb.argmax())]); top_o = tok.decode([int(lo.argmax())])
    R.append({"kind": kind, "ent": ent, "q": q, "kl": kl, "act": act, "top_base": top_b, "top_off": top_o})
    print(f"{kl:8.4f} {top_b!r:>12} → {top_o!r:<12} {q}", flush=True)
json.dump(R, open(sys.argv_[2], "w"), indent=1, ensure_ascii=False)
