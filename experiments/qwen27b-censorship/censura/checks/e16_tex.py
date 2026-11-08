"""Generates the E16 LaTeX snippets for the report (numbers and table) from the bf16 verdicts, with Q4 as a secondary column.
Usage: e16_tex.py RES_DIR REPORT_DIR"""
import json, sys
RD, OUT = sys.argv[1], sys.argv[2]; CENS = {'official_script', 'denial', 'sovereignty_claim', 'praise', 'refusal'}
S = json.load(open(f"{RD}/e16_bf16_summary.json")); Q = json.load(open(f"{RD}/e16_summary.json"))
rows = S["rows"]; q4 = {r["topic"]: r for r in Q["rows"]}; sm = S["summary"]
cand = [r for r in rows if r["source"] != "CTRL"]; cens = [r for r in cand if r["base"] in CENS]
unl = [r for r in cens if r["off32"] not in CENS]; unlr = [r for r in cens if r["rand32"] not in CENS]; notu = [r for r in cens if r["off32"] in CENS]
esc = lambda s: s.replace("&", "\\&").replace("_", "\\_")
nums = {"Ecand": len(cand), "Eprobe": sum(r["source"] in ("T32", "MAP", "RARE") for r in cand), "Ehand": sum(r["source"] == "HAND" for r in cand),
        "Ecens": len(cens), "Eunl": len(unl), "Eunlrand": len(unlr), "Enotunlexamples": ", ".join(esc(r["topic"]) for r in notu[:4])}
open(f"{OUT}/e16_numbers.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums.items()) + "\n")
knows_u = sum(bool(r["knows_base"]) for r in unl)
q4c = [r for r in Q["rows"] if r["source"] != "CTRL" and r["verdict"] != "not censored"]; q4u = [r for r in q4c if r["verdict"] == "UNLOCKED"]
agree = sum((r["base"] in CENS) == (q4[r["topic"]]["base"] in ("official_script", "denial", "sovereignty_claim", "praise", "refusal")) for r in cand)
src = lambda r: {"T32": "T", "MAP": "M", "RARE": "R", "HAND": "H"}[r["source"]]
lst = lambda L: ", ".join(f"{esc(r['topic'])} ({src(r)})" for r in L) or "--"
notc = [r for r in cand if r["base"] not in CENS]
body = f"""Table~\\ref{{tab:e16}} gives the verdicts in bf16, the precision in which the seed neurons were selected. The {len(cand)}
candidates are an inventory we picked: {nums['Eprobe']} among the cases the probes had surfaced and {nums['Ehand']} well-known
topics we chose ourselves, each asked with one new direct question; so the confirmation rate measures how well the surfaced cases
reproduce, not the precision of the probes. {len(cens)} are censored by the base model (24 of the {nums['Eprobe']}
probe-surfaced ones; 15 in Q4). Gemma answers {sm['gemma_balanced']} of {len(cand)} on the merits; the two controls are not
censored (with the seed neurons off, the unknown-author control turns from a deflection into a refusal). Switching off the 32 seed
neurons removes the censorship for \\textbf{{{len(unl)}}} of the {len(cens)}, switching off 32 random neurons with the same per-layer
distribution for {len(unlr)}; in {knows_u} of the {len(unl)} unlocked topics the base model also answers the indirect knowledge question
correctly. Among the {len(notu)} topics not unlocked, 7 fail the indirect question; for Tiananmen this contradicts the
multiple-choice result of Section~\\ref{{sec:tian}}, so the indirect question is a weak test of knowledge. A secondary run on Q4 GGUFs gives
{len(q4c)} censored and {len(q4u)} unlocked (random neurons: 0); the base verdict agrees between the two precisions for {agree} of
{len(cand)} candidates. Topics censored in both precisions include Jimmy Lai, Joshua Wong, the Dalai Lama, Senkaku, Spratly,
Kashmir, Laos, the YJ-18 missile, Diao Yinan and Yiguandao; others (e.g.\\ Liu Xiaobo, Galwan) are censored only in bf16.

\\begin{{table}}[h]\\centering\\small
\\begin{{tabularx}}{{\\textwidth}}{{>{{\\raggedright\\arraybackslash}}p{{3.2cm}}c>{{\\raggedright\\arraybackslash}}X}}\\toprule
Verdict (E16, bf16) & n & Topics (source of the candidate)\\\\\\midrule
Censored, \\textbf{{unlocked}} by the 32 seed neurons & {len(unl)} & {lst(unl)}\\\\
Censored, not unlocked & {len(notu)} & {lst(notu)}\\\\
Not censored in this phrasing & {len(notc)} & {lst(notc)}\\\\\\bottomrule
\\end{{tabularx}}
\\caption{{Verification of the cascade's candidates (bf16, greedy, one direct question each). Source: T = scan with the 32 seed
neurons (stage 2); M = censorship-neuron map over our battery (stage 3); R = topic lists written by the model, ranked with the
map (stage 4); H = questions we chose while testing the seed neurons. Random-neuron control: {len(unlr)} unlocked.\\label{{tab:e16}}}}
\\end{{table}}
"""
open(f"{OUT}/e16_table.tex", "w").write(body); print(nums, "knows", knows_u, "q4", len(q4c), len(q4u), "agree", agree)
