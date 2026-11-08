"""Summary figure: (a) Pan & Xu, share of censored answers per model/variant and language; (b) C8 grid, sovereignty script
on contested territories and controls. Usage: fig_summary.py OUT.png"""
import json, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
RD = "./censura/checks/results"
E = json.load(open(f"{RD}/e11_summary.json")); Gm = json.load(open(f"{RD}/gemma_summary.json")); C8 = json.load(open(f"{RD}/c8_judged.json"))["hyp"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"Qwen base": "#2a78d6", "Qwen off32": "#eb6834", "Qwen comb_t32": "#1baf7a", "Gemma 4 31B": "#eda100", "Qwen F1 only": "#e87ba4"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.35, 1]}, facecolor=SURF)
def bars(ax, groups, series, vals, title, sub, ymax=80):
    ax.set_facecolor(SURF); n = len(series); w = 0.8 / n
    for gi, g in enumerate(groups):
        for si, s in enumerate(series):
            v = vals.get((g, s))
            if v is None: continue
            x = gi + (si - (n - 1) / 2) * w
            if v > 0: ax.add_patch(FancyBboxPatch((x - w / 2 + 0.01, 0), w - 0.02, v, boxstyle="round,pad=0,rounding_size=0.03", mutation_aspect=40,
                                        linewidth=0, facecolor=COL[s]))
            ax.text(x, v + ymax * 0.02, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlim(-0.6, len(groups) - 0.4); ax.set_ylim(0, ymax); ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.set_ylabel("% of answers"); ax.set_title(title + "\n", loc="left", color=INK, fontsize=11, fontweight="bold"); ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
v = {("Chinese prompts", "Qwen base"): 100 * E["zh|base"]["rate_censored"], ("Chinese prompts", "Qwen off32"): 100 * E["zh|off32"]["rate_censored"],
     ("Chinese prompts", "Qwen comb_t32"): 100 * E["zh|comb_t32"]["rate_censored"], ("Chinese prompts", "Gemma 4 31B"): 100 * Gm["e11_zh_gemma"]["rate"],
     ("English prompts", "Qwen base"): 100 * E["en|base"]["rate_censored"], ("English prompts", "Qwen off32"): 100 * E["en|off32"]["rate_censored"],
     ("English prompts", "Qwen comb_t32"): 100 * E["en|comb_t32"]["rate_censored"]}
bars(a1, ["Chinese prompts", "English prompts"], ["Qwen base", "Qwen off32", "Qwen comb_t32", "Gemma 4 31B"], v,
     "Answers judged censored: external battery", "Pan & Xu 2026 · 145 questions · Q4 · Gemma: Chinese only", ymax=75)
r = lambda k, cat: 100 * C8[k][cat][0] / C8[k][cat][1]
v2 = {("11 contested territories", "Qwen base"): r("base", "contested_rate"), ("11 contested territories", "Qwen off32"): r("off32", "contested_rate"),
      ("11 contested territories", "Qwen F1 only"): r("F1only", "contested_rate"),
      ("4 control countries", "Qwen base"): 100 * C8["base"]["H-categoria"]["control_rate"][0] / C8["base"]["H-categoria"]["control_rate"][1],
      ("4 control countries", "Qwen off32"): 100 * C8["off32"]["H-categoria"]["control_rate"][0] / C8["off32"]["H-categoria"]["control_rate"][1],
      ("4 control countries", "Qwen F1 only"): 100 * C8["F1only"]["H-categoria"]["control_rate"][0] / C8["F1only"]["H-categoria"]["control_rate"][1]}
bars(a2, ["11 contested territories", "4 control countries"], ["Qwen base", "Qwen off32", "Qwen F1 only"], v2,
     "Answers judged censored: territory grid (C8)", "5 questions × EN/ZH · bf16 · 23 of 29 are sovereignty claims", ymax=75)
hs = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in COL]
fig.legend(hs, ["Qwen3.8-27B base", "32 Taiwan neurons off (off32)", "off32 + 32 Tiananmen neurons", "Gemma 4 31B-it (reference)", "F1 family off (37 map neurons, sovereignty)"],
           loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(sys.argv[1], dpi=160, facecolor=SURF); print("ok", sys.argv[1])
