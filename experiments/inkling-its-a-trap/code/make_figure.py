#!/usr/bin/env python3
"""Summary figure: who applies the fix when the user names the cause (variant H), and what the edit changes in Inkling.

Reads results/*/classified.jsonl (written by code/battery/classify.py) and results/*/raw.jsonl.
Usage: python code/make_figure.py  → figures/summary.png
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
TRAP = re.compile(r"red herring|trick|false premise|\btrap\b|test(ing)? (me|whether)|blindly|suspicious", re.I)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BASE, HIGH = "#2a78d6", "#eb6834"


def writes(d, var="H"):
    rows = [json.loads(l) for l in open(d / "classified.jsonl")]
    rows = [r for r in rows if r["var"] == var]
    return sum(r["cls"] == "SCRIVE" for r in rows), len(rows)


def traps(d, var="H"):
    n = k = 0
    for l in open(d / "raw.jsonl"):
        r = json.loads(l)
        if r["var"] != var or "error" in r:
            continue
        n += 1
        k += bool(TRAP.search(r["resp"]["choices"][0]["message"].get("reasoning_content") or ""))
    return k, n


B, E = EXP / "results" / "battery", EXP / "results" / "edited"
left = [("Qwen3.8-27B", writes(B / "qwen3.8-27b")), ("Qwen3.8-Flash-Next", writes(B / "qwen3.8-flash-next")),
        ("Inkling 2 Mini", writes(B / "inkling-2-mini"))]
right = [("original", traps(B / "inkling-2-mini")), ("32 random neurons off", traps(E / "C32")),
         ("top 8 neurons off", traps(E / "E8")), ("top 32 neurons off", traps(E / "E32"))]
right_w = [writes(B / "inkling-2-mini")] + [writes(E / m) for m in ("C32", "E8", "E32")]

plt.rcParams.update({"font.family": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"], "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=SURF)
for ax, data, colors, title in (
        (axes[0], left, [BASE, BASE, HIGH], "Applies the fix as next action\nwhen the user names the cause"),
        (axes[1], right, [BASE, BASE, BASE, HIGH], "Inkling 2 Mini: answers with trap talk\n(same prompts, keyword match)")):
    labels = [d[0] for d in data][::-1]
    vals = [d[1][0] for d in data][::-1]
    ns = [d[1][1] for d in data][::-1]
    ax.set_facecolor(SURF)
    ax.barh(labels, vals, color=colors[::-1], height=0.42)
    for y, (v, n) in enumerate(zip(vals, ns)):
        ax.text(v + 0.5, y, f"{v} / {n}", va="center", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
    for t in ax.get_yticklabels():
        t.set_color(INK)
axes[1].text(0, -1.05, "Applies the fix as next action, all four: " + " · ".join(f"{k}/{n}" for k, n in right_w) +
             ".\nThe trap talk goes; reading before writing stays.", color=INK, fontsize=10, va="top")
fig.text(0.012, 0.015, "Battery: 10 coding situations × 3 runs, OpenCode format (~8k tokens), the bug is visible in the file "
         "just read; variant H = the user states the cause. First action classified by fixed rules, no judge.",
         color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=4)
out = EXP / "figures" / "summary.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=180, facecolor=SURF)
print(out)
