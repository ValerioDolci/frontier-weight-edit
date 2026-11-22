#!/usr/bin/env python3
"""Extract the "apply / hesitate" decision points from Inkling's reasoning in the short battery (22/11/2026).

For every Inkling response, find the first paragraph boundary ("\\n\\n") that comes AFTER a paragraph in which the fix is
already written (fix / should be / bug / instead of ...) and BEFORE a paragraph that starts with a hesitation (But, Wait,
Also, However, Hmm, Actually, "Let me check/look/verify/see..."). Prefix = battery prompt + reasoning up to that boundary;
the decision token is the first one of the following paragraph.
Categories: DIFF = cause stated by the user (variants E, H) · THOR = cause found by the model itself (A, C, D, G) ·
RUN = can execute (B).
Output: {id, sid, var, rep, cat, reasoning_prefix, next_text, prompt_id} per line.

Usage: extract_pivots.py [--raw results/battery/inkling-2-mini/raw.jsonl] [--out results/pivots/pivot_points.jsonl]
"""
import argparse, json, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(os.path.dirname(HERE))   # experiment root
FIX = re.compile(r"\bfix\b|should be|\bbug\b|instead of|\bwrong\b|\bcorrect(ion)?\b", re.I)
HES = re.compile(r"(But|Wait|Also|However|Hmm|Actually|Before|Let me (check|look|verify|see|also|confirm|make sure|search|read|examine|inspect)|Let's (check|look|verify|see|also))\b")
CAT = {"E": "DIFF", "H": "DIFF", "A": "THOR", "C": "THOR", "D": "THOR", "G": "THOR", "B": "RUN"}


def pivot(t):
    for m in re.finditer(r"\n\n", t):
        p = m.end()
        if FIX.search(t[:p]) and HES.match(t[p:]):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(EXP, "results", "battery", "inkling-2-mini", "raw.jsonl"))
    ap.add_argument("--out", default=os.path.join(EXP, "results", "pivots", "pivot_points.jsonl"))
    a = ap.parse_args()
    rows, tot, hit, first = [], collections.Counter(), collections.Counter(), collections.Counter()
    for l in open(a.raw):
        r = json.loads(l)
        if "error" in r or r["var"] not in CAT:
            continue
        t = r["resp"]["choices"][0]["message"].get("reasoning_content") or ""
        tot[r["var"]] += 1
        p = pivot(t)
        if p is None:
            continue
        hit[r["var"]] += 1
        first[t[p:].split()[0]] += 1
        rows.append({"id": f"{r['id']}_r{r['rep']}", "sid": r["sid"], "var": r["var"], "rep": r["rep"], "cat": CAT[r["var"]],
                     "prompt_id": r["id"], "reasoning_prefix": t[:p], "next_text": t[p:p + 160]})
    with open(a.out, "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("points per variant:", {v: f"{hit[v]}/{tot[v]}" for v in sorted(tot)})
    print("per category:", dict(collections.Counter(x["cat"] for x in rows)))
    print("situations covered (DIFF):", len({x["sid"] for x in rows if x["cat"] == "DIFF"}))
    print("first word:", first.most_common(12))
    print("->", a.out)


if __name__ == "__main__":
    main()
