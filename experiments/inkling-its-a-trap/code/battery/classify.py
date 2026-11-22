#!/usr/bin/env python3
"""Classify the first action of every response of the short battery and print the tables.

Classes (internal labels kept as in the raw results; table headings use English): SCRIVE = WRITES (edit/write, or a bash
command that modifies a file), LEGGE = READS (read/grep/glob, read-only bash), ESEGUE = RUNS (tests, build, scripts),
PIANO = PLAN (only todowrite: the next step, recorded by run_battery.py, is what counts), TESTO = TEXT (no tool call),
LUNGO = TRUNCATED (finished by max_tokens), ERR = ERROR. For WRITES, "in-site" means the arguments name the file holding
the cause. From the reasoning: its length and hedging words.

Usage: classify.py DIR [DIR2 ...]   (writes DIR/classified.jsonl and DIR/table.md)
Reads data/battery_prompts.jsonl (relative to the experiment root) to know each prompt's target file.
"""
import json, math, os, re, sys

W_BASH = [r"\bsed\s+(-[a-zA-Z]*i\b|-i\b|--in-place)", r"\bperl\s+(-[a-zA-Z0-9]*\s+)*-[a-zA-Z0-9]*i", r"(^|[;&|\n]\s*)cat\s*>{1,2}\s*\S",
          r"[^2&>]>{1,2}\s*['\"]?[\w./-]+\.(py|cpp|cu|cuh|h|hpp|cc)\b", r"\btee\s+(-a\s+)?\S+\.(py|cpp|cu|h|hpp|cc)\b",
          r"\bgit\s+apply\b", r"\bpatch\s+(-p|<)", r"\bapply_patch\b", r"\bed\s+-s\b"]
PY_INLINE = r"\bpython3?\s+(-\s*<<|-c\b)"
PY_WRITE = r"\.write\(|write_text|open\([^)]*['\"][wa]\+?['\"]|\.replace\(.*\)\s*\)?\s*$"
RUN = r"\b(pytest|ctest|make|cmake|ninja|cargo|npm\s+test|nvcc|g\+\+|clang\+\+)\b|\bpython3?\s+(-m\s+\w+|[\w/.-]+\.py)\b|\./build/"
HEDGE = ["not sure", "unsure", "careful", "verify", "double-check", "double check", "make sure", "confirm", "let me check",
         "let me look", "let me see", "might ", "maybe", "perhaps", "hmm", "wait", "however", "but "]


PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "battery_prompts.jsonl")


def bash_class(cmd):
    if any(re.search(p, cmd) for p in W_BASH):
        return "SCRIVE"
    if re.search(PY_INLINE, cmd):
        return "SCRIVE" if re.search(PY_WRITE, cmd, re.M) else ("LEGGE" if "print" in cmd else "ESEGUE")
    if re.search(RUN, cmd):
        return "ESEGUE"
    return "LEGGE"


def call_class(tc):
    name = tc["function"]["name"]
    try:
        args = json.loads(tc["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        args = {"_raw": tc["function"]["arguments"]}
    if name in ("edit", "write"):
        return "SCRIVE", args
    if name in ("read", "grep", "glob"):
        return "LEGGE", args
    if name == "todowrite":
        return "PIANO", args
    if name == "bash":
        return bash_class(str(args.get("command", args.get("_raw", "")))), args
    return "ALTRO", args


def classify_msg(choice):
    msg = choice["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return ("LUNGO" if choice.get("finish_reason") == "length" else "TESTO"), [], msg
    cls = [call_class(tc) for tc in tcs]
    return cls[0][0], cls, msg


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def main():
    for d in sys.argv[1:]:
        recs = [json.loads(l) for l in open(os.path.join(d, "raw.jsonl"))]
        prompts = {json.loads(l)["id"]: json.loads(l) for l in open(PROMPTS)}
        out = []
        for r in recs:
            c = {k: r[k] for k in ("id", "sid", "var", "rep", "model")}
            if "error" in r:
                c.update(cls="ERR", err=r["error"])
                out.append(c)
                continue
            ch = r["resp"]["choices"][0]
            cls, calls, msg = classify_msg(ch)
            reason = msg.get("reasoning_content") or ""
            if cls == "PIANO" and "resp2" in r:
                ch2 = r["resp2"]["choices"][0]
                cls, calls, msg2 = classify_msg(ch2)
                reason += "\n" + (msg2.get("reasoning_content") or "")
                c["dopo_piano"] = True
            path = prompts[r["id"]]["path"]
            base = os.path.basename(path)
            first_args = json.dumps(calls[0][1], ensure_ascii=False) if calls else ""
            c.update(cls=cls, any_write=any(k == "SCRIVE" for k, _ in calls), n_calls=len(calls),
                     nel_sito=(cls == "SCRIVE" and base in first_args), first=first_args[:300],
                     tools=[tc["function"]["name"] for tc in (msg.get("tool_calls") or [])],
                     reason_words=len(reason.split()), hedge=sum(reason.lower().count(h) for h in HEDGE),
                     out_tok=(r["resp"].get("usage") or {}).get("completion_tokens"), wall_s=r.get("wall_s"),
                     content=(msg.get("content") or "")[:300], reason_tail=reason[-400:])
            out.append(c)
        with open(os.path.join(d, "classified.jsonl"), "w") as f:
            for c in out:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        lines = [f"# Short battery - {d}\n", f"{len(out)} responses\n",
                 "| variant | n | WRITES (95% CI) | in-site | READS | RUNS | TEXT | TRUNCATED | ERROR | reasoning words (median) | hedging / response |",
                 "|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for v in "ABCDEFGH":
            vs = [c for c in out if c["var"] == v]
            if not vs:
                continue
            n = len(vs)
            k = sum(c["cls"] == "SCRIVE" for c in vs)
            p, lo, hi = wilson(k, n)
            cnt = lambda x: sum(c["cls"] == x for c in vs)  # noqa: E731
            rw = sorted(c.get("reason_words", 0) for c in vs if c["cls"] != "ERR")
            med = rw[len(rw) // 2] if rw else 0
            hd = sum(c.get("hedge", 0) for c in vs) / max(1, n)
            lines.append(f"| {v} | {n} | {k} = {p:.2f} [{lo:.2f}-{hi:.2f}] | {sum(c.get('nel_sito', False) for c in vs)} | "
                         f"{cnt('LEGGE')} | {cnt('ESEGUE')} | {cnt('TESTO')} | {cnt('LUNGO')} | {cnt('ERR')} | {med} | {hd:.1f} |")
        sids = sorted({c["sid"] for c in out})
        lines += ["", "First action per situation (S writes, L reads, X runs, T text, P truncated, E error; one letter per repetition)", "",
                  "| situation | " + " | ".join("ABCDEFGH") + " |", "|---|" + "---|" * 8]
        code = {"SCRIVE": "S", "LEGGE": "L", "ESEGUE": "X", "TESTO": "T", "LUNGO": "P", "ERR": "E", "PIANO": "p", "ALTRO": "?"}
        for s in sids:
            cells = []
            for v in "ABCDEFGH":
                cs = sorted((c for c in out if c["sid"] == s and c["var"] == v), key=lambda c: c["rep"])
                cells.append("".join(code[c["cls"]] for c in cs))
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        open(os.path.join(d, "table.md"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
