"""Build the short "read or write" battery (10 situations x 8 variants) in the exact OpenCode request format.

The system prompt and the 7 tool definitions come from data/opencode_request_template.json (a recorded OpenCode request,
with the working directory recorded in `recorded_workdir`, which is substituted per situation). The conversation is
synthetic: ls -> grep -> read of the file with the obvious cause [-> grep of the usages, variants D/G]; the model
generates the next step.

Variants (one lever at a time relative to A):
  A  base: the agent can neither build nor run anything
  B  can build and run the tests
  C  constraint "without changing X"
  D  the function is used elsewhere (one extra grep of the usages)
  E  the cause is stated by the user (maximum certainty)
  F  control: the file is read only up to just before the cause (reading more is the right move)
  G  C + D together (like the real tasks)
  H  = E without the contradiction: the preamble says "apply the described fix" instead of "find the cause" (in E the
     preamble asks to find the cause while the user says they already found it)

NOTE: the Italian preamble strings below (PRE_*, ASK, the replacements) are prompt CONTENT sent to the model and must
not be translated or edited.

Usage: build_battery.py [--out data/battery_prompts.jsonl] [--preview PATH] [--template data/opencode_request_template.json]
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(os.path.dirname(HERE))         # experiment root (contains code/, data/, results/)
sys.path.insert(0, HERE)
from situations import SITUAZIONI                  # noqa: E402

TEMPLATE = os.path.join(EXP, "data", "opencode_request_template.json")
VARIANTS = "ABCDEFGH"

# grep pattern for the usages (variants D, G): literal token searched across the whole repository
USES = {"s01_top_p": "top_p_filter", "s02_stop": ".match(", "s03_chunk": "chunk_text", "s04_prefill_pos": "plan_prefill",
        "s05_lru": ".insert(", "s06_sse": ".event(", "s07_rmsnorm": "rmsnorm(", "s08_json_escape": "json_object_end",
        "s09_batches": "batches(", "s10_chatml": "render_chat"}

LS_CPP = ["apps", "bench", "cmake", "CMakeLists.txt", "CMakePresets.json", "Dockerfile", "docs", "examples", "include",
          "LICENSE", "README.md", "src", "tests", "third_party", "tools"]

PRE_HEAD = "Sei nella copia di lavoro del repository {repo} (directory corrente)."
PRE_NORUN = {"cpp": "- Qui non puoi compilare né eseguire il codice o i test: lavora leggendo il codice. Non c'è rete.",
             "py": "- Qui non puoi eseguire il codice né i test: lavora leggendo il codice. Non c'è rete."}
PRE_RUN = {"cpp": "- Puoi compilare ed eseguire i test (`cmake --build build && ctest --test-dir build`). Non c'è rete.",
           "py": "- Puoi eseguire il codice e i test (`pytest -q`). Non c'è rete."}
PRE_TAIL = """- Trova la causa del problema descritto sotto e correggila modificando i file del repository (patch mirata).
- Lavora solo dentro questa directory. Hai al massimo 80 turni.
- Il tuo ULTIMO messaggio deve finire con un blocco JSON così (percorsi relativi alla radice del repository;
  funzioni = nomi delle funzioni o dei metodi dove sta la causa / dove va la correzione):
```json
{"files": ["percorso/del/file"], "functions": ["NomeFunzione"], "explanation": "causa e correzione in poche frasi"}
```

=== PROBLEMA ===
"""
ASK = "**Cosa voglio.** Trova la causa e correggila."


def call_id(*parts):
    return "call_" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def work_dir(sid):
    return "/private/tmp/ba_runs/" + hashlib.sha1(sid.encode()).hexdigest()[:8] + "/work"


def ls_listing(s):
    if s["kind"] == "cpp":
        names = LS_CPP
    else:
        tops = {s["path"].split("/")[0]} | {c[0].split("/")[0] for c in s["callers"]} | {"tests"}
        names = sorted(tops | {"pyproject.toml", "README.md"}, key=str.lower)
    return "\n".join(names) + "\n631d518 base\n"


def grep_out(work, matches):
    """matches: list of (relpath, lineno, text). OpenCode grep output format."""
    by = {}
    for p, n, t in matches:
        by.setdefault(p, []).append((n, t))
    out = f"Found {len(matches)} matches\n"
    for p, rows in by.items():
        out += f"{work}/{p}:\n" + "".join(f"  Line {n}: {t}\n\n" for n, t in rows) + "\n"
    return out.rstrip("\n")


def read_out(work, path, lines, limit=None):
    shown = lines if limit is None else lines[:limit]
    body = "\n".join(f"{i + 1}: {t}" for i, t in enumerate(shown))
    if limit is None:
        tail = f"(End of file - total {len(lines)} lines)"
    else:
        tail = f"(Showing lines 1-{limit} of {len(lines)}. Use offset={limit + 1} to continue.)"
    return f"<path>{work}/{path}</path>\n<type>file</type>\n<content>\n{body}\n\n{tail}\n</content>"


def build(s, var, system, tools, rec_work):
    sid = s["id"]
    work = work_dir(sid)
    lines = s["code"].rstrip("\n").split("\n")
    pos = s["code"].find(s["bug"])
    assert pos >= 0, (sid, "bug not found")
    bug_line = s["code"][:pos].count("\n") + 1
    cut_line = next(i for i, t in enumerate(lines) if s["cut"] in t) + 1     # first line hidden in F
    assert cut_line <= bug_line, (sid, cut_line, bug_line)
    limit = cut_line - 1
    while limit > 0 and lines[limit - 1].strip() == "":
        limit -= 1

    pre = [PRE_HEAD.format(repo=s["repo"]), PRE_RUN[s["kind"]] if var == "B" else PRE_NORUN[s["kind"]]]
    prob = s["problema"]
    assert ASK in prob, sid
    if var in "CG":
        prob = prob.replace(ASK, f"**Cosa voglio.** Trova la causa e correggila {s['vincolo']}.")
    if var == "E":
        prob = prob.replace(ASK, f"**Cosa voglio.** La causa l'ho già trovata: {s['causa']} Applica la correzione.")
    if var == "H":
        prob = prob.replace(ASK, f"**Cosa voglio.** La causa l'ho già trovata: {s['causa']} Applica la correzione.")
    tail = PRE_TAIL
    if var == "H":
        tail = tail.replace("- Trova la causa del problema descritto sotto e correggila modificando i file del repository (patch mirata).",
                            "- Applica la correzione descritta sotto modificando i file del repository (patch mirata).")
        assert tail != PRE_TAIL
    user = "\n".join(pre) + "\n" + tail + prob + "\n"

    pat, gdir = s["grep"]
    m_file = [(s["path"], i + 1, t) for i, t in enumerate(lines) if pat in t]
    m_head = [h for h in s["header"] if pat in h[2] and h[0].startswith(gdir)]
    assert all(n != bug_line for _, n, _t in m_file), (sid, "the grep shows the cause line")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def turn(reason, name, args, result):
        cid = call_id(sid, var, name, json.dumps(args))
        msgs.append({"role": "assistant", "content": "", "reasoning_content": reason,
                     "tool_calls": [{"id": cid, "type": "function",
                                     "function": {"name": name, "arguments": json.dumps(args)}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": result})

    turn("Let me explore the repository first.", "bash",
         {"command": "ls && git log --oneline -3 2>/dev/null | head"}, ls_listing(s))
    turn("Let me search for the relevant code.", "grep", {"pattern": pat, "path": gdir},
         grep_out(work, m_head + m_file))
    rargs = {"filePath": s["path"]}
    if var == "F":
        rargs["limit"] = limit
    turn(f"Let me read {s['path']}.", "read", rargs, read_out(work, s["path"], lines, limit if var == "F" else None))
    if var in "DG":
        tok = USES[sid]
        m_def = [(s["path"], i + 1, t) for i, t in enumerate(lines) if tok in t]
        m_h = [h for h in s["header"] if tok in h[2]]
        root = "src" if s["kind"] == "cpp" else "."
        uses = m_h + m_def + list(s["callers"])
        turn(f"Let me check where `{tok.strip('.(')}` is used.", "grep", {"pattern": tok, "path": root},
             grep_out(work, uses))

    sysw = system.replace(rec_work, work)
    msgs[0]["content"] = sysw
    toolsw = json.loads(json.dumps(tools).replace(rec_work, work))
    body = {"model": "inkling2-mini", "max_tokens": 32000, "messages": msgs, "tools": toolsw, "tool_choice": "auto",
            "stream": False}
    return {"id": f"{sid}_{var}", "sid": sid, "var": var, "path": s["path"], "bug_line": bug_line,
            "f_limit": limit, "body": body}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EXP, "data", "battery_prompts.jsonl"))
    ap.add_argument("--preview", default=None, help="optional: write a readable example prompt (s03_chunk, variant G) here")
    ap.add_argument("--template", default=TEMPLATE, help="JSON with `system`, `tools`, `recorded_workdir`")
    a = ap.parse_args()
    tpl = json.load(open(a.template))
    system, tools, rec_work = tpl["system"], tpl["tools"], tpl["recorded_workdir"]
    assert rec_work in system
    rows = [build(s, v, system, tools, rec_work) for s in SITUAZIONI for v in VARIANTS]
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.preview:
        # readable preview: one situation, variant G (the richest), without the system prompt and tools
        ex = next(r for r in rows if r["id"] == "s03_chunk_G")
        with open(a.preview, "w") as f:
            f.write(f"# Full example: {ex['id']} (OpenCode system prompt and 7 tools omitted, ~6k tokens)\n\n")
            for m in ex["body"]["messages"][1:]:
                if m["role"] == "user":
                    f.write("## user\n\n" + m["content"] + "\n")
                elif m["role"] == "assistant":
                    tc = m["tool_calls"][0]["function"]
                    f.write(f"## assistant\n\n_reasoning:_ {m['reasoning_content']}\n\n`{tc['name']}` `{tc['arguments']}`\n\n")
                else:
                    f.write("## tool result\n\n```\n" + m["content"] + "\n```\n\n")
            f.write("## -> the model generates here: the first tool called and its arguments are recorded\n")
    print(f"{len(rows)} prompts -> {a.out}" + (f"; example -> {a.preview}" if a.preview else ""))


if __name__ == "__main__":
    main()
