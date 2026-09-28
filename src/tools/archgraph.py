#!/usr/bin/env python3
"""archgraph.py - the architecture of src/core as Mermaid graphs and a text map.

Scans src/core (every .h / .c), resolves every quoted #include the way the
build does (the same resolver and zone rule as baseline/hygiene.py), and
writes, under docs/architecture/generated/:

  zones.md          the six zones (common, split, il, bridge, wisdom2, front)
                    and the include edges between them, checked against the
                    dependency rules
  folders.md        every core folder as a node, grouped by zone; an edge is
                    "folder A includes a file of folder B", labelled with the
                    number of such includes
  folders/*.md      one diagram per folder: its files and their includes;
                    targets outside the folder are collapsed to their folder
  map.md            the plain-text index: every file with its one-line role
                    (the first sentence of its opening comment), what it
                    includes and what includes it. Written to be read by a
                    person or pasted to an AI; nothing needs rendering.
  functions.md      every function defined in src/core: file, line, the core
                    functions it calls, the ones it references by name
                    (thread-pool tasks, dispatch tables, function pointers)
                    and its callers
  calls/*.md        call graphs from the entry points: vfft_create,
                    vfft_execute and each side of their layout fork
  graph.json        the same data, for any other tool

SVG copies, for reading without a Mermaid viewer, go to docs/architecture/svg/
with the same layout (a markdown file with several diagrams gives one SVG per
section). They are rendered by mermaid-cli (https://github.com/mermaid-js/
mermaid-cli): set MMDC to its mmdc, or put mmdc on PATH, or let the tool run
`npx -y @mermaid-js/mermaid-cli`. A Chromium is needed; set CHROME to its
binary if puppeteer cannot find one. svg/manifest.json records a hash of each
diagram's source, so --check reports stale SVGs without needing mermaid-cli.

The call graph is best effort, not a compiler: both sides of every #if are
read, and calls through function pointers or macros are not followed.

Everything is derived from the code, so the output is never edited by hand.
Only the standard library is used; Python 3.8+.

USAGE (from anywhere in the repo)
  python src/tools/archgraph.py               regenerate docs/architecture/generated/
  python src/tools/archgraph.py --check       exit 1 and list the stale files if the
                                              committed output differs from the code
  python src/tools/archgraph.py --focus X     print one Mermaid graph of X and its
                                              neighbours to stdout; X is a file
                                              (ztt.h), a folder (split/real) or a zone
                                              (il). --depth N widens it (default 1).
  python src/tools/archgraph.py --svg         also render every diagram to
                                              docs/architecture/svg/ (needs mermaid-cli:
                                              $MMDC, mmdc on PATH, or npx; see below)
  python src/tools/archgraph.py --calls F     print the call graph of function F (as deep
                                              as fits ~70 nodes, or --depth N levels);
                                              --up: its callers instead
"""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import tempfile
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "baseline"))
import hygiene   # noqa: E402  (the include resolver and the zone rule)
import toolchain  # noqa: E402

ROOT = toolchain.ROOT
CORE = toolchain.CORE
OUT = os.path.join(ROOT, "docs", "architecture", "generated")
SVG = os.path.join(ROOT, "docs", "architecture", "svg")
ZONES = ["common", "split", "il", "bridge", "wisdom2", "front"]
ZONE_ROLE = {
    "common": "shared by both layouts: ABI types, math, support, the wisdom2 store core, the plan struct",
    "split": "the SPLIT library (separate re[] / im[] planes)",
    "il": "the INTERLEAVED library (z[] of (re, im) pairs)",
    "bridge": "the only place besides the front door that sees both layouts (temporary)",
    "wisdom2": "front-side wisdom glue spanning both layouts (legacy readers, migration)",
    "front": "the public front door: vfft.c and its helpers",
}


def zone_of(path):
    rel = os.path.relpath(path, CORE).replace(os.sep, "/")
    if rel.split("/", 1)[0] == "wisdom2":
        return "wisdom2"
    return hygiene._zone(path)


def allowed(src_zone, dst_zone, dst_path):
    """hygiene.dep_violations' rule; the front door and wisdom2 may include anything."""
    if src_zone in ("front", "wisdom2"):
        return True
    return (dst_zone == src_zone or dst_zone == "common"
            or (src_zone == "bridge" and dst_zone in ("split", "il"))
            or (src_zone == "il" and dst_zone == "bridge"
                and os.path.basename(dst_path) == "real_doors.h"))


def folder_of(rel):
    d = os.path.dirname(rel)
    return d if d else "(core)"


_TOP = re.compile(r"^(?:\s*(?:#[^\n]*)?\n)*\s*(/\*.*?\*/|(?://[^\n]*\n)+)", re.S)
_DECOR = re.compile(r"^[\s=*\-─═━_#~]*$")


def role_of(text, name):
    """The file's one-line role: the first paragraph of its opening comment (the
    comment may follow the include guard), without the 'name.h -' prefix and
    banner lines; its first sentence unless that is a short tag ('INTERNAL.')."""
    m = _TOP.match(text)
    if not m:
        return ""
    body = re.sub(r"^/\*+!?|\*+/$", "", m.group(1).strip())
    para = []
    for raw in body.split("\n"):
        l = re.sub(r"^\s*(\*+|//+)\s?", "", raw).strip()
        l = re.sub(r"^[─═━=\-]{2,}\s*|\s*[─═━=\-]{2,}$", "", l).strip()
        if not l or _DECOR.match(l):
            if para:
                break
            continue
        para.append(l)
    s = " ".join(para)
    s = re.sub(r"^@?(file|brief)\s+", "", s)
    base = re.escape(os.path.splitext(name)[0])
    s = re.sub(r"^%s(\.[ch])?\s*(\(.*?\))?\s*(—|--|-|:)\s*" % base, "", s)
    m2 = re.match(r"(.+?[.!?])(\s|$)", s)
    if m2 and len(m2.group(1)) >= 25:
        s = m2.group(1)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= 180 else s[:177].rstrip() + "..."


# ---------------------------------------------------------------- functions
# Best effort, not a compiler: comments, strings and preprocessor lines are
# blanked, both sides of every #if are read, and a call is a core function's
# name followed by '('. A core function named WITHOUT a call (handed to the
# thread pool, stored in a dispatch table or a plan's function pointer) is a
# reference. Calls through function pointers and macros are not followed.
_KW = {"if", "for", "while", "switch", "return", "sizeof", "_Alignof", "alignof", "__attribute__",
       "defined", "else", "do", "case", "__declspec", "_Static_assert", "static_assert",
       "typeof", "__typeof__"}
_HEAD = re.compile(r"([A-Za-z_]\w*)\s*\(([^;{}()]|\([^;{}()]*\))*\)\s*(__attribute__\s*\(\(.*?\)\)\s*)*$", re.S)
_WORD = re.compile(r"\b([A-Za-z_]\w*)\b(\s*\()?")


def _clean(t):
    out, i, n = [], 0, len(t)
    while i < n:
        if t.startswith("/*", i):
            j = t.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", t[i:j]))
            i = j
        elif t.startswith("//", i):
            j = t.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif t[i] in "\"'":
            q, j = t[i], i + 1
            while j < n and t[j] != q and t[j] != "\n":
                j += 2 if t[j] == "\\" else 1
            out.append(q + re.sub(r"[^\n]", " ", t[i + 1:j]) + q)
            i = j + 1
        else:
            out.append(t[i])
            i += 1
    lines, cont = "".join(out).split("\n"), False
    for k, l in enumerate(lines):
        if cont or l.lstrip().startswith("#"):
            cont = l.rstrip().endswith("\\")
            lines[k] = ""
    return "\n".join(lines)


def _defs(text):
    """[(name, first line, body)] of the functions defined at file scope."""
    s = _clean(text)
    res, depth, last, name, start = [], 0, 0, None, 0
    for i, c in enumerate(s):
        if c == "{":
            if depth == 0:
                head = s[last:i]
                m = _HEAD.search(head)
                h = head.strip()
                name = (m.group(1) if m and m.group(1) not in _KW and "=" not in h
                        and not re.match(r"^(typedef|struct|union|enum)\b", h) else None)
                start = i
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
            if depth == 0:
                if name:
                    res.append((name, s[:start].count("\n") + 1, s[start:i + 1]))
                name, last = None, i + 1
        elif c == ";" and depth == 0:
            last = i + 1
    return res


def scan_functions(files, texts):
    funcs = {}
    bodies = []
    for rel in sorted(files):
        for name, line, body in _defs(texts[rel]):
            f = funcs.setdefault(name, {"file": rel, "line": line, "calls": set(), "refs": set()})
            bodies.append((name, body))
    names = set(funcs)
    for name, body in bodies:
        for m in _WORD.finditer(body):
            t = m.group(1)
            if t in names and t != name:
                (funcs[name]["calls"] if m.group(2) else funcs[name]["refs"]).add(t)
    for f in funcs.values():
        f["refs"] -= f["calls"]
        f["callers"] = set()
    for name, f in funcs.items():
        for t in f["calls"] | f["refs"]:
            funcs[t]["callers"].add(name)
    for f in funcs.values():
        for k in ("calls", "refs", "callers"):
            f[k] = sorted(f[k])
    return funcs


def scan():
    dirs = toolchain.include_dirs()
    files = {}
    for p in hygiene.core_headers():
        rel = os.path.relpath(p, CORE).replace(os.sep, "/")
        text = open(p, encoding="utf-8", errors="replace").read()
        files[rel] = {"path": p, "zone": zone_of(p), "folder": folder_of(rel),
                      "role": role_of(text, os.path.basename(rel)),
                      "lines": text.count("\n"), "includes": [], "external": [], "text": text}
    for rel, f in files.items():
        seen = set()
        for inc in hygiene._INC.findall(f["text"]):
            tgt = hygiene.resolve(inc, f["path"], dirs)
            if tgt and os.path.normpath(tgt).startswith(os.path.normpath(CORE) + os.sep):
                t = os.path.relpath(tgt, CORE).replace(os.sep, "/")
                if t in files and t != rel and t not in seen:
                    seen.add(t)
                    f["includes"].append(t)
            elif inc not in f["external"]:
                f["external"].append(inc)
    texts = {rel: f.pop("text") for rel, f in files.items()}
    for f in files.values():
        f["included_by"] = []
    for rel, f in sorted(files.items()):
        for t in f["includes"]:
            files[t]["included_by"].append(rel)
    return files, scan_functions(files, texts)


# ---------------------------------------------------------------- Mermaid
def nid(s):
    return "n_" + re.sub(r"[^A-Za-z0-9_]", "_", s)


def lbl(s):
    return s.replace('"', "#quot;")


def zones_md(files):
    edges, bad = {}, []
    for rel, f in sorted(files.items()):
        for t in f["includes"]:
            a, b = f["zone"], files[t]["zone"]
            if a != b:
                edges[(a, b)] = edges.get((a, b), 0) + 1
            if not allowed(a, b, t):
                bad.append("%s (%s) includes %s (%s)" % (rel, a, t, b))
    counts = {z: sum(1 for f in files.values() if f["zone"] == z) for z in ZONES}
    out = ["# Zones", "",
           "Generated by `src/tools/archgraph.py` from the includes in `src/core`. Do not edit.", "",
           "An edge A --> B means files of zone A include files of zone B; the label is the",
           "number of such includes. The dependency rules (checked by",
           "`src/tools/baseline/hygiene.py`): `common` includes only `common`; `split` and `il`",
           "include themselves and `common` (`il` may also use `bridge/real_doors.h`); `bridge`",
           "may include both layouts; `wisdom2` and the front door may include anything.", "",
           "```mermaid", "flowchart TD"]
    for z in ZONES:
        out.append('    %s["%s (%d files)<br/>%s"]' % (nid(z), z, counts[z], lbl(ZONE_ROLE[z])))
    for (a, b), n in sorted(edges.items(), key=lambda e: (ZONES.index(e[0][0]), ZONES.index(e[0][1]))):
        out.append("    %s -->|%d| %s" % (nid(a), n, nid(b)))
    out += ["```", ""]
    out.append("Rule violations: " + ("none." if not bad else "%d" % len(bad)))
    out += ["- " + b for b in bad]
    return "\n".join(out) + "\n"


def folders_md(files):
    """One diagram per zone: its folders and the includes among them; includes
    that leave the zone are collapsed to one node per target zone. The small
    zones (bridge, wisdom2, front) share one last diagram."""
    folders = {}
    for f in files.values():
        folders.setdefault(f["folder"], {"zone": f["zone"], "n": 0})["n"] += 1
    fedge, zedge = {}, {}
    for f in files.values():
        for t in f["includes"]:
            a, b = f["folder"], files[t]["folder"]
            if a == b:
                continue
            if files[t]["zone"] == f["zone"]:
                fedge[(a, b)] = fedge.get((a, b), 0) + 1
            else:
                k = (a, files[t]["zone"])
                zedge[k] = zedge.get(k, 0) + 1
    out = ["# Folders", "",
           "Generated by `src/tools/archgraph.py`. Do not edit.", "",
           "One diagram per zone. A box is a folder of `src/core` (with its file count); an",
           "edge A --> B means files in A include files in B, labelled with the number of",
           "includes. Includes that leave the zone end in a rounded node for the target zone.",
           "Per-folder file diagrams are in `folders/`, the full lists in `map.md`.", ""]
    groups = [("split", ["split"]), ("il", ["il"]), ("common", ["common"]),
              ("bridge, wisdom2 and the front door", ["bridge", "wisdom2", "front"])]
    for title, zs in groups:
        mine = sorted(k for k, v in folders.items() if v["zone"] in zs)
        out += ["## %s" % title, "", "```mermaid", "flowchart LR"]
        for k in mine:
            name = "src/core (vfft.c, ...)" if k == "(core)" else k
            out.append('    %s["%s (%d)"]' % (nid(k), name, folders[k]["n"]))
        targets = sorted({z for (a, z) in zedge if a in mine})
        for z in targets:
            out.append('    %s(["%s"])' % (nid("zone_" + z), z))
        for (a, b), n in sorted(fedge.items()):
            if a in mine:
                out.append("    %s -->|%d| %s" % (nid(a), n, nid(b)))
        for (a, z), n in sorted(zedge.items()):
            if a in mine:
                out.append("    %s -->|%d| %s" % (nid(a), n, nid("zone_" + z)))
        out += ["```", ""]
    return "\n".join(out)


def folder_file(folder):
    return "core" if folder == "(core)" else folder.replace("/", "__")


def folder_md(files, folder):
    mine = sorted(r for r, f in files.items() if f["folder"] == folder)
    out = ["# %s" % ("src/core (top level)" if folder == "(core)" else "src/core/" + folder), "",
           "Generated by `src/tools/archgraph.py`. Do not edit. Zone: `%s`." % files[mine[0]]["zone"], "",
           "Files of this folder and their includes. Includes of files in other folders are",
           "collapsed to that folder (a rounded node); the full lists are in `../map.md`.", ""]
    for r in mine:
        role = files[r]["role"]
        out.append("- `%s`%s" % (os.path.basename(r), (": " + role) if role else ""))
    out += ["", "```mermaid", "flowchart LR"]
    ext = set()
    for r in mine:
        out.append('    %s["%s"]' % (nid(r), os.path.basename(r)))
    lines = []
    for r in mine:
        for t in files[r]["includes"]:
            if files[t]["folder"] == folder:
                lines.append("    %s --> %s" % (nid(r), nid(t)))
            else:
                ext.add(files[t]["folder"])
                lines.append("    %s --> %s" % (nid(r), nid("dir_" + files[t]["folder"])))
    for e in sorted(ext):
        out.append('    %s("%s/")' % (nid("dir_" + e), e))
    out += sorted(set(lines)) + ["```"]
    return "\n".join(out) + "\n"


def map_md(files):
    out = ["# Map of src/core", "",
           "Generated by `src/tools/archgraph.py`. Do not edit.", "",
           "Every file of `src/core` with its role (the first sentence of its opening comment),",
           "what it includes and what includes it. Paths are relative to `src/core`. External",
           "includes (codelet registries, generated headers, the public API) are listed as such.", ""]
    for z in ZONES:
        zf = sorted(r for r, f in files.items() if f["zone"] == z)
        if not zf:
            continue
        out += ["## %s: %s" % (z, ZONE_ROLE[z]), ""]
        folder = None
        for r in zf:
            f = files[r]
            if f["folder"] != folder:
                folder = f["folder"]
                out += ["### %s" % folder, ""]
            out.append("- **%s** (%d lines): %s" % (r, f["lines"], f["role"] or "(no opening comment)"))
            if f["includes"]:
                out.append("  - includes: " + ", ".join(f["includes"]))
            if f["external"]:
                out.append("  - external: " + ", ".join(f["external"]))
            if f["included_by"]:
                out.append("  - included by: " + ", ".join(f["included_by"]))
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- call views
HELPER_FANIN = 6     # a function with this many callers or more is a shared helper
CALL_VIEWS = [
    ("create", "vfft_create", "vfft_create: validation, the layout fork, the tiers"),
    ("execute", "vfft_execute", "vfft_execute: the signature check and the layout fork"),
    ("create_split", "_vfft_split_create", "the SPLIT side of the create fork"),
    ("create_il", "_vfft_il_create", "the INTERLEAVED side of the create fork"),
    ("create_real", "_vfft_create_real", "the 1D real create (through the bridge)"),
    ("execute_split", "_vfft_split_execute", "the SPLIT side of the execute fork"),
    ("execute_il", "_vfft_il_execute", "the INTERLEAVED side of the execute fork"),
    ("execute_real", "_vfft_real_bridge_execute", "the 1D real execute (through the bridge)"),
]


def _helper(funcs, name):
    return len(funcs[name]["callers"]) >= HELPER_FANIN


def call_graph(files, funcs, entry, up=False, maxn=70, maxdepth=4):
    """BFS from entry (callees, or callers with up=True), shared helpers left
    out, as deep as fits in maxn nodes. Returns (mermaid lines, depth, notes)."""
    def nxt(n):
        if up:
            return [(c, "call" if n in funcs[c]["calls"] else "ref") for c in funcs[n]["callers"]]
        return ([(c, "call") for c in funcs[n]["calls"]] + [(c, "ref") for c in funcs[n]["refs"]])
    for depth in range(maxdepth, 0, -1):
        seen, frontier, edges, helpers = {entry: 0}, [entry], set(), set()
        for d in range(1, depth + 1):
            new = []
            for n in frontier:
                for c, kind in nxt(n):
                    if not up and _helper(funcs, c):
                        helpers.add(c)
                        continue
                    edges.add((n, c, kind) if not up else (c, n, kind))
                    if c not in seen:
                        seen[c] = d
                        new.append(c)
            frontier = new
        if len(seen) <= maxn or depth == 1:
            break
    cut = {}
    for n, d in seen.items():
        if d == depth:
            more = [c for c, _ in nxt(n) if c not in seen and (up or not _helper(funcs, c))]
            if more:
                cut[n] = len(set(more))
    lines = ["flowchart LR"]
    byfolder = {}
    for n in seen:
        byfolder.setdefault(files[funcs[n]["file"]]["folder"], []).append(n)
    for fo in sorted(byfolder):
        lines.append('    subgraph %s["%s"]' % (nid("f_" + fo), "src/core" if fo == "(core)" else fo))
        for n in sorted(byfolder[fo]):
            label = "%s%s<br/>%s" % (n, (" +%d" % cut[n]) if n in cut else "",
                                     os.path.basename(funcs[n]["file"]))
            lines.append('        %s["%s"]%s' % (nid("fn_" + n), lbl(label), ":::entry" if n == entry else ""))
        lines.append("    end")
    for a, b, kind in sorted(edges):
        lines.append("    %s %s %s" % (nid("fn_" + a), "-->" if kind == "call" else "-.->", nid("fn_" + b)))
    lines.append("    classDef entry stroke-width:3px")
    return lines, depth, sorted(helpers), cut


def call_view_md(files, funcs, entry, title):
    lines, depth, helpers, cut = call_graph(files, funcs, entry)
    f = funcs[entry]
    out = ["# %s" % title, "",
           "Generated by `src/tools/archgraph.py`. Do not edit.", "",
           "What `%s` (`%s`, line %d) calls, %d level%s deep. A solid arrow is a direct call; a"
           % (entry, f["file"], f["line"], depth, "" if depth == 1 else "s"),
           "dashed arrow is a function passed or stored by name (a thread-pool task, a dispatch",
           "table entry, a plan's function pointer). `+N` on a node: N more callees not drawn",
           "(see `functions.md`, or run `python src/tools/archgraph.py --calls NAME`).", "",
           "```mermaid"] + lines + ["```", ""]
    if helpers:
        out += ["Shared helpers left out (%d or more callers each): %s." % (
            HELPER_FANIN, ", ".join("`%s`" % h for h in helpers)), ""]
    out += ["Best effort: calls through function pointers and macros are not followed; both",
            "sides of every `#if` are read."]
    return "\n".join(out) + "\n"


def functions_md(files, funcs):
    out = ["# Functions of src/core", "",
           "Generated by `src/tools/archgraph.py`. Do not edit.", "",
           "Every function defined in `src/core`, by file: its line, the core functions it calls,",
           "the ones it references by name without calling (tasks, table entries, function",
           "pointers) and its callers. Best effort: calls through function pointers and macros",
           "are not followed. %d functions." % len(funcs), ""]
    byfile = {}
    for n, f in funcs.items():
        byfile.setdefault(f["file"], []).append(n)
    for z in ZONES:
        for rel in sorted(r for r in byfile if files[r]["zone"] == z):
            out += ["## %s" % rel, ""]
            for n in sorted(byfile[rel], key=lambda x: funcs[x]["line"]):
                f = funcs[n]
                parts = []
                if f["calls"]:
                    parts.append("calls " + ", ".join(f["calls"]))
                if f["refs"]:
                    parts.append("refs " + ", ".join(f["refs"]))
                if f["callers"]:
                    c = f["callers"]
                    parts.append("called by " + ", ".join(c[:15]) + (" (+%d more)" % (len(c) - 15) if len(c) > 15 else ""))
                out.append("- `%s` (line %d)%s" % (n, f["line"], (": " + "; ".join(parts)) if parts else ""))
            out.append("")
    return "\n".join(out)


def outputs(files, funcs):
    res = {"zones.md": zones_md(files), "folders.md": folders_md(files), "map.md": map_md(files),
           "functions.md": functions_md(files, funcs)}
    for key, entry, title in CALL_VIEWS:
        if entry in funcs:
            res["calls/%s.md" % key] = call_view_md(files, funcs, entry, title)
    for folder in sorted({f["folder"] for f in files.values()}):
        res["folders/%s.md" % folder_file(folder)] = folder_md(files, folder)
    data = {r: {k: v for k, v in f.items() if k != "path"} for r, f in sorted(files.items())}
    res["graph.json"] = json.dumps({"zones": ZONE_ROLE, "files": data, "functions": funcs},
                                   indent=1, sort_keys=True) + "\n"
    return res


# ---------------------------------------------------------------- focus
def focus(files, what, depth):
    what = what.strip("/")
    if what in ZONES:
        seed = {r for r, f in files.items() if f["zone"] == what}
    elif any(f["folder"] == what for f in files.values()):
        seed = {r for r, f in files.items() if f["folder"] == what}
    else:
        seed = {r for r in files if r == what or os.path.basename(r) == what}
    if not seed:
        sys.exit("archgraph: no file, folder or zone named %r" % what)
    nodes = set(seed)
    frontier = set(seed)
    for _ in range(depth):
        nxt = set()
        for r in frontier:
            nxt.update(files[r]["includes"])
            nxt.update(x for x in files[r]["included_by"] if files[x]["zone"] != "front")
        nxt -= nodes
        nodes |= nxt
        frontier = nxt
    out = ["```mermaid", "flowchart LR"]
    for fo in sorted({files[r]["folder"] for r in nodes}):
        out.append('    subgraph %s["%s"]' % (nid("f_" + fo), fo))
        for r in sorted(x for x in nodes if files[x]["folder"] == fo):
            style = ":::seed" if r in seed else ""
            out.append('        %s["%s"]%s' % (nid(r), os.path.basename(r), style))
        out.append("    end")
    for r in sorted(nodes):
        for t in files[r]["includes"]:
            if t in nodes:
                out.append("    %s --> %s" % (nid(r), nid(t)))
    out += ["    classDef seed stroke-width:3px", "```"]
    print("\n".join(out))
    print()
    for r in sorted(seed):
        print("- %s: %s" % (r, files[r]["role"]))


# ---------------------------------------------------------------- SVG
_BLOCK = re.compile(r"(?:^## (.+)\n(?:.*\n)*?)?```mermaid\n(.*?)```", re.M)


def svg_jobs(want):
    """{svg path relative to svg/: mermaid source} for every diagram in the output."""
    jobs = {}
    for name, text in sorted(want.items()):
        if not name.endswith(".md"):
            continue
        blocks, section = [], None
        for line_block in re.split(r"(?m)^(?=## |```mermaid)", text):
            if line_block.startswith("## "):
                section = line_block[3:].split("\n", 1)[0].strip()
            m = re.match(r"```mermaid\n(.*?)```", line_block, re.S)
            if m:
                blocks.append((section, m.group(1)))
        base = name[:-3]
        for i, (sec, src) in enumerate(blocks):
            if len(blocks) == 1:
                out = base + ".svg"
            else:
                slug = re.sub(r"[^a-z0-9]+", "_", (sec or str(i + 1)).lower()).strip("_")
                out = "%s-%s.svg" % (base, slug)
            jobs[out] = src
    return jobs


def svg_manifest(jobs):
    return json.dumps({k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in sorted(jobs.items())},
                      indent=1, sort_keys=True) + "\n"


def _mmdc():
    if os.environ.get("MMDC"):
        return [os.environ["MMDC"]]
    if shutil.which("mmdc"):
        return [shutil.which("mmdc")]
    if shutil.which("npx"):
        return [shutil.which("npx"), "-y", "@mermaid-js/mermaid-cli"]
    sys.exit("archgraph: --svg needs mermaid-cli (set MMDC, put mmdc on PATH, or install node for npx)")


def render_svgs(jobs):
    cmd = _mmdc()
    tmp = tempfile.mkdtemp(prefix="archgraph_")
    cfg = {"args": ["--no-sandbox"]}
    if os.environ.get("CHROME"):
        cfg["executablePath"] = os.environ["CHROME"]
    pp = os.path.join(tmp, "puppeteer.json")
    with open(pp, "w") as fh:
        json.dump(cfg, fh)
    # plain SVG <text> labels instead of HTML in <foreignObject>: image viewers
    # and editors that are not browsers draw foreignObject as empty boxes
    mc = os.path.join(tmp, "mermaid.json")
    with open(mc, "w") as fh:
        json.dump({"htmlLabels": False, "flowchart": {"htmlLabels": False, "wrappingWidth": 600}}, fh)

    def one(item):
        k, (name, src) = item
        inp = os.path.join(tmp, "%d.mmd" % k)
        with open(inp, "w", encoding="utf-8") as fh:
            fh.write(src)
        out = os.path.join(SVG, name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        r = subprocess.run(cmd + ["-q", "-p", pp, "-c", mc, "-b", "white", "-i", inp, "-o", out],
                           capture_output=True, text=True)
        if r.returncode == 0:
            # mermaid-cli embeds its web font as base64 (~250 KB per file); the
            # SVG names arial / sans-serif after it, so drop the embedded copy
            with open(out, encoding="utf-8") as fh:
                svg = fh.read()
            svg = re.sub(r"@font-face\s*\{[^}]*\}", "", svg)
            with open(out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(svg)
        return name, r.returncode, (r.stderr or r.stdout)[-300:]

    try:
        with concurrent.futures.ThreadPoolExecutor(4) as ex:
            res = list(ex.map(one, enumerate(sorted(jobs.items()))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = [r for r in res if r[1]]
    for name, _, err in bad:
        print("archgraph: render FAILED for %s: %s" % (name, err.strip()))
    if bad:
        sys.exit(1)
    for dp, _, fs in os.walk(SVG):
        for fn in fs:
            rel = os.path.relpath(os.path.join(dp, fn), SVG).replace(os.sep, "/")
            if rel not in jobs and rel != "manifest.json":
                os.remove(os.path.join(dp, fn))
    with open(os.path.join(SVG, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg_manifest(jobs))
    print("archgraph: rendered %d SVG(s) to %s" % (len(jobs), os.path.relpath(SVG, ROOT)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--focus")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--calls", help="print the call graph of one function")
    ap.add_argument("--svg", action="store_true", help="also render docs/architecture/svg/")
    ap.add_argument("--up", action="store_true", help="with --calls: its callers instead")
    a = ap.parse_args()
    files, funcs = scan()
    if a.focus:
        focus(files, a.focus, a.depth)
        return 0
    if a.calls:
        if a.calls not in funcs:
            sys.exit("archgraph: no function named %r in src/core" % a.calls)
        lines, depth, helpers, _ = call_graph(files, funcs, a.calls, up=a.up,
                                              maxn=10 ** 6 if a.depth > 1 else 70,
                                              maxdepth=a.depth if a.depth > 1 else 4)
        print("```mermaid\n" + "\n".join(lines) + "\n```")
        f = funcs[a.calls]
        print("\n%s: %s line %d; %d level(s) of %s" % (a.calls, f["file"], f["line"], depth,
                                                      "callers" if a.up else "callees"))
        if helpers:
            print("shared helpers left out: " + ", ".join(helpers))
        return 0
    want = outputs(files, funcs)
    if a.check:
        stale = []
        for name, text in sorted(want.items()):
            p = os.path.join(OUT, name)
            if not os.path.isfile(p) or open(p, encoding="utf-8").read() != text:
                stale.append(name)
        if os.path.isdir(OUT):
            for dp, _, fs in os.walk(OUT):
                for fn in fs:
                    rel = os.path.relpath(os.path.join(dp, fn), OUT).replace(os.sep, "/")
                    if rel not in want:
                        stale.append(rel + " (no longer generated)")
        mf = os.path.join(SVG, "manifest.json")
        if os.path.isdir(SVG) and (not os.path.isfile(mf) or
                                   open(mf, encoding="utf-8").read() != svg_manifest(svg_jobs(want))):
            stale.append("../svg/ (run with --svg to re-render)")
        if stale:
            print("archgraph: %d stale file(s) in docs/architecture/generated/ "
                  "(run python src/tools/archgraph.py):" % len(stale))
            for s in stale:
                print("  " + s)
            return 1
        print("archgraph: docs/architecture/generated/ is current")
        return 0
    for name, text in want.items():
        p = os.path.join(OUT, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    for dp, _, fs in os.walk(OUT):
        for fn in fs:
            rel = os.path.relpath(os.path.join(dp, fn), OUT).replace(os.sep, "/")
            if rel not in want:
                os.remove(os.path.join(dp, fn))
    print("archgraph: wrote %d file(s) to %s" % (len(want), os.path.relpath(OUT, ROOT)))
    if a.svg:
        render_svgs(svg_jobs(want))
    return 0


if __name__ == "__main__":
    sys.exit(main())
