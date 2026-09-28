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
  graph.json        the same data, for any other tool

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
"""
import argparse
import json
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
    for f in files.values():
        del f["text"]
        f["included_by"] = []
    for rel, f in sorted(files.items()):
        for t in f["includes"]:
            files[t]["included_by"].append(rel)
    return files


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


def outputs(files):
    res = {"zones.md": zones_md(files), "folders.md": folders_md(files), "map.md": map_md(files)}
    for folder in sorted({f["folder"] for f in files.values()}):
        res["folders/%s.md" % folder_file(folder)] = folder_md(files, folder)
    data = {r: {k: v for k, v in f.items() if k != "path"} for r, f in sorted(files.items())}
    res["graph.json"] = json.dumps({"zones": ZONE_ROLE, "files": data}, indent=1, sort_keys=True) + "\n"
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--focus")
    ap.add_argument("--depth", type=int, default=1)
    a = ap.parse_args()
    files = scan()
    if a.focus:
        focus(files, a.focus, a.depth)
        return 0
    want = outputs(files)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
