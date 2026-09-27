#!/usr/bin/env python3
"""relayout.py - move core files by a map and rewrite every include that names
them by path, in the style it was written.

WHY A TOOL
----------
Phases 2-3 of docs/roadmap/layout_separation_plan.md move ~100 headers. The
build puts every src/core directory on -I, so a BARE include ("ztt.h") keeps
resolving wherever the file goes (basenames are unique, hygiene.py enforces it).
What breaks is a PATH-QUALIFIED include - about 36 files in core, plus gates,
benches and the gauntlet that reach in by path ("oop/ztt.h",
"../../wisdom2/wisdom2_fftnd.h", "../../src/core/oop/il2p.h"). By hand that is
the kind of edit a gate catches only as a build break in a file nobody built.

Each quoted include containing '/' is resolved against the OLD tree exactly as
the compiler would (the including file's directory first, then the -I set). If
it lands on a moved file, it is rewritten to reach the new location in the SAME
style: relative to the including file if it was written that way, else relative
to the -I directory it resolved through. Includes are never turned bare or
path-qualified; the text changes only where the path must.

MAP FORMAT: one move per line, paths relative to src/core, '#' comments:
    support/threads.h        common/support/threads.h
A directory maps every file under it.

USAGE
  python relayout.py MAP [--dry-run]      (from anywhere; operates on the repo)
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toolchain  # noqa: E402

ROOT, CORE = toolchain.ROOT, toolchain.CORE
SCAN = [CORE, os.path.join(ROOT, "gauntlet"), os.path.join(ROOT, "build_tuned"),
        os.path.join(ROOT, "src", "tools")]
_INC = re.compile(r'(#\s*include\s+")([^"]*/[^"]*)(")')


def load_map(path):
    moves = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        a, b = line.split()
        src, dst = os.path.join(CORE, a), os.path.join(CORE, b)
        if os.path.isdir(src):
            for dp, _, fs in os.walk(src):
                for f in fs:
                    p = os.path.join(dp, f)
                    moves[os.path.normpath(p)] = os.path.normpath(
                        os.path.join(dst, os.path.relpath(p, src)))
        elif os.path.isfile(src):
            moves[os.path.normpath(src)] = os.path.normpath(dst)
        else:
            raise SystemExit("relayout: %s does not exist" % a)
    return moves


def sources():
    for d in SCAN:
        for dp, dns, fs in os.walk(d):
            dns[:] = [x for x in dns if x not in ("_build", ".obj", "__pycache__")]
            for f in fs:
                if f.endswith((".c", ".h")):
                    yield os.path.normpath(os.path.join(dp, f))


def plan(moves):
    dirs = [os.path.normpath(d) for d in toolchain.include_dirs()]
    edits = {}
    for f in sources():
        text = open(f, encoding="utf-8", errors="surrogateescape").read()
        new_f = moves.get(f, f)
        changed = False

        def fix(m):
            nonlocal changed
            inc = m.group(2)
            base = None
            tgt = os.path.normpath(os.path.join(os.path.dirname(f), inc))
            if os.path.isfile(tgt):
                style = "file"
            else:
                style = None
                for d in dirs:
                    t = os.path.normpath(os.path.join(d, inc))
                    if os.path.isfile(t):
                        tgt, style, base = t, "dir", d
                        break
            if style is None:
                return m.group(0)
            new_t = moves.get(tgt, tgt)
            if new_t == tgt and new_f == f:
                return m.group(0)
            if style == "file":
                rel = os.path.relpath(new_t, os.path.dirname(new_f))
            else:
                new_base = moves.get(base, base)     # an -I dir does not move
                rel = os.path.relpath(new_t, new_base)
                if rel.startswith(".."):
                    # the -I dir it went through no longer holds it: reach it
                    # from src/core, which is always on the path
                    rel = os.path.relpath(new_t, CORE)
            rel = rel.replace(os.sep, "/")
            if rel != inc:
                changed = True
                return m.group(1) + rel + m.group(3)
            return m.group(0)

        new_text = _INC.sub(fix, text)
        if changed:
            edits[f] = new_text
    return edits


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    moves = load_map(sys.argv[1])
    dry = "--dry-run" in sys.argv
    edits = plan(moves)
    for f, t in sorted(edits.items()):
        print("rewrite %s" % os.path.relpath(f, ROOT))
    for a, b in sorted(moves.items()):
        print("move    %s -> %s" % (os.path.relpath(a, ROOT), os.path.relpath(b, ROOT)))
    if dry:
        return 0
    for f, t in edits.items():
        with open(f, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
            fh.write(t)
    for a, b in sorted(moves.items()):
        os.makedirs(os.path.dirname(b), exist_ok=True)
        r = subprocess.run(["git", "mv", a, b], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("git mv %s failed: %s" % (a, r.stderr))
    # drop directories the moves emptied
    for dp, dns, fs in sorted(os.walk(CORE), key=lambda x: -len(x[0])):
        if not os.listdir(dp):
            os.rmdir(dp)
    print("relayout: %d files moved, %d files rewritten" % (len(moves), len(edits)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
