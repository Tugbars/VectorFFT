#!/usr/bin/env python3
"""hygiene.py - source-level checks of rung R0: header names, includes, and the
layout dependency rules.

  dup_basenames()        two headers under src/core with one basename. The
                         build puts EVERY core directory on -I, so a duplicate
                         silently changes which file `#include "x.h"` means.
  unresolved_includes()  every quoted #include (core, gauntlet, benches, tools)
                         that resolves to no file: a move that forgot an
                         include path. Compared against the reference, since a
                         few are optional (guarded) by design.
  dep_violations()       docs/roadmap/layout_separation_plan.md section 4:
                           common/ includes only common/
                           split/  never includes il/ or bridge/
                           il/     never includes split/ (bridge/ is allowed only
                                   through bridge/real_doors.h)
                         Active only once src/core/common exists.

USAGE
  python hygiene.py            print all three
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toolchain  # noqa: E402

ROOT = toolchain.ROOT
CORE = toolchain.CORE
_INC = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
SCAN_DIRS = [CORE, os.path.join(ROOT, "gauntlet"), os.path.join(ROOT, "build_tuned", "benches"),
             os.path.join(ROOT, "src", "tools")]


def _files(d, exts=(".h", ".c")):
    for dp, _, fs in os.walk(d):
        for f in fs:
            if f.endswith(exts):
                yield os.path.join(dp, f)


def core_headers():
    return sorted(_files(CORE))


def dup_basenames():
    seen = {}
    for p in core_headers():
        seen.setdefault(os.path.basename(p), []).append(os.path.relpath(p, ROOT))
    return {b: ps for b, ps in seen.items() if len(ps) > 1}


def resolve(inc, from_file, dirs):
    cands = [os.path.join(os.path.dirname(from_file), inc)] + [os.path.join(d, inc) for d in dirs]
    for c in cands:
        if os.path.isfile(c):
            return os.path.normpath(c)
    return None


def unresolved_includes():
    dirs = toolchain.include_dirs()
    rows = set()
    for d in SCAN_DIRS:
        if not os.path.isdir(d):
            continue
        for f in _files(d):
            if "/_build/" in f or "/.obj/" in f:
                continue
            try:
                text = open(f, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for inc in _INC.findall(text):
                if resolve(inc, f, dirs) is None:
                    rows.add("%s: %s" % (os.path.relpath(f, ROOT), inc))
    return sorted(rows)


def _zone(path):
    rel = os.path.relpath(path, CORE).replace(os.sep, "/")
    top = rel.split("/", 1)[0]
    return top if top in ("common", "split", "il", "bridge") else "front"


def dep_violations():
    if not os.path.isdir(os.path.join(CORE, "common")):
        return None                     # the rules start with the new tree
    dirs = toolchain.include_dirs()
    bad = []
    for f in core_headers():
        z = _zone(f)
        if z == "front":
            continue
        text = open(f, encoding="utf-8", errors="replace").read()
        for inc in _INC.findall(text):
            tgt = resolve(inc, f, dirs)
            if tgt is None or not tgt.startswith(CORE):
                continue
            tz = _zone(tgt)
            ok = (tz == z or tz == "common"
                  or (z == "bridge" and tz in ("split", "il"))
                  or (z == "il" and tz == "bridge"
                      and os.path.basename(tgt) == "real_doors.h"))
            if not ok:
                bad.append("%s (%s) includes %s (%s)"
                           % (os.path.relpath(f, ROOT), z, os.path.relpath(tgt, ROOT), tz))
    return bad


if __name__ == "__main__":
    print("duplicate basenames:", dup_basenames() or "none")
    u = unresolved_includes()
    print("unresolved includes: %d" % len(u))
    for r in u:
        print("  " + r)
    d = dep_violations()
    print("dependency rules:", "not active (no src/core/common yet)" if d is None
          else ("clean" if not d else "%d violations" % len(d)))
    for r in d or []:
        print("  " + r)
