#!/usr/bin/env python3
"""step_gate.py - is this step of the layout separation a behaviour change?

Captures the current tree (capture_ref.py, the same code that captured the
reference) into a scratch directory and compares it with a reference, rung by
rung (docs/roadmap/layout_separation_plan.md, section 7):

  R0 hygiene    no duplicate core header names; no NEW unresolved include; the
                race census found the same number of sites (a file may lose
                sites only if another gains them); no new warnings; from phase 7
                on, the dependency rules (hygiene.py)
  R1 bytes      the codelets (always identical), then the two vfft.c objects.
                Identical objects make R2-R4 hold by construction
  R2 source     vfft.i identical; or, with --allow-reorder, the same top-level
                declarations and macros in another order. R2b: the object's
                read-only strings, always identical (no path may leak)
  R3 objects    obj_equiv --strict-data on both objects: code, immediates,
                struct offsets, constants by content, data objects
  R4 censuses   symbols (undefined, mutable, defined), race protocols with
                fn= masked, struct layout
  R5 semantics  golden bits, plan fingerprints, the API sweep, the wisdom
                replay, the store round trip (and gates.txt if both have it),
                byte-identical after LF normalization

Every rung runs and is reported; the verdict is PASS only when every GATED rung
is green. What a step is allowed to change is stated on the command line, never
inferred:

  --allow-reorder          R2 may pass on the sorted form
  --allow-changed F1,F2    R3 may report these functions (and only these)
                           changed, gone or new - a split step names the
                           functions it splits
  --rename-map FILE        R3/R4: "old new" per line, applied to the reference
  --allow-defined          R4: the defined-symbol census may change (a split
                           adds functions); undefined and mutable never may
  --allow-layout           R4: layout.txt may change (phase 8, the struct split)
  --enforce-deps           R0: the common/split/il dependency rules are GATED
                           (phase 7 on); before that they are reported, the
                           list of violations being the remaining work
  --code-change            a step that changes code ON PURPOSE (deletion, a
                           function split): R2, R3 code and the defined census
                           are reported, not gated; data objects, undefined and
                           mutable symbols, read-only strings and every R5
                           artifact stay gated. The step's commit lists the
                           changed functions
  --allow-census-move      R0: race sites may move between files (counts kept)
  --allow-census-removed F1.h,..  R0: these files were DELETED; their reference
                           sites are subtracted before the totals are compared

USAGE
  python step_gate.py --ref REFDIR --isa avx2|avx512 [--scratch DIR]
        [--cur DIR (compare an existing capture; no capture)] [--no-semantic]
        [--jobs J] [--repeat N] [--cmake-dir DIR] [--gates] [allowances]
Exit 0 on PASS. Writes DIR/step_gate.json and prints one table.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toolchain  # noqa: E402
import capture_ref  # noqa: E402
import hygiene  # noqa: E402


def opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def read(d, f):
    p = os.path.join(d, f)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return fh.read().replace(b"\r\n", b"\n")


def same(ref, cur, f):
    a, b = read(ref, f), read(cur, f)
    if a is None and b is None:
        return None                        # not captured on either side
    return a == b


def diff_lines(ref, cur, f, limit=12):
    a = (read(ref, f) or b"").decode("utf-8", "replace").splitlines()
    b = (read(cur, f) or b"").decode("utf-8", "replace").splitlines()
    sa, sb = set(a), set(b)
    out = ["- " + l for l in a if l not in sb][:limit] + ["+ " + l for l in b if l not in sa][:limit]
    return out


class Gate:
    def __init__(self):
        self.rows = []

    def add(self, rung, check, ok, detail="", gated=True):
        state = "skip" if ok is None else ("PASS" if ok else ("FAIL" if gated else "note"))
        self.rows.append(dict(rung=rung, check=check, state=state, detail=detail, gated=gated))

    def verdict(self):
        return all(r["state"] != "FAIL" for r in self.rows)


def compare(ref, cur, allow, strict_objdump):
    g = Gate()

    # ---- R0 hygiene
    dup = read(cur, "dup_basenames.txt")
    g.add("R0", "no duplicate core header names", dup is not None and dup.strip() == b"",
          (dup or b"").decode()[:300])
    new_unres = [l[2:] for l in diff_lines(ref, cur, "includes_unresolved.txt", 50) if l.startswith("+")]
    g.add("R0", "no new unresolved #include", not new_unres, "; ".join(new_unres[:6]))
    rc = _census_counts(ref, allow["census_removed"]), _census_counts(cur, [])
    moved = rc[0][1] != rc[1][1]
    g.add("R0", "race sites: same total", rc[0][0] == rc[1][0],
          "timing/verdicts %s -> %s%s" % (rc[0][0], rc[1][0],
          (" (reference minus deleted %s)" % ",".join(allow["census_removed"]))
          if allow["census_removed"] else ""))
    g.add("R0", "race sites: same files", not moved,
          "; ".join(diff_lines(ref, cur, "race_census_files.txt", 6)),
          gated=not allow["census_move"])
    new_warn = [l for l in diff_lines(ref, cur, "warnings.txt", 50) if l.startswith("+")]
    gone_warn = [l for l in diff_lines(ref, cur, "warnings.txt", 50) if l.startswith("-")]
    g.add("R0", "no new warnings", not new_warn,
          "; ".join(new_warn[:6]) or ("%d warnings gone" % len(gone_warn) if gone_warn else ""))
    dv = hygiene.dep_violations()
    g.add("R0", "layout dependency rules", None if dv is None else not dv,
          ("%d violations: " % len(dv) if dv else "") + "; ".join((dv or [])[:4]),
          gated=allow["enforce_deps"])

    # ---- R1 bytes
    g.add("R1", "codelet objects + libdagcodelets.a identical", same(ref, cur, "codelets.sha"),
          "; ".join(diff_lines(ref, cur, "codelets.sha", 4)))
    objs = {}
    for o in ("vfft_O2.o", "vfft_O3native.o"):
        a, b = read(ref, o), read(cur, o)
        objs[o] = a is not None and a == b
        g.add("R1", "%s byte-identical" % o, objs[o], gated=False)
    bytes_ok = all(objs.values())
    g.add("R1", "harness binaries identical", same(ref, cur, "bins.sha"), gated=False)
    g.add("R1", "cmake artifacts identical", same(ref, cur, "cmake.sha"),
          "; ".join(diff_lines(ref, cur, "cmake.sha", 4)), gated=False)

    # ---- R2 preprocessed source
    i_same = same(ref, cur, "vfft.i")
    if bytes_ok or i_same:
        g.add("R2", "vfft.i identical" if i_same else "vfft.i (objects identical)", True)
    else:
        sorted_ok = same(ref, cur, "vfft.i.sorted") and _macros(ref) == _macros(cur)
        g.add("R2", "vfft.i identical", False, "text differs",
              gated=not (allow["reorder"] or allow["code_change"]))
        g.add("R2", "same declarations and macros (reorder)", sorted_ok,
              "; ".join(diff_lines(ref, cur, "vfft.i.sorted", 3) + diff_lines(ref, cur, "vfft.macros", 3))[:600],
              gated=not allow["code_change"])
    g.add("R2b", "read-only strings identical", same(ref, cur, "rodata_strings.txt"),
          "; ".join(diff_lines(ref, cur, "rodata_strings.txt", 4)))

    # ---- R3 strict objects
    for o in ("vfft_O2.o", "vfft_O3native.o"):
        if objs[o]:
            g.add("R3", "%s strict-equivalent" % o, True, "byte-identical")
            continue
        ok, detail, data_ok = _strict(os.path.join(ref, o), os.path.join(cur, o), allow, strict_objdump)
        g.add("R3", "%s strict-equivalent" % o, ok, detail, gated=not allow["code_change"])
        g.add("R3", "%s data objects unchanged" % o, data_ok, "")

    # ---- R4 censuses
    g.add("R4", "undefined symbols identical", same(ref, cur, "sym_undefined.txt"),
          "; ".join(diff_lines(ref, cur, "sym_undefined.txt", 6)))
    g.add("R4", "mutable objects identical", same(ref, cur, "sym_mutable.txt"),
          "; ".join(diff_lines(ref, cur, "sym_mutable.txt", 6)))
    g.add("R4", "defined symbols identical", _renamed_same(ref, cur, "sym_defined.txt", allow["rename"]),
          "; ".join(diff_lines(ref, cur, "sym_defined.txt", 6)),
          gated=not (allow["defined"] or allow["code_change"]))
    g.add("R4", "race protocols identical (fn masked)", _census_masked(ref) == _census_masked(cur),
          "; ".join(diff_lines(ref, cur, "race_census.txt", 4)),
          gated=not allow["census_removed"])
    g.add("R4", "struct layout identical", same(ref, cur, "layout.txt"),
          "; ".join(diff_lines(ref, cur, "layout.txt", 6)), gated=not allow["layout"])

    # ---- R5 semantics
    for f in ("golden_bits.txt", "fp_replay.txt", "api_sweep.txt", "wisdom_replay.txt",
              "roundtrip.txt", "gates.txt"):
        s = same(ref, cur, f)
        if s is None or (read(ref, f) is None) != (read(cur, f) is None):
            g.add("R5", f, None, "not captured on both sides")
            continue
        g.add("R5", f, s, "; ".join(diff_lines(ref, cur, f, 6))[:800])
    return g


_GUARD = re.compile(r"^#define [A-Z0-9_]+_H\s*$")


def _macros(d):
    """The -dM macro set without empty include guards: carving a header out
    adds its guard and nothing else a reader could observe."""
    t = (read(d, "vfft.macros") or b"").decode().splitlines()
    return [l for l in t if not _GUARD.match(l)]


def _census_counts(d, removed):
    t = (read(d, "race_census_files.txt") or b"").decode().splitlines()
    tot = [0, 0]
    files = []
    for l in t:
        p = l.split()
        if len(p) == 3 and not l.startswith("#") and p[0] not in removed:
            tot[0] += int(p[1])
            tot[1] += int(p[2])
            files.append(l)
    return tuple(tot), files


def _census_masked(d):
    t = (read(d, "race_census.txt") or b"").decode().splitlines()
    return sorted(re.sub(r"\bfn=\S*", "fn=*", re.sub(r"(verdict .*?)fn=\S*", r"\1", l)) for l in t)


def _renamed_same(ref, cur, f, rename_path):
    a = (read(ref, f) or b"").decode().splitlines()
    b = (read(cur, f) or b"").decode().splitlines()
    if rename_path:
        m = {}
        for line in open(rename_path):
            p = line.split()
            if len(p) == 2 and not line.startswith("#"):
                m[p[0]] = p[1]
        a = sorted(" ".join(x.split()[:-1] + [m.get(x.split()[-1], x.split()[-1])]) for x in a if x.split())
    return a == b


def _strict(a, b, allow, objdump):
    cmd = [sys.executable, os.path.join(HERE, "obj_equiv.py"), a, b, "--strict-data",
           "--objdump", objdump]
    if allow["rename"]:
        cmd += ["--rename-map", allow["rename"]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = r.stdout.splitlines()
    named = [l.split(":", 1)[1].strip() for l in lines
             if l.strip().startswith(("CHANGED", "DISAPPEARED", "APPEARED"))]
    summary = "; ".join(l for l in lines if l.startswith(("functions:", "data objects:")))
    dl = [l for l in lines if l.startswith("data objects:")]
    data_ok = bool(dl) and "(changed 0, gone 0, new 0)" in dl[0]
    if r.returncode == 0:
        return True, summary, True
    extra = [n for n in named if n not in allow["changed"]]
    if allow["changed"] and not extra:
        return True, summary + " (all changes allowed: %s)" % ",".join(named[:8]), data_ok
    return False, summary + " | " + ", ".join(extra[:10]), data_ok


def main():
    ref = opt("--ref")
    isa = opt("--isa")
    if not ref or isa not in toolchain.ISAS:
        print(__doc__)
        return 2
    ref = os.path.abspath(ref)
    cur = opt("--cur")
    if not cur:
        cur = os.path.abspath(opt("--scratch") or tempfile.mkdtemp(prefix="vfft_step_"))
        store = os.path.join(ref, "sweep_store")
        capture_ref.capture(isa, cur, semantic="--no-semantic" not in sys.argv,
                            sweep_store=store if os.path.isdir(store) else None,
                            repeat=int(opt("--repeat", "6")), jobs=int(opt("--jobs", "4")),
                            cmake_dir=opt("--cmake-dir"), run_gates="--gates" in sys.argv)
    allow = dict(reorder="--allow-reorder" in sys.argv,
                 changed=[x for x in (opt("--allow-changed") or "").split(",") if x],
                 rename=opt("--rename-map"), defined="--allow-defined" in sys.argv,
                 layout="--allow-layout" in sys.argv,
                 census_move="--allow-census-move" in sys.argv,
                 code_change="--code-change" in sys.argv,
                 enforce_deps="--enforce-deps" in sys.argv,
                 census_removed=[x for x in (opt("--allow-census-removed") or "").split(",") if x])
    g = compare(ref, cur, allow, toolchain.objdump())

    print("\n%-4s %-6s %-48s %s" % ("rung", "state", "check", "detail"))
    for r in g.rows:
        print("%-4s %-6s %-48s %s" % (r["rung"], r["state"], r["check"], r["detail"][:160]))
    ok = g.verdict()
    print("\nSTEP GATE: %s   (ref %s, cur %s)" % ("PASS" if ok else "FAIL", ref, cur))
    with open(os.path.join(cur, "step_gate.json"), "w") as f:
        json.dump(dict(ref=ref, cur=cur, isa=isa, allow=allow, pass_=ok, rows=g.rows), f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
