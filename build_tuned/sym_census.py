#!/usr/bin/env python3
"""sym_census.py - stable symbol and mutable-object censuses for an object file.

WHY NORMALIZATION IS REQUIRED
-----------------------------
A raw `nm | sort` is NOT a stable artifact. GCC emits static tables and switch
tables under auto-numbered names - CSWTCH.1398, primes.4521, POOL.907 - and the
numeric suffix is an arbitrary per-translation-unit counter. Move any code and
the counter shifts, so a pure relocation shows up as

    < r CSWTCH.1398
    > r CSWTCH.1399

which is a diff with no meaning whatsoever. Measured on the step-5 pilot move:
178 of 2650 defined symbols carry such a suffix, and a 16-line relocation that
obj_equiv proved EQUIVALENT still moved one of them. A census that cries wolf
on a provably-identical move is worse than no census, because the next real
finding gets waved through with it.

WHAT IS NORMALIZED, AND WHAT DELIBERATELY IS NOT
-----------------------------------------------
Only a TRAILING run of digits is collapsed to `.N`:

    CSWTCH.1398                  -> CSWTCH.N              (counter, meaningless)
    _calibrate_c2c.constprop.0   -> _calibrate_c2c.constprop.N

The optimisation MARKERS survive - `.part`, `.constprop`, `.isra`. Those are
semantic: a function that stops being cloned, or starts being, is a real change
to what the compiler did and the census must still show it. Only the counter
after the marker is dropped.

Symbol CLASS is kept (T/t/B/b/D/d/R/r), so external-vs-static is still visible.
That matters: a `b` becoming a `B` means a static gained external linkage, which
is exactly the property the migration relies on to turn a duplicated counter
into a link error rather than a silent second copy.

USAGE
  python build_tuned/sym_census.py <obj> --defined     defined symbols
  python build_tuned/sym_census.py <obj> --undefined   undefined symbols
  python build_tuned/sym_census.py <obj> --mutable     file-scope mutable objects
"""
import re
import os
import subprocess
# env NM overrides the historical mingw152 path (2026-09-03).
DEFAULT_NM = os.environ.get("NM", "C:/mingw152/mingw64/bin/nm.exe")
import sys

TRAILING_NUM = re.compile(r"\.\d+$")


def normalize(name):
    """Collapse a trailing counter; keep .part/.constprop/.isra markers."""
    prev = None
    while prev != name:                 # ".constprop.0" -> ".constprop"
        prev = name
        name = TRAILING_NUM.sub(".N", name)
        if name.endswith(".N.N"):
            name = name[:-2]
    return name


def census(obj, mode, nm=DEFAULT_NM):
    if mode == "undefined":
        out = subprocess.run([nm, "-u", obj], capture_output=True, text=True,
                             check=True).stdout
        rows = {normalize(l.split()[-1]) for l in out.splitlines() if l.strip()}
        return sorted(rows)

    out = subprocess.run([nm, "--defined-only", obj], capture_output=True,
                         text=True, check=True).stdout
    rows = []
    for l in out.splitlines():
        p = l.split()
        if len(p) < 2:
            continue
        cls, name = p[-2], p[-1]
        if len(cls) != 1:
            continue
        if mode == "mutable" and cls not in "bBdD":
            continue
        rows.append("%s %s" % (cls, normalize(name)))
    return sorted(rows)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    modes = [a[2:] for a in sys.argv[1:] if a.startswith("--")]
    if len(args) != 1 or len(modes) != 1 or modes[0] not in (
            "defined", "undefined", "mutable"):
        print(__doc__)
        return 2
    for row in census(args[0], modes[0]):
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
