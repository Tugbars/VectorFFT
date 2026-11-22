"""nd_sweep.py -- the correctness sweep of the rank-4 to 7 complex transforms: nd_probe over a fixed list of shapes.

Every shape runs with estimate plans (VFFT_FLAG_ESTIMATE, nothing measured, no wisdom) out of place on one thread,
and a part of them in place and at 4 and 8 threads. A shape of up to DIRECT_MAX points is checked against the direct
DFT along every axis, a larger one against the library's 1D plans line by line. The list is fixed: it comes from
arithmetic alone, the same on every machine.

    python tests/rank_n/nd_sweep.py [--probe build/nd_probe.exe] [--quick] [--store DIR]

--store DIR also plans a small part of the list by measuring, in that scratch wisdom folder (never the library's own).
Prints one line a failure and a summary; exit 1 when any run failed.
"""
import argparse
import os
import subprocess
import sys

DIRECT_MAX = 70000      # points: the direct reference is O(points x extent) an axis


def shapes():
    out = []
    # powers of two
    out += ["4x4x4x4", "8x8x8x8", "16x16x16x16", "32x32x32x32", "2x4x8x16", "16x8x4x2", "32x4x4x4", "4x4x4x32",
            "2x2x2x2", "64x8x8x8", "8x8x8x64", "4x16x4x16",
            "4x4x4x4x4", "8x8x8x8x8", "16x16x16x16x16", "2x4x8x4x2", "16x2x2x2x16",
            "4x4x4x4x4x4", "8x8x8x8x8x8", "2x4x2x4x2x4",
            "2x2x2x2x2x2x2", "4x4x4x4x4x4x4", "8x8x8x8x8x8x8", "2x4x8x2x4x8x2"]
    # composites and odd extents
    out += ["3x5x7x9", "9x7x5x3", "6x10x14x18", "5x5x5x5", "7x7x7x7", "12x12x12x12", "10x10x10x10", "15x6x4x9",
            "25x4x6x10", "20x20x20x20", "24x18x12x6",
            "3x4x5x6x7", "7x6x5x4x3", "6x6x6x6x6", "10x10x10x10x10", "5x4x3x4x5", "12x3x5x2x9",
            "2x3x4x5x6x7", "6x6x6x6x6x6", "3x3x3x3x3x3", "7x2x5x2x3x2",
            "3x4x2x5x2x3x4", "3x3x3x3x3x3x3", "2x3x2x3x2x3x2", "5x4x3x2x3x4x5"]
    # a prime extent above 47 on each axis (no kernel of its own: a Rader stage, Bluestein, the turned prime pass)
    out += ["53x4x4x4", "4x53x4x4", "4x4x53x4", "4x4x4x53", "61x6x5x4", "5x6x7x61", "127x8x8x8", "8x8x8x127",
            "4x4x4x4x53", "53x4x4x4x4", "3x4x59x4x3", "2x2x2x2x2x2x53", "53x2x2x2x2x2x2", "4x4x4x67x4x4",
            "101x3x3x3", "3x3x3x101"]
    # primes up to 47 and their products
    out += ["17x4x4x4", "4x4x4x17", "31x5x5x5", "47x4x3x2", "13x11x7x5", "23x4x4x4x4", "4x4x4x4x29", "37x3x2x3x2x3"]
    # larger arrays (the reference by lines)
    out += ["64x64x64x64", "48x48x48x48", "128x32x32x32", "32x32x32x128", "100x40x20x10", "12x12x12x12x12",
            "32x16x16x16x16", "16x16x16x16x32", "10x10x10x10x10x10", "8x8x8x8x8x16", "6x6x6x6x6x6x6", "4x4x4x8x8x8x8",
            "256x16x16x16", "16x16x16x256", "81x27x9x3"]
    return out


def points(sh):
    p = 1
    for t in sh.split("x"):
        p *= int(t)
    return p


def run(probe, sh, extra):
    ref = "direct" if points(sh) <= DIRECT_MAX else "lines"
    cmd = [probe, sh, "--ref", ref, "--reps", "1"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    lines = [l for l in r.stdout.splitlines() if l.strip()]   # the result line is the probe's stdout; a measured
                                                              # create's notices go to stderr
    last = lines[-1] if lines else "(no result line) " + " | ".join(r.stderr.splitlines()[-2:])
    return r.returncode == 0 and last.rstrip().endswith("ok"), last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="build/nd_probe.exe")
    ap.add_argument("--quick", action="store_true", help="arrays of up to 300,000 points only")
    ap.add_argument("--store", default=None, help="a scratch wisdom folder: also plan a part of the list by measuring")
    a = ap.parse_args()
    a.probe = os.path.abspath(a.probe)   # a bare relative name is not found where the current folder is off the path
    lst =[s for s in shapes() if not a.quick or points(s) <= 300000]
    runs = []
    for i, sh in enumerate(lst):
        runs.append((sh, ["--est"]))
        if i % 2 == 0:
            runs.append((sh, ["--est", "--ip"]))
        if points(sh) >= 256:
            runs.append((sh, ["--est", "--threads", "4"]))
            runs.append((sh, ["--est", "--threads", "8"]))
        if i % 3 == 0:
            runs.append((sh, ["--est", "--threads", "8", "--ip"]))
        if a.store and points(sh) <= 70000 and i % 4 == 0:
            runs.append((sh, ["--store", a.store]))
            runs.append((sh, ["--store", a.store, "--threads", "8"]))
    bad = 0
    for sh, extra in runs:
        ok, last = run(a.probe, sh, extra)
        if not ok:
            bad += 1
            print("FAIL %s %s: %s" % (sh, " ".join(extra), last))
            sys.stdout.flush()
    print("%d shapes, %d runs, %d failed" % (len(lst), len(runs), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
