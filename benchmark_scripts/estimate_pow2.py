"""estimate_pow2.py -- the estimate against the planner at powers of two: the library's estimate plan
(VFFT_FLAG_ESTIMATE, planned by formulas, nothing measured) against its own measured plan (the planner's,
banked in the wisdom store) for the 1D, 2D and 3D complex transform, in one process per set:

    python benchmark_scripts/estimate_pow2.py [options]
    ./benchmark_scripts/run_estimate_pow2.sh [options]

    --isa avx512|avx2   the codelet tree to compile (default avx512)
    --sets 1d,2d,3d     which sets (default all three)
    --rounds R          timed rounds per cell (default 15)
    --threads T         2D and 3D at T threads (default 1; the 1D tool times one thread only, so 1d is skipped)
    --store DIR         the wisdom store to start from (default: this CPU's shipped folder, as the library names it)
    --cells "..."       replace the lists: N for 1D, N1xN2 for 2D, N1xN2xN3 for 3D
    --name NAME         the run directory under gauntlet/results (default estimate-pow2-<isa>-<date>)
    --no-build          use the tools as built

The tools are src/estimate/estimate_race.c (1D) and src/estimate/estimate_race_nd.c (2D, 3D). Per cell each
times three plans, the banked plan (B), the estimate plan (E) and the banked plan created again (B2, the noise
floor), and prints E/B: below 1 the estimate plan is faster. They run against a COPY of the store in the run
directory, never the shipped one. A cell whose banked row is missing has its plan measured by the planner
first: the first pass at one round races and banks those rows into the copy (the tools skip a cell whose
create measured), the second pass times. The 1D and the one-thread protocol pin logical CPU 2 at HIGH
priority; run on a quiet machine. Windows only (the tools use the Windows timer and affinity calls).
Output: <run>/1d.txt, 2d.txt, 3d.txt (each pass), and SUMMARY.md with every cell's E/B and the geometric
means. Needs the gauntlet built for the store lookup (build_avx512.sh / build_avx2.sh) unless --store is given.
"""
import argparse
import datetime
import math
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAUNTLET = os.path.join(ROOT, "gauntlet")
EST = os.path.join(ROOT, "src", "estimate")
EXE = ".exe" if os.name == "nt" else ""

# every power of two 64 .. 2^23 is the 1D tool's own default list
CELLS_2D = ["32x32", "64x64", "128x128", "256x256", "512x512", "1024x1024", "2048x2048",
            "64x256", "256x64", "128x512", "512x128", "32x512", "512x32", "16x1024", "1024x16",
            "4x2048", "2048x4", "128x2048", "2048x128", "8x8192", "8192x8", "64x4096", "4096x64",
            "256x4096", "4096x256", "1024x4096", "4096x1024", "512x8192", "8192x512"]
CELLS_3D = ["8x8x8", "16x16x16", "32x32x32", "64x64x64", "128x128x128",
            "4x256x256", "256x4x256", "256x256x4", "8x64x64", "64x8x64", "64x64x8",
            "16x128x128", "128x16x128", "128x128x16", "32x64x128", "128x64x32",
            "512x4x4", "4x4x512", "1024x16x16", "16x16x1024", "256x32x32"]


def build(isa):
    env = os.environ.copy()
    env["VFFT_ISA"] = isa
    env["PYTHONUNBUFFERED"] = "1"
    for stem in ("estimate_race", "estimate_race_nd"):
        cmd = [sys.executable, os.path.join(GAUNTLET, "build.py"), "--compile", "--src", os.path.join(EST, stem + ".c")]
        print("== build %s (%s)" % (stem, isa), flush=True)
        if subprocess.call(cmd, cwd=ROOT, env=env) != 0:
            sys.exit("the build of %s failed" % stem)


def shipped_store():
    sys.path.insert(0, GAUNTLET)
    try:
        from wisdom_folder import shipped_folder
        folder, identity = shipped_folder()
    except SystemExit as e:
        sys.exit("%s\n(or name the store with --store DIR)" % e)
    print("store: %s\n  identity: %s" % (folder, identity))
    return folder


def run(tool, args, log, title):
    """one pass of a tool, its output to the screen and appended to log"""
    cmd = [os.path.join(EST, tool + EXE)] + args
    print("\n== %s: %s" % (title, " ".join(cmd)), flush=True)
    with open(log, "a", encoding="utf-8", errors="replace") as f:
        f.write("\n== %s: %s\n" % (title, " ".join(cmd)))
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = p.wait()
    if rc != 0:
        print("(%s exited %d)" % (tool, rc))


LINE = re.compile(r"^\s*(\S+)\s+B\s+([\d.]+)\s+E\s+([\d.]+)\s+B2\s+([\d.]+)\s+E/B ([\d.]+) \[([\d.]+)\.\.([\d.]+)\]\s+B2/B ([\d.]+)")


def summary_rows(log):
    """the timed pass's summary lines (its '# summary' block; none if the timed pass did not finish)"""
    text = open(log, encoding="utf-8", errors="replace").read()
    block = text.rsplit("== TIMED PASS", 1)[-1].split("# summary", 1)
    if len(block) < 2:
        return []
    rows = []
    for ln in block[1].splitlines()[1:]:
        m = LINE.match(ln)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(5)), float(m.group(6)),
                         float(m.group(7)), float(m.group(8))))
    return rows


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isa", choices=["avx2", "avx512"], default="avx512")
    ap.add_argument("--sets", default="1d,2d,3d")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--store")
    ap.add_argument("--cells")
    ap.add_argument("--name")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    if os.name != "nt":
        sys.exit("the estimate tools time with the Windows timer and affinity calls: Windows only")
    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    if a.threads > 1 and "1d" in sets:
        print("1d skipped: the 1D tool times one thread only")
        sets.remove("1d")
    lists = {"1d": [], "2d": CELLS_2D, "3d": CELLS_3D}
    if a.cells:
        cs = a.cells.replace(",", " ").split()
        lists = {"1d": [c for c in cs if "x" not in c], "2d": [c for c in cs if c.count("x") == 1],
                 "3d": [c for c in cs if c.count("x") == 2]}
        sets = [s for s in sets if lists[s]]

    if not a.no_build:
        build(a.isa)
    src = os.path.abspath(a.store) if a.store else shipped_store()
    if not os.path.isdir(src):
        sys.exit("no store at %s" % src)
    name = a.name or "estimate-pow2-%s-%s" % (a.isa, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    out = os.path.join(GAUNTLET, "results", name)
    store = os.path.join(out, "store")
    if os.path.exists(store):
        sys.exit("%s exists: pick another --name" % out)
    shutil.copytree(src, store)
    print("run: %s\n  store copied from %s" % (out, src))

    for s in sets:
        log = os.path.join(out, s + ".txt")
        t = ["T=%d" % a.threads] if a.threads > 1 else []
        if s == "1d":
            tool, cells = "estimate_race", lists["1d"]
            first = [store, "1", "c2c"] + cells
            timed = [store, str(a.rounds), "c2c"] + cells
        else:
            tool, cells = "estimate_race_nd", lists[s]
            first = [store, "1"] + t + cells
            timed = [store, str(a.rounds)] + t + cells
        run(tool, first, log, "%s: bank the missing rows (one round)" % s)
        run(tool, timed, log, "TIMED PASS %s, %d rounds" % (s, a.rounds))

    md, allv = [], []
    for s in sets:
        rows = summary_rows(os.path.join(out, s + ".txt"))
        if not rows:
            md += ["## %s" % s.upper(), "", "no timed cells (see %s.txt)" % s, ""]
            continue
        v = [r[3] for r in rows]
        allv += v
        md += ["## %s: %d cells, E/B %.3f (geometric mean), %d within 5%%, %d above 1.10; control B2/B %.3f" % (
            s.upper(), len(rows), gm(v), sum(x <= 1.05 for x in v), sum(x > 1.10 for x in v), gm([r[6] for r in rows])),
               "", "| cell | planner ns | estimate ns | E/B | range | B2/B |", "|---|---:|---:|---:|---|---:|"]
        md += ["| %s | %.1f | %.1f | %.3f | %.3f .. %.3f | %.3f |" % r for r in rows]
        md.append("")
    head = ["# Estimate against the planner, powers of two (%s, %s)" % (a.isa, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
            "",
            "Complex transforms, natural order, out of place, %s; store copied from `%s`; %d rounds. E/B = the "
            "estimate plan's time over the planner's banked plan's (below 1: the estimate is faster); B2/B = the "
            "banked plan created again, the noise floor." % (
                "one thread" if a.threads <= 1 else "%d threads" % a.threads, src, a.rounds),
            "",
            "**All sets: %d cells, E/B %.3f (geometric mean).**" % (len(allv), gm(allv)) if allv else "**No timed cells.**",
            ""]
    with open(os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(head + md) + "\n")
    print("\n" + "\n".join(head[4:5] + [l for l in md if l.startswith("## ")]))
    print("written %s" % os.path.join(out, "SUMMARY.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
