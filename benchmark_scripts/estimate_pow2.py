"""estimate_pow2.py -- the estimate against the planner at powers of two: the library's estimate plan
(VFFT_FLAG_ESTIMATE, planned by formulas, nothing measured) against its own measured plan (the planner's,
banked in the wisdom store), in one process per set:

    python benchmark_scripts/estimate_pow2.py [options]
    ./benchmark_scripts/run_estimate_pow2.sh [options]

    --isa avx512|avx2   the codelet tree to compile (default avx512)
    --sets LIST         which sets, from 1d, 2d, 3d (complex) and r2d, r3d (real: R2C and C2R); default 1d,2d,3d
    --real              the real sets alone (--sets r2d,r3d)
    --rounds R          timed rounds per cell (default 15)
    --threads T         the 2D and 3D sets at T threads (default 1; the 1D tool times one thread only, so 1d is skipped)
    --store DIR         the wisdom store to start from (default: this CPU's shipped folder, as the library names it)
    --cells "..."       replace the lists: N for 1D, N1xN2 for the 2D sets, N1xN2xN3 for the 3D sets
    --name NAME         the run directory under gauntlet/results (default estimate-pow2-<isa>-<date>)
    --no-build          use the tools as built

The tools are src/estimate/estimate_race.c (1D complex), estimate_race_nd.c (2D and 3D complex) and
estimate_race_real.c (2D and 3D R2C and C2R). Per cell each times three plans, the banked plan (B), the estimate
plan (E) and the banked plan created again (B2, the noise floor), and prints E/B: below 1 the estimate plan is
faster. The real tool also checks every estimate plan against a long-double DFT (R2C: 48 sampled bins; C2R: the
round trip) and reports an estimate create the library refused, so a kernel this build lacks shows up there.
They run against a COPY of the store in the run directory, never the shipped one. A cell whose banked row is
missing has its plan measured by the planner first: the first pass at one round races and banks those rows into
the copy (the tools skip a cell whose create measured), the second pass times. The one-thread protocol pins
logical CPU 2 at HIGH priority; run on a quiet machine. Windows only (the tools use the Windows timer and
affinity calls). Output: <run>/<set>.txt (the real sets: <set>-r2c.txt, <set>-c2r.txt), and SUMMARY.md with
every cell's E/B and the geometric means. The store is the folder the library itself names for this CPU and build
(the tools' --where); the tools run with the toolchain's folder first on PATH, as the gauntlet runs its own.
"""
import argparse
import datetime
import importlib.util
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
SETS = ("1d", "2d", "3d", "r2d", "r3d")

# every power of two 64 .. 2^23 is the 1D tool's own default list
CELLS_2D = ["32x32", "64x64", "128x128", "256x256", "512x512", "1024x1024", "2048x2048",
            "64x256", "256x64", "128x512", "512x128", "32x512", "512x32", "16x1024", "1024x16",
            "4x2048", "2048x4", "128x2048", "2048x128", "8x8192", "8192x8", "64x4096", "4096x64",
            "256x4096", "4096x256", "1024x4096", "4096x1024", "512x8192", "8192x512"]
CELLS_3D = ["8x8x8", "16x16x16", "32x32x32", "64x64x64", "128x128x128",
            "4x256x256", "256x4x256", "256x256x4", "8x64x64", "64x8x64", "64x64x8",
            "16x128x128", "128x16x128", "128x128x16", "32x64x128", "128x64x32",
            "512x4x4", "4x4x512", "1024x16x16", "16x16x1024", "256x32x32"]
# the real sets: the planes and volumes the real estimate's rules branch on (the tall column, the one-stage column,
# the 128-point leaf, the fused walk over rows of 32, the half-store leaf, the strips, the one-stage axis 0)
CELLS_R2D = ["32x32", "64x64", "128x128", "256x256", "512x512", "1024x1024", "2048x2048",
             "64x256", "256x64", "16x1024", "1024x16", "16x4096", "32x512", "512x32", "1024x32", "128x2048",
             "2048x128", "128x8192", "64x4096", "4096x64", "4096x16", "8192x64", "8x8192", "8192x8",
             "1024x4096", "4096x1024"]
CELLS_R3D = ["8x8x8", "16x16x16", "32x32x32", "64x64x64", "128x128x128",
             "4x256x256", "256x4x256", "256x256x4", "16x64x64", "32x64x128", "64x32x256", "16x128x128",
             "128x16x128", "128x128x16", "128x64x32", "1024x16x16", "16x16x1024", "512x4x4", "16x256x256"]
TOOL = {"1d": "estimate_race", "2d": "estimate_race_nd", "3d": "estimate_race_nd", "r2d": "estimate_race_real",
        "r3d": "estimate_race_real"}


def build(isa, sets):
    env = os.environ.copy()
    env["VFFT_ISA"] = isa
    env["PYTHONUNBUFFERED"] = "1"
    for stem in sorted(set(TOOL[s] for s in sets)):
        cmd = [sys.executable, os.path.join(GAUNTLET, "build.py"), "--compile", "--src", os.path.join(EST, stem + ".c")]
        print("== build %s (%s)" % (stem, isa), flush=True)
        if subprocess.call(cmd, cwd=ROOT, env=env) != 0:
            sys.exit("the build of %s failed" % stem)


def tool_env():
    """the environment the tools run in: the toolchain's folder first on PATH (gauntlet/build.py's compiler) -- a
    MinGW-built tool loads its runtime (libwinpthread-1.dll) through PATH and, started without it, dies before main()
    with no output, as gauntlet.py says of its bench processes -- and no VFFT_WISDOM_DIR (the library's own store)"""
    spec = importlib.util.spec_from_file_location("gauntlet_build", os.path.join(GAUNTLET, "build.py"))
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    cc = b.detect_toolchain()["cc"]
    env = os.environ.copy()
    env.pop("VFFT_WISDOM_DIR", None)
    if os.path.dirname(cc):
        env["PATH"] = os.path.dirname(os.path.abspath(cc)) + os.pathsep + env.get("PATH", "")
    return env


def shipped_store(tool, env):
    """this CPU's folder of the library's own store, as the library names it: asked of the estimate tool just built
    (--where, recal_1d_probe's question), so the gauntlet's own tools need not be built"""
    exe = os.path.join(EST, tool + EXE)
    r = subprocess.run([exe, "--where"], capture_output=True, text=True, errors="replace", env=env)
    out = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
    if r.returncode != 0 or "folder" not in out:
        sys.exit("%s --where failed (exit %d): %s\n(or name the store with --store DIR)" % (
            exe, r.returncode, (r.stderr or r.stdout).strip()[-400:]))
    print("store: %s\n  identity: %s" % (out["folder"], out.get("identity", "")))
    return os.path.normpath(out["folder"])


ENV = None   # main: tool_env()


def run(tool, args, log, title):
    """one pass of a tool, its output to the screen and appended to log"""
    cmd = [os.path.join(EST, tool + EXE)] + args
    print("\n== %s: %s" % (title, " ".join(cmd)), flush=True)
    with open(log, "a", encoding="utf-8", errors="replace") as f:
        f.write("\n== %s: %s\n" % (title, " ".join(cmd)))
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                             env=ENV)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = p.wait()
    if rc != 0:
        print("(%s exited %d)" % (tool, rc))


LINE = re.compile(r"^\s*(\S+)\s+B\s+([\d.]+)\s+E\s+([\d.]+)\s+B2\s+([\d.]+)\s+E/B ([\d.]+) \[([\d.]+)\.\.([\d.]+)\]\s+B2/B ([\d.]+)"
                  r"(?:.*check (\S+) (PASS|FAIL))?")


def timed_pass(log):
    """the timed pass's output (empty if it did not run)"""
    text = open(log, encoding="utf-8", errors="replace").read()
    parts = text.rsplit("== TIMED PASS", 1)
    return parts[1] if len(parts) == 2 else ""


def summary_rows(block):
    """the timed pass's summary lines (its '# summary' block; none if the pass did not finish)"""
    parts = block.split("# summary", 1)
    rows = []
    for ln in (parts[1].splitlines()[1:] if len(parts) == 2 else []):
        m = LINE.match(ln)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(5)), float(m.group(6)),
                         float(m.group(7)), float(m.group(8)), m.group(9) or "", m.group(10) or ""))
    return rows


def gm(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isa", choices=["avx2", "avx512"], default="avx512")
    ap.add_argument("--sets", default="1d,2d,3d")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--store")
    ap.add_argument("--cells")
    ap.add_argument("--name")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    if os.name != "nt":
        sys.exit("the estimate tools time with the Windows timer and affinity calls: Windows only")
    sets = ["r2d", "r3d"] if a.real else [s.strip() for s in a.sets.split(",") if s.strip()]
    bad = [s for s in sets if s not in SETS]
    if bad:
        sys.exit("unknown set(s) %s: pick from %s" % (", ".join(bad), ", ".join(SETS)))
    if a.threads > 1 and "1d" in sets:
        print("1d skipped: the 1D tool times one thread only")
        sets.remove("1d")
    lists = {"1d": [], "2d": CELLS_2D, "3d": CELLS_3D, "r2d": CELLS_R2D, "r3d": CELLS_R3D}
    if a.cells:
        cs = a.cells.replace(",", " ").split()
        c1, c2, c3 = [c for c in cs if "x" not in c], [c for c in cs if c.count("x") == 1], [c for c in cs if c.count("x") == 2]
        lists = {"1d": c1, "2d": c2, "3d": c3, "r2d": c2, "r3d": c3}
        sets = [s for s in sets if lists[s]]

    if not a.no_build:
        build(a.isa, sets)
    global ENV
    ENV = tool_env()
    src = os.path.abspath(a.store) if a.store else shipped_store(TOOL[sets[0]], ENV)
    if not os.path.isdir(src):
        sys.exit("no store at %s" % src)
    name = a.name or "estimate-pow2-%s-%s" % (a.isa, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    out = os.path.join(GAUNTLET, "results", name)
    store = os.path.join(out, "store")
    if os.path.exists(store):
        sys.exit("%s exists: pick another --name" % out)
    shutil.copytree(src, store)
    print("run: %s\n  store copied from %s" % (out, src))

    t = ["T=%d" % a.threads] if a.threads > 1 else []
    parts = []   # (title, log)
    for s in sets:
        cells = lists[s]
        if s == "1d":
            jobs = [("1D complex", "1d.txt", [store, "{R}", "c2c"] + cells)]
        elif s in ("2d", "3d"):
            jobs = [("%s complex" % s.upper(), s + ".txt", [store, "{R}"] + t + cells)]
        else:
            jobs = [("%s real %s" % (s[1:].upper(), d.upper()), "%s-%s.txt" % (s, d), [store, "{R}"] + t + [d] + cells)
                    for d in ("r2c", "c2r")]
        for title, logname, args in jobs:
            log = os.path.join(out, logname)
            run(TOOL[s], [x.replace("{R}", "1") for x in args], log, "%s: bank the missing rows (one round)" % title)
            run(TOOL[s], [x.replace("{R}", str(a.rounds)) for x in args], log, "TIMED PASS %s, %d rounds" % (title, a.rounds))
            parts.append((title, log))

    md, allv, problems = [], [], []
    for title, log in parts:
        block = timed_pass(log)
        rows = summary_rows(block)
        refused = [ln.strip() for ln in block.splitlines()
                   if re.search(r"refused|MEASURED", ln) and not ln.lstrip().startswith("#")]
        fails = [r for r in rows if r[8] == "FAIL"]
        problems += ["%s: %s" % (title, r) for r in refused] + ["%s: %s check %s FAIL" % (title, r[0], r[7]) for r in fails]
        if not rows:
            md += ["## %s" % title, "", "no timed cells (see %s)" % os.path.basename(log), ""]
            continue
        v = [r[3] for r in rows]
        allv += v
        real = any(r[8] for r in rows)
        md += ["## %s: %d cells, E/B %.3f (geometric mean), %d within 5%%, %d above 1.10; control B2/B %.3f%s" % (
            title, len(rows), gm(v), sum(x <= 1.05 for x in v), sum(x > 1.10 for x in v), gm([r[6] for r in rows]),
            "; checks %d PASS, %d FAIL" % (len(rows) - len(fails), len(fails)) if real else ""), ""]
        md += ["| cell | planner ns | estimate ns | E/B | range | B2/B |%s" % (" check |" if real else ""),
               "|---|---:|---:|---:|---|---:|%s" % ("---|" if real else "")]
        md += ["| %s | %.1f | %.1f | %.3f | %.3f .. %.3f | %.3f |" % r[:7] + (" %s %s |" % (r[7], r[8]) if real else "")
               for r in rows]
        md.append("")
    head = ["# Estimate against the planner, powers of two (%s, %s)" % (a.isa, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
            "",
            "Natural order, out of place, %s; store copied from `%s`; %d rounds. E/B = the estimate plan's time over "
            "the planner's banked plan's (below 1: the estimate is faster); B2/B = the banked plan created again, the "
            "noise floor." % ("one thread" if a.threads <= 1 else "%d threads" % a.threads, src, a.rounds),
            "",
            "**All sets: %d cells, E/B %.3f (geometric mean).**" % (len(allv), gm(allv)) if allv else "**No timed cells.**",
            ""]
    if problems:
        head += ["**Estimate refusals, measured creates and failed checks (%d):**" % len(problems), ""] + \
                ["- %s" % p for p in problems] + [""]
    with open(os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(head + md) + "\n")
    print("\n" + "\n".join([head[4]] + [l for l in md if l.startswith("## ")] +
                           (["%d estimate refusals / measured creates / failed checks: see SUMMARY.md" % len(problems)] if problems else [])))
    print("written %s" % os.path.join(out, "SUMMARY.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
