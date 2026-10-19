#!/usr/bin/env python3
"""run_suite.py -- the five benchmark pictures of the repository's README on this machine, against FFTW or MKL (benchmark_scripts/README.md).

    python benchmark_scripts/run_suite.py --cmp fftw --yes                     # all five sets
    python benchmark_scripts/run_suite.py --cmp mkl --sets 2d,2d-mt8 --yes     # two of them
    python benchmark_scripts/run_suite.py --cmp fftw --plots-only              # redraw from runs already finished

Each set is one command of the gauntlet (gauntlet/gauntlet.py: the README's own command with
--cmp added), then the plot script of that picture (src/tools/plots/), drawn into the run
directory, then a SUMMARY.md there:

    1d        run --cells 2..4096              gauntlet/results/<prefix>_1d/      vectorfft-<cmp>-gflops.svg
    accuracy  verify --cells 2..4096           the 1d run's directory             vectorfft-<cmp>-precision.svg
    2d        run --group 2d-pow2              gauntlet/results/<prefix>_2d/      vectorfft-<cmp>-2d-pow2.svg + .png
    2d-mt8    run --group 2d-pow2 --threads 8  gauntlet/results/<prefix>_2d_mt8/  vectorfft-<cmp>-2d-mt-forest.svg + .png (with the 2d run's csv)
    3d        run --group 3d-pow2              gauntlet/results/<prefix>_3d/      vectorfft-<cmp>-3d-pow2.svg + .png

The sets run one after the other, in that order, the driver's output streaming through; the
first failure stops the suite with the failing command printed. Nothing is merged into the
library's shipped wisdom (src/wisdom/): a run keeps its verdicts in its own store.

    --cmp fftw|mkl       the comparator (required)
    --sets a,b,...       the sets to run, of 1d, accuracy, 2d, 2d-mt8, 3d (default: all five)
    --threads N          the thread count of the 2d-mt8 set (default 8)
    --name-prefix NAME   the run directories' prefix (default <cmp>_<isa>_<date>, the ISA from build_tools.py's record)
    --fftw-dll PATH      the FFTW shared library (default: the one build_tools.py recorded)
    --yes                do not stop for the driver's duration prompt
    --plots-only         draw the pictures and write the summaries from runs already finished; nothing runs
    --cells SPEC         replace every set's cell list, for a quick check (e.g. --cells 64,4096; 32x32,64x64 for 2d; 16x16x16 for 3d)
"""
import argparse
import ctypes
import datetime
import glob
import json
import os
import platform
import subprocess
import sys

from build_tools import CONFIG, GAUNTLET, ROOT, fftw_version_string, harness

RESULTS = "gauntlet/results"             # relative to ROOT: every command runs from there
DRIVER = "gauntlet/gauntlet.py"
PLOTS = "src/tools/plots"
EXE = ".exe" if os.name == "nt" else ""
TOOLS = ("bench_1d_vs_mkl", "recal_1d_probe", "k1_fwd_ref_probe")
SETS = ("1d", "accuracy", "2d", "2d-mt8", "3d")
LABEL = {"fftw": "FFTW", "mkl": "MKL"}   # the comparator's name on a picture
ABOUT = {
    "1d": "1D c2c, every length N from 2 to 4,096, throughput in GFLOPS.",
    "accuracy": "The forward 1D c2c transform, every N from 2 to 4,096, against a long-double reference.",
    "2d": "2D c2c, every power-of-two plane up to 4M points (159 planes), the speedup per plane.",
    "2d-mt8": "The same 159 planes at %d threads, the comparator at %d threads too, beside the one-thread run.",
    "3d": "3D c2c, every power-of-two volume up to 4M points (1,288 volumes), the speedup per volume.",
}


def suffix(dims, threads, cmp):
    """gauntlet.py's Run.sfx for the suite's contracts: _2d / _3d, _mt<T>, _<cmp> (nothing for mkl)"""
    return ("_%dd" % dims if dims >= 2 else "") + ("_mt%d" % threads if threads > 1 else "") + ("_" + cmp if cmp != "mkl" else "")


class Set:
    """one set: the driver's verb and arguments, its run directory, the files the plot reads and writes"""

    def __init__(self, name, cmp, prefix, threads, cells):
        self.name = name
        self.cmp = cmp
        dims = {"1d": 1, "accuracy": 1, "2d": 2, "2d-mt8": 2, "3d": 3}[name]
        self.threads = threads if name == "2d-mt8" else 1
        self.sfx = suffix(dims, self.threads, cmp)
        self.run = prefix + {"1d": "_1d", "accuracy": "_1d", "2d": "_2d", "2d-mt8": "_2d_mt%d" % self.threads, "3d": "_3d"}[name]
        self.rel = RESULTS + "/" + self.run
        self.dir = os.path.join(ROOT, RESULTS, self.run)
        self.verb = "verify" if name == "accuracy" else "run"
        self.cells = cells
        self.cell_args = ["--cells", cells] if cells else {
            "1d": ["--cells", "2..4096"], "accuracy": ["--cells", "2..4096"],
            "2d": ["--group", "2d-pow2"], "2d-mt8": ["--group", "2d-pow2"], "3d": ["--group", "3d-pow2"]}[name]
        self.about = ABOUT[name] % (self.threads, self.threads) if name == "2d-mt8" else ABOUT[name]
        self.csv = self.rel + "/" + ("verify" if name == "accuracy" else "gauntlet") + self.sfx + ".csv"
        self.report = None if name == "accuracy" else self.rel + "/report" + self.sfx + ".md"
        self.one_thread_csv = None       # the forest's one-thread record (the 2d set's csv)

    def command(self, yes):
        return ([DRIVER, self.verb] + self.cell_args + (["--threads", str(self.threads)] if self.threads > 1 else [])
                + ["--cmp", self.cmp, "--name", self.run] + (["--yes"] if yes else []))

    def estimate_command(self):
        """the driver's own duration estimate for the set's cells (the cells verb; it runs nothing)"""
        return [DRIVER, "cells"] + self.cell_args

    def plot(self):
        """(the plot script's command, the picture's base name)"""
        label = LABEL[self.cmp]
        base = self.rel + "/vectorfft-" + self.cmp
        if self.name == "1d":
            return [PLOTS + "/gen_gflops.py", self.csv, "--out", base + "-gflops.svg", "--cmp", label, "--logy"], base + "-gflops.svg"
        if self.name == "accuracy":
            return [PLOTS + "/gen_precision.py", self.csv, "--out", base + "-precision.svg"], base + "-precision.svg"
        if self.name == "2d":
            return [PLOTS + "/gen_2d_pow2.py", self.report, base + "-2d-pow2", "--cmp", label], base + "-2d-pow2.svg + .png"
        if self.name == "2d-mt8":
            return [PLOTS + "/gen_forest_2d_mt.py", self.one_thread_csv, self.csv, base + "-2d-mt-forest", "--cmp", label], base + "-2d-mt-forest.svg + .png"
        return [PLOTS + "/gen_3d_pow2.py", self.csv, base + "-3d-pow2", "--cmp", label], base + "-3d-pow2.svg + .png"


# ── the machine, for the summary ───────────────────────────────────────────

def cpu_name():
    if os.name == "nt":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    except OSError:
        return ""


def mkl_version():
    """MKL's own version string, asked of the mkl_rt library of the install gauntlet/build.py finds;
    the install's path when the call is not possible"""
    inc, lib = harness().find_mkl()
    if not inc:
        return "Intel MKL (install not found)"
    root = os.path.realpath(os.path.dirname(str(inc)))
    for d in (os.path.join(root, "bin"), str(lib), os.path.join(root, "lib", "intel64")):
        for pat in ("mkl_rt*.dll", "libmkl_rt.so*", "libmkl_rt*.dylib"):
            for path in sorted(glob.glob(os.path.join(d, pat))):
                try:
                    if os.name == "nt":
                        os.add_dll_directory(d)
                    rt = ctypes.CDLL(path)
                    buf = ctypes.create_string_buffer(512)
                    rt.mkl_get_version_string(buf, 512)
                    s = buf.value.decode("ascii", "replace").strip()
                    if s:
                        return "%s (%s)" % (s, path)
                except (OSError, AttributeError):
                    continue
    return "Intel MKL at %s" % root


def fftw_has_threads(path):
    """the threads API in the library itself (a build with --with-combined-threads exports fftw_plan_with_nthreads)"""
    with open(path, "rb") as f:
        return b"fftw_plan_with_nthreads" in f.read()


# ── running and writing ────────────────────────────────────────────────────

def shown(cmd):
    return " ".join(["python"] + [a.replace(os.sep, "/") for a in cmd])


def fail(cmd, rc):
    print("\nfailed (exit %d): %s" % (rc, shown(cmd)), flush=True)
    sys.exit(1)


def run(cmd, env=None, capture=False):
    """a command from the repository's root; the output streams through unless captured"""
    full = [sys.executable] + cmd
    if capture:
        r = subprocess.run(full, cwd=ROOT, env=env, capture_output=True, text=True, errors="replace")
        return r.returncode, r.stdout + r.stderr
    return subprocess.run(full, cwd=ROOT, env=env).returncode, ""


def write_summary(s, machine, lines):
    """SUMMARY.md in the run directory: the machine, then one section per set that ran there
    (the 1d and accuracy sets share a directory; a section is replaced when its set runs again)"""
    path = os.path.join(s.dir, "SUMMARY.md")
    sections = {}
    if os.path.isfile(path):
        for part in open(path, encoding="utf-8").read().split("\n## ")[1:]:
            sections[part.split("\n", 1)[0].strip()] = "## " + part.rstrip() + "\n"
    sections[s.name] = "## %s\n\n%s\n\n%s\n" % (s.name, s.about, "\n".join("- " + l for l in lines))
    head = "# %s\n\nWritten by benchmark_scripts/run_suite.py, %s.\n\n%s\n\n" % (
        s.run, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "\n".join("- " + l for l in machine))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(head + "\n".join(sections[k] for k in SETS if k in sections))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmp", choices=["fftw", "mkl"], required=True, help="the comparator")
    ap.add_argument("--sets", default=",".join(SETS), help="which sets, of %s (default all)" % ", ".join(SETS))
    ap.add_argument("--threads", type=int, default=8, help="the thread count of the 2d-mt8 set (default 8)")
    ap.add_argument("--name-prefix", help="the run directories' prefix (default <cmp>_<isa>_<date>)")
    ap.add_argument("--fftw-dll", help="the FFTW shared library (default: build_tools.py's record)")
    ap.add_argument("--yes", action="store_true", help="pass the driver's --yes (no duration prompt)")
    ap.add_argument("--plots-only", action="store_true", help="draw from finished runs; nothing runs")
    ap.add_argument("--cells", help="replace every set's cell list, for a quick check")
    args = ap.parse_args()

    names = [n.strip() for n in args.sets.split(",") if n.strip()]
    unknown = [n for n in names if n not in SETS]
    if unknown:
        sys.exit("unknown set %s: the sets are %s" % (", ".join(unknown), ", ".join(SETS)))
    names = [n for n in SETS if n in names]
    if not os.path.isfile(CONFIG):
        sys.exit("no %s: run build_tools.py first (benchmark_scripts/README.md)" % CONFIG)
    config = json.load(open(CONFIG, encoding="utf-8"))
    isa, cmp = config["isa"], args.cmp
    prefix = args.name_prefix or "%s_%s_%s" % (cmp, isa, datetime.date.today().strftime("%Y-%m-%d"))

    env = os.environ.copy()
    fftw_dll = args.fftw_dll or config.get("fftw_dll")
    if cmp == "fftw":
        if not fftw_dll:
            sys.exit("--cmp fftw needs the FFTW library: pass --fftw-dll, or give it to build_tools.py")
        fftw_dll = os.path.abspath(fftw_dll)
        if not os.path.isfile(fftw_dll):
            sys.exit("FFTW library not found: %s" % fftw_dll)
        env["VFFT_FFTW_DLL"] = fftw_dll
        comparator = "FFTW %s (%s)" % (fftw_version_string(fftw_dll) or "version string not found", fftw_dll)
    else:
        if not config.get("mkl"):
            sys.exit("the tools were built without MKL (build_tools.py's record): build again with --mkl, or run --cmp fftw")
        comparator = mkl_version()

    if not args.plots_only:
        missing = [t for t in TOOLS if not os.path.isfile(os.path.join(GAUNTLET, t + EXE))]
        if missing:
            sys.exit("missing %s in gauntlet/: run build_tools.py first" % ", ".join(missing))
        if "2d-mt8" in names and cmp == "fftw" and not fftw_has_threads(fftw_dll):
            sys.exit("the 2d-mt8 set needs an FFTW with the threads API in the same library (a build with "
                     "--with-combined-threads, benchmark_scripts/README.md); %s has none" % fftw_dll)

    plan = [Set(n, cmp, prefix, args.threads, args.cells) for n in names]
    for s in plan:
        if s.name == "2d-mt8":
            one = Set("2d", cmp, prefix, args.threads, args.cells)
            s.one_thread_csv = one.csv
            if "2d" not in names and not os.path.isfile(os.path.join(ROOT, one.csv)):
                sys.exit("the 2d-mt8 picture sets the threaded run beside the one-thread run, and %s does not exist: "
                         "run --sets 2d first, or both together (--sets 2d,2d-mt8)" % one.csv)

    commit = git_commit()
    machine = ["machine: %s, %s, %d logical CPUs" % (cpu_name(), platform.platform(), os.cpu_count() or 0),
               "VectorFFT: codelets for %s, compiler %s, tools built %s (build_tools.py)%s" % (
                   isa, config.get("cc", "gcc"), config.get("built", "?"), ", commit " + commit if commit else ""),
               "comparator: " + comparator]
    print("\n".join(machine))
    print("sets: %s; run directories: %s/%s_*" % (", ".join(names), RESULTS, prefix), flush=True)

    for s in plan:
        print("\n==== %s: %s" % (s.name, s.about), flush=True)
        cmd = s.command(args.yes)
        if not args.plots_only:
            if s.verb == "run":
                rc, out = run(s.estimate_command(), capture=True)
                print("the driver's estimate: " + out.strip().splitlines()[-1] if rc == 0 and out.strip() else "", flush=True)
            print(shown(cmd), flush=True)
            rc, _ = run(cmd, env=env)
            if rc != 0:
                fail(cmd, rc)
        plot_cmd, picture = s.plot()
        print(shown(plot_cmd), flush=True)
        rc, out = run(plot_cmd, capture=True)
        print(out.strip())
        if rc != 0:
            fail(plot_cmd, rc)
        lines = ["command: `%s`" % shown(cmd)] + (["cells: %s (a quick check, not the README's set)" % s.cells] if s.cells else [])
        lines += ["csv: %s" % s.csv] + (["report: %s" % s.report] if s.report else []) + (["one-thread csv: %s" % s.one_thread_csv] if s.one_thread_csv else [])
        lines += ["picture: %s" % picture, "plot: `%s`" % shown(plot_cmd), "the plot script says: %s" % (out.strip().splitlines()[-1] if out.strip() else "")]
        if args.plots_only:
            lines.append("drawn again with --plots-only, %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print("summary: " + write_summary(s, machine, lines).replace(os.sep, "/"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
