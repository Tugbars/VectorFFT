#!/usr/bin/env python3
"""run_suite.py -- the five benchmark pictures of the repository's README on this machine, against FFTW or MKL (benchmark_scripts/README.md).

    python benchmark_scripts/run_suite.py --cmp fftw --yes                     # all five sets
    python benchmark_scripts/run_suite.py --cmp mkl --sets 2d,2d-mt8 --yes     # two of them
    python benchmark_scripts/run_suite.py --cmp fftw --plots-only              # redraw from runs already finished

Each set is one command of the gauntlet (gauntlet/gauntlet.py: the README's own command with
--cmp added), then the plot script of that picture (src/tools/plots/), drawn into the run
directory, then a SUMMARY.md there:

    1d        run --cells 2..4096              gauntlet/results/<prefix>_1d/        vectorfft-<cmp>-gflops.svg
    accuracy  verify --cells 2..4096           the 1d run's directory               vectorfft-<cmp>-precision.svg
    2d        run --group 2d-pow2              gauntlet/results/<prefix>_2d/        vectorfft-<cmp>-2d-pow2.svg + .png
    2d-mt     run --group 2d-pow2 --threads T  gauntlet/results/<prefix>_2d_mt<T>/  vectorfft-<cmp>-2d-mt-forest.svg + .png (with the 2d run's csv)
    3d        run --group 3d-pow2              gauntlet/results/<prefix>_3d/        vectorfft-<cmp>-3d-pow2.svg + .png

T, the thread count of the 2d-mt set, is the number of physical cores of this machine's
fastest core class (read from the OS: 8 on an i9-14900KF with 8 P-cores, 6 on a six-core
Zen 4), printed when the suite starts; --threads N overrides it.

The sets run one after the other, in that order, the driver's output streaming through; the
first failure stops the suite with the failing command printed. Nothing is merged into the
library's shipped wisdom (src/wisdom/): a run keeps its verdicts in its own store. With
--cmp fftw the FFTW library is found, loaded and checked before anything runs (exit 2 with
what to do when it is missing, or lacks the threads API the 2d-mt set needs).

    --cmp fftw|mkl       the comparator (required)
    --sets a,b,...       the sets to run, of 1d, accuracy, 2d, 2d-mt, 3d (default: all five)
    --threads N          the thread count of the 2d-mt set (default: the physical cores of the fastest core class)
    --name-prefix NAME   the run directories' prefix (default <cmp>_<isa>_<date>, the ISA from build_tools.py's record)
    --fftw-dll PATH      an FFTW shared library of your own (default: build_tools.py's record, then $VFFT_FFTW_DLL, then
                         the FFTW shipped with the repository on Windows (gauntlet/fftw/), vcpkg's, the distribution's on Linux)
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
import struct
import subprocess
import sys

from build_tools import (CONFIG, GAUNTLET, ROOT, fftw_has_threads, fftw_load, fftw_missing, fftw_no_threads,
                         fftw_version_of, harness, resolve_fftw)

RESULTS = "gauntlet/results"             # relative to ROOT: every command runs from there
DRIVER = "gauntlet/gauntlet.py"
PLOTS = "src/tools/plots"
EXE = ".exe" if os.name == "nt" else ""
TOOLS = ("bench_1d_vs_mkl", "recal_1d_probe", "k1_fwd_ref_probe")
SETS = ("1d", "accuracy", "2d", "2d-mt", "3d")
LABEL = {"fftw": "FFTW", "mkl": "MKL"}   # the comparator's name on a picture
ABOUT = {
    "1d": "1D c2c, every length N from 2 to 4,096, throughput in GFLOPS.",
    "accuracy": "The forward 1D c2c transform, every N from 2 to 4,096, against a long-double reference.",
    "2d": "2D c2c, every power-of-two plane up to 4M points (159 planes), the speedup per plane.",
    "2d-mt": "The same 159 planes at %d threads, the comparator at %d threads too, beside the one-thread run.",
    "3d": "3D c2c, every power-of-two volume up to 4M points (1,288 volumes), the speedup per volume.",
}


def physical_cores():
    """the physical cores of this machine's fastest core class, from the OS. Windows:
    GetLogicalProcessorInformationEx, one RelationProcessorCore record per core with its
    EfficiencyClass (the highest class is the fastest: the P-cores of a hybrid Intel part,
    every core of an AMD part). Linux: the distinct (physical id, core id) pairs of
    /proc/cpuinfo, among the CPUs of /sys/devices/cpu_core/cpus when a hybrid kernel lists
    them. Else the logical count."""
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            need = ctypes.c_ulong(0)
            k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(need))       # RelationProcessorCore = 0
            buf = ctypes.create_string_buffer(need.value)
            if k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(need)):
                classes, off = {}, 0
                while off < need.value:
                    rel, size = struct.unpack_from("II", buf, off)                # Relationship, Size
                    if rel == 0:
                        eff = buf.raw[off + 9]                                    # PROCESSOR_RELATIONSHIP.EfficiencyClass
                        classes[eff] = classes.get(eff, 0) + 1
                    off += size
                if classes:
                    return classes[max(classes)]
        except (OSError, AttributeError, struct.error):
            pass
    try:
        allowed = None
        if os.path.isfile("/sys/devices/cpu_core/cpus"):
            allowed = set()
            for part in open("/sys/devices/cpu_core/cpus").read().strip().split(","):
                a, _, b = part.partition("-")
                allowed.update(range(int(a), int(b or a) + 1))
        cores, cpu, phys, core = set(), None, "0", None
        for line in list(open("/proc/cpuinfo", encoding="utf-8", errors="replace")) + ["\n"]:
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if key == "processor":
                cpu, phys, core = int(val), "0", None
            elif key == "physical id":
                phys = val
            elif key == "core id":
                core = val
            elif not line.strip() and cpu is not None:
                if core is not None and (allowed is None or cpu in allowed):
                    cores.add((phys, core))
                cpu = None
        if cores:
            return len(cores)
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def suffix(dims, threads, cmp):
    """gauntlet.py's Run.sfx for the suite's contracts: _2d / _3d, _mt<T>, _<cmp> (nothing for mkl)"""
    return ("_%dd" % dims if dims >= 2 else "") + ("_mt%d" % threads if threads > 1 else "") + ("_" + cmp if cmp != "mkl" else "")


class Set:
    """one set: the driver's verb and arguments, its run directory, the files the plot reads and writes"""

    def __init__(self, name, cmp, prefix, threads, cells):
        self.name = name
        self.cmp = cmp
        dims = {"1d": 1, "accuracy": 1, "2d": 2, "2d-mt": 2, "3d": 3}[name]
        self.threads = threads if name == "2d-mt" else 1
        self.sfx = suffix(dims, self.threads, cmp)
        self.run = prefix + {"1d": "_1d", "accuracy": "_1d", "2d": "_2d", "2d-mt": "_2d_mt%d" % self.threads, "3d": "_3d"}[name]
        self.rel = RESULTS + "/" + self.run
        self.dir = os.path.join(ROOT, RESULTS, self.run)
        self.verb = "verify" if name == "accuracy" else "run"
        self.cells = cells
        self.cell_args = ["--cells", cells] if cells else {
            "1d": ["--cells", "2..4096"], "accuracy": ["--cells", "2..4096"],
            "2d": ["--group", "2d-pow2"], "2d-mt": ["--group", "2d-pow2"], "3d": ["--group", "3d-pow2"]}[name]
        self.about = ABOUT[name] % (self.threads, self.threads) if name == "2d-mt" else ABOUT[name]
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
        if self.name == "2d-mt":
            return [PLOTS + "/gen_forest_2d_mt.py", self.one_thread_csv, self.csv, base + "-2d-mt-forest", "--cmp", label,
                    "--threads", str(self.threads)], base + "-2d-mt-forest.svg + .png"
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


# ── running and writing ────────────────────────────────────────────────────

def shown(cmd):
    return " ".join(["python"] + [a.replace(os.sep, "/") for a in cmd])


def fail(cmd, rc):
    print("\nfailed (exit %d): %s" % (rc, shown(cmd)), flush=True)
    sys.exit(1)


def run(cmd, env=None, capture=False):
    """a command from the repository's root; the output streams through unless captured,
    then (exit code, stdout, stderr) comes back"""
    full = [sys.executable] + cmd
    if capture:
        r = subprocess.run(full, cwd=ROOT, env=env, capture_output=True, text=True, errors="replace")
        return r.returncode, r.stdout, r.stderr
    return subprocess.run(full, cwd=ROOT, env=env).returncode, "", ""


def last_line(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


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
    ap.add_argument("--threads", type=int, help="the thread count of the 2d-mt set (default: this machine's physical cores of its fastest core class)")
    ap.add_argument("--name-prefix", help="the run directories' prefix (default <cmp>_<isa>_<date>)")
    ap.add_argument("--fftw-dll", help="an FFTW shared library of your own (default: build_tools.py's record, VFFT_FFTW_DLL, then the one shipped with the repository on Windows, the distribution's on Linux)")
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
    threads = args.threads or physical_cores()
    if "2d-mt" in names:
        print("threads for the 2d-mt set: %d (%s)" % (threads, "--threads" if args.threads else "the physical cores of this machine's fastest core class"))

    env = os.environ.copy()
    fftw_dll, fftw_lib = None, None
    if cmp == "fftw":
        # the FFTW library is found, loaded and checked here, before any run: the bench
        # never discovers a missing library cell by cell
        fftw_dll, tried = resolve_fftw(args.fftw_dll, config.get("fftw_dll"))
        if fftw_dll:
            fftw_dll = os.path.abspath(fftw_dll) if os.path.dirname(fftw_dll) else fftw_dll
            fftw_lib, why = fftw_load(fftw_dll)
            if fftw_lib is None and not args.plots_only:
                print("FFTW library %s could not be loaded: %s" % (fftw_dll, why), file=sys.stderr)
                print(fftw_missing(tried), file=sys.stderr)
                sys.exit(2)
        elif not args.plots_only:
            print(fftw_missing(tried), file=sys.stderr)
            sys.exit(2)
        if fftw_lib is not None:
            env["VFFT_FFTW_DLL"] = fftw_dll
            comparator = "FFTW %s (%s)" % (fftw_version_of(fftw_lib, fftw_dll), fftw_dll)
        else:
            comparator = "FFTW (the library was not found when the pictures were drawn)"
    else:
        if not config.get("mkl"):
            sys.exit("the tools were built without MKL (build_tools.py's record): build again with --mkl, or run --cmp fftw")
        comparator = mkl_version()

    if not args.plots_only:
        missing = [t for t in TOOLS if not os.path.isfile(os.path.join(GAUNTLET, t + EXE))]
        if missing:
            sys.exit("missing %s in gauntlet/: run build_tools.py first" % ", ".join(missing))
        if "2d-mt" in names and cmp == "fftw" and not fftw_has_threads(fftw_lib):
            print(fftw_no_threads(fftw_dll), file=sys.stderr)
            sys.exit(2)

    plan = [Set(n, cmp, prefix, threads, args.cells) for n in names]
    for s in plan:
        if s.name == "2d-mt":
            one = Set("2d", cmp, prefix, threads, args.cells)
            s.one_thread_csv = one.csv
            if "2d" not in names and not os.path.isfile(os.path.join(ROOT, one.csv)):
                sys.exit("the 2d-mt picture sets the threaded run beside the one-thread run, and %s does not exist: "
                         "run --sets 2d first, or both together (--sets 2d,2d-mt)" % one.csv)

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
                rc, out, _ = run(s.estimate_command(), capture=True)
                if rc == 0 and last_line(out):
                    print("the driver's estimate: " + last_line(out), flush=True)
            print(shown(cmd), flush=True)
            rc, _, _ = run(cmd, env=env)
            if rc != 0:
                fail(cmd, rc)
        plot_cmd, picture = s.plot()
        print(shown(plot_cmd), flush=True)
        rc, out, err = run(plot_cmd, capture=True)
        print((out + err).strip())
        if rc != 0:
            fail(plot_cmd, rc)
        lines = ["command: `%s`" % shown(cmd)] + (["cells: %s (a quick check, not the README's set)" % s.cells] if s.cells else [])
        lines += ["csv: %s" % s.csv] + (["report: %s" % s.report] if s.report else []) + (["one-thread csv: %s" % s.one_thread_csv] if s.one_thread_csv else [])
        lines += ["picture: %s" % picture, "plot: `%s`" % shown(plot_cmd), "the plot script says: %s" % last_line(out)]
        if args.plots_only:
            lines.append("drawn again with --plots-only, %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print("summary: " + write_summary(s, machine, lines).replace(os.sep, "/"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
