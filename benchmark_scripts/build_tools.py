#!/usr/bin/env python3
"""build_tools.py -- build the gauntlet's three tools for the benchmark scripts (benchmark_scripts/README.md).

    python benchmark_scripts/build_tools.py --fftw-dll <path to fftw3.dll or libfftw3.so>
    python benchmark_scripts/build_tools.py --isa avx512 --fftw-dll <path>     # the AVX-512 codelet tree
    python benchmark_scripts/build_tools.py --no-mkl                            # without MKL, even when it is installed

Builds bench_1d_vs_mkl, recal_1d_probe and k1_fwd_ref_probe through gauntlet/build.py, one
after the other, with the library (--vfft) and with MKL (--mkl, the bench and the verify
probe) when MKL is on. The binaries land in gauntlet/, where gauntlet.py looks for them.
What was chosen is written to benchmark_scripts/.config.json, which run_suite.py reads: the
ISA, whether MKL is built in, the FFTW library's path and its version string, the compiler.

    --isa avx2|avx512   the codelet tree to compile (VFFT_ISA for build.py; default avx2)
    --mkl / --no-mkl    build the MKL arm in; default: on when MKL is found (MKLROOT or the oneAPI default install)
    --fftw-dll PATH     the FFTW shared library a --cmp fftw run binds at run time (VFFT_FFTW_DLL)
    --cc NAME           the compiler (CC for build.py; default gcc, the known MinGW-w64 / MSYS2 locations searched on Windows)

The CMake target vfft-gauntlet builds the same binaries (gauntlet/README.md); this script is
the one-command path and never calls CMake.
"""
import argparse
import datetime
import importlib.util
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GAUNTLET = os.path.join(ROOT, "gauntlet")
CONFIG = os.path.join(HERE, ".config.json")
# (source stem, built with --mkl when MKL is on): the calibrator never links MKL
TOOLS = (("bench_1d_vs_mkl", True), ("recal_1d_probe", False), ("k1_fwd_ref_probe", True))


def fftw_version_string(path):
    """the fftw_version string inside the library file, e.g. fftw-3.3.10-sse2-avx-avx2-avx512:
    the version, then the SIMD sets the library was built with (the bench prints the same
    string when it binds the library). Empty when the file holds none."""
    with open(path, "rb") as f:
        data = f.read()
    found = re.findall(rb"fftw-3\.[0-9][0-9A-Za-z_.\-]*", data)
    return max(found, key=len).decode("ascii", "replace") if found else ""


def harness():
    """gauntlet/build.py as a module: its find_mkl() is where MKL is looked for"""
    spec = importlib.util.spec_from_file_location("gauntlet_build", os.path.join(GAUNTLET, "build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isa", choices=["avx2", "avx512"], default="avx2", help="the codelet tree to compile (default avx2)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mkl", dest="mkl", action="store_true", default=None, help="build the MKL arm in (default: when MKL is found)")
    g.add_argument("--no-mkl", dest="mkl", action="store_false", help="leave MKL out")
    ap.add_argument("--fftw-dll", help="the FFTW shared library for --cmp fftw runs (recorded for run_suite.py)")
    ap.add_argument("--cc", help="the compiler (default: gcc)")
    args = ap.parse_args()

    fftw_dll, fftw_version = None, ""
    if args.fftw_dll:
        fftw_dll = os.path.abspath(args.fftw_dll)
        if not os.path.isfile(fftw_dll):
            sys.exit("FFTW library not found: %s" % fftw_dll)
        fftw_version = fftw_version_string(fftw_dll)
        print("FFTW: %s (%s)" % (fftw_version or "no fftw-3. version string in the file; is this FFTW?", fftw_dll))
    else:
        print("FFTW: no --fftw-dll given; run_suite.py --cmp fftw will need one")

    mkl = args.mkl
    if mkl is None:
        inc, lib = harness().find_mkl()
        mkl = bool(inc and lib)
        print("MKL: %s" % ("found at %s, built in" % os.path.dirname(str(inc)) if mkl else "not found (MKLROOT unset, no oneAPI default install); left out"))
    else:
        print("MKL: %s" % ("built in (--mkl)" if mkl else "left out (--no-mkl)"))
    print("ISA: %s" % args.isa)

    env = os.environ.copy()
    env["VFFT_ISA"] = args.isa
    env["PYTHONUNBUFFERED"] = "1"      # build.py's lines arrive as it prints them
    if args.cc:
        env["CC"] = args.cc

    results = []
    for stem, takes_mkl in TOOLS:
        cmd = ([sys.executable, os.path.join(GAUNTLET, "build.py"), "--compile", "--vfft"]
               + (["--mkl"] if mkl and takes_mkl else [])
               + ["--src", os.path.join(GAUNTLET, stem + ".c")])
        print("\n== %s\n%s" % (stem, " ".join(cmd)), flush=True)
        last = ""
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.startswith("[compile]"):
                last = line.strip()
        rc = p.wait()
        results.append((stem, rc, last))
        if rc != 0:
            break

    print("\nresults:")
    for stem, rc, last in results:
        print("  %-18s %s" % (stem, last or "exit %d" % rc))
    if len(results) < len(TOOLS) or any(rc for _, rc, _ in results):
        sys.exit("a build failed; nothing was recorded in %s" % CONFIG)

    config = {"isa": args.isa, "mkl": mkl, "fftw_dll": fftw_dll, "fftw_version": fftw_version,
              "cc": args.cc or "gcc", "built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print("recorded %s" % CONFIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
