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
import ctypes
import ctypes.util
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
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return ""
    found = re.findall(rb"fftw-3\.[0-9][0-9A-Za-z_.\-]*", data)
    return max(found, key=len).decode("ascii", "replace") if found else ""


# ── the FFTW library: found, loaded and checked before any run starts ──────

def fftw_default():
    """where FFTW usually is: vcpkg's library on Windows, the distribution's on Linux"""
    if os.name == "nt":
        return r"C:\vcpkg\installed\x64-windows\bin\fftw3.dll"
    return ctypes.util.find_library("fftw3") or "libfftw3.so.3"


def resolve_fftw(given, recorded):
    """the FFTW library, the first that exists of: --fftw-dll, build_tools.py's record,
    $VFFT_FFTW_DLL, the usual place. A bare library name (Linux: libfftw3.so.3) is left to
    the loader. Returns (path or None, the candidates tried)."""
    tried = []
    if given:                       # named on the command line: that one, or nothing
        return (given if not os.path.dirname(given) or os.path.isfile(given) else None), ["--fftw-dll: " + given]
    for what, p in (("benchmark_scripts/.config.json", recorded),
                    ("VFFT_FFTW_DLL", os.environ.get("VFFT_FFTW_DLL")), ("the usual place", fftw_default())):
        if p:
            tried.append("%s: %s" % (what, p))
            if not os.path.dirname(p) or os.path.isfile(p):
                return p, tried
    return None, tried


def fftw_load(path):
    """load the library as the bench will, by its path, its own directory on the DLL search
    path on Windows. Returns (the library, '') or (None, why)."""
    try:
        if os.path.dirname(path):
            if not os.path.isfile(path):
                return None, "no such file"
            if os.name == "nt":
                os.add_dll_directory(os.path.dirname(os.path.abspath(path)))
        return ctypes.CDLL(path), ""
    except OSError as e:
        return None, str(e)


def fftw_version_of(lib, path):
    """FFTW's own version string: the fftw_version symbol of the loaded library, a char
    array (what the bench prints at bind, read at the symbol's address as the bench does);
    the string found in the file when the symbol is not there"""
    try:
        return ctypes.string_at(ctypes.addressof(ctypes.c_char.in_dll(lib, "fftw_version"))).decode("ascii", "replace")
    except (ValueError, AttributeError):
        return fftw_version_string(path) or "version string not found"


def fftw_has_threads(lib):
    """the threads API in the same library (fftw_init_threads is exported by a build with
    --with-combined-threads); a separate libfftw3_threads does not count, the bench loads one file"""
    return hasattr(lib, "fftw_init_threads")


def fftw_missing(tried):
    return ("FFTW was not found. Tried " + "; ".join(tried) + ".\n"
            "Point at it with --fftw-dll <path> or VFFT_FFTW_DLL=<path>. To get one:\n"
            "  Windows: vcpkg install fftw3:x64-windows (a serial AVX2 build; C:\\vcpkg\\installed\\x64-windows\\bin\\fftw3.dll)\n"
            "  Linux:   the distribution's package, e.g. apt install libfftw3-3\n"
            "  for the threaded picture or AVX-512: build FFTW 3.3.10 from source with the configure flags in benchmark_scripts/README.md")


def fftw_no_threads(path):
    return ("the threaded picture needs FFTW's threads API in the same library, and %s has none (no fftw_init_threads).\n"
            "Build FFTW 3.3.10 with --enable-threads --with-combined-threads (CMake: -DENABLE_THREADS=ON -DWITH_COMBINED_THREADS=ON),\n"
            "the configure flags in benchmark_scripts/README.md, and point at it with --fftw-dll; or leave the 2d-mt set out." % path)


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
        fftw_dll = os.path.abspath(args.fftw_dll) if os.path.dirname(args.fftw_dll) or os.path.isfile(args.fftw_dll) else args.fftw_dll
        lib, why = fftw_load(fftw_dll)
        if lib is None:
            print("FFTW library %s could not be loaded: %s" % (fftw_dll, why), file=sys.stderr)
            print(fftw_missing(["--fftw-dll: " + fftw_dll]), file=sys.stderr)
            sys.exit(2)
        fftw_version = fftw_version_of(lib, fftw_dll)
        print("FFTW: %s (%s); threads API %s" % (fftw_version, fftw_dll,
              "present" if fftw_has_threads(lib) else "absent (the threaded picture needs it)"))
    else:
        print("FFTW: no --fftw-dll given; run_suite.py --cmp fftw will need one (it looks in VFFT_FFTW_DLL and the usual place too)")

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
