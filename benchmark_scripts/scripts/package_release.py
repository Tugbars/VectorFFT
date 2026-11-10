"""package_release.py -- the vectorfft_vs_* benchmark programs as a download (benchmark_scripts/README.md,
"The download"): one folder and one archive that run on any x86-64 machine with AVX2, no compiler, no Python.

    python benchmark_scripts/scripts/package_release.py --build <build dir> [--out <dir>] [--name <name>]

<build dir> is a CMake build of this repository made for other machines and with both instruction sets:

    cmake -B <build dir> -DVFFT_NATIVE=OFF -DVFFT_ISA=avx2 -DVFFT_BENCH_BOTH_ISAS=ON -DVFFT_ENABLE_FLOAT=ON
    cmake --build <build dir> --target <every vectorfft_vs_* program>

built WITHOUT Intel MKL: every program runs the same bench, and a bench linked to MKL does not start on a machine
without it, the FFTW programs included (the script refuses such a bench). The package:

    <name>/run_benchmark.bat | run.sh   the menu (Windows | Linux)
           README.txt, BUILD.txt, LICENSE
           vectorfft_vs_*[.exe]         the programs (vectorfft_launcher.h)
           avx2/, avx512/               each instruction set's worker, bench and planning tool, with the compiler's
                                        runtime libraries they load (MinGW: libwinpthread-1.dll)
           fftw/                        Windows: FFTW 3.3.10 double and single precision, with its license
           wisdom/                      src/wisdom/: the planner's measured plans of the CPUs the repository knows

and <out>/<name>.zip (Windows) or <name>.tar.gz (Linux), with <archive>.sha256. The programs find all of it
beside themselves (vectorfft_launcher.h vl_worker, vectorfft_bench.h vs_locate). Default --out: <build dir>/package.
"""
import argparse
import datetime
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # benchmark_scripts/scripts/ -> the repository
WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
ISAS = ("avx2", "avx512")


def cache(build):
    """the build's CMakeCache.txt as name -> value"""
    out = {}
    p = os.path.join(build, "CMakeCache.txt")
    if not os.path.isfile(p):
        sys.exit("%s is not a CMake build directory (no CMakeCache.txt)" % build)
    for line in io.open(p, encoding="utf-8", errors="replace"):
        m = re.match(r"^([A-Za-z0-9_.-]+):[A-Z]+=(.*)$", line.rstrip("\r\n"))
        if m:
            out[m.group(1)] = m.group(2)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dll_deps(objdump, exe):
    """the DLL names an executable imports (objdump -p), Windows only"""
    if not WIN or not objdump:
        return []
    r = subprocess.run([objdump, "-p", exe], capture_output=True, text=True, errors="replace")
    return re.findall(r"DLL Name:\s*(\S+)", r.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True, help="the CMake build directory (both instruction sets, no MKL)")
    ap.add_argument("--out", help="where the package folder and archive go (default: <build>/package)")
    ap.add_argument("--name", help="the package's name (default: vectorfft-benchmarks-<windows|linux>-x64)")
    a = ap.parse_args()
    build = os.path.abspath(a.build)
    out = os.path.abspath(a.out or os.path.join(build, "package"))
    name = a.name or "vectorfft-benchmarks-%s-x64" % ("windows" if WIN else "linux")
    c = cache(build)

    # the two instruction sets: this build's, and the second build's folder <build>/<other isa>/
    subs = [i for i in ISAS if os.path.isdir(os.path.join(build, i, "benchmark_scripts"))]
    if len(subs) != 1:
        sys.exit("%s holds no second instruction set's build (<build>/avx2/ or <build>/avx512/): configure it with "
                 "-DVFFT_BENCH_BOTH_ISAS=ON and build the programs" % build)
    other = subs[0]
    main_isa = "avx512" if other == "avx2" else "avx2"
    tools = {main_isa: build, other: os.path.join(build, other)}

    pkg = os.path.join(out, name)
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    os.makedirs(pkg)
    objdump = c.get("CMAKE_OBJDUMP")
    cc = c.get("CMAKE_C_COMPILER", "")
    cc_bin = os.path.dirname(cc)

    # the programs
    progs = sorted(f for f in os.listdir(os.path.join(build, "benchmark_scripts"))
                   if f.startswith("vectorfft_vs_") and f.endswith(EXE) and (EXE or "." not in f))
    if not progs:
        sys.exit("no vectorfft_vs_* program in %s/benchmark_scripts: build them first" % build)
    for f in progs:
        shutil.copy2(os.path.join(build, "benchmark_scripts", f), pkg)

    # each instruction set's worker, bench and planning tool, with the compiler's runtime libraries they import
    for isa, b in tools.items():
        d = os.path.join(pkg, isa)
        os.makedirs(d)
        for rel in ("benchmark_scripts/vectorfft_bench_worker", "gauntlet/bench_1d_vs_mkl", "gauntlet/recal_1d_probe"):
            src = os.path.join(b, rel + EXE)
            if not os.path.isfile(src):
                sys.exit("missing %s: build the programs (they build the worker, the bench and the planning tool)" % src)
            shutil.copy2(src, d)
            for dll in dll_deps(objdump, src):
                if re.match(r"(?i)mkl_|libiomp", dll):
                    sys.exit("%s imports %s: a bench linked to Intel MKL does not start on a machine without MKL, the "
                             "FFTW programs included. Configure the package's build without MKL." % (src, dll))
                local = os.path.join(cc_bin, dll)
                if cc_bin and os.path.isfile(local) and not os.path.isfile(os.path.join(d, dll)):
                    shutil.copy2(local, d)   # the compiler's runtime (Windows finds a DLL beside the program)

    # FFTW (Windows: the repository's builds; elsewhere the system's libfftw3 / libfftw3f)
    if WIN:
        d = os.path.join(pkg, "fftw")
        os.makedirs(d)
        for f in ("libfftw3-3.dll", "libfftw3f-3.dll", "COPYING", "COPYRIGHT", "README.md"):
            shutil.copy2(os.path.join(ROOT, "gauntlet", "fftw", f), d)

    # the shipped wisdom: every CPU's folder, never the lock files or backups
    shutil.copytree(os.path.join(ROOT, "src", "wisdom"), os.path.join(pkg, "wisdom"),
                    ignore=shutil.ignore_patterns("*.lock", "*.bak*", "*.tmp"))

    # the menu, the README, the license
    rel = os.path.join(ROOT, "benchmark_scripts", "release")
    if WIN:
        shutil.copy2(os.path.join(rel, "run_benchmark.bat"), pkg)
    else:
        shutil.copy2(os.path.join(rel, "run.sh"), pkg)
        os.chmod(os.path.join(pkg, "run.sh"), 0o755)
    shutil.copy2(os.path.join(rel, "README.txt"), pkg)
    shutil.copy2(os.path.join(ROOT, "LICENSE"), pkg)

    # BUILD.txt: what this package is, from which source
    def git(*args):
        r = subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    ccv = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[:1] if cc else []
    lines = [
        "VectorFFT benchmark programs",
        "",
        "built      %s UTC" % datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "commit     %s%s" % (git("rev-parse", "HEAD"), " (with uncommitted changes)" if git("status", "--porcelain", "--", "src", "include", "gauntlet", "benchmark_scripts") else ""),
        "describe   %s" % git("describe", "--always", "--tags"),
        "compiler   %s" % (ccv[0] if ccv else cc),
        "built for  any x86-64 CPU with AVX2 and FMA%s" % ("" if c.get("VFFT_NATIVE", "ON") in ("OFF", "0", "FALSE") else " -- WARNING: VFFT_NATIVE was ON, these binaries may need the build machine's CPU"),
        "versions   %s and %s (the program picks AVX-512 when the CPU runs it)" % ISAS,
        "FP32       %s" % ("yes" if c.get("VFFT_ENABLE_FLOAT") in ("ON", "1", "TRUE") else "no"),
        "comparator FFTW 3.3.10 (%s); Intel MKL: not in this package (build from source with MKL)" % ("fftw/" if WIN else "the system's libfftw3"),
        "",
        "files (SHA-256):",
    ]
    for dp, dn, fn in os.walk(pkg):
        dn.sort()
        for f in sorted(fn):
            if os.path.relpath(dp, pkg).startswith("wisdom"):
                continue
            p = os.path.join(dp, f)
            lines.append("  %s  %s" % (sha256(p), os.path.relpath(p, pkg).replace(os.sep, "/")))
    io.open(os.path.join(pkg, "BUILD.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")

    # the archive and its checksum
    if WIN:
        arc = os.path.join(out, name + ".zip")
        with zipfile.ZipFile(arc, "w", zipfile.ZIP_DEFLATED) as z:
            for dp, dn, fn in os.walk(pkg):
                for f in fn:
                    p = os.path.join(dp, f)
                    z.write(p, os.path.relpath(p, out))
    else:
        arc = os.path.join(out, name + ".tar.gz")
        with tarfile.open(arc, "w:gz") as t:
            t.add(pkg, arcname=name)
    h = sha256(arc)
    io.open(arc + ".sha256", "w", encoding="utf-8", newline="\n").write("%s  %s\n" % (h, os.path.basename(arc)))
    print("package  %s" % pkg)
    print("archive  %s (%.1f MB)" % (arc, os.path.getsize(arc) / 1048576.0))
    print("sha256   %s" % h)
    print("programs %s" % ", ".join(f[:-len(EXE)] if EXE else f for f in progs))


if __name__ == "__main__":
    main()
