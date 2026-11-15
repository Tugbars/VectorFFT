"""make_download.py -- the ready-built download of VectorFFT for the system it runs on.

It builds the library with -DVFFT_BUILD_FRONT=ON (libvfft in front of libvfft_avx2 and libvfft_avx512, for any
CPU with AVX2 and FMA), puts what a user needs into one folder named vectorfft, checks that folder by building
and running check.c in a copy of it, and packs it:

    Windows   vectorfft-<version>-windows-x64.zip
    Linux     vectorfft-<version>-linux-x64.tar.gz

The folder holds the three library files (on Windows also libvfft.dll.a, which the compiler reads), vfft.h and
vfft_diagnostics.h, the wisdom folder (the shards of every CPU folder of src/wisdom, as cmake --install copies
them), LICENSE, README.txt, and the check: check.c with check.bat or check.sh, and on Windows check.exe.

    python src/tools/release/make_download.py                  build in <repo>/build-download, pack there
    python src/tools/release/make_download.py --float          with the FP32 library (vfftf.h)
    python src/tools/release/make_download.py --build <dir>    another build folder (keep its path short on Windows)
    python src/tools/release/make_download.py --pack-only      pack an existing build again

Needs CMake, Ninja, GCC and, on Windows, Python 3 for the build itself. The Linux archive runs on a Linux whose
C library is at least as new as the one that built it; the script prints the version it needs."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WINDOWS = os.name == "nt"


def run(cmd, cwd=None, env=None, quiet=False):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                       stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None, text=True)
    if r.returncode != 0:
        if quiet and r.stdout:
            print(r.stdout[-3000:])
        raise SystemExit("make_download: %s failed (exit %d)" % (cmd[0], r.returncode))
    return r.stdout or ""


def version():
    m = re.search(r"project\(VectorFFT_dag\s+VERSION\s+(\d+)\.(\d+)\.(\d+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    if not m:
        raise SystemExit("make_download: no version in CMakeLists.txt")
    return m.groups()


def readme(ver, with_float):
    lib = "libvfft.dll, libvfft_avx2.dll, libvfft_avx512.dll" if WINDOWS else "libvfft.so, libvfft_avx2.so, libvfft_avx512.so"
    lines = [
        "VectorFFT %s (beta), ready built, for %s x64" % (ver, "Windows" if WINDOWS else "Linux"),
        "",
        "Needs a CPU with AVX2 and FMA (Intel Core since 2013, any AMD Ryzen) and GCC" + (" (MinGW-w64)." if WINDOWS else "."),
        "",
        "1. Check that it works here: " + ("double-click check.bat" if WINDOWS else "sh check.sh"),
        "",
        "2. Build a program. Save it in this folder, then:",
        "",
        "     gcc app.c -I. -L. -lvfft -o app.exe" if WINDOWS else "     gcc app.c -I. -L. -lvfft -lm -Wl,-rpath,'$ORIGIN' -o app",
        "     .\\app.exe" if WINDOWS else "     ./app",
        "",
        "   From another folder, name this one: -I<this folder> -L<this folder>, and "
        + ("put <this folder> on PATH" if WINDOWS else "-Wl,-rpath,<this folder>") + ".",
        "",
        "3. Keep these together: %s and the folder wisdom." % lib,
        "   wisdom holds the plans measured on your CPU. The first use of a transform is measured for seconds",
        "   and saved there; after that it is read at once. Put this folder where you can write.",
        "",
        "In this folder",
        "  vfft.h              the header your program includes",
        "  %-19s the library your program links" % ("libvfft.dll" if WINDOWS else "libvfft.so"),
        "  libvfft_avx2, libvfft_avx512   the two libraries it loads from, by what your CPU runs",
    ]
    if WINDOWS:
        lines.append("  libvfft.dll.a       what GCC reads when it links your program")
    if with_float:
        lines.append("  vfftf.h             the FP32 (float) functions: 1D complex, power-of-two lengths")
    lines += [
        "  vfft_diagnostics.h  extra facts for logs and tests; not needed to compute a transform",
        "  wisdom              the plans, one folder for each CPU",
        "  check.c, " + ("check.bat, check.exe" if WINDOWS else "check.sh") + "   the check of step 1",
        "",
        "This is a beta: when you replace these files with a newer version, compile your program again.",
        "",
        "The guide: https://github.com/Tugbars/VectorFFT (how-to-use/VectorFFT-Guide.pdf, Chapter 1).",
        "",
    ]
    return "\n".join(lines)


def stage(build, out, ver, with_float):
    """The folder vectorfft as a user gets it."""
    major, minor, _ = ver
    v = ".".join(ver)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if WINDOWS:
        names = ["libvfft.dll", "libvfft.dll.a", "libvfft_avx2.dll", "libvfft_avx512.dll"]
    else:
        names = ["libvfft.so.%s" % v, "libvfft_avx2.so", "libvfft_avx512.so"]
    for n in names:
        src = build / n
        if not src.is_file() or src.stat().st_size == 0:
            raise SystemExit("make_download: the build has no %s" % n)
        shutil.copy2(src, out / n)
    if not WINDOWS:     # the names a program is linked and started with
        os.symlink("libvfft.so.%s" % v, out / ("libvfft.so.%s.%s" % (major, minor)))
        os.symlink("libvfft.so.%s.%s" % (major, minor), out / "libvfft.so")
    for h in ["vfft.h", "vfft_diagnostics.h"] + (["vfftf.h"] if with_float else []):
        shutil.copy2(ROOT / "include" / h, out / h)
    shards = 0
    for pat in ("wisdom2_*.txt", "wisdom2f_*.txt"):
        for f in sorted((ROOT / "src" / "wisdom").glob("*/" + pat)):
            dst = out / "wisdom" / f.parent.name / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            shards += 1
    if not shards:
        raise SystemExit("make_download: src/wisdom holds no shards")
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")
    (out / "README.txt").write_text(readme(v, with_float), encoding="utf-8", newline="\r\n" if WINDOWS else "\n")
    shutil.copy2(HERE / "check.c", out / "check.c")
    if WINDOWS:
        # cmd reads a batch file with Windows line ends
        text = (HERE / "check.bat").read_text(encoding="utf-8").replace("\r\n", "\n")
        (out / "check.bat").write_text(text, encoding="utf-8", newline="\r\n")
    else:
        text = (HERE / "check.sh").read_text(encoding="utf-8").replace("\r\n", "\n")
        (out / "check.sh").write_text(text, encoding="utf-8", newline="\n")
        os.chmod(out / "check.sh", 0o755)
    return shards


def check(folder, compiler, with_float, keep_exe):
    """Build check.c against the folder and run it there; returns the program's output."""
    flags = ["-DCHECK_FLOAT"] if with_float else []
    if WINDOWS:
        cmd = [compiler, "-O2", "check.c"] + flags + ["-I.", "-L.", "-lvfft", "-o", "check.exe"]
        exe = folder / "check.exe"
    else:
        cmd = [compiler, "-O2", "check.c"] + flags + ["-I.", "-L.", "-lvfft", "-lm", "-Wl,-rpath,$ORIGIN", "-o", "check"]
        exe = folder / "check"
    run(cmd, cwd=folder)
    env = {k: v for k, v in os.environ.items() if k not in ("VFFT_ISA", "VFFT_WISDOM_DIR")}
    r = subprocess.run([str(exe)], cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    if r.returncode != 0:
        raise SystemExit("make_download: the check failed in %s (exit %d): nothing is packed" % (folder, r.returncode))
    if not keep_exe:
        exe.unlink()
    return r.stdout


def needs_glibc(folder):
    """The newest C library version the Linux libraries ask for."""
    newest = (0, 0)
    for so in folder.glob("libvfft*.so*"):
        if so.is_symlink():
            continue
        out = subprocess.run(["objdump", "-T", str(so)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout
        for a, b in re.findall(r"GLIBC_(\d+)\.(\d+)", out):
            newest = max(newest, (int(a), int(b)))
    return "%d.%d" % newest


def main():
    ap = argparse.ArgumentParser(description="Build and pack the ready-built download of VectorFFT.")
    ap.add_argument("--build", default=str(ROOT / "build-download"), help="the build folder")
    ap.add_argument("--float", action="store_true", help="with the FP32 library (vfftf.h)")
    ap.add_argument("--compiler", default="gcc")
    ap.add_argument("--pack-only", action="store_true", help="the build folder is built already")
    a = ap.parse_args()
    build = Path(a.build).resolve()
    ver = version()
    v = ".".join(ver)
    name = "vectorfft-%s-%s-x64" % (v, "windows" if WINDOWS else "linux")

    if not a.pack_only:
        print("== build (two instruction sets: this takes a few minutes)", flush=True)
        run(["cmake", "-S", ROOT, "-B", build, "-G", "Ninja", "-DCMAKE_C_COMPILER=" + a.compiler,
             "-DCMAKE_BUILD_TYPE=Release", "-DVFFT_BUILD_FRONT=ON", "-DVFFT_BUILD_GAUNTLET=OFF",
             "-DVFFT_BUILD_EXAMPLES=OFF", "-DVFFT_INSTALL=OFF", "-DVFFT_ENABLE_FLOAT=" + ("ON" if a.float else "OFF")],
            quiet=True)
        run(["cmake", "--build", build], quiet=True)
    elif a.float != ((build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace").count("VFFT_ENABLE_FLOAT:BOOL=ON") > 0):
        raise SystemExit("make_download: --float does not match how %s was built" % build)

    print("== the folder", flush=True)
    folder = build / "download" / "vectorfft"
    shards = stage(build, folder, ver, a.float)

    print("== the check, in a copy (the folder that is packed keeps the wisdom as shipped)", flush=True)
    trial = build / "download-check" / "vectorfft"
    if trial.parent.exists():
        shutil.rmtree(trial.parent)
    shutil.copytree(folder, trial, symlinks=True)
    check(trial, a.compiler, a.float, keep_exe=True)
    if WINDOWS:     # the ready check program, for a PC without a compiler
        shutil.copy2(trial / "check.exe", folder / "check.exe")

    print("== pack", flush=True)
    if WINDOWS:
        archive = build / (name + ".zip")
        if archive.exists():
            archive.unlink()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    z.write(f, Path("vectorfft") / f.relative_to(folder))
    else:
        archive = build / (name + ".tar.gz")
        if archive.exists():
            archive.unlink()
        with tarfile.open(archive, "w:gz") as t:
            t.add(folder, arcname="vectorfft")
    files = sorted(p.name for p in folder.iterdir())
    print("%s  (%.1f MB)" % (archive, archive.stat().st_size / 1048576.0))
    print("in it: vectorfft/ with %s; wisdom: %d shards" % (", ".join(files), shards))
    if not WINDOWS:
        print("this archive needs a Linux with the C library (glibc) %s or newer" % needs_glibc(folder))
    return 0


if __name__ == "__main__":
    sys.exit(main())
