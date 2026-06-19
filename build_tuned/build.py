"""
build.py — compile + run the new tuned core test program.

Stays completely separate from CMake. Mirrors the pattern used in
src/vectorfft_tune/common/bench.py: detect compiler, set flags, compile,
run.

Include path order matters
--------------------------
src/core/                          (new — 6 headers override production)
src/stride-fft/core/               (production — supplies factorizer.h,
                                    threads.h, env.h, prefetch.h, etc.)
src/stride-fft/codelets/{isa}/     (production codelets — n1/aux variants)
src/vectorfft_tune/generated/r{R}/ (per-host dispatchers + plan_wisdom)

Anything resolvable in src/core/ shadows the production version, so the
new planner.h, registry.h, executor.h, dp_planner.h, exhaustive.h get
picked up. Everything else falls through to production.

Usage:
    set CC=icx
    python build.py            # compile + run
    python build.py --compile  # compile only, don't run
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent  # repo root: highSpeedFFT/

# ── RE-POINTED to the dag-fft-compiler tree (2026-06-16) ──────────────
# Production (src/core + src/vectorfft_tune) is being retired; this harness
# now builds against dag-fft-compiler. KEY MODEL DIFFERENCE: production
# header-INCLUDED its SIMD codelets via per-radix -I dirs; dag's codelets are
# separately-compiled .c files (codelets/inplace/{isa}/*.c) that get LINKED in
# — the generated registry_{isa}.h holds externs + the init that wires them.
DAG          = ROOT / 'src' / 'dag-fft-compiler'
DAG_CORE     = DAG / 'core'
DAG_GEN      = DAG / 'generator' / 'generated'   # generated registry + spike_wisdom
DAG_ISA      = os.environ.get('VFFT_ISA', 'avx2')  # avx2 | avx512
DAG_CODELETS = DAG / 'codelets' / 'inplace' / DAG_ISA


def dag_codelet_srcs() -> list[str]:
    """All dag SIMD codelet .c files to compile+link (the registry references
    them by symbol). ~300 c2c files + the real-FFT families. Compiled in one
    invocation. The rfft (r2cf + hc2hc) and c2r (r2cb + hc2hc-bwd) codelets live
    in sibling dirs; including them here lets r2c/c2r benches link the same lib."""
    dirs = [
        DAG_CODELETS,                          # c2c in-place
        DAG / 'codelets' / 'rfft' / DAG_ISA,   # r2c forward: r2cf leaf + hc2hc DIT
        DAG / 'codelets' / 'c2r'  / DAG_ISA,   # c2r inverse: r2cb leaf + hc2hc DIF/ranged
        DAG / 'codelets' / 'oop'  / DAG_ISA,   # OOP c2c: n1 + t1p (LEAF/BAILEY2 kinds)
    ]
    srcs: list[str] = []
    for d in dirs:
        if d.is_dir():
            srcs += [str(p) for p in d.glob('*.c')]
        else:
            print(f'  [warn] codelet dir missing: {d}', file=sys.stderr)
    return sorted(srcs)


def dag_codelet_lib(tc) -> str | None:
    """Compile the dag codelets into a CACHED static lib. They're OCaml-
    generated and unchanged between calibration runs, so the lib rebuilds only
    when a codelet .c is newer than it — turning a driver rebuild from a ~100s
    recompile-everything into a fast relink. Delete src/dag-fft-compiler/.obj
    to force a clean rebuild (e.g. after the OCaml pipeline regenerates)."""
    srcs = [Path(s) for s in dag_codelet_srcs()]
    if not srcs:
        return None
    objdir = DAG / '.obj' / DAG_ISA
    objdir.mkdir(parents=True, exist_ok=True)
    lib = objdir / 'libdagcodelets.a'
    newest = max(s.stat().st_mtime for s in srcs)
    if lib.exists() and lib.stat().st_mtime >= newest:
        print(f'  [codelets] cached lib ({len(srcs)} codelets)')
        return str(lib)
    print(f'  [codelets] building static lib from {len(srcs)} codelets '
          f'(one-time ~100s; cached after) ...', flush=True)
    for old in objdir.glob('*.o'):   # clear stale objects (regen may rename/drop)
        old.unlink()
    cflags = ['-O3', '-mavx2', '-mfma', '-march=native', '-fpermissive', '-w']
    srcs_rsp = objdir / '_srcs.rsp'
    srcs_rsp.write_text('\n'.join(s.as_posix() for s in srcs), encoding='ascii')
    r = subprocess.run([tc['cc']] + cflags + build_includes() + ['-c', f'@{srcs_rsp}'],
                       cwd=str(objdir), capture_output=True, text=True,
                       encoding='utf-8', errors='replace', env=build_env(tc))
    if r.returncode != 0:
        print(f'  [codelets] compile FAILED:\n{r.stderr[:800]}', file=sys.stderr)
        return None
    objs = sorted(objdir.glob('*.o'))
    ar = str(Path(tc['cc']).with_name(Path(tc['cc']).name.replace('gcc', 'ar')))
    objs_rsp = objdir / '_objs.rsp'
    objs_rsp.write_text('\n'.join(o.as_posix() for o in objs), encoding='ascii')
    if lib.exists():
        lib.unlink()
    ra = subprocess.run([ar, 'rcs', str(lib), f'@{objs_rsp}'],
                        capture_output=True, text=True, encoding='utf-8', errors='replace')
    if ra.returncode != 0:
        print(f'  [codelets] ar FAILED:\n{ra.stderr[:800]}', file=sys.stderr)
        return None
    print(f'  [codelets] lib built ({len(objs)} objects)')
    return str(lib)


def dag_write_jit_rsp():
    """Point the JIT runtime's codelet response file (jit/generated/codelets.rsp,
    its default VFFT_PROTO_JIT_CODELETS) at build.py's CACHED .obj objects — so the
    --jit build config is fully self-contained in build_tuned (no separate
    jit/build_codelets.ps1 step). The JIT's runtime `gcc -shared` links these .o
    into each emitted single-plan .dll."""
    objdir = DAG / '.obj' / DAG_ISA
    objs = sorted(objdir.glob('*.o'))
    if not objs:
        print('  [jit] no codelet objects to point at (run a build first)', file=sys.stderr)
        return
    rsp = DAG / 'jit' / 'generated' / 'codelets.rsp'
    rsp.parent.mkdir(parents=True, exist_ok=True)
    rsp.write_text('\n'.join(o.as_posix() for o in objs), encoding='ascii')
    print(f'  [jit] codelets.rsp -> {len(objs)} cached objects')


def detect_toolchain():
    # dag dev compiler is gcc (mingw 15.2); production used icx. Override via CC.
    _default_cc = (r'C:\mingw152\mingw64\bin\gcc.exe' if os.name == 'nt' else 'gcc')
    cc = os.environ.get('CC', _default_cc)
    cc_basename = Path(cc).name.lower()
    is_windows = os.name == 'nt'
    is_icx = 'icx' in cc_basename
    is_msvc_style = cc_basename in ('cl', 'cl.exe', 'icx-cl', 'icx-cl.exe', 'icl', 'icl.exe')
    return {
        'cc': cc, 'is_windows': is_windows,
        'is_icx': is_icx, 'is_msvc_style': is_msvc_style,
    }


def build_includes() -> list[str]:
    """-I list for the dag-fft-compiler build. dag headers cross-reference each
    other (and the generated registry) BARE (#include "executor.h"), so every
    core subfolder must be on the -I path. core/ is organized into subfolders
    (engine/, support/, planning/, transforms/{real,trig,fft2d}/, primes/, oop/);
    we walk core/ recursively so a future reorg needs no build edit. SIMD codelets
    are LINKED .c files (see dag_codelet_srcs), not header-included."""
    core_dirs = [DAG_CORE] + sorted(d for d in DAG_CORE.rglob('*') if d.is_dir())
    inc = [str(ROOT / 'include'), str(DAG), str(DAG_GEN)] + [str(d) for d in core_dirs]
    return [f'-I{p}' for p in inc]


def find_mkl():
    """Locate MKL include + lib dirs. Returns (inc_dir, lib_dir) or
    (None, None) if not found. Mirrors the discovery hints in
    src/stride-fft/CMakeLists.txt."""
    mklroot = os.environ.get('MKLROOT')
    inc_candidates = []
    lib_candidates = []
    if mklroot:
        inc_candidates += [Path(mklroot) / 'include']
        lib_candidates += [Path(mklroot) / 'lib',
                           Path(mklroot) / 'lib' / 'intel64']
    inc_candidates += [
        Path(r'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\include'),
        Path(r'C:\Program Files\Intel\oneAPI\mkl\latest\include'),
    ]
    lib_candidates += [
        Path(r'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\lib'),
        Path(r'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\lib\intel64'),
        Path(r'C:\Program Files\Intel\oneAPI\mkl\latest\lib'),
        Path(r'C:\Program Files\Intel\oneAPI\mkl\latest\lib\intel64'),
    ]
    inc = next((p for p in inc_candidates if (p / 'mkl_dfti.h').is_file()), None)
    lib = next((p for p in lib_candidates if (p / 'mkl_intel_ilp64.lib').is_file()
                                          or (p / 'libmkl_intel_ilp64.a').is_file()
                                          or (p / 'libmkl_intel_ilp64.so').is_file()), None)
    return inc, lib


def find_fftw():
    """Locate FFTW3 (vcpkg install). Returns (inc, lib_dir, dll_dir)."""
    candidates = [
        Path(r'C:\vcpkg\installed\x64-windows'),
        Path(r'C:\Users\Tugbars\Desktop\highSpeedFFT\vcpkg\installed\x64-windows'),
    ]
    for root in candidates:
        if (root / 'include' / 'fftw' / 'fftw3.h').is_file():
            return root / 'include' / 'fftw', root / 'lib', root / 'bin'
    return None, None, None


def build_cmd(tc, src_c, out_bin, mkl=False, fftw=False, jit=False, extra_srcs=None):
    mkl_inc, mkl_lib = (None, None)
    fftw_inc, fftw_lib, fftw_dll = (None, None, None)
    if mkl:
        mkl_inc, mkl_lib = find_mkl()
        if not mkl_inc or not mkl_lib:
            print('  [error] --mkl requested but MKL not found',
                  file=sys.stderr)
            print(f'  set MKLROOT or install Intel oneAPI MKL', file=sys.stderr)
            sys.exit(2)
        print(f'  [mkl] include: {mkl_inc}')
        print(f'  [mkl] libs:    {mkl_lib}')
    if fftw:
        fftw_inc, fftw_lib, fftw_dll = find_fftw()
        if not fftw_inc or not fftw_lib:
            print('  [error] --fftw requested but FFTW3 not found in vcpkg',
                  file=sys.stderr)
            sys.exit(2)
        print(f'  [fftw] include: {fftw_inc}')
        print(f'  [fftw] libs:    {fftw_lib}')

    if tc['is_msvc_style']:
        # MSVC-style: /I instead of -I, /Fe for output
        flags = ['/O2', '/arch:AVX2', '/fp:fast', '/wd4244', '/wd4267']
        inc = [a.replace('-I', '/I') for a in build_includes()]
        if mkl:
            flags += ['/DVFFT_HAS_MKL', '/DMKL_ILP64']
            inc += [f'/I{mkl_inc}']
        all_srcs = [str(src_c)] + [str(s) for s in (extra_srcs or [])] + dag_codelet_srcs()
        cmd = [tc['cc']] + flags + inc + all_srcs + [f'/Fe:{out_bin}']
        if mkl:
            cmd += [f'/link', f'/LIBPATH:{mkl_lib}',
                    'mkl_intel_ilp64.lib', 'mkl_sequential.lib', 'mkl_core.lib']
        return cmd

    # GCC-style (icx, gcc, clang).
    # _CRT_SECURE_NO_WARNINGS suppresses MSVC's fopen/sscanf deprecation
    # warnings — they spam thousands of lines and bury real errors.
    flags = ['-O3', '-mavx2', '-mfma', '-march=native', '-fpermissive',
             '-D_CRT_SECURE_NO_WARNINGS',
             '-Wno-overflow', '-Wno-implicit-function-declaration',
             '-Wno-unused-function', '-Wno-unknown-argument',
             '-Wno-incompatible-pointer-types',  # gcc-15: dag codelets' aligned-store casts
             '-Wno-deprecated-declarations']
    if mkl:
        # LP64 (mkl_rt), NOT ILP64: ILP64's 8-byte MKL_LONG corrupts the DFTI
        # strides array -> "Inconsistent configuration parameters" at DftiCommit.
        flags += ['-DVFFT_HAS_MKL', f'-I{mkl_inc}']
    if fftw:
        flags += ['-DVFFT_HAS_FFTW', f'-I{fftw_inc}']
    # Driver + extras compiled here; dag codelets come from a CACHED static lib
    # (built once by dag_codelet_lib) so a driver rebuild is a fast relink, not
    # a ~100s recompile of ~300 codelets.
    if jit:
        flags = flags + ['-DVFFT_USE_JIT']   # bench resolves via vfft_proto_plan_jit_fwd
    base_srcs = [str(src_c)] + [str(s) for s in (extra_srcs or [])]
    cmd = [tc['cc']] + flags + build_includes() + base_srcs
    lib = dag_codelet_lib(tc)
    if lib:
        cmd.append(lib)
    if jit:
        dag_write_jit_rsp()                  # JIT runtime links build.py's cached .o
    cmd += ['-o', str(out_bin)]
    if tc['is_windows'] and tc['is_icx']:
        cmd.append('-fuse-ld=lld')
    if mkl:
        # LP64 single dynamic lib (mkl_rt). Runtime needs <mkl>/latest/bin on PATH
        # (mkl_rt.2.dll) + the mingw bin (libwinpthread-1.dll).
        if tc['is_windows']:
            cmd += [str(Path(mkl_lib) / 'mkl_rt.lib')]
        else:
            cmd += [f'-L{mkl_lib}', '-lmkl_rt', '-lpthread', '-lm', '-ldl']
    if fftw:
        if tc['is_windows']:
            cmd += [str(Path(fftw_lib) / 'fftw3.lib')]
        else:
            cmd += [f'-L{fftw_lib}', '-lfftw3', '-lm']
    # -lm for gcc (mingw on Windows has libm.a; Linux needs it). NOT for MSVC
    # or icx-on-Windows (MSVC CRT supplies libm).
    if not tc['is_msvc_style'] and not (tc['is_windows'] and tc['is_icx']):
        cmd.append('-lm')
    return cmd


def build_env(tc):
    """Build subprocess env. On Windows + ICX, ensure LIB contains the
    Intel oneAPI runtime library directory so lld-link can find
    libircmt.lib, svml_dispmt.lib, libmmt.lib. setvars.bat normally does
    this; we replicate the minimum needed when called from a plain cmd."""
    env = os.environ.copy()
    if not tc['is_windows'] or not tc['is_icx']:
        return env
    # Build LIB path covering: oneAPI runtime, MSVC, Windows SDK (um + ucrt).
    # setvars.bat / vcvarsall.bat normally set this; we replicate it.
    lib_paths = []

    # oneAPI compiler runtime
    for p in (r'C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\lib',
              r'C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib',
              r'C:\Program Files\Intel\oneAPI\compiler\2025.3\lib',
              r'C:\Program Files\Intel\oneAPI\compiler\latest\lib'):
        if Path(p).is_dir():
            lib_paths.append(p)
            break

    # MSVC C runtime — pick the highest-versioned MSVC under VS Community
    msvc_root = Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC')
    if msvc_root.is_dir():
        versions = sorted([d for d in msvc_root.iterdir() if d.is_dir()],
                          reverse=True)
        if versions:
            msvc_lib = versions[0] / 'lib' / 'x64'
            if msvc_lib.is_dir():
                lib_paths.append(str(msvc_lib))

    # Windows SDK (kernel32.lib, uuid.lib, etc.) — pick highest version
    sdk_root = Path(r'C:\Program Files (x86)\Windows Kits\10\Lib')
    if sdk_root.is_dir():
        versions = sorted([d for d in sdk_root.iterdir() if d.is_dir()],
                          reverse=True)
        if versions:
            for sub in ('um', 'ucrt'):
                p = versions[0] / sub / 'x64'
                if p.is_dir():
                    lib_paths.append(str(p))

    if not lib_paths:
        print('  [warn] no system lib dirs found — link may fail')
        return env

    existing = env.get('LIB', '')
    new_lib = ';'.join(lib_paths)
    env['LIB'] = new_lib + (';' + existing if existing else '')
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compile', action='store_true',
                    help='Compile only, do not run')
    ap.add_argument('--src', default=str(HERE / 'test' / 'test_tuned_core.c'),
                    help='Source file to build. Bench files live in build_tuned/, '
                         'tests live in build_tuned/test/.')
    ap.add_argument('--mkl', action='store_true',
                    help='Link Intel MKL (ILP64 sequential). Adds '
                         '-DVFFT_HAS_MKL -DMKL_ILP64 and the three '
                         'libs (mkl_intel_ilp64, mkl_sequential, mkl_core). '
                         'Requires MKLROOT or oneAPI default install path.')
    ap.add_argument('--fftw', action='store_true',
                    help='Link FFTW3 (vcpkg double-precision). Adds '
                         '-DVFFT_HAS_FFTW and fftw3.lib.')
    ap.add_argument('--jit', action='store_true',
                    help='JIT build config: defines VFFT_USE_JIT (bench resolves '
                         'plans via vfft_proto_plan_jit_fwd) + points the JIT '
                         'runtime at build.py-cached codelet objects. All in build_tuned.')
    ap.add_argument('--vfft', action='store_true',
                    help='Compile src/vfft.c alongside the source. Use this '
                         'when the source file uses the public vfft.h API '
                         '(opaque-handle entry points).')
    args = ap.parse_args()

    tc = detect_toolchain()
    print('-' * 60)
    print(' Tuned core build harness')
    print('-' * 60)
    print(f' Compiler:  {tc["cc"]} ({"MSVC-style" if tc["is_msvc_style"] else "GCC-style"})')
    print(f' Source:    {args.src}')
    print(f' Includes:  {len(build_includes())} dirs')
    print('-' * 60)

    src = Path(args.src).resolve()
    if not src.exists():
        print(f'[error] source not found: {src}', file=sys.stderr)
        return 1

    stem = src.stem
    out_bin = src.parent / (stem + '.exe' if tc['is_windows'] else stem)
    extra_srcs = []
    if args.vfft:
        extra_srcs.append(ROOT / 'src' / 'vfft.c')
    cmd = build_cmd(tc, src, out_bin, mkl=args.mkl, fftw=args.fftw, jit=args.jit,
                    extra_srcs=extra_srcs)

    print(f'[compile] {tc["cc"]} ... -> {out_bin.name}', flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True,
                            text=True, encoding='utf-8', errors='replace',
                            env=build_env(tc))
    if result.returncode != 0:
        print(f'[compile] FAILED ({time.time()-t0:.1f}s)')
        # Print stderr in full — Intel ICE/include errors get cut off otherwise
        print(result.stderr[:8000])
        return 1
    print(f'[compile] OK ({time.time()-t0:.1f}s)')
    if result.stderr.strip():
        # Warnings only — show the first few lines so user sees them
        head = '\n'.join(result.stderr.splitlines()[:15])
        if head:
            print(f'[compile] warnings:\n{head}')

    if args.compile:
        return 0

    print(f'[run] {out_bin}', flush=True)
    run_env = os.environ.copy()
    if args.fftw:
        _, _, fftw_dll = find_fftw()
        if fftw_dll and fftw_dll.is_dir():
            run_env['PATH'] = str(fftw_dll) + os.pathsep + run_env.get('PATH', '')
    rc = subprocess.run([str(out_bin)], env=run_env).returncode
    print(f'[run] exit code {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
