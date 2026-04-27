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

CORE_NEW       = ROOT / 'src' / 'core'
CORE_PROD      = ROOT / 'src' / 'stride-fft' / 'core'
TUNED_GEN      = ROOT / 'src' / 'vectorfft_tune' / 'generated'

# stride-fft codelet dirs are NOT on the include path for AVX2/AVX-512
# builds. All SIMD codelets come from vectorfft_tune/generated/r{R}/.
# For scalar fallback (no AVX2/AVX-512) the registry's scalar branch
# still includes stride-fft scalar headers — that path is handled in
# build_includes() below by adding the scalar codelet dir conditionally.
CODELETS_SCALAR = ROOT / 'src' / 'stride-fft' / 'codelets' / 'scalar'

# Tuned radixes whose dispatcher dirs need -I. These match the wisdom_bridge
# include list and the registry's _REG_TUNED_FULL[_*] calls.
TUNED_RADIXES = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 20, 25, 32, 64]


def detect_toolchain():
    cc = os.environ.get('CC', 'icx')
    cc_basename = Path(cc).name.lower()
    is_windows = os.name == 'nt'
    is_icx = 'icx' in cc_basename
    is_msvc_style = cc_basename in ('cl', 'cl.exe', 'icx-cl', 'icx-cl.exe', 'icl', 'icl.exe')
    return {
        'cc': cc, 'is_windows': is_windows,
        'is_icx': is_icx, 'is_msvc_style': is_msvc_style,
    }


def build_includes() -> list[str]:
    """Build the -I flag list. Order matters — first match wins.

    Layout (AVX2 / AVX-512 build):
        src/core                                 (new core overrides)
        src/stride-fft/core                      (factorizer, threads, env...)
        src/vectorfft_tune/generated/r2          (R=2 bootstrap)
        src/vectorfft_tune/generated/r{3..64}    (per-radix codelets +
                                                  aux + dispatchers)

    No stride-fft/codelets/avx2 on the path — the new core is fully
    self-reliant on vectorfft_tune for SIMD codelets.
    """
    inc = [str(CORE_NEW), str(CORE_PROD)]
    # R=2 first (single-variant bootstrap)
    r2 = TUNED_GEN / 'r2'
    if r2.is_dir():
        inc.append(str(r2))
    for r in TUNED_RADIXES:
        d = TUNED_GEN / f'r{r}'
        if d.is_dir():
            inc.append(str(d))
        else:
            print(f'  [warn] missing dispatcher dir: {d}', file=sys.stderr)
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


def build_cmd(tc, src_c, out_bin, mkl=False):
    mkl_inc, mkl_lib = (None, None)
    if mkl:
        mkl_inc, mkl_lib = find_mkl()
        if not mkl_inc or not mkl_lib:
            print('  [error] --mkl requested but MKL not found',
                  file=sys.stderr)
            print(f'  set MKLROOT or install Intel oneAPI MKL', file=sys.stderr)
            sys.exit(2)
        print(f'  [mkl] include: {mkl_inc}')
        print(f'  [mkl] libs:    {mkl_lib}')

    if tc['is_msvc_style']:
        # MSVC-style: /I instead of -I, /Fe for output
        flags = ['/O2', '/arch:AVX2', '/fp:fast', '/wd4244', '/wd4267']
        inc = [a.replace('-I', '/I') for a in build_includes()]
        if mkl:
            flags += ['/DVFFT_HAS_MKL', '/DMKL_ILP64']
            inc += [f'/I{mkl_inc}']
        cmd = [tc['cc']] + flags + inc + [str(src_c), f'/Fe:{out_bin}']
        if mkl:
            cmd += [f'/link', f'/LIBPATH:{mkl_lib}',
                    'mkl_intel_ilp64.lib', 'mkl_sequential.lib', 'mkl_core.lib']
        return cmd

    # GCC-style (icx, gcc, clang).
    # _CRT_SECURE_NO_WARNINGS suppresses MSVC's fopen/sscanf deprecation
    # warnings — they spam thousands of lines and bury real errors.
    flags = ['-O2', '-mavx2', '-mfma',
             '-D_CRT_SECURE_NO_WARNINGS',
             '-Wno-overflow', '-Wno-implicit-function-declaration',
             '-Wno-unused-function', '-Wno-unknown-argument',
             '-Wno-deprecated-declarations']
    if mkl:
        flags += ['-DVFFT_HAS_MKL', '-DMKL_ILP64', f'-I{mkl_inc}']
    cmd = [tc['cc']] + flags + build_includes() + [str(src_c), '-o', str(out_bin)]
    if tc['is_windows'] and tc['is_icx']:
        cmd.append('-fuse-ld=lld')
    if mkl:
        if tc['is_windows']:
            # lld-link doesn't honor -L; pass full lib paths instead.
            cmd += [str(Path(mkl_lib) / 'mkl_intel_ilp64.lib'),
                    str(Path(mkl_lib) / 'mkl_sequential.lib'),
                    str(Path(mkl_lib) / 'mkl_core.lib')]
        else:
            cmd += [f'-L{mkl_lib}',
                    '-lmkl_intel_ilp64', '-lmkl_sequential', '-lmkl_core',
                    '-lpthread', '-lm', '-ldl']
    if not tc['is_windows']:
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
    ap.add_argument('--src', default=str(HERE / 'test_tuned_core.c'))
    ap.add_argument('--mkl', action='store_true',
                    help='Link Intel MKL (ILP64 sequential). Adds '
                         '-DVFFT_HAS_MKL -DMKL_ILP64 and the three '
                         'libs (mkl_intel_ilp64, mkl_sequential, mkl_core). '
                         'Requires MKLROOT or oneAPI default install path.')
    args = ap.parse_args()

    tc = detect_toolchain()
    print('-' * 60)
    print(' Tuned core build harness')
    print('-' * 60)
    print(f' Compiler:  {tc["cc"]} ({"MSVC-style" if tc["is_msvc_style"] else "GCC-style"})')
    print(f' Source:    {args.src}')
    print(f' Includes:  {len(build_includes())} dirs')
    print('-' * 60)

    src = Path(args.src)
    if not src.exists():
        print(f'[error] source not found: {src}', file=sys.stderr)
        return 1

    stem = src.stem
    out_bin = HERE / (stem + '.exe' if tc['is_windows'] else stem)
    cmd = build_cmd(tc, src, out_bin, mkl=args.mkl)

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
    rc = subprocess.run([str(out_bin)]).returncode
    print(f'[run] exit code {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
