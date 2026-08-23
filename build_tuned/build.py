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

Environment:
    VFFT_BUILD_JOBS   compile parallelism (default: CPU count; 1 = serial)
    VFFT_ISA          avx2 | avx512   VFFT_ASAN   build with AddressSanitizer
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
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
DAG          = ROOT / 'src' / 'dag-fft-compiler'   # the compiler (generator + generated + jit)
DAG_CORE     = ROOT / 'src' / 'core'               # the runtime library (moved out of the compiler)
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
        DAG / 'codelets' / 'strided' / DAG_ISA,  # strided rows (6a35-6a45): c2c mono + r2c/c2r two-for-one
        DAG / 'codelets' / 'il'   / DAG_ISA,   # RETIRED (derived population deleted 2026-07-24); dir kept as the once-home
        DAG / 'codelets' / 'zil'  / DAG_ISA / 'pure_il',         # PURE IL, packed complex throughout (codelet_cil.ml + codelet_zil.ml)
        DAG / 'codelets' / 'zil'  / DAG_ISA / 'pure_il' / 'tangent',  # tangent-interior variants (il_kv variant 3); see that dir's README
        DAG / 'codelets' / 'zil'  / DAG_ISA / 'boundary_split',  # cascade N>=2048: IL at the edges, SPLIT interior (codelet_zsplit.ml)
        DAG / 'codelets' / 'trig' / DAG_ISA,   # trig (DCT/DST) specializations
    ]
    srcs: list[str] = []
    for d in dirs:
        if d.is_dir():
            # *_emit.c are generator bit-gate twins of the hand codelets — same symbol
            # names, so linking both would collide; the gates compile them standalone.
            srcs += [str(p) for p in d.glob('*.c') if not p.name.endswith('_emit.c')]
        else:
            print(f'  [warn] codelet dir missing: {d}', file=sys.stderr)
    return sorted(srcs)


# ── Parallel + incremental compile machinery ────────────────────────────────
# Until 2026-08-23 every .c went through a SINGLE gcc process: 899 codelets in
# one `gcc -c @srcs.rsp`, then the driver TUs in one more. gcc does not
# parallelise inside a process, so 31 of this box's 32 hardware threads sat
# idle -- measured 317s for the codelet corpus and 137s for the two driver TUs.
# Compilation is now per-object, depfile-tracked, and spread over a pool.

def _jobs() -> int:
    """Compile parallelism. VFFT_BUILD_JOBS overrides; set it to 1 to get the
    old one-at-a-time behaviour back when bisecting a compiler problem."""
    j = os.environ.get('VFFT_BUILD_JOBS')
    if j:
        return max(1, int(j))
    return max(1, os.cpu_count() or 4)


def _flags_key(parts) -> str:
    """Short hash of everything that changes an object's contents (compiler,
    flags, -I list). Objects keyed by this can never be silently reused across
    a flag change -- e.g. a --mkl object vs the same TU built without it."""
    return hashlib.sha1(repr(list(parts)).encode('utf-8')).hexdigest()[:12]


def _deps_of(dep_file: Path) -> list[Path]:
    """Parse a gcc -MMD depfile into the files the object depends on.

    Hand-tokenised rather than split(), because both Windows hazards are live
    here: paths carry drive-letter colons (handled by giving -MT a colon-free
    target, so partition(':') finds the real separator), and the MKL include
    dir lives under "Program Files (x86)", which gcc emits with the space
    backslash-escaped. A naive whitespace split shreds that into non-existent
    fragments, every object then looks stale, and the cache never hits."""
    BS = chr(92)
    try:
        txt = dep_file.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    txt = txt.replace(BS + chr(13) + chr(10), ' ').replace(BS + chr(10), ' ')
    _, sep, rhs = txt.partition(':')
    if not sep:
        return []
    out, cur, i = [], '', 0
    while i < len(rhs):
        c = rhs[i]
        if c == BS and i + 1 < len(rhs) and rhs[i + 1] in ' ' + BS + '#':
            cur += rhs[i + 1]
            i += 2
            continue
        if c.isspace():
            if cur:
                out.append(cur)
                cur = ''
            i += 1
            continue
        cur += c
        i += 1
    if cur:
        out.append(cur)
    return [Path(t) for t in out]


def _is_stale(src: Path, obj: Path) -> bool:
    """Rebuild obj? Every uncertainty answers YES -- a needless rebuild costs
    seconds, a wrong cache hit links a stale object. The depfile is what makes
    reuse safe at all: editing a core header invalidates the objects that
    include it, which an mtime check on the .c alone would miss entirely."""
    if not obj.exists():
        return True
    dep = obj.with_suffix('.d')
    if not dep.exists():
        return True                        # no dependency record -> cannot trust it
    deps = _deps_of(dep)
    if not deps:
        return True                        # unparseable -> rebuild
    try:
        o_mt = obj.stat().st_mtime
    except OSError:
        return True
    for d in [src] + deps:
        try:
            if d.stat().st_mtime > o_mt:
                return True
        except OSError:
            return True                    # a dependency vanished -> rebuild
    return False


def _compile_many(cmds: list, env, label: str) -> bool:
    """Run compile commands across a pool (the work happens in the gcc
    subprocesses, so threads are the right handle). True on success."""
    if not cmds:
        return True
    fails, warns = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=_jobs()) as ex:
        futs = [ex.submit(subprocess.run, c, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', env=env)
                for c in cmds]
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            if r.returncode != 0:
                fails.append(r.stderr)
            elif r.stderr.strip():
                warns.append(r.stderr)
    if fails:
        print(f'  [{label}] compile FAILED ({len(fails)} of {len(cmds)}):',
              file=sys.stderr)
        print(fails[0][:4000], file=sys.stderr)
        return False
    if warns:
        print(f'  [{label}] warnings in {len(warns)} TU(s); first:')
        print('\n'.join(warns[0].splitlines()[:10]))
    return True


def _compile_cmd(tc, src: Path, obj: Path, flags: list) -> list:
    """One TU -> one object, plus the depfile that drives _is_stale. -MT gets
    the bare object NAME so the depfile target carries no drive-letter colon
    (see _deps_of)."""
    return ([tc['cc']] + flags +
            ['-MMD', '-MF', str(obj.with_suffix('.d')), '-MT', obj.name,
             '-c', str(src), '-o', str(obj)])


def dag_codelet_lib(tc) -> str | None:
    """Compile the dag codelets into a CACHED static lib. They're OCaml-
    generated and unchanged between calibration runs, so a driver rebuild is a
    fast relink rather than a recompile of the corpus. Delete
    src/dag-fft-compiler/.obj to force a clean rebuild.

    Per-object and parallel since 2026-08-23. The previous version took the max
    mtime over all sources and, if ANY single file was newer, deleted every .o
    and rebuilt the whole corpus in one serial gcc -- regenerating one codelet
    cost a measured 317s (the "one-time ~100s" in the old docstring dated from
    when this tree held ~300 codelets, not 899). Now only stale objects
    recompile, across _jobs() processes: a clean build is ~45s and a
    one-codelet regen is ~1s.

    The objdir path is deliberately unchanged -- dag_write_jit_rsp points the
    JIT runtime at these same .o files.
    """
    srcs = [Path(s) for s in dag_codelet_srcs()]
    if not srcs:
        return None
    objdir = DAG / '.obj' / DAG_ISA
    objdir.mkdir(parents=True, exist_ok=True)
    lib = objdir / 'libdagcodelets.a'

    cflags = ['-O3', '-mavx2', '-mfma', '-march=native', '-fpermissive', '-w']
    if os.environ.get('VFFT_ASAN'):
        cflags += ['-fsanitize=address', '-g', '-fno-omit-frame-pointer']
    flags = cflags + build_includes()

    # This objdir has a FIXED path (the JIT rsp points at it), so the flag key
    # can't live in the path the way it does for the driver objects. Stamp it
    # in a file instead and wipe on mismatch -- otherwise an ASAN or -I change
    # would quietly relink objects built under the old flags.
    key = _flags_key([tc['cc']] + flags)
    stamp = objdir / '_flags.key'
    if stamp.exists() and stamp.read_text(encoding='ascii').strip() != key:
        print('  [codelets] flags changed -> full rebuild')
        for old in list(objdir.glob('*.o')) + list(objdir.glob('*.d')):
            old.unlink()
        if lib.exists():
            lib.unlink()

    # Objects are named by source stem; dag_codelet_srcs() has no duplicate
    # basenames across families, which the old single-gcc build already relied
    # on (it wrote every .o into one cwd).
    pairs = [(s, objdir / (s.stem + '.o')) for s in srcs]

    # Drop objects whose source is gone -- a regen that renames or retires a
    # codelet would otherwise leave it in the archive forever.
    keep = {o.name for (_, o) in pairs}
    for old in list(objdir.glob('*.o')):
        if old.name not in keep:
            old.unlink()
            old.with_suffix('.d').unlink(missing_ok=True)

    stale = [(s, o) for (s, o) in pairs if _is_stale(s, o)]
    if not stale and lib.exists():
        print(f'  [codelets] cached lib ({len(srcs)} codelets)')
        stamp.write_text(key, encoding='ascii')
        return str(lib)

    if stale:
        print(f'  [codelets] compiling {len(stale)} of {len(srcs)} codelets '
              f'on {_jobs()} jobs ...', flush=True)
        t0 = time.time()
        if not _compile_many([_compile_cmd(tc, s, o, flags) for (s, o) in stale],
                             build_env(tc), 'codelets'):
            return None
        print(f'  [codelets] compiled {len(stale)} in {time.time() - t0:.1f}s')

    objs = sorted(o for (_, o) in pairs)
    ar = str(Path(tc['cc']).with_name(Path(tc['cc']).name.replace('gcc', 'ar')))
    # Per-process temp names, then an atomic replace. Builds DO run concurrently
    # here (a second session was observed mid-build on 2026-08-23), and the old
    # unlink-then-ar sequence left a window in which the archive did not exist:
    # a concurrent link in that window fails outright. os.replace is atomic
    # within a volume, so a reader sees either the old lib or the new one.
    objs_rsp = objdir / f'_objs.{os.getpid()}.rsp'
    tmp_lib = objdir / f'libdagcodelets.{os.getpid()}.a'
    objs_rsp.write_text('\n'.join(o.as_posix() for o in objs), encoding='ascii')
    if tmp_lib.exists():
        tmp_lib.unlink()
    ra = subprocess.run([ar, 'rcs', str(tmp_lib), f'@{objs_rsp}'],
                        capture_output=True, text=True, encoding='utf-8', errors='replace')
    objs_rsp.unlink(missing_ok=True)
    if ra.returncode != 0:
        tmp_lib.unlink(missing_ok=True)
        print(f'  [codelets] ar FAILED:\n{ra.stderr[:800]}', file=sys.stderr)
        return None
    os.replace(tmp_lib, lib)
    stamp.write_text(key, encoding='ascii')
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
    inc = [str(ROOT / 'include'), str(DAG), str(DAG_GEN), str(DAG / 'jit')] + [str(d) for d in core_dirs]
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


def build_cmd(tc, src_c, out_bin, mkl=False, fftw=False, jit=False, extra_srcs=None,
              split=False):
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
    if os.environ.get('VFFT_ASAN'):
        flags += ['-fsanitize=address', '-g', '-fno-omit-frame-pointer']
    if mkl:
        # LP64 (mkl_rt), NOT ILP64: ILP64's 8-byte MKL_LONG corrupts the DFTI
        # strides array -> "Inconsistent configuration parameters" at DftiCommit.
        flags += ['-DVFFT_HAS_MKL', f'-I{mkl_inc}']
    if fftw:
        flags += ['-DVFFT_HAS_FFTW', f'-I{fftw_inc}']
    if jit:
        flags = flags + ['-DVFFT_USE_JIT']   # bench resolves via vfft_proto_plan_jit_fwd
    base_srcs = [str(src_c)] + [str(s) for s in (extra_srcs or [])]
    cflags = flags + build_includes()

    # Everything from here down is LINK input. It is kept separate from cflags
    # so build_gcc() can compile the driver TUs to cached objects in parallel
    # and then link them; dag codelets still come from the CACHED static lib.
    link_args = []
    lib = dag_codelet_lib(tc)
    if lib:
        link_args.append(lib)
    if jit:
        dag_write_jit_rsp()                  # JIT runtime links build.py's cached .o
    if tc['is_windows'] and tc['is_icx']:
        link_args.append('-fuse-ld=lld')
    if mkl:
        # LP64 single dynamic lib (mkl_rt). Runtime needs <mkl>/latest/bin on PATH
        # (mkl_rt.2.dll) + the mingw bin (libwinpthread-1.dll).
        if tc['is_windows']:
            link_args += [str(Path(mkl_lib) / 'mkl_rt.lib')]
        else:
            link_args += [f'-L{mkl_lib}', '-lmkl_rt', '-lpthread', '-lm', '-ldl']
    if fftw:
        if tc['is_windows']:
            link_args += [str(Path(fftw_lib) / 'fftw3.lib')]
        else:
            link_args += [f'-L{fftw_lib}', '-lfftw3', '-lm']
    # -lm for gcc (mingw on Windows has libm.a; Linux needs it). NOT for MSVC
    # or icx-on-Windows (MSVC CRT supplies libm).
    if not tc['is_msvc_style'] and not (tc['is_windows'] and tc['is_icx']):
        link_args.append('-lm')

    if split:
        return cflags, base_srcs, link_args
    # Single-command form, unchanged in argument ORDER (sources before the
    # codelet lib, lib before MKL/FFTW, -lm last): kept for callers that want
    # one compile+link invocation.
    return [tc['cc']] + cflags + base_srcs + ['-o', str(out_bin)] + link_args


def build_gcc(tc, out_bin, cflags, base_srcs, link_args) -> bool:
    """Compile each driver TU to a CACHED object, in parallel, then link.

    The old path handed gcc every source in one command, so the driver TUs --
    for the vs-MKL bench that is src/core/vfft.c plus the bench itself -- were
    compiled back to back inside one process: 137s measured, on EVERY build,
    even when only one of them had actually changed. Split and cached, an
    unchanged rebuild is a ~0.3s relink and a one-file edit costs just that
    file.

    Objects live under a dir named by a hash of compiler+flags+includes, so a
    --mkl build and a --fftw build (or an ASAN one) can never reuse each
    other's objects."""
    objdir = HERE / '.obj' / _flags_key([tc['cc']] + cflags)
    objdir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for s in base_srcs:
        sp = Path(s)
        # Sources CAN share a stem across trees, so disambiguate the object
        # name with a short hash of the full path.
        pairs.append((sp, objdir / f'{sp.stem}-{_flags_key([str(sp)])[:6]}.o'))

    stale = [(s, o) for (s, o) in pairs if _is_stale(s, o)]
    if stale:
        print(f'  [driver] compiling {len(stale)} of {len(pairs)} TUs on '
              f'{_jobs()} jobs: {", ".join(s.name for (s, _) in stale)}', flush=True)
        t0 = time.time()
        if not _compile_many([_compile_cmd(tc, s, o, cflags) for (s, o) in stale],
                             build_env(tc), 'driver'):
            return False
        print(f'  [driver] compiled {len(stale)} in {time.time() - t0:.1f}s')
    else:
        print(f'  [driver] {len(pairs)} TU(s) cached')

    # cflags go on the link line too: that is what the old single-command form
    # did, and it is what carries -fsanitize=address (VFFT_ASAN) through to the
    # link. The compile-only flags among them are inert here.
    link = ([tc['cc']] + cflags + [str(o) for (_, o) in pairs]
            + ['-o', str(out_bin)] + link_args)
    r = subprocess.run(link, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', env=build_env(tc))
    if r.returncode != 0:
        print(f'  [driver] link FAILED:', file=sys.stderr)
        print(r.stderr[:4000], file=sys.stderr)
        return False
    if r.stderr.strip():
        print('  [driver] link warnings:')
        print('\n'.join(r.stderr.splitlines()[:10]))
    return True


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
                    help='Compile src/core/vfft.c alongside the source. Use this '
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
        extra_srcs.append(DAG_CORE / 'vfft.c')   # canonical src/core/vfft.c (old src/vfft.c retired)
    print(f'[compile] {tc["cc"]} ... -> {out_bin.name}', flush=True)
    t0 = time.time()

    if tc['is_msvc_style']:
        # MSVC compiles the whole corpus in one cl invocation (no cached lib on
        # that path), so it stays a single command.
        cmd = build_cmd(tc, src, out_bin, mkl=args.mkl, fftw=args.fftw,
                        jit=args.jit, extra_srcs=extra_srcs)
        result = subprocess.run(cmd, capture_output=True,
                                text=True, encoding='utf-8', errors='replace',
                                env=build_env(tc))
        if result.returncode != 0:
            print(f'[compile] FAILED ({time.time()-t0:.1f}s)')
            # Print stderr in full — Intel ICE/include errors get cut off otherwise
            print(result.stderr[:8000])
            return 1
        if result.stderr.strip():
            # Warnings only — show the first few lines so user sees them
            head = '\n'.join(result.stderr.splitlines()[:15])
            if head:
                print(f'[compile] warnings:\n{head}')
    else:
        cflags, base_srcs, link_args = build_cmd(
            tc, src, out_bin, mkl=args.mkl, fftw=args.fftw, jit=args.jit,
            extra_srcs=extra_srcs, split=True)
        if not build_gcc(tc, out_bin, cflags, base_srcs, link_args):
            print(f'[compile] FAILED ({time.time()-t0:.1f}s)')
            return 1

    print(f'[compile] OK ({time.time()-t0:.1f}s)')

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
