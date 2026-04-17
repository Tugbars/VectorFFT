"""
run_r32_bench.py — full R=32 bench with parallel compile and incremental
checkpointing. Designed to survive turn boundaries:
  - Build phase: parallel per-codelet compile, run in background with poll
  - Run phase: C harness writes measurements.json incrementally,
    one line per (candidate, ios, me, dir); can resume from partial data.

Usage:
    python run_r32_bench.py --phase generate
    python run_r32_bench.py --phase build
    python run_r32_bench.py --phase run  (can be resumed)
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import candidates

HERE = Path(__file__).parent
GEN = HERE / 'gen_radix32_buffered.py'
STAGING = HERE / 'staging'
BUILD = HERE / 'build'
OBJS = BUILD / 'objs'
MEAS = HERE / 'measurements.jsonl'  # jsonl for streaming append


# ────────────────────────────────────────────────────
# Phase 1: generate
# ────────────────────────────────────────────────────
def phase_generate():
    STAGING.mkdir(exist_ok=True)
    cands = candidates.enumerate_all()
    print(f"[generate] {len(cands)} candidates")
    t0 = time.time()
    for i, c in enumerate(cands):
        r = subprocess.run([sys.executable, str(GEN)] + c.cli_args(),
                           capture_output=True, timeout=60)
        if r.returncode != 0:
            print(f"  FAIL {c.id()}: {r.stderr.decode('utf-8','replace')[:200]}")
            sys.exit(1)
        (STAGING / c.header_name()).write_bytes(r.stdout)
        if i % 20 == 19 or i == len(cands)-1:
            print(f"  [{i+1}/{len(cands)}] {time.time()-t0:.1f}s")
    print(f"[generate] done in {time.time()-t0:.1f}s")


# ────────────────────────────────────────────────────
# Phase 2: parallel build
# ────────────────────────────────────────────────────

HARNESS_PREAMBLE = r'''
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>

#ifdef _WIN32
  #include <windows.h>
  #include <malloc.h>
#endif

/* CPUID via the compiler's preferred intrinsic header */
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
  #include <intrin.h>
  static void _cpuid_count(int leaf, int subleaf, int *eax, int *ebx, int *ecx, int *edx) {
      int r[4]; __cpuidex(r, leaf, subleaf);
      *eax = r[0]; *ebx = r[1]; *ecx = r[2]; *edx = r[3];
  }
#else
  #include <cpuid.h>
  static void _cpuid_count(int leaf, int subleaf, int *eax, int *ebx, int *ecx, int *edx) {
      unsigned int a, b, c, d;
      __cpuid_count(leaf, subleaf, a, b, c, d);
      *eax = (int)a; *ebx = (int)b; *ecx = (int)c; *edx = (int)d;
  }
#endif

/* ─── timing: portable high-resolution monotonic clock in ns ─── */
#ifdef _WIN32
static double now_ns(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)freq.QuadPart;
}
#else
static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/* ─── aligned alloc / free ─── */
#ifdef _WIN32
static void *aalloc(size_t b) {
    void *p = _aligned_malloc(b, 64);
    if (p) memset(p, 0, b);
    return p;
}
static void afree(void *p) { _aligned_free(p); }
#else
static void *aalloc(size_t b) {
    void *p = NULL;
    if (posix_memalign(&p, 64, b) != 0) return NULL;
    memset(p, 0, b); return p;
}
static void afree(void *p) { free(p); }
#endif

static int _dcmp(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static int have_avx512(void) {
    int a, b, c, d;
    /* OSXSAVE required for AVX/AVX-512 state save */
    _cpuid_count(1, 0, &a, &b, &c, &d);
    if (!(c & (1 << 27))) return 0;
    /* XGETBV to check XCR0 bits for ZMM state saving */
    unsigned long long xcr0;
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
    xcr0 = _xgetbv(0);
#else
    unsigned int lo, hi;
    __asm__ volatile ("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    xcr0 = ((unsigned long long)hi << 32) | lo;
#endif
    /* Bits 1(SSE) 2(AVX) 5(opmask) 6(zmm hi) 7(zmm16-31) */
    if ((xcr0 & 0xE6) != 0xE6) return 0;
    _cpuid_count(7, 0, &a, &b, &c, &d);
    return ((b & (1<<16)) != 0) && ((b & (1<<17)) != 0);
}

typedef void (*t1_fn)(double*, double*, const double*, const double*, size_t, size_t);
#define R 32
#define PI 3.14159265358979323846

static void fill_twiddles(double *Wr, double *Wi, size_t me) {
    for (size_t n = 0; n < R-1; n++) for (size_t m = 0; m < me; m++) {
        double g = -2.0 * PI * (double)(n+1) * (double)m / (double)(R*me);
        Wr[n*me+m] = cos(g); Wi[n*me+m] = sin(g);
    }
}

static double measure(t1_fn fn, size_t ios, size_t me) {
    size_t an = R * (ios > me ? ios : me);
    double *rio_re = aalloc(an*8), *rio_im = aalloc(an*8);
    double *src_re = aalloc(an*8), *src_im = aalloc(an*8);
    double *Wr = aalloc((R-1)*me*8), *Wi = aalloc((R-1)*me*8);
    unsigned s = 12345;
    for (size_t i = 0; i < an; i++) {
        s = s*1103515245u + 12345u;
        src_re[i] = ((double)(s>>16) / 32768.0) - 1.0;
        s = s*1103515245u + 12345u;
        src_im[i] = ((double)(s>>16) / 32768.0) - 1.0;
    }
    fill_twiddles(Wr, Wi, me);
    size_t work = R * me;
    int reps = (int)(1000000.0 / ((double)work + 1));
    if (reps < 20) reps = 20;
    if (reps > 10000) reps = 10000;
    for (int i = 0; i < 100; i++) {
        memcpy(rio_re, src_re, an*8); memcpy(rio_im, src_im, an*8);
        fn(rio_re, rio_im, Wr, Wi, ios, me);
    }
    double s1[21], s2[21];
    for (int i = 0; i < 21; i++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(rio_re, src_re, an*8); memcpy(rio_im, src_im, an*8);
            fn(rio_re, rio_im, Wr, Wi, ios, me);
        }
        s1[i] = (now_ns() - t0) / reps;
    }
    for (int i = 0; i < 21; i++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(rio_re, src_re, an*8); memcpy(rio_im, src_im, an*8);
        }
        s2[i] = (now_ns() - t0) / reps;
    }
    qsort(s1, 21, sizeof(double), _dcmp);
    qsort(s2, 21, sizeof(double), _dcmp);
    double net = s1[10] - s2[10];
    if (net < 0) net = s1[10];
    afree(rio_re); afree(rio_im); afree(src_re); afree(src_im); afree(Wr); afree(Wi);
    return net;
}
'''


def _wrapper_c(candidate):
    """Single-codelet wrapper: #define static to force external linkage."""
    return (f'#define static\n'
            f'#include "{(STAGING / candidate.header_name()).resolve()}"\n')


# ────────────────────────────────────────────────────
# Compiler detection — GCC, Clang, Intel ICX on any OS
# ────────────────────────────────────────────────────
def _detect_toolchain():
    """Returns a dict describing the compiler setup:
      cc: the command name ('gcc', 'icx', 'clang', etc.)
      msvc_style: True if flags use /O2 style (cl, icl, icx-cl)
      is_windows, is_icx, is_gcc, is_clang: bool
      toolchain_name: human-readable description
    """
    cc = os.environ.get('CC', 'gcc')
    cc_basename = Path(cc).name.lower()
    msvc_style = cc_basename in ('cl', 'cl.exe', 'icl', 'icl.exe',
                                  'icx-cl', 'icx-cl.exe')
    is_windows = os.name == 'nt'
    is_icx = 'icx' in cc_basename
    is_gcc = 'gcc' in cc_basename
    is_clang = 'clang' in cc_basename

    if is_icx and msvc_style:
        name = 'Intel ICX (MSVC-style, icx-cl)'
    elif is_icx:
        name = 'Intel ICX (GCC-style)'
    elif msvc_style:
        name = 'Microsoft MSVC (cl.exe)'
    elif is_clang:
        name = 'LLVM Clang'
    elif is_gcc:
        name = 'GNU GCC'
    else:
        name = f'{cc} (unknown driver style)'

    return {
        'cc': cc, 'cc_basename': cc_basename,
        'msvc_style': msvc_style,
        'is_windows': is_windows, 'is_icx': is_icx,
        'is_gcc': is_gcc, 'is_clang': is_clang,
        'toolchain_name': name,
    }


def _compile_cmd(src, out, isa, tc):
    """Build the compile command line for the given toolchain.
    Produces a compile-to-object command (no linking)."""
    if tc['msvc_style']:
        # MSVC-style: /c for compile-only, /Fo for output
        flags = ['/c', '/O2', '/fp:fast', '/wd4244', '/wd4267']
        if isa == 'avx512':
            flags.append('/arch:AVX512')
        else:
            flags.append('/arch:AVX2')
        return [tc['cc']] + flags + [str(src), f'/Fo:{out}']
    else:
        # GCC-style: -c for compile-only, -o for output
        flags = ['-c', '-O3', '-mtune=native', '-mavx2', '-mfma',
                 '-Wno-overflow', '-Wno-implicit-function-declaration',
                 '-Wno-unused-function', '-Wno-unknown-argument']
        if isa == 'avx512':
            flags += ['-mavx512f', '-mavx512dq']
        return [tc['cc']] + flags + [str(src), '-o', str(out)]


def _link_cmd(obj_files, harness_obj, out_bin, tc):
    """Build the link command line for the given toolchain."""
    if tc['msvc_style']:
        # MSVC-style linking
        return [tc['cc']] + obj_files + [str(harness_obj),
                f'/Fe:{out_bin}',
                '/link', '/FORCE:MULTIPLE']
    else:
        # GCC-style linking. LLD for Windows+ICX to avoid needing link.exe
        # (works if lld-link.exe is on PATH; setvars.bat from oneAPI ensures this).
        cmd = [tc['cc']] + obj_files + [str(harness_obj),
               '-Wl,--allow-multiple-definition',
               '-o', str(out_bin)]
        if tc['is_windows'] and tc['is_icx']:
            cmd.append('-fuse-ld=lld')
        if not tc['is_windows']:
            cmd.append('-lm')
        return cmd


def _harness_compile_cmd(src, out, tc):
    """Compile just the harness (AVX2 baseline, no per-codelet flags)."""
    if tc['msvc_style']:
        flags = ['/c', '/O2', '/arch:AVX2', '/fp:fast']
        return [tc['cc']] + flags + [str(src), f'/Fo:{out}']
    else:
        flags = ['-c', '-O2', '-mavx2', '-mfma',
                 '-Wno-unknown-argument']
        return [tc['cc']] + flags + [str(src), '-o', str(out)]


def _run_with_progress(procs, label):
    """Wait for a list of subprocesses, printing progress."""
    total = len(procs)
    done = 0
    fails = []
    while procs:
        time.sleep(0.5)
        still = []
        for src, p in procs:
            rc = p.poll()
            if rc is None:
                still.append((src, p))
            else:
                done += 1
                if rc != 0:
                    err = p.stderr.read().decode('utf-8','replace')[:400]
                    fails.append((src, err))
                if done % 20 == 0 or done == total:
                    print(f"  [{label}] {done}/{total}")
        procs = still
    return fails


def phase_build():
    BUILD.mkdir(exist_ok=True)
    OBJS.mkdir(exist_ok=True)
    cands = candidates.enumerate_all()

    # Detect toolchain (gcc/icx/msvc, GCC-style vs MSVC-style flags)
    tc = _detect_toolchain()
    print(f"[build] toolchain: {tc['toolchain_name']}")
    if tc['is_windows'] and tc['is_icx']:
        print(f"[build] linker: LLD (-fuse-ld=lld, bundled with ICX)")
        print(f"[build] note: ensure Intel oneAPI env is set (setvars.bat sourced)")
    elif tc['msvc_style']:
        print(f"[build] linker: Microsoft link.exe")
        print(f"[build] note: ensure VS Developer Command Prompt or vcvarsall sourced")

    # Binary suffix: .exe on Windows
    out_bin = BUILD / ('bench.exe' if tc['is_windows'] else 'bench')
    obj_ext = '.obj' if tc['msvc_style'] else '.o'

    # Write wrapper .c files, one per codelet
    wrappers = []
    for c in cands:
        wc = OBJS / f'{c.id()}.c'
        wc.write_text(_wrapper_c(c), encoding='utf-8')
        wrappers.append((c, wc))

    # Parallel compile — batch dispatch
    print(f"[build] parallel compile of {len(wrappers)} codelet wrappers", flush=True)
    t0 = time.time()
    maxN = max(1, os.cpu_count() or 2)

    procs_all = []
    def spawn(c, wc):
        obj = OBJS / f'{c.id()}{obj_ext}'
        err_log = OBJS / f'{c.id()}.err'
        errf = open(err_log, 'wb')
        cmd = _compile_cmd(wc, obj, c.isa, tc)
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errf)
        p._errfile = errf
        p._errpath = err_log
        p._wc = wc
        return p

    active = []
    pending = list(wrappers)
    completed = 0
    while pending or active:
        while pending and len(active) < maxN:
            c, wc = pending.pop(0)
            active.append(spawn(c, wc))
        time.sleep(0.1)
        still = []
        for p in active:
            rc = p.poll()
            if rc is None:
                still.append(p)
            else:
                p._errfile.close()
                completed += 1
                procs_all.append(p)
                if completed % 20 == 0 or completed == len(wrappers):
                    print(f"  [build] {completed}/{len(wrappers)}  ({time.time()-t0:.1f}s)", flush=True)
        active = still

    fails = []
    for p in procs_all:
        if p.returncode != 0:
            err = p._errpath.read_text(errors='replace')[:400] if p._errpath.exists() else ''
            fails.append((p._wc.name, err))
    if fails:
        print(f"[build] {len(fails)} codelet compile failures:")
        for n, e in fails[:3]:
            print(f"  {n}: {e}")
        sys.exit(1)
    print(f"[build] codelets compiled in {time.time()-t0:.1f}s")

    # Write and compile the main harness
    cand_entries = []
    for c in cands:
        fns = c.function_names()
        req512 = 1 if c.isa == 'avx512' else 0
        cand_entries.append(
            f'extern void {fns["fwd"]}(double*, double*, const double*, const double*, size_t, size_t);\n'
            f'extern void {fns["bwd"]}(double*, double*, const double*, const double*, size_t, size_t);\n'
        )
    extern_block = ''.join(cand_entries)

    table_rows = []
    for c in cands:
        fns = c.function_names()
        req512 = 1 if c.isa == 'avx512' else 0
        table_rows.append(
            f'    {{"{c.id()}", "{c.isa}", {fns["fwd"]}, {fns["bwd"]}, {req512}}},\n'
        )
    table = ''.join(table_rows)

    pts = candidates.sweep_points()
    sweep = ''.join(f'    {{{ios}, {me}}},\n' for ios, me in pts)

    harness = HARNESS_PREAMBLE + extern_block + f'''
typedef struct {{ const char *id; const char *isa; t1_fn fwd, bwd; int req512; }} cand_t;
static cand_t CANDS[] = {{
{table}    {{NULL,NULL,NULL,NULL,0}}
}};
typedef struct {{ size_t ios, me; }} sp_t;
static sp_t SW[] = {{
{sweep}    {{0,0}}
}};

int main(int argc, char **argv) {{
    int skip512 = !have_avx512();
    int n_cand = sizeof(CANDS)/sizeof(CANDS[0]) - 1;
    int n_sp = sizeof(SW)/sizeof(SW[0]) - 1;
    const char *outpath = argc > 1 ? argv[1] : "measurements.jsonl";

    /* Read existing measurements (resume support) */
    char done_set[{len(cands)}][2] = {{{{0}}}};  /* [cand_idx][dir] */
    /* Actually use a set by parsing existing jsonl */
    FILE *rd = fopen(outpath, "r");
    if (rd) {{
        char line[1024];
        while (fgets(line, sizeof(line), rd)) {{
            /* parse: {{"id":"...", "ios":X, "me":Y, "dir":"fwd", ...}} */
            /* simple: just skip — we'll count lines and assume in-order */
            /* Better: we append so re-runs just keep going */
        }}
        fclose(rd);
    }}

    /* Count already-done measurements to skip */
    int already = 0;
    FILE *cnt = fopen(outpath, "r");
    if (cnt) {{
        int ch;
        while ((ch = fgetc(cnt)) != EOF) if (ch == '\\n') already++;
        fclose(cnt);
    }}
    fprintf(stderr, "harness: %d candidates, %d sweeps, %d dirs; skipping %d done\\n",
            n_cand, n_sp, 2, already);

    FILE *out = fopen(outpath, "a");
    if (!out) {{ perror("fopen"); return 1; }}

    int total_emitted = 0;
    for (int ci = 0; ci < n_cand; ci++) {{
        cand_t *c = &CANDS[ci];
        if (c->req512 && skip512) continue;
        fprintf(stderr, "[%d/%d] %s\\n", ci+1, n_cand, c->id);
        for (int sp = 0; sp < n_sp; sp++) {{
            size_t ios = SW[sp].ios, me = SW[sp].me;
            for (int d = 0; d < 2; d++) {{
                if (total_emitted < already) {{ total_emitted++; continue; }}
                t1_fn fn = (d == 0) ? c->fwd : c->bwd;
                double ns = measure(fn, ios, me);
                fprintf(out, "{{\\"id\\":\\"%s\\",\\"ios\\":%zu,\\"me\\":%zu,\\"dir\\":\\"%s\\",\\"ns\\":%.2f}}\\n",
                        c->id, ios, me, d == 0 ? "fwd" : "bwd", ns);
                fflush(out);
                total_emitted++;
            }}
        }}
    }}
    fclose(out);
    return 0;
}}
'''
    hc = BUILD / 'harness.c'
    hc.write_text(harness, encoding='utf-8')

    # Compile harness (AVX2 baseline — no AVX-512 in glue code)
    hobj = OBJS / ('harness' + obj_ext)
    hcmd = _harness_compile_cmd(hc, hobj, tc)
    r = subprocess.run(hcmd, capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print(f"[build] harness compile failed:\n{r.stderr[:2000]}")
        sys.exit(1)
    print(f"[build] harness compiled")

    # Link everything. --allow-multiple-definition / /FORCE:MULTIPLE tolerates
    # duplicate twiddle-table constants (W32_* symbols) that appear in every
    # codelet's .o (they were 'static const' in the headers; '#define static'
    # trick promoted them to external linkage; all copies are bit-identical).
    obj_files = [str(OBJS / f'{c.id()}{obj_ext}') for c in cands]
    lcmd = _link_cmd(obj_files, hobj, out_bin, tc)
    r = subprocess.run(lcmd, capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print(f"[build] link failed:\n{r.stderr[:2000]}")
        sys.exit(1)
    print(f"[build] linked → {out_bin}, total build time {time.time()-t0:.1f}s")


def phase_run():
    bench = BUILD / ('bench.exe' if os.name == 'nt' else 'bench')
    if not bench.exists():
        print(f"[run] bench binary missing at {bench}; run --phase build first")
        sys.exit(1)
    print(f"[run] executing {bench}")
    t0 = time.time()
    log = BUILD / 'run.log'
    with open(log, 'w', encoding='utf-8') as lf:
        p = subprocess.Popen([str(bench), str(MEAS)],
                             stdout=lf, stderr=lf)
        while p.poll() is None:
            time.sleep(2)
            if MEAS.exists():
                with open(MEAS, encoding='utf-8') as f:
                    n = sum(1 for _ in f)
                print(f"  progress: {n} measurements  ({time.time()-t0:.0f}s)", flush=True)
    if p.returncode != 0:
        print(f"[run] bench exited with code {p.returncode}")
        print(f"       rerun --phase run to resume (measurements.jsonl is checkpointed)")
        sys.exit(1)
    print(f"[run] done in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('generate','build','run','all'),
                    default='all')
    args = ap.parse_args()

    # Startup banner — show environment so users can diagnose issues
    # before long-running phases start.
    import platform
    cc = os.environ.get('CC', 'gcc')
    print("-" * 64)
    print(" VectorFFT codelet bench (R=32)")
    print("-" * 64)
    print(f" Platform:  {platform.system()} {platform.machine()}")
    print(f" Python:    {platform.python_version()} ({sys.executable})")
    print(f" Compiler:  {cc} (set CC env var to change)")
    print(f" Generator: {GEN}")
    print("-" * 64)
    if os.name == 'nt' and 'icx' in Path(cc).name.lower():
        print(" Using Intel ICX on Windows: LLD linker via -fuse-ld=lld.")
        print(" If linking fails, ensure you launched cmd from:")
        print("   - Intel oneAPI command prompt, OR")
        print("   - sourced setvars.bat in this shell.")
        print("-" * 64)
    elif os.name == 'nt' and Path(cc).name.lower() in ('cl', 'icx-cl', 'icl'):
        print(" Using MSVC-style driver on Windows.")
        print(" If linking fails, ensure link.exe is on PATH:")
        print("   - launch from 'Developer Command Prompt for VS', OR")
        print("   - sourced vcvarsall.bat in this shell.")
        print("-" * 64)
    print()

    if args.phase in ('generate', 'all'): phase_generate()
    if args.phase in ('build', 'all'): phase_build()
    if args.phase in ('run', 'all'): phase_run()