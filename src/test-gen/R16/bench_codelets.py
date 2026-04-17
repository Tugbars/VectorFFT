"""
bench_codelets.py — generate, build, run the R=16 codelet bench.

Three phases:
  1. GENERATE — for each candidate, invoke the generator and stage its .h
  2. BUILD    — emit one C harness including all staged headers,
                compile it (one executable, one compile, runtime dispatch)
  3. RUN      — execute the harness, capture JSON output to measurements.json

Each phase is separately callable. Run with --help for options.

Output artifact: measurements.json, consumed by select_codelets.py.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Import candidates (same directory)
sys.path.insert(0, str(Path(__file__).parent))
import candidates


# ─────────────────────────────────────────────────────────────────────
# Defaults / paths
# ─────────────────────────────────────────────────────────────────────

BENCH_ROOT = Path(__file__).parent
STAGING = BENCH_ROOT / 'staging'
BUILD = BENCH_ROOT / 'build'
# Default: look for the generator next to this script.
# Override via --generator /path/to/gen_radix16_buffered.py
GENERATOR_DEFAULT = BENCH_ROOT / 'gen_radix16_buffered.py'


# ─────────────────────────────────────────────────────────────────────
# Phase 1: GENERATE
# ─────────────────────────────────────────────────────────────────────

def phase_generate(generator: Path, verbose: bool = True) -> None:
    """Run the generator for each candidate, write to staging/."""
    if not generator.exists():
        raise FileNotFoundError(
            f"R=16 generator not found: {generator}\n"
            f"Either place gen_radix16_buffered.py in the bench directory, "
            f"or pass --generator /path/to/gen_radix16_buffered.py"
        )
    STAGING.mkdir(exist_ok=True)

    cands = candidates.enumerate_all()
    if verbose:
        print(f"[generate] {len(cands)} candidates → {STAGING}/")

    t0 = time.time()
    for i, c in enumerate(cands):
        out_path = STAGING / c.header_name()
        # Use sys.executable so we invoke the same Python interpreter
        # that's running this script — works on any OS regardless of
        # whether 'python' or 'python3' is the conventional name.
        cmd = [sys.executable, str(generator)] + c.cli_args()
        try:
            # Capture as bytes, not text — avoids the Windows cp1252/UTF-8
            # encoding trap where the generator may emit em-dashes, box-drawing
            # chars, etc. in whatever encoding Python defaulted to. The bytes
            # written to the .h file will be whatever the generator produced;
            # the C compiler handles it regardless (it's in comments anyway).
            result = subprocess.run(cmd, capture_output=True,
                                    timeout=60, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  FAIL {c.id()}")
            err = e.stderr.decode('utf-8', errors='replace') if e.stderr else ''
            print(f"    stderr: {err[:300]}")
            raise
        # Write raw bytes to preserve whatever encoding the generator used
        out_path.write_bytes(result.stdout)

        if verbose and (i % 5 == 4 or i == len(cands) - 1):
            print(f"  [{i+1}/{len(cands)}] {c.id()}")

    if verbose:
        print(f"[generate] done in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────
# Phase 2: BUILD — emit a C harness including all staged codelets
# ─────────────────────────────────────────────────────────────────────

HARNESS_HEAD = r'''
/* Auto-generated bench harness. Do not edit. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#ifdef _WIN32
  #include <windows.h>
  #include <malloc.h>
#endif

#define static /* flatten static decls so everything links */
'''

HARNESS_INCLUDES_TEMPLATE = '#include "{header_path}"\n'

HARNESS_BODY = r'''
/* ─── timing: portable high-resolution monotonic clock in ns ─── */
#ifdef _WIN32
static double now_ns(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    /* ns = ticks * 1e9 / freq */
    return (double)t.QuadPart * 1e9 / (double)freq.QuadPart;
}
#else
static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/* ─── aligned alloc / free ─── */
#ifdef _WIN32
static void *aalloc(size_t bytes) {
    void *p = _aligned_malloc(bytes, 64);
    if (p) memset(p, 0, bytes);
    return p;
}
static void afree(void *p) { _aligned_free(p); }
#else
static void *aalloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    memset(p, 0, bytes);
    return p;
}
static void afree(void *p) { free(p); }
#endif

/* ─── runtime CPU feature check for AVX-512 ───
 * Uses direct CPUID instruction rather than __builtin_cpu_supports()
 * (which on some Windows compilers requires __cpu_model runtime symbol
 * that LLD can't resolve).
 *
 * AVX-512F  = CPUID.7.0:EBX[16]
 * AVX-512DQ = CPUID.7.0:EBX[17]
 * OSXSAVE is checked too: CPUID.1:ECX[27]
 * XGETBV XCR0 bits 5,6,7 must be set for the OS to save ZMM state.
 */
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
  #include <intrin.h>
  static void _cpuid(int out[4], int leaf, int subleaf) {
      __cpuidex(out, leaf, subleaf);
  }
#else
  #include <cpuid.h>
  static void _cpuid(int out[4], int leaf, int subleaf) {
      __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
  }
#endif

static int have_avx512(void) {
    int r[4];
    /* Check OSXSAVE first (bit 27 of ECX from CPUID.1) */
    _cpuid(r, 1, 0);
    if (!(r[2] & (1 << 27))) return 0;  /* OS doesn't support XSAVE */
    /* XGETBV to check XCR0 bits for ZMM state saving */
    unsigned long long xcr0;
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
    xcr0 = _xgetbv(0);
#else
    unsigned int lo, hi;
    __asm__ volatile ("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    xcr0 = ((unsigned long long)hi << 32) | lo;
#endif
    /* Bits 1 (SSE), 2 (AVX), 5 (opmask), 6 (zmm hi), 7 (zmm 16-31) */
    const unsigned long long want = (1ULL << 1) | (1ULL << 2)
                                  | (1ULL << 5) | (1ULL << 6) | (1ULL << 7);
    if ((xcr0 & want) != want) return 0;
    /* Now check actual CPU support via CPUID.7.0:EBX */
    _cpuid(r, 7, 0);
    const int has_avx512f  = (r[1] & (1 << 16)) != 0;
    const int has_avx512dq = (r[1] & (1 << 17)) != 0;
    return has_avx512f && has_avx512dq;
}

/* ─── dispatch tables ─── */
typedef void (*t1_fn)(double *rio_re, double *rio_im,
                      const double *W_re, const double *W_im,
                      size_t ios, size_t me);

typedef struct {
    const char *id;
    const char *isa;
    t1_fn fwd;
    t1_fn bwd;
    int requires_avx512;
} candidate_entry_t;

/* Populated by emit_dispatch_table() below */
static const candidate_entry_t CANDIDATES[] = {
CANDIDATE_ENTRIES
    {NULL, NULL, NULL, NULL, 0}
};

static int N_CANDIDATES =
    sizeof(CANDIDATES)/sizeof(CANDIDATES[0]) - 1;

/* ─── sweep points ─── */
typedef struct { size_t ios, me; } sweep_point_t;
static const sweep_point_t SWEEP[] = {
SWEEP_POINTS
    {0, 0}
};
static int N_SWEEP = sizeof(SWEEP)/sizeof(SWEEP[0]) - 1;

/* ─── twiddle table setup ─── */
/* For R=16, W_re/im has 15 rows × me columns, in [(j-1)*me + m] layout */
#define R 16
#define PI 3.14159265358979323846
static void fill_twiddles(double *Wr, double *Wi, size_t me) {
    for (size_t n = 0; n < R - 1; n++) {
        for (size_t m = 0; m < me; m++) {
            double g = -2.0 * PI * (double)(n+1) * (double)m
                       / (double)(R * me);
            Wr[n*me + m] = cos(g);
            Wi[n*me + m] = sin(g);
        }
    }
}

/* ─── comparator for qsort (top-level; ICX/clang reject nested fns) ─── */
static int _dcmp(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/* ─── measurement of one (candidate, ios, me) ─── */
static double measure(t1_fn fn, size_t ios, size_t me, int bwd) {
    size_t alloc_N = R * (ios > me ? ios : me);
    double *rio_re = aalloc(alloc_N * sizeof(double));
    double *rio_im = aalloc(alloc_N * sizeof(double));
    double *src_re = aalloc(alloc_N * sizeof(double));
    double *src_im = aalloc(alloc_N * sizeof(double));
    double *Wr = aalloc((R-1) * me * sizeof(double));
    double *Wi = aalloc((R-1) * me * sizeof(double));
    (void)bwd;

    /* Check all allocations succeeded */
    if (!rio_re || !rio_im || !src_re || !src_im || !Wr || !Wi) {
        fprintf(stderr, "    aalloc FAILED at ios=%zu me=%zu (alloc_N=%zu bytes_each=%zu)\n",
                ios, me, alloc_N, alloc_N * sizeof(double));
        fflush(stderr);
        if (rio_re) afree(rio_re);
        if (rio_im) afree(rio_im);
        if (src_re) afree(src_re);
        if (src_im) afree(src_im);
        if (Wr)     afree(Wr);
        if (Wi)     afree(Wi);
        return -1.0;
    }

    /* Random data — don't want zero inputs (FMA throughput differs) */
    unsigned s = 12345;
    for (size_t i = 0; i < alloc_N; i++) {
        s = s * 1103515245u + 12345u;
        src_re[i] = ((double)(s >> 16) / 32768.0) - 1.0;
        s = s * 1103515245u + 12345u;
        src_im[i] = ((double)(s >> 16) / 32768.0) - 1.0;
    }
    fill_twiddles(Wr, Wi, me);

    /* Reps scale so each timed region is ~1-5ms */
    size_t work = R * me;
    int reps = (int)(1000000.0 / ((double)work + 1));
    if (reps < 20) reps = 20;
    if (reps > 10000) reps = 10000;

    /* Warmup — get caches/TLB warm, branch predictor trained */
    for (int i = 0; i < 100; i++) {
        memcpy(rio_re, src_re, alloc_N * sizeof(double));
        memcpy(rio_im, src_im, alloc_N * sizeof(double));
        fn(rio_re, rio_im, Wr, Wi, ios, me);
    }

    /* Measure — median of 21 samples */
    double samples[21];
    for (int i = 0; i < 21; i++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(rio_re, src_re, alloc_N * sizeof(double));
            memcpy(rio_im, src_im, alloc_N * sizeof(double));
            fn(rio_re, rio_im, Wr, Wi, ios, me);
        }
        double ns_per_rep = (now_ns() - t0) / reps;
        samples[i] = ns_per_rep;
    }
    /* Also measure memcpy-only to subtract */
    double mc_samples[21];
    for (int i = 0; i < 21; i++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(rio_re, src_re, alloc_N * sizeof(double));
            memcpy(rio_im, src_im, alloc_N * sizeof(double));
        }
        mc_samples[i] = (now_ns() - t0) / reps;
    }

    /* Median */
    qsort(samples, 21, sizeof(double), _dcmp);
    qsort(mc_samples, 21, sizeof(double), _dcmp);
    double net = samples[10] - mc_samples[10];
    if (net < 0) net = samples[10];  /* memcpy subtraction guard */

    afree(rio_re); afree(rio_im); afree(src_re); afree(src_im);
    afree(Wr); afree(Wi);
    return net;
}

/* ─── main: sweep and emit JSON ─── */
int main(int argc, char **argv) {
    int skip_avx512 = !have_avx512();

    fprintf(stderr, "harness: %d candidates, %d sweep points, avx512=%s\n",
            N_CANDIDATES, N_SWEEP, skip_avx512 ? "MISSING (skipping)" : "OK");

    printf("{\n");
    printf("  \"avx512_available\": %s,\n", skip_avx512 ? "false" : "true");
    printf("  \"measurements\": [\n");

    int first = 1;
    for (int ci = 0; ci < N_CANDIDATES; ci++) {
        const candidate_entry_t *c = &CANDIDATES[ci];
        if (c->requires_avx512 && skip_avx512) {
            fprintf(stderr, "  [skip avx512] %s\n", c->id);
            continue;
        }
        fprintf(stderr, "  [%d/%d] %s\n", ci+1, N_CANDIDATES, c->id);
        fflush(stderr);

        for (int sp = 0; sp < N_SWEEP; sp++) {
            size_t ios = SWEEP[sp].ios;
            size_t me  = SWEEP[sp].me;

            fprintf(stderr, "    ios=%zu me=%zu fwd...", ios, me);
            fflush(stderr);
            double fwd_ns = measure(c->fwd, ios, me, 0);
            fprintf(stderr, " bwd...");
            fflush(stderr);
            double bwd_ns = measure(c->bwd, ios, me, 1);
            fprintf(stderr, " done (fwd=%.0f bwd=%.0f ns)\n", fwd_ns, bwd_ns);
            fflush(stderr);

            if (!first) printf(",\n");
            first = 0;
            printf("    {\"id\": \"%s\", \"ios\": %zu, \"me\": %zu, "
                   "\"fwd_ns\": %.2f, \"bwd_ns\": %.2f}",
                   c->id, ios, me, fwd_ns, bwd_ns);
            fflush(stdout);
        }
    }
    printf("\n  ]\n}\n");
    return 0;
}
'''


def phase_build(verbose: bool = True) -> Path:
    """Emit the harness C file and compile it to build/bench.exe."""
    BUILD.mkdir(exist_ok=True)
    harness_c = BUILD / 'harness.c'
    harness_bin = BUILD / 'bench'

    cands = candidates.enumerate_all()

    # 1. Build includes section
    includes = ''.join(
        HARNESS_INCLUDES_TEMPLATE.format(
            header_path=str((STAGING / c.header_name()).resolve()))
        for c in cands
    )

    # 2. Build dispatch table entries
    entries = []
    for c in cands:
        fns = c.function_names()
        requires_avx512 = 1 if c.isa == 'avx512' else 0
        entries.append(
            f'    {{"{c.id()}", "{c.isa}", '
            f'{fns["fwd"]}, {fns["bwd"]}, {requires_avx512}}},\n'
        )
    entries_c = ''.join(entries)

    # 3. Build sweep table entries
    pts = candidates.sweep_points()
    sweep_c = ''.join(f'    {{{ios}, {me}}},\n' for ios, me in pts)

    # 4. Assemble full source
    src = HARNESS_HEAD + includes + HARNESS_BODY
    src = src.replace('CANDIDATE_ENTRIES', entries_c)
    src = src.replace('SWEEP_POINTS', sweep_c)
    harness_c.write_text(src, encoding='utf-8')

    if verbose:
        print(f"[build] harness source: {harness_c} "
              f"({harness_c.stat().st_size} bytes)")

    # 5. Compile. Use aggressive opt + native arch. Per-function target
    #    attributes on AVX-512 codelets let the harness compile even on
    #    non-AVX-512 hosts; CPUID gates runtime dispatch.
    #
    # We support three driver styles:
    #   GCC/Clang on Linux (gcc, clang, icx):      -O3 ... -lm
    #   GCC/Clang/ICX on Windows (mingw-gcc, icx): -O3 ... (no -lm)
    #   MSVC-style on Windows (cl, icl, icx-cl):   /O2 /arch:AVX512
    cc = os.environ.get('CC', 'gcc')
    cc_basename = Path(cc).name.lower()
    msvc_style = cc_basename in ('cl', 'cl.exe', 'icl', 'icl.exe',
                                  'icx-cl', 'icx-cl.exe')
    is_windows = os.name == 'nt'
    is_icx = 'icx' in cc_basename
    is_gcc = 'gcc' in cc_basename

    # Identify toolchain for user-facing messages
    if is_icx and msvc_style:
        toolchain_name = 'Intel ICX (MSVC-style, icx-cl)'
    elif is_icx:
        toolchain_name = 'Intel ICX (GCC-style)'
    elif msvc_style:
        toolchain_name = 'Microsoft MSVC (cl.exe)'
    elif 'clang' in cc_basename:
        toolchain_name = 'LLVM Clang'
    elif is_gcc:
        toolchain_name = 'GNU GCC'
    else:
        toolchain_name = f'{cc} (unknown driver style)'

    # Binary suffix: .exe on Windows
    if is_windows and not str(harness_bin).endswith('.exe'):
        harness_bin = Path(str(harness_bin) + '.exe')

    if msvc_style:
        # MSVC/ICX-cl driver: /O2 + /arch:AVX512 enables all AVX-512 subsets.
        cflags = ['/O2', '/arch:AVX512', '/fp:fast',
                  '/wd4244', '/wd4267']  # silence int-narrowing warnings
        cmd = [cc] + cflags + [str(harness_c),
                               f'/Fe:{harness_bin}',
                               f'/Fo:{BUILD}\\']
    else:
        # GCC-style driver (gcc, clang, icx with GCC-like flags).
        # Baseline ISA: -mavx2 + -mfma only. We do NOT pass -march=native
        # because on some hosts that would enable -mavx512f globally,
        # which means the compiler could emit AVX-512 instructions in
        # glue code (memcpy inlining, outer loops) that would SIGILL on
        # AVX2-only CPUs. AVX-512 codelets compile correctly via
        # per-function __attribute__((target("avx512f,avx512dq"))).
        #
        # But we DO pass -mtune=native so the compiler schedules the
        # generated code for the host microarchitecture (cache latencies,
        # execution port layout, front-end width). This is the best of
        # both worlds: correct baseline + host-tuned codegen.
        cflags = ['-O3', '-mtune=native', '-mavx2', '-mfma',
                  '-Wno-overflow', '-Wno-implicit-function-declaration',
                  '-Wno-unknown-argument']
        # On Windows with ICX, use the bundled LLD linker to avoid the need
        # for Visual Studio's link.exe.
        if is_windows and is_icx:
            cflags.append('-fuse-ld=lld')
        cmd = [cc] + cflags + [str(harness_c), '-o', str(harness_bin)]
        # -lm only needed on Linux; on Windows the C runtime has libm built in
        if not is_windows:
            cmd.append('-lm')

    if verbose:
        print(f"[build] toolchain: {toolchain_name}")
        # Compiler-specific notes so users know what to expect
        if is_icx and is_windows:
            print(f"[build] linker: LLD (bundled with ICX, -fuse-ld=lld)")
            print(f"[build] note: if linking fails, ensure lld-link.exe is on PATH,")
            print(f"[build]       or invoke via Intel oneAPI command prompt.")
        elif msvc_style:
            print(f"[build] linker: Microsoft link.exe (requires VS Build Tools on PATH)")
            print(f"[build] note: run this from 'Developer Command Prompt for VS' if link fails.")
        elif is_gcc and is_windows:
            print(f"[build] linker: MinGW ld (GCC's default)")
        elif is_windows:
            print(f"[build] linker: {cc}'s default")
        else:
            print(f"[build] linker: system ld (Linux/Unix)")
        print(f"[build] command: {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True,
                            text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        print(f"[build] compile FAILED")
        print(f"  stderr: {result.stderr[:2000]}")
        raise RuntimeError("compile failed")
    if verbose:
        print(f"[build] compiled in {time.time()-t0:.1f}s → {harness_bin}")
    return harness_bin


# ─────────────────────────────────────────────────────────────────────
# Phase 3: RUN
# ─────────────────────────────────────────────────────────────────────

def phase_run(harness_bin: Path, verbose: bool = True) -> dict:
    """Execute the harness and parse its JSON output.

    Streams stderr live (harness prints progress there) so the user sees
    progress as the bench runs. Captures stdout (the JSON) separately.
    """
    if verbose:
        print(f"[run] executing {harness_bin}")
        print(f"[run] streaming progress from harness (stderr) below:")
    t0 = time.time()

    # Launch harness with stdout captured (JSON), stderr inheriting current
    # process's stderr (so progress lines print live to the user's terminal).
    proc = subprocess.Popen(
        [str(harness_bin)],
        stdout=subprocess.PIPE,
        stderr=None,  # inherit — progress prints live
        text=False,   # read stdout as bytes to avoid encoding issues
    )
    try:
        stdout_bytes, _ = proc.communicate(timeout=3600)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise

    if proc.returncode != 0:
        raise RuntimeError(f"harness exited with code {proc.returncode} "
                           f"(see stderr above for progress before crash)")

    # Decode stdout as UTF-8 (harness emits JSON, ASCII-safe)
    stdout_text = stdout_bytes.decode('utf-8', errors='replace')
    try:
        data = json.loads(stdout_text)
    except json.JSONDecodeError as e:
        print(f"[run] harness stdout was not valid JSON:")
        print(stdout_text[:2000])
        raise RuntimeError(f"JSON parse failed: {e}")

    if verbose:
        print(f"[run] done in {time.time()-t0:.1f}s — "
              f"{len(data['measurements'])} measurements")
    return data


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generator', default=str(GENERATOR_DEFAULT),
                    help='Path to gen_radix16_buffered.py')
    ap.add_argument('--output', default='measurements.json',
                    help='Where to write the final measurements')
    ap.add_argument('--skip-generate', action='store_true',
                    help='Assume staging/ is already populated')
    ap.add_argument('--skip-build', action='store_true',
                    help='Assume build/bench already exists')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    verbose = not args.quiet

    # Startup banner — show what environment we're running in so users
    # can diagnose issues before long-running phases start.
    if verbose:
        import platform
        cc = os.environ.get('CC', 'gcc')
        print("─" * 64)
        print(f" VectorFFT codelet bench (R=16)")
        print("─" * 64)
        print(f" Platform:  {platform.system()} {platform.machine()}")
        print(f" Python:    {platform.python_version()} ({sys.executable})")
        print(f" Compiler:  {cc} (set CC env var to change)")
        print(f" Generator: {args.generator}")
        print("─" * 64)
        # Specific env-setup hints per platform
        if os.name == 'nt' and 'icx' in Path(cc).name.lower():
            print(" Using Intel ICX on Windows: linker will be LLD (-fuse-ld=lld)")
            print(" If linking fails, ensure you launched this cmd from:")
            print("   - Intel oneAPI command prompt, OR")
            print("   - sourced setvars.bat in this shell, OR")
            print("   - lld-link.exe is on PATH.")
            print("─" * 64)
        elif os.name == 'nt' and Path(cc).name.lower() in ('cl', 'icx-cl', 'icl'):
            print(" Using MSVC-style driver on Windows.")
            print(" If linking fails, ensure link.exe is on PATH:")
            print("   - launch from 'Developer Command Prompt for VS', OR")
            print("   - sourced vcvarsall.bat in this shell.")
            print("─" * 64)
        elif os.name == 'nt':
            print(" Using GCC-style driver on Windows (MinGW, or other).")
            print(" If linking fails, ensure the linker is on PATH.")
            print("─" * 64)
        print()

    if not args.skip_generate:
        phase_generate(Path(args.generator), verbose=verbose)

    # Default path when --skip-build is used: find whichever bench/bench.exe exists
    harness_bin = BUILD / ('bench.exe' if os.name == 'nt' else 'bench')
    if not args.skip_build:
        harness_bin = phase_build(verbose=verbose)

    data = phase_run(harness_bin, verbose=verbose)

    out = Path(args.output)
    out.write_text(json.dumps(data, indent=2), encoding='utf-8')
    if verbose:
        print(f"[done] measurements → {out}")


if __name__ == '__main__':
    main()