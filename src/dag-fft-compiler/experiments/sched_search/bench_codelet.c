/* bench_codelet.c — representative single-codelet timer (Phase 0.1).
 *
 * Schedule-search experiment harness. Times ONE codelet (loaded from a DLL by
 * symbol) over a production-shaped K-batch, on the native-Windows production
 * target, so the measurement reflects what ships — not a naked microbench.
 *
 * Design (see docs/roadmap/schedule_search_plan.md):
 *   - dlopen-by-symbol (LoadLibrary/GetProcAddress) so the SAME binary benches
 *     any candidate codelet .dll — this is also the Phase 1.3 per-candidate
 *     turnaround path (no relink per variant).
 *   - leaf isolation: N = R, the codelet processes K parallel R-point DFTs;
 *     buffer footprint R*K per channel spans L1/L2/L3 as K is swept, which is
 *     how we expose whether a schedule's spill delta matters under memory
 *     pressure (the executor/memory-bound regime).
 *   - native-Windows timing: QueryPerformanceCounter (ns) + __rdtsc (cycles),
 *     core-0 pin, high priority, best-of-N batches with auto-calibrated iters.
 *   - CSV out (one row) so the Phase 0.3 correlation script can consume it.
 *
 * Codelet ABI (in-place split-complex, mirrors score_and_time_plans.c):
 *   void fn(double* re, double* im,
 *           const double* tw_re, const double* tw_im,
 *           size_t ios, size_t me);
 *   me  = total butterflies in the stage (the K-batch loop is INSIDE the codelet)
 *   ios = leg stride in doubles
 *   For a standalone leaf (N=R): me = K, ios = K.
 *
 * Usage:
 *   bench_codelet <dll> <symbol> <R> <K> [batches] [target_ns]
 * Example:
 *   bench_codelet exp.dll expsched_r32_v0007_n1_fwd_avx2 32 256
 *
 * Build (native Windows / mingw gcc, fixed flags):
 *   gcc -O3 -march=native -funroll-loops bench_codelet.c -o bench_codelet.exe
 * (the codelet DLL is compiled separately with the SAME fixed flags so the
 *  only thing that varies between candidates is instruction order.)
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef _WIN32
  #include <windows.h>
  #include <intrin.h>
#else
  #include <dlfcn.h>
  #include <time.h>
  #include <x86intrin.h>
  #include <pthread.h>
#endif

typedef void (*codelet_fn)(double *, double *,
                           const double *, const double *,
                           size_t, size_t);

/* ---- platform shims (timing / affinity / dlopen) ---- */
#ifdef _WIN32
static double qpc_freq(void) {
    LARGE_INTEGER f; QueryPerformanceFrequency(&f); return (double)f.QuadPart;
}
static int64_t qpc_now(void) {
    LARGE_INTEGER c; QueryPerformanceCounter(&c); return (int64_t)c.QuadPart;
}
static void pin_and_prioritize(int core) {
    /* Match bench_1d_vs_mkl.c: pin single-thread runs to a quiet P-core
     * (default P-core 2). Core 0 is OS-contended and produces bimodal timings
     * even at a locked 5.7 GHz. */
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1u << core);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
}
static void pace(int ms) {
    if (ms > 0) Sleep((DWORD)ms);
}
static void *dl_open(const char *p) { return (void *)LoadLibraryA(p); }
static void *dl_sym(void *h, const char *s) {
    return (void *)GetProcAddress((HMODULE)h, s);
}
static void *xalloc(size_t b) { return _aligned_malloc(b, 64); }
static void  xfree(void *p)   { _aligned_free(p); }
#else
static double qpc_freq(void) { return 1.0e9; }
static int64_t qpc_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}
static void pin_and_prioritize(int core) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}
static void pace(int ms) {
    if (ms > 0) { struct timespec ts = {ms/1000, (long)(ms%1000)*1000000L};
                  nanosleep(&ts, NULL); }
}
static void *dl_open(const char *p) { return dlopen(p, RTLD_NOW); }
static void *dl_sym(void *h, const char *s) { return dlsym(h, s); }
static void *xalloc(size_t b) { void *p = NULL;
    if (posix_memalign(&p, 64, b)) p = NULL; return p; }
static void  xfree(void *p) { free(p); }
#endif

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
            "usage: %s <dll> <symbol> <R> <K> [batches=21] [target_ns=5e7]\n",
            argv[0]);
        return 1;
    }
    const char *dll = argv[1];
    const char *sym = argv[2];
    int R = atoi(argv[3]);
    size_t K = (size_t)strtoull(argv[4], NULL, 10);
    int batches = (argc > 5) ? atoi(argv[5]) : 21;
    double target_ns = (argc > 6) ? atof(argv[6]) : 5.0e7; /* 50 ms */
    int core = (argc > 7) ? atoi(argv[7]) : 2;   /* quiet P-core (bench_1d_vs_mkl convention) */
    int pace_ms = (argc > 8) ? atoi(argv[8]) : 5; /* inter-batch idle, lets the core settle */

    void *h = dl_open(dll);
    if (!h) { fprintf(stderr, "load failed: %s\n", dll); return 2; }
    codelet_fn fn = (codelet_fn)dl_sym(h, sym);
    if (!fn) { fprintf(stderr, "symbol not found: %s\n", sym); return 3; }

    /* Leaf isolation: N=R, K parallel R-point DFTs. me=K, ios=K. */
    size_t me = K;
    size_t ios = K;
    size_t nelem = (size_t)R * K + 64;
    double *re    = (double *)xalloc(nelem * sizeof(double));
    double *im    = (double *)xalloc(nelem * sizeof(double));
    double *tw_re = (double *)xalloc(nelem * sizeof(double));
    double *tw_im = (double *)xalloc(nelem * sizeof(double));
    if (!re || !im || !tw_re || !tw_im) { fprintf(stderr, "alloc failed\n"); return 4; }
    for (size_t i = 0; i < nelem; i++) {
        re[i] = 1.0 + 1e-3 * (double)(i & 1023);
        im[i] = 0.5 - 1e-3 * (double)(i & 511);
        tw_re[i] = 0.70710678; tw_im[i] = -0.70710678;
    }

    pin_and_prioritize(core);
    double freq = qpc_freq();

    /* Warmup: drive the pinned core to steady state (longer than a few calls). */
    for (int i = 0; i < 4096; i++) fn(re, im, tw_re, tw_im, ios, me);

    /* Auto-calibrate iters so one batch ~= target_ns. */
    size_t iters = 1;
    for (;;) {
        int64_t t0 = qpc_now();
        for (size_t i = 0; i < iters; i++) fn(re, im, tw_re, tw_im, ios, me);
        int64_t t1 = qpc_now();
        double ns = (double)(t1 - t0) / freq * 1e9;
        if (ns >= target_ns || iters >= (size_t)1 << 30) break;
        iters = (ns < target_ns / 8.0) ? iters * 8 : iters * 2;
    }

    /* best-of-N batches; record ns/call and cycles/call per batch. */
    double *ns_per = (double *)malloc((size_t)batches * sizeof(double));
    double *cyc_per = (double *)malloc((size_t)batches * sizeof(double));
    for (int b = 0; b < batches; b++) {
        pace(pace_ms);  /* inter-batch idle (bench_1d_vs_mkl pacing) */
        uint64_t c0 = __rdtsc();
        int64_t  t0 = qpc_now();
        for (size_t i = 0; i < iters; i++) fn(re, im, tw_re, tw_im, ios, me);
        int64_t  t1 = qpc_now();
        uint64_t c1 = __rdtsc();
        ns_per[b]  = (double)(t1 - t0) / freq * 1e9 / (double)iters;
        cyc_per[b] = (double)(c1 - c0) / (double)iters;
    }

    qsort(ns_per, batches, sizeof(double), cmp_double);
    qsort(cyc_per, batches, sizeof(double), cmp_double);
    double ns_min = ns_per[0];
    double ns_med = ns_per[batches / 2];
    double mean = 0.0; for (int b = 0; b < batches; b++) mean += ns_per[b];
    mean /= batches;
    double var = 0.0; for (int b = 0; b < batches; b++) {
        double d = ns_per[b] - mean; var += d * d;
    }
    double sd = (batches > 1) ? sqrt(var / (batches - 1)) : 0.0;
    double cyc_min = cyc_per[0];

    /* CSV: symbol,R,K,iters,batches,ns_min,ns_med,ns_sd,ns_cov_pct,cyc_min */
    printf("%s,%d,%zu,%zu,%d,%.3f,%.3f,%.4f,%.3f,%.1f\n",
           sym, R, K, iters, batches, ns_min, ns_med, sd,
           ns_med > 0 ? 100.0 * sd / ns_med : 0.0, cyc_min);

    free(ns_per); free(cyc_per);
    xfree(re); xfree(im); xfree(tw_re); xfree(tw_im);
    return 0;
}
