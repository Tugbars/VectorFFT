/* microbench_codelet.c — isolated single-codelet latency microbench.
 * FFTW-free, MKL-free, bundle-header-free: it only externs ONE codelet symbol
 * and times it. Purpose: A/B the SAME symbol built from two different codelet
 * trees (OLD src/prototype vs NEW src/dag-fft-compiler) so the ONLY variable is
 * the codelet machine code — the executor, planner, twiddle layout, buffer,
 * compiler, and flags are all held identical by the driver (ab_codelets.sh /
 * ab_codelets_icx.ps1).
 *
 * Portable: POSIX (gcc/clang in WSL) AND Windows (Intel ICX / MSVC). The timer
 * is QueryPerformanceCounter on Windows, clock_gettime(MONOTONIC) elsewhere;
 * aligned alloc is _aligned_malloc vs posix_memalign.
 *
 * Scope: the in-place c2c family (n1 / t1 / t1s / log3 / dit / dif / fwd / bwd),
 * which all share the 6-arg in-place rio ABI:
 *     void FN(double* rio_re, double* rio_im,
 *             const double* tw_re, const double* tw_im, size_t ios, size_t me)
 *
 * Working set is deliberately tiny (me=8, ios=8) so the buffers stay L1/L2
 * resident: this isolates the codelet's compute + instruction-scheduling (the
 * thing the new compiler changed) rather than memory bandwidth. Both trees see
 * the identical resident buffer, so any memory effect is common-mode.
 *
 * FTZ/DAZ are forced on so that magnitudes drifting toward denormals over the
 * repeated in-place passes never trip the microcode assist (an unrelated
 * confound). Values may saturate to inf/nan over many passes — fine, AVX FMA
 * throughput is unaffected, and correctness is gated separately.
 *
 * Config macros (set by the driver via -D / -D...):
 *   RN  = radix (buffer sizing)
 *   FN  = exported codelet symbol (e.g. radix16_t1_dit_fwd_avx2)
 *   T1S = 1 if this is a t1s (scalar-twiddle) codelet, else 0 (flat me-strided)
 * Precision knobs (env, optional): MB_REPS_BUDGET, MB_BESTOF.
 *
 * Output: exactly one line "ns=<min ns per call>" on stdout. Nothing else.
 */
#ifndef _WIN32
#  define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>   /* FTZ/DAZ control + codelet intrinsics */
#ifdef _WIN32
#  include <windows.h>
#else
#  include <time.h>
#endif

#define MPI 3.14159265358979323846

#ifndef RN
#define RN 16
#endif
#ifndef FN
#define FN radix16_t1_dit_fwd_avx2
#endif
#ifndef T1S
#define T1S 0
#endif

/* 6-arg in-place rio signature shared by n1/t1/t1s/log3. */
extern void FN(double *, double *, const double *, const double *, size_t, size_t);

static void *xaligned(size_t align, size_t sz) {
#ifdef _WIN32
    return _aligned_malloc(sz, align);
#else
    void *p = NULL;
    if (posix_memalign(&p, align, sz) != 0) return NULL;
    return p;
#endif
}
static void xfree(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static double now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

int main(void) {
    const int N = RN;
    const size_t me = 8, ios = 8;
    const size_t buf = (size_t)N * ios;

    /* Denormals-are-zero / flush-to-zero: remove the only memory-independent
     * confound (denormal assist) so old vs new compares pure codegen. */
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    double *re    = (double *)xaligned(64, buf * sizeof(double));
    double *im    = (double *)xaligned(64, buf * sizeof(double));
    double *tw_re = (double *)xaligned(64, ((size_t)N * me + 128) * sizeof(double));
    double *tw_im = (double *)xaligned(64, ((size_t)N * me + 128) * sizeof(double));
    if (!re || !im || !tw_re || !tw_im) { fprintf(stderr, "alloc failed\n"); return 2; }

    /* Benign deterministic data; correctness is gated elsewhere. */
    for (size_t i = 0; i < buf; i++) {
        re[i] = (double)((i * 2654435761u) & 0xffff) / 65535.0 - 0.5;
        im[i] = (double)((i * 40503u)      & 0xffff) / 65535.0 - 0.5;
    }
    /* Twiddle slot s holds W_N^{s+1}; t1/log3 read flat me-strided, t1s scalar. */
    for (int s = 0; s < N; s++) {
        double ang = -2.0 * MPI * (double)(s + 1) / (double)N;
        double c = cos(ang), sn = sin(ang);
        if (T1S) { tw_re[s] = c; tw_im[s] = sn; }
        else for (size_t b = 0; b < me; b++) {
            tw_re[(size_t)s * me + b] = c;
            tw_im[(size_t)s * me + b] = sn;
        }
    }

    /* Scale reps so each timed burst does a comparable amount of work across
     * radices (~constant total butterfly count), with a generous floor.
     * Precision knobs (env, optional): MB_REPS_BUDGET raises the numerator =>
     * longer bursts => tighter floor (default 2e6); MB_BESTOF raises the
     * internal best-of count (default 11). Used by the radix-verify re-run. */
    double reps_budget = 2000000.0;
    int    bestof      = 11;
    const char *e;
    if ((e = getenv("MB_REPS_BUDGET")) && atof(e) > 0) reps_budget = atof(e);
    if ((e = getenv("MB_BESTOF"))      && atoi(e) > 0) bestof      = atoi(e);

    long reps = (long)(reps_budget / (double)(buf + 1));
    if (reps < 500)      reps = 500;
    if (reps > 50000000) reps = 50000000;

    /* warmup (also pages in + warms caches/branch predictors) */
    for (long w = 0; w < reps; w++) FN(re, im, tw_re, tw_im, ios, me);

    /* best-of-N internal floor: rejects upward outliers (interrupts, throttle) */
    double best = 1e30;
    for (int t = 0; t < bestof; t++) {
        double t0 = now_ns();
        for (long i = 0; i < reps; i++) FN(re, im, tw_re, tw_im, ios, me);
        double ns = (now_ns() - t0) / (double)reps;
        if (ns < best) best = ns;
    }

    printf("ns=%.4f\n", best);
    xfree(re); xfree(im); xfree(tw_re); xfree(tw_im);
    return 0;
}
