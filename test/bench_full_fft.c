/**
 * bench_full_fft.c — VectorFFT full-radix benchmark vs FFTW
 *
 * Wired radixes:
 *   R=2,3,4,5,7,8,10,16,25,32 — fused DIT tw + fused DIF tw (scalar/AVX2/AVX-512)
 *   R=11,13,17,19,23           — genfft notw (no fused tw)
 *   R=64,128                   — N1-only (innermost stage, K=1)
 *
 * Tests:
 *   1. Correctness: VectorFFT fwd vs FFTW fwd
 *   2. Roundtrip:   VectorFFT fwd->bwd vs identity
 *   3. Benchmark:   VectorFFT vs FFTW (ns, min-of-5, FFTW_ESTIMATE)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * ISA compatibility preamble
 *
 * Three dispatch headers (radix7, radix32, vfft_register_codelets) each
 * define vfft_isa_level_t and vfft_detect_isa() without a shared guard,
 * causing redefinition errors when all are included in one TU.
 *
 * Fix: define the canonical enum + function here first, then redirect
 * each header's duplicate definition to a throwaway private name via
 * #define / #undef around that specific include.
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum {
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2   = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_level_t;
#endif

/* Single canonical detect function — compile-time, consistent with -m flags. */
static inline vfft_isa_level_t vfft_detect_isa(void)
{
#if defined(__AVX512F__)
    return VFFT_ISA_AVX512;
#elif defined(__AVX2__)
    return VFFT_ISA_AVX2;
#else
    return VFFT_ISA_SCALAR;
#endif
}

/* ── DIT dispatch: notw + fused twiddle ─────────────────────────────── */
#include "fft_radix2_dispatch.h"
#include "fft_radix3_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"

/* radix7 defines vfft_detect_isa without VFFT_ISA_DETECT_DEFINED guard */
#define vfft_detect_isa _bench_detect_isa_r7
#include "fft_radix7_dispatch.h"
#undef vfft_detect_isa

#include "fft_radix8_dispatch.h"
#include "fft_radix16_dispatch.h"   /* already guarded by VFFT_ISA_DETECT_DEFINED */

/* radix32 same issue as radix7 */
#define vfft_detect_isa _bench_detect_isa_r32
#include "fft_radix32_dispatch.h"
#undef vfft_detect_isa
#include "fft_radix10_dispatch.h"
#include "fft_radix25_dispatch.h"

/* ── DIF dispatch: fused twiddle after butterfly ────────────────────── */
#include "fft_radix2_dif_dispatch.h"
#include "fft_radix3_dif_dispatch.h"
#include "fft_radix4_dif_dispatch.h"
#include "fft_radix5_dif_dispatch.h"
#include "fft_radix7_dif_dispatch.h"
#include "fft_radix8_dif_dispatch.h"
#include "fft_radix16_dif_dispatch.h"
#include "fft_radix32_dif_dispatch.h"
#include "fft_radix10_dif_dispatch.h"
#include "fft_radix25_dif_dispatch.h"

/* ── Genfft primes: notw only, no fused tw ──────────────────────────── */
#include "fft_radix11_genfft.h"
#include "fft_radix13_genfft.h"
#include "fft_radix17_genfft.h"
#include "fft_radix19_genfft.h"
#include "fft_radix23_genfft.h"

/* ── N1 large: innermost stage only (K=1) ───────────────────────────── */
#include "fft_radix64_n1.h"
#include "fft_radix128_n1.h"

/* ── Planner ─────────────────────────────────────────────────────────── */
#include "vfft_planner.h"

/* register_codelets also defines vfft_detect_isa without guard */
#define vfft_detect_isa _bench_detect_isa_reg
#include "vfft_register_codelets.h"
#undef vfft_detect_isa

/* ═══════════════════════════════════════════════════════════════════════
 * Timing
 * ═══════════════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#include <windows.h>
static double get_ns(void)
{
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#include <time.h>
static double get_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static double *aa64(size_t n)
{
    double *p = (double *)vfft_aligned_alloc(64, n * sizeof(double));
    memset(p, 0, n * sizeof(double));
    return p;
}

static void factstr(const vfft_plan *plan, char *buf, size_t bufsz)
{
    int pos = 0;
    for (size_t s = 0; s < plan->nstages; s++) {
        if (s) pos += snprintf(buf + pos, bufsz - (size_t)pos, "x");
        pos += snprintf(buf + pos, bufsz - (size_t)pos, "%zu",
                        plan->stages[s].radix);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Correctness
 * ═══════════════════════════════════════════════════════════════════════ */

static int test_correctness(size_t N, const vfft_codelet_registry *reg)
{
    double *ir = aa64(N), *ii = aa64(N);
    double *vr = aa64(N), *vi = aa64(N);

    srand(42 + (unsigned)N);
    for (size_t i = 0; i < N; i++) {
        ir[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        ii[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    /* FFTW reference */
    fftw_complex *fin  = fftw_alloc_complex(N);
    fftw_complex *fout = fftw_alloc_complex(N);
    fftw_plan fp = fftw_plan_dft_1d((int)N, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE);
    for (size_t i = 0; i < N; i++) { fin[i][0] = ir[i]; fin[i][1] = ii[i]; }
    fftw_execute(fp);

    /* VectorFFT forward */
    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) {
        printf("  N=%-6zu  PLAN FAILED\n", N);
        fftw_destroy_plan(fp); fftw_free(fin); fftw_free(fout);
        vfft_aligned_free(ir); vfft_aligned_free(ii);
        vfft_aligned_free(vr); vfft_aligned_free(vi);
        return 0;
    }
    vfft_execute_fwd(plan, ir, ii, vr, vi);

    /* Forward error */
    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(vr[i] - fout[i][0]), fabs(vi[i] - fout[i][1]));
        double m = fmax(fabs(fout[i][0]), fabs(fout[i][1]));
        if (e > err) err = e;
        if (m > mag) mag = m;
    }
    double fwd_rel = mag > 0 ? err / mag : err;
    double tol     = 1e-12 * (1.0 + log2((double)N));

    /* Roundtrip */
    double *rr = aa64(N), *ri = aa64(N);
    vfft_execute_bwd(plan, vr, vi, rr, ri);
    double rt_err = 0;
    for (size_t i = 0; i < N; i++) {
        rr[i] /= (double)N;
        ri[i] /= (double)N;
        double e = fmax(fabs(ir[i] - rr[i]), fabs(ii[i] - ri[i]));
        if (e > rt_err) rt_err = e;
    }
    double rt_rel = mag > 0 ? rt_err / mag : rt_err;

    int pass = (fwd_rel < tol) && (rt_rel < tol);

    /* Count fused stages */
    int n_dit = 0, n_dif = 0;
    for (size_t s = 0; s < plan->nstages; s++) {
        if (plan->stages[s].tw_fwd     && plan->stages[s].K > 1) n_dit++;
        if (plan->stages[s].tw_dif_bwd && plan->stages[s].K > 1) n_dif++;
    }

    char fact[64] = "";
    factstr(plan, fact, sizeof(fact));

    printf("  N=%-6zu  %-16s  stg=%zu dit=%d dif=%d  fwd=%.1e  rt=%.1e  %s\n",
           N, fact, plan->nstages, n_dit, n_dif, fwd_rel, rt_rel,
           pass ? "PASS" : "FAIL");

    vfft_plan_destroy(plan);
    fftw_destroy_plan(fp); fftw_free(fin); fftw_free(fout);
    vfft_aligned_free(ir); vfft_aligned_free(ii);
    vfft_aligned_free(vr); vfft_aligned_free(vi);
    vfft_aligned_free(rr); vfft_aligned_free(ri);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark
 * ═══════════════════════════════════════════════════════════════════════ */

static void bench(size_t N, const vfft_codelet_registry *reg)
{
    double *ir = aa64(N), *ii = aa64(N);
    double *vr = aa64(N), *vi = aa64(N);
    double *br = aa64(N), *bi = aa64(N);

    srand((unsigned)N);
    for (size_t i = 0; i < N; i++) {
        ir[i] = (double)rand() / RAND_MAX;
        ii[i] = (double)rand() / RAND_MAX;
    }

    fftw_complex *fin  = fftw_alloc_complex(N);
    fftw_complex *fout = fftw_alloc_complex(N);
    fftw_complex *bout = fftw_alloc_complex(N);
    fftw_plan fp_fwd = fftw_plan_dft_1d((int)N, fin,  fout, FFTW_FORWARD,  FFTW_ESTIMATE);
    fftw_plan fp_bwd = fftw_plan_dft_1d((int)N, fout, bout, FFTW_BACKWARD, FFTW_ESTIMATE);
    for (size_t i = 0; i < N; i++) { fin[i][0] = ir[i]; fin[i][1] = ii[i]; }

    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) {
        printf("  N=%-6zu  PLAN FAILED\n", N);
        fftw_destroy_plan(fp_fwd); fftw_destroy_plan(fp_bwd);
        fftw_free(fin); fftw_free(fout); fftw_free(bout);
        vfft_aligned_free(ir); vfft_aligned_free(ii);
        vfft_aligned_free(vr); vfft_aligned_free(vi);
        vfft_aligned_free(br); vfft_aligned_free(bi);
        return;
    }

    int reps;
    if (N <= 512)
        reps = 5000;
    else if (N <= 2048)
        reps = 2000;
    else if (N <= 8192)
        reps = 1000;
    else if (N <= 32768)
        reps = 200;
    else
        reps = 100;

    /* Warm up */
    for (int r = 0; r < 5; r++) {
        vfft_execute_fwd(plan, ir, ii, vr, vi);
        vfft_execute_bwd(plan, vr, vi, br, bi);
        fftw_execute(fp_fwd);
        fftw_execute(fp_bwd);
    }

    double t0, t1, best;

#define BENCH_LOOP(var, body)                               \
    best = 1e18;                                            \
    for (int _trial = 0; _trial < 5; _trial++) {           \
        t0 = get_ns();                                      \
        for (int _r = 0; _r < reps; _r++) { body; }        \
        t1 = get_ns();                                      \
        double _ns = (t1 - t0) / reps;                     \
        if (_ns < best) best = _ns;                         \
    }                                                       \
    (var) = best

    double vfft_fwd_ns, vfft_rt_ns, fftw_fwd_ns, fftw_rt_ns;

    BENCH_LOOP(vfft_fwd_ns, vfft_execute_fwd(plan, ir, ii, vr, vi));
    BENCH_LOOP(vfft_rt_ns,  vfft_execute_fwd(plan, ir, ii, vr, vi);
                            vfft_execute_bwd(plan, vr, vi, br, bi));
    BENCH_LOOP(fftw_fwd_ns, fftw_execute(fp_fwd));
    BENCH_LOOP(fftw_rt_ns,  fftw_execute(fp_fwd); fftw_execute(fp_bwd));

#undef BENCH_LOOP

    char fact[64] = "";
    factstr(plan, fact, sizeof(fact));

#define CLR_GREEN "\033[92m"
#define CLR_RESET "\033[0m"
    double fwd_x = fftw_fwd_ns / vfft_fwd_ns;
    double rt_x  = fftw_rt_ns  / vfft_rt_ns;
    printf("  N=%-6zu  %-16s  vfft_fwd=%7.0f  fftw_fwd=%7.0f  fwd_x=%s%.2f%s"
           "  vfft_rt=%7.0f  fftw_rt=%7.0f  rt_x=%s%.2f%s\n",
           N, fact,
           vfft_fwd_ns, fftw_fwd_ns,
           fwd_x > 1.4 ? CLR_GREEN : "", fwd_x, fwd_x > 1.4 ? CLR_RESET : "",
           vfft_rt_ns, fftw_rt_ns,
           rt_x  > 1.4 ? CLR_GREEN : "", rt_x,  rt_x  > 1.4 ? CLR_RESET : "");
#undef CLR_GREEN
#undef CLR_RESET

    vfft_plan_destroy(plan);
    fftw_destroy_plan(fp_fwd); fftw_destroy_plan(fp_bwd);
    fftw_free(fin); fftw_free(fout); fftw_free(bout);
    vfft_aligned_free(ir); vfft_aligned_free(ii);
    vfft_aligned_free(vr); vfft_aligned_free(vi);
    vfft_aligned_free(br); vfft_aligned_free(bi);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  VectorFFT Full-Radix Benchmark\n");
    printf("  R={2,3,4,5,7,8,10,16,25,32} fused DIT+DIF  "
           "R={11,13,17,19,23} genfft  R={64,128} N1\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");

    vfft_codelet_registry reg;
    vfft_register_all(&reg);
    printf("Registry:\n");
    vfft_print_registry(&reg);
    printf("\n");

    static const size_t test_Ns[] = {
        /* Pure pow2 — exercises R=128,32,16,8,4,2
         *   256  = 128x2
         *   512  = 128x4
         *   1024 = 128x8
         *   2048 = 128x16
         *   4096 = 128x32
         *   8192 = 128x32x2
         *  16384 = 128x32x4
         *  32768 = 128x32x8                                             */
        256, 512, 1024, 2048, 4096, 8192, 16384, 32768,

        /* R=64 innermost (128 does not divide N)
         *   320 = 64x5    448 = 64x7                                   */
        320, 448,

        /* R=5 */
        200, 400, 1000, 2000, 5000, 10000,

        /* R=7 */
        224, 896, 1792, 3584,

        /* Genfft primes: N = prime * pow2 (pow2 >= 8 for SIMD K)
         * R=11 */  88, 704, 5632,
        /* R=13 */ 104, 832, 6656,
        /* R=17 */ 136, 1088,
        /* R=19 */ 152, 1216,
        /* R=23 */ 184, 1472,

        /* R=10 */
        80, 640, 4000, 8000,

        /* R=25 */
        200, 800, 5000, 20000,

        /* Large mixed */
        40000,
    };
    size_t nN = sizeof(test_Ns) / sizeof(test_Ns[0]);

    /* ── Correctness ── */
    printf("── Correctness: VectorFFT fwd vs FFTW + roundtrip ──\n\n");
    int pass = 0, total = 0;
    for (size_t i = 0; i < nN; i++) {
        total++;
        pass += test_correctness(test_Ns[i], &reg);
    }
    printf("\n  %d/%d %s\n\n", pass, total,
           pass == total ? "ALL PASSED" : "FAILURES ABOVE");

    if (pass != total) {
        printf("Correctness failures — skipping benchmarks.\n");
        return 1;
    }

    /* ── Benchmark ── */
    printf("── Benchmark: VectorFFT vs FFTW (ns, min-of-5, FFTW_ESTIMATE) ──\n");
    printf("  %-8s  %-16s  %9s  %9s  %6s  %9s  %9s  %6s\n",
           "N", "factors", "vfft_fwd", "fftw_fwd", "fwd_x",
           "vfft_rt", "fftw_rt", "rt_x");
    printf("  ──────────────────────────────────────────────────────────────"
           "────────────────────\n");

    for (size_t i = 0; i < nN; i++)
        bench(test_Ns[i], &reg);

    return 0;
}
