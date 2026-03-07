/**
 * bench_full_fft.c — Full-N FFT benchmark: VectorFFT vs FFTW
 *
 * Available optimized codelets: R=2,4,5,8 (scalar + AVX-512)
 * with fused DIT tw + DIF backward.
 *
 * All other radixes fall back to naive codelets in the planner.
 *
 * Tests:
 *   1. Correctness: VectorFFT fwd vs FFTW fwd
 *   2. Roundtrip: VectorFFT fwd→bwd vs identity
 *   3. Benchmark: VectorFFT vs FFTW (ns, min-of-5)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* DIT dispatch (notw + tw) */
#include "fft_radix2_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"
#include "fft_radix8_dispatch.h"

/* DIF dispatch (tw after butterfly) */
#include "fft_radix2_dif_dispatch.h"
#include "fft_radix4_dif_dispatch.h"
#include "fft_radix5_dif_dispatch.h"
#include "fft_radix8_dif_dispatch.h"

/* Planner + registry */
#include "vfft_planner.h"
#include "vfft_register_codelets.h"

/* ═══════════════════════════════════════════════════════════════ */

static double *aa64(size_t n) {
    double *p = (double *)vfft_aligned_alloc(64, n * sizeof(double));
    memset(p, 0, n * sizeof(double));
    return p;
}

static double get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ═══════════════════════════════════════════════════════════════
 * CORRECTNESS: compare VectorFFT forward vs FFTW forward
 * ═══════════════════════════════════════════════════════════════ */

static int test_correctness(size_t N, const vfft_codelet_registry *reg) {
    double *ir = aa64(N), *ii = aa64(N);
    double *vr = aa64(N), *vi = aa64(N);

    srand(42 + (unsigned)N);
    for (size_t i = 0; i < N; i++) {
        ir[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        ii[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    /* FFTW reference */
    fftw_complex *fin = fftw_alloc_complex(N);
    fftw_complex *fout = fftw_alloc_complex(N);
    fftw_plan fp = fftw_plan_dft_1d((int)N, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE);
    for (size_t i = 0; i < N; i++) { fin[i][0] = ir[i]; fin[i][1] = ii[i]; }
    fftw_execute(fp);

    /* VectorFFT forward */
    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) {
        printf("  N=%-6zu  PLAN FAILED\n", N);
        fftw_destroy_plan(fp); fftw_free(fin); fftw_free(fout);
        return 0;
    }
    vfft_execute_fwd(plan, ir, ii, vr, vi);

    /* Compare */
    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++) {
        double er = fabs(vr[i] - fout[i][0]);
        double ei = fabs(vi[i] - fout[i][1]);
        double e = fmax(er, ei);
        double m = fmax(fabs(fout[i][0]), fabs(fout[i][1]));
        if (e > err) err = e;
        if (m > mag) mag = m;
    }
    double rel = mag > 0 ? err / mag : err;
    double tol = 1e-12 * (1.0 + log2((double)N));

    /* Roundtrip */
    double *rr = aa64(N), *ri = aa64(N);
    vfft_execute_bwd(plan, vr, vi, rr, ri);
    double rt_err = 0;
    for (size_t i = 0; i < N; i++) {
        rr[i] /= (double)N; ri[i] /= (double)N;
        double e = fmax(fabs(ir[i] - rr[i]), fabs(ii[i] - ri[i]));
        if (e > rt_err) rt_err = e;
    }
    double rt_rel = mag > 0 ? rt_err / mag : rt_err;

    int pass = (rel < tol) && (rt_rel < tol);

    /* Count fused stages */
    int n_dit = 0, n_dif = 0;
    for (size_t s = 0; s < plan->nstages; s++) {
        if (plan->stages[s].tw_fwd && plan->stages[s].K > 1) n_dit++;
        if (plan->stages[s].tw_dif_bwd && plan->stages[s].K > 1) n_dif++;
    }

    printf("  N=%-6zu %zu stg  dit=%d dif=%d  fwd=%.1e  rt=%.1e  %s\n",
           N, plan->nstages, n_dit, n_dif, rel, rt_rel,
           pass ? "PASS" : "FAIL");

    vfft_plan_destroy(plan);
    fftw_destroy_plan(fp); fftw_free(fin); fftw_free(fout);
    vfft_aligned_free(ir); vfft_aligned_free(ii);
    vfft_aligned_free(vr); vfft_aligned_free(vi);
    vfft_aligned_free(rr); vfft_aligned_free(ri);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 * BENCHMARK: VectorFFT vs FFTW
 * ═══════════════════════════════════════════════════════════════ */

static void bench(size_t N, const vfft_codelet_registry *reg) {
    double *ir = aa64(N), *ii = aa64(N);
    double *vr = aa64(N), *vi = aa64(N);
    double *br = aa64(N), *bi = aa64(N);

    for (size_t i = 0; i < N; i++) {
        ir[i] = (double)rand() / RAND_MAX;
        ii[i] = (double)rand() / RAND_MAX;
    }

    /* FFTW setup */
    fftw_complex *fin = fftw_alloc_complex(N);
    fftw_complex *fout = fftw_alloc_complex(N);
    fftw_complex *bout = fftw_alloc_complex(N);
    fftw_plan fp_fwd = fftw_plan_dft_1d((int)N, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan fp_bwd = fftw_plan_dft_1d((int)N, fout, bout, FFTW_BACKWARD, FFTW_ESTIMATE);
    for (size_t i = 0; i < N; i++) { fin[i][0] = ir[i]; fin[i][1] = ii[i]; }

    /* VectorFFT setup */
    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) { printf("  N=%-6zu  PLAN FAILED\n", N); return; }

    int reps;
    if      (N <= 256)   reps = 200000;
    else if (N <= 1024)  reps = 100000;
    else if (N <= 4096)  reps = 50000;
    else if (N <= 16384) reps = 10000;
    else if (N <= 65536) reps = 5000;
    else                 reps = 1000;

    /* Warm up */
    for (int r = 0; r < 5; r++) {
        vfft_execute_fwd(plan, ir, ii, vr, vi);
        vfft_execute_bwd(plan, vr, vi, br, bi);
        fftw_execute(fp_fwd);
        fftw_execute(fp_bwd);
    }

    double t0, t1, best;

    /* VectorFFT forward */
    best = 1e18;
    for (int trial = 0; trial < 5; trial++) {
        t0 = get_ns();
        for (int r = 0; r < reps; r++)
            vfft_execute_fwd(plan, ir, ii, vr, vi);
        t1 = get_ns();
        double ns = (t1 - t0) / reps;
        if (ns < best) best = ns;
    }
    double vfft_fwd_ns = best;

    /* VectorFFT roundtrip (fwd + bwd) */
    best = 1e18;
    for (int trial = 0; trial < 5; trial++) {
        t0 = get_ns();
        for (int r = 0; r < reps; r++) {
            vfft_execute_fwd(plan, ir, ii, vr, vi);
            vfft_execute_bwd(plan, vr, vi, br, bi);
        }
        t1 = get_ns();
        double ns = (t1 - t0) / reps;
        if (ns < best) best = ns;
    }
    double vfft_rt_ns = best;

    /* FFTW forward */
    best = 1e18;
    for (int trial = 0; trial < 5; trial++) {
        t0 = get_ns();
        for (int r = 0; r < reps; r++)
            fftw_execute(fp_fwd);
        t1 = get_ns();
        double ns = (t1 - t0) / reps;
        if (ns < best) best = ns;
    }
    double fftw_fwd_ns = best;

    /* FFTW roundtrip */
    best = 1e18;
    for (int trial = 0; trial < 5; trial++) {
        t0 = get_ns();
        for (int r = 0; r < reps; r++) {
            fftw_execute(fp_fwd);
            fftw_execute(fp_bwd);
        }
        t1 = get_ns();
        double ns = (t1 - t0) / reps;
        if (ns < best) best = ns;
    }
    double fftw_rt_ns = best;

    /* Print factorization */
    char fact_str[128] = "";
    int pos = 0;
    for (size_t s = 0; s < plan->nstages; s++) {
        if (s > 0) pos += snprintf(fact_str + pos, sizeof(fact_str) - pos, "x");
        pos += snprintf(fact_str + pos, sizeof(fact_str) - pos, "%zu",
                        plan->stages[s].radix);
    }

    printf("  N=%-6zu  %-12s  vfft_fwd=%7.0f  fftw_fwd=%7.0f  ratio=%.2fx  "
           "vfft_rt=%7.0f  fftw_rt=%7.0f  rt_ratio=%.2fx\n",
           N, fact_str,
           vfft_fwd_ns, fftw_fwd_ns, fftw_fwd_ns / vfft_fwd_ns,
           vfft_rt_ns, fftw_rt_ns, fftw_rt_ns / vfft_rt_ns);

    vfft_plan_destroy(plan);
    fftw_destroy_plan(fp_fwd); fftw_destroy_plan(fp_bwd);
    fftw_free(fin); fftw_free(fout); fftw_free(bout);
    vfft_aligned_free(ir); vfft_aligned_free(ii);
    vfft_aligned_free(vr); vfft_aligned_free(vi);
    vfft_aligned_free(br); vfft_aligned_free(bi);
}

int main(void) {
    printf("════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  VectorFFT Full-N Benchmark — R={2,4,5,8} optimized, fused DIT tw + DIF bwd\n");
    printf("════════════════════════════════════════════════════════════════════════════════════════════\n\n");

    vfft_codelet_registry reg;
    vfft_register_all(&reg);
    printf("Registry:\n");
    vfft_print_registry(&reg);
    printf("\n");

    /* Sizes that decompose cleanly into R={2,4,5,8} */
    size_t test_Ns[] = {
        /* Pure pow2 (R=8,4,2) */
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        /* With R=5 */
        40, 80, 160, 320, 640, 1280, 2560, 5120,
        /* Mixed */
        100, 200, 400, 500, 800, 1000, 2000, 4000, 5000, 8000,
        10000, 20000, 40000,
    };
    size_t nN = sizeof(test_Ns) / sizeof(test_Ns[0]);

    /* ── Correctness ── */
    printf("── Correctness: VectorFFT fwd vs FFTW + roundtrip ──\n\n");
    int p = 0, t = 0;
    for (size_t i = 0; i < nN; i++) {
        t++; p += test_correctness(test_Ns[i], &reg);
    }
    printf("\n  %d/%d %s\n\n", p, t, p == t ? "ALL PASSED" : "FAILURES");

    if (p != t) {
        printf("CORRECTNESS FAILURES — skipping benchmarks.\n");
        return 1;
    }

    /* ── Benchmark ── */
    printf("── Benchmark: VectorFFT vs FFTW (ns, min-of-5, FFTW_ESTIMATE) ──\n");
    printf("  %-8s  %-12s  %9s  %9s  %7s  %9s  %9s  %9s\n",
           "N", "factors", "vfft_fwd", "fftw_fwd", "fwd_×", "vfft_rt", "fftw_rt", "rt_×");
    printf("  ──────────────────────────────────────────────────────────────────────────────────────\n");

    for (size_t i = 0; i < nN; i++) {
        bench(test_Ns[i], &reg);
    }

    return 0;
}
