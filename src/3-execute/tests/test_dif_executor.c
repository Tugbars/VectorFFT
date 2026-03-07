/**
 * test_dif_executor.c — Verify DIF backward executor
 *
 * Tests:
 *   1. DIF backward matches naive IDFT
 *   2. DIT forward → DIF backward roundtrip = identity
 *   3. inv_perm correctness
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

static double *aa64(size_t n) {
    double *p = (double *)vfft_aligned_alloc(64, n * sizeof(double));
    memset(p, 0, n * sizeof(double)); return p;
}

static void naive_dft(const double *ir, const double *ii,
                      double *nr, double *ni, size_t N, int dir) {
    double sign = (dir < 0) ? -1.0 : 1.0;
    for (size_t m = 0; m < N; m++) {
        double sr = 0, si = 0;
        for (size_t n = 0; n < N; n++) {
            double a = sign * 2.0 * M_PI * (double)m * (double)n / (double)N;
            sr += ir[n]*cos(a) - ii[n]*sin(a);
            si += ir[n]*sin(a) + ii[n]*cos(a);
        }
        nr[m] = sr; ni[m] = si;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Test 1: inv_perm correctness
 * ═══════════════════════════════════════════════════════════════ */

static int test_inv_perm(size_t N, const vfft_codelet_registry *reg) {
    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) { printf("  N=%-6zu  PLAN FAILED\n", N); return 0; }

    int pass = 1;
    if (plan->perm && plan->inv_perm) {
        /* Check: perm[inv_perm[i]] == i for all i */
        for (size_t i = 0; i < N; i++) {
            if (plan->perm[plan->inv_perm[i]] != i) {
                printf("  N=%-6zu  FAIL: perm[inv_perm[%zu]] = %zu != %zu\n",
                       N, i, plan->perm[plan->inv_perm[i]], i);
                pass = 0; break;
            }
        }
        /* Check: inv_perm[perm[i]] == i for all i */
        if (pass) {
            for (size_t i = 0; i < N; i++) {
                if (plan->inv_perm[plan->perm[i]] != i) {
                    printf("  N=%-6zu  FAIL: inv_perm[perm[%zu]] = %zu != %zu\n",
                           N, i, plan->inv_perm[plan->perm[i]], i);
                    pass = 0; break;
                }
            }
        }
    } else if (!plan->perm && !plan->inv_perm) {
        /* Single stage — no perm needed */
    } else {
        printf("  N=%-6zu  FAIL: perm and inv_perm not both present\n", N);
        pass = 0;
    }

    if (pass)
        printf("  N=%-6zu  %zu stg  inv_perm OK\n", N, plan->nstages);

    vfft_plan_destroy(plan);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 * Test 2: DIF backward matches naive IDFT
 * ═══════════════════════════════════════════════════════════════ */

static int test_dif_bwd(size_t N, const vfft_codelet_registry *reg) {
    double *ir = aa64(N), *ii_ = aa64(N);
    double *fr = aa64(N), *fi = aa64(N);
    double *ref_r = aa64(N), *ref_i = aa64(N);
    double *got_r = aa64(N), *got_i = aa64(N);

    srand(42 + (unsigned)N);
    for (size_t i = 0; i < N; i++) {
        ir[i] = (double)rand()/RAND_MAX*2.0-1.0;
        ii_[i] = (double)rand()/RAND_MAX*2.0-1.0;
    }

    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) { printf("  N=%-6zu  PLAN FAILED\n", N); return 0; }

    /* Forward DFT to get frequency domain */
    vfft_execute_fwd(plan, ir, ii_, fr, fi);

    /* Reference: naive IDFT */
    naive_dft(fr, fi, ref_r, ref_i, N, +1);

    /* DIF backward */
    vfft_execute_bwd(plan, fr, fi, got_r, got_i);

    /* Compare (IDFT result, unnormalized) */
    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(ref_r[i]-got_r[i]), fabs(ref_i[i]-got_i[i]));
        double m = fmax(fabs(ref_r[i]), fabs(ref_i[i]));
        if (e > err) err = e;
        if (m > mag) mag = m;
    }
    double rel = mag > 0 ? err/mag : err;
    double tol = 1e-12 * (1.0 + log2((double)N));
    int pass = rel < tol;

    /* Count DIF stages used */
    int n_dif = 0;
    for (size_t s = 0; s < plan->nstages; s++)
        if (plan->stages[s].tw_dif_bwd && plan->stages[s].K > 1)
            n_dif++;

    printf("  N=%-6zu  %zu stg  %d dif  bwd=%.1e  %s\n",
           N, plan->nstages, n_dif, rel, pass ? "PASS" : "FAIL");

    vfft_plan_destroy(plan);
    vfft_aligned_free(ir); vfft_aligned_free(ii_);
    vfft_aligned_free(fr); vfft_aligned_free(fi);
    vfft_aligned_free(ref_r); vfft_aligned_free(ref_i);
    vfft_aligned_free(got_r); vfft_aligned_free(got_i);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 * Test 3: DIT forward → DIF backward roundtrip
 * ═══════════════════════════════════════════════════════════════ */

static int test_roundtrip(size_t N, const vfft_codelet_registry *reg) {
    double *ir = aa64(N), *ii_ = aa64(N);
    double *fr = aa64(N), *fi = aa64(N);
    double *rr = aa64(N), *ri = aa64(N);

    srand(7777 + (unsigned)N);
    for (size_t i = 0; i < N; i++) {
        ir[i] = (double)rand()/RAND_MAX*2.0-1.0;
        ii_[i] = (double)rand()/RAND_MAX*2.0-1.0;
    }

    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) { printf("  N=%-6zu  PLAN FAILED\n", N); return 0; }

    /* Forward (DIT) */
    vfft_execute_fwd(plan, ir, ii_, fr, fi);

    /* Backward (DIF) */
    vfft_execute_bwd(plan, fr, fi, rr, ri);

    /* Normalize */
    for (size_t i = 0; i < N; i++) {
        rr[i] /= (double)N;
        ri[i] /= (double)N;
    }

    /* Compare with original */
    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(ir[i]-rr[i]), fabs(ii_[i]-ri[i]));
        double m = fmax(fabs(ir[i]), fabs(ii_[i]));
        if (e > err) err = e;
        if (m > mag) mag = m;
    }
    double rel = mag > 0 ? err/mag : err;
    double tol = 1e-12 * (1.0 + log2((double)N));
    int pass = rel < tol;

    printf("  N=%-6zu  %zu stg  roundtrip=%.1e  %s\n",
           N, plan->nstages, rel, pass ? "PASS" : "FAIL");

    vfft_plan_destroy(plan);
    vfft_aligned_free(ir); vfft_aligned_free(ii_);
    vfft_aligned_free(fr); vfft_aligned_free(fi);
    vfft_aligned_free(rr); vfft_aligned_free(ri);
    return pass;
}

int main(void) {
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  VectorFFT DIF Backward Executor Test\n");
    printf("════════════════════════════════════════════════════════════════\n\n");

    vfft_codelet_registry reg;
    vfft_register_all(&reg);

    printf("Registry:\n");
    vfft_print_registry(&reg);
    printf("\n");

    int p = 0, t = 0;

    size_t Ns[] = {
        /* Single stage (no perm) */
        2, 4, 5, 8,
        /* Two stages */
        10, 16, 20, 25, 32, 40, 50, 64,
        /* Three+ stages */
        100, 125, 128, 200, 250, 256, 400, 500, 512,
        /* Large */
        1000, 1024, 2000, 2048, 4096, 8000, 8192,
    };
    size_t nN = sizeof(Ns)/sizeof(Ns[0]);

    printf("── 1. Inverse permutation correctness ──\n\n");
    for (size_t i = 0; i < nN; i++) { t++; p += test_inv_perm(Ns[i], &reg); }

    printf("\n── 2. DIF backward vs naive IDFT ──\n\n");
    for (size_t i = 0; i < nN; i++) { t++; p += test_dif_bwd(Ns[i], &reg); }

    printf("\n── 3. DIT forward → DIF backward roundtrip ──\n\n");
    for (size_t i = 0; i < nN; i++) { t++; p += test_roundtrip(Ns[i], &reg); }

    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("  %d/%d %s\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    printf("════════════════════════════════════════════════════════════════\n");
    return p != t;
}
