/**
 * test_tw_fused.c — Verify fused tw codelet path in planner
 *
 * Compares: naive registry (notw+separate twiddle) vs optimized registry
 * (fused tw codelets for R=2,4,5,8). Both must match naive DFT reference.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "vfft_planner.h"
#include "fft_radix2_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"
#include "fft_radix8_dispatch.h"
#include "vfft_register_codelets.h"

static double *aa64(size_t n) {
    double *p = (double *)vfft_aligned_alloc(64, n * sizeof(double));
    memset(p, 0, n * sizeof(double)); return p;
}

static void naive_dft(const double *ir, const double *ii,
                      double *nr, double *ni, size_t N) {
    for (size_t m = 0; m < N; m++) {
        double sr = 0, si = 0;
        for (size_t n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)m * (double)n / (double)N;
            sr += ir[n]*cos(a) - ii[n]*sin(a);
            si += ir[n]*sin(a) + ii[n]*cos(a);
        }
        nr[m] = sr; ni[m] = si;
    }
}

typedef struct {
    const char *label;
    size_t N;
    const char *expect_tw;  /* which radices should use fused tw */
} test_case_t;

static int run_test(size_t N, const vfft_codelet_registry *reg_naive,
                    const vfft_codelet_registry *reg_opt, int verbose) {
    double *ir = aa64(N), *ii_ = aa64(N);
    double *ref_r = aa64(N), *ref_i = aa64(N);
    double *naive_r = aa64(N), *naive_i = aa64(N);
    double *opt_r = aa64(N), *opt_i = aa64(N);

    srand(42 + (unsigned)N);
    for (size_t i = 0; i < N; i++) {
        ir[i] = (double)rand()/RAND_MAX*2.0 - 1.0;
        ii_[i] = (double)rand()/RAND_MAX*2.0 - 1.0;
    }

    /* Reference: naive DFT */
    naive_dft(ir, ii_, ref_r, ref_i, N);

    /* Path 1: naive registry (notw + separate twiddle) */
    vfft_plan *plan_naive = vfft_plan_create(N, reg_naive);
    if (!plan_naive) { printf("  N=%-6zu  NAIVE PLAN FAILED\n", N); return 0; }
    vfft_execute_fwd(plan_naive, ir, ii_, naive_r, naive_i);

    /* Path 2: optimized registry (fused tw codelets where available) */
    vfft_plan *plan_opt = vfft_plan_create(N, reg_opt);
    if (!plan_opt) { printf("  N=%-6zu  OPT PLAN FAILED\n", N); return 0; }

    if (verbose) {
        printf("  N=%-6zu plan: ", N);
        for (size_t s = 0; s < plan_opt->nstages; s++) {
            if (s > 0) printf(" x ");
            printf("r%zu(K=%zu%s)", plan_opt->stages[s].radix,
                   plan_opt->stages[s].K,
                   (plan_opt->stages[s].tw_fwd && plan_opt->stages[s].K > 1) ? ",TW" : "");
        }
        printf("\n");
    }

    vfft_execute_fwd(plan_opt, ir, ii_, opt_r, opt_i);

    /* Check both against reference */
    double err_naive = 0, err_opt = 0, mag = 0;
    for (size_t i = 0; i < N; i++) {
        double en = fmax(fabs(naive_r[i]-ref_r[i]), fabs(naive_i[i]-ref_i[i]));
        double eo = fmax(fabs(opt_r[i]-ref_r[i]), fabs(opt_i[i]-ref_i[i]));
        double m = fmax(fabs(ref_r[i]), fabs(ref_i[i]));
        if (en > err_naive) err_naive = en;
        if (eo > err_opt) err_opt = eo;
        if (m > mag) mag = m;
    }
    double rel_naive = mag > 0 ? err_naive/mag : err_naive;
    double rel_opt   = mag > 0 ? err_opt/mag   : err_opt;
    double tol = 1e-12 * (1.0 + log2((double)N));

    int pass_naive = rel_naive < tol;
    int pass_opt   = rel_opt < tol;
    int pass = pass_naive && pass_opt;

    /* Check fused path was actually used */
    int n_fused = 0;
    for (size_t s = 0; s < plan_opt->nstages; s++)
        if (plan_opt->stages[s].tw_fwd && plan_opt->stages[s].K > 1)
            n_fused++;

    printf("  N=%-6zu  %zu stg  %d fused  naive=%.1e  opt=%.1e  %s\n",
           N, plan_opt->nstages, n_fused, rel_naive, rel_opt,
           pass ? "PASS" : "FAIL");

    /* Also test roundtrip */
    double *bwd_r = aa64(N), *bwd_i = aa64(N);
    vfft_execute_bwd(plan_opt, opt_r, opt_i, bwd_r, bwd_i);
    double rt_err = 0;
    for (size_t i = 0; i < N; i++) {
        bwd_r[i] /= (double)N; bwd_i[i] /= (double)N;
        double e = fmax(fabs(ir[i]-bwd_r[i]), fabs(ii_[i]-bwd_i[i]));
        if (e > rt_err) rt_err = e;
    }
    double rt_rel = mag > 0 ? rt_err/mag : rt_err;
    int rt_pass = rt_rel < tol;
    if (!rt_pass) {
        printf("         roundtrip FAIL: rel=%.1e\n", rt_rel);
        pass = 0;
    }

    vfft_plan_destroy(plan_naive);
    vfft_plan_destroy(plan_opt);
    vfft_aligned_free(ir); vfft_aligned_free(ii_);
    vfft_aligned_free(ref_r); vfft_aligned_free(ref_i);
    vfft_aligned_free(naive_r); vfft_aligned_free(naive_i);
    vfft_aligned_free(opt_r); vfft_aligned_free(opt_i);
    vfft_aligned_free(bwd_r); vfft_aligned_free(bwd_i);
    return pass;
}

int main(void) {
    printf("=== Fused TW Codelet Test ===\n\n");

    /* Naive registry: notw + separate twiddle for all stages */
    vfft_codelet_registry reg_naive;
    vfft_registry_init_naive(&reg_naive);

    /* Optimized registry: fused tw where available */
    vfft_codelet_registry reg_opt;
    vfft_register_all(&reg_opt);

    printf("Registered tw codelets:\n");
    for (size_t r = 2; r < 64; r++) {
        if (reg_opt.tw_fwd[r])
            printf("  R=%-3zu tw_fwd=yes tw_bwd=%s\n", r,
                   reg_opt.tw_bwd[r] ? "yes" : "no");
    }
    printf("\n");

    int p = 0, t = 0;

    /* Multi-stage sizes that exercise tw codelets */
    printf("-- Multi-stage (fused tw expected) --\n");
    size_t fused_Ns[] = {
        /* R=2 tw: */ 2*2*2*2, 2*8,
        /* R=4 tw: */ 4*4, 4*8, 4*32,
        /* R=5 tw: */ 5*8, 5*4, 5*2, 5*32, 5*5*8,
        /* R=8 tw: */ 8*8, 8*4, 8*32, 8*5,
        /* Mixed:  */ 2*4*5*8, 4*5*8*2, 8*8*8, 32*8, 32*32,
        /* Large:  */ 1024, 2048, 4096, 8192,
    };
    for (size_t i = 0; i < sizeof(fused_Ns)/sizeof(fused_Ns[0]); i++) {
        t++; p += run_test(fused_Ns[i], &reg_naive, &reg_opt, 1);
    }

    /* Sizes with no tw benefit (single stage = no twiddles) */
    printf("\n-- Single stage (no tw needed) --\n");
    size_t single_Ns[] = {2, 4, 5, 7, 8, 11, 13, 32};
    for (size_t i = 0; i < sizeof(single_Ns)/sizeof(single_Ns[0]); i++) {
        t++; p += run_test(single_Ns[i], &reg_naive, &reg_opt, 1);
    }

    /* Sizes with odd K (SIMD alignment test) */
    printf("\n-- SIMD alignment stress (N=prime*pow2) --\n");
    size_t align_Ns[] = {5*8, 5*16, 5*32, 5*64, 7*8, 7*16, 11*8, 13*8};
    for (size_t i = 0; i < sizeof(align_Ns)/sizeof(align_Ns[0]); i++) {
        t++; p += run_test(align_Ns[i], &reg_naive, &reg_opt, 1);
    }

    printf("\n=== %d/%d %s ===\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    return p != t;
}
