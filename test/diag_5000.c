/**
 * diag_5000.c — Minimal diagnostic for N=5000 crash
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum {
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2 = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_level_t;
#endif
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

#include "fft_radix2_dispatch.h"
#include "fft_radix3_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"
#define vfft_detect_isa _diag_isa_r7
#include "fft_radix7_dispatch.h"
#undef vfft_detect_isa
#include "fft_radix8_dispatch.h"
#include "fft_radix16_dispatch.h"
#define vfft_detect_isa _diag_isa_r32
#include "fft_radix32_dispatch.h"
#undef vfft_detect_isa
#include "fft_radix10_dispatch.h"
#include "fft_radix25_dispatch.h"
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
#include "fft_radix11_genfft.h"
#include "fft_radix13_genfft.h"
#include "fft_radix17_genfft.h"
#include "fft_radix19_genfft.h"
#include "fft_radix23_genfft.h"
#include "fft_radix64_n1.h"
#include "fft_radix128_n1.h"
#include "vfft_planner.h"
#define vfft_detect_isa _diag_isa_reg
#include "vfft_register_codelets.h"
#undef vfft_detect_isa

int main(void)
{
    printf("=== N=5000 diagnostic ===\n");
    fflush(stdout);

    vfft_codelet_registry reg;
    vfft_register_all(&reg);
    printf("Registry OK\n");
    fflush(stdout);

    size_t N = 5000;

    /* Test factorization */
    vfft_factorization fact;
    int rc = vfft_factorize(N, &reg, &fact);
    printf("vfft_factorize(5000) = %d, nfactors=%zu\n", rc, fact.nfactors);
    printf("  factors:");
    for (size_t i = 0; i < fact.nfactors; i++)
        printf(" %zu", fact.factors[i]);
    printf("\n");
    fflush(stdout);

    /* Test plan creation */
    printf("Creating plan...\n");
    fflush(stdout);
    vfft_plan *plan = vfft_plan_create(N, &reg);
    if (!plan) {
        printf("  PLAN FAILED\n");
        return 1;
    }
    printf("Plan OK: %zu stages\n", plan->nstages);
    for (size_t s = 0; s < plan->nstages; s++) {
        printf("  stage %zu: R=%zu K=%zu fwd=%p tw_fwd=%p\n",
               s, plan->stages[s].radix, plan->stages[s].K,
               (void*)plan->stages[s].fwd, (void*)plan->stages[s].tw_fwd);
    }
    fflush(stdout);

    /* Allocate test data */
    double *ir = (double *)vfft_aligned_alloc(64, N * 8);
    double *ii = (double *)vfft_aligned_alloc(64, N * 8);
    double *vr = (double *)vfft_aligned_alloc(64, N * 8);
    double *vi = (double *)vfft_aligned_alloc(64, N * 8);
    for (size_t i = 0; i < N; i++) {
        ir[i] = (double)(i % 17) / 17.0;
        ii[i] = (double)(i % 13) / 13.0;
    }
    printf("Data allocated\n");
    fflush(stdout);

    /* Manual stage-by-stage execution to find crash */
    printf("Manual stage-by-stage forward...\n");
    fflush(stdout);

    double *src_re = plan->buf_a_re, *src_im = plan->buf_a_im;
    double *dst_re = plan->buf_b_re, *dst_im = plan->buf_b_im;

    /* Apply permutation */
    for (size_t i = 0; i < N; i++) {
        src_re[i] = ir[plan->perm[i]];
        src_im[i] = ii[plan->perm[i]];
    }
    printf("  Permutation applied OK\n");
    fflush(stdout);

    for (size_t s = 0; s < plan->nstages; s++) {
        const vfft_stage *st = &plan->stages[s];
        size_t R = st->radix, K = st->K;
        size_t n_outer = N / (R * K);
        int is_last = (s == plan->nstages - 1);

        printf("  Stage %zu: R=%zu K=%zu n_outer=%zu is_last=%d\n", s, R, K, n_outer, is_last);
        printf("    tw_fwd=%p tw_re=%p walk=%p\n",
               (void*)st->tw_fwd, (void*)st->tw_re, (void*)st->walk);
        fflush(stdout);

        if (is_last) {
            dst_re = vr;
            dst_im = vi;
        }

        for (size_t g = 0; g < n_outer; g++) {
            size_t off = g * R * K;
            if (K > 1 && st->tw_re && st->tw_fwd) {
                if (g == 0) { printf("    Calling tw_fwd (fused) g=0...\n"); fflush(stdout); }
                st->tw_fwd(src_re + off, src_im + off,
                           dst_re + off, dst_im + off,
                           st->tw_re, st->tw_im, K);
                if (g == 0) { printf("    tw_fwd g=0 OK\n"); fflush(stdout); }
            } else if (K > 1 && st->tw_re) {
                if (g == 0) { printf("    Calling notw+twiddle g=0...\n"); fflush(stdout); }
                vfft_apply_twiddles_dispatch(
                    src_re + off, src_im + off,
                    st->tw_re, st->tw_im, R, K, 0);
                st->fwd(src_re + off, src_im + off,
                        dst_re + off, dst_im + off, K);
                if (g == 0) { printf("    notw+twiddle g=0 OK\n"); fflush(stdout); }
            } else {
                if (g == 0) { printf("    Calling notw g=0...\n"); fflush(stdout); }
                st->fwd(src_re + off, src_im + off,
                        dst_re + off, dst_im + off, K);
                if (g == 0) { printf("    notw g=0 OK\n"); fflush(stdout); }
            }
        }
        printf("  Stage %zu complete\n", s);
        fflush(stdout);

        if (!is_last) {
            double *t;
            t = src_re; src_re = dst_re; dst_re = t;
            t = src_im; src_im = dst_im; dst_im = t;
        }
    }
    printf("Forward OK, vr[0]=%.6f, vi[0]=%.6f\n", vr[0], vi[0]);
    fflush(stdout);

    /* Execute backward - manual DIF stage-by-stage */
    double *rr = (double *)vfft_aligned_alloc(64, N * 8);
    double *ri_buf = (double *)vfft_aligned_alloc(64, N * 8);
    printf("Executing backward (DIF, manual)...\n");
    fflush(stdout);

    /* Check all bwd codelets */
    int have_all_bwd = 1;
    for (size_t s = 0; s < plan->nstages; s++) {
        printf("  stage %zu: bwd=%p tw_dif_bwd=%p\n",
               s, (void*)plan->stages[s].bwd, (void*)plan->stages[s].tw_dif_bwd);
        if (!plan->stages[s].bwd) have_all_bwd = 0;
    }
    printf("  have_all_bwd=%d\n", have_all_bwd);
    fflush(stdout);

    /* DIF backward: natural input */
    src_re = plan->buf_a_re; src_im = plan->buf_a_im;
    dst_re = plan->buf_b_re; dst_im = plan->buf_b_im;
    memcpy(src_re, vr, N * sizeof(double));
    memcpy(src_im, vi, N * sizeof(double));

    /* Process outer to inner */
    for (int s = (int)plan->nstages - 1; s >= 0; s--) {
        const vfft_stage *st = &plan->stages[s];
        size_t R = st->radix, K = st->K;
        size_t n_outer = N / (R * K);

        printf("  DIF Stage %d: R=%zu K=%zu n_outer=%zu\n", s, R, K, n_outer);
        printf("    bwd=%p tw_dif_bwd=%p tw_re=%p\n",
               (void*)st->bwd, (void*)st->tw_dif_bwd, (void*)st->tw_re);
        fflush(stdout);

        for (size_t g = 0; g < n_outer; g++) {
            size_t off = g * R * K;
            if (K > 1 && st->tw_re && st->tw_dif_bwd) {
                if (g == 0) { printf("    DIF fused tw_dif_bwd g=0...\n"); fflush(stdout); }
                st->tw_dif_bwd(src_re + off, src_im + off,
                               dst_re + off, dst_im + off,
                               st->tw_re, st->tw_im, K);
                if (g == 0) { printf("    DIF fused g=0 OK\n"); fflush(stdout); }
            } else if (K > 1 && st->tw_re) {
                if (g == 0) { printf("    DIF notw+tw g=0...\n"); fflush(stdout); }
                st->bwd(src_re + off, src_im + off,
                        dst_re + off, dst_im + off, K);
                vfft_apply_twiddles_dispatch(
                    dst_re + off, dst_im + off,
                    st->tw_re, st->tw_im, R, K, 1);
                if (g == 0) { printf("    DIF notw+tw g=0 OK\n"); fflush(stdout); }
            } else {
                if (g == 0) { printf("    DIF notw g=0...\n"); fflush(stdout); }
                st->bwd(src_re + off, src_im + off,
                        dst_re + off, dst_im + off, K);
                if (g == 0) { printf("    DIF notw g=0 OK\n"); fflush(stdout); }
            }
        }
        printf("  DIF Stage %d complete\n", s);
        fflush(stdout);

        double *t;
        t = src_re; src_re = dst_re; dst_re = t;
        t = src_im; src_im = dst_im; dst_im = t;
    }

    /* Apply inverse permutation */
    for (size_t i = 0; i < N; i++) {
        rr[i] = src_re[plan->inv_perm[i]];
        ri_buf[i] = src_im[plan->inv_perm[i]];
    }
    printf("Backward (manual split) OK\n");
    fflush(stdout);

    /* Now test the IL DIF bwd codelet directly */
    printf("\nTesting IL DIF bwd codelet for R=8 K=625...\n");
    fflush(stdout);
    {
        size_t K = 625, R = 8;
        size_t stage_N = R * K; /* 5000 */
        double *il_in = (double *)vfft_aligned_alloc(64, 2 * stage_N * sizeof(double));
        double *il_out = (double *)vfft_aligned_alloc(64, 2 * stage_N * sizeof(double));
        /* Fill with test data */
        for (size_t i = 0; i < 2 * stage_N; i++) {
            il_in[i] = (double)(i % 37) / 37.0;
            il_out[i] = 0.0;
        }
        printf("  IL buffers allocated and filled\n");
        fflush(stdout);

        /* Get the DIF bwd IL function pointer from plan */
        vfft_tw_il_codelet_fn il_fn = plan->stages[2].tw_dif_bwd_il;
        printf("  tw_dif_bwd_il = %p\n", (void*)il_fn);
        fflush(stdout);

        if (il_fn) {
            printf("  Calling IL DIF bwd codelet with K=%zu...\n", K);
            fflush(stdout);
            il_fn(il_in, il_out, plan->stages[2].tw_re, plan->stages[2].tw_im, K);
            printf("  IL DIF bwd codelet OK\n");
            fflush(stdout);
        } else {
            printf("  IL DIF bwd codelet is NULL\n");
        }
        vfft_aligned_free(il_in);
        vfft_aligned_free(il_out);
    }

    /* Now test actual vfft_execute_bwd */
    printf("\nTesting vfft_execute_bwd...\n");
    fflush(stdout);
    {
        double *rr2 = (double *)vfft_aligned_alloc(64, N * 8);
        double *ri2 = (double *)vfft_aligned_alloc(64, N * 8);
        vfft_execute_bwd(plan, vr, vi, rr2, ri2);
        printf("vfft_execute_bwd OK\n");
        vfft_aligned_free(rr2);
        vfft_aligned_free(ri2);
    }
    fflush(stdout);

    /* Check roundtrip */
    double max_err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fabs(ir[i] - rr[i]/N) + fabs(ii[i] - ri_buf[i]/N);
        if (e > max_err) max_err = e;
    }
    printf("Roundtrip max error: %.3e\n", max_err);

    vfft_plan_destroy(plan);
    vfft_aligned_free(ir); vfft_aligned_free(ii);
    vfft_aligned_free(vr); vfft_aligned_free(vi);
    vfft_aligned_free(rr); vfft_aligned_free(ri_buf);

    printf("=== DONE ===\n");
    return 0;
}
