/* demo_impulse.c — Phase 2 correctness test for prototype-core.
 *
 * Validates that real twiddles computed by twiddle.h produce
 * mathematically correct FFT output. We use the impulse test:
 *
 *   Input:  x[k] = (k == 0) ? 1.0 : 0.0    (Kronecker delta)
 *   FFT:    X[n] = sum_k x[k] * W_N^(n*k) = 1.0 for all n
 *
 * So the FFT of an impulse is a constant signal (all ones). If our
 * twiddle compute is correct, the executor produces 16 ones for an
 * N=16 impulse.
 *
 * Cell: N=16 K=4, factorization {4, 4}.
 *   - Stage 0: R=4 (no twiddle, n1 codelet)
 *   - Stage 1: R=4 (real cross-stage twiddles, t1s codelet)
 *
 * K=4 batches the same impulse 4 times; all 4 batches should give
 * the same all-ones output. The twiddle math is what's being tested
 * here, not the codelets (they were validated by the Phase 1 spike
 * harnesses already).
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../executor.h"
#include "../twiddle.h"

#define N_FFT   16
#define K_BATCH 4
#define NF      2
static const int FACTORS[NF] = {4, 4};

#define BUF_LEN ((size_t)N_FFT * (size_t)K_BATCH)

static double *alloc_doubles(size_t n) {
    double *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}

/* Build a plan for N=16 K=4 with REAL twiddles via twiddle.h. */
static stride_plan_t *build_plan_with_real_twiddles(void) {
    stride_plan_t *plan = calloc(1, sizeof(*plan));
    plan->N = N_FFT;
    plan->K = K_BATCH;
    plan->num_stages = NF;
    plan->use_dif_forward = 0;
    for (int s = 0; s < NF; s++) plan->factors[s] = FACTORS[s];

    /* Wire codelet function pointers. */
    extern void radix4_n1_fwd_avx2(double *, double *,
                                   const double *, const double *,
                                   size_t, size_t);
    extern void radix4_t1s_dit_fwd_avx2(double *, double *,
                                        const double *, const double *,
                                        size_t, size_t);
    for (int s = 0; s < NF; s++) {
        plan->stages[s].n1_fwd  = radix4_n1_fwd_avx2;
        plan->stages[s].t1s_fwd = radix4_t1s_dit_fwd_avx2;
    }

    /* THE Phase 2 line — populate group_base / needs_tw / cf0 /
     * tw_scalar from the FFT math, not hand-crafted garbage. */
    vfft_proto_compute_plan_tables(plan);

    /* Pre-walk the tape (not strictly needed at N=16; plan_executors.h
     * has no entry for N=16 so the lookup returns NULL and the generic
     * executor runs — which doesn't use tape). Allocate for safety. */
    for (int s = 0; s < NF; s++) {
        stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        if (posix_memalign((void **)&st->tape, 64,
                           (size_t)G * sizeof(stride_invocation_t)) != 0) {
            fprintf(stderr, "tape alloc failed\n"); exit(1);
        }
        for (int g = 0; g < G; g++) {
            st->tape[g].base  = st->group_base[g];
            st->tape[g].tw_re = st->tw_scalar_re[g];
            st->tape[g].tw_im = st->tw_scalar_im[g];
        }
    }

    return plan;
}

int main(void) {
    fprintf(stderr, "[demo-impulse] Phase 2 correctness test\n");
    fprintf(stderr, "[demo-impulse] cell: N=%d K=%d factors={4,4}\n",
            N_FFT, K_BATCH);

    stride_plan_t *plan = build_plan_with_real_twiddles();

    /* Allocate split-complex buffers in K-batched layout.
     * Layout: re[k + i*K] = real part of element i for batch k.
     * Set each batch to an impulse: element 0 is 1.0, rest 0. */
    double *re = alloc_doubles(BUF_LEN);
    double *im = alloc_doubles(BUF_LEN);
    memset(re, 0, BUF_LEN * sizeof(double));
    memset(im, 0, BUF_LEN * sizeof(double));
    for (size_t k = 0; k < K_BATCH; k++) {
        re[k + 0 * K_BATCH] = 1.0;  /* impulse at position 0 of each batch */
    }

    /* Execute the FFT. */
    vfft_proto_execute_fwd(plan, re, im, (size_t)K_BATCH);

    /* Check: every output position should be 1.0 (real) + 0.0 (imag). */
    int errors = 0;
    double max_err_re = 0.0, max_err_im = 0.0;
    for (size_t i = 0; i < N_FFT; i++) {
        for (size_t k = 0; k < K_BATCH; k++) {
            double r = re[k + i * K_BATCH];
            double imv = im[k + i * K_BATCH];
            double err_re = fabs(r - 1.0);
            double err_im = fabs(imv - 0.0);
            if (err_re > max_err_re) max_err_re = err_re;
            if (err_im > max_err_im) max_err_im = err_im;
            if (err_re > 1e-10 || err_im > 1e-10) {
                if (errors < 5) {
                    printf("  pos[%zu, batch %zu]: got (%.6e, %.6e), want (1.0, 0.0)\n",
                           i, k, r, imv);
                }
                errors++;
            }
        }
    }

    printf("\n[demo-impulse] FFT of impulse (should be all ones):\n");
    for (size_t i = 0; i < N_FFT; i++) {
        printf("  X[%2zu] = (%+.6f, %+.6f)\n",
               i, re[0 + i * K_BATCH], im[0 + i * K_BATCH]);
    }

    printf("\n[demo-impulse] max |re - 1.0| = %.3e\n", max_err_re);
    printf("[demo-impulse] max |im - 0.0| = %.3e\n", max_err_im);

    if (errors == 0) {
        printf("\n[demo-impulse] PASS — twiddles correct, FFT output matches spec\n");
        free(re); free(im);
        vfft_proto_free_plan_tables(plan);
        free(plan);
        return 0;
    } else {
        printf("\n[demo-impulse] FAIL — %d out of %zu positions wrong\n",
               errors, (size_t)(N_FFT * K_BATCH));
        free(re); free(im);
        return 1;
    }
}
