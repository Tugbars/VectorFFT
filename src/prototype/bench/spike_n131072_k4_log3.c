/* spike_n131072_k4_log3.c — (B)+(A) validation on LOG3 variant.
 *
 * Synthetic plan: same N=131072 K=4 8-stage shape as the T1S/FLAT
 * benches, but with all-LOG3 inner-stage variants. Wisdom doesn't
 * pick this exact configuration (LOG3 is rarely used for many-inner-
 * stage plans — it's mostly the innermost-stage choice in 3-stage
 * plans like R=13×R=13×R=13), but isolates the LOG3 codepath at
 * maximum invocation count.
 *
 * The LOG3 codepath in production applies cf0 to ALL R legs (not just
 * leg 0 like T1S), then calls the log3 variant codelet which internally
 * derives non-power-of-2 twiddles. In this spike cf0=(1.0,0.0) so the
 * cf branch is skipped; the measurement isolates the codelet call +
 * wrapper bookkeeping difference between baseline and spike.
 *
 * Comparison:
 *   - baseline : indirect call through st->t1s_fwd (wired to log3 codelet)
 *                + scattered per-group loads
 *   - spike    : direct call + tape walk
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "plan_executors.h"

/* Forward-declare the LOG3 synthetic executor by name (the lookup
 * disambiguates by (N,K,factors,dif) only, so there's a collision with
 * the T1S and FLAT entries at this shape; call directly). */
extern void exec_n131072_k4_44448444_v01111111_dit_fwd_avx2(
    const stride_plan_t *plan, double *re, double *im,
    size_t slice_K, size_t full_K, int start_stage);

#define N_FFT 131072
#define K_BATCH 4
#define NF 8
static const int FACTORS[NF] = {4, 4, 4, 4, 8, 4, 4, 4};

#define BUF_LEN ((size_t)N_FFT * (size_t)K_BATCH)

static double *alloc_doubles(size_t n) {
    double *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) exit(1);
    return p;
}
static size_t *alloc_size_t(size_t n) {
    size_t *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(size_t)) != 0) exit(1);
    return p;
}

static stride_plan_t *build_plan(void)
{
    stride_plan_t *plan = calloc(1, sizeof(*plan));
    plan->N = N_FFT;
    plan->K = K_BATCH;
    plan->num_stages = NF;
    plan->use_dif_forward = 0;
    for (int s = 0; s < NF; s++) plan->factors[s] = FACTORS[s];

    size_t strides[NF];
    size_t prod = 1;
    for (int s = NF - 1; s >= 0; s--) {
        strides[s] = (size_t)K_BATCH * prod;
        prod *= (size_t)FACTORS[s];
    }

    for (int s = 0; s < NF; s++) {
        const int R = FACTORS[s];
        stride_stage_t *st = &plan->stages[s];
        st->radix      = R;
        st->stride     = strides[s];
        st->num_groups = N_FFT / R;

        if (R == 4) {
            extern void radix4_n1_fwd_avx2(double *, double *,
                                           const double *, const double *,
                                           size_t, size_t);
            extern void radix4_t1_dit_log3_fwd_avx2(double *, double *,
                                                    const double *, const double *,
                                                    size_t, size_t);
            st->n1_fwd  = radix4_n1_fwd_avx2;
            /* Baseline calls the LOG3 codelet through this slot. */
            st->t1s_fwd = radix4_t1_dit_log3_fwd_avx2;
        } else if (R == 8) {
            extern void radix8_n1_fwd_avx2(double *, double *,
                                           const double *, const double *,
                                           size_t, size_t);
            extern void radix8_t1_dit_log3_fwd_avx2(double *, double *,
                                                    const double *, const double *,
                                                    size_t, size_t);
            st->n1_fwd  = radix8_n1_fwd_avx2;
            st->t1s_fwd = radix8_t1_dit_log3_fwd_avx2;
        }

        const int G = st->num_groups;
        st->group_base   = alloc_size_t((size_t)G);
        st->needs_tw     = malloc((size_t)G * sizeof(int));
        st->cf0_re       = alloc_doubles((size_t)G);
        st->cf0_im       = alloc_doubles((size_t)G);
        st->tw_scalar_re = malloc((size_t)G * sizeof(double *));
        st->tw_scalar_im = malloc((size_t)G * sizeof(double *));

        for (int g = 0; g < G; g++) {
            size_t base = (size_t)g * K_BATCH;
            size_t max_used = (size_t)(R - 1) * strides[s] + (size_t)K_BATCH;
            size_t headroom = (max_used < BUF_LEN) ? (BUF_LEN - max_used) : 1;
            base = base % headroom;
            st->group_base[g] = base;
            st->needs_tw[g] = 1;
            st->cf0_re[g]   = 1.0;
            st->cf0_im[g]   = 0.0;
        }

        /* LOG3 codelets read tw_re[j*me + k] at sparse j slots — same
         * vector-load layout as t1 flat. Pool is generous enough that
         * the codelet's reads stay in bounds for this radix/me. */
        double *pool_re = alloc_doubles((size_t)(R - 1) * 64);
        double *pool_im = alloc_doubles((size_t)(R - 1) * 64);
        for (int j = 0; j < (R - 1) * 64; j++) {
            pool_re[j] = 0.7071 + 0.001 * (j & 7);
            pool_im[j] = 0.7071 - 0.001 * (j & 7);
        }
        for (int g = 0; g < G; g++) {
            st->tw_scalar_re[g] = pool_re;
            st->tw_scalar_im[g] = pool_im;
        }

        if (posix_memalign((void **)&st->tape, 64,
                           (size_t)G * sizeof(stride_invocation_t)) != 0) exit(1);
        for (int g = 0; g < G; g++) {
            st->tape[g].base  = st->group_base[g];
            st->tape[g].tw_re = st->tw_scalar_re[g];
            st->tape[g].tw_im = st->tw_scalar_im[g];
        }
    }
    return plan;
}

/* Baseline LOG3 executor — mirrors production's else-if branch at
 * executor.h:420-437. Indirect call through st->t1s_fwd (wired to the
 * log3 codelet). cf branch skipped at runtime since cf0 is trivial. */
static void baseline_log3_exec(const stride_plan_t *plan, double *re, double *im,
                                size_t slice_K, int start_stage)
{
    for (int s = start_stage; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];
            if (s == 0) {
                st->n1_fwd(base_re, base_im, NULL, NULL, st->stride, slice_K);
                continue;
            }
            double cfr = st->cf0_re[g];
            double cfi = st->cf0_im[g];
            if (cfr != 1.0 || cfi != 0.0) {
                /* LOG3 production path applies cf0 to ALL R legs */
                for (int j = 0; j < st->radix; j++) {
                    _stride_cmul_scalar_inplace(
                        base_re + (size_t)j * st->stride,
                        base_im + (size_t)j * st->stride,
                        slice_K, cfr, cfi);
                }
            }
            st->t1s_fwd(base_re, base_im,
                        st->tw_scalar_re[g], st->tw_scalar_im[g],
                        st->stride, slice_K);
        }
    }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static double median(double *xs, int n) {
    for (int i = 1; i < n; i++) {
        double v = xs[i]; int j = i - 1;
        while (j >= 0 && xs[j] > v) { xs[j+1] = xs[j]; j--; }
        xs[j+1] = v;
    }
    return xs[n/2];
}

int main(void) {
    fprintf(stderr, "[spike-log3] N=%d K=%d, 8-stage all-LOG3 inner\n", N_FFT, K_BATCH);
    stride_plan_t *plan = build_plan();

    double *re_a = alloc_doubles(BUF_LEN);
    double *im_a = alloc_doubles(BUF_LEN);
    double *re_b = alloc_doubles(BUF_LEN);
    double *im_b = alloc_doubles(BUF_LEN);
    for (size_t i = 0; i < BUF_LEN; i++) {
        re_a[i] = re_b[i] = 1.0 + 1e-9 * (double)i;
        im_a[i] = im_b[i] = -1.0 + 1e-9 * (double)i;
    }

    const int WARMUP = 5;
    const int RUNS   = 11;
    const int REPS_PER_RUN = 20;
    double t_spike[16], t_base[16];

    for (int w = 0; w < WARMUP; w++) {
        exec_n131072_k4_44448444_v01111111_dit_fwd_avx2(plan, re_a, im_a,
                                                         (size_t)K_BATCH, (size_t)K_BATCH, 0);
        baseline_log3_exec(plan, re_b, im_b, (size_t)K_BATCH, 0);
    }

    for (int r = 0; r < RUNS; r++) {
        double t0 = now_sec();
        for (int i = 0; i < REPS_PER_RUN; i++)
            exec_n131072_k4_44448444_v01111111_dit_fwd_avx2(plan, re_a, im_a,
                                                             (size_t)K_BATCH, (size_t)K_BATCH, 0);
        double t1 = now_sec();
        for (int i = 0; i < REPS_PER_RUN; i++)
            baseline_log3_exec(plan, re_b, im_b, (size_t)K_BATCH, 0);
        double t2 = now_sec();
        t_spike[r] = (t1 - t0) / REPS_PER_RUN;
        t_base[r]  = (t2 - t1) / REPS_PER_RUN;
    }

    double med_spike = median(t_spike, RUNS);
    double med_base  = median(t_base,  RUNS);

    printf("N=%d K=%d 8-stage all-LOG3 inner-stage plan, %d reps/run x %d runs\n",
           N_FFT, K_BATCH, REPS_PER_RUN, RUNS);
    printf("  baseline (indirect log3):  %12.0f ns/FFT\n", med_base  * 1e9);
    printf("  spike    (direct + tape):  %12.0f ns/FFT\n", med_spike * 1e9);
    printf("  speedup:                   %12.3fx\n", med_base / med_spike);
    printf("  wrapper share recovered:   %.1f%%\n",
           (1.0 - med_spike / med_base) * 100.0);

    printf("\nrun  spike(ns)     base(ns)\n");
    for (int r = 0; r < RUNS; r++)
        printf(" %2d  %10.0f  %10.0f\n", r, t_spike[r]*1e9, t_base[r]*1e9);

    free(re_a); free(im_a); free(re_b); free(im_b);
    return 0;
}
