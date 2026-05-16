/* spike_n131072_k4_flat.c — (B)+(A) validation on FLAT variant.
 *
 * Synthetic plan: same N=131072 K=4 8-stage structure as the T1S bench,
 * but with all-FLAT inner-stage variants. Wisdom doesn't pick this exact
 * configuration (T1S beats FLAT at K=4), but exercising the FLAT codepath
 * at maximum invocation count (~256K codelet calls per FFT) is the
 * cleanest test of whether (B)+(A) helps the FLAT path the way it helps
 * the T1S path.
 *
 * FLAT path's wrapper has MORE per-group work than T1S — it stages
 * (R-1) scalar twiddles into a tw_buf via _stride_broadcast_2 before
 * calling the t1 codelet. So if (B)+(A) recovers wrapper share, it
 * should show ≥ the 5% we saw on T1S.
 *
 * Comparison:
 *   - baseline : indirect call + scattered loads + production-style
 *                K-blocked broadcast staging
 *   - spike    : direct call + tape walk + inline broadcast staging
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "plan_executors.h"

/* Forward-declare the SYNTHETIC executor by exact name. plan_executors.h's
 * lookup function disambiguates by (N, K, factors, dif) only — for the
 * T1S vs FLAT collision at N=131072 K=4 we have to call the FLAT one
 * directly. The lookup mismatch is a known limitation of the spike. */
extern void exec_n131072_k4_44448444_v00000000_dit_fwd_avx2(
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
            extern void radix4_t1_dit_fwd_avx2(double *, double *,
                                               const double *, const double *,
                                               size_t, size_t);
            st->n1_fwd  = radix4_n1_fwd_avx2;
            /* Baseline uses t1 (FLAT) variant codelet, not t1s. */
            st->t1s_fwd = radix4_t1_dit_fwd_avx2;
        } else if (R == 8) {
            extern void radix8_n1_fwd_avx2(double *, double *,
                                           const double *, const double *,
                                           size_t, size_t);
            extern void radix8_t1_dit_fwd_avx2(double *, double *,
                                               const double *, const double *,
                                               size_t, size_t);
            st->n1_fwd  = radix8_n1_fwd_avx2;
            st->t1s_fwd = radix8_t1_dit_fwd_avx2;
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

/* Baseline FLAT executor — mirrors production's else-if branch at
 * executor.h:455-491. Indirect call through st->t1s_fwd (which is
 * actually the t1 codelet — pointer name is just a leftover from the
 * struct definition shared with the T1S bench). K-blocked tw_buf
 * staging happens INSIDE the per-group loop, paid 256K times per FFT. */
static void baseline_flat_exec(const stride_plan_t *plan, double *re, double *im,
                                size_t slice_K, int start_stage)
{
    for (int s = start_stage; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        const int Rm1 = st->radix - 1;
        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];
            if (s == 0) {
                /* Stage 0 always uses n1 — same in baseline and spike. */
                st->n1_fwd(base_re, base_im, NULL, NULL, st->stride, slice_K);
                continue;
            }
            /* FLAT path: K-blocked staging. tw_buf sized for max R=8. */
            double tw_buf_re[7 * VFFT_PROTO_TW_BLOCK_K];
            double tw_buf_im[7 * VFFT_PROTO_TW_BLOCK_K];
            const double *stw_r = st->tw_scalar_re[g];
            const double *stw_i = st->tw_scalar_im[g];
            for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                size_t this_K = (slice_K - kb < VFFT_PROTO_TW_BLOCK_K)
                                ? (slice_K - kb) : VFFT_PROTO_TW_BLOCK_K;
                for (int j = 0; j < Rm1; j++) {
                    _stride_broadcast_2(tw_buf_re + (size_t)j * this_K,
                                        tw_buf_im + (size_t)j * this_K,
                                        this_K, stw_r[j], stw_i[j]);
                }
                /* indirect call through stage's codelet pointer */
                st->t1s_fwd(base_re + kb, base_im + kb,
                            tw_buf_re, tw_buf_im,
                            st->stride, this_K);
            }
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
    fprintf(stderr, "[spike-flat] building plan: N=%d K=%d, 8-stage all-FLAT inner\n",
            N_FFT, K_BATCH);
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
        exec_n131072_k4_44448444_v00000000_dit_fwd_avx2(plan, re_a, im_a,
                                                         (size_t)K_BATCH, (size_t)K_BATCH, 0);
        baseline_flat_exec(plan, re_b, im_b, (size_t)K_BATCH, 0);
    }

    for (int r = 0; r < RUNS; r++) {
        double t0 = now_sec();
        for (int i = 0; i < REPS_PER_RUN; i++)
            exec_n131072_k4_44448444_v00000000_dit_fwd_avx2(plan, re_a, im_a,
                                                             (size_t)K_BATCH, (size_t)K_BATCH, 0);
        double t1 = now_sec();
        for (int i = 0; i < REPS_PER_RUN; i++)
            baseline_flat_exec(plan, re_b, im_b, (size_t)K_BATCH, 0);
        double t2 = now_sec();
        t_spike[r] = (t1 - t0) / REPS_PER_RUN;
        t_base[r]  = (t2 - t1) / REPS_PER_RUN;
    }

    double med_spike = median(t_spike, RUNS);
    double med_base  = median(t_base,  RUNS);

    printf("N=%d K=%d 8-stage all-FLAT inner-stage plan, %d reps/run x %d runs\n",
           N_FFT, K_BATCH, REPS_PER_RUN, RUNS);
    printf("  baseline (indirect, K-block stg): %12.0f ns/FFT\n", med_base  * 1e9);
    printf("  spike    (direct, tape + stg):    %12.0f ns/FFT\n", med_spike * 1e9);
    printf("  speedup:                          %12.3fx\n", med_base / med_spike);
    printf("  wrapper share recovered:          %.1f%%\n",
           (1.0 - med_spike / med_base) * 100.0);

    printf("\nrun  spike(ns)     base(ns)\n");
    for (int r = 0; r < RUNS; r++)
        printf(" %2d  %10.0f  %10.0f\n", r, t_spike[r]*1e9, t_base[r]*1e9);

    free(re_a); free(im_a); free(re_b); free(im_b);
    return 0;
}
