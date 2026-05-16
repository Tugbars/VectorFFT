/* spike_n1024_k128.c — Plan B+A validation on N=1024 K=128 + Plan C comparison.
 *
 * Three executors timed on the same data:
 *   - Baseline : indirect calls, scattered loads (production-style)
 *   - Spike    : direct calls, tape walk (Plan B+A)
 *   - Plan C   : single call to monolithic radix1024_n1_fwd_avx2
 *                (does the entire batched FFT in one codelet — no inter-
 *                stage boundary, zero executor wrapper work)
 *
 * As with spike_n131072_k4.c, this is a TIMING-only bench. Twiddles are
 * filled with arbitrary values; correctness is a separate workstream.
 *
 * For Plan C: the existing radix1024_n1_fwd_avx2 codelet from the
 * xl_pow2 family is what an MKL-style monolithic kernel looks like for
 * N=1024. Wisdom doesn't pick it (prefers the 5-stage decomposition);
 * the bench measures *why*.
 *
 * Compile:
 *   gcc-15 -O2 -mavx2 -mfma -Wno-incompatible-pointer-types \
 *     -I src/prototype/generated \
 *     src/prototype/bench/spike_n1024_k128.c \
 *     src/prototype/codelets/avx2/small_pow2/r4_n1_fwd.c \
 *     src/prototype/codelets/avx2/small_pow2/r4_t1s_dit_fwd.c \
 *     src/prototype/codelets/avx2/xl_pow2/r1024_n1_fwd.c \
 *     -o build_tuned/spike_n1024_k128 -lm
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "plan_executors.h"

/* Plan C: monolithic codelet. Declared here since we don't include it via
 * the auto-emitted plan_executors.h (which only references codelets used
 * by the specialized executors, not by Plan C). */
extern void radix1024_n1_fwd_avx2(double *rio_re, double *rio_im,
                                   const double *tw_re, const double *tw_im,
                                   size_t ios, size_t me);

/* ── Plan setup: N=1024 K=128, factorization 4×4×4×4×4 (5 stages) ──
 *
 * Strides:
 *   stride[s] = K * product(R_{s+1}..R_{S-1})
 *   stride[0] = 128 * (4*4*4*4) = 32768
 *   stride[1] = 128 * (4*4*4)   = 8192
 *   stride[2] = 128 * (4*4)     = 2048
 *   stride[3] = 128 * 4         = 512
 *   stride[4] = 128 * 1         = 128
 *
 * num_groups: N/R = 256 for every stage.
 */

#define N_FFT   1024
#define K_BATCH 128
#define NF      5
static const int FACTORS[NF] = {4, 4, 4, 4, 4};

#define BUF_LEN ((size_t)N_FFT * (size_t)K_BATCH)  /* 131072 doubles per re/im */

static double *alloc_doubles(size_t n) {
    double *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed for %zu doubles\n", n);
        exit(1);
    }
    return p;
}

static size_t *alloc_size_t(size_t n) {
    size_t *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(size_t)) != 0) {
        fprintf(stderr, "alloc failed for %zu size_t\n", n);
        exit(1);
    }
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
            extern void radix4_t1s_dit_fwd_avx2(double *, double *,
                                                const double *, const double *,
                                                size_t, size_t);
            st->n1_fwd  = radix4_n1_fwd_avx2;
            st->t1s_fwd = radix4_t1s_dit_fwd_avx2;
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

        /* Pre-walked tape for (B)+(A) spike. */
        if (posix_memalign((void **)&st->tape, 64,
                           (size_t)G * sizeof(stride_invocation_t)) != 0) {
            fprintf(stderr, "tape alloc failed for stage %d\n", s); exit(1);
        }
        for (int g = 0; g < G; g++) {
            st->tape[g].base  = st->group_base[g];
            st->tape[g].tw_re = st->tw_scalar_re[g];
            st->tape[g].tw_im = st->tw_scalar_im[g];
        }
    }
    return plan;
}

static void baseline_exec(const stride_plan_t *plan, double *re, double *im,
                          size_t slice_K, int start_stage)
{
    for (int s = start_stage; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];
            if (!st->needs_tw[g]) {
                st->n1_fwd(base_re, base_im, NULL, NULL, st->stride, slice_K);
            } else {
                double cfr = st->cf0_re[g];
                double cfi = st->cf0_im[g];
                if (cfr != 1.0 || cfi != 0.0)
                    _stride_cmul_scalar_inplace(base_re, base_im, slice_K, cfr, cfi);
                st->t1s_fwd(base_re, base_im,
                            st->tw_scalar_re[g], st->tw_scalar_im[g],
                            st->stride, slice_K);
            }
        }
    }
}

/* Plan C: single call to the monolithic N=1024 codelet. Handles the
 * full batched FFT — no inter-stage executor wrapper at all. */
static inline void plan_c_exec(double *re, double *im, size_t K)
{
    /* For a K-batched layout, ios = K (stride between FFT-axis elements),
     * me = K (process K batch elements per inner SIMD iteration of the
     * codelet's k-loop). */
    radix1024_n1_fwd_avx2(re, im, NULL, NULL, (size_t)K, (size_t)K);
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
    fprintf(stderr, "[spike] N=%d K=%d, 5-stage 4×4×4×4×4 plan + Plan C R=1024 mono\n",
            N_FFT, K_BATCH);
    stride_plan_t *plan = build_plan();
    vfft_proto_exec_fn spike_fn = vfft_proto_lookup_fwd_avx2(plan);
    if (!spike_fn) {
        fprintf(stderr, "[spike] FATAL: lookup returned NULL\n");
        return 1;
    }
    fprintf(stderr, "[spike] specialized executor resolved: %p\n", (void *)spike_fn);

    double *re_a = alloc_doubles(BUF_LEN);  /* baseline */
    double *im_a = alloc_doubles(BUF_LEN);
    double *re_b = alloc_doubles(BUF_LEN);  /* spike    */
    double *im_b = alloc_doubles(BUF_LEN);
    double *re_c = alloc_doubles(BUF_LEN);  /* Plan C   */
    double *im_c = alloc_doubles(BUF_LEN);
    for (size_t i = 0; i < BUF_LEN; i++) {
        re_a[i] = re_b[i] = re_c[i] = 1.0 + 1e-9 * (double)i;
        im_a[i] = im_b[i] = im_c[i] = -1.0 + 1e-9 * (double)i;
    }

    const int WARMUP = 10;
    const int RUNS   = 11;
    const int REPS_PER_RUN = 200;  /* N=1024 is small — need more reps for stable timing */
    double t_spike[16], t_base[16], t_planc[16];

    for (int w = 0; w < WARMUP; w++) {
        spike_fn(plan, re_a, im_a, (size_t)K_BATCH, (size_t)K_BATCH, 0);
        baseline_exec(plan, re_b, im_b, (size_t)K_BATCH, 0);
        plan_c_exec(re_c, im_c, (size_t)K_BATCH);
    }

    /* Interleave the three timed segments per run. */
    for (int r = 0; r < RUNS; r++) {
        double t0 = now_sec();
        for (int i = 0; i < REPS_PER_RUN; i++)
            spike_fn(plan, re_a, im_a, (size_t)K_BATCH, (size_t)K_BATCH, 0);
        double t1 = now_sec();
        for (int i = 0; i < REPS_PER_RUN; i++)
            baseline_exec(plan, re_b, im_b, (size_t)K_BATCH, 0);
        double t2 = now_sec();
        for (int i = 0; i < REPS_PER_RUN; i++)
            plan_c_exec(re_c, im_c, (size_t)K_BATCH);
        double t3 = now_sec();
        t_spike[r] = (t1 - t0) / REPS_PER_RUN;
        t_base[r]  = (t2 - t1) / REPS_PER_RUN;
        t_planc[r] = (t3 - t2) / REPS_PER_RUN;
    }

    double med_spike = median(t_spike, RUNS);
    double med_base  = median(t_base,  RUNS);
    double med_planc = median(t_planc, RUNS);

    printf("N=%d K=%d (5-stage 4×4×4×4×4 plan), %d reps/run x %d runs\n",
           N_FFT, K_BATCH, REPS_PER_RUN, RUNS);
    printf("                            median ns/FFT       speedup vs baseline\n");
    printf("  baseline (indirect)       %12.0f       1.000x\n", med_base  * 1e9);
    printf("  spike    (B+A direct)     %12.0f       %.3fx\n",
           med_spike * 1e9, med_base / med_spike);
    printf("  Plan C   (R=1024 mono)    %12.0f       %.3fx\n",
           med_planc * 1e9, med_base / med_planc);
    printf("\nspike  vs baseline:  %.1f%% improvement (Plan B+A wrapper recovery)\n",
           (1.0 - med_spike / med_base) * 100.0);
    printf("Plan C vs baseline:  %.1f%% (negative = Plan C is slower)\n",
           (1.0 - med_planc / med_base) * 100.0);
    printf("Plan C vs spike:     %.1f%% (positive = monolithic still wins after B+A)\n",
           (1.0 - med_planc / med_spike) * 100.0);

    printf("\nrun  spike(ns)     base(ns)     planc(ns)\n");
    for (int r = 0; r < RUNS; r++)
        printf(" %2d  %10.0f  %10.0f  %10.0f\n",
               r, t_spike[r]*1e9, t_base[r]*1e9, t_planc[r]*1e9);

    free(re_a); free(im_a); free(re_b); free(im_b); free(re_c); free(im_c);
    return 0;
}
