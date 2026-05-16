/* spike_n131072_k4.c — Plan B validation bench.
 *
 * Standalone benchmark for the plan-shaped specialized executor vs a
 * generic-style (function-pointer-indirected) baseline. Same plan, same
 * memory access pattern, same codelet bodies executing — the ONLY
 * difference is call style and per-group branch tree.
 *
 * The architectural claim being tested:
 *   Plan-shaped emission turns the executor's per-group indirect call
 *   + 4-branch variant tree into a direct call + one branch, recovering
 *   most of the 21% _stride_execute_fwd_slice_from share documented in
 *   docs/dev/vtune_n131072_k4_vfft_vs_mkl.md.
 *
 * NOT TESTED HERE:
 *   - Correctness — twiddles are filled with arbitrary values. The
 *     codelets execute the same instructions regardless of twiddle data,
 *     so wall-time delta is meaningful even with garbage twiddles.
 *     Correctness validation is a separate workstream against production
 *     planner output.
 *
 * Compile (Linux, AVX2):
 *   gcc-15 -O2 -mavx2 -mfma -Wno-incompatible-pointer-types \
 *     -I src/prototype/generated \
 *     src/prototype/bench/spike_n131072_k4.c \
 *     src/prototype/codelets/avx2/small_pow2/r4_n1_fwd.c \
 *     src/prototype/codelets/avx2/small_pow2/r4_t1s_dit_fwd.c \
 *     src/prototype/codelets/avx2/small_pow2/r8_n1_fwd.c \
 *     src/prototype/codelets/avx2/small_pow2/r8_t1s_dit_fwd.c \
 *     -o build_tuned/spike_n131072_k4 -lm
 */
/* posix_memalign + clock_gettime + _POSIX_TIMERS need these macros
 * defined BEFORE any system header is parsed. */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE  1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "plan_executors.h"

/* ── Plan setup ───────────────────────────────────────────────────────
 *
 * N=131072 K=4 factorization 4×4×4×4×8×4×4×4, all-T1S inner stages.
 *
 * Stride per stage (K-batched DIT layout):
 *   stride[s] = K * product(R_{s+1}..R_{S-1})
 *
 * Stage:  0       1      2     3    4   5   6  7
 * R:      4       4      4     4    8   4   4  4
 * stride: 131072  32768  8192  2048 256 64  16 4
 * groups: 32768   32768  32768 32768 16384 32768 32768 32768
 *
 * group_base[g]: for the timing spike we use g*stride mod N*K so the
 * codelet reads/writes inside the buffer. Real production plans use
 * a more complex bit-reversal-like indexing; for *timing* the volume
 * of memory traffic and number of butterflies is what matters, both of
 * which are identical between the two executors.
 */

#define N_FFT 131072
#define K_BATCH 4
#define NF 8
static const int FACTORS[NF] = {4, 4, 4, 4, 8, 4, 4, 4};

/* Buffer size: each of re/im holds N*K doubles. */
#define BUF_LEN ((size_t)N_FFT * (size_t)K_BATCH)

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

/* Build the plan with realistic strides and per-group bookkeeping.
 * Allocates twiddle scalars (filled with arbitrary values), group_base
 * arrays, and codelet function pointers for the baseline executor. */
static stride_plan_t *build_plan(void)
{
    stride_plan_t *plan = calloc(1, sizeof(*plan));
    plan->N = N_FFT;
    plan->K = K_BATCH;
    plan->num_stages = NF;
    plan->use_dif_forward = 0;
    for (int s = 0; s < NF; s++) plan->factors[s] = FACTORS[s];

    /* Compute strides: stride[s] = K * product(R_{s+1}..R_{S-1}) */
    size_t strides[NF];
    size_t prod = 1;
    for (int s = NF - 1; s >= 0; s--) {
        strides[s] = (size_t)K_BATCH * prod;
        prod *= (size_t)FACTORS[s];
    }

    /* Set per-stage data. */
    for (int s = 0; s < NF; s++) {
        const int R = FACTORS[s];
        stride_stage_t *st = &plan->stages[s];
        st->radix      = R;
        st->stride     = strides[s];
        st->num_groups = N_FFT / R;

        /* Codelet function pointers for baseline. Wire by radix. */
        if (R == 4) {
            extern void radix4_n1_fwd_avx2(double *, double *,
                                           const double *, const double *,
                                           size_t, size_t);
            extern void radix4_t1s_dit_fwd_avx2(double *, double *,
                                                const double *, const double *,
                                                size_t, size_t);
            st->n1_fwd  = radix4_n1_fwd_avx2;
            st->t1s_fwd = radix4_t1s_dit_fwd_avx2;
        } else if (R == 8) {
            extern void radix8_n1_fwd_avx2(double *, double *,
                                           const double *, const double *,
                                           size_t, size_t);
            extern void radix8_t1s_dit_fwd_avx2(double *, double *,
                                                const double *, const double *,
                                                size_t, size_t);
            st->n1_fwd  = radix8_n1_fwd_avx2;
            st->t1s_fwd = radix8_t1s_dit_fwd_avx2;
        }

        /* Per-group arrays. */
        const int G = st->num_groups;
        st->group_base   = alloc_size_t((size_t)G);
        st->needs_tw     = malloc((size_t)G * sizeof(int));
        st->cf0_re       = alloc_doubles((size_t)G);
        st->cf0_im       = alloc_doubles((size_t)G);
        st->tw_scalar_re = malloc((size_t)G * sizeof(double *));
        st->tw_scalar_im = malloc((size_t)G * sizeof(double *));

        /* group_base[g]: stride out groups so each leg lands inside the
         * buffer. With stride = K*prod and R legs per group, the group's
         * footprint is R*stride. Spacing groups by K (the K-batch unit)
         * keeps each leg's last write at base + (R-1)*stride + K, which
         * stays within the buffer for all stages (verified below). */
        for (int g = 0; g < G; g++) {
            size_t base = (size_t)g * K_BATCH;
            /* Wrap modulo a safe ceiling so we never run past buffer end. */
            size_t max_used = (size_t)(R - 1) * strides[s] + (size_t)K_BATCH;
            size_t headroom = (max_used < BUF_LEN) ? (BUF_LEN - max_used) : 1;
            base = base % headroom;
            st->group_base[g] = base;
            /* All groups take the t1s path with trivial cf0 — that's the
             * fast path we want to measure. Stage 0 ignores needs_tw (it
             * always uses n1 regardless). */
            st->needs_tw[g] = 1;
            st->cf0_re[g]   = 1.0;
            st->cf0_im[g]   = 0.0;
        }

        /* Per-group scalar twiddles: each group has (R-1) doubles for
         * the leg's scalar twiddle pair (re, im). Allocate one pool;
         * point each group's slot at the same shared filler — values
         * don't matter for timing. */
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

        /* Pre-walk the tape (Plan B+A): pack per-group runtime values
         * into a flat sequential array. The spike executor walks this
         * directly, replacing 4-5 scattered loads per group with one
         * sequential 24-byte struct load that HW prefetcher handles. */
        if (posix_memalign((void **)&st->tape, 64,
                           (size_t)G * sizeof(stride_invocation_t)) != 0) {
            fprintf(stderr, "tape alloc failed for stage %d (%d groups)\n", s, G);
            exit(1);
        }
        for (int g = 0; g < G; g++) {
            st->tape[g].base  = st->group_base[g];
            st->tape[g].tw_re = st->tw_scalar_re[g];
            st->tape[g].tw_im = st->tw_scalar_im[g];
        }
    }
    return plan;
}

/* ── Baseline executor (function-pointer indirected) ─────────────────
 *
 * Models the production generic executor's hot loop: at each (stage,
 * group), branch on needs_tw, then call the codelet through the per-
 * stage function pointer. Compiler emits `call *reg` (indirect) for the
 * codelet call sites; no inlining possible. */
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
                if (cfr != 1.0 || cfi != 0.0) {
                    _stride_cmul_scalar_inplace(base_re, base_im, slice_K, cfr, cfi);
                }
                st->t1s_fwd(base_re, base_im,
                            st->tw_scalar_re[g], st->tw_scalar_im[g],
                            st->stride, slice_K);
            }
        }
    }
}

/* Hide the spike executor function pointer behind volatile so the
 * compiler can't fold the two invocations into one. */
static vfft_proto_exec_fn get_spike_exec(const stride_plan_t *plan) {
    return vfft_proto_lookup_fwd_avx2(plan);
}

/* ── Timing ──────────────────────────────────────────────────────── */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double median(double *xs, int n) {
    /* simple insertion sort */
    for (int i = 1; i < n; i++) {
        double v = xs[i]; int j = i - 1;
        while (j >= 0 && xs[j] > v) { xs[j+1] = xs[j]; j--; }
        xs[j+1] = v;
    }
    return xs[n/2];
}

int main(void) {
    fprintf(stderr, "[spike] building plan for N=%d K=%d (8-stage 4×4×4×4×8×4×4×4)\n",
            N_FFT, K_BATCH);
    stride_plan_t *plan = build_plan();
    vfft_proto_exec_fn spike_fn = get_spike_exec(plan);
    if (!spike_fn) {
        fprintf(stderr, "[spike] FATAL: lookup returned NULL — plan tuple mismatch\n");
        return 1;
    }
    fprintf(stderr, "[spike] specialized executor resolved: %p\n", (void *)spike_fn);

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

    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        spike_fn(plan, re_a, im_a, (size_t)K_BATCH, (size_t)K_BATCH, 0);
        baseline_exec(plan, re_b, im_b, (size_t)K_BATCH, 0);
    }

    /* Timed runs — interleave spike and baseline to fairly share thermal /
     * frequency drift. */
    for (int r = 0; r < RUNS; r++) {
        double t0 = now_sec();
        for (int i = 0; i < REPS_PER_RUN; i++)
            spike_fn(plan, re_a, im_a, (size_t)K_BATCH, (size_t)K_BATCH, 0);
        double t1 = now_sec();
        for (int i = 0; i < REPS_PER_RUN; i++)
            baseline_exec(plan, re_b, im_b, (size_t)K_BATCH, 0);
        double t2 = now_sec();
        t_spike[r] = (t1 - t0) / REPS_PER_RUN;
        t_base[r]  = (t2 - t1) / REPS_PER_RUN;
    }

    double med_spike = median(t_spike, RUNS);
    double med_base  = median(t_base,  RUNS);
    double speedup   = med_base / med_spike;
    double wrapper_share_recovered = (1.0 - med_spike / med_base) * 100.0;

    printf("N=%d K=%d 8-stage plan, %d reps/run x %d runs (median ns/FFT)\n",
           N_FFT, K_BATCH, REPS_PER_RUN, RUNS);
    printf("  baseline (indirect): %12.0f ns/FFT\n", med_base  * 1e9);
    printf("  spike    (direct):   %12.0f ns/FFT\n", med_spike * 1e9);
    printf("  speedup:             %12.3fx\n", speedup);
    printf("  wrapper share recovered: %.1f%%\n", wrapper_share_recovered);

    /* Show distribution */
    printf("\nrun  spike(ns)     base(ns)\n");
    for (int r = 0; r < RUNS; r++)
        printf(" %2d  %10.0f  %10.0f\n", r, t_spike[r]*1e9, t_base[r]*1e9);

    free(re_a); free(im_a); free(re_b); free(im_b);
    return 0;
}
