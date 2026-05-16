/* demo_n1024_k128.c — Phase 1 prototype-core demonstration.
 *
 * Smallest possible exercise of prototype-core: build a hand-crafted
 * plan for N=1024 K=128 (matching the second spike entry in
 * plan_executors.h), run it through vfft_proto_execute_fwd, and
 * confirm:
 *
 *   1. The plan-shaped specialization fast path is selected (the
 *      lookup returns non-NULL because the plan matches one of the
 *      hardcoded wisdom entries).
 *   2. The codelets execute without crashing.
 *   3. Wall-time per execute is in the expected range (~100 µs on
 *      Raptor Lake, per the spike measurements).
 *
 * This is NOT a correctness test — twiddles are filled with arbitrary
 * values so the FFT output is meaningless. Phase 2 (twiddle compute)
 * adds real twiddles; until then, "doesn't crash" is the demo's bar.
 *
 * Build via cost_model-style script:
 *   bash demo/build_demo.sh
 * Run:
 *   build_tuned/demo_n1024_k128
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

/* Pull in prototype-core's library — plan + executor + executor_generic
 * are header-only static-inline today, so just one include. */
#include "../executor.h"

/* ─── Plan parameters ──────────────────────────────────────────────── */

#define N_FFT   1024
#define K_BATCH 128
#define NF      5
static const int FACTORS[NF] = {4, 4, 4, 4, 4};

#define BUF_LEN ((size_t)N_FFT * (size_t)K_BATCH)

/* ─── Allocation helpers ──────────────────────────────────────────── */

static double *alloc_doubles(size_t n) {
    double *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}

static size_t *alloc_size_t(size_t n) {
    size_t *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(size_t)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}

/* ─── Plan construction ────────────────────────────────────────────── */

/* Build a stride_plan_t matching the hardcoded N=1024 K=128 wisdom
 * entry in plan_executors.h:
 *   factors=[4,4,4,4,4], variants=[FLAT, T1S, T1S, T1S, T1S]
 *
 * Same pattern as the spike harness `bench/spike_n1024_k128.c::build_plan`.
 * In Phase 3 the planner replaces this hand-construction; for Phase 1
 * we just want to demonstrate that the executor consumes a plan struct
 * and dispatches correctly. */
static stride_plan_t *build_demo_plan(void) {
    stride_plan_t *plan = calloc(1, sizeof(*plan));
    plan->N = N_FFT;
    plan->K = K_BATCH;
    plan->num_stages = NF;
    plan->use_dif_forward = 0;
    for (int s = 0; s < NF; s++) plan->factors[s] = FACTORS[s];

    /* Stride per stage: K * product(R_{s+1}..R_{S-1}) */
    size_t strides[NF];
    size_t prod = 1;
    for (int s = NF - 1; s >= 0; s--) {
        strides[s] = (size_t)K_BATCH * prod;
        prod *= (size_t)FACTORS[s];
    }

    /* Codelet symbols — the same names registry_avx2.h exposes. */
    extern void radix4_n1_fwd_avx2(double *, double *,
                                   const double *, const double *,
                                   size_t, size_t);
    extern void radix4_t1s_dit_fwd_avx2(double *, double *,
                                        const double *, const double *,
                                        size_t, size_t);

    for (int s = 0; s < NF; s++) {
        const int R = FACTORS[s];
        stride_stage_t *st = &plan->stages[s];
        st->radix      = R;
        st->stride     = strides[s];
        st->num_groups = N_FFT / R;
        st->n1_fwd     = radix4_n1_fwd_avx2;
        st->t1s_fwd    = radix4_t1s_dit_fwd_avx2;

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
            st->group_base[g] = base % headroom;
            st->needs_tw[g]   = 1;
            st->cf0_re[g]     = 1.0;
            st->cf0_im[g]     = 0.0;
        }

        /* Per-group scalar twiddles — shared pool, arbitrary values
         * (this is timing-only; no correctness check). */
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

        /* Pre-walked tape for the (B)+(A) plan-shaped fast path. */
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

/* ─── Timing helper ────────────────────────────────────────────────── */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ─── Main ─────────────────────────────────────────────────────────── */

int main(void) {
    fprintf(stderr, "[demo] prototype-core Phase 1 smoke test\n");
    fprintf(stderr, "[demo] cell: N=%d K=%d, 5-stage 4x4x4x4x4 plan\n",
            N_FFT, K_BATCH);

    /* Verify the lookup will find a specialization for this plan. */
    stride_plan_t *plan = build_demo_plan();
    vfft_proto_exec_fn fn = vfft_proto_lookup_fwd_avx2(plan);
    if (fn) {
        fprintf(stderr, "[demo] specialization MATCHED: fn=%p (fast path)\n",
                (void *)fn);
    } else {
        fprintf(stderr, "[demo] specialization NOT found, will use generic\n");
    }

    /* Allocate buffers and run. */
    double *re = alloc_doubles(BUF_LEN);
    double *im = alloc_doubles(BUF_LEN);
    for (size_t i = 0; i < BUF_LEN; i++) {
        re[i] = 1.0 + 1e-9 * (double)i;
        im[i] = -1.0 + 1e-9 * (double)i;
    }

    /* Warm up the cache + branch predictor. */
    for (int w = 0; w < 5; w++) {
        vfft_proto_execute_fwd(plan, re, im, (size_t)K_BATCH);
    }

    /* Timed runs — single dispatch per run. */
    const int RUNS = 1000;
    double t0 = now_sec();
    for (int r = 0; r < RUNS; r++) {
        vfft_proto_execute_fwd(plan, re, im, (size_t)K_BATCH);
    }
    double t1 = now_sec();
    double per_call_us = (t1 - t0) / RUNS * 1e6;

    printf("[demo] %d executions completed without crashing\n", RUNS);
    printf("[demo] mean wall time: %.1f us per FFT\n", per_call_us);
    printf("[demo] (expected ~100 us on Raptor Lake; spike harness measured 106 us baseline)\n");

    free(re); free(im);
    return 0;
}
