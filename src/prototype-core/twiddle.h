/* twiddle.h — per-stage layout + twiddle compute for 1D C2C plans.
 *
 * Phase 2 port of src/core/executor.h:plan_compute_groups +
 * plan_compute_twiddles_c. Faithfully reproduces production's Method C
 * twiddle generation for the T1S codepath:
 *
 *   group_base[g]    offset into re/im buffer for group g
 *   needs_tw[g]      1 if group requires twiddle multiplication
 *   cf0_re/im[g]     leg-0 common factor (scalar)
 *   tw_scalar_re/im[g]  (R-1) scalar twiddles for legs 1..R-1
 *
 * Method C: cf0 carries the j-independent part; per-leg scalars carry
 * the j-linear part. T1S codelets multiply by these scalars internally
 * via _mm256_set1_pd broadcast.
 *
 * Phase 2 scope (deliberately narrow):
 *   - DIT orientation only (DIF deferred)
 *   - Forward direction only (bwd deferred)
 *   - Method C scalar twiddles only (FLAT grp_tw_re/im / cf_all
 *     populated as NULL — they'd be needed for FLAT/n1_fallback paths,
 *     which Phase 2.5 adds)
 *   - LOG3 not yet wired (it shares the same exponent math but stores
 *     per_leg WITHOUT cf baked in — see commented branch in production)
 *
 * Memory ownership: each stage's per-group arrays are allocated here
 * via calloc. Phase 2.5 will introduce a tw_pool_re/im allocation
 * pattern matching production for memory efficiency; for now each
 * group's (R-1) scalar pair gets its own tiny calloc. The caller is
 * responsible for vfft_proto_plan_destroy() to free.
 */
#ifndef VFFT_PROTO_CORE_TWIDDLE_H
#define VFFT_PROTO_CORE_TWIDDLE_H

#include "plan.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ────────────────────────────────────────────────────────────────────
 * Per-stage layout: radix, stride, num_groups, group_base[].
 *
 * Mirrors src/core/executor.h:plan_compute_groups. The layout depends
 * on K (batch size) and the full factorization — each "other" stage
 * contributes a stride to base offsets.
 * ──────────────────────────────────────────────────────────────────── */
static inline void vfft_proto_compute_groups(stride_plan_t *plan, int s)
{
    stride_stage_t *st = &plan->stages[s];
    const int    nf = plan->num_stages;
    const size_t K  = plan->K;
    const int    N  = plan->N;
    const int    R  = plan->factors[s];

    /* dim_stride[d] = K × product(factors[d+1..nf-1]). Stage d's stride
     * is the distance (in doubles) between adjacent butterfly legs. */
    size_t dim_stride[STRIDE_MAX_STAGES];
    {
        size_t acc = K;
        for (int d = nf - 1; d >= 0; d--) {
            dim_stride[d] = acc;
            acc *= plan->factors[d];
        }
    }

    st->radix      = R;
    st->stride     = dim_stride[s];
    st->num_groups = N / R;

    st->group_base = (size_t *)malloc((size_t)st->num_groups * sizeof(size_t));

    /* For each group g, base = sum over "other" (non-s) stages of
     * counter[d] × other_strides[d]. counter[] is a mixed-radix
     * iteration in row-major order. */
    int    other_sizes  [STRIDE_MAX_STAGES];
    size_t other_strides[STRIDE_MAX_STAGES];
    int    n_other = 0;
    for (int d = 0; d < nf; d++) {
        if (d != s) {
            other_sizes  [n_other] = plan->factors[d];
            other_strides[n_other] = dim_stride[d];
            n_other++;
        }
    }

    int counter[STRIDE_MAX_STAGES];
    memset(counter, 0, sizeof(counter));

    for (int g = 0; g < st->num_groups; g++) {
        size_t base = 0;
        for (int d = 0; d < n_other; d++)
            base += (size_t)counter[d] * other_strides[d];
        st->group_base[g] = base;

        /* Increment mixed-radix counter (rightmost digit varies fastest). */
        for (int d = n_other - 1; d >= 0; d--) {
            counter[d]++;
            if (counter[d] < other_sizes[d]) break;
            counter[d] = 0;
        }
    }
}

/* ────────────────────────────────────────────────────────────────────
 * Method C twiddle compute (T1S subset).
 *
 * For each group g in stage s, computes the cross-stage twiddle that
 * pre-multiplies the input data. Method C factors it as:
 *
 *   cf0[g]        = W_N^{k_prev × ow_prev × lower_data_pos}    (j-independent)
 *   per_leg[j][g] = W_N^{k_prev × ow_prev × j × S_s}           (j-linear, j>0)
 *
 * The COMBINED twiddle is cf0 × per_leg[j]; the codelet applies cf0
 * to ALL R legs (via cf0_re/im[g] + the T1S scalar-broadcast path)
 * then multiplies legs 1..R-1 by tw_scalar_re/im[g][j-1].
 *
 * Mirrors src/core/executor.h:plan_compute_twiddles_c lines 1301-1530,
 * keeping only the scalar twiddle path (tw_scalar_re/im) — FLAT's
 * grp_tw_re/im and the n1_fallback's cf_all_re/im are NOT populated
 * (left NULL). Phase 2.5 adds those.
 * ──────────────────────────────────────────────────────────────────── */
static inline void vfft_proto_compute_twiddles_dit(stride_plan_t *plan, int s)
{
    stride_stage_t *st = &plan->stages[s];
    const int    nf = plan->num_stages;
    const int    N  = plan->N;
    const int    R  = st->radix;
    const int    ng = st->num_groups;

    /* Allocate per-group arrays. T1S subset only. */
    st->needs_tw     = (int *)     calloc((size_t)ng, sizeof(int));
    st->cf0_re       = (double *)  calloc((size_t)ng, sizeof(double));
    st->cf0_im       = (double *)  calloc((size_t)ng, sizeof(double));
    st->tw_scalar_re = (double **) calloc((size_t)ng, sizeof(double *));
    st->tw_scalar_im = (double **) calloc((size_t)ng, sizeof(double *));

    if (s == 0) {
        /* First stage: no twiddle. cf0 = 1.0, all groups skip twiddle. */
        for (int g = 0; g < ng; g++) {
            st->cf0_re[g] = 1.0;
            st->cf0_im[g] = 0.0;
            /* needs_tw[g] stays 0 — first stage uses n1 codelet. */
        }
        return;
    }

    /* S_s    = product(factors[s+1..nf-1])
     * ow_prev = product(factors[0..s-2])
     *
     * The cross-stage twiddle for stage s carries:
     *   exponent = k_prev × ow_prev × (j × S_s + lower_data_pos)
     * where k_prev is the digit in dim s-1 and lower_data_pos
     * accumulates the digits of dims s+1..nf-1.
     */
    int S_s = 1;
    for (int d = s + 1; d < nf; d++)
        S_s *= plan->factors[d];
    int ow_prev = 1;
    for (int d = 0; d < s - 1; d++)
        ow_prev *= plan->factors[d];

    int other_sizes[STRIDE_MAX_STAGES];
    int n_other = 0;
    for (int d = 0; d < nf; d++)
        if (d != s)
            other_sizes[n_other++] = plan->factors[d];

    /* Walk groups via mixed-radix counter. */
    int counter[STRIDE_MAX_STAGES];
    memset(counter, 0, sizeof(counter));

    for (int g = 0; g < ng; g++) {
        /* Decode counter into k_prev and lower_data_pos. */
        int k_prev = 0;
        int lower_data_pos = 0;
        int ci = 0;
        for (int d = 0; d < nf; d++) {
            if (d == s) continue;
            if (d == s - 1) k_prev = counter[ci];
            if (d > s) {
                int w = 1;
                for (int d2 = d + 1; d2 < nf; d2++)
                    w *= plan->factors[d2];
                lower_data_pos += counter[ci] * w;
            }
            ci++;
        }

        if (k_prev == 0) {
            /* Trivial twiddle row — codelet uses n1 codepath. */
            st->needs_tw[g] = 0;
            st->cf0_re[g]   = 1.0;
            st->cf0_im[g]   = 0.0;
        } else {
            st->needs_tw[g] = 1;

            /* Leg-0 common factor. */
            int cf_exp = (int)(((long long)k_prev * ow_prev * lower_data_pos) % N);
            if (cf_exp < 0) cf_exp += N;
            double cf_angle = -2.0 * M_PI * (double)cf_exp / (double)N;
            double cfr = cos(cf_angle);
            double cfi = sin(cf_angle);
            st->cf0_re[g] = cfr;
            st->cf0_im[g] = cfi;

            /* Per-leg scalar twiddles (legs 1..R-1). Method C: combined
             * = cf0 × per_leg[j]. */
            double *stw_r = (double *)calloc((size_t)(R - 1), sizeof(double));
            double *stw_i = (double *)calloc((size_t)(R - 1), sizeof(double));
            st->tw_scalar_re[g] = stw_r;
            st->tw_scalar_im[g] = stw_i;

            for (int j = 1; j < R; j++) {
                int leg_exp = (int)(((long long)k_prev * ow_prev * j * S_s) % N);
                if (leg_exp < 0) leg_exp += N;
                double leg_angle = -2.0 * M_PI * (double)leg_exp / (double)N;
                double lr = cos(leg_angle);
                double li = sin(leg_angle);
                /* Combined = cf × per_leg (Method C). */
                double wr = cfr * lr - cfi * li;
                double wi = cfr * li + cfi * lr;
                stw_r[j - 1] = wr;
                stw_i[j - 1] = wi;
            }
        }

        /* Increment mixed-radix counter. */
        for (int d = n_other - 1; d >= 0; d--) {
            counter[d]++;
            if (counter[d] < other_sizes[d]) break;
            counter[d] = 0;
        }
    }
}

/* ────────────────────────────────────────────────────────────────────
 * Top-level driver — call after the plan's factors[] / num_stages / K
 * / N are set. Populates every stage's runtime tables.
 * ──────────────────────────────────────────────────────────────────── */
static inline void vfft_proto_compute_plan_tables(stride_plan_t *plan)
{
    for (int s = 0; s < plan->num_stages; s++) {
        vfft_proto_compute_groups(plan, s);
        vfft_proto_compute_twiddles_dit(plan, s);
    }
}

/* ────────────────────────────────────────────────────────────────────
 * Memory cleanup. Pairs with vfft_proto_compute_plan_tables.
 * ──────────────────────────────────────────────────────────────────── */
static inline void vfft_proto_free_plan_tables(stride_plan_t *plan)
{
    for (int s = 0; s < plan->num_stages; s++) {
        stride_stage_t *st = &plan->stages[s];
        if (st->tw_scalar_re) {
            for (int g = 0; g < st->num_groups; g++) {
                free(st->tw_scalar_re[g]);
                free(st->tw_scalar_im[g]);
            }
        }
        free(st->tw_scalar_re); free(st->tw_scalar_im);
        free(st->group_base);
        free(st->needs_tw);
        free(st->cf0_re); free(st->cf0_im);
        if (st->tape) free(st->tape);
    }
}

#endif /* VFFT_PROTO_CORE_TWIDDLE_H */
