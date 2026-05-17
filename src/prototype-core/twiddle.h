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
/* Ported verbatim from src/core/executor.h:plan_compute_twiddles_c with
 * mechanical renames. The production version handles every edge case
 * we've been re-discovering (k_prev=0 rows, pool sizing by n_tw_groups
 * counted upfront, per-stage single allocation, cf_all for backward).
 *
 * Differences from production:
 *   - vfft_proto_posix_memalign instead of STRIDE_ALIGNED_ALLOC (same
 *     semantics, different shim name for Windows _aligned_malloc).
 *   - Production stores combined twiddle in tw_pool (K-replicated) when
 *     not log3, and raw per_leg when log3. We keep that exact storage
 *     convention since the executor reads it via the same indexing. */
static inline void vfft_proto_compute_twiddles_dit(stride_plan_t *plan, int s)
{
    stride_stage_t *st = &plan->stages[s];
    const int nf = plan->num_stages;
    const size_t K = plan->K;
    const int N = plan->N;
    const int R = st->radix;
    const int ng = st->num_groups;

    st->needs_tw = (int *)calloc(ng, sizeof(int));
    st->grp_tw_re = (double **)calloc(ng, sizeof(double *));
    st->grp_tw_im = (double **)calloc(ng, sizeof(double *));
    st->tw_scalar_re = (double **)calloc(ng, sizeof(double *));
    st->tw_scalar_im = (double **)calloc(ng, sizeof(double *));
    st->cf0_re = (double *)calloc(ng, sizeof(double));
    st->cf0_im = (double *)calloc(ng, sizeof(double));

    if (s == 0) {
        /* First stage: no twiddles. */
        st->tw_pool_re = st->tw_pool_im = NULL;
        st->tw_scalar_pool_re = st->tw_scalar_pool_im = NULL;
        st->cf_all_re = st->cf_all_im = NULL;
        for (int g = 0; g < ng; g++) {
            st->cf0_re[g] = 1.0;
            st->cf0_im[g] = 0.0;
        }
        return;
    }

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

    /* Count twiddled groups for pool allocation. */
    int n_tw_groups = 0;
    {
        int counter[STRIDE_MAX_STAGES];
        memset(counter, 0, sizeof(counter));
        for (int g = 0; g < ng; g++) {
            int k_prev = 0;
            {
                int ci = 0;
                for (int d = 0; d < nf; d++) {
                    if (d == s) continue;
                    if (d == s - 1) k_prev = counter[ci];
                    ci++;
                }
            }
            if (k_prev != 0) n_tw_groups++;
            for (int d = n_other - 1; d >= 0; d--) {
                counter[d]++;
                if (counter[d] < other_sizes[d]) break;
                counter[d] = 0;
            }
        }
    }

    /* Allocate twiddle pools sized for the actual number of twiddled
     * groups (not ng) — production pattern. Avoids reserving slots for
     * k_prev=0 rows that never use them. */
    size_t per_grp = (size_t)(R - 1) * K;
    size_t scalar_per_grp = (size_t)(R - 1);

    if (n_tw_groups > 0) {
        vfft_proto_posix_memalign((void **)&st->tw_pool_re, 64,
            (size_t)n_tw_groups * per_grp * sizeof(double));
        vfft_proto_posix_memalign((void **)&st->tw_pool_im, 64,
            (size_t)n_tw_groups * per_grp * sizeof(double));
        vfft_proto_posix_memalign((void **)&st->tw_scalar_pool_re, 64,
            (size_t)n_tw_groups * scalar_per_grp * sizeof(double));
        vfft_proto_posix_memalign((void **)&st->tw_scalar_pool_im, 64,
            (size_t)n_tw_groups * scalar_per_grp * sizeof(double));
    } else {
        st->tw_pool_re = st->tw_pool_im = NULL;
        st->tw_scalar_pool_re = st->tw_scalar_pool_im = NULL;
    }

    /* Backward cf: full per-element twiddle for all groups. */
    st->cf_all_re = (double *)calloc((size_t)ng * R * K, sizeof(double));
    st->cf_all_im = (double *)calloc((size_t)ng * R * K, sizeof(double));

    /* Fill per-group data. */
    int counter[STRIDE_MAX_STAGES];
    memset(counter, 0, sizeof(counter));
    int tw_idx = 0;

    for (int g = 0; g < ng; g++) {
        int k_prev = 0;
        int lower_data_pos = 0;
        {
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
        }

        /* Backward: fill full combined twiddle for all legs. */
        for (int j = 0; j < R; j++) {
            int tw_exp = (int)(((long long)k_prev * ow_prev * (j * S_s + lower_data_pos)) % N);
            if (tw_exp < 0) tw_exp += N;
            double angle = -2.0 * M_PI * (double)tw_exp / (double)N;
            double wr = cos(angle), wi = sin(angle);
            for (size_t kk = 0; kk < K; kk++) {
                st->cf_all_re[(size_t)g * R * K + (size_t)j * K + kk] = wr;
                st->cf_all_im[(size_t)g * R * K + (size_t)j * K + kk] = wi;
            }
        }

        if (k_prev == 0) {
            st->needs_tw[g] = 0;
            st->cf0_re[g] = 1.0;
            st->cf0_im[g] = 0.0;
        } else {
            st->needs_tw[g] = 1;

            /* Common factor for leg 0. */
            int cf_exp = (int)(((long long)k_prev * ow_prev * lower_data_pos) % N);
            if (cf_exp < 0) cf_exp += N;
            double cf_angle = -2.0 * M_PI * (double)cf_exp / (double)N;
            double cfr = cos(cf_angle), cfi = sin(cf_angle);
            st->cf0_re[g] = cfr;
            st->cf0_im[g] = cfi;

            /* Twiddle tables for legs 1..R-1 — pointers into pools. */
            double *tw_r = st->tw_pool_re + (size_t)tw_idx * per_grp;
            double *tw_i = st->tw_pool_im + (size_t)tw_idx * per_grp;
            st->grp_tw_re[g] = tw_r;
            st->grp_tw_im[g] = tw_i;

            double *stw_r = st->tw_scalar_pool_re + (size_t)tw_idx * scalar_per_grp;
            double *stw_i = st->tw_scalar_pool_im + (size_t)tw_idx * scalar_per_grp;
            st->tw_scalar_re[g] = stw_r;
            st->tw_scalar_im[g] = stw_i;

            /* tw_scalar ALWAYS stores combined cf*per_leg[j] for legs 1..R-1
             * (regardless of use_log3). This unifies the bwd executor path:
             * t1s_bwd codelet expects combined twiddles. Forward T1S also uses
             * this format. Forward LOG3 uses grp_tw (filled separately below)
             * with raw per_leg — that path doesn't read tw_scalar. */
            for (int j = 1; j < R; j++) {
                int leg_exp = (int)(((long long)k_prev * ow_prev * j * S_s) % N);
                if (leg_exp < 0) leg_exp += N;
                double leg_angle = -2.0 * M_PI * (double)leg_exp / (double)N;
                double lr = cos(leg_angle), li = sin(leg_angle);
                double wr = cfr * lr - cfi * li;
                double wi = cfr * li + cfi * lr;
                stw_r[j - 1] = wr;
                stw_i[j - 1] = wi;
            }
            if (st->use_log3) {
                /* Log3 grp_tw: raw per_leg (no cf baked in) — fwd LOG3 codelet
                 * expects this format and the executor pre-applies cf to all R
                 * legs before calling the codelet. */
                for (int j = 1; j < R; j++) {
                    int leg_exp = (int)(((long long)k_prev * ow_prev * j * S_s) % N);
                    if (leg_exp < 0) leg_exp += N;
                    double leg_angle = -2.0 * M_PI * (double)leg_exp / (double)N;
                    double lr = cos(leg_angle), li = sin(leg_angle);
                    size_t base_idx = (size_t)(j - 1) * K;
                    for (size_t kk = 0; kk < K; kk++) {
                        tw_r[base_idx + kk] = lr;
                        tw_i[base_idx + kk] = li;
                    }
                }
            } else {
                /* Flat grp_tw: combined cf*per_leg, K-replicated. */
                for (int j = 1; j < R; j++) {
                    size_t base_idx = (size_t)(j - 1) * K;
                    for (size_t kk = 0; kk < K; kk++) {
                        tw_r[base_idx + kk] = stw_r[j - 1];
                        tw_i[base_idx + kk] = stw_i[j - 1];
                    }
                }
            }
            tw_idx++;
        }

        for (int d = n_other - 1; d >= 0; d--) {
            counter[d]++;
            if (counter[d] < other_sizes[d]) break;
            counter[d] = 0;
        }
    }
}

/* (The previous hand-built version's tail is removed — the production
 * port above is complete and self-contained.) */

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
        /* Pools allocated via vfft_proto_posix_memalign — MUST use the
         * matching aligned-free on Windows. Calling plain free() on
         * _aligned_malloc memory is UB and corrupts the heap. */
        if (st->tw_scalar_pool_re) vfft_proto_aligned_free(st->tw_scalar_pool_re);
        if (st->tw_scalar_pool_im) vfft_proto_aligned_free(st->tw_scalar_pool_im);
        if (st->tw_pool_re)        vfft_proto_aligned_free(st->tw_pool_re);
        if (st->tw_pool_im)        vfft_proto_aligned_free(st->tw_pool_im);
        /* Pointer arrays + cf_all were allocated with calloc — plain free. */
        free(st->tw_scalar_re); free(st->tw_scalar_im);
        free(st->grp_tw_re);    free(st->grp_tw_im);
        free(st->cf_all_re);    free(st->cf_all_im);
        free(st->group_base);
        free(st->needs_tw);
        free(st->cf0_re); free(st->cf0_im);
        if (st->tape) vfft_proto_aligned_free(st->tape);
    }
}

#endif /* VFFT_PROTO_CORE_TWIDDLE_H */
