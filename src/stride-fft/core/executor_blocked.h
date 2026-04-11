/**
 * executor_blocked.h — Cache-blocked FFT executor
 *
 * Two-phase execution that reduces DTLB and L1 miss overhead:
 *
 *   Phase 1 (wide):    stages 0..split-1, full sweep over buffer.
 *                      These stages have large strides (L2/L3 working set)
 *                      and can't be blocked.
 *
 *   Phase 2 (blocked): stages split..last, processed in L1-resident blocks.
 *                      For each block of groups at the split stage, execute
 *                      ALL remaining stages before moving to the next block.
 *                      Data stays in L1/TLB across stage transitions.
 *
 * When to use:
 *   - Small K (4-8) with large N (>1024): DTLB store overhead dominates
 *   - Factorizations with >=2 L1-resident stages to block together
 *   - Determined by joint calibration in planner_blocked.h
 *
 * When NOT to use:
 *   - Large K (>=16): TLB entries reused naturally within each stage sweep
 *   - Few blockable stages: overhead exceeds benefit
 *   - The standard executor (executor.h) is faster — wisdom decides
 *
 * VTune evidence (N=16384 K=4):
 *   Standard: 48.4% retiring, 32.4% DTLB store overhead
 *   Blocked:  reduces DTLB store overhead by concentrating stores to
 *             same pages across consecutive stages.
 */
#ifndef STRIDE_EXECUTOR_BLOCKED_H
#define STRIDE_EXECUTOR_BLOCKED_H

#include "executor.h"

/* ═══════════════════════════════════════════════════════════════
 * SINGLE GROUP EXECUTION
 *
 * Shared by both standard and blocked paths. Handles all twiddle
 * variants: notw, t1s (scalar broadcast), t1 (full twiddle),
 * n1_fallback, log3.
 * ═══════════════════════════════════════════════════════════════ */

static inline void _stride_exec_group_fwd(const stride_stage_t *st,
                                           double *re, double *im,
                                           int g, size_t K, size_t full_K) {
    double *base_re = re + st->group_base[g];
    double *base_im = im + st->group_base[g];

    if (!st->needs_tw[g]) {
        st->n1_fwd(base_re, base_im, base_re, base_im,
                   st->stride, st->stride, K);
        return;
    }

    if (st->use_n1_fallback) {
        const int R = st->radix;
        const double *cfr = st->cf_all_re + (size_t)g * R * full_K;
        const double *cfi = st->cf_all_im + (size_t)g * R * full_K;
        for (int j = 0; j < R; j++) {
            double *lr = base_re + (size_t)j * st->stride;
            double *li = base_im + (size_t)j * st->stride;
            const double *wr = cfr + (size_t)j * full_K;
            const double *wi = cfi + (size_t)j * full_K;
            for (size_t kk = 0; kk < K; kk++) {
                double tr = lr[kk];
                lr[kk] = tr * wr[kk] - li[kk] * wi[kk];
                li[kk] = tr * wi[kk] + li[kk] * wr[kk];
            }
        }
        st->n1_fwd(base_re, base_im, base_re, base_im,
                   st->stride, st->stride, K);
        return;
    }

    if (st->use_log3) {
        double cfr = st->cf0_re[g], cfi = st->cf0_im[g];
        if (cfr != 1.0 || cfi != 0.0) {
            const int R = st->radix;
            for (int j = 0; j < R; j++) {
                double *lr = base_re + (size_t)j * st->stride;
                double *li = base_im + (size_t)j * st->stride;
                for (size_t kk = 0; kk < K; kk++) {
                    double tr = lr[kk];
                    lr[kk] = tr * cfr - li[kk] * cfi;
                    li[kk] = tr * cfi + li[kk] * cfr;
                }
            }
        }
        st->t1_fwd(base_re, base_im,
                   st->grp_tw_re[g], st->grp_tw_im[g],
                   st->stride, K);
        return;
    }

    /* Common factor on leg 0 */
    double cfr = st->cf0_re[g], cfi = st->cf0_im[g];
    if (cfr != 1.0 || cfi != 0.0) {
        for (size_t kk = 0; kk < K; kk++) {
            double tr = base_re[kk];
            base_re[kk] = tr * cfr - base_im[kk] * cfi;
            base_im[kk] = tr * cfi + base_im[kk] * cfr;
        }
    }

    /* Scalar twiddle broadcast path */
    if (st->t1s_fwd && st->tw_scalar_re && st->tw_scalar_re[g]
#ifdef STRIDE_FORCE_TEMP_BUFFER
        && 0
#endif
       ) {
        st->t1s_fwd(base_re, base_im,
                     st->tw_scalar_re[g], st->tw_scalar_im[g],
                     st->stride, K);
        return;
    }

    /* Scalar twiddle with temp buffer broadcast */
    if (st->tw_scalar_re && st->tw_scalar_re[g]) {
        #ifndef STRIDE_TW_BLOCK_K
        #define STRIDE_TW_BLOCK_K 64
        #endif
        const int Rm1 = st->radix - 1;
        const double *stw_r = st->tw_scalar_re[g];
        const double *stw_i = st->tw_scalar_im[g];
        double tw_buf_re[63 * STRIDE_TW_BLOCK_K];
        double tw_buf_im[63 * STRIDE_TW_BLOCK_K];
        for (size_t kb = 0; kb < K; kb += STRIDE_TW_BLOCK_K) {
            size_t this_K = K - kb;
            if (this_K > STRIDE_TW_BLOCK_K) this_K = STRIDE_TW_BLOCK_K;
            for (int j = 0; j < Rm1; j++) {
                double wr = stw_r[j], wi = stw_i[j];
                size_t base = (size_t)j * this_K;
                for (size_t kk = 0; kk < this_K; kk++) {
                    tw_buf_re[base + kk] = wr;
                    tw_buf_im[base + kk] = wi;
                }
            }
            st->t1_fwd(base_re + kb, base_im + kb,
                       tw_buf_re, tw_buf_im, st->stride, this_K);
        }
        return;
    }

    /* Full twiddle table path */
    st->t1_fwd(base_re, base_im,
               st->grp_tw_re[g], st->grp_tw_im[g],
               st->stride, K);
}

static inline void _stride_exec_group_bwd(const stride_stage_t *st,
                                           double *re, double *im,
                                           int g, size_t K, size_t full_K) {
    double *base_re = re + st->group_base[g];
    double *base_im = im + st->group_base[g];
    const int R = st->radix;

    st->n1_bwd(base_re, base_im, base_re, base_im,
               st->stride, st->stride, K);

    if (st->needs_tw[g] && st->cf_all_re) {
        const double *cfr = st->cf_all_re + (size_t)g * R * full_K;
        const double *cfi = st->cf_all_im + (size_t)g * R * full_K;
        for (int j = 0; j < R; j++) {
            double *lr = base_re + (size_t)j * st->stride;
            double *li = base_im + (size_t)j * st->stride;
            const double *wr = cfr + (size_t)j * full_K;
            const double *wi = cfi + (size_t)j * full_K;
            for (size_t kk = 0; kk < K; kk++) {
                double tr = lr[kk];
                lr[kk] = tr * wr[kk] + li[kk] * wi[kk];
                li[kk] = li[kk] * wr[kk] - tr * wi[kk];
            }
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
 * BINARY SEARCH — find first group with group_base >= lo
 *
 * Used to locate which groups of a later stage fall within
 * a block's element range, without scanning all groups.
 * ═══════════════════════════════════════════════════════════════ */

static inline int _blocked_find_first(const stride_stage_t *st, size_t lo) {
    int left = 0, right = st->num_groups;
    while (left < right) {
        int mid = (left + right) / 2;
        if (st->group_base[mid] < lo) left = mid + 1;
        else right = mid;
    }
    return left;
}


/* ═══════════════════════════════════════════════════════════════
 * BLOCKED FORWARD EXECUTOR
 *
 * split_stage:  first stage to block (stages before are wide)
 * block_groups: number of split-stage groups per block
 *
 * For each block:
 *   1. Execute split_stage groups [gb, gb+block_groups)
 *   2. For each later stage, binary-search the group range
 *      within the block's element span, execute those groups
 *
 * This keeps data in L1/TLB across stage transitions within a block.
 * ═══════════════════════════════════════════════════════════════ */

static inline void _stride_execute_fwd_blocked(const stride_plan_t *plan,
                                                double *re, double *im,
                                                int split_stage,
                                                int block_groups) {
    const size_t K = plan->K;

    /* Phase 1: wide stages — full sweep */
    for (int s = 0; s < split_stage; s++) {
        const stride_stage_t *st = &plan->stages[s];
        for (int g = 0; g < st->num_groups; g++)
            _stride_exec_group_fwd(st, re, im, g, K, K);
    }

    if (split_stage >= plan->num_stages) return;

    /* Phase 2: blocked stages */
    const stride_stage_t *split_st = &plan->stages[split_stage];
    int total_groups = split_st->num_groups;

    for (int gb = 0; gb < total_groups; gb += block_groups) {
        int g_end = gb + block_groups;
        if (g_end > total_groups) g_end = total_groups;

        /* Element range for this block */
        size_t block_lo = split_st->group_base[gb];
        size_t block_hi = split_st->group_base[g_end - 1] +
                          (size_t)(split_st->radix - 1) * split_st->stride + K;

        /* Execute split-stage groups directly (no search needed) */
        for (int g = gb; g < g_end; g++)
            _stride_exec_group_fwd(split_st, re, im, g, K, K);

        /* Later stages: binary search for groups within block range */
        for (int s = split_stage + 1; s < plan->num_stages; s++) {
            const stride_stage_t *st = &plan->stages[s];
            int g_start = _blocked_find_first(st, block_lo);
            for (int g = g_start; g < st->num_groups; g++) {
                if (st->group_base[g] >= block_hi) break;
                _stride_exec_group_fwd(st, re, im, g, K, K);
            }
        }
    }
}

static inline void _stride_execute_bwd_blocked(const stride_plan_t *plan,
                                                double *re, double *im,
                                                int split_stage,
                                                int block_groups) {
    const size_t K = plan->K;

    /* Phase 1: blocked stages (reverse order of forward) */
    if (split_stage < plan->num_stages) {
        const stride_stage_t *split_st = &plan->stages[split_stage];
        int total_groups = split_st->num_groups;

        for (int gb = 0; gb < total_groups; gb += block_groups) {
            int g_end = gb + block_groups;
            if (g_end > total_groups) g_end = total_groups;

            size_t block_lo = split_st->group_base[gb];
            size_t block_hi = split_st->group_base[g_end - 1] +
                              (size_t)(split_st->radix - 1) * split_st->stride + K;

            /* Later stages first (reverse), then split stage */
            for (int s = plan->num_stages - 1; s > split_stage; s--) {
                const stride_stage_t *st = &plan->stages[s];
                int g_start = _blocked_find_first(st, block_lo);
                for (int g = g_start; g < st->num_groups; g++) {
                    if (st->group_base[g] >= block_hi) break;
                    _stride_exec_group_bwd(st, re, im, g, K, K);
                }
            }

            for (int g = gb; g < g_end; g++)
                _stride_exec_group_bwd(split_st, re, im, g, K, K);
        }
    }

    /* Phase 2: wide stages — full sweep (reverse order) */
    for (int s = split_stage - 1; s >= 0; s--) {
        const stride_stage_t *st = &plan->stages[s];
        for (int g = 0; g < st->num_groups; g++)
            _stride_exec_group_bwd(st, re, im, g, K, K);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * SPLIT POINT COMPUTATION
 *
 * Finds the first stage whose per-group working set fits in L1.
 * Returns num_stages if no stage fits (= no blocking).
 * ═══════════════════════════════════════════════════════════════ */

#ifndef STRIDE_BLOCKED_L1_BYTES
#define STRIDE_BLOCKED_L1_BYTES (48 * 1024)
#endif

static inline int _stride_find_split(const stride_plan_t *plan) {
    for (int s = 0; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];
        size_t ws = (size_t)st->radix * st->stride * plan->K * 2 * sizeof(double);
        if (ws <= STRIDE_BLOCKED_L1_BYTES)
            return s;
    }
    return plan->num_stages;  /* no blocking */
}

static inline int _stride_compute_block_groups(const stride_plan_t *plan,
                                                int split_stage) {
    if (split_stage >= plan->num_stages) return 1;
    const stride_stage_t *st = &plan->stages[split_stage];
    size_t per_grp = (size_t)st->radix * st->stride * plan->K * 2 * sizeof(double);
    int bg = (int)(STRIDE_BLOCKED_L1_BYTES / per_grp);
    if (bg < 1) bg = 1;
    if (bg > st->num_groups) bg = st->num_groups;
    return bg;
}


#endif /* STRIDE_EXECUTOR_BLOCKED_H */
