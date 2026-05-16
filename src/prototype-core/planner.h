/* planner.h — 1D C2C plan construction for prototype-core.
 *
 * Given (N, K), produce a fully-populated stride_plan_t ready to feed
 * into vfft_proto_execute_fwd. Three entry points:
 *
 *   vfft_proto_auto_plan(N, K, reg)            — wisdom first, else estimate
 *   vfft_proto_wise_plan(N, K, reg, wis)       — strict wisdom, NULL if missing
 *   vfft_proto_estimate_plan(N, K, reg)        — cost-model only
 *
 * All three:
 *   1. Decide factorization (wisdom's, or greedy largest-first into
 *      available radixes).
 *   2. Allocate stride_plan_t + per-stage layout via twiddle.h's
 *      vfft_proto_compute_groups.
 *   3. Wire codelet function pointers from the registry — currently
 *      T1S only (the dominant variant in production wisdom).
 *   4. Compute twiddle tables via vfft_proto_compute_twiddles_dit.
 *   5. Pre-walk the (B)+(A) tape for plan_executors.h lookups.
 *
 * Scope:
 *   - Factorizable N (radixes 2..512). Non-factorable / prime N
 *     returns NULL — caller can fall back to production for those.
 *   - Forward direction only (bwd lands in a later phase).
 *   - T1S variant for inner stages (matches the executor's current
 *     codepath). Future phase extends to FLAT and LOG3 variants;
 *     wisdom entries that prefer those are honored with T1S substitution
 *     (correctness-preserving, may sacrifice some perf).
 *   - DIT orientation (DIF deferred).
 */
#ifndef VFFT_PROTO_CORE_PLANNER_H
#define VFFT_PROTO_CORE_PLANNER_H

#include "plan.h"
#include "twiddle.h"
#include "wisdom_reader.h"
#include "../prototype/generated/registry.h"  /* IWYU pragma: keep */
#include <stdlib.h>

/* Available radixes (must match the registry's standard set, sorted
 * largest-first for greedy factorization). */
static const int VFFT_PROTO_AVAILABLE_RADIXES[] = {
    64, 32, 25, 20, 19, 17, 16, 13, 12, 11, 10, 8, 7, 6, 5, 4, 3, 2
};
#define VFFT_PROTO_N_RADIXES \
    (sizeof(VFFT_PROTO_AVAILABLE_RADIXES) / sizeof(int))

/* SIMD-aware reorder: pow2 innermost (after factorization). Mirrors
 * production's SIMD reorder pass. */
static inline void vfft_proto_reorder_pow2_innermost(int *factors, int nf) {
    /* Move pow2 factors to the END (innermost = last stage). Stable
     * sort that preserves relative order of same-class factors. */
    int tmp[STRIDE_MAX_STAGES];
    int oi = 0;
    /* Non-pow2 first (outermost). */
    for (int i = 0; i < nf; i++) {
        int r = factors[i];
        if ((r & (r - 1)) != 0) tmp[oi++] = r;
    }
    /* Pow2 last (innermost). */
    for (int i = 0; i < nf; i++) {
        int r = factors[i];
        if ((r & (r - 1)) == 0) tmp[oi++] = r;
    }
    memcpy(factors, tmp, (size_t)nf * sizeof(int));
}

/* Greedy largest-first factorization of N into VFFT_PROTO_AVAILABLE_RADIXES.
 * Returns nf > 0 on success, 0 if N has a prime factor we can't cover. */
static inline int vfft_proto_factorize(int N, int *factors) {
    int nf = 0;
    int remaining = N;
    for (size_t i = 0; i < VFFT_PROTO_N_RADIXES && remaining > 1; i++) {
        int r = VFFT_PROTO_AVAILABLE_RADIXES[i];
        while (remaining % r == 0) {
            if (nf >= STRIDE_MAX_STAGES) return 0;
            factors[nf++] = r;
            remaining /= r;
        }
    }
    if (remaining != 1) return 0;  /* unfactorable */
    vfft_proto_reorder_pow2_innermost(factors, nf);
    return nf;
}

/* Wire codelet function pointers for one stage based on the radix. */
static inline void vfft_proto_wire_stage_codelets(
    stride_stage_t *st, int R, const vfft_proto_registry_t *reg)
{
    /* T1S-only for now (matches executor_generic.h's codepath). FLAT
     * and LOG3 substitute T1S — correctness-preserving. */
    st->n1_fwd  = reg->n1_fwd[R];
    st->t1s_fwd = reg->t1s_dit_fwd[R];
}

/* Build a plan with explicit factorization + registry. Returns NULL on
 * allocation failure or if any required codelet slot is empty in reg. */
static inline stride_plan_t *vfft_proto_plan_create(
    int N, size_t K, const int *factors, int nf,
    const vfft_proto_registry_t *reg)
{
    if (nf <= 0 || nf >= STRIDE_MAX_STAGES) return NULL;

    /* Verify every radix has the codelets we need. */
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        if (!reg->n1_fwd[R] || !reg->t1s_dit_fwd[R]) return NULL;
    }

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(*plan));
    plan->N = N;
    plan->K = K;
    plan->num_stages = nf;
    plan->use_dif_forward = 0;
    for (int s = 0; s < nf; s++) plan->factors[s] = factors[s];

    /* Wire codelets + compute layout + compute twiddles per stage. */
    for (int s = 0; s < nf; s++) {
        vfft_proto_wire_stage_codelets(&plan->stages[s], factors[s], reg);
    }
    vfft_proto_compute_plan_tables(plan);

    /* Pre-walk the (B)+(A) tape so plan_executors.h's specialized path
     * can dispatch directly when the lookup matches. */
    for (int s = 0; s < nf; s++) {
        stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        if (posix_memalign((void **)&st->tape, 64,
                           (size_t)G * sizeof(stride_invocation_t)) != 0) {
            /* Tape allocation failure isn't fatal — generic executor
             * doesn't use it. Leave NULL. */
            st->tape = NULL;
            continue;
        }
        for (int g = 0; g < G; g++) {
            st->tape[g].base  = st->group_base[g];
            st->tape[g].tw_re = st->tw_scalar_re[g];
            st->tape[g].tw_im = st->tw_scalar_im[g];
        }
    }

    return plan;
}

/* Wisdom-first plan creation. If wis has an entry for (N, K), use its
 * factorization. Otherwise fall back to greedy factorize. Returns NULL
 * for cells we can't handle (non-factorable / prime). */
static inline stride_plan_t *vfft_proto_auto_plan(
    int N, size_t K,
    const vfft_proto_registry_t *reg,
    const vfft_proto_wisdom_t *wis)
{
    /* 1. Wisdom hit. */
    if (wis) {
        const vfft_proto_wisdom_entry_t *e =
            vfft_proto_wisdom_lookup(wis, N, K);
        if (e && e->nf > 0 && !e->use_dif_forward) {
            stride_plan_t *plan = vfft_proto_plan_create(N, K, e->factors, e->nf, reg);
            if (plan) return plan;
            /* Fall through if codelet shape didn't match. */
        }
    }

    /* 2. Estimate mode: greedy factorize. */
    int factors[STRIDE_MAX_STAGES];
    int nf = vfft_proto_factorize(N, factors);
    if (nf == 0) return NULL;  /* non-factorable — caller falls back */
    return vfft_proto_plan_create(N, K, factors, nf, reg);
}

/* Strict wisdom: returns NULL if (N, K) not in wisdom. */
static inline stride_plan_t *vfft_proto_wise_plan(
    int N, size_t K,
    const vfft_proto_registry_t *reg,
    const vfft_proto_wisdom_t *wis)
{
    if (!wis) return NULL;
    const vfft_proto_wisdom_entry_t *e = vfft_proto_wisdom_lookup(wis, N, K);
    if (!e || e->nf == 0 || e->use_dif_forward) return NULL;
    return vfft_proto_plan_create(N, K, e->factors, e->nf, reg);
}

/* Estimate-only plan: ignores wisdom even if present. */
static inline stride_plan_t *vfft_proto_estimate_plan(
    int N, size_t K, const vfft_proto_registry_t *reg)
{
    int factors[STRIDE_MAX_STAGES];
    int nf = vfft_proto_factorize(N, factors);
    if (nf == 0) return NULL;
    return vfft_proto_plan_create(N, K, factors, nf, reg);
}

/* Destroy a plan built by any of the above. Frees twiddle tables, tape,
 * group_base, then the plan itself. */
static inline void vfft_proto_plan_destroy(stride_plan_t *plan) {
    if (!plan) return;
    vfft_proto_free_plan_tables(plan);
    free(plan);
}

#endif /* VFFT_PROTO_CORE_PLANNER_H */
