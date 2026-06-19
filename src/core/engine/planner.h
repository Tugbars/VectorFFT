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
 *   2. Decide per-stage variant assignment (wisdom's variants[], or
 *      default T1S everywhere in estimate mode).
 *   3. Allocate stride_plan_t + per-stage layout via twiddle.h's
 *      vfft_proto_compute_groups.
 *   4. Wire codelet function pointers from the registry per variant:
 *        FLAT  → reg->t1_dit_fwd[R]      (t1_fwd slot)
 *        LOG3  → reg->t1_dit_log3_fwd[R] (t1_fwd slot, use_log3=1)
 *        T1S   → reg->t1s_dit_fwd[R]    (t1s_fwd slot)
 *   5. Compute twiddle tables via vfft_proto_compute_twiddles_dit.
 *   6. Pre-walk the (B)+(A) tape for plan_executors.h lookups.
 *
 * Scope:
 *   - Factorizable N (radixes 2..512). Non-factorable / prime N
 *     returns NULL — caller can fall back to production for those.
 *   - Forward direction only (bwd lands in a later phase).
 *   - Variants FLAT (0), LOG3 (1), T1S (2). BUF (3) falls back to T1S.
 *   - DIT orientation (DIF deferred).
 */
#ifndef VFFT_PROTO_CORE_PLANNER_H
#define VFFT_PROTO_CORE_PLANNER_H

#include "plan.h"
#include "twiddle.h"
#include "wisdom_reader.h"
#include "registry.h"  /* IWYU pragma: keep */
#include <stdlib.h>

/* Variant codes match the wisdom file format. */
#define VFFT_PROTO_VARIANT_FLAT 0
#define VFFT_PROTO_VARIANT_LOG3 1
#define VFFT_PROTO_VARIANT_T1S  2
#define VFFT_PROTO_VARIANT_BUF  3

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

/* Greedy largest-first factorization of N into VFFT_PROTO_AVAILABLE_RADIXES. */
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

/* Wire codelet function pointers for one stage based on radix + variant.
 * Returns 0 if the variant's required codelet is missing in reg.
 *
 * use_dif_forward selects DIF (post-twiddle) vs DIT (pre-twiddle) codelet
 * family. Production parity: DIF only supports FLAT and LOG3 (no T1S);
 * T1S in DIF mode falls back to FLAT. */
static inline int vfft_proto_wire_stage_codelets(
    stride_stage_t *st, int R, int variant, int use_dif_forward,
    const vfft_proto_registry_t *reg)
{
    st->n1_fwd  = reg->n1_fwd[R];
    st->n1_bwd  = reg->n1_bwd[R];
    if (!st->n1_fwd) return 0;

    /* Clear all variant slots first. */
    st->t1_fwd    = NULL;
    st->t1_bwd    = NULL;
    st->t1s_fwd   = NULL;
    st->use_log3  = 0;

    if (use_dif_forward) {
        switch (variant) {
        case VFFT_PROTO_VARIANT_LOG3:
            st->t1_fwd   = reg->t1_dif_log3_fwd[R];
            st->t1_bwd   = reg->t1_dif_log3_bwd[R];
            st->use_log3 = 1;
            return st->t1_fwd != NULL && st->t1_bwd != NULL;
        case VFFT_PROTO_VARIANT_FLAT:
        case VFFT_PROTO_VARIANT_T1S:   /* DIF has no T1S — fall back to FLAT */
        case VFFT_PROTO_VARIANT_BUF:
        default:
            st->t1_fwd = reg->t1_dif_fwd[R];
            st->t1_bwd = reg->t1_dif_bwd[R];
            return st->t1_fwd != NULL && st->t1_bwd != NULL;
        }
    }

    /* DIT (default). */
    switch (variant) {
    case VFFT_PROTO_VARIANT_LOG3:
        st->t1_fwd   = reg->t1_dit_log3_fwd[R];
        st->t1_bwd   = reg->t1_dit_log3_bwd[R];
        st->use_log3 = 1;
        return st->t1_fwd != NULL;
    case VFFT_PROTO_VARIANT_FLAT:
        st->t1_fwd = reg->t1_dit_fwd[R];
        st->t1_bwd = reg->t1_dit_bwd[R];
        return st->t1_fwd != NULL;
    case VFFT_PROTO_VARIANT_T1S:
    case VFFT_PROTO_VARIANT_BUF:  /* BUF not implemented — fall back to T1S */
    default:
        st->t1s_fwd = reg->t1s_dit_fwd[R];
        /* t1_bwd for fallback path (also lets DIT-bwd Tier 1 work). */
        st->t1_bwd  = reg->t1_dit_bwd[R];
        if (st->t1s_fwd) return 1;
        /* T1S unavailable for this radix — try FLAT as last resort. */
        st->t1_fwd = reg->t1_dit_fwd[R];
        return st->t1_fwd != NULL;
    }
}

/* Full plan create with DIF orientation flag. */
static inline stride_plan_t *vfft_proto_plan_create_ex(
    int N, size_t K, const int *factors, const int *variants, int nf,
    int use_dif_forward, const vfft_proto_registry_t *reg)
{
    if (nf <= 0 || nf >= STRIDE_MAX_STAGES) return NULL;

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(*plan));
    plan->N = N;
    plan->K = K;
    plan->num_stages = nf;
    plan->use_dif_forward = use_dif_forward;
    for (int s = 0; s < nf; s++) plan->factors[s] = factors[s];

    /* Wire codelets per stage. For DIF the no-twiddle stage is the LAST
     * (s == nf-1); for DIT it's the first (s == 0). Variant choice on the
     * no-twiddle stage is moot but still wired for consistency. */
    for (int s = 0; s < nf; s++) {
        int v = variants ? variants[s] : VFFT_PROTO_VARIANT_T1S;
        if (!vfft_proto_wire_stage_codelets(&plan->stages[s],
                                             factors[s], v,
                                             use_dif_forward, reg)) {
            free(plan);
            return NULL;
        }
    }

    /* Compute layout then twiddles (dispatched by use_dif_forward). */
    for (int s = 0; s < nf; s++) {
        vfft_proto_compute_groups(plan, s);
        if (use_dif_forward) {
            vfft_proto_compute_twiddles_dif(plan, s);
        } else {
            vfft_proto_compute_twiddles_dit(plan, s);
        }
    }

    /* Pre-walk the (B)+(A) tape. Tape is T1S-shaped (scalar twiddles);
     * for FLAT/LOG3 stages the tape entries point at grp_tw arrays
     * instead, but the specialized executor lookup will only fire for
     * tape shapes its emitted variant matches, so we populate scalars
     * unconditionally and let the lookup miss for non-T1S plans. */
    for (int s = 0; s < nf; s++) {
        stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        if (vfft_proto_posix_memalign((void **)&st->tape, 64,
                           (size_t)G * sizeof(stride_invocation_t)) != 0) {
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

/* Build a plan with default DIT orientation (use_dif_forward = 0). */
static inline stride_plan_t *vfft_proto_plan_create(
    int N, size_t K, const int *factors, const int *variants, int nf,
    const vfft_proto_registry_t *reg)
{
    return vfft_proto_plan_create_ex(N, K, factors, variants, nf,
                                      /*use_dif_forward=*/0, reg);
}

/* Wisdom-first plan creation. */
static inline stride_plan_t *vfft_proto_auto_plan(
    int N, size_t K,
    const vfft_proto_registry_t *reg,
    const vfft_proto_wisdom_t *wis)
{
    if (wis) {
        const vfft_proto_wisdom_entry_t *e =
            vfft_proto_wisdom_lookup(wis, N, K);
        if (e && e->nf > 0) {
            /* Honor the wisdom's orientation: plan_create_ex carries use_dif.
             * DIF execution is validated (baked/JIT/generic DIF all roundtrip);
             * MEASURE records DIF winners, so the runtime must build them. */
            stride_plan_t *plan = vfft_proto_plan_create_ex(
                N, K, e->factors, e->variants, e->nf, e->use_dif_forward, reg);
            if (plan) return plan;
        }
    }

    int factors[STRIDE_MAX_STAGES];
    int nf = vfft_proto_factorize(N, factors);
    if (nf == 0) return NULL;
    return vfft_proto_plan_create(N, K, factors, /*variants=*/NULL, nf, reg);
}

/* Strict wisdom: returns NULL if (N, K) not in wisdom. */
static inline stride_plan_t *vfft_proto_wise_plan(
    int N, size_t K,
    const vfft_proto_registry_t *reg,
    const vfft_proto_wisdom_t *wis)
{
    if (!wis) return NULL;
    const vfft_proto_wisdom_entry_t *e = vfft_proto_wisdom_lookup(wis, N, K);
    if (!e || e->nf == 0) return NULL;
    return vfft_proto_plan_create_ex(N, K, e->factors, e->variants, e->nf,
                                     e->use_dif_forward, reg);
}

/* Estimate-only plan: ignores wisdom, defaults to T1S everywhere. */
static inline stride_plan_t *vfft_proto_estimate_plan(
    int N, size_t K, const vfft_proto_registry_t *reg)
{
    int factors[STRIDE_MAX_STAGES];
    int nf = vfft_proto_factorize(N, factors);
    if (nf == 0) return NULL;
    return vfft_proto_plan_create(N, K, factors, /*variants=*/NULL, nf, reg);
}

static inline void vfft_proto_plan_destroy(stride_plan_t *plan) {
    if (!plan) return;
    /* Override plans (Rader/Bluestein/DCT) own their data via override_destroy;
     * they have no staged tables (num_stages=0). Honor it FIRST or we leak the
     * override_data + its inner plan. Mirrors production's stride_plan_destroy
     * (src/core/executor.h) and the bridge's stride_plan_destroy. */
    if (plan->override_destroy) {
        plan->override_destroy(plan->override_data);
        free(plan);
        return;
    }
    vfft_proto_free_plan_tables(plan);
    free(plan);
}

#endif /* VFFT_PROTO_CORE_PLANNER_H */
