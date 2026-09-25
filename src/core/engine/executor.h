/* executor.h — single-threaded, in-place 1D C2C execution dispatch.
 *
 * Entry points: vfft_proto_execute_fwd() / vfft_proto_execute_bwd().
 *
 *   1. A plan with an override backend (Rader/Bluestein/DCT/...) runs its
 *      own execute fn.
 *   2. Otherwise the Tier-1 lookup (plan_executors.h; AVX-512 first when
 *      compiled in, then AVX2) returns the specialized (B)+(A) plan-shaped
 *      executor if one was emitted for this plan's shape: 5-6% faster on
 *      T1S/FLAT cells.
 *   3. Otherwise (cold cell) the generic loop in executor_generic.h, DIT or
 *      DIF by the plan's orientation, which handles every plan shape.
 */
#ifndef VFFT_PROTO_CORE_EXECUTOR_H
#define VFFT_PROTO_CORE_EXECUTOR_H

#include "plan.h"
#include "executor_generic.h"  // IWYU pragma: keep

/* Forward execution.
 *
 * Inputs (in-place):
 *   plan     — fully populated stride_plan_t from the planner (or
 *              hand-constructed for testing)
 *   re/im    — split-complex buffers of size plan->N * slice_K doubles
 *              each. Will be overwritten with the transform.
 *   slice_K  — K batches to process (may be ≤ plan->K for split
 *              execution)
 */
/* Compile-time Tier 1 lookup selector: pick AVX-512 when available,
 * fall back to AVX-2 otherwise. Both lookups exist in plan_executors.h;
 * the AVX-512 set is guarded by #ifdef __AVX512F__. */
static inline vfft_proto_exec_fn
_vfft_proto_lookup_fwd(const stride_plan_t *plan)
{
#if defined(__AVX512F__)
    vfft_proto_exec_fn fn = vfft_proto_lookup_fwd_avx512(plan);
    if (fn) return fn;
#endif
    return vfft_proto_lookup_fwd_avx2(plan);
}

static inline vfft_proto_exec_fn
_vfft_proto_lookup_bwd(const stride_plan_t *plan)
{
#if defined(__AVX512F__)
    vfft_proto_exec_fn fn = vfft_proto_lookup_bwd_avx512(plan);
    if (fn) return fn;
#endif
    return vfft_proto_lookup_bwd_avx2(plan);
}

static inline void vfft_proto_execute_fwd(const stride_plan_t *plan,
                                           double *re, double *im,
                                           size_t slice_K)
{
    /* Override backend (Rader/Bluestein/DCT/...): the plan carries its own
     * execute fn and handles the full transform + any internal threading, so
     * slice_K is moot here. */
    if (plan->override_fwd) {
        plan->override_fwd(plan->override_data, re, im);
        return;
    }

    /* DIF dispatch (Tier 1 specialization supported via lookup). */
    if (plan->use_dif_forward) {
        vfft_proto_exec_fn fn = _vfft_proto_lookup_fwd(plan);
        if (fn) {
            fn(plan, re, im, slice_K, plan->K, /*start_stage=*/0);
            return;
        }
        vfft_proto_execute_fwd_generic_dif(plan, re, im, slice_K);
        return;
    }

    /* DIT path. */
    vfft_proto_exec_fn fn = _vfft_proto_lookup_fwd(plan);
    if (fn) {
        fn(plan, re, im, slice_K, plan->K, /*start_stage=*/0);
        return;
    }
    vfft_proto_execute_fwd_generic(plan, re, im, slice_K);
}

/* Backward execution (unnormalized): fwd then bwd yields the original
 * input × N. Same dispatch as forward. */
static inline void vfft_proto_execute_bwd(const stride_plan_t *plan,
                                           double *re, double *im,
                                           size_t slice_K)
{
    if (plan->override_bwd) {
        plan->override_bwd(plan->override_data, re, im);
        return;
    }

    /* DIF dispatch (Tier 1 specialization supported via lookup). */
    if (plan->use_dif_forward) {
        vfft_proto_exec_fn fn = _vfft_proto_lookup_bwd(plan);
        if (fn) {
            fn(plan, re, im, slice_K, plan->K, /*start_stage=*/0);
            return;
        }
        vfft_proto_execute_bwd_generic_dif(plan, re, im, slice_K);
        return;
    }

    /* DIT path. */
    vfft_proto_exec_fn fn = _vfft_proto_lookup_bwd(plan);
    if (fn) {
        fn(plan, re, im, slice_K, plan->K, /*start_stage=*/0);
        return;
    }
    vfft_proto_execute_bwd_generic(plan, re, im, slice_K);
}

#endif /* VFFT_PROTO_CORE_EXECUTOR_H */
