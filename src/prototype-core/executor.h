/* executor.h — 1D C2C execution dispatch for prototype-core.
 *
 * Public entry point: vfft_proto_execute_fwd().
 *
 * Two-tier dispatch:
 *
 *   1. Call vfft_proto_lookup_fwd_avx2(plan) from plan_executors.h.
 *      Returns a specialized executor (the (B)+(A) plan-shaped fast
 *      path) if one was emitted for this plan's shape, NULL otherwise.
 *
 *   2. If non-NULL: invoke it. Saves 5-6% wall time on T1S/FLAT cells
 *      per the spike measurements in docs/61.
 *
 *   3. If NULL (cold cell): fall back to vfft_proto_execute_fwd_generic
 *      from executor_generic.h. The generic loop handles every plan
 *      shape; the specialized path is a strict optimization on top.
 *
 * Phase 1 scope:
 *   - 1D C2C only
 *   - In-place (re/im are both input and output)
 *   - Single-threaded
 *   - DIT orientation (DIF deferred to a later phase)
 *   - Forward direction only (bwd lands alongside in a later phase)
 *
 * Phase 1 simplifying assumption:
 *   The lookup picks avx2; AVX-512 dispatch lives in the per-ISA
 *   registry but isn't wired through here yet. Build per-ISA binaries
 *   for now (same model as production).
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
 *              execution, but Phase 1 uses slice_K == plan->K)
 */
static inline void vfft_proto_execute_fwd(const stride_plan_t *plan,
                                           double *re, double *im,
                                           size_t slice_K)
{
    /* Tier 1: try the (B)+(A) plan-shaped specialization. */
    vfft_proto_exec_fn fn = vfft_proto_lookup_fwd_avx2(plan);
    if (fn) {
        fn(plan, re, im, slice_K, plan->K, /*start_stage=*/0);
        return;
    }

    /* Tier 2: cold cell — generic per-stage loop. */
    vfft_proto_execute_fwd_generic(plan, re, im, slice_K);
}

#endif /* VFFT_PROTO_CORE_EXECUTOR_H */
