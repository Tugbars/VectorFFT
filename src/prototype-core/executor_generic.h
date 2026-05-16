/* executor_generic.h — cold-cell fallback executor for 1D C2C.
 *
 * Per-stage loop, function-pointer indirected codelet dispatch. Same
 * shape as the spike harnesses' `baseline_exec` (see
 * src/prototype/bench/spike_n131072_k4.c) — refactored into a reusable
 * library function so consumers don't reinvent it.
 *
 * Slower than the (B)+(A) plan-shaped specialization in plan_executors.h
 * (the 5-6% wrapper share documented in docs/61). The generic loop is
 * the CORRECTNESS BASELINE — it handles every plan shape the planner
 * produces, including cells that don't have a specialization emitted.
 *
 * Phase 1 scope:
 *   - T1S variant inner-stage codepath (the most common; 60% of wisdom
 *     stage assignments)
 *   - n1 codepath for stage 0 + groups where needs_tw=0
 *   - cf0 scalar prep when non-trivial
 *
 * Phase 2 (or 1.5) extensions:
 *   - FLAT variant codepath (with K-blocked _stride_broadcast_2 staging)
 *   - LOG3 variant codepath (with all-leg cf application)
 *   - use_n1_fallback path (R=64 large-K case)
 */
#ifndef VFFT_PROTO_CORE_EXECUTOR_GENERIC_H
#define VFFT_PROTO_CORE_EXECUTOR_GENERIC_H

#include "plan.h"

/* Forward executor — runs all stages of the plan over (re, im) buffers
 * of length plan->N * slice_K doubles each (split-complex layout).
 *
 * Phase 1 simplifying assumptions:
 *   - All inner-stage codelets are T1S (st->t1s_fwd set)
 *   - cf0 may be trivial or not; we handle both
 *   - No DIF orientation (DIT only)
 *   - No n1_fallback / log3 / FLAT codepaths yet — those land in 1.5/2
 */
static inline void vfft_proto_execute_fwd_generic(const stride_plan_t *plan,
                                                   double *re, double *im,
                                                   size_t slice_K)
{
    for (int s = 0; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];

            if (!st->needs_tw[g]) {
                /* No-twiddle path: use n1 codelet (stage 0 always, +
                 * inner-stage groups whose effective twiddle is trivial). */
                st->n1_fwd(base_re, base_im, NULL, NULL,
                           st->stride, slice_K);
                continue;
            }

            /* Inner-stage T1S codepath. cf0 scalar prep if non-trivial. */
            double cfr = st->cf0_re[g];
            double cfi = st->cf0_im[g];
            if (cfr != 1.0 || cfi != 0.0) {
                _stride_cmul_scalar_inplace(base_re, base_im, slice_K,
                                            cfr, cfi);
            }
            st->t1s_fwd(base_re, base_im,
                        st->tw_scalar_re[g], st->tw_scalar_im[g],
                        st->stride, slice_K);
        }
    }
}

#endif /* VFFT_PROTO_CORE_EXECUTOR_GENERIC_H */
