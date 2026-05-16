/* executor_generic.h — cold-cell fallback executor.
 *
 * [Phase 1 placeholder — will be populated when Phase 1 lands.]
 *
 * For plans that don't have a matching specialization in
 * plan_executors.h (i.e., the wisdom-portfolio plan-shaped fast path
 * doesn't cover this (N, K, factors, variants) tuple), this is the
 * correctness baseline.
 *
 * Conceptually mirrors src/core/executor.h's _stride_execute_fwd_slice_from:
 * a per-stage loop that dispatches each group through the runtime
 * variant branch tree. Slower than the specialized path (this is the
 * 5-6% wrapper share the (B)+(A) spike recovers) but works for every
 * plan shape the planner produces.
 *
 * Uses ../prototype/generated/registry.h to look up codelet function
 * pointers — so the codelet inventory and the executor are loosely
 * coupled: regen the registry, rebuild here, no source edits needed.
 *
 * Scope: 1D C2C, DIT. Same as executor.h.
 */
#ifndef VFFT_PROTO_CORE_EXECUTOR_GENERIC_H
#define VFFT_PROTO_CORE_EXECUTOR_GENERIC_H

#include "plan.h"

/* Phase 1 will add:
 *
 *   void vfft_proto_execute_fwd_generic(const stride_plan_t *plan,
 *                                       double *re, double *im,
 *                                       size_t slice_K);
 *   void vfft_proto_execute_bwd_generic(const stride_plan_t *plan,
 *                                       double *re, double *im,
 *                                       size_t slice_K);
 *
 * Each iterates plan->stages[s] and dispatches the variant codepath
 * (FLAT / LOG3 / T1S / n1_fallback) per group, using the codelet
 * pointers stored in stage->n1_fwd / t1_fwd / t1s_fwd / etc.
 */

#endif /* VFFT_PROTO_CORE_EXECUTOR_GENERIC_H */
