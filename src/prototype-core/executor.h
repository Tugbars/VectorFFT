/* executor.h — 1D C2C execution dispatch for prototype-core.
 *
 * [Phase 1 placeholder — will be populated when Phase 1 lands.]
 *
 * Public entry point:
 *
 *   void vfft_proto_execute_fwd(const stride_plan_t *plan,
 *                               double *re, double *im, size_t slice_K);
 *
 * Internally:
 *   1. Call vfft_proto_lookup_fwd_<isa>(plan) from plan_executors.h.
 *   2. If non-NULL: invoke the specialized plan-shaped executor (the
 *      (B)+(A) tape-walk fast path; saves 5-6% wall time on T1S/FLAT
 *      cells per the spike measurements in docs/61).
 *   3. If NULL (cold cell, no specialization for this plan shape):
 *      fall back to vfft_proto_execute_fwd_generic in executor_generic.h.
 *
 * Two-tier dispatch keeps the spike's specialized fast path as a
 * strict optimization layer; the generic path is the correctness
 * baseline that handles every plan shape the planner produces.
 *
 * Scope mirrors plan.h's: 1D C2C, in-place, single-threaded, DIT
 * (DIF deferred). Backward direction will land alongside the
 * forward implementation when Phase 1 ships.
 */
#ifndef VFFT_PROTO_CORE_EXECUTOR_H
#define VFFT_PROTO_CORE_EXECUTOR_H

#include "plan.h"

/* Phase 1 will add:
 *
 *   void vfft_proto_execute_fwd(const stride_plan_t *plan,
 *                               double *re, double *im,
 *                               size_t slice_K);
 *   void vfft_proto_execute_bwd(const stride_plan_t *plan,
 *                               double *re, double *im,
 *                               size_t slice_K);
 *
 * Internally: lookup specialized; fall back to generic.
 */

#endif /* VFFT_PROTO_CORE_EXECUTOR_H */
