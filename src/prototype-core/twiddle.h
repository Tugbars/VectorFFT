/* twiddle.h — per-stage twiddle table computation for 1D C2C plans.
 *
 * [Phase 2 placeholder — will be populated when Phase 2 lands.]
 *
 * Given a plan struct with factors[] and num_stages already set, this
 * computes the per-stage runtime tables:
 *
 *   - group_base[g]      offset into re/im for group g
 *   - needs_tw[g]        1 if group's twiddle row is non-trivial
 *   - cf0_re/im[g]       leg-0 common factor per group (scalar)
 *   - tw_scalar_re/im[g] per-leg scalars for the T1S variant ((R-1) doubles per group)
 *   - grp_tw_re/im[g]    per-element twiddles for the FLAT variant
 *   - cf_all_re/im       all-leg twiddles for the n1_fallback path (R=64 large-K case)
 *
 * Mirrors src/core/'s plan_compute_twiddles_c for 1D C2C; ports it
 * here so prototype-core is independent of src/core/. R2C-specific,
 * 2D-specific, and trig-specific twiddle layouts are NOT ported —
 * those add separately when their respective workstreams come.
 *
 * Scope: 1D C2C, DIT (DIF deferred to Phase 1+epsilon).
 */
#ifndef VFFT_PROTO_CORE_TWIDDLE_H
#define VFFT_PROTO_CORE_TWIDDLE_H

#include "plan.h"

/* Phase 2 will add:
 *
 *   void vfft_proto_compute_twiddles_dit(stride_plan_t *plan);
 *   void vfft_proto_compute_twiddles_dif(stride_plan_t *plan);
 *
 * Both walk plan->stages[s] in order and populate the per-stage runtime
 * tables. Memory is owned by the plan; freed via stride_plan_destroy.
 */

#endif /* VFFT_PROTO_CORE_TWIDDLE_H */
