/* plan.h — stride_plan_t / stride_stage_t for prototype-core 1D C2C.
 *
 * Phase 1 strategy: reuse the minimal `stride_plan_t` / `stride_stage_t`
 * defined in `src/prototype/generated/plan_executors.h`. That struct
 * already has the fields needed for T1S-variant execution and the
 * (B)+(A) tape walk: `group_base`, `needs_tw`, `cf0_re/im`,
 * `tw_scalar_re/im`, `tape`, plus codelet function pointer slots
 * `n1_fwd` and `t1s_fwd`.
 *
 * Phase 2 (twiddle compute) extends this struct with the production-
 * shape fields: `grp_tw_re/im` (per-element twiddle arrays for the
 * FLAT variant), `cf_all_re/im` (used by the n1_fallback path on
 * large-K R=64 cells), `tw_pool_re/im` allocation pools. When Phase 2
 * lands, plan.h becomes the canonical definition site and
 * plan_executors.h gets re-emitted to consume the types from here
 * via its existing `#ifndef VFFT_PROTO_USE_PRODUCTION_PLAN_T` guard.
 */
#ifndef VFFT_PROTO_CORE_PLAN_H
#define VFFT_PROTO_CORE_PLAN_H

/* Pull in the minimal type definitions + SIMD helper stubs the
 * (B)+(A) plan-shaped executors compile against. This is a
 * re-export — plan.h's job is to be the prototype-core entry point
 * for the plan types, even though we don't reference any symbols
 * from plan_executors.h ourselves yet. The IWYU pragma below tells
 * include-what-you-use tooling that the consumers of plan.h are
 * intended to inherit these symbols. */
#include "../prototype/generated/plan_executors.h"  // IWYU pragma: export

#endif /* VFFT_PROTO_CORE_PLAN_H */
