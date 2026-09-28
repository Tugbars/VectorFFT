/* plan.h — the entry point for stride_plan_t / stride_stage_t (1D C2C).
 */
#ifndef VFFT_PROTO_CORE_PLAN_H
#define VFFT_PROTO_CORE_PLAN_H

/* Re-export: the generated plan_executors.h defines the plan types, the SIMD
 * helpers and the (B)+(A) plan-shaped executors. Consumers of plan.h inherit
 * these symbols (hence the IWYU export pragma). */
#include "plan_executors.h"  // IWYU pragma: export

#include "common/support/zalloc.h" /* vfft_aligned_alloc / vfft_aligned_free: the one allocator */

#endif /* VFFT_PROTO_CORE_PLAN_H */
