/* jit_prelude.h — shared include surface for JIT-compiled single-plan executors.
 *
 * The JIT'd executor receives a `plan` pointer FROM THE CORE, so it must compile
 * against the SAME stride_plan_t the core uses. core/plan.h includes
 * plan_executors.h WITHOUT VFFT_PROTO_USE_PRODUCTION_PLAN_T, so the standalone
 * struct defined there is exactly what the core compiles against — ABI parity.
 * plan_executors.h also carries the VFFT_PROTO_STAGE_* macros + SIMD stubs the
 * executor body needs. The JIT translation unit is compiled with -I<this dir>,
 * so the "../core/plan.h" below resolves relative to here.
 *
 * Spike note: this also drags in the baked static executors (unused →
 * -Wno-unused-function). A later cut can extract just {struct, macros} into a
 * lean header; the generated machine code is identical either way. */
#ifndef VFFT_PROTO_JIT_PRELUDE_H
#define VFFT_PROTO_JIT_PRELUDE_H
#include "../core/plan.h"
#endif
