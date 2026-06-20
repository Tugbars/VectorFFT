/* jit_prelude.h — shared include surface for JIT-compiled single-plan executors.
 *
 * The JIT'd executor receives a `plan` pointer FROM THE CORE, so it must compile
 * against the SAME stride_plan_t the core uses. That standalone struct (plus the
 * VFFT_PROTO_STAGE_* macros, SIMD stubs, and lookup tables the executor body
 * needs) is defined in plan_executors.h, which the core also compiles against —
 * so including it directly gives ABI parity. plan_executors.h is self-contained
 * (only stddef/stdlib/immintrin + its own STRIDE_MAX_STAGES), so the runtime JIT
 * compile needs just -I<generated/> for this to resolve.
 *
 * (Previously this pulled "../core/plan.h"; the core tree moved to src/core/ and
 * plan.h became a thin re-export of plan_executors.h, so we include the source of
 * truth directly. The runtime gcc is given -I<generated/> via VFFT_PROTO_JIT_GENINC
 * in jit_runtime.h.)
 *
 * Spike note: this also drags in the baked static executors (unused →
 * -Wno-unused-function). A later cut can extract just {struct, macros} into a
 * lean header; the generated machine code is identical either way. */
#ifndef VFFT_PROTO_JIT_PRELUDE_H
#define VFFT_PROTO_JIT_PRELUDE_H
#include "plan_executors.h"
#endif
