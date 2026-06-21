/* rfft_jit_prelude.h — include surface for JIT-compiled rfft single-plan executors.
 *
 * The JIT'd rfft executor receives an `rfft_plan_t *` from the engine, so it must
 * compile against the SAME struct + helpers (rfft_mid_column, the VFFT_RFFT_* macros,
 * the codelet typedefs) and against rfft_execute_fwd_packed for its Kb!=K fallback.
 * All of that lives in rfft.h, which is self-contained (system headers only), so the
 * runtime JIT compile needs just -I<core/transforms/real> for this to resolve.
 *
 * Mirrors jit_prelude.h (c2c), which pulls plan_executors.h. */
#ifndef RFFT_JIT_PRELUDE_H
#define RFFT_JIT_PRELUDE_H
#include "rfft.h"
#endif
