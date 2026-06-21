/* c2r_jit_prelude.h — include surface for JIT-compiled c2r single-plan executors.
 *
 * The JIT'd c2r executor receives a `c2r_plan_t *`, so it must compile against that
 * struct + c2r_mid_inv_column + c2r_execute_packed (Kb!=K fallback) + the codelet
 * typedefs. All in c2r.h (which pulls rfft.h for the shared base/types), so the
 * runtime compile needs just -I<core/transforms/real>. Mirrors rfft_jit_prelude.h. */
#ifndef C2R_JIT_PRELUDE_H
#define C2R_JIT_PRELUDE_H
#include "c2r.h"
#endif
