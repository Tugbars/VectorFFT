/* c2r_jit_runtime.h — runtime JIT resolve for the c2r (inverse real FFT).
 *
 * Thin wrapper over rfft_jit_runtime.h's generic _real_jit_resolve (shared emit/
 * compile/dlopen/cache machinery). Include AFTER c2r.h (needs c2r_plan_t). Gated by
 * the caller under VFFT_USE_JIT, same as the rfft/c2c runtimes. */
#ifndef VFFT_C2R_JIT_RUNTIME_H
#define VFFT_C2R_JIT_RUNTIME_H

#include "rfft_jit_runtime.h"   /* _real_jit_resolve + dlopen macros + shared registry */

typedef void (*c2r_jit_fn)(const c2r_plan_t *, const double *, double *);

/* Resolve the c2r winner's JIT executor (emit_c2r_jit.py -> c2r_jit_exec). NULL ->
 * caller falls back to c2r_execute_packed. */
static inline c2r_jit_fn
vfft_c2r_jit_resolve(int N, size_t K, const int *factors, int nf,
                     const int *variants, const char *isa) {
    return (c2r_jit_fn)_real_jit_resolve(N, K, factors, nf, variants, isa,
        "c2r_jit_prelude.h", "emit_c2r_jit.py", "c2r_jit_exec", "c2rjit", "c2r", /*mode=*/NULL);
}

#endif /* VFFT_C2R_JIT_RUNTIME_H */
