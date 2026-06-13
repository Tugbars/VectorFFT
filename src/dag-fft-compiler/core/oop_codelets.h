/* oop_codelets.h — ABI-typed registry struct for the OOP codelet family.
 *
 * The norm (sections 62-64): every distinct codelet ABI gets its own typed
 * slot; the registrar is auto-emitted from coverage. This is the struct the
 * auto-emitted oop_registry_<isa>.h populates. It coexists with the
 * hand-written oop_leaf_registry.h (the Bailey fast-path switch) during the
 * transition; the auto registry is the coverage-complete one.
 *
 * TWO ABIs in this family (this is why spec needs its own slots):
 *   - REGULAR (11-arg, vfft_oop11_fn): runtime strides.
 *       fn(src_re, src_im, dst_re, dst_im, W_re, W_im,
 *          in_leg_stride, in_group_stride, out_leg_stride, out_group_stride, me)
 *   - SPEC (7-arg, vfft_oop7_fn): strides BAKED IN at codegen (rv = r*8), so
 *       no runtime stride params.
 *       fn(src_re, src_im, dst_re, dst_im, W_re, W_im, me)
 * Mixing them is a compile error — which is the point.
 *
 * Within each ABI, three kinds:
 *   - n1   : no-twiddle leaf (W_re = W_im = NULL at call).
 *   - t1p  : twiddled stage, FLAT. The DEFAULT twiddle codelet (fewer FMA).
 *   - t1p_log3 : twiddled stage, LOG3. A port-rebalancing OPTIMIZATION that
 *       spends idle FMA-port slack to relieve LOAD-port pressure (more FMA,
 *       fewer twiddle loads). The planner should pick this ONLY when a stage
 *       is load-bound with FMA slack; otherwise flat t1p wins. See the
 *       selection TODO in oop_plan.h / the cost model (memboundness signal).
 *       It is NOT a strict upgrade over flat — that was the old Bailey
 *       hardcode's wrong assumption.
 */
#ifndef VFFT_OOP_CODELETS_H
#define VFFT_OOP_CODELETS_H

#include <stddef.h>

#ifndef VFFT_OOP_MAX_RADIX
#define VFFT_OOP_MAX_RADIX 128
#endif

/* 11-arg runtime-stride ABI (regular OOP). Matches oop_leaf_registry.h's
 * vfft_oop11_fn; redeclared here under a stable name so this header is
 * self-contained. They are the same type. */
typedef void (*vfft_oop11_fn)(const double *, const double *,
                              double *, double *,
                              const double *, const double *,
                              size_t, size_t, size_t, size_t, size_t);

/* 7-arg baked-stride ABI (spec OOP, rv = r*8). */
typedef void (*vfft_oop7_fn)(const double *, const double *,
                             double *, double *,
                             const double *, const double *,
                             size_t);

typedef struct {
    /* regular (11-arg, runtime strides) */
    vfft_oop11_fn n1[VFFT_OOP_MAX_RADIX + 1];
    vfft_oop11_fn t1p[VFFT_OOP_MAX_RADIX + 1];       /* flat: the default */
    vfft_oop11_fn t1p_log3[VFFT_OOP_MAX_RADIX + 1];  /* opt-in port rebalance */
    /* spec (7-arg, baked strides) — distinct ABI, distinct slots */
    vfft_oop7_fn n1_spec[VFFT_OOP_MAX_RADIX + 1];
    vfft_oop7_fn t1p_spec[VFFT_OOP_MAX_RADIX + 1];
    vfft_oop7_fn t1p_log3_spec[VFFT_OOP_MAX_RADIX + 1];
} oop_codelets_t;

#endif /* VFFT_OOP_CODELETS_H */
