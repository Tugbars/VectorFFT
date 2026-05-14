/* radix_cpe.h — placeholder for prototype CPE numbers.
 *
 * AUTO-GENERATED in spirit. Today's contents are zero — measure_cpe.c
 * has not yet been retargeted to the prototype codelets, so we have
 * no real per-radix cycle measurements. The factorizer falls back to
 * ops/SIMD-width from radix_profile.h when entries here are 0.0,
 * which gives a working (if approximate) cost model until measurement
 * lands.
 *
 * Two-column shape (vs production's four columns):
 *   cyc_n1     — first-stage (no-twiddle) codelet at K=256
 *   cyc_inner  — inner-stage codelet at K=256
 *
 * The collapse is intentional. The planner doesn't consult per-stage
 * variant predicates anymore (wisdom_bridge.h retired); the codelet
 * generator picks one inner-stage variant per radix, and cyc_inner
 * reflects whichever one it picked. See cost_model/README.md and
 * cost_model/factorizer.h for the full rationale.
 *
 * Calibration fingerprint (to be populated by measure_cpe):
 * Host OS:    (unmeasured)
 * Host CPU:   (unmeasured)
 * ISA tag:    (unmeasured)
 * Eff. freq:  (unmeasured)
 * Max CV:     (unmeasured)
 * Algorithm-class per radix: (unmeasured)
 * Emission mode: M-active (VFFT_USE_REGALLOC=1 VFFT_USE_REGALLOC_M5=1)
 * Date (UTC): (unmeasured)
 */
#ifndef STRIDE_RADIX_CPE_H
#define STRIDE_RADIX_CPE_H

#include "radix_profile.h"

typedef struct {
    double cyc_n1;
    double cyc_inner;
} stride_radix_cpe_t;

/* Empty tables — every entry falls back to ops/SIMD via the profile
 * table until measure_cpe is retargeted and run on a quiet host. */
static const stride_radix_cpe_t stride_radix_cpe_avx2[STRIDE_RADIX_PROFILE_MAX_R] = {
    /* populate via cost_model/measure_cpe */
};

static const stride_radix_cpe_t stride_radix_cpe_avx512[STRIDE_RADIX_PROFILE_MAX_R] = {
    /* populate via cost_model/measure_cpe */
};

#endif /* STRIDE_RADIX_CPE_H */
