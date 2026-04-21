/* vfft_r7_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=7.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * Derived from cross-protocol comparison at each sweep point. The
 * planner should consult these AFTER the codelet-level dispatcher
 * has been selected — they drive twiddle-table layout and the
 * K-blocked execution path.
 */
#ifndef VFFT_R7_PLAN_WISDOM_H
#define VFFT_R7_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      56    56   t1s               108.1
 *      56    64   t1s               100.4
 *      56   392   flat              110.0
 *      56   448   t1s               108.8
 *     112   112   t1s               198.2
 *     112   120   t1s               207.1
 *     112   784   t1s               217.6
 *     112   896   t1s               218.9
 *     224   224   t1s               404.8
 *     224   232   t1s               409.1
 *     224  1568   t1s               435.6
 *     224  1792   t1s               403.9
 *     448   448   t1s               849.7
 *     448   456   t1s               808.8
 *     448  3136   t1s               900.5
 *     448  3584   t1s              1579.7
 *     896   896   t1s              1779.1
 *     896   904   t1s              1619.8
 *     896  6272   t1s              1731.6
 *     896  7168   t1s              3084.7
 *    1792  1792   t1s              3470.7
 *    1792  1800   t1s              3644.5
 *    1792 12544   t1s              3215.2
 *    1792 14336   t1s              3856.3
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix7_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed log3 never wins on this host. */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix7_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {56, 112, 224, 448, 896, 1792} */
    if (me >= 56 && me <= 1792) return 1;
    return 0;
}

#endif /* VFFT_R7_PLAN_WISDOM_H */
