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
 *      56    56   t1s               105.9
 *      56    64   t1s               105.3
 *      56   392   t1s               111.2
 *      56   448   flat              105.7
 *     112   112   t1s               199.4
 *     112   120   t1s               195.7
 *     112   784   flat              213.8
 *     112   896   flat              214.5
 *     224   224   t1s               399.5
 *     224   232   t1s               407.7
 *     224  1568   flat              420.5
 *     224  1792   t1s               425.1
 *     448   448   t1s               819.5
 *     448   456   t1s               867.5
 *     448  3136   t1s               915.1
 *     448  3584   t1s              1573.7
 *     896   896   t1s              1781.7
 *     896   904   t1s              1721.6
 *     896  6272   t1s              1754.6
 *     896  7168   t1s              3028.9
 *    1792  1792   t1s              3584.2
 *    1792  1800   t1s              3357.7
 *    1792 12544   t1s              3379.7
 *    1792 14336   t1s              3286.3
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
