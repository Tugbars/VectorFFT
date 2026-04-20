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
 *      56    56   t1s                97.6
 *      56    64   t1s                97.5
 *      56   392   t1s                97.8
 *      56   448   t1s                98.0
 *     112   112   t1s               197.4
 *     112   120   t1s               197.0
 *     112   784   t1s               192.9
 *     112   896   t1s               194.2
 *     224   224   t1s               387.4
 *     224   232   t1s               397.4
 *     224  1568   t1s               394.0
 *     224  1792   t1s               391.5
 *     448   448   t1s               766.7
 *     448   456   t1s               780.4
 *     448  3136   t1s               769.0
 *     448  3584   log3             1530.8
 *     896   896   t1s              1633.1
 *     896   904   t1s              1618.5
 *     896  6272   t1s              1624.8
 *     896  7168   t1s              2741.5
 *    1792  1792   t1s              3166.4
 *    1792  1800   t1s              3231.3
 *    1792 12544   t1s              3413.3
 *    1792 14336   t1s              3110.0
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix7_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
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
