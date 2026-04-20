/* vfft_r5_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=5.
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
#ifndef VFFT_R5_PLAN_WISDOM_H
#define VFFT_R5_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      40    40   t1s                37.5
 *      40    48   t1s                36.9
 *      40   200   t1s                36.9
 *      40   320   t1s                37.3
 *      80    80   t1s                73.2
 *      80    88   flat               73.9
 *      80   400   t1s                74.8
 *      80   640   flat               73.6
 *     160   160   t1s               145.0
 *     160   168   t1s               148.2
 *     160   800   flat              146.7
 *     160  1280   t1s               145.0
 *     320   320   flat              287.5
 *     320   328   flat              288.5
 *     320  1600   t1s               289.2
 *     320  2560   t1s               296.8
 *     640   640   t1s               601.6
 *     640   648   t1s               582.8
 *     640  3200   t1s               584.6
 *     640  5120   flat              708.0
 *    1280  1280   t1s              1179.8
 *    1280  1288   t1s              1166.9
 *    1280  6400   t1s              1214.2
 *    1280 10240   flat             1481.0
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix5_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed log3 never wins on this host. */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix5_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {40, 80, 160, 320, 640, 1280} */
    if (me >= 40 && me <= 1280) return 1;
    return 0;
}

#endif /* VFFT_R5_PLAN_WISDOM_H */
