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
 *      40    40   flat               39.2
 *      40    48   t1s                42.9
 *      40   200   flat               41.0
 *      40   320   t1s                38.9
 *      80    80   t1s                77.7
 *      80    88   flat               79.0
 *      80   400   flat               78.5
 *      80   640   flat               83.9
 *     160   160   t1s               151.4
 *     160   168   t1s               161.3
 *     160   800   flat              158.0
 *     160  1280   flat              147.1
 *     320   320   flat              305.1
 *     320   328   t1s               311.7
 *     320  1600   t1s               322.5
 *     320  2560   t1s               329.7
 *     640   640   t1s               657.1
 *     640   648   t1s               626.4
 *     640  3200   t1s               670.4
 *     640  5120   t1s               618.9
 *    1280  1280   t1s              1383.8
 *    1280  1288   t1s              1290.3
 *    1280  6400   t1s              1335.0
 *    1280 10240   t1s              1360.1
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
    /* Bench wins at me ∈ {40, 160, 320, 640, 1280} */
    if (me >= 40 && me <= 1280) return 1;
    return 0;
}

#endif /* VFFT_R5_PLAN_WISDOM_H */
