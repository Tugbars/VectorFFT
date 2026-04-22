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
 *      40    40   t1s                37.2
 *      40    48   t1s                37.2
 *      40   200   t1s                37.2
 *      40   320   t1s                37.0
 *      80    80   t1s                73.3
 *      80    88   flat               74.1
 *      80   400   t1s                73.2
 *      80   640   t1s                72.8
 *     160   160   flat              145.4
 *     160   168   t1s               145.7
 *     160   800   flat              147.3
 *     160  1280   t1s               146.0
 *     320   320   flat              290.0
 *     320   328   flat              292.8
 *     320  1600   t1s               288.4
 *     320  2560   t1s               296.0
 *     640   640   t1s               580.5
 *     640   648   t1s               585.4
 *     640  3200   t1s               633.0
 *     640  5120   t1s               603.7
 *    1280  1280   t1s              1230.2
 *    1280  1288   t1s              1185.9
 *    1280  6400   t1s              1210.6
 *    1280 10240   t1s              1241.2
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
