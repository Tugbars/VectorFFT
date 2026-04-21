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
 *      40    40   t1s                37.9
 *      40    48   flat               38.3
 *      40   200   flat               38.2
 *      40   320   flat               38.0
 *      80    80   flat               73.9
 *      80    88   flat               76.4
 *      80   400   flat               74.4
 *      80   640   flat               77.7
 *     160   160   flat              146.0
 *     160   168   t1s               148.7
 *     160   800   flat              156.0
 *     160  1280   flat              146.2
 *     320   320   flat              290.6
 *     320   328   flat              291.3
 *     320  1600   t1s               306.1
 *     320  2560   t1s               308.8
 *     640   640   t1s               582.8
 *     640   648   t1s               579.9
 *     640  3200   t1s               653.4
 *     640  5120   t1s               651.9
 *    1280  1280   t1s              1171.3
 *    1280  1288   t1s              1287.5
 *    1280  6400   t1s              1196.5
 *    1280 10240   t1s              1242.8
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
    /* Bench wins at me ∈ {320, 640, 1280} */
    if (me == 320 || me == 640 || me == 1280) return 1;
    return 0;
}

#endif /* VFFT_R5_PLAN_WISDOM_H */
