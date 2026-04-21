/* vfft_r64_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=64.
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
#ifndef VFFT_R64_PLAN_WISDOM_H
#define VFFT_R64_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      64    64   t1s              2046.0
 *      64    72   t1s              1849.6
 *      64   512   t1s              4739.5
 *     128   128   t1s              8893.0
 *     128   136   t1s              3538.5
 *     128  1024   t1s              9256.8
 *     256   256   flat            19450.0
 *     256   264   log3             9885.8
 *     256  2048   log3            23766.6
 *     512   512   log3            42854.7
 *     512   520   log3            19442.2
 *     512  4096   log3            50445.7
 *    1024  1024   log3            79146.9
 *    1024  1032   log3            37281.6
 *    1024  8192   flat           148913.3
 *    2048  2048   log3           199429.7
 *    2048  2056   log3            98778.9
 *    2048 16384   flat           295996.9
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix64_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256, 512, 1024, 2048} */
    if (me >= 256 && me <= 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix64_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 128} */
    if (me == 64 || me == 128) return 1;
    return 0;
}

#endif /* VFFT_R64_PLAN_WISDOM_H */
