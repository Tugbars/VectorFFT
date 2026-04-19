/* vfft_r16_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=16.
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
#ifndef VFFT_R16_PLAN_WISDOM_H
#define VFFT_R16_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      64    64   t1s               239.3
 *      64    72   flat              264.9
 *      64   512   t1s               817.3
 *     128   128   t1s               479.6
 *     128   136   t1s               539.9
 *     128  1024   log3             1643.9
 *     256   256   log3             3818.4
 *     256   264   log3             1400.8
 *     256  2048   log3             3343.7
 *     512   512   flat             6936.1
 *     512   520   log3             2883.2
 *     512  4096   log3             6365.0
 *    1024  1024   log3            13337.4
 *    1024  1032   log3             5894.1
 *    1024  8192   flat            15912.7
 *    2048  2048   log3            29935.7
 *    2048  2056   log3            12232.0
 *    2048 16384   log3            32878.7
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix16_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256, 512, 1024, 2048} */
    if (me >= 256 && me <= 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix16_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 128} */
    if (me == 64 || me == 128) return 1;
    return 0;
}

#endif /* VFFT_R16_PLAN_WISDOM_H */
