/* vfft_r32_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=32.
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
#ifndef VFFT_R32_PLAN_WISDOM_H
#define VFFT_R32_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      64    64   t1s               696.6
 *      64    72   t1s               689.3
 *      64   512   t1s              1775.5
 *     128   128   log3             1998.4
 *     128   136   t1s              1438.7
 *     128  1024   t1s              3661.6
 *     256   256   log3             7965.5
 *     256   264   flat             4283.9
 *     256  2048   log3             7853.6
 *     512   512   flat            15157.6
 *     512   520   log3             7957.4
 *     512  4096   flat            17841.1
 *    1024  1024   flat            30031.4
 *    1024  1032   log3            15508.1
 *    1024  8192   flat            39475.8
 *    2048  2048   log3            62425.8
 *    2048  2056   log3            31212.1
 *    2048 16384   flat           109664.8
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix32_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256, 2048} */
    if (me == 256 || me == 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix32_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 128} */
    if (me == 64 || me == 128) return 1;
    return 0;
}

#endif /* VFFT_R32_PLAN_WISDOM_H */
