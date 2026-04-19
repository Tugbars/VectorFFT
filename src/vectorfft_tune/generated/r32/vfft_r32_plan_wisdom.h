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
 *      64    64   t1s               662.0
 *      64    72   t1s               736.2
 *      64   512   t1s              1717.3
 *     128   128   t1s              2319.7
 *     128   136   t1s              1412.5
 *     128  1024   t1s              3569.4
 *     256   256   flat             7242.3
 *     256   264   log3             3794.4
 *     256  2048   log3             7855.3
 *     512   512   log3            15351.1
 *     512   520   log3             7799.9
 *     512  4096   log3            18746.0
 *    1024  1024   log3            30142.4
 *    1024  1032   log3            15785.3
 *    1024  8192   flat            40418.8
 *    2048  2048   log3            71467.2
 *    2048  2056   log3            33403.9
 *    2048 16384   log3           122168.8
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix32_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256, 512, 1024, 2048} */
    if (me >= 256 && me <= 2048) return 1;
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
