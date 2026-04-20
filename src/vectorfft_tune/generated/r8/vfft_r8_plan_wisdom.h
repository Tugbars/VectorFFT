/* vfft_r8_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=8.
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
#ifndef VFFT_R8_PLAN_WISDOM_H
#define VFFT_R8_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      64    64   t1s               224.4
 *      64    72   t1s               219.8
 *      64   512   log3              437.2
 *     128   128   t1s               434.2
 *     128   136   log3              457.4
 *     128  1024   log3              862.5
 *     256   256   log3              906.7
 *     256   264   log3              904.2
 *     256  2048   log3             3971.9
 *     512   512   log3             3608.0
 *     512   520   log3             2079.5
 *     512  4096   log3             7847.2
 *    1024  1024   log3             7029.8
 *    1024  1032   log3             4043.2
 *    1024  8192   log3            15960.6
 *    2048  2048   log3            31634.2
 *    2048  2056   log3            10823.6
 *    2048 16384   log3            31561.3
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix8_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {128, 256, 512, 1024, 2048} */
    if (me >= 128 && me <= 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix8_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64} */
    if (me == 64) return 1;
    return 0;
}

#endif /* VFFT_R8_PLAN_WISDOM_H */
