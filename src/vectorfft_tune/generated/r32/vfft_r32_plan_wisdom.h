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
 *      64    64   t1s               632.2
 *      64    72   t1s               791.3
 *      64   512   t1s              1897.3
 *     128   128   flat             2680.8
 *     128   136   t1s              1558.4
 *     128  1024   log3             3925.5
 *     256   256   log3             6392.3
 *     256   264   log3             3659.0
 *     256  2048   flat             8258.6
 *     512   512   log3            15295.1
 *     512   520   log3             8066.4
 *     512  4096   flat            19422.6
 *    1024  1024   flat            30952.9
 *    1024  1032   log3            16074.3
 *    1024  8192   log3            41223.0
 *    2048  2048   log3            62679.7
 *    2048  2056   log3            32975.0
 *    2048 16384   flat           117386.7
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
    /* Bench wins at me ∈ {64} */
    if (me == 64) return 1;
    return 0;
}

#endif /* VFFT_R32_PLAN_WISDOM_H */
