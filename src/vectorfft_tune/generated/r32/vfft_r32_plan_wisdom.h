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
 *      64    64   t1s              1719.7
 *      64    72   t1s              1718.8
 *      64   512   t1s              4869.4
 *     128   128   t1s              3745.0
 *     128   136   t1s              3472.2
 *     128  1024   t1s              9776.6
 *     256   256   log3            15011.7
 *     256   264   log3             7972.5
 *     256  2048   log3            20297.6
 *     512   512   log3            39854.2
 *     512   520   log3            16477.0
 *     512  4096   log3            39751.3
 *    1024  1024   log3            77351.1
 *    1024  1032   log3            34442.5
 *    1024  8192   log3            80616.4
 *    2048  2048   log3           160091.1
 *    2048  2056   log3            72098.4
 *    2048 16384   log3           163659.7
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
