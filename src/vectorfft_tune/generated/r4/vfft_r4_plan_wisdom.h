/* vfft_r4_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=4.
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
#ifndef VFFT_R4_PLAN_WISDOM_H
#define VFFT_R4_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      64    64   t1s                30.4
 *      64    72   t1s                29.3
 *      64   512   t1s                29.3
 *     128   128   flat               67.2
 *     128   136   t1s                58.0
 *     128  1024   flat               60.7
 *     256   256   flat              122.8
 *     256   264   flat              118.7
 *     256  2048   flat              119.9
 *     512   512   flat              273.2
 *     512   520   flat              271.0
 *     512  4096   flat              282.4
 *    1024  1024   log3              711.1
 *    1024  1032   log3              733.4
 *    1024  8192   log3              725.5
 *    2048  2048   log3             1489.8
 *    2048  2056   log3             1456.5
 *    2048 16384   log3             1477.0
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix4_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {1024, 2048} */
    if (me == 1024 || me == 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix4_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64} */
    if (me == 64) return 1;
    return 0;
}

#endif /* VFFT_R4_PLAN_WISDOM_H */
