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
 *      64    64   t1s                73.8
 *      64    72   t1s                72.6
 *      64   512   t1s                74.0
 *     128   128   t1s               143.0
 *     128   136   t1s               146.6
 *     128  1024   t1s               143.3
 *     256   256   flat              329.5
 *     256   264   flat              328.7
 *     256  2048   flat              327.0
 *     512   512   flat              655.8
 *     512   520   flat              669.9
 *     512  4096   log3              675.8
 *    1024  1024   log3             1663.5
 *    1024  1032   log3             1627.3
 *    1024  8192   log3             1634.5
 *    2048  2048   log3             3326.9
 *    2048  2056   log3             3215.7
 *    2048 16384   flat             3498.3
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
    /* Bench wins at me ∈ {64, 128} */
    if (me == 64 || me == 128) return 1;
    return 0;
}

#endif /* VFFT_R4_PLAN_WISDOM_H */
