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
 *      64    64   flat              292.3
 *      64    72   flat              273.3
 *      64   512   flat              818.5
 *     128   128   t1s               562.3
 *     128   136   t1s               574.1
 *     128  1024   t1s              1699.0
 *     256   256   flat             3061.4
 *     256   264   log3             1405.6
 *     256  2048   flat             3600.1
 *     512   512   flat             6908.6
 *     512   520   log3             3056.6
 *     512  4096   flat             6315.6
 *    1024  1024   flat            13190.7
 *    1024  1032   log3             6226.0
 *    1024  8192   flat            13709.1
 *    2048  2048   log3            27521.1
 *    2048  2056   log3            11930.6
 *    2048 16384   flat            30468.8
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix16_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {2048} */
    if (me == 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix16_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {128} */
    if (me == 128) return 1;
    return 0;
}

#endif /* VFFT_R16_PLAN_WISDOM_H */
