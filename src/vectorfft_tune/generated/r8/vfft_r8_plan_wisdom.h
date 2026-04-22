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
 *      64    64   t1s                98.3
 *      64    72   flat               93.5
 *      64   512   log3              221.0
 *     128   128   t1s               166.8
 *     128   136   t1s               176.8
 *     128  1024   log3              442.3
 *     256   256   log3              416.3
 *     256   264   log3              416.7
 *     256  2048   flat              894.8
 *     512   512   log3             1731.2
 *     512   520   log3              961.4
 *     512  4096   flat             1720.6
 *    1024  1024   log3             3751.6
 *    1024  1032   flat             1905.5
 *    1024  8192   log3             2631.8
 *    2048  2048   flat             7262.5
 *    2048  2056   log3             3876.7
 *    2048 16384   flat            15711.4
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix8_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256, 512, 1024} */
    if (me == 256 || me == 512 || me == 1024) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix8_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {128} */
    if (me == 128) return 1;
    return 0;
}

#endif /* VFFT_R8_PLAN_WISDOM_H */
