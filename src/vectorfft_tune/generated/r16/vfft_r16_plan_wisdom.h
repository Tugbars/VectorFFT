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
 *      64    64   t1s               247.4
 *      64    72   t1s               245.0
 *      64   512   t1s               815.9
 *     128   128   t1s               488.3
 *     128   136   t1s               499.7
 *     128  1024   t1s              1621.3
 *     256   256   log3             3607.2
 *     256   264   log3             1457.7
 *     256  2048   log3             3413.0
 *     512   512   flat             6782.9
 *     512   520   log3             3031.2
 *     512  4096   flat             6430.0
 *    1024  1024   flat            15181.2
 *    1024  1032   log3             6188.9
 *    1024  8192   flat            13329.9
 *    2048  2048   flat            27342.4
 *    2048  2056   log3            15745.6
 *    2048 16384   flat            35513.9
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix16_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256} */
    if (me == 256) return 1;
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
