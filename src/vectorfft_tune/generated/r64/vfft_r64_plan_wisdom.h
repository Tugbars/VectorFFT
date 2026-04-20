/* vfft_r64_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=64.
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
#ifndef VFFT_R64_PLAN_WISDOM_H
#define VFFT_R64_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      64    64   t1s              4683.0
 *      64    72   t1s              4360.0
 *      64   512   flat            11078.2
 *     128   128   t1s             15505.1
 *     128   136   t1s              8891.1
 *     128  1024   flat            22449.3
 *     256   256   flat            40080.2
 *     256   264   log3            19505.8
 *     256  2048   flat            44559.6
 *     512   512   log3            93756.4
 *     512   520   log3            42605.0
 *     512  4096   flat            92149.6
 *    1024  1024   log3           184002.7
 *    1024  1032   log3            84148.1
 *    1024  8192   log3           185782.8
 *    2048  2048   log3           395413.4
 *    2048  2056   log3           178482.1
 *    2048 16384   log3           398118.2
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix64_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {512, 1024, 2048} */
    if (me == 512 || me == 1024 || me == 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix64_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 128} */
    if (me == 64 || me == 128) return 1;
    return 0;
}

#endif /* VFFT_R64_PLAN_WISDOM_H */
