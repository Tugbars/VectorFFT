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
 *      64    64   flat               91.2
 *      64    72   t1s                86.5
 *      64   512   log3              230.5
 *     128   128   t1s               166.9
 *     128   136   flat              181.6
 *     128  1024   flat              417.2
 *     256   256   flat              413.2
 *     256   264   log3              414.6
 *     256  2048   flat              861.9
 *     512   512   log3             1746.8
 *     512   520   log3              929.6
 *     512  4096   flat             1701.1
 *    1024  1024   log3             3357.9
 *    1024  1032   log3             1858.7
 *    1024  8192   log3             2458.3
 *    2048  2048   log3             6880.7
 *    2048  2056   log3             3851.3
 *    2048 16384   flat            15103.1
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix8_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {512, 1024, 2048} */
    if (me == 512 || me == 1024 || me == 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix8_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

#endif /* VFFT_R8_PLAN_WISDOM_H */
