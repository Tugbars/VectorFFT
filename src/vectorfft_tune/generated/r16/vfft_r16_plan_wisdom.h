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
 *      64    64   log3              650.3
 *      64    72   log3              657.4
 *      64   512   t1s              1598.0
 *     128   128   log3             1302.3
 *     128   136   log3             1298.7
 *     128  1024   t1s              3771.3
 *     256   256   log3             4992.2
 *     256   264   log3             2875.5
 *     256  2048   flat             7171.8
 *     512   512   flat            13717.6
 *     512   520   log3             5868.6
 *     512  4096   flat            15106.6
 *    1024  1024   flat            27841.6
 *    1024  1032   log3            11536.2
 *    1024  8192   flat            30950.9
 *    2048  2048   flat            62410.9
 *    2048  2056   log3            25194.9
 *    2048 16384   flat            63299.1
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix16_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 128, 256} */
    if (me == 64 || me == 128 || me == 256) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix16_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

#endif /* VFFT_R16_PLAN_WISDOM_H */
