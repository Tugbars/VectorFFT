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
 *      64    64   flat               92.0
 *      64    72   flat               99.4
 *      64   512   flat              229.0
 *     128   128   t1s               173.8
 *     128   136   t1s               181.2
 *     128  1024   log3              439.6
 *     256   256   flat              429.4
 *     256   264   flat              435.2
 *     256  2048   log3              899.7
 *     512   512   flat             1771.6
 *     512   520   flat             1047.4
 *     512  4096   flat             1819.6
 *    1024  1024   log3             3680.2
 *    1024  1032   log3             2089.1
 *    1024  8192   flat             3681.6
 *    2048  2048   log3             6781.3
 *    2048  2056   flat             4104.1
 *    2048 16384   flat            16412.7
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix8_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {1024} */
    if (me == 1024) return 1;
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
