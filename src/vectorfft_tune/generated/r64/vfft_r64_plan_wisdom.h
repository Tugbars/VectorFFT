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
 *      64    64   t1s              1986.1
 *      64    72   t1s              1822.4
 *      64   512   log3             4718.6
 *     128   128   log3             8810.9
 *     128   136   t1s              4789.8
 *     128  1024   log3             9757.5
 *     256   256   log3            15560.3
 *     256   264   log3            10855.9
 *     256  2048   log3            25199.8
 *     512   512   flat            41511.3
 *     512   520   log3            20292.8
 *     512  4096   log3            64750.4
 *    1024  1024   log3            85343.0
 *    1024  1032   log3            39751.2
 *    1024  8192   log3           146220.3
 *    2048  2048   log3           202425.0
 *    2048  2056   log3            91275.8
 *    2048 16384   flat           322400.0
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix64_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {128, 256, 512, 1024, 2048} */
    if (me >= 128 && me <= 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix64_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64} */
    if (me == 64) return 1;
    return 0;
}

#endif /* VFFT_R64_PLAN_WISDOM_H */
