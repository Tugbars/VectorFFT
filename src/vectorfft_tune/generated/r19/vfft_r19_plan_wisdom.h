/* vfft_r19_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=19.
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
#ifndef VFFT_R19_PLAN_WISDOM_H
#define VFFT_R19_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   flat              100.1
 *       8    16   t1s               107.1
 *       8   152   flat               96.0
 *       8   256   log3              109.5
 *      16    16   flat              188.6
 *      16    24   t1s               190.4
 *      16   304   t1s               202.6
 *      16   512   log3              345.2
 *      32    32   log3              414.0
 *      32    40   t1s               378.7
 *      32   608   t1s               422.5
 *      32  1024   log3              730.4
 *      64    64   flat              734.0
 *      64    72   t1s               738.5
 *      64  1216   t1s               777.2
 *      64  2048   log3             1439.3
 *     128   128   t1s              1468.8
 *     128   136   log3             1604.5
 *     128  2432   t1s              1622.5
 *     128  4096   flat             3070.3
 *     256   256   t1s              4983.8
 *     256   264   log3             3375.2
 *     256  4864   t1s              4608.9
 *     256  8192   flat             6825.7
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix19_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {32} */
    if (me == 32) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix19_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {16, 32, 64, 128, 256} */
    if (me >= 16 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R19_PLAN_WISDOM_H */
