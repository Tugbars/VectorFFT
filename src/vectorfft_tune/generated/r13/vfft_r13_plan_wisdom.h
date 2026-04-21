/* vfft_r13_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=13.
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
#ifndef VFFT_R13_PLAN_WISDOM_H
#define VFFT_R13_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   t1s                51.8
 *       8    16   flat               53.4
 *       8   104   flat               53.0
 *       8   256   flat               54.6
 *      16    16   t1s                92.1
 *      16    24   flat               96.8
 *      16   208   flat              100.9
 *      16   512   t1s               234.7
 *      32    32   t1s               172.5
 *      32    40   t1s               179.3
 *      32   416   flat              209.8
 *      32  1024   flat              472.2
 *      64    64   flat              343.2
 *      64    72   flat              357.9
 *      64   832   flat              421.3
 *      64  2048   log3              876.7
 *     128   128   t1s               687.4
 *     128   136   flat              715.5
 *     128  1664   log3              828.9
 *     128  4096   t1s              1833.5
 *     256   256   t1s              1527.3
 *     256   264   t1s              1963.4
 *     256  3328   log3             2615.1
 *     256  8192   t1s              3445.4
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix13_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix13_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {16, 32, 128, 256} */
    if (me >= 16 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R13_PLAN_WISDOM_H */
