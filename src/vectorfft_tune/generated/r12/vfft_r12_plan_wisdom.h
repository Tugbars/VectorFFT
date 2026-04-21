/* vfft_r12_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=12.
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
#ifndef VFFT_R12_PLAN_WISDOM_H
#define VFFT_R12_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   flat               30.0
 *       8    16   t1s                32.8
 *       8    96   flat               30.7
 *       8   256   flat               29.1
 *      16    16   t1s                56.3
 *      16    24   t1s                55.1
 *      16   192   flat               56.5
 *      16   512   t1s               183.8
 *      32    32   flat              107.5
 *      32    40   flat              108.2
 *      32   384   t1s               107.8
 *      32  1024   t1s               308.1
 *      64    64   flat              221.1
 *      64    72   t1s               204.8
 *      64   768   t1s               222.6
 *      64  2048   log3              693.8
 *     128   128   t1s               416.7
 *     128   136   t1s               408.7
 *     128  1536   flat             1305.0
 *     128  4096   t1s              1393.8
 *     256   256   t1s               822.0
 *     256   264   t1s               814.7
 *     256  3072   t1s              2725.4
 *     256  8192   log3             2124.2
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix12_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix12_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {16, 32, 64, 128, 256} */
    if (me >= 16 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R12_PLAN_WISDOM_H */
