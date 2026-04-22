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
 *       8     8   flat               29.8
 *       8    16   t1s                30.0
 *       8    96   t1s                30.4
 *       8   256   flat               30.2
 *      16    16   t1s                54.2
 *      16    24   t1s                54.8
 *      16   192   t1s                55.1
 *      16   512   t1s               172.9
 *      32    32   t1s               106.1
 *      32    40   t1s               106.0
 *      32   384   t1s               109.1
 *      32  1024   log3              315.1
 *      64    64   t1s               211.0
 *      64    72   t1s               208.2
 *      64   768   t1s               212.7
 *      64  2048   t1s               719.0
 *     128   128   t1s               405.3
 *     128   136   t1s               405.2
 *     128  1536   log3             1273.4
 *     128  4096   log3             1410.0
 *     256   256   t1s               805.9
 *     256   264   t1s               804.6
 *     256  3072   log3             2700.0
 *     256  8192   t1s              1917.2
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix12_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {128} */
    if (me == 128) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix12_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8, 16, 32, 64, 128, 256} */
    if (me >= 8 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R12_PLAN_WISDOM_H */
