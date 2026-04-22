/* vfft_r20_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=20.
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
#ifndef VFFT_R20_PLAN_WISDOM_H
#define VFFT_R20_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   t1s                60.9
 *       8    16   t1s                62.0
 *       8   160   flat               60.4
 *       8   256   t1s                62.2
 *      16    16   t1s               110.6
 *      16    24   t1s               108.0
 *      16   320   t1s               109.4
 *      16   512   flat              271.1
 *      32    32   t1s               225.3
 *      32    40   t1s               201.9
 *      32   640   t1s               221.4
 *      32  1024   log3              547.3
 *      64    64   t1s               430.7
 *      64    72   t1s               422.7
 *      64  1280   t1s              1014.0
 *      64  2048   t1s              1109.2
 *     128   128   t1s               809.1
 *     128   136   t1s               826.0
 *     128  2560   log3             2111.3
 *     128  4096   t1s              2058.2
 *     256   256   t1s              4262.9
 *     256   264   t1s              2145.7
 *     256  5120   t1s              4448.9
 *     256  8192   t1s              5165.8
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix20_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix20_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8, 16, 32, 64, 128, 256} */
    if (me >= 8 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R20_PLAN_WISDOM_H */
