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
 *       8     8   t1s                62.4
 *       8    16   flat               58.1
 *       8   160   flat               61.4
 *       8   256   flat               62.6
 *      16    16   flat              119.6
 *      16    24   t1s               113.2
 *      16   320   t1s               112.0
 *      16   512   log3              287.0
 *      32    32   flat              229.5
 *      32    40   t1s               222.1
 *      32   640   t1s               229.0
 *      32  1024   t1s               562.1
 *      64    64   t1s               387.7
 *      64    72   t1s               392.4
 *      64  1280   flat              977.5
 *      64  2048   log3             1142.2
 *     128   128   t1s               777.1
 *     128   136   t1s               807.6
 *     128  2560   log3             2049.2
 *     128  4096   t1s              1966.8
 *     256   256   log3             3997.3
 *     256   264   t1s              1729.2
 *     256  5120   t1s              4161.3
 *     256  8192   t1s              5130.9
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix20_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix20_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {16, 32, 64, 128, 256} */
    if (me >= 16 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R20_PLAN_WISDOM_H */
