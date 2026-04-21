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
 *       8     8   t1s                60.1
 *       8    16   flat               61.4
 *       8   160   t1s                59.2
 *       8   256   t1s                61.3
 *      16    16   t1s               106.8
 *      16    24   flat              115.4
 *      16   320   t1s               105.7
 *      16   512   t1s               270.3
 *      32    32   t1s               224.3
 *      32    40   t1s               217.9
 *      32   640   t1s               220.5
 *      32  1024   log3              516.1
 *      64    64   t1s               410.0
 *      64    72   t1s               390.4
 *      64  1280   log3              910.6
 *      64  2048   log3             1070.7
 *     128   128   t1s               782.0
 *     128   136   t1s               851.8
 *     128  2560   log3             2192.2
 *     128  4096   t1s              2006.8
 *     256   256   log3             4337.7
 *     256   264   t1s              1885.5
 *     256  5120   t1s              4168.5
 *     256  8192   t1s              5095.9
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix20_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64} */
    if (me == 64) return 1;
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
