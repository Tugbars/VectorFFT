/* vfft_r25_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=25.
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
#ifndef VFFT_R25_PLAN_WISDOM_H
#define VFFT_R25_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   flat               79.0
 *       8    16   flat               80.5
 *       8   200   t1s                92.3
 *       8   256   t1s               131.0
 *      16    16   flat              152.6
 *      16    24   flat              156.3
 *      16   400   flat              155.7
 *      16   512   t1s               337.0
 *      32    32   flat              305.6
 *      32    40   flat              306.0
 *      32   800   t1s               315.3
 *      32  1024   log3              714.9
 *      64    64   flat              629.4
 *      64    72   t1s               609.3
 *      64  1600   t1s               623.0
 *      64  2048   t1s              1291.7
 *     128   128   t1s              1319.1
 *     128   136   t1s              1296.6
 *     128  3200   t1s              1368.4
 *     128  4096   t1s              2758.5
 *     256   256   t1s              5222.4
 *     256   264   t1s              2691.1
 *     256  6400   t1s              6107.6
 *     256  8192   t1s              6807.3
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix25_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix25_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8, 64, 128, 256} */
    if (me >= 8 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R25_PLAN_WISDOM_H */
