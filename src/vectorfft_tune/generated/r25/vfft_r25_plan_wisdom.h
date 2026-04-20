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
 *       8     8   t1s                87.0
 *       8    16   t1s                86.9
 *       8   200   flat               80.2
 *       8   256   log3              106.9
 *      16    16   t1s               158.1
 *      16    24   flat              152.7
 *      16   400   flat              152.0
 *      16   512   t1s               339.2
 *      32    32   flat              292.4
 *      32    40   flat              290.9
 *      32   800   flat              295.6
 *      32  1024   t1s               705.0
 *      64    64   flat              596.3
 *      64    72   flat              616.3
 *      64  1600   t1s               621.5
 *      64  2048   t1s              1308.8
 *     128   128   log3             1463.5
 *     128   136   t1s              1215.0
 *     128  3200   t1s              1368.4
 *     128  4096   t1s              2606.1
 *     256   256   t1s              4834.4
 *     256   264   t1s              2618.2
 *     256  6400   t1s              5547.9
 *     256  8192   t1s              7086.8
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix25_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix25_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8, 16, 64, 128, 256} */
    if (me >= 8 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R25_PLAN_WISDOM_H */
