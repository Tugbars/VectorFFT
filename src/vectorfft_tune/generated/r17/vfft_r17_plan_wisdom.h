/* vfft_r17_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=17.
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
#ifndef VFFT_R17_PLAN_WISDOM_H
#define VFFT_R17_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   flat               74.8
 *       8    16   flat               76.9
 *       8   136   flat               75.2
 *       8   256   flat               75.9
 *      16    16   flat              146.5
 *      16    24   flat              148.3
 *      16   272   flat              148.8
 *      16   512   log3              301.0
 *      32    32   t1s               284.7
 *      32    40   t1s               284.4
 *      32   544   flat              299.3
 *      32  1024   flat              590.3
 *      64    64   t1s               559.0
 *      64    72   t1s               563.0
 *      64  1088   flat              597.0
 *      64  2048   t1s              1096.2
 *     128   128   log3             1227.6
 *     128   136   log3             1229.1
 *     128  2176   t1s              1169.5
 *     128  4096   t1s              2245.3
 *     256   256   log3             4415.0
 *     256   264   t1s              2638.2
 *     256  4352   t1s              2439.5
 *     256  8192   t1s              5305.4
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix17_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {128} */
    if (me == 128) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix17_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {32, 64, 128, 256} */
    if (me >= 32 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R17_PLAN_WISDOM_H */
