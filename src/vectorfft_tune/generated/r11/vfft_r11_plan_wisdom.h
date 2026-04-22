/* vfft_r11_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=11.
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
#ifndef VFFT_R11_PLAN_WISDOM_H
#define VFFT_R11_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   flat               37.8
 *       8    16   flat               37.7
 *       8    88   t1s                39.1
 *       8   256   flat               38.0
 *      16    16   flat               67.8
 *      16    24   flat               66.7
 *      16   176   t1s                69.3
 *      16   512   t1s               173.7
 *      32    32   flat              126.6
 *      32    40   flat              126.7
 *      32   352   flat              128.8
 *      32  1024   log3              300.4
 *      64    64   t1s               252.3
 *      64    72   flat              263.7
 *      64   704   flat              257.8
 *      64  2048   flat              690.4
 *     128   128   flat              506.5
 *     128   136   t1s               487.6
 *     128  1408   t1s               502.1
 *     128  4096   log3             1353.7
 *     256   256   t1s              1124.1
 *     256   264   log3             1247.3
 *     256  2816   t1s              1221.8
 *     256  8192   t1s              1395.4
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix11_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix11_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {16, 128, 256} */
    if (me == 16 || me == 128 || me == 256) return 1;
    return 0;
}

#endif /* VFFT_R11_PLAN_WISDOM_H */
