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
 *       8     8   flat               38.1
 *       8    16   flat               40.6
 *       8    88   flat               43.0
 *       8   256   flat               40.4
 *      16    16   flat               70.3
 *      16    24   flat               74.2
 *      16   176   t1s                69.8
 *      16   512   log3              207.0
 *      32    32   flat              127.6
 *      32    40   flat              147.8
 *      32   352   flat              140.7
 *      32  1024   log3              312.4
 *      64    64   flat              292.6
 *      64    72   flat              254.3
 *      64   704   flat              263.5
 *      64  2048   t1s               662.8
 *     128   128   flat              557.6
 *     128   136   flat              491.7
 *     128  1408   flat              591.3
 *     128  4096   log3             1386.0
 *     256   256   t1s              1167.3
 *     256   264   t1s              1011.6
 *     256  2816   t1s              1220.4
 *     256  8192   log3             1804.2
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix11_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix11_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256} */
    if (me == 256) return 1;
    return 0;
}

#endif /* VFFT_R11_PLAN_WISDOM_H */
