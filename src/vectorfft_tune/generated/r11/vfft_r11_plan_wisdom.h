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
 *       8     8   flat               37.5
 *       8    16   flat               38.6
 *       8    88   flat               37.3
 *       8   256   flat               37.0
 *      16    16   flat               69.5
 *      16    24   t1s                68.5
 *      16   176   flat               67.0
 *      16   512   log3              176.6
 *      32    32   flat              134.7
 *      32    40   flat              127.9
 *      32   352   t1s               128.9
 *      32  1024   flat              297.9
 *      64    64   flat              248.1
 *      64    72   t1s               249.4
 *      64   704   t1s               252.5
 *      64  2048   t1s               672.4
 *     128   128   flat              486.8
 *     128   136   t1s               488.1
 *     128  1408   t1s               493.2
 *     128  4096   log3             1327.7
 *     256   256   log3             1239.8
 *     256   264   t1s               969.7
 *     256  2816   t1s              1021.6
 *     256  8192   log3             1802.5
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix11_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256} */
    if (me == 256) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix11_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 128, 256} */
    if (me == 64 || me == 128 || me == 256) return 1;
    return 0;
}

#endif /* VFFT_R11_PLAN_WISDOM_H */
