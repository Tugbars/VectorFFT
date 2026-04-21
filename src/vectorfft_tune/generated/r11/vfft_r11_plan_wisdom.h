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
 *       8     8   flat               39.3
 *       8    16   flat               40.6
 *       8    88   flat               46.3
 *       8   256   flat               40.2
 *      16    16   flat               72.7
 *      16    24   t1s                73.0
 *      16   176   flat               70.4
 *      16   512   flat              199.7
 *      32    32   flat              138.7
 *      32    40   flat              135.4
 *      32   352   flat              138.0
 *      32  1024   log3              313.7
 *      64    64   flat              292.0
 *      64    72   flat              264.1
 *      64   704   t1s               276.2
 *      64  2048   log3              668.7
 *     128   128   flat              538.0
 *     128   136   flat              503.3
 *     128  1408   t1s               595.5
 *     128  4096   t1s              1371.1
 *     256   256   t1s              1134.3
 *     256   264   log3             1256.1
 *     256  2816   t1s              1238.5
 *     256  8192   log3             1471.5
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
    /* Bench wins at me ∈ {128, 256} */
    if (me == 128 || me == 256) return 1;
    return 0;
}

#endif /* VFFT_R11_PLAN_WISDOM_H */
