/* vfft_r19_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=19.
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
#ifndef VFFT_R19_PLAN_WISDOM_H
#define VFFT_R19_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   flat               97.7
 *       8    16   log3              111.7
 *       8   152   flat              113.0
 *       8   256   t1s               107.2
 *      16    16   t1s               197.6
 *      16    24   flat              192.4
 *      16   304   flat              215.1
 *      16   512   log3              368.3
 *      32    32   flat              379.8
 *      32    40   flat              373.8
 *      32   608   t1s               399.1
 *      32  1024   flat              693.6
 *      64    64   flat              753.7
 *      64    72   flat              753.2
 *      64  1216   flat              812.8
 *      64  2048   log3             1422.4
 *     128   128   t1s              1509.6
 *     128   136   log3             1681.7
 *     128  2432   t1s              1650.5
 *     128  4096   t1s              2714.7
 *     256   256   t1s              4912.4
 *     256   264   log3             3609.2
 *     256  4864   t1s              3345.7
 *     256  8192   log3             7867.3
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix19_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256} */
    if (me == 256) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix19_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {128, 256} */
    if (me == 128 || me == 256) return 1;
    return 0;
}

#endif /* VFFT_R19_PLAN_WISDOM_H */
