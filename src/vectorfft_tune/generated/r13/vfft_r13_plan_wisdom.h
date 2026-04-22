/* vfft_r13_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=13.
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
#ifndef VFFT_R13_PLAN_WISDOM_H
#define VFFT_R13_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   flat               51.7
 *       8    16   flat               49.8
 *       8   104   flat               52.1
 *       8   256   flat               52.1
 *      16    16   flat               90.4
 *      16    24   flat               95.3
 *      16   208   flat              103.5
 *      16   512   log3              219.7
 *      32    32   flat              177.8
 *      32    40   t1s               176.9
 *      32   416   t1s               185.3
 *      32  1024   log3              418.2
 *      64    64   flat              379.1
 *      64    72   t1s               337.0
 *      64   832   t1s               375.9
 *      64  2048   log3              849.6
 *     128   128   t1s               687.9
 *     128   136   t1s               772.9
 *     128  1664   log3              818.4
 *     128  4096   log3             1690.2
 *     256   256   log3             1580.9
 *     256   264   t1s              1390.7
 *     256  3328   log3             2036.4
 *     256  8192   log3             3796.7
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix13_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {128, 256} */
    if (me == 128 || me == 256) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix13_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {32, 64, 128} */
    if (me == 32 || me == 64 || me == 128) return 1;
    return 0;
}

#endif /* VFFT_R13_PLAN_WISDOM_H */
