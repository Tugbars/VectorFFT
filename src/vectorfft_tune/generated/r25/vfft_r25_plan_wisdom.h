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
 *       8     8   flat               80.0
 *       8    16   flat               82.7
 *       8   200   flat               82.3
 *       8   256   flat              139.8
 *      16    16   flat              153.8
 *      16    24   flat              154.7
 *      16   400   flat              160.5
 *      16   512   log3              344.2
 *      32    32   flat              301.1
 *      32    40   t1s               303.8
 *      32   800   flat              307.8
 *      32  1024   t1s               696.5
 *      64    64   t1s               616.6
 *      64    72   t1s               628.5
 *      64  1600   t1s               640.6
 *      64  2048   t1s              1359.4
 *     128   128   t1s              1239.7
 *     128   136   t1s              1287.7
 *     128  3200   t1s              1354.5
 *     128  4096   t1s              2730.0
 *     256   256   t1s              4916.1
 *     256   264   t1s              2690.0
 *     256  6400   flat             5970.8
 *     256  8192   flat             7004.5
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix25_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix25_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {32, 64, 128, 256} */
    if (me >= 32 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R25_PLAN_WISDOM_H */
