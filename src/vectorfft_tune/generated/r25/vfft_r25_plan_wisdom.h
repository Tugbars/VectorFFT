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
 *       8     8   flat               78.7
 *       8    16   flat               79.8
 *       8   200   log3               92.2
 *       8   256   log3              145.2
 *      16    16   flat              150.9
 *      16    24   flat              155.7
 *      16   400   flat              154.5
 *      16   512   flat              335.0
 *      32    32   flat              307.2
 *      32    40   flat              297.6
 *      32   800   t1s               336.6
 *      32  1024   t1s               700.1
 *      64    64   t1s               604.5
 *      64    72   flat              625.8
 *      64  1600   t1s               616.1
 *      64  2048   t1s              1389.9
 *     128   128   t1s              1258.1
 *     128   136   t1s              1279.3
 *     128  3200   t1s              1393.8
 *     128  4096   flat             2760.5
 *     256   256   t1s              4805.9
 *     256   264   t1s              2751.7
 *     256  6400   t1s              5931.8
 *     256  8192   flat             7100.4
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix25_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8} */
    if (me == 8) return 1;
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
