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
 *       8     8   flat               78.5
 *       8    16   flat               79.9
 *       8   200   flat               79.1
 *       8   256   flat              124.7
 *      16    16   flat              152.3
 *      16    24   t1s               165.7
 *      16   400   flat              159.3
 *      16   512   t1s               338.6
 *      32    32   flat              298.9
 *      32    40   flat              295.6
 *      32   800   flat              300.1
 *      32  1024   t1s               653.4
 *      64    64   t1s               588.1
 *      64    72   t1s               609.9
 *      64  1600   flat              675.2
 *      64  2048   t1s              1427.5
 *     128   128   t1s              1215.4
 *     128   136   t1s              1225.3
 *     128  3200   t1s              1306.2
 *     128  4096   t1s              2702.0
 *     256   256   t1s              5083.2
 *     256   264   t1s              2607.1
 *     256  6400   t1s              5575.8
 *     256  8192   t1s              6697.0
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix25_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed log3 never wins on this host. */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix25_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {16, 64, 128, 256} */
    if (me >= 16 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R25_PLAN_WISDOM_H */
