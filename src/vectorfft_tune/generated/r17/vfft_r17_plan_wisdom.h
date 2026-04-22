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
 *       8     8   flat               74.4
 *       8    16   flat               76.5
 *       8   136   flat               76.5
 *       8   256   flat               77.4
 *      16    16   flat              148.0
 *      16    24   flat              147.9
 *      16   272   flat              152.6
 *      16   512   flat              299.6
 *      32    32   flat              294.0
 *      32    40   flat              289.0
 *      32   544   flat              304.9
 *      32  1024   flat              609.6
 *      64    64   flat              569.9
 *      64    72   log3              642.6
 *      64  1088   flat              607.3
 *      64  2048   t1s              1110.9
 *     128   128   t1s              1150.8
 *     128   136   flat             1318.8
 *     128  2176   flat             1357.5
 *     128  4096   t1s              2321.3
 *     256   256   log3             4576.7
 *     256   264   flat             2796.2
 *     256  4352   flat             2909.3
 *     256  8192   t1s              5695.3
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix17_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix17_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {128} */
    if (me == 128) return 1;
    return 0;
}

#endif /* VFFT_R17_PLAN_WISDOM_H */
