/* vfft_r6_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=6.
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
#ifndef VFFT_R6_PLAN_WISDOM_H
#define VFFT_R6_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   t1s                10.8
 *       8    16   flat               11.2
 *       8    80   t1s                11.0
 *       8   256   t1s                11.4
 *      16    16   t1s                19.7
 *      16    24   t1s                19.9
 *      16   160   t1s                20.0
 *      16   512   t1s                19.9
 *      32    32   t1s                36.7
 *      32    40   t1s                37.9
 *      32   320   t1s                36.5
 *      32  1024   t1s                36.6
 *      64    64   t1s                72.3
 *      64    72   t1s                73.1
 *      64   640   t1s                72.0
 *      64  2048   t1s                72.8
 *     128   128   flat              141.3
 *     128   136   flat              140.2
 *     128  1280   flat              139.0
 *     128  4096   t1s               150.0
 *     256   256   flat              277.1
 *     256   264   flat              278.4
 *     256  2560   t1s               379.9
 *     256  8192   t1s               360.5
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix6_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed log3 never wins on this host. */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix6_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8, 16, 32, 64, 256} */
    if (me >= 8 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R6_PLAN_WISDOM_H */
