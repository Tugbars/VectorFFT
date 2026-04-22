/* vfft_r10_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=10.
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
#ifndef VFFT_R10_PLAN_WISDOM_H
#define VFFT_R10_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   flat               25.4
 *       8    16   flat               25.2
 *       8    80   flat               25.3
 *       8   256   flat               26.0
 *      16    16   flat               45.5
 *      16    24   flat               45.9
 *      16   160   flat               46.1
 *      16   512   flat              146.2
 *      32    32   flat               88.4
 *      32    40   flat               86.6
 *      32   320   flat               88.2
 *      32  1024   t1s               220.6
 *      64    64   t1s               178.2
 *      64    72   flat              169.6
 *      64   640   flat              184.7
 *      64  2048   t1s               367.3
 *     128   128   flat              337.2
 *     128   136   flat              341.3
 *     128  1280   t1s               382.2
 *     128  4096   t1s               692.7
 *     256   256   t1s               750.4
 *     256   264   t1s               720.9
 *     256  2560   t1s              1458.1
 *     256  8192   t1s               914.6
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix10_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed log3 never wins on this host. */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix10_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 128, 256} */
    if (me == 64 || me == 128 || me == 256) return 1;
    return 0;
}

#endif /* VFFT_R10_PLAN_WISDOM_H */
