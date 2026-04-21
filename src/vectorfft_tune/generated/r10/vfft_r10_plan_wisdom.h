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
 *       8     8   flat               27.9
 *       8    16   flat               26.6
 *       8    80   flat               27.3
 *       8   256   flat               28.2
 *      16    16   flat               50.4
 *      16    24   flat               47.9
 *      16   160   flat               47.3
 *      16   512   t1s               163.0
 *      32    32   flat               96.8
 *      32    40   flat               88.6
 *      32   320   flat               93.3
 *      32  1024   t1s               217.7
 *      64    64   flat              202.0
 *      64    72   flat              176.9
 *      64   640   flat              188.5
 *      64  2048   t1s               387.2
 *     128   128   flat              360.9
 *     128   136   t1s               346.6
 *     128  1280   t1s               390.2
 *     128  4096   flat              785.6
 *     256   256   t1s               805.5
 *     256   264   t1s               714.8
 *     256  2560   t1s              1708.5
 *     256  8192   t1s               858.5
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
    /* Bench wins at me ∈ {128, 256} */
    if (me == 128 || me == 256) return 1;
    return 0;
}

#endif /* VFFT_R10_PLAN_WISDOM_H */
