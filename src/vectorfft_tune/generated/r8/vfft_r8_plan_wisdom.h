/* vfft_r8_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=8.
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
#ifndef VFFT_R8_PLAN_WISDOM_H
#define VFFT_R8_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      64    64   flat              229.6
 *      64    72   flat              232.7
 *      64   512   flat              503.9
 *     128   128   flat              484.8
 *     128   136   flat              492.2
 *     128  1024   flat             1079.6
 *     256   256   flat             1042.5
 *     256   264   flat             1040.3
 *     256  2048   flat             4080.7
 *     512   512   flat             4222.6
 *     512   520   flat             2119.3
 *     512  4096   flat             8568.6
 *    1024  1024   flat             9230.7
 *    1024  1032   flat             3982.3
 *    1024  8192   flat            16879.6
 *    2048  2048   flat            34116.1
 *    2048  2056   flat            14893.7
 *    2048 16384   flat            35213.4
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix8_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed log3 never wins on this host. */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix8_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed t1s never wins on this host. */
    return 0;
}

#endif /* VFFT_R8_PLAN_WISDOM_H */
