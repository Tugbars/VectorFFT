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
 *      64    64   flat               90.0
 *      64    72   flat               91.6
 *      64   512   flat              232.9
 *     128   128   flat              171.7
 *     128   136   flat              177.1
 *     128  1024   flat              421.7
 *     256   256   flat              460.0
 *     256   264   flat              419.0
 *     256  2048   flat              831.1
 *     512   512   flat             1811.3
 *     512   520   flat             1028.1
 *     512  4096   flat             1745.0
 *    1024  1024   flat             3675.7
 *    1024  1032   flat             2120.5
 *    1024  8192   flat             3451.8
 *    2048  2048   flat             7285.2
 *    2048  2056   flat             4459.6
 *    2048 16384   flat            14874.4
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
