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
 *      64    64   flat               90.4
 *      64    72   flat               91.3
 *      64   512   flat              233.0
 *     128   128   flat              178.4
 *     128   136   t1s               167.2
 *     128  1024   flat              419.2
 *     256   256   flat              410.8
 *     256   264   flat              402.9
 *     256  2048   log3              876.2
 *     512   512   log3             1703.0
 *     512   520   log3              938.6
 *     512  4096   flat             1739.5
 *    1024  1024   flat             3383.0
 *    1024  1032   log3             1927.6
 *    1024  8192   flat             3458.4
 *    2048  2048   log3             6672.3
 *    2048  2056   log3             3909.5
 *    2048 16384   flat            15096.6
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix8_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {512, 2048} */
    if (me == 512 || me == 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix8_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

#endif /* VFFT_R8_PLAN_WISDOM_H */
