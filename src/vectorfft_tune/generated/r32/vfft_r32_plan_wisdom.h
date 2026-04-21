/* vfft_r32_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=32.
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
#ifndef VFFT_R32_PLAN_WISDOM_H
#define VFFT_R32_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      64    64   t1s               705.3
 *      64    72   t1s               655.5
 *      64   512   t1s              1862.3
 *     128   128   t1s              2260.6
 *     128   136   t1s              1431.7
 *     128  1024   flat             3802.3
 *     256   256   log3             6359.6
 *     256   264   log3             4266.2
 *     256  2048   log3             7964.4
 *     512   512   flat            18073.6
 *     512   520   log3             8365.6
 *     512  4096   flat            18225.9
 *    1024  1024   flat            31035.7
 *    1024  1032   log3            16638.1
 *    1024  8192   flat            40418.8
 *    2048  2048   flat            67232.8
 *    2048  2056   log3            36124.6
 *    2048 16384   log3           117947.7
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix32_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256, 2048} */
    if (me == 256 || me == 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix32_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 128} */
    if (me == 64 || me == 128) return 1;
    return 0;
}

#endif /* VFFT_R32_PLAN_WISDOM_H */
