/* vfft_r4_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=4.
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
#ifndef VFFT_R4_PLAN_WISDOM_H
#define VFFT_R4_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      64    64   t1s                70.9
 *      64    72   t1s                69.2
 *      64   512   t1s                67.9
 *     128   128   t1s               135.0
 *     128   136   t1s               134.3
 *     128  1024   t1s               133.8
 *     256   256   flat              329.4
 *     256   264   flat              320.6
 *     256  2048   flat              324.3
 *     512   512   flat              648.6
 *     512   520   flat              655.3
 *     512  4096   log3              648.4
 *    1024  1024   log3             1585.7
 *    1024  1032   log3             1597.2
 *    1024  8192   log3             1608.1
 *    2048  2048   flat             3325.3
 *    2048  2056   log3             3165.7
 *    2048 16384   log3             3261.4
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix4_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {1024, 2048} */
    if (me == 1024 || me == 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix4_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 128} */
    if (me == 64 || me == 128) return 1;
    return 0;
}

#endif /* VFFT_R4_PLAN_WISDOM_H */
