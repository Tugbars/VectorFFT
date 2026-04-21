/* vfft_r16_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=16.
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
#ifndef VFFT_R16_PLAN_WISDOM_H
#define VFFT_R16_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      64    64   t1s               249.0
 *      64    72   flat              278.7
 *      64   512   flat              836.7
 *     128   128   t1s               575.2
 *     128   136   t1s               495.6
 *     128  1024   t1s              1639.7
 *     256   256   flat             3145.6
 *     256   264   log3             1576.4
 *     256  2048   flat             3682.6
 *     512   512   flat             7228.8
 *     512   520   log3             3338.3
 *     512  4096   flat             6814.8
 *    1024  1024   flat            14700.8
 *    1024  1032   flat             6905.0
 *    1024  8192   flat            15601.6
 *    2048  2048   flat            27585.4
 *    2048  2056   log3            12597.8
 *    2048 16384   log3            34642.8
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix16_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {2048} */
    if (me == 2048) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix16_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {128} */
    if (me == 128) return 1;
    return 0;
}

#endif /* VFFT_R16_PLAN_WISDOM_H */
