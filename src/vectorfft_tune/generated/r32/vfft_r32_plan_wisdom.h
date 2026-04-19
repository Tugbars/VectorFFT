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
 *      64    64   t1s               662.0
 *      64    72   t1s               728.2
 *      64   512   flat             1898.6
 *     128   128   t1s              2408.2
 *     128   136   log3             1737.8
 *     128  1024   t1s              3697.7
 *     256   256   flat             6481.8
 *     256   264   log3             3920.4
 *     256  2048   flat             7963.8
 *     512   512   log3            17763.2
 *     512   520   log3             7840.4
 *     512  4096   flat            17954.3
 *    1024  1024   flat            35335.7
 *    1024  1032   log3            17347.3
 *    1024  8192   flat            43575.8
 *    2048  2048   flat            64714.8
 *    2048  2056   log3            40224.6
 *    2048 16384   flat           117681.2
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix32_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {512} */
    if (me == 512) return 1;
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
