/* vfft_r6_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=6.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * log3 queries come in three flavors:
 *   radix6_prefer_dit_log3   — DIT-log3 codelet won
 *   radix6_prefer_dif_log3   — DIF-log3 codelet won
 *   radix6_prefer_log3       — either of the above (union)
 * A planner whose forward executor is DIT-structured should query
 * prefer_dit_log3 to avoid activating log3 on cells where only
 * DIF-log3 was the winner (it would substitute DIT-log3 there).
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
 *   me    ios   winning_protocol   log3_family   winning_ns
 *       8     8   t1s            -                 24.8
 *       8    16   t1s            -                 24.9
 *       8    80   flat           -                 25.4
 *       8   256   flat           -                 27.1
 *      16    16   t1s            -                 41.5
 *      16    24   t1s            -                 42.1
 *      16   160   t1s            -                 44.5
 *      16   512   t1s            -                 44.6
 *      32    32   t1s            -                 81.6
 *      32    40   t1s            -                 83.8
 *      32   320   t1s            -                 84.2
 *      32  1024   t1s            -                 83.0
 *      64    64   t1s            -                156.4
 *      64    72   t1s            -                164.5
 *      64   640   t1s            -                156.0
 *      64  2048   t1s            -                160.5
 *     128   128   t1s            -                326.3
 *     128   136   t1s            -                313.3
 *     128  1280   t1s            -                314.2
 *     128  4096   t1s            -                397.1
 *     256   256   t1s            -                632.2
 *     256   264   t1s            -                611.2
 *     256  2560   t1s            -                780.0
 *     256  8192   t1s            -                945.8
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix6_prefer_dit_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIT-log3 never wins on this host. */
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix6_prefer_dif_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix6_prefer_log3(size_t me, size_t ios) {
    return radix6_prefer_dit_log3(me, ios)
        || radix6_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix6_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8, 16, 32, 64, 128, 256} */
    if (me >= 8 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R6_PLAN_WISDOM_H */
