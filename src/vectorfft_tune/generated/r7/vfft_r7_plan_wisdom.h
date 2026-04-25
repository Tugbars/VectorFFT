/* vfft_r7_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=7.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * log3 queries come in three flavors:
 *   radix7_prefer_dit_log3   — DIT-log3 codelet won
 *   radix7_prefer_dif_log3   — DIF-log3 codelet won
 *   radix7_prefer_log3       — either of the above (union)
 * A planner whose forward executor is DIT-structured should query
 * prefer_dit_log3 to avoid activating log3 on cells where only
 * DIF-log3 was the winner (it would substitute DIT-log3 there).
 *
 * Derived from cross-protocol comparison at each sweep point. The
 * planner should consult these AFTER the codelet-level dispatcher
 * has been selected — they drive twiddle-table layout and the
 * K-blocked execution path.
 */
#ifndef VFFT_R7_PLAN_WISDOM_H
#define VFFT_R7_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *      56    56   t1s            -                255.0
 *      56    64   t1s            -                253.4
 *      56   392   t1s            -                254.0
 *      56   448   t1s            -                256.2
 *     112   112   t1s            -                506.2
 *     112   120   t1s            -                503.3
 *     112   784   t1s            -                505.9
 *     112   896   t1s            -                507.1
 *     224   224   t1s            -               1010.4
 *     224   232   t1s            -                995.3
 *     224  1568   t1s            -               1010.7
 *     224  1792   t1s            -               1015.8
 *     448   448   t1s            -               2035.2
 *     448   456   t1s            -               1998.5
 *     448  3136   t1s            -               1994.8
 *     448  3584   flat           -               6006.1
 *     896   896   t1s            -               4091.1
 *     896   904   t1s            -               4110.8
 *     896  6272   t1s            -               4206.8
 *     896  7168   t1s            -              11855.0
 *    1792  1792   t1s            -               8387.7
 *    1792  1800   t1s            -               8386.3
 *    1792 12544   t1s            -               8340.6
 *    1792 14336   t1s            -              23575.7
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix7_prefer_dit_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIT-log3 never wins on this host. */
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix7_prefer_dif_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix7_prefer_log3(size_t me, size_t ios) {
    return radix7_prefer_dit_log3(me, ios)
        || radix7_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix7_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {56, 112, 224, 448, 896, 1792} */
    if (me >= 56 && me <= 1792) return 1;
    return 0;
}

#endif /* VFFT_R7_PLAN_WISDOM_H */
