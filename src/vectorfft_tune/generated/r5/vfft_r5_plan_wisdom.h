/* vfft_r5_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=5.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * log3 queries come in three flavors:
 *   radix5_prefer_dit_log3   — DIT-log3 codelet won
 *   radix5_prefer_dif_log3   — DIF-log3 codelet won
 *   radix5_prefer_log3       — either of the above (union)
 * A planner whose forward executor is DIT-structured should query
 * prefer_dit_log3 to avoid activating log3 on cells where only
 * DIF-log3 was the winner (it would substitute DIT-log3 there).
 *
 * Derived from cross-protocol comparison at each sweep point. The
 * planner should consult these AFTER the codelet-level dispatcher
 * has been selected — they drive twiddle-table layout and the
 * K-blocked execution path.
 */
#ifndef VFFT_R5_PLAN_WISDOM_H
#define VFFT_R5_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *      40    40   t1s            -                 84.4
 *      40    48   t1s            -                 83.7
 *      40   200   t1s            -                 82.7
 *      40   320   t1s            -                 83.2
 *      80    80   t1s            -                161.2
 *      80    88   t1s            -                161.4
 *      80   400   t1s            -                159.3
 *      80   640   t1s            -                162.8
 *     160   160   t1s            -                307.2
 *     160   168   t1s            -                310.3
 *     160   800   t1s            -                303.2
 *     160  1280   t1s            -                310.6
 *     320   320   t1s            -                615.8
 *     320   328   t1s            -                600.1
 *     320  1600   t1s            -                609.9
 *     320  2560   t1s            -                609.7
 *     640   640   t1s            -               1252.8
 *     640   648   t1s            -               1257.5
 *     640  3200   t1s            -               1216.5
 *     640  5120   t1s            -               1249.9
 *    1280  1280   t1s            -               2443.9
 *    1280  1288   t1s            -               2499.8
 *    1280  6400   t1s            -               2631.2
 *    1280 10240   t1s            -               2773.5
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix5_prefer_dit_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIT-log3 never wins on this host. */
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix5_prefer_dif_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix5_prefer_log3(size_t me, size_t ios) {
    return radix5_prefer_dit_log3(me, ios)
        || radix5_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix5_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {40, 80, 160, 320, 640, 1280} */
    if (me >= 40 && me <= 1280) return 1;
    return 0;
}

#endif /* VFFT_R5_PLAN_WISDOM_H */
