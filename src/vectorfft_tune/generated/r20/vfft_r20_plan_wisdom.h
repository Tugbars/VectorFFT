/* vfft_r20_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=20.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * log3 queries come in three flavors:
 *   radix20_prefer_dit_log3   — DIT-log3 codelet won
 *   radix20_prefer_dif_log3   — DIF-log3 codelet won
 *   radix20_prefer_log3       — either of the above (union)
 * A planner whose forward executor is DIT-structured should query
 * prefer_dit_log3 to avoid activating log3 on cells where only
 * DIF-log3 was the winner (it would substitute DIT-log3 there).
 *
 * Derived from cross-protocol comparison at each sweep point. The
 * planner should consult these AFTER the codelet-level dispatcher
 * has been selected — they drive twiddle-table layout and the
 * K-blocked execution path.
 */
#ifndef VFFT_R20_PLAN_WISDOM_H
#define VFFT_R20_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *       8     8   t1s            -                187.3
 *       8    16   t1s            -                191.1
 *       8   160   t1s            -                185.4
 *       8   256   t1s            -                252.2
 *      16    16   t1s            -                358.5
 *      16    24   t1s            -                350.5
 *      16   320   t1s            -                358.9
 *      16   512   t1s            -                663.4
 *      32    32   t1s            -                674.3
 *      32    40   t1s            -                693.0
 *      32   640   t1s            -                734.3
 *      32  1024   t1s            -               1478.1
 *      64    64   t1s            -               1377.2
 *      64    72   t1s            -               1331.7
 *      64  1280   t1s            -               2392.9
 *      64  2048   t1s            -               2932.4
 *     128   128   t1s            -               2868.0
 *     128   136   t1s            -               2760.5
 *     128  2560   t1s            -               5957.1
 *     128  4096   t1s            -               5885.4
 *     256   256   t1s            -               8420.9
 *     256   264   t1s            -               5533.7
 *     256  5120   t1s            -              11890.5
 *     256  8192   t1s            -              12061.3
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix20_prefer_dit_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIT-log3 never wins on this host. */
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix20_prefer_dif_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix20_prefer_log3(size_t me, size_t ios) {
    return radix20_prefer_dit_log3(me, ios)
        || radix20_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix20_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8, 16, 32, 64, 128, 256} */
    if (me >= 8 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R20_PLAN_WISDOM_H */
