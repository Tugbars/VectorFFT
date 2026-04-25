/* vfft_r12_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=12.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * log3 queries come in three flavors:
 *   radix12_prefer_dit_log3   — DIT-log3 codelet won
 *   radix12_prefer_dif_log3   — DIF-log3 codelet won
 *   radix12_prefer_log3       — either of the above (union)
 * A planner whose forward executor is DIT-structured should query
 * prefer_dit_log3 to avoid activating log3 on cells where only
 * DIF-log3 was the winner (it would substitute DIT-log3 there).
 *
 * Derived from cross-protocol comparison at each sweep point. The
 * planner should consult these AFTER the codelet-level dispatcher
 * has been selected — they drive twiddle-table layout and the
 * K-blocked execution path.
 */
#ifndef VFFT_R12_PLAN_WISDOM_H
#define VFFT_R12_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *       8     8   t1s            -                 64.7
 *       8    16   t1s            -                 64.0
 *       8    96   t1s            -                 63.8
 *       8   256   t1s            -                 62.6
 *      16    16   t1s            -                122.6
 *      16    24   t1s            -                130.7
 *      16   192   t1s            -                126.4
 *      16   512   flat           -                305.9
 *      32    32   t1s            -                267.5
 *      32    40   t1s            -                248.3
 *      32   384   t1s            -                247.1
 *      32  1024   flat           -                612.0
 *      64    64   t1s            -                515.7
 *      64    72   t1s            -                497.8
 *      64   768   t1s            -                505.0
 *      64  2048   log3           dit             1585.1
 *     128   128   t1s            -                969.6
 *     128   136   t1s            -               1066.0
 *     128  1536   log3           dit             3105.6
 *     128  4096   log3           dit             2878.8
 *     256   256   t1s            -               1990.3
 *     256   264   t1s            -               2071.4
 *     256  3072   log3           dit             5932.7
 *     256  8192   log3           dit             6127.1
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix12_prefer_dit_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {128, 256} */
    if (me == 128 || me == 256) return 1;
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix12_prefer_dif_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix12_prefer_log3(size_t me, size_t ios) {
    return radix12_prefer_dit_log3(me, ios)
        || radix12_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix12_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8, 16, 32, 64, 128, 256} */
    if (me >= 8 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R12_PLAN_WISDOM_H */
