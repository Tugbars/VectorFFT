/* vfft_r11_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=11.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * log3 queries come in three flavors:
 *   radix11_prefer_dit_log3   — DIT-log3 codelet won
 *   radix11_prefer_dif_log3   — DIF-log3 codelet won
 *   radix11_prefer_log3       — either of the above (union)
 * A planner whose forward executor is DIT-structured should query
 * prefer_dit_log3 to avoid activating log3 on cells where only
 * DIF-log3 was the winner (it would substitute DIT-log3 there).
 *
 * Derived from cross-protocol comparison at each sweep point. The
 * planner should consult these AFTER the codelet-level dispatcher
 * has been selected — they drive twiddle-table layout and the
 * K-blocked execution path.
 */
#ifndef VFFT_R11_PLAN_WISDOM_H
#define VFFT_R11_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *       8     8   t1s            -                 92.2
 *       8    16   t1s            -                 92.4
 *       8    88   t1s            -                 92.3
 *       8   256   t1s            -                 92.6
 *      16    16   t1s            -                181.1
 *      16    24   t1s            -                182.4
 *      16   176   t1s            -                181.0
 *      16   512   log3           dit              288.4
 *      32    32   t1s            -                358.1
 *      32    40   t1s            -                360.6
 *      32   352   t1s            -                352.3
 *      32  1024   flat           -                516.1
 *      64    64   t1s            -                697.2
 *      64    72   t1s            -                747.1
 *      64   704   t1s            -                711.5
 *      64  2048   log3           dit             1111.1
 *     128   128   t1s            -               1387.2
 *     128   136   t1s            -               1396.8
 *     128  1408   t1s            -               1399.7
 *     128  4096   log3           dit             2232.4
 *     256   256   t1s            -               2791.2
 *     256   264   t1s            -               2797.7
 *     256  2816   t1s            -               3363.5
 *     256  8192   log3           dit             4956.8
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix11_prefer_dit_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix11_prefer_dif_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix11_prefer_log3(size_t me, size_t ios) {
    return radix11_prefer_dit_log3(me, ios)
        || radix11_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix11_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8, 16, 32, 64, 128, 256} */
    if (me >= 8 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R11_PLAN_WISDOM_H */
