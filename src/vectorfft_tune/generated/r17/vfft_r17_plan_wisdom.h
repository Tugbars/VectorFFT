/* vfft_r17_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=17.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * log3 queries come in three flavors:
 *   radix17_prefer_dit_log3   — DIT-log3 codelet won
 *   radix17_prefer_dif_log3   — DIF-log3 codelet won
 *   radix17_prefer_log3       — either of the above (union)
 * A planner whose forward executor is DIT-structured should query
 * prefer_dit_log3 to avoid activating log3 on cells where only
 * DIF-log3 was the winner (it would substitute DIT-log3 there).
 *
 * Derived from cross-protocol comparison at each sweep point. The
 * planner should consult these AFTER the codelet-level dispatcher
 * has been selected — they drive twiddle-table layout and the
 * K-blocked execution path.
 */
#ifndef VFFT_R17_PLAN_WISDOM_H
#define VFFT_R17_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *       8     8   t1s            -                173.6
 *       8    16   t1s            -                180.1
 *       8   136   t1s            -                174.3
 *       8   256   t1s            -                178.5
 *      16    16   flat           -                380.3
 *      16    24   t1s            -                337.9
 *      16   272   t1s            -                337.4
 *      16   512   log3           dit              556.0
 *      32    32   t1s            -                670.6
 *      32    40   t1s            -                688.4
 *      32   544   t1s            -                678.9
 *      32  1024   log3           dit             1293.2
 *      64    64   t1s            -               1318.5
 *      64    72   t1s            -               1347.5
 *      64  1088   t1s            -               1375.7
 *      64  2048   t1s            -               2662.6
 *     128   128   t1s            -               2787.7
 *     128   136   t1s            -               2764.9
 *     128  2176   t1s            -               2820.8
 *     128  4096   log3           dit             5253.0
 *     256   256   log3           dit             6642.2
 *     256   264   t1s            -               5449.5
 *     256  4352   log3           dit             8898.2
 *     256  8192   log3           dit            10405.9
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix17_prefer_dit_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256} */
    if (me == 256) return 1;
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix17_prefer_dif_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix17_prefer_log3(size_t me, size_t ios) {
    return radix17_prefer_dit_log3(me, ios)
        || radix17_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix17_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8, 16, 32, 64, 128} */
    if (me >= 8 && me <= 128) return 1;
    return 0;
}

#endif /* VFFT_R17_PLAN_WISDOM_H */
