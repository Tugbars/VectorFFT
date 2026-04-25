/* vfft_r13_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=13.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * Predicates:
 *   radix13_prefer_dit_log3 — DIT-log3 codelet won as log3 representative
 *                              AND log3 was cross-protocol winner.
 *   radix13_prefer_dif_log3 — collapsed to always-0 in v1.0 (DIF filter).
 *   radix13_prefer_log3     — union of the above (= dit_log3 in v1.0).
 *   radix13_prefer_t1s      — t1s protocol won.
 *   radix13_prefer_buf      — flat won cross-protocol AND t1_buf_dit
 *                              dispatcher won within flat (vs t1_dit).
 *                              Always-0 for radixes without buf variants.
 *
 * DIF filter (v1.0): DIF variants are excluded from winner selection
 * because the DIT-structured forward executor cannot substitute them
 * per stage. See dit_dif_design_note.md.
 *
 * Derived from cross-protocol comparison at each sweep point. The
 * planner should consult these AFTER the codelet-level dispatcher
 * has been selected — they drive twiddle-table layout and the
 * K-blocked execution path.
 */
#ifndef VFFT_R13_PLAN_WISDOM_H
#define VFFT_R13_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *       8     8   t1s            -                114.4
 *       8    16   t1s            -                117.6
 *       8   104   t1s            -                114.6
 *       8   256   t1s            -                109.7
 *      16    16   t1s            -                212.3
 *      16    24   t1s            -                218.5
 *      16   208   t1s            -                217.9
 *      16   512   t1s            -                379.6
 *      32    32   t1s            -                421.9
 *      32    40   t1s            -                428.9
 *      32   416   t1s            -                422.1
 *      32  1024   log3           dit              793.2
 *      64    64   t1s            -                856.6
 *      64    72   t1s            -                873.3
 *      64   832   t1s            -                858.2
 *      64  2048   t1s            -               1769.5
 *     128   128   t1s            -               1720.3
 *     128   136   t1s            -               1667.1
 *     128  1664   t1s            -               1698.5
 *     128  4096   t1s            -               3564.0
 *     256   256   log3           dit             3956.2
 *     256   264   t1s            -               3335.9
 *     256  3328   t1s            -               5919.4
 *     256  8192   t1s            -               7345.4
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix13_prefer_dit_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix13_prefer_dif_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix13_prefer_log3(size_t me, size_t ios) {
    return radix13_prefer_dit_log3(me, ios)
        || radix13_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix13_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {8, 16, 32, 64, 128, 256} */
    if (me >= 8 && me <= 256) return 1;
    return 0;
}

/* Should the planner use t1_buf_dit (buffered flat) at (me, ios)?
 * True when the flat protocol wins cross-protocol AND the t1_buf_dit
 * dispatcher beats t1_dit baseline within the flat protocol (per the
 * within-flat per-dispatcher comparison with 2%% tie threshold). */
static inline int radix13_prefer_buf(size_t me, size_t ios) {
    (void)me; (void)ios;
    /* No t1_buf_dit dispatcher in this radix's portfolio. */
    return 0;
}

#endif /* VFFT_R13_PLAN_WISDOM_H */
