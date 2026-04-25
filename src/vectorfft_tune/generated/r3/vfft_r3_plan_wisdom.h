/* vfft_r3_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=3.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * Predicates:
 *   radix3_prefer_dit_log3 — DIT-log3 codelet won as log3 representative
 *                              AND log3 was cross-protocol winner.
 *   radix3_prefer_dif_log3 — collapsed to always-0 in v1.0 (DIF filter).
 *   radix3_prefer_log3     — union of the above (= dit_log3 in v1.0).
 *   radix3_prefer_t1s      — t1s protocol won.
 *   radix3_prefer_buf      — flat won cross-protocol AND t1_buf_dit
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
#ifndef VFFT_R3_PLAN_WISDOM_H
#define VFFT_R3_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *      24    24   t1s            -                 25.1
 *      24    32   t1s            -                 24.8
 *      24    72   t1s            -                 24.6
 *      24   192   t1s            -                 24.4
 *      48    48   t1s            -                 47.1
 *      48    56   t1s            -                 47.0
 *      48   144   t1s            -                 47.2
 *      48   384   flat           -                 47.7
 *      96    96   flat           -                 93.5
 *      96   104   flat           -                 93.1
 *      96   288   t1s            -                 93.5
 *      96   768   t1s            -                 92.7
 *     192   192   log3           dit              185.8
 *     192   200   flat           -                184.7
 *     192   576   t1s            -                182.7
 *     192  1536   log3           dit              182.8
 *     384   384   t1s            -                363.9
 *     384   392   t1s            -                366.2
 *     384  1152   flat           -                367.3
 *     384  3072   t1s            -                363.8
 *     768   768   log3           dit              766.4
 *     768   776   t1s            -                725.9
 *     768  2304   t1s            -                736.9
 *     768  6144   t1s            -                736.6
 *    1536  1536   t1s            -               1611.1
 *    1536  1544   t1s            -               1730.7
 *    1536  4608   t1s            -               1834.8
 *    1536 12288   t1s            -               1590.3
 *    3072  3072   t1s            -               3402.4
 *    3072  3080   t1s            -               3392.6
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix3_prefer_dit_log3(size_t me, size_t ios) {
    /* Sparse DIT-log3 wins at 1 specific (me, ios) cells: (768,768) */
    if ((me == 768 && ios == 768)) return 1;
    /* Bench wins at me ∈ {192} */
    if (me == 192) return 1;
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix3_prefer_dif_log3(size_t me, size_t ios) {
    (void)me; (void)ios;
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix3_prefer_log3(size_t me, size_t ios) {
    return radix3_prefer_dit_log3(me, ios)
        || radix3_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix3_prefer_t1s(size_t me, size_t ios) {
    /* Sparse t1s wins at 1 specific (me, ios) cells: (192,576) */
    if ((me == 192 && ios == 576)) return 1;
    /* Bench wins at me ∈ {24, 48, 96, 384, 768, 1536, 3072} */
    if (me >= 24 && me <= 3072) return 1;
    return 0;
}

/* Should the planner use t1_buf_dit (buffered flat) at (me, ios)?
 * True when the flat protocol wins cross-protocol AND the t1_buf_dit
 * dispatcher beats t1_dit baseline within the flat protocol (per the
 * within-flat per-dispatcher comparison with 2%% tie threshold). */
static inline int radix3_prefer_buf(size_t me, size_t ios) {
    (void)me; (void)ios;
    /* No t1_buf_dit dispatcher in this radix's portfolio. */
    return 0;
}

#endif /* VFFT_R3_PLAN_WISDOM_H */
