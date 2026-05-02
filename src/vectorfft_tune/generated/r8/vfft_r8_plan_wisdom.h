/* vfft_r8_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=8.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * Predicates:
 *   radix8_prefer_dit_log3 — DIT-log3 codelet won as log3 representative
 *                              AND log3 was cross-protocol winner.
 *   radix8_prefer_dif_log3 — collapsed to always-0 in v1.0 (DIF filter).
 *   radix8_prefer_log3     — union of the above (= dit_log3 in v1.0).
 *   radix8_prefer_t1s      — t1s protocol won.
 *   radix8_prefer_buf      — flat won cross-protocol AND t1_buf_dit
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
#ifndef VFFT_R8_PLAN_WISDOM_H
#define VFFT_R8_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *      64    64   t1s            -                 91.0
 *      64    72   t1s            -                 87.0
 *      64   512   flat           -                114.0
 *      96    96   t1s            -                132.5
 *      96   104   t1s            -                131.6
 *      96   768   flat           -                133.9
 *     128   128   t1s            -                169.1
 *     128   136   t1s            -                175.2
 *     128  1024   flat           -                438.7
 *     192   192   flat           -                276.6
 *     192   200   flat           -                276.8
 *     192  1536   flat           -                668.1
 *     256   256   flat           -                402.2
 *     256   264   flat           -                421.3
 *     256  2048   flat           -                875.9
 *     384   384   log3           dit              738.7
 *     384   392   flat           -                716.8
 *     384  3072   flat           -               1226.1
 *     512   512   flat           -               1113.3
 *     512   520   flat           -               1018.5
 *     512  4096   flat           -               1727.4
 *     768   768   log3           dit             1459.5
 *     768   776   flat           -               1440.4
 *     768  6144   flat           -               2720.7
 *    1024  1024   flat           -               3547.6
 *    1024  1032   flat           -               1983.5
 *    1024  8192   flat           -               4025.6
 *    1536  1536   flat           -               5176.0
 *    1536  1544   log3           dit             2818.1
 *    1536 12288   flat           -               5992.2
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix8_prefer_dit_log3(size_t me, size_t ios) {
    /* Sparse DIT-log3 wins at 4 specific (me, ios) cells: (384,384), (768,768), (1536,1544), (2048,2056) */
    if ((me == 384 && ios == 384) || (me == 768 && ios == 768) || (me == 1536 && ios == 1544) || (me == 2048 && ios == 2056)) return 1;
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix8_prefer_dif_log3(size_t me, size_t ios) {
    (void)me; (void)ios;
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix8_prefer_log3(size_t me, size_t ios) {
    return radix8_prefer_dit_log3(me, ios)
        || radix8_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix8_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* rules are me-only */
    /* Bench wins at me ∈ {64, 96, 128} */
    if (me == 64 || me == 96 || me == 128) return 1;
    return 0;
}

/* Should the planner use t1_buf_dit (buffered flat) at (me, ios)?
 * True when the flat protocol wins cross-protocol AND the t1_buf_dit
 * dispatcher beats t1_dit baseline within the flat protocol (per the
 * within-flat per-dispatcher comparison with 2%% tie threshold). */
static inline int radix8_prefer_buf(size_t me, size_t ios) {
    (void)me; (void)ios;
    /* No t1_buf_dit dispatcher in this radix's portfolio. */
    return 0;
}

#endif /* VFFT_R8_PLAN_WISDOM_H */
