/* vfft_r4_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=4.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * Predicates:
 *   radix4_prefer_dit_log3 — DIT-log3 codelet won as log3 representative
 *                              AND log3 was cross-protocol winner.
 *   radix4_prefer_dif_log3 — collapsed to always-0 in v1.0 (DIF filter).
 *   radix4_prefer_log3     — union of the above (= dit_log3 in v1.0).
 *   radix4_prefer_t1s      — t1s protocol won.
 *   radix4_prefer_buf      — flat won cross-protocol AND t1_buf_dit
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
#ifndef VFFT_R4_PLAN_WISDOM_H
#define VFFT_R4_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *      64    64   flat           -                 29.4
 *      64    72   t1s            -                 29.6
 *      64   512   t1s            -                 29.6
 *      96    96   t1s            -                 45.6
 *      96   104   t1s            -                 43.9
 *      96   768   t1s            -                 44.0
 *     128   128   t1s            -                 58.4
 *     128   136   t1s            -                 58.4
 *     128  1024   t1s            -                 57.6
 *     192   192   flat           -                 89.0
 *     192   200   flat           -                 90.9
 *     192  1536   flat           -                 90.4
 *     256   256   flat           -                118.8
 *     256   264   flat           -                121.4
 *     256  2048   flat           -                119.9
 *     384   384   flat           -                180.6
 *     384   392   flat           -                179.0
 *     384  3072   flat           -                181.7
 *     512   512   flat           -                286.5
 *     512   520   flat           -                310.9
 *     512  4096   flat           -                289.7
 *     768   768   log3           dit              531.1
 *     768   776   log3           dit              573.1
 *     768  6144   log3           dit              522.2
 *    1024  1024   log3           dit              772.1
 *    1024  1032   log3           dit              745.1
 *    1024  8192   log3           dit              757.7
 *    1536  1536   flat           -               1152.6
 *    1536  1544   flat           -               1144.9
 *    1536 12288   log3           dit             1126.9
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix4_prefer_dit_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {768, 1024, 2048} */
    if (me == 768 || me == 1024 || me == 2048) return 1;
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix4_prefer_dif_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix4_prefer_log3(size_t me, size_t ios) {
    return radix4_prefer_dit_log3(me, ios)
        || radix4_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix4_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 96, 128} */
    if (me == 64 || me == 96 || me == 128) return 1;
    return 0;
}

/* Should the planner use t1_buf_dit (buffered flat) at (me, ios)?
 * True when the flat protocol wins cross-protocol AND the t1_buf_dit
 * dispatcher beats t1_dit baseline within the flat protocol (per the
 * within-flat per-dispatcher comparison with 2%% tie threshold). */
static inline int radix4_prefer_buf(size_t me, size_t ios) {
    (void)me; (void)ios;
    /* No t1_buf_dit dispatcher in this radix's portfolio. */
    return 0;
}

#endif /* VFFT_R4_PLAN_WISDOM_H */
