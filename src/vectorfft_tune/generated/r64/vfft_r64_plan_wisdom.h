/* vfft_r64_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=64.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * Predicates:
 *   radix64_prefer_dit_log3 — DIT-log3 codelet won as log3 representative
 *                              AND log3 was cross-protocol winner.
 *   radix64_prefer_dif_log3 — collapsed to always-0 in v1.0 (DIF filter).
 *   radix64_prefer_log3     — union of the above (= dit_log3 in v1.0).
 *   radix64_prefer_t1s      — t1s protocol won.
 *   radix64_prefer_buf      — flat won cross-protocol AND t1_buf_dit
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
#ifndef VFFT_R64_PLAN_WISDOM_H
#define VFFT_R64_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *      64    64   t1s            -               2542.4
 *      64    72   t1s            -               1867.2
 *      64   512   t1s            -               4731.2
 *      96    96   t1s            -               3109.9
 *      96   104   t1s            -               2707.4
 *      96   768   flat           -               6966.6
 *     128   128   t1s            -               9048.3
 *     128   136   t1s            -               3877.0
 *     128  1024   t1s            -              10020.1
 *     192   192   log3           dit             8295.3
 *     192   200   log3           dit             7620.5
 *     192  1536   flat           -              16055.0
 *     256   256   flat           -              19504.7
 *     256   264   log3           dit             9914.9
 *     256  2048   log3           dit            24025.4
 *     384   384   log3           dit            29857.2
 *     384   392   log3           dit            15170.1
 *     384  3072   log3           dit            35137.7
 *     512   512   log3           dit            41818.4
 *     512   520   log3           dit            18932.4
 *     512  4096   log3           dit            49456.6
 *     768   768   log3           dit            56160.2
 *     768   776   log3           dit            30422.5
 *     768  6144   log3           dit            71559.4
 *    1024  1024   log3           dit            84084.4
 *    1024  1032   log3           dit            40150.8
 *    1024  8192   flat           -             161673.4
 *    1536  1536   log3           dit           120911.7
 *    1536  1544   log3           dit            61530.1
 *    1536 12288   log3           dit           155503.1
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix64_prefer_dit_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {192, 256, 384, 512, 768, 1024, 1536} */
    if (me >= 192 && me <= 1536) return 1;
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix64_prefer_dif_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix64_prefer_log3(size_t me, size_t ios) {
    return radix64_prefer_dit_log3(me, ios)
        || radix64_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix64_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 96, 128} */
    if (me == 64 || me == 96 || me == 128) return 1;
    return 0;
}

/* Should the planner use t1_buf_dit (buffered flat) at (me, ios)?
 * True when the flat protocol wins cross-protocol AND the t1_buf_dit
 * dispatcher beats t1_dit baseline within the flat protocol (per the
 * within-flat per-dispatcher comparison with 2%% tie threshold). */
static inline int radix64_prefer_buf(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

#endif /* VFFT_R64_PLAN_WISDOM_H */
