/* vfft_r16_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=16.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * Predicates:
 *   radix16_prefer_dit_log3 — DIT-log3 codelet won as log3 representative
 *                              AND log3 was cross-protocol winner.
 *   radix16_prefer_dif_log3 — collapsed to always-0 in v1.0 (DIF filter).
 *   radix16_prefer_log3     — union of the above (= dit_log3 in v1.0).
 *   radix16_prefer_t1s      — t1s protocol won.
 *   radix16_prefer_buf      — flat won cross-protocol AND t1_buf_dit
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
#ifndef VFFT_R16_PLAN_WISDOM_H
#define VFFT_R16_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *      64    64   log3           dit              991.6
 *      64    72   log3           dit              946.4
 *      64   512   log3           dit             1959.5
 *      96    96   log3           dit             1395.5
 *      96   104   log3           dit             1416.8
 *      96   768   log3           dit             1825.5
 *     128   128   log3           dit             2015.2
 *     128   136   log3           dit             1942.1
 *     128  1024   log3           dit             4534.5
 *     192   192   log3           dit             2945.4
 *     192   200   log3           dit             2963.6
 *     192  1536   log3           dit             7145.0
 *     256   256   log3           dit             5220.9
 *     256   264   log3           dit             3978.2
 *     256  2048   log3           dit             9349.0
 *     384   384   log3           dit             5918.9
 *     384   392   log3           dit             5998.2
 *     384  3072   log3           dit            14132.9
 *     512   512   log3           dit            16497.7
 *     512   520   log3           dit             7950.3
 *     512  4096   log3           dit            18173.4
 *     768   768   log3           dit            15363.7
 *     768   776   log3           dit            11719.8
 *     768  6144   log3           dit            28529.1
 *    1024  1024   log3           dit            38447.2
 *    1024  1032   log3           dit            16180.0
 *    1024  8192   log3           dit            37948.7
 *    1536  1536   log3           dit            60694.5
 *    1536  1544   log3           dit            25995.8
 *    1536 12288   log3           dit            56006.6
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix16_prefer_dit_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048} */
    if (me >= 64 && me <= 2048) return 1;
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix16_prefer_dif_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix16_prefer_log3(size_t me, size_t ios) {
    return radix16_prefer_dit_log3(me, ios)
        || radix16_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix16_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed t1s never wins on this host. */
    return 0;
}

/* Should the planner use t1_buf_dit (buffered flat) at (me, ios)?
 * True when the flat protocol wins cross-protocol AND the t1_buf_dit
 * dispatcher beats t1_dit baseline within the flat protocol (per the
 * within-flat per-dispatcher comparison with 2%% tie threshold). */
static inline int radix16_prefer_buf(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed buf never wins on this host. */
    return 0;
}

#endif /* VFFT_R16_PLAN_WISDOM_H */
