/* vfft_r32_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=32.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * Predicates:
 *   radix32_prefer_dit_log3 — DIT-log3 codelet won as log3 representative
 *                              AND log3 was cross-protocol winner.
 *   radix32_prefer_dif_log3 — collapsed to always-0 in v1.0 (DIF filter).
 *   radix32_prefer_log3     — union of the above (= dit_log3 in v1.0).
 *   radix32_prefer_t1s      — t1s protocol won.
 *   radix32_prefer_buf      — flat won cross-protocol AND t1_buf_dit
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
#ifndef VFFT_R32_PLAN_WISDOM_H
#define VFFT_R32_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *      64    64   t1s            -                717.5
 *      64    72   t1s            -                763.3
 *      64   512   t1s            -               1900.5
 *      96    96   t1s            -               1366.8
 *      96   104   t1s            -               1289.2
 *      96   768   t1s            -               2271.7
 *     128   128   log3           dit             2034.1
 *     128   136   log3           dit             1811.5
 *     128  1024   t1s            -               4121.7
 *     192   192   flat           -               2879.9
 *     192   200   log3           dit             3050.4
 *     192  1536   log3           dit             6174.7
 *     256   256   log3           dit             6957.1
 *     256   264   log3           dit             4717.2
 *     256  2048   log3           dit             8711.2
 *     384   384   log3           dit             6740.4
 *     384   392   flat           -               7396.5
 *     384  3072   flat           -              11675.4
 *     512   512   log3           dit            15970.0
 *     512   520   log3           dit             8438.5
 *     512  4096   flat           -              18813.5
 *     768   768   log3           dit            20543.0
 *     768   776   log3           dit            13653.4
 *     768  6144   flat           -              29055.3
 *    1024  1024   log3           dit            32618.8
 *    1024  1032   log3           dit            16150.8
 *    1024  8192   flat           -              41255.9
 *    1536  1536   log3           dit            49650.4
 *    1536  1544   log3           dit            25107.8
 *    1536 12288   flat           -              64150.4
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix32_prefer_dit_log3(size_t me, size_t ios) {
    /* Sparse DIT-log3 wins at 1 specific (me, ios) cells: (384,384) */
    if ((me == 384 && ios == 384)) return 1;
    /* Bench wins at me ∈ {128, 192, 256, 512, 768, 1024, 1536, 2048} */
    if (me >= 128 && me <= 2048) return 1;
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix32_prefer_dif_log3(size_t me, size_t ios) {
    (void)me; (void)ios;
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix32_prefer_log3(size_t me, size_t ios) {
    return radix32_prefer_dit_log3(me, ios)
        || radix32_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix32_prefer_t1s(size_t me, size_t ios) {
    /* Sparse t1s wins at 1 specific (me, ios) cells: (128,1024) */
    if ((me == 128 && ios == 1024)) return 1;
    /* Bench wins at me ∈ {64, 96} */
    if (me == 64 || me == 96) return 1;
    return 0;
}

/* Should the planner use t1_buf_dit (buffered flat) at (me, ios)?
 * True when the flat protocol wins cross-protocol AND the t1_buf_dit
 * dispatcher beats t1_dit baseline within the flat protocol (per the
 * within-flat per-dispatcher comparison with 2%% tie threshold). */
static inline int radix32_prefer_buf(size_t me, size_t ios) {
    /* Sparse buf wins at 3 specific (me, ios) cells: (512,4096), (768,6144), (1536,12288) */
    if ((me == 512 && ios == 4096) || (me == 768 && ios == 6144) || (me == 1536 && ios == 12288)) return 1;
    return 0;
}

#endif /* VFFT_R32_PLAN_WISDOM_H */
