/* vfft_r25_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=25.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * Predicates:
 *   radix25_prefer_dit_log3 — DIT-log3 codelet won as log3 representative
 *                              AND log3 was cross-protocol winner.
 *   radix25_prefer_dif_log3 — collapsed to always-0 in v1.0 (DIF filter).
 *   radix25_prefer_log3     — union of the above (= dit_log3 in v1.0).
 *   radix25_prefer_t1s      — t1s protocol won.
 *   radix25_prefer_buf      — flat won cross-protocol AND t1_buf_dit
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
#ifndef VFFT_R25_PLAN_WISDOM_H
#define VFFT_R25_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   log3_family   winning_ns
 *       8     8   t1s            -                200.9
 *       8    16   t1s            -                189.5
 *       8   200   t1s            -                196.9
 *       8   256   log3           dit              284.8
 *      16    16   t1s            -                368.0
 *      16    24   t1s            -                378.4
 *      16   400   t1s            -                385.1
 *      16   512   flat           -                739.3
 *      32    32   t1s            -                735.3
 *      32    40   t1s            -                735.6
 *      32   800   t1s            -                736.7
 *      32  1024   flat           -               1851.5
 *      64    64   t1s            -               1539.9
 *      64    72   t1s            -               1525.2
 *      64  1600   t1s            -               1547.6
 *      64  2048   flat           -               3519.4
 *     128   128   t1s            -               2978.9
 *     128   136   t1s            -               3080.7
 *     128  3200   t1s            -               3837.3
 *     128  4096   flat           -               7219.2
 *     256   256   t1s            -               9875.4
 *     256   264   t1s            -               6173.0
 *     256  6400   t1s            -              11978.4
 *     256  8192   flat           -              14725.2
 */

/* Should the planner use DIT-log3 protocol at (me, ios)? */
/* Safe for today's executor: activates DIT-log3 codelet on the forward path. */
static inline int radix25_prefer_dit_log3(size_t me, size_t ios) {
    /* Sparse DIT-log3 wins at 1 specific (me, ios) cells: (8,256) */
    if ((me == 8 && ios == 256)) return 1;
    return 0;
}

/* Should the planner use DIF-log3 protocol at (me, ios)? */
/* NOT yet consumable by the default executor (forward path is DIT).
 * Requires a DIF-forward executor path to activate. Informational. */
static inline int radix25_prefer_dif_log3(size_t me, size_t ios) {
    (void)me; (void)ios;
    /* Bench showed DIF-log3 never wins on this host. */
    return 0;
}

/* Should the planner use any log3 protocol at (me, ios)?
 * Backward-compatible union: returns 1 if DIT-log3 or DIF-log3 wins. */
static inline int radix25_prefer_log3(size_t me, size_t ios) {
    return radix25_prefer_dit_log3(me, ios)
        || radix25_prefer_dif_log3(me, ios);
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix25_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* rules are me-only */
    /* Bench wins at me ∈ {8, 16, 32, 64, 128, 256} */
    if (me >= 8 && me <= 256) return 1;
    return 0;
}

/* Should the planner use t1_buf_dit (buffered flat) at (me, ios)?
 * True when the flat protocol wins cross-protocol AND the t1_buf_dit
 * dispatcher beats t1_dit baseline within the flat protocol (per the
 * within-flat per-dispatcher comparison with 2%% tie threshold). */
static inline int radix25_prefer_buf(size_t me, size_t ios) {
    (void)me; (void)ios;
    /* No t1_buf_dit dispatcher in this radix's portfolio. */
    return 0;
}

#endif /* VFFT_R25_PLAN_WISDOM_H */
