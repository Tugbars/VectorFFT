/* vfft_r12_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=12.
 * Bench host: unknown
 *
 * Each function returns 1 if the planner should use the named protocol
 * for a stage with the given (me, ios), 0 otherwise.
 *
 * Derived from cross-protocol comparison at each sweep point. The
 * planner should consult these AFTER the codelet-level dispatcher
 * has been selected — they drive twiddle-table layout and the
 * K-blocked execution path.
 */
#ifndef VFFT_R12_PLAN_WISDOM_H
#define VFFT_R12_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   flat               28.7
 *       8    16   flat               28.8
 *       8    96   flat               30.9
 *       8   256   flat               29.2
 *      16    16   flat               54.4
 *      16    24   t1s                52.9
 *      16   192   t1s                52.8
 *      16   512   t1s               203.1
 *      32    32   t1s               103.0
 *      32    40   t1s               101.2
 *      32   384   t1s               103.5
 *      32  1024   t1s               265.1
 *      64    64   t1s               200.7
 *      64    72   t1s               199.3
 *      64   768   t1s               211.0
 *      64  2048   flat              609.7
 *     128   128   t1s               388.2
 *     128   136   t1s               384.9
 *     128  1536   log3             1278.5
 *     128  4096   flat             1225.6
 *     256   256   t1s               796.6
 *     256   264   t1s               819.1
 *     256  3072   log3             2597.1
 *     256  8192   log3             1520.6
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix12_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {256} */
    if (me == 256) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix12_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {16, 32, 64, 128, 256} */
    if (me >= 16 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R12_PLAN_WISDOM_H */
