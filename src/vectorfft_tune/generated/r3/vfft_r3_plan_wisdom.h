/* vfft_r3_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=3.
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
#ifndef VFFT_R3_PLAN_WISDOM_H
#define VFFT_R3_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *      24    24   flat                9.2
 *      24    32   flat                9.3
 *      24    72   flat                9.6
 *      24   192   flat                9.7
 *      48    48   flat               18.2
 *      48    56   t1s                17.4
 *      48   144   flat               19.0
 *      48   384   flat               17.6
 *      96    96   t1s                34.3
 *      96   104   flat               35.4
 *      96   288   flat               34.0
 *      96   768   t1s                34.6
 *     192   192   t1s                67.8
 *     192   200   flat               68.6
 *     192   576   flat               66.0
 *     192  1536   flat               66.1
 *     384   384   flat              134.9
 *     384   392   flat              129.8
 *     384  1152   flat              136.0
 *     384  3072   flat              133.7
 *     768   768   t1s               258.9
 *     768   776   t1s               257.6
 *     768  2304   t1s               268.1
 *     768  6144   t1s               276.3
 *    1536  1536   t1s               840.2
 *    1536  1544   log3              845.7
 *    1536  4608   t1s               850.6
 *    1536 12288   t1s               779.3
 *    3072  3072   t1s              1731.2
 *    3072  3080   t1s              1672.3
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix3_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix3_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {96, 768, 1536, 3072} */
    if (me >= 96 && me <= 3072) return 1;
    return 0;
}

#endif /* VFFT_R3_PLAN_WISDOM_H */
