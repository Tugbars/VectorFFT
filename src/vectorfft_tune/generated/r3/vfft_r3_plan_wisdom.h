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
 *      24    24   t1s                24.6
 *      24    32   t1s                24.5
 *      24    72   flat               25.4
 *      24   192   t1s                24.7
 *      48    48   flat               47.0
 *      48    56   t1s                46.0
 *      48   144   t1s                47.5
 *      48   384   flat               46.4
 *      96    96   t1s                92.6
 *      96   104   flat               92.6
 *      96   288   flat               90.5
 *      96   768   flat               92.6
 *     192   192   t1s               180.9
 *     192   200   log3              183.2
 *     192   576   flat              184.1
 *     192  1536   t1s               180.4
 *     384   384   flat              363.8
 *     384   392   log3              363.5
 *     384  1152   t1s               366.3
 *     384  3072   log3              363.4
 *     768   768   t1s               731.7
 *     768   776   t1s               731.3
 *     768  2304   t1s               733.8
 *     768  6144   t1s               729.8
 *    1536  1536   t1s              1650.3
 *    1536  1544   t1s              1694.7
 *    1536  4608   t1s              1656.6
 *    1536 12288   t1s              1585.5
 *    3072  3072   log3             3748.2
 *    3072  3080   t1s              3402.3
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix3_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {384} */
    if (me == 384) return 1;
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix3_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {24, 48, 192, 768, 1536, 3072} */
    if (me >= 24 && me <= 3072) return 1;
    return 0;
}

#endif /* VFFT_R3_PLAN_WISDOM_H */
