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
 *      24    24   t1s                 8.9
 *      24    32   t1s                 9.0
 *      24    72   t1s                 9.1
 *      24   192   t1s                 8.9
 *      48    48   t1s                16.7
 *      48    56   t1s                16.7
 *      48   144   t1s                16.8
 *      48   384   flat               16.9
 *      96    96   flat               32.8
 *      96   104   t1s                33.4
 *      96   288   t1s                32.6
 *      96   768   flat               32.5
 *     192   192   flat               65.3
 *     192   200   flat               63.7
 *     192   576   flat               63.8
 *     192  1536   flat               64.7
 *     384   384   flat              127.8
 *     384   392   t1s               129.0
 *     384  1152   flat              127.7
 *     384  3072   t1s               127.9
 *     768   768   t1s               257.1
 *     768   776   t1s               253.8
 *     768  2304   t1s               258.6
 *     768  6144   t1s               259.8
 *    1536  1536   t1s               741.8
 *    1536  1544   t1s               770.2
 *    1536  4608   t1s               749.6
 *    1536 12288   t1s               751.2
 *    3072  3072   t1s              1585.8
 *    3072  3080   t1s              1586.5
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix3_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix3_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {24, 48, 96, 384, 768, 1536, 3072} */
    if (me >= 24 && me <= 3072) return 1;
    return 0;
}

#endif /* VFFT_R3_PLAN_WISDOM_H */
