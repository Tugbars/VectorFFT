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
 *      24    24   t1s                 9.2
 *      24    32   t1s                 8.9
 *      24    72   flat                9.2
 *      24   192   flat                9.2
 *      48    48   t1s                16.8
 *      48    56   flat               16.9
 *      48   144   t1s                17.1
 *      48   384   flat               17.0
 *      96    96   flat               34.2
 *      96   104   flat               32.6
 *      96   288   flat               32.6
 *      96   768   flat               33.0
 *     192   192   t1s                64.8
 *     192   200   t1s                67.3
 *     192   576   flat               64.3
 *     192  1536   flat               67.6
 *     384   384   t1s               127.9
 *     384   392   flat              126.0
 *     384  1152   flat              127.9
 *     384  3072   t1s               129.3
 *     768   768   t1s               259.1
 *     768   776   t1s               269.0
 *     768  2304   t1s               253.9
 *     768  6144   t1s               265.1
 *    1536  1536   t1s               762.3
 *    1536  1544   t1s               785.2
 *    1536  4608   t1s               771.3
 *    1536 12288   t1s               785.4
 *    3072  3072   t1s              1583.4
 *    3072  3080   t1s              1603.5
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix3_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench showed log3 never wins on this host. */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix3_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {24, 48, 192, 384, 768, 1536, 3072} */
    if (me >= 24 && me <= 3072) return 1;
    return 0;
}

#endif /* VFFT_R3_PLAN_WISDOM_H */
