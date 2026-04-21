/* vfft_r10_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=10.
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
#ifndef VFFT_R10_PLAN_WISDOM_H
#define VFFT_R10_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   flat               28.1
 *       8    16   flat               25.5
 *       8    80   flat               27.2
 *       8   256   flat               27.1
 *      16    16   flat               50.0
 *      16    24   flat               48.3
 *      16   160   flat               50.0
 *      16   512   t1s               160.7
 *      32    32   flat               97.0
 *      32    40   flat               87.9
 *      32   320   t1s               100.0
 *      32  1024   t1s               215.4
 *      64    64   flat              181.0
 *      64    72   flat              178.0
 *      64   640   flat              184.8
 *      64  2048   t1s               384.0
 *     128   128   flat              384.9
 *     128   136   flat              344.3
 *     128  1280   t1s               377.7
 *     128  4096   log3             1069.7
 *     256   256   t1s               769.9
 *     256   264   t1s               723.3
 *     256  2560   t1s              1347.4
 *     256  8192   t1s               984.5
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix10_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix10_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {32, 256} */
    if (me == 32 || me == 256) return 1;
    return 0;
}

#endif /* VFFT_R10_PLAN_WISDOM_H */
