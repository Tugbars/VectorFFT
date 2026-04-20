/* vfft_r20_plan_wisdom.h
 *
 * Auto-generated plan-protocol selection for R=20.
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
#ifndef VFFT_R20_PLAN_WISDOM_H
#define VFFT_R20_PLAN_WISDOM_H

#include <stddef.h>

/* Cross-protocol comparison (fwd direction, AVX2):
 *   me    ios   winning_protocol   winning_ns
 *       8     8   t1s                58.8
 *       8    16   flat               60.6
 *       8   160   flat               58.3
 *       8   256   flat               58.9
 *      16    16   t1s               113.5
 *      16    24   flat              110.0
 *      16   320   t1s               105.8
 *      16   512   flat              278.9
 *      32    32   flat              211.0
 *      32    40   flat              212.4
 *      32   640   t1s               218.4
 *      32  1024   flat              517.2
 *      64    64   t1s               418.3
 *      64    72   t1s               382.9
 *      64  1280   flat              931.4
 *      64  2048   log3             1040.9
 *     128   128   t1s               815.4
 *     128   136   t1s               820.1
 *     128  2560   flat             2117.6
 *     128  4096   t1s              1941.1
 *     256   256   flat             4119.1
 *     256   264   t1s              1667.2
 *     256  5120   t1s              4178.2
 *     256  8192   t1s              5233.8
 */

/* Should the planner use log3 protocol at (me, ios)? */
static inline int radix20_prefer_log3(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    return 0;
}

/* Should the planner use t1s protocol at (me, ios)? */
static inline int radix20_prefer_t1s(size_t me, size_t ios) {
    (void)ios;  /* may be unused if rules are me-only */
    /* Bench wins at me ∈ {16, 64, 128, 256} */
    if (me >= 16 && me <= 256) return 1;
    return 0;
}

#endif /* VFFT_R20_PLAN_WISDOM_H */
