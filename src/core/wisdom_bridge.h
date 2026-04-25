/**
 * wisdom_bridge.h — Plan-time protocol selection bridge (Phase 2).
 *
 * The codelet-side bench produces per-radix plan-wisdom headers:
 *
 *   generated/rN/vfft_rN_plan_wisdom.h
 *
 * Each exports static-inline predicates indexed by (me, ios):
 *
 *   radixN_prefer_dit_log3(me, ios)   — DIT-log3 won the bench at (me,ios)
 *   radixN_prefer_dif_log3(me, ios)   — DIF-log3 won the bench at (me,ios)
 *   radixN_prefer_log3(me, ios)       — DIT or DIF log3 won (union)
 *   radixN_prefer_t1s(me, ios)        — t1s protocol won
 *
 * This header includes all 17 portfolio plan_wisdom headers and dispatches
 * a single entry point per protocol that the planner calls without
 * knowing which radix has which wisdom available.
 *
 * Phase 2 uses stride_prefer_dit_log3 only:
 * ---------------------------------------------------------------------
 * The forward executor path is DIT-structured. It can run the DIT-log3
 * codelet directly through reg->t1_fwd_log3[R], applying cf pre-butterfly
 * (executor.h lines 200-217). DIF-log3 is a different codelet that
 * applies twiddles post-butterfly; it is NOT swappable with DIT-log3 —
 * cross-validation confirms they produce different output buffers.
 * Activating DIF-log3 in the executor requires a separate code path
 * (Phase 3 work) and is out of scope here.
 *
 * stride_prefer_dif_log3 is exposed for future Phase 3 integration and
 * for diagnostic counting (how many cells does the executor's current
 * DIT-only design leave on the table?). The Phase 2 planner does not
 * call it.
 *
 * Missing wisdom (unknown R, or radix doesn't emit wisdom because it
 * has only one protocol worth benching, e.g. R=4) yields 0 from both
 * queries — the planner falls back to the legacy "flat via t1_fwd,
 * with t1s attached where available" behaviour, which is exactly the
 * pre-wisdom planner.
 *
 * Build note
 * ==========
 * The -I path used by the build that consumes this must include
 * generated/rN/ for every N in the portfolio. The registry.h already
 * arranges -I for codelet headers; plan_wisdom lives alongside.
 *
 * Single translation unit rule
 * ============================
 * The radixN_prefer_* functions are static inline (no external linkage).
 * Taking their addresses at registry-init time only works if they're
 * visible in that TU. Every TU that includes registry.h gets its own
 * copies, which is fine — we never compare these pointers.
 */
#ifndef VFFT_WISDOM_BRIDGE_H
#define VFFT_WISDOM_BRIDGE_H

#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════
 * PLAN-WISDOM HEADERS PER RADIX
 *
 * Each radix in the portfolio that has multiple protocols measured
 * contributes a plan_wisdom header. The list below must stay in sync
 * with the portfolio.
 * ═══════════════════════════════════════════════════════════════ */

#include "vfft_r3_plan_wisdom.h"
#include "vfft_r5_plan_wisdom.h"
#include "vfft_r6_plan_wisdom.h"
#include "vfft_r7_plan_wisdom.h"
#include "vfft_r10_plan_wisdom.h"
#include "vfft_r11_plan_wisdom.h"
#include "vfft_r12_plan_wisdom.h"
#include "vfft_r13_plan_wisdom.h"
#include "vfft_r16_plan_wisdom.h"
#include "vfft_r17_plan_wisdom.h"
#include "vfft_r19_plan_wisdom.h"
#include "vfft_r20_plan_wisdom.h"
#include "vfft_r25_plan_wisdom.h"
#include "vfft_r32_plan_wisdom.h"
#include "vfft_r64_plan_wisdom.h"

/* R=4 and R=8: if the bench produced plan_wisdom for these, uncomment.
 *   #include "vfft_r4_plan_wisdom.h"
 *   #include "vfft_r8_plan_wisdom.h"
 * Default behaviour without these includes is conservative: stride_prefer_*
 * returns 0 for R=4 and R=8, planner falls back to flat. */

/* ═══════════════════════════════════════════════════════════════
 * PROTOCOL QUERIES
 *
 * The planner calls these at each stage with the runtime-effective
 * me (per-thread slice size) and the stage's ios (stride between
 * butterfly legs). The result drives:
 *   - which codelet slot to read from the registry
 *   - whether to set the log3_mask bit for this stage in plan_create
 * ═══════════════════════════════════════════════════════════════ */

/* Should the planner use DIT-log3 protocol at this (me, ios)?
 *
 * SAFE FOR PHASE 2: returns 1 only at cells where the DIT-log3 codelet
 * specifically was the bench winner. The forward executor path is
 * DIT-structured; activating log3 here means substituting in the
 * DIT-log3 codelet (via reg->t1_fwd_log3[R]) which the executor
 * already supports. This is the conservative, regression-free query.
 */
static inline int stride_prefer_dit_log3(int R, size_t me, size_t ios) {
    switch (R) {
        case 3:  return radix3_prefer_dit_log3(me, ios);
        case 5:  return radix5_prefer_dit_log3(me, ios);
        case 6:  return radix6_prefer_dit_log3(me, ios);
        case 7:  return radix7_prefer_dit_log3(me, ios);
        case 10: return radix10_prefer_dit_log3(me, ios);
        case 11: return radix11_prefer_dit_log3(me, ios);
        case 12: return radix12_prefer_dit_log3(me, ios);
        case 13: return radix13_prefer_dit_log3(me, ios);
        case 16: return radix16_prefer_dit_log3(me, ios);
        case 17: return radix17_prefer_dit_log3(me, ios);
        case 19: return radix19_prefer_dit_log3(me, ios);
        case 20: return radix20_prefer_dit_log3(me, ios);
        case 25: return radix25_prefer_dit_log3(me, ios);
        case 32: return radix32_prefer_dit_log3(me, ios);
        case 64: return radix64_prefer_dit_log3(me, ios);
        default: return 0;
    }
}

/* Should the planner use DIF-log3 protocol at this (me, ios)?
 *
 * NOT YET CONSUMABLE BY THE EXECUTOR. The forward path is DIT-structured;
 * DIF-log3 is a different codelet (different output buffer, even given
 * the same input + same twiddle table). Including DIF-log3 in the
 * planner's selection would require new executor and twiddle-build paths
 * — Phase 3 work.
 *
 * This query is exposed for diagnostics: how many cells could a future
 * DIF-log3-aware executor capture? Phase 2 planners do not call it.
 */
static inline int stride_prefer_dif_log3(int R, size_t me, size_t ios) {
    switch (R) {
        case 3:  return radix3_prefer_dif_log3(me, ios);
        case 5:  return radix5_prefer_dif_log3(me, ios);
        case 6:  return radix6_prefer_dif_log3(me, ios);
        case 7:  return radix7_prefer_dif_log3(me, ios);
        case 10: return radix10_prefer_dif_log3(me, ios);
        case 11: return radix11_prefer_dif_log3(me, ios);
        case 12: return radix12_prefer_dif_log3(me, ios);
        case 13: return radix13_prefer_dif_log3(me, ios);
        case 16: return radix16_prefer_dif_log3(me, ios);
        case 17: return radix17_prefer_dif_log3(me, ios);
        case 19: return radix19_prefer_dif_log3(me, ios);
        case 20: return radix20_prefer_dif_log3(me, ios);
        case 25: return radix25_prefer_dif_log3(me, ios);
        case 32: return radix32_prefer_dif_log3(me, ios);
        case 64: return radix64_prefer_dif_log3(me, ios);
        default: return 0;
    }
}

/* Should the planner prefer t1s protocol? Returns 1 if yes.
 *
 * Note: the existing planner already attaches t1s whenever a t1s codelet
 * is registered for the radix; the executor prefers t1s over t1 at runtime.
 * This query is informational — it lets a future planner *disable* t1s at
 * sweep points where flat or log3 actually wins. Not used by Phase 2.
 */
static inline int stride_prefer_t1s(int R, size_t me, size_t ios) {
    switch (R) {
        case 3:  return radix3_prefer_t1s(me, ios);
        case 5:  return radix5_prefer_t1s(me, ios);
        case 6:  return radix6_prefer_t1s(me, ios);
        case 7:  return radix7_prefer_t1s(me, ios);
        case 10: return radix10_prefer_t1s(me, ios);
        case 11: return radix11_prefer_t1s(me, ios);
        case 12: return radix12_prefer_t1s(me, ios);
        case 13: return radix13_prefer_t1s(me, ios);
        case 16: return radix16_prefer_t1s(me, ios);
        case 17: return radix17_prefer_t1s(me, ios);
        case 19: return radix19_prefer_t1s(me, ios);
        case 20: return radix20_prefer_t1s(me, ios);
        case 25: return radix25_prefer_t1s(me, ios);
        /* R=32 and R=64 plan_wisdom does not export prefer_t1s
         * because t1s isn't competitive at those radixes. */
        case 32: return 0;
        case 64: return 0;
        default: return 0;
    }
}

#endif /* VFFT_WISDOM_BRIDGE_H */
