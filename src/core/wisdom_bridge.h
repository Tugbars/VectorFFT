/**
 * wisdom_bridge.h — Plan-time protocol selection bridge.
 *
 * The codelet-side bench produces per-radix plan-wisdom headers:
 *
 *   generated/r{N}/vfft_r{N}_plan_wisdom.h
 *
 * Each exports static-inline predicates indexed by (me, ios):
 *
 *   radix{N}_prefer_dit_log3   DIT-log3 won as log3 representative AND
 *                              log3 was cross-protocol winner
 *   radix{N}_prefer_dif_log3   collapsed to always-0 in v1.0 (DIF filter)
 *   radix{N}_prefer_log3       union of the above (= dit_log3 in v1.0)
 *   radix{N}_prefer_t1s        t1s protocol won
 *   radix{N}_prefer_buf        flat won cross-protocol AND t1_buf_dit
 *                              dispatcher won within flat (vs t1_dit).
 *                              Always-0 for radixes without buf variants.
 *
 * This header dispatches a single entry point per predicate that the
 * planner calls without knowing which radix has which wisdom available.
 *
 * Contract with the planner
 * -------------------------
 * The planner calls these at each stage with the runtime-effective me
 * (per-thread slice size) and the stage's ios (stride between butterfly
 * legs). The returned bit drives:
 *   - which codelet slot to read from the registry
 *   - whether to set the log3_mask bit for this stage in plan_create
 *
 * Wiring status (this version)
 * ----------------------------
 * - stride_prefer_dit_log3:  used by the new core/planner.h — Phase 2
 *                            integration, regression-free.
 * - stride_prefer_dif_log3:  always-0 in v1.0 (DIF filter excludes DIF
 *                            variants from winner selection). Kept in
 *                            the API for v1.1 when a DIF-forward
 *                            executor path lands.
 * - stride_prefer_t1s:       informational — the executor today
 *                            unconditionally prefers t1s at runtime
 *                            when t1s_fwd is non-NULL. A future
 *                            planner can use this to *disable* t1s
 *                            attachment at cells where flat or log3
 *                            actually wins.
 * - stride_prefer_buf:       not yet consumed by the planner. Phase 2.1
 *                            (planner.h extension) will consult this to
 *                            choose t1_buf_fwd over t1_fwd at cells
 *                            where flat wins cross-protocol AND buf
 *                            wins within flat. Wired here so the
 *                            planner change is a single small edit.
 *
 * Missing-radix fallback
 * ----------------------
 * For radixes without a plan_wisdom header (R=2 in the v1.0 portfolio),
 * each query returns 0 from the default switch case. The planner's
 * fallback behaviour for R=2 is the legacy flat path — exactly the
 * pre-wisdom planner.
 */
#ifndef VFFT_WISDOM_BRIDGE_H
#define VFFT_WISDOM_BRIDGE_H

#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════
 * PLAN-WISDOM HEADERS PER RADIX
 *
 * Every radix in the v1.0 portfolio has a plan_wisdom header. The
 * list below covers the entire tune portfolio (R=3..R=64, 16 radixes).
 * R=2 is the only untuned radix and has no plan_wisdom header.
 * ═══════════════════════════════════════════════════════════════ */

#include "vfft_r3_plan_wisdom.h"
#include "vfft_r4_plan_wisdom.h"
#include "vfft_r5_plan_wisdom.h"
#include "vfft_r6_plan_wisdom.h"
#include "vfft_r7_plan_wisdom.h"
#include "vfft_r8_plan_wisdom.h"
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

/* ═══════════════════════════════════════════════════════════════
 * PROTOCOL QUERIES
 *
 * Every predicate covers every tuned radix. The portfolio's wisdom
 * always emits all four predicates (prefer_dit_log3, prefer_dif_log3,
 * prefer_t1s, prefer_buf), with no-op stubs for radixes that don't
 * have a particular dispatcher (e.g. prefer_buf returns 0 for R=3
 * because R=3 has no t1_buf_dit family).
 * ═══════════════════════════════════════════════════════════════ */

/* DIT-log3: log3 won cross-protocol AND winner came from DIT family.
 * Activates the DIT-log3 codelet via reg->t1_fwd_log3[R]. */
static inline int stride_prefer_dit_log3(int R, size_t me, size_t ios) {
    switch (R) {
        case 3:  return radix3_prefer_dit_log3(me, ios);
        case 4:  return radix4_prefer_dit_log3(me, ios);
        case 5:  return radix5_prefer_dit_log3(me, ios);
        case 6:  return radix6_prefer_dit_log3(me, ios);
        case 7:  return radix7_prefer_dit_log3(me, ios);
        case 8:  return radix8_prefer_dit_log3(me, ios);
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

/* DIF-log3: collapsed to always-0 in v1.0 (DIF filter in select_and_emit
 * excludes DIF variants from winner selection). Kept for v1.1 when a
 * DIF-forward executor path lands. */
static inline int stride_prefer_dif_log3(int R, size_t me, size_t ios) {
    switch (R) {
        case 3:  return radix3_prefer_dif_log3(me, ios);
        case 4:  return radix4_prefer_dif_log3(me, ios);
        case 5:  return radix5_prefer_dif_log3(me, ios);
        case 6:  return radix6_prefer_dif_log3(me, ios);
        case 7:  return radix7_prefer_dif_log3(me, ios);
        case 8:  return radix8_prefer_dif_log3(me, ios);
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

/* t1s: scalar-broadcast twiddle protocol won. Critical at small me —
 * t1s wins broadly across the portfolio (every radix's wisdom shows
 * t1s wins at me ∈ {8..256} on this host's measurements). */
static inline int stride_prefer_t1s(int R, size_t me, size_t ios) {
    switch (R) {
        case 3:  return radix3_prefer_t1s(me, ios);
        case 4:  return radix4_prefer_t1s(me, ios);
        case 5:  return radix5_prefer_t1s(me, ios);
        case 6:  return radix6_prefer_t1s(me, ios);
        case 7:  return radix7_prefer_t1s(me, ios);
        case 8:  return radix8_prefer_t1s(me, ios);
        case 10: return radix10_prefer_t1s(me, ios);
        case 11: return radix11_prefer_t1s(me, ios);
        case 12: return radix12_prefer_t1s(me, ios);
        case 13: return radix13_prefer_t1s(me, ios);
        case 16: return radix16_prefer_t1s(me, ios);
        case 17: return radix17_prefer_t1s(me, ios);
        case 19: return radix19_prefer_t1s(me, ios);
        case 20: return radix20_prefer_t1s(me, ios);
        case 25: return radix25_prefer_t1s(me, ios);
        case 32: return radix32_prefer_t1s(me, ios);
        case 64: return radix64_prefer_t1s(me, ios);
        default: return 0;
    }
}

/* Buffered-flat: flat won cross-protocol AND t1_buf_dit dispatcher
 * won within flat (vs t1_dit baseline). Returns 0 for radixes without
 * a t1_buf_dit family — the wisdom emits a no-op stub there. */
static inline int stride_prefer_buf(int R, size_t me, size_t ios) {
    switch (R) {
        case 3:  return radix3_prefer_buf(me, ios);
        case 4:  return radix4_prefer_buf(me, ios);
        case 5:  return radix5_prefer_buf(me, ios);
        case 6:  return radix6_prefer_buf(me, ios);
        case 7:  return radix7_prefer_buf(me, ios);
        case 8:  return radix8_prefer_buf(me, ios);
        case 10: return radix10_prefer_buf(me, ios);
        case 11: return radix11_prefer_buf(me, ios);
        case 12: return radix12_prefer_buf(me, ios);
        case 13: return radix13_prefer_buf(me, ios);
        case 16: return radix16_prefer_buf(me, ios);
        case 17: return radix17_prefer_buf(me, ios);
        case 19: return radix19_prefer_buf(me, ios);
        case 20: return radix20_prefer_buf(me, ios);
        case 25: return radix25_prefer_buf(me, ios);
        case 32: return radix32_prefer_buf(me, ios);
        case 64: return radix64_prefer_buf(me, ios);
        default: return 0;
    }
}

#endif /* VFFT_WISDOM_BRIDGE_H */
