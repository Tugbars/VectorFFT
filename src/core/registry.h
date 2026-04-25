/**
 * src/core/registry.h — codelet registry for the orchestrator-tuned executor.
 *
 * Wires the per-host codelet dispatchers emitted by vectorfft_tune into the
 * registry that the planner and executor consume. Each tuned radix's t1
 * slots point at static-inline dispatchers from
 * `vectorfft_tune/generated/r{R}/vfft_r{R}_{family}_dispatch_{isa}.h`.
 * Each dispatcher is a per-(me, ios) selector over the variants within
 * one dispatcher family; calling it costs one inlined branch tree plus a
 * codelet call.
 *
 * Differences from stride-fft/core/registry.h
 * -------------------------------------------
 * 1. New slot: `t1_buf_fwd` / `t1_buf_bwd` (buffered-flat dispatcher).
 *    Populated for radixes that have a t1_buf_dit family in the tune
 *    portfolio (R=16 today; R=25/R=32/R=64 once locally regenerated).
 *    NULL elsewhere. Currently no consumer reads this slot — Phase 2.1
 *    (Python-side wisdom emit) needs to land before the planner consults
 *    `radix{R}_prefer_buf(me, ios)` to choose between t1_dit and
 *    t1_buf_dit. The slot is wired but inert until then.
 *
 * 2. No `t1_dif` slot. DIF codelets exist as bench artifacts (see
 *    `vfft_r{R}_t1_dif_dispatch_*.h` for benchmarking) but the
 *    DIT-structured forward executor cannot substitute them per stage —
 *    DIT and DIF compute different output buffers given the same input
 *    and twiddles. Filtered for v1.0 per `dit_dif_design_note.md`.
 *
 * 3. Tuned vs untuned radixes
 *      Tuned (dispatchers + plan_wisdom): 3, 5, 6, 7, 10..13, 16, 17,
 *        19, 20, 25, 32, 64.  Same set the wisdom_bridge.h consults.
 *      Untuned (production codelets used directly): 2, 4, 8.
 *    For R=32 and R=64, dispatcher headers exist only after the bench is
 *    run locally for those radixes. Build error on missing header is the
 *    intentional signal — run the orchestrator before consuming this
 *    registry.
 *
 * Auxiliary slots (n1, n1_scaled, t1_oop) come from the production
 * codelet headers under stride-fft/codelets/{isa}/. The tune generator
 * does not emit those variants — they're used by R2C, C2R, and 2D paths
 * that have separate dispatcher concerns outside this work.
 *
 * Build requirements
 * ------------------
 * The CMake target consuming this header must add to its include path:
 *   - src/stride-fft/core                          (executor.h)
 *   - src/stride-fft/codelets/{isa}                (production codelets)
 *   - src/vectorfft_tune/generated/r{R} per tuned R (dispatcher .h files)
 *
 * The dispatchers are static inline so there is nothing to link — the
 * compiler inlines the winning codelet at the call site.
 */
#ifndef VFFT_CORE_REGISTRY_H
#define VFFT_CORE_REGISTRY_H

#include "executor.h"
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * ISA DETECTION
 *
 * Priority: AVX-512 > AVX2 > scalar. Mirrors stride-fft/core/registry.h.
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) && defined(__AVX512DQ__)
  #define VFFT_ISA_AVX512 1
  #define VFFT_ISA_TAG avx512
#elif defined(__AVX2__) && defined(__FMA__)
  #define VFFT_ISA_AVX2 1
  #define VFFT_ISA_TAG avx2
#else
  #define VFFT_ISA_SCALAR 1
  #define VFFT_ISA_TAG scalar
#endif

#define _VFFT_PASTE3(a,b,c)  a##b##c
#define _VFFT_PASTE(a,b,c)   _VFFT_PASTE3(a,b,c)
#define VFFT_FN(base)        _VFFT_PASTE(base, _, VFFT_ISA_TAG)

/* ═══════════════════════════════════════════════════════════════
 * UNTUNED RADIX HEADERS — production all-in-one codelets for R=2/4/8
 * ═══════════════════════════════════════════════════════════════ */

#if defined(VFFT_ISA_AVX512)
  #include "fft_radix2_avx512.h"
  #include "fft_radix4_avx512.h"
  #include "fft_radix8_avx512.h"
#elif defined(VFFT_ISA_AVX2)
  #include "fft_radix2_avx2.h"
  #include "fft_radix4_avx2.h"
  #include "fft_radix8_avx2.h"
#else
  #include "fft_radix2_scalar.h"
  #include "fft_radix4_scalar.h"
  #include "fft_radix8_scalar.h"
#endif

/* ═══════════════════════════════════════════════════════════════
 * AUXILIARY HEADERS — production n1, n1_scaled, t1_oop
 *
 * The tune generator does not emit these variants. They're consumed by
 * R2C/C2R fused pack-unpack and by the 2D FFT strided executor. Pulled
 * in from the production codelet directory for every tuned radix that
 * has them (the same per-variant header list as stride-fft/registry.h).
 * ═══════════════════════════════════════════════════════════════ */

#if defined(VFFT_ISA_AVX512)
  #include "fft_radix3_avx512_ct_n1.h"
  #include "fft_radix3_avx512_ct_t1_oop_dit.h"
  #include "fft_radix3_avx512_ct_n1_scaled.h"
  #include "fft_radix5_avx512_ct_n1.h"
  #include "fft_radix5_avx512_ct_t1_oop_dit.h"
  #include "fft_radix5_avx512_ct_n1_scaled.h"
  #include "fft_radix6_avx512_ct_n1.h"
  #include "fft_radix6_avx512_ct_t1_oop_dit.h"
  #include "fft_radix6_avx512_ct_n1_scaled.h"
  #include "fft_radix7_avx512_ct_n1.h"
  #include "fft_radix7_avx512_ct_t1_oop_dit.h"
  #include "fft_radix7_avx512_ct_n1_scaled.h"
  #include "fft_radix10_avx512_ct_n1.h"
  #include "fft_radix10_avx512_ct_t1_oop_dit.h"
  #include "fft_radix10_avx512_ct_n1_scaled.h"
  #include "fft_radix11_avx512_ct_n1.h"
  #include "fft_radix11_avx512_ct_t1_oop_dit.h"
  #include "fft_radix11_avx512_ct_n1_scaled.h"
  #include "fft_radix12_avx512_ct_n1.h"
  #include "fft_radix12_avx512_ct_t1_oop_dit.h"
  #include "fft_radix12_avx512_ct_n1_scaled.h"
  #include "fft_radix13_avx512_ct_n1.h"
  #include "fft_radix13_avx512_ct_t1_oop_dit.h"
  #include "fft_radix13_avx512_ct_n1_scaled.h"
  #include "fft_radix16_avx512_ct_n1.h"
  #include "fft_radix16_avx512_ct_t1_oop_dit.h"
  #include "fft_radix16_avx512_ct_n1_scaled.h"
  #include "fft_radix17_avx512_ct_n1.h"
  #include "fft_radix17_avx512_ct_t1_oop_dit.h"
  #include "fft_radix17_avx512_ct_n1_scaled.h"
  #include "fft_radix19_avx512_ct_n1.h"
  #include "fft_radix19_avx512_ct_t1_oop_dit.h"
  #include "fft_radix19_avx512_ct_n1_scaled.h"
  #include "fft_radix20_avx512_ct_n1.h"
  #include "fft_radix20_avx512_ct_t1_oop_dit.h"
  #include "fft_radix20_avx512_ct_n1_scaled.h"
  #include "fft_radix25_avx512_ct_n1.h"
  #include "fft_radix25_avx512_ct_t1_oop_dit.h"
  #include "fft_radix25_avx512_ct_n1_scaled.h"
  #include "fft_radix32_avx512_ct_n1.h"
  #include "fft_radix32_avx512_ct_t1_oop_dit.h"
  #include "fft_radix32_avx512_ct_n1_scaled.h"
  #include "fft_radix64_avx512_ct_n1.h"
  #include "fft_radix64_avx512_ct_t1_oop_dit.h"
  #include "fft_radix64_avx512_ct_n1_scaled.h"
#elif defined(VFFT_ISA_AVX2)
  #include "fft_radix3_avx2_ct_n1.h"
  #include "fft_radix3_avx2_ct_t1_oop_dit.h"
  #include "fft_radix3_avx2_ct_n1_scaled.h"
  #include "fft_radix5_avx2_ct_n1.h"
  #include "fft_radix5_avx2_ct_t1_oop_dit.h"
  #include "fft_radix5_avx2_ct_n1_scaled.h"
  #include "fft_radix6_avx2_ct_n1.h"
  #include "fft_radix6_avx2_ct_t1_oop_dit.h"
  #include "fft_radix6_avx2_ct_n1_scaled.h"
  #include "fft_radix7_avx2_ct_n1.h"
  #include "fft_radix7_avx2_ct_t1_oop_dit.h"
  #include "fft_radix7_avx2_ct_n1_scaled.h"
  #include "fft_radix10_avx2_ct_n1.h"
  #include "fft_radix10_avx2_ct_t1_oop_dit.h"
  #include "fft_radix10_avx2_ct_n1_scaled.h"
  #include "fft_radix11_avx2_ct_n1.h"
  #include "fft_radix11_avx2_ct_t1_oop_dit.h"
  #include "fft_radix11_avx2_ct_n1_scaled.h"
  #include "fft_radix12_avx2_ct_n1.h"
  #include "fft_radix12_avx2_ct_t1_oop_dit.h"
  #include "fft_radix12_avx2_ct_n1_scaled.h"
  #include "fft_radix13_avx2_ct_n1.h"
  #include "fft_radix13_avx2_ct_t1_oop_dit.h"
  #include "fft_radix13_avx2_ct_n1_scaled.h"
  #include "fft_radix16_avx2_ct_n1.h"
  #include "fft_radix16_avx2_ct_t1_oop_dit.h"
  #include "fft_radix16_avx2_ct_n1_scaled.h"
  #include "fft_radix17_avx2_ct_n1.h"
  #include "fft_radix17_avx2_ct_t1_oop_dit.h"
  #include "fft_radix17_avx2_ct_n1_scaled.h"
  #include "fft_radix19_avx2_ct_n1.h"
  #include "fft_radix19_avx2_ct_t1_oop_dit.h"
  #include "fft_radix19_avx2_ct_n1_scaled.h"
  #include "fft_radix20_avx2_ct_n1.h"
  #include "fft_radix20_avx2_ct_t1_oop_dit.h"
  #include "fft_radix20_avx2_ct_n1_scaled.h"
  #include "fft_radix25_avx2_ct_n1.h"
  #include "fft_radix25_avx2_ct_t1_oop_dit.h"
  #include "fft_radix25_avx2_ct_n1_scaled.h"
  #include "fft_radix32_avx2_ct_n1.h"
  #include "fft_radix32_avx2_ct_t1_oop_dit.h"
  #include "fft_radix32_avx2_ct_n1_scaled.h"
  #include "fft_radix64_avx2_ct_n1.h"
  #include "fft_radix64_avx2_ct_t1_oop_dit.h"
  #include "fft_radix64_avx2_ct_n1_scaled.h"
#else
  #include "fft_radix3_scalar_ct_n1.h"
  #include "fft_radix3_scalar_ct_t1_oop_dit.h"
  #include "fft_radix3_scalar_ct_n1_scaled.h"
  #include "fft_radix5_scalar_ct_n1.h"
  #include "fft_radix5_scalar_ct_t1_oop_dit.h"
  #include "fft_radix5_scalar_ct_n1_scaled.h"
  #include "fft_radix6_scalar_ct_n1.h"
  #include "fft_radix6_scalar_ct_t1_oop_dit.h"
  #include "fft_radix6_scalar_ct_n1_scaled.h"
  #include "fft_radix7_scalar_ct_n1.h"
  #include "fft_radix7_scalar_ct_t1_oop_dit.h"
  #include "fft_radix7_scalar_ct_n1_scaled.h"
  #include "fft_radix10_scalar_ct_n1.h"
  #include "fft_radix10_scalar_ct_t1_oop_dit.h"
  #include "fft_radix10_scalar_ct_n1_scaled.h"
  #include "fft_radix11_scalar_ct_n1.h"
  #include "fft_radix11_scalar_ct_t1_oop_dit.h"
  #include "fft_radix11_scalar_ct_n1_scaled.h"
  #include "fft_radix12_scalar_ct_n1.h"
  #include "fft_radix12_scalar_ct_t1_oop_dit.h"
  #include "fft_radix12_scalar_ct_n1_scaled.h"
  #include "fft_radix13_scalar_ct_n1.h"
  #include "fft_radix13_scalar_ct_t1_oop_dit.h"
  #include "fft_radix13_scalar_ct_n1_scaled.h"
  #include "fft_radix16_scalar_ct_n1.h"
  #include "fft_radix16_scalar_ct_t1_oop_dit.h"
  #include "fft_radix16_scalar_ct_n1_scaled.h"
  #include "fft_radix17_scalar_ct_n1.h"
  #include "fft_radix17_scalar_ct_t1_oop_dit.h"
  #include "fft_radix17_scalar_ct_n1_scaled.h"
  #include "fft_radix19_scalar_ct_n1.h"
  #include "fft_radix19_scalar_ct_t1_oop_dit.h"
  #include "fft_radix19_scalar_ct_n1_scaled.h"
  #include "fft_radix20_scalar_ct_n1.h"
  #include "fft_radix20_scalar_ct_t1_oop_dit.h"
  #include "fft_radix20_scalar_ct_n1_scaled.h"
  #include "fft_radix25_scalar_ct_n1.h"
  #include "fft_radix25_scalar_ct_t1_oop_dit.h"
  #include "fft_radix25_scalar_ct_n1_scaled.h"
  #include "fft_radix32_scalar_ct_n1.h"
  #include "fft_radix32_scalar_ct_t1_oop_dit.h"
  #include "fft_radix32_scalar_ct_n1_scaled.h"
  #include "fft_radix64_scalar_ct_n1.h"
  #include "fft_radix64_scalar_ct_t1_oop_dit.h"
  #include "fft_radix64_scalar_ct_n1_scaled.h"
#endif

/* ═══════════════════════════════════════════════════════════════
 * TUNED RADIX DISPATCHER HEADERS
 *
 * Each dispatcher header is static-inline and pulls in the unified
 * codelet header `fft_radix{R}_{isa}.h` from the same generated/r{R}/
 * directory. That unified header contains every variant for the radix
 * (t1_dit baseline, log3, isub2, log_half, t1s, buf tile64/tile128,
 * and DIF families). Including a dispatcher therefore makes every
 * variant available; the dispatcher's branch tree picks one per call.
 *
 * Per-radix include lists below match the dispatchers actually emitted
 * by `select_and_emit.py`. R=16 is the only radix today with a
 * t1_buf_dit family — its include block has the extra dispatcher.
 *
 * R=32 and R=64 dispatcher headers exist only after the orchestrator
 * has been run locally for those radixes. If they're missing, the build
 * will fail with "no such file" pointing at the missing header — the
 * fix is to run the bench, not to edit this file.
 * ═══════════════════════════════════════════════════════════════ */

#if defined(VFFT_ISA_AVX512)
  #include "vfft_r3_t1_dit_dispatch_avx512.h"
  #include "vfft_r3_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r3_t1s_dit_dispatch_avx512.h"
  #include "vfft_r5_t1_dit_dispatch_avx512.h"
  #include "vfft_r5_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r5_t1s_dit_dispatch_avx512.h"
  #include "vfft_r6_t1_dit_dispatch_avx512.h"
  #include "vfft_r6_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r6_t1s_dit_dispatch_avx512.h"
  #include "vfft_r7_t1_dit_dispatch_avx512.h"
  #include "vfft_r7_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r7_t1s_dit_dispatch_avx512.h"
  #include "vfft_r10_t1_dit_dispatch_avx512.h"
  #include "vfft_r10_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r10_t1s_dit_dispatch_avx512.h"
  #include "vfft_r11_t1_dit_dispatch_avx512.h"
  #include "vfft_r11_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r11_t1s_dit_dispatch_avx512.h"
  #include "vfft_r12_t1_dit_dispatch_avx512.h"
  #include "vfft_r12_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r12_t1s_dit_dispatch_avx512.h"
  #include "vfft_r13_t1_dit_dispatch_avx512.h"
  #include "vfft_r13_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r13_t1s_dit_dispatch_avx512.h"
  #include "vfft_r16_t1_dit_dispatch_avx512.h"
  #include "vfft_r16_t1_buf_dit_dispatch_avx512.h"
  #include "vfft_r16_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r16_t1s_dit_dispatch_avx512.h"
  #include "vfft_r17_t1_dit_dispatch_avx512.h"
  #include "vfft_r17_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r17_t1s_dit_dispatch_avx512.h"
  #include "vfft_r19_t1_dit_dispatch_avx512.h"
  #include "vfft_r19_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r19_t1s_dit_dispatch_avx512.h"
  #include "vfft_r20_t1_dit_dispatch_avx512.h"
  #include "vfft_r20_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r20_t1s_dit_dispatch_avx512.h"
  #include "vfft_r25_t1_dit_dispatch_avx512.h"
  #include "vfft_r25_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r25_t1s_dit_dispatch_avx512.h"
  #include "vfft_r32_t1_dit_dispatch_avx512.h"
  #include "vfft_r32_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r64_t1_dit_dispatch_avx512.h"
  #include "vfft_r64_t1_dit_log3_dispatch_avx512.h"
#elif defined(VFFT_ISA_AVX2)
  #include "vfft_r3_t1_dit_dispatch_avx2.h"
  #include "vfft_r3_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r3_t1s_dit_dispatch_avx2.h"
  #include "vfft_r5_t1_dit_dispatch_avx2.h"
  #include "vfft_r5_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r5_t1s_dit_dispatch_avx2.h"
  #include "vfft_r6_t1_dit_dispatch_avx2.h"
  #include "vfft_r6_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r6_t1s_dit_dispatch_avx2.h"
  #include "vfft_r7_t1_dit_dispatch_avx2.h"
  #include "vfft_r7_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r7_t1s_dit_dispatch_avx2.h"
  #include "vfft_r10_t1_dit_dispatch_avx2.h"
  #include "vfft_r10_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r10_t1s_dit_dispatch_avx2.h"
  #include "vfft_r11_t1_dit_dispatch_avx2.h"
  #include "vfft_r11_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r11_t1s_dit_dispatch_avx2.h"
  #include "vfft_r12_t1_dit_dispatch_avx2.h"
  #include "vfft_r12_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r12_t1s_dit_dispatch_avx2.h"
  #include "vfft_r13_t1_dit_dispatch_avx2.h"
  #include "vfft_r13_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r13_t1s_dit_dispatch_avx2.h"
  #include "vfft_r16_t1_dit_dispatch_avx2.h"
  #include "vfft_r16_t1_buf_dit_dispatch_avx2.h"
  #include "vfft_r16_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r16_t1s_dit_dispatch_avx2.h"
  #include "vfft_r17_t1_dit_dispatch_avx2.h"
  #include "vfft_r17_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r17_t1s_dit_dispatch_avx2.h"
  #include "vfft_r19_t1_dit_dispatch_avx2.h"
  #include "vfft_r19_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r19_t1s_dit_dispatch_avx2.h"
  #include "vfft_r20_t1_dit_dispatch_avx2.h"
  #include "vfft_r20_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r20_t1s_dit_dispatch_avx2.h"
  #include "vfft_r25_t1_dit_dispatch_avx2.h"
  #include "vfft_r25_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r25_t1s_dit_dispatch_avx2.h"
  #include "vfft_r32_t1_dit_dispatch_avx2.h"
  #include "vfft_r32_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r64_t1_dit_dispatch_avx2.h"
  #include "vfft_r64_t1_dit_log3_dispatch_avx2.h"
#endif
/* No scalar dispatchers — the tune generator targets only AVX2/AVX-512.
 * Scalar builds use the production codelets via the untuned path. */

/* ═══════════════════════════════════════════════════════════════
 * REGISTRY STRUCTURE
 *
 * Same as stride-fft/core/registry.h plus `t1_buf_fwd` / `t1_buf_bwd`.
 * The new buf slot is populated where a t1_buf_dit dispatcher exists
 * (R=16 today). It's read by the planner once Phase 2.1 lands wisdom
 * with `radix{R}_prefer_buf(me, ios)` — until then no executor path
 * consults this slot.
 * ═══════════════════════════════════════════════════════════════ */

#define STRIDE_REG_MAX_RADIX 128

typedef struct {
    stride_n1_fn n1_fwd[STRIDE_REG_MAX_RADIX];
    stride_n1_fn n1_bwd[STRIDE_REG_MAX_RADIX];

    /* t1 (DIT-flat) baseline — t1_dit dispatcher (or production codelet
     * for untuned R=2/4/8). */
    stride_t1_fn t1_fwd[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1_bwd[STRIDE_REG_MAX_RADIX];

    /* t1_buf (DIT-flat buffered) — t1_buf_dit dispatcher.
     * Populated for R=16 today; NULL elsewhere. Phase 2.1 (wisdom emit)
     * + planner prefer_buf consultation will activate this slot. */
    stride_t1_fn t1_buf_fwd[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1_buf_bwd[STRIDE_REG_MAX_RADIX];

    /* t1 log3 — t1_dit_log3 dispatcher. */
    stride_t1_fn t1_fwd_log3[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1_bwd_log3[STRIDE_REG_MAX_RADIX];

    /* t1s (scalar-broadcast) — t1s_dit dispatcher. */
    stride_t1_fn t1s_fwd[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1s_bwd[STRIDE_REG_MAX_RADIX];

    /* Auxiliary slots — production codelets. Not in the tune portfolio. */
    stride_t1_oop_fn t1_oop_fwd[STRIDE_REG_MAX_RADIX];
    stride_t1_oop_fn t1_oop_bwd[STRIDE_REG_MAX_RADIX];
    stride_n1_scaled_fn n1_scaled_fwd[STRIDE_REG_MAX_RADIX];
    stride_n1_scaled_fn n1_scaled_bwd[STRIDE_REG_MAX_RADIX];

    /* No t1_dif / t1_dif_log3 slots. DIF codelets exist in
     * vectorfft_tune as bench artifacts but are not executor-callable
     * per stage in v1.0. See dit_dif_design_note.md. */
} stride_registry_t;

static const int STRIDE_AVAILABLE_RADIXES[] = {
    64, 32, 25, 20, 19, 17, 16, 13, 12, 11, 10, 8, 7, 6, 5, 4, 3, 2, 0
};

/* ═══════════════════════════════════════════════════════════════
 * REGISTRATION MACROS
 *
 * Tuned variants point at the dispatcher symbols emitted by
 * vectorfft_tune. Untuned variants point at the production codelets
 * directly (R=2/4/8 only).
 *
 * VFFT_FN(base) pastes the ISA suffix:
 *   VFFT_FN(vfft_r16_t1_dit_dispatch_fwd) -> vfft_r16_t1_dit_dispatch_fwd_avx2
 *   VFFT_FN(radix4_t1_dit_fwd)            -> radix4_t1_dit_fwd_avx2
 * ═══════════════════════════════════════════════════════════════ */

#define _REG_N1(R) \
    reg->n1_fwd[R] = (stride_n1_fn)VFFT_FN(radix##R##_n1_fwd); \
    reg->n1_bwd[R] = (stride_n1_fn)VFFT_FN(radix##R##_n1_bwd);

/* Tuned t1: dispatcher symbol (15 radixes that have the dispatcher) */
#define _REG_TUNED_T1(R) \
    reg->t1_fwd[R] = (stride_t1_fn)VFFT_FN(vfft_r##R##_t1_dit_dispatch_fwd); \
    reg->t1_bwd[R] = (stride_t1_fn)VFFT_FN(vfft_r##R##_t1_dit_dispatch_bwd);

#define _REG_TUNED_T1_LOG3(R) \
    reg->t1_fwd_log3[R] = (stride_t1_fn)VFFT_FN(vfft_r##R##_t1_dit_log3_dispatch_fwd); \
    reg->t1_bwd_log3[R] = (stride_t1_fn)VFFT_FN(vfft_r##R##_t1_dit_log3_dispatch_bwd);

#define _REG_TUNED_T1S(R) \
    reg->t1s_fwd[R] = (stride_t1_fn)VFFT_FN(vfft_r##R##_t1s_dit_dispatch_fwd); \
    reg->t1s_bwd[R] = (stride_t1_fn)VFFT_FN(vfft_r##R##_t1s_dit_dispatch_bwd);

#define _REG_TUNED_T1_BUF(R) \
    reg->t1_buf_fwd[R] = (stride_t1_fn)VFFT_FN(vfft_r##R##_t1_buf_dit_dispatch_fwd); \
    reg->t1_buf_bwd[R] = (stride_t1_fn)VFFT_FN(vfft_r##R##_t1_buf_dit_dispatch_bwd);

/* Untuned t1: raw production codelet (R=2/4/8) */
#define _REG_RAW_T1(R) \
    reg->t1_fwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_fwd); \
    reg->t1_bwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_bwd);

#define _REG_RAW_T1_LOG3(R) \
    reg->t1_fwd_log3[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_log3_fwd); \
    reg->t1_bwd_log3[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_log3_bwd);

/* Auxiliary slots: production codelets (every radix that has them) */
#define _REG_T1_OOP(R) \
    reg->t1_oop_fwd[R] = (stride_t1_oop_fn)VFFT_FN(radix##R##_t1_oop_dit_fwd); \
    reg->t1_oop_bwd[R] = (stride_t1_oop_fn)VFFT_FN(radix##R##_t1_oop_dit_bwd);

#define _REG_N1_SCALED(R) \
    reg->n1_scaled_fwd[R] = (stride_n1_scaled_fn)VFFT_FN(radix##R##_n1_scaled_fwd); \
    reg->n1_scaled_bwd[R] = (stride_n1_scaled_fn)VFFT_FN(radix##R##_n1_scaled_bwd);

/* Bundle macros */
#define _REG_TUNED_FULL(R) \
    _REG_N1(R) _REG_TUNED_T1(R) _REG_TUNED_T1_LOG3(R) _REG_TUNED_T1S(R) \
    _REG_T1_OOP(R) _REG_N1_SCALED(R)

/* R=32, R=64: have t1_dit + log3 dispatchers but no t1s in the portfolio */
#define _REG_TUNED_NO_T1S(R) \
    _REG_N1(R) _REG_TUNED_T1(R) _REG_TUNED_T1_LOG3(R) \
    _REG_T1_OOP(R) _REG_N1_SCALED(R)

/* R=16 is the only radix with a buf dispatcher today */
#define _REG_TUNED_FULL_WITH_BUF(R) \
    _REG_N1(R) _REG_TUNED_T1(R) _REG_TUNED_T1_BUF(R) _REG_TUNED_T1_LOG3(R) \
    _REG_TUNED_T1S(R) _REG_T1_OOP(R) _REG_N1_SCALED(R)

/* Untuned bundle: production codelets all the way down */
#define _REG_RAW_FULL(R)    _REG_N1(R) _REG_RAW_T1(R) _REG_RAW_T1_LOG3(R)
#define _REG_RAW_NO_LOG3(R) _REG_N1(R) _REG_RAW_T1(R)

/* ═══════════════════════════════════════════════════════════════
 * REGISTRY INITIALIZATION
 * ═══════════════════════════════════════════════════════════════ */

static void stride_registry_init(stride_registry_t *reg) {
    memset(reg, 0, sizeof(*reg));

    /* Untuned radixes — production codelets directly */
    _REG_RAW_NO_LOG3(2)                 _REG_T1_OOP(2)  _REG_N1_SCALED(2)
    _REG_RAW_FULL(4)                    _REG_T1_OOP(4)  _REG_N1_SCALED(4)
    _REG_RAW_NO_LOG3(8)                 _REG_T1_OOP(8)  _REG_N1_SCALED(8)

    /* Tuned radixes — dispatcher symbols */
    _REG_TUNED_FULL(3)
    _REG_TUNED_FULL(5)
    _REG_TUNED_FULL(6)
    _REG_TUNED_FULL(7)
    _REG_TUNED_FULL(10)
    _REG_TUNED_FULL(11)
    _REG_TUNED_FULL(12)
    _REG_TUNED_FULL(13)
    _REG_TUNED_FULL_WITH_BUF(16)        /* R=16 is the only buf today */
    _REG_TUNED_FULL(17)
    _REG_TUNED_FULL(19)
    _REG_TUNED_FULL(20)
    _REG_TUNED_FULL(25)
    _REG_TUNED_NO_T1S(32)               /* R=32: no t1s in tune portfolio */
    _REG_TUNED_NO_T1S(64)               /* R=64: no t1s in tune portfolio */
}

#undef _REG_N1
#undef _REG_TUNED_T1
#undef _REG_TUNED_T1_LOG3
#undef _REG_TUNED_T1S
#undef _REG_TUNED_T1_BUF
#undef _REG_RAW_T1
#undef _REG_RAW_T1_LOG3
#undef _REG_T1_OOP
#undef _REG_N1_SCALED
#undef _REG_TUNED_FULL
#undef _REG_TUNED_NO_T1S
#undef _REG_TUNED_FULL_WITH_BUF
#undef _REG_RAW_FULL
#undef _REG_RAW_NO_LOG3

/* ═══════════════════════════════════════════════════════════════
 * QUERIES
 * ═══════════════════════════════════════════════════════════════ */

static inline int stride_registry_has(const stride_registry_t *reg, int radix) {
    return radix > 0 && radix < STRIDE_REG_MAX_RADIX && reg->n1_fwd[radix] != NULL;
}

static inline int stride_registry_has_t1(const stride_registry_t *reg, int radix) {
    return radix > 0 && radix < STRIDE_REG_MAX_RADIX && reg->t1_fwd[radix] != NULL;
}

/* Does this radix have a buffered-flat dispatcher in the registry?
 * Used by future planner prefer_buf consultation (Phase 2.1+). */
static inline int stride_registry_has_t1_buf(const stride_registry_t *reg, int radix) {
    return radix > 0 && radix < STRIDE_REG_MAX_RADIX && reg->t1_buf_fwd[radix] != NULL;
}

#endif /* VFFT_CORE_REGISTRY_H */
