/**
 * src/core/registry.h — codelet registry for the orchestrator-tuned executor.
 *
 * Wires the per-host codelet dispatchers emitted by vectorfft_tune into the
 * registry that the planner and executor consume. Each radix's t1 slots
 * point at static-inline dispatchers from
 * `vectorfft_tune/generated/r{R}/vfft_r{R}_{family}_dispatch_{isa}.h`.
 * Each dispatcher is a per-(me, ios) selector over the variants within
 * one dispatcher family; calling it costs one inlined branch tree plus a
 * codelet call.
 *
 * Differences from stride-fft/core/registry.h
 * -------------------------------------------
 * 1. New slot: `t1_buf_fwd` / `t1_buf_bwd` (buffered-flat dispatcher).
 *    Populated for radixes with a t1_buf_dit family (R=16, R=32, R=64).
 *    NULL elsewhere. Currently no consumer reads this slot — Phase 2.1
 *    (Python-side wisdom emit + planner consultation) needs to land
 *    before the planner consults `radix{R}_prefer_buf(me, ios)` to
 *    choose between t1_dit and t1_buf_dit. The slot is wired but inert
 *    until then.
 *
 * 2. No `t1_dif` slot. DIF codelets exist as bench artifacts (the
 *    `vfft_r{R}_t1_dif_dispatch_*.h` files) but the DIT-structured
 *    forward executor cannot substitute them per stage — DIT and DIF
 *    compute different output buffers given the same input and twiddles.
 *    Filtered for v1.0 per `dit_dif_design_note.md`.
 *
 * 3. Tuned vs untuned radixes
 *    Tuned (dispatchers + plan_wisdom): 3, 4, 5, 6, 7, 8, 10, 11, 12,
 *      13, 16, 17, 19, 20, 25, 32, 64. Every radix in the tune portfolio
 *      has all three core dispatchers (t1_dit, t1_dit_log3, t1s_dit);
 *      R=16/R=32/R=64 also have t1_buf_dit.
 *    Untuned (production codelets used directly): R=2 only.
 *
 * Auxiliary slots (n1, n1_scaled, t1_oop) come from the production
 * codelet headers under stride-fft/codelets/{isa}/. The tune generator
 * does not emit those variants — they're used by R2C, C2R, and 2D
 * paths that have separate dispatcher concerns outside this work.
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
/* Use the SAME guard as production registry.h (stride-fft/core/registry.h)
 * so that when production headers (e.g. factorizer.h) sibling-include
 * "registry.h" and find the production version, the guard is already set
 * and the production content is skipped. The new core's planner.h includes
 * THIS file first via the -I order, ensuring this version wins. */
#ifndef STRIDE_REGISTRY_H
#define STRIDE_REGISTRY_H

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
 * RADIX HEADERS — all from vectorfft_tune/generated/r{R}/
 *
 * As of v1.0, the new core pulls every SIMD codelet from the
 * vectorfft_tune tree. No #include from src/stride-fft/codelets/.
 *
 * R=2: copied once from production into generated/r2/ (bootstrap —
 *      no orchestrator coverage for R=2, single variant only).
 * R=4, R=8: legacy unified header with all variants baked in (n1,
 *      n1_scaled, t1_oop_dit, plus tuned variants).
 * R=3..R=64 except 4/8: bench-portfolio variants in unified header
 *      fft_radix{R}_{isa}.h, aux variants (n1, n1_scaled, t1_oop_dit)
 *      as separate per-variant headers fft_radix{R}_{isa}_ct_*.h.
 *      Both produced by phase_generate in common/bench.py and copied
 *      to generated/r{R}/ by phase_emit.
 *
 * Scalar fallback path remains tied to stride-fft/codelets/scalar/
 * because the orchestrator only targets AVX2 / AVX-512.
 * ═══════════════════════════════════════════════════════════════ */

/* R=2 — single variant, no tune dispatcher needed. */
#if defined(VFFT_ISA_AVX512)
  #include "fft_radix2_avx512.h"
#elif defined(VFFT_ISA_AVX2)
  #include "fft_radix2_avx2.h"
#else
  #include "fft_radix2_scalar.h"
#endif

/* ═══════════════════════════════════════════════════════════════
 * AUXILIARY HEADERS — n1, n1_scaled, t1_oop (R≥3, except R=4/R=8)
 *
 * These are NOT bench-tuned (single variant per radix) but are needed
 * by the new core's R2C/C2R/2D paths. Emitted by bench.py's
 * phase_generate alongside the unified bench-portfolio header, copied
 * to generated/r{R}/ by phase_emit.
 *
 * For R=4 and R=8 these are baked into the legacy unified header and
 * pulled in transitively via the dispatcher includes below.
 * ═══════════════════════════════════════════════════════════════ */

#if defined(VFFT_ISA_AVX512)
  #define _VFFT_AUX_HDR(R) \
      "fft_radix" #R "_avx512_ct_n1.h"
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
  /* R=4 and R=8 use legacy fft_radix4_avx512.h / fft_radix8_avx512.h
   * which include n1 inline — no separate ct_n1 headers. */
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
  /* R=4 and R=8: NOT included here. The vectorfft_tune dispatcher headers
   * (vfft_r{4,8}_*_dispatch_avx2.h) sibling-include the generated unified
   * `fft_radix{4,8}_avx2.h` from generated/r{4,8}/, which has the FULL
   * variant set (log1, log3, t1s, plus n1 / n1_scaled / t1_oop aux).
   * The production legacy `src/stride-fft/codelets/avx2/fft_radix{4,8}_avx2.h`
   * uses the same #include guard `FFT_RADIX{4,8}_AVX2_H` but with FEWER
   * variants. Including the production header here would set the guard
   * first, causing the dispatcher's include of the unified version to
   * be skipped — and tuned dispatchers would then reference undefined
   * symbols (radix4_t1s_dit_fwd_avx2, radix8_t1_dit_log1_fwd_avx2, etc.).
   * Letting the dispatcher pull the unified version first sidesteps this. */
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
  #include "fft_radix4_scalar.h"
  #include "fft_radix8_scalar.h"
  /* Scalar fallback: production t1_dit / t1_dit_log3 / t1s_dit codelets.
   * The tune generator only emits SIMD dispatchers, so scalar builds use
   * production codelets directly. R=32 and R=64 have no scalar t1s
   * codelet — those slots stay NULL via macro gating below. */
  #include "fft_radix3_scalar_ct_t1_dit.h"
  #include "fft_radix3_scalar_ct_t1_dit_log3.h"
  #include "fft_radix3_scalar_ct_t1s_dit.h"
  #include "fft_radix5_scalar_ct_t1_dit.h"
  #include "fft_radix5_scalar_ct_t1_dit_log3.h"
  #include "fft_radix5_scalar_ct_t1s_dit.h"
  #include "fft_radix6_scalar_ct_t1_dit.h"
  #include "fft_radix6_scalar_ct_t1_dit_log3.h"
  #include "fft_radix6_scalar_ct_t1s_dit.h"
  #include "fft_radix7_scalar_ct_t1_dit.h"
  #include "fft_radix7_scalar_ct_t1_dit_log3.h"
  #include "fft_radix7_scalar_ct_t1s_dit.h"
  #include "fft_radix10_scalar_ct_t1_dit.h"
  #include "fft_radix10_scalar_ct_t1_dit_log3.h"
  #include "fft_radix10_scalar_ct_t1s_dit.h"
  #include "fft_radix11_scalar_ct_t1_dit.h"
  #include "fft_radix11_scalar_ct_t1_dit_log3.h"
  #include "fft_radix11_scalar_ct_t1s_dit.h"
  #include "fft_radix12_scalar_ct_t1_dit.h"
  #include "fft_radix12_scalar_ct_t1_dit_log3.h"
  #include "fft_radix12_scalar_ct_t1s_dit.h"
  #include "fft_radix13_scalar_ct_t1_dit.h"
  #include "fft_radix13_scalar_ct_t1_dit_log3.h"
  #include "fft_radix13_scalar_ct_t1s_dit.h"
  #include "fft_radix16_scalar_ct_t1_dit.h"
  #include "fft_radix16_scalar_ct_t1_dit_log3.h"
  #include "fft_radix16_scalar_ct_t1s_dit.h"
  #include "fft_radix17_scalar_ct_t1_dit.h"
  #include "fft_radix17_scalar_ct_t1_dit_log3.h"
  #include "fft_radix17_scalar_ct_t1s_dit.h"
  #include "fft_radix19_scalar_ct_t1_dit.h"
  #include "fft_radix19_scalar_ct_t1_dit_log3.h"
  #include "fft_radix19_scalar_ct_t1s_dit.h"
  #include "fft_radix20_scalar_ct_t1_dit.h"
  #include "fft_radix20_scalar_ct_t1_dit_log3.h"
  #include "fft_radix20_scalar_ct_t1s_dit.h"
  #include "fft_radix25_scalar_ct_t1_dit.h"
  #include "fft_radix25_scalar_ct_t1_dit_log3.h"
  #include "fft_radix25_scalar_ct_t1s_dit.h"
  #include "fft_radix32_scalar_ct_t1_dit.h"
  #include "fft_radix32_scalar_ct_t1_dit_log3.h"
  /* R=32 has no scalar t1s codelet */
  #include "fft_radix64_scalar_ct_t1_dit.h"
  #include "fft_radix64_scalar_ct_t1_dit_log3.h"
  /* R=64 has no scalar t1s codelet */
#endif

/* ═══════════════════════════════════════════════════════════════
 * TUNED RADIX DISPATCHER HEADERS
 *
 * Every tuned radix has three core dispatchers: t1_dit, t1_dit_log3,
 * t1s_dit. R=16/R=32/R=64 additionally have t1_buf_dit. Per-radix
 * include lists below match what the orchestrator emits.
 *
 * Each dispatcher header is static-inline and pulls in the unified
 * codelet header `fft_radix{R}_{isa}.h` from the same generated/r{R}/
 * directory. The unified header contains every variant for the radix;
 * including a dispatcher therefore makes every variant available, and
 * the dispatcher's branch tree picks one per call.
 *
 * Note: these headers are emitted by per-host calibration. They must
 * exist at build time — if missing, run `python orchestrator.py` for
 * the missing radixes. Build error on missing header is the intended
 * signal, not a bug.
 * ═══════════════════════════════════════════════════════════════ */

#if defined(VFFT_ISA_AVX512)
  #include "vfft_r3_t1_dit_dispatch_avx512.h"
  #include "vfft_r3_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r3_t1s_dit_dispatch_avx512.h"
  #include "vfft_r4_t1_dit_dispatch_avx512.h"
  #include "vfft_r4_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r4_t1s_dit_dispatch_avx512.h"
  #include "vfft_r5_t1_dit_dispatch_avx512.h"
  #include "vfft_r5_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r5_t1s_dit_dispatch_avx512.h"
  #include "vfft_r6_t1_dit_dispatch_avx512.h"
  #include "vfft_r6_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r6_t1s_dit_dispatch_avx512.h"
  #include "vfft_r7_t1_dit_dispatch_avx512.h"
  #include "vfft_r7_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r7_t1s_dit_dispatch_avx512.h"
  #include "vfft_r8_t1_dit_dispatch_avx512.h"
  #include "vfft_r8_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r8_t1s_dit_dispatch_avx512.h"
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
  #include "vfft_r32_t1_buf_dit_dispatch_avx512.h"
  #include "vfft_r32_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r32_t1s_dit_dispatch_avx512.h"
  #include "vfft_r64_t1_dit_dispatch_avx512.h"
  #include "vfft_r64_t1_buf_dit_dispatch_avx512.h"
  #include "vfft_r64_t1_dit_log3_dispatch_avx512.h"
  #include "vfft_r64_t1s_dit_dispatch_avx512.h"
#elif defined(VFFT_ISA_AVX2)
  #include "vfft_r3_t1_dit_dispatch_avx2.h"
  #include "vfft_r3_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r3_t1s_dit_dispatch_avx2.h"
  #include "vfft_r4_t1_dit_dispatch_avx2.h"
  #include "vfft_r4_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r4_t1s_dit_dispatch_avx2.h"
  #include "vfft_r5_t1_dit_dispatch_avx2.h"
  #include "vfft_r5_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r5_t1s_dit_dispatch_avx2.h"
  #include "vfft_r6_t1_dit_dispatch_avx2.h"
  #include "vfft_r6_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r6_t1s_dit_dispatch_avx2.h"
  #include "vfft_r7_t1_dit_dispatch_avx2.h"
  #include "vfft_r7_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r7_t1s_dit_dispatch_avx2.h"
  #include "vfft_r8_t1_dit_dispatch_avx2.h"
  #include "vfft_r8_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r8_t1s_dit_dispatch_avx2.h"
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
  #include "vfft_r32_t1_buf_dit_dispatch_avx2.h"
  #include "vfft_r32_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r32_t1s_dit_dispatch_avx2.h"
  #include "vfft_r64_t1_dit_dispatch_avx2.h"
  #include "vfft_r64_t1_buf_dit_dispatch_avx2.h"
  #include "vfft_r64_t1_dit_log3_dispatch_avx2.h"
  #include "vfft_r64_t1s_dit_dispatch_avx2.h"
#endif
/* No scalar dispatchers — the tune generator targets only AVX2/AVX-512.
 * Scalar builds use the production codelets via the untuned path. */

/* ═══════════════════════════════════════════════════════════════
 * REGISTRY STRUCTURE
 *
 * Same as stride-fft/core/registry.h plus `t1_buf_fwd` / `t1_buf_bwd`.
 * The new buf slot is populated where a t1_buf_dit dispatcher exists
 * (R=16, R=32, R=64). It's read by the planner once Phase 2.1 lands
 * wisdom with `radix{R}_prefer_buf(me, ios)` — until then no executor
 * path consults this slot.
 * ═══════════════════════════════════════════════════════════════ */

#define STRIDE_REG_MAX_RADIX 128

typedef struct {
    stride_n1_fn n1_fwd[STRIDE_REG_MAX_RADIX];
    stride_n1_fn n1_bwd[STRIDE_REG_MAX_RADIX];

    /* t1 (DIT-flat) baseline — t1_dit dispatcher (or production codelet
     * for R=2). */
    stride_t1_fn t1_fwd[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1_bwd[STRIDE_REG_MAX_RADIX];

    /* t1_buf (DIT-flat buffered) — t1_buf_dit dispatcher.
     * Populated for R=16, R=32, R=64; NULL elsewhere. Phase 2.1 (wisdom
     * emit) + planner prefer_buf consultation will activate this slot. */
    stride_t1_fn t1_buf_fwd[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1_buf_bwd[STRIDE_REG_MAX_RADIX];

    /* t1 log3 — t1_dit_log3 dispatcher. */
    stride_t1_fn t1_fwd_log3[STRIDE_REG_MAX_RADIX];
    stride_t1_fn t1_bwd_log3[STRIDE_REG_MAX_RADIX];

    /* t1s (scalar-broadcast) — t1s_dit dispatcher.
     * Populated for every tuned radix (3..64). t1s wins broadly at
     * small me — even at R=32/R=64 (me ∈ {64, 96, 128}) — so this
     * slot is not optional. */
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
 * vectorfft_tune. R=2 (the only untuned radix) points at the raw
 * production codelet directly.
 *
 * VFFT_FN(base) pastes the ISA suffix:
 *   VFFT_FN(vfft_r16_t1_dit_dispatch_fwd) -> vfft_r16_t1_dit_dispatch_fwd_avx2
 *   VFFT_FN(radix2_t1_dit_fwd)            -> radix2_t1_dit_fwd_avx2
 * ═══════════════════════════════════════════════════════════════ */

#define _REG_N1(R) \
    reg->n1_fwd[R] = (stride_n1_fn)VFFT_FN(radix##R##_n1_fwd); \
    reg->n1_bwd[R] = (stride_n1_fn)VFFT_FN(radix##R##_n1_bwd);

/* Tuned t1 macros differ by ISA:
 *   AVX2/AVX-512: point at the per-host dispatcher symbol
 *   scalar:       fall back to raw production codelet (the tune
 *                 generator targets only SIMD ISAs; scalar builds
 *                 are development/portability paths and don't need
 *                 per-cell dispatch).
 */
#if defined(VFFT_ISA_SCALAR)
  #define _REG_TUNED_T1(R) \
      reg->t1_fwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_fwd); \
      reg->t1_bwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_bwd);
  #define _REG_TUNED_T1_LOG3(R) \
      reg->t1_fwd_log3[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_log3_fwd); \
      reg->t1_bwd_log3[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_log3_bwd);
  #define _REG_TUNED_T1S(R) \
      reg->t1s_fwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1s_dit_fwd); \
      reg->t1s_bwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1s_dit_bwd);
  /* No buf codelets in production scalar — slot stays NULL. */
  #define _REG_TUNED_T1_BUF(R) /* no-op for scalar */
#else
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
#endif

/* R=2 (untuned): raw production codelet */
#define _REG_RAW_T1(R) \
    reg->t1_fwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_fwd); \
    reg->t1_bwd[R] = (stride_t1_fn)VFFT_FN(radix##R##_t1_dit_bwd);

/* Auxiliary slots: production codelets (every radix that has them) */
#define _REG_T1_OOP(R) \
    reg->t1_oop_fwd[R] = (stride_t1_oop_fn)VFFT_FN(radix##R##_t1_oop_dit_fwd); \
    reg->t1_oop_bwd[R] = (stride_t1_oop_fn)VFFT_FN(radix##R##_t1_oop_dit_bwd);

#define _REG_N1_SCALED(R) \
    reg->n1_scaled_fwd[R] = (stride_n1_scaled_fn)VFFT_FN(radix##R##_n1_scaled_fwd); \
    reg->n1_scaled_bwd[R] = (stride_n1_scaled_fn)VFFT_FN(radix##R##_n1_scaled_bwd);

/* Bundle macros for tuned radixes.
 * _REG_TUNED_FULL: t1_dit + t1_dit_log3 + t1s_dit dispatchers.
 *                  Auxiliary t1_oop + n1_scaled where they exist.
 * _REG_TUNED_FULL_WITH_BUF: above plus t1_buf_dit dispatcher (R=16/32/64). */
#define _REG_TUNED_FULL(R) \
    _REG_N1(R) _REG_TUNED_T1(R) _REG_TUNED_T1_LOG3(R) _REG_TUNED_T1S(R) \
    _REG_T1_OOP(R) _REG_N1_SCALED(R)

#define _REG_TUNED_FULL_WITH_BUF(R) \
    _REG_N1(R) _REG_TUNED_T1(R) _REG_TUNED_T1_BUF(R) \
    _REG_TUNED_T1_LOG3(R) _REG_TUNED_T1S(R) \
    _REG_T1_OOP(R) _REG_N1_SCALED(R)

/* Bundle macro identical to _REG_TUNED_FULL — kept as a separate name
 * because R=4/R=8 use the legacy all-in-one fft_radix{4,8}_avx2.h
 * headers (vs per-variant ct_n1/ct_t1_oop/ct_n1_scaled headers for
 * other radixes). Both header forms expose radix{R}_t1_oop_dit_*
 * and radix{R}_n1_scaled_* symbols, so the macro itself is the same. */
#define _REG_TUNED_FULL_LEGACY_HDR(R) \
    _REG_N1(R) _REG_TUNED_T1(R) _REG_TUNED_T1_LOG3(R) _REG_TUNED_T1S(R) \
    _REG_T1_OOP(R) _REG_N1_SCALED(R)

/* R=2 untuned: production codelet for t1, no log3, no t1s, no buf */
#define _REG_RAW_R2() \
    _REG_N1(2) _REG_RAW_T1(2)

/* ═══════════════════════════════════════════════════════════════
 * REGISTRY INITIALIZATION
 *
 * Per-radix wiring summary (tune portfolio bench, this host):
 *   R=2       : untuned (raw production t1_dit only)
 *   R=4       : tuned (t1_dit, log3, t1s; no buf, no aux n1_scaled/oop)
 *   R=8       : same as R=4
 *   R=16/32/64: tuned with buf (t1_dit, t1_buf_dit, log3, t1s, aux)
 *   others    : tuned without buf (t1_dit, log3, t1s, aux)
 * ═══════════════════════════════════════════════════════════════ */

static void stride_registry_init(stride_registry_t *reg) {
    memset(reg, 0, sizeof(*reg));

    /* R=2: untuned. */
    _REG_RAW_R2()

    /* R=3..R=64: tuned via dispatchers. */
    _REG_TUNED_FULL(3)
    _REG_TUNED_FULL_LEGACY_HDR(4)          /* aux variants live in legacy hdr */
    _REG_TUNED_FULL(5)
    _REG_TUNED_FULL(6)
    _REG_TUNED_FULL(7)
    _REG_TUNED_FULL_LEGACY_HDR(8)          /* aux variants live in legacy hdr */
    _REG_TUNED_FULL(10)
    _REG_TUNED_FULL(11)
    _REG_TUNED_FULL(12)
    _REG_TUNED_FULL(13)
    _REG_TUNED_FULL_WITH_BUF(16)
    _REG_TUNED_FULL(17)
    _REG_TUNED_FULL(19)
    _REG_TUNED_FULL(20)
    _REG_TUNED_FULL(25)
#if defined(VFFT_ISA_SCALAR)
    /* R=32 and R=64 have no scalar t1s codelet — the production scalar
     * portfolio never built one. Wire t1_dit + t1_dit_log3 only; t1s
     * slot stays NULL. The SIMD path uses tuned dispatchers which DO
     * include t1s for these radixes (per orchestrator measurements). */
    _REG_N1(32) _REG_TUNED_T1(32) _REG_TUNED_T1_LOG3(32)
        _REG_T1_OOP(32) _REG_N1_SCALED(32)
    _REG_N1(64) _REG_TUNED_T1(64) _REG_TUNED_T1_LOG3(64)
        _REG_T1_OOP(64) _REG_N1_SCALED(64)
#else
    _REG_TUNED_FULL_WITH_BUF(32)
    _REG_TUNED_FULL_WITH_BUF(64)
#endif
}

#undef _REG_N1
#undef _REG_TUNED_T1
#undef _REG_TUNED_T1_LOG3
#undef _REG_TUNED_T1S
#undef _REG_TUNED_T1_BUF
#undef _REG_RAW_T1
#undef _REG_T1_OOP
#undef _REG_N1_SCALED
#undef _REG_TUNED_FULL
#undef _REG_TUNED_FULL_WITH_BUF
#undef _REG_TUNED_FULL_LEGACY_HDR
#undef _REG_RAW_R2

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

#endif /* STRIDE_REGISTRY_H */
