/* estimate_plan.h — V4 cost-model-driven estimate planner for prototype-core.
 *
 * Enumerates every factorization of N into available radixes (up to
 * STRIDE_MAX_STAGES stages), scores each with the V4 cost model
 * (V2 structure + wide-radix-outer penalty + per-stage buffer-pass cost),
 * picks the lowest-scoring shape, and builds a plan from it.
 *
 * NO MEASUREMENT — pure cost-model scoring. Plan creation is microseconds.
 *
 * Trade-off vs DP planner:
 *   - DP measures sub-plans, picks based on measured wall time at each
 *     cache slot. Right family but biases against wide-radix-innermost
 *     due to first-stage-first recursion + sub-cache pruning.
 *   - V4 estimate is purely model-based. Captures "fewer stages wins"
 *     and "wide-radix-outer is bad" through explicit physical terms
 *     (buffer-pass-per-stage, wide-radix-load-stall). Right family,
 *     captures the recipe pattern your wisdom encodes.
 *
 * Use case: cells without wisdom coverage. Estimate-without-calibration.
 *
 * Variant axis: NOT searched here. V4 picks the min-CPE variant per
 * stage via _radix_butterfly_cost; the resulting plan is built with
 * variants=NULL (T1S defaults). For variant-aware plans, layer a
 * measurement step on top — that's the hybrid path, not this header.
 *
 * Bandwidth/latency constants: calibrated for Raptor Lake @ 5.7 GHz.
 * On other hosts these should be re-probed; see header below. */
#ifndef VFFT_PROTO_CORE_ESTIMATE_PLAN_H
#define VFFT_PROTO_CORE_ESTIMATE_PLAN_H

#include "plan.h"
#include "twiddle.h"
#include "planner.h"
#include "../prototype/cost_model/factorizer.h"  /* stride_cpu_info_t,
                                                  * _radix_butterfly_cost,
                                                  * stride_detect_cpu */
#include "../prototype/cost_model/generated/radix_memboundness.h"  /* measured
                                                  * per-(R, tier) CPE inflation
                                                  * factors; replaces V4's
                                                  * wide_penalty heuristic for
                                                  * R ≥ 16 */
#include "registry.h"
#include <stdlib.h>
#include <string.h>

/* ─────────────────────────────────────────────────────────────────
 * V4 SCORING — same formula as demo_cost_model_rank.c
 *
 *   per-stage score = data_cost + tw_cost + dtlb_cost + wide_penalty
 *                   + buffer_pass_cost_per_stage
 *
 * Bandwidth (bytes/cycle at 5.7 GHz):
 *   L1   ~45    (256 GB/s)
 *   L2   ~14    (80 GB/s)
 *   L3    ~7    (40 GB/s)
 *   DRAM  ~9    (50 GB/s — DDR5 single channel)
 * ───────────────────────────────────────────────────────────────── */

static inline double _vfft_proto_v4_score(
    const int *factors, int nf, size_t K, int N,
    const stride_cpu_info_t *cpu)
{
    const size_t l1 = cpu->l1d_bytes;
    const size_t l2 = cpu->l2_bytes;
    const size_t l3 = cpu->l3_bytes;
    const int dtlb_entries     = (cpu->dtlb_entries    > 0) ? cpu->dtlb_entries    : 96;
    const int dtlb_miss_cycles = (cpu->dtlb_miss_cycles > 0) ? cpu->dtlb_miss_cycles : 7;
#if defined(__AVX512F__)
    const int isa_avx512 = 1;
#else
    const int isa_avx512 = 0;
#endif
    double score = 0.0;
    const size_t total_bytes = (size_t)N * K * 16;
    const double fraction_hot = (total_bytes > 0 && l1 < total_bytes)
                              ? (double)l1 / (double)total_bytes : 1.0;
    const double cf_buffer = (total_bytes > l3) ? 4.0 : 0.0;

    /* extra_lat_* and baseline_R were V4's wide_penalty knobs (2026-05-17
     * onward subsumed by the measured radix_memboundness.h table for R ≥ 16
     * and dropped for R < 16). Left as documentation of what the heuristic
     * looked like before the measurement; see [v4_wide_penalty_artifact]. */

    /* Buffer-pass cost per stage transition.
     *
     * Was hardcoded `2 × total_bytes / bytes_per_cycle` where
     * bytes_per_cycle = 45/14/7/9 from spec sheets. Now reads the
     * MEASURED stride-1 memcpy throughput from radix_memboundness.h's
     * companion table (2026-05-17 onward). Reality on Raptor Lake with
     * AVX2 memcpy is ~6-15× faster than spec sheets predicted, because
     * memcpy uses streaming stores + HW prefetcher gets it right.
     *
     * The 2× factor (one read + one write per stage) stays — that's a
     * structural property of in-place FFT, not measurement-dependent. */
    int total_tier = (total_bytes <= l1) ? STRIDE_MB_TIER_L1
                   : (total_bytes <= l2) ? STRIDE_MB_TIER_L2
                   : (total_bytes <= l3) ? STRIDE_MB_TIER_L3
                                          : STRIDE_MB_TIER_DRAM;
    double cyc_per_byte_pass = stride_cache_cyc_per_byte_avx2[total_tier];
    if (cyc_per_byte_pass <= 0.0) {
        /* Fall back to spec-sheet constants if the measurement table is
         * empty (e.g., other ISA, or before measure_memboundness was run). */
        double bpc = (total_bytes <= l1) ? 45.0
                   : (total_bytes <= l2) ? 14.0
                   : (total_bytes <= l3) ?  7.0 : 9.0;
        cyc_per_byte_pass = 1.0 / bpc;
    }
    const double buffer_pass_cost_per_stage =
        2.0 * (double)total_bytes * cyc_per_byte_pass;

    size_t accumulated_K = K;
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        int groups = N / R;
        size_t stride = K;
        for (int d = s + 1; d < nf; d++) stride *= factors[d];

        size_t me_s = (size_t)(N / R) * K;
        double bf_cost = _radix_butterfly_cost(R, s, me_s, isa_avx512);

        size_t ws = (size_t)R * stride * 16;
        int tier = (ws <= l1) ? STRIDE_MB_TIER_L1
                 : (ws <= l2) ? STRIDE_MB_TIER_L2
                 : (ws <= l3) ? STRIDE_MB_TIER_L3
                              : STRIDE_MB_TIER_DRAM;

        /* Cache scaling factor. For R ≥ 16 we use the MEASURED memory-
         * boundness factor from radix_memboundness.h (2026-05-17 onward).
         * For smaller R (no measurement table entry), fall back to the
         * heuristic cf_stage. The mb_factor SUBSUMES what V4's wide_penalty
         * term used to add: the codelet's actual per-tier slowdown is
         * captured directly, not approximated via (R-4) × extra_lat. */
        double cache_scale;
        if (R >= 16 && R < STRIDE_RADIX_PROFILE_MAX_R
            && stride_radix_memboundness_avx2[R].factor[STRIDE_MB_TIER_L1] > 0.0)
        {
            cache_scale = stride_radix_memboundness_avx2[R].factor[tier];
            if (cache_scale < 1.0) cache_scale = 1.0;  /* guard against table holes */
        } else {
            /* Small-radix heuristic, unchanged from original V4. */
            cache_scale = (ws <= l1) ? 1.0
                        : (ws <= l2) ? 1.4
                        : (ws <= l3) ? 2.3
                                     : 4.0;
        }
        double cf_cold = (cache_scale > cf_buffer) ? cache_scale : cf_buffer;

        double cf_eff = (s == 0) ? cf_cold
                                  : fraction_hot * 1.0 + (1.0 - fraction_hot) * cf_cold;

        double data_cost = (double)groups * (double)K * bf_cost * cf_eff;

        /* dtlb_cost REMOVED (2026-05-17).
         *
         * Old term charged `groups × (pages_per_group − 96) × 7 cyc`
         * for any group whose stride spans more pages than the L1 DTLB
         * (96 entries on Raptor Lake). At N=131072 K=4 [2,4,64,16,16],
         * the S1 R=4 stage spans 512 pages per group → V4 charged 95M
         * cycles for that ONE stage's "page walks". Patient measurement
         * for that plan is 1.78 ms — V4 was 31× off.
         *
         * Why it was wrong: Raptor Lake's STLB holds 2048 entries
         * (vs DTLB's 96); page walks are nearly never required for
         * working sets under ~8 MB. Plus the HW prefetcher prefetches
         * page-table entries for predictable strided patterns. The
         * `7 × excess` formula models neither.
         *
         * Could be re-added with proper STLB modeling + HW-prefetch
         * awareness, but for now the term is doing strictly more
         * harm than good. */

        double tw_cost = 0.0;
        if (s > 0) {
            /* Twiddle-load cost. With HW prefetcher + sequential access
             * pattern + dual load ports, the per-element cost is well
             * under 1 cyc — measured at K=256 inner stages it's effectively
             * hidden behind compute. The old V4 charged `4 × (R-1) × acc_K`
             * for L1-spilled twiddle tables, which dominates the score for
             * deep plans (e.g., N=131072 K=4 inner R=16 has acc_K=32768
             * and tw_cost was ~2M cyc) and breaks rankings vs measurement.
             *
             * Scale down to 0.5 cyc/element when in L1, 1.0 when spilled.
             * Mirrors HW reality where the codelet's twiddle stream is
             * mostly bandwidth-not-latency-bound.
             */
            size_t tw_bytes = (size_t)(R - 1) * accumulated_K * 16;
            if (tw_bytes > l1) tw_cost = (double)(R - 1) * accumulated_K * 1.0;
            else               tw_cost = (double)(R - 1) * accumulated_K * 0.5;
        }

        score += data_cost + tw_cost + buffer_pass_cost_per_stage;
        accumulated_K *= R;
    }
    return score;
}

/* ─────────────────────────────────────────────────────────────────
 * FACTORIZATION ENUMERATION + SCORING
 *
 * Recursively walks every ordered factorization of N using available
 * radixes from the registry. Up to STRIDE_MAX_STAGES stages. Keeps the
 * single lowest-scoring shape.
 * ───────────────────────────────────────────────────────────────── */

typedef struct {
    int N;
    size_t K;
    const stride_cpu_info_t *cpu;
    const vfft_proto_registry_t *reg;
    int current[STRIDE_MAX_STAGES];
    int best[STRIDE_MAX_STAGES];
    int best_nf;
    double best_score;
} _vfft_proto_estimate_search_t;

static inline void _vfft_proto_estimate_recurse(
    _vfft_proto_estimate_search_t *s, int remaining, int nf)
{
    if (remaining == 1) {
        if (nf == 0) return;
        double sc = _vfft_proto_v4_score(s->current, nf, s->K, s->N, s->cpu);
        if (sc < s->best_score) {
            s->best_score = sc;
            s->best_nf = nf;
            memcpy(s->best, s->current, (size_t)nf * sizeof(int));
        }
        return;
    }
    if (nf >= STRIDE_MAX_STAGES) return;

    /* Try every radix R that divides `remaining` and is in the registry. */
    for (int R = 2; R < VFFT_PROTO_REG_MAX_RADIX; R++) {
        if (remaining % R != 0) continue;
        if (!s->reg->n1_fwd[R]) continue;
        if (!s->reg->t1s_dit_fwd[R] && R < remaining) continue;  /* inner needs t1s */
        s->current[nf] = R;
        _vfft_proto_estimate_recurse(s, remaining / R, nf + 1);
    }
}

/* Public: V4-cost-model estimate planner. Returns a plan built from
 * the lowest-V4-scoring factorization of N. NULL if N is unfactorable
 * with available radixes. */
static inline stride_plan_t *vfft_proto_estimate_plan_v4(
    int N, size_t K, const vfft_proto_registry_t *reg)
{
    if (N <= 1) return NULL;

    stride_cpu_info_t cpu = stride_detect_cpu();
    _vfft_proto_estimate_search_t s;
    memset(&s, 0, sizeof(s));
    s.N = N; s.K = K; s.cpu = &cpu; s.reg = reg;
    s.best_nf = 0;
    s.best_score = 1e30;

    _vfft_proto_estimate_recurse(&s, N, 0);

    if (s.best_nf == 0) return NULL;  /* unfactorable */
    return vfft_proto_plan_create(N, K, s.best, /*variants=*/NULL, s.best_nf, reg);
}

/* DIT/DIF-aware verbose variant. Cost model is orientation-agnostic
 * (factorization shape determines score, not orient), so estimate picks
 * the same factors for both orientations — only the built plan differs. */
static inline stride_plan_t *vfft_proto_estimate_plan_v4_verbose_ex(
    int N, size_t K, int use_dif_forward,
    const vfft_proto_registry_t *reg,
    int *out_factors, int *out_nf, double *out_score)
{
    if (N <= 1) {
        if (out_nf) *out_nf = 0;
        return NULL;
    }
    stride_cpu_info_t cpu = stride_detect_cpu();
    _vfft_proto_estimate_search_t s;
    memset(&s, 0, sizeof(s));
    s.N = N; s.K = K; s.cpu = &cpu; s.reg = reg;
    s.best_nf = 0;
    s.best_score = 1e30;

    _vfft_proto_estimate_recurse(&s, N, 0);

    if (s.best_nf == 0) {
        if (out_nf) *out_nf = 0;
        return NULL;
    }
    if (out_factors) memcpy(out_factors, s.best, (size_t)s.best_nf * sizeof(int));
    if (out_nf)      *out_nf = s.best_nf;
    if (out_score)   *out_score = s.best_score;
    return vfft_proto_plan_create_ex(N, K, s.best, /*variants=*/NULL,
                                      s.best_nf, use_dif_forward, reg);
}

/* Back-compat: DIT-only verbose. */
static inline stride_plan_t *vfft_proto_estimate_plan_v4_verbose(
    int N, size_t K, const vfft_proto_registry_t *reg,
    int *out_factors, int *out_nf, double *out_score)
{
    return vfft_proto_estimate_plan_v4_verbose_ex(
        N, K, /*use_dif_forward=*/0, reg, out_factors, out_nf, out_score);
}

#endif /* VFFT_PROTO_CORE_ESTIMATE_PLAN_H */
