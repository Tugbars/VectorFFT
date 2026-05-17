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
#include "../prototype/generated/registry.h"
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

    const double extra_lat_l2   = 7.0;
    const double extra_lat_l3   = 35.0;
    const double extra_lat_dram = 150.0;
    const int    baseline_R     = 4;

    /* Buffer-pass cost per stage transition. */
    double bytes_per_cycle;
    if      (total_bytes <= l1) bytes_per_cycle = 45.0;
    else if (total_bytes <= l2) bytes_per_cycle = 14.0;
    else if (total_bytes <= l3) bytes_per_cycle =  7.0;
    else                        bytes_per_cycle =  9.0;
    const double buffer_pass_cost_per_stage =
        2.0 * (double)total_bytes / bytes_per_cycle;

    size_t accumulated_K = K;
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        int groups = N / R;
        size_t stride = K;
        for (int d = s + 1; d < nf; d++) stride *= factors[d];

        size_t me_s = (size_t)(N / R) * K;
        double bf_cost = _radix_butterfly_cost(R, s, me_s, isa_avx512);

        size_t ws = (size_t)R * stride * 16;
        double cf_stage;
        double extra_lat;
        if      (ws <= l1) { cf_stage = 1.0; extra_lat = 0.0; }
        else if (ws <= l2) { cf_stage = 1.4; extra_lat = extra_lat_l2; }
        else if (ws <= l3) { cf_stage = 2.3; extra_lat = extra_lat_l3; }
        else               { cf_stage = 4.0; extra_lat = extra_lat_dram; }
        double cf_cold = (cf_stage > cf_buffer) ? cf_stage : cf_buffer;

        double cf_eff = (s == 0) ? cf_cold
                                  : fraction_hot * 1.0 + (1.0 - fraction_hot) * cf_cold;

        double data_cost = (double)groups * (double)K * bf_cost * cf_eff;

        int extra_loads = R - baseline_R;
        if (extra_loads < 0) extra_loads = 0;
        double wide_penalty = (double)extra_loads * extra_lat
                            * (double)groups * (double)K;

        double dtlb_cost = 0.0;
        if (s > 0) {
            size_t bytes_spanned = (size_t)R * stride * 8;
            int pages_per_group  = (int)((bytes_spanned + 4095) / 4096);
            if (pages_per_group > dtlb_entries) {
                int excess = pages_per_group - dtlb_entries;
                dtlb_cost = (double)groups * (double)excess * (double)dtlb_miss_cycles;
            }
        }

        double tw_cost = 0.0;
        if (s > 0) {
            size_t tw_bytes = (size_t)(R - 1) * accumulated_K * 16;
            if (tw_bytes > l1) tw_cost = (double)(R - 1) * accumulated_K * 4.0;
            else               tw_cost = (double)(R - 1) * accumulated_K;
        }

        score += data_cost + tw_cost + dtlb_cost + wide_penalty
              + buffer_pass_cost_per_stage;
        accumulated_K *= R;
    }
    return score;
}

/* ─────────────────────────────────────────────────────────────────
 * V5 SCORING — V4 minus wide_penalty term.
 *
 * wide_penalty was added in V4 to model the assumption that wide
 * codelets (R≥8) take an extra-latency hit when their working set
 * spills out of L1. That assumption was true BEFORE commit c18b7c1
 * (2026-05-15) "Force YMM register allocation in AVX2 codelets" —
 * the wide codelets used to suffer compiler-induced register spills,
 * which surfaced as load-port pressure × cache-tier latency.
 *
 * Post-c18b7c1, the spill-analyzer-driven YMM-pinning produces
 * spill-FREE wide codelets. The CPE measurement (radix_cpe.h,
 * 2026-05-15 23:11) confirms it: R=16 cyc_t1s is 13.5/13.6/13.8
 * across me=256/4096/65536 (flat); R=32 is 73.6/74.4/74.2 (flat).
 * No me-curve = no spill-driven degradation.
 *
 * So wide_penalty is now an artifact penalizing a pattern that
 * doesn't exist. V5 drops it. data_cost already captures the per-R
 * compute cost via measured CPE × cf_eff cache scaling.
 *
 * Concrete consequence: cells where the true winner has R=8 outer
 * + larger tail (e.g., N=16384 K=4 patient winner [4,8,32,16]) move
 * from V4-rank ~21 to V5-rank in single digits.
 *
 * NOTE: This is calibrated for current spill-free AVX2 codelets on
 * Raptor Lake. If a future regalloc change reintroduces spills, or
 * if AVX-512 codelets show different behavior, the wide_penalty
 * term may need to come back (perhaps gated on a per-ISA, per-R
 * spill-fingerprint). For now: data, not heuristics.
 * ───────────────────────────────────────────────────────────────── */

static inline double _vfft_proto_v5_score(
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

    double bytes_per_cycle;
    if      (total_bytes <= l1) bytes_per_cycle = 45.0;
    else if (total_bytes <= l2) bytes_per_cycle = 14.0;
    else if (total_bytes <= l3) bytes_per_cycle =  7.0;
    else                        bytes_per_cycle =  9.0;
    const double buffer_pass_cost_per_stage =
        2.0 * (double)total_bytes / bytes_per_cycle;

    size_t accumulated_K = K;
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        int groups = N / R;
        size_t stride = K;
        for (int d = s + 1; d < nf; d++) stride *= factors[d];

        size_t me_s = (size_t)(N / R) * K;
        double bf_cost = _radix_butterfly_cost(R, s, me_s, isa_avx512);

        size_t ws = (size_t)R * stride * 16;
        double cf_stage;
        if      (ws <= l1) cf_stage = 1.0;
        else if (ws <= l2) cf_stage = 1.4;
        else if (ws <= l3) cf_stage = 2.3;
        else               cf_stage = 4.0;
        double cf_cold = (cf_stage > cf_buffer) ? cf_stage : cf_buffer;

        double cf_eff = (s == 0) ? cf_cold
                                  : fraction_hot * 1.0 + (1.0 - fraction_hot) * cf_cold;

        double data_cost = (double)groups * (double)K * bf_cost * cf_eff;

        double dtlb_cost = 0.0;
        if (s > 0) {
            size_t bytes_spanned = (size_t)R * stride * 8;
            int pages_per_group  = (int)((bytes_spanned + 4095) / 4096);
            if (pages_per_group > dtlb_entries) {
                int excess = pages_per_group - dtlb_entries;
                dtlb_cost = (double)groups * (double)excess * (double)dtlb_miss_cycles;
            }
        }

        double tw_cost = 0.0;
        if (s > 0) {
            size_t tw_bytes = (size_t)(R - 1) * accumulated_K * 16;
            if (tw_bytes > l1) tw_cost = (double)(R - 1) * accumulated_K * 4.0;
            else               tw_cost = (double)(R - 1) * accumulated_K;
        }

        score += data_cost + tw_cost + dtlb_cost + buffer_pass_cost_per_stage;
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

/* Variant that also reports the picked factorization (for diagnostics). */
static inline stride_plan_t *vfft_proto_estimate_plan_v4_verbose(
    int N, size_t K, const vfft_proto_registry_t *reg,
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
    return vfft_proto_plan_create(N, K, s.best, /*variants=*/NULL, s.best_nf, reg);
}

#endif /* VFFT_PROTO_CORE_ESTIMATE_PLAN_H */
