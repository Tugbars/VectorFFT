/* demo_cost_model_rank.c — smoke test: does prototype's cost model
 * (src/prototype/cost_model/factorizer.h) rank [4,4,4,16] above
 * [4,4,4,4,4] for N=1024 K=128?
 *
 * If yes, we know the cost model captures the stride context that DP's
 * recursive-bench misses, and the hybrid (cost-model rank → measure
 * top-K) approach is justified.
 *
 * NOTE: this driver uses the cost model ONLY — no plan creation, no
 * measurement, no prototype-core executor. The cost model is fully
 * standalone (its own stride_registry_t stub at factorizer.h:67).
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../prototype/cost_model/factorizer.h"

/* Walk every factorization of N (recursively), score each via V1 and V2,
 * print all of them sorted by V2 score. */

#define MAX_DECOMPOSITIONS_DUMP 2048

typedef struct {
    int factors[FACT_MAX_STAGES];
    int nf;
    double prod_v1_score;   /* production V1: tiers 1.0/3.0/10.0, no L3 awareness */
    double v1_score;        /* prototype V1: recalibrated tiers 1.0/1.4/2.3/4.0 + L3 */
    double v2_score;        /* prototype V2: V1 + DTLB + carry */
    double v2_harsh_score;  /* prototype V2 but with prod V1's harsh tiers */
    double v3_score;        /* V2 + wide-radix penalty (R-4) × tier_latency × groups × K */
    double v4_score;        /* V3 but tier comes from PAIR (current+next) residency */
} _cand_t;

/* V3 — V2 with an extra term that punishes wide-radix codelets when
 * working set exceeds L1.
 *
 * The mechanism: an R-codelet does R loads + R stores per butterfly.
 * When the working set is in L1 those loads cost ~5 cyc each (hidden
 * by OoO). When working set is in L2 each load costs ~12 cyc (extra
 * 7 cyc). In L3, ~40 cyc (extra 35). The codelet stalls on these
 * loads in proportion to (R - baseline_R), because the baseline R=4
 * codelet was the calibration point for isolated CPE.
 *
 * Per-stage penalty = (R - 4) × extra_latency(tier) × groups × K
 *
 * Effect: R=8 / R=16 at out-of-L1 tiers eat additional cost
 * proportional to extra loads × tier latency gap. R=4 at any tier:
 * zero extra penalty. R=16 at L1: also zero (its ws fits, so the
 * "extra latency" is zero). The penalty is large precisely in the
 * regime your recipe forbids: wide radixes at outer stages. */
static double v3_score(const int *factors, int nf, size_t K,
                        int N, const stride_cpu_info_t *cpu)
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

    /* tier latency gap from L1, cycles. Calibrated for Raptor Lake;
     * platform-portable if probed at calibrate time. */
    const double extra_lat_l2  = 7.0;
    const double extra_lat_l3  = 35.0;
    const double extra_lat_dram = 150.0;
    const int   baseline_R     = 4;

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

        /* NEW: wide-radix-outer penalty. */
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

        score += data_cost + tw_cost + dtlb_cost + wide_penalty;
        accumulated_K *= R;
    }
    return score;
}

/* V4 — V3 + per-stage cross-stage transition cost.
 *
 * Each stage is one full pass through the N*K complex buffer. Between
 * adjacent stages the buffer transits cache. The READ+WRITE bandwidth
 * cost per stage = buffer_bytes / effective_BW(buffer_tier), expressed
 * in cycles via clock_rate / bandwidth.
 *
 * Effect: directly penalizes plans with more stages, by exactly the
 * amount of memory-bandwidth time each extra pass costs. Empirically
 * tips [4,4,4,16] (4 stages) above [4,4,4,4,4] (5 stages) because the
 * 5-stage plan pays one extra buffer pass.
 *
 * Bandwidth numbers calibrated for Raptor Lake @ 5.7 GHz (i9-14900KF):
 *   L1  ~256 GB/s ~ 45 bytes/cycle
 *   L2  ~ 80 GB/s ~ 14 bytes/cycle
 *   L3  ~ 40 GB/s ~  7 bytes/cycle
 *   DRAM~ 50 GB/s ~  9 bytes/cycle
 * Other hosts would re-probe these via a bandwidth probe at calibrate
 * time. The TIER is determined by the whole buffer size, not per-stage
 * ws — the buffer is what transits cache between stages, and its
 * residency is fixed across all stages of a plan. */
static double v4_score(const int *factors, int nf, size_t K,
                        int N, const stride_cpu_info_t *cpu)
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

    const double extra_lat_l2  = 7.0;
    const double extra_lat_l3  = 35.0;
    const double extra_lat_dram = 150.0;
    const int   baseline_R     = 4;

    /* Buffer-pass cost: bytes per stage transit / bytes-per-cycle at
     * the cache tier that holds the WHOLE buffer. Same for every stage
     * (the buffer is whole-plan, not per-stage). */
    double bytes_per_cycle;
    if      (total_bytes <= l1) bytes_per_cycle = 45.0;   /* L1 ~256 GB/s */
    else if (total_bytes <= l2) bytes_per_cycle = 14.0;   /* L2 ~80 GB/s  */
    else if (total_bytes <= l3) bytes_per_cycle =  7.0;   /* L3 ~40 GB/s  */
    else                        bytes_per_cycle =  9.0;   /* DRAM ~50 GB/s */
    /* Per-stage buffer transit cost: read+write the whole buffer once
     * per stage. Factor of 2 accounts for read+write (each stage does
     * both). */
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

        /* Per-stage buffer-pass cost — what tips fewer-stage plans. */
        score += data_cost + tw_cost + dtlb_cost + wide_penalty
              + buffer_pass_cost_per_stage;
        accumulated_K *= R;
    }
    return score;
}

/* V2 structure (DTLB + hot-set carry + L3-bandwidth floor) but with
 * production V1's HARSH cache tiers (1.0 / 3.0 / 10.0). Tests whether
 * the bias against wide-radix-outer plans tightens when outer-stage
 * cost is penalized more aggressively. */
static double v2_harsh_score(const int *factors, int nf, size_t K,
                              int N, const stride_cpu_info_t *cpu)
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
    /* Harsher buffer-streaming floor too: 10.0 instead of 4.0. */
    const double cf_buffer = (total_bytes > l3) ? 10.0 : 0.0;

    size_t accumulated_K = K;

    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        int groups = N / R;
        size_t stride = K;
        for (int d = s + 1; d < nf; d++) stride *= factors[d];

        size_t me_s = (size_t)(N / R) * K;
        double bf_cost = _radix_butterfly_cost(R, s, me_s, isa_avx512);

        size_t ws = (size_t)R * stride * 16;
        /* PRODUCTION V1 tiers — 3 levels, no L3 distinction. */
        double cf_stage;
        if      (ws <= l1) cf_stage = 1.0;
        else if (ws <= l2) cf_stage = 3.0;
        else               cf_stage = 10.0;
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

        score += data_cost + tw_cost + dtlb_cost;
        accumulated_K *= R;
    }
    return score;
}

/* Faithful re-implementation of production's stride_score_factorization
 * (src/core/factorizer.h:354) using the same CPE table (same hardware).
 * The only difference from prototype V1 is cache_factor tiers + no L3
 * awareness + no buffer floor. */
static double production_v1_score(const int *factors, int nf, size_t K,
                                   int N, const stride_cpu_info_t *cpu)
{
    const size_t l1 = cpu->l1d_bytes;
    const size_t l2 = cpu->l2_bytes;
#if defined(__AVX512F__)
    const int isa_avx512 = 1;
#else
    const int isa_avx512 = 0;
#endif
    double score = 0.0;
    size_t accumulated_K = K;

    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        int groups = N / R;
        size_t stride = K;
        for (int d = s + 1; d < nf; d++) stride *= factors[d];

        /* Use the same CPE lookup the prototype uses (same hardware). */
        size_t me_s = (size_t)(N / R) * K;
        double bf_cost = _radix_butterfly_cost(R, s, me_s, isa_avx512);

        size_t ws = (size_t)R * stride * 16;

        /* PRODUCTION tier values: 1.0 / 3.0 / 10.0, no L3 distinction. */
        double cache_factor;
        if      (ws <= l1) cache_factor = 1.0;
        else if (ws <= l2) cache_factor = 3.0;
        else               cache_factor = 10.0;

        double data_cost = (double)groups * (double)K * bf_cost * cache_factor;

        double tw_cost = 0.0;
        if (s > 0) {
            size_t tw_bytes = (size_t)(R - 1) * accumulated_K * 16;
            if (tw_bytes > l1) tw_cost = (double)(R - 1) * accumulated_K * 4.0;
            else               tw_cost = (double)(R - 1) * accumulated_K;
        }

        score += data_cost + tw_cost;
        accumulated_K *= R;
    }
    return score;
}

typedef struct {
    _cand_t cands[MAX_DECOMPOSITIONS_DUMP];
    int n;
    int N;
    size_t K;
    const stride_cpu_info_t *cpu;
    int current[FACT_MAX_STAGES];
} _enum_t;

static void _enum_recurse(_enum_t *e, int remaining, int nf) {
    if (remaining == 1) {
        if (nf == 0) return;
        if (e->n >= MAX_DECOMPOSITIONS_DUMP) return;
        _cand_t *c = &e->cands[e->n++];
        c->nf = nf;
        memcpy(c->factors, e->current, (size_t)nf * sizeof(int));
        c->prod_v1_score = production_v1_score(c->factors, nf, e->K, e->N, e->cpu);
        c->v1_score      = stride_score_factorization(c->factors, nf, e->K, e->N, e->cpu);
        c->v2_score      = stride_score_factorization_v2(c->factors, nf, e->K, e->N, e->cpu);
        c->v2_harsh_score = v2_harsh_score(c->factors, nf, e->K, e->N, e->cpu);
        c->v3_score      = v3_score(c->factors, nf, e->K, e->N, e->cpu);
        c->v4_score      = v4_score(c->factors, nf, e->K, e->N, e->cpu);
        return;
    }
    if (nf >= FACT_MAX_STAGES) return;
    for (int R = 2; R < STRIDE_REG_MAX_RADIX; R++) {
        if (remaining % R != 0) continue;
        if (!_radix_has_codelet(R)) continue;
        e->current[nf] = R;
        _enum_recurse(e, remaining / R, nf + 1);
    }
}

static int _cmp_prod_v1(const void *a, const void *b) {
    double xa = ((const _cand_t *)a)->prod_v1_score;
    double xb = ((const _cand_t *)b)->prod_v1_score;
    return (xa < xb) ? -1 : (xa > xb) ? 1 : 0;
}
static int _cmp_v2_harsh(const void *a, const void *b) {
    double xa = ((const _cand_t *)a)->v2_harsh_score;
    double xb = ((const _cand_t *)b)->v2_harsh_score;
    return (xa < xb) ? -1 : (xa > xb) ? 1 : 0;
}
static int _cmp_v3(const void *a, const void *b) {
    double xa = ((const _cand_t *)a)->v3_score;
    double xb = ((const _cand_t *)b)->v3_score;
    return (xa < xb) ? -1 : (xa > xb) ? 1 : 0;
}
static int _cmp_v4(const void *a, const void *b) {
    double xa = ((const _cand_t *)a)->v4_score;
    double xb = ((const _cand_t *)b)->v4_score;
    return (xa < xb) ? -1 : (xa > xb) ? 1 : 0;
}
static int _cmp_v2(const void *a, const void *b) {
    double xa = ((const _cand_t *)a)->v2_score;
    double xb = ((const _cand_t *)b)->v2_score;
    return (xa < xb) ? -1 : (xa > xb) ? 1 : 0;
}

int main(int argc, char **argv) {
    int N = 1024;
    size_t K = 128;
    if (argc >= 3) { N = atoi(argv[1]); K = (size_t)atoll(argv[2]); }

    stride_cpu_info_t cpu = stride_detect_cpu();
    printf("[cost-model-rank] N=%d K=%zu\n", N, (size_t)K);
    printf("  CPU: L1=%zu L2=%zu L3=%zu  DTLB=%d entries (%d cyc/miss)\n\n",
           cpu.l1d_bytes, cpu.l2_bytes, cpu.l3_bytes,
           cpu.dtlb_entries, cpu.dtlb_miss_cycles);

    _enum_t e;
    memset(&e, 0, sizeof(e));
    e.N = N; e.K = K; e.cpu = &cpu;
    _enum_recurse(&e, N, 0);

    int show = (e.n < 20) ? e.n : 20;
    int target[] = {4, 4, 4, 16};
    int target_nf = 4;

    /* Helper: find the rank of [4,4,4,16] in the current sort order. */
#define FIND_TARGET_RANK(rankvar)                                             \
    do {                                                                      \
        rankvar = -1;                                                         \
        for (int i = 0; i < e.n; i++) {                                       \
            if (e.cands[i].nf != target_nf) continue;                         \
            int match = 1;                                                    \
            for (int s = 0; s < target_nf; s++)                               \
                if (e.cands[i].factors[s] != target[s]) { match = 0; break; } \
            if (match) { rankvar = i + 1; break; }                            \
        }                                                                     \
    } while (0)

    /* Sort by V4 (pair-aware tier). */
    qsort(e.cands, e.n, sizeof(_cand_t), _cmp_v4);
    printf("=== Ranked by V4 (V3 + PAIR-aware tier) ===\n");
    printf("  rank  factors                  V4              V3              proto_V2\n");
    printf("  ───────────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < show; i++) {
        char buf[64] = {0};
        int pos = 0;
        for (int s = 0; s < e.cands[i].nf; s++)
            pos += snprintf(buf + pos, sizeof(buf) - pos, "%s%d",
                            s ? "x" : "", e.cands[i].factors[s]);
        printf("  %3d   %-24s %12.0f   %12.0f   %12.0f\n",
               i + 1, buf, e.cands[i].v4_score,
               e.cands[i].v3_score, e.cands[i].v2_score);
    }
    int rank_v4; FIND_TARGET_RANK(rank_v4);

    qsort(e.cands, e.n, sizeof(_cand_t), _cmp_v3);
    int rank_v3; FIND_TARGET_RANK(rank_v3);
    qsort(e.cands, e.n, sizeof(_cand_t), _cmp_v2_harsh);
    int rank_v2_harsh; FIND_TARGET_RANK(rank_v2_harsh);
    qsort(e.cands, e.n, sizeof(_cand_t), _cmp_prod_v1);
    int rank_prod_v1; FIND_TARGET_RANK(rank_prod_v1);
    qsort(e.cands, e.n, sizeof(_cand_t), _cmp_v2);
    int rank_proto_v2; FIND_TARGET_RANK(rank_proto_v2);

    printf("\n  measured winner [4,4,4,16] rank by model:\n");
    printf("    production V1 (1/3/10 tiers, no L3)   : rank %d / %d\n",
           rank_prod_v1, e.n);
    printf("    prototype V2 (1/1.4/2.3/4 + DTLB)     : rank %d / %d\n",
           rank_proto_v2, e.n);
    printf("    V2-harsh (V2 + production tiers)      : rank %d / %d\n",
           rank_v2_harsh, e.n);
    printf("    V3 (V2 + wide-R-outer penalty)        : rank %d / %d\n",
           rank_v3, e.n);
    printf("    V4 (V3 + pair-aware tier)             : rank %d / %d\n",
           rank_v4, e.n);
    return 0;
}
