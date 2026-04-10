/**
 * bench_blocked_executor.c — Blocked vs standard executor for N=16384
 *
 * Tests whether stage-blocking reduces DTLB store overhead.
 * Uses precomputed group ranges per block (no O(n^2) scanning).
 *
 * Usage:
 *   vfft_bench_blocked [K] [duration_ms]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/compat.h"
#include "../core/env.h"
#include "../core/planner.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * Execute one group of one stage (shared by both executors)
 * ═══════════════════════════════════════════════════════════════ */

static inline void _exec_group(const stride_stage_t *st, double *re, double *im,
                                int g, size_t K) {
    double *base_re = re + st->group_base[g];
    double *base_im = im + st->group_base[g];

    if (!st->needs_tw[g]) {
        st->n1_fwd(base_re, base_im, base_re, base_im,
                   st->stride, st->stride, K);
        return;
    }

    double cfr = st->cf0_re[g], cfi = st->cf0_im[g];
    if (cfr != 1.0 || cfi != 0.0) {
        for (size_t kk = 0; kk < K; kk++) {
            double tr = base_re[kk];
            base_re[kk] = tr * cfr - base_im[kk] * cfi;
            base_im[kk] = tr * cfi + base_im[kk] * cfr;
        }
    }

    if (st->t1s_fwd && st->tw_scalar_re && st->tw_scalar_re[g]) {
        st->t1s_fwd(base_re, base_im,
                     st->tw_scalar_re[g], st->tw_scalar_im[g],
                     st->stride, K);
    } else if (st->grp_tw_re && st->grp_tw_re[g]) {
        st->t1_fwd(base_re, base_im,
                   st->grp_tw_re[g], st->grp_tw_im[g],
                   st->stride, K);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * BLOCKED EXECUTOR
 *
 * Key insight: for stages after the split point, groups are ordered
 * so that group_base increases monotonically within each "block" of
 * the earlier stage's output. We exploit this by tracking group
 * index ranges per block instead of scanning all groups.
 *
 * For each block of split-stage groups [gb, gb+block_groups):
 *   - Compute the element range [lo, hi) they cover
 *   - For later stages, find the contiguous group range whose
 *     group_base falls within [lo, hi) using binary search
 * ═══════════════════════════════════════════════════════════════ */

/* Binary search: first group with group_base >= lo */
static int _find_first_group(const stride_stage_t *st, size_t lo) {
    int left = 0, right = st->num_groups;
    while (left < right) {
        int mid = (left + right) / 2;
        if (st->group_base[mid] < lo) left = mid + 1;
        else right = mid;
    }
    return left;
}

static void execute_fwd_blocked(const stride_plan_t *plan,
                                 double *re, double *im,
                                 int split_stage, int block_groups) {
    const size_t K = plan->K;

    /* Phase 1: wide stages */
    for (int s = 0; s < split_stage; s++) {
        const stride_stage_t *st = &plan->stages[s];
        for (int g = 0; g < st->num_groups; g++)
            _exec_group(st, re, im, g, K);
    }

    if (split_stage >= plan->num_stages) return;

    /* Phase 2: blocked */
    const stride_stage_t *split_st = &plan->stages[split_stage];
    int total_split_groups = split_st->num_groups;

    for (int gb = 0; gb < total_split_groups; gb += block_groups) {
        int g_end = gb + block_groups;
        if (g_end > total_split_groups) g_end = total_split_groups;

        /* Element range for this block */
        size_t block_lo = split_st->group_base[gb];
        size_t block_hi = split_st->group_base[g_end - 1] +
                          (size_t)(split_st->radix - 1) * split_st->stride + K;

        /* Execute split stage groups directly */
        for (int g = gb; g < g_end; g++)
            _exec_group(split_st, re, im, g, K);

        /* For later stages: binary search for group range within block */
        for (int s = split_stage + 1; s < plan->num_stages; s++) {
            const stride_stage_t *st = &plan->stages[s];
            int g_start = _find_first_group(st, block_lo);
            for (int g = g_start; g < st->num_groups; g++) {
                if (st->group_base[g] >= block_hi) break;
                _exec_group(st, re, im, g, K);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════ */

static void format_factors(char *buf, const int *factors, int nf) {
    buf[0] = 0;
    for (int s = 0; s < nf; s++) {
        char tmp[16];
        sprintf(tmp, "%s%d", s ? "x" : "", factors[s]);
        strcat(buf, tmp);
    }
}

int main(int argc, char **argv) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);

    int N = 16384;
    size_t K = 4;
    int duration_ms = 2000;

    if (argc > 1) K = (size_t)atoi(argv[1]);
    if (argc > 2) duration_ms = atoi(argv[2]);

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("=== Blocked Executor Benchmark: N=%d K=%zu ===\n\n", N, K);

    /* Factorizations to test. 16384 = 2^14 */
    int factorizations[][6] = {
        {8, 32, 64, 0, 0, 0},       /* 3 stages */
        {2, 8, 16, 64, 0, 0},       /* 4 stages */
        {4, 4, 16, 64, 0, 0},       /* 4 stages */
        {4, 4, 4, 4, 64, 0},        /* 5 stages */
        {4, 4, 4, 8, 16, 0},        /* 5 stages */
        {16, 16, 64, 0, 0, 0},      /* 3 stages */
        {2, 4, 4, 4, 4, 16},        /* 6 stages */
    };
    int n_facts = sizeof(factorizations) / sizeof(factorizations[0]);

    size_t total = (size_t)N * K;
    double *re     = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im     = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ref_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ref_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *chk_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *chk_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    srand(42);
    for (size_t i = 0; i < total; i++) {
        ref_re[i] = (double)rand() / RAND_MAX - 0.5;
        ref_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    printf("%-22s %3s  %10s %10s %7s  %-30s\n",
           "Factorization", "nf", "standard", "blocked", "speedup", "stages");
    printf("────────────────────── ───  ────────── ────────── ───────  "
           "──────────────────────────────\n");

    for (int fi = 0; fi < n_facts; fi++) {
        int nf = 0;
        int prod = 1;
        while (nf < 6 && factorizations[fi][nf] != 0) {
            prod *= factorizations[fi][nf];
            nf++;
        }
        if (prod != N) continue;

        stride_plan_t *plan = _stride_build_plan(N, K, factorizations[fi], nf, &reg);
        if (!plan) continue;

        char fstr[64];
        format_factors(fstr, factorizations[fi], nf);

        /* Stage info */
        char stage_info[128] = "";
        for (int s = 0; s < plan->num_stages; s++) {
            const stride_stage_t *st = &plan->stages[s];
            size_t ws = (size_t)st->radix * st->stride * K * 16;
            char tmp[24];
            sprintf(tmp, "%sR%d:%s", s ? " " : "",
                    st->radix,
                    ws <= 48*1024 ? "L1" : ws <= 2*1024*1024 ? "L2" : "L3");
            strcat(stage_info, tmp);
        }

        /* Find split */
        int split = plan->num_stages;
        for (int s = 0; s < plan->num_stages; s++) {
            size_t ws = (size_t)plan->stages[s].radix *
                        plan->stages[s].stride * K * 16;
            if (ws <= 48 * 1024) { split = s; break; }
        }

        int block_groups = 1;
        if (split < plan->num_stages) {
            const stride_stage_t *st = &plan->stages[split];
            size_t per_grp = (size_t)st->radix * st->stride * K * 16;
            block_groups = (int)(48 * 1024 / per_grp);
            if (block_groups < 1) block_groups = 1;
            if (block_groups > st->num_groups) block_groups = st->num_groups;
        }

        /* Correctness */
        memcpy(re, ref_re, total * sizeof(double));
        memcpy(im, ref_im, total * sizeof(double));
        _stride_execute_fwd_slice(plan, re, im, K, K);
        memcpy(chk_re, re, total * sizeof(double));
        memcpy(chk_im, im, total * sizeof(double));

        memcpy(re, ref_re, total * sizeof(double));
        memcpy(im, ref_im, total * sizeof(double));
        execute_fwd_blocked(plan, re, im, split, block_groups);

        double max_err = 0;
        for (size_t i = 0; i < total; i++) {
            double d = fabs(re[i] - chk_re[i]);
            if (d > max_err) max_err = d;
            d = fabs(im[i] - chk_im[i]);
            if (d > max_err) max_err = d;
        }

        if (max_err > 1e-8) {
            printf("%-22s %3d  FAIL err=%.2e\n", fstr, nf, max_err);
            stride_plan_destroy(plan);
            continue;
        }

        /* Benchmark standard */
        for (int w = 0; w < 500; w++)
            _stride_execute_fwd_slice(plan, re, im, K, K);

        int reps = 5000;
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            _stride_execute_fwd_slice(plan, re, im, K, K);
        double calib = (now_ns() - t0) / reps;
        reps = (int)((double)duration_ms * 1e6 / calib);
        if (reps < 1000) reps = 1000;

        double best_std = 1e18;
        for (int trial = 0; trial < 5; trial++) {
            t0 = now_ns();
            for (int r = 0; r < reps; r++)
                _stride_execute_fwd_slice(plan, re, im, K, K);
            double ns = (now_ns() - t0) / reps;
            if (ns < best_std) best_std = ns;
        }

        /* Benchmark blocked */
        for (int w = 0; w < 500; w++)
            execute_fwd_blocked(plan, re, im, split, block_groups);

        double best_blk = 1e18;
        for (int trial = 0; trial < 5; trial++) {
            t0 = now_ns();
            for (int r = 0; r < reps; r++)
                execute_fwd_blocked(plan, re, im, split, block_groups);
            double ns = (now_ns() - t0) / reps;
            if (ns < best_blk) best_blk = ns;
        }

        printf("%-22s %3d  %9.0f %9.0f %6.2fx  %s\n",
               fstr, nf, best_std, best_blk, best_std / best_blk, stage_info);

        stride_plan_destroy(plan);
    }

    /* ── K sweep with best factorization (4x4x16x64) ── */
    printf("\n── K sweep: 4x4x16x64 (best blocked absolute time) ──\n\n");
    printf("%-6s %10s %10s %7s\n", "K", "standard", "blocked", "speedup");
    printf("────── ────────── ────────── ───────\n");

    int sweep_factors[] = {4, 4, 16, 64};
    int sweep_nf = 4;
    size_t sweep_Ks[] = {4, 8, 16, 32, 64, 128, 256};
    int n_sweep = sizeof(sweep_Ks) / sizeof(sweep_Ks[0]);

    for (int ki = 0; ki < n_sweep; ki++) {
        size_t Kv = sweep_Ks[ki];
        size_t tot = (size_t)N * Kv;

        double *sre = (double *)STRIDE_ALIGNED_ALLOC(64, tot * sizeof(double));
        double *sim = (double *)STRIDE_ALIGNED_ALLOC(64, tot * sizeof(double));
        for (size_t i = 0; i < tot; i++) {
            sre[i] = (double)rand() / RAND_MAX - 0.5;
            sim[i] = (double)rand() / RAND_MAX - 0.5;
        }

        stride_plan_t *p = _stride_build_plan(N, Kv, sweep_factors, sweep_nf, &reg);
        if (!p) { STRIDE_ALIGNED_FREE(sre); STRIDE_ALIGNED_FREE(sim); continue; }

        /* Find split */
        int sp = p->num_stages;
        for (int s = 0; s < p->num_stages; s++) {
            size_t ws = (size_t)p->stages[s].radix * p->stages[s].stride * Kv * 16;
            if (ws <= 48 * 1024) { sp = s; break; }
        }
        int bg = 1;
        if (sp < p->num_stages) {
            size_t per_grp = (size_t)p->stages[sp].radix *
                             p->stages[sp].stride * Kv * 16;
            bg = (int)(48 * 1024 / per_grp);
            if (bg < 1) bg = 1;
            if (bg > p->stages[sp].num_groups) bg = p->stages[sp].num_groups;
        }

        /* Bench standard */
        for (int w = 0; w < 500; w++)
            _stride_execute_fwd_slice(p, sre, sim, Kv, Kv);
        int reps = 5000;
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            _stride_execute_fwd_slice(p, sre, sim, Kv, Kv);
        double calib = (now_ns() - t0) / reps;
        reps = (int)((double)duration_ms * 1e6 / calib);
        if (reps < 500) reps = 500;

        double best_s = 1e18;
        for (int trial = 0; trial < 5; trial++) {
            t0 = now_ns();
            for (int r = 0; r < reps; r++)
                _stride_execute_fwd_slice(p, sre, sim, Kv, Kv);
            double ns = (now_ns() - t0) / reps;
            if (ns < best_s) best_s = ns;
        }

        /* Bench blocked */
        for (int w = 0; w < 500; w++)
            execute_fwd_blocked(p, sre, sim, sp, bg);
        double best_b = 1e18;
        for (int trial = 0; trial < 5; trial++) {
            t0 = now_ns();
            for (int r = 0; r < reps; r++)
                execute_fwd_blocked(p, sre, sim, sp, bg);
            double ns = (now_ns() - t0) / reps;
            if (ns < best_b) best_b = ns;
        }

        printf("%-6zu %9.0f %9.0f %6.2fx  (split@%d, blk=%d)\n",
               Kv, best_s, best_b, best_s / best_b, sp, bg);

        stride_plan_destroy(p);
        STRIDE_ALIGNED_FREE(sre); STRIDE_ALIGNED_FREE(sim);
    }

    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    STRIDE_ALIGNED_FREE(ref_re); STRIDE_ALIGNED_FREE(ref_im);
    STRIDE_ALIGNED_FREE(chk_re); STRIDE_ALIGNED_FREE(chk_im);
    return 0;
}
