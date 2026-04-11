/**
 * planner_blocked.h — Joint calibration: factorization × executor × split
 *
 * Extends the standard planner with blocked executor awareness.
 * For each candidate factorization, benchmarks BOTH the standard sweep
 * executor and the blocked executor at every valid split point, picking
 * the globally optimal (factorization, executor, split) triple.
 *
 * The winner is stored in an extended wisdom entry that tells the
 * executor at runtime which path to take — zero overhead for sizes
 * where blocking doesn't help (wisdom says "don't block").
 *
 * Usage:
 *   stride_blocked_wisdom_t bwis;
 *   stride_blocked_wisdom_init(&bwis);
 *   stride_blocked_wisdom_load(&bwis, "vfft_blocked_wisdom.txt");
 *
 *   // Calibrate missing entries (joint search)
 *   stride_blocked_calibrate(&bwis, N, K, &reg, force, verbose, threshold, path);
 *
 *   // Build plan with optimal executor selection
 *   stride_plan_t *plan = stride_blocked_wise_plan(N, K, &reg, &bwis);
 *   // plan->use_blocked, plan->split_stage, plan->block_groups are set
 */
#ifndef STRIDE_PLANNER_BLOCKED_H
#define STRIDE_PLANNER_BLOCKED_H

#include "executor_blocked.h"
#include "exhaustive.h"
#include "dp_planner.h"

/* ═══════════════════════════════════════════════════════════════
 * EXTENDED WISDOM — stores executor selection alongside factorization
 * ═══════════════════════════════════════════════════════════════ */

#define BLOCKED_WISDOM_MAX 256
#define BLOCKED_WISDOM_VERSION 3

typedef struct {
    int N;
    size_t K;
    int factors[FACT_MAX_STAGES];
    int nfactors;
    double best_ns;

    /* Executor selection (new fields) */
    int use_blocked;     /* 0 = standard sweep, 1 = blocked */
    int split_stage;     /* first blocked stage (ignored if use_blocked=0) */
    int block_groups;    /* groups per block at split stage */
} stride_blocked_entry_t;

typedef struct {
    stride_blocked_entry_t entries[BLOCKED_WISDOM_MAX];
    int count;
} stride_blocked_wisdom_t;

static void stride_blocked_wisdom_init(stride_blocked_wisdom_t *w) {
    w->count = 0;
}

static const stride_blocked_entry_t *stride_blocked_wisdom_lookup(
        const stride_blocked_wisdom_t *w, int N, size_t K) {
    for (int i = 0; i < w->count; i++)
        if (w->entries[i].N == N && w->entries[i].K == K)
            return &w->entries[i];
    return NULL;
}

static void stride_blocked_wisdom_add(stride_blocked_wisdom_t *w,
                                       const stride_blocked_entry_t *e) {
    /* Update existing */
    for (int i = 0; i < w->count; i++) {
        if (w->entries[i].N == e->N && w->entries[i].K == e->K) {
            if (e->best_ns < w->entries[i].best_ns)
                w->entries[i] = *e;
            return;
        }
    }
    /* Insert new */
    if (w->count < BLOCKED_WISDOM_MAX)
        w->entries[w->count++] = *e;
}

/* ═══════════════════════════════════════════════════════════════
 * FILE I/O
 *
 * Format (version 3):
 *   @version 3
 *   # N K nf f0 f1 ... best_ns use_blocked split_stage block_groups
 *   16384 4 4 4 4 16 64 120339.00 1 2 3
 * ═══════════════════════════════════════════════════════════════ */

static int stride_blocked_wisdom_save(const stride_blocked_wisdom_t *w,
                                       const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "@version %d\n", BLOCKED_WISDOM_VERSION);
    fprintf(f, "# VectorFFT blocked wisdom — %d entries\n", w->count);
    fprintf(f, "# N K nf factors... best_ns use_blocked split_stage block_groups\n");
    for (int i = 0; i < w->count; i++) {
        const stride_blocked_entry_t *e = &w->entries[i];
        fprintf(f, "%d %zu %d", e->N, e->K, e->nfactors);
        for (int j = 0; j < e->nfactors; j++)
            fprintf(f, " %d", e->factors[j]);
        fprintf(f, " %.2f %d %d %d\n",
                e->best_ns, e->use_blocked, e->split_stage, e->block_groups);
    }
    fclose(f);
    return 0;
}

static int stride_blocked_wisdom_load(stride_blocked_wisdom_t *w,
                                       const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    int version_ok = 0;

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (line[0] == '@') {
            int ver = 0;
            if (sscanf(line, "@version %d", &ver) == 1 && ver == BLOCKED_WISDOM_VERSION)
                version_ok = 1;
            continue;
        }
        if (!version_ok) { fclose(f); return -1; }

        stride_blocked_entry_t e;
        memset(&e, 0, sizeof(e));
        int pos = 0, n;
        if (sscanf(line, "%d %zu %d%n", &e.N, &e.K, &e.nfactors, &n) < 3)
            continue;
        pos = n;
        if (e.nfactors < 1 || e.nfactors > FACT_MAX_STAGES) continue;
        int ok = 1;
        for (int j = 0; j < e.nfactors; j++) {
            if (sscanf(line + pos, "%d%n", &e.factors[j], &n) < 1) { ok = 0; break; }
            pos += n;
        }
        if (!ok) continue;
        if (sscanf(line + pos, "%lf %d %d %d",
                   &e.best_ns, &e.use_blocked, &e.split_stage, &e.block_groups) < 4)
            continue;
        stride_blocked_wisdom_add(w, &e);
    }
    fclose(f);
    return version_ok ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════
 * BENCHMARK HELPERS
 * ═══════════════════════════════════════════════════════════════ */

/* Bench a plan with the standard executor */
static double _blocked_bench_standard(stride_plan_t *plan, int N, size_t K,
                                       double *re, double *im,
                                       const double *orig_re,
                                       const double *orig_im) {
    size_t total = (size_t)N * K;
    int reps = (int)(5e5 / (total + 1));
    if (reps < 10) reps = 10;
    if (reps > 50000) reps = 50000;

    /* Warmup */
    for (int w = 0; w < 5; w++) {
        memcpy(re, orig_re, total * sizeof(double));
        memcpy(im, orig_im, total * sizeof(double));
        _stride_execute_fwd_slice(plan, re, im, K, K);
    }

    double best = 1e18;
    for (int t = 0; t < 3; t++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            _stride_execute_fwd_slice(plan, re, im, K, K);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* Bench a plan with the blocked executor at a given split point */
static double _blocked_bench_blocked(stride_plan_t *plan, int N, size_t K,
                                      double *re, double *im,
                                      const double *orig_re,
                                      const double *orig_im,
                                      int split_stage, int block_groups) {
    size_t total = (size_t)N * K;
    int reps = (int)(5e5 / (total + 1));
    if (reps < 10) reps = 10;
    if (reps > 50000) reps = 50000;

    for (int w = 0; w < 5; w++) {
        memcpy(re, orig_re, total * sizeof(double));
        memcpy(im, orig_im, total * sizeof(double));
        _stride_execute_fwd_blocked(plan, re, im, split_stage, block_groups);
    }

    double best = 1e18;
    for (int t = 0; t < 3; t++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            _stride_execute_fwd_blocked(plan, re, im, split_stage, block_groups);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* ═══════════════════════════════════════════════════════════════
 * JOINT CALIBRATION
 *
 * For each candidate factorization:
 *   1. Build plan
 *   2. Bench with standard executor
 *   3. For each valid split point, bench with blocked executor
 *   4. Record the globally best (factorization, executor, split)
 *
 * Valid split points: stages where working set first fits in L1.
 * We also try split+1 in case blocking fewer stages is better.
 * ═══════════════════════════════════════════════════════════════ */

static double stride_blocked_calibrate(
        stride_blocked_wisdom_t *bwis, int N, size_t K,
        const stride_registry_t *reg,
        stride_dp_context_t *dp_ctx,
        int force, int verbose, int exhaustive_threshold,
        const char *save_path)
{
    /* Skip if already calibrated */
    if (!force) {
        const stride_blocked_entry_t *e = stride_blocked_wisdom_lookup(bwis, N, K);
        if (e) return e->best_ns;
    }

    /* Step 1: enumerate all factorizations (reuse exhaustive infrastructure) */
    factorization_list_t *flist = (factorization_list_t *)malloc(sizeof(*flist));
    stride_enumerate_factorizations(N, reg, flist);

    if (verbose)
        printf("  N=%d K=%zu: %d decompositions, joint search...\n", N, K, flist->count);

    /* Allocate benchmark buffers */
    size_t total = (size_t)N * K;
    double *re      = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im      = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *orig_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *orig_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    stride_blocked_entry_t global_best;
    memset(&global_best, 0, sizeof(global_best));
    global_best.N = N;
    global_best.K = K;
    global_best.best_ns = 1e18;

    int total_candidates = 0;

    for (int fi = 0; fi < flist->count; fi++) {
        const stride_factorization_t *f = &flist->results[fi];

        /* Try all orderings of this factorization */
        permutation_list_t *plist = (permutation_list_t *)malloc(sizeof(*plist));
        stride_gen_permutations(f->factors, f->nfactors, plist);

        for (int pi = 0; pi < plist->count; pi++) {
            const int *factors = plist->perms[pi];
            int nf = f->nfactors;

            /* Verify product */
            int prod = 1;
            for (int s = 0; s < nf; s++) prod *= factors[s];
            if (prod != N) continue;

            stride_plan_t *plan = _stride_build_plan(N, K, factors, nf, reg);
            if (!plan) continue;
            total_candidates++;

            /* Bench standard executor */
            double std_ns = _blocked_bench_standard(plan, N, K, re, im,
                                                     orig_re, orig_im);
            if (std_ns < global_best.best_ns) {
                global_best.best_ns = std_ns;
                global_best.nfactors = nf;
                memcpy(global_best.factors, factors, nf * sizeof(int));
                global_best.use_blocked = 0;
                global_best.split_stage = nf;
                global_best.block_groups = 0;
            }

            /* Try blocked executor at each valid split point */
            for (int sp = 0; sp < plan->num_stages; sp++) {
                size_t ws = (size_t)plan->stages[sp].radix *
                            plan->stages[sp].stride * K * 2 * sizeof(double);
                if (ws > STRIDE_BLOCKED_L1_BYTES) continue;

                /* Compute block groups for this split */
                size_t per_grp = (size_t)plan->stages[sp].radix *
                                 plan->stages[sp].stride * K * 2 * sizeof(double);
                int bg = (int)(STRIDE_BLOCKED_L1_BYTES / per_grp);
                if (bg < 1) bg = 1;
                if (bg > plan->stages[sp].num_groups)
                    bg = plan->stages[sp].num_groups;

                double blk_ns = _blocked_bench_blocked(plan, N, K, re, im,
                                                        orig_re, orig_im,
                                                        sp, bg);
                if (blk_ns < global_best.best_ns) {
                    global_best.best_ns = blk_ns;
                    global_best.nfactors = nf;
                    memcpy(global_best.factors, factors, nf * sizeof(int));
                    global_best.use_blocked = 1;
                    global_best.split_stage = sp;
                    global_best.block_groups = bg;
                }
            }

            stride_plan_destroy(plan);
        }
        free(plist);
    }

    free(flist);
    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    STRIDE_ALIGNED_FREE(orig_re);
    STRIDE_ALIGNED_FREE(orig_im);

    if (global_best.best_ns >= 1e17) return 1e18;

    if (verbose) {
        printf("  Best: ");
        for (int s = 0; s < global_best.nfactors; s++)
            printf("%s%d", s ? "x" : "", global_best.factors[s]);
        printf(" = %.0f ns, executor=%s",
               global_best.best_ns,
               global_best.use_blocked ? "blocked" : "standard");
        if (global_best.use_blocked)
            printf(" (split@%d, blk=%d)", global_best.split_stage,
                   global_best.block_groups);
        printf(" (%d candidates)\n", total_candidates);
    }

    stride_blocked_wisdom_add(bwis, &global_best);

    if (save_path)
        stride_blocked_wisdom_save(bwis, save_path);

    return global_best.best_ns;
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN BUILDER — uses blocked wisdom to select executor
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *stride_blocked_wise_plan(
        int N, size_t K,
        const stride_registry_t *reg,
        const stride_blocked_wisdom_t *bwis)
{
    const stride_blocked_entry_t *e = stride_blocked_wisdom_lookup(bwis, N, K);
    if (!e) return NULL;

    stride_plan_t *plan = _stride_build_plan(N, K, e->factors, e->nfactors, reg);
    if (!plan) return NULL;

    /* Store executor selection in the plan */
    plan->use_blocked = e->use_blocked;
    plan->split_stage = e->split_stage;
    plan->block_groups = e->block_groups;

    return plan;
}


#endif /* STRIDE_PLANNER_BLOCKED_H */
