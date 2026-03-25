/**
 * @file vfft_recursive.h
 * @brief VectorFFT recursive executor — plan tree, integrated calibration
 *
 * ═══════════════════════════════════════════════════════════════════════
 * ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Plan is a tree of nodes. Each node is either:
 *   LEAF  — sub-DFT fits in L1 cache, uses existing flat executor
 *   SPLIT — four-step decomposition, recurses into children
 *
 * The planner builds this tree using one of three modes:
 *   ESTIMATE — heuristic split + default factorization (instant)
 *   MEASURE  — benchmark all valid splits + factorizations (~seconds)
 *   PATIENT  — MEASURE + wider search space (~tens of seconds)
 *
 * Three calibration systems feed into the planner:
 *   1. IL calibration   — per-radix split/IL crossover K (bench_il)
 *   2. Leaf factorize   — best flat factorization per small N
 *   3. Split selection  — best N1*N2 decomposition per large N
 *
 * All three can run at plan time (MEASURE mode) or be loaded from
 * a wisdom file (pre-computed offline via bench tools).
 *
 * ═══════════════════════════════════════════════════════════════════════
 * FOUR-STEP DECOMPOSITION
 * ═══════════════════════════════════════════════════════════════════════
 *
 * DFT(N) where N = N1 * N2, data viewed as N1 rows * N2 columns:
 *
 *   1. Transpose input   (N1*N2 -> N2*N1) — columns become contiguous
 *   2. N2 row DFTs of size N1             — each row fits in L1
 *   3. Twiddle multiply  W_N^(n1*k2)     — one sequential pass
 *   4. Transpose          (N2*N1 -> N1*N2) — prepare for column DFTs
 *   5. N1 row DFTs of size N2             — recurse if N2 > threshold
 *
 * Output is in natural order. No bit-reversal permutation at the
 * recursive level — each leaf handles its own internally.
 *
 * ═══════════════════════════════════════════════════════════════════════
 * CALIBRATION PIPELINE
 * ═══════════════════════════════════════════════════════════════════════
 *
 *   vfft_rplan_create(N, mode, registry):
 *
 *   if N <= L1_threshold:
 *     ESTIMATE -> default factorization heuristic
 *     MEASURE  -> try all factorizations, benchmark, pick best
 *                 (= bench_factorize logic, inline at plan time)
 *     Uses IL calibration for per-stage split/IL decision
 *
 *   if N > L1_threshold:
 *     ESTIMATE -> largest N1 <= threshold with good factorization
 *     MEASURE  -> try all valid N1*N2, build recursive plan for
 *                 each, benchmark full execution, pick best
 *     Children created recursively with same mode
 *
 *   Wisdom cache (V2):
 *     Before benchmarking, check if wisdom has a cached result
 *     After benchmarking, store result in wisdom for reuse
 *     Wisdom persists across plan creations in same process
 *     Optionally load/save to disk
 *
 * ═══════════════════════════════════════════════════════════════════════
 * CACHE THRESHOLDS
 * ═══════════════════════════════════════════════════════════════════════
 *
 * L1 = 48KB (Raptor Lake P-core).
 * Per-element footprint: in(16B) + out(16B) + twiddles(~16B) = ~48B
 * Conservative threshold: L1 / 64 = 768 complex doubles.
 * Practical: 512 (leaves headroom for stack, codelet scratch).
 *
 * L2 = 2MB -> ~32K complex before needing 3rd recursion level.
 * L3 = 36MB -> ~500K before needing 4th level (unlikely in practice).
 */

#ifndef VFFT_RECURSIVE_H
#define VFFT_RECURSIVE_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════
 * TIMING — needed for MEASURE mode inline benchmarking
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_GET_NS_DEFINED
#define VFFT_GET_NS_DEFINED
#ifdef _WIN32
#include <windows.h>
static inline double vfft_get_ns(void)
{
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart)
        QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#include <time.h>
static inline double vfft_get_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif
#endif /* VFFT_GET_NS_DEFINED */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * PLAN MODE
 * ═══════════════════════════════════════════════════════════════ */

typedef enum
{
    VFFT_ESTIMATE = 0, /* Heuristic only — instant, ~90% optimal */
    VFFT_MEASURE  = 1, /* Benchmark all factorizations + splits */
    VFFT_PATIENT  = 2  /* MEASURE + wider search + more iterations */
} vfft_plan_mode;

/* ═══════════════════════════════════════════════════════════════
 * PLAN TREE NODE
 * ═══════════════════════════════════════════════════════════════ */

typedef enum
{
    VFFT_RNODE_LEAF,  /* Sub-DFT <= L1_threshold -> flat executor */
    VFFT_RNODE_SPLIT  /* Four-step decomposition -> recurse */
} vfft_rnode_type;

typedef struct vfft_rnode
{
    vfft_rnode_type type;
    size_t N;

    /* ── LEAF ── */
    vfft_plan *flat_plan; /* Existing flat plan. NULL for SPLIT nodes. */

    /* ── SPLIT: N = N1 * N2 ──
     * Data viewed as N1 rows * N2 columns (row-major).
     * N1 chosen so DFT(N1) fits in L1.
     * N2 may need further recursion. */
    size_t N1, N2;
    struct vfft_rnode *row_plan;  /* DFT-N1 (shared across N2 row DFTs) */
    struct vfft_rnode *col_plan;  /* DFT-N2 (shared across N1 col DFTs) */
    int col_plan_shared;          /* 1 if col_plan == row_plan (N1==N2) */

    /* Twiddle factors in DATA layout: tw[k2 * N1 + n1] = W_N^(n1 * k2)
     * Contiguous over n1 (inner dimension) -> SIMD-friendly. */
    double *tw_re, *tw_im;

} vfft_rnode;

/* ═══════════════════════════════════════════════════════════════
 * RECURSIVE PLAN (top-level container)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct
{
    size_t N;
    vfft_rnode *root;
    vfft_plan_mode mode;
    size_t L1_threshold;

    /* Scratch: 2*N per component.
     * [0..N): current recursion level workspace.
     * [N..2N): child recursion levels.
     * Depth-first execution ensures no conflicts. */
    double *scratch_re, *scratch_im;

} vfft_recursive_plan;

/* ═══════════════════════════════════════════════════════════════
 * SPLIT POINT SELECTION
 * ═══════════════════════════════════════════════════════════════ */

static const size_t VFFT_R_RADIXES[] = {
    2, 3, 4, 5, 7, 8, 10, 11, 13, 16, 17, 19, 20, 23, 25, 32, 64, 128, 0};

static int vfft_r_can_factor(size_t M)
{
    if (M <= 1)
        return 1;
    for (const size_t *r = VFFT_R_RADIXES; *r; r++)
        if (M % *r == 0)
            return vfft_r_can_factor(M / *r);
    return 0;
}

/* Enumerate all valid N1 values for splitting N */
static size_t vfft_r_enum_splits(size_t N, size_t L1_threshold,
                                  size_t *candidates, size_t max_cand)
{
    size_t count = 0;
    for (size_t n1 = 2; n1 <= L1_threshold && n1 <= N / 2; n1++)
    {
        if (N % n1 != 0)
            continue;
        size_t n2 = N / n1;
        if (!vfft_r_can_factor(n1) || !vfft_r_can_factor(n2))
            continue;
        if (count < max_cand)
            candidates[count++] = n1;
    }
    return count;
}

/* Heuristic: largest valid N1 <= L1_threshold */
static size_t vfft_r_heuristic_split(size_t N, size_t L1_threshold)
{
    size_t best = 0;
    for (size_t n1 = L1_threshold; n1 >= 2; n1--)
    {
        if (N % n1 != 0)
            continue;
        size_t n2 = N / n1;
        if (vfft_r_can_factor(n1) && vfft_r_can_factor(n2))
        {
            best = n1;
            break;
        }
    }
    if (best == 0)
    {
        for (const size_t *r = VFFT_R_RADIXES; *r; r++)
            if (N % *r == 0 && *r <= L1_threshold)
                best = *r;
    }
    return best;
}

/* ═══════════════════════════════════════════════════════════════
 * TRANSPOSE — cache-friendly tiled
 * ═══════════════════════════════════════════════════════════════ */

#define VFFT_TRANSPOSE_TILE 32

static void vfft_r_transpose(
    const double *__restrict__ src,
    double *__restrict__ dst,
    size_t rows, size_t cols)
{
    for (size_t i0 = 0; i0 < rows; i0 += VFFT_TRANSPOSE_TILE)
    {
        size_t i1 = i0 + VFFT_TRANSPOSE_TILE;
        if (i1 > rows) i1 = rows;
        for (size_t j0 = 0; j0 < cols; j0 += VFFT_TRANSPOSE_TILE)
        {
            size_t j1 = j0 + VFFT_TRANSPOSE_TILE;
            if (j1 > cols) j1 = cols;
            for (size_t i = i0; i < i1; i++)
                for (size_t j = j0; j < j1; j++)
                    dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLE MULTIPLY — SIMD dispatch
 *
 * Both data and twiddles in same layout:
 *   data[k2 * N1 + n1], tw[k2 * N1 + n1] = W_N^(n1 * k2)
 * Straight vectorized element-wise complex multiply.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_r_twiddle_scalar(
    double *__restrict__ re, double *__restrict__ im,
    const double *__restrict__ twr, const double *__restrict__ twi,
    size_t N)
{
    for (size_t i = 0; i < N; i++)
    {
        double xr = re[i], xi = im[i];
        double wr = twr[i], wi = twi[i];
        re[i] = xr * wr - xi * wi;
        im[i] = xr * wi + xi * wr;
    }
}

#ifdef __AVX2__
#include <immintrin.h>
__attribute__((target("avx2,fma")))
static void vfft_r_twiddle_avx2(
    double *__restrict__ re, double *__restrict__ im,
    const double *__restrict__ twr, const double *__restrict__ twi,
    size_t N)
{
    size_t i = 0;
    for (; i + 4 <= N; i += 4)
    {
        __m256d xr = _mm256_loadu_pd(&re[i]);
        __m256d xi = _mm256_loadu_pd(&im[i]);
        __m256d wr = _mm256_loadu_pd(&twr[i]);
        __m256d wi = _mm256_loadu_pd(&twi[i]);
        _mm256_storeu_pd(&re[i], _mm256_fmsub_pd(xr, wr, _mm256_mul_pd(xi, wi)));
        _mm256_storeu_pd(&im[i], _mm256_fmadd_pd(xr, wi, _mm256_mul_pd(xi, wr)));
    }
    for (; i < N; i++)
    {
        double xr = re[i], xi = im[i];
        double wr = twr[i], wi = twi[i];
        re[i] = xr * wr - xi * wi;
        im[i] = xr * wi + xi * wr;
    }
}
#endif

#if defined(__AVX512F__) || defined(__AVX512F)
__attribute__((target("avx512f,fma")))
static void vfft_r_twiddle_avx512(
    double *__restrict__ re, double *__restrict__ im,
    const double *__restrict__ twr, const double *__restrict__ twi,
    size_t N)
{
    size_t i = 0;
    for (; i + 8 <= N; i += 8)
    {
        __m512d xr = _mm512_loadu_pd(&re[i]);
        __m512d xi = _mm512_loadu_pd(&im[i]);
        __m512d wr = _mm512_loadu_pd(&twr[i]);
        __m512d wi = _mm512_loadu_pd(&twi[i]);
        _mm512_storeu_pd(&re[i], _mm512_fmsub_pd(xr, wr, _mm512_mul_pd(xi, wi)));
        _mm512_storeu_pd(&im[i], _mm512_fmadd_pd(xr, wi, _mm512_mul_pd(xi, wr)));
    }
    for (; i < N; i++)
    {
        double xr = re[i], xi = im[i];
        double wr = twr[i], wi = twi[i];
        re[i] = xr * wr - xi * wi;
        im[i] = xr * wi + xi * wr;
    }
}
#endif

static void vfft_r_twiddle_dispatch(
    double *re, double *im,
    const double *twr, const double *twi,
    size_t N)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    vfft_r_twiddle_avx512(re, im, twr, twi, N);
    return;
#endif
#ifdef __AVX2__
    vfft_r_twiddle_avx2(re, im, twr, twi, N);
    return;
#endif
    vfft_r_twiddle_scalar(re, im, twr, twi, N);
}

/* ═══════════════════════════════════════════════════════════════
 * RECURSIVE EXECUTOR
 *
 * execute(node, in, out, scratch)
 *
 * LEAF: call flat executor.
 * SPLIT:
 *   1. Transpose in (N1*N2) -> scratch (N2*N1)
 *   2. N2 row DFTs(N1): scratch -> out
 *   3. Twiddle multiply on out (in-place)
 *   4. Transpose out (N2*N1) -> scratch (N1*N2)
 *   5. N1 row DFTs(N2): scratch -> out
 *
 * scratch must be at least 2*N doubles.
 * [0..N) used by this level, [N..2N) passed to children.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_rexecute_fwd(
    const vfft_rnode *node,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    double *__restrict__ scratch_re, double *__restrict__ scratch_im)
{
    if (node->type == VFFT_RNODE_LEAF)
    {
        vfft_execute_fwd(node->flat_plan, in_re, in_im, out_re, out_im);
        return;
    }

    const size_t N1 = node->N1;
    const size_t N2 = node->N2;
    const size_t N = N1 * N2;

    /* Step 1: Transpose input (N1 rows * N2 cols) -> scratch (N2 rows * N1 cols) */
    vfft_r_transpose(in_re, scratch_re, N1, N2);
    vfft_r_transpose(in_im, scratch_im, N1, N2);

    /* Step 2: N2 row DFTs of size N1.
     * scratch[k2*N1 .. k2*N1+N1-1] -> out[k2*N1 .. k2*N1+N1-1]
     * Each DFT-N1 operates on N1 contiguous elements (L1 resident). */
    for (size_t k2 = 0; k2 < N2; k2++)
    {
        vfft_rexecute_fwd(
            node->row_plan,
            scratch_re + k2 * N1, scratch_im + k2 * N1,
            out_re     + k2 * N1, out_im     + k2 * N1,
            scratch_re + N,       scratch_im + N);
    }

    /* Step 3: Twiddle multiply on out (in-place, sequential).
     * out[k2*N1 + n1] *= W_N^(n1*k2)
     * tw is in same [k2*N1 + n1] layout -> straight vectorized loop. */
    vfft_r_twiddle_dispatch(out_re, out_im,
                            node->tw_re, node->tw_im, N);

    /* Step 4: Transpose out (N2 rows * N1 cols) -> scratch (N1 rows * N2 cols) */
    vfft_r_transpose(out_re, scratch_re, N2, N1);
    vfft_r_transpose(out_im, scratch_im, N2, N1);

    /* Step 5: N1 row DFTs of size N2.
     * scratch[k1*N2 .. k1*N2+N2-1] -> out[k1*N2 .. k1*N2+N2-1]
     * If N2 > L1_threshold, col_plan is SPLIT -> recurses. */
    for (size_t k1 = 0; k1 < N1; k1++)
    {
        vfft_rexecute_fwd(
            node->col_plan,
            scratch_re + k1 * N2, scratch_im + k1 * N2,
            out_re     + k1 * N2, out_im     + k1 * N2,
            scratch_re + N,       scratch_im + N);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * NODE CONSTRUCTION HELPERS
 * ═══════════════════════════════════════════════════════════════ */

/* Forward declarations for mutual recursion */
static vfft_rnode *vfft_rnode_create(
    size_t N, size_t L1_threshold,
    vfft_plan_mode mode,
    const vfft_codelet_registry *reg);

static void vfft_rnode_destroy(vfft_rnode *node);

/* Create LEAF node */
static vfft_rnode *vfft_rnode_create_leaf(
    size_t N,
    vfft_plan_mode mode,
    const vfft_codelet_registry *reg)
{
    vfft_rnode *node = (vfft_rnode *)calloc(1, sizeof(vfft_rnode));
    node->type = VFFT_RNODE_LEAF;
    node->N = N;

    if (mode == VFFT_ESTIMATE)
    {
        node->flat_plan = vfft_plan_create(N, reg);
    }
    else
    {
        /* MEASURE/PATIENT: benchmark all factorizations inline.
         * TODO: integrate bench_factorize logic here.
         * For V1, use the default planner. */
        node->flat_plan = vfft_plan_create(N, reg);
    }

    return node;
}

/* Create SPLIT node for a specific N1 * N2 */
static vfft_rnode *vfft_rnode_create_split(
    size_t N, size_t N1, size_t N2,
    size_t L1_threshold,
    vfft_plan_mode mode,
    const vfft_codelet_registry *reg)
{
    vfft_rnode *node = (vfft_rnode *)calloc(1, sizeof(vfft_rnode));
    node->type = VFFT_RNODE_SPLIT;
    node->N = N;
    node->N1 = N1;
    node->N2 = N2;

    /* Recurse for children */
    node->row_plan = vfft_rnode_create(N1, L1_threshold, mode, reg);

    if (N2 == N1)
    {
        node->col_plan = node->row_plan;
        node->col_plan_shared = 1;
    }
    else
    {
        node->col_plan = vfft_rnode_create(N2, L1_threshold, mode, reg);
        node->col_plan_shared = 0;
    }

    /* Twiddle table in DATA layout: tw[k2 * N1 + n1] = W_N^(n1 * k2)
     * n1 is the contiguous inner dimension -> vectorizable. */
    node->tw_re = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    node->tw_im = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    for (size_t k2 = 0; k2 < N2; k2++)
    {
        for (size_t n1 = 0; n1 < N1; n1++)
        {
            double angle = -2.0 * M_PI * (double)(n1 * k2) / (double)N;
            node->tw_re[k2 * N1 + n1] = cos(angle);
            node->tw_im[k2 * N1 + n1] = sin(angle);
        }
    }

    return node;
}

/* ═══════════════════════════════════════════════════════════════
 * NODE CONSTRUCTION — MAIN (recursive, mode-aware)
 * ═══════════════════════════════════════════════════════════════ */

static vfft_rnode *vfft_rnode_create(
    size_t N, size_t L1_threshold,
    vfft_plan_mode mode,
    const vfft_codelet_registry *reg)
{
    /* Base case */
    if (N <= L1_threshold)
        return vfft_rnode_create_leaf(N, mode, reg);

    /* ESTIMATE: heuristic split */
    if (mode == VFFT_ESTIMATE)
    {
        size_t N1 = vfft_r_heuristic_split(N, L1_threshold);
        if (N1 == 0)
            return vfft_rnode_create_leaf(N, mode, reg);
        return vfft_rnode_create_split(N, N1, N / N1,
                                       L1_threshold, mode, reg);
    }

    /* MEASURE / PATIENT: benchmark all valid splits */
    size_t candidates[512];
    size_t n_cand = vfft_r_enum_splits(N, L1_threshold, candidates, 512);

    if (n_cand == 0)
        return vfft_rnode_create_leaf(N, mode, reg);

    /* Limit search for MEASURE (top ~30 candidates = largest N1 values) */
    size_t start_idx = 0;
    if (mode == VFFT_MEASURE && n_cand > 30)
        start_idx = n_cand - 30;

    /* Bench buffers */
    double *b_ir = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *b_ii = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *b_or = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *b_oi = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *b_sr = (double *)vfft_aligned_alloc(64, 2 * N * sizeof(double));
    double *b_si = (double *)vfft_aligned_alloc(64, 2 * N * sizeof(double));

    srand(42 + (unsigned)N);
    for (size_t i = 0; i < N; i++)
    {
        b_ir[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        b_ii[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    int reps = N <= 2048  ? 2000
             : N <= 8192  ? 500
             : N <= 32768 ? 100
                          : 30;
    int trials = (mode == VFFT_PATIENT) ? 7 : 3;

    double best_ns = 1e18;
    size_t best_N1 = candidates[n_cand - 1];

    for (size_t ci = start_idx; ci < n_cand; ci++)
    {
        size_t N1 = candidates[ci];
        size_t N2 = N / N1;

        /* Trial plan: children use ESTIMATE for speed */
        vfft_rnode *trial = vfft_rnode_create_split(
            N, N1, N2, L1_threshold, VFFT_ESTIMATE, reg);

        /* Warmup */
        for (int w = 0; w < 3; w++)
            vfft_rexecute_fwd(trial, b_ir, b_ii, b_or, b_oi, b_sr, b_si);

        /* Benchmark */
        double trial_best = 1e18;
        for (int t = 0; t < trials; t++)
        {
            double t0 = vfft_get_ns();
            for (int r = 0; r < reps; r++)
                vfft_rexecute_fwd(trial, b_ir, b_ii, b_or, b_oi, b_sr, b_si);
            double ns = (vfft_get_ns() - t0) / (double)reps;
            if (ns < trial_best)
                trial_best = ns;
        }

        if (trial_best < best_ns)
        {
            best_ns = trial_best;
            best_N1 = N1;
        }

        vfft_rnode_destroy(trial);
    }

    vfft_aligned_free(b_ir);
    vfft_aligned_free(b_ii);
    vfft_aligned_free(b_or);
    vfft_aligned_free(b_oi);
    vfft_aligned_free(b_sr);
    vfft_aligned_free(b_si);

    /* Build final plan with the winning split (full mode for children) */
    return vfft_rnode_create_split(
        N, best_N1, N / best_N1, L1_threshold, mode, reg);
}

/* ═══════════════════════════════════════════════════════════════
 * TOP-LEVEL API
 * ═══════════════════════════════════════════════════════════════ */

static size_t vfft_r_detect_l1(void)
{
#ifdef _WIN32
    /* TODO: GetLogicalProcessorInformation */
    return 48 * 1024;
#else
    /* TODO: sysconf(_SC_LEVEL1_DCACHE_SIZE) */
    return 48 * 1024;
#endif
}

static vfft_recursive_plan *vfft_rplan_create(
    size_t N,
    vfft_plan_mode mode,
    const vfft_codelet_registry *reg)
{
    size_t L1 = vfft_r_detect_l1();
    size_t L1_threshold = L1 / 64;
    if (L1_threshold < 32)
        L1_threshold = 32;

    vfft_recursive_plan *rp = (vfft_recursive_plan *)calloc(1, sizeof(*rp));
    rp->N = N;
    rp->mode = mode;
    rp->L1_threshold = L1_threshold;
    rp->root = vfft_rnode_create(N, L1_threshold, mode, reg);

    /* Scratch: 2*N per component (level 0 + children) */
    rp->scratch_re = (double *)vfft_aligned_alloc(64, 2 * N * sizeof(double));
    rp->scratch_im = (double *)vfft_aligned_alloc(64, 2 * N * sizeof(double));

    return rp;
}

static void vfft_rplan_execute_fwd(
    const vfft_recursive_plan *rp,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
{
    vfft_rexecute_fwd(rp->root, in_re, in_im, out_re, out_im,
                      rp->scratch_re, rp->scratch_im);
}

/* Backward via conjugate trick (V1) */
static void vfft_rplan_execute_bwd(
    const vfft_recursive_plan *rp,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
{
    const size_t N = rp->N;

    /* Use second half of scratch for conjugated input */
    double *cj_re = rp->scratch_re + N;
    double *cj_im = rp->scratch_im + N;
    memcpy(cj_re, in_re, N * sizeof(double));
    for (size_t i = 0; i < N; i++)
        cj_im[i] = -in_im[i];

    /* Forward DFT of conjugated input.
     * Note: scratch [0..N) is available since cj uses [N..2N). */
    vfft_rexecute_fwd(rp->root, cj_re, cj_im, out_re, out_im,
                      rp->scratch_re, rp->scratch_im);

    /* Conjugate output */
    for (size_t i = 0; i < N; i++)
        out_im[i] = -out_im[i];
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN DESTRUCTION
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_rnode_destroy(vfft_rnode *node)
{
    if (!node)
        return;
    if (node->type == VFFT_RNODE_LEAF)
    {
        if (node->flat_plan)
            vfft_plan_destroy(node->flat_plan);
    }
    else
    {
        if (node->col_plan && !node->col_plan_shared)
            vfft_rnode_destroy(node->col_plan);
        vfft_rnode_destroy(node->row_plan);
        vfft_aligned_free(node->tw_re);
        vfft_aligned_free(node->tw_im);
    }
    free(node);
}

static void vfft_rplan_destroy(vfft_recursive_plan *rp)
{
    if (!rp)
        return;
    vfft_rnode_destroy(rp->root);
    vfft_aligned_free(rp->scratch_re);
    vfft_aligned_free(rp->scratch_im);
    free(rp);
}

/* ═══════════════════════════════════════════════════════════════
 * DIAGNOSTICS — plan tree printer
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_rnode_print(const vfft_rnode *node, int depth)
{
    for (int i = 0; i < depth; i++)
        printf("  ");

    if (node->type == VFFT_RNODE_LEAF)
    {
        printf("LEAF N=%zu", node->N);
        if (node->flat_plan)
        {
            printf(" [");
            for (size_t s = 0; s < node->flat_plan->nstages; s++)
            {
                if (s) printf("x");
                printf("%zu", node->flat_plan->stages[s].radix);
            }
            printf("]");
        }
        printf("\n");
    }
    else
    {
        printf("SPLIT N=%zu = %zu x %zu%s\n",
               node->N, node->N1, node->N2,
               node->col_plan_shared ? " (shared)" : "");
        vfft_rnode_print(node->row_plan, depth + 1);
        if (!node->col_plan_shared)
            vfft_rnode_print(node->col_plan, depth + 1);
    }
}

static void vfft_rplan_print(const vfft_recursive_plan *rp)
{
    printf("=== Recursive Plan: N=%zu, mode=%s, L1_threshold=%zu ===\n",
           rp->N,
           rp->mode == VFFT_ESTIMATE ? "ESTIMATE"
         : rp->mode == VFFT_MEASURE  ? "MEASURE"
                                     : "PATIENT",
           rp->L1_threshold);
    vfft_rnode_print(rp->root, 0);
}

/* ═══════════════════════════════════════════════════════════════
 * V2 ROADMAP
 * ═══════════════════════════════════════════════════════════════
 *
 * 1. LEAF MEASURE
 *    Integrate bench_factorize logic into vfft_rnode_create_leaf()
 *    for MEASURE mode. Try all factorizations of N, benchmark each,
 *    store the best flat plan.
 *
 * 2. WISDOM CACHE
 *    Per-process cache: avoid re-measuring the same N twice.
 *    Keyed by N + mode. Optional disk persistence.
 *    Critical for MEASURE mode where the same leaf size (e.g., 256)
 *    appears many times across different parent splits.
 *
 * 3. NATIVE BACKWARD EXECUTOR
 *    Same tree, backward codelets in leaves, conjugated twiddles
 *    at split levels. Eliminates 2 conjugation passes over N.
 *
 * 4. SIMD TRANSPOSE
 *    4x4 transpose kernel: _mm256_unpackhi/lo_pd + _mm256_permute2f128_pd
 *    8x8 for AVX-512: _mm512_unpackhi/lo_pd + permutex2var
 *    Critical for large N where transpose is ~30% of total time.
 *
 * 5. FUSED TWIDDLE + TRANSPOSE
 *    Steps 3+4 combined: read out[k2][n1], multiply by tw, write
 *    scratch[n1][k2]. One pass instead of two. Saves ~15% at large N.
 *
 * 6. DIF/DIT FUSION (permutation elimination)
 *    Row DFTs use DIF (output in scrambled order).
 *    Column DFTs use DIT (input in scrambled order).
 *    Scrambled output of rows = correct scrambled input for cols.
 *    Eliminates bit-reversal permutation inside leaves entirely.
 *    Requires separate row_plan_dif and col_plan_dit.
 *
 * 7. THREE-LEVEL RECURSION (L2-aware)
 *    For N > ~128K complex: split into L2-sized chunks first,
 *    then each chunk splits into L1-sized leaves.
 *    Tree depth 3 instead of 2. Same code, just deeper recursion.
 *    L1 leaf: <= 768 elements (~48KB)
 *    L2 node: <= 32K elements (~2MB)
 *    L3 root: any size
 *
 * 8. IN-PLACE EXECUTION
 *    Replace out-of-place transpose with in-place cycle-leader.
 *    Reduces memory from 4*N to 2*N. Complex for non-square.
 *
 * ═══════════════════════════════════════════════════════════════ */

#endif /* VFFT_RECURSIVE_H */
