/**
 * stride_fft2d.h -- 2D FFT via Bailey transpose decomposition
 *
 * In-place, same layout (row-major in, row-major out).
 *
 * Algorithm (Bailey):
 *   Forward:
 *     1. Column FFTs (N1-point, K=N2) — native, no transpose needed.
 *     2. Transpose N1×N2 → N2×N1 to scratch.
 *     3. Row FFTs on transposed scratch (N2-point, K=N1) — native.
 *     4. Transpose N2×N1 → N1×N2 back to data.
 *
 *   Backward (reverse of forward):
 *     1. Transpose N1×N2 → N2×N1 to scratch.
 *     2. Row IFFTs on scratch (N2-point, K=N1) — native.
 *     3. Transpose N2×N1 → N1×N2 back to data.
 *     4. Column IFFTs (N1-point, K=N2) — native.
 *
 * Both FFT phases are native column FFTs (stride = K). The transpose
 * converts row access patterns to column access patterns, eliminating
 * the gather/scatter that was 80% of execution time in the old tiled
 * approach.
 *
 * Data layout (split-complex):
 *   re[i * N2 + j]  for i=0..N1-1, j=0..N2-1
 *   im[i * N2 + j]  same
 *
 * Normalization: bwd(fwd(x)) = N1*N2 * x.
 */
#ifndef STRIDE_FFT2D_H
#define STRIDE_FFT2D_H

#include "executor.h"
#include "transpose.h"

/* ═══════════════════════════════════════════════════════════════
 * 2D PLAN DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int N1;                    /* rows (axis-0 FFT length) */
    int N2;                    /* cols (axis-1 FFT length) */

    stride_plan_t *plan_col;   /* N1-point FFT, K = N2 (column FFTs, native) */
    stride_plan_t *plan_row;   /* N2-point FFT, K = N1 (row FFTs on transposed data) */

    double *scratch_re;        /* N1 * N2 scratch for transpose */
    double *scratch_im;
} stride_fft2d_data_t;


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_execute_fwd(void *data, double *re, double *im) {
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)data;
    const size_t N1 = (size_t)d->N1;
    const size_t N2 = (size_t)d->N2;

    /* 1. Column FFTs: N1-point, K=N2 — native layout */
    stride_execute_fwd(d->plan_col, re, im);

    /* 2. Transpose N1×N2 → N2×N1 to scratch */
    stride_transpose_pair(re, im, d->scratch_re, d->scratch_im,
                          N2, N1, N1, N2);

    /* 3. Row FFTs on transposed scratch: N2-point, K=N1 — native */
    stride_execute_fwd(d->plan_row, d->scratch_re, d->scratch_im);

    /* 4. Transpose N2×N1 → N1×N2 back to data */
    stride_transpose_pair(d->scratch_re, d->scratch_im, re, im,
                          N1, N2, N2, N1);
}

static void _fft2d_execute_bwd(void *data, double *re, double *im) {
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)data;
    const size_t N1 = (size_t)d->N1;
    const size_t N2 = (size_t)d->N2;

    /* 1. Transpose N1×N2 → N2×N1 to scratch */
    stride_transpose_pair(re, im, d->scratch_re, d->scratch_im,
                          N2, N1, N1, N2);

    /* 2. Row IFFTs on scratch: N2-point, K=N1 — native */
    stride_execute_bwd(d->plan_row, d->scratch_re, d->scratch_im);

    /* 3. Transpose N2×N1 → N1×N2 back to data */
    stride_transpose_pair(d->scratch_re, d->scratch_im, re, im,
                          N1, N2, N2, N1);

    /* 4. Column IFFTs: N1-point, K=N2 — native */
    stride_execute_bwd(d->plan_col, re, im);
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_destroy(void *data) {
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)data;
    if (!d) return;
    if (d->plan_col) stride_plan_destroy(d->plan_col);
    if (d->plan_row) stride_plan_destroy(d->plan_row);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 * ═══════════════════════════════════════════════════════════════ */

/* Internal: build 2D plan from two pre-built sub-plans */
static stride_plan_t *_fft2d_build(int N1, int N2,
                                    stride_plan_t *plan_col,
                                    stride_plan_t *plan_row) {
    stride_fft2d_data_t *d =
        (stride_fft2d_data_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;

    d->N1 = N1;
    d->N2 = N2;
    d->plan_col = plan_col;
    d->plan_row = plan_row;

    /* Scratch: full N1*N2 for transpose (split-complex) */
    size_t total = (size_t)N1 * N2;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    if (!d->scratch_re || !d->scratch_im) {
        _fft2d_destroy(d);
        return NULL;
    }

    /* Wrap in override plan */
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _fft2d_destroy(d); return NULL; }

    plan->N = N1 * N2;
    plan->K = 1;
    plan->num_stages = 0;
    plan->override_fwd     = _fft2d_execute_fwd;
    plan->override_bwd     = _fft2d_execute_bwd;
    plan->override_destroy = _fft2d_destroy;
    plan->override_data    = d;

    return plan;
}

/** Default 2D plan — uses exhaustive search for sub-plans.
 *
 * 2D sub-plans have large K (K=N2 or K=N1) where the heuristic
 * factorizer picks R=32 which regresses badly due to stride pressure.
 * Exhaustive search avoids this (finds [4,4,4,4] instead of [2,4,32]
 * for N=256, K=256 — 24% faster). The search cost is negligible since
 * 2D plans are created once and sub-plan N is typically moderate. */
static stride_plan_t *stride_plan_2d(
        int N1, int N2,
        const stride_registry_t *reg)
{
    if (N1 < 1 || N2 < 1) return NULL;

    /* Try exhaustive first, fall back to heuristic */
    stride_plan_t *pc = stride_exhaustive_plan(N1, (size_t)N2, reg);
    if (!pc) pc = stride_auto_plan(N1, (size_t)N2, reg);
    if (!pc) return NULL;

    stride_plan_t *pr = stride_exhaustive_plan(N2, (size_t)N1, reg);
    if (!pr) pr = stride_auto_plan(N2, (size_t)N1, reg);
    if (!pr) { stride_plan_destroy(pc); return NULL; }

    return _fft2d_build(N1, N2, pc, pr);
}

/** Fast 2D plan — heuristic only, no exhaustive search. */
static stride_plan_t *stride_plan_2d_heuristic(
        int N1, int N2,
        const stride_registry_t *reg)
{
    if (N1 < 1 || N2 < 1) return NULL;

    stride_plan_t *pc = stride_auto_plan(N1, (size_t)N2, reg);
    if (!pc) return NULL;
    stride_plan_t *pr = stride_auto_plan(N2, (size_t)N1, reg);
    if (!pr) { stride_plan_destroy(pc); return NULL; }

    return _fft2d_build(N1, N2, pc, pr);
}


#endif /* STRIDE_FFT2D_H */
