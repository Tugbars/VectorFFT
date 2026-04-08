/**
 * stride_fft2d.h -- 2D FFT with two methods: tiled and Bailey
 *
 * Both methods use the same column FFT (native, K=N2). They differ
 * only in how row FFTs are performed:
 *
 *   Tiled (default): for each tile of B rows:
 *     1. Gather B rows → scratch via SIMD transpose (B×N2 → N2×B)
 *     2. N2-point FFT with K=B on scratch
 *     3. Scatter scratch back via SIMD transpose (N2×B → B×N2)
 *
 *   Bailey: two full-matrix transposes bracket a single large-K row FFT.
 *     1. Transpose N1×N2 → N2×N1 to scratch
 *     2. N2-point FFT with K=N1 on scratch
 *     3. Transpose N2×N1 → N1×N2 back
 *
 * Default: tiled with B=8. On Intel i9-14900KF (AVX2), tiled B=8 beats
 * both Bailey and MKL at all tested sizes (32² to 1024², 1.08-1.63x
 * over MKL). Small tiles keep the working set in L1/L2 and the SIMD
 * 4×4/8×4 transpose kernels make gather/scatter nearly free.
 *
 * NOTE: this was benchmarked on a single CPU. Other architectures
 * (different L1/L2 sizes, memory subsystems) may prefer Bailey at
 * very large sizes or a different tile B. Override FFT2D_DEFAULT_TILE
 * at build time to tune.
 *
 * Data layout (split-complex):
 *   re[i * N2 + j]  for i=0..N1-1, j=0..N2-1
 *
 * Normalization: bwd(fwd(x)) = N1*N2 * x.
 */
#ifndef STRIDE_FFT2D_H
#define STRIDE_FFT2D_H

#include "executor.h"
#include "transpose.h"

/* Working-set threshold for regime selection (bytes).
 * Below this: tiled gather/scatter. Above: Bailey transpose. */
#ifndef FFT2D_TILED_WS_BYTES
#define FFT2D_TILED_WS_BYTES (256 * 1024)
#endif

/* Minimum tile height for SIMD efficiency. */
#define FFT2D_MIN_TILE 4

/* ═══════════════════════════════════════════════════════════════
 * 2D PLAN DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int N1;                    /* rows (axis-0 FFT length) */
    int N2;                    /* cols (axis-1 FFT length) */
    int use_bailey;            /* 1 = full transpose, 0 = tiled */

    stride_plan_t *plan_col;   /* N1-point FFT, K = N2 (column FFTs, native) */
    stride_plan_t *plan_row;   /* N2-point FFT, K = N1 (Bailey) or K = B (tiled) */

    size_t B;                  /* tile height (tiled mode only) */
    double *scratch_re;        /* Bailey: N1*N2. Tiled: N2*B */
    double *scratch_im;
} stride_fft2d_data_t;


/* ═══════════════════════════════════════════════════════════════
 * TILED EXECUTOR — gather/scatter via SIMD transpose
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_tiled_fwd(stride_fft2d_data_t *d,
                              double *re, double *im) {
    const int N1 = d->N1, N2 = d->N2;
    const size_t B = d->B;

    for (size_t i = 0; i < (size_t)N1; i += B) {
        size_t this_B = B;
        if (i + B > (size_t)N1) this_B = (size_t)N1 - i;

        /* Gather: transpose this_B × N2 → N2 × B scratch.
         * ld_dst = B (not this_B!) so the plan's K=B layout is correct.
         * Positions this_B..B-1 in each block are unused. */
        stride_transpose_pair(
            re + i * N2, im + i * N2,
            d->scratch_re, d->scratch_im,
            (size_t)N2, B,
            this_B, (size_t)N2);

        /* FFT on scratch: N2-point, slice_K=this_B of the K=B plan */
        _stride_execute_fwd_slice(d->plan_row,
                                  d->scratch_re, d->scratch_im,
                                  this_B, B);

        /* Scatter: transpose N2 × B scratch → this_B × N2 back.
         * ld_src = B to match the plan's layout. */
        stride_transpose_pair(
            d->scratch_re, d->scratch_im,
            re + i * N2, im + i * N2,
            B, (size_t)N2,
            (size_t)N2, this_B);
    }
}

static void _fft2d_tiled_bwd(stride_fft2d_data_t *d,
                              double *re, double *im) {
    const int N1 = d->N1, N2 = d->N2;
    const size_t B = d->B;

    for (size_t i = 0; i < (size_t)N1; i += B) {
        size_t this_B = B;
        if (i + B > (size_t)N1) this_B = (size_t)N1 - i;

        stride_transpose_pair(
            re + i * N2, im + i * N2,
            d->scratch_re, d->scratch_im,
            (size_t)N2, B,
            this_B, (size_t)N2);

        _stride_execute_bwd_slice(d->plan_row,
                                  d->scratch_re, d->scratch_im,
                                  this_B, B);

        stride_transpose_pair(
            d->scratch_re, d->scratch_im,
            re + i * N2, im + i * N2,
            B, (size_t)N2,
            (size_t)N2, this_B);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * BAILEY EXECUTOR — full-matrix transpose
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_bailey_fwd(stride_fft2d_data_t *d,
                               double *re, double *im) {
    const size_t N1 = (size_t)d->N1, N2 = (size_t)d->N2;

    stride_transpose_pair(re, im, d->scratch_re, d->scratch_im,
                          N2, N1, N1, N2);
    stride_execute_fwd(d->plan_row, d->scratch_re, d->scratch_im);
    stride_transpose_pair(d->scratch_re, d->scratch_im, re, im,
                          N1, N2, N2, N1);
}

static void _fft2d_bailey_bwd(stride_fft2d_data_t *d,
                               double *re, double *im) {
    const size_t N1 = (size_t)d->N1, N2 = (size_t)d->N2;

    stride_transpose_pair(re, im, d->scratch_re, d->scratch_im,
                          N2, N1, N1, N2);
    stride_execute_bwd(d->plan_row, d->scratch_re, d->scratch_im);
    stride_transpose_pair(d->scratch_re, d->scratch_im, re, im,
                          N1, N2, N2, N1);
}


/* ═══════════════════════════════════════════════════════════════
 * DISPATCH
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_execute_fwd(void *data, double *re, double *im) {
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)data;
    stride_execute_fwd(d->plan_col, re, im);
    if (d->use_bailey)
        _fft2d_bailey_fwd(d, re, im);
    else
        _fft2d_tiled_fwd(d, re, im);
}

static void _fft2d_execute_bwd(void *data, double *re, double *im) {
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)data;
    if (d->use_bailey)
        _fft2d_bailey_bwd(d, re, im);
    else
        _fft2d_tiled_bwd(d, re, im);
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

static stride_plan_t *_fft2d_wrap(stride_fft2d_data_t *d) {
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _fft2d_destroy(d); return NULL; }
    plan->N = d->N1 * d->N2;
    plan->K = 1;
    plan->num_stages = 0;
    plan->override_fwd     = _fft2d_execute_fwd;
    plan->override_bwd     = _fft2d_execute_bwd;
    plan->override_destroy = _fft2d_destroy;
    plan->override_data    = d;
    return plan;
}

/* Default tile height. Benchmarked: B=8 wins at large sizes (256+),
 * B=16 wins at mid sizes (64-128). B=8 is the safe default — it
 * keeps the tile (N2×8×16 bytes) well within L1 for any N2 ≤ 256
 * and within L2 for N2 ≤ 2048. Override at build time if needed. */
#ifndef FFT2D_DEFAULT_TILE
#define FFT2D_DEFAULT_TILE 8
#endif

static size_t _fft2d_choose_tile(int N2, int N1) {
    size_t B = FFT2D_DEFAULT_TILE;
    if (B > (size_t)N1) B = (size_t)N1;
    if (B < FFT2D_MIN_TILE) B = FFT2D_MIN_TILE;
    return B;
}

/** Default 2D plan — tiled with exhaustive sub-plan search.
 *  Beats MKL 1.08-1.63x on i9-14900KF (AVX2). */
static stride_plan_t *stride_plan_2d(
        int N1, int N2,
        const stride_registry_t *reg)
{
    if (N1 < 1 || N2 < 1) return NULL;

    stride_fft2d_data_t *d =
        (stride_fft2d_data_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;

    d->N1 = N1;
    d->N2 = N2;

    /* Column FFTs: always N1-point, K=N2 */
    d->plan_col = stride_exhaustive_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) d->plan_col = stride_auto_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) { free(d); return NULL; }

    /* Tiled approach: small tiles (B=8) beat Bailey at all tested sizes.
     * Keeps tile working set in L1/L2, SIMD transpose kernels make
     * gather/scatter nearly free. */
    d->use_bailey = 0;
    d->B = _fft2d_choose_tile(N2, N1);

    d->plan_row = stride_exhaustive_plan(N2, d->B, reg);
    if (!d->plan_row) d->plan_row = stride_auto_plan(N2, d->B, reg);
    if (!d->plan_row) { stride_plan_destroy(d->plan_col); free(d); return NULL; }

    size_t tile_sz = (size_t)N2 * d->B;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, tile_sz * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, tile_sz * sizeof(double));

    if (!d->scratch_re || !d->scratch_im) {
        _fft2d_destroy(d);
        return NULL;
    }

    return _fft2d_wrap(d);
}

/** Bailey 2D plan — always uses full-matrix transpose.
 *  Provided as alternative for architectures where large-K FFT + cache-
 *  oblivious transpose outperforms the tiled approach (e.g. CPUs with
 *  large L2/L3 and fast HW prefetch). Uses exhaustive sub-plan search. */
static stride_plan_t *stride_plan_2d_bailey(
        int N1, int N2,
        const stride_registry_t *reg)
{
    if (N1 < 1 || N2 < 1) return NULL;

    stride_fft2d_data_t *d =
        (stride_fft2d_data_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;

    d->N1 = N1;
    d->N2 = N2;
    d->use_bailey = 1;

    d->plan_col = stride_exhaustive_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) d->plan_col = stride_auto_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) { free(d); return NULL; }

    d->plan_row = stride_exhaustive_plan(N2, (size_t)N1, reg);
    if (!d->plan_row) d->plan_row = stride_auto_plan(N2, (size_t)N1, reg);
    if (!d->plan_row) { stride_plan_destroy(d->plan_col); free(d); return NULL; }

    d->B = (size_t)N1;
    size_t total = (size_t)N1 * N2;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    if (!d->scratch_re || !d->scratch_im) {
        _fft2d_destroy(d);
        return NULL;
    }

    return _fft2d_wrap(d);
}


#endif /* STRIDE_FFT2D_H */
