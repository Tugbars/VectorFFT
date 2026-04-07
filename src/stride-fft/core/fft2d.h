/**
 * stride_fft2d.h -- 2D FFT via row-column decomposition
 *
 * In-place, same layout (row-major in, row-major out).
 *
 * Algorithm:
 *   Forward:
 *     1. Axis 0 (columns, length N1): native 1D FFT with K=N2.
 *        Data layout re[i*N2 + j] = re[n*K + k] — free.
 *     2. Axis 1 (rows, length N2): tile-walk B rows at a time.
 *        Gather B rows to N2×B scratch, 1D FFT with K=B, scatter back.
 *
 *   Backward: same but with bwd executor.
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

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* ═══════════════════════════════════════════════════════════════
 * 2D PLAN DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int N1;                 /* rows (axis-0 FFT length) */
    int N2;                 /* cols (axis-1 FFT length) */

    stride_plan_t *plan_ax0;   /* N1-point FFT, K = N2 (column FFT) */
    stride_plan_t *plan_ax1;   /* N2-point FFT, K = B  (row FFT, tiled) */

    size_t B;               /* tile height for axis-1 (rows per tile) */
    double *scratch_re;     /* N2 * B scratch for axis-1 tile */
    double *scratch_im;
} stride_fft2d_data_t;


/* ═══════════════════════════════════════════════════════════════
 * AXIS-1 TILED EXECUTOR
 *
 * For each tile of B rows:
 *   1. Gather: read B rows at stride N2 → dense scratch at stride B
 *   2. FFT: full 1D FFT on scratch (K=B)
 *   3. Scatter: write scratch back to original positions at stride N2
 * ═══════════════════════════════════════════════════════════════ */

/* Gather B rows from re[row*N2 + col] to scratch[col*B + row_in_tile] */
static void _fft2d_gather(const double * __restrict__ src,
                           double * __restrict__ dst,
                           int N2, size_t B, size_t N2_stride)
{
    for (size_t b = 0; b < B; b++) {
        const double *row = src + b * N2_stride;
        size_t j = 0;
#if defined(__AVX512F__)
        for (; j + 8 <= (size_t)N2; j += 8) {
            /* Load 8 contiguous from one row, store strided into scratch */
            __m512d v = _mm512_loadu_pd(row + j);
            /* Scatter to dst[j*B + b], dst[(j+1)*B + b], ... — not contiguous.
             * For B >= 4, write 8 scalars. For large B, consider transposed layout. */
            for (int q = 0; q < 8; q++)
                dst[(j + q) * B + b] = row[j + q];
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        for (; j + 4 <= (size_t)N2; j += 4) {
            /* Same issue: dst positions are B apart, not contiguous */
            dst[(j + 0) * B + b] = row[j + 0];
            dst[(j + 1) * B + b] = row[j + 1];
            dst[(j + 2) * B + b] = row[j + 2];
            dst[(j + 3) * B + b] = row[j + 3];
        }
#endif
        for (; j < (size_t)N2; j++)
            dst[j * B + b] = row[j];
    }
}

/* Scatter: scratch[col*B + row_in_tile] → re[row*N2 + col] */
static void _fft2d_scatter(const double * __restrict__ src,
                            double * __restrict__ dst,
                            int N2, size_t B, size_t N2_stride)
{
    for (size_t b = 0; b < B; b++) {
        double *row = dst + b * N2_stride;
        size_t j = 0;
#if defined(__AVX2__) || defined(__AVX512F__)
        for (; j + 4 <= (size_t)N2; j += 4) {
            row[j + 0] = src[(j + 0) * B + b];
            row[j + 1] = src[(j + 1) * B + b];
            row[j + 2] = src[(j + 2) * B + b];
            row[j + 3] = src[(j + 3) * B + b];
        }
#endif
        for (; j < (size_t)N2; j++)
            row[j] = src[j * B + b];
    }
}

/* Axis-1 forward: tile-walk B rows at a time.
 * Gather B rows → dense scratch, FFT with K=B, scatter back.
 * TODO: fuse first/last stage with gather/scatter via n1_fwd(is=N2, os=B)
 *       to eliminate 2 of the 4 copy passes per tile. */
static void _fft2d_axis1_fwd(
        const stride_plan_t *plan, double *re, double *im,
        double *sr, double *si,
        int N1, int N2, size_t B)
{
    for (size_t b0 = 0; b0 < (size_t)N1; b0 += B) {
        size_t this_B = B;
        if (b0 + B > (size_t)N1) this_B = (size_t)N1 - b0;

        double *tile_re = re + b0 * N2;
        double *tile_im = im + b0 * N2;

        _fft2d_gather(tile_re, sr, N2, this_B, (size_t)N2);
        _fft2d_gather(tile_im, si, N2, this_B, (size_t)N2);
        stride_execute_fwd(plan, sr, si);
        _fft2d_scatter(sr, tile_re, N2, this_B, (size_t)N2);
        _fft2d_scatter(si, tile_im, N2, this_B, (size_t)N2);
    }
}

static void _fft2d_axis1_bwd(
        const stride_plan_t *plan, double *re, double *im,
        double *sr, double *si,
        int N1, int N2, size_t B)
{
    for (size_t b0 = 0; b0 < (size_t)N1; b0 += B) {
        size_t this_B = B;
        if (b0 + B > (size_t)N1) this_B = (size_t)N1 - b0;

        double *tile_re = re + b0 * N2;
        double *tile_im = im + b0 * N2;

        /* Gather → backward FFT → scatter */
        _fft2d_gather(tile_re, sr, N2, this_B, (size_t)N2);
        _fft2d_gather(tile_im, si, N2, this_B, (size_t)N2);
        stride_execute_bwd(plan, sr, si);
        _fft2d_scatter(sr, tile_re, N2, this_B, (size_t)N2);
        _fft2d_scatter(si, tile_im, N2, this_B, (size_t)N2);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_execute_fwd(void *data, double *re, double *im) {
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)data;

    /* Axis 0: column FFT, length N1, K=N2. Native — just call execute. */
    stride_execute_fwd(d->plan_ax0, re, im);

    /* Axis 1: row FFT, length N2, tiled with B rows at a time. */
    _fft2d_axis1_fwd(d->plan_ax1, re, im,
                            d->scratch_re, d->scratch_im,
                            d->N1, d->N2, d->B);
}

static void _fft2d_execute_bwd(void *data, double *re, double *im) {
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)data;

    /* Axis 1 backward (rows) first — reverse order of forward */
    _fft2d_axis1_bwd(d->plan_ax1, re, im,
                            d->scratch_re, d->scratch_im,
                            d->N1, d->N2, d->B);

    /* Axis 0 backward (columns) */
    stride_execute_bwd(d->plan_ax0, re, im);
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_destroy(void *data) {
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)data;
    if (!d) return;
    if (d->plan_ax0) stride_plan_destroy(d->plan_ax0);
    if (d->plan_ax1) stride_plan_destroy(d->plan_ax1);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 * ═══════════════════════════════════════════════════════════════ */

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

    /* Axis 0: N1-point FFT with K = N2 (column FFT, native layout) */
    d->plan_ax0 = stride_auto_plan(N1, (size_t)N2, reg);
    if (!d->plan_ax0) { free(d); return NULL; }

    /* Axis 1: N2-point FFT with K = B (row FFT, tiled).
     * B = number of rows per tile, chosen to fit scratch in L2. */
    d->B = _bluestein_block_size(N2, (size_t)N1);
    d->plan_ax1 = stride_auto_plan(N2, d->B, reg);
    if (!d->plan_ax1) { stride_plan_destroy(d->plan_ax0); free(d); return NULL; }

    /* Scratch: N2 * B doubles for axis-1 tile */
    size_t scratch_sz = (size_t)N2 * d->B;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_sz * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_sz * sizeof(double));
    if (!d->scratch_re || !d->scratch_im) {
        _fft2d_destroy(d);
        return NULL;
    }

    /* Wrap in override plan */
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _fft2d_destroy(d); return NULL; }

    plan->N = N1 * N2;     /* total size for normalization */
    plan->K = 1;            /* 2D plan has no batching (for now) */
    plan->num_stages = 0;
    plan->override_fwd     = _fft2d_execute_fwd;
    plan->override_bwd     = _fft2d_execute_bwd;
    plan->override_destroy = _fft2d_destroy;
    plan->override_data    = d;

    return plan;
}


#endif /* STRIDE_FFT2D_H */
