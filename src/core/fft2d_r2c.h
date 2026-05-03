/**
 * fft2d_r2c.h -- 2D Real-to-Complex / Complex-to-Real FFT
 *
 * Two phases (matching FFTW convention: reduce along inner axis):
 *
 *   Forward (R2C 2D):  N1*N2 reals -> N1 * (N2/2+1) complex
 *     Phase 1: tiled R2C row pass — for each tile of B rows, transpose
 *              real input B*N2 -> N2*B, run 1D R2C (N=N2, K=B), transpose
 *              split-complex output (N2/2+1)*B -> B*(N2/2+1).
 *     Phase 2: 1D C2C col pass — N1-point complex FFT batched K=(N2/2+1).
 *              No transpose needed; layout already matches.
 *
 *   Backward (C2R 2D):  N1 * (N2/2+1) complex -> N1*N2 reals
 *     Phase 1: 1D C2C col IFFT — N1-point batched K=(N2/2+1).
 *     Phase 2: tiled C2R row pass, processed in REVERSE tile order
 *              (scatter writes longer rows than gather reads, so reverse
 *              avoids overwriting future tiles' input).
 *
 * Layout (split-batched):
 *   real input:   real[i * N2 + j]            for i=0..N1-1, j=0..N2-1
 *   complex out:  re[i*(N2/2+1) + f], im same  for i=0..N1-1, f=0..N2/2
 *
 * In-place semantics:
 *   forward: caller passes (re, im); re sized to hold max(N1*N2,
 *            N1*(N2/2+1)) = N1*N2 reals on input, becomes N1*(N2/2+1)
 *            Re bins on output. im sized N1*(N2/2+1).
 *   backward: same buffers, layout transitions back.
 *
 * Constraint: N2 must be even (inherits 1D R2C even-N constraint).
 *
 * Threading:
 *   Phase 1: tile-parallel — same model as 2D C2C tiled row pass.
 *   Phase 2: K-split via the C2C executor — K=(N2/2+1) usually large enough.
 *
 * Reuses transpose.h (8x4 line-filling kernel) and r2c.h.
 */
#ifndef STRIDE_FFT2D_R2C_H
#define STRIDE_FFT2D_R2C_H

#include "executor.h"
#include "transpose.h"
#include "r2c.h"
#include <stdio.h>  /* DIAG */

#ifndef FFT2D_R2C_DEFAULT_TILE
#define FFT2D_R2C_DEFAULT_TILE 8
#endif

#ifndef FFT2D_R2C_MIN_TILE
#define FFT2D_R2C_MIN_TILE 4
#endif

#ifndef FFT2D_R2C_MAX_THREADS
#define FFT2D_R2C_MAX_THREADS 64
#endif


/* ═══════════════════════════════════════════════════════════════
 * 2D R2C PLAN DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int N1;                       /* rows */
    int N2;                       /* cols (must be even) */
    size_t B;                     /* row tile height */

    int num_scratch;              /* per-thread scratch slots */
    size_t tile_real_sz;          /* N2 * B */
    size_t tile_complex_sz;       /* (N2/2+1) * B */
    double *scratch_re;           /* num_scratch * tile_real_sz doubles */
    double *scratch_im;           /* num_scratch * tile_complex_sz doubles */

    stride_plan_t *plan_r2c;      /* N=N2, K=B, R2C inner */
    stride_plan_t *plan_col;      /* N=N1, K=N2/2+1, C2C col */
} stride_fft2d_r2c_data_t;


/* ═══════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════ */

static inline double *_fft2d_r2c_scratch_re(stride_fft2d_r2c_data_t *d, int t) {
    return d->scratch_re + (size_t)t * d->tile_real_sz;
}
static inline double *_fft2d_r2c_scratch_im(stride_fft2d_r2c_data_t *d, int t) {
    return d->scratch_im + (size_t)t * d->tile_complex_sz;
}

/* Run inner R2C single-threaded on the caller's thread, regardless of the
 * global num_threads setting. Tile-parallel outer wants no nested dispatch. */
static inline void _fft2d_r2c_inner_fwd(stride_plan_t *plan, double *re, double *im) {
    stride_r2c_data_t *d = (stride_r2c_data_t *)plan->override_data;
    _r2c_worker_arg_t a = { d, re, im, 0, d->K, 0 };
    _r2c_worker_fwd(&a);
}
static inline void _fft2d_r2c_inner_bwd(stride_plan_t *plan, double *re, double *im) {
    stride_r2c_data_t *d = (stride_r2c_data_t *)plan->override_data;
    _r2c_worker_arg_t a = { d, re, im, 0, d->K, 0 };
    _r2c_worker_bwd(&a);
}


/* ═══════════════════════════════════════════════════════════════
 * TILED ROW PASS — forward (R2C)
 *
 * For tile of B rows starting at i_tile:
 *   gather real B x N2 -> scratch_re (N2 x B)
 *   inner R2C  (in-place on scratch_re/scratch_im, N=N2 K=B)
 *   scatter scratch (N2/2+1 x B split-complex) -> out (B x (N2/2+1))
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_r2c_tiled_fwd_range(stride_fft2d_r2c_data_t *d,
                                        double *re, double *im,
                                        double *sr, double *si,
                                        size_t row_start, size_t row_end)
{
    const int N2 = d->N2;
    const int halfN_plus1 = N2 / 2 + 1;
    const size_t B = d->B;

    for (size_t i = row_start; i < row_end; i += B) {
        size_t this_B = B;
        if (i + B > row_end) this_B = row_end - i;

        /* Gather: real B x N2 -> scratch_re N2 x B (single-plane transpose). */
        stride_transpose(re + i * (size_t)N2, (size_t)N2,
                         sr, B, this_B, (size_t)N2);

        /* Inner R2C in-place on scratch. After: sr[f*B + k_local] holds Re bins,
         * si[f*B + k_local] holds Im bins for f=0..N2/2. */
        _fft2d_r2c_inner_fwd(d->plan_r2c, sr, si);

        /* Scatter split-complex: (N2/2+1) x B -> B x (N2/2+1).
         * Output layout: re[i*(N2/2+1) + f]. */
        stride_transpose_pair(sr, si,
                              re + i * (size_t)halfN_plus1,
                              im + i * (size_t)halfN_plus1,
                              B, (size_t)halfN_plus1,
                              (size_t)halfN_plus1, this_B);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * TILED ROW PASS — backward (C2R)
 *
 * Process tiles in REVERSE order: scatter writes wider rows (N2 reals)
 * than gather reads (N2/2+1 complex). Forward order would overwrite
 * future tiles' input. Reverse order keeps later tiles' read region
 * intact until they are processed.
 *
 * For tile of B rows starting at i_tile:
 *   gather complex B x (N2/2+1) -> scratch (N2/2+1 x B split)
 *   inner C2R (in-place on scratch_re/scratch_im, N=N2 K=B)
 *   scatter scratch_re (N2 x B real) -> re (B x N2 real)
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_r2c_tiled_bwd_range(stride_fft2d_r2c_data_t *d,
                                        double *re, double *im,
                                        double *sr, double *si,
                                        size_t row_start, size_t row_end)
{
    const int N2 = d->N2;
    const int halfN_plus1 = N2 / 2 + 1;
    const size_t B = d->B;

    /* Iterate tiles in reverse to avoid in-place aliasing. Each tile
     * starts at i = row_start + k*B; we walk k from largest down to 0. */
    if (row_end <= row_start) return;
    size_t span = row_end - row_start;
    size_t n_tiles = (span + B - 1) / B;

    for (size_t k = n_tiles; k > 0; k--) {
        size_t i = row_start + (k - 1) * B;
        size_t this_B = B;
        if (i + B > row_end) this_B = row_end - i;

        /* Gather split-complex: B x (N2/2+1) -> (N2/2+1) x B. */
        stride_transpose_pair(re + i * (size_t)halfN_plus1,
                              im + i * (size_t)halfN_plus1,
                              sr, si,
                              (size_t)halfN_plus1, B,
                              this_B, (size_t)halfN_plus1);

        /* Inner C2R in-place on scratch. After: sr[j*B + k_local] holds
         * the j-th real sample for tile-row k_local. */
        _fft2d_r2c_inner_bwd(d->plan_r2c, sr, si);

        /* Scatter real: N2 x B -> B x N2. */
        stride_transpose(sr, B, re + i * (size_t)N2, (size_t)N2,
                         (size_t)N2, this_B);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * TILE-PARALLEL THREADING (forward)
 *
 * Tiles are independent. Distribute across threads, each owns a scratch slot.
 * For backward, reverse tile order is required for in-place safety, so
 * THREADING IS DISABLED in backward (single-threaded path). v1.1 candidate.
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    stride_fft2d_r2c_data_t *d;
    double *re, *im;
    double *sr, *si;
    size_t row_start, row_end;
} _fft2d_r2c_tile_arg_t;

static void _fft2d_r2c_tile_fwd_trampoline(void *arg) {
    _fft2d_r2c_tile_arg_t *a = (_fft2d_r2c_tile_arg_t *)arg;
    _fft2d_r2c_tiled_fwd_range(a->d, a->re, a->im, a->sr, a->si,
                                a->row_start, a->row_end);
}

static void _fft2d_r2c_tiled_fwd_mt(stride_fft2d_r2c_data_t *d,
                                     double *re, double *im) {
    const size_t N1 = (size_t)d->N1;
    const size_t B = d->B;
    int T = stride_get_num_threads();
    if (T > d->num_scratch) T = d->num_scratch;

    size_t n_tiles = (N1 + B - 1) / B;
    if (T <= 1 || n_tiles <= 1) {
        _fft2d_r2c_tiled_fwd_range(d, re, im,
                                    d->scratch_re, d->scratch_im,
                                    0, N1);
        return;
    }

    _fft2d_r2c_tile_arg_t args[FFT2D_R2C_MAX_THREADS];
    int n_dispatch = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t tiles_start = (n_tiles * t) / T;
        size_t tiles_end   = (n_tiles * (t + 1)) / T;
        size_t row_start   = tiles_start * B;
        size_t row_end     = tiles_end * B;
        if (row_end > N1) row_end = N1;
        if (row_start >= N1) break;

        args[t].d = d;
        args[t].re = re;
        args[t].im = im;
        args[t].sr = _fft2d_r2c_scratch_re(d, t);
        args[t].si = _fft2d_r2c_scratch_im(d, t);
        args[t].row_start = row_start;
        args[t].row_end = row_end;
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _fft2d_r2c_tile_fwd_trampoline, &args[t]);
        n_dispatch++;
    }
    {
        size_t row_end = ((n_tiles * 1) / T) * B;
        if (row_end > N1) row_end = N1;
        _fft2d_r2c_tiled_fwd_range(d, re, im,
                                    d->scratch_re, d->scratch_im,
                                    0, row_end);
    }
    if (n_dispatch > 0) _stride_pool_wait_all();
}


/* ═══════════════════════════════════════════════════════════════
 * DISPATCH — forward and backward
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_r2c_execute_fwd(void *data, double *re, double *im) {
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)data;

    /* Phase 1: tiled R2C row pass. After: (re, im) holds N1*(N2/2+1) complex. */
    _fft2d_r2c_tiled_fwd_mt(d, re, im);

    /* DIAG */
    if (d->N1 == 16 && d->N2 == 16) {
        int hp1 = d->N2 / 2 + 1;
        fprintf(stderr, "[fwd post-row] re:");
        for (int i = 0; i < d->N1 * hp1; i++) fprintf(stderr, " %.2f", re[i]);
        fprintf(stderr, "\n[fwd post-row] im:");
        for (int i = 0; i < d->N1 * hp1; i++) fprintf(stderr, " %.2f", im[i]);
        fprintf(stderr, "\n"); fflush(stderr);
    }

    /* Phase 2: C2C col FFT (batched, K=N2/2+1). */
    stride_execute_fwd(d->plan_col, re, im);
}

static void _fft2d_r2c_execute_bwd(void *data, double *re, double *im) {
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)data;

    /* Phase 1: C2C col IFFT (batched, K=N2/2+1). */
    stride_execute_bwd(d->plan_col, re, im);

    /* Phase 2: tiled C2R row pass — REVERSE tile order for in-place safety.
     * Single-threaded for v1.0 (reverse iteration + tile parallelism is
     * solvable but not in scope here). */
    _fft2d_r2c_tiled_bwd_range(d, re, im,
                                d->scratch_re, d->scratch_im,
                                0, (size_t)d->N1);
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_r2c_destroy(void *data) {
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)data;
    if (!d) return;
    if (d->plan_r2c) stride_plan_destroy(d->plan_r2c);
    if (d->plan_col) stride_plan_destroy(d->plan_col);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Caller provides inner plans:
 *   plan_r2c: N=N2, K=B, R2C (will be owned by 2D plan)
 *   plan_col: N=N1, K=N2/2+1, C2C (will be owned by 2D plan)
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *stride_plan_2d_r2c_from(int N1, int N2, size_t B,
                                               stride_plan_t *plan_r2c,
                                               stride_plan_t *plan_col)
{
    /* Caller must ensure B == plan_r2c->K (they index the same scratch).
     * No clamping here — clamping would silently break the layout invariant. */
    if (N1 < 2 || N2 < 2 || (N2 & 1) || !plan_r2c || !plan_col ||
        B < 2 || B > (size_t)N1) {
        if (plan_r2c) stride_plan_destroy(plan_r2c);
        if (plan_col) stride_plan_destroy(plan_col);
        return NULL;
    }

    stride_fft2d_r2c_data_t *d =
        (stride_fft2d_r2c_data_t *)calloc(1, sizeof(*d));
    if (!d) {
        stride_plan_destroy(plan_r2c);
        stride_plan_destroy(plan_col);
        return NULL;
    }
    d->N1 = N1;
    d->N2 = N2;
    d->B = B;
    d->plan_r2c = plan_r2c;
    d->plan_col = plan_col;

    d->tile_real_sz = (size_t)N2 * B;
    d->tile_complex_sz = (size_t)(N2 / 2 + 1) * B;

    int T = stride_get_num_threads();
    if (T > FFT2D_R2C_MAX_THREADS) T = FFT2D_R2C_MAX_THREADS;
    if (T < 1) T = 1;
    d->num_scratch = T;

    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64,
        (size_t)T * d->tile_real_sz * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64,
        (size_t)T * d->tile_complex_sz * sizeof(double));
    if (!d->scratch_re || !d->scratch_im) {
        _fft2d_r2c_destroy(d);
        return NULL;
    }

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _fft2d_r2c_destroy(d); return NULL; }

    plan->N = N1 * N2;
    plan->K = 1;
    plan->num_stages = 0;
    plan->override_fwd     = _fft2d_r2c_execute_fwd;
    plan->override_bwd     = _fft2d_r2c_execute_bwd;
    plan->override_destroy = _fft2d_r2c_destroy;
    plan->override_data    = d;

    return plan;
}


/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE API
 *
 * stride_execute_2d_r2c(plan, real_in, out_re, out_im):
 *   real_in: N1*N2 reals.
 *   out_re, out_im: each N1*(N2/2+1) doubles.
 *
 * stride_execute_2d_c2r(plan, in_re, in_im, real_out):
 *   in_re, in_im: each N1*(N2/2+1) doubles.
 *   real_out: N1*N2 reals.
 *
 * Both wrappers allocate temp scratch internally because the in-place
 * override requires the re buffer to be sized for the LARGER of input
 * (N1*N2 reals for forward) or output (N1*N2 reals for backward).
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_2d_r2c(const stride_plan_t *plan,
                                          const double *real_in,
                                          double *out_re, double *out_im)
{
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)plan->override_data;
    size_t real_sz = (size_t)d->N1 * (size_t)d->N2;
    size_t cplx_sz = (size_t)d->N1 * (size_t)(d->N2 / 2 + 1);
    /* In-place override needs re sized real_sz (= max(real_sz, cplx_sz)).
     * Use a scratch re buffer, then copy the lower cplx_sz doubles to out_re. */
    double *re_tmp = (double *)STRIDE_ALIGNED_ALLOC(64, real_sz * sizeof(double));
    if (!re_tmp) return;
    memcpy(re_tmp, real_in, real_sz * sizeof(double));
    plan->override_fwd(plan->override_data, re_tmp, out_im);
    memcpy(out_re, re_tmp, cplx_sz * sizeof(double));
    STRIDE_ALIGNED_FREE(re_tmp);
}

static inline void stride_execute_2d_c2r(const stride_plan_t *plan,
                                          const double *in_re, const double *in_im,
                                          double *real_out)
{
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)plan->override_data;
    size_t real_sz = (size_t)d->N1 * (size_t)d->N2;
    size_t cplx_sz = (size_t)d->N1 * (size_t)(d->N2 / 2 + 1);
    /* real_out is sized real_sz; also serves as the re scratch (real_sz >= cplx_sz). */
    memcpy(real_out, in_re, cplx_sz * sizeof(double));
    /* Need a temp im buffer (override_bwd writes to im too). */
    double *im_tmp = (double *)STRIDE_ALIGNED_ALLOC(64, cplx_sz * sizeof(double));
    if (!im_tmp) return;
    memcpy(im_tmp, in_im, cplx_sz * sizeof(double));
    plan->override_bwd(plan->override_data, real_out, im_tmp);
    STRIDE_ALIGNED_FREE(im_tmp);
}


#endif /* STRIDE_FFT2D_R2C_H */
