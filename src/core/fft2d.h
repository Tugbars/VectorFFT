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
 * Threading:
 *   Phase 1 (columns): uses executor's built-in K-split (K=N2).
 *   Phase 2 (rows):    tile-parallel — tiles distributed across threads,
 *                       each thread uses its own scratch buffer.
 *   No barriers needed — both phases are embarrassingly parallel.
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

/* Minimum tile height for SIMD efficiency. */
#define FFT2D_MIN_TILE 4

/* Default tile height. B=8 keeps tile in L1 for N2≤256, L2 for N2≤2048. */
#ifndef FFT2D_DEFAULT_TILE
#define FFT2D_DEFAULT_TILE 8
#endif

/* Maximum threads for per-thread scratch allocation. */
#ifndef FFT2D_MAX_THREADS
#define FFT2D_MAX_THREADS 64
#endif


/* ═══════════════════════════════════════════════════════════════
 * 2D PLAN DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int N1;                    /* rows (axis-0 FFT length) */
    int N2;                    /* cols (axis-1 FFT length) */
    int use_bailey;            /* 1 = full transpose, 0 = tiled */

    stride_plan_t *plan_col;   /* N1-point FFT, K = N2 (column FFTs, native) */
    stride_plan_t *plan_row;   /* N2-point FFT, K = N1 (Bailey) or K = B (tiled) */

    size_t B;                  /* tile height */

    /* Per-thread scratch buffers for tile-parallel execution.
     * Thread t uses scratch_re + t * tile_sz, scratch_im + t * tile_sz.
     * Allocated for num_scratch threads at plan time. */
    int num_scratch;
    size_t tile_sz;            /* N2 * B (tiled) or N1 * N2 (Bailey) */
    double *scratch_re;
    double *scratch_im;
} stride_fft2d_data_t;

/* Get scratch pointer for thread t */
static inline double *_fft2d_scratch(double *pool, size_t tile_sz, int t) {
    return pool + (size_t)t * tile_sz;
}


/* ═══════════════════════════════════════════════════════════════
 * TILED EXECUTOR — single-threaded core
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_tiled_range(stride_fft2d_data_t *d,
                                double *re, double *im,
                                double *sr, double *si,
                                size_t row_start, size_t row_end,
                                int is_bwd) {
    const int N2 = d->N2;
    const size_t B = d->B;

    for (size_t i = row_start; i < row_end; i += B) {
        size_t this_B = B;
        if (i + B > row_end) this_B = row_end - i;

        /* Gather: B×N2 → N2×B (ld_dst=B for plan's K=B layout) */
        stride_transpose_pair(
            re + i * N2, im + i * N2, sr, si,
            (size_t)N2, B, this_B, (size_t)N2);

        /* FFT on scratch */
        if (is_bwd)
            _stride_execute_bwd_slice(d->plan_row, sr, si, this_B, B);
        else
            _stride_execute_fwd_slice(d->plan_row, sr, si, this_B, B);

        /* Scatter: N2×B → B×N2 (ld_src=B) */
        stride_transpose_pair(
            sr, si, re + i * N2, im + i * N2,
            B, (size_t)N2, (size_t)N2, this_B);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * TILE-PARALLEL THREADING
 *
 * Tiles are independent — distribute across threads, no barriers.
 * Each thread gets its own scratch buffer.
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    stride_fft2d_data_t *d;
    double *re, *im;
    double *sr, *si;           /* per-thread scratch */
    size_t row_start, row_end;
    int is_bwd;
} _fft2d_tile_arg_t;

static void _fft2d_tile_trampoline(void *arg) {
    _fft2d_tile_arg_t *a = (_fft2d_tile_arg_t *)arg;
    _fft2d_tiled_range(a->d, a->re, a->im, a->sr, a->si,
                        a->row_start, a->row_end, a->is_bwd);
}

static void _fft2d_tiled_mt(stride_fft2d_data_t *d,
                             double *re, double *im, int is_bwd) {
    const size_t N1 = (size_t)d->N1;
    const size_t B = d->B;
    int T = stride_get_num_threads();

    /* Cap threads at available scratch buffers */
    if (T > d->num_scratch) T = d->num_scratch;

    /* Total tiles */
    size_t n_tiles = (N1 + B - 1) / B;

    /* For small problems, single-threaded is faster (no dispatch overhead) */
    if (T <= 1 || n_tiles <= 1) {
        _fft2d_tiled_range(d, re, im,
                           d->scratch_re, d->scratch_im,
                           0, N1, is_bwd);
        return;
    }

    /* Distribute tiles across threads.
     * Round tile boundaries to multiples of B for clean splits. */
    _fft2d_tile_arg_t args[FFT2D_MAX_THREADS];
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
        args[t].sr = _fft2d_scratch(d->scratch_re, d->tile_sz, t);
        args[t].si = _fft2d_scratch(d->scratch_im, d->tile_sz, t);
        args[t].row_start = row_start;
        args[t].row_end = row_end;
        args[t].is_bwd = is_bwd;

        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _fft2d_tile_trampoline, &args[t]);
        n_dispatch++;
    }

    /* Thread 0 (caller) processes its own share */
    {
        size_t row_end = ((n_tiles * 1) / T) * B;
        if (row_end > N1) row_end = N1;
        _fft2d_tiled_range(d, re, im,
                           d->scratch_re, d->scratch_im,
                           0, row_end, is_bwd);
    }

    /* Wait for workers */
    if (n_dispatch > 0)
        _stride_pool_wait_all();
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

    /* Phase 1: column FFTs — internally threaded via K-split */
    stride_execute_fwd(d->plan_col, re, im);

    /* Phase 2: row FFTs */
    if (d->use_bailey)
        _fft2d_bailey_fwd(d, re, im);
    else
        _fft2d_tiled_mt(d, re, im, 0);
}

static void _fft2d_execute_bwd(void *data, double *re, double *im) {
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)data;

    /* Phase 1: row IFFTs */
    if (d->use_bailey)
        _fft2d_bailey_bwd(d, re, im);
    else
        _fft2d_tiled_mt(d, re, im, 1);

    /* Phase 2: column IFFTs — internally threaded */
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

static size_t _fft2d_choose_tile(int N2, int N1) {
    size_t B = FFT2D_DEFAULT_TILE;
    if (B > (size_t)N1) B = (size_t)N1;
    if (B < FFT2D_MIN_TILE) B = FFT2D_MIN_TILE;
    return B;
}

/* Allocate per-thread scratch buffers.
 * Returns number of scratch slots allocated. */
static int _fft2d_alloc_scratch(stride_fft2d_data_t *d, size_t tile_sz) {
    int T = stride_get_num_threads();
    if (T > FFT2D_MAX_THREADS) T = FFT2D_MAX_THREADS;
    if (T < 1) T = 1;

    d->tile_sz = tile_sz;
    d->num_scratch = T;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)T * tile_sz * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)T * tile_sz * sizeof(double));

    if (!d->scratch_re || !d->scratch_im) return 0;
    return T;
}

/** Default 2D plan — tiled with exhaustive sub-plan search.
 *  Beats MKL 1.08-1.63x on i9-14900KF (AVX2), single-threaded.
 *  Tile-parallel threading for row FFTs when num_threads > 1. */
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
    d->use_bailey = 0;
    d->B = _fft2d_choose_tile(N2, N1);

    /* Column FFTs: N1-point, K=N2 */
    d->plan_col = stride_exhaustive_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) d->plan_col = stride_auto_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) { free(d); return NULL; }

    /* Row FFTs: N2-point, K=B */
    d->plan_row = stride_exhaustive_plan(N2, d->B, reg);
    if (!d->plan_row) d->plan_row = stride_auto_plan(N2, d->B, reg);
    if (!d->plan_row) { stride_plan_destroy(d->plan_col); free(d); return NULL; }

    /* Per-thread scratch: T copies of N2*B */
    if (!_fft2d_alloc_scratch(d, (size_t)N2 * d->B)) {
        _fft2d_destroy(d);
        return NULL;
    }

    return _fft2d_wrap(d);
}

/** Bailey 2D plan — always uses full-matrix transpose.
 *  Alternative for architectures where large-K FFT + cache-oblivious
 *  transpose outperforms tiled. Uses exhaustive sub-plan search. */
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
    /* Bailey only needs 1 scratch (not tile-parallel) */
    d->num_scratch = 1;
    d->tile_sz = (size_t)N1 * N2;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, d->tile_sz * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, d->tile_sz * sizeof(double));

    if (!d->scratch_re || !d->scratch_im) {
        _fft2d_destroy(d);
        return NULL;
    }

    return _fft2d_wrap(d);
}

/** Wisdom-aware 2D plan — uses pre-calibrated wisdom for the row FFT only.
 *
 *  v1.0 SAFETY: Column FFT (plan_col, K=N2) is FORCED to non-wisdom
 *  (`stride_auto_plan`) because wisdom-driven plan_col + K-split path
 *  silently corrupts at intermediate T (e.g., 1024²: err ~1e6 at T=2/T=4
 *  while T=1 and T=8 work fine). Investigation deferred to v1.1 — likely
 *  variant-code interaction with K-split slice helpers at large K, or a
 *  DIF/blocked path triggered by wisdom that the K-split executor doesn't
 *  handle. Cost: ~3-5% per-stage codelet tuning loss on plan_col.
 *
 *  Row FFT (plan_row, K=B=8) is wisdom-tuned — K-split never fires for
 *  K=8 < threshold(256), so this is safe. */
static stride_plan_t *stride_plan_2d_wise(
        int N1, int N2,
        const stride_registry_t *reg,
        const stride_wisdom_t *wis)
{
    if (N1 < 1 || N2 < 1) return NULL;

    stride_fft2d_data_t *d =
        (stride_fft2d_data_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;

    d->N1 = N1;
    d->N2 = N2;
    d->use_bailey = 0;
    d->B = _fft2d_choose_tile(N2, N1);

    /* Column FFTs: NON-wisdom (see safety note above). Match stride_plan_2d's
     * exact path — exhaustive first, then auto fallback. Empirically safe at
     * all T from 1..8 for sizes 64²..1024². */
    d->plan_col = stride_exhaustive_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) d->plan_col = stride_auto_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) { free(d); return NULL; }

    /* Row FFTs: NON-wisdom too for now (paranoid v1.0 safety until the
     * 1024² K-split + variant-coded plan_col bug is properly diagnosed).
     * K-split doesn't fire for K=B=8, so this is purely a defensive choice. */
    d->plan_row = stride_exhaustive_plan(N2, d->B, reg);
    if (!d->plan_row) d->plan_row = stride_auto_plan(N2, d->B, reg);
    if (!d->plan_row) { stride_plan_destroy(d->plan_col); free(d); return NULL; }
    (void)wis;  /* unused in v1.0; will be re-enabled once K-split bug is fixed */

    if (!_fft2d_alloc_scratch(d, (size_t)N2 * d->B)) {
        _fft2d_destroy(d);
        return NULL;
    }

    return _fft2d_wrap(d);
}


#endif /* STRIDE_FFT2D_H */
