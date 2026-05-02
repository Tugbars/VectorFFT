/**
 * stride_dct.h -- Discrete Cosine Transforms (real-to-real)
 *
 * Currently implements DCT-II only (the JPEG/MPEG/AAC workhorse).
 * Convention matches FFTW's REDFT10:
 *
 *   Y[k] = 2 * sum_{n=0..N-1} x[n] * cos( π k (2n+1) / (2N) )    for k=0..N-1
 *
 * Algorithm: 2N-point R2C trick.
 *   1. Build even extension: y[m] = x[m] for m<N, y[m] = x[2N-1-m] for m>=N
 *   2. R2C of 2N reals -> Y_2N[0..N] complex bins
 *   3. Post-process: DCT_II[k] = 2*(cos(πk/(2N))*Re(Y_2N[k]) + sin(πk/(2N))*Im(Y_2N[k]))
 *
 * Cost: dominated by the 2N-point R2C (which is ≈ 1× cost of an N-point
 * complex FFT since R2C of 2N reals is internally an N-point complex FFT).
 *
 * Layout (split-complex batched, real input):
 *   in[n * K + k]  for n=0..N-1, k=0..K-1
 *   out[n * K + k]  for n=0..N-1 (DCT bins), same layout
 *
 * Inverse (DCT-III, FFTW REDFT01) is NOT implemented in v1.0 — defer to next pass.
 *
 * MT: defers all MT to the inner R2C plan (which has its own outer-K dispatcher).
 * Even extension and post-process are O(NK) memory passes, single-threaded for now.
 */
#ifndef STRIDE_DCT_H
#define STRIDE_DCT_H

#include "executor.h"
#include "r2c.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/* ═══════════════════════════════════════════════════════════════
 * DCT-II DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int N;                 /* DCT-II size (input length, output bin count) */
    size_t K;              /* batch count */
    int n_threads;         /* T_plan snapshot (R2C inner uses this) */

    /* Post-process twiddles: cos(πk/(2N)) and sin(πk/(2N)) for k=0..N-1 */
    double *cos_tw;
    double *sin_tw;

    /* Internal scratch for 2N-point R2C (sized 2N*K each) */
    double *ext_re;
    double *ext_im;

    stride_plan_t *r2c_plan;   /* 2N-point R2C plan with batch K */
} stride_dct2_data_t;


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- FORWARD DCT-II (in-place over re; im unused)
 * ═══════════════════════════════════════════════════════════════ */

static void _dct2_execute_fwd(void *data, double *re, double *im) {
    (void)im;   /* DCT is real-only; im argument unused */
    stride_dct2_data_t *d = (stride_dct2_data_t *)data;
    const int N = d->N;
    const size_t K = d->K;
    const size_t twoN = 2 * (size_t)N;

    /* 1. Even extension into ext_re. Lower half copies x as-is;
     *    upper half mirrors (y[2N-1-m] = x[m]). */
    for (int m = 0; m < N; m++) {
        memcpy(d->ext_re + (size_t)m * K,
               re + (size_t)m * K,
               K * sizeof(double));
    }
    for (int m = 0; m < N; m++) {
        memcpy(d->ext_re + (size_t)(2 * N - 1 - m) * K,
               re + (size_t)m * K,
               K * sizeof(double));
    }
    /* ext_im is freq-output workspace; R2C overwrites it. No init needed. */

    /* 2. R2C of size 2N. After this, ext_re/ext_im hold the freq output
     *    at the first (N+1)*K positions. */
    stride_execute_fwd(d->r2c_plan, d->ext_re, d->ext_im);

    /* 3. Post-process: write DCT-II bins back into the user's re buffer.
     *    DCT_II_FFTW[k] = cos[k] * Re(Y[k]) + sin[k] * Im(Y[k])  for k=0..N-1
     *    The factor of 2 from FFTW's `Y[k] = 2 * sum(...)` definition is
     *    already absorbed by the doubling effect of the even extension on
     *    the 2N-point R2C output. Note: bin N (Nyquist of the 2N R2C)
     *    is not used by DCT-II. */
    for (int k = 0; k < N; k++) {
        const double c = d->cos_tw[k];
        const double s = d->sin_tw[k];
        const double *yr = d->ext_re + (size_t)k * K;
        const double *yi = d->ext_im + (size_t)k * K;
        double *out = re + (size_t)k * K;
        for (size_t j = 0; j < K; j++)
            out[j] = c * yr[j] + s * yi[j];
    }
    (void)twoN;
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _dct2_destroy(void *data) {
    stride_dct2_data_t *d = (stride_dct2_data_t *)data;
    if (!d) return;
    free(d->cos_tw);
    free(d->sin_tw);
    STRIDE_ALIGNED_FREE(d->ext_re);
    STRIDE_ALIGNED_FREE(d->ext_im);
    if (d->r2c_plan) stride_plan_destroy(d->r2c_plan);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Caller passes N (DCT size), K (batch), and an inner R2C plan for size 2N
 * with batch K. The plan owns the inner R2C plan from this point on
 * (destroyed via plan_destroy).
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *stride_dct2_plan(int N, size_t K, stride_plan_t *r2c_plan_2N)
{
    if (N < 1 || !r2c_plan_2N) {
        if (r2c_plan_2N) stride_plan_destroy(r2c_plan_2N);
        return NULL;
    }

    stride_dct2_data_t *d =
        (stride_dct2_data_t *)calloc(1, sizeof(*d));
    if (!d) { stride_plan_destroy(r2c_plan_2N); return NULL; }

    d->N = N;
    d->K = K;
    d->r2c_plan = r2c_plan_2N;

    int T_plan = stride_get_num_threads();
    if (T_plan < 1) T_plan = 1;
    d->n_threads = T_plan;

    /* Post-process twiddles */
    d->cos_tw = (double *)malloc((size_t)N * sizeof(double));
    d->sin_tw = (double *)malloc((size_t)N * sizeof(double));
    if (!d->cos_tw || !d->sin_tw) { _dct2_destroy(d); return NULL; }
    for (int k = 0; k < N; k++) {
        double angle = M_PI * (double)k / (2.0 * (double)N);
        d->cos_tw[k] = cos(angle);
        d->sin_tw[k] = sin(angle);
    }

    /* Internal scratch: 2N * K each (for in-place R2C of size 2N).
     * R2C's convenience wrapper uses N*K-sized buffers (where N=2N here). */
    size_t ext_sz = (size_t)(2 * N) * K;
    d->ext_re = (double *)STRIDE_ALIGNED_ALLOC(64, ext_sz * sizeof(double));
    d->ext_im = (double *)STRIDE_ALIGNED_ALLOC(64, ext_sz * sizeof(double));
    if (!d->ext_re || !d->ext_im) { _dct2_destroy(d); return NULL; }

    /* Wrap with override pointers */
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _dct2_destroy(d); return NULL; }

    plan->N = N;
    plan->K = K;
    plan->num_stages = 0;
    plan->override_fwd     = _dct2_execute_fwd;
    plan->override_bwd     = NULL;   /* DCT-III not implemented yet */
    plan->override_destroy = _dct2_destroy;
    plan->override_data    = d;

    return plan;
}


/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE API
 *
 * stride_execute_dct2: separate input and output buffers (or same for
 * in-place). Wraps the override path.
 *
 * NOTE: bwd not implemented in v1.0. DCT-III (FFTW REDFT01, the inverse
 * of DCT-II) is the natural follow-up.
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_dct2(const stride_plan_t *plan,
                                        const double *in, double *out) {
    size_t NK = (size_t)plan->N * plan->K;
    if (in != out) memcpy(out, in, NK * sizeof(double));
    plan->override_fwd(plan->override_data, out, NULL);
}


#endif /* STRIDE_DCT_H */
