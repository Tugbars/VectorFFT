/**
 * dct4.h -- DCT-IV (real-to-real, FFTW REDFT11 convention)
 *
 *   Y[k] = 2 * sum_{n=0..N-1} x[n] * cos(pi*(2k+1)*(2n+1)/(4N))   for k=0..N-1
 *
 * DCT-IV is involutory up to scale 2N: DCT-IV(DCT-IV(x))/(2N) = x.
 * Used in MDCT (modified DCT) for audio codecs (MP3/AAC/Vorbis/Opus).
 *
 * Algorithm (v1.0 — correctness-first, ~2x R2C cost):
 *   Identity: cos((2k+1)(2n+1) pi/(4N))
 *           = cos(pi*(2k+1)/(4N)) * cos(pi*n*(2k+1)/(2N))
 *           - sin(pi*(2k+1)/(4N)) * sin(pi*n*(2k+1)/(2N))
 *
 *   So DCT-IV[k] = cos(pi*(2k+1)/(4N)) * A[k]
 *                - sin(pi*(2k+1)/(4N)) * B[k]
 *
 *   where  A[k] = 2 * sum_{n=0..N-1} x[n] * cos(pi*n*(2k+1)/(2N))
 *               = x[0] + DCT-III(x)[k]
 *   and    B[k] = 2 * sum_{n=1..N-1} x[n] * sin(pi*n*(2k+1)/(2N))
 *               = DST-III(x')[k]   with x' = (x[1], x[2], ..., x[N-1], 0)
 *
 * Cost: 2 R2C-based transforms (DCT-III + DST-III) + O(N*K) twiddle.
 * v1.1 candidate: Lee 1984 reduces to a single N/2-point complex FFT.
 *
 * Constraint: N must be even (inherits R2C even-N).
 *
 * Layout (split-batched, real input/output):
 *   in[n*K + k]  for n=0..N-1, k=0..K-1
 *   out[n*K + k] same shape
 */
#ifndef STRIDE_DCT4_H
#define STRIDE_DCT4_H

#include "executor.h"
#include "dct.h"
#include "dst.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int N;                  /* DCT-IV size (must be even) */
    size_t K;               /* batch count */
    double *cos_tw;         /* cos(pi*(2k+1)/(4N)) for k=0..N-1 */
    double *sin_tw;         /* sin(pi*(2k+1)/(4N)) for k=0..N-1 */
    double *A_buf;          /* N*K — holds X[0] + DCT-III(X) */
    double *B_buf;          /* N*K — holds DST-III(X') */
    stride_plan_t *dct_plan; /* owned: DCT-II/III plan */
    stride_plan_t *dst_plan; /* owned: DST-II/III plan */
} stride_dct4_data_t;


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- DCT-IV (involutory; same routine for fwd and bwd).
 * ═══════════════════════════════════════════════════════════════ */

static void _dct4_execute(void *data, double *re, double *im) {
    (void)im;
    stride_dct4_data_t *d = (stride_dct4_data_t *)data;
    const int N = d->N;
    const size_t K = d->K;

    /* 1. Compute A[k] = X[0] + DCT-III(X)[k].
     *    Copy re into A_buf, run DCT-III on A_buf, then add re[0..K) to every row. */
    memcpy(d->A_buf, re, (size_t)N * K * sizeof(double));
    d->dct_plan->override_bwd(d->dct_plan->override_data, d->A_buf, NULL);

    {
        const double *X0 = re;  /* re[0..K) holds x[0] for each batch element */
        for (int k = 0; k < N; k++) {
            double *Arow = d->A_buf + (size_t)k * K;
            for (size_t b = 0; b < K; b++) Arow[b] += X0[b];
        }
    }

    /* 2. Build X' = (x[1], x[2], ..., x[N-1], 0) into B_buf, then run DST-III. */
    for (int n = 0; n < N - 1; n++) {
        memcpy(d->B_buf + (size_t)n       * K,
               re        + (size_t)(n + 1) * K,
               K * sizeof(double));
    }
    memset(d->B_buf + (size_t)(N - 1) * K, 0, K * sizeof(double));

    d->dst_plan->override_bwd(d->dst_plan->override_data, d->B_buf, NULL);

    /* 3. Combine: re[k] = cos_tw[k] * A[k] - sin_tw[k] * B[k]. */
    for (int k = 0; k < N; k++) {
        const double cw = d->cos_tw[k];
        const double sw = d->sin_tw[k];
        const double *A = d->A_buf + (size_t)k * K;
        const double *B = d->B_buf + (size_t)k * K;
        double       *out = re     + (size_t)k * K;
        for (size_t b = 0; b < K; b++)
            out[b] = cw * A[b] - sw * B[b];
    }
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _dct4_destroy(void *data) {
    stride_dct4_data_t *d = (stride_dct4_data_t *)data;
    if (!d) return;
    free(d->cos_tw);
    free(d->sin_tw);
    STRIDE_ALIGNED_FREE(d->A_buf);
    STRIDE_ALIGNED_FREE(d->B_buf);
    if (d->dct_plan) stride_plan_destroy(d->dct_plan);
    if (d->dst_plan) stride_plan_destroy(d->dst_plan);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Caller provides a DCT-II/III plan and a DST-II/III plan; the DCT-IV
 * plan owns both from here. Constraint: N must be even.
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *stride_dct4_plan(int N, size_t K,
                                       stride_plan_t *dct_plan,
                                       stride_plan_t *dst_plan)
{
    if (N < 2 || (N & 1) || !dct_plan || !dst_plan) {
        if (dct_plan) stride_plan_destroy(dct_plan);
        if (dst_plan) stride_plan_destroy(dst_plan);
        return NULL;
    }

    stride_dct4_data_t *d = (stride_dct4_data_t *)calloc(1, sizeof(*d));
    if (!d) {
        stride_plan_destroy(dct_plan);
        stride_plan_destroy(dst_plan);
        return NULL;
    }

    d->N = N;
    d->K = K;
    d->dct_plan = dct_plan;
    d->dst_plan = dst_plan;

    d->cos_tw = (double *)malloc((size_t)N * sizeof(double));
    d->sin_tw = (double *)malloc((size_t)N * sizeof(double));
    if (!d->cos_tw || !d->sin_tw) { _dct4_destroy(d); return NULL; }
    for (int k = 0; k < N; k++) {
        double a = M_PI * (double)(2 * k + 1) / (4.0 * (double)N);
        d->cos_tw[k] = cos(a);
        d->sin_tw[k] = sin(a);
    }

    size_t buf_sz = (size_t)N * K;
    d->A_buf = (double *)STRIDE_ALIGNED_ALLOC(64, buf_sz * sizeof(double));
    d->B_buf = (double *)STRIDE_ALIGNED_ALLOC(64, buf_sz * sizeof(double));
    if (!d->A_buf || !d->B_buf) { _dct4_destroy(d); return NULL; }

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _dct4_destroy(d); return NULL; }

    plan->N = N;
    plan->K = K;
    plan->num_stages = 0;
    plan->override_fwd     = _dct4_execute;   /* DCT-IV is involutory */
    plan->override_bwd     = _dct4_execute;   /* same routine; caller divides by 2N */
    plan->override_destroy = _dct4_destroy;
    plan->override_data    = d;

    return plan;
}


/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE API
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_dct4(const stride_plan_t *plan,
                                        const double *in, double *out) {
    size_t NK = (size_t)plan->N * plan->K;
    if (in != out) memcpy(out, in, NK * sizeof(double));
    plan->override_fwd(plan->override_data, out, NULL);
}


#endif /* STRIDE_DCT4_H */
