/**
 * dht.h -- Discrete Hartley Transform (real-to-real)
 *
 * Convention matches FFTW's FFTW_DHT:
 *   H[k] = sum_{n=0..N-1} x[n] * (cos(2*pi*k*n/N) + sin(2*pi*k*n/N))
 *
 * Self-inverse up to 1/N: DHT(DHT(x)) = N * x. Apply forward twice and
 * divide by N to recover the original (FFTW unnormalized convention).
 *
 * Algorithm: built directly on N-point R2C — no twiddles, no pre-permute.
 * Given X = R2C(x) with X[N-k] = conj(X[k]):
 *   H[0]    = Re(X[0])                  (Im(X[0]) = 0 for real input)
 *   H[k]    = Re(X[k]) - Im(X[k])       for k = 1 .. N/2 - 1
 *   H[N-k]  = Re(X[k]) + Im(X[k])       for k = 1 .. N/2 - 1
 *   H[N/2]  = Re(X[N/2])                (Im = 0 at Nyquist when N even)
 *
 * Cost: one N-point R2C plus one O(N*K) butterfly pass. ~2x faster than
 * a full N-point complex FFT.
 *
 * Constraint: N must be even (our R2C requires even input size).
 *
 * Layout (split-batched, real input/output):
 *   in[n*K + k]  for n=0..N-1, k=0..K-1
 *   out[n*K + k] same shape
 */
#ifndef STRIDE_DHT_H
#define STRIDE_DHT_H

#include "executor.h"
#include "r2c.h"

typedef struct {
    int N;                 /* DHT size (must be even) */
    size_t K;              /* batch count */
    int n_threads;         /* T_plan snapshot — caps execute-time T */
    double *buf_re;        /* N*K scratch for input copy + R2C Re bins */
    double *buf_im;        /* (N/2+1)*K scratch for R2C Im bins */
    stride_plan_t *r2c_plan;
} stride_dht_data_t;


/* ═══════════════════════════════════════════════════════════════
 * MT INFRASTRUCTURE — K-split worker for the post-butterfly pass.
 *
 * The pre phase is one big memcpy of N*K doubles — bandwidth-bound, not
 * worth parallelizing (1 sequential memcpy beats T smaller memcpys when
 * memory bandwidth is the limit). Only the post-butterfly is K-split.
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    stride_dht_data_t *d;
    double *re;
    size_t k0, k1;
} _dht_slice_arg_t;

static inline int _dht_mt_threads(int n_threads_plan, int N, size_t K) {
    int T = stride_get_num_threads();
    if (T > n_threads_plan) T = n_threads_plan;
    if (T > _stride_pool_size + 1) T = _stride_pool_size + 1;
    if (T < 1) T = 1;
    if (T > 1 && (size_t)N * K < (size_t)8192 * (size_t)T) T = 1;
    return T;
}

/* Post-butterfly K-slice: H[i]=Re-Im, H[N-i]=Re+Im, plus boundary cases. */
static void _dht_worker_post(void *arg) {
    _dht_slice_arg_t *a = (_dht_slice_arg_t *)arg;
    stride_dht_data_t *d = a->d;
    const int N = d->N;
    const int halfN = N / 2;
    const size_t K = d->K;
    const size_t k0 = a->k0, k1 = a->k1, slice = k1 - k0;
    double *re = a->re;
    const double *buf_re = d->buf_re;
    const double *buf_im = d->buf_im;

    /* H[0] = Re(X[0]) */
    memcpy(re + k0, buf_re + k0, slice * sizeof(double));

    for (int i = 1; i < halfN; i++) {
        const double *Zr = buf_re + (size_t)i * K;
        const double *Zi = buf_im + (size_t)i * K;
        double *out_lo   = re + (size_t)i       * K;
        double *out_hi   = re + (size_t)(N - i) * K;
        for (size_t k = k0; k < k1; k++) {
            double aa = Zr[k];
            double bb = Zi[k];
            out_lo[k] = aa - bb;
            out_hi[k] = aa + bb;
        }
    }

    /* H[N/2] = Re(X[N/2]) */
    memcpy(re + (size_t)halfN * K + k0,
           buf_re + (size_t)halfN * K + k0,
           slice * sizeof(double));
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- DHT (forward; same routine serves as backward via
 *            self-inverse property — caller divides by N).
 * ═══════════════════════════════════════════════════════════════ */

static void _dht_execute(void *data, double *re, double *im) {
    (void)im;
    stride_dht_data_t *d = (stride_dht_data_t *)data;
    const int N = d->N;
    const size_t K = d->K;
    const int halfN = N / 2;

    /* 1. Copy real input into scratch and run N-point R2C in-place on scratch.
     *    After this:
     *      buf_re[i*K..(i+1)*K) = Re(X[i]) for i = 0 .. N/2
     *      buf_im[i*K..(i+1)*K) = Im(X[i]) for i = 0 .. N/2  (Im[0]=Im[N/2]=0)
     *    Higher rows of buf_re (i > N/2) hold residual input — unused.
     *    The pre memcpy is bandwidth-bound and runs single-threaded; the
     *    inner R2C uses its own MT internally on the same pool. */
    memcpy(d->buf_re, re, (size_t)N * K * sizeof(double));
    stride_execute_fwd(d->r2c_plan, d->buf_re, d->buf_im);

    /* 2. Post-butterfly: K-split across threads when worth it. */
    int T = _dht_mt_threads(d->n_threads, N, K);
    if (T > 1) {
        _dht_slice_arg_t args[64];
        for (int t = 0; t < T; t++) {
            args[t].d  = d;
            args[t].re = re;
            args[t].k0 = (K * (size_t)t)       / (size_t)T;
            args[t].k1 = (K * (size_t)(t + 1)) / (size_t)T;
        }
        for (int t = 1; t < T; t++)
            _stride_pool_dispatch(&_stride_workers[t - 1],
                                  _dht_worker_post, &args[t]);
        _dht_worker_post(&args[0]);
        _stride_pool_wait_all();
        return;
    }

    /* Single-threaded post-butterfly path:
     *      H[0] = Re(X[0])
     *      H[i] = Re(X[i]) - Im(X[i])
     *      H[N-i] = Re(X[i]) + Im(X[i])
     *      H[N/2] = Re(X[N/2])
     */
    for (size_t k = 0; k < K; k++)
        re[k] = d->buf_re[k];

    for (int i = 1; i < halfN; i++) {
        const double *Zr = d->buf_re + (size_t)i * K;
        const double *Zi = d->buf_im + (size_t)i * K;
        double *out_lo   = re        + (size_t)i       * K;
        double *out_hi   = re        + (size_t)(N - i) * K;
        for (size_t k = 0; k < K; k++) {
            double a = Zr[k];
            double b = Zi[k];
            out_lo[k] = a - b;
            out_hi[k] = a + b;
        }
    }

    {
        const double *Zr = d->buf_re + (size_t)halfN * K;
        double *out      = re        + (size_t)halfN * K;
        for (size_t k = 0; k < K; k++)
            out[k] = Zr[k];
    }
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _dht_destroy(void *data) {
    stride_dht_data_t *d = (stride_dht_data_t *)data;
    if (!d) return;
    STRIDE_ALIGNED_FREE(d->buf_re);
    STRIDE_ALIGNED_FREE(d->buf_im);
    if (d->r2c_plan) stride_plan_destroy(d->r2c_plan);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Caller provides an N-point R2C plan; the DHT plan owns it from here.
 * Constraint: N must be even.
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *stride_dht_plan(int N, size_t K, stride_plan_t *r2c_plan_N)
{
    if (N < 2 || (N & 1) || !r2c_plan_N) {
        if (r2c_plan_N) stride_plan_destroy(r2c_plan_N);
        return NULL;
    }

    stride_dht_data_t *d = (stride_dht_data_t *)calloc(1, sizeof(*d));
    if (!d) { stride_plan_destroy(r2c_plan_N); return NULL; }

    d->N = N;
    d->K = K;
    d->r2c_plan = r2c_plan_N;

    int T_plan = stride_get_num_threads();
    if (T_plan < 1) T_plan = 1;
    d->n_threads = T_plan;

    /* buf_re: N*K (large enough for the input copy that R2C consumes).
     * buf_im: (N/2+1)*K (matches R2C's split Im output). */
    size_t buf_re_sz = (size_t)N * K;
    size_t buf_im_sz = (size_t)(N / 2 + 1) * K;
    d->buf_re = (double *)STRIDE_ALIGNED_ALLOC(64, buf_re_sz * sizeof(double));
    d->buf_im = (double *)STRIDE_ALIGNED_ALLOC(64, buf_im_sz * sizeof(double));
    if (!d->buf_re || !d->buf_im) { _dht_destroy(d); return NULL; }

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _dht_destroy(d); return NULL; }

    plan->N = N;
    plan->K = K;
    plan->num_stages = 0;
    plan->override_fwd     = _dht_execute;   /* DHT is self-inverse */
    plan->override_bwd     = _dht_execute;   /* same routine; caller divides by N */
    plan->override_destroy = _dht_destroy;
    plan->override_data    = d;

    return plan;
}


/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE API
 *
 * stride_execute_dht(plan, in, out): copy in -> out, then in-place DHT on out.
 *   To recover x from H = DHT(x), apply DHT again and divide by N.
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_dht(const stride_plan_t *plan,
                                       const double *in, double *out) {
    size_t NK = (size_t)plan->N * plan->K;
    if (in != out) memcpy(out, in, NK * sizeof(double));
    plan->override_fwd(plan->override_data, out, NULL);
}


#endif /* STRIDE_DHT_H */
