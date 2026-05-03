/**
 * dct4.h -- DCT-IV (real-to-real, FFTW REDFT11 convention)
 *
 *   Y[k] = 2 * sum_{n=0..N-1} x[n] * cos(pi*(2k+1)*(2n+1)/(4N))   for k=0..N-1
 *
 * Involutory up to scale 2N: DCT-IV(DCT-IV(x))/(2N) = x.
 * Used in MDCT (modified DCT) for audio codecs (MP3/AAC/Vorbis/Opus).
 *
 * Algorithm: Lee 1984 -- single N/2-point complex FFT plus pre/post twiddles.
 *
 * Derivation (in summary form):
 *   Define z[m] = x[2m] - i*x[N-1-2m]    for m = 0..N/2-1
 *
 *   Pair Y[2k'] (cos-based) with Y[N-1-2k'] (sin-based via the (-1)^n
 *   index identity) into one complex sequence:
 *     Z[k'] := Y[2k'] + i*Y[N-1-2k']
 *
 *   Index manipulation collapses the four (n,k) parities into:
 *     Z[k'] = 2 * exp(i*pi*(4k'+1)/(4N)) * IFFT_{N/2}(psi)[k']
 *
 *   where psi[m] = z[m] * exp(i*pi*m/N)   (pre-twiddle by exp(i*pi*m/N))
 *   and  IFFT_{N/2}(psi)[k'] = sum_m psi[m] * exp(+2*pi*i*m*k'/(N/2))
 *                              (unnormalized backward FFT in our convention).
 *
 *   Y[2k']     = Re(Z[k'])
 *   Y[N-1-2k'] = Im(Z[k'])
 *
 * Cost: 1 N/2-point complex FFT + 2 O(N*K) twiddle passes. ~2x faster than
 * the textbook DCT-III + DST-III combo (which costs 2 R2C-based transforms).
 *
 * Constraint: N must be even.
 *
 * Layout (split-batched, real input/output):
 *   in[n*K + k]  for n=0..N-1, k=0..K-1
 *   out[n*K + k] same shape
 */
#ifndef STRIDE_DCT4_H
#define STRIDE_DCT4_H

#include "executor.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int N;                  /* DCT-IV size (must be even) */
    size_t K;               /* batch count */
    /* Pre-twiddle: exp(i*pi*m/N) for m=0..N/2-1 */
    double *pre_cos;
    double *pre_sin;
    /* Post-twiddle: exp(i*pi*(4k'+1)/(4N)) for k'=0..N/2-1.
     * Includes the 2x prefactor: stored as 2*cos(...) and 2*sin(...). */
    double *post_cos2;
    double *post_sin2;
    /* Scratch for N/2-point complex FFT (split-complex). */
    double *psi_re;
    double *psi_im;
    stride_plan_t *fft_plan; /* owned: N/2-point complex FFT plan */
} stride_dct4_data_t;


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- DCT-IV via Lee 1984.
 *
 * 1. Pre-twiddle: psi[m] = (x[2m] - i*x[N-1-2m]) * exp(i*pi*m/N)
 * 2. N/2-point backward FFT (e^{+2*pi*i/M}, unnormalized).
 * 3. Post-twiddle + unpack:
 *      P[k'] = 2 * exp(i*pi*(4k'+1)/(4N)) * W[k']
 *      Y[2k']     = Re(P[k'])
 *      Y[N-1-2k'] = Im(P[k'])
 * ═══════════════════════════════════════════════════════════════ */

static void _dct4_execute(void *data, double *re, double *im) {
    (void)im;
    stride_dct4_data_t *d = (stride_dct4_data_t *)data;
    const int N = d->N;
    const int halfN = N / 2;
    const size_t K = d->K;

    /* 1. Pre-twiddle. Reads re[2m*K..] and re[(N-1-2m)*K..]; writes psi scratch.
     *    psi_re[m] = x[2m]*cos(pi*m/N) + x[N-1-2m]*sin(pi*m/N)
     *    psi_im[m] = x[2m]*sin(pi*m/N) - x[N-1-2m]*cos(pi*m/N)
     */
    for (int m = 0; m < halfN; m++) {
        const double *x_e = re + (size_t)(2 * m)         * K;  /* x[2m] */
        const double *x_o = re + (size_t)(N - 1 - 2 * m) * K;  /* x[N-1-2m] */
        double *psi_r = d->psi_re + (size_t)m * K;
        double *psi_i = d->psi_im + (size_t)m * K;
        const double cm = d->pre_cos[m];
        const double sm = d->pre_sin[m];
        for (size_t k = 0; k < K; k++) {
            const double xe = x_e[k];
            const double xo = x_o[k];
            psi_r[k] = xe * cm + xo * sm;
            psi_i[k] = xe * sm - xo * cm;
        }
    }

    /* 2. DIAG: backward via conj+fwd+conj to isolate executor bug.
     *    bwd(psi)[k'] = conj(fwd(conj(psi))[k']) */
    for (size_t i = 0; i < (size_t)halfN * K; i++) d->psi_im[i] = -d->psi_im[i];
    stride_execute_fwd(d->fft_plan, d->psi_re, d->psi_im);
    for (size_t i = 0; i < (size_t)halfN * K; i++) d->psi_im[i] = -d->psi_im[i];

    /* 3. Post-twiddle + unpack.
     *    For each k'=0..halfN-1:
     *      Y[2k']     = 2 * (post_cos[k']*W_re - post_sin[k']*W_im)
     *      Y[N-1-2k'] = 2 * (post_cos[k']*W_im + post_sin[k']*W_re)
     *    (post_cos2/post_sin2 already include the 2x.)
     */
    for (int kp = 0; kp < halfN; kp++) {
        const double pc = d->post_cos2[kp];
        const double ps = d->post_sin2[kp];
        const double *Wr = d->psi_re + (size_t)kp * K;
        const double *Wi = d->psi_im + (size_t)kp * K;
        double *out_lo = re + (size_t)(2 * kp)         * K;
        double *out_hi = re + (size_t)(N - 1 - 2 * kp) * K;
        for (size_t k = 0; k < K; k++) {
            const double wr = Wr[k];
            const double wi = Wi[k];
            out_lo[k] = pc * wr - ps * wi;
            out_hi[k] = pc * wi + ps * wr;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _dct4_destroy(void *data) {
    stride_dct4_data_t *d = (stride_dct4_data_t *)data;
    if (!d) return;
    free(d->pre_cos);
    free(d->pre_sin);
    free(d->post_cos2);
    free(d->post_sin2);
    STRIDE_ALIGNED_FREE(d->psi_re);
    STRIDE_ALIGNED_FREE(d->psi_im);
    if (d->fft_plan) stride_plan_destroy(d->fft_plan);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Caller provides an N/2-point complex FFT plan. The DCT-IV plan owns
 * it from here. Constraint: N must be even.
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *stride_dct4_plan(int N, size_t K, stride_plan_t *fft_plan_halfN)
{
    if (N < 2 || (N & 1) || !fft_plan_halfN) {
        if (fft_plan_halfN) stride_plan_destroy(fft_plan_halfN);
        return NULL;
    }

    stride_dct4_data_t *d = (stride_dct4_data_t *)calloc(1, sizeof(*d));
    if (!d) { stride_plan_destroy(fft_plan_halfN); return NULL; }

    d->N = N;
    d->K = K;
    d->fft_plan = fft_plan_halfN;

    const int halfN = N / 2;

    d->pre_cos = (double *)malloc((size_t)halfN * sizeof(double));
    d->pre_sin = (double *)malloc((size_t)halfN * sizeof(double));
    d->post_cos2 = (double *)malloc((size_t)halfN * sizeof(double));
    d->post_sin2 = (double *)malloc((size_t)halfN * sizeof(double));
    if (!d->pre_cos || !d->pre_sin || !d->post_cos2 || !d->post_sin2) {
        _dct4_destroy(d); return NULL;
    }
    for (int m = 0; m < halfN; m++) {
        const double a = M_PI * (double)m / (double)N;
        d->pre_cos[m] = cos(a);
        d->pre_sin[m] = sin(a);
    }
    for (int kp = 0; kp < halfN; kp++) {
        const double a = M_PI * (double)(4 * kp + 1) / (4.0 * (double)N);
        d->post_cos2[kp] = 2.0 * cos(a);
        d->post_sin2[kp] = 2.0 * sin(a);
    }

    size_t psi_sz = (size_t)halfN * K;
    d->psi_re = (double *)STRIDE_ALIGNED_ALLOC(64, psi_sz * sizeof(double));
    d->psi_im = (double *)STRIDE_ALIGNED_ALLOC(64, psi_sz * sizeof(double));
    if (!d->psi_re || !d->psi_im) { _dct4_destroy(d); return NULL; }

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
