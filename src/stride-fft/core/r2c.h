/**
 * stride_r2c.h -- Real-to-Complex (R2C) and Complex-to-Real (C2R) FFT
 *
 * Converts an N-point real FFT into one N/2-point complex FFT plus a
 * post-process butterfly, exploiting Hermitian symmetry for 2x speedup.
 *
 * Algorithm (pair-packing):
 *   Forward (R2C): N reals -> N/2+1 complex
 *     1. Pack pairs: z[n] = x[2n] + i*x[2n+1]
 *     2. N/2-point complex FFT of z
 *     3. Post-process: extract X[k] from Z[k] via twiddle butterfly
 *
 *   Backward (C2R): N/2+1 complex -> N reals
 *     1. Pre-process: reconstruct Z from X (reverse butterfly)
 *     2. N/2-point complex IFFT of Z
 *     3. Unpack: x[2n] = 2*Re(z[n]), x[2n+1] = 2*Im(z[n])
 *
 * Normalization: bwd(fwd(x)) = N * x (consistent with complex convention).
 *
 * Data layout (split-complex, batched):
 *   Real input:    real[n * K + k]  for n=0..N-1, k=0..K-1
 *   Complex output: re[f * K + k], im[f * K + k]  for f=0..N/2, k=0..K-1
 *
 * N must be even. For odd N, use the complex FFT with im=0.
 */
#ifndef STRIDE_R2C_H
#define STRIDE_R2C_H

#include "executor.h"

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif


/* ═══════════════════════════════════════════════════════════════
 * R2C DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int N;                  /* original real-FFT size (must be even) */
    int half_N;             /* N/2 (inner FFT size) */
    size_t K;               /* batch count */
    size_t B;               /* block size for cache-friendly execution */

    double *tw_re;          /* N/2 twiddle factors: W_N^k = cos(-2*pi*k/N) */
    double *tw_im;          /* N/2 twiddle factors: sin(-2*pi*k/N) */

    int *perm;              /* N/2 digit-reversal permutation: natural → DIT output order */
    int *iperm;             /* N/2 inverse permutation: DIT output → natural order */

    double *scratch_re;     /* N/2 * B scratch for inner FFT */
    double *scratch_im;
    double *c2r_im_buf;    /* (N/2+1) * K pre-allocated temp for stride_execute_c2r */

    stride_plan_t *inner;   /* N/2-point complex FFT plan with K = B */
} stride_r2c_data_t;


/* ═══════════════════════════════════════════════════════════════
 * TWIDDLE PRECOMPUTATION
 * ═══════════════════════════════════════════════════════════════ */

/* Compute mixed-radix digit-reversal permutation.
 * For DIT forward: output[perm[n]] = DFT[n].
 * So to read DFT[n] from the output, access output[perm[n]].
 * iperm is the inverse: output[k] = DFT[iperm[k]]. */
static void _r2c_compute_perm(const int *factors, int nf, int N,
                               int *perm, int *iperm) {
    for (int n = 0; n < N; n++) {
        int idx = n, rev = 0, radix_product = 1;
        for (int s = 0; s < nf; s++) {
            int R = factors[s];
            int digit = idx % R;
            idx /= R;
            rev += digit * (N / (radix_product * R));
            radix_product *= R;
        }
        perm[n] = rev;
    }
    for (int n = 0; n < N; n++)
        iperm[perm[n]] = n;
}

static void _r2c_init_twiddles(int N, double *tw_re, double *tw_im) {
    int half_N = N / 2;
    for (int k = 0; k < half_N; k++) {
        double angle = -2.0 * M_PI * (double)k / (double)N;
        tw_re[k] = cos(angle);
        tw_im[k] = sin(angle);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * POST-PROCESS (forward R2C): Z[0..N/2-1] -> X[0..N/2]
 *
 * Given the N/2-point complex FFT output Z in scratch, compute
 * the N-point real FFT output X in the output arrays.
 *
 * DC and Nyquist bins are purely real:
 *   X[0]   = Re(Z[0]) + Im(Z[0])
 *   X[N/2] = Re(Z[0]) - Im(Z[0])
 *
 * For k=1..N/2-1, butterfly pairs (k, N/2-k):
 *   E = (Z[k] + conj(Z[N/2-k])) / 2       (even part)
 *   O = (Z[k] - conj(Z[N/2-k])) / 2       (odd part)
 *   X[k] = E + W_N^k * (-i * O)
 * ═══════════════════════════════════════════════════════════════ */

static void _r2c_postprocess(
        const double * __restrict__ z_re,
        const double * __restrict__ z_im,
        double * __restrict__ out_re,
        double * __restrict__ out_im,
        const double * __restrict__ tw_re,
        const double * __restrict__ tw_im,
        const int * __restrict__ perm,
        int half_N, size_t K, size_t B, size_t b0)
{
    const size_t out_len = (size_t)(half_N + 1);
    (void)out_len;

    /* DC (f=0) and Nyquist (f=N/2) — Z[0] is at perm[0] in DIT output */
    size_t z0_off = (size_t)perm[0] * B;
    size_t nyq_off = (size_t)half_N * K + b0;
    {
        size_t k = 0;
#if defined(__AVX512F__)
        for (; k + 8 <= B; k += 8) {
            __m512d zr = _mm512_load_pd(z_re + z0_off + k);
            __m512d zi = _mm512_load_pd(z_im + z0_off + k);
            _mm512_storeu_pd(out_re + b0 + k, _mm512_add_pd(zr, zi));
            _mm512_storeu_pd(out_im + b0 + k, _mm512_setzero_pd());
            _mm512_storeu_pd(out_re + nyq_off + k, _mm512_sub_pd(zr, zi));
            _mm512_storeu_pd(out_im + nyq_off + k, _mm512_setzero_pd());
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        for (; k + 4 <= B; k += 4) {
            __m256d zr = _mm256_load_pd(z_re + z0_off + k);
            __m256d zi = _mm256_load_pd(z_im + z0_off + k);
            _mm256_storeu_pd(out_re + b0 + k, _mm256_add_pd(zr, zi));
            _mm256_storeu_pd(out_im + b0 + k, _mm256_setzero_pd());
            _mm256_storeu_pd(out_re + nyq_off + k, _mm256_sub_pd(zr, zi));
            _mm256_storeu_pd(out_im + nyq_off + k, _mm256_setzero_pd());
        }
#endif
        for (; k < B; k++) {
            out_re[b0 + k] = z_re[z0_off + k] + z_im[z0_off + k];
            out_im[b0 + k] = 0.0;
            out_re[nyq_off + k] = z_re[z0_off + k] - z_im[z0_off + k];
            out_im[nyq_off + k] = 0.0;
        }
    }

    /* Butterfly pairs (f, N/2-f) for f=1..N/2-1 */
    for (int f = 1; f < half_N; f++) {
        int mirror = half_N - f;
        if (f > mirror) break;

        const double wr = tw_re[f], wi = tw_im[f];
        size_t f_off  = (size_t)perm[f] * B;
        size_t m_off  = (size_t)perm[mirror] * B;
        size_t fo_off = (size_t)f * K + b0;
        size_t mo_off = (size_t)mirror * K + b0;

        size_t k = 0;
#if defined(__AVX512F__)
        {
            __m512d half_v = _mm512_set1_pd(0.5);
            __m512d vwr    = _mm512_set1_pd(wr);
            __m512d vwi    = _mm512_set1_pd(wi);

            for (; k + 8 <= B; k += 8) {
                __m512d Zfr = _mm512_load_pd(z_re + f_off + k);
                __m512d Zfi = _mm512_load_pd(z_im + f_off + k);
                __m512d Zmr = _mm512_load_pd(z_re + m_off + k);
                __m512d Zmi = _mm512_load_pd(z_im + m_off + k);

                __m512d Er = _mm512_mul_pd(_mm512_add_pd(Zfr, Zmr), half_v);
                __m512d Ei = _mm512_mul_pd(_mm512_sub_pd(Zfi, Zmi), half_v);
                __m512d Or = _mm512_mul_pd(_mm512_sub_pd(Zfr, Zmr), half_v);
                __m512d Oi = _mm512_mul_pd(_mm512_add_pd(Zfi, Zmi), half_v);

                /* -i*O = (Oi, -Or) */
                __m512d niOr = Oi;
                __m512d neg_Or = _mm512_sub_pd(_mm512_setzero_pd(), Or);

                __m512d Tr = _mm512_fmsub_pd(vwr, niOr, _mm512_mul_pd(vwi, neg_Or));
                __m512d Ti = _mm512_fmadd_pd(vwr, neg_Or, _mm512_mul_pd(vwi, niOr));

                _mm512_storeu_pd(out_re + fo_off + k, _mm512_add_pd(Er, Tr));
                _mm512_storeu_pd(out_im + fo_off + k, _mm512_add_pd(Ei, Ti));

                if (f != mirror) {
                    __m512d vwrm = _mm512_set1_pd(tw_re[mirror]);
                    __m512d vwim = _mm512_set1_pd(tw_im[mirror]);

                    __m512d Emr = Er;
                    __m512d Emi = _mm512_sub_pd(_mm512_setzero_pd(), Ei);
                    __m512d neg_Or2 = neg_Or;   /* -Or */
                    __m512d Omi = Oi;

                    __m512d niOmr = Omi;
                    __m512d niOmi = _mm512_sub_pd(_mm512_setzero_pd(), neg_Or2); /* Or */

                    __m512d Tmr = _mm512_fmsub_pd(vwrm, niOmr, _mm512_mul_pd(vwim, niOmi));
                    __m512d Tmi = _mm512_fmadd_pd(vwrm, niOmi, _mm512_mul_pd(vwim, niOmr));

                    _mm512_storeu_pd(out_re + mo_off + k, _mm512_add_pd(Emr, Tmr));
                    _mm512_storeu_pd(out_im + mo_off + k, _mm512_add_pd(Emi, Tmi));
                }
            }
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        {
            __m256d half_v = _mm256_set1_pd(0.5);
            __m256d sign   = _mm256_set1_pd(-0.0);
            __m256d vwr    = _mm256_set1_pd(wr);
            __m256d vwi    = _mm256_set1_pd(wi);

            for (; k + 4 <= B; k += 4) {
                __m256d Zfr = _mm256_load_pd(z_re + f_off + k);
                __m256d Zfi = _mm256_load_pd(z_im + f_off + k);
                __m256d Zmr = _mm256_load_pd(z_re + m_off + k);
                __m256d Zmi = _mm256_load_pd(z_im + m_off + k);

                __m256d Er = _mm256_mul_pd(_mm256_add_pd(Zfr, Zmr), half_v);
                __m256d Ei = _mm256_mul_pd(_mm256_sub_pd(Zfi, Zmi), half_v);
                __m256d Or = _mm256_mul_pd(_mm256_sub_pd(Zfr, Zmr), half_v);
                __m256d Oi = _mm256_mul_pd(_mm256_add_pd(Zfi, Zmi), half_v);

                /* -i*O = (Oi, -Or) */
                __m256d niOr = Oi;
                __m256d niOi = _mm256_xor_pd(Or, sign);

                __m256d Tr = _mm256_fmsub_pd(vwr, niOr, _mm256_mul_pd(vwi, niOi));
                __m256d Ti = _mm256_fmadd_pd(vwr, niOi, _mm256_mul_pd(vwi, niOr));

                _mm256_storeu_pd(out_re + fo_off + k, _mm256_add_pd(Er, Tr));
                _mm256_storeu_pd(out_im + fo_off + k, _mm256_add_pd(Ei, Ti));

                if (f != mirror) {
                    __m256d vwrm = _mm256_set1_pd(tw_re[mirror]);
                    __m256d vwim = _mm256_set1_pd(tw_im[mirror]);

                    __m256d Emr = Er;
                    __m256d Emi = _mm256_xor_pd(Ei, sign);
                    __m256d Omr = _mm256_xor_pd(Or, sign);
                    __m256d Omi = Oi;

                    __m256d niOmr = Omi;
                    __m256d niOmi = _mm256_xor_pd(Omr, sign);

                    __m256d Tmr = _mm256_fmsub_pd(vwrm, niOmr, _mm256_mul_pd(vwim, niOmi));
                    __m256d Tmi = _mm256_fmadd_pd(vwrm, niOmi, _mm256_mul_pd(vwim, niOmr));

                    _mm256_storeu_pd(out_re + mo_off + k, _mm256_add_pd(Emr, Tmr));
                    _mm256_storeu_pd(out_im + mo_off + k, _mm256_add_pd(Emi, Tmi));
                }
            }
        }
#endif
        /* Scalar tail */
        for (; k < B; k++) {
            double Zfr = z_re[f_off+k], Zfi = z_im[f_off+k];
            double Zmr = z_re[m_off+k], Zmi = z_im[m_off+k];
            double Er = (Zfr + Zmr) * 0.5, Ei = (Zfi - Zmi) * 0.5;
            double Or = (Zfr - Zmr) * 0.5, Oi = (Zfi + Zmi) * 0.5;
            double niOr = Oi, niOi = -Or;
            double Tr = wr*niOr - wi*niOi, Ti = wr*niOi + wi*niOr;
            out_re[fo_off+k] = Er + Tr;
            out_im[fo_off+k] = Ei + Ti;
            if (f != mirror) {
                double wrm = tw_re[mirror], wim = tw_im[mirror];
                double Emr = Er, Emi = -Ei, Omr = -Or, Omi = Oi;
                double niOmr = Omi, niOmi = -Omr;
                double Tmr = wrm*niOmr - wim*niOmi, Tmi = wrm*niOmi + wim*niOmr;
                out_re[mo_off+k] = Emr + Tmr;
                out_im[mo_off+k] = Emi + Tmi;
            }
        }

        /* If f == mirror (N divisible by 4, f = N/4), the loop handles it
         * via the f != mirror guard — the single bin is computed once. */
    }
}


/* ═══════════════════════════════════════════════════════════════
 * PRE-PROCESS (backward C2R): X[0..N/2] -> Z[0..N/2-1]
 *
 * Reverse of post-process: reconstruct Z from X.
 *   Z[0] = (X[0].re + X[N/2].re) + i*(X[0].re - X[N/2].re)
 *   Z[f] = E - W_N^f * (-i * O)   ... (inversion of forward)
 *
 * More precisely, from X[f] and X[N/2-f]:
 *   E = (X[f] + conj(X[N/2-f])) / 2
 *   D = (X[f] - conj(X[N/2-f])) / 2
 *   Xo = conj(W_N^f) * D
 *   Z[f] = E + i * Xo = (E.re - Xo.im) + i*(E.im + Xo.re)
 * ═══════════════════════════════════════════════════════════════ */

static void _r2c_preprocess(
        const double * __restrict__ in_re,
        const double * __restrict__ in_im,
        double * __restrict__ z_re,
        double * __restrict__ z_im,
        const double * __restrict__ tw_re,
        const double * __restrict__ tw_im,
        const int * __restrict__ perm,
        int half_N, size_t K, size_t B, size_t b0)
{
    /* DC: Z[0] written to permuted position perm[0] */
    {
        size_t z0_out = (size_t)perm[0] * B;
        size_t nyq = (size_t)half_N * K + b0;
        size_t k = 0;
#if defined(__AVX512F__)
        for (; k + 8 <= B; k += 8) {
            __m512d x0 = _mm512_loadu_pd(in_re + b0 + k);
            __m512d xn = _mm512_loadu_pd(in_re + nyq + k);
            _mm512_store_pd(z_re + z0_out + k, _mm512_add_pd(x0, xn));
            _mm512_store_pd(z_im + z0_out + k, _mm512_sub_pd(x0, xn));
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        for (; k + 4 <= B; k += 4) {
            __m256d x0 = _mm256_loadu_pd(in_re + b0 + k);
            __m256d xn = _mm256_loadu_pd(in_re + nyq + k);
            _mm256_store_pd(z_re + z0_out + k, _mm256_add_pd(x0, xn));
            _mm256_store_pd(z_im + z0_out + k, _mm256_sub_pd(x0, xn));
        }
#endif
        for (; k < B; k++) {
            double x0r = in_re[b0 + k];
            double xnr = in_re[nyq + k];
            z_re[z0_out + k] = x0r + xnr;
            z_im[z0_out + k] = x0r - xnr;
        }
    }

    /* Butterfly pairs — write Z[f] to perm[f], Z[mirror] to perm[mirror] */
    for (int f = 1; f < half_N; f++) {
        int mirror = half_N - f;
        size_t fi = (size_t)f * K + b0;
        size_t mi = (size_t)mirror * K + b0;
        size_t fo = (size_t)perm[f] * B;
        size_t mo = (size_t)perm[mirror] * B;
        (void)mo;

        /* conj(W) for this bin */
        double cwr = tw_re[f], cwi = -tw_im[f];

        size_t k = 0;
#if defined(__AVX512F__)
        {
            __m512d half_v = _mm512_set1_pd(0.5);
            __m512d vcwr   = _mm512_set1_pd(cwr);
            __m512d vcwi   = _mm512_set1_pd(cwi);
            for (; k + 8 <= B; k += 8) {
                __m512d Xfr = _mm512_loadu_pd(in_re + fi + k);
                __m512d Xfi = _mm512_loadu_pd(in_im + fi + k);
                __m512d Xmr = _mm512_loadu_pd(in_re + mi + k);
                __m512d Xmi = _mm512_loadu_pd(in_im + mi + k);

                __m512d Er = _mm512_mul_pd(_mm512_add_pd(Xfr, Xmr), half_v);
                __m512d Ei = _mm512_mul_pd(_mm512_sub_pd(Xfi, Xmi), half_v);
                __m512d Dr = _mm512_mul_pd(_mm512_sub_pd(Xfr, Xmr), half_v);
                __m512d Di = _mm512_mul_pd(_mm512_add_pd(Xfi, Xmi), half_v);

                __m512d Xor = _mm512_fmsub_pd(vcwr, Dr, _mm512_mul_pd(vcwi, Di));
                __m512d Xoi = _mm512_fmadd_pd(vcwr, Di, _mm512_mul_pd(vcwi, Dr));

                _mm512_store_pd(z_re + fo + k, _mm512_sub_pd(Er, Xoi));
                _mm512_store_pd(z_im + fo + k, _mm512_add_pd(Ei, Xor));
            }
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        {
            __m256d half_v = _mm256_set1_pd(0.5);
            __m256d vcwr   = _mm256_set1_pd(cwr);
            __m256d vcwi   = _mm256_set1_pd(cwi);
            for (; k + 4 <= B; k += 4) {
                __m256d Xfr = _mm256_loadu_pd(in_re + fi + k);
                __m256d Xfi = _mm256_loadu_pd(in_im + fi + k);
                __m256d Xmr = _mm256_loadu_pd(in_re + mi + k);
                __m256d Xmi = _mm256_loadu_pd(in_im + mi + k);

                __m256d Er = _mm256_mul_pd(_mm256_add_pd(Xfr, Xmr), half_v);
                __m256d Ei = _mm256_mul_pd(_mm256_sub_pd(Xfi, Xmi), half_v);
                __m256d Dr = _mm256_mul_pd(_mm256_sub_pd(Xfr, Xmr), half_v);
                __m256d Di = _mm256_mul_pd(_mm256_add_pd(Xfi, Xmi), half_v);

                __m256d Xor = _mm256_fmsub_pd(vcwr, Dr, _mm256_mul_pd(vcwi, Di));
                __m256d Xoi = _mm256_fmadd_pd(vcwr, Di, _mm256_mul_pd(vcwi, Dr));

                _mm256_store_pd(z_re + fo + k, _mm256_sub_pd(Er, Xoi));
                _mm256_store_pd(z_im + fo + k, _mm256_add_pd(Ei, Xor));
            }
        }
#endif
        for (; k < B; k++) {
            double Xfr = in_re[fi+k], Xfi = in_im[fi+k];
            double Xmr = in_re[mi+k], Xmi = in_im[mi+k];

            double Er = (Xfr + Xmr) * 0.5;
            double Ei = (Xfi - Xmi) * 0.5;
            double Dr = (Xfr - Xmr) * 0.5;
            double Di = (Xfi + Xmi) * 0.5;

            double Xor = cwr * Dr - cwi * Di;
            double Xoi = cwr * Di + cwi * Dr;

            z_re[fo+k] = Er - Xoi;
            z_im[fo+k] = Ei + Xor;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- FORWARD R2C (block-walk)
 * ═══════════════════════════════════════════════════════════════ */

static void _r2c_execute_fwd(void *data, double *re, double *im) {
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    const int N = d->N, halfN = d->half_N;
    const size_t K = d->K, B = d->B;
    double *sr = d->scratch_re, *si = d->scratch_im;

    for (size_t b0 = 0; b0 < K; b0 += B) {
        /* 1. Pack pairs: z[n] = x[2n] + i*x[2n+1] */
        for (int n = 0; n < halfN; n++) {
            const double *even = re + (size_t)(2*n) * K + b0;
            const double *odd  = re + (size_t)(2*n+1) * K + b0;
            double *dst_r = sr + (size_t)n * B;
            double *dst_i = si + (size_t)n * B;
            size_t k = 0;
#if defined(__AVX512F__)
            for (; k + 8 <= B; k += 8) {
                _mm512_store_pd(dst_r + k, _mm512_loadu_pd(even + k));
                _mm512_store_pd(dst_i + k, _mm512_loadu_pd(odd  + k));
            }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
            for (; k + 4 <= B; k += 4) {
                _mm256_store_pd(dst_r + k, _mm256_loadu_pd(even + k));
                _mm256_store_pd(dst_i + k, _mm256_loadu_pd(odd  + k));
            }
#endif
            for (; k < B; k++) {
                dst_r[k] = even[k];
                dst_i[k] = odd[k];
            }
        }

        /* 2. N/2-point complex FFT (DIT forward — output in digit-reversed order) */
        stride_execute_fwd(d->inner, sr, si);

        /* 3. Post-process with permuted indices: reads Z[perm[f]] directly,
         *    no separate permutation pass needed. */
        _r2c_postprocess(sr, si, re, im, d->tw_re, d->tw_im, d->perm,
                         halfN, K, B, b0);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- BACKWARD C2R (block-walk)
 *
 * Unnormalized: output = N * original_input.
 * Caller divides by N to normalize (consistent with complex bwd).
 * ═══════════════════════════════════════════════════════════════ */

static void _r2c_execute_bwd(void *data, double *re, double *im) {
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    const int N = d->N, halfN = d->half_N;
    const size_t K = d->K, B = d->B;
    double *sr = d->scratch_re, *si = d->scratch_im;

    for (size_t b0 = 0; b0 < K; b0 += B) {
        /* 1. Pre-process: reconstruct Z and write to permuted positions
         *    (DIF bwd expects digit-reversed input). No separate permutation pass. */
        _r2c_preprocess(re, im, sr, si, d->tw_re, d->tw_im, d->perm,
                        halfN, K, B, b0);

        /* 2. N/2-point complex IFFT (unnormalized: gives halfN * z[n]) */
        stride_execute_bwd(d->inner, sr, si);

        /* 3. Unpack: x[2n] = 2*Re(z[n]), x[2n+1] = 2*Im(z[n])
         *    Factor 2: inner bwd gives halfN * z[n], we need N * x[n].
         *    Since z[n] = x[2n] + i*x[2n+1], halfN * z[n] -> need *2 for N. */
        for (int n = 0; n < halfN; n++) {
            const double *src_r = sr + (size_t)n * B;
            const double *src_i = si + (size_t)n * B;
            double *even = re + (size_t)(2*n) * K + b0;
            double *odd  = re + (size_t)(2*n+1) * K + b0;
            size_t k = 0;
#if defined(__AVX512F__)
            {
                __m512d two = _mm512_set1_pd(2.0);
                for (; k + 8 <= B; k += 8) {
                    _mm512_storeu_pd(even + k, _mm512_mul_pd(two, _mm512_load_pd(src_r + k)));
                    _mm512_storeu_pd(odd  + k, _mm512_mul_pd(two, _mm512_load_pd(src_i + k)));
                }
            }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
            {
                __m256d two = _mm256_set1_pd(2.0);
                for (; k + 4 <= B; k += 4) {
                    _mm256_storeu_pd(even + k, _mm256_mul_pd(two, _mm256_load_pd(src_r + k)));
                    _mm256_storeu_pd(odd  + k, _mm256_mul_pd(two, _mm256_load_pd(src_i + k)));
                }
            }
#endif
            for (; k < B; k++) {
                even[k] = 2.0 * src_r[k];
                odd[k]  = 2.0 * src_i[k];
            }
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _r2c_destroy(void *data) {
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    if (!d) return;
    STRIDE_ALIGNED_FREE(d->tw_re);
    STRIDE_ALIGNED_FREE(d->tw_im);
    free(d->perm);
    free(d->iperm);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    if (d->inner) stride_plan_destroy(d->inner);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Parameters:
 *   N         - real FFT size (must be even)
 *   K         - batch count
 *   block_K   - block size for cache-friendly execution
 *   inner_plan - N/2-point complex FFT plan with K = block_K
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *stride_r2c_plan(
        int N, size_t K, size_t block_K,
        stride_plan_t *inner_plan)
{
    if (N < 2 || (N & 1)) {
        if (inner_plan) stride_plan_destroy(inner_plan);
        return NULL;
    }

    stride_r2c_data_t *d =
        (stride_r2c_data_t *)calloc(1, sizeof(*d));
    if (!d) { stride_plan_destroy(inner_plan); return NULL; }

    int halfN = N / 2;
    d->N = N;
    d->half_N = halfN;
    d->K = K;
    d->B = block_K;
    d->inner = inner_plan;

    /* Twiddle factors: W_N^k for k=0..N/2-1 */
    d->tw_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)halfN * sizeof(double));
    d->tw_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)halfN * sizeof(double));
    _r2c_init_twiddles(N, d->tw_re, d->tw_im);

    /* Digit-reversal permutation from inner plan's factorization.
     * Override plans (Bluestein/Rader) produce natural-order output,
     * so permutation is identity for those. */
    d->perm  = (int *)malloc((size_t)halfN * sizeof(int));
    d->iperm = (int *)malloc((size_t)halfN * sizeof(int));
    if (inner_plan->num_stages > 0) {
        _r2c_compute_perm(inner_plan->factors, inner_plan->num_stages, halfN,
                          d->perm, d->iperm);
    } else {
        /* Override plan: output is already natural order */
        for (int i = 0; i < halfN; i++)
            d->perm[i] = d->iperm[i] = i;
    }

    /* Scratch: N/2 * block_K */
    size_t scratch_sz = (size_t)halfN * block_K;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_sz * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_sz * sizeof(double));

    /* Build plan shell */
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _r2c_destroy(d); return NULL; }

    plan->N = N;
    plan->K = K;
    plan->num_stages = 0;
    plan->override_fwd     = _r2c_execute_fwd;
    plan->override_bwd     = _r2c_execute_bwd;
    plan->override_destroy = _r2c_destroy;
    plan->override_data    = d;

    return plan;
}


/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE API
 *
 * stride_execute_r2c: explicit 3-pointer (real_in -> complex_out)
 * stride_execute_c2r: explicit 3-pointer (complex_in -> real_out)
 *
 * These copy real_in -> out_re (which must be N*K), then call
 * the in-place override. For zero-copy, use stride_execute_fwd
 * directly with the in-place convention.
 * ═══════════════════════════════════════════════════════════════ */

static inline void stride_execute_r2c(const stride_plan_t *plan,
                                       const double *real_in,
                                       double *out_re, double *out_im) {
    size_t NK = (size_t)plan->N * plan->K;
    memcpy(out_re, real_in, NK * sizeof(double));
    plan->override_fwd(plan->override_data, out_re, out_im);
}

static inline void stride_execute_c2r(const stride_plan_t *plan,
                                       const double *in_re, const double *in_im,
                                       double *real_out) {
    size_t halfN_plus1_K = (size_t)(plan->N / 2 + 1) * plan->K;
    /* Copy complex input to real_out (re) and a temp buffer (im).
     * The backward override reads from (re, im) and writes real to re. */
    memcpy(real_out, in_re, halfN_plus1_K * sizeof(double));
    /* Need writable im — allocate temp if in_im is const */
    double *im_buf = (double *)STRIDE_ALIGNED_ALLOC(64, halfN_plus1_K * sizeof(double));
    memcpy(im_buf, in_im, halfN_plus1_K * sizeof(double));
    plan->override_bwd(plan->override_data, real_out, im_buf);
    STRIDE_ALIGNED_FREE(im_buf);
}


#endif /* STRIDE_R2C_H */
