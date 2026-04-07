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

typedef struct
{
    int N;      /* original real-FFT size (must be even) */
    int half_N; /* N/2 (inner FFT size) */
    size_t K;   /* batch count */
    size_t B;   /* block size for cache-friendly execution */

    double *tw_re; /* N/2 twiddle factors: W_N^k = cos(-2*pi*k/N) */
    double *tw_im; /* N/2 twiddle factors: sin(-2*pi*k/N) */

    int *perm;  /* N/2 digit-reversal permutation: natural → DIT output order */
    int *iperm; /* N/2 inverse permutation: DIT output → natural order */

    double *scratch_re; /* N/2 * B scratch for inner FFT */
    double *scratch_im;
    double *c2r_im_buf; /* (N/2+1) * K pre-allocated temp for stride_execute_c2r */

    stride_plan_t *inner; /* N/2-point complex FFT plan with K = B */
} stride_r2c_data_t;

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLE PRECOMPUTATION
 * ═══════════════════════════════════════════════════════════════ */

/* Compute mixed-radix digit-reversal permutation.
 * For DIT forward: output[perm[n]] = DFT[n].
 * So to read DFT[n] from the output, access output[perm[n]].
 * iperm is the inverse: output[k] = DFT[iperm[k]]. */
static void _r2c_compute_perm(const int *factors, int nf, int N,
                              int *perm, int *iperm)
{
    for (int n = 0; n < N; n++)
    {
        int idx = n, rev = 0, radix_product = 1;
        for (int s = 0; s < nf; s++)
        {
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

static void _r2c_init_twiddles(int N, double *tw_re, double *tw_im)
{
    int half_N = N / 2;
    for (int k = 0; k < half_N; k++)
    {
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
    const double *__restrict__ z_re,
    const double *__restrict__ z_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    const int *__restrict__ iperm,
    const int *__restrict__ perm,
    int half_N, size_t K, size_t B, size_t b0)
{
    /* DC (f=0) and Nyquist (f=N/2).
     * perm[0]=0 always (digit-reversal of 0 is 0), so Z[0] is at scratch[0]. */
    {
        size_t nyq_off = (size_t)half_N * K + b0;
        size_t k = 0;
#if defined(__AVX512F__)
        for (; k + 8 <= B; k += 8)
        {
            __m512d zr = _mm512_load_pd(z_re + k);
            __m512d zi = _mm512_load_pd(z_im + k);
            _mm512_storeu_pd(out_re + b0 + k, _mm512_add_pd(zr, zi));
            _mm512_storeu_pd(out_im + b0 + k, _mm512_setzero_pd());
            _mm512_storeu_pd(out_re + nyq_off + k, _mm512_sub_pd(zr, zi));
            _mm512_storeu_pd(out_im + nyq_off + k, _mm512_setzero_pd());
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        for (; k + 4 <= B; k += 4)
        {
            __m256d zr = _mm256_load_pd(z_re + k);
            __m256d zi = _mm256_load_pd(z_im + k);
            _mm256_storeu_pd(out_re + b0 + k, _mm256_add_pd(zr, zi));
            _mm256_storeu_pd(out_im + b0 + k, _mm256_setzero_pd());
            _mm256_storeu_pd(out_re + nyq_off + k, _mm256_sub_pd(zr, zi));
            _mm256_storeu_pd(out_im + nyq_off + k, _mm256_setzero_pd());
        }
#endif
        for (; k < B; k++)
        {
            out_re[b0 + k] = z_re[k] + z_im[k];
            out_im[b0 + k] = 0.0;
            out_re[nyq_off + k] = z_re[k] - z_im[k];
            out_im[nyq_off + k] = 0.0;
        }
    }

    /* Iterate sequentially through scratch for cache-friendly primary reads.
     * Process pairs: when iperm[p] = f and f <= mirror, compute both X[f]
     * and X[mirror] from the same Z loads. Skip when f > mirror (already done).
     * Result: sequential primary reads, each Z element loaded once,
     * mirror twiddle hoisted outside k-loop. */
    for (int p = 1; p < half_N; p++) {
        int f = iperm[p];
        int mirror = half_N - f;
        if (f > mirror) continue;   /* already processed as partner */

        size_t z_f    = (size_t)p * B;                 /* sequential read */
        size_t z_m    = (size_t)perm[mirror] * B;      /* scattered read */
        size_t fo_off = (size_t)f * K + b0;
        size_t mo_off = (size_t)mirror * K + b0;

        const double wr = tw_re[f], wi = tw_im[f];
        int do_mirror = (f != mirror);

        /* Hoist mirror twiddle broadcasts outside k-loop */
        double wrm = 0, wim = 0;
        if (do_mirror) { wrm = tw_re[mirror]; wim = tw_im[mirror]; }

        size_t k = 0;
#if defined(__AVX512F__)
        {
            __m512d half_v = _mm512_set1_pd(0.5);
            __m512d vwr    = _mm512_set1_pd(wr);
            __m512d vwi    = _mm512_set1_pd(wi);
            __m512d vwrm, vwim;
            if (do_mirror) { vwrm = _mm512_set1_pd(wrm); vwim = _mm512_set1_pd(wim); }

            for (; k + 8 <= B; k += 8) {
                __m512d Zfr = _mm512_load_pd(z_re + z_f + k);
                __m512d Zfi = _mm512_load_pd(z_im + z_f + k);
                __m512d Zmr = _mm512_load_pd(z_re + z_m + k);
                __m512d Zmi = _mm512_load_pd(z_im + z_m + k);

                __m512d Er = _mm512_mul_pd(_mm512_add_pd(Zfr, Zmr), half_v);
                __m512d Ei = _mm512_mul_pd(_mm512_sub_pd(Zfi, Zmi), half_v);
                __m512d Or = _mm512_mul_pd(_mm512_sub_pd(Zfr, Zmr), half_v);
                __m512d Oi = _mm512_mul_pd(_mm512_add_pd(Zfi, Zmi), half_v);

                __m512d niOr  = Oi;
                __m512d neg_Or = _mm512_sub_pd(_mm512_setzero_pd(), Or);

                __m512d Tr = _mm512_fmsub_pd(vwr, niOr, _mm512_mul_pd(vwi, neg_Or));
                __m512d Ti = _mm512_fmadd_pd(vwr, neg_Or, _mm512_mul_pd(vwi, niOr));

                _mm512_storeu_pd(out_re + fo_off + k, _mm512_add_pd(Er, Tr));
                _mm512_storeu_pd(out_im + fo_off + k, _mm512_add_pd(Ei, Ti));

                if (do_mirror) {
                    __m512d Emi   = _mm512_sub_pd(_mm512_setzero_pd(), Ei);
                    __m512d niOmr = Oi;
                    __m512d niOmi = _mm512_sub_pd(_mm512_setzero_pd(), neg_Or); /* Or */

                    __m512d Tmr = _mm512_fmsub_pd(vwrm, niOmr, _mm512_mul_pd(vwim, niOmi));
                    __m512d Tmi = _mm512_fmadd_pd(vwrm, niOmi, _mm512_mul_pd(vwim, niOmr));

                    _mm512_storeu_pd(out_re + mo_off + k, _mm512_add_pd(Er, Tmr));
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
            __m256d vwrm, vwim;
            if (do_mirror) { vwrm = _mm256_set1_pd(wrm); vwim = _mm256_set1_pd(wim); }

            for (; k + 4 <= B; k += 4) {
                __m256d Zfr = _mm256_load_pd(z_re + z_f + k);
                __m256d Zfi = _mm256_load_pd(z_im + z_f + k);
                __m256d Zmr = _mm256_load_pd(z_re + z_m + k);
                __m256d Zmi = _mm256_load_pd(z_im + z_m + k);

                __m256d Er = _mm256_mul_pd(_mm256_add_pd(Zfr, Zmr), half_v);
                __m256d Ei = _mm256_mul_pd(_mm256_sub_pd(Zfi, Zmi), half_v);
                __m256d Or = _mm256_mul_pd(_mm256_sub_pd(Zfr, Zmr), half_v);
                __m256d Oi = _mm256_mul_pd(_mm256_add_pd(Zfi, Zmi), half_v);

                __m256d niOr = Oi;
                __m256d niOi = _mm256_xor_pd(Or, sign);

                __m256d Tr = _mm256_fmsub_pd(vwr, niOr, _mm256_mul_pd(vwi, niOi));
                __m256d Ti = _mm256_fmadd_pd(vwr, niOi, _mm256_mul_pd(vwi, niOr));

                _mm256_storeu_pd(out_re + fo_off + k, _mm256_add_pd(Er, Tr));
                _mm256_storeu_pd(out_im + fo_off + k, _mm256_add_pd(Ei, Ti));

                if (do_mirror) {
                    __m256d Emi   = _mm256_xor_pd(Ei, sign);
                    __m256d Omr   = _mm256_xor_pd(Or, sign);
                    __m256d niOmr = Oi;
                    __m256d niOmi = _mm256_xor_pd(Omr, sign); /* Or */

                    __m256d Tmr = _mm256_fmsub_pd(vwrm, niOmr, _mm256_mul_pd(vwim, niOmi));
                    __m256d Tmi = _mm256_fmadd_pd(vwrm, niOmi, _mm256_mul_pd(vwim, niOmr));

                    _mm256_storeu_pd(out_re + mo_off + k, _mm256_add_pd(Er, Tmr));
                    _mm256_storeu_pd(out_im + mo_off + k, _mm256_add_pd(Emi, Tmi));
                }
            }
        }
#endif
        for (; k < B; k++) {
            double Zfr = z_re[z_f+k], Zfi = z_im[z_f+k];
            double Zmr = z_re[z_m+k], Zmi = z_im[z_m+k];
            double Er = (Zfr + Zmr) * 0.5, Ei = (Zfi - Zmi) * 0.5;
            double Or = (Zfr - Zmr) * 0.5, Oi = (Zfi + Zmi) * 0.5;
            double niOr = Oi, niOi = -Or;
            double Tr = wr*niOr - wi*niOi, Ti = wr*niOi + wi*niOr;
            out_re[fo_off+k] = Er + Tr;
            out_im[fo_off+k] = Ei + Ti;
            if (do_mirror) {
                double Emr = Er, Emi = -Ei, Omr = -Or, Omi = Oi;
                double niOmr = Omi, niOmi = -Omr;
                double Tmr = wrm*niOmr - wim*niOmi, Tmi = wrm*niOmi + wim*niOmr;
                out_re[mo_off+k] = Emr + Tmr;
                out_im[mo_off+k] = Emi + Tmi;
            }
        }
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
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ z_re,
    double *__restrict__ z_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    const int *__restrict__ perm,
    int half_N, size_t K, size_t B, size_t b0)
{
    /* DC: Z[0] written to permuted position perm[0] */
    {
        size_t z0_out = (size_t)perm[0] * B;
        size_t nyq = (size_t)half_N * K + b0;
        size_t k = 0;
#if defined(__AVX512F__)
        for (; k + 8 <= B; k += 8)
        {
            __m512d x0 = _mm512_loadu_pd(in_re + b0 + k);
            __m512d xn = _mm512_loadu_pd(in_re + nyq + k);
            _mm512_store_pd(z_re + z0_out + k, _mm512_add_pd(x0, xn));
            _mm512_store_pd(z_im + z0_out + k, _mm512_sub_pd(x0, xn));
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        for (; k + 4 <= B; k += 4)
        {
            __m256d x0 = _mm256_loadu_pd(in_re + b0 + k);
            __m256d xn = _mm256_loadu_pd(in_re + nyq + k);
            _mm256_store_pd(z_re + z0_out + k, _mm256_add_pd(x0, xn));
            _mm256_store_pd(z_im + z0_out + k, _mm256_sub_pd(x0, xn));
        }
#endif
        for (; k < B; k++)
        {
            double x0r = in_re[b0 + k];
            double xnr = in_re[nyq + k];
            z_re[z0_out + k] = x0r + xnr;
            z_im[z0_out + k] = x0r - xnr;
        }
    }

    /* Butterfly pairs — process (f, mirror) together, write both Z values.
     * Each X element is loaded once. Mirror twiddles hoisted before k-loop. */
    for (int f = 1; f < half_N; f++)
    {
        int mirror = half_N - f;
        if (f > mirror)
            break;

        size_t fi = (size_t)f * K + b0;
        size_t mi = (size_t)mirror * K + b0;
        size_t fo = (size_t)perm[f] * B;
        size_t mo = (size_t)perm[mirror] * B;

        double cwr = tw_re[f], cwi = -tw_im[f];
        int do_mirror = (f != mirror);

        /* Hoist mirror twiddle broadcasts outside k-loop */
        double cwr_m = 0, cwi_m = 0;
        if (do_mirror)
        {
            cwr_m = tw_re[mirror];
            cwi_m = -tw_im[mirror];
        }

        size_t k = 0;
#if defined(__AVX512F__)
        {
            __m512d half_v = _mm512_set1_pd(0.5);
            __m512d sign = _mm512_set1_pd(-0.0);
            __m512d vcwr = _mm512_set1_pd(cwr);
            __m512d vcwi = _mm512_set1_pd(cwi);
            __m512d vcwr_m, vcwi_m;
            if (do_mirror)
            {
                vcwr_m = _mm512_set1_pd(cwr_m);
                vcwi_m = _mm512_set1_pd(cwi_m);
            }
            for (; k + 8 <= B; k += 8)
            {
                __m512d Xfr = _mm512_loadu_pd(in_re + fi + k);
                __m512d Xfi = _mm512_loadu_pd(in_im + fi + k);
                __m512d Xmr = _mm512_loadu_pd(in_re + mi + k);
                __m512d Xmi = _mm512_loadu_pd(in_im + mi + k);

                __m512d Er = _mm512_mul_pd(_mm512_add_pd(Xfr, Xmr), half_v);
                __m512d Ei = _mm512_mul_pd(_mm512_sub_pd(Xfi, Xmi), half_v);
                __m512d Dr = _mm512_mul_pd(_mm512_sub_pd(Xfr, Xmr), half_v);
                __m512d Di = _mm512_mul_pd(_mm512_add_pd(Xfi, Xmi), half_v);

                __m512d Xor_f = _mm512_fmsub_pd(vcwr, Dr, _mm512_mul_pd(vcwi, Di));
                __m512d Xoi_f = _mm512_fmadd_pd(vcwr, Di, _mm512_mul_pd(vcwi, Dr));

                _mm512_store_pd(z_re + fo + k, _mm512_sub_pd(Er, Xoi_f));
                _mm512_store_pd(z_im + fo + k, _mm512_add_pd(Ei, Xor_f));

                if (do_mirror)
                {
                    __m512d neg_Dr = _mm512_sub_pd(_mm512_setzero_pd(), Dr);
                    __m512d Xor_m = _mm512_fmsub_pd(vcwr_m, neg_Dr, _mm512_mul_pd(vcwi_m, Di));
                    __m512d Xoi_m = _mm512_fmadd_pd(vcwr_m, Di, _mm512_mul_pd(vcwi_m, neg_Dr));
                    __m512d neg_Ei = _mm512_sub_pd(_mm512_setzero_pd(), Ei);
                    _mm512_store_pd(z_re + mo + k, _mm512_sub_pd(Er, Xoi_m));
                    _mm512_store_pd(z_im + mo + k, _mm512_add_pd(neg_Ei, Xor_m));
                }
            }
        }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
        {
            __m256d half_v = _mm256_set1_pd(0.5);
            __m256d sign = _mm256_set1_pd(-0.0);
            __m256d vcwr = _mm256_set1_pd(cwr);
            __m256d vcwi = _mm256_set1_pd(cwi);
            __m256d vcwr_m, vcwi_m;
            if (do_mirror)
            {
                vcwr_m = _mm256_set1_pd(cwr_m);
                vcwi_m = _mm256_set1_pd(cwi_m);
            }
            for (; k + 4 <= B; k += 4)
            {
                __m256d Xfr = _mm256_loadu_pd(in_re + fi + k);
                __m256d Xfi = _mm256_loadu_pd(in_im + fi + k);
                __m256d Xmr = _mm256_loadu_pd(in_re + mi + k);
                __m256d Xmi = _mm256_loadu_pd(in_im + mi + k);

                __m256d Er = _mm256_mul_pd(_mm256_add_pd(Xfr, Xmr), half_v);
                __m256d Ei = _mm256_mul_pd(_mm256_sub_pd(Xfi, Xmi), half_v);
                __m256d Dr = _mm256_mul_pd(_mm256_sub_pd(Xfr, Xmr), half_v);
                __m256d Di = _mm256_mul_pd(_mm256_add_pd(Xfi, Xmi), half_v);

                __m256d Xor_f = _mm256_fmsub_pd(vcwr, Dr, _mm256_mul_pd(vcwi, Di));
                __m256d Xoi_f = _mm256_fmadd_pd(vcwr, Di, _mm256_mul_pd(vcwi, Dr));

                _mm256_store_pd(z_re + fo + k, _mm256_sub_pd(Er, Xoi_f));
                _mm256_store_pd(z_im + fo + k, _mm256_add_pd(Ei, Xor_f));

                if (do_mirror)
                {
                    __m256d neg_Dr = _mm256_xor_pd(Dr, sign);
                    __m256d Xor_m = _mm256_fmsub_pd(vcwr_m, neg_Dr, _mm256_mul_pd(vcwi_m, Di));
                    __m256d Xoi_m = _mm256_fmadd_pd(vcwr_m, Di, _mm256_mul_pd(vcwi_m, neg_Dr));
                    __m256d neg_Ei = _mm256_xor_pd(Ei, sign);
                    _mm256_store_pd(z_re + mo + k, _mm256_sub_pd(Er, Xoi_m));
                    _mm256_store_pd(z_im + mo + k, _mm256_add_pd(neg_Ei, Xor_m));
                }
            }
        }
#endif
        for (; k < B; k++)
        {
            double Xfr = in_re[fi + k], Xfi = in_im[fi + k];
            double Xmr = in_re[mi + k], Xmi = in_im[mi + k];

            double Er = (Xfr + Xmr) * 0.5;
            double Ei = (Xfi - Xmi) * 0.5;
            double Dr = (Xfr - Xmr) * 0.5;
            double Di = (Xfi + Xmi) * 0.5;

            double Xor_f = cwr * Dr - cwi * Di;
            double Xoi_f = cwr * Di + cwi * Dr;
            z_re[fo + k] = Er - Xoi_f;
            z_im[fo + k] = Ei + Xor_f;

            if (do_mirror)
            {
                double Xor_m = cwr_m * (-Dr) - cwi_m * Di;
                double Xoi_m = cwr_m * Di + cwi_m * (-Dr);
                z_re[mo + k] = Er - Xoi_m;
                z_im[mo + k] = -Ei + Xor_m;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- FORWARD R2C (block-walk)
 * ═══════════════════════════════════════════════════════════════ */

/* ── Fused first stage: read from input at stride 2K, write to scratch at stride B.
 *    Stage 0 is ALWAYS twiddle-free (cf0=1, needs_tw=0 for all groups).
 *    So all groups use n1_fwd with is=input_leg_stride, os=scratch_leg_stride.
 *    Eliminates the entire O(halfN*B) pack pass. ── */
static inline void _r2c_fused_first_stage(
        const stride_plan_t *inner, double *re,
        double *sr, double *si,
        size_t K, size_t B, size_t b0)
{
    const stride_stage_t *st = &inner->stages[0];
    const int ngroups = st->num_groups;
    const size_t scratch_leg_stride = st->stride;        /* distance between legs in scratch */
    const size_t elem_per_leg = scratch_leg_stride / B;   /* element spacing per leg */
    const size_t input_leg_stride = elem_per_leg * 2 * K; /* distance between legs in input */

    for (int g = 0; g < ngroups; g++) {
        size_t scratch_base = st->group_base[g];
        size_t first_elem = scratch_base / B;
        size_t in_re_off = first_elem * 2 * K + b0;

        st->n1_fwd(re + in_re_off, re + K + in_re_off,
                   sr + scratch_base, si + scratch_base,
                   input_leg_stride, scratch_leg_stride, B);
    }
}

/* Remaining stages use _stride_execute_fwd_slice_from(plan, sr, si, B, B, 1)
 * defined in executor.h — no duplicated executor code needed. */

static void _r2c_execute_fwd(void *data, double *re, double *im)
{
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    const int N = d->N, halfN = d->half_N;
    const size_t K = d->K, B = d->B;
    double *sr = d->scratch_re, *si = d->scratch_im;

    for (size_t b0 = 0; b0 < K; b0 += B)
    {
        /* 1. Fused first stage: read from input at stride 2K,
         *    twiddle + butterfly, write to scratch at stride B.
         *    Eliminates the O(halfN*B) pack pass entirely. */
        if (d->inner->num_stages > 0 && d->inner->stages[0].n1_fwd) {
            _r2c_fused_first_stage(d->inner, re, sr, si, K, B, b0);
            /* Run stages 1+ on dense scratch via the real executor */
            _stride_execute_fwd_slice_from(d->inner, sr, si, B, B, 1);
        } else {
            /* Fallback: explicit pack + full inner FFT (for plans without t1_oop) */
            for (int n = 0; n < halfN; n++) {
                const double *even = re + (size_t)(2 * n) * K + b0;
                const double *odd  = re + (size_t)(2 * n + 1) * K + b0;
                double *dst_r = sr + (size_t)n * B;
                double *dst_i = si + (size_t)n * B;
                size_t k = 0;
#if defined(__AVX512F__)
                for (; k + 8 <= B; k += 8) {
                    _mm512_store_pd(dst_r + k, _mm512_loadu_pd(even + k));
                    _mm512_store_pd(dst_i + k, _mm512_loadu_pd(odd + k));
                }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
                for (; k + 4 <= B; k += 4) {
                    _mm256_store_pd(dst_r + k, _mm256_loadu_pd(even + k));
                    _mm256_store_pd(dst_i + k, _mm256_loadu_pd(odd + k));
                }
#endif
                for (; k < B; k++) { dst_r[k] = even[k]; dst_i[k] = odd[k]; }
            }
            stride_execute_fwd(d->inner, sr, si);
        }

        /* 3. Post-process with permuted indices */
        _r2c_postprocess(sr, si, re, im, d->tw_re, d->tw_im, d->iperm, d->perm,
                         halfN, K, B, b0);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTE -- BACKWARD C2R (block-walk)
 *
 * Unnormalized: output = N * original_input.
 * Caller divides by N to normalize (consistent with complex bwd).
 * ═══════════════════════════════════════════════════════════════ */

/* ── Fused last stage (backward): DIF butterfly + ×2 scale + strided write.
 *    Stage 0 is twiddle-free and is the LAST stage in DIF order.
 *    n1_scaled_bwd reads from scratch at stride B, writes to output at stride 2K
 *    with output *= 2.0. Eliminates the O(halfN*B) unpack pass. ── */
static inline void _r2c_fused_last_stage(
        const stride_plan_t *inner, double *re,
        double *sr, double *si,
        size_t K, size_t B, size_t b0)
{
    const stride_stage_t *st = &inner->stages[0];
    const int ngroups = st->num_groups;
    const size_t scratch_leg_stride = st->stride;
    const size_t elem_per_leg = scratch_leg_stride / B;
    const size_t output_leg_stride = elem_per_leg * 2 * K;

    for (int g = 0; g < ngroups; g++) {
        size_t scratch_base = st->group_base[g];
        size_t first_elem = scratch_base / B;
        size_t out_off = first_elem * 2 * K + b0;

        st->n1_scaled_bwd(sr + scratch_base, si + scratch_base,
                          re + out_off, re + K + out_off,
                          scratch_leg_stride, output_leg_stride, B, 2.0);
    }
}

static void _r2c_execute_bwd(void *data, double *re, double *im)
{
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    const int N = d->N, halfN = d->half_N;
    const size_t K = d->K, B = d->B;
    double *sr = d->scratch_re, *si = d->scratch_im;

    for (size_t b0 = 0; b0 < K; b0 += B)
    {
        /* 1. Pre-process: reconstruct Z and write to permuted positions
         *    (DIF bwd expects digit-reversed input). No separate permutation pass. */
        _r2c_preprocess(re, im, sr, si, d->tw_re, d->tw_im, d->perm,
                        halfN, K, B, b0);

        /* 2+3. Fused IFFT + unpack: run stages num_stages-1..1 on scratch,
         *       then stage 0 writes ×2 scaled output directly at stride 2K. */
        if (d->inner->num_stages > 0 && d->inner->stages[0].n1_scaled_bwd) {
            _stride_execute_bwd_slice_until(d->inner, sr, si, B, B, 1);
            _r2c_fused_last_stage(d->inner, re, sr, si, K, B, b0);
        } else {
            /* Fallback: full IFFT + separate unpack */
            stride_execute_bwd(d->inner, sr, si);
            for (int n = 0; n < halfN; n++) {
                const double *src_r = sr + (size_t)n * B;
                const double *src_i = si + (size_t)n * B;
                double *even = re + (size_t)(2 * n) * K + b0;
                double *odd  = re + (size_t)(2 * n + 1) * K + b0;
                size_t k = 0;
#if defined(__AVX512F__)
                {
                    __m512d two = _mm512_set1_pd(2.0);
                    for (; k + 8 <= B; k += 8) {
                        _mm512_storeu_pd(even + k, _mm512_mul_pd(two, _mm512_load_pd(src_r + k)));
                        _mm512_storeu_pd(odd + k, _mm512_mul_pd(two, _mm512_load_pd(src_i + k)));
                    }
                }
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
                {
                    __m256d two = _mm256_set1_pd(2.0);
                    for (; k + 4 <= B; k += 4) {
                        _mm256_storeu_pd(even + k, _mm256_mul_pd(two, _mm256_load_pd(src_r + k)));
                        _mm256_storeu_pd(odd + k, _mm256_mul_pd(two, _mm256_load_pd(src_i + k)));
                    }
                }
#endif
                for (; k < B; k++) { even[k] = 2.0 * src_r[k]; odd[k] = 2.0 * src_i[k]; }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _r2c_destroy(void *data)
{
    stride_r2c_data_t *d = (stride_r2c_data_t *)data;
    if (!d)
        return;
    STRIDE_ALIGNED_FREE(d->tw_re);
    STRIDE_ALIGNED_FREE(d->tw_im);
    free(d->perm);
    free(d->iperm);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    STRIDE_ALIGNED_FREE(d->c2r_im_buf);
    if (d->inner)
        stride_plan_destroy(d->inner);
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
    if (N < 2 || (N & 1))
    {
        if (inner_plan)
            stride_plan_destroy(inner_plan);
        return NULL;
    }

    stride_r2c_data_t *d =
        (stride_r2c_data_t *)calloc(1, sizeof(*d));
    if (!d)
    {
        stride_plan_destroy(inner_plan);
        return NULL;
    }

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
    d->perm = (int *)malloc((size_t)halfN * sizeof(int));
    d->iperm = (int *)malloc((size_t)halfN * sizeof(int));
    if (inner_plan->num_stages > 0)
    {
        _r2c_compute_perm(inner_plan->factors, inner_plan->num_stages, halfN,
                          d->perm, d->iperm);
    }
    else
    {
        /* Override plan: output is already natural order */
        for (int i = 0; i < halfN; i++)
            d->perm[i] = d->iperm[i] = i;
    }

    /* Scratch: N/2 * block_K */
    size_t scratch_sz = (size_t)halfN * block_K;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_sz * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, scratch_sz * sizeof(double));

    /* Pre-allocated im buffer for stride_execute_c2r (avoids malloc per call) */
    d->c2r_im_buf = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)(halfN + 1) * K * sizeof(double));

    /* Build plan shell */
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan)
    {
        _r2c_destroy(d);
        return NULL;
    }

    plan->N = N;
    plan->K = K;
    plan->num_stages = 0;
    plan->override_fwd = _r2c_execute_fwd;
    plan->override_bwd = _r2c_execute_bwd;
    plan->override_destroy = _r2c_destroy;
    plan->override_data = d;

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
                                      double *out_re, double *out_im)
{
    size_t NK = (size_t)plan->N * plan->K;
    memcpy(out_re, real_in, NK * sizeof(double));
    plan->override_fwd(plan->override_data, out_re, out_im);
}

static inline void stride_execute_c2r(const stride_plan_t *plan,
                                      const double *in_re, const double *in_im,
                                      double *real_out)
{
    stride_r2c_data_t *d = (stride_r2c_data_t *)plan->override_data;
    size_t halfN_plus1_K = (size_t)(plan->N / 2 + 1) * plan->K;
    /* Copy complex input to real_out (re) and pre-allocated im buffer.
     * No malloc/free per call — the im buffer lives in the plan. */
    memcpy(real_out, in_re, halfN_plus1_K * sizeof(double));
    memcpy(d->c2r_im_buf, in_im, halfN_plus1_K * sizeof(double));
    plan->override_bwd(plan->override_data, real_out, d->c2r_im_buf);
}

#endif /* STRIDE_R2C_H */
