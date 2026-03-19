/**
 * @file fft_radix8_avx2_tw_packed_il.h
 * @brief DFT-8 AVX2 packed-twiddle interleaved codelet
 *
 * Combines three ideas:
 *   1. Packed twiddle table: all 7 twiddles per k-group stored sequentially
 *      → 7 aligned loads, zero strided access, zero derivation math
 *   2. Interleaved data layout: {re,im,re,im,...}
 *      → half the memory streams vs split re/im
 *   3. genfft DAG butterfly with native FMA intrinsics
 *      → optimal op count from Frigo & Johnson's scheduler
 *
 * Twiddle table format (built by radix8_pack_twiddles):
 *   ptw[group*28 + n*4 ... +3] = {W^(n+1)_re_k, W^(n+1)_im_k, W^(n+1)_re_{k+1}, W^(n+1)_im_{k+1}}
 *   group = k/2, n = 0..6 (7 twiddles), 28 doubles = 224 bytes per group
 *
 * Complex multiply: fmaddsub(x, dup_re(w), mul(flip(x), dup_im(w)))
 *   Single instruction on FMA3: fmaddsub does {a*b-c, a*b+c} per 128-bit lane
 *
 * Register budget (AVX2, 16 YMM):
 *   Phase 1 (tw load+apply): 1 tw(2→dies) + 1 data(2→dies) → xN grows 0→8
 *     Peak: ~4 YMM (tw + rN + current xN being written)
 *     After: 8 YMM (x0..x7) + 1 (cW8) = 9
 *   Phase 2 (butterfly): 8 data + ~5 intermediates + 1 const = ~14
 *   → ZERO SPILLS
 *
 * vs old IL (log3 strided): 7 tw + 8 data + 1 const = 16 (tight) + derivation ops
 * vs split (log3 strided):  14 tw + 16 data + 2 const = 32 → 6+ spills
 */

#ifndef FFT_RADIX8_AVX2_TW_PACKED_IL_H
#define FFT_RADIX8_AVX2_TW_PACKED_IL_H

#include <stddef.h>
#include <immintrin.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLE TABLE PACKING
 *
 * Converts strided split tw_re[(n-1)*K+k], tw_im[(n-1)*K+k]
 * into packed interleaved: ptw[group*28 + n*4 + 0..3]
 *   = {W^(n+1)_re_k, W^(n+1)_im_k, W^(n+1)_re_{k+1}, W^(n+1)_im_{k+1}}
 *
 * Can also build directly from angles (used in plan creation).
 * ═══════════════════════════════════════════════════════════════ */

/** Allocate and build packed twiddle table from scratch.
 *  Returns aligned pointer. Caller must free with aligned_free. */
static inline double *radix8_build_packed_tw(size_t K)
{
    double *ptw;
    size_t nb = K / 2;
    size_t total = nb * 28;  /* 7 twiddles × 4 doubles per group */
#ifdef _WIN32
    ptw = (double *)_aligned_malloc(total * sizeof(double), 64);
#else
    posix_memalign((void **)&ptw, 64, total * sizeof(double));
#endif
    double N = (double)(8 * K);
    for (size_t g = 0; g < nb; g++) {
        size_t k = g * 2;
        for (int n = 1; n <= 7; n++) {
            double a0 = -2.0 * M_PI * (double)(n * k) / N;
            double a1 = -2.0 * M_PI * (double)(n * (k + 1)) / N;
            ptw[g * 28 + (n - 1) * 4 + 0] = cos(a0);
            ptw[g * 28 + (n - 1) * 4 + 1] = sin(a0);
            ptw[g * 28 + (n - 1) * 4 + 2] = cos(a1);
            ptw[g * 28 + (n - 1) * 4 + 3] = sin(a1);
        }
    }
    return ptw;
}

/** Convert existing strided split tables to packed format. */
static inline double *radix8_pack_twiddles(
    const double *tw_re, const double *tw_im, size_t K)
{
    double *ptw;
    size_t nb = K / 2;
    size_t total = nb * 28;
#ifdef _WIN32
    ptw = (double *)_aligned_malloc(total * sizeof(double), 64);
#else
    posix_memalign((void **)&ptw, 64, total * sizeof(double));
#endif
    for (size_t g = 0; g < nb; g++) {
        size_t k = g * 2;
        for (int n = 0; n < 7; n++) {
            ptw[g * 28 + n * 4 + 0] = tw_re[n * K + k];
            ptw[g * 28 + n * 4 + 1] = tw_im[n * K + k];
            ptw[g * 28 + n * 4 + 2] = tw_re[n * K + k + 1];
            ptw[g * 28 + n * 4 + 3] = tw_im[n * K + k + 1];
        }
    }
    return ptw;
}

/* ═══════════════════════════════════════════════════════════════
 * INTERLEAVED COMPLEX MULTIPLY
 * ═══════════════════════════════════════════════════════════════ */

/* Forward: (x_re + j*x_im) * (w_re + j*w_im) */
#define R8P_CMUL(x, w) \
    _mm256_fmaddsub_pd((x), _mm256_movedup_pd(w), \
        _mm256_mul_pd(_mm256_permute_pd((x), 0x5), _mm256_permute_pd((w), 0xF)))

/* Conjugate: (x_re + j*x_im) * (w_re - j*w_im) */
#define R8P_CMULJ(x, w) \
    _mm256_fmsubadd_pd((x), _mm256_movedup_pd(w), \
        _mm256_mul_pd(_mm256_permute_pd((x), 0x5), _mm256_permute_pd((w), 0xF)))

/* ±j multiply for interleaved: VFMAI(b,c) = c + j*b */
#define R8P_ADDI(c, b) _mm256_addsub_pd((c), _mm256_permute_pd((b), 0x5))
/* c - j*b */
#define R8P_SUBI(c, b) _mm256_sub_pd((c), \
    _mm256_addsub_pd(_mm256_setzero_pd(), _mm256_permute_pd((b), 0x5)))

/* ═══════════════════════════════════════════════════════════════
 * FORWARD DIT CODELET — PACKED TWIDDLES + INTERLEAVED DATA
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2,fma")))
static void
radix8_tw_packed_il_fwd_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    const double * __restrict__ ptw,
    size_t K)
{
    const __m256d cW8 = _mm256_set1_pd(0.70710678118654752440);

    for (size_t k = 0; k < K; k += 2) {
        const double *tw = ptw + (k / 2) * 28;

        /* Load twiddles sequentially — 7 aligned loads, all adjacent */
        const __m256d w1 = _mm256_load_pd(tw +  0);  /* W^1 */
        const __m256d w2 = _mm256_load_pd(tw +  4);  /* W^2 */
        const __m256d w3 = _mm256_load_pd(tw +  8);  /* W^3 */
        const __m256d w4 = _mm256_load_pd(tw + 12);  /* W^4 */
        const __m256d w5 = _mm256_load_pd(tw + 16);  /* W^5 */
        const __m256d w6 = _mm256_load_pd(tw + 20);  /* W^6 */
        const __m256d w7 = _mm256_load_pd(tw + 24);  /* W^7 */

        /* Load data + apply twiddles */
        const __m256d x0 = _mm256_load_pd(&in[2 * (0 * K + k)]);
        const __m256d x1 = R8P_CMUL(_mm256_load_pd(&in[2 * (1 * K + k)]), w1);
        const __m256d x2 = R8P_CMUL(_mm256_load_pd(&in[2 * (2 * K + k)]), w2);
        const __m256d x3 = R8P_CMUL(_mm256_load_pd(&in[2 * (3 * K + k)]), w3);
        const __m256d x4 = R8P_CMUL(_mm256_load_pd(&in[2 * (4 * K + k)]), w4);
        const __m256d x5 = R8P_CMUL(_mm256_load_pd(&in[2 * (5 * K + k)]), w5);
        const __m256d x6 = R8P_CMUL(_mm256_load_pd(&in[2 * (6 * K + k)]), w6);
        const __m256d x7 = R8P_CMUL(_mm256_load_pd(&in[2 * (7 * K + k)]), w7);

        /* ── genfft DFT-8 DAG (forward) ── */
        const __m256d T3 = _mm256_sub_pd(x0, x4);
        const __m256d Tj = _mm256_add_pd(x0, x4);
        const __m256d Te = _mm256_sub_pd(x2, x6);
        const __m256d Tk = _mm256_add_pd(x2, x6);

        const __m256d T6 = _mm256_sub_pd(x1, x5);
        const __m256d T9 = _mm256_sub_pd(x7, x3);
        const __m256d Ta = _mm256_add_pd(T6, T9);
        const __m256d Tn = _mm256_add_pd(x7, x3);
        const __m256d Tf = _mm256_sub_pd(T9, T6);
        const __m256d Tm = _mm256_add_pd(x1, x5);

        const __m256d Tb = _mm256_fmadd_pd(cW8, Ta, T3);
        const __m256d Tg = _mm256_fnmadd_pd(cW8, Tf, Te);
        const __m256d y1 = R8P_SUBI(Tb, Tg);
        const __m256d y7 = R8P_ADDI(Tb, Tg);

        const __m256d Tp = _mm256_sub_pd(Tj, Tk);
        const __m256d Tq = _mm256_sub_pd(Tn, Tm);
        const __m256d y6 = R8P_SUBI(Tp, Tq);
        const __m256d y2 = R8P_ADDI(Tp, Tq);

        const __m256d Th = _mm256_fnmadd_pd(cW8, Ta, T3);
        const __m256d Ti = _mm256_fmadd_pd(cW8, Tf, Te);
        const __m256d y5 = R8P_SUBI(Th, Ti);
        const __m256d y3 = R8P_ADDI(Th, Ti);

        const __m256d Tl = _mm256_add_pd(Tj, Tk);
        const __m256d To = _mm256_add_pd(Tm, Tn);
        const __m256d y4 = _mm256_sub_pd(Tl, To);
        const __m256d y0 = _mm256_add_pd(Tl, To);

        _mm256_store_pd(&out[2 * (0 * K + k)], y0);
        _mm256_store_pd(&out[2 * (1 * K + k)], y1);
        _mm256_store_pd(&out[2 * (2 * K + k)], y2);
        _mm256_store_pd(&out[2 * (3 * K + k)], y3);
        _mm256_store_pd(&out[2 * (4 * K + k)], y4);
        _mm256_store_pd(&out[2 * (5 * K + k)], y5);
        _mm256_store_pd(&out[2 * (6 * K + k)], y6);
        _mm256_store_pd(&out[2 * (7 * K + k)], y7);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * BACKWARD DIT CODELET — CONJUGATE TWIDDLES
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2,fma")))
static void
radix8_tw_packed_il_bwd_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    const double * __restrict__ ptw,
    size_t K)
{
    const __m256d cW8 = _mm256_set1_pd(0.70710678118654752440);

    for (size_t k = 0; k < K; k += 2) {
        const double *tw = ptw + (k / 2) * 28;

        const __m256d w1 = _mm256_load_pd(tw +  0);
        const __m256d w2 = _mm256_load_pd(tw +  4);
        const __m256d w3 = _mm256_load_pd(tw +  8);
        const __m256d w4 = _mm256_load_pd(tw + 12);
        const __m256d w5 = _mm256_load_pd(tw + 16);
        const __m256d w6 = _mm256_load_pd(tw + 20);
        const __m256d w7 = _mm256_load_pd(tw + 24);

        /* Conjugate multiply for backward */
        const __m256d x0 = _mm256_load_pd(&in[2 * (0 * K + k)]);
        const __m256d x1 = R8P_CMULJ(_mm256_load_pd(&in[2 * (1 * K + k)]), w1);
        const __m256d x2 = R8P_CMULJ(_mm256_load_pd(&in[2 * (2 * K + k)]), w2);
        const __m256d x3 = R8P_CMULJ(_mm256_load_pd(&in[2 * (3 * K + k)]), w3);
        const __m256d x4 = R8P_CMULJ(_mm256_load_pd(&in[2 * (4 * K + k)]), w4);
        const __m256d x5 = R8P_CMULJ(_mm256_load_pd(&in[2 * (5 * K + k)]), w5);
        const __m256d x6 = R8P_CMULJ(_mm256_load_pd(&in[2 * (6 * K + k)]), w6);
        const __m256d x7 = R8P_CMULJ(_mm256_load_pd(&in[2 * (7 * K + k)]), w7);

        /* ── genfft DFT-8 DAG (backward) ── */
        const __m256d T3 = _mm256_sub_pd(x0, x4);
        const __m256d Tj = _mm256_add_pd(x0, x4);
        const __m256d Te = _mm256_sub_pd(x2, x6);
        const __m256d Tk = _mm256_add_pd(x2, x6);

        const __m256d T6 = _mm256_sub_pd(x1, x5);
        const __m256d T9 = _mm256_sub_pd(x7, x3);
        const __m256d Ta = _mm256_add_pd(T6, T9);
        const __m256d Tn = _mm256_add_pd(x7, x3);
        const __m256d Tf = _mm256_sub_pd(T6, T9);
        const __m256d Tm = _mm256_add_pd(x1, x5);

        const __m256d Tb = _mm256_fnmadd_pd(cW8, Ta, T3);
        const __m256d Tg = _mm256_fnmadd_pd(cW8, Tf, Te);
        const __m256d y3 = R8P_SUBI(Tb, Tg);
        const __m256d y5 = R8P_ADDI(Tb, Tg);

        const __m256d Tp = _mm256_add_pd(Tj, Tk);
        const __m256d Tq = _mm256_add_pd(Tm, Tn);
        const __m256d y4 = _mm256_sub_pd(Tp, Tq);
        const __m256d y0 = _mm256_add_pd(Tp, Tq);

        const __m256d Th = _mm256_fmadd_pd(cW8, Ta, T3);
        const __m256d Ti = _mm256_fmadd_pd(cW8, Tf, Te);
        const __m256d y1 = R8P_ADDI(Th, Ti);
        const __m256d y7 = R8P_SUBI(Th, Ti);

        const __m256d Tl = _mm256_sub_pd(Tj, Tk);
        const __m256d To = _mm256_sub_pd(Tm, Tn);
        const __m256d y6 = R8P_SUBI(Tl, To);
        const __m256d y2 = R8P_ADDI(Tl, To);

        _mm256_store_pd(&out[2 * (0 * K + k)], y0);
        _mm256_store_pd(&out[2 * (1 * K + k)], y1);
        _mm256_store_pd(&out[2 * (2 * K + k)], y2);
        _mm256_store_pd(&out[2 * (3 * K + k)], y3);
        _mm256_store_pd(&out[2 * (4 * K + k)], y4);
        _mm256_store_pd(&out[2 * (5 * K + k)], y5);
        _mm256_store_pd(&out[2 * (6 * K + k)], y6);
        _mm256_store_pd(&out[2 * (7 * K + k)], y7);
    }
}

#undef R8P_CMUL
#undef R8P_CMULJ
#undef R8P_ADDI
#undef R8P_SUBI

#endif /* FFT_RADIX8_AVX2_TW_PACKED_IL_H */
