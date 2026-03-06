/**
 * @file fft_radix32_avx512_tw_pack_walk.h
 * @brief DFT-32 AVX-512 packed data + walked twiddles (zero tw table)
 *
 * ═══════════════════════════════════════════════════════════════════
 * ALGORITHM
 * ═══════════════════════════════════════════════════════════════════
 *
 * Data must be in packed layout (block*32*8 contiguous).
 * Twiddles are generated on-the-fly from 5 walking base accumulators:
 *   b[0]=W^1, b[1]=W^2, b[2]=W^4, b[3]=W^8, b[4]=W^16
 *
 * Per block:
 *   1. Derive all 31 twiddles from 5 bases via binary tree:
 *      W^3=W^1*W^2, W^5=W^1*W^4, W^6=W^2*W^4, W^7=W^3*W^4
 *      W^{9..15}=W^{n-8}*W^8, W^{17..31}=W^{n-16}*W^16
 *      → 26 cmuls total
 *   2. Call flat DFT-32 kernel on packed block with derived twiddles
 *   3. Walk: b[i] *= step[i]  → 5 cmuls
 *
 * Total overhead: 31 cmuls/block vs 31 loads/block for table approach.
 * Wins when tw table (31*K*16 bytes) exceeds L2 (K > ~512).
 *
 * ═══════════════════════════════════════════════════════════════════
 * REQUIRES
 * ═══════════════════════════════════════════════════════════════════
 *
 * Include the flat kernel BEFORE this header:
 *   #include "fft_radix32_avx512_tw_ladder_v2.h"  (for flat kernel)
 *   — or any header that provides:
 *     radix32_tw_flat_dit_kernel_fwd_avx512(...)
 *     radix32_tw_flat_dit_kernel_bwd_avx512(...)
 */

#ifndef FFT_RADIX32_AVX512_TW_PACK_WALK_H
#define FFT_RADIX32_AVX512_TW_PACK_WALK_H

#include <stddef.h>
#include <immintrin.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * WALK PLAN
 *
 * step[i] = W_{32K}^{pow[i]*8}   (advance 8 lanes = 1 AVX-512 vector)
 * init[i][j] = W_{32K}^{pow[i]*j}  j=0..7  (initial twiddle vector)
 *
 * pow = {1, 2, 4, 8, 16}
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    double step_re[5];
    double step_im[5];
    double init_re[5][8];
    double init_im[5][8];
} radix32_walk_plan_t;

static inline void radix32_walk_plan_init(
    radix32_walk_plan_t *plan, size_t K)
{
    const size_t NN = 32 * K;
    const int pows[5] = {1, 2, 4, 8, 16};
    for (int i = 0; i < 5; i++) {
        double a_step = -2.0 * M_PI * (double)(pows[i] * 8) / (double)NN;
        plan->step_re[i] = cos(a_step);
        plan->step_im[i] = sin(a_step);
        for (int j = 0; j < 8; j++) {
            double a = -2.0 * M_PI * (double)(pows[i] * j) / (double)NN;
            plan->init_re[i][j] = cos(a);
            plan->init_im[i][j] = sin(a);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * COMPLEX MULTIPLY HELPERS
 * ═══════════════════════════════════════════════════════════════ */

#ifndef R32W_CMUL_DEFINED
#define R32W_CMUL_DEFINED
#define R32W_CMUL_RE(ar,ai,br,bi) \
    _mm512_fnmadd_pd(ai, bi, _mm512_mul_pd(ar, br))
#define R32W_CMUL_IM(ar,ai,br,bi) \
    _mm512_fmadd_pd(ar, bi, _mm512_mul_pd(ai, br))
#endif

/* ═══════════════════════════════════════════════════════════════
 * DERIVE 31 TWIDDLES FROM 5 BASES
 *
 * Binary tree:
 *   Level 0: b1, b2, b4 directly
 *   Level 1: b3 = b1*b2
 *   Level 2: b5=b1*b4, b6=b2*b4, b7=b3*b4
 *   Level 3: b9..b15 = b{n-8}*b8
 *   Level 4: b17..b31 = b{n-16}*b16
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,fma")))
static inline void r32w_derive_twiddles(
    const __m512d b_re[5], const __m512d b_im[5],
    double * __restrict__ tw_re, double * __restrict__ tw_im)
{
    /* Store direct bases: W^1 → slot 0, W^2 → slot 1, W^4 → slot 3, W^8 → slot 7, W^16 → slot 15 */
    _mm512_store_pd(&tw_re[0*8], b_re[0]);  _mm512_store_pd(&tw_im[0*8], b_im[0]);
    _mm512_store_pd(&tw_re[1*8], b_re[1]);  _mm512_store_pd(&tw_im[1*8], b_im[1]);
    _mm512_store_pd(&tw_re[3*8], b_re[2]);  _mm512_store_pd(&tw_im[3*8], b_im[2]);
    _mm512_store_pd(&tw_re[7*8], b_re[3]);  _mm512_store_pd(&tw_im[7*8], b_im[3]);
    _mm512_store_pd(&tw_re[15*8], b_re[4]); _mm512_store_pd(&tw_im[15*8], b_im[4]);

    /* W^3 = W^1 * W^2 */
    __m512d w3r = R32W_CMUL_RE(b_re[0],b_im[0],b_re[1],b_im[1]);
    __m512d w3i = R32W_CMUL_IM(b_re[0],b_im[0],b_re[1],b_im[1]);
    _mm512_store_pd(&tw_re[2*8], w3r);  _mm512_store_pd(&tw_im[2*8], w3i);

    /* W^5 = W^1 * W^4 */
    __m512d w5r = R32W_CMUL_RE(b_re[0],b_im[0],b_re[2],b_im[2]);
    __m512d w5i = R32W_CMUL_IM(b_re[0],b_im[0],b_re[2],b_im[2]);
    _mm512_store_pd(&tw_re[4*8], w5r);  _mm512_store_pd(&tw_im[4*8], w5i);

    /* W^6 = W^2 * W^4 */
    __m512d w6r = R32W_CMUL_RE(b_re[1],b_im[1],b_re[2],b_im[2]);
    __m512d w6i = R32W_CMUL_IM(b_re[1],b_im[1],b_re[2],b_im[2]);
    _mm512_store_pd(&tw_re[5*8], w6r);  _mm512_store_pd(&tw_im[5*8], w6i);

    /* W^7 = W^3 * W^4 */
    __m512d w7r = R32W_CMUL_RE(w3r,w3i,b_re[2],b_im[2]);
    __m512d w7i = R32W_CMUL_IM(w3r,w3i,b_re[2],b_im[2]);
    _mm512_store_pd(&tw_re[6*8], w7r);  _mm512_store_pd(&tw_im[6*8], w7i);

    /* W^9..W^15 = W^{n-8} * W^8 */
    for (int n = 9; n <= 15; n++) {
        __m512d sr = _mm512_load_pd(&tw_re[(n-9)*8]);
        __m512d si = _mm512_load_pd(&tw_im[(n-9)*8]);
        _mm512_store_pd(&tw_re[(n-1)*8], R32W_CMUL_RE(sr,si,b_re[3],b_im[3]));
        _mm512_store_pd(&tw_im[(n-1)*8], R32W_CMUL_IM(sr,si,b_re[3],b_im[3]));
    }

    /* W^17..W^31 = W^{n-16} * W^16 */
    for (int n = 17; n <= 31; n++) {
        __m512d sr = _mm512_load_pd(&tw_re[(n-17)*8]);
        __m512d si = _mm512_load_pd(&tw_im[(n-17)*8]);
        _mm512_store_pd(&tw_re[(n-1)*8], R32W_CMUL_RE(sr,si,b_re[4],b_im[4]));
        _mm512_store_pd(&tw_im[(n-1)*8], R32W_CMUL_IM(sr,si,b_re[4],b_im[4]));
    }
}

/* ═══════════════════════════════════════════════════════════════
 * FORWARD PACK+WALK KERNEL
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,fma")))
static inline void
radix32_tw_pack_walk_fwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix32_walk_plan_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 8;

    /* Load step broadcasts */
    __m512d stp_re[5], stp_im[5];
    for (int i = 0; i < 5; i++) {
        stp_re[i] = _mm512_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm512_set1_pd(plan->step_im[i]);
    }

    /* Initialize base accumulators */
    __m512d b_re[5], b_im[5];
    for (int i = 0; i < 5; i++) {
        b_re[i] = _mm512_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm512_loadu_pd(plan->init_im[i]);
    }

    /* Stack scratch for derived twiddles (31×8 = 248 doubles × 2 = 3968 bytes) */
    double __attribute__((aligned(64))) tw_re[31*8];
    double __attribute__((aligned(64))) tw_im[31*8];

    for (size_t blk = 0; blk < nb; blk++) {
        /* Derive 31 twiddles from 5 bases */
        r32w_derive_twiddles(b_re, b_im, tw_re, tw_im);

        /* Flat DFT-32 kernel on packed block */
        radix32_tw_flat_dit_kernel_fwd_avx512(
            in_re + blk*256, in_im + blk*256,
            out_re + blk*256, out_im + blk*256,
            tw_re, tw_im, 8);

        /* Walk: advance 5 bases by step */
        for (int i = 0; i < 5; i++) {
            __m512d nr = R32W_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m512d ni = R32W_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * BACKWARD PACK+WALK KERNEL
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,fma")))
static inline void
radix32_tw_pack_walk_bwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix32_walk_plan_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 8;

    __m512d stp_re[5], stp_im[5];
    for (int i = 0; i < 5; i++) {
        stp_re[i] = _mm512_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm512_set1_pd(plan->step_im[i]);
    }

    __m512d b_re[5], b_im[5];
    for (int i = 0; i < 5; i++) {
        b_re[i] = _mm512_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm512_loadu_pd(plan->init_im[i]);
    }

    double __attribute__((aligned(64))) tw_re[31*8];
    double __attribute__((aligned(64))) tw_im[31*8];

    for (size_t blk = 0; blk < nb; blk++) {
        r32w_derive_twiddles(b_re, b_im, tw_re, tw_im);

        radix32_tw_flat_dit_kernel_bwd_avx512(
            in_re + blk*256, in_im + blk*256,
            out_re + blk*256, out_im + blk*256,
            tw_re, tw_im, 8);

        for (int i = 0; i < 5; i++) {
            __m512d nr = R32W_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m512d ni = R32W_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

#undef R32W_CMUL_RE
#undef R32W_CMUL_IM

#endif /* FFT_RADIX32_AVX512_TW_PACK_WALK_H */
