/**
 * @file fft_r16_avx512_il_tw.h
 * @brief DFT-16 AVX-512 fused TW, native IL — single pass
 *
 * Fuses external twiddle multiply + DFT-16 butterfly into one k-loop.
 * tw_il layout: tw_il[(m-1)*K*2 + k*2 + {0=re,1=im}]  (pre-interleaved at plan time)
 *
 * Per k-iteration:
 *   1. Load 16 complex from in[] (16 ZMM loads)
 *   2. Apply 15 external twiddles via IL cmul (arms 1..15)
 *   3. 4×4 CT DFT-16 with internal W16 twiddles
 *   4. Store 16 complex to out[] (16 ZMM stores)
 *
 * Single memory pass. Half the bandwidth of split TW.
 * k-step = 4 (AVX-512 IL: 4 complex per ZMM)
 */
#ifndef FFT_R16_AVX512_IL_TW_H
#define FFT_R16_AVX512_IL_TW_H
#include <immintrin.h>
#include <stddef.h>

static const double _r16tw_W1r =  0.92387953251128675613;
static const double _r16tw_W1i = -0.38268343236508977173;
static const double _r16tw_W3r =  0.38268343236508977173;
static const double _r16tw_W3i = -0.92387953251128675613;
static const double _r16tw_S2  =  0.70710678118654752440;

/* ═══════════════════════════════════════════════════════════════
 * DIT FORWARD: twiddle BEFORE butterfly
 * Signature: vfft_tw_il_native_codelet_fn
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_tw_dit_fwd_il_avx512(
    const double * __restrict__ in,
    double * __restrict__ out,
    const double * __restrict__ tw_il,
    size_t K)
{
    const __m512d sign_odd = _mm512_castsi512_pd(_mm512_set_epi64(
        (long long)0x8000000000000000ULL, 0,
        (long long)0x8000000000000000ULL, 0,
        (long long)0x8000000000000000ULL, 0,
        (long long)0x8000000000000000ULL, 0));
    const __m512d sign_all = _mm512_castsi512_pd(
        _mm512_set1_epi64((long long)0x8000000000000000ULL));
    const __m512d sign_even = _mm512_castsi512_pd(_mm512_set_epi64(
        0, (long long)0x8000000000000000ULL,
        0, (long long)0x8000000000000000ULL,
        0, (long long)0x8000000000000000ULL,
        0, (long long)0x8000000000000000ULL));

    const __m512d vw1_rr = _mm512_set1_pd(_r16tw_W1r);
    const __m512d vw1_in = _mm512_xor_pd(_mm512_set1_pd(_r16tw_W1i), sign_even);
    const __m512d vw3_rr = _mm512_set1_pd(_r16tw_W3r);
    const __m512d vw3_in = _mm512_xor_pd(_mm512_set1_pd(_r16tw_W3i), sign_even);
    const __m512d vs2    = _mm512_set1_pd(_r16tw_S2);

    /* IL ×(-j): [re,im] → [im,-re] */
    #define TW_NJ(z, d) { d = _mm512_xor_pd(_mm512_permute_pd(z, 0x55), sign_odd); }

    /* IL cmul by constant W */
    #define TW_CMUL(z, wrr, win, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        d = _mm512_fmadd_pd(z, wrr, _mm512_mul_pd(zs, win)); \
    }
    #define TW_CMUL_NEG(z, wrr, win, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        d = _mm512_xor_pd(_mm512_fmadd_pd(z, wrr, _mm512_mul_pd(zs, win)), sign_all); \
    }

    /* IL cmul by runtime twiddle from tw_il */
    #define TW_CMUL_RT(z, tw, d) { \
        __m512d wr = _mm512_permute_pd(tw, 0x00); \
        __m512d wi = _mm512_xor_pd(_mm512_permute_pd(tw, 0xFF), sign_even); \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        d = _mm512_fmadd_pd(z, wr, _mm512_mul_pd(zs, wi)); \
    }

    /* IL ×W8 */
    #define TW_W8(z, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        __m512d sum = _mm512_add_pd(z, zs); \
        __m512d dif = _mm512_sub_pd(z, zs); \
        d = _mm512_mul_pd(_mm512_mask_blend_pd(0xAA, sum, dif), vs2); \
    }

    /* IL ×W8³ */
    #define TW_W83(z, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        __m512d sum = _mm512_add_pd(z, zs); \
        __m512d dif = _mm512_sub_pd(zs, z); \
        d = _mm512_mul_pd(_mm512_mask_blend_pd(0xAA, dif, _mm512_xor_pd(sum, sign_all)), vs2); \
    }

    /* IL DFT-4 */
    #define TW_DFT4(z0,z1,z2,z3, d0,d1,d2,d3) { \
        __m512d s02 = _mm512_add_pd(z0, z2); \
        __m512d d02 = _mm512_sub_pd(z0, z2); \
        __m512d s13 = _mm512_add_pd(z1, z3); \
        __m512d d13 = _mm512_sub_pd(z1, z3); \
        __m512d nj_d13; TW_NJ(d13, nj_d13); \
        d0 = _mm512_add_pd(s02, s13); \
        d2 = _mm512_sub_pd(s02, s13); \
        d1 = _mm512_add_pd(d02, nj_d13); \
        d3 = _mm512_sub_pd(d02, nj_d13); \
    }

    for (size_t k = 0; k < K; k += 4) {
        size_t off = k * 2;

        /* ══ Load 16 inputs + apply external twiddles ══
         * arm 0: no twiddle. arms 1..15: cmul by tw_il. */
        __m512d x0  = _mm512_load_pd(&in[( 0*K)*2 + off]);
        __m512d x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

        #define TW_LOAD_AND_MUL(arm, dst) { \
            __m512d raw = _mm512_load_pd(&in[(arm*K)*2 + off]); \
            __m512d tw  = _mm512_load_pd(&tw_il[(((size_t)(arm)-1)*K)*2 + off]); \
            TW_CMUL_RT(raw, tw, dst); \
        }

        TW_LOAD_AND_MUL( 1, x1);
        TW_LOAD_AND_MUL( 2, x2);
        TW_LOAD_AND_MUL( 3, x3);
        TW_LOAD_AND_MUL( 4, x4);
        TW_LOAD_AND_MUL( 5, x5);
        TW_LOAD_AND_MUL( 6, x6);
        TW_LOAD_AND_MUL( 7, x7);
        TW_LOAD_AND_MUL( 8, x8);
        TW_LOAD_AND_MUL( 9, x9);
        TW_LOAD_AND_MUL(10, x10);
        TW_LOAD_AND_MUL(11, x11);
        TW_LOAD_AND_MUL(12, x12);
        TW_LOAD_AND_MUL(13, x13);
        TW_LOAD_AND_MUL(14, x14);
        TW_LOAD_AND_MUL(15, x15);

        #undef TW_LOAD_AND_MUL

        /* ══ Pass 1: 4× DFT-4 rows + internal W16 twiddles ══ */

        /* Row k1=0: DFT-4 on [0,4,8,12], no internal twiddle */
        __m512d r00, r01, r02, r03;
        TW_DFT4(x0, x4, x8, x12, r00, r01, r02, r03)

        /* Row k1=1: DFT-4 on [1,5,9,13], twiddle W¹,W²,W³ */
        __m512d r10, r11, r12, r13;
        {
            __m512d d0,d1,d2,d3;
            TW_DFT4(x1, x5, x9, x13, d0, d1, d2, d3)
            r10 = d0;
            TW_CMUL(d1, vw1_rr, vw1_in, r11);
            TW_W8(d2, r12);
            TW_CMUL(d3, vw3_rr, vw3_in, r13);
        }

        /* Row k1=2: DFT-4 on [2,6,10,14], twiddle W²,W⁴,W⁶ */
        __m512d r20, r21, r22, r23;
        {
            __m512d d0,d1,d2,d3;
            TW_DFT4(x2, x6, x10, x14, d0, d1, d2, d3)
            r20 = d0;
            TW_W8(d1, r21);
            __m512d nj; TW_NJ(d2, nj); r22 = nj;
            TW_W83(d3, r23);
        }

        /* Row k1=3: DFT-4 on [3,7,11,15], twiddle W³,W⁶,W⁹ */
        __m512d r30, r31, r32, r33;
        {
            __m512d d0,d1,d2,d3;
            TW_DFT4(x3, x7, x11, x15, d0, d1, d2, d3)
            r30 = d0;
            TW_CMUL(d1, vw3_rr, vw3_in, r31);
            TW_W83(d2, r32);
            TW_CMUL_NEG(d3, vw1_rr, vw1_in, r33);
        }

        /* ══ Pass 2: 4× DFT-4 columns → output ══ */
        {
            __m512d d0,d1,d2,d3;
            TW_DFT4(r00,r10,r20,r30, d0,d1,d2,d3)
            _mm512_store_pd(&out[( 0*K)*2 + off], d0);
            _mm512_store_pd(&out[( 4*K)*2 + off], d1);
            _mm512_store_pd(&out[( 8*K)*2 + off], d2);
            _mm512_store_pd(&out[(12*K)*2 + off], d3);
        }
        {
            __m512d d0,d1,d2,d3;
            TW_DFT4(r01,r11,r21,r31, d0,d1,d2,d3)
            _mm512_store_pd(&out[( 1*K)*2 + off], d0);
            _mm512_store_pd(&out[( 5*K)*2 + off], d1);
            _mm512_store_pd(&out[( 9*K)*2 + off], d2);
            _mm512_store_pd(&out[(13*K)*2 + off], d3);
        }
        {
            __m512d d0,d1,d2,d3;
            TW_DFT4(r02,r12,r22,r32, d0,d1,d2,d3)
            _mm512_store_pd(&out[( 2*K)*2 + off], d0);
            _mm512_store_pd(&out[( 6*K)*2 + off], d1);
            _mm512_store_pd(&out[(10*K)*2 + off], d2);
            _mm512_store_pd(&out[(14*K)*2 + off], d3);
        }
        {
            __m512d d0,d1,d2,d3;
            TW_DFT4(r03,r13,r23,r33, d0,d1,d2,d3)
            _mm512_store_pd(&out[( 3*K)*2 + off], d0);
            _mm512_store_pd(&out[( 7*K)*2 + off], d1);
            _mm512_store_pd(&out[(11*K)*2 + off], d2);
            _mm512_store_pd(&out[(15*K)*2 + off], d3);
        }
    }

    #undef TW_NJ
    #undef TW_CMUL
    #undef TW_CMUL_NEG
    #undef TW_CMUL_RT
    #undef TW_W8
    #undef TW_W83
    #undef TW_DFT4
}

#endif /* FFT_R16_AVX512_IL_TW_H */
