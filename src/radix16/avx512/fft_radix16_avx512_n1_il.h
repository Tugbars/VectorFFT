/**
 * @file fft_r16_avx512_il.h
 * @brief DFT-16 AVX-512 interleaved CT — native IL, zero permutex2var
 *
 * IL format: {re0,im0,re1,im1,...} — one ZMM = 4 complex doubles
 * k-step = 4 (half of split, but half the streams)
 *
 * DFT-4 in IL: add/sub on complex pairs, ×(-j) = permute+sign-flip
 * W16 twiddles via log3 cmul: 1 permute + 1 mul + 1 fmadd
 *
 * Register-only design: 16 complex × 1 ZMM = 16 ZMM intermediates
 * + constants + temps = ~24 ZMM peak, fits in 32.
 */
#ifndef FFT_R16_AVX512_IL_H
#define FFT_R16_AVX512_IL_H
#include <immintrin.h>
#include <stddef.h>

/* IL twiddle constants: broadcast as [val, val, val, val, ...] */
static const double _r16il_W1r  =  0.92387953251128675613;  /* cos(π/8) */
static const double _r16il_W1i  = -0.38268343236508977173;  /* -sin(π/8) */
static const double _r16il_W3r  =  0.38268343236508977173;  /* cos(3π/8) */
static const double _r16il_W3i  = -0.92387953251128675613;  /* -sin(3π/8) */
static const double _r16il_S2   =  0.70710678118654752440;  /* 1/√2 */

/* ═══════════════════════════════════════════════════════════════
 * N1 FORWARD (notw, interleaved)
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix16_ct_n1_fwd_il_avx512(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    /* Sign masks for IL operations:
     * sign_odd  = {+0, -0, +0, -0, ...}  negate im slots (positions 1,3,5,7)
     * sign_even = {-0, +0, -0, +0, ...}  negate re slots (positions 0,2,4,6) */
    const __m512d sign_odd  = _mm512_castsi512_pd(_mm512_set_epi64(
        (long long)0x8000000000000000ULL, 0,
        (long long)0x8000000000000000ULL, 0,
        (long long)0x8000000000000000ULL, 0,
        (long long)0x8000000000000000ULL, 0));
    const __m512d sign_even = _mm512_castsi512_pd(_mm512_set_epi64(
        0, (long long)0x8000000000000000ULL,
        0, (long long)0x8000000000000000ULL,
        0, (long long)0x8000000000000000ULL,
        0, (long long)0x8000000000000000ULL));

    /* IL cmul constants: wr broadcast + wi_neg = [-wi, wi, -wi, wi, ...] */
    const __m512d vw1_rr = _mm512_set1_pd(_r16il_W1r);
    const __m512d vw1_in = _mm512_xor_pd(_mm512_set1_pd(_r16il_W1i), sign_even);
    const __m512d vw3_rr = _mm512_set1_pd(_r16il_W3r);
    const __m512d vw3_in = _mm512_xor_pd(_mm512_set1_pd(_r16il_W3i), sign_even);
    const __m512d vs2    = _mm512_set1_pd(_r16il_S2);

    /* IL ×(-j): [re,im] → [im,-re]
     * = permute pairs + negate new im (odd) slots */
    #define IL_NJ(z, d) { \
        d = _mm512_xor_pd(_mm512_permute_pd(z, 0x55), sign_odd); \
    }

    /* IL cmul by constant W = (wr, wi):
     * z_swap = permute(z)
     * result = z * wr_broadcast + z_swap * wi_neg_broadcast
     * wi_neg = [-wi, wi, -wi, wi, ...] so: 
     *   even slots: re*wr + im*(-wi) = re*wr - im*wi ✓
     *   odd  slots: im*wr + re*wi                    ✓ */
    #define IL_CMUL(z, wrr, win, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        d = _mm512_fmadd_pd(z, wrr, _mm512_mul_pd(zs, win)); \
    }

    /* IL cmul by -W: negate result */
    #define IL_CMUL_NEG(z, wrr, win, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        d = _mm512_xor_pd(_mm512_fmadd_pd(z, wrr, _mm512_mul_pd(zs, win)), \
            _mm512_castsi512_pd(_mm512_set1_epi64((long long)0x8000000000000000ULL))); \
    }

    /* IL ×W8: (a+bi)(1-i)/√2 = ((a+b)/√2, (b-a)/√2)
     * z = [re, im, ...], z_swap = [im, re, ...]
     * sum  = z + z_swap = [re+im, im+re, ...]  (same both slots)
     * dif  = z - z_swap = [re-im, im-re, ...]
     * blend even from sum, odd from dif → [re+im, im-re, ...]
     * result = blend * s2 */
    #define IL_W8(z, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        __m512d sum = _mm512_add_pd(z, zs); \
        __m512d dif = _mm512_sub_pd(z, zs); \
        d = _mm512_mul_pd(_mm512_mask_blend_pd(0xAA, sum, dif), vs2); \
    }

    /* IL ×W8³: (-1-i)/√2 → re'=(b-a)/√2, im'=-(a+b)/√2
     * = [im-re, -(re+im), ...] * s2
     * blend even from diff, odd from -sum */
    #define IL_W83(z, d) { \
        __m512d zs = _mm512_permute_pd(z, 0x55); \
        __m512d sum = _mm512_add_pd(z, zs); \
        __m512d dif = _mm512_sub_pd(zs, z); \
        __m512d neg_sum = _mm512_xor_pd(sum, _mm512_castsi512_pd(_mm512_set1_epi64((long long)0x8000000000000000ULL))); \
        d = _mm512_mul_pd(_mm512_mask_blend_pd(0xAA, dif, neg_sum), vs2); \
    }

    /* IL DFT-4 macro: 4 complex inputs → 4 complex outputs
     * ×(-j) fused inline */
    #define IL_DFT4(z0,z1,z2,z3, d0,d1,d2,d3) { \
        __m512d s02 = _mm512_add_pd(z0, z2); \
        __m512d d02 = _mm512_sub_pd(z0, z2); \
        __m512d s13 = _mm512_add_pd(z1, z3); \
        __m512d d13 = _mm512_sub_pd(z1, z3); \
        __m512d nj_d13; IL_NJ(d13, nj_d13); \
        d0 = _mm512_add_pd(s02, s13); \
        d2 = _mm512_sub_pd(s02, s13); \
        d1 = _mm512_add_pd(d02, nj_d13); \
        d3 = _mm512_sub_pd(d02, nj_d13); \
    }

    for (size_t k = 0; k < K; k += 4) {
        size_t off = k * 2;  /* IL offset: k complex = k*2 doubles */

        /* ══ Row k1=0: DFT-4 on [0,4,8,12], no twiddle ══ */
        __m512d r00, r01, r02, r03;
        {
            __m512d z0 = _mm512_load_pd(&in[( 0*K)*2 + off]);
            __m512d z1 = _mm512_load_pd(&in[( 4*K)*2 + off]);
            __m512d z2 = _mm512_load_pd(&in[( 8*K)*2 + off]);
            __m512d z3 = _mm512_load_pd(&in[(12*K)*2 + off]);
            IL_DFT4(z0,z1,z2,z3, r00,r01,r02,r03)
        }

        /* ══ Row k1=1: DFT-4 on [1,5,9,13], twiddle W¹,W²,W³ ══ */
        __m512d r10, r11, r12, r13;
        {
            __m512d z0 = _mm512_load_pd(&in[( 1*K)*2 + off]);
            __m512d z1 = _mm512_load_pd(&in[( 5*K)*2 + off]);
            __m512d z2 = _mm512_load_pd(&in[( 9*K)*2 + off]);
            __m512d z3 = _mm512_load_pd(&in[(13*K)*2 + off]);
            __m512d d0,d1,d2,d3;
            IL_DFT4(z0,z1,z2,z3, d0,d1,d2,d3)
            r10 = d0;
            IL_CMUL(d1, vw1_rr, vw1_in, r11);
            IL_W8(d2, r12);
            IL_CMUL(d3, vw3_rr, vw3_in, r13);
        }

        /* ══ Row k1=2: DFT-4 on [2,6,10,14], twiddle W²,W⁴,W⁶ ══ */
        __m512d r20, r21, r22, r23;
        {
            __m512d z0 = _mm512_load_pd(&in[( 2*K)*2 + off]);
            __m512d z1 = _mm512_load_pd(&in[( 6*K)*2 + off]);
            __m512d z2 = _mm512_load_pd(&in[(10*K)*2 + off]);
            __m512d z3 = _mm512_load_pd(&in[(14*K)*2 + off]);
            __m512d d0,d1,d2,d3;
            IL_DFT4(z0,z1,z2,z3, d0,d1,d2,d3)
            r20 = d0;
            IL_W8(d1, r21);
            __m512d nj; IL_NJ(d2, nj); r22 = nj;
            IL_W83(d3, r23);
        }

        /* ══ Row k1=3: DFT-4 on [3,7,11,15], twiddle W³,W⁶,W⁹ ══ */
        __m512d r30, r31, r32, r33;
        {
            __m512d z0 = _mm512_load_pd(&in[( 3*K)*2 + off]);
            __m512d z1 = _mm512_load_pd(&in[( 7*K)*2 + off]);
            __m512d z2 = _mm512_load_pd(&in[(11*K)*2 + off]);
            __m512d z3 = _mm512_load_pd(&in[(15*K)*2 + off]);
            __m512d d0,d1,d2,d3;
            IL_DFT4(z0,z1,z2,z3, d0,d1,d2,d3)
            r30 = d0;
            IL_CMUL(d1, vw3_rr, vw3_in, r31);
            IL_W83(d2, r32);
            IL_CMUL_NEG(d3, vw1_rr, vw1_in, r33);
        }

        /* ══ Pass 2: 4× DFT-4 columns → output ══ */

        /* Column n2=0 → out[0,4,8,12] */
        {
            __m512d d0,d1,d2,d3;
            IL_DFT4(r00,r10,r20,r30, d0,d1,d2,d3)
            _mm512_store_pd(&out[( 0*K)*2 + off], d0);
            _mm512_store_pd(&out[( 4*K)*2 + off], d1);
            _mm512_store_pd(&out[( 8*K)*2 + off], d2);
            _mm512_store_pd(&out[(12*K)*2 + off], d3);
        }
        /* Column n2=1 → out[1,5,9,13] */
        {
            __m512d d0,d1,d2,d3;
            IL_DFT4(r01,r11,r21,r31, d0,d1,d2,d3)
            _mm512_store_pd(&out[( 1*K)*2 + off], d0);
            _mm512_store_pd(&out[( 5*K)*2 + off], d1);
            _mm512_store_pd(&out[( 9*K)*2 + off], d2);
            _mm512_store_pd(&out[(13*K)*2 + off], d3);
        }
        /* Column n2=2 → out[2,6,10,14] */
        {
            __m512d d0,d1,d2,d3;
            IL_DFT4(r02,r12,r22,r32, d0,d1,d2,d3)
            _mm512_store_pd(&out[( 2*K)*2 + off], d0);
            _mm512_store_pd(&out[( 6*K)*2 + off], d1);
            _mm512_store_pd(&out[(10*K)*2 + off], d2);
            _mm512_store_pd(&out[(14*K)*2 + off], d3);
        }
        /* Column n2=3 → out[3,7,11,15] */
        {
            __m512d d0,d1,d2,d3;
            IL_DFT4(r03,r13,r23,r33, d0,d1,d2,d3)
            _mm512_store_pd(&out[( 3*K)*2 + off], d0);
            _mm512_store_pd(&out[( 7*K)*2 + off], d1);
            _mm512_store_pd(&out[(11*K)*2 + off], d2);
            _mm512_store_pd(&out[(15*K)*2 + off], d3);
        }
    }

    #undef IL_NJ
    #undef IL_CMUL
    #undef IL_CMUL_NEG
    #undef IL_W8
    #undef IL_W83
    #undef IL_DFT4
}

#endif /* FFT_R16_AVX512_IL_H */
