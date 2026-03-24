/**
 * @file fft_r16_avx2_dag_il.h
 * @brief DFT-16 AVX2 DAG in native IL — 72 ops, k-step=2
 *
 * Direct translation of the split-radix algebraic DAG to IL format.
 * Same math as FFTW's genfft output, but:
 *   - Direct AVX2 FMA intrinsics (no abstraction layer)
 *   - Aligned loads on 32-byte boundaries
 *   - No function pointer / planner overhead
 *   - ×j fused via vpermilpd + vxorpd (single-cycle on port 5)
 *
 * k-step=2: each YMM = [re0,im0,re1,im1] = 2 complex doubles
 *
 * IL multiply-by-constant tricks:
 *   VFMA(c, A, B) = c*A + B  (element-wise, c is broadcast scalar)
 *   VFMAI(B, A)   = A + j*B  (IL: permute B, sign-flip, add)
 *   VFNMSI(B, A)  = A - j*B  (IL: permute B, sign-flip, sub)
 */
#ifndef FFT_R16_AVX2_DAG_IL_H
#define FFT_R16_AVX2_DAG_IL_H
#include <immintrin.h>
#include <stddef.h>

__attribute__((target("avx2,fma")))
static void radix16_dag_n1_fwd_il_avx2(
    const double * __restrict__ in,
    double * __restrict__ out,
    size_t K)
{
    const __m256d KT = _mm256_set1_pd(0.41421356237309504880168872420969808);
    const __m256d KS = _mm256_set1_pd(0.70710678118654752440084436210484904);
    const __m256d KC = _mm256_set1_pd(0.92387953251128675612818318939678829);

    /* Sign masks for IL ×j operations */
    const __m256d sign_odd = _mm256_castsi256_pd(_mm256_set_epi64x(
        (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL, 0));
    const __m256d sign_even = _mm256_castsi256_pd(_mm256_set_epi64x(
        0, (long long)0x8000000000000000ULL, 0, (long long)0x8000000000000000ULL));

    /* IL ×(+j): [re,im] → [-im,re] = permute + negate even (re positions get -im) */
    #define FMAI(B, A, D)  { __m256d jB = _mm256_xor_pd(_mm256_permute_pd(B,0x5),sign_even); D = _mm256_add_pd(A, jB); }
    /* IL ×(-j): [re,im] → [im,-re] = permute + negate odd (im positions get -re) */
    #define FNMSI(B, A, D) { __m256d jB = _mm256_xor_pd(_mm256_permute_pd(B,0x5),sign_odd); D = _mm256_add_pd(A, jB); }

    for (size_t k = 0; k < K; k += 2) {
        size_t off = k * 2;  /* IL offset */

        /* ═══ Block 1: {0,4,8,12} ═══ */
        __m256d T7, TU, Tz, TH;
        {
            __m256d T1 = _mm256_load_pd(&in[( 0*K)*2+off]);
            __m256d T2 = _mm256_load_pd(&in[( 8*K)*2+off]);
            __m256d T3 = _mm256_add_pd(T1, T2);
            __m256d T4 = _mm256_load_pd(&in[( 4*K)*2+off]);
            __m256d T5 = _mm256_load_pd(&in[(12*K)*2+off]);
            __m256d T6 = _mm256_add_pd(T4, T5);
            T7 = _mm256_sub_pd(T3, T6);
            TU = _mm256_sub_pd(T4, T5);
            Tz = _mm256_add_pd(T3, T6);
            TH = _mm256_sub_pd(T1, T2);
        }

        /* ═══ Block 2: {2,6,10,14} ═══ */
        __m256d Tu, TV, TA, TK;
        {
            __m256d To = _mm256_load_pd(&in[(14*K)*2+off]);
            __m256d Tp = _mm256_load_pd(&in[( 6*K)*2+off]);
            __m256d Tq = _mm256_add_pd(To, Tp);
            __m256d TJ = _mm256_sub_pd(To, Tp);
            __m256d Tr = _mm256_load_pd(&in[( 2*K)*2+off]);
            __m256d Ts = _mm256_load_pd(&in[(10*K)*2+off]);
            __m256d Tt = _mm256_add_pd(Tr, Ts);
            __m256d TI = _mm256_sub_pd(Tr, Ts);
            Tu = _mm256_sub_pd(Tq, Tt);
            TV = _mm256_sub_pd(TJ, TI);
            TA = _mm256_add_pd(Tt, Tq);
            TK = _mm256_add_pd(TI, TJ);
        }

        /* ═══ Block 3: {1,5,9,13} + tan trick ═══ */
        __m256d Te, TX, TC, TO;
        {
            __m256d T8 = _mm256_load_pd(&in[( 1*K)*2+off]);
            __m256d T9 = _mm256_load_pd(&in[( 9*K)*2+off]);
            __m256d Ta = _mm256_add_pd(T8, T9);
            __m256d TM = _mm256_sub_pd(T8, T9);
            __m256d Tb = _mm256_load_pd(&in[( 5*K)*2+off]);
            __m256d Tc = _mm256_load_pd(&in[(13*K)*2+off]);
            __m256d Td = _mm256_add_pd(Tb, Tc);
            __m256d TN = _mm256_sub_pd(Tb, Tc);
            Te = _mm256_sub_pd(Ta, Td);
            TX = _mm256_fmadd_pd(KT, TM, TN);     /* tan*TM + TN */
            TC = _mm256_add_pd(Ta, Td);
            TO = _mm256_fnmadd_pd(KT, TN, TM);     /* TM - tan*TN */
        }

        /* ═══ Block 4: {3,7,11,15} + tan trick ═══ */
        __m256d Tl, TY, TD, TR;
        {
            __m256d Tf = _mm256_load_pd(&in[(15*K)*2+off]);
            __m256d Tg = _mm256_load_pd(&in[( 7*K)*2+off]);
            __m256d Th = _mm256_add_pd(Tf, Tg);
            __m256d TP = _mm256_sub_pd(Tf, Tg);
            __m256d Ti = _mm256_load_pd(&in[( 3*K)*2+off]);
            __m256d Tj = _mm256_load_pd(&in[(11*K)*2+off]);
            __m256d Tk = _mm256_add_pd(Ti, Tj);
            __m256d TQ = _mm256_sub_pd(Tj, Ti);  /* NOTE: reversed */
            Tl = _mm256_sub_pd(Th, Tk);
            TY = _mm256_fmadd_pd(KT, TP, TQ);
            TD = _mm256_add_pd(Th, Tk);
            TR = _mm256_fnmadd_pd(KT, TQ, TP);
        }

        /* ═══ Output 1: arms 0, 8 ═══ */
        {
            __m256d TB = _mm256_add_pd(Tz, TA);
            __m256d TE = _mm256_add_pd(TC, TD);
            _mm256_store_pd(&out[( 8*K)*2+off], _mm256_sub_pd(TB, TE));
            _mm256_store_pd(&out[( 0*K)*2+off], _mm256_add_pd(TB, TE));
        }

        /* ═══ Output 2: arms 4, 12 (×j fused) ═══ */
        {
            __m256d TF = _mm256_sub_pd(Tz, TA);
            __m256d TG = _mm256_sub_pd(TD, TC);
            __m256d out12, out4;
            FNMSI(TG, TF, out12)  /* TF - j*TG */
            FMAI(TG, TF, out4)    /* TF + j*TG */
            _mm256_store_pd(&out[(12*K)*2+off], out12);
            _mm256_store_pd(&out[( 4*K)*2+off], out4);
        }

        /* ═══ Output 3: arms 2, 6, 10, 14 ═══ */
        {
            __m256d Tm = _mm256_add_pd(Te, Tl);
            __m256d Tn = _mm256_fnmadd_pd(KS, Tm, T7);
            __m256d Tx = _mm256_fmadd_pd(KS, Tm, T7);
            __m256d Tv = _mm256_sub_pd(Tl, Te);
            __m256d Tw = _mm256_fnmadd_pd(KS, Tv, Tu);
            __m256d Ty = _mm256_fmadd_pd(KS, Tv, Tu);
            __m256d out6, out2, out10, out14;
            FNMSI(Tw, Tn, out6)
            FMAI(Ty, Tx, out2)
            FMAI(Tw, Tn, out10)
            FNMSI(Ty, Tx, out14)
            _mm256_store_pd(&out[( 6*K)*2+off], out6);
            _mm256_store_pd(&out[( 2*K)*2+off], out2);
            _mm256_store_pd(&out[(10*K)*2+off], out10);
            _mm256_store_pd(&out[(14*K)*2+off], out14);
        }

        /* ═══ Output 4: arms 1, 7, 9, 15 ═══ */
        {
            __m256d TL = _mm256_fmadd_pd(KS, TK, TH);
            __m256d TS = _mm256_add_pd(TO, TR);
            __m256d TT = _mm256_fnmadd_pd(KC, TS, TL);
            __m256d T11 = _mm256_fmadd_pd(KC, TS, TL);
            __m256d TW = _mm256_fnmadd_pd(KS, TV, TU);
            __m256d TZ = _mm256_sub_pd(TX, TY);
            __m256d T10 = _mm256_fnmadd_pd(KC, TZ, TW);
            __m256d T12 = _mm256_fmadd_pd(KC, TZ, TW);
            __m256d out9, out15, out7, out1;
            FNMSI(T10, TT, out9)
            FMAI(T12, T11, out15)
            FMAI(T10, TT, out7)
            FNMSI(T12, T11, out1)
            _mm256_store_pd(&out[( 9*K)*2+off], out9);
            _mm256_store_pd(&out[(15*K)*2+off], out15);
            _mm256_store_pd(&out[( 7*K)*2+off], out7);
            _mm256_store_pd(&out[( 1*K)*2+off], out1);
        }

        /* ═══ Output 5: arms 3, 5, 11, 13 ═══ */
        {
            __m256d T13 = _mm256_fnmadd_pd(KS, TK, TH);
            __m256d T14 = _mm256_add_pd(TX, TY);
            __m256d T15 = _mm256_fnmadd_pd(KC, T14, T13);
            __m256d T19 = _mm256_fmadd_pd(KC, T14, T13);
            __m256d T16 = _mm256_fmadd_pd(KS, TV, TU);
            __m256d T17 = _mm256_sub_pd(TR, TO);
            __m256d T18 = _mm256_fnmadd_pd(KC, T17, T16);
            __m256d T1a = _mm256_fmadd_pd(KC, T17, T16);
            __m256d out5, out13, out11, out3;
            FNMSI(T18, T15, out5)
            FNMSI(T1a, T19, out13)
            FMAI(T18, T15, out11)
            FMAI(T1a, T19, out3)
            _mm256_store_pd(&out[( 5*K)*2+off], out5);
            _mm256_store_pd(&out[(13*K)*2+off], out13);
            _mm256_store_pd(&out[(11*K)*2+off], out11);
            _mm256_store_pd(&out[( 3*K)*2+off], out3);
        }
    }
    #undef FMAI
    #undef FNMSI
}

#endif /* FFT_R16_AVX2_DAG_IL_H */
