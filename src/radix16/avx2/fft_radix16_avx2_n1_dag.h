/**
 * @file fft_r16_avx2_dag.h
 * @brief DFT-16 AVX2 DAG-style codelet — split format, k-step=4
 *
 * Mathematical structure from DFT-16 split-radix with algebraic optimization:
 *   - tan(π/8) trick: W₁₆¹ and W₃ via 2 FMA instead of 4 ops
 *   - ×W₈ folded into subsequent FMA with cos(π/8)
 *   - 72 complex ops (same as FFTW's genfft DAG)
 *   - Split (re/im separate) with k-step=4
 *
 * Register strategy: let the compiler manage allocation.
 * 16 outer-scope complex vars (32 YMM) > 16 regs → compiler spills.
 * But with scoped blocks and -variables-4-style scheduling, most
 * spills stay in L1 and the compiler picks optimal spill points.
 *
 * Constants:
 *   KT = tan(π/8) = √2 - 1 ≈ 0.4142   (for W₁₆¹ and W₁₆³ twiddles)
 *   KS = 1/√2        ≈ 0.7071   (for ×W₈ = ×(1-j)/√2)
 *   KC = cos(π/8)    ≈ 0.9239   (for final twiddle stage)
 */
#ifndef FFT_R16_AVX2_DAG_H
#define FFT_R16_AVX2_DAG_H
#include <immintrin.h>
#include <stddef.h>

__attribute__((target("avx2,fma")))
static void radix16_dag_n1_fwd_avx2(
    const double * __restrict__ ri, const double * __restrict__ ii,
    double * __restrict__ ro, double * __restrict__ io,
    size_t K)
{
    const __m256d KT = _mm256_set1_pd(0.41421356237309504880168872420969808);  /* tan(π/8) */
    const __m256d KS = _mm256_set1_pd(0.70710678118654752440084436210484904);  /* 1/√2 */
    const __m256d KC = _mm256_set1_pd(0.92387953251128675612818318939678829);  /* cos(π/8) */

    for (size_t k = 0; k < K; k += 4) {
        /* ═══ Block 1: even-even group {0,4,8,12} ═══ */
        __m256d T7r, T7i, TUr, TUi, Tzr, Tzi, THr, THi;
        {
            __m256d T1r = _mm256_load_pd(&ri[ 0*K+k]), T1i = _mm256_load_pd(&ii[ 0*K+k]);
            __m256d T2r = _mm256_load_pd(&ri[ 8*K+k]), T2i = _mm256_load_pd(&ii[ 8*K+k]);
            __m256d T3r = _mm256_add_pd(T1r, T2r),     T3i = _mm256_add_pd(T1i, T2i);
            __m256d T4r = _mm256_load_pd(&ri[ 4*K+k]), T4i = _mm256_load_pd(&ii[ 4*K+k]);
            __m256d T5r = _mm256_load_pd(&ri[12*K+k]), T5i = _mm256_load_pd(&ii[12*K+k]);
            __m256d T6r = _mm256_add_pd(T4r, T5r),     T6i = _mm256_add_pd(T4i, T5i);
            T7r = _mm256_sub_pd(T3r, T6r);  T7i = _mm256_sub_pd(T3i, T6i);
            TUr = _mm256_sub_pd(T4r, T5r);  TUi = _mm256_sub_pd(T4i, T5i);
            Tzr = _mm256_add_pd(T3r, T6r);  Tzi = _mm256_add_pd(T3i, T6i);
            THr = _mm256_sub_pd(T1r, T2r);  THi = _mm256_sub_pd(T1i, T2i);
        }

        /* ═══ Block 2: even-odd group {2,6,10,14} ═══ */
        __m256d Tur, Tui, TVr, TVi, TAr, TAi, TKr, TKi;
        {
            __m256d Tor = _mm256_load_pd(&ri[14*K+k]), Toi = _mm256_load_pd(&ii[14*K+k]);
            __m256d Tpr = _mm256_load_pd(&ri[ 6*K+k]), Tpi = _mm256_load_pd(&ii[ 6*K+k]);
            __m256d Tqr = _mm256_add_pd(Tor, Tpr),     Tqi = _mm256_add_pd(Toi, Tpi);
            __m256d TJr = _mm256_sub_pd(Tor, Tpr),     TJi = _mm256_sub_pd(Toi, Tpi);
            __m256d Trr = _mm256_load_pd(&ri[ 2*K+k]), Tri = _mm256_load_pd(&ii[ 2*K+k]);
            __m256d Tsr = _mm256_load_pd(&ri[10*K+k]), Tsi = _mm256_load_pd(&ii[10*K+k]);
            __m256d Ttr = _mm256_add_pd(Trr, Tsr),     Tti = _mm256_add_pd(Tri, Tsi);
            __m256d TIr = _mm256_sub_pd(Trr, Tsr),     TIi = _mm256_sub_pd(Tri, Tsi);
            Tur = _mm256_sub_pd(Tqr, Ttr);  Tui = _mm256_sub_pd(Tqi, Tti);
            TVr = _mm256_sub_pd(TJr, TIr);  TVi = _mm256_sub_pd(TJi, TIi);
            TAr = _mm256_add_pd(Ttr, Tqr);  TAi = _mm256_add_pd(Tti, Tqi);
            TKr = _mm256_add_pd(TIr, TJr);  TKi = _mm256_add_pd(TIi, TJi);
        }

        /* ═══ Block 3: odd-even group {1,5,9,13} + tan trick ═══ */
        __m256d Ter, Tei, TXr, TXi, TCr, TCi, TOr, TOi;
        {
            __m256d T8r = _mm256_load_pd(&ri[ 1*K+k]), T8i = _mm256_load_pd(&ii[ 1*K+k]);
            __m256d T9r = _mm256_load_pd(&ri[ 9*K+k]), T9i = _mm256_load_pd(&ii[ 9*K+k]);
            __m256d Tar = _mm256_add_pd(T8r, T9r),     Tai = _mm256_add_pd(T8i, T9i);
            __m256d TMr = _mm256_sub_pd(T8r, T9r),     TMi = _mm256_sub_pd(T8i, T9i);
            __m256d Tbr = _mm256_load_pd(&ri[ 5*K+k]), Tbi = _mm256_load_pd(&ii[ 5*K+k]);
            __m256d Tcr = _mm256_load_pd(&ri[13*K+k]), Tci = _mm256_load_pd(&ii[13*K+k]);
            __m256d Tdr = _mm256_add_pd(Tbr, Tcr),     Tdi = _mm256_add_pd(Tbi, Tci);
            __m256d TNr = _mm256_sub_pd(Tbr, Tcr),     TNi = _mm256_sub_pd(Tbi, Tci);
            Ter = _mm256_sub_pd(Tar, Tdr);  Tei = _mm256_sub_pd(Tai, Tdi);
            /* tan trick: TX = tan*TM + TN, TO = TM - tan*TN */
            TXr = _mm256_fmadd_pd(KT, TMr, TNr);  TXi = _mm256_fmadd_pd(KT, TMi, TNi);
            TCr = _mm256_add_pd(Tar, Tdr);         TCi = _mm256_add_pd(Tai, Tdi);
            TOr = _mm256_fnmadd_pd(KT, TNr, TMr);  TOi = _mm256_fnmadd_pd(KT, TNi, TMi);
        }

        /* ═══ Block 4: odd-odd group {3,7,11,15} + tan trick ═══ */
        __m256d Tlr, Tli, TYr, TYi, TDr, TDi, TRr, TRi;
        {
            __m256d Tfr = _mm256_load_pd(&ri[15*K+k]), Tfi = _mm256_load_pd(&ii[15*K+k]);
            __m256d Tgr = _mm256_load_pd(&ri[ 7*K+k]), Tgi = _mm256_load_pd(&ii[ 7*K+k]);
            __m256d Thr = _mm256_add_pd(Tfr, Tgr),     Thi = _mm256_add_pd(Tfi, Tgi);
            __m256d TPr = _mm256_sub_pd(Tfr, Tgr),     TPi = _mm256_sub_pd(Tfi, Tgi);
            __m256d Tir = _mm256_load_pd(&ri[ 3*K+k]), Tii_ = _mm256_load_pd(&ii[ 3*K+k]);
            __m256d Tjr = _mm256_load_pd(&ri[11*K+k]), Tji = _mm256_load_pd(&ii[11*K+k]);
            __m256d Tkr = _mm256_add_pd(Tir, Tjr),     Tki = _mm256_add_pd(Tii_, Tji);
            /* NOTE: TQ = Tj - Ti (reversed vs other blocks!) */
            __m256d TQr = _mm256_sub_pd(Tjr, Tir),     TQi = _mm256_sub_pd(Tji, Tii_);
            Tlr = _mm256_sub_pd(Thr, Tkr);  Tli = _mm256_sub_pd(Thi, Tki);
            TYr = _mm256_fmadd_pd(KT, TPr, TQr);   TYi = _mm256_fmadd_pd(KT, TPi, TQi);
            TDr = _mm256_add_pd(Thr, Tkr);          TDi = _mm256_add_pd(Thi, Tki);
            TRr = _mm256_fnmadd_pd(KT, TQr, TPr);   TRi = _mm256_fnmadd_pd(KT, TQi, TPi);
        }

        /* ═══ Output block 1: arms 0, 8 (pure add/sub) ═══ */
        {
            __m256d TBr = _mm256_add_pd(Tzr, TAr),  TBi = _mm256_add_pd(Tzi, TAi);
            __m256d TEr = _mm256_add_pd(TCr, TDr),  TEi = _mm256_add_pd(TCi, TDi);
            _mm256_store_pd(&ro[ 8*K+k], _mm256_sub_pd(TBr, TEr));
            _mm256_store_pd(&io[ 8*K+k], _mm256_sub_pd(TBi, TEi));
            _mm256_store_pd(&ro[ 0*K+k], _mm256_add_pd(TBr, TEr));
            _mm256_store_pd(&io[ 0*K+k], _mm256_add_pd(TBi, TEi));
        }

        /* ═══ Output block 2: arms 4, 12 (×j fused) ═══ */
        {
            __m256d TFr = _mm256_sub_pd(Tzr, TAr),  TFi = _mm256_sub_pd(Tzi, TAi);
            __m256d TGr = _mm256_sub_pd(TDr, TCr),  TGi = _mm256_sub_pd(TDi, TCi);
            /* out[12] = TF - j*TG: re = TFr + TGi, im = TFi - TGr */
            _mm256_store_pd(&ro[12*K+k], _mm256_add_pd(TFr, TGi));
            _mm256_store_pd(&io[12*K+k], _mm256_sub_pd(TFi, TGr));
            /* out[4]  = TF + j*TG: re = TFr - TGi, im = TFi + TGr */
            _mm256_store_pd(&ro[ 4*K+k], _mm256_sub_pd(TFr, TGi));
            _mm256_store_pd(&io[ 4*K+k], _mm256_add_pd(TFi, TGr));
        }

        /* ═══ Output block 3: arms 2, 6, 10, 14 (×W₈ via FMA with 1/√2) ═══ */
        {
            __m256d Tmr = _mm256_add_pd(Ter, Tlr),  Tmi = _mm256_add_pd(Tei, Tli);
            /* Tn = T7 - (1/√2)*Tm, Tx = T7 + (1/√2)*Tm */
            __m256d Tnr = _mm256_fnmadd_pd(KS, Tmr, T7r),  Tni = _mm256_fnmadd_pd(KS, Tmi, T7i);
            __m256d Txr = _mm256_fmadd_pd(KS, Tmr, T7r),   Txi = _mm256_fmadd_pd(KS, Tmi, T7i);
            __m256d Tvr = _mm256_sub_pd(Tlr, Ter),  Tvi = _mm256_sub_pd(Tli, Tei);
            __m256d Twr = _mm256_fnmadd_pd(KS, Tvr, Tur),  Twi = _mm256_fnmadd_pd(KS, Tvi, Tui);
            __m256d Tyr = _mm256_fmadd_pd(KS, Tvr, Tur),   Tyi = _mm256_fmadd_pd(KS, Tvi, Tui);
            /* out[6]  = Tn - j*Tw */
            _mm256_store_pd(&ro[ 6*K+k], _mm256_add_pd(Tnr, Twi));
            _mm256_store_pd(&io[ 6*K+k], _mm256_sub_pd(Tni, Twr));
            /* out[2]  = Tx + j*Ty */
            _mm256_store_pd(&ro[ 2*K+k], _mm256_sub_pd(Txr, Tyi));
            _mm256_store_pd(&io[ 2*K+k], _mm256_add_pd(Txi, Tyr));
            /* out[10] = Tn + j*Tw */
            _mm256_store_pd(&ro[10*K+k], _mm256_sub_pd(Tnr, Twi));
            _mm256_store_pd(&io[10*K+k], _mm256_add_pd(Tni, Twr));
            /* out[14] = Tx - j*Ty */
            _mm256_store_pd(&ro[14*K+k], _mm256_add_pd(Txr, Tyi));
            _mm256_store_pd(&io[14*K+k], _mm256_sub_pd(Txi, Tyr));
        }

        /* ═══ Output block 4: arms 1, 7, 9, 15 (cos(π/8) stage) ═══ */
        {
            __m256d TLr = _mm256_fmadd_pd(KS, TKr, THr),   TLi = _mm256_fmadd_pd(KS, TKi, THi);
            __m256d TSr = _mm256_add_pd(TOr, TRr),          TSi = _mm256_add_pd(TOi, TRi);
            __m256d TTr = _mm256_fnmadd_pd(KC, TSr, TLr),   TTi = _mm256_fnmadd_pd(KC, TSi, TLi);
            __m256d T11r = _mm256_fmadd_pd(KC, TSr, TLr),   T11i = _mm256_fmadd_pd(KC, TSi, TLi);
            __m256d TWr = _mm256_fnmadd_pd(KS, TVr, TUr),   TWi = _mm256_fnmadd_pd(KS, TVi, TUi);
            __m256d TZr = _mm256_sub_pd(TXr, TYr),          TZi = _mm256_sub_pd(TXi, TYi);
            __m256d T10r = _mm256_fnmadd_pd(KC, TZr, TWr),  T10i = _mm256_fnmadd_pd(KC, TZi, TWi);
            __m256d T12r = _mm256_fmadd_pd(KC, TZr, TWr),   T12i = _mm256_fmadd_pd(KC, TZi, TWi);
            /* out[9]  = TT - j*T10 */
            _mm256_store_pd(&ro[ 9*K+k], _mm256_add_pd(TTr, T10i));
            _mm256_store_pd(&io[ 9*K+k], _mm256_sub_pd(TTi, T10r));
            /* out[15] = T11 + j*T12 */
            _mm256_store_pd(&ro[15*K+k], _mm256_sub_pd(T11r, T12i));
            _mm256_store_pd(&io[15*K+k], _mm256_add_pd(T11i, T12r));
            /* out[7]  = TT + j*T10 */
            _mm256_store_pd(&ro[ 7*K+k], _mm256_sub_pd(TTr, T10i));
            _mm256_store_pd(&io[ 7*K+k], _mm256_add_pd(TTi, T10r));
            /* out[1]  = T11 - j*T12 */
            _mm256_store_pd(&ro[ 1*K+k], _mm256_add_pd(T11r, T12i));
            _mm256_store_pd(&io[ 1*K+k], _mm256_sub_pd(T11i, T12r));
        }

        /* ═══ Output block 5: arms 3, 5, 11, 13 (cos(π/8) stage, other diagonal) ═══ */
        {
            __m256d T13r = _mm256_fnmadd_pd(KS, TKr, THr),  T13i = _mm256_fnmadd_pd(KS, TKi, THi);
            __m256d T14r = _mm256_add_pd(TXr, TYr),         T14i = _mm256_add_pd(TXi, TYi);
            __m256d T15r = _mm256_fnmadd_pd(KC, T14r, T13r), T15i = _mm256_fnmadd_pd(KC, T14i, T13i);
            __m256d T19r = _mm256_fmadd_pd(KC, T14r, T13r),  T19i = _mm256_fmadd_pd(KC, T14i, T13i);
            __m256d T16r = _mm256_fmadd_pd(KS, TVr, TUr),   T16i = _mm256_fmadd_pd(KS, TVi, TUi);
            __m256d T17r = _mm256_sub_pd(TRr, TOr),          T17i = _mm256_sub_pd(TRi, TOi);
            __m256d T18r = _mm256_fnmadd_pd(KC, T17r, T16r), T18i = _mm256_fnmadd_pd(KC, T17i, T16i);
            __m256d T1ar = _mm256_fmadd_pd(KC, T17r, T16r),  T1ai = _mm256_fmadd_pd(KC, T17i, T16i);
            /* out[5]  = T15 - j*T18 */
            _mm256_store_pd(&ro[ 5*K+k], _mm256_add_pd(T15r, T18i));
            _mm256_store_pd(&io[ 5*K+k], _mm256_sub_pd(T15i, T18r));
            /* out[13] = T19 - j*T1a */
            _mm256_store_pd(&ro[13*K+k], _mm256_add_pd(T19r, T1ai));
            _mm256_store_pd(&io[13*K+k], _mm256_sub_pd(T19i, T1ar));
            /* out[11] = T15 + j*T18 */
            _mm256_store_pd(&ro[11*K+k], _mm256_sub_pd(T15r, T18i));
            _mm256_store_pd(&io[11*K+k], _mm256_add_pd(T15i, T18r));
            /* out[3]  = T19 + j*T1a */
            _mm256_store_pd(&ro[ 3*K+k], _mm256_sub_pd(T19r, T1ai));
            _mm256_store_pd(&io[ 3*K+k], _mm256_add_pd(T19i, T1ar));
        }
    }
}

#endif /* FFT_R16_AVX2_DAG_H */
