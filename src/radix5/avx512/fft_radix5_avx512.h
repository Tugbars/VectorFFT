/**
 * @file  fft_radix5_avx512.h
 * @brief Radix-5 FFT butterfly using Winograd (WFTA) — AVX-512 (8-wide), U=2
 *
 * ALGORITHM: Direct closed-form DFT-5, identical math to AVX2 version.
 *
 * AVX-512 ADVANTAGES EXPLOITED:
 * ═════════════════════════════
 *   1. 32 ZMM registers → U=2 with hoisted constants, ZERO spills guaranteed
 *   2. 8-wide vectors   → 2× throughput per instruction vs AVX2
 *   3. Masked load/store → eliminates scalar tail entirely (2 paths, not 3)
 *   4. Software prefetch → explicit guidance for 10-14 concurrent streams
 *
 * REGISTER BUDGET (32 ZMM):
 *   Hoisted constants:  4  (vQA, vQB, vS1, vS2)  → ZMM28-31
 *   U=2 peak (N1):     24  (12 per butterfly)      → total 28, 4 spare
 *   U=2 peak (fwd tw): 28  (twiddle derivation)    → total 32, exact fit
 *   U=2 peak (bwd tw): 18  (post-twiddle drains)   → total 22, 10 spare
 *
 * LOOP STRUCTURE:
 *   U=2 main  (k+=16)  → full 8-wide U=1 (at most once)  → masked tail (1..7)
 *   Only 2.5 code paths vs AVX2's 3 (U=2 → U=1 → scalar)
 *
 * TWIDDLE LAYOUT: BLOCKED2  (same as AVX2)
 *   tw1[k] = W^k, tw2[k] = W^{2k}. W3=W1·W2, W4=W2² derived inline.
 *
 * FUNCTIONS:
 *   radix5_wfta_fwd_avx512      — forward, twiddled (BLOCKED2)
 *   radix5_wfta_bwd_avx512      — backward, twiddled (IDFT-5 → post-conj-twiddle)
 *   radix5_wfta_fwd_avx512_N1   — forward, no twiddles
 *   radix5_wfta_bwd_avx512_N1   — backward, no twiddles
 */

#ifndef FFT_RADIX5_AVX512_H
#define FFT_RADIX5_AVX512_H

#include <immintrin.h>

#ifndef R5_BUTTERFLY_API
#define R5_BUTTERFLY_API static inline __attribute__((always_inline))
#endif

/* ================================================================== */
/*  WFTA constants for DFT-5                                           */
/* ================================================================== */
#define R5_C1 0.30901699437494742410    /* cos(2π/5) = (√5-1)/4       */
#define R5_C2 (-0.80901699437494742410) /* cos(4π/5) = -(√5+1)/4      */
#define R5_SIN1 0.95105651629515357212  /* sin(2π/5)                   */
#define R5_SIN2 0.58778525229247312917  /* sin(4π/5)                   */
#define R5_QA (-0.25)                   /* (C1+C2)/2 = -1/4           */
#define R5_QB 0.55901699437494742410    /* (C1-C2)/2 = √5/4           */

/* ================================================================== */
/*  Complex multiply helpers — 512-bit FMA                             */
/* ================================================================== */
#define R5_CMUL_512(ar, ai, wr, wi, tr, ti)                            \
    do                                                                 \
    {                                                                  \
        (ti) = _mm512_fmadd_pd((ar), (wi), _mm512_mul_pd((ai), (wr))); \
        (tr) = _mm512_fmsub_pd((ar), (wr), _mm512_mul_pd((ai), (wi))); \
    } while (0)

#define R5_CMULJ_512(ar, ai, wr, wi, tr, ti)                           \
    do                                                                 \
    {                                                                  \
        (tr) = _mm512_fmadd_pd((ai), (wi), _mm512_mul_pd((ar), (wr))); \
        (ti) = _mm512_fmsub_pd((ai), (wr), _mm512_mul_pd((ar), (wi))); \
    } while (0)

/* ================================================================== */
/*  Software prefetch — distance in doubles (64 = 512 bytes ahead)     */
/* ================================================================== */
#ifndef R5_PF_DIST
#define R5_PF_DIST 64
#endif

#define R5_PF(ptr, off) \
    _mm_prefetch((const char *)&(ptr)[(off) + R5_PF_DIST], _MM_HINT_T0)

#define R5_PREFETCH_N1(off) \
    do                      \
    {                       \
        R5_PF(a_re, off);   \
        R5_PF(a_im, off);   \
        R5_PF(b_re, off);   \
        R5_PF(b_im, off);   \
        R5_PF(c_re, off);   \
        R5_PF(c_im, off);   \
        R5_PF(d_re, off);   \
        R5_PF(d_im, off);   \
        R5_PF(e_re, off);   \
        R5_PF(e_im, off);   \
    } while (0)

#define R5_PREFETCH_TW(off)  \
    do                       \
    {                        \
        R5_PREFETCH_N1(off); \
        R5_PF(tw1_re, off);  \
        R5_PF(tw1_im, off);  \
        R5_PF(tw2_re, off);  \
        R5_PF(tw2_im, off);  \
    } while (0)

/* ================================================================== */
/*  Load + sum/diff — full 8-wide (unmasked)                           */
/* ================================================================== */
#define R5_LOAD_SD_512(P, off)                           \
    do                                                   \
    {                                                    \
        P##_ar = _mm512_loadu_pd(&a_re[(off)]);          \
        P##_ai = _mm512_loadu_pd(&a_im[(off)]);          \
        {                                                \
            __m512d _br = _mm512_loadu_pd(&b_re[(off)]); \
            __m512d _bi = _mm512_loadu_pd(&b_im[(off)]); \
            __m512d _er = _mm512_loadu_pd(&e_re[(off)]); \
            __m512d _ei = _mm512_loadu_pd(&e_im[(off)]); \
            P##_s1r = _mm512_add_pd(_br, _er);           \
            P##_s1i = _mm512_add_pd(_bi, _ei);           \
            P##_d1r = _mm512_sub_pd(_br, _er);           \
            P##_d1i = _mm512_sub_pd(_bi, _ei);           \
        }                                                \
        {                                                \
            __m512d _cr = _mm512_loadu_pd(&c_re[(off)]); \
            __m512d _ci = _mm512_loadu_pd(&c_im[(off)]); \
            __m512d _dr = _mm512_loadu_pd(&d_re[(off)]); \
            __m512d _di = _mm512_loadu_pd(&d_im[(off)]); \
            P##_s2r = _mm512_add_pd(_cr, _dr);           \
            P##_s2i = _mm512_add_pd(_ci, _di);           \
            P##_d2r = _mm512_sub_pd(_cr, _dr);           \
            P##_d2i = _mm512_sub_pd(_ci, _di);           \
        }                                                \
    } while (0)

/* Load + sum/diff — masked (1..7 valid lanes, rest zeroed) */
#define R5_LOAD_SD_512M(P, off, mask)                                  \
    do                                                                 \
    {                                                                  \
        P##_ar = _mm512_maskz_loadu_pd((mask), &a_re[(off)]);          \
        P##_ai = _mm512_maskz_loadu_pd((mask), &a_im[(off)]);          \
        {                                                              \
            __m512d _br = _mm512_maskz_loadu_pd((mask), &b_re[(off)]); \
            __m512d _bi = _mm512_maskz_loadu_pd((mask), &b_im[(off)]); \
            __m512d _er = _mm512_maskz_loadu_pd((mask), &e_re[(off)]); \
            __m512d _ei = _mm512_maskz_loadu_pd((mask), &e_im[(off)]); \
            P##_s1r = _mm512_add_pd(_br, _er);                         \
            P##_s1i = _mm512_add_pd(_bi, _ei);                         \
            P##_d1r = _mm512_sub_pd(_br, _er);                         \
            P##_d1i = _mm512_sub_pd(_bi, _ei);                         \
        }                                                              \
        {                                                              \
            __m512d _cr = _mm512_maskz_loadu_pd((mask), &c_re[(off)]); \
            __m512d _ci = _mm512_maskz_loadu_pd((mask), &c_im[(off)]); \
            __m512d _dr = _mm512_maskz_loadu_pd((mask), &d_re[(off)]); \
            __m512d _di = _mm512_maskz_loadu_pd((mask), &d_im[(off)]); \
            P##_s2r = _mm512_add_pd(_cr, _dr);                         \
            P##_s2i = _mm512_add_pd(_ci, _di);                         \
            P##_d2r = _mm512_sub_pd(_cr, _dr);                         \
            P##_d2i = _mm512_sub_pd(_ci, _di);                         \
        }                                                              \
    } while (0)

/* ================================================================== */
/*  Cosine chain — stores y0, produces t1/t2                           */
/*  Uses hoisted vQA, vQB (must be in scope)                           */
/*  Entry: 10 (ar,ai,s1×2,d1×2,s2×2,d2×2)                            */
/*  Exit:   8 (d1×2,d2×2,t1×2,t2×2)                                   */
/* ================================================================== */
#define R5_COSINE_CHAIN_512(P, off)                                     \
    do                                                                  \
    {                                                                   \
        __m512d P##_Ar = _mm512_add_pd(P##_s1r, P##_s2r);               \
        __m512d P##_Ai = _mm512_add_pd(P##_s1i, P##_s2i);               \
        _mm512_storeu_pd(&y0_re[(off)], _mm512_add_pd(P##_ar, P##_Ar)); \
        _mm512_storeu_pd(&y0_im[(off)], _mm512_add_pd(P##_ai, P##_Ai)); \
        __m512d P##_Br = _mm512_sub_pd(P##_s1r, P##_s2r);               \
        __m512d P##_Bi = _mm512_sub_pd(P##_s1i, P##_s2i);               \
        __m512d P##_comr = _mm512_fmadd_pd(vQA, P##_Ar, P##_ar);        \
        __m512d P##_comi = _mm512_fmadd_pd(vQA, P##_Ai, P##_ai);        \
        __m512d P##_mbr = _mm512_mul_pd(vQB, P##_Br);                   \
        __m512d P##_mbi = _mm512_mul_pd(vQB, P##_Bi);                   \
        P##_t1r = _mm512_add_pd(P##_comr, P##_mbr);                     \
        P##_t1i = _mm512_add_pd(P##_comi, P##_mbi);                     \
        P##_t2r = _mm512_sub_pd(P##_comr, P##_mbr);                     \
        P##_t2i = _mm512_sub_pd(P##_comi, P##_mbi);                     \
    } while (0)

/* Cosine chain — masked y0 stores */
#define R5_COSINE_CHAIN_512M(P, off, mask)                       \
    do                                                           \
    {                                                            \
        __m512d P##_Ar = _mm512_add_pd(P##_s1r, P##_s2r);        \
        __m512d P##_Ai = _mm512_add_pd(P##_s1i, P##_s2i);        \
        _mm512_mask_storeu_pd(&y0_re[(off)], (mask),             \
                              _mm512_add_pd(P##_ar, P##_Ar));    \
        _mm512_mask_storeu_pd(&y0_im[(off)], (mask),             \
                              _mm512_add_pd(P##_ai, P##_Ai));    \
        __m512d P##_Br = _mm512_sub_pd(P##_s1r, P##_s2r);        \
        __m512d P##_Bi = _mm512_sub_pd(P##_s1i, P##_s2i);        \
        __m512d P##_comr = _mm512_fmadd_pd(vQA, P##_Ar, P##_ar); \
        __m512d P##_comi = _mm512_fmadd_pd(vQA, P##_Ai, P##_ai); \
        __m512d P##_mbr = _mm512_mul_pd(vQB, P##_Br);            \
        __m512d P##_mbi = _mm512_mul_pd(vQB, P##_Bi);            \
        P##_t1r = _mm512_add_pd(P##_comr, P##_mbr);              \
        P##_t1i = _mm512_add_pd(P##_comi, P##_mbi);              \
        P##_t2r = _mm512_sub_pd(P##_comr, P##_mbr);              \
        P##_t2i = _mm512_sub_pd(P##_comi, P##_mbi);              \
    } while (0)

/* ================================================================== */
/*  Sine chain — register-only, no masked variant needed               */
/*  Uses hoisted vS1, vS2                                              */
/* ================================================================== */
#define R5_SINE_CHAIN_512(P)                                    \
    do                                                          \
    {                                                           \
        P##_v1r = _mm512_fmadd_pd(vS1, P##_d1r,                 \
                                  _mm512_mul_pd(vS2, P##_d2r)); \
        P##_v1i = _mm512_fmadd_pd(vS1, P##_d1i,                 \
                                  _mm512_mul_pd(vS2, P##_d2i)); \
        P##_v2r = _mm512_fmsub_pd(vS2, P##_d1r,                 \
                                  _mm512_mul_pd(vS1, P##_d2r)); \
        P##_v2i = _mm512_fmsub_pd(vS2, P##_d1i,                 \
                                  _mm512_mul_pd(vS1, P##_d2i)); \
    } while (0)

/* ================================================================== */
/*  Split stores — unmasked and masked variants                        */
/* ================================================================== */
#define R5_STORE_14_FWD_512(P, off)                                       \
    do                                                                    \
    {                                                                     \
        _mm512_storeu_pd(&y1_re[(off)], _mm512_add_pd(P##_t1r, P##_v1i)); \
        _mm512_storeu_pd(&y1_im[(off)], _mm512_sub_pd(P##_t1i, P##_v1r)); \
        _mm512_storeu_pd(&y4_re[(off)], _mm512_sub_pd(P##_t1r, P##_v1i)); \
        _mm512_storeu_pd(&y4_im[(off)], _mm512_add_pd(P##_t1i, P##_v1r)); \
    } while (0)
#define R5_STORE_23_FWD_512(P, off)                                       \
    do                                                                    \
    {                                                                     \
        _mm512_storeu_pd(&y2_re[(off)], _mm512_add_pd(P##_t2r, P##_v2i)); \
        _mm512_storeu_pd(&y2_im[(off)], _mm512_sub_pd(P##_t2i, P##_v2r)); \
        _mm512_storeu_pd(&y3_re[(off)], _mm512_sub_pd(P##_t2r, P##_v2i)); \
        _mm512_storeu_pd(&y3_im[(off)], _mm512_add_pd(P##_t2i, P##_v2r)); \
    } while (0)
#define R5_STORE_14_BWD_512(P, off)                                       \
    do                                                                    \
    {                                                                     \
        _mm512_storeu_pd(&y1_re[(off)], _mm512_sub_pd(P##_t1r, P##_v1i)); \
        _mm512_storeu_pd(&y1_im[(off)], _mm512_add_pd(P##_t1i, P##_v1r)); \
        _mm512_storeu_pd(&y4_re[(off)], _mm512_add_pd(P##_t1r, P##_v1i)); \
        _mm512_storeu_pd(&y4_im[(off)], _mm512_sub_pd(P##_t1i, P##_v1r)); \
    } while (0)
#define R5_STORE_23_BWD_512(P, off)                                       \
    do                                                                    \
    {                                                                     \
        _mm512_storeu_pd(&y2_re[(off)], _mm512_sub_pd(P##_t2r, P##_v2i)); \
        _mm512_storeu_pd(&y2_im[(off)], _mm512_add_pd(P##_t2i, P##_v2r)); \
        _mm512_storeu_pd(&y3_re[(off)], _mm512_add_pd(P##_t2r, P##_v2i)); \
        _mm512_storeu_pd(&y3_im[(off)], _mm512_sub_pd(P##_t2i, P##_v2r)); \
    } while (0)

/* Masked store variants */
#define R5_STORE_14_FWD_512M(P, off, mask)                      \
    do                                                          \
    {                                                           \
        _mm512_mask_storeu_pd(&y1_re[(off)], (mask),            \
                              _mm512_add_pd(P##_t1r, P##_v1i)); \
        _mm512_mask_storeu_pd(&y1_im[(off)], (mask),            \
                              _mm512_sub_pd(P##_t1i, P##_v1r)); \
        _mm512_mask_storeu_pd(&y4_re[(off)], (mask),            \
                              _mm512_sub_pd(P##_t1r, P##_v1i)); \
        _mm512_mask_storeu_pd(&y4_im[(off)], (mask),            \
                              _mm512_add_pd(P##_t1i, P##_v1r)); \
    } while (0)
#define R5_STORE_23_FWD_512M(P, off, mask)                      \
    do                                                          \
    {                                                           \
        _mm512_mask_storeu_pd(&y2_re[(off)], (mask),            \
                              _mm512_add_pd(P##_t2r, P##_v2i)); \
        _mm512_mask_storeu_pd(&y2_im[(off)], (mask),            \
                              _mm512_sub_pd(P##_t2i, P##_v2r)); \
        _mm512_mask_storeu_pd(&y3_re[(off)], (mask),            \
                              _mm512_sub_pd(P##_t2r, P##_v2i)); \
        _mm512_mask_storeu_pd(&y3_im[(off)], (mask),            \
                              _mm512_add_pd(P##_t2i, P##_v2r)); \
    } while (0)
#define R5_STORE_14_BWD_512M(P, off, mask)                      \
    do                                                          \
    {                                                           \
        _mm512_mask_storeu_pd(&y1_re[(off)], (mask),            \
                              _mm512_sub_pd(P##_t1r, P##_v1i)); \
        _mm512_mask_storeu_pd(&y1_im[(off)], (mask),            \
                              _mm512_add_pd(P##_t1i, P##_v1r)); \
        _mm512_mask_storeu_pd(&y4_re[(off)], (mask),            \
                              _mm512_add_pd(P##_t1r, P##_v1i)); \
        _mm512_mask_storeu_pd(&y4_im[(off)], (mask),            \
                              _mm512_sub_pd(P##_t1i, P##_v1r)); \
    } while (0)
#define R5_STORE_23_BWD_512M(P, off, mask)                      \
    do                                                          \
    {                                                           \
        _mm512_mask_storeu_pd(&y2_re[(off)], (mask),            \
                              _mm512_sub_pd(P##_t2r, P##_v2i)); \
        _mm512_mask_storeu_pd(&y2_im[(off)], (mask),            \
                              _mm512_add_pd(P##_t2i, P##_v2r)); \
        _mm512_mask_storeu_pd(&y3_re[(off)], (mask),            \
                              _mm512_add_pd(P##_t2r, P##_v2i)); \
        _mm512_mask_storeu_pd(&y3_im[(off)], (mask),            \
                              _mm512_sub_pd(P##_t2i, P##_v2r)); \
    } while (0)

/* ================================================================== */
/*  Twiddle load macros — full, partial-BE, deferred-CD, masked        */
/* ================================================================== */

/* Full twiddle + s/d: load W1,W2, derive W3,W4, apply to b,c,d,e */
#define R5_LOAD_TW_SD_512(P, off)                                \
    do                                                           \
    {                                                            \
        P##_ar = _mm512_loadu_pd(&a_re[(off)]);                  \
        P##_ai = _mm512_loadu_pd(&a_im[(off)]);                  \
        {                                                        \
            __m512d _w1r = _mm512_loadu_pd(&tw1_re[(off)]);      \
            __m512d _w1i = _mm512_loadu_pd(&tw1_im[(off)]);      \
            __m512d _w2r = _mm512_loadu_pd(&tw2_re[(off)]);      \
            __m512d _w2i = _mm512_loadu_pd(&tw2_im[(off)]);      \
            __m512d _tbr, _tbi;                                  \
            {                                                    \
                __m512d _br = _mm512_loadu_pd(&b_re[(off)]);     \
                __m512d _bi = _mm512_loadu_pd(&b_im[(off)]);     \
                R5_CMUL_512(_br, _bi, _w1r, _w1i, _tbr, _tbi);   \
            }                                                    \
            __m512d _tdr, _tdi;                                  \
            {                                                    \
                __m512d _w3r, _w3i;                              \
                R5_CMUL_512(_w1r, _w1i, _w2r, _w2i, _w3r, _w3i); \
                __m512d _dr = _mm512_loadu_pd(&d_re[(off)]);     \
                __m512d _di = _mm512_loadu_pd(&d_im[(off)]);     \
                R5_CMUL_512(_dr, _di, _w3r, _w3i, _tdr, _tdi);   \
            }                                                    \
            __m512d _tcr, _tci;                                  \
            {                                                    \
                __m512d _cr = _mm512_loadu_pd(&c_re[(off)]);     \
                __m512d _ci = _mm512_loadu_pd(&c_im[(off)]);     \
                R5_CMUL_512(_cr, _ci, _w2r, _w2i, _tcr, _tci);   \
            }                                                    \
            __m512d _ter, _tei;                                  \
            {                                                    \
                __m512d _w4r, _w4i;                              \
                R5_CMUL_512(_w2r, _w2i, _w2r, _w2i, _w4r, _w4i); \
                __m512d _er = _mm512_loadu_pd(&e_re[(off)]);     \
                __m512d _ei = _mm512_loadu_pd(&e_im[(off)]);     \
                R5_CMUL_512(_er, _ei, _w4r, _w4i, _ter, _tei);   \
            }                                                    \
            P##_s1r = _mm512_add_pd(_tbr, _ter);                 \
            P##_s1i = _mm512_add_pd(_tbi, _tei);                 \
            P##_d1r = _mm512_sub_pd(_tbr, _ter);                 \
            P##_d1i = _mm512_sub_pd(_tbi, _tei);                 \
            P##_s2r = _mm512_add_pd(_tcr, _tdr);                 \
            P##_s2i = _mm512_add_pd(_tci, _tdi);                 \
            P##_d2r = _mm512_sub_pd(_tcr, _tdr);                 \
            P##_d2i = _mm512_sub_pd(_tci, _tdi);                 \
        }                                                        \
    } while (0)

/* Full twiddle + s/d — masked */
#define R5_LOAD_TW_SD_512M(P, off, mask)                                   \
    do                                                                     \
    {                                                                      \
        P##_ar = _mm512_maskz_loadu_pd((mask), &a_re[(off)]);              \
        P##_ai = _mm512_maskz_loadu_pd((mask), &a_im[(off)]);              \
        {                                                                  \
            __m512d _w1r = _mm512_maskz_loadu_pd((mask), &tw1_re[(off)]);  \
            __m512d _w1i = _mm512_maskz_loadu_pd((mask), &tw1_im[(off)]);  \
            __m512d _w2r = _mm512_maskz_loadu_pd((mask), &tw2_re[(off)]);  \
            __m512d _w2i = _mm512_maskz_loadu_pd((mask), &tw2_im[(off)]);  \
            __m512d _tbr, _tbi;                                            \
            {                                                              \
                __m512d _br = _mm512_maskz_loadu_pd((mask), &b_re[(off)]); \
                __m512d _bi = _mm512_maskz_loadu_pd((mask), &b_im[(off)]); \
                R5_CMUL_512(_br, _bi, _w1r, _w1i, _tbr, _tbi);             \
            }                                                              \
            __m512d _tdr, _tdi;                                            \
            {                                                              \
                __m512d _w3r, _w3i;                                        \
                R5_CMUL_512(_w1r, _w1i, _w2r, _w2i, _w3r, _w3i);           \
                __m512d _dr = _mm512_maskz_loadu_pd((mask), &d_re[(off)]); \
                __m512d _di = _mm512_maskz_loadu_pd((mask), &d_im[(off)]); \
                R5_CMUL_512(_dr, _di, _w3r, _w3i, _tdr, _tdi);             \
            }                                                              \
            __m512d _tcr, _tci;                                            \
            {                                                              \
                __m512d _cr = _mm512_maskz_loadu_pd((mask), &c_re[(off)]); \
                __m512d _ci = _mm512_maskz_loadu_pd((mask), &c_im[(off)]); \
                R5_CMUL_512(_cr, _ci, _w2r, _w2i, _tcr, _tci);             \
            }                                                              \
            __m512d _ter, _tei;                                            \
            {                                                              \
                __m512d _w4r, _w4i;                                        \
                R5_CMUL_512(_w2r, _w2i, _w2r, _w2i, _w4r, _w4i);           \
                __m512d _er = _mm512_maskz_loadu_pd((mask), &e_re[(off)]); \
                __m512d _ei = _mm512_maskz_loadu_pd((mask), &e_im[(off)]); \
                R5_CMUL_512(_er, _ei, _w4r, _w4i, _ter, _tei);             \
            }                                                              \
            P##_s1r = _mm512_add_pd(_tbr, _ter);                           \
            P##_s1i = _mm512_add_pd(_tbi, _tei);                           \
            P##_d1r = _mm512_sub_pd(_tbr, _ter);                           \
            P##_d1i = _mm512_sub_pd(_tbi, _tei);                           \
            P##_s2r = _mm512_add_pd(_tcr, _tdr);                           \
            P##_s2i = _mm512_add_pd(_tci, _tdi);                           \
            P##_d2r = _mm512_sub_pd(_tcr, _tdr);                           \
            P##_d2i = _mm512_sub_pd(_tci, _tdi);                           \
        }                                                                  \
    } while (0)

/* Partial twiddle: only b,e → s1,d1 (for U=2 phase 2) */
#define R5_LOAD_TW_SD_PARTIAL_BE_512(P, off)                     \
    do                                                           \
    {                                                            \
        P##_ar = _mm512_loadu_pd(&a_re[(off)]);                  \
        P##_ai = _mm512_loadu_pd(&a_im[(off)]);                  \
        {                                                        \
            __m512d _w1r = _mm512_loadu_pd(&tw1_re[(off)]);      \
            __m512d _w1i = _mm512_loadu_pd(&tw1_im[(off)]);      \
            __m512d _w2r = _mm512_loadu_pd(&tw2_re[(off)]);      \
            __m512d _w2i = _mm512_loadu_pd(&tw2_im[(off)]);      \
            __m512d _tbr, _tbi;                                  \
            {                                                    \
                __m512d _br = _mm512_loadu_pd(&b_re[(off)]);     \
                __m512d _bi = _mm512_loadu_pd(&b_im[(off)]);     \
                R5_CMUL_512(_br, _bi, _w1r, _w1i, _tbr, _tbi);   \
            }                                                    \
            __m512d _ter, _tei;                                  \
            {                                                    \
                __m512d _w4r, _w4i;                              \
                R5_CMUL_512(_w2r, _w2i, _w2r, _w2i, _w4r, _w4i); \
                __m512d _er = _mm512_loadu_pd(&e_re[(off)]);     \
                __m512d _ei = _mm512_loadu_pd(&e_im[(off)]);     \
                R5_CMUL_512(_er, _ei, _w4r, _w4i, _ter, _tei);   \
            }                                                    \
            P##_s1r = _mm512_add_pd(_tbr, _ter);                 \
            P##_s1i = _mm512_add_pd(_tbi, _tei);                 \
            P##_d1r = _mm512_sub_pd(_tbr, _ter);                 \
            P##_d1i = _mm512_sub_pd(_tbi, _tei);                 \
        }                                                        \
    } while (0)

/* Deferred twiddle: c,d → s2,d2 (for U=2 phase 4, reloads W1,W2) */
#define R5_LOAD_TW_SD_DEFERRED_CD_512(P, off)                    \
    do                                                           \
    {                                                            \
        {                                                        \
            __m512d _w1r = _mm512_loadu_pd(&tw1_re[(off)]);      \
            __m512d _w1i = _mm512_loadu_pd(&tw1_im[(off)]);      \
            __m512d _w2r = _mm512_loadu_pd(&tw2_re[(off)]);      \
            __m512d _w2i = _mm512_loadu_pd(&tw2_im[(off)]);      \
            __m512d _tdr, _tdi;                                  \
            {                                                    \
                __m512d _w3r, _w3i;                              \
                R5_CMUL_512(_w1r, _w1i, _w2r, _w2i, _w3r, _w3i); \
                __m512d _dr = _mm512_loadu_pd(&d_re[(off)]);     \
                __m512d _di = _mm512_loadu_pd(&d_im[(off)]);     \
                R5_CMUL_512(_dr, _di, _w3r, _w3i, _tdr, _tdi);   \
            }                                                    \
            __m512d _tcr, _tci;                                  \
            {                                                    \
                __m512d _cr = _mm512_loadu_pd(&c_re[(off)]);     \
                __m512d _ci = _mm512_loadu_pd(&c_im[(off)]);     \
                R5_CMUL_512(_cr, _ci, _w2r, _w2i, _tcr, _tci);   \
            }                                                    \
            P##_s2r = _mm512_add_pd(_tcr, _tdr);                 \
            P##_s2i = _mm512_add_pd(_tci, _tdi);                 \
            P##_d2r = _mm512_sub_pd(_tcr, _tdr);                 \
            P##_d2i = _mm512_sub_pd(_tci, _tdi);                 \
        }                                                        \
    } while (0)

/* ================================================================== */
/*  Forward N1 — no twiddles, U=2 pipeline + masked tail               */
/* ================================================================== */

R5_BUTTERFLY_API __attribute__((target("avx512f"))) void radix5_wfta_fwd_avx512_N1(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    int K)
{
    const __m512d vQA = _mm512_set1_pd(R5_QA);
    const __m512d vQB = _mm512_set1_pd(R5_QB);
    const __m512d vS1 = _mm512_set1_pd(R5_SIN1);
    const __m512d vS2 = _mm512_set1_pd(R5_SIN2);

    int k = 0;

    /* ── U=2 main loop: 16 elements per iteration ── */
    for (; k + 15 < K; k += 16)
    {
        R5_PREFETCH_N1(k);

        /* Phase 1: A full load + cosine → A=8 */
        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD_512(A, k);
        R5_COSINE_CHAIN_512(A, k);

        /* Phase 2: B partial load → B=6, total=14+4=18 */
        __m512d B_ar, B_ai, B_s1r, B_s1i, B_d1r, B_d1i;
        B_ar = _mm512_loadu_pd(&a_re[k + 8]);
        B_ai = _mm512_loadu_pd(&a_im[k + 8]);
        {
            __m512d _br = _mm512_loadu_pd(&b_re[k + 8]);
            __m512d _bi = _mm512_loadu_pd(&b_im[k + 8]);
            __m512d _er = _mm512_loadu_pd(&e_re[k + 8]);
            __m512d _ei = _mm512_loadu_pd(&e_im[k + 8]);
            B_s1r = _mm512_add_pd(_br, _er);
            B_s1i = _mm512_add_pd(_bi, _ei);
            B_d1r = _mm512_sub_pd(_br, _er);
            B_d1i = _mm512_sub_pd(_bi, _ei);
        }

        /* Phase 3: A sine chain — FMAs overlap with B's L1 data */
        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);

        /* Phase 4: A stores y1,y4 → A=4.  B loads c,d → B=10 */
        R5_STORE_14_FWD_512(A, k);

        __m512d B_s2r, B_s2i, B_d2r, B_d2i;
        {
            __m512d _cr = _mm512_loadu_pd(&c_re[k + 8]);
            __m512d _ci = _mm512_loadu_pd(&c_im[k + 8]);
            __m512d _dr = _mm512_loadu_pd(&d_re[k + 8]);
            __m512d _di = _mm512_loadu_pd(&d_im[k + 8]);
            B_s2r = _mm512_add_pd(_cr, _dr);
            B_s2i = _mm512_add_pd(_ci, _di);
            B_d2r = _mm512_sub_pd(_cr, _dr);
            B_d2i = _mm512_sub_pd(_ci, _di);
        }

        /* Phase 5: A stores y2,y3 → A=0.  B computes solo */
        R5_STORE_23_FWD_512(A, k);

        __m512d B_t1r, B_t1i, B_t2r, B_t2i;
        R5_COSINE_CHAIN_512(B, k + 8);
        __m512d B_v1r, B_v1i, B_v2r, B_v2i;
        R5_SINE_CHAIN_512(B);
        R5_STORE_14_FWD_512(B, k + 8);
        R5_STORE_23_FWD_512(B, k + 8);
    }

    /* ── U=1 cleanup (at most one full 8-wide iteration) ── */
    if (k + 7 < K)
    {
        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD_512(A, k);
        R5_COSINE_CHAIN_512(A, k);
        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);
        R5_STORE_14_FWD_512(A, k);
        R5_STORE_23_FWD_512(A, k);
        k += 8;
    }

    /* ── Masked tail (1..7 elements) ── */
    if (k < K)
    {
        __mmask8 mask = (__mmask8)((1u << (K - k)) - 1);
        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD_512M(A, k, mask);
        R5_COSINE_CHAIN_512M(A, k, mask);
        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);
        R5_STORE_14_FWD_512M(A, k, mask);
        R5_STORE_23_FWD_512M(A, k, mask);
    }
}

/* ================================================================== */
/*  Backward N1 — no twiddles, U=2 pipeline + masked tail              */
/* ================================================================== */

R5_BUTTERFLY_API __attribute__((target("avx512f"))) void radix5_wfta_bwd_avx512_N1(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    int K)
{
    const __m512d vQA = _mm512_set1_pd(R5_QA);
    const __m512d vQB = _mm512_set1_pd(R5_QB);
    const __m512d vS1 = _mm512_set1_pd(R5_SIN1);
    const __m512d vS2 = _mm512_set1_pd(R5_SIN2);

    int k = 0;

    for (; k + 15 < K; k += 16)
    {
        R5_PREFETCH_N1(k);

        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD_512(A, k);
        R5_COSINE_CHAIN_512(A, k);

        __m512d B_ar, B_ai, B_s1r, B_s1i, B_d1r, B_d1i;
        B_ar = _mm512_loadu_pd(&a_re[k + 8]);
        B_ai = _mm512_loadu_pd(&a_im[k + 8]);
        {
            __m512d _br = _mm512_loadu_pd(&b_re[k + 8]);
            __m512d _bi = _mm512_loadu_pd(&b_im[k + 8]);
            __m512d _er = _mm512_loadu_pd(&e_re[k + 8]);
            __m512d _ei = _mm512_loadu_pd(&e_im[k + 8]);
            B_s1r = _mm512_add_pd(_br, _er);
            B_s1i = _mm512_add_pd(_bi, _ei);
            B_d1r = _mm512_sub_pd(_br, _er);
            B_d1i = _mm512_sub_pd(_bi, _ei);
        }

        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);
        R5_STORE_14_BWD_512(A, k);

        __m512d B_s2r, B_s2i, B_d2r, B_d2i;
        {
            __m512d _cr = _mm512_loadu_pd(&c_re[k + 8]);
            __m512d _ci = _mm512_loadu_pd(&c_im[k + 8]);
            __m512d _dr = _mm512_loadu_pd(&d_re[k + 8]);
            __m512d _di = _mm512_loadu_pd(&d_im[k + 8]);
            B_s2r = _mm512_add_pd(_cr, _dr);
            B_s2i = _mm512_add_pd(_ci, _di);
            B_d2r = _mm512_sub_pd(_cr, _dr);
            B_d2i = _mm512_sub_pd(_ci, _di);
        }

        R5_STORE_23_BWD_512(A, k);

        __m512d B_t1r, B_t1i, B_t2r, B_t2i;
        R5_COSINE_CHAIN_512(B, k + 8);
        __m512d B_v1r, B_v1i, B_v2r, B_v2i;
        R5_SINE_CHAIN_512(B);
        R5_STORE_14_BWD_512(B, k + 8);
        R5_STORE_23_BWD_512(B, k + 8);
    }

    if (k + 7 < K)
    {
        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD_512(A, k);
        R5_COSINE_CHAIN_512(A, k);
        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);
        R5_STORE_14_BWD_512(A, k);
        R5_STORE_23_BWD_512(A, k);
        k += 8;
    }

    if (k < K)
    {
        __mmask8 mask = (__mmask8)((1u << (K - k)) - 1);
        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD_512M(A, k, mask);
        R5_COSINE_CHAIN_512M(A, k, mask);
        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);
        R5_STORE_14_BWD_512M(A, k, mask);
        R5_STORE_23_BWD_512M(A, k, mask);
    }
}

/* ================================================================== */
/*  Forward twiddled — BLOCKED2, U=2 + masked tail                     */
/* ================================================================== */

R5_BUTTERFLY_API __attribute__((target("avx512f"))) void radix5_wfta_fwd_avx512(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    const double *restrict tw1_re, const double *restrict tw1_im,
    const double *restrict tw2_re, const double *restrict tw2_im,
    int K)
{
    const __m512d vQA = _mm512_set1_pd(R5_QA);
    const __m512d vQB = _mm512_set1_pd(R5_QB);
    const __m512d vS1 = _mm512_set1_pd(R5_SIN1);
    const __m512d vS2 = _mm512_set1_pd(R5_SIN2);

    int k = 0;

    for (; k + 15 < K; k += 16)
    {
        R5_PREFETCH_TW(k);

        /* A: full twiddle + s/d + cosine → A=8 */
        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_TW_SD_512(A, k);
        R5_COSINE_CHAIN_512(A, k);

        /* B partial: a,b,e twiddle → s1,d1 */
        __m512d B_ar, B_ai, B_s1r, B_s1i, B_d1r, B_d1i;
        R5_LOAD_TW_SD_PARTIAL_BE_512(B, k + 8);

        /* A sine chain */
        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);

        /* A stores y1,y4.  B deferred c,d twiddle */
        R5_STORE_14_FWD_512(A, k);

        __m512d B_s2r, B_s2i, B_d2r, B_d2i;
        R5_LOAD_TW_SD_DEFERRED_CD_512(B, k + 8);

        /* A stores y2,y3.  B solo. */
        R5_STORE_23_FWD_512(A, k);

        __m512d B_t1r, B_t1i, B_t2r, B_t2i;
        R5_COSINE_CHAIN_512(B, k + 8);
        __m512d B_v1r, B_v1i, B_v2r, B_v2i;
        R5_SINE_CHAIN_512(B);
        R5_STORE_14_FWD_512(B, k + 8);
        R5_STORE_23_FWD_512(B, k + 8);
    }

    /* U=1 cleanup */
    if (k + 7 < K)
    {
        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_TW_SD_512(A, k);
        R5_COSINE_CHAIN_512(A, k);
        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);
        R5_STORE_14_FWD_512(A, k);
        R5_STORE_23_FWD_512(A, k);
        k += 8;
    }

    /* Masked tail */
    if (k < K)
    {
        __mmask8 mask = (__mmask8)((1u << (K - k)) - 1);
        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_TW_SD_512M(A, k, mask);
        R5_COSINE_CHAIN_512M(A, k, mask);
        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);
        R5_STORE_14_FWD_512M(A, k, mask);
        R5_STORE_23_FWD_512M(A, k, mask);
    }
}

/* ================================================================== */
/*  Backward twiddled — IDFT-5 → post-conj-twiddle, U=2 + masked tail */
/*                                                                     */
/*  U=2 interleaving (AVX-512 only — impossible on AVX2):             */
/*    A: IDFT-5 → raw r1..r4 (8 regs)                                */
/*    A: load W1,W2                     (A=12)                        */
/*    B: load a,b,e → s1,d1             (B=6, total=22/32)           */
/*    A: y1=r1·conj(W1) store, W3, y3 store  (A drains → 4)          */
/*    B: load c,d → s2,d2               (B=10, total=18/32)          */
/*    A: y2 store, W4, y4 store         (A→0)                        */
/*    B: IDFT-5 + post-conj-twiddle solo                              */
/* ================================================================== */

R5_BUTTERFLY_API __attribute__((target("avx512f"))) void radix5_wfta_bwd_avx512(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    const double *restrict tw1_re, const double *restrict tw1_im,
    const double *restrict tw2_re, const double *restrict tw2_im,
    int K)
{
    const __m512d vQA = _mm512_set1_pd(R5_QA);
    const __m512d vQB = _mm512_set1_pd(R5_QB);
    const __m512d vS1 = _mm512_set1_pd(R5_SIN1);
    const __m512d vS2 = _mm512_set1_pd(R5_SIN2);

    int k = 0;

    for (; k + 15 < K; k += 16)
    {
        R5_PREFETCH_TW(k);

        /* ═══ A: IDFT-5 core → raw r1..r4 ═══ */
        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD_512(A, k);
        R5_COSINE_CHAIN_512(A, k); /* stores y0 (DC, no twiddle) */

        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);

        /* Backward +i → raw IDFT outputs */
        __m512d A_r1r = _mm512_sub_pd(A_t1r, A_v1i);
        __m512d A_r1i = _mm512_add_pd(A_t1i, A_v1r);
        __m512d A_r4r = _mm512_add_pd(A_t1r, A_v1i);
        __m512d A_r4i = _mm512_sub_pd(A_t1i, A_v1r);
        __m512d A_r2r = _mm512_sub_pd(A_t2r, A_v2i);
        __m512d A_r2i = _mm512_add_pd(A_t2i, A_v2r);
        __m512d A_r3r = _mm512_add_pd(A_t2r, A_v2i);
        __m512d A_r3i = _mm512_sub_pd(A_t2i, A_v2r);
        /* A = 8 (r1..r4).  t,v dead.  +4 const = 12 total. */

        /* ═══ A load W1,W2 + B partial load ═══ */
        __m512d w1r = _mm512_loadu_pd(&tw1_re[k]);
        __m512d w1i = _mm512_loadu_pd(&tw1_im[k]);
        __m512d w2r = _mm512_loadu_pd(&tw2_re[k]);
        __m512d w2i = _mm512_loadu_pd(&tw2_im[k]);
        /* A = 12.  Total = 16. */

        __m512d B_ar, B_ai, B_s1r, B_s1i, B_d1r, B_d1i;
        B_ar = _mm512_loadu_pd(&a_re[k + 8]);
        B_ai = _mm512_loadu_pd(&a_im[k + 8]);
        {
            __m512d _br = _mm512_loadu_pd(&b_re[k + 8]);
            __m512d _bi = _mm512_loadu_pd(&b_im[k + 8]);
            __m512d _er = _mm512_loadu_pd(&e_re[k + 8]);
            __m512d _ei = _mm512_loadu_pd(&e_im[k + 8]);
            B_s1r = _mm512_add_pd(_br, _er);
            B_s1i = _mm512_add_pd(_bi, _ei);
            B_d1r = _mm512_sub_pd(_br, _er);
            B_d1i = _mm512_sub_pd(_bi, _ei);
        }
        /* A=12, B=6 → 18+4 = 22 total ✓ */

        /* ═══ A: y1=r1·conj(W1); W3=W1·W2, y3=r3·conj(W3) ═══ */
        {
            __m512d o1r, o1i;
            R5_CMULJ_512(A_r1r, A_r1i, w1r, w1i, o1r, o1i);
            _mm512_storeu_pd(&y1_re[k], o1r);
            _mm512_storeu_pd(&y1_im[k], o1i);
        }
        {
            __m512d w3r, w3i;
            R5_CMUL_512(w1r, w1i, w2r, w2i, w3r, w3i);
            __m512d o3r, o3i;
            R5_CMULJ_512(A_r3r, A_r3i, w3r, w3i, o3r, o3i);
            _mm512_storeu_pd(&y3_re[k], o3r);
            _mm512_storeu_pd(&y3_im[k], o3i);
        }
        /* r1,r3,W1,W3 dead.  A = 6 (r2,r4,W2).  B=6.  Total 16. */

        /* ═══ B loads c,d → s2,d2 ═══ */
        __m512d B_s2r, B_s2i, B_d2r, B_d2i;
        {
            __m512d _cr = _mm512_loadu_pd(&c_re[k + 8]);
            __m512d _ci = _mm512_loadu_pd(&c_im[k + 8]);
            __m512d _dr = _mm512_loadu_pd(&d_re[k + 8]);
            __m512d _di = _mm512_loadu_pd(&d_im[k + 8]);
            B_s2r = _mm512_add_pd(_cr, _dr);
            B_s2i = _mm512_add_pd(_ci, _di);
            B_d2r = _mm512_sub_pd(_cr, _dr);
            B_d2i = _mm512_sub_pd(_ci, _di);
        }
        /* A=6, B=10 → 20 total */

        /* ═══ A: y2=r2·conj(W2); W4=W2², y4=r4·conj(W4) ═══ */
        {
            __m512d o2r, o2i;
            R5_CMULJ_512(A_r2r, A_r2i, w2r, w2i, o2r, o2i);
            _mm512_storeu_pd(&y2_re[k], o2r);
            _mm512_storeu_pd(&y2_im[k], o2i);
        }
        {
            __m512d w4r, w4i;
            R5_CMUL_512(w2r, w2i, w2r, w2i, w4r, w4i);
            __m512d o4r, o4i;
            R5_CMULJ_512(A_r4r, A_r4i, w4r, w4i, o4r, o4i);
            _mm512_storeu_pd(&y4_re[k], o4r);
            _mm512_storeu_pd(&y4_im[k], o4i);
        }
        /* A = 0.  B = 10 + 4 const = 14 total. */

        /* ═══ B: solo IDFT-5 + post-conj-twiddle ═══ */
        __m512d B_t1r, B_t1i, B_t2r, B_t2i;
        R5_COSINE_CHAIN_512(B, k + 8);

        __m512d B_v1r, B_v1i, B_v2r, B_v2i;
        R5_SINE_CHAIN_512(B);

        __m512d B_r1r = _mm512_sub_pd(B_t1r, B_v1i);
        __m512d B_r1i = _mm512_add_pd(B_t1i, B_v1r);
        __m512d B_r4r = _mm512_add_pd(B_t1r, B_v1i);
        __m512d B_r4i = _mm512_sub_pd(B_t1i, B_v1r);
        __m512d B_r2r = _mm512_sub_pd(B_t2r, B_v2i);
        __m512d B_r2i = _mm512_add_pd(B_t2i, B_v2r);
        __m512d B_r3r = _mm512_add_pd(B_t2r, B_v2i);
        __m512d B_r3i = _mm512_sub_pd(B_t2i, B_v2r);

        {
            __m512d bw1r = _mm512_loadu_pd(&tw1_re[k + 8]);
            __m512d bw1i = _mm512_loadu_pd(&tw1_im[k + 8]);
            __m512d bw2r = _mm512_loadu_pd(&tw2_re[k + 8]);
            __m512d bw2i = _mm512_loadu_pd(&tw2_im[k + 8]);

            __m512d o1r, o1i;
            R5_CMULJ_512(B_r1r, B_r1i, bw1r, bw1i, o1r, o1i);
            _mm512_storeu_pd(&y1_re[k + 8], o1r);
            _mm512_storeu_pd(&y1_im[k + 8], o1i);

            __m512d bw3r, bw3i;
            R5_CMUL_512(bw1r, bw1i, bw2r, bw2i, bw3r, bw3i);
            __m512d o3r, o3i;
            R5_CMULJ_512(B_r3r, B_r3i, bw3r, bw3i, o3r, o3i);
            _mm512_storeu_pd(&y3_re[k + 8], o3r);
            _mm512_storeu_pd(&y3_im[k + 8], o3i);

            __m512d o2r, o2i;
            R5_CMULJ_512(B_r2r, B_r2i, bw2r, bw2i, o2r, o2i);
            _mm512_storeu_pd(&y2_re[k + 8], o2r);
            _mm512_storeu_pd(&y2_im[k + 8], o2i);

            __m512d bw4r, bw4i;
            R5_CMUL_512(bw2r, bw2i, bw2r, bw2i, bw4r, bw4i);
            __m512d o4r, o4i;
            R5_CMULJ_512(B_r4r, B_r4i, bw4r, bw4i, o4r, o4i);
            _mm512_storeu_pd(&y4_re[k + 8], o4r);
            _mm512_storeu_pd(&y4_im[k + 8], o4i);
        }
    }

    /* ── U=1 cleanup ── */
    if (k + 7 < K)
    {
        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD_512(A, k);
        R5_COSINE_CHAIN_512(A, k);

        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);

        __m512d r1r = _mm512_sub_pd(A_t1r, A_v1i);
        __m512d r1i = _mm512_add_pd(A_t1i, A_v1r);
        __m512d r4r = _mm512_add_pd(A_t1r, A_v1i);
        __m512d r4i = _mm512_sub_pd(A_t1i, A_v1r);
        __m512d r2r = _mm512_sub_pd(A_t2r, A_v2i);
        __m512d r2i = _mm512_add_pd(A_t2i, A_v2r);
        __m512d r3r = _mm512_add_pd(A_t2r, A_v2i);
        __m512d r3i = _mm512_sub_pd(A_t2i, A_v2r);

        __m512d w1r = _mm512_loadu_pd(&tw1_re[k]);
        __m512d w1i = _mm512_loadu_pd(&tw1_im[k]);
        __m512d w2r = _mm512_loadu_pd(&tw2_re[k]);
        __m512d w2i = _mm512_loadu_pd(&tw2_im[k]);

        {
            __m512d o1r, o1i;
            R5_CMULJ_512(r1r, r1i, w1r, w1i, o1r, o1i);
            _mm512_storeu_pd(&y1_re[k], o1r);
            _mm512_storeu_pd(&y1_im[k], o1i);
        }

        {
            __m512d w3r, w3i;
            R5_CMUL_512(w1r, w1i, w2r, w2i, w3r, w3i);
            __m512d o3r, o3i;
            R5_CMULJ_512(r3r, r3i, w3r, w3i, o3r, o3i);
            _mm512_storeu_pd(&y3_re[k], o3r);
            _mm512_storeu_pd(&y3_im[k], o3i);
        }

        {
            __m512d o2r, o2i;
            R5_CMULJ_512(r2r, r2i, w2r, w2i, o2r, o2i);
            _mm512_storeu_pd(&y2_re[k], o2r);
            _mm512_storeu_pd(&y2_im[k], o2i);
        }

        {
            __m512d w4r, w4i;
            R5_CMUL_512(w2r, w2i, w2r, w2i, w4r, w4i);
            __m512d o4r, o4i;
            R5_CMULJ_512(r4r, r4i, w4r, w4i, o4r, o4i);
            _mm512_storeu_pd(&y4_re[k], o4r);
            _mm512_storeu_pd(&y4_im[k], o4i);
        }

        k += 8;
    }

    /* ── Masked tail ── */
    if (k < K)
    {
        __mmask8 mask = (__mmask8)((1u << (K - k)) - 1);

        __m512d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m512d A_s2r, A_s2i, A_d2r, A_d2i;
        __m512d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD_512M(A, k, mask);
        R5_COSINE_CHAIN_512M(A, k, mask);

        __m512d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN_512(A);

        __m512d r1r = _mm512_sub_pd(A_t1r, A_v1i);
        __m512d r1i = _mm512_add_pd(A_t1i, A_v1r);
        __m512d r4r = _mm512_add_pd(A_t1r, A_v1i);
        __m512d r4i = _mm512_sub_pd(A_t1i, A_v1r);
        __m512d r2r = _mm512_sub_pd(A_t2r, A_v2i);
        __m512d r2i = _mm512_add_pd(A_t2i, A_v2r);
        __m512d r3r = _mm512_add_pd(A_t2r, A_v2i);
        __m512d r3i = _mm512_sub_pd(A_t2i, A_v2r);

        __m512d w1r = _mm512_maskz_loadu_pd(mask, &tw1_re[k]);
        __m512d w1i = _mm512_maskz_loadu_pd(mask, &tw1_im[k]);
        __m512d w2r = _mm512_maskz_loadu_pd(mask, &tw2_re[k]);
        __m512d w2i = _mm512_maskz_loadu_pd(mask, &tw2_im[k]);

        {
            __m512d o1r, o1i;
            R5_CMULJ_512(r1r, r1i, w1r, w1i, o1r, o1i);
            _mm512_mask_storeu_pd(&y1_re[k], mask, o1r);
            _mm512_mask_storeu_pd(&y1_im[k], mask, o1i);
        }

        {
            __m512d w3r, w3i;
            R5_CMUL_512(w1r, w1i, w2r, w2i, w3r, w3i);
            __m512d o3r, o3i;
            R5_CMULJ_512(r3r, r3i, w3r, w3i, o3r, o3i);
            _mm512_mask_storeu_pd(&y3_re[k], mask, o3r);
            _mm512_mask_storeu_pd(&y3_im[k], mask, o3i);
        }

        {
            __m512d o2r, o2i;
            R5_CMULJ_512(r2r, r2i, w2r, w2i, o2r, o2i);
            _mm512_mask_storeu_pd(&y2_re[k], mask, o2r);
            _mm512_mask_storeu_pd(&y2_im[k], mask, o2i);
        }

        {
            __m512d w4r, w4i;
            R5_CMUL_512(w2r, w2i, w2r, w2i, w4r, w4i);
            __m512d o4r, o4i;
            R5_CMULJ_512(r4r, r4i, w4r, w4i, o4r, o4i);
            _mm512_mask_storeu_pd(&y4_re[k], mask, o4r);
            _mm512_mask_storeu_pd(&y4_im[k], mask, o4i);
        }
    }
}

#endif /* FFT_RADIX5_AVX512_H */