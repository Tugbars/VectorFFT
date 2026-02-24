/**
 * @file  fft_radix5_avx2.h
 * @brief Radix-5 FFT butterfly using Winograd (WFTA) — AVX2 (4-wide), U=2 pipeline
 *
 * ALGORITHM: Direct closed-form DFT-5 exploiting conjugate symmetry of roots.
 *   4 real constants (cos/sin of 2π/5 and 4π/5) give all 5 outputs.
 *   Factored cosine terms: QA=(C1+C2)/2=-1/4,  QB=(C1-C2)/2=√5/4
 *
 * U=2 PIPELINE (zero-spill schedule):
 * ═══════════════════════════════════
 *   Phase 1: A load+s/d+cosine chain → t1/t2      (A=8 regs)
 *   Phase 2: B partial load (a,b,e → s1,d1)        (A=8, B=6 → 14)
 *   Phase 3: A sine chain (v1,v2 FMAs)             (A=8, B=6 → 14)
 *   Phase 4: A stores y1/y4 + B loads c,d → s2/d2  (A=4, B=10 → 14)
 *   Phase 5: A stores y2/y3, B computes solo        (A=0, B≤12)
 *
 *   Peak register pressure: 14/16 YMM — zero stack spills.
 *   Constants via memory-broadcast operands (vbroadcastsd from .rodata).
 *
 * TWIDDLE LAYOUT: BLOCKED2
 *   tw1[k] = W^k, tw2[k] = W^{2k}. W3=W1·W2, W4=W2² derived inline.
 *
 * FUNCTIONS:
 *   radix5_wfta_fwd_avx2      — forward, twiddled (BLOCKED2)
 *   radix5_wfta_bwd_avx2      — backward, twiddled (IDFT-5 → post-conj-twiddle)
 *   radix5_wfta_fwd_avx2_N1   — forward, no twiddles (K=1 stage-0)
 *   radix5_wfta_bwd_avx2_N1   — backward, no twiddles
 */

#ifndef FFT_RADIX5_AVX2_H
#define FFT_RADIX5_AVX2_H

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

/* Memory-broadcast constant — compiler emits vbroadcastsd, zero reg cost */
#define R5_BCAST(val) _mm256_set1_pd(val)

/* ================================================================== */
/*  Complex multiply helpers (FMA)                                     */
/* ================================================================== */
#define R5_CMUL(ar, ai, wr, wi, tr, ti)                                \
    do                                                                 \
    {                                                                  \
        (ti) = _mm256_fmadd_pd((ar), (wi), _mm256_mul_pd((ai), (wr))); \
        (tr) = _mm256_fmsub_pd((ar), (wr), _mm256_mul_pd((ai), (wi))); \
    } while (0)

#define R5_CMULJ(ar, ai, wr, wi, tr, ti)                               \
    do                                                                 \
    {                                                                  \
        (tr) = _mm256_fmadd_pd((ai), (wi), _mm256_mul_pd((ar), (wr))); \
        (ti) = _mm256_fmsub_pd((ai), (wr), _mm256_mul_pd((ar), (wi))); \
    } while (0)

/* ================================================================== */
/*  Scalar helpers (tails + K=1)                                       */
/* ================================================================== */

static inline void r5_scalar_cmul(double ar, double ai, double wr, double wi,
                                  double *tr, double *ti)
{
    *tr = ar * wr - ai * wi;
    *ti = ar * wi + ai * wr;
}

static inline void r5_scalar_cmulj(double ar, double ai, double wr, double wi,
                                   double *tr, double *ti)
{
    *tr = ar * wr + ai * wi;
    *ti = ai * wr - ar * wi;
}

static inline void r5_scalar_core_fwd(
    double ar, double ai, double br, double bi,
    double cr, double ci, double dr, double di,
    double er, double ei,
    double *y0r, double *y0i, double *y1r, double *y1i,
    double *y2r, double *y2i, double *y3r, double *y3i,
    double *y4r, double *y4i)
{
    double s1r = br + er, s1i = bi + ei;
    double d1r = br - er, d1i = bi - ei;
    double s2r = cr + dr, s2i = ci + di;
    double d2r = cr - dr, d2i = ci - di;
    double Ar = s1r + s2r, Ai = s1i + s2i;
    *y0r = ar + Ar;
    *y0i = ai + Ai;
    double Br = s1r - s2r, Bi = s1i - s2i;
    double comr = ar + R5_QA * Ar, comi = ai + R5_QA * Ai;
    double mbr = R5_QB * Br, mbi = R5_QB * Bi;
    double t1r = comr + mbr, t1i = comi + mbi;
    double t2r = comr - mbr, t2i = comi - mbi;
    double v1r = R5_SIN1 * d1r + R5_SIN2 * d2r;
    double v1i = R5_SIN1 * d1i + R5_SIN2 * d2i;
    double v2r = R5_SIN2 * d1r - R5_SIN1 * d2r;
    double v2i = R5_SIN2 * d1i - R5_SIN1 * d2i;
    *y1r = t1r + v1i;
    *y1i = t1i - v1r;
    *y4r = t1r - v1i;
    *y4i = t1i + v1r;
    *y2r = t2r + v2i;
    *y2i = t2i - v2r;
    *y3r = t2r - v2i;
    *y3i = t2i + v2r;
}

static inline void r5_scalar_core_bwd(
    double ar, double ai, double br, double bi,
    double cr, double ci, double dr, double di,
    double er, double ei,
    double *y0r, double *y0i, double *y1r, double *y1i,
    double *y2r, double *y2i, double *y3r, double *y3i,
    double *y4r, double *y4i)
{
    double s1r = br + er, s1i = bi + ei;
    double d1r = br - er, d1i = bi - ei;
    double s2r = cr + dr, s2i = ci + di;
    double d2r = cr - dr, d2i = ci - di;
    double Ar = s1r + s2r, Ai = s1i + s2i;
    *y0r = ar + Ar;
    *y0i = ai + Ai;
    double Br = s1r - s2r, Bi = s1i - s2i;
    double comr = ar + R5_QA * Ar, comi = ai + R5_QA * Ai;
    double mbr = R5_QB * Br, mbi = R5_QB * Bi;
    double t1r = comr + mbr, t1i = comi + mbi;
    double t2r = comr - mbr, t2i = comi - mbi;
    double v1r = R5_SIN1 * d1r + R5_SIN2 * d2r;
    double v1i = R5_SIN1 * d1i + R5_SIN2 * d2i;
    double v2r = R5_SIN2 * d1r - R5_SIN1 * d2r;
    double v2i = R5_SIN2 * d1i - R5_SIN1 * d2i;
    *y1r = t1r - v1i;
    *y1i = t1i + v1r;
    *y4r = t1r + v1i;
    *y4i = t1i - v1r;
    *y2r = t2r - v2i;
    *y2i = t2i + v2r;
    *y3r = t2r + v2i;
    *y3i = t2i - v2r;
}

/* ================================================================== */
/*  Building-block macros for U=2 scheduling                           */
/* ================================================================== */

/* Load 5 legs → s/d pairs.  Produces 10 register-variables:          */
/*   P_ar, P_ai, P_s1r, P_s1i, P_d1r, P_d1i,                        */
/*   P_s2r, P_s2i, P_d2r, P_d2i                                      */
#define R5_LOAD_SD(P, off)                               \
    do                                                   \
    {                                                    \
        P##_ar = _mm256_loadu_pd(&a_re[(off)]);          \
        P##_ai = _mm256_loadu_pd(&a_im[(off)]);          \
        {                                                \
            __m256d _br = _mm256_loadu_pd(&b_re[(off)]); \
            __m256d _bi = _mm256_loadu_pd(&b_im[(off)]); \
            __m256d _er = _mm256_loadu_pd(&e_re[(off)]); \
            __m256d _ei = _mm256_loadu_pd(&e_im[(off)]); \
            P##_s1r = _mm256_add_pd(_br, _er);           \
            P##_s1i = _mm256_add_pd(_bi, _ei);           \
            P##_d1r = _mm256_sub_pd(_br, _er);           \
            P##_d1i = _mm256_sub_pd(_bi, _ei);           \
        }                                                \
        {                                                \
            __m256d _cr = _mm256_loadu_pd(&c_re[(off)]); \
            __m256d _ci = _mm256_loadu_pd(&c_im[(off)]); \
            __m256d _dr = _mm256_loadu_pd(&d_re[(off)]); \
            __m256d _di = _mm256_loadu_pd(&d_im[(off)]); \
            P##_s2r = _mm256_add_pd(_cr, _dr);           \
            P##_s2i = _mm256_add_pd(_ci, _di);           \
            P##_d2r = _mm256_sub_pd(_cr, _dr);           \
            P##_d2i = _mm256_sub_pd(_ci, _di);           \
        }                                                \
    } while (0)

/* Cosine chain: s/d → Ar,y0(stored),Br,com,mb,t1,t2                  */
/* Entry: 10 (ar,ai,s1×2,d1×2,s2×2,d2×2)                             */
/* Exit:   8 (d1×2,d2×2,t1×2,t2×2)  — ar,ai,s1,s2 consumed           */
#define R5_COSINE_CHAIN(P, off)                                              \
    do                                                                       \
    {                                                                        \
        __m256d P##_Ar = _mm256_add_pd(P##_s1r, P##_s2r);                    \
        __m256d P##_Ai = _mm256_add_pd(P##_s1i, P##_s2i);                    \
        _mm256_storeu_pd(&y0_re[(off)], _mm256_add_pd(P##_ar, P##_Ar));      \
        _mm256_storeu_pd(&y0_im[(off)], _mm256_add_pd(P##_ai, P##_Ai));      \
        __m256d P##_Br = _mm256_sub_pd(P##_s1r, P##_s2r);                    \
        __m256d P##_Bi = _mm256_sub_pd(P##_s1i, P##_s2i);                    \
        __m256d P##_comr = _mm256_fmadd_pd(R5_BCAST(R5_QA), P##_Ar, P##_ar); \
        __m256d P##_comi = _mm256_fmadd_pd(R5_BCAST(R5_QA), P##_Ai, P##_ai); \
        __m256d P##_mbr = _mm256_mul_pd(R5_BCAST(R5_QB), P##_Br);            \
        __m256d P##_mbi = _mm256_mul_pd(R5_BCAST(R5_QB), P##_Bi);            \
        P##_t1r = _mm256_add_pd(P##_comr, P##_mbr);                          \
        P##_t1i = _mm256_add_pd(P##_comi, P##_mbi);                          \
        P##_t2r = _mm256_sub_pd(P##_comr, P##_mbr);                          \
        P##_t2i = _mm256_sub_pd(P##_comi, P##_mbi);                          \
    } while (0)

/* Sine chain: d1/d2 → v1/v2 (d1,d2 conceptually dead after)          */
/* Register count unchanged (4 in, 4 out)                              */
#define R5_SINE_CHAIN(P)                                                      \
    do                                                                        \
    {                                                                         \
        P##_v1r = _mm256_fmadd_pd(R5_BCAST(R5_SIN1), P##_d1r,                 \
                                  _mm256_mul_pd(R5_BCAST(R5_SIN2), P##_d2r)); \
        P##_v1i = _mm256_fmadd_pd(R5_BCAST(R5_SIN1), P##_d1i,                 \
                                  _mm256_mul_pd(R5_BCAST(R5_SIN2), P##_d2i)); \
        P##_v2r = _mm256_fmsub_pd(R5_BCAST(R5_SIN2), P##_d1r,                 \
                                  _mm256_mul_pd(R5_BCAST(R5_SIN1), P##_d2r)); \
        P##_v2i = _mm256_fmsub_pd(R5_BCAST(R5_SIN2), P##_d1i,                 \
                                  _mm256_mul_pd(R5_BCAST(R5_SIN1), P##_d2i)); \
    } while (0)

/* Split stores: y1/y4 pair, then y2/y3 pair — enables interleaving */
#define R5_STORE_14_FWD(P, off)                                           \
    do                                                                    \
    {                                                                     \
        _mm256_storeu_pd(&y1_re[(off)], _mm256_add_pd(P##_t1r, P##_v1i)); \
        _mm256_storeu_pd(&y1_im[(off)], _mm256_sub_pd(P##_t1i, P##_v1r)); \
        _mm256_storeu_pd(&y4_re[(off)], _mm256_sub_pd(P##_t1r, P##_v1i)); \
        _mm256_storeu_pd(&y4_im[(off)], _mm256_add_pd(P##_t1i, P##_v1r)); \
    } while (0)
#define R5_STORE_23_FWD(P, off)                                           \
    do                                                                    \
    {                                                                     \
        _mm256_storeu_pd(&y2_re[(off)], _mm256_add_pd(P##_t2r, P##_v2i)); \
        _mm256_storeu_pd(&y2_im[(off)], _mm256_sub_pd(P##_t2i, P##_v2r)); \
        _mm256_storeu_pd(&y3_re[(off)], _mm256_sub_pd(P##_t2r, P##_v2i)); \
        _mm256_storeu_pd(&y3_im[(off)], _mm256_add_pd(P##_t2i, P##_v2r)); \
    } while (0)
#define R5_STORE_14_BWD(P, off)                                           \
    do                                                                    \
    {                                                                     \
        _mm256_storeu_pd(&y1_re[(off)], _mm256_sub_pd(P##_t1r, P##_v1i)); \
        _mm256_storeu_pd(&y1_im[(off)], _mm256_add_pd(P##_t1i, P##_v1r)); \
        _mm256_storeu_pd(&y4_re[(off)], _mm256_add_pd(P##_t1r, P##_v1i)); \
        _mm256_storeu_pd(&y4_im[(off)], _mm256_sub_pd(P##_t1i, P##_v1r)); \
    } while (0)
#define R5_STORE_23_BWD(P, off)                                           \
    do                                                                    \
    {                                                                     \
        _mm256_storeu_pd(&y2_re[(off)], _mm256_sub_pd(P##_t2r, P##_v2i)); \
        _mm256_storeu_pd(&y2_im[(off)], _mm256_add_pd(P##_t2i, P##_v2r)); \
        _mm256_storeu_pd(&y3_re[(off)], _mm256_add_pd(P##_t2r, P##_v2i)); \
        _mm256_storeu_pd(&y3_im[(off)], _mm256_sub_pd(P##_t2i, P##_v2r)); \
    } while (0)

/* ================================================================== */
/*  Forward N1 — no twiddles, U=2 pipeline                             */
/* ================================================================== */

R5_BUTTERFLY_API __attribute__((target("avx2,fma"))) void radix5_wfta_fwd_avx2_N1(
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
    int k = 0;

    /* ── U=2 main loop: 8 elements per iteration ── */
    for (; k + 7 < K; k += 8)
    {

        /* ═══ Phase 1: A full load + s/d + cosine → 8 regs ═══ */
        __m256d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m256d A_s2r, A_s2i, A_d2r, A_d2i;
        __m256d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD(A, k);      /* A = 10 */
        R5_COSINE_CHAIN(A, k); /* A = 8 (d1×2,d2×2,t1×2,t2×2) */

        /* ═══ Phase 2: B partial load (a,b,e → s1,d1) ═══ */
        /*     A=8, B=6 → 14 ✓                              */
        __m256d B_ar, B_ai, B_s1r, B_s1i, B_d1r, B_d1i;
        B_ar = _mm256_loadu_pd(&a_re[k + 4]);
        B_ai = _mm256_loadu_pd(&a_im[k + 4]);
        {
            __m256d _br = _mm256_loadu_pd(&b_re[k + 4]);
            __m256d _bi = _mm256_loadu_pd(&b_im[k + 4]);
            __m256d _er = _mm256_loadu_pd(&e_re[k + 4]);
            __m256d _ei = _mm256_loadu_pd(&e_im[k + 4]);
            B_s1r = _mm256_add_pd(_br, _er);
            B_s1i = _mm256_add_pd(_bi, _ei);
            B_d1r = _mm256_sub_pd(_br, _er);
            B_d1i = _mm256_sub_pd(_bi, _ei);
        }

        /* ═══ Phase 3: A sine chain — FMAs overlap with B's L1 data ═══ */
        /*     A=8, B=6 → 14 ✓                                          */
        __m256d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN(A);
        /* A=8 (t1×2,t2×2,v1×2,v2×2). d1_A,d2_A dead. */

        /* ═══ Phase 4: A stores y1,y4 + B loads c,d → s2,d2 ═══ */
        /*     A→4, B→10 → 14 ✓                                  */
        R5_STORE_14_FWD(A, k);
        /* A=4 (t2×2,v2×2) */

        __m256d B_s2r, B_s2i, B_d2r, B_d2i;
        {
            __m256d _cr = _mm256_loadu_pd(&c_re[k + 4]);
            __m256d _ci = _mm256_loadu_pd(&c_im[k + 4]);
            __m256d _dr = _mm256_loadu_pd(&d_re[k + 4]);
            __m256d _di = _mm256_loadu_pd(&d_im[k + 4]);
            B_s2r = _mm256_add_pd(_cr, _dr);
            B_s2i = _mm256_add_pd(_ci, _di);
            B_d2r = _mm256_sub_pd(_cr, _dr);
            B_d2i = _mm256_sub_pd(_ci, _di);
        }

        /* ═══ Phase 5: A stores y2,y3 → A=0.  B computes solo ═══ */
        R5_STORE_23_FWD(A, k);

        __m256d B_t1r, B_t1i, B_t2r, B_t2i;
        R5_COSINE_CHAIN(B, k + 4); /* B peaks 12 */
        __m256d B_v1r, B_v1i, B_v2r, B_v2i;
        R5_SINE_CHAIN(B);
        R5_STORE_14_FWD(B, k + 4);
        R5_STORE_23_FWD(B, k + 4);
    }

    /* ── U=1 cleanup ── */
    if (k + 3 < K)
    {
        __m256d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m256d A_s2r, A_s2i, A_d2r, A_d2i;
        __m256d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD(A, k);
        R5_COSINE_CHAIN(A, k);
        __m256d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN(A);
        R5_STORE_14_FWD(A, k);
        R5_STORE_23_FWD(A, k);
        k += 4;
    }

    /* ── Scalar tail ── */
    for (; k < K; k++)
    {
        r5_scalar_core_fwd(a_re[k], a_im[k], b_re[k], b_im[k],
                           c_re[k], c_im[k], d_re[k], d_im[k],
                           e_re[k], e_im[k],
                           &y0_re[k], &y0_im[k], &y1_re[k], &y1_im[k],
                           &y2_re[k], &y2_im[k], &y3_re[k], &y3_im[k],
                           &y4_re[k], &y4_im[k]);
    }
}

/* ================================================================== */
/*  Backward N1 — no twiddles, U=2 pipeline                            */
/* ================================================================== */

R5_BUTTERFLY_API __attribute__((target("avx2,fma"))) void radix5_wfta_bwd_avx2_N1(
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
    int k = 0;

    for (; k + 7 < K; k += 8)
    {
        __m256d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m256d A_s2r, A_s2i, A_d2r, A_d2i;
        __m256d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD(A, k);
        R5_COSINE_CHAIN(A, k);

        __m256d B_ar, B_ai, B_s1r, B_s1i, B_d1r, B_d1i;
        B_ar = _mm256_loadu_pd(&a_re[k + 4]);
        B_ai = _mm256_loadu_pd(&a_im[k + 4]);
        {
            __m256d _br = _mm256_loadu_pd(&b_re[k + 4]);
            __m256d _bi = _mm256_loadu_pd(&b_im[k + 4]);
            __m256d _er = _mm256_loadu_pd(&e_re[k + 4]);
            __m256d _ei = _mm256_loadu_pd(&e_im[k + 4]);
            B_s1r = _mm256_add_pd(_br, _er);
            B_s1i = _mm256_add_pd(_bi, _ei);
            B_d1r = _mm256_sub_pd(_br, _er);
            B_d1i = _mm256_sub_pd(_bi, _ei);
        }

        __m256d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN(A);

        R5_STORE_14_BWD(A, k);

        __m256d B_s2r, B_s2i, B_d2r, B_d2i;
        {
            __m256d _cr = _mm256_loadu_pd(&c_re[k + 4]);
            __m256d _ci = _mm256_loadu_pd(&c_im[k + 4]);
            __m256d _dr = _mm256_loadu_pd(&d_re[k + 4]);
            __m256d _di = _mm256_loadu_pd(&d_im[k + 4]);
            B_s2r = _mm256_add_pd(_cr, _dr);
            B_s2i = _mm256_add_pd(_ci, _di);
            B_d2r = _mm256_sub_pd(_cr, _dr);
            B_d2i = _mm256_sub_pd(_ci, _di);
        }

        R5_STORE_23_BWD(A, k);

        __m256d B_t1r, B_t1i, B_t2r, B_t2i;
        R5_COSINE_CHAIN(B, k + 4);
        __m256d B_v1r, B_v1i, B_v2r, B_v2i;
        R5_SINE_CHAIN(B);
        R5_STORE_14_BWD(B, k + 4);
        R5_STORE_23_BWD(B, k + 4);
    }

    if (k + 3 < K)
    {
        __m256d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m256d A_s2r, A_s2i, A_d2r, A_d2i;
        __m256d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD(A, k);
        R5_COSINE_CHAIN(A, k);
        __m256d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN(A);
        R5_STORE_14_BWD(A, k);
        R5_STORE_23_BWD(A, k);
        k += 4;
    }

    for (; k < K; k++)
    {
        r5_scalar_core_bwd(a_re[k], a_im[k], b_re[k], b_im[k],
                           c_re[k], c_im[k], d_re[k], d_im[k],
                           e_re[k], e_im[k],
                           &y0_re[k], &y0_im[k], &y1_re[k], &y1_im[k],
                           &y2_re[k], &y2_im[k], &y3_re[k], &y3_im[k],
                           &y4_re[k], &y4_im[k]);
    }
}

/* ================================================================== */
/*  Twiddle application macro — sequential W derivation                */
/*                                                                     */
/*  Load W1,W2 → b*W1 → W3=W1*W2 (W1 dead) → d*W3 →                 */
/*               c*W2  → W4=W2*W2 (W2 dead) → e*W4                   */
/*                                                                     */
/*  Produces: P_ar, P_ai, P_s1r..P_d2i (same as R5_LOAD_SD)          */
/*  Peak during twiddle: ~12 regs                                      */
/* ================================================================== */
#define R5_LOAD_TW_SD(P, off)                                \
    do                                                       \
    {                                                        \
        P##_ar = _mm256_loadu_pd(&a_re[(off)]);              \
        P##_ai = _mm256_loadu_pd(&a_im[(off)]);              \
        {                                                    \
            __m256d _w1r = _mm256_loadu_pd(&tw1_re[(off)]);  \
            __m256d _w1i = _mm256_loadu_pd(&tw1_im[(off)]);  \
            __m256d _w2r = _mm256_loadu_pd(&tw2_re[(off)]);  \
            __m256d _w2i = _mm256_loadu_pd(&tw2_im[(off)]);  \
            /* tb = b * W1 */                                \
            __m256d _tbr, _tbi;                              \
            {                                                \
                __m256d _br = _mm256_loadu_pd(&b_re[(off)]); \
                __m256d _bi = _mm256_loadu_pd(&b_im[(off)]); \
                R5_CMUL(_br, _bi, _w1r, _w1i, _tbr, _tbi);   \
            }                                                \
            /* W3 = W1*W2, td = d*W3 — W1 dead after */      \
            __m256d _tdr, _tdi;                              \
            {                                                \
                __m256d _w3r, _w3i;                          \
                R5_CMUL(_w1r, _w1i, _w2r, _w2i, _w3r, _w3i); \
                __m256d _dr = _mm256_loadu_pd(&d_re[(off)]); \
                __m256d _di = _mm256_loadu_pd(&d_im[(off)]); \
                R5_CMUL(_dr, _di, _w3r, _w3i, _tdr, _tdi);   \
            }                                                \
            /* tc = c * W2 */                                \
            __m256d _tcr, _tci;                              \
            {                                                \
                __m256d _cr = _mm256_loadu_pd(&c_re[(off)]); \
                __m256d _ci = _mm256_loadu_pd(&c_im[(off)]); \
                R5_CMUL(_cr, _ci, _w2r, _w2i, _tcr, _tci);   \
            }                                                \
            /* W4 = W2*W2, te = e*W4 — W2 dead after */      \
            __m256d _ter, _tei;                              \
            {                                                \
                __m256d _w4r, _w4i;                          \
                R5_CMUL(_w2r, _w2i, _w2r, _w2i, _w4r, _w4i); \
                __m256d _er = _mm256_loadu_pd(&e_re[(off)]); \
                __m256d _ei = _mm256_loadu_pd(&e_im[(off)]); \
                R5_CMUL(_er, _ei, _w4r, _w4i, _ter, _tei);   \
            }                                                \
            /* s/d from twiddled legs */                     \
            P##_s1r = _mm256_add_pd(_tbr, _ter);             \
            P##_s1i = _mm256_add_pd(_tbi, _tei);             \
            P##_d1r = _mm256_sub_pd(_tbr, _ter);             \
            P##_d1i = _mm256_sub_pd(_tbi, _tei);             \
            P##_s2r = _mm256_add_pd(_tcr, _tdr);             \
            P##_s2i = _mm256_add_pd(_tci, _tdi);             \
            P##_d2r = _mm256_sub_pd(_tcr, _tdr);             \
            P##_d2i = _mm256_sub_pd(_tci, _tdi);             \
        }                                                    \
    } while (0)

/* Partial twiddle: only b,e → s1,d1 (for U=2 phase 2) */
#define R5_LOAD_TW_SD_PARTIAL_BE(P, off)                     \
    do                                                       \
    {                                                        \
        P##_ar = _mm256_loadu_pd(&a_re[(off)]);              \
        P##_ai = _mm256_loadu_pd(&a_im[(off)]);              \
        {                                                    \
            __m256d _w1r = _mm256_loadu_pd(&tw1_re[(off)]);  \
            __m256d _w1i = _mm256_loadu_pd(&tw1_im[(off)]);  \
            __m256d _w2r = _mm256_loadu_pd(&tw2_re[(off)]);  \
            __m256d _w2i = _mm256_loadu_pd(&tw2_im[(off)]);  \
            /* tb = b * W1 */                                \
            __m256d _tbr, _tbi;                              \
            {                                                \
                __m256d _br = _mm256_loadu_pd(&b_re[(off)]); \
                __m256d _bi = _mm256_loadu_pd(&b_im[(off)]); \
                R5_CMUL(_br, _bi, _w1r, _w1i, _tbr, _tbi);   \
            }                                                \
            /* W4 = W2*W2, te = e * W4 */                    \
            __m256d _ter, _tei;                              \
            {                                                \
                __m256d _w4r, _w4i;                          \
                R5_CMUL(_w2r, _w2i, _w2r, _w2i, _w4r, _w4i); \
                __m256d _er = _mm256_loadu_pd(&e_re[(off)]); \
                __m256d _ei = _mm256_loadu_pd(&e_im[(off)]); \
                R5_CMUL(_er, _ei, _w4r, _w4i, _ter, _tei);   \
            }                                                \
            P##_s1r = _mm256_add_pd(_tbr, _ter);             \
            P##_s1i = _mm256_add_pd(_tbi, _tei);             \
            P##_d1r = _mm256_sub_pd(_tbr, _ter);             \
            P##_d1i = _mm256_sub_pd(_tbi, _tei);             \
        }                                                    \
    } while (0)

/* Deferred twiddle: c,d → s2,d2 (for U=2 phase 4, reloads W1,W2) */
#define R5_LOAD_TW_SD_DEFERRED_CD(P, off)                    \
    do                                                       \
    {                                                        \
        {                                                    \
            __m256d _w1r = _mm256_loadu_pd(&tw1_re[(off)]);  \
            __m256d _w1i = _mm256_loadu_pd(&tw1_im[(off)]);  \
            __m256d _w2r = _mm256_loadu_pd(&tw2_re[(off)]);  \
            __m256d _w2i = _mm256_loadu_pd(&tw2_im[(off)]);  \
            /* W3 = W1*W2, td = d*W3 */                      \
            __m256d _tdr, _tdi;                              \
            {                                                \
                __m256d _w3r, _w3i;                          \
                R5_CMUL(_w1r, _w1i, _w2r, _w2i, _w3r, _w3i); \
                __m256d _dr = _mm256_loadu_pd(&d_re[(off)]); \
                __m256d _di = _mm256_loadu_pd(&d_im[(off)]); \
                R5_CMUL(_dr, _di, _w3r, _w3i, _tdr, _tdi);   \
            }                                                \
            /* tc = c * W2 */                                \
            __m256d _tcr, _tci;                              \
            {                                                \
                __m256d _cr = _mm256_loadu_pd(&c_re[(off)]); \
                __m256d _ci = _mm256_loadu_pd(&c_im[(off)]); \
                R5_CMUL(_cr, _ci, _w2r, _w2i, _tcr, _tci);   \
            }                                                \
            P##_s2r = _mm256_add_pd(_tcr, _tdr);             \
            P##_s2i = _mm256_add_pd(_tci, _tdi);             \
            P##_d2r = _mm256_sub_pd(_tcr, _tdr);             \
            P##_d2i = _mm256_sub_pd(_tci, _tdi);             \
        }                                                    \
    } while (0)

/* ================================================================== */
/*  Forward twiddled — BLOCKED2, U=2 pipeline                          */
/* ================================================================== */

R5_BUTTERFLY_API __attribute__((target("avx2,fma"))) void radix5_wfta_fwd_avx2(
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
    int k = 0;

    for (; k + 7 < K; k += 8)
    {

        /* ═══ A: full twiddle + s/d + cosine → 8 regs ═══ */
        __m256d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m256d A_s2r, A_s2i, A_d2r, A_d2i;
        __m256d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_TW_SD(A, k);
        R5_COSINE_CHAIN(A, k);

        /* ═══ B partial: a,b,e twiddle → s1,d1 (6 regs) ═══ */
        __m256d B_ar, B_ai, B_s1r, B_s1i, B_d1r, B_d1i;
        R5_LOAD_TW_SD_PARTIAL_BE(B, k + 4);
        /* A=8, B=6 → 14 ✓ */

        /* ═══ A sine chain ═══ */
        __m256d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN(A);

        /* ═══ A stores y1,y4 → A=4.  B deferred c,d twiddle → B=10 ═══ */
        R5_STORE_14_FWD(A, k);

        __m256d B_s2r, B_s2i, B_d2r, B_d2i;
        R5_LOAD_TW_SD_DEFERRED_CD(B, k + 4);
        /* A=4, B=10 → 14 ✓ */

        /* ═══ A stores y2,y3.  B solo. ═══ */
        R5_STORE_23_FWD(A, k);

        __m256d B_t1r, B_t1i, B_t2r, B_t2i;
        R5_COSINE_CHAIN(B, k + 4);
        __m256d B_v1r, B_v1i, B_v2r, B_v2i;
        R5_SINE_CHAIN(B);
        R5_STORE_14_FWD(B, k + 4);
        R5_STORE_23_FWD(B, k + 4);
    }

    /* ── U=1 cleanup ── */
    for (; k + 3 < K; k += 4)
    {
        __m256d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m256d A_s2r, A_s2i, A_d2r, A_d2i;
        __m256d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_TW_SD(A, k);
        R5_COSINE_CHAIN(A, k);
        __m256d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN(A);
        R5_STORE_14_FWD(A, k);
        R5_STORE_23_FWD(A, k);
    }

    /* Scalar tail */
    for (; k < K; k++)
    {
        double tbr, tbi, tcr, tci, tdr, tdi, ter, tei;
        double w3r, w3i, w4r, w4i;
        r5_scalar_cmul(tw1_re[k], tw1_im[k], tw2_re[k], tw2_im[k], &w3r, &w3i);
        r5_scalar_cmul(tw2_re[k], tw2_im[k], tw2_re[k], tw2_im[k], &w4r, &w4i);
        r5_scalar_cmul(b_re[k], b_im[k], tw1_re[k], tw1_im[k], &tbr, &tbi);
        r5_scalar_cmul(c_re[k], c_im[k], tw2_re[k], tw2_im[k], &tcr, &tci);
        r5_scalar_cmul(d_re[k], d_im[k], w3r, w3i, &tdr, &tdi);
        r5_scalar_cmul(e_re[k], e_im[k], w4r, w4i, &ter, &tei);
        r5_scalar_core_fwd(a_re[k], a_im[k], tbr, tbi, tcr, tci,
                           tdr, tdi, ter, tei,
                           &y0_re[k], &y0_im[k], &y1_re[k], &y1_im[k],
                           &y2_re[k], &y2_im[k], &y3_re[k], &y3_im[k],
                           &y4_re[k], &y4_im[k]);
    }
}

/* ================================================================== */
/*  Backward twiddled — IDFT-5 → post-multiply conj(W), U=2           */
/*                                                                     */
/*  Note: backward twiddle ordering is different from forward.         */
/*  Forward = pre-twiddle → DFT-5.                                    */
/*  Backward = IDFT-5 → post-conj-twiddle.                            */
/*  The post-multiply naturally drains registers as each y is stored.  */
/*  U=2: A finishes IDFT-5+post-twiddle, then B runs solo.            */
/*  Less interleaving opportunity than forward, but OoO engine         */
/*  overlaps B's loads with A's final post-twiddle stores.             */
/* ================================================================== */

R5_BUTTERFLY_API __attribute__((target("avx2,fma"))) void radix5_wfta_bwd_avx2(
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
    int k = 0;

    for (; k + 7 < K; k += 8)
    {

        /* ═══ A: IDFT-5 core ═══ */
        __m256d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m256d A_s2r, A_s2i, A_d2r, A_d2i;
        __m256d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD(A, k);
        R5_COSINE_CHAIN(A, k); /* stores y0 (no twiddle on DC) */

        __m256d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN(A);

        /* Backward +i → raw IDFT outputs */
        __m256d A_r1r = _mm256_sub_pd(A_t1r, A_v1i);
        __m256d A_r1i = _mm256_add_pd(A_t1i, A_v1r);
        __m256d A_r4r = _mm256_add_pd(A_t1r, A_v1i);
        __m256d A_r4i = _mm256_sub_pd(A_t1i, A_v1r);
        __m256d A_r2r = _mm256_sub_pd(A_t2r, A_v2i);
        __m256d A_r2i = _mm256_add_pd(A_t2i, A_v2r);
        __m256d A_r3r = _mm256_add_pd(A_t2r, A_v2i);
        __m256d A_r3i = _mm256_sub_pd(A_t2i, A_v2r);
        /* A = 8 regs (r1..r4).  t,v dead. */

        /* Post-multiply by conj(W): sequential drain */
        {
            __m256d w1r = _mm256_loadu_pd(&tw1_re[k]);
            __m256d w1i = _mm256_loadu_pd(&tw1_im[k]);
            __m256d w2r = _mm256_loadu_pd(&tw2_re[k]);
            __m256d w2i = _mm256_loadu_pd(&tw2_im[k]);

            __m256d o1r, o1i;
            R5_CMULJ(A_r1r, A_r1i, w1r, w1i, o1r, o1i);
            _mm256_storeu_pd(&y1_re[k], o1r);
            _mm256_storeu_pd(&y1_im[k], o1i);

            __m256d w3r, w3i;
            R5_CMUL(w1r, w1i, w2r, w2i, w3r, w3i);
            __m256d o3r, o3i;
            R5_CMULJ(A_r3r, A_r3i, w3r, w3i, o3r, o3i);
            _mm256_storeu_pd(&y3_re[k], o3r);
            _mm256_storeu_pd(&y3_im[k], o3i);

            __m256d o2r, o2i;
            R5_CMULJ(A_r2r, A_r2i, w2r, w2i, o2r, o2i);
            _mm256_storeu_pd(&y2_re[k], o2r);
            _mm256_storeu_pd(&y2_im[k], o2i);

            __m256d w4r, w4i;
            R5_CMUL(w2r, w2i, w2r, w2i, w4r, w4i);
            __m256d o4r, o4i;
            R5_CMULJ(A_r4r, A_r4i, w4r, w4i, o4r, o4i);
            _mm256_storeu_pd(&y4_re[k], o4r);
            _mm256_storeu_pd(&y4_im[k], o4i);
        }
        /* A = 0 */

        /* ═══ B: solo IDFT-5 + post-conj-twiddle ═══ */
        __m256d B_ar, B_ai, B_s1r, B_s1i, B_d1r, B_d1i;
        __m256d B_s2r, B_s2i, B_d2r, B_d2i;
        __m256d B_t1r, B_t1i, B_t2r, B_t2i;
        R5_LOAD_SD(B, k + 4);
        R5_COSINE_CHAIN(B, k + 4);

        __m256d B_v1r, B_v1i, B_v2r, B_v2i;
        R5_SINE_CHAIN(B);

        __m256d B_r1r = _mm256_sub_pd(B_t1r, B_v1i);
        __m256d B_r1i = _mm256_add_pd(B_t1i, B_v1r);
        __m256d B_r4r = _mm256_add_pd(B_t1r, B_v1i);
        __m256d B_r4i = _mm256_sub_pd(B_t1i, B_v1r);
        __m256d B_r2r = _mm256_sub_pd(B_t2r, B_v2i);
        __m256d B_r2i = _mm256_add_pd(B_t2i, B_v2r);
        __m256d B_r3r = _mm256_add_pd(B_t2r, B_v2i);
        __m256d B_r3i = _mm256_sub_pd(B_t2i, B_v2r);

        {
            __m256d w1r = _mm256_loadu_pd(&tw1_re[k + 4]);
            __m256d w1i = _mm256_loadu_pd(&tw1_im[k + 4]);
            __m256d w2r = _mm256_loadu_pd(&tw2_re[k + 4]);
            __m256d w2i = _mm256_loadu_pd(&tw2_im[k + 4]);

            __m256d o1r, o1i;
            R5_CMULJ(B_r1r, B_r1i, w1r, w1i, o1r, o1i);
            _mm256_storeu_pd(&y1_re[k + 4], o1r);
            _mm256_storeu_pd(&y1_im[k + 4], o1i);

            __m256d w3r, w3i;
            R5_CMUL(w1r, w1i, w2r, w2i, w3r, w3i);
            __m256d o3r, o3i;
            R5_CMULJ(B_r3r, B_r3i, w3r, w3i, o3r, o3i);
            _mm256_storeu_pd(&y3_re[k + 4], o3r);
            _mm256_storeu_pd(&y3_im[k + 4], o3i);

            __m256d o2r, o2i;
            R5_CMULJ(B_r2r, B_r2i, w2r, w2i, o2r, o2i);
            _mm256_storeu_pd(&y2_re[k + 4], o2r);
            _mm256_storeu_pd(&y2_im[k + 4], o2i);

            __m256d w4r, w4i;
            R5_CMUL(w2r, w2i, w2r, w2i, w4r, w4i);
            __m256d o4r, o4i;
            R5_CMULJ(B_r4r, B_r4i, w4r, w4i, o4r, o4i);
            _mm256_storeu_pd(&y4_re[k + 4], o4r);
            _mm256_storeu_pd(&y4_im[k + 4], o4i);
        }
    }

    /* ── U=1 cleanup ── */
    for (; k + 3 < K; k += 4)
    {
        __m256d A_ar, A_ai, A_s1r, A_s1i, A_d1r, A_d1i;
        __m256d A_s2r, A_s2i, A_d2r, A_d2i;
        __m256d A_t1r, A_t1i, A_t2r, A_t2i;
        R5_LOAD_SD(A, k);
        R5_COSINE_CHAIN(A, k);

        __m256d A_v1r, A_v1i, A_v2r, A_v2i;
        R5_SINE_CHAIN(A);

        __m256d r1r = _mm256_sub_pd(A_t1r, A_v1i);
        __m256d r1i = _mm256_add_pd(A_t1i, A_v1r);
        __m256d r4r = _mm256_add_pd(A_t1r, A_v1i);
        __m256d r4i = _mm256_sub_pd(A_t1i, A_v1r);
        __m256d r2r = _mm256_sub_pd(A_t2r, A_v2i);
        __m256d r2i = _mm256_add_pd(A_t2i, A_v2r);
        __m256d r3r = _mm256_add_pd(A_t2r, A_v2i);
        __m256d r3i = _mm256_sub_pd(A_t2i, A_v2r);

        __m256d w1r = _mm256_loadu_pd(&tw1_re[k]);
        __m256d w1i = _mm256_loadu_pd(&tw1_im[k]);
        __m256d w2r = _mm256_loadu_pd(&tw2_re[k]);
        __m256d w2i = _mm256_loadu_pd(&tw2_im[k]);

        __m256d o1r, o1i;
        R5_CMULJ(r1r, r1i, w1r, w1i, o1r, o1i);
        _mm256_storeu_pd(&y1_re[k], o1r);
        _mm256_storeu_pd(&y1_im[k], o1i);

        __m256d w3r, w3i;
        R5_CMUL(w1r, w1i, w2r, w2i, w3r, w3i);
        __m256d o3r, o3i;
        R5_CMULJ(r3r, r3i, w3r, w3i, o3r, o3i);
        _mm256_storeu_pd(&y3_re[k], o3r);
        _mm256_storeu_pd(&y3_im[k], o3i);

        __m256d o2r, o2i;
        R5_CMULJ(r2r, r2i, w2r, w2i, o2r, o2i);
        _mm256_storeu_pd(&y2_re[k], o2r);
        _mm256_storeu_pd(&y2_im[k], o2i);

        __m256d w4r, w4i;
        R5_CMUL(w2r, w2i, w2r, w2i, w4r, w4i);
        __m256d o4r, o4i;
        R5_CMULJ(r4r, r4i, w4r, w4i, o4r, o4i);
        _mm256_storeu_pd(&y4_re[k], o4r);
        _mm256_storeu_pd(&y4_im[k], o4i);
    }

    /* Scalar tail */
    for (; k < K; k++)
    {
        double r0r, r0i, r1r, r1i, r2r, r2i, r3r, r3i, r4r, r4i;
        r5_scalar_core_bwd(a_re[k], a_im[k], b_re[k], b_im[k],
                           c_re[k], c_im[k], d_re[k], d_im[k],
                           e_re[k], e_im[k],
                           &r0r, &r0i, &r1r, &r1i, &r2r, &r2i,
                           &r3r, &r3i, &r4r, &r4i);

        double w3r, w3i, w4r, w4i;
        r5_scalar_cmul(tw1_re[k], tw1_im[k], tw2_re[k], tw2_im[k], &w3r, &w3i);
        r5_scalar_cmul(tw2_re[k], tw2_im[k], tw2_re[k], tw2_im[k], &w4r, &w4i);

        y0_re[k] = r0r;
        y0_im[k] = r0i;
        r5_scalar_cmulj(r1r, r1i, tw1_re[k], tw1_im[k], &y1_re[k], &y1_im[k]);
        r5_scalar_cmulj(r2r, r2i, tw2_re[k], tw2_im[k], &y2_re[k], &y2_im[k]);
        r5_scalar_cmulj(r3r, r3i, w3r, w3i, &y3_re[k], &y3_im[k]);
        r5_scalar_cmulj(r4r, r4i, w4r, w4i, &y4_re[k], &y4_im[k]);
    }
}

#endif /* FFT_RADIX5_AVX2_H */