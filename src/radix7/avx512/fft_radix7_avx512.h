/**
 * @file  fft_radix7_avx512.h
 * @brief Radix-7 FFT butterfly using Rader's algorithm — AVX-512 (8-wide, U2)
 *
 * ARCHITECTURAL REVOLUTION - Generation 3:
 * ===========================================
 * TRUE SoA-in-register: re/im stay separate throughout
 * Unit-stride loads: killed gathers for stage twiddles
 * 8-wide processing: full 512-bit loads (8 doubles)
 * U2 pipeline: dual butterflies (k, k+8) saturating dual FMA ports
 * Aligned loads/stores: guaranteed alignment
 *
 * ALL RADIX-7 OPTIMIZATIONS:
 * ===========================
 * P0: Pre-split Rader broadcasts — kernel as aligned double[8] octets
 * P0: Round-robin convolution — 2-wave interleaved FMA schedule
 * P0: U2 shared kernel loads — A and B butterflies share kernel octets
 * P1: Tree y0 sum — binary tree DC accumulation (3 levels vs 6)
 *
 * Register budget (32 ZMM on AVX-512):
 *   U=2 pointwise (tightest point):
 *     12 ZMM: A's s0-s5 (re+im)
 *     12 ZMM: B's s0-s5 (re+im)
 *      6 ZMM: kernel loads (shared, per wave)
 *      2 ZMM: temporaries
 *     = 32 at peak — fits exactly.
 *
 *   U=2 DFT6 overlap:
 *     12 ZMM: first butterfly's DFT6 working set
 *     12 ZMM: second butterfly's DFT6 working set
 *      2 ZMM: DFT3 constants (shared)
 *      4 ZMM: W6 twiddle constants (shared)
 *      2 ZMM: temporaries
 *     = 32 — tight, x0 reloaded from L1 after Rader phase.
 *
 * TARGET: High-end Xeons with 2x 512-bit FMA units, 32 ZMM regs
 */

#ifndef FFT_RADIX7_AVX512_H
#define FFT_RADIX7_AVX512_H

#if defined(__AVX512F__)
#include <immintrin.h>

#define R7_512_VW 8  /* vector width: doubles per ZMM */
#define R7_512_U2 16 /* U=2 stride: 2 × VW           */

/* ================================================================== */
/*  Pre-split constant octets  (P0: aligned loads, ports 2/3)          */
/* ================================================================== */

#define R7Q8(name, v)                                          \
    static const double __attribute__((aligned(64))) name[8] = \
        {v, v, v, v, v, v, v, v}

/* DFT3 */
R7Q8(R7_C3_Q8, -0.5);
R7Q8(R7_S3_Q8, 0.86602540378443864676);

/* W6 forward twiddles */
R7Q8(R7_W61R_Q8, 0.5);
R7Q8(R7_W61I_Q8, -0.86602540378443864676);
R7Q8(R7_W62R_Q8, -0.5);
R7Q8(R7_W62I_Q8, -0.86602540378443864676);

/* W6 backward (conjugate) twiddles */
R7Q8(R7_W61I_N_Q8, 0.86602540378443864676);
R7Q8(R7_W62I_N_Q8, 0.86602540378443864676);

/* Forward Rader kernel — 12 octets */
R7Q8(RK7_FR0, -1.66666666666666657415e-01);
R7Q8(RK7_FR1, +4.06688893057589595514e-01);
R7Q8(RK7_FR2, +3.95078234262700001000e-01);
R7Q8(RK7_FR3, +1.85037170770859413132e-17);
R7Q8(RK7_FR4, +3.95078234262699945489e-01);
R7Q8(RK7_FR5, -4.06688893057589151425e-01);
R7Q8(RK7_FI0, -9.25185853854297065662e-18);
R7Q8(RK7_FI1, -1.70436465311965518188e-01);
R7Q8(RK7_FI2, -1.95851048647464526242e-01);
R7Q8(RK7_FI3, -4.40958551844098212147e-01);
R7Q8(RK7_FI4, +1.95851048647464942576e-01);
R7Q8(RK7_FI5, -1.70436465311965684721e-01);

/* Backward Rader kernel — 12 octets */
R7Q8(RK7_BR0, -1.66666666666666657415e-01);
R7Q8(RK7_BR1, -4.06688893057589595514e-01);
R7Q8(RK7_BR2, +3.95078234262700167534e-01);
R7Q8(RK7_BR3, +9.25185853854297158106e-17);
R7Q8(RK7_BR4, +3.95078234262700223045e-01);
R7Q8(RK7_BR5, +4.06688893057589540003e-01);
R7Q8(RK7_BI0, +9.25185853854297065662e-18);
R7Q8(RK7_BI1, +1.70436465311965656966e-01);
R7Q8(RK7_BI2, -1.95851048647464304198e-01);
R7Q8(RK7_BI3, +4.40958551844098767258e-01);
R7Q8(RK7_BI4, +1.95851048647464359709e-01);
R7Q8(RK7_BI5, +1.70436465311965323899e-01);

#undef R7Q8

/* Scalar kernel arrays — for scalar tail */
static const double RK7S_FWD_RE[6] = {
    -1.66666666666666657415e-01, +4.06688893057589595514e-01,
    +3.95078234262700001000e-01, +1.85037170770859413132e-17,
    +3.95078234262699945489e-01, -4.06688893057589151425e-01};
static const double RK7S_FWD_IM[6] = {
    -9.25185853854297065662e-18, -1.70436465311965518188e-01,
    -1.95851048647464526242e-01, -4.40958551844098212147e-01,
    +1.95851048647464942576e-01, -1.70436465311965684721e-01};
static const double RK7S_BWD_RE[6] = {
    -1.66666666666666657415e-01, -4.06688893057589595514e-01,
    +3.95078234262700167534e-01, +9.25185853854297158106e-17,
    +3.95078234262700223045e-01, +4.06688893057589540003e-01};
static const double RK7S_BWD_IM[6] = {
    +9.25185853854297065662e-18, +1.70436465311965656966e-01,
    -1.95851048647464304198e-01, +4.40958551844098767258e-01,
    +1.95851048647464359709e-01, +1.70436465311965323899e-01};

/* ================================================================== */
/*  AVX-512 complex multiply (FMA)                                     */
/* ================================================================== */

static inline __attribute__((always_inline)) void cmul_512(__m512d ar, __m512d ai, __m512d wr, __m512d wi,
                                                           __m512d *cr, __m512d *ci)
{
    __m512d t0 = _mm512_mul_pd(ai, wi);
    *cr = _mm512_fmsub_pd(ar, wr, t0);
    __m512d t1 = _mm512_mul_pd(ai, wr);
    *ci = _mm512_fmadd_pd(ar, wi, t1);
}

/* ================================================================== */
/*  DFT6 forward — single butterfly, ZMM                               */
/* ================================================================== */

#define DFT6_FWD_512(x0r, x0i, x1r, x1i, x2r, x2i,                              \
                     x3r, x3i, x4r, x4i, x5r, x5i)                              \
    do                                                                          \
    {                                                                           \
        __m512d vC3 = _mm512_load_pd(R7_C3_Q8);                                 \
        __m512d vS3 = _mm512_load_pd(R7_S3_Q8);                                 \
        __m512d se_r = _mm512_add_pd(x2r, x4r), se_i = _mm512_add_pd(x2i, x4i); \
        __m512d de_r = _mm512_sub_pd(x2r, x4r), de_i = _mm512_sub_pd(x2i, x4i); \
        __m512d E0r = _mm512_add_pd(x0r, se_r), E0i = _mm512_add_pd(x0i, se_i); \
        __m512d be_r = _mm512_fmadd_pd(vC3, se_r, x0r);                         \
        __m512d be_i = _mm512_fmadd_pd(vC3, se_i, x0i);                         \
        __m512d E1r = _mm512_fmadd_pd(vS3, de_i, be_r);                         \
        __m512d E1i = _mm512_fnmadd_pd(vS3, de_r, be_i);                        \
        __m512d E2r = _mm512_fnmadd_pd(vS3, de_i, be_r);                        \
        __m512d E2i = _mm512_fmadd_pd(vS3, de_r, be_i);                         \
        __m512d so_r = _mm512_add_pd(x3r, x5r), so_i = _mm512_add_pd(x3i, x5i); \
        __m512d do_r = _mm512_sub_pd(x3r, x5r), do_i = _mm512_sub_pd(x3i, x5i); \
        __m512d O0r = _mm512_add_pd(x1r, so_r), O0i = _mm512_add_pd(x1i, so_i); \
        __m512d bo_r = _mm512_fmadd_pd(vC3, so_r, x1r);                         \
        __m512d bo_i = _mm512_fmadd_pd(vC3, so_i, x1i);                         \
        __m512d O1r = _mm512_fmadd_pd(vS3, do_i, bo_r);                         \
        __m512d O1i = _mm512_fnmadd_pd(vS3, do_r, bo_i);                        \
        __m512d O2r = _mm512_fnmadd_pd(vS3, do_i, bo_r);                        \
        __m512d O2i = _mm512_fmadd_pd(vS3, do_r, bo_i);                         \
        x0r = _mm512_add_pd(E0r, O0r);                                          \
        x0i = _mm512_add_pd(E0i, O0i);                                          \
        x3r = _mm512_sub_pd(E0r, O0r);                                          \
        x3i = _mm512_sub_pd(E0i, O0i);                                          \
        {                                                                       \
            __m512d W1r = _mm512_load_pd(R7_W61R_Q8);                           \
            __m512d W1i = _mm512_load_pd(R7_W61I_Q8);                           \
            __m512d T1r, T1i;                                                   \
            cmul_512(O1r, O1i, W1r, W1i, &T1r, &T1i);                           \
            x1r = _mm512_add_pd(E1r, T1r);                                      \
            x1i = _mm512_add_pd(E1i, T1i);                                      \
            x4r = _mm512_sub_pd(E1r, T1r);                                      \
            x4i = _mm512_sub_pd(E1i, T1i);                                      \
        }                                                                       \
        {                                                                       \
            __m512d W2r = _mm512_load_pd(R7_W62R_Q8);                           \
            __m512d W2i = _mm512_load_pd(R7_W62I_Q8);                           \
            __m512d T2r, T2i;                                                   \
            cmul_512(O2r, O2i, W2r, W2i, &T2r, &T2i);                           \
            x2r = _mm512_add_pd(E2r, T2r);                                      \
            x2i = _mm512_add_pd(E2i, T2i);                                      \
            x5r = _mm512_sub_pd(E2r, T2r);                                      \
            x5i = _mm512_sub_pd(E2i, T2i);                                      \
        }                                                                       \
    } while (0)

/* ================================================================== */
/*  DFT6 backward — conjugate twiddles                                 */
/* ================================================================== */

#define DFT6_BWD_512(x0r, x0i, x1r, x1i, x2r, x2i,                              \
                     x3r, x3i, x4r, x4i, x5r, x5i)                              \
    do                                                                          \
    {                                                                           \
        __m512d vC3 = _mm512_load_pd(R7_C3_Q8);                                 \
        __m512d vS3 = _mm512_load_pd(R7_S3_Q8);                                 \
        __m512d se_r = _mm512_add_pd(x2r, x4r), se_i = _mm512_add_pd(x2i, x4i); \
        __m512d de_r = _mm512_sub_pd(x2r, x4r), de_i = _mm512_sub_pd(x2i, x4i); \
        __m512d E0r = _mm512_add_pd(x0r, se_r), E0i = _mm512_add_pd(x0i, se_i); \
        __m512d be_r = _mm512_fmadd_pd(vC3, se_r, x0r);                         \
        __m512d be_i = _mm512_fmadd_pd(vC3, se_i, x0i);                         \
        __m512d E1r = _mm512_fnmadd_pd(vS3, de_i, be_r);                        \
        __m512d E1i = _mm512_fmadd_pd(vS3, de_r, be_i);                         \
        __m512d E2r = _mm512_fmadd_pd(vS3, de_i, be_r);                         \
        __m512d E2i = _mm512_fnmadd_pd(vS3, de_r, be_i);                        \
        __m512d so_r = _mm512_add_pd(x3r, x5r), so_i = _mm512_add_pd(x3i, x5i); \
        __m512d do_r = _mm512_sub_pd(x3r, x5r), do_i = _mm512_sub_pd(x3i, x5i); \
        __m512d O0r = _mm512_add_pd(x1r, so_r), O0i = _mm512_add_pd(x1i, so_i); \
        __m512d bo_r = _mm512_fmadd_pd(vC3, so_r, x1r);                         \
        __m512d bo_i = _mm512_fmadd_pd(vC3, so_i, x1i);                         \
        __m512d O1r = _mm512_fnmadd_pd(vS3, do_i, bo_r);                        \
        __m512d O1i = _mm512_fmadd_pd(vS3, do_r, bo_i);                         \
        __m512d O2r = _mm512_fmadd_pd(vS3, do_i, bo_r);                         \
        __m512d O2i = _mm512_fnmadd_pd(vS3, do_r, bo_i);                        \
        x0r = _mm512_add_pd(E0r, O0r);                                          \
        x0i = _mm512_add_pd(E0i, O0i);                                          \
        x3r = _mm512_sub_pd(E0r, O0r);                                          \
        x3i = _mm512_sub_pd(E0i, O0i);                                          \
        {                                                                       \
            __m512d W1r = _mm512_load_pd(R7_W61R_Q8);                           \
            __m512d W1i = _mm512_load_pd(R7_W61I_N_Q8);                         \
            __m512d T1r, T1i;                                                   \
            cmul_512(O1r, O1i, W1r, W1i, &T1r, &T1i);                           \
            x1r = _mm512_add_pd(E1r, T1r);                                      \
            x1i = _mm512_add_pd(E1i, T1i);                                      \
            x4r = _mm512_sub_pd(E1r, T1r);                                      \
            x4i = _mm512_sub_pd(E1i, T1i);                                      \
        }                                                                       \
        {                                                                       \
            __m512d W2r = _mm512_load_pd(R7_W62R_Q8);                           \
            __m512d W2i = _mm512_load_pd(R7_W62I_N_Q8);                         \
            __m512d T2r, T2i;                                                   \
            cmul_512(O2r, O2i, W2r, W2i, &T2r, &T2i);                           \
            x2r = _mm512_add_pd(E2r, T2r);                                      \
            x2i = _mm512_add_pd(E2i, T2i);                                      \
            x5r = _mm512_sub_pd(E2r, T2r);                                      \
            x5i = _mm512_sub_pd(E2i, T2i);                                      \
        }                                                                       \
    } while (0)

/* ================================================================== */
/*  Round-robin pointwise — U=1 (single butterfly)                     */
/* ================================================================== */

#define PW_RR6_512(s0r, s0i, s1r, s1i, s2r, s2i,                              \
                   s3r, s3i, s4r, s4i, s5r, s5i,                              \
                   KR0, KI0, KR1, KI1, KR2, KI2,                              \
                   KR3, KI3, KR4, KI4, KR5, KI5)                              \
    do                                                                        \
    {                                                                         \
        __m512d k0r = _mm512_load_pd(KR0), k0i = _mm512_load_pd(KI0);         \
        __m512d k1r = _mm512_load_pd(KR1), k1i = _mm512_load_pd(KI1);         \
        __m512d k2r = _mm512_load_pd(KR2), k2i = _mm512_load_pd(KI2);         \
        __m512d p0a = _mm512_mul_pd(s0i, k0i), p0b = _mm512_mul_pd(s0i, k0r); \
        __m512d p1a = _mm512_mul_pd(s1i, k1i), p1b = _mm512_mul_pd(s1i, k1r); \
        __m512d p2a = _mm512_mul_pd(s2i, k2i), p2b = _mm512_mul_pd(s2i, k2r); \
        __m512d r0r = _mm512_fmsub_pd(s0r, k0r, p0a);                         \
        __m512d r0i = _mm512_fmadd_pd(s0r, k0i, p0b);                         \
        __m512d r1r = _mm512_fmsub_pd(s1r, k1r, p1a);                         \
        __m512d r1i = _mm512_fmadd_pd(s1r, k1i, p1b);                         \
        __m512d r2r = _mm512_fmsub_pd(s2r, k2r, p2a);                         \
        __m512d r2i = _mm512_fmadd_pd(s2r, k2i, p2b);                         \
        s0r = r0r;                                                            \
        s0i = r0i;                                                            \
        s1r = r1r;                                                            \
        s1i = r1i;                                                            \
        s2r = r2r;                                                            \
        s2i = r2i;                                                            \
        __m512d k3r = _mm512_load_pd(KR3), k3i = _mm512_load_pd(KI3);         \
        __m512d k4r = _mm512_load_pd(KR4), k4i = _mm512_load_pd(KI4);         \
        __m512d k5r = _mm512_load_pd(KR5), k5i = _mm512_load_pd(KI5);         \
        __m512d p3a = _mm512_mul_pd(s3i, k3i), p3b = _mm512_mul_pd(s3i, k3r); \
        __m512d p4a = _mm512_mul_pd(s4i, k4i), p4b = _mm512_mul_pd(s4i, k4r); \
        __m512d p5a = _mm512_mul_pd(s5i, k5i), p5b = _mm512_mul_pd(s5i, k5r); \
        __m512d r3r = _mm512_fmsub_pd(s3r, k3r, p3a);                         \
        __m512d r3i = _mm512_fmadd_pd(s3r, k3i, p3b);                         \
        __m512d r4r = _mm512_fmsub_pd(s4r, k4r, p4a);                         \
        __m512d r4i = _mm512_fmadd_pd(s4r, k4i, p4b);                         \
        __m512d r5r = _mm512_fmsub_pd(s5r, k5r, p5a);                         \
        __m512d r5i = _mm512_fmadd_pd(s5r, k5i, p5b);                         \
        s3r = r3r;                                                            \
        s3i = r3i;                                                            \
        s4r = r4r;                                                            \
        s4i = r4i;                                                            \
        s5r = r5r;                                                            \
        s5i = r5i;                                                            \
    } while (0)

/* ================================================================== */
/*  Round-robin pointwise — U=2 (dual butterfly, shared kernel loads)  */
/*                                                                     */
/*  A (k) and B (k+8) share all 12 kernel loads.                      */
/*  Within each wave, A products interleave with B products:           */
/*    A.mul→B.mul→A.fma→B.fma maximizes ILP across both FMA ports.    */
/*                                                                     */
/*  Register peak: 12(A) + 12(B) + 6(kernel) + 2(temps) = 32         */
/* ================================================================== */

#define PW_RR6_U2(a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, a4r, a4i, a5r, a5i,   \
                  b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i, b4r, b4i, b5r, b5i,   \
                  KR0, KI0, KR1, KI1, KR2, KI2,                                 \
                  KR3, KI3, KR4, KI4, KR5, KI5)                                 \
    do                                                                          \
    {                                                                           \
        /* Wave 1: slots 0,1,2 — shared kernel loads */                         \
        __m512d k0r = _mm512_load_pd(KR0), k0i = _mm512_load_pd(KI0);           \
        __m512d k1r = _mm512_load_pd(KR1), k1i = _mm512_load_pd(KI1);           \
        __m512d k2r = _mm512_load_pd(KR2), k2i = _mm512_load_pd(KI2);           \
        /* A mul phase */                                                       \
        __m512d ap0a = _mm512_mul_pd(a0i, k0i), ap0b = _mm512_mul_pd(a0i, k0r); \
        __m512d ap1a = _mm512_mul_pd(a1i, k1i), ap1b = _mm512_mul_pd(a1i, k1r); \
        __m512d ap2a = _mm512_mul_pd(a2i, k2i), ap2b = _mm512_mul_pd(a2i, k2r); \
        /* B mul phase (interleaved — fills A's FMA latency bubbles) */         \
        __m512d bp0a = _mm512_mul_pd(b0i, k0i), bp0b = _mm512_mul_pd(b0i, k0r); \
        __m512d bp1a = _mm512_mul_pd(b1i, k1i), bp1b = _mm512_mul_pd(b1i, k1r); \
        __m512d bp2a = _mm512_mul_pd(b2i, k2i), bp2b = _mm512_mul_pd(b2i, k2r); \
        /* A FMA phase */                                                       \
        a0r = _mm512_fmsub_pd(a0r, k0r, ap0a);                                  \
        a0i = _mm512_fmadd_pd(a0i, k0i, ap0b);                                  \
        /* ^^^ WRONG: a0i was consumed by mul above, now overwritten.           \
         * Fix: use saved ar for FMA, compute into temps. */                    \
        (void)0;                                                                \
    } while (0)

/* The approach above has a hazard — we need ar alive for ci=ar*ki+ai*kr
 * but ar gets clobbered by cr=ar*kr-ai*ki.  The round-robin fixes this
 * by computing all products BEFORE any writeback.  Rewrite: */

#undef PW_RR6_U2

#define PW_RR6_U2(a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, a4r, a4i, a5r, a5i,   \
                  b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i, b4r, b4i, b5r, b5i,   \
                  KR0, KI0, KR1, KI1, KR2, KI2,                                 \
                  KR3, KI3, KR4, KI4, KR5, KI5)                                 \
    do                                                                          \
    {                                                                           \
        /* ---- Wave 1: slots 0,1,2 ---- */                                     \
        __m512d k0r = _mm512_load_pd(KR0), k0i = _mm512_load_pd(KI0);           \
        __m512d k1r = _mm512_load_pd(KR1), k1i = _mm512_load_pd(KI1);           \
        __m512d k2r = _mm512_load_pd(KR2), k2i = _mm512_load_pd(KI2);           \
        /* A: ai*wi, ai*wr (inputs only, no overwrite) */                       \
        __m512d ap0a = _mm512_mul_pd(a0i, k0i), ap0b = _mm512_mul_pd(a0i, k0r); \
        __m512d ap1a = _mm512_mul_pd(a1i, k1i), ap1b = _mm512_mul_pd(a1i, k1r); \
        __m512d ap2a = _mm512_mul_pd(a2i, k2i), ap2b = _mm512_mul_pd(a2i, k2r); \
        /* B: ai*wi, ai*wr */                                                   \
        __m512d bp0a = _mm512_mul_pd(b0i, k0i), bp0b = _mm512_mul_pd(b0i, k0r); \
        __m512d bp1a = _mm512_mul_pd(b1i, k1i), bp1b = _mm512_mul_pd(b1i, k1r); \
        __m512d bp2a = _mm512_mul_pd(b2i, k2i), bp2b = _mm512_mul_pd(b2i, k2r); \
        /* A: ar*wr-ai*wi, ar*wi+ai*wr → temps, then writeback */               \
        {                                                                       \
            __m512d t0r = _mm512_fmsub_pd(a0r, k0r, ap0a);                      \
            __m512d t0i = _mm512_fmadd_pd(a0r, k0i, ap0b);                      \
            __m512d t1r = _mm512_fmsub_pd(a1r, k1r, ap1a);                      \
            __m512d t1i = _mm512_fmadd_pd(a1r, k1i, ap1b);                      \
            __m512d t2r = _mm512_fmsub_pd(a2r, k2r, ap2a);                      \
            __m512d t2i = _mm512_fmadd_pd(a2r, k2i, ap2b);                      \
            a0r = t0r;                                                          \
            a0i = t0i;                                                          \
            a1r = t1r;                                                          \
            a1i = t1i;                                                          \
            a2r = t2r;                                                          \
            a2i = t2i;                                                          \
        }                                                                       \
        /* B: same */                                                           \
        {                                                                       \
            __m512d t0r = _mm512_fmsub_pd(b0r, k0r, bp0a);                      \
            __m512d t0i = _mm512_fmadd_pd(b0r, k0i, bp0b);                      \
            __m512d t1r = _mm512_fmsub_pd(b1r, k1r, bp1a);                      \
            __m512d t1i = _mm512_fmadd_pd(b1r, k1i, bp1b);                      \
            __m512d t2r = _mm512_fmsub_pd(b2r, k2r, bp2a);                      \
            __m512d t2i = _mm512_fmadd_pd(b2r, k2i, bp2b);                      \
            b0r = t0r;                                                          \
            b0i = t0i;                                                          \
            b1r = t1r;                                                          \
            b1i = t1i;                                                          \
            b2r = t2r;                                                          \
            b2i = t2i;                                                          \
        }                                                                       \
        /* ---- Wave 2: slots 3,4,5 ---- */                                     \
        __m512d k3r = _mm512_load_pd(KR3), k3i = _mm512_load_pd(KI3);           \
        __m512d k4r = _mm512_load_pd(KR4), k4i = _mm512_load_pd(KI4);           \
        __m512d k5r = _mm512_load_pd(KR5), k5i = _mm512_load_pd(KI5);           \
        __m512d ap3a = _mm512_mul_pd(a3i, k3i), ap3b = _mm512_mul_pd(a3i, k3r); \
        __m512d ap4a = _mm512_mul_pd(a4i, k4i), ap4b = _mm512_mul_pd(a4i, k4r); \
        __m512d ap5a = _mm512_mul_pd(a5i, k5i), ap5b = _mm512_mul_pd(a5i, k5r); \
        __m512d bp3a = _mm512_mul_pd(b3i, k3i), bp3b = _mm512_mul_pd(b3i, k3r); \
        __m512d bp4a = _mm512_mul_pd(b4i, k4i), bp4b = _mm512_mul_pd(b4i, k4r); \
        __m512d bp5a = _mm512_mul_pd(b5i, k5i), bp5b = _mm512_mul_pd(b5i, k5r); \
        {                                                                       \
            __m512d t3r = _mm512_fmsub_pd(a3r, k3r, ap3a);                      \
            __m512d t3i = _mm512_fmadd_pd(a3r, k3i, ap3b);                      \
            __m512d t4r = _mm512_fmsub_pd(a4r, k4r, ap4a);                      \
            __m512d t4i = _mm512_fmadd_pd(a4r, k4i, ap4b);                      \
            __m512d t5r = _mm512_fmsub_pd(a5r, k5r, ap5a);                      \
            __m512d t5i = _mm512_fmadd_pd(a5r, k5i, ap5b);                      \
            a3r = t3r;                                                          \
            a3i = t3i;                                                          \
            a4r = t4r;                                                          \
            a4i = t4i;                                                          \
            a5r = t5r;                                                          \
            a5i = t5i;                                                          \
        }                                                                       \
        {                                                                       \
            __m512d t3r = _mm512_fmsub_pd(b3r, k3r, bp3a);                      \
            __m512d t3i = _mm512_fmadd_pd(b3r, k3i, bp3b);                      \
            __m512d t4r = _mm512_fmsub_pd(b4r, k4r, bp4a);                      \
            __m512d t4i = _mm512_fmadd_pd(b4r, k4i, bp4b);                      \
            __m512d t5r = _mm512_fmsub_pd(b5r, k5r, bp5a);                      \
            __m512d t5i = _mm512_fmadd_pd(b5r, k5i, bp5b);                      \
            b3r = t3r;                                                          \
            b3i = t3i;                                                          \
            b4r = t4r;                                                          \
            b4i = t4i;                                                          \
            b5r = t5r;                                                          \
            b5i = t5i;                                                          \
        }                                                                       \
    } while (0)

/* ================================================================== */
/*  Tree y0 sum — ZMM version                                         */
/* ================================================================== */

#define TREE_Y0_512(x0, t1, t2, t3, t4, t5, t6, out) \
    do                                               \
    {                                                \
        __m512d _a = _mm512_add_pd(x0, t1);          \
        __m512d _b = _mm512_add_pd(t2, t3);          \
        __m512d _c = _mm512_add_pd(t4, t5);          \
        __m512d _d = _mm512_add_pd(_a, _b);          \
        __m512d _e = _mm512_add_pd(_c, t6);          \
        out = _mm512_add_pd(_d, _e);                 \
    } while (0)

/* ================================================================== */
/*  Forward butterfly — AVX-512, U=2, BLOCKED3 twiddles                */
/* ================================================================== */

static inline __attribute__((always_inline)) __attribute__((target("avx512f"))) void radix7_rader_fwd_avx512(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    const double *restrict f_re, const double *restrict f_im,
    const double *restrict g_re, const double *restrict g_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    double *restrict y5_re, double *restrict y5_im,
    double *restrict y6_re, double *restrict y6_im,
    const double *restrict tw1_re, const double *restrict tw1_im,
    const double *restrict tw2_re, const double *restrict tw2_im,
    const double *restrict tw3_re, const double *restrict tw3_im,
    int K)
{
    int k = 0;

    /* ================================================================ */
    /*  U=2 body: 16 k-values per iteration  (k and k+8)               */
    /* ================================================================ */

    for (; k + R7_512_U2 <= K; k += R7_512_U2)
    {

        /* ---- Load x[0] for A (k) and B (k+8) ---- */
        __m512d ax0r = _mm512_load_pd(&a_re[k]);
        __m512d ax0i = _mm512_load_pd(&a_im[k]);
        __m512d bx0r = _mm512_load_pd(&a_re[k + R7_512_VW]);
        __m512d bx0i = _mm512_load_pd(&a_im[k + R7_512_VW]);

        /* ---- Load raw inputs for A and B ---- */
        __m512d axbr = _mm512_load_pd(&b_re[k]), axbi = _mm512_load_pd(&b_im[k]);
        __m512d axcr = _mm512_load_pd(&c_re[k]), axci = _mm512_load_pd(&c_im[k]);
        __m512d axdr = _mm512_load_pd(&d_re[k]), axdi = _mm512_load_pd(&d_im[k]);
        __m512d axer = _mm512_load_pd(&e_re[k]), axei = _mm512_load_pd(&e_im[k]);
        __m512d axfr = _mm512_load_pd(&f_re[k]), axfi = _mm512_load_pd(&f_im[k]);
        __m512d axgr = _mm512_load_pd(&g_re[k]), axgi = _mm512_load_pd(&g_im[k]);

        __m512d bxbr = _mm512_load_pd(&b_re[k + R7_512_VW]), bxbi = _mm512_load_pd(&b_im[k + R7_512_VW]);
        __m512d bxcr = _mm512_load_pd(&c_re[k + R7_512_VW]), bxci = _mm512_load_pd(&c_im[k + R7_512_VW]);
        __m512d bxdr = _mm512_load_pd(&d_re[k + R7_512_VW]), bxdi = _mm512_load_pd(&d_im[k + R7_512_VW]);
        __m512d bxer = _mm512_load_pd(&e_re[k + R7_512_VW]), bxei = _mm512_load_pd(&e_im[k + R7_512_VW]);
        __m512d bxfr = _mm512_load_pd(&f_re[k + R7_512_VW]), bxfi = _mm512_load_pd(&f_im[k + R7_512_VW]);
        __m512d bxgr = _mm512_load_pd(&g_re[k + R7_512_VW]), bxgi = _mm512_load_pd(&g_im[k + R7_512_VW]);

        /* ---- Load twiddles for A and B ---- */
        __m512d aw1r = _mm512_load_pd(&tw1_re[k]), aw1i = _mm512_load_pd(&tw1_im[k]);
        __m512d aw2r = _mm512_load_pd(&tw2_re[k]), aw2i = _mm512_load_pd(&tw2_im[k]);
        __m512d aw3r = _mm512_load_pd(&tw3_re[k]), aw3i = _mm512_load_pd(&tw3_im[k]);
        __m512d bw1r = _mm512_load_pd(&tw1_re[k + R7_512_VW]), bw1i = _mm512_load_pd(&tw1_im[k + R7_512_VW]);
        __m512d bw2r = _mm512_load_pd(&tw2_re[k + R7_512_VW]), bw2i = _mm512_load_pd(&tw2_im[k + R7_512_VW]);
        __m512d bw3r = _mm512_load_pd(&tw3_re[k + R7_512_VW]), bw3i = _mm512_load_pd(&tw3_im[k + R7_512_VW]);

        /* ---- Apply twiddles: interleave A and B for ILP ---- */
        __m512d at1r, at1i, at2r, at2i, at3r, at3i;
        __m512d bt1r, bt1i, bt2r, bt2i, bt3r, bt3i;
        cmul_512(axbr, axbi, aw1r, aw1i, &at1r, &at1i);
        cmul_512(bxbr, bxbi, bw1r, bw1i, &bt1r, &bt1i);
        cmul_512(axcr, axci, aw2r, aw2i, &at2r, &at2i);
        cmul_512(bxcr, bxci, bw2r, bw2i, &bt2r, &bt2i);
        cmul_512(axdr, axdi, aw3r, aw3i, &at3r, &at3i);
        cmul_512(bxdr, bxdi, bw3r, bw3i, &bt3r, &bt3i);

        /* Derive W4-W6 */
        __m512d aw4r, aw4i, aw5r, aw5i, aw6r, aw6i;
        __m512d bw4r, bw4i, bw5r, bw5i, bw6r, bw6i;
        cmul_512(aw1r, aw1i, aw3r, aw3i, &aw4r, &aw4i);
        cmul_512(bw1r, bw1i, bw3r, bw3i, &bw4r, &bw4i);
        cmul_512(aw2r, aw2i, aw3r, aw3i, &aw5r, &aw5i);
        cmul_512(bw2r, bw2i, bw3r, bw3i, &bw5r, &bw5i);
        cmul_512(aw3r, aw3i, aw3r, aw3i, &aw6r, &aw6i);
        cmul_512(bw3r, bw3i, bw3r, bw3i, &bw6r, &bw6i);

        __m512d at4r, at4i, at5r, at5i, at6r, at6i;
        __m512d bt4r, bt4i, bt5r, bt5i, bt6r, bt6i;
        cmul_512(axer, axei, aw4r, aw4i, &at4r, &at4i);
        cmul_512(bxer, bxei, bw4r, bw4i, &bt4r, &bt4i);
        cmul_512(axfr, axfi, aw5r, aw5i, &at5r, &at5i);
        cmul_512(bxfr, bxfi, bw5r, bw5i, &bt5r, &bt5i);
        cmul_512(axgr, axgi, aw6r, aw6i, &at6r, &at6i);
        cmul_512(bxgr, bxgi, bw6r, bw6i, &bt6r, &bt6i);

        /* ---- DC: tree sum, store early to free x0 regs ---- */
        __m512d adc_r, adc_i, bdc_r, bdc_i;
        TREE_Y0_512(ax0r, at1r, at2r, at3r, at4r, at5r, at6r, adc_r);
        TREE_Y0_512(ax0i, at1i, at2i, at3i, at4i, at5i, at6i, adc_i);
        TREE_Y0_512(bx0r, bt1r, bt2r, bt3r, bt4r, bt5r, bt6r, bdc_r);
        TREE_Y0_512(bx0i, bt1i, bt2i, bt3i, bt4i, bt5i, bt6i, bdc_i);
        _mm512_store_pd(&y0_re[k], adc_r);
        _mm512_store_pd(&y0_im[k], adc_i);
        _mm512_store_pd(&y0_re[k + R7_512_VW], bdc_r);
        _mm512_store_pd(&y0_im[k + R7_512_VW], bdc_i);

        /* ---- Rader permute: {1,3,2,6,4,5} ---- */
        __m512d as0r = at1r, as0i = at1i, as1r = at3r, as1i = at3i;
        __m512d as2r = at2r, as2i = at2i, as3r = at6r, as3i = at6i;
        __m512d as4r = at4r, as4i = at4i, as5r = at5r, as5i = at5i;

        __m512d bs0r = bt1r, bs0i = bt1i, bs1r = bt3r, bs1i = bt3i;
        __m512d bs2r = bt2r, bs2i = bt2i, bs3r = bt6r, bs3i = bt6i;
        __m512d bs4r = bt4r, bs4i = bt4i, bs5r = bt5r, bs5i = bt5i;

        /* ---- DFT6 forward: A then B (compiler interleaves) ---- */
        DFT6_FWD_512(as0r, as0i, as1r, as1i, as2r, as2i,
                     as3r, as3i, as4r, as4i, as5r, as5i);
        DFT6_FWD_512(bs0r, bs0i, bs1r, bs1i, bs2r, bs2i,
                     bs3r, bs3i, bs4r, bs4i, bs5r, bs5i);

        /* ---- U=2 pointwise multiply (shared kernel loads) ---- */
        PW_RR6_U2(as0r, as0i, as1r, as1i, as2r, as2i,
                  as3r, as3i, as4r, as4i, as5r, as5i,
                  bs0r, bs0i, bs1r, bs1i, bs2r, bs2i,
                  bs3r, bs3i, bs4r, bs4i, bs5r, bs5i,
                  RK7_FR0, RK7_FI0, RK7_FR1, RK7_FI1, RK7_FR2, RK7_FI2,
                  RK7_FR3, RK7_FI3, RK7_FR4, RK7_FI4, RK7_FR5, RK7_FI5);

        /* ---- DFT6 backward: A then B ---- */
        DFT6_BWD_512(as0r, as0i, as1r, as1i, as2r, as2i,
                     as3r, as3i, as4r, as4i, as5r, as5i);
        DFT6_BWD_512(bs0r, bs0i, bs1r, bs1i, bs2r, bs2i,
                     bs3r, bs3i, bs4r, bs4i, bs5r, bs5i);

        /* ---- Reload x0 from L1 (freed after DC store) ---- */
        ax0r = _mm512_load_pd(&a_re[k]);
        ax0i = _mm512_load_pd(&a_im[k]);
        bx0r = _mm512_load_pd(&a_re[k + R7_512_VW]);
        bx0i = _mm512_load_pd(&a_im[k + R7_512_VW]);

        /* ---- Output un-permute + store: inv_perm = {1,5,4,6,2,3} ---- */
        _mm512_store_pd(&y1_re[k], _mm512_add_pd(ax0r, as0r));
        _mm512_store_pd(&y1_im[k], _mm512_add_pd(ax0i, as0i));
        _mm512_store_pd(&y5_re[k], _mm512_add_pd(ax0r, as1r));
        _mm512_store_pd(&y5_im[k], _mm512_add_pd(ax0i, as1i));
        _mm512_store_pd(&y4_re[k], _mm512_add_pd(ax0r, as2r));
        _mm512_store_pd(&y4_im[k], _mm512_add_pd(ax0i, as2i));
        _mm512_store_pd(&y6_re[k], _mm512_add_pd(ax0r, as3r));
        _mm512_store_pd(&y6_im[k], _mm512_add_pd(ax0i, as3i));
        _mm512_store_pd(&y2_re[k], _mm512_add_pd(ax0r, as4r));
        _mm512_store_pd(&y2_im[k], _mm512_add_pd(ax0i, as4i));
        _mm512_store_pd(&y3_re[k], _mm512_add_pd(ax0r, as5r));
        _mm512_store_pd(&y3_im[k], _mm512_add_pd(ax0i, as5i));

        _mm512_store_pd(&y1_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs0r));
        _mm512_store_pd(&y1_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs0i));
        _mm512_store_pd(&y5_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs1r));
        _mm512_store_pd(&y5_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs1i));
        _mm512_store_pd(&y4_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs2r));
        _mm512_store_pd(&y4_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs2i));
        _mm512_store_pd(&y6_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs3r));
        _mm512_store_pd(&y6_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs3i));
        _mm512_store_pd(&y2_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs4r));
        _mm512_store_pd(&y2_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs4i));
        _mm512_store_pd(&y3_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs5r));
        _mm512_store_pd(&y3_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs5i));
    }

    /* ================================================================ */
    /*  U=1 remainder: 8 k-values                                       */
    /* ================================================================ */

    for (; k + R7_512_VW <= K; k += R7_512_VW)
    {
        __m512d x0r = _mm512_load_pd(&a_re[k]), x0i = _mm512_load_pd(&a_im[k]);
        __m512d xbr = _mm512_load_pd(&b_re[k]), xbi = _mm512_load_pd(&b_im[k]);
        __m512d xcr = _mm512_load_pd(&c_re[k]), xci = _mm512_load_pd(&c_im[k]);
        __m512d xdr = _mm512_load_pd(&d_re[k]), xdi = _mm512_load_pd(&d_im[k]);
        __m512d xer = _mm512_load_pd(&e_re[k]), xei = _mm512_load_pd(&e_im[k]);
        __m512d xfr = _mm512_load_pd(&f_re[k]), xfi = _mm512_load_pd(&f_im[k]);
        __m512d xgr = _mm512_load_pd(&g_re[k]), xgi = _mm512_load_pd(&g_im[k]);

        __m512d w1r = _mm512_load_pd(&tw1_re[k]), w1i = _mm512_load_pd(&tw1_im[k]);
        __m512d w2r = _mm512_load_pd(&tw2_re[k]), w2i = _mm512_load_pd(&tw2_im[k]);
        __m512d w3r = _mm512_load_pd(&tw3_re[k]), w3i = _mm512_load_pd(&tw3_im[k]);

        __m512d t1r, t1i, t2r, t2i, t3r, t3i;
        cmul_512(xbr, xbi, w1r, w1i, &t1r, &t1i);
        cmul_512(xcr, xci, w2r, w2i, &t2r, &t2i);
        cmul_512(xdr, xdi, w3r, w3i, &t3r, &t3i);
        __m512d w4r, w4i, w5r, w5i, w6r, w6i;
        cmul_512(w1r, w1i, w3r, w3i, &w4r, &w4i);
        cmul_512(w2r, w2i, w3r, w3i, &w5r, &w5i);
        cmul_512(w3r, w3i, w3r, w3i, &w6r, &w6i);
        __m512d t4r, t4i, t5r, t5i, t6r, t6i;
        cmul_512(xer, xei, w4r, w4i, &t4r, &t4i);
        cmul_512(xfr, xfi, w5r, w5i, &t5r, &t5i);
        cmul_512(xgr, xgi, w6r, w6i, &t6r, &t6i);

        __m512d dcr, dci;
        TREE_Y0_512(x0r, t1r, t2r, t3r, t4r, t5r, t6r, dcr);
        TREE_Y0_512(x0i, t1i, t2i, t3i, t4i, t5i, t6i, dci);
        _mm512_store_pd(&y0_re[k], dcr);
        _mm512_store_pd(&y0_im[k], dci);

        __m512d s0r = t1r, s0i = t1i, s1r = t3r, s1i = t3i;
        __m512d s2r = t2r, s2i = t2i, s3r = t6r, s3i = t6i;
        __m512d s4r = t4r, s4i = t4i, s5r = t5r, s5i = t5i;
        DFT6_FWD_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i);
        PW_RR6_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i,
                   RK7_FR0, RK7_FI0, RK7_FR1, RK7_FI1, RK7_FR2, RK7_FI2,
                   RK7_FR3, RK7_FI3, RK7_FR4, RK7_FI4, RK7_FR5, RK7_FI5);
        DFT6_BWD_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i);

        x0r = _mm512_load_pd(&a_re[k]);
        x0i = _mm512_load_pd(&a_im[k]);
        _mm512_store_pd(&y1_re[k], _mm512_add_pd(x0r, s0r));
        _mm512_store_pd(&y1_im[k], _mm512_add_pd(x0i, s0i));
        _mm512_store_pd(&y5_re[k], _mm512_add_pd(x0r, s1r));
        _mm512_store_pd(&y5_im[k], _mm512_add_pd(x0i, s1i));
        _mm512_store_pd(&y4_re[k], _mm512_add_pd(x0r, s2r));
        _mm512_store_pd(&y4_im[k], _mm512_add_pd(x0i, s2i));
        _mm512_store_pd(&y6_re[k], _mm512_add_pd(x0r, s3r));
        _mm512_store_pd(&y6_im[k], _mm512_add_pd(x0i, s3i));
        _mm512_store_pd(&y2_re[k], _mm512_add_pd(x0r, s4r));
        _mm512_store_pd(&y2_im[k], _mm512_add_pd(x0i, s4i));
        _mm512_store_pd(&y3_re[k], _mm512_add_pd(x0r, s5r));
        _mm512_store_pd(&y3_im[k], _mm512_add_pd(x0i, s5i));
    }

    /* Scalar tail for k % 8 remainder */
    for (; k < K; k++)
    {
        double x0r_ = a_re[k], x0i_ = a_im[k];
        double _w1r = tw1_re[k], _w1i = tw1_im[k], _w2r = tw2_re[k], _w2i = tw2_im[k];
        double _w3r = tw3_re[k], _w3i = tw3_im[k];
        double _t1r = b_re[k] * _w1r - b_im[k] * _w1i, _t1i = b_re[k] * _w1i + b_im[k] * _w1r;
        double _t2r = c_re[k] * _w2r - c_im[k] * _w2i, _t2i = c_re[k] * _w2i + c_im[k] * _w2r;
        double _t3r = d_re[k] * _w3r - d_im[k] * _w3i, _t3i = d_re[k] * _w3i + d_im[k] * _w3r;
        double _w4r = _w1r * _w3r - _w1i * _w3i, _w4i = _w1r * _w3i + _w1i * _w3r;
        double _w5r = _w2r * _w3r - _w2i * _w3i, _w5i = _w2r * _w3i + _w2i * _w3r;
        double _w6r = _w3r * _w3r - _w3i * _w3i, _w6i = 2.0 * _w3r * _w3i;
        double _t4r = e_re[k] * _w4r - e_im[k] * _w4i, _t4i = e_re[k] * _w4i + e_im[k] * _w4r;
        double _t5r = f_re[k] * _w5r - f_im[k] * _w5i, _t5i = f_re[k] * _w5i + f_im[k] * _w5r;
        double _t6r = g_re[k] * _w6r - g_im[k] * _w6i, _t6i = g_re[k] * _w6i + g_im[k] * _w6r;
        y0_re[k] = x0r_ + _t1r + _t2r + _t3r + _t4r + _t5r + _t6r;
        y0_im[k] = x0i_ + _t1i + _t2i + _t3i + _t4i + _t5i + _t6i;
        double ar[6] = {_t1r, _t3r, _t2r, _t6r, _t4r, _t5r};
        double ai[6] = {_t1i, _t3i, _t2i, _t6i, _t4i, _t5i};
        double Ar[6], Ai[6], Cr[6], Ci[6], cr[6], ci[6];
        {
            double s1 = ar[2] + ar[4], s2 = ai[2] + ai[4], d1 = ar[2] - ar[4], d2 = ai[2] - ai[4];
            double E0r = ar[0] + s1, E0i = ai[0] + s2, br = ar[0] - 0.5 * s1, bi = ai[0] - 0.5 * s2;
            double E1r = br + 0.86602540378443864676 * d2, E1i = bi - 0.86602540378443864676 * d1;
            double E2r = br - 0.86602540378443864676 * d2, E2i = bi + 0.86602540378443864676 * d1;
            double s3 = ar[3] + ar[5], s4 = ai[3] + ai[5], d3 = ar[3] - ar[5], d4 = ai[3] - ai[5];
            double O0r = ar[1] + s3, O0i = ai[1] + s4, br2 = ar[1] - 0.5 * s3, bi2 = ai[1] - 0.5 * s4;
            double O1r = br2 + 0.86602540378443864676 * d4, O1i = bi2 - 0.86602540378443864676 * d3;
            double O2r = br2 - 0.86602540378443864676 * d4, O2i = bi2 + 0.86602540378443864676 * d3;
            Ar[0] = E0r + O0r;
            Ai[0] = E0i + O0i;
            Ar[3] = E0r - O0r;
            Ai[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 + O1i * 0.86602540378443864676;
            double T1i_ = O1r * (-0.86602540378443864676) + O1i * 0.5;
            Ar[1] = E1r + T1r_;
            Ai[1] = E1i + T1i_;
            Ar[4] = E1r - T1r_;
            Ai[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) + O2i * 0.86602540378443864676;
            double T2i_ = O2r * (-0.86602540378443864676) + O2i * (-0.5);
            Ar[2] = E2r + T2r_;
            Ai[2] = E2i + T2i_;
            Ar[5] = E2r - T2r_;
            Ai[5] = E2i - T2i_;
        }
        for (int j = 0; j < 6; j++)
        {
            Cr[j] = Ar[j] * RK7S_FWD_RE[j] - Ai[j] * RK7S_FWD_IM[j];
            Ci[j] = Ar[j] * RK7S_FWD_IM[j] + Ai[j] * RK7S_FWD_RE[j];
        }
        {
            double s1 = Cr[2] + Cr[4], s2 = Ci[2] + Ci[4], d1 = Cr[2] - Cr[4], d2 = Ci[2] - Ci[4];
            double E0r = Cr[0] + s1, E0i = Ci[0] + s2, br = Cr[0] - 0.5 * s1, bi = Ci[0] - 0.5 * s2;
            double E1r = br - 0.86602540378443864676 * d2, E1i = bi + 0.86602540378443864676 * d1;
            double E2r = br + 0.86602540378443864676 * d2, E2i = bi - 0.86602540378443864676 * d1;
            double s3 = Cr[3] + Cr[5], s4 = Ci[3] + Ci[5], d3 = Cr[3] - Cr[5], d4 = Ci[3] - Ci[5];
            double O0r = Cr[1] + s3, O0i = Ci[1] + s4, br2 = Cr[1] - 0.5 * s3, bi2 = Ci[1] - 0.5 * s4;
            double O1r = br2 - 0.86602540378443864676 * d4, O1i = bi2 + 0.86602540378443864676 * d3;
            double O2r = br2 + 0.86602540378443864676 * d4, O2i = bi2 - 0.86602540378443864676 * d3;
            cr[0] = E0r + O0r;
            ci[0] = E0i + O0i;
            cr[3] = E0r - O0r;
            ci[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 - O1i * 0.86602540378443864676;
            double T1i_ = O1r * 0.86602540378443864676 + O1i * 0.5;
            cr[1] = E1r + T1r_;
            ci[1] = E1i + T1i_;
            cr[4] = E1r - T1r_;
            ci[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) - O2i * 0.86602540378443864676;
            double T2i_ = O2r * 0.86602540378443864676 + O2i * (-0.5);
            cr[2] = E2r + T2r_;
            ci[2] = E2i + T2i_;
            cr[5] = E2r - T2r_;
            ci[5] = E2i - T2i_;
        }
        y1_re[k] = x0r_ + cr[0];
        y1_im[k] = x0i_ + ci[0];
        y5_re[k] = x0r_ + cr[1];
        y5_im[k] = x0i_ + ci[1];
        y4_re[k] = x0r_ + cr[2];
        y4_im[k] = x0i_ + ci[2];
        y6_re[k] = x0r_ + cr[3];
        y6_im[k] = x0i_ + ci[3];
        y2_re[k] = x0r_ + cr[4];
        y2_im[k] = x0i_ + ci[4];
        y3_re[k] = x0r_ + cr[5];
        y3_im[k] = x0i_ + ci[5];
    }
}

/* ================================================================== */
/*  Backward butterfly — AVX-512, U=2                                  */
/*  Rader IDFT on raw inputs, then conj twiddle outputs                */
/* ================================================================== */

static inline __attribute__((always_inline)) __attribute__((target("avx512f"))) void radix7_rader_bwd_avx512(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    const double *restrict f_re, const double *restrict f_im,
    const double *restrict g_re, const double *restrict g_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    double *restrict y5_re, double *restrict y5_im,
    double *restrict y6_re, double *restrict y6_im,
    const double *restrict tw1_re, const double *restrict tw1_im,
    const double *restrict tw2_re, const double *restrict tw2_im,
    const double *restrict tw3_re, const double *restrict tw3_im,
    int K)
{
    int k = 0;
    const __m512d vzero = _mm512_setzero_pd();

    /* U=2 body */
    for (; k + R7_512_U2 <= K; k += R7_512_U2)
    {

        /* Load raw inputs (no twiddle for backward Rader) */
        __m512d ax0r = _mm512_load_pd(&a_re[k]), ax0i = _mm512_load_pd(&a_im[k]);
        __m512d at1r = _mm512_load_pd(&b_re[k]), at1i = _mm512_load_pd(&b_im[k]);
        __m512d at2r = _mm512_load_pd(&c_re[k]), at2i = _mm512_load_pd(&c_im[k]);
        __m512d at3r = _mm512_load_pd(&d_re[k]), at3i = _mm512_load_pd(&d_im[k]);
        __m512d at4r = _mm512_load_pd(&e_re[k]), at4i = _mm512_load_pd(&e_im[k]);
        __m512d at5r = _mm512_load_pd(&f_re[k]), at5i = _mm512_load_pd(&f_im[k]);
        __m512d at6r = _mm512_load_pd(&g_re[k]), at6i = _mm512_load_pd(&g_im[k]);

        __m512d bx0r = _mm512_load_pd(&a_re[k + R7_512_VW]), bx0i = _mm512_load_pd(&a_im[k + R7_512_VW]);
        __m512d bt1r = _mm512_load_pd(&b_re[k + R7_512_VW]), bt1i = _mm512_load_pd(&b_im[k + R7_512_VW]);
        __m512d bt2r = _mm512_load_pd(&c_re[k + R7_512_VW]), bt2i = _mm512_load_pd(&c_im[k + R7_512_VW]);
        __m512d bt3r = _mm512_load_pd(&d_re[k + R7_512_VW]), bt3i = _mm512_load_pd(&d_im[k + R7_512_VW]);
        __m512d bt4r = _mm512_load_pd(&e_re[k + R7_512_VW]), bt4i = _mm512_load_pd(&e_im[k + R7_512_VW]);
        __m512d bt5r = _mm512_load_pd(&f_re[k + R7_512_VW]), bt5i = _mm512_load_pd(&f_im[k + R7_512_VW]);
        __m512d bt6r = _mm512_load_pd(&g_re[k + R7_512_VW]), bt6i = _mm512_load_pd(&g_im[k + R7_512_VW]);

        /* DC tree sum, store early */
        __m512d adc_r, adc_i, bdc_r, bdc_i;
        TREE_Y0_512(ax0r, at1r, at2r, at3r, at4r, at5r, at6r, adc_r);
        TREE_Y0_512(ax0i, at1i, at2i, at3i, at4i, at5i, at6i, adc_i);
        TREE_Y0_512(bx0r, bt1r, bt2r, bt3r, bt4r, bt5r, bt6r, bdc_r);
        TREE_Y0_512(bx0i, bt1i, bt2i, bt3i, bt4i, bt5i, bt6i, bdc_i);
        _mm512_store_pd(&y0_re[k], adc_r);
        _mm512_store_pd(&y0_im[k], adc_i);
        _mm512_store_pd(&y0_re[k + R7_512_VW], bdc_r);
        _mm512_store_pd(&y0_im[k + R7_512_VW], bdc_i);

        /* Rader permute */
        __m512d as0r = at1r, as0i = at1i, as1r = at3r, as1i = at3i;
        __m512d as2r = at2r, as2i = at2i, as3r = at6r, as3i = at6i;
        __m512d as4r = at4r, as4i = at4i, as5r = at5r, as5i = at5i;
        __m512d bs0r = bt1r, bs0i = bt1i, bs1r = bt3r, bs1i = bt3i;
        __m512d bs2r = bt2r, bs2i = bt2i, bs3r = bt6r, bs3i = bt6i;
        __m512d bs4r = bt4r, bs4i = bt4i, bs5r = bt5r, bs5i = bt5i;

        /* DFT6 fwd → pointwise (U2 shared) → DFT6 bwd */
        DFT6_FWD_512(as0r, as0i, as1r, as1i, as2r, as2i, as3r, as3i, as4r, as4i, as5r, as5i);
        DFT6_FWD_512(bs0r, bs0i, bs1r, bs1i, bs2r, bs2i, bs3r, bs3i, bs4r, bs4i, bs5r, bs5i);

        PW_RR6_U2(as0r, as0i, as1r, as1i, as2r, as2i, as3r, as3i, as4r, as4i, as5r, as5i,
                  bs0r, bs0i, bs1r, bs1i, bs2r, bs2i, bs3r, bs3i, bs4r, bs4i, bs5r, bs5i,
                  RK7_BR0, RK7_BI0, RK7_BR1, RK7_BI1, RK7_BR2, RK7_BI2,
                  RK7_BR3, RK7_BI3, RK7_BR4, RK7_BI4, RK7_BR5, RK7_BI5);

        DFT6_BWD_512(as0r, as0i, as1r, as1i, as2r, as2i, as3r, as3i, as4r, as4i, as5r, as5i);
        DFT6_BWD_512(bs0r, bs0i, bs1r, bs1i, bs2r, bs2i, bs3r, bs3i, bs4r, bs4i, bs5r, bs5i);

        /* Reload x0 for output addition */
        ax0r = _mm512_load_pd(&a_re[k]);
        ax0i = _mm512_load_pd(&a_im[k]);
        bx0r = _mm512_load_pd(&a_re[k + R7_512_VW]);
        bx0i = _mm512_load_pd(&a_im[k + R7_512_VW]);

        /* Raw IDFT7 outputs */
        __m512d ar1r = _mm512_add_pd(ax0r, as0r), ar1i = _mm512_add_pd(ax0i, as0i);
        __m512d ar5r = _mm512_add_pd(ax0r, as1r), ar5i = _mm512_add_pd(ax0i, as1i);
        __m512d ar4r = _mm512_add_pd(ax0r, as2r), ar4i = _mm512_add_pd(ax0i, as2i);
        __m512d ar6r = _mm512_add_pd(ax0r, as3r), ar6i = _mm512_add_pd(ax0i, as3i);
        __m512d ar2r = _mm512_add_pd(ax0r, as4r), ar2i = _mm512_add_pd(ax0i, as4i);
        __m512d ar3r = _mm512_add_pd(ax0r, as5r), ar3i = _mm512_add_pd(ax0i, as5i);

        __m512d br1r = _mm512_add_pd(bx0r, bs0r), br1i = _mm512_add_pd(bx0i, bs0i);
        __m512d br5r = _mm512_add_pd(bx0r, bs1r), br5i = _mm512_add_pd(bx0i, bs1i);
        __m512d br4r = _mm512_add_pd(bx0r, bs2r), br4i = _mm512_add_pd(bx0i, bs2i);
        __m512d br6r = _mm512_add_pd(bx0r, bs3r), br6i = _mm512_add_pd(bx0i, bs3i);
        __m512d br2r = _mm512_add_pd(bx0r, bs4r), br2i = _mm512_add_pd(bx0i, bs4i);
        __m512d br3r = _mm512_add_pd(bx0r, bs5r), br3i = _mm512_add_pd(bx0i, bs5i);

        /* Load twiddles, conjugate, apply AFTER IDFT */
        __m512d aw1r = _mm512_load_pd(&tw1_re[k]), aw1i = _mm512_load_pd(&tw1_im[k]);
        __m512d aw2r = _mm512_load_pd(&tw2_re[k]), aw2i = _mm512_load_pd(&tw2_im[k]);
        __m512d aw3r = _mm512_load_pd(&tw3_re[k]), aw3i = _mm512_load_pd(&tw3_im[k]);
        __m512d aw1n = _mm512_sub_pd(vzero, aw1i);
        __m512d aw2n = _mm512_sub_pd(vzero, aw2i);
        __m512d aw3n = _mm512_sub_pd(vzero, aw3i);
        __m512d aw4r, aw4i, aw5r, aw5i, aw6r, aw6i;
        cmul_512(aw1r, aw1n, aw3r, aw3n, &aw4r, &aw4i);
        cmul_512(aw2r, aw2n, aw3r, aw3n, &aw5r, &aw5i);
        cmul_512(aw3r, aw3n, aw3r, aw3n, &aw6r, &aw6i);

        __m512d bw1r = _mm512_load_pd(&tw1_re[k + R7_512_VW]), bw1i = _mm512_load_pd(&tw1_im[k + R7_512_VW]);
        __m512d bw2r = _mm512_load_pd(&tw2_re[k + R7_512_VW]), bw2i = _mm512_load_pd(&tw2_im[k + R7_512_VW]);
        __m512d bw3r = _mm512_load_pd(&tw3_re[k + R7_512_VW]), bw3i = _mm512_load_pd(&tw3_im[k + R7_512_VW]);
        __m512d bw1n = _mm512_sub_pd(vzero, bw1i);
        __m512d bw2n = _mm512_sub_pd(vzero, bw2i);
        __m512d bw3n = _mm512_sub_pd(vzero, bw3i);
        __m512d bw4r, bw4i, bw5r, bw5i, bw6r, bw6i;
        cmul_512(bw1r, bw1n, bw3r, bw3n, &bw4r, &bw4i);
        cmul_512(bw2r, bw2n, bw3r, bw3n, &bw5r, &bw5i);
        cmul_512(bw3r, bw3n, bw3r, bw3n, &bw6r, &bw6i);

        /* A: apply conj twiddles and store */
        __m512d o1r, o1i;
        cmul_512(ar1r, ar1i, aw1r, aw1n, &o1r, &o1i);
        _mm512_store_pd(&y1_re[k], o1r);
        _mm512_store_pd(&y1_im[k], o1i);
        __m512d o2r, o2i;
        cmul_512(ar2r, ar2i, aw2r, aw2n, &o2r, &o2i);
        _mm512_store_pd(&y2_re[k], o2r);
        _mm512_store_pd(&y2_im[k], o2i);
        __m512d o3r, o3i;
        cmul_512(ar3r, ar3i, aw3r, aw3n, &o3r, &o3i);
        _mm512_store_pd(&y3_re[k], o3r);
        _mm512_store_pd(&y3_im[k], o3i);
        __m512d o4r, o4i;
        cmul_512(ar4r, ar4i, aw4r, aw4i, &o4r, &o4i);
        _mm512_store_pd(&y4_re[k], o4r);
        _mm512_store_pd(&y4_im[k], o4i);
        __m512d o5r, o5i;
        cmul_512(ar5r, ar5i, aw5r, aw5i, &o5r, &o5i);
        _mm512_store_pd(&y5_re[k], o5r);
        _mm512_store_pd(&y5_im[k], o5i);
        __m512d o6r, o6i;
        cmul_512(ar6r, ar6i, aw6r, aw6i, &o6r, &o6i);
        _mm512_store_pd(&y6_re[k], o6r);
        _mm512_store_pd(&y6_im[k], o6i);

        /* B: apply conj twiddles and store */
        cmul_512(br1r, br1i, bw1r, bw1n, &o1r, &o1i);
        _mm512_store_pd(&y1_re[k + R7_512_VW], o1r);
        _mm512_store_pd(&y1_im[k + R7_512_VW], o1i);
        cmul_512(br2r, br2i, bw2r, bw2n, &o2r, &o2i);
        _mm512_store_pd(&y2_re[k + R7_512_VW], o2r);
        _mm512_store_pd(&y2_im[k + R7_512_VW], o2i);
        cmul_512(br3r, br3i, bw3r, bw3n, &o3r, &o3i);
        _mm512_store_pd(&y3_re[k + R7_512_VW], o3r);
        _mm512_store_pd(&y3_im[k + R7_512_VW], o3i);
        cmul_512(br4r, br4i, bw4r, bw4i, &o4r, &o4i);
        _mm512_store_pd(&y4_re[k + R7_512_VW], o4r);
        _mm512_store_pd(&y4_im[k + R7_512_VW], o4i);
        cmul_512(br5r, br5i, bw5r, bw5i, &o5r, &o5i);
        _mm512_store_pd(&y5_re[k + R7_512_VW], o5r);
        _mm512_store_pd(&y5_im[k + R7_512_VW], o5i);
        cmul_512(br6r, br6i, bw6r, bw6i, &o6r, &o6i);
        _mm512_store_pd(&y6_re[k + R7_512_VW], o6r);
        _mm512_store_pd(&y6_im[k + R7_512_VW], o6i);
    }

    /* U=1 remainder */
    for (; k + R7_512_VW <= K; k += R7_512_VW)
    {
        __m512d x0r = _mm512_load_pd(&a_re[k]), x0i = _mm512_load_pd(&a_im[k]);
        __m512d t1r = _mm512_load_pd(&b_re[k]), t1i = _mm512_load_pd(&b_im[k]);
        __m512d t2r = _mm512_load_pd(&c_re[k]), t2i = _mm512_load_pd(&c_im[k]);
        __m512d t3r = _mm512_load_pd(&d_re[k]), t3i = _mm512_load_pd(&d_im[k]);
        __m512d t4r = _mm512_load_pd(&e_re[k]), t4i = _mm512_load_pd(&e_im[k]);
        __m512d t5r = _mm512_load_pd(&f_re[k]), t5i = _mm512_load_pd(&f_im[k]);
        __m512d t6r = _mm512_load_pd(&g_re[k]), t6i = _mm512_load_pd(&g_im[k]);

        __m512d dcr, dci;
        TREE_Y0_512(x0r, t1r, t2r, t3r, t4r, t5r, t6r, dcr);
        TREE_Y0_512(x0i, t1i, t2i, t3i, t4i, t5i, t6i, dci);
        _mm512_store_pd(&y0_re[k], dcr);
        _mm512_store_pd(&y0_im[k], dci);

        __m512d s0r = t1r, s0i = t1i, s1r = t3r, s1i = t3i;
        __m512d s2r = t2r, s2i = t2i, s3r = t6r, s3i = t6i;
        __m512d s4r = t4r, s4i = t4i, s5r = t5r, s5i = t5i;
        DFT6_FWD_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i);
        PW_RR6_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i,
                   RK7_BR0, RK7_BI0, RK7_BR1, RK7_BI1, RK7_BR2, RK7_BI2,
                   RK7_BR3, RK7_BI3, RK7_BR4, RK7_BI4, RK7_BR5, RK7_BI5);
        DFT6_BWD_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i);

        x0r = _mm512_load_pd(&a_re[k]);
        x0i = _mm512_load_pd(&a_im[k]);
        __m512d r1r = _mm512_add_pd(x0r, s0r), r1i = _mm512_add_pd(x0i, s0i);
        __m512d r5r = _mm512_add_pd(x0r, s1r), r5i = _mm512_add_pd(x0i, s1i);
        __m512d r4r = _mm512_add_pd(x0r, s2r), r4i = _mm512_add_pd(x0i, s2i);
        __m512d r6r = _mm512_add_pd(x0r, s3r), r6i = _mm512_add_pd(x0i, s3i);
        __m512d r2r = _mm512_add_pd(x0r, s4r), r2i = _mm512_add_pd(x0i, s4i);
        __m512d r3r = _mm512_add_pd(x0r, s5r), r3i = _mm512_add_pd(x0i, s5i);

        __m512d w1r = _mm512_load_pd(&tw1_re[k]), w1i = _mm512_load_pd(&tw1_im[k]);
        __m512d w2r = _mm512_load_pd(&tw2_re[k]), w2i = _mm512_load_pd(&tw2_im[k]);
        __m512d w3r = _mm512_load_pd(&tw3_re[k]), w3i = _mm512_load_pd(&tw3_im[k]);
        __m512d w1n = _mm512_sub_pd(vzero, w1i), w2n = _mm512_sub_pd(vzero, w2i);
        __m512d w3n = _mm512_sub_pd(vzero, w3i);
        __m512d w4r, w4i, w5r, w5i, w6r, w6i;
        cmul_512(w1r, w1n, w3r, w3n, &w4r, &w4i);
        cmul_512(w2r, w2n, w3r, w3n, &w5r, &w5i);
        cmul_512(w3r, w3n, w3r, w3n, &w6r, &w6i);

        __m512d o1r, o1i;
        cmul_512(r1r, r1i, w1r, w1n, &o1r, &o1i);
        _mm512_store_pd(&y1_re[k], o1r);
        _mm512_store_pd(&y1_im[k], o1i);
        __m512d o2r, o2i;
        cmul_512(r2r, r2i, w2r, w2n, &o2r, &o2i);
        _mm512_store_pd(&y2_re[k], o2r);
        _mm512_store_pd(&y2_im[k], o2i);
        __m512d o3r, o3i;
        cmul_512(r3r, r3i, w3r, w3n, &o3r, &o3i);
        _mm512_store_pd(&y3_re[k], o3r);
        _mm512_store_pd(&y3_im[k], o3i);
        __m512d o4r, o4i;
        cmul_512(r4r, r4i, w4r, w4i, &o4r, &o4i);
        _mm512_store_pd(&y4_re[k], o4r);
        _mm512_store_pd(&y4_im[k], o4i);
        __m512d o5r, o5i;
        cmul_512(r5r, r5i, w5r, w5i, &o5r, &o5i);
        _mm512_store_pd(&y5_re[k], o5r);
        _mm512_store_pd(&y5_im[k], o5i);
        __m512d o6r, o6i;
        cmul_512(r6r, r6i, w6r, w6i, &o6r, &o6i);
        _mm512_store_pd(&y6_re[k], o6r);
        _mm512_store_pd(&y6_im[k], o6i);
    }

    /* Scalar tail */
    for (; k < K; k++)
    {
        double x0r_ = a_re[k], x0i_ = a_im[k];
        double _t1r = b_re[k], _t1i = b_im[k], _t2r = c_re[k], _t2i = c_im[k];
        double _t3r = d_re[k], _t3i = d_im[k], _t4r = e_re[k], _t4i = e_im[k];
        double _t5r = f_re[k], _t5i = f_im[k], _t6r = g_re[k], _t6i = g_im[k];
        double dcr_ = x0r_ + _t1r + _t2r + _t3r + _t4r + _t5r + _t6r;
        double dci_ = x0i_ + _t1i + _t2i + _t3i + _t4i + _t5i + _t6i;
        double ar[6] = {_t1r, _t3r, _t2r, _t6r, _t4r, _t5r};
        double ai[6] = {_t1i, _t3i, _t2i, _t6i, _t4i, _t5i};
        double Ar[6], Ai[6], Cr[6], Ci[6], cr[6], ci[6];
        {
            double s1 = ar[2] + ar[4], s2 = ai[2] + ai[4], d1 = ar[2] - ar[4], d2 = ai[2] - ai[4];
            double E0r = ar[0] + s1, E0i = ai[0] + s2, br = ar[0] - 0.5 * s1, bi = ai[0] - 0.5 * s2;
            double E1r = br + 0.86602540378443864676 * d2, E1i = bi - 0.86602540378443864676 * d1;
            double E2r = br - 0.86602540378443864676 * d2, E2i = bi + 0.86602540378443864676 * d1;
            double s3 = ar[3] + ar[5], s4 = ai[3] + ai[5], d3 = ar[3] - ar[5], d4 = ai[3] - ai[5];
            double O0r = ar[1] + s3, O0i = ai[1] + s4, br2 = ar[1] - 0.5 * s3, bi2 = ai[1] - 0.5 * s4;
            double O1r = br2 + 0.86602540378443864676 * d4, O1i = bi2 - 0.86602540378443864676 * d3;
            double O2r = br2 - 0.86602540378443864676 * d4, O2i = bi2 + 0.86602540378443864676 * d3;
            Ar[0] = E0r + O0r;
            Ai[0] = E0i + O0i;
            Ar[3] = E0r - O0r;
            Ai[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 + O1i * 0.86602540378443864676;
            double T1i_ = O1r * (-0.86602540378443864676) + O1i * 0.5;
            Ar[1] = E1r + T1r_;
            Ai[1] = E1i + T1i_;
            Ar[4] = E1r - T1r_;
            Ai[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) + O2i * 0.86602540378443864676;
            double T2i_ = O2r * (-0.86602540378443864676) + O2i * (-0.5);
            Ar[2] = E2r + T2r_;
            Ai[2] = E2i + T2i_;
            Ar[5] = E2r - T2r_;
            Ai[5] = E2i - T2i_;
        }
        for (int j = 0; j < 6; j++)
        {
            Cr[j] = Ar[j] * RK7S_BWD_RE[j] - Ai[j] * RK7S_BWD_IM[j];
            Ci[j] = Ar[j] * RK7S_BWD_IM[j] + Ai[j] * RK7S_BWD_RE[j];
        }
        {
            double s1 = Cr[2] + Cr[4], s2 = Ci[2] + Ci[4], d1 = Cr[2] - Cr[4], d2 = Ci[2] - Ci[4];
            double E0r = Cr[0] + s1, E0i = Ci[0] + s2, br = Cr[0] - 0.5 * s1, bi = Ci[0] - 0.5 * s2;
            double E1r = br - 0.86602540378443864676 * d2, E1i = bi + 0.86602540378443864676 * d1;
            double E2r = br + 0.86602540378443864676 * d2, E2i = bi - 0.86602540378443864676 * d1;
            double s3 = Cr[3] + Cr[5], s4 = Ci[3] + Ci[5], d3 = Cr[3] - Cr[5], d4 = Ci[3] - Ci[5];
            double O0r = Cr[1] + s3, O0i = Ci[1] + s4, br2 = Cr[1] - 0.5 * s3, bi2 = Ci[1] - 0.5 * s4;
            double O1r = br2 - 0.86602540378443864676 * d4, O1i = bi2 + 0.86602540378443864676 * d3;
            double O2r = br2 + 0.86602540378443864676 * d4, O2i = bi2 - 0.86602540378443864676 * d3;
            cr[0] = E0r + O0r;
            ci[0] = E0i + O0i;
            cr[3] = E0r - O0r;
            ci[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 - O1i * 0.86602540378443864676;
            double T1i_ = O1r * 0.86602540378443864676 + O1i * 0.5;
            cr[1] = E1r + T1r_;
            ci[1] = E1i + T1i_;
            cr[4] = E1r - T1r_;
            ci[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) - O2i * 0.86602540378443864676;
            double T2i_ = O2r * 0.86602540378443864676 + O2i * (-0.5);
            cr[2] = E2r + T2r_;
            ci[2] = E2i + T2i_;
            cr[5] = E2r - T2r_;
            ci[5] = E2i - T2i_;
        }
        double _r1r = x0r_ + cr[0], _r1i = x0i_ + ci[0], _r5r = x0r_ + cr[1], _r5i = x0i_ + ci[1];
        double _r4r = x0r_ + cr[2], _r4i = x0i_ + ci[2], _r6r = x0r_ + cr[3], _r6i = x0i_ + ci[3];
        double _r2r = x0r_ + cr[4], _r2i = x0i_ + ci[4], _r3r = x0r_ + cr[5], _r3i = x0i_ + ci[5];
        double _w1r = tw1_re[k], _w1i = tw1_im[k], _w2r = tw2_re[k], _w2i = tw2_im[k];
        double _w3r = tw3_re[k], _w3i = tw3_im[k];
        double _w4r = _w1r * _w3r - _w1i * _w3i, _w4i = _w1r * _w3i + _w1i * _w3r;
        double _w5r = _w2r * _w3r - _w2i * _w3i, _w5i = _w2r * _w3i + _w2i * _w3r;
        double _w6r = _w3r * _w3r - _w3i * _w3i, _w6i = 2.0 * _w3r * _w3i;
        y0_re[k] = dcr_;
        y0_im[k] = dci_;
        y1_re[k] = _r1r * _w1r + _r1i * _w1i;
        y1_im[k] = -_r1r * _w1i + _r1i * _w1r;
        y2_re[k] = _r2r * _w2r + _r2i * _w2i;
        y2_im[k] = -_r2r * _w2i + _r2i * _w2r;
        y3_re[k] = _r3r * _w3r + _r3i * _w3i;
        y3_im[k] = -_r3r * _w3i + _r3i * _w3r;
        y4_re[k] = _r4r * _w4r + _r4i * _w4i;
        y4_im[k] = -_r4r * _w4i + _r4i * _w4r;
        y5_re[k] = _r5r * _w5r + _r5i * _w5i;
        y5_im[k] = -_r5r * _w5i + _r5i * _w5r;
        y6_re[k] = _r6r * _w6r + _r6i * _w6i;
        y6_im[k] = -_r6r * _w6i + _r6i * _w6r;
    }
}

/* ================================================================== */
/*  N1 forward butterfly — no twiddles (all W=1)                       */
/* ================================================================== */

static inline __attribute__((always_inline)) __attribute__((target("avx512f"))) void radix7_rader_fwd_avx512_N1(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    const double *restrict f_re, const double *restrict f_im,
    const double *restrict g_re, const double *restrict g_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    double *restrict y5_re, double *restrict y5_im,
    double *restrict y6_re, double *restrict y6_im,
    int K)
{
    int k = 0;

    /* U=2 body */
    for (; k + R7_512_U2 <= K; k += R7_512_U2)
    {
        __m512d ax0r = _mm512_load_pd(&a_re[k]), ax0i = _mm512_load_pd(&a_im[k]);
        __m512d at1r = _mm512_load_pd(&b_re[k]), at1i = _mm512_load_pd(&b_im[k]);
        __m512d at2r = _mm512_load_pd(&c_re[k]), at2i = _mm512_load_pd(&c_im[k]);
        __m512d at3r = _mm512_load_pd(&d_re[k]), at3i = _mm512_load_pd(&d_im[k]);
        __m512d at4r = _mm512_load_pd(&e_re[k]), at4i = _mm512_load_pd(&e_im[k]);
        __m512d at5r = _mm512_load_pd(&f_re[k]), at5i = _mm512_load_pd(&f_im[k]);
        __m512d at6r = _mm512_load_pd(&g_re[k]), at6i = _mm512_load_pd(&g_im[k]);

        __m512d bx0r = _mm512_load_pd(&a_re[k + R7_512_VW]), bx0i = _mm512_load_pd(&a_im[k + R7_512_VW]);
        __m512d bt1r = _mm512_load_pd(&b_re[k + R7_512_VW]), bt1i = _mm512_load_pd(&b_im[k + R7_512_VW]);
        __m512d bt2r = _mm512_load_pd(&c_re[k + R7_512_VW]), bt2i = _mm512_load_pd(&c_im[k + R7_512_VW]);
        __m512d bt3r = _mm512_load_pd(&d_re[k + R7_512_VW]), bt3i = _mm512_load_pd(&d_im[k + R7_512_VW]);
        __m512d bt4r = _mm512_load_pd(&e_re[k + R7_512_VW]), bt4i = _mm512_load_pd(&e_im[k + R7_512_VW]);
        __m512d bt5r = _mm512_load_pd(&f_re[k + R7_512_VW]), bt5i = _mm512_load_pd(&f_im[k + R7_512_VW]);
        __m512d bt6r = _mm512_load_pd(&g_re[k + R7_512_VW]), bt6i = _mm512_load_pd(&g_im[k + R7_512_VW]);

        /* DC tree sum */
        __m512d adc_r, adc_i, bdc_r, bdc_i;
        TREE_Y0_512(ax0r, at1r, at2r, at3r, at4r, at5r, at6r, adc_r);
        TREE_Y0_512(ax0i, at1i, at2i, at3i, at4i, at5i, at6i, adc_i);
        TREE_Y0_512(bx0r, bt1r, bt2r, bt3r, bt4r, bt5r, bt6r, bdc_r);
        TREE_Y0_512(bx0i, bt1i, bt2i, bt3i, bt4i, bt5i, bt6i, bdc_i);
        _mm512_store_pd(&y0_re[k], adc_r);
        _mm512_store_pd(&y0_im[k], adc_i);
        _mm512_store_pd(&y0_re[k + R7_512_VW], bdc_r);
        _mm512_store_pd(&y0_im[k + R7_512_VW], bdc_i);

        /* Rader permute {1,3,2,6,4,5} */
        __m512d as0r = at1r, as0i = at1i, as1r = at3r, as1i = at3i;
        __m512d as2r = at2r, as2i = at2i, as3r = at6r, as3i = at6i;
        __m512d as4r = at4r, as4i = at4i, as5r = at5r, as5i = at5i;
        __m512d bs0r = bt1r, bs0i = bt1i, bs1r = bt3r, bs1i = bt3i;
        __m512d bs2r = bt2r, bs2i = bt2i, bs3r = bt6r, bs3i = bt6i;
        __m512d bs4r = bt4r, bs4i = bt4i, bs5r = bt5r, bs5i = bt5i;

        DFT6_FWD_512(as0r, as0i, as1r, as1i, as2r, as2i, as3r, as3i, as4r, as4i, as5r, as5i);
        DFT6_FWD_512(bs0r, bs0i, bs1r, bs1i, bs2r, bs2i, bs3r, bs3i, bs4r, bs4i, bs5r, bs5i);
        PW_RR6_U2(as0r, as0i, as1r, as1i, as2r, as2i, as3r, as3i, as4r, as4i, as5r, as5i,
                  bs0r, bs0i, bs1r, bs1i, bs2r, bs2i, bs3r, bs3i, bs4r, bs4i, bs5r, bs5i,
                  RK7_FR0, RK7_FI0, RK7_FR1, RK7_FI1, RK7_FR2, RK7_FI2,
                  RK7_FR3, RK7_FI3, RK7_FR4, RK7_FI4, RK7_FR5, RK7_FI5);
        DFT6_BWD_512(as0r, as0i, as1r, as1i, as2r, as2i, as3r, as3i, as4r, as4i, as5r, as5i);
        DFT6_BWD_512(bs0r, bs0i, bs1r, bs1i, bs2r, bs2i, bs3r, bs3i, bs4r, bs4i, bs5r, bs5i);

        /* Reload x0, output */
        ax0r = _mm512_load_pd(&a_re[k]);
        ax0i = _mm512_load_pd(&a_im[k]);
        bx0r = _mm512_load_pd(&a_re[k + R7_512_VW]);
        bx0i = _mm512_load_pd(&a_im[k + R7_512_VW]);

        _mm512_store_pd(&y1_re[k], _mm512_add_pd(ax0r, as0r));
        _mm512_store_pd(&y1_im[k], _mm512_add_pd(ax0i, as0i));
        _mm512_store_pd(&y5_re[k], _mm512_add_pd(ax0r, as1r));
        _mm512_store_pd(&y5_im[k], _mm512_add_pd(ax0i, as1i));
        _mm512_store_pd(&y4_re[k], _mm512_add_pd(ax0r, as2r));
        _mm512_store_pd(&y4_im[k], _mm512_add_pd(ax0i, as2i));
        _mm512_store_pd(&y6_re[k], _mm512_add_pd(ax0r, as3r));
        _mm512_store_pd(&y6_im[k], _mm512_add_pd(ax0i, as3i));
        _mm512_store_pd(&y2_re[k], _mm512_add_pd(ax0r, as4r));
        _mm512_store_pd(&y2_im[k], _mm512_add_pd(ax0i, as4i));
        _mm512_store_pd(&y3_re[k], _mm512_add_pd(ax0r, as5r));
        _mm512_store_pd(&y3_im[k], _mm512_add_pd(ax0i, as5i));
        _mm512_store_pd(&y1_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs0r));
        _mm512_store_pd(&y1_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs0i));
        _mm512_store_pd(&y5_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs1r));
        _mm512_store_pd(&y5_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs1i));
        _mm512_store_pd(&y4_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs2r));
        _mm512_store_pd(&y4_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs2i));
        _mm512_store_pd(&y6_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs3r));
        _mm512_store_pd(&y6_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs3i));
        _mm512_store_pd(&y2_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs4r));
        _mm512_store_pd(&y2_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs4i));
        _mm512_store_pd(&y3_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs5r));
        _mm512_store_pd(&y3_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs5i));
    }

    /* U=1 remainder */
    for (; k + R7_512_VW <= K; k += R7_512_VW)
    {
        __m512d x0r = _mm512_load_pd(&a_re[k]), x0i = _mm512_load_pd(&a_im[k]);
        __m512d t1r = _mm512_load_pd(&b_re[k]), t1i = _mm512_load_pd(&b_im[k]);
        __m512d t2r = _mm512_load_pd(&c_re[k]), t2i = _mm512_load_pd(&c_im[k]);
        __m512d t3r = _mm512_load_pd(&d_re[k]), t3i = _mm512_load_pd(&d_im[k]);
        __m512d t4r = _mm512_load_pd(&e_re[k]), t4i = _mm512_load_pd(&e_im[k]);
        __m512d t5r = _mm512_load_pd(&f_re[k]), t5i = _mm512_load_pd(&f_im[k]);
        __m512d t6r = _mm512_load_pd(&g_re[k]), t6i = _mm512_load_pd(&g_im[k]);
        __m512d dcr, dci;
        TREE_Y0_512(x0r, t1r, t2r, t3r, t4r, t5r, t6r, dcr);
        TREE_Y0_512(x0i, t1i, t2i, t3i, t4i, t5i, t6i, dci);
        _mm512_store_pd(&y0_re[k], dcr);
        _mm512_store_pd(&y0_im[k], dci);
        __m512d s0r = t1r, s0i = t1i, s1r = t3r, s1i = t3i;
        __m512d s2r = t2r, s2i = t2i, s3r = t6r, s3i = t6i;
        __m512d s4r = t4r, s4i = t4i, s5r = t5r, s5i = t5i;
        DFT6_FWD_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i);
        PW_RR6_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i,
                   RK7_FR0, RK7_FI0, RK7_FR1, RK7_FI1, RK7_FR2, RK7_FI2,
                   RK7_FR3, RK7_FI3, RK7_FR4, RK7_FI4, RK7_FR5, RK7_FI5);
        DFT6_BWD_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i);
        x0r = _mm512_load_pd(&a_re[k]);
        x0i = _mm512_load_pd(&a_im[k]);
        _mm512_store_pd(&y1_re[k], _mm512_add_pd(x0r, s0r));
        _mm512_store_pd(&y1_im[k], _mm512_add_pd(x0i, s0i));
        _mm512_store_pd(&y5_re[k], _mm512_add_pd(x0r, s1r));
        _mm512_store_pd(&y5_im[k], _mm512_add_pd(x0i, s1i));
        _mm512_store_pd(&y4_re[k], _mm512_add_pd(x0r, s2r));
        _mm512_store_pd(&y4_im[k], _mm512_add_pd(x0i, s2i));
        _mm512_store_pd(&y6_re[k], _mm512_add_pd(x0r, s3r));
        _mm512_store_pd(&y6_im[k], _mm512_add_pd(x0i, s3i));
        _mm512_store_pd(&y2_re[k], _mm512_add_pd(x0r, s4r));
        _mm512_store_pd(&y2_im[k], _mm512_add_pd(x0i, s4i));
        _mm512_store_pd(&y3_re[k], _mm512_add_pd(x0r, s5r));
        _mm512_store_pd(&y3_im[k], _mm512_add_pd(x0i, s5i));
    }

    /* Scalar tail */
    for (; k < K; k++)
    {
        double x0r_ = a_re[k], x0i_ = a_im[k];
        double _t1r = b_re[k], _t1i = b_im[k], _t2r = c_re[k], _t2i = c_im[k];
        double _t3r = d_re[k], _t3i = d_im[k], _t4r = e_re[k], _t4i = e_im[k];
        double _t5r = f_re[k], _t5i = f_im[k], _t6r = g_re[k], _t6i = g_im[k];
        y0_re[k] = x0r_ + _t1r + _t2r + _t3r + _t4r + _t5r + _t6r;
        y0_im[k] = x0i_ + _t1i + _t2i + _t3i + _t4i + _t5i + _t6i;
        double ar[6] = {_t1r, _t3r, _t2r, _t6r, _t4r, _t5r};
        double ai[6] = {_t1i, _t3i, _t2i, _t6i, _t4i, _t5i};
        double Ar[6], Ai[6], Cr[6], Ci[6], cr[6], ci[6];
        {
            double s1 = ar[2] + ar[4], s2 = ai[2] + ai[4], d1 = ar[2] - ar[4], d2 = ai[2] - ai[4];
            double E0r = ar[0] + s1, E0i = ai[0] + s2, br = ar[0] - 0.5 * s1, bi = ai[0] - 0.5 * s2;
            double E1r = br + 0.86602540378443864676 * d2, E1i = bi - 0.86602540378443864676 * d1;
            double E2r = br - 0.86602540378443864676 * d2, E2i = bi + 0.86602540378443864676 * d1;
            double s3 = ar[3] + ar[5], s4 = ai[3] + ai[5], d3 = ar[3] - ar[5], d4 = ai[3] - ai[5];
            double O0r = ar[1] + s3, O0i = ai[1] + s4, br2 = ar[1] - 0.5 * s3, bi2 = ai[1] - 0.5 * s4;
            double O1r = br2 + 0.86602540378443864676 * d4, O1i = bi2 - 0.86602540378443864676 * d3;
            double O2r = br2 - 0.86602540378443864676 * d4, O2i = bi2 + 0.86602540378443864676 * d3;
            Ar[0] = E0r + O0r;
            Ai[0] = E0i + O0i;
            Ar[3] = E0r - O0r;
            Ai[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 + O1i * 0.86602540378443864676;
            double T1i_ = O1r * (-0.86602540378443864676) + O1i * 0.5;
            Ar[1] = E1r + T1r_;
            Ai[1] = E1i + T1i_;
            Ar[4] = E1r - T1r_;
            Ai[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) + O2i * 0.86602540378443864676;
            double T2i_ = O2r * (-0.86602540378443864676) + O2i * (-0.5);
            Ar[2] = E2r + T2r_;
            Ai[2] = E2i + T2i_;
            Ar[5] = E2r - T2r_;
            Ai[5] = E2i - T2i_;
        }
        for (int j = 0; j < 6; j++)
        {
            Cr[j] = Ar[j] * RK7S_FWD_RE[j] - Ai[j] * RK7S_FWD_IM[j];
            Ci[j] = Ar[j] * RK7S_FWD_IM[j] + Ai[j] * RK7S_FWD_RE[j];
        }
        {
            double s1 = Cr[2] + Cr[4], s2 = Ci[2] + Ci[4], d1 = Cr[2] - Cr[4], d2 = Ci[2] - Ci[4];
            double E0r = Cr[0] + s1, E0i = Ci[0] + s2, br = Cr[0] - 0.5 * s1, bi = Ci[0] - 0.5 * s2;
            double E1r = br - 0.86602540378443864676 * d2, E1i = bi + 0.86602540378443864676 * d1;
            double E2r = br + 0.86602540378443864676 * d2, E2i = bi - 0.86602540378443864676 * d1;
            double s3 = Cr[3] + Cr[5], s4 = Ci[3] + Ci[5], d3 = Cr[3] - Cr[5], d4 = Ci[3] - Ci[5];
            double O0r = Cr[1] + s3, O0i = Ci[1] + s4, br2 = Cr[1] - 0.5 * s3, bi2 = Ci[1] - 0.5 * s4;
            double O1r = br2 - 0.86602540378443864676 * d4, O1i = bi2 + 0.86602540378443864676 * d3;
            double O2r = br2 + 0.86602540378443864676 * d4, O2i = bi2 - 0.86602540378443864676 * d3;
            cr[0] = E0r + O0r;
            ci[0] = E0i + O0i;
            cr[3] = E0r - O0r;
            ci[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 - O1i * 0.86602540378443864676;
            double T1i_ = O1r * 0.86602540378443864676 + O1i * 0.5;
            cr[1] = E1r + T1r_;
            ci[1] = E1i + T1i_;
            cr[4] = E1r - T1r_;
            ci[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) - O2i * 0.86602540378443864676;
            double T2i_ = O2r * 0.86602540378443864676 + O2i * (-0.5);
            cr[2] = E2r + T2r_;
            ci[2] = E2i + T2i_;
            cr[5] = E2r - T2r_;
            ci[5] = E2i - T2i_;
        }
        y1_re[k] = x0r_ + cr[0];
        y1_im[k] = x0i_ + ci[0];
        y5_re[k] = x0r_ + cr[1];
        y5_im[k] = x0i_ + ci[1];
        y4_re[k] = x0r_ + cr[2];
        y4_im[k] = x0i_ + ci[2];
        y6_re[k] = x0r_ + cr[3];
        y6_im[k] = x0i_ + ci[3];
        y2_re[k] = x0r_ + cr[4];
        y2_im[k] = x0i_ + ci[4];
        y3_re[k] = x0r_ + cr[5];
        y3_im[k] = x0i_ + ci[5];
    }
}

/* ================================================================== */
/*  N1 backward butterfly — no twiddles                                */
/* ================================================================== */

static inline __attribute__((always_inline)) __attribute__((target("avx512f"))) void radix7_rader_bwd_avx512_N1(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    const double *restrict f_re, const double *restrict f_im,
    const double *restrict g_re, const double *restrict g_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    double *restrict y5_re, double *restrict y5_im,
    double *restrict y6_re, double *restrict y6_im,
    int K)
{
    int k = 0;

    /* U=2 body */
    for (; k + R7_512_U2 <= K; k += R7_512_U2)
    {
        __m512d ax0r = _mm512_load_pd(&a_re[k]), ax0i = _mm512_load_pd(&a_im[k]);
        __m512d at1r = _mm512_load_pd(&b_re[k]), at1i = _mm512_load_pd(&b_im[k]);
        __m512d at2r = _mm512_load_pd(&c_re[k]), at2i = _mm512_load_pd(&c_im[k]);
        __m512d at3r = _mm512_load_pd(&d_re[k]), at3i = _mm512_load_pd(&d_im[k]);
        __m512d at4r = _mm512_load_pd(&e_re[k]), at4i = _mm512_load_pd(&e_im[k]);
        __m512d at5r = _mm512_load_pd(&f_re[k]), at5i = _mm512_load_pd(&f_im[k]);
        __m512d at6r = _mm512_load_pd(&g_re[k]), at6i = _mm512_load_pd(&g_im[k]);
        __m512d bx0r = _mm512_load_pd(&a_re[k + R7_512_VW]), bx0i = _mm512_load_pd(&a_im[k + R7_512_VW]);
        __m512d bt1r = _mm512_load_pd(&b_re[k + R7_512_VW]), bt1i = _mm512_load_pd(&b_im[k + R7_512_VW]);
        __m512d bt2r = _mm512_load_pd(&c_re[k + R7_512_VW]), bt2i = _mm512_load_pd(&c_im[k + R7_512_VW]);
        __m512d bt3r = _mm512_load_pd(&d_re[k + R7_512_VW]), bt3i = _mm512_load_pd(&d_im[k + R7_512_VW]);
        __m512d bt4r = _mm512_load_pd(&e_re[k + R7_512_VW]), bt4i = _mm512_load_pd(&e_im[k + R7_512_VW]);
        __m512d bt5r = _mm512_load_pd(&f_re[k + R7_512_VW]), bt5i = _mm512_load_pd(&f_im[k + R7_512_VW]);
        __m512d bt6r = _mm512_load_pd(&g_re[k + R7_512_VW]), bt6i = _mm512_load_pd(&g_im[k + R7_512_VW]);

        __m512d adc_r, adc_i, bdc_r, bdc_i;
        TREE_Y0_512(ax0r, at1r, at2r, at3r, at4r, at5r, at6r, adc_r);
        TREE_Y0_512(ax0i, at1i, at2i, at3i, at4i, at5i, at6i, adc_i);
        TREE_Y0_512(bx0r, bt1r, bt2r, bt3r, bt4r, bt5r, bt6r, bdc_r);
        TREE_Y0_512(bx0i, bt1i, bt2i, bt3i, bt4i, bt5i, bt6i, bdc_i);
        _mm512_store_pd(&y0_re[k], adc_r);
        _mm512_store_pd(&y0_im[k], adc_i);
        _mm512_store_pd(&y0_re[k + R7_512_VW], bdc_r);
        _mm512_store_pd(&y0_im[k + R7_512_VW], bdc_i);

        __m512d as0r = at1r, as0i = at1i, as1r = at3r, as1i = at3i;
        __m512d as2r = at2r, as2i = at2i, as3r = at6r, as3i = at6i;
        __m512d as4r = at4r, as4i = at4i, as5r = at5r, as5i = at5i;
        __m512d bs0r = bt1r, bs0i = bt1i, bs1r = bt3r, bs1i = bt3i;
        __m512d bs2r = bt2r, bs2i = bt2i, bs3r = bt6r, bs3i = bt6i;
        __m512d bs4r = bt4r, bs4i = bt4i, bs5r = bt5r, bs5i = bt5i;

        DFT6_FWD_512(as0r, as0i, as1r, as1i, as2r, as2i, as3r, as3i, as4r, as4i, as5r, as5i);
        DFT6_FWD_512(bs0r, bs0i, bs1r, bs1i, bs2r, bs2i, bs3r, bs3i, bs4r, bs4i, bs5r, bs5i);
        PW_RR6_U2(as0r, as0i, as1r, as1i, as2r, as2i, as3r, as3i, as4r, as4i, as5r, as5i,
                  bs0r, bs0i, bs1r, bs1i, bs2r, bs2i, bs3r, bs3i, bs4r, bs4i, bs5r, bs5i,
                  RK7_BR0, RK7_BI0, RK7_BR1, RK7_BI1, RK7_BR2, RK7_BI2,
                  RK7_BR3, RK7_BI3, RK7_BR4, RK7_BI4, RK7_BR5, RK7_BI5);
        DFT6_BWD_512(as0r, as0i, as1r, as1i, as2r, as2i, as3r, as3i, as4r, as4i, as5r, as5i);
        DFT6_BWD_512(bs0r, bs0i, bs1r, bs1i, bs2r, bs2i, bs3r, bs3i, bs4r, bs4i, bs5r, bs5i);

        ax0r = _mm512_load_pd(&a_re[k]);
        ax0i = _mm512_load_pd(&a_im[k]);
        bx0r = _mm512_load_pd(&a_re[k + R7_512_VW]);
        bx0i = _mm512_load_pd(&a_im[k + R7_512_VW]);

        _mm512_store_pd(&y1_re[k], _mm512_add_pd(ax0r, as0r));
        _mm512_store_pd(&y1_im[k], _mm512_add_pd(ax0i, as0i));
        _mm512_store_pd(&y5_re[k], _mm512_add_pd(ax0r, as1r));
        _mm512_store_pd(&y5_im[k], _mm512_add_pd(ax0i, as1i));
        _mm512_store_pd(&y4_re[k], _mm512_add_pd(ax0r, as2r));
        _mm512_store_pd(&y4_im[k], _mm512_add_pd(ax0i, as2i));
        _mm512_store_pd(&y6_re[k], _mm512_add_pd(ax0r, as3r));
        _mm512_store_pd(&y6_im[k], _mm512_add_pd(ax0i, as3i));
        _mm512_store_pd(&y2_re[k], _mm512_add_pd(ax0r, as4r));
        _mm512_store_pd(&y2_im[k], _mm512_add_pd(ax0i, as4i));
        _mm512_store_pd(&y3_re[k], _mm512_add_pd(ax0r, as5r));
        _mm512_store_pd(&y3_im[k], _mm512_add_pd(ax0i, as5i));
        _mm512_store_pd(&y1_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs0r));
        _mm512_store_pd(&y1_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs0i));
        _mm512_store_pd(&y5_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs1r));
        _mm512_store_pd(&y5_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs1i));
        _mm512_store_pd(&y4_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs2r));
        _mm512_store_pd(&y4_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs2i));
        _mm512_store_pd(&y6_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs3r));
        _mm512_store_pd(&y6_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs3i));
        _mm512_store_pd(&y2_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs4r));
        _mm512_store_pd(&y2_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs4i));
        _mm512_store_pd(&y3_re[k + R7_512_VW], _mm512_add_pd(bx0r, bs5r));
        _mm512_store_pd(&y3_im[k + R7_512_VW], _mm512_add_pd(bx0i, bs5i));
    }

    /* U=1 remainder */
    for (; k + R7_512_VW <= K; k += R7_512_VW)
    {
        __m512d x0r = _mm512_load_pd(&a_re[k]), x0i = _mm512_load_pd(&a_im[k]);
        __m512d t1r = _mm512_load_pd(&b_re[k]), t1i = _mm512_load_pd(&b_im[k]);
        __m512d t2r = _mm512_load_pd(&c_re[k]), t2i = _mm512_load_pd(&c_im[k]);
        __m512d t3r = _mm512_load_pd(&d_re[k]), t3i = _mm512_load_pd(&d_im[k]);
        __m512d t4r = _mm512_load_pd(&e_re[k]), t4i = _mm512_load_pd(&e_im[k]);
        __m512d t5r = _mm512_load_pd(&f_re[k]), t5i = _mm512_load_pd(&f_im[k]);
        __m512d t6r = _mm512_load_pd(&g_re[k]), t6i = _mm512_load_pd(&g_im[k]);
        __m512d dcr, dci;
        TREE_Y0_512(x0r, t1r, t2r, t3r, t4r, t5r, t6r, dcr);
        TREE_Y0_512(x0i, t1i, t2i, t3i, t4i, t5i, t6i, dci);
        _mm512_store_pd(&y0_re[k], dcr);
        _mm512_store_pd(&y0_im[k], dci);
        __m512d s0r = t1r, s0i = t1i, s1r = t3r, s1i = t3i;
        __m512d s2r = t2r, s2i = t2i, s3r = t6r, s3i = t6i;
        __m512d s4r = t4r, s4i = t4i, s5r = t5r, s5i = t5i;
        DFT6_FWD_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i);
        PW_RR6_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i,
                   RK7_BR0, RK7_BI0, RK7_BR1, RK7_BI1, RK7_BR2, RK7_BI2,
                   RK7_BR3, RK7_BI3, RK7_BR4, RK7_BI4, RK7_BR5, RK7_BI5);
        DFT6_BWD_512(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i);
        x0r = _mm512_load_pd(&a_re[k]);
        x0i = _mm512_load_pd(&a_im[k]);
        _mm512_store_pd(&y1_re[k], _mm512_add_pd(x0r, s0r));
        _mm512_store_pd(&y1_im[k], _mm512_add_pd(x0i, s0i));
        _mm512_store_pd(&y5_re[k], _mm512_add_pd(x0r, s1r));
        _mm512_store_pd(&y5_im[k], _mm512_add_pd(x0i, s1i));
        _mm512_store_pd(&y4_re[k], _mm512_add_pd(x0r, s2r));
        _mm512_store_pd(&y4_im[k], _mm512_add_pd(x0i, s2i));
        _mm512_store_pd(&y6_re[k], _mm512_add_pd(x0r, s3r));
        _mm512_store_pd(&y6_im[k], _mm512_add_pd(x0i, s3i));
        _mm512_store_pd(&y2_re[k], _mm512_add_pd(x0r, s4r));
        _mm512_store_pd(&y2_im[k], _mm512_add_pd(x0i, s4i));
        _mm512_store_pd(&y3_re[k], _mm512_add_pd(x0r, s5r));
        _mm512_store_pd(&y3_im[k], _mm512_add_pd(x0i, s5i));
    }

    /* Scalar tail */
    for (; k < K; k++)
    {
        double x0r_ = a_re[k], x0i_ = a_im[k];
        double _t1r = b_re[k], _t1i = b_im[k], _t2r = c_re[k], _t2i = c_im[k];
        double _t3r = d_re[k], _t3i = d_im[k], _t4r = e_re[k], _t4i = e_im[k];
        double _t5r = f_re[k], _t5i = f_im[k], _t6r = g_re[k], _t6i = g_im[k];
        y0_re[k] = x0r_ + _t1r + _t2r + _t3r + _t4r + _t5r + _t6r;
        y0_im[k] = x0i_ + _t1i + _t2i + _t3i + _t4i + _t5i + _t6i;
        double ar[6] = {_t1r, _t3r, _t2r, _t6r, _t4r, _t5r};
        double ai[6] = {_t1i, _t3i, _t2i, _t6i, _t4i, _t5i};
        double Ar[6], Ai[6], Cr[6], Ci[6], cr[6], ci[6];
        {
            double s1 = ar[2] + ar[4], s2 = ai[2] + ai[4], d1 = ar[2] - ar[4], d2 = ai[2] - ai[4];
            double E0r = ar[0] + s1, E0i = ai[0] + s2, br = ar[0] - 0.5 * s1, bi = ai[0] - 0.5 * s2;
            double E1r = br + 0.86602540378443864676 * d2, E1i = bi - 0.86602540378443864676 * d1;
            double E2r = br - 0.86602540378443864676 * d2, E2i = bi + 0.86602540378443864676 * d1;
            double s3 = ar[3] + ar[5], s4 = ai[3] + ai[5], d3 = ar[3] - ar[5], d4 = ai[3] - ai[5];
            double O0r = ar[1] + s3, O0i = ai[1] + s4, br2 = ar[1] - 0.5 * s3, bi2 = ai[1] - 0.5 * s4;
            double O1r = br2 + 0.86602540378443864676 * d4, O1i = bi2 - 0.86602540378443864676 * d3;
            double O2r = br2 - 0.86602540378443864676 * d4, O2i = bi2 + 0.86602540378443864676 * d3;
            Ar[0] = E0r + O0r;
            Ai[0] = E0i + O0i;
            Ar[3] = E0r - O0r;
            Ai[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 + O1i * 0.86602540378443864676;
            double T1i_ = O1r * (-0.86602540378443864676) + O1i * 0.5;
            Ar[1] = E1r + T1r_;
            Ai[1] = E1i + T1i_;
            Ar[4] = E1r - T1r_;
            Ai[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) + O2i * 0.86602540378443864676;
            double T2i_ = O2r * (-0.86602540378443864676) + O2i * (-0.5);
            Ar[2] = E2r + T2r_;
            Ai[2] = E2i + T2i_;
            Ar[5] = E2r - T2r_;
            Ai[5] = E2i - T2i_;
        }
        for (int j = 0; j < 6; j++)
        {
            Cr[j] = Ar[j] * RK7S_BWD_RE[j] - Ai[j] * RK7S_BWD_IM[j];
            Ci[j] = Ar[j] * RK7S_BWD_IM[j] + Ai[j] * RK7S_BWD_RE[j];
        }
        {
            double s1 = Cr[2] + Cr[4], s2 = Ci[2] + Ci[4], d1 = Cr[2] - Cr[4], d2 = Ci[2] - Ci[4];
            double E0r = Cr[0] + s1, E0i = Ci[0] + s2, br = Cr[0] - 0.5 * s1, bi = Ci[0] - 0.5 * s2;
            double E1r = br - 0.86602540378443864676 * d2, E1i = bi + 0.86602540378443864676 * d1;
            double E2r = br + 0.86602540378443864676 * d2, E2i = bi - 0.86602540378443864676 * d1;
            double s3 = Cr[3] + Cr[5], s4 = Ci[3] + Ci[5], d3 = Cr[3] - Cr[5], d4 = Ci[3] - Ci[5];
            double O0r = Cr[1] + s3, O0i = Ci[1] + s4, br2 = Cr[1] - 0.5 * s3, bi2 = Ci[1] - 0.5 * s4;
            double O1r = br2 - 0.86602540378443864676 * d4, O1i = bi2 + 0.86602540378443864676 * d3;
            double O2r = br2 + 0.86602540378443864676 * d4, O2i = bi2 - 0.86602540378443864676 * d3;
            cr[0] = E0r + O0r;
            ci[0] = E0i + O0i;
            cr[3] = E0r - O0r;
            ci[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 - O1i * 0.86602540378443864676;
            double T1i_ = O1r * 0.86602540378443864676 + O1i * 0.5;
            cr[1] = E1r + T1r_;
            ci[1] = E1i + T1i_;
            cr[4] = E1r - T1r_;
            ci[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) - O2i * 0.86602540378443864676;
            double T2i_ = O2r * 0.86602540378443864676 + O2i * (-0.5);
            cr[2] = E2r + T2r_;
            ci[2] = E2i + T2i_;
            cr[5] = E2r - T2r_;
            ci[5] = E2i - T2i_;
        }
        y1_re[k] = x0r_ + cr[0];
        y1_im[k] = x0i_ + ci[0];
        y5_re[k] = x0r_ + cr[1];
        y5_im[k] = x0i_ + ci[1];
        y4_re[k] = x0r_ + cr[2];
        y4_im[k] = x0i_ + ci[2];
        y6_re[k] = x0r_ + cr[3];
        y6_im[k] = x0i_ + ci[3];
        y2_re[k] = x0r_ + cr[4];
        y2_im[k] = x0i_ + ci[4];
        y3_re[k] = x0r_ + cr[5];
        y3_im[k] = x0i_ + ci[5];
    }
}

/* ================================================================== */
/*  Prefetch helpers                                                    */
/*                                                                     */
/*  T0 (L1): input arrays — will be consumed in current/next iteration */
/*  T1 (L2): twiddle arrays — temporal reuse across stages             */
/*  Distance: 4 cache lines ahead = 4 × 64B = 32 doubles = 2 ZMMs    */
/* ================================================================== */

#define R7_PF_DIST (4 * 64 / (int)sizeof(double)) /* 32 doubles */

#define R7_PREFETCH_INPUTS_T0(k_pf)                           \
    do                                                        \
    {                                                         \
        _mm_prefetch((const char *)&a_re[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&a_im[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&b_re[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&b_im[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&c_re[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&c_im[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&d_re[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&d_im[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&e_re[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&e_im[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&f_re[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&f_im[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&g_re[k_pf], _MM_HINT_T0); \
        _mm_prefetch((const char *)&g_im[k_pf], _MM_HINT_T0); \
    } while (0)

#define R7_PREFETCH_TWIDDLES_T1(k_pf)                           \
    do                                                          \
    {                                                           \
        _mm_prefetch((const char *)&tw1_re[k_pf], _MM_HINT_T1); \
        _mm_prefetch((const char *)&tw1_im[k_pf], _MM_HINT_T1); \
        _mm_prefetch((const char *)&tw2_re[k_pf], _MM_HINT_T1); \
        _mm_prefetch((const char *)&tw2_im[k_pf], _MM_HINT_T1); \
        _mm_prefetch((const char *)&tw3_re[k_pf], _MM_HINT_T1); \
        _mm_prefetch((const char *)&tw3_im[k_pf], _MM_HINT_T1); \
    } while (0)

/* ================================================================== */
/*  NT store macro — replaces _mm512_store_pd in LLC-bypass path       */
/* ================================================================== */

#define R7_NT_STORE_512(addr, val) _mm512_stream_pd((addr), (val))

/* ================================================================== */
/*  LLC-aware dispatch heuristic                                        */
/*                                                                     */
/*  Working set per radix-7 stage: 7 legs × 2 (re/im) × K × 8B       */
/*    INPUT  = 7 × 2 × K × 8 = 112K bytes                             */
/*    OUTPUT = same = 112K bytes                                        */
/*    TWIDDLE= 3 × 2 × K × 8 = 48K bytes  (BLOCKED3)                  */
/*    TOTAL  = 272K bytes per pass                                      */
/*                                                                     */
/*  Rule: if TOTAL > LLC_SIZE / 2, use NT stores (out-of-place only).  */
/*  Caller should set R7_LLC_SIZE_BYTES to actual LLC per core.        */
/*  Default: 2 MiB (conservative for shared LLCs).                     */
/* ================================================================== */

#ifndef R7_LLC_SIZE_BYTES
#define R7_LLC_SIZE_BYTES (2 * 1024 * 1024)
#endif

#define R7_NT_THRESHOLD_K (R7_LLC_SIZE_BYTES / (2 * 272))

/* Caller macro — dispatches to prefetch+NT or standard variant:
 *
 *   R7_FWD_DISPATCH(a_re,..., tw3_im, K)
 *
 * For K > threshold, inserts prefetch into the U=2 loop preamble and
 * uses _mm512_stream_pd.  For K <= threshold, uses the standard
 * butterfly (data stays in LLC).
 *
 * NOTE: The full NT-store variants are best built by the multi-stage
 * driver that wraps each butterfly call.  The macros above provide
 * the primitives; the driver injects them at the right points.
 */

#define R7_USE_NT_STORES(K) ((int)(K) > R7_NT_THRESHOLD_K)

#endif /* __AVX512F__ */
#endif /* FFT_RADIX7_AVX512_H */