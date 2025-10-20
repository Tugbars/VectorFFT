```c
//==============================================================================
// fft_radix11_macros.h - Shared Macros for Radix-11 Butterflies
//==============================================================================
//
// ALGORITHM: Radix-11 DFT with 5 symmetric pairs (NOT Rader's)
//   Prime 11 uses direct geometric decomposition with symmetry
//   - Form 5 pairs: (b,k), (c,j), (d,i), (e,h), (f,g)
//   - Cosine coefficients: C11_1..C11_5
//   - Sine coefficients: S11_1..S11_5
//   - Exploits conjugate symmetry: Y_m and Y_{11-m}
//
// USAGE:
//   #include "fft_radix11_macros.h" in both fft_radix11_fv.c and fft_radix11_bv.c
//

#ifndef FFT_RADIX11_MACROS_H
#define FFT_RADIX11_MACROS_H

#include "simd_math.h"

//==============================================================================
// GEOMETRIC CONSTANTS - IDENTICAL for forward/inverse
//==============================================================================

#define C11_1 0.8412535328311812   // cos(2π/11)
#define C11_2 0.4154150130018864   // cos(4π/11)
#define C11_3 -0.14231483827328514 // cos(6π/11)
#define C11_4 -0.6548607339452850  // cos(8π/11)
#define C11_5 -0.9594929736144974  // cos(10π/11)

#define S11_1 0.5406408174555976  // sin(2π/11)
#define S11_2 0.9096319953545184  // sin(4π/11)
#define S11_3 0.9898214418809327  // sin(6π/11)
#define S11_4 0.7557495743542583  // sin(8π/11)
#define S11_5 0.28173255684142967 // sin(10π/11)

//==============================================================================
// COMPLEX MULTIPLICATION - FMA-optimized (IDENTICAL)
//==============================================================================

#ifdef __AVX2__
#define CMUL_FMA_R11(out, a, w)                                      \
    do                                                               \
    {                                                                \
        __m256d ar = _mm256_unpacklo_pd(a, a);                       \
        __m256d ai = _mm256_unpackhi_pd(a, a);                       \
        __m256d wr = _mm256_unpacklo_pd(w, w);                       \
        __m256d wi = _mm256_unpackhi_pd(w, w);                       \
        __m256d re = _mm256_fmsub_pd(ar, wr, _mm256_mul_pd(ai, wi)); \
        __m256d im = _mm256_fmadd_pd(ar, wi, _mm256_mul_pd(ai, wr)); \
        (out) = _mm256_unpacklo_pd(re, im);                          \
    } while (0)
#endif

//==============================================================================
// ROTATION HELPERS - Direction-specific
//==============================================================================

#ifdef __AVX2__
// Apply -i rotation: (a+bi)*(-i) = b - ai
#define ROT_NEG_I_AVX2(v, out)                                     \
    do                                                             \
    {                                                              \
        __m256d _t = _mm256_permute_pd((v), 0b0101);               \
        const __m256d _mask = _mm256_set_pd(0.0, -0.0, 0.0, -0.0); \
        (out) = _mm256_xor_pd(_t, _mask);                          \
    } while (0)

// Apply +i rotation: (a+bi)*(+i) = -b + ai
#define ROT_POS_I_AVX2(v, out)                                     \
    do                                                             \
    {                                                              \
        __m256d _t = _mm256_permute_pd((v), 0b0101);               \
        const __m256d _mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); \
        (out) = _mm256_xor_pd(_t, _mask);                          \
    } while (0)
#endif

//==============================================================================
// RADIX-11 BUTTERFLY CORE - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Compute symmetric pairs and Y0 (DC component)
 *
 * Forms 5 symmetric pairs:
 * - Pair 0: (b, k) → t0 = b+k, s0 = b-k
 * - Pair 1: (c, j) → t1 = c+j, s1 = c-j
 * - Pair 2: (d, i) → t2 = d+i, s2 = d-i
 * - Pair 3: (e, h) → t3 = e+h, s3 = e-h
 * - Pair 4: (f, g) → t4 = f+g, s4 = f-g
 *
 * Y0 = a + t0 + t1 + t2 + t3 + t4
 */
#ifdef __AVX2__
#define RADIX11_BUTTERFLY_CORE_AVX2(a, b, c, d, e, f, g, h, i, j, xk,            \
                                    t0, t1, t2, t3, t4,                          \
                                    s0, s1, s2, s3, s4, y0)                      \
    do                                                                           \
    {                                                                            \
        t0 = _mm256_add_pd(b, xk); /* b + k */                                   \
        t1 = _mm256_add_pd(c, j);  /* c + j */                                   \
        t2 = _mm256_add_pd(d, i);  /* d + i */                                   \
        t3 = _mm256_add_pd(e, h);  /* e + h */                                   \
        t4 = _mm256_add_pd(f, g);  /* f + g */                                   \
                                                                                 \
        s0 = _mm256_sub_pd(b, xk); /* b - k */                                   \
        s1 = _mm256_sub_pd(c, j);  /* c - j */                                   \
        s2 = _mm256_sub_pd(d, i);  /* d - i */                                   \
        s3 = _mm256_sub_pd(e, h);  /* e - h */                                   \
        s4 = _mm256_sub_pd(f, g);  /* f - g */                                   \
                                                                                 \
        __m256d sum_t = _mm256_add_pd(_mm256_add_pd(t0, t1),                     \
                                      _mm256_add_pd(_mm256_add_pd(t2, t3), t4)); \
        y0 = _mm256_add_pd(a, sum_t);                                            \
    } while (0)
#endif

//==============================================================================
// COMPUTE REAL PARTS - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Compute real part for symmetric pair m
 *
 * Real_m = a + C11_i0*t0 + C11_i1*t1 + C11_i2*t2 + C11_i3*t3 + C11_i4*t4
 *
 * Coefficients cycle through different permutations for m=1..5
 */
#ifdef __AVX2__
// Pair 1: Y_1, Y_10 (coefficients: c1, c2, c3, c4, c5)
#define RADIX11_REAL_PAIR1_AVX2(a, t0, t1, t2, t3, t4, real_out)                                                                                            \
    do                                                                                                                                                      \
    {                                                                                                                                                       \
        const __m256d vc1 = _mm256_set1_pd(C11_1);                                                                                                          \
        const __m256d vc2 = _mm256_set1_pd(C11_2);                                                                                                          \
        const __m256d vc3 = _mm256_set1_pd(C11_3);                                                                                                          \
        const __m256d vc4 = _mm256_set1_pd(C11_4);                                                                                                          \
        const __m256d vc5 = _mm256_set1_pd(C11_5);                                                                                                          \
        real_out = _mm256_add_pd(a, _mm256_fmadd_pd(vc1, t0,                                                                                                \
                                                    _mm256_fmadd_pd(vc2, t1, _mm256_fmadd_pd(vc3, t2, _mm256_fmadd_pd(vc4, t3, _mm256_mul_pd(vc5, t4)))))); \
    } while (0)

// Pair 2: Y_2, Y_9 (coefficients: c2, c4, c5, c3, c1)
#define RADIX11_REAL_PAIR2_AVX2(a, t0, t1, t2, t3, t4, real_out)                                                                                            \
    do                                                                                                                                                      \
    {                                                                                                                                                       \
        const __m256d vc2 = _mm256_set1_pd(C11_2);                                                                                                          \
        const __m256d vc4 = _mm256_set1_pd(C11_4);                                                                                                          \
        const __m256d vc5 = _mm256_set1_pd(C11_5);                                                                                                          \
        const __m256d vc3 = _mm256_set1_pd(C11_3);                                                                                                          \
        const __m256d vc1 = _mm256_set1_pd(C11_1);                                                                                                          \
        real_out = _mm256_add_pd(a, _mm256_fmadd_pd(vc2, t0,                                                                                                \
                                                    _mm256_fmadd_pd(vc4, t1, _mm256_fmadd_pd(vc5, t2, _mm256_fmadd_pd(vc3, t3, _mm256_mul_pd(vc1, t4)))))); \
    } while (0)

// Pair 3: Y_3, Y_8 (coefficients: c3, c5, c2, c1, c4)
#define RADIX11_REAL_PAIR3_AVX2(a, t0, t1, t2, t3, t4, real_out)                                                                                            \
    do                                                                                                                                                      \
    {                                                                                                                                                       \
        const __m256d vc3 = _mm256_set1_pd(C11_3);                                                                                                          \
        const __m256d vc5 = _mm256_set1_pd(C11_5);                                                                                                          \
        const __m256d vc2 = _mm256_set1_pd(C11_2);                                                                                                          \
        const __m256d vc1 = _mm256_set1_pd(C11_1);                                                                                                          \
        const __m256d vc4 = _mm256_set1_pd(C11_4);                                                                                                          \
        real_out = _mm256_add_pd(a, _mm256_fmadd_pd(vc3, t0,                                                                                                \
                                                    _mm256_fmadd_pd(vc5, t1, _mm256_fmadd_pd(vc2, t2, _mm256_fmadd_pd(vc1, t3, _mm256_mul_pd(vc4, t4)))))); \
    } while (0)

// Pair 4: Y_4, Y_7 (coefficients: c4, c3, c1, c5, c2)
#define RADIX11_REAL_PAIR4_AVX2(a, t0, t1, t2, t3, t4, real_out)                                                                                            \
    do                                                                                                                                                      \
    {                                                                                                                                                       \
        const __m256d vc4 = _mm256_set1_pd(C11_4);                                                                                                          \
        const __m256d vc3 = _mm256_set1_pd(C11_3);                                                                                                          \
        const __m256d vc1 = _mm256_set1_pd(C11_1);                                                                                                          \
        const __m256d vc5 = _mm256_set1_pd(C11_5);                                                                                                          \
        const __m256d vc2 = _mm256_set1_pd(C11_2);                                                                                                          \
        real_out = _mm256_add_pd(a, _mm256_fmadd_pd(vc4, t0,                                                                                                \
                                                    _mm256_fmadd_pd(vc3, t1, _mm256_fmadd_pd(vc1, t2, _mm256_fmadd_pd(vc5, t3, _mm256_mul_pd(vc2, t4)))))); \
    } while (0)

// Pair 5: Y_5, Y_6 (coefficients: c5, c1, c4, c2, c3)
#define RADIX11_REAL_PAIR5_AVX2(a, t0, t1, t2, t3, t4, real_out)                                                                                            \
    do                                                                                                                                                      \
    {                                                                                                                                                       \
        const __m256d vc5 = _mm256_set1_pd(C11_5);                                                                                                          \
        const __m256d vc1 = _mm256_set1_pd(C11_1);                                                                                                          \
        const __m256d vc4 = _mm256_set1_pd(C11_4);                                                                                                          \
        const __m256d vc2 = _mm256_set1_pd(C11_2);                                                                                                          \
        const __m256d vc3 = _mm256_set1_pd(C11_3);                                                                                                          \
        real_out = _mm256_add_pd(a, _mm256_fmadd_pd(vc5, t0,                                                                                                \
                                                    _mm256_fmadd_pd(vc1, t1, _mm256_fmadd_pd(vc4, t2, _mm256_fmadd_pd(vc2, t3, _mm256_mul_pd(vc3, t4)))))); \
    } while (0)
#endif

//==============================================================================
// COMPUTE IMAGINARY ROTATION - DIRECTION-SPECIFIC
//==============================================================================

/**
 * @brief Compute imaginary rotation base for pair m
 *
 * Base_m = S11_i0*s0 + S11_i1*s1 + S11_i2*s2 + S11_i3*s3 + S11_i4*s4
 * Then apply ±i rotation based on direction
 */
#ifdef __AVX2__
// Pair 1 - FORWARD (applies -i rotation)
#define RADIX11_IMAG_PAIR1_FV_AVX2(s0, s1, s2, s3, s4, rot_out)                                                                                        \
    do                                                                                                                                                 \
    {                                                                                                                                                  \
        const __m256d vs1 = _mm256_set1_pd(S11_1);                                                                                                     \
        const __m256d vs2 = _mm256_set1_pd(S11_2);                                                                                                     \
        const __m256d vs3 = _mm256_set1_pd(S11_3);                                                                                                     \
        const __m256d vs4 = _mm256_set1_pd(S11_4);                                                                                                     \
        const __m256d vs5 = _mm256_set1_pd(S11_5);                                                                                                     \
        __m256d base = _mm256_fmadd_pd(vs1, s0, _mm256_fmadd_pd(vs2, s1, _mm256_fmadd_pd(vs3, s2, _mm256_fmadd_pd(vs4, s3, _mm256_mul_pd(vs5, s4))))); \
        ROT_NEG_I_AVX2(base, rot_out);                                                                                                                 \
    } while (0)

// Pair 1 - INVERSE (applies +i rotation)
#define RADIX11_IMAG_PAIR1_BV_AVX2(s0, s1, s2, s3, s4, rot_out)                                                                                        \
    do                                                                                                                                                 \
    {                                                                                                                                                  \
        const __m256d vs1 = _mm256_set1_pd(S11_1);                                                                                                     \
        const __m256d vs2 = _mm256_set1_pd(S11_2);                                                                                                     \
        const __m256d vs3 = _mm256_set1_pd(S11_3);                                                                                                     \
        const __m256d vs4 = _mm256_set1_pd(S11_4);                                                                                                     \
        const __m256d vs5 = _mm256_set1_pd(S11_5);                                                                                                     \
        __m256d base = _mm256_fmadd_pd(vs1, s0, _mm256_fmadd_pd(vs2, s1, _mm256_fmadd_pd(vs3, s2, _mm256_fmadd_pd(vs4, s3, _mm256_mul_pd(vs5, s4))))); \
        ROT_POS_I_AVX2(base, rot_out);                                                                                                                 \
    } while (0)

// Pair 2 - FORWARD (sines: s2, s4, s5, s3, s1)
#define RADIX11_IMAG_PAIR2_FV_AVX2(s0, s1, s2, s3, s4, rot_out)                                                                                        \
    do                                                                                                                                                 \
    {                                                                                                                                                  \
        const __m256d vs2 = _mm256_set1_pd(S11_2);                                                                                                     \
        const __m256d vs4 = _mm256_set1_pd(S11_4);                                                                                                     \
        const __m256d vs5 = _mm256_set1_pd(S11_5);                                                                                                     \
        const __m256d vs3 = _mm256_set1_pd(S11_3);                                                                                                     \
        const __m256d vs1 = _mm256_set1_pd(S11_1);                                                                                                     \
        __m256d base = _mm256_fmadd_pd(vs2, s0, _mm256_fmadd_pd(vs4, s1, _mm256_fmadd_pd(vs5, s2, _mm256_fmadd_pd(vs3, s3, _mm256_mul_pd(vs1, s4))))); \
        ROT_NEG_I_AVX2(base, rot_out);                                                                                                                 \
    } while (0)

// Pair 2 - INVERSE
#define RADIX11_IMAG_PAIR2_BV_AVX2(s0, s1, s2, s3, s4, rot_out)                                                                                        \
    do                                                                                                                                                 \
    {                                                                                                                                                  \
        const __m256d vs2 = _mm256_set1_pd(S11_2);                                                                                                     \
        const __m256d vs4 = _mm256_set1_pd(S11_4);                                                                                                     \
        const __m256d vs5 = _mm256_set1_pd(S11_5);                                                                                                     \
        const __m256d vs3 = _mm256_set1_pd(S11_3);                                                                                                     \
        const __m256d vs1 = _mm256_set1_pd(S11_1);                                                                                                     \
        __m256d base = _mm256_fmadd_pd(vs2, s0, _mm256_fmadd_pd(vs4, s1, _mm256_fmadd_pd(vs5, s2, _mm256_fmadd_pd(vs3, s3, _mm256_mul_pd(vs1, s4))))); \
        ROT_POS_I_AVX2(base, rot_out);                                                                                                                 \
    } while (0)

// Pair 3 - FORWARD (sines: s3, s5, s2, s1, s4)
#define RADIX11_IMAG_PAIR3_FV_AVX2(s0, s1, s2, s3, s4, rot_out)                                                                                        \
    do                                                                                                                                                 \
    {                                                                                                                                                  \
        const __m256d vs3 = _mm256_set1_pd(S11_3);                                                                                                     \
        const __m256d vs5 = _mm256_set1_pd(S11_5);                                                                                                     \
        const __m256d vs2 = _mm256_set1_pd(S11_2);                                                                                                     \
        const __m256d vs1 = _mm256_set1_pd(S11_1);                                                                                                     \
        const __m256d vs4 = _mm256_set1_pd(S11_4);                                                                                                     \
        __m256d base = _mm256_fmadd_pd(vs3, s0, _mm256_fmadd_pd(vs5, s1, _mm256_fmadd_pd(vs2, s2, _mm256_fmadd_pd(vs1, s3, _mm256_mul_pd(vs4, s4))))); \
        ROT_NEG_I_AVX2(base, rot_out);                                                                                                                 \
    } while (0)

// Pair 3 - INVERSE
#define RADIX11_IMAG_PAIR3_BV_AVX2(s0, s1, s2, s3, s4, rot_out)                                                                                        \
    do                                                                                                                                                 \
    {                                                                                                                                                  \
        const __m256d vs3 = _mm256_set1_pd(S11_3);                                                                                                     \
        const __m256d vs5 = _mm256_set1_pd(S11_5);                                                                                                     \
        const __m256d vs2 = _mm256_set1_pd(S11_2);                                                                                                     \
        const __m256d vs1 = _mm256_set1_pd(S11_1);                                                                                                     \
        const __m256d vs4 = _mm256_set1_pd(S11_4);                                                                                                     \
        __m256d base = _mm256_fmadd_pd(vs3, s0, _mm256_fmadd_pd(vs5, s1, _mm256_fmadd_pd(vs2, s2, _mm256_fmadd_pd(vs1, s3, _mm256_mul_pd(vs4, s4))))); \
        ROT_POS_I_AVX2(base, rot_out);                                                                                                                 \
    } while (0)

// Pair 4 - FORWARD (sines: s4, s3, s1, s5, s2)
#define RADIX11_IMAG_PAIR4_FV_AVX2(s0, s1, s2, s3, s4, rot_out)                                                                                        \
    do                                                                                                                                                 \
    {                                                                                                                                                  \
        const __m256d vs4 = _mm256_set1_pd(S11_4);                                                                                                     \
        const __m256d vs3 = _mm256_set1_pd(S11_3);                                                                                                     \
        const __m256d vs1 = _mm256_set1_pd(S11_1);                                                                                                     \
        const __m256d vs5 = _mm256_set1_pd(S11_5);                                                                                                     \
        const __m256d vs2 = _mm256_set1_pd(S11_2);                                                                                                     \
        __m256d base = _mm256_fmadd_pd(vs4, s0, _mm256_fmadd_pd(vs3, s1, _mm256_fmadd_pd(vs1, s2, _mm256_fmadd_pd(vs5, s3, _mm256_mul_pd(vs2, s4))))); \
        ROT_NEG_I_AVX2(base, rot_out);                                                                                                                 \
    } while (0)

// Pair 4 - INVERSE
#define RADIX11_IMAG_PAIR4_BV_AVX2(s0, s1, s2, s3, s4, rot_out)                                                                                        \
    do                                                                                                                                                 \
    {                                                                                                                                                  \
        const __m256d vs4 = _mm256_set1_pd(S11_4);                                                                                                     \
        const __m256d vs3 = _mm256_set1_pd(S11_3);                                                                                                     \
        const __m256d vs1 = _mm256_set1_pd(S11_1);                                                                                                     \
        const __m256d vs5 = _mm256_set1_pd(S11_5);                                                                                                     \
        const __m256d vs2 = _mm256_set1_pd(S11_2);                                                                                                     \
        __m256d base = _mm256_fmadd_pd(vs4, s0, _mm256_fmadd_pd(vs3, s1, _mm256_fmadd_pd(vs1, s2, _mm256_fmadd_pd(vs5, s3, _mm256_mul_pd(vs2, s4))))); \
        ROT_POS_I_AVX2(base, rot_out);                                                                                                                 \
    } while (0)

// Pair 5 - FORWARD (sines: s5, s1, s4, s2, s3)
#define RADIX11_IMAG_PAIR5_FV_AVX2(s0, s1, s2, s3, s4, rot_out)                                                                                        \
    do                                                                                                                                                 \
    {                                                                                                                                                  \
        const __m256d vs5 = _mm256_set1_pd(S11_5);                                                                                                     \
        const __m256d vs1 = _mm256_set1_pd(S11_1);                                                                                                     \
        const __m256d vs4 = _mm256_set1_pd(S11_4);                                                                                                     \
        const __m256d vs2 = _mm256_set1_pd(S11_2);                                                                                                     \
        const __m256d vs3 = _mm256_set1_pd(S11_3);                                                                                                     \
        __m256d base = _mm256_fmadd_pd(vs5, s0, _mm256_fmadd_pd(vs1, s1, _mm256_fmadd_pd(vs4, s2, _mm256_fmadd_pd(vs2, s3, _mm256_mul_pd(vs3, s4))))); \
        ROT_NEG_I_AVX2(base, rot_out);                                                                                                                 \
    } while (0)

// Pair 5 - INVERSE
#define RADIX11_IMAG_PAIR5_BV_AVX2(s0, s1, s2, s3, s4, rot_out)                                                                                        \
    do                                                                                                                                                 \
    {                                                                                                                                                  \
        const __m256d vs5 = _mm256_set1_pd(S11_5);                                                                                                     \
        const __m256d vs1 = _mm256_set1_pd(S11_1);                                                                                                     \
        const __m256d vs4 = _mm256_set1_pd(S11_4);                                                                                                     \
        const __m256d vs2 = _mm256_set1_pd(S11_2);                                                                                                     \
        const __m256d vs3 = _mm256_set1_pd(S11_3);                                                                                                     \
        __m256d base = _mm256_fmadd_pd(vs5, s0, _mm256_fmadd_pd(vs1, s1, _mm256_fmadd_pd(vs4, s2, _mm256_fmadd_pd(vs2, s3, _mm256_mul_pd(vs3, s4))))); \
        ROT_POS_I_AVX2(base, rot_out);                                                                                                                 \
    } while (0)
#endif

//==============================================================================
// OUTPUT ASSEMBLY - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Assemble conjugate pair outputs
 *
 * Y_m      = real + rot
 * Y_{11-m} = real - rot
 */
#ifdef __AVX2__
#define RADIX11_ASSEMBLE_PAIR_AVX2(real, rot, y_m, y_11m) \
    do                                                    \
    {                                                     \
        y_m = _mm256_add_pd(real, rot);                   \
        y_11m = _mm256_sub_pd(real, rot);                 \
    } while (0)
#endif

//==============================================================================
// APPLY PRECOMPUTED TWIDDLES - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief AVX2: Apply stage twiddles for 2 butterflies (kk and kk+1)
 *
 * stage_tw layout: [W^(1*k), ..., W^(10*k)] for each k
 */
#ifdef __AVX2__
#define APPLY_STAGE_TWIDDLES_R11_AVX2(kk, b, c, d, e, f, g, h, i, j, xk, stage_tw)           \
    do                                                                                       \
    {                                                                                        \
        if (sub_len > 1)                                                                     \
        {                                                                                    \
            __m256d w1 = load2_aos(&stage_tw[(kk) * 10 + 0], &stage_tw[(kk + 1) * 10 + 0]);  \
            __m256d w2 = load2_aos(&stage_tw[(kk) * 10 + 1], &stage_tw[(kk + 1) * 10 + 1]);  \
            __m256d w3 = load2_aos(&stage_tw[(kk) * 10 + 2], &stage_tw[(kk + 1) * 10 + 2]);  \
            __m256d w4 = load2_aos(&stage_tw[(kk) * 10 + 3], &stage_tw[(kk + 1) * 10 + 3]);  \
            __m256d w5 = load2_aos(&stage_tw[(kk) * 10 + 4], &stage_tw[(kk + 1) * 10 + 4]);  \
            __m256d w6 = load2_aos(&stage_tw[(kk) * 10 + 5], &stage_tw[(kk + 1) * 10 + 5]);  \
            __m256d w7 = load2_aos(&stage_tw[(kk) * 10 + 6], &stage_tw[(kk + 1) * 10 + 6]);  \
            __m256d w8 = load2_aos(&stage_tw[(kk) * 10 + 7], &stage_tw[(kk + 1) * 10 + 7]);  \
            __m256d w9 = load2_aos(&stage_tw[(kk) * 10 + 8], &stage_tw[(kk + 1) * 10 + 8]);  \
            __m256d w10 = load2_aos(&stage_tw[(kk) * 10 + 9], &stage_tw[(kk + 1) * 10 + 9]); \
                                                                                             \
            CMUL_FMA_R11(b, b, w1);                                                          \
            CMUL_FMA_R11(c, c, w2);                                                          \
            CMUL_FMA_R11(d, d, w3);                                                          \
            CMUL_FMA_R11(e, e, w4);                                                          \
            CMUL_FMA_R11(f, f, w5);                                                          \
            CMUL_FMA_R11(g, g, w6);                                                          \
            CMUL_FMA_R11(h, h, w7);                                                          \
            CMUL_FMA_R11(i, i, w8);                                                          \
            CMUL_FMA_R11(j, j, w9);                                                          \
            CMUL_FMA_R11(xk, xk, w10);                                                       \
        }                                                                                    \
    } while (0)
#endif

//==============================================================================
// DATA MOVEMENT - IDENTICAL for forward/inverse
//==============================================================================

#ifdef __AVX2__
#define LOAD_11_LANES_AVX2(kk, K, sub_outputs, a, b, c, d, e, f, g, h, i, j, xk)      \
    do                                                                                \
    {                                                                                 \
        a = load2_aos(&sub_outputs[kk], &sub_outputs[(kk) + 1]);                      \
        b = load2_aos(&sub_outputs[(kk) + K], &sub_outputs[(kk) + 1 + K]);            \
        c = load2_aos(&sub_outputs[(kk) + 2 * K], &sub_outputs[(kk) + 1 + 2 * K]);    \
        d = load2_aos(&sub_outputs[(kk) + 3 * K], &sub_outputs[(kk) + 1 + 3 * K]);    \
        e = load2_aos(&sub_outputs[(kk) + 4 * K], &sub_outputs[(kk) + 1 + 4 * K]);    \
        f = load2_aos(&sub_outputs[(kk) + 5 * K], &sub_outputs[(kk) + 1 + 5 * K]);    \
        g = load2_aos(&sub_outputs[(kk) + 6 * K], &sub_outputs[(kk) + 1 + 6 * K]);    \
        h = load2_aos(&sub_outputs[(kk) + 7 * K], &sub_outputs[(kk) + 1 + 7 * K]);    \
        i = load2_aos(&sub_outputs[(kk) + 8 * K], &sub_outputs[(kk) + 1 + 8 * K]);    \
        j = load2_aos(&sub_outputs[(kk) + 9 * K], &sub_outputs[(kk) + 1 + 9 * K]);    \
        xk = load2_aos(&sub_outputs[(kk) + 10 * K], &sub_outputs[(kk) + 1 + 10 * K]); \
    } while (0)

#define STORE_11_LANES_AVX2(kk, K, output, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) \
    do                                                                                  \
    {                                                                                   \
        STOREU_PD(&output[kk].re, y0);                                                  \
        STOREU_PD(&output[(kk) + K].re, y1);                                            \
        STOREU_PD(&output[(kk) + 2 * K].re, y2);                                        \
        STOREU_PD(&output[(kk) + 3 * K].re, y3);                                        \
        STOREU_PD(&output[(kk) + 4 * K].re, y4);                                        \
        STOREU_PD(&output[(kk) + 5 * K].re, y5);                                        \
        STOREU_PD(&output[(kk) + 6 * K].re, y6);                                        \
        STOREU_PD(&output[(kk) + 7 * K].re, y7);                                        \
        STOREU_PD(&output[(kk) + 8 * K].re, y8);                                        \
        STOREU_PD(&output[(kk) + 9 * K].re, y9);                                        \
        STOREU_PD(&output[(kk) + 10 * K].re, y10);                                      \
    } while (0)
#endif

//==============================================================================
// PREFETCHING
//==============================================================================

#define PREFETCH_L1_R11 8
#define PREFETCH_L2_R11 32
#define PREFETCH_L3_R11 64

#ifdef __AVX2__
#define PREFETCH_11_LANES_R11(k, K, distance, sub_outputs)                                           \
    do                                                                                               \
    {                                                                                                \
        if ((k) + (distance) < (K))                                                                  \
        {                                                                                            \
            /* Adaptive prefetch based on K size */                                                  \
            int hint_type = ((K) > 256) ? _MM_HINT_T1 : /* L2 for large K */                         \
                                ((K) > 64) ? _MM_HINT_T0                                             \
                                           :  /* L1 for medium K */                                  \
                                _MM_HINT_NTA; /* Non-temporal for tiny K */                          \
            for (int _lane = 0; _lane < 11; _lane++)                                                 \
            {                                                                                        \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + _lane * (K)], hint_type); \
            }                                                                                        \
        }                                                                                            \
    } while (0)
#endif

//==============================================================================
// COMPLETE SCALAR BUTTERFLIES
//==============================================================================

//==============================================================================
// HELPER: Compute one symmetric pair (scalar)
//==============================================================================

#define RADIX11_COMPUTE_PAIR_SCALAR(b2r, b2i, k2r, k2i, c2r, c2i, j2r, j2i,         \
                                    d2r, d2i, i2r, i2i, e2r, e2i, h2r, h2i,         \
                                    f2r, f2i, g2r, g2i, a_re, a_im,                 \
                                    C1, C2, C3, C4, C5,                             \
                                    S1, S2, S3, S4, S5,                             \
                                    is_forward, out_plus, out_minus)                \
    do                                                                              \
    {                                                                               \
        double t0r = b2r + k2r, t0i = b2i + k2i;                                    \
        double t1r = c2r + j2r, t1i = c2i + j2i;                                    \
        double t2r = d2r + i2r, t2i = d2i + i2i;                                    \
        double t3r = e2r + h2r, t3i = e2i + h2i;                                    \
        double t4r = f2r + g2r, t4i = f2i + g2i;                                    \
        double s0r = b2r - k2r, s0i = b2i - k2i;                                    \
        double s1r = c2r - j2r, s1i = c2i - j2i;                                    \
        double s2r = d2r - i2r, s2i = d2i - i2i;                                    \
        double s3r = e2r - h2r, s3i = e2i - h2i;                                    \
        double s4r = f2r - g2r, s4i = f2i - g2i;                                    \
        double realr = a_re + C1 * t0r + C2 * t1r + C3 * t2r + C4 * t3r + C5 * t4r; \
        double reali = a_im + C1 * t0i + C2 * t1i + C3 * t2i + C4 * t3i + C5 * t4i; \
        double baser = S1 * s0r + S2 * s1r + S3 * s2r + S4 * s3r + S5 * s4r;        \
        double basei = S1 * s0i + S2 * s1i + S3 * s2i + S4 * s3i + S5 * s4i;        \
        double rotr, roti;                                                          \
        if (is_forward)                                                             \
        {                                                                           \
            rotr = basei; /* -i: (re,im)*(-i) = (im, -re) */                        \
            roti = -baser;                                                          \
        }                                                                           \
        else                                                                        \
        {                                                                           \
            rotr = -basei; /* +i: (re,im)*(+i) = (-im, re) */                       \
            roti = baser;                                                           \
        }                                                                           \
        out_plus = (fft_data){realr + rotr, reali + roti};                          \
        out_minus = (fft_data){realr - rotr, reali - roti};                         \
    } while (0)

//==============================================================================
// COMPLETE SCALAR BUTTERFLIES (Refactored)
//==============================================================================

/**
 * @brief Scalar radix-11 butterfly (FORWARD)
 *
 * Applies -i rotation to imaginary components
 */
#define RADIX11_BUTTERFLY_SCALAR_FV(k, K, sub_outputs, stage_tw, output_buffer)            \
    do                                                                                     \
    {                                                                                      \
        /* Load 11 lanes */                                                                \
        const fft_data a = sub_outputs[k];                                                 \
        const fft_data b = sub_outputs[k + K];                                             \
        const fft_data c = sub_outputs[k + 2 * K];                                         \
        const fft_data d = sub_outputs[k + 3 * K];                                         \
        const fft_data e = sub_outputs[k + 4 * K];                                         \
        const fft_data f = sub_outputs[k + 5 * K];                                         \
        const fft_data g = sub_outputs[k + 6 * K];                                         \
        const fft_data h = sub_outputs[k + 7 * K];                                         \
        const fft_data i = sub_outputs[k + 8 * K];                                         \
        const fft_data j = sub_outputs[k + 9 * K];                                         \
        const fft_data xk = sub_outputs[k + 10 * K];                                       \
                                                                                           \
        /* Load twiddles W^(1*k) through W^(10*k) */                                       \
        const fft_data w1 = stage_tw[10 * k + 0];                                          \
        const fft_data w2 = stage_tw[10 * k + 1];                                          \
        const fft_data w3 = stage_tw[10 * k + 2];                                          \
        const fft_data w4 = stage_tw[10 * k + 3];                                          \
        const fft_data w5 = stage_tw[10 * k + 4];                                          \
        const fft_data w6 = stage_tw[10 * k + 5];                                          \
        const fft_data w7 = stage_tw[10 * k + 6];                                          \
        const fft_data w8 = stage_tw[10 * k + 7];                                          \
        const fft_data w9 = stage_tw[10 * k + 8];                                          \
        const fft_data w10 = stage_tw[10 * k + 9];                                         \
                                                                                           \
        /* Twiddle multiply */                                                             \
        double b2r = b.re * w1.re - b.im * w1.im;                                          \
        double b2i = b.re * w1.im + b.im * w1.re;                                          \
        double c2r = c.re * w2.re - c.im * w2.im;                                          \
        double c2i = c.re * w2.im + c.im * w2.re;                                          \
        double d2r = d.re * w3.re - d.im * w3.im;                                          \
        double d2i = d.re * w3.im + d.im * w3.re;                                          \
        double e2r = e.re * w4.re - e.im * w4.im;                                          \
        double e2i = e.re * w4.im + e.im * w4.re;                                          \
        double f2r = f.re * w5.re - f.im * w5.im;                                          \
        double f2i = f.re * w5.im + f.im * w5.re;                                          \
        double g2r = g.re * w6.re - g.im * w6.im;                                          \
        double g2i = g.re * w6.im + g.im * w6.re;                                          \
        double h2r = h.re * w7.re - h.im * w7.im;                                          \
        double h2i = h.re * w7.im + h.im * w7.re;                                          \
        double i2r = i.re * w8.re - i.im * w8.im;                                          \
        double i2i = i.re * w8.im + i.im * w8.re;                                          \
        double j2r = j.re * w9.re - j.im * w9.im;                                          \
        double j2i = j.re * w9.im + j.im * w9.re;                                          \
        double k2r = xk.re * w10.re - xk.im * w10.im;                                      \
        double k2i = xk.re * w10.im + xk.im * w10.re;                                      \
                                                                                           \
        /* Y0 (DC component) */                                                            \
        output_buffer[k] = (fft_data){                                                     \
            a.re + ((b2r + k2r) + (c2r + j2r) + (d2r + i2r) + (e2r + h2r) + (f2r + g2r)),  \
            a.im + ((b2i + k2i) + (c2i + j2i) + (d2i + i2i) + (e2i + h2i) + (f2i + g2i))}; \
                                                                                           \
        /* Compute all 5 pairs using helper macro */                                       \
        fft_data y1, y10;                                                                  \
        RADIX11_COMPUTE_PAIR_SCALAR(b2r, b2i, k2r, k2i, c2r, c2i, j2r, j2i,                \
                                    d2r, d2i, i2r, i2i, e2r, e2i, h2r, h2i,                \
                                    f2r, f2i, g2r, g2i, a.re, a.im,                        \
                                    C11_1, C11_2, C11_3, C11_4, C11_5,                     \
                                    S11_1, S11_2, S11_3, S11_4, S11_5,                     \
                                    1, y1, y10);                                           \
        output_buffer[k + K] = y1;                                                         \
        output_buffer[k + 10 * K] = y10;                                                   \
                                                                                           \
        fft_data y2, y9;                                                                   \
        RADIX11_COMPUTE_PAIR_SCALAR(b2r, b2i, k2r, k2i, c2r, c2i, j2r, j2i,                \
                                    d2r, d2i, i2r, i2i, e2r, e2i, h2r, h2i,                \
                                    f2r, f2i, g2r, g2i, a.re, a.im,                        \
                                    C11_2, C11_4, C11_5, C11_3, C11_1,                     \
                                    S11_2, S11_4, S11_5, S11_3, S11_1,                     \
                                    1, y2, y9);                                            \
        output_buffer[k + 2 * K] = y2;                                                     \
        output_buffer[k + 9 * K] = y9;                                                     \
                                                                                           \
        fft_data y3, y8;                                                                   \
        RADIX11_COMPUTE_PAIR_SCALAR(b2r, b2i, k2r, k2i, c2r, c2i, j2r, j2i,                \
                                    d2r, d2i, i2r, i2i, e2r, e2i, h2r, h2i,                \
                                    f2r, f2i, g2r, g2i, a.re, a.im,                        \
                                    C11_3, C11_5, C11_2, C11_1, C11_4,                     \
                                    S11_3, S11_5, S11_2, S11_1, S11_4,                     \
                                    1, y3, y8);                                            \
        output_buffer[k + 3 * K] = y3;                                                     \
        output_buffer[k + 8 * K] = y8;                                                     \
                                                                                           \
        fft_data y4, y7;                                                                   \
        RADIX11_COMPUTE_PAIR_SCALAR(b2r, b2i, k2r, k2i, c2r, c2i, j2r, j2i,                \
                                    d2r, d2i, i2r, i2i, e2r, e2i, h2r, h2i,                \
                                    f2r, f2i, g2r, g2i, a.re, a.im,                        \
                                    C11_4, C11_3, C11_1, C11_5, C11_2,                     \
                                    S11_4, S11_3, S11_1, S11_5, S11_2,                     \
                                    1, y4, y7);                                            \
        output_buffer[k + 4 * K] = y4;                                                     \
        output_buffer[k + 7 * K] = y7;                                                     \
                                                                                           \
        fft_data y5, y6;                                                                   \
        RADIX11_COMPUTE_PAIR_SCALAR(b2r, b2i, k2r, k2i, c2r, c2i, j2r, j2i,                \
                                    d2r, d2i, i2r, i2i, e2r, e2i, h2r, h2i,                \
                                    f2r, f2i, g2r, g2i, a.re, a.im,                        \
                                    C11_5, C11_1, C11_4, C11_2, C11_3,                     \
                                    S11_5, S11_1, S11_4, S11_2, S11_3,                     \
                                    1, y5, y6);                                            \
        output_buffer[k + 5 * K] = y5;                                                     \
        output_buffer[k + 6 * K] = y6;                                                     \
    } while (0)

/**
 * @brief Scalar radix-11 butterfly (INVERSE)
 *
 * Applies +i rotation to imaginary components
 */
#define RADIX11_BUTTERFLY_SCALAR_BV(k, K, sub_outputs, stage_tw, output_buffer)            \
    do                                                                                     \
    {                                                                                      \
        /* Load 11 lanes */                                                                \
        const fft_data a = sub_outputs[k];                                                 \
        const fft_data b = sub_outputs[k + K];                                             \
        const fft_data c = sub_outputs[k + 2 * K];                                         \
        const fft_data d = sub_outputs[k + 3 * K];                                         \
        const fft_data e = sub_outputs[k + 4 * K];                                         \
        const fft_data f = sub_outputs[k + 5 * K];                                         \
        const fft_data g = sub_outputs[k + 6 * K];                                         \
        const fft_data h = sub_outputs[k + 7 * K];                                         \
        const fft_data i = sub_outputs[k + 8 * K];                                         \
        const fft_data j = sub_outputs[k + 9 * K];                                         \
        const fft_data xk = sub_outputs[k + 10 * K];                                       \
                                                                                           \
        /* Load twiddles */                                                                \
        const fft_data w1 = stage_tw[10 * k + 0];                                          \
        const fft_data w2 = stage_tw[10 * k + 1];                                          \
        const fft_data w3 = stage_tw[10 * k + 2];                                          \
        const fft_data w4 = stage_tw[10 * k + 3];                                          \
        const fft_data w5 = stage_tw[10 * k + 4];                                          \
        const fft_data w6 = stage_tw[10 * k + 5];                                          \
        const fft_data w7 = stage_tw[10 * k + 6];                                          \
        const fft_data w8 = stage_tw[10 * k + 7];                                          \
        const fft_data w9 = stage_tw[10 * k + 8];                                          \
        const fft_data w10 = stage_tw[10 * k + 9];                                         \
                                                                                           \
        /* Twiddle multiply */                                                             \
        double b2r = b.re * w1.re - b.im * w1.im;                                          \
        double b2i = b.re * w1.im + b.im * w1.re;                                          \
        double c2r = c.re * w2.re - c.im * w2.im;                                          \
        double c2i = c.re * w2.im + c.im * w2.re;                                          \
        double d2r = d.re * w3.re - d.im * w3.im;                                          \
        double d2i = d.re * w3.im + d.im * w3.re;                                          \
        double e2r = e.re * w4.re - e.im * w4.im;                                          \
        double e2i = e.re * w4.im + e.im * w4.re;                                          \
        double f2r = f.re * w5.re - f.im * w5.im;                                          \
        double f2i = f.re * w5.im + f.im * w5.re;                                          \
        double g2r = g.re * w6.re - g.im * w6.im;                                          \
        double g2i = g.re * w6.im + g.im * w6.re;                                          \
        double h2r = h.re * w7.re - h.im * w7.im;                                          \
        double h2i = h.re * w7.im + h.im * w7.re;                                          \
        double i2r = i.re * w8.re - i.im * w8.im;                                          \
        double i2i = i.re * w8.im + i.im * w8.re;                                          \
        double j2r = j.re * w9.re - j.im * w9.im;                                          \
        double j2i = j.re * w9.im + j.im * w9.re;                                          \
        double k2r = xk.re * w10.re - xk.im * w10.im;                                      \
        double k2i = xk.re * w10.im + xk.im * w10.re;                                      \
                                                                                           \
        /* Y0 (DC component) */                                                            \
        output_buffer[k] = (fft_data){                                                     \
            a.re + ((b2r + k2r) + (c2r + j2r) + (d2r + i2r) + (e2r + h2r) + (f2r + g2r)),  \
            a.im + ((b2i + k2i) + (c2i + j2i) + (d2i + i2i) + (e2i + h2i) + (f2i + g2i))}; \
                                                                                           \
        /* Compute all 5 pairs - INVERSE uses is_forward=0 */                              \
        fft_data y1, y10;                                                                  \
        RADIX11_COMPUTE_PAIR_SCALAR(b2r, b2i, k2r, k2i, c2r, c2i, j2r, j2i,                \
                                    d2r, d2i, i2r, i2i, e2r, e2i, h2r, h2i,                \
                                    f2r, f2i, g2r, g2i, a.re, a.im,                        \
                                    C11_1, C11_2, C11_3, C11_4, C11_5,                     \
                                    S11_1, S11_2, S11_3, S11_4, S11_5,                     \
                                    0, y1, y10);                                           \
        output_buffer[k + K] = y1;                                                         \
        output_buffer[k + 10 * K] = y10;                                                   \
                                                                                           \
        fft_data y2, y9;                                                                   \
        RADIX11_COMPUTE_PAIR_SCALAR(b2r, b2i, k2r, k2i, c2r, c2i, j2r, j2i,                \
                                    d2r, d2i, i2r, i2i, e2r, e2i, h2r, h2i,                \
                                    f2r, f2i, g2r, g2i, a.re, a.im,                        \
                                    C11_2, C11_4, C11_5, C11_3, C11_1,                     \
                                    S11_2, S11_4, S11_5, S11_3, S11_1,                     \
                                    0, y2, y9);                                            \
        output_buffer[k + 2 * K] = y2;                                                     \
        output_buffer[k + 9 * K] = y9;                                                     \
                                                                                           \
        fft_data y3, y8;                                                                   \
        RADIX11_COMPUTE_PAIR_SCALAR(b2r, b2i, k2r, k2i, c2r, c2i, j2r, j2i,                \
                                    d2r, d2i, i2r, i2i, e2r, e2i, h2r, h2i,                \
                                    f2r, f2i, g2r, g2i, a.re, a.im,                        \
                                    C11_3, C11_5, C11_2, C11_1, C11_4,                     \
                                    S11_3, S11_5, S11_2, S11_1, S11_4,                     \
                                    0, y3, y8);                                            \
        output_buffer[k + 3 * K] = y3;                                                     \
        output_buffer[k + 8 * K] = y8;                                                     \
                                                                                           \
        fft_data y4, y7;                                                                   \
        RADIX11_COMPUTE_PAIR_SCALAR(b2r, b2i, k2r, k2i, c2r, c2i, j2r, j2i,                \
                                    d2r, d2i, i2r, i2i, e2r, e2i, h2r, h2i,                \
                                    f2r, f2i, g2r, g2i, a.re, a.im,                        \
                                    C11_4, C11_3, C11_1, C11_5, C11_2,                     \
                                    S11_4, S11_3, S11_1, S11_5, S11_2,                     \
                                    0, y4, y7);                                            \
        output_buffer[k + 4 * K] = y4;                                                     \
        output_buffer[k + 7 * K] = y7;                                                     \
                                                                                           \
        fft_data y5, y6;                                                                   \
        RADIX11_COMPUTE_PAIR_SCALAR(b2r, b2i, k2r, k2i, c2r, c2i, j2r, j2i,                \
                                    d2r, d2i, i2r, i2i, e2r, e2i, h2r, h2i,                \
                                    f2r, f2i, g2r, g2i, a.re, a.im,                        \
                                    C11_5, C11_1, C11_4, C11_2, C11_3,                     \
                                    S11_5, S11_1, S11_4, S11_2, S11_3,                     \
                                    0, y5, y6);                                            \
        output_buffer[k + 5 * K] = y5;                                                     \
        output_buffer[k + 6 * K] = y6;                                                     \
    } while (0)

#endif // FFT_RADIX11_MACROS_H
