/**
 * @file fft_radix32_avx2_n1.c
 * @brief Twiddle-less radix-32 butterfly for AVX2 (n1 codelet)
 *
 * Bottom-of-recursion 32-point DFT with zero twiddle memory traffic.
 * All W_32^k constants are hardcoded in the instruction stream.
 *
 * Architecture: 4×8 decomposition (DIT-4 → twiddle → DIF-8)
 *   Pass 1: Radix-4 DIT on 8 groups (no stage twiddles — n1 codelet)
 *   Inter:  Hardcoded W_32^{g·b} applied per-bin
 *   Pass 2: Radix-8 DIF on 4 bins
 *
 * Temp buffer: 32×4 doubles = 1KB re + 1KB im, stack-allocated, L1-resident.
 * No twiddle arrays, no prefetch overhead, no multi-mode system.
 *
 * @note Reuses butterfly cores from fft_radix32_avx2.c
 */

#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

#ifndef TARGET_AVX2_FMA
#define TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#endif
#ifndef FORCE_INLINE
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif
#ifndef RESTRICT
#define RESTRICT __restrict__
#endif
#ifndef ALIGNAS
#define ALIGNAS(n) __attribute__((aligned(n)))
#endif

/*==========================================================================
 * COMPLEX ARITHMETIC
 *=========================================================================*/

TARGET_AVX2_FMA
FORCE_INLINE void cmul_v256(
    __m256d ar, __m256d ai, __m256d br, __m256d bi,
    __m256d *RESTRICT cr, __m256d *RESTRICT ci)
{
    __m256d ai_bi = _mm256_mul_pd(ai, bi);
    __m256d ai_br = _mm256_mul_pd(ai, br);
    *cr = _mm256_fmsub_pd(ar, br, ai_bi);
    *ci = _mm256_fmadd_pd(ar, bi, ai_br);
}

/*==========================================================================
 * RADIX-4 DIT CORES
 *=========================================================================*/

TARGET_AVX2_FMA
FORCE_INLINE void radix4_dit_core_forward_avx2(
    __m256d x0r, __m256d x0i, __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i, __m256d x3r, __m256d x3i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i)
{
    __m256d t0r = _mm256_add_pd(x0r, x2r), t0i = _mm256_add_pd(x0i, x2i);
    __m256d t1r = _mm256_sub_pd(x0r, x2r), t1i = _mm256_sub_pd(x0i, x2i);
    __m256d t2r = _mm256_add_pd(x1r, x3r), t2i = _mm256_add_pd(x1i, x3i);
    __m256d t3r = _mm256_sub_pd(x1r, x3r), t3i = _mm256_sub_pd(x1i, x3i);
    *y0r = _mm256_add_pd(t0r, t2r); *y0i = _mm256_add_pd(t0i, t2i);
    *y1r = _mm256_add_pd(t1r, t3i); *y1i = _mm256_sub_pd(t1i, t3r);
    *y2r = _mm256_sub_pd(t0r, t2r); *y2i = _mm256_sub_pd(t0i, t2i);
    *y3r = _mm256_sub_pd(t1r, t3i); *y3i = _mm256_add_pd(t1i, t3r);
}

TARGET_AVX2_FMA
FORCE_INLINE void radix4_dit_core_backward_avx2(
    __m256d x0r, __m256d x0i, __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i, __m256d x3r, __m256d x3i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i)
{
    __m256d t0r = _mm256_add_pd(x0r, x2r), t0i = _mm256_add_pd(x0i, x2i);
    __m256d t1r = _mm256_sub_pd(x0r, x2r), t1i = _mm256_sub_pd(x0i, x2i);
    __m256d t2r = _mm256_add_pd(x1r, x3r), t2i = _mm256_add_pd(x1i, x3i);
    __m256d t3r = _mm256_sub_pd(x1r, x3r), t3i = _mm256_sub_pd(x1i, x3i);
    *y0r = _mm256_add_pd(t0r, t2r); *y0i = _mm256_add_pd(t0i, t2i);
    *y1r = _mm256_sub_pd(t1r, t3i); *y1i = _mm256_add_pd(t1i, t3r);
    *y2r = _mm256_sub_pd(t0r, t2r); *y2i = _mm256_sub_pd(t0i, t2i);
    *y3r = _mm256_add_pd(t1r, t3i); *y3i = _mm256_sub_pd(t1i, t3r);
}

/*==========================================================================
 * RADIX-8 DIF CORES
 *=========================================================================*/

TARGET_AVX2_FMA
FORCE_INLINE void radix8_dif_core_forward_avx2(
    __m256d x0r, __m256d x0i, __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i, __m256d x3r, __m256d x3i,
    __m256d x4r, __m256d x4i, __m256d x5r, __m256d x5i,
    __m256d x6r, __m256d x6i, __m256d x7r, __m256d x7i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i,
    __m256d *RESTRICT y4r, __m256d *RESTRICT y4i,
    __m256d *RESTRICT y5r, __m256d *RESTRICT y5i,
    __m256d *RESTRICT y6r, __m256d *RESTRICT y6i,
    __m256d *RESTRICT y7r, __m256d *RESTRICT y7i)
{
    const __m256d W8r = _mm256_set1_pd( 0.70710678118654752440);
    const __m256d W8i = _mm256_set1_pd(-0.70710678118654752440);
    const __m256d NEG = _mm256_set1_pd(-0.0);

    __m256d a0r = _mm256_add_pd(x0r, x4r), a0i = _mm256_add_pd(x0i, x4i);
    __m256d a4r = _mm256_sub_pd(x0r, x4r), a4i = _mm256_sub_pd(x0i, x4i);
    __m256d a1r = _mm256_add_pd(x1r, x5r), a1i = _mm256_add_pd(x1i, x5i);
    __m256d a5r = _mm256_sub_pd(x1r, x5r), a5i = _mm256_sub_pd(x1i, x5i);
    __m256d a2r = _mm256_add_pd(x2r, x6r), a2i = _mm256_add_pd(x2i, x6i);
    __m256d a6r = _mm256_sub_pd(x2r, x6r), a6i = _mm256_sub_pd(x2i, x6i);
    __m256d a3r = _mm256_add_pd(x3r, x7r), a3i = _mm256_add_pd(x3i, x7i);
    __m256d a7r = _mm256_sub_pd(x3r, x7r), a7i = _mm256_sub_pd(x3i, x7i);

    __m256d b5r, b5i; cmul_v256(a5r, a5i, W8r, W8i, &b5r, &b5i);
    __m256d b6r = a6i, b6i = _mm256_xor_pd(a6r, NEG);
    __m256d nW8 = _mm256_xor_pd(W8r, NEG);
    __m256d b7r, b7i; cmul_v256(a7r, a7i, nW8, W8i, &b7r, &b7i);

    __m256d e0r = _mm256_add_pd(a0r, a2r), e0i = _mm256_add_pd(a0i, a2i);
    __m256d e1r = _mm256_sub_pd(a0r, a2r), e1i = _mm256_sub_pd(a0i, a2i);
    __m256d e2r = _mm256_add_pd(a1r, a3r), e2i = _mm256_add_pd(a1i, a3i);
    __m256d e3r = _mm256_sub_pd(a1r, a3r), e3i = _mm256_sub_pd(a1i, a3i);

    *y0r = _mm256_add_pd(e0r, e2r); *y0i = _mm256_add_pd(e0i, e2i);
    *y2r = _mm256_add_pd(e1r, e3i); *y2i = _mm256_sub_pd(e1i, e3r);
    *y4r = _mm256_sub_pd(e0r, e2r); *y4i = _mm256_sub_pd(e0i, e2i);
    *y6r = _mm256_sub_pd(e1r, e3i); *y6i = _mm256_add_pd(e1i, e3r);

    __m256d o0r = _mm256_add_pd(a4r, b6r), o0i = _mm256_add_pd(a4i, b6i);
    __m256d o1r = _mm256_sub_pd(a4r, b6r), o1i = _mm256_sub_pd(a4i, b6i);
    __m256d o2r = _mm256_add_pd(b5r, b7r), o2i = _mm256_add_pd(b5i, b7i);
    __m256d o3r = _mm256_sub_pd(b5r, b7r), o3i = _mm256_sub_pd(b5i, b7i);

    *y1r = _mm256_add_pd(o0r, o2r); *y1i = _mm256_add_pd(o0i, o2i);
    *y3r = _mm256_add_pd(o1r, o3i); *y3i = _mm256_sub_pd(o1i, o3r);
    *y5r = _mm256_sub_pd(o0r, o2r); *y5i = _mm256_sub_pd(o0i, o2i);
    *y7r = _mm256_sub_pd(o1r, o3i); *y7i = _mm256_add_pd(o1i, o3r);
}

TARGET_AVX2_FMA
FORCE_INLINE void radix8_dif_core_backward_avx2(
    __m256d x0r, __m256d x0i, __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i, __m256d x3r, __m256d x3i,
    __m256d x4r, __m256d x4i, __m256d x5r, __m256d x5i,
    __m256d x6r, __m256d x6i, __m256d x7r, __m256d x7i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i,
    __m256d *RESTRICT y4r, __m256d *RESTRICT y4i,
    __m256d *RESTRICT y5r, __m256d *RESTRICT y5i,
    __m256d *RESTRICT y6r, __m256d *RESTRICT y6i,
    __m256d *RESTRICT y7r, __m256d *RESTRICT y7i)
{
    const __m256d W8r = _mm256_set1_pd(0.70710678118654752440);
    const __m256d W8i = _mm256_set1_pd(0.70710678118654752440); /* conjugated */
    const __m256d NEG = _mm256_set1_pd(-0.0);

    __m256d a0r = _mm256_add_pd(x0r, x4r), a0i = _mm256_add_pd(x0i, x4i);
    __m256d a4r = _mm256_sub_pd(x0r, x4r), a4i = _mm256_sub_pd(x0i, x4i);
    __m256d a1r = _mm256_add_pd(x1r, x5r), a1i = _mm256_add_pd(x1i, x5i);
    __m256d a5r = _mm256_sub_pd(x1r, x5r), a5i = _mm256_sub_pd(x1i, x5i);
    __m256d a2r = _mm256_add_pd(x2r, x6r), a2i = _mm256_add_pd(x2i, x6i);
    __m256d a6r = _mm256_sub_pd(x2r, x6r), a6i = _mm256_sub_pd(x2i, x6i);
    __m256d a3r = _mm256_add_pd(x3r, x7r), a3i = _mm256_add_pd(x3i, x7i);
    __m256d a7r = _mm256_sub_pd(x3r, x7r), a7i = _mm256_sub_pd(x3i, x7i);

    __m256d b5r, b5i; cmul_v256(a5r, a5i, W8r, W8i, &b5r, &b5i);
    __m256d b6r = _mm256_xor_pd(a6i, NEG), b6i = a6r;
    __m256d nW8 = _mm256_xor_pd(W8r, NEG);
    __m256d b7r, b7i; cmul_v256(a7r, a7i, nW8, W8i, &b7r, &b7i);

    __m256d e0r = _mm256_add_pd(a0r, a2r), e0i = _mm256_add_pd(a0i, a2i);
    __m256d e1r = _mm256_sub_pd(a0r, a2r), e1i = _mm256_sub_pd(a0i, a2i);
    __m256d e2r = _mm256_add_pd(a1r, a3r), e2i = _mm256_add_pd(a1i, a3i);
    __m256d e3r = _mm256_sub_pd(a1r, a3r), e3i = _mm256_sub_pd(a1i, a3i);

    *y0r = _mm256_add_pd(e0r, e2r); *y0i = _mm256_add_pd(e0i, e2i);
    *y2r = _mm256_sub_pd(e1r, e3i); *y2i = _mm256_add_pd(e1i, e3r);
    *y4r = _mm256_sub_pd(e0r, e2r); *y4i = _mm256_sub_pd(e0i, e2i);
    *y6r = _mm256_add_pd(e1r, e3i); *y6i = _mm256_sub_pd(e1i, e3r);

    __m256d o0r = _mm256_add_pd(a4r, b6r), o0i = _mm256_add_pd(a4i, b6i);
    __m256d o1r = _mm256_sub_pd(a4r, b6r), o1i = _mm256_sub_pd(a4i, b6i);
    __m256d o2r = _mm256_add_pd(b5r, b7r), o2i = _mm256_add_pd(b5i, b7i);
    __m256d o3r = _mm256_sub_pd(b5r, b7r), o3i = _mm256_sub_pd(b5i, b7i);

    *y1r = _mm256_add_pd(o0r, o2r); *y1i = _mm256_add_pd(o0i, o2i);
    *y3r = _mm256_sub_pd(o1r, o3i); *y3i = _mm256_add_pd(o1i, o3r);
    *y5r = _mm256_sub_pd(o0r, o2r); *y5i = _mm256_sub_pd(o0i, o2i);
    *y7r = _mm256_add_pd(o1r, o3i); *y7i = _mm256_sub_pd(o1i, o3r);
}

/*==========================================================================
 * W_32 CONSTANTS
 *
 * W_32^k = cos(πk/16) − i·sin(πk/16) (forward)
 * Symmetry: W^{k+16} = −W^k,  W^8 = −j
 *=========================================================================*/

#define W32_C1  0.98078528040323044295   /* cos( π/16) */
#define W32_S1  0.19509032201612826785   /* sin( π/16) */
#define W32_C2  0.92387953251128675613   /* cos( π/8 ) */
#define W32_S2  0.38268343236508977173   /* sin( π/8 ) */
#define W32_C3  0.83146961230254523708   /* cos(3π/16) */
#define W32_S3  0.55557023301960222474   /* sin(3π/16) */
#define W32_C4  0.70710678118654752440   /* cos( π/4 ) = √2/2 */
#define W32_S4  0.70710678118654752440   /* sin( π/4 ) */
#define W32_C5  0.55557023301960222474   /* cos(5π/16) */
#define W32_S5  0.83146961230254523708   /* sin(5π/16) */
#define W32_C6  0.38268343236508977173   /* cos(3π/8 ) */
#define W32_S6  0.92387953251128675613   /* sin(3π/8 ) */
#define W32_C7  0.19509032201612826785   /* cos(7π/16) */
#define W32_S7  0.98078528040323044295   /* sin(7π/16) */

/*==========================================================================
 * ROTATION HELPERS — free or nearly free
 *=========================================================================*/

/** Multiply by −j: (a+jb)·(−j) = b − ja.  Cost: 1 XOR */
TARGET_AVX2_FMA static FORCE_INLINE void
rot_neg_j(__m256d xr, __m256d xi, __m256d *yr, __m256d *yi)
{
    *yr = xi;
    *yi = _mm256_xor_pd(xr, _mm256_set1_pd(-0.0));
}

/** Multiply by +j: (a+jb)·(+j) = −b + ja.  Cost: 1 XOR */
TARGET_AVX2_FMA static FORCE_INLINE void
rot_pos_j(__m256d xr, __m256d xi, __m256d *yr, __m256d *yi)
{
    *yr = _mm256_xor_pd(xi, _mm256_set1_pd(-0.0));
    *yi = xr;
}

/*==========================================================================
 * TEMP BUFFER LAYOUT (n1 codelet, K=1 vector = 4 doubles)
 *
 * Bin-major: 32 temp "stripes" × 4 doubles each = 128 doubles
 * Stripe index = bin*8 + group
 * Addressing: tmp[stripe * 4]  (no +k offset since K=1 vector width)
 *=========================================================================*/

#define T(stripe) ((stripe) * 4)

/*==========================================================================
 * LOAD 8 TEMP STRIPES FOR A BIN
 *=========================================================================*/

#define LOAD_BIN(tmp_re, tmp_im, base,                              \
    x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,                      \
    x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i)                      \
do {                                                                \
    x0r = _mm256_load_pd(&(tmp_re)[T((base)+0)]);                  \
    x0i = _mm256_load_pd(&(tmp_im)[T((base)+0)]);                  \
    x1r = _mm256_load_pd(&(tmp_re)[T((base)+1)]);                  \
    x1i = _mm256_load_pd(&(tmp_im)[T((base)+1)]);                  \
    x2r = _mm256_load_pd(&(tmp_re)[T((base)+2)]);                  \
    x2i = _mm256_load_pd(&(tmp_im)[T((base)+2)]);                  \
    x3r = _mm256_load_pd(&(tmp_re)[T((base)+3)]);                  \
    x3i = _mm256_load_pd(&(tmp_im)[T((base)+3)]);                  \
    x4r = _mm256_load_pd(&(tmp_re)[T((base)+4)]);                  \
    x4i = _mm256_load_pd(&(tmp_im)[T((base)+4)]);                  \
    x5r = _mm256_load_pd(&(tmp_re)[T((base)+5)]);                  \
    x5i = _mm256_load_pd(&(tmp_im)[T((base)+5)]);                  \
    x6r = _mm256_load_pd(&(tmp_re)[T((base)+6)]);                  \
    x6i = _mm256_load_pd(&(tmp_im)[T((base)+6)]);                  \
    x7r = _mm256_load_pd(&(tmp_re)[T((base)+7)]);                  \
    x7i = _mm256_load_pd(&(tmp_im)[T((base)+7)]);                  \
} while (0)

/*==========================================================================
 * STORE 8 OUTPUTS FOR A BIN
 *=========================================================================*/

#define STORE_BIN(out_re, out_im, os, base,                         \
    y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,                       \
    y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i)                       \
do {                                                                \
    _mm256_store_pd(&(out_re)[((base)+0)*(os)], y0r);              \
    _mm256_store_pd(&(out_im)[((base)+0)*(os)], y0i);              \
    _mm256_store_pd(&(out_re)[((base)+1)*(os)], y1r);              \
    _mm256_store_pd(&(out_im)[((base)+1)*(os)], y1i);              \
    _mm256_store_pd(&(out_re)[((base)+2)*(os)], y2r);              \
    _mm256_store_pd(&(out_im)[((base)+2)*(os)], y2i);              \
    _mm256_store_pd(&(out_re)[((base)+3)*(os)], y3r);              \
    _mm256_store_pd(&(out_im)[((base)+3)*(os)], y3i);              \
    _mm256_store_pd(&(out_re)[((base)+4)*(os)], y4r);              \
    _mm256_store_pd(&(out_im)[((base)+4)*(os)], y4i);              \
    _mm256_store_pd(&(out_re)[((base)+5)*(os)], y5r);              \
    _mm256_store_pd(&(out_im)[((base)+5)*(os)], y5i);              \
    _mm256_store_pd(&(out_re)[((base)+6)*(os)], y6r);              \
    _mm256_store_pd(&(out_im)[((base)+6)*(os)], y6i);              \
    _mm256_store_pd(&(out_re)[((base)+7)*(os)], y7r);              \
    _mm256_store_pd(&(out_im)[((base)+7)*(os)], y7i);              \
} while (0)

/*==========================================================================
 * TWIDDLE APPLICATION HELPER
 * cmul with compile-time constant broadcast
 *=========================================================================*/

#define CMUL_CONST(xr, xi, c_re, c_im) do {                        \
    __m256d _tr, _ti;                                               \
    cmul_v256(xr, xi,                                               \
              _mm256_set1_pd(c_re), _mm256_set1_pd(c_im),          \
              &_tr, &_ti);                                          \
    xr = _tr; xi = _ti;                                            \
} while (0)

/*==========================================================================
 * FORWARD: PER-BIN TWIDDLE + RADIX-8 DIF
 *
 * Inter-stage twiddle for group g, bin b: W_32^{g·b}
 * Group 0 always gets W^0 = 1 → skip
 *=========================================================================*/

/**
 * Bin 0: all twiddles W_32^0 = 1 → pure radix-8 DIF
 */
TARGET_AVX2_FMA static FORCE_INLINE void
n1_bin0_fwd(const double *tr, const double *ti,
            double *or_, double *oi, size_t os)
{
    __m256d x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i;
    __m256d x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
    LOAD_BIN(tr, ti, 0,
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);

    __m256d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;
    __m256d y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i;
    radix8_dif_core_forward_avx2(
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        &y0r,&y0i,&y1r,&y1i,&y2r,&y2i,&y3r,&y3i,
        &y4r,&y4i,&y5r,&y5i,&y6r,&y6i,&y7r,&y7i);

    STORE_BIN(or_, oi, os, 0,
        y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,
        y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i);
}

/**
 * Bin 1: W_32^{0,1,2,3,4,5,6,7} on groups 0..7
 */
TARGET_AVX2_FMA static FORCE_INLINE void
n1_bin1_fwd(const double *tr, const double *ti,
            double *or_, double *oi, size_t os)
{
    __m256d x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i;
    __m256d x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
    LOAD_BIN(tr, ti, 8,
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);

    /* g=0: W^0=1 skip */
    CMUL_CONST(x1r, x1i,  W32_C1, -W32_S1); /* g=1: W^1 */
    CMUL_CONST(x2r, x2i,  W32_C2, -W32_S2); /* g=2: W^2 */
    CMUL_CONST(x3r, x3i,  W32_C3, -W32_S3); /* g=3: W^3 */
    CMUL_CONST(x4r, x4i,  W32_C4, -W32_S4); /* g=4: W^4 */
    CMUL_CONST(x5r, x5i,  W32_C5, -W32_S5); /* g=5: W^5 */
    CMUL_CONST(x6r, x6i,  W32_C6, -W32_S6); /* g=6: W^6 */
    CMUL_CONST(x7r, x7i,  W32_C7, -W32_S7); /* g=7: W^7 */

    __m256d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;
    __m256d y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i;
    radix8_dif_core_forward_avx2(
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        &y0r,&y0i,&y1r,&y1i,&y2r,&y2i,&y3r,&y3i,
        &y4r,&y4i,&y5r,&y5i,&y6r,&y6i,&y7r,&y7i);

    STORE_BIN(or_, oi, os, 8,
        y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,
        y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i);
}

/**
 * Bin 2: W_32^{0,2,4,6,8,10,12,14}
 * g=4: W^8 = −j (free rotation)
 * g≥5: W^{k} with k>8, use W^{k} = −W^{k−16} symmetry
 */
TARGET_AVX2_FMA static FORCE_INLINE void
n1_bin2_fwd(const double *tr, const double *ti,
            double *or_, double *oi, size_t os)
{
    __m256d x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i;
    __m256d x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
    LOAD_BIN(tr, ti, 16,
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);

    /* g=0: W^0=1 */
    CMUL_CONST(x1r, x1i,  W32_C2, -W32_S2); /* g=1: W^2  */
    CMUL_CONST(x2r, x2i,  W32_C4, -W32_S4); /* g=2: W^4  */
    CMUL_CONST(x3r, x3i,  W32_C6, -W32_S6); /* g=3: W^6  */
    rot_neg_j(x4r, x4i, &x4r, &x4i);        /* g=4: W^8 = −j */
    /* g=5: W^10 = −W^{−6} → (−cos3π/8, −sin3π/8) */
    CMUL_CONST(x5r, x5i, -W32_C6, -W32_S6);
    /* g=6: W^12 = −W^{−4} → (−√2/2, −√2/2) */
    CMUL_CONST(x6r, x6i, -W32_C4, -W32_S4);
    /* g=7: W^14 = −W^{−2} → (−cosπ/8, −sinπ/8) */
    CMUL_CONST(x7r, x7i, -W32_C2, -W32_S2);

    __m256d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;
    __m256d y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i;
    radix8_dif_core_forward_avx2(
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        &y0r,&y0i,&y1r,&y1i,&y2r,&y2i,&y3r,&y3i,
        &y4r,&y4i,&y5r,&y5i,&y6r,&y6i,&y7r,&y7i);

    STORE_BIN(or_, oi, os, 16,
        y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,
        y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i);
}

/**
 * Bin 3: W_32^{0,3,6,9,12,15,18,21}
 *
 * W^9  = (−sinπ/16, −cosπ/16)
 * W^12 = (−√2/2,    −√2/2)
 * W^15 = (−cosπ/16, −sinπ/16)
 * W^18 = −W^2  = (−cosπ/8,  +sinπ/8)
 * W^21 = −W^5  = (−cos5π/16,+sin5π/16)
 */
TARGET_AVX2_FMA static FORCE_INLINE void
n1_bin3_fwd(const double *tr, const double *ti,
            double *or_, double *oi, size_t os)
{
    __m256d x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i;
    __m256d x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
    LOAD_BIN(tr, ti, 24,
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);

    /* g=0: W^0=1 */
    CMUL_CONST(x1r, x1i,  W32_C3, -W32_S3); /* g=1: W^3  */
    CMUL_CONST(x2r, x2i,  W32_C6, -W32_S6); /* g=2: W^6  */
    CMUL_CONST(x3r, x3i, -W32_S1, -W32_C1); /* g=3: W^9  */
    CMUL_CONST(x4r, x4i, -W32_C4, -W32_S4); /* g=4: W^12 */
    CMUL_CONST(x5r, x5i, -W32_C1, -W32_S1); /* g=5: W^15 */
    CMUL_CONST(x6r, x6i, -W32_C2, +W32_S2); /* g=6: W^18 = −W^2 */
    CMUL_CONST(x7r, x7i, -W32_C5, +W32_S5); /* g=7: W^21 = −W^5 */

    __m256d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;
    __m256d y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i;
    radix8_dif_core_forward_avx2(
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        &y0r,&y0i,&y1r,&y1i,&y2r,&y2i,&y3r,&y3i,
        &y4r,&y4i,&y5r,&y5i,&y6r,&y6i,&y7r,&y7i);

    STORE_BIN(or_, oi, os, 24,
        y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,
        y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i);
}

/*==========================================================================
 * BACKWARD: PER-BIN TWIDDLE + RADIX-8 DIF
 * All twiddle imag parts sign-flipped (conjugated roots of unity)
 *=========================================================================*/

TARGET_AVX2_FMA static FORCE_INLINE void
n1_bin0_bwd(const double *tr, const double *ti,
            double *or_, double *oi, size_t os)
{
    __m256d x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i;
    __m256d x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
    LOAD_BIN(tr, ti, 0,
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);

    __m256d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;
    __m256d y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i;
    radix8_dif_core_backward_avx2(
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        &y0r,&y0i,&y1r,&y1i,&y2r,&y2i,&y3r,&y3i,
        &y4r,&y4i,&y5r,&y5i,&y6r,&y6i,&y7r,&y7i);

    STORE_BIN(or_, oi, os, 0,
        y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,
        y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i);
}

TARGET_AVX2_FMA static FORCE_INLINE void
n1_bin1_bwd(const double *tr, const double *ti,
            double *or_, double *oi, size_t os)
{
    __m256d x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i;
    __m256d x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
    LOAD_BIN(tr, ti, 8,
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);

    CMUL_CONST(x1r, x1i,  W32_C1, +W32_S1);
    CMUL_CONST(x2r, x2i,  W32_C2, +W32_S2);
    CMUL_CONST(x3r, x3i,  W32_C3, +W32_S3);
    CMUL_CONST(x4r, x4i,  W32_C4, +W32_S4);
    CMUL_CONST(x5r, x5i,  W32_C5, +W32_S5);
    CMUL_CONST(x6r, x6i,  W32_C6, +W32_S6);
    CMUL_CONST(x7r, x7i,  W32_C7, +W32_S7);

    __m256d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;
    __m256d y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i;
    radix8_dif_core_backward_avx2(
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        &y0r,&y0i,&y1r,&y1i,&y2r,&y2i,&y3r,&y3i,
        &y4r,&y4i,&y5r,&y5i,&y6r,&y6i,&y7r,&y7i);

    STORE_BIN(or_, oi, os, 8,
        y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,
        y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i);
}

TARGET_AVX2_FMA static FORCE_INLINE void
n1_bin2_bwd(const double *tr, const double *ti,
            double *or_, double *oi, size_t os)
{
    __m256d x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i;
    __m256d x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
    LOAD_BIN(tr, ti, 16,
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);

    CMUL_CONST(x1r, x1i,  W32_C2, +W32_S2); /* W^{-2} */
    CMUL_CONST(x2r, x2i,  W32_C4, +W32_S4); /* W^{-4} */
    CMUL_CONST(x3r, x3i,  W32_C6, +W32_S6); /* W^{-6} */
    rot_pos_j(x4r, x4i, &x4r, &x4i);        /* W^{-8} = +j */
    CMUL_CONST(x5r, x5i, -W32_C6, +W32_S6); /* W^{-10} */
    CMUL_CONST(x6r, x6i, -W32_C4, +W32_S4); /* W^{-12} */
    CMUL_CONST(x7r, x7i, -W32_C2, +W32_S2); /* W^{-14} */

    __m256d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;
    __m256d y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i;
    radix8_dif_core_backward_avx2(
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        &y0r,&y0i,&y1r,&y1i,&y2r,&y2i,&y3r,&y3i,
        &y4r,&y4i,&y5r,&y5i,&y6r,&y6i,&y7r,&y7i);

    STORE_BIN(or_, oi, os, 16,
        y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,
        y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i);
}

TARGET_AVX2_FMA static FORCE_INLINE void
n1_bin3_bwd(const double *tr, const double *ti,
            double *or_, double *oi, size_t os)
{
    __m256d x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i;
    __m256d x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
    LOAD_BIN(tr, ti, 24,
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);

    CMUL_CONST(x1r, x1i,  W32_C3, +W32_S3); /* W^{-3}  */
    CMUL_CONST(x2r, x2i,  W32_C6, +W32_S6); /* W^{-6}  */
    CMUL_CONST(x3r, x3i, -W32_S1, +W32_C1); /* W^{-9}  */
    CMUL_CONST(x4r, x4i, -W32_C4, +W32_S4); /* W^{-12} */
    CMUL_CONST(x5r, x5i, -W32_C1, +W32_S1); /* W^{-15} */
    CMUL_CONST(x6r, x6i, -W32_C2, -W32_S2); /* W^{-18} = −W^{-2} */
    CMUL_CONST(x7r, x7i, -W32_C5, -W32_S5); /* W^{-21} = −W^{-5} */

    __m256d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;
    __m256d y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i;
    radix8_dif_core_backward_avx2(
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,
        x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        &y0r,&y0i,&y1r,&y1i,&y2r,&y2i,&y3r,&y3i,
        &y4r,&y4i,&y5r,&y5i,&y6r,&y6i,&y7r,&y7i);

    STORE_BIN(or_, oi, os, 24,
        y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,
        y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i);
}

/*==========================================================================
 * PASS 1: TWIDDLE-LESS RADIX-4 DIT (8 groups, no stage twiddles)
 *
 * Group g reads stripes {g, g+8, g+16, g+24} from input (stride = 8 * os)
 * and writes bin-major: bin b → tmp[b*8 + g]
 *
 * For n1 codelet K = vector_width = 4 doubles, so the "loop" is one
 * iteration per group — no pipelining needed.
 *=========================================================================*/

#define PASS1_GROUP(g, dir)                                                 \
do {                                                                        \
    __m256d x0r = _mm256_load_pd(&in_re[( (g)      ) * is]);               \
    __m256d x0i = _mm256_load_pd(&in_im[( (g)      ) * is]);               \
    __m256d x1r = _mm256_load_pd(&in_re[( (g) +  8 ) * is]);               \
    __m256d x1i = _mm256_load_pd(&in_im[( (g) +  8 ) * is]);               \
    __m256d x2r = _mm256_load_pd(&in_re[( (g) + 16 ) * is]);               \
    __m256d x2i = _mm256_load_pd(&in_im[( (g) + 16 ) * is]);               \
    __m256d x3r = _mm256_load_pd(&in_re[( (g) + 24 ) * is]);               \
    __m256d x3i = _mm256_load_pd(&in_im[( (g) + 24 ) * is]);               \
    __m256d y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i;                              \
    radix4_dit_core_##dir##_avx2(                                           \
        x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,                                 \
        &y0r,&y0i,&y1r,&y1i,&y2r,&y2i,&y3r,&y3i);                         \
    /* bin-major: bin b → stripe b*8+g */                                   \
    _mm256_store_pd(&tmp_re[T(0*8+(g))], y0r);                             \
    _mm256_store_pd(&tmp_im[T(0*8+(g))], y0i);                             \
    _mm256_store_pd(&tmp_re[T(1*8+(g))], y1r);                             \
    _mm256_store_pd(&tmp_im[T(1*8+(g))], y1i);                             \
    _mm256_store_pd(&tmp_re[T(2*8+(g))], y2r);                             \
    _mm256_store_pd(&tmp_im[T(2*8+(g))], y2i);                             \
    _mm256_store_pd(&tmp_re[T(3*8+(g))], y3r);                             \
    _mm256_store_pd(&tmp_im[T(3*8+(g))], y3i);                             \
} while (0)

/*==========================================================================
 * PUBLIC API: TWIDDLE-LESS RADIX-32 (n1 CODELET)
 *=========================================================================*/

/**
 * @brief Twiddle-less 32-point FFT — FORWARD
 *
 * Computes 4-wide 32-point DFTs (4 independent transforms in parallel).
 *
 * @param in_re   Input real,  accessed as in_re[stripe * in_stride]
 * @param in_im   Input imag
 * @param out_re  Output real, accessed as out_re[stripe * out_stride]
 * @param out_im  Output imag
 * @param in_stride   Stride between input stripes (in doubles)
 * @param out_stride  Stride between output stripes (in doubles)
 */
TARGET_AVX2_FMA
void fft_radix32_n1_forward_avx2(
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    size_t in_stride,
    size_t out_stride)
{
    const size_t is = in_stride;
    const size_t os = out_stride;

    /* Stack temp: 32 stripes × 4 doubles = 128 per component, L1-resident */
    ALIGNAS(32) double tmp_re[128];
    ALIGNAS(32) double tmp_im[128];

    /* PASS 1: Radix-4 DIT on 8 groups (no stage twiddles) */
    PASS1_GROUP(0, forward);
    PASS1_GROUP(1, forward);
    PASS1_GROUP(2, forward);
    PASS1_GROUP(3, forward);
    PASS1_GROUP(4, forward);
    PASS1_GROUP(5, forward);
    PASS1_GROUP(6, forward);
    PASS1_GROUP(7, forward);

    /* PASS 2: Per-bin hardcoded twiddle + radix-8 DIF */
    n1_bin0_fwd(tmp_re, tmp_im, out_re, out_im, os);
    n1_bin1_fwd(tmp_re, tmp_im, out_re, out_im, os);
    n1_bin2_fwd(tmp_re, tmp_im, out_re, out_im, os);
    n1_bin3_fwd(tmp_re, tmp_im, out_re, out_im, os);

    _mm256_zeroupper();
}

/**
 * @brief Twiddle-less 32-point FFT — BACKWARD (IFFT without 1/N scaling)
 */
TARGET_AVX2_FMA
void fft_radix32_n1_backward_avx2(
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    size_t in_stride,
    size_t out_stride)
{
    const size_t is = in_stride;
    const size_t os = out_stride;

    ALIGNAS(32) double tmp_re[128];
    ALIGNAS(32) double tmp_im[128];

    /* PASS 1: Radix-4 DIT BACKWARD */
    PASS1_GROUP(0, backward);
    PASS1_GROUP(1, backward);
    PASS1_GROUP(2, backward);
    PASS1_GROUP(3, backward);
    PASS1_GROUP(4, backward);
    PASS1_GROUP(5, backward);
    PASS1_GROUP(6, backward);
    PASS1_GROUP(7, backward);

    /* PASS 2: Per-bin conjugated twiddle + radix-8 DIF backward */
    n1_bin0_bwd(tmp_re, tmp_im, out_re, out_im, os);
    n1_bin1_bwd(tmp_re, tmp_im, out_re, out_im, os);
    n1_bin2_bwd(tmp_re, tmp_im, out_re, out_im, os);
    n1_bin3_bwd(tmp_re, tmp_im, out_re, out_im, os);

    _mm256_zeroupper();
}

#undef T
#undef LOAD_BIN
#undef STORE_BIN
#undef CMUL_CONST
#undef PASS1_GROUP
