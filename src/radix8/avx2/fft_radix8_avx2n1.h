/**
 * @file fft_radix8_avx2_n1.h
 * @brief Twiddle-less Radix-8 AVX2 Stage Drivers (N1 variant)
 *
 * For the first stage of a mixed-radix FFT where all stage twiddles are unity
 * (W_N^(j·0) = 1 for the outermost stage where k ranges within stride-1).
 *
 * Performance vs twiddled version:
 * - Zero twiddle loads (saves 4-8 cache lines/iteration)
 * - Zero complex multiplications for twiddles (saves 14 FMA/iteration)
 * - ~50-70% faster than BLOCKED4 for same K
 *
 * Optimizations:
 * ✅ U=2 software pipelining (overlapped loads/compute/stores)
 * ✅ Two-wave stores (register pressure control)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ Adaptive prefetch (NTA for streaming, T0 for reuse)
 * ✅ No-unroll pragmas (preserve instruction scheduling)
 * ✅ Fast W8 micro-kernels (add/sub, no cmul)
 * ✅ zeroupper after NT stores
 *
 * @note Requires AVX2 + FMA3 support (Haswell+, Zen1+)
 * @note Include fft_radix8_avx2_blocked_hybrid_fixed.h first for shared primitives
 */

#ifndef FFT_RADIX8_AVX2_N1_H
#define FFT_RADIX8_AVX2_N1_H

#include "fft_radix8_avx2_blocked_hybrid_fixed.h"

/* N1 prefetch distance: larger than twiddled since no twiddle loads compete */
#ifndef RADIX8_PREFETCH_DISTANCE_AVX2_N1
#define RADIX8_PREFETCH_DISTANCE_AVX2_N1 32
#endif

/*============================================================================
 * N1 FORWARD STAGE DRIVER
 *
 * Pure radix-8 DIT butterfly with no stage twiddle multiplication.
 * Split decomposition: even radix-4 (x0,x2,x4,x6) + odd radix-4 (x1,x3,x5,x7)
 * then W8 on odd outputs, combine via add/sub.
 *============================================================================*/
TARGET_AVX2_FMA
__attribute__((optimize("no-unroll-loops")))
static void
radix8_stage_n1_forward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_N1;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    /*======================================================================
     * PROLOGUE: Load first iteration inputs (no twiddles to load)
     *======================================================================*/
    __m256d nx0r = LDPD(&in_re[0*K]); __m256d nx0i = LDPD(&in_im[0*K]);
    __m256d nx1r = LDPD(&in_re[1*K]); __m256d nx1i = LDPD(&in_im[1*K]);
    __m256d nx2r = LDPD(&in_re[2*K]); __m256d nx2i = LDPD(&in_im[2*K]);
    __m256d nx3r = LDPD(&in_re[3*K]); __m256d nx3i = LDPD(&in_im[3*K]);
    __m256d nx4r = LDPD(&in_re[4*K]); __m256d nx4i = LDPD(&in_im[4*K]);
    __m256d nx5r = LDPD(&in_re[5*K]); __m256d nx5i = LDPD(&in_im[5*K]);
    __m256d nx6r = LDPD(&in_re[6*K]); __m256d nx6i = LDPD(&in_im[6*K]);
    __m256d nx7r = LDPD(&in_re[7*K]); __m256d nx7i = LDPD(&in_im[7*K]);

    /*======================================================================
     * STEADY-STATE U=2 LOOP
     *======================================================================*/
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4) {
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m256d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m256d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        const size_t kn = k + 4;

        /* Even radix-4: x0, x2, x4, x6 */
        __m256d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m256d SF = _mm256_set1_pd(-0.0);
            radix4_core_avx2(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                             &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i, SF);
        }

        /* Load next even inputs (overlap with compute) */
        nx0r = LDPD(&in_re[0*K+kn]); nx0i = LDPD(&in_im[0*K+kn]);
        nx2r = LDPD(&in_re[2*K+kn]); nx2i = LDPD(&in_im[2*K+kn]);
        nx4r = LDPD(&in_re[4*K+kn]); nx4i = LDPD(&in_im[4*K+kn]);
        nx6r = LDPD(&in_re[6*K+kn]); nx6i = LDPD(&in_im[6*K+kn]);

        /* Odd radix-4: x1, x3, x5, x7 */
        __m256d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m256d SF = _mm256_set1_pd(-0.0);
            radix4_core_avx2(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                             &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i, SF);
        }

        /* W8 twiddles on odd outputs (fast add/sub micro-kernel) */
        w8_apply_fast_forward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        /* Load next half-odd */
        nx1r = LDPD(&in_re[1*K+kn]); nx1i = LDPD(&in_im[1*K+kn]);
        nx3r = LDPD(&in_re[3*K+kn]); nx3i = LDPD(&in_im[3*K+kn]);

        /* Store Wave A: outputs 0-3 */
        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i, o1i));
        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i, o3i));

        /* Load remaining next odd (register pressure relieved by Wave A stores) */
        nx5r = LDPD(&in_re[5*K+kn]); nx5i = LDPD(&in_im[5*K+kn]);
        nx7r = LDPD(&in_re[7*K+kn]); nx7i = LDPD(&in_im[7*K+kn]);

        /* Store Wave B: outputs 4-7 */
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i, o3i));

        /* Prefetch: data only (no twiddle tables) */
        if (kn + prefetch_dist < K) {
            RADIX8_PF((const char *)&in_re[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_re[1*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[1*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_re[2*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[2*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_re[3*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[3*K+kn+prefetch_dist]);
        }
    }

    /*======================================================================
     * EPILOGUE: Final iteration (no next loads needed)
     *======================================================================*/
    {
        const size_t k = K - 4;
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m256d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m256d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;

        __m256d e0r,e0i, e1r,e1i, e2r,e2i, e3r,e3i;
        { const __m256d SF = _mm256_set1_pd(-0.0);
          radix4_core_avx2(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                           &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i, SF); }

        __m256d o0r,o0i, o1r,o1i, o2r,o2i, o3r,o3i;
        { const __m256d SF = _mm256_set1_pd(-0.0);
          radix4_core_avx2(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                           &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i, SF); }

        w8_apply_fast_forward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r,o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i,o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r,o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i,o1i));
        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r,o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i,o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r,o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i,o3i));
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r,o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i,o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r,o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i,o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r,o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i,o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r,o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i,o3i));
    }

    if (use_nt) { _mm_sfence(); _mm256_zeroupper(); }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

/*============================================================================
 * N1 BACKWARD STAGE DRIVER
 *
 * Same structure as forward but:
 * - radix4_core gets zero sign_mask (backward direction)
 * - Uses w8_apply_fast_backward_avx2 instead of forward
 *============================================================================*/
TARGET_AVX2_FMA
__attribute__((optimize("no-unroll-loops")))
static void
radix8_stage_n1_backward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_N1;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    /* Backward sign: XOR(-0.0, -0.0) = 0.0 → no flip in radix4_core */
    const __m256d ZERO = _mm256_setzero_pd();

    /*======================================================================
     * PROLOGUE
     *======================================================================*/
    __m256d nx0r = LDPD(&in_re[0*K]); __m256d nx0i = LDPD(&in_im[0*K]);
    __m256d nx1r = LDPD(&in_re[1*K]); __m256d nx1i = LDPD(&in_im[1*K]);
    __m256d nx2r = LDPD(&in_re[2*K]); __m256d nx2i = LDPD(&in_im[2*K]);
    __m256d nx3r = LDPD(&in_re[3*K]); __m256d nx3i = LDPD(&in_im[3*K]);
    __m256d nx4r = LDPD(&in_re[4*K]); __m256d nx4i = LDPD(&in_im[4*K]);
    __m256d nx5r = LDPD(&in_re[5*K]); __m256d nx5i = LDPD(&in_im[5*K]);
    __m256d nx6r = LDPD(&in_re[6*K]); __m256d nx6i = LDPD(&in_im[6*K]);
    __m256d nx7r = LDPD(&in_re[7*K]); __m256d nx7i = LDPD(&in_im[7*K]);

    /*======================================================================
     * STEADY-STATE U=2 LOOP
     *======================================================================*/
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4) {
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m256d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m256d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        const size_t kn = k + 4;

        /* Even radix-4 (backward: zero sign_mask) */
        __m256d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        radix4_core_avx2(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                         &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i, ZERO);

        /* Load next even inputs */
        nx0r = LDPD(&in_re[0*K+kn]); nx0i = LDPD(&in_im[0*K+kn]);
        nx2r = LDPD(&in_re[2*K+kn]); nx2i = LDPD(&in_im[2*K+kn]);
        nx4r = LDPD(&in_re[4*K+kn]); nx4i = LDPD(&in_im[4*K+kn]);
        nx6r = LDPD(&in_re[6*K+kn]); nx6i = LDPD(&in_im[6*K+kn]);

        /* Odd radix-4 (backward) */
        __m256d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        radix4_core_avx2(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                         &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i, ZERO);

        /* W8 backward twiddles */
        w8_apply_fast_backward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        /* Load next half-odd */
        nx1r = LDPD(&in_re[1*K+kn]); nx1i = LDPD(&in_im[1*K+kn]);
        nx3r = LDPD(&in_re[3*K+kn]); nx3i = LDPD(&in_im[3*K+kn]);

        /* Store Wave A */
        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i, o1i));
        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i, o3i));

        /* Load remaining next odd */
        nx5r = LDPD(&in_re[5*K+kn]); nx5i = LDPD(&in_im[5*K+kn]);
        nx7r = LDPD(&in_re[7*K+kn]); nx7i = LDPD(&in_im[7*K+kn]);

        /* Store Wave B */
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i, o3i));

        /* Prefetch: data only */
        if (kn + prefetch_dist < K) {
            RADIX8_PF((const char *)&in_re[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_re[1*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[1*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_re[2*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[2*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_re[3*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[3*K+kn+prefetch_dist]);
        }
    }

    /*======================================================================
     * EPILOGUE
     *======================================================================*/
    {
        const size_t k = K - 4;
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m256d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m256d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;

        __m256d e0r,e0i, e1r,e1i, e2r,e2i, e3r,e3i;
        radix4_core_avx2(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                         &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i, ZERO);

        __m256d o0r,o0i, o1r,o1i, o2r,o2i, o3r,o3i;
        radix4_core_avx2(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                         &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i, ZERO);

        w8_apply_fast_backward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r,o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i,o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r,o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i,o1i));
        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r,o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i,o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r,o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i,o3i));
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r,o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i,o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r,o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i,o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r,o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i,o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r,o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i,o3i));
    }

    if (use_nt) { _mm_sfence(); _mm256_zeroupper(); }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

#endif /* FFT_RADIX8_AVX2_N1_H */