/**
 * @file fft_radix8_avx512_n1.h
 * @brief Twiddle-less Radix-8 AVX-512 Stage Drivers (N1 variant)
 *
 * For the first stage of a mixed-radix FFT where all stage twiddles are unity.
 * Zero twiddle loads, zero twiddle multiplications.
 *
 * Optimizations:
 * ✅ U=2 software pipelining (8-wide vectors, stride 8)
 * ✅ Two-wave stores (register pressure control)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ Adaptive prefetch NTA/T0
 * ✅ No-unroll pragmas
 * ✅ Fast W8 micro-kernels (add/sub, no cmul)
 *
 * @note Requires AVX-512F + AVX-512DQ + FMA
 * @note K must be multiple of 8, minimum 16
 * @note Include fft_radix8_avx512_blocked_hybrid_fixed.h first for shared primitives
 */

#ifndef FFT_RADIX8_AVX512_N1_H
#define FFT_RADIX8_AVX512_N1_H

#include "fft_radix8_avx512_blocked_hybrid_fixed.h"

/*============================================================================
 * N1 FORWARD (AVX-512) - Twiddle-less first/last stage
 *============================================================================*/
TARGET_AVX512
__attribute__((optimize("no-unroll-loops")))
static void
radix8_stage_n1_forward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define ST(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX512_B4; /* reuse B4 distance */
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;

#define STS(p, v) (use_nt ? _mm512_stream_pd(p, v) : ST(p, v))

    /* PROLOGUE */
    __m512d nx0r=LD(&in_re[0*K]), nx0i=LD(&in_im[0*K]);
    __m512d nx1r=LD(&in_re[1*K]), nx1i=LD(&in_im[1*K]);
    __m512d nx2r=LD(&in_re[2*K]), nx2i=LD(&in_im[2*K]);
    __m512d nx3r=LD(&in_re[3*K]), nx3i=LD(&in_im[3*K]);
    __m512d nx4r=LD(&in_re[4*K]), nx4i=LD(&in_im[4*K]);
    __m512d nx5r=LD(&in_re[5*K]), nx5i=LD(&in_im[5*K]);
    __m512d nx6r=LD(&in_re[6*K]), nx6i=LD(&in_im[6*K]);
    __m512d nx7r=LD(&in_re[7*K]), nx7i=LD(&in_im[7*K]);

#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 8 < K; k += 8) {
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        const size_t kn = k + 8;

        __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                             &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, SF); }

        nx0r=LD(&in_re[0*K+kn]); nx0i=LD(&in_im[0*K+kn]);
        nx2r=LD(&in_re[2*K+kn]); nx2i=LD(&in_im[2*K+kn]);
        nx4r=LD(&in_re[4*K+kn]); nx4i=LD(&in_im[4*K+kn]);
        nx6r=LD(&in_re[6*K+kn]); nx6i=LD(&in_im[6*K+kn]);

        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                             &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, SF); }

        w8_apply_fast_forward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        nx1r=LD(&in_re[1*K+kn]); nx1i=LD(&in_im[1*K+kn]);
        nx3r=LD(&in_re[3*K+kn]); nx3i=LD(&in_im[3*K+kn]);

        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r)); STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r)); STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));
        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r)); STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r)); STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));

        nx5r=LD(&in_re[5*K+kn]); nx5i=LD(&in_im[5*K+kn]);
        nx7r=LD(&in_re[7*K+kn]); nx7i=LD(&in_im[7*K+kn]);

        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r)); STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r)); STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r)); STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r)); STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));

        if (kn + prefetch_dist < K) {
            RADIX8_PF512((const char *)&in_re[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&in_im[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&in_re[4*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&in_im[4*K+kn+prefetch_dist]);
        }
    }

    /* EPILOGUE */
    {
        size_t k = K - 8;
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;

        __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                             &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, SF); }
        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                             &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, SF); }
        w8_apply_fast_forward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r)); STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r)); STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));
        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r)); STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r)); STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));
        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r)); STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r)); STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r)); STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r)); STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));
    }

    if (use_nt) _mm_sfence();

#undef LD
#undef ST
#undef STS
}

/*============================================================================
 * N1 BACKWARD (AVX-512) - Twiddle-less first/last stage
 *============================================================================*/
TARGET_AVX512
__attribute__((optimize("no-unroll-loops")))
static void
radix8_stage_n1_backward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define ST(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX512_B4;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;

#define STS(p, v) (use_nt ? _mm512_stream_pd(p, v) : ST(p, v))

    const __m512d ZERO = _mm512_setzero_pd();

    /* PROLOGUE */
    __m512d nx0r=LD(&in_re[0*K]), nx0i=LD(&in_im[0*K]);
    __m512d nx1r=LD(&in_re[1*K]), nx1i=LD(&in_im[1*K]);
    __m512d nx2r=LD(&in_re[2*K]), nx2i=LD(&in_im[2*K]);
    __m512d nx3r=LD(&in_re[3*K]), nx3i=LD(&in_im[3*K]);
    __m512d nx4r=LD(&in_re[4*K]), nx4i=LD(&in_im[4*K]);
    __m512d nx5r=LD(&in_re[5*K]), nx5i=LD(&in_im[5*K]);
    __m512d nx6r=LD(&in_re[6*K]), nx6i=LD(&in_im[6*K]);
    __m512d nx7r=LD(&in_re[7*K]), nx7i=LD(&in_im[7*K]);

#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 8 < K; k += 8) {
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        const size_t kn = k + 8;

        __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        radix4_core_avx512(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                           &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, ZERO);

        nx0r=LD(&in_re[0*K+kn]); nx0i=LD(&in_im[0*K+kn]);
        nx2r=LD(&in_re[2*K+kn]); nx2i=LD(&in_im[2*K+kn]);
        nx4r=LD(&in_re[4*K+kn]); nx4i=LD(&in_im[4*K+kn]);
        nx6r=LD(&in_re[6*K+kn]); nx6i=LD(&in_im[6*K+kn]);

        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        radix4_core_avx512(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                           &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, ZERO);

        w8_apply_fast_backward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        nx1r=LD(&in_re[1*K+kn]); nx1i=LD(&in_im[1*K+kn]);
        nx3r=LD(&in_re[3*K+kn]); nx3i=LD(&in_im[3*K+kn]);

        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r)); STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r)); STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));
        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r)); STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r)); STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));

        nx5r=LD(&in_re[5*K+kn]); nx5i=LD(&in_im[5*K+kn]);
        nx7r=LD(&in_re[7*K+kn]); nx7i=LD(&in_im[7*K+kn]);

        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r)); STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r)); STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r)); STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r)); STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));

        if (kn + prefetch_dist < K) {
            RADIX8_PF512((const char *)&in_re[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&in_im[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&in_re[4*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&in_im[4*K+kn+prefetch_dist]);
        }
    }

    /* EPILOGUE */
    {
        size_t k = K - 8;
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;

        __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        radix4_core_avx512(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                           &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, ZERO);
        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        radix4_core_avx512(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                           &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, ZERO);
        w8_apply_fast_backward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r)); STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r)); STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));
        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r)); STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r)); STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));
        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r)); STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r)); STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r)); STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r)); STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));
    }

    if (use_nt) _mm_sfence();

#undef LD
#undef ST
#undef STS
}


#endif /* FFT_RADIX8_AVX512_N1_H */