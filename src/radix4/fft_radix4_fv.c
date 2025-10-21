//==============================================================================
// fft_radix4_fv.c - Forward Radix-4 Butterfly (SPLIT-FORM OPTIMIZED)
//==============================================================================

#include "fft_radix4.h"
#include "simd_math.h"
#include "fft_radix4_macros.h"

void fft_radix4_fv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_twiddles_soa *restrict stage_tw,
    int sub_len)
{
    // Alignment hints (data comes pre-aligned from planner)
    output_buffer = __builtin_assume_aligned(output_buffer, 32);
    sub_outputs = __builtin_assume_aligned(sub_outputs, 32);

    const int K = sub_len;
    int k = 0;

    //==========================================================================
    // DECIDE: Streaming vs Normal stores (OUTSIDE loop - no branch in hot path)
    //==========================================================================
    const int use_streaming = (K >= RADIX4_STREAM_THRESHOLD);

    //==========================================================================
    // AVX-512 PATH: 8 butterflies per iteration (unroll-by-2 for latency hiding)
    //==========================================================================
#ifdef __AVX512F__
    if (use_streaming)
    {
        // Streaming stores for large K (bypass cache)
        for (; k + 7 < K; k += 8)
        {
            RADIX4_PIPELINE_8_FV_AVX512_STREAM(k, K, sub_outputs, stage_tw, output_buffer);
        }
        _mm_sfence(); // Single fence after all streaming stores
    }
    else
    {
        // Normal stores for moderate K
        for (; k + 7 < K; k += 8)
        {
            RADIX4_PIPELINE_8_FV_AVX512_SPLIT(k, K, sub_outputs, stage_tw, output_buffer);
        }
    }

    // Handle remaining 4-butterfly groups
    if (use_streaming)
    {
        for (; k + 3 < K; k += 4)
        {
            // Fall back to single unit for tail (no unroll)
            __m512d a, b, c, d;
            LOAD_4_LANES_AVX512(k, K, sub_outputs, a, b, c, d);
            __m512d ar = split_re_avx512(a), ai = split_im_avx512(a);
            __m512d br = split_re_avx512(b), bi = split_im_avx512(b);
            __m512d cr = split_re_avx512(c), ci = split_im_avx512(c);
            __m512d dr = split_re_avx512(d), di = split_im_avx512(d);

            __m512d w1r = _mm512_loadu_pd(&stage_tw->re[0 * K + k]);
            __m512d w1i = _mm512_loadu_pd(&stage_tw->im[0 * K + k]);
            __m512d w2r = _mm512_loadu_pd(&stage_tw->re[1 * K + k]);
            __m512d w2i = _mm512_loadu_pd(&stage_tw->im[1 * K + k]);
            __m512d w3r = _mm512_loadu_pd(&stage_tw->re[2 * K + k]);
            __m512d w3i = _mm512_loadu_pd(&stage_tw->im[2 * K + k]);

            __m512d tBr, tBi, tCr, tCi, tDr, tDi;
            CMUL_SPLIT_AVX512(br, bi, w1r, w1i, tBr, tBi);
            CMUL_SPLIT_AVX512(cr, ci, w2r, w2i, tCr, tCi);
            CMUL_SPLIT_AVX512(dr, di, w3r, w3i, tDr, tDi);

            __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            RADIX4_BFLY_SPLIT_FV_AVX512(ar, ai, tBr, tBi, tCr, tCi, tDr, tDi,
                                        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i);

            _mm512_stream_pd(&output_buffer[k].re, join_ri_avx512(y0r, y0i));
            _mm512_stream_pd(&output_buffer[k + K].re, join_ri_avx512(y1r, y1i));
            _mm512_stream_pd(&output_buffer[k + 2 * K].re, join_ri_avx512(y2r, y2i));
            _mm512_stream_pd(&output_buffer[k + 3 * K].re, join_ri_avx512(y3r, y3i));
        }
        _mm_sfence();
    }
    else
    {
        for (; k + 3 < K; k += 4)
        {
            __m512d a, b, c, d;
            LOAD_4_LANES_AVX512(k, K, sub_outputs, a, b, c, d);
            __m512d ar = split_re_avx512(a), ai = split_im_avx512(a);
            __m512d br = split_re_avx512(b), bi = split_im_avx512(b);
            __m512d cr = split_re_avx512(c), ci = split_im_avx512(c);
            __m512d dr = split_re_avx512(d), di = split_im_avx512(d);

            __m512d w1r = _mm512_loadu_pd(&stage_tw->re[0 * K + k]);
            __m512d w1i = _mm512_loadu_pd(&stage_tw->im[0 * K + k]);
            __m512d w2r = _mm512_loadu_pd(&stage_tw->re[1 * K + k]);
            __m512d w2i = _mm512_loadu_pd(&stage_tw->im[1 * K + k]);
            __m512d w3r = _mm512_loadu_pd(&stage_tw->re[2 * K + k]);
            __m512d w3i = _mm512_loadu_pd(&stage_tw->im[2 * K + k]);

            __m512d tBr, tBi, tCr, tCi, tDr, tDi;
            CMUL_SPLIT_AVX512(br, bi, w1r, w1i, tBr, tBi);
            CMUL_SPLIT_AVX512(cr, ci, w2r, w2i, tCr, tCi);
            CMUL_SPLIT_AVX512(dr, di, w3r, w3i, tDr, tDi);

            __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            RADIX4_BFLY_SPLIT_FV_AVX512(ar, ai, tBr, tBi, tCr, tCi, tDr, tDi,
                                        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i);

            STOREU_PD512(&output_buffer[k].re, join_ri_avx512(y0r, y0i));
            STOREU_PD512(&output_buffer[k + K].re, join_ri_avx512(y1r, y1i));
            STOREU_PD512(&output_buffer[k + 2 * K].re, join_ri_avx512(y2r, y2i));
            STOREU_PD512(&output_buffer[k + 3 * K].re, join_ri_avx512(y3r, y3i));
        }
    }
#endif

    //==========================================================================
    // AVX2 PATH: 4 butterflies per iteration (unroll-by-2)
    //==========================================================================
#ifdef __AVX2__
    if (use_streaming)
    {
        for (; k + 3 < K; k += 4)
        {
            RADIX4_PIPELINE_4_FV_AVX2_STREAM(k, K, sub_outputs, stage_tw, output_buffer);
        }
        _mm_sfence();
    }
    else
    {
        for (; k + 3 < K; k += 4)
        {
            RADIX4_PIPELINE_4_FV_AVX2_SPLIT(k, K, sub_outputs, stage_tw, output_buffer);
        }
    }

    // Handle remaining 2-butterfly groups
    if (use_streaming)
    {
        for (; k + 1 < K; k += 2)
        {
            __m256d a, b, c, d;
            LOAD_4_LANES_AVX2(k, K, sub_outputs, a, b, c, d);
            __m256d ar = split_re_avx2(a), ai = split_im_avx2(a);
            __m256d br = split_re_avx2(b), bi = split_im_avx2(b);
            __m256d cr = split_re_avx2(c), ci = split_im_avx2(c);
            __m256d dr = split_re_avx2(d), di = split_im_avx2(d);

            __m256d w1r = _mm256_loadu_pd(&stage_tw->re[0 * K + k]);
            __m256d w1i = _mm256_loadu_pd(&stage_tw->im[0 * K + k]);
            __m256d w2r = _mm256_loadu_pd(&stage_tw->re[1 * K + k]);
            __m256d w2i = _mm256_loadu_pd(&stage_tw->im[1 * K + k]);
            __m256d w3r = _mm256_loadu_pd(&stage_tw->re[2 * K + k]);
            __m256d w3i = _mm256_loadu_pd(&stage_tw->im[2 * K + k]);

            __m256d tBr, tBi, tCr, tCi, tDr, tDi;
            CMUL_SPLIT_AVX2(br, bi, w1r, w1i, tBr, tBi);
            CMUL_SPLIT_AVX2(cr, ci, w2r, w2i, tCr, tCi);
            CMUL_SPLIT_AVX2(dr, di, w3r, w3i, tDr, tDi);

            __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            RADIX4_BFLY_SPLIT_FV_AVX2(ar, ai, tBr, tBi, tCr, tCi, tDr, tDi,
                                      y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i);

            _mm256_stream_pd(&output_buffer[k].re, join_ri_avx2(y0r, y0i));
            _mm256_stream_pd(&output_buffer[k + K].re, join_ri_avx2(y1r, y1i));
            _mm256_stream_pd(&output_buffer[k + 2 * K].re, join_ri_avx2(y2r, y2i));
            _mm256_stream_pd(&output_buffer[k + 3 * K].re, join_ri_avx2(y3r, y3i));
        }
        _mm_sfence();
    }
    else
    {
        for (; k + 1 < K; k += 2)
        {
            __m256d a, b, c, d;
            LOAD_4_LANES_AVX2(k, K, sub_outputs, a, b, c, d);
            __m256d ar = split_re_avx2(a), ai = split_im_avx2(a);
            __m256d br = split_re_avx2(b), bi = split_im_avx2(b);
            __m256d cr = split_re_avx2(c), ci = split_im_avx2(c);
            __m256d dr = split_re_avx2(d), di = split_im_avx2(d);

            __m256d w1r = _mm256_loadu_pd(&stage_tw->re[0 * K + k]);
            __m256d w1i = _mm256_loadu_pd(&stage_tw->im[0 * K + k]);
            __m256d w2r = _mm256_loadu_pd(&stage_tw->re[1 * K + k]);
            __m256d w2i = _mm256_loadu_pd(&stage_tw->im[1 * K + k]);
            __m256d w3r = _mm256_loadu_pd(&stage_tw->re[2 * K + k]);
            __m256d w3i = _mm256_loadu_pd(&stage_tw->im[2 * K + k]);

            __m256d tBr, tBi, tCr, tCi, tDr, tDi;
            CMUL_SPLIT_AVX2(br, bi, w1r, w1i, tBr, tBi);
            CMUL_SPLIT_AVX2(cr, ci, w2r, w2i, tCr, tCi);
            CMUL_SPLIT_AVX2(dr, di, w3r, w3i, tDr, tDi);

            __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            RADIX4_BFLY_SPLIT_FV_AVX2(ar, ai, tBr, tBi, tCr, tCi, tDr, tDi,
                                      y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i);

            STOREU_PD(&output_buffer[k].re, join_ri_avx2(y0r, y0i));
            STOREU_PD(&output_buffer[k + K].re, join_ri_avx2(y1r, y1i));
            STOREU_PD(&output_buffer[k + 2 * K].re, join_ri_avx2(y2r, y2i));
            STOREU_PD(&output_buffer[k + 3 * K].re, join_ri_avx2(y3r, y3i));
        }
    }
#endif

    //==========================================================================
    // SCALAR TAIL: Process remaining single butterflies
    //==========================================================================
    for (; k < K; k++)
    {
        RADIX4_BUTTERFLY_SCALAR_FV_SOA(k, K, sub_outputs, stage_tw, output_buffer);
    }
}