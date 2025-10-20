//==============================================================================
// fft_radix16_bv.c - Inverse Radix-16 Butterfly (Optimized)
//==============================================================================

#include "fft_radix16.h"
#include "simd_math.h"
#include "fft_radix16_macros.h"

void fft_radix16_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int sub_len)
{
    output_buffer = __builtin_assume_aligned(output_buffer, 32);
    sub_outputs = __builtin_assume_aligned(sub_outputs, 32);
    stage_tw = __builtin_assume_aligned(stage_tw, 32);
    
    const int K = sub_len;
    int k = 0;
    const int use_streaming = (K >= STREAM_THRESHOLD_R16);

#ifdef __AVX512F__
    const __m512d rot_mask = _mm512_set_pd(0.0, -0.0, 0.0, -0.0,
                                           0.0, -0.0, 0.0, -0.0);
    const __m512d neg_mask = _mm512_set1_pd(-0.0);
    
    for (; k + 3 < K; k += 4)
    {
        PREFETCH_16_LANES_AVX512(k, K, PREFETCH_L1_AVX512, sub_outputs, _MM_HINT_T0);
        PREFETCH_STAGE_TW_AVX512(k, PREFETCH_L1_AVX512, stage_tw);
        
        if (use_streaming) {
            RADIX16_PIPELINE_4_BV_AVX512_STREAM(k, K, sub_outputs, stage_tw, output_buffer, rot_mask, neg_mask);
        } else {
            RADIX16_PIPELINE_4_BV_AVX512(k, K, sub_outputs, stage_tw, output_buffer, rot_mask, neg_mask);
        }
    }
    
    if (use_streaming) {
        _mm_sfence();
    }
#endif

#ifdef __AVX2__
    const __m256d rot_mask = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
    const __m256d neg_mask = _mm256_set1_pd(-0.0);
    
    for (; k + 1 < K; k += 2)
    {
        PREFETCH_16_LANES(k, K, PREFETCH_L1, sub_outputs, _MM_HINT_T0);
        PREFETCH_STAGE_TW_AVX2(k, PREFETCH_L1, stage_tw);
        
        __m256d x[16];
        LOAD_16_LANES_AVX2(k, K, sub_outputs, x);
        APPLY_STAGE_TWIDDLES_R16_AVX2(k, x, stage_tw);
        
        __m256d y[16];
        RADIX4_BUTTERFLY_AVX2(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask);
        RADIX4_BUTTERFLY_AVX2(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask);
        RADIX4_BUTTERFLY_AVX2(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask);
        RADIX4_BUTTERFLY_AVX2(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask);
        
        APPLY_W4_INTERMEDIATE_BV_AVX2_HOISTED(y, neg_mask);
        
        __m256d temp[4];
        for (int m = 0; m < 4; m++)
        {
            RADIX4_BUTTERFLY_AVX2(y[m], y[m + 4], y[m + 8], y[m + 12],
                                 temp[0], temp[1], temp[2], temp[3], rot_mask);
            y[m] = temp[0];
            y[m + 4] = temp[1];
            y[m + 8] = temp[2];
            y[m + 12] = temp[3];
        }
        
        if (use_streaming) {
            STORE_16_LANES_AVX2_STREAM(k, K, output_buffer, y);
        } else {
            STORE_16_LANES_AVX2(k, K, output_buffer, y);
        }
    }
    
    if (use_streaming) {
        _mm_sfence();
    }
#endif

    for (; k < K; k++)
    {
        fft_data x[16];
        for (int lane = 0; lane < 16; lane++) {
            x[lane] = sub_outputs[k + lane * K];
        }
        
        APPLY_STAGE_TWIDDLES_R16_SCALAR(k, x, stage_tw);
        
        fft_data y[16];
        RADIX4_BUTTERFLY_SCALAR(
            x[0].re, x[0].im, x[4].re, x[4].im, x[8].re, x[8].im, x[12].re, x[12].im,
            y[0].re, y[0].im, y[1].re, y[1].im, y[2].re, y[2].im, y[3].re, y[3].im, +1);
        RADIX4_BUTTERFLY_SCALAR(
            x[1].re, x[1].im, x[5].re, x[5].im, x[9].re, x[9].im, x[13].re, x[13].im,
            y[4].re, y[4].im, y[5].re, y[5].im, y[6].re, y[6].im, y[7].re, y[7].im, +1);
        RADIX4_BUTTERFLY_SCALAR(
            x[2].re, x[2].im, x[6].re, x[6].im, x[10].re, x[10].im, x[14].re, x[14].im,
            y[8].re, y[8].im, y[9].re, y[9].im, y[10].re, y[10].im, y[11].re, y[11].im, +1);
        RADIX4_BUTTERFLY_SCALAR(
            x[3].re, x[3].im, x[7].re, x[7].im, x[11].re, x[11].im, x[15].re, x[15].im,
            y[12].re, y[12].im, y[13].re, y[13].im, y[14].re, y[14].im, y[15].re, y[15].im, +1);
        
        APPLY_W4_INTERMEDIATE_BV_SCALAR(y);
        
        fft_data z[16];
        for (int m = 0; m < 4; m++)
        {
            RADIX4_BUTTERFLY_SCALAR(
                y[m].re, y[m].im, y[m + 4].re, y[m + 4].im,
                y[m + 8].re, y[m + 8].im, y[m + 12].re, y[m + 12].im,
                z[m].re, z[m].im, z[m + 4].re, z[m + 4].im,
                z[m + 8].re, z[m + 8].im, z[m + 12].re, z[m + 12].im, +1);
        }
        
        for (int lane = 0; lane < 16; lane++) {
            output_buffer[k + lane * K] = z[lane];
        }
    }
}