PREFETCH USAGE NOTE
===================

All pipeline macros now include a prefetch_dist parameter.

MACRO SIGNATURES (UPDATED):
---------------------------

AVX-512:
  RADIX2_PIPELINE_8_NATIVE_SOA_AVX512(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist)
  RADIX2_PIPELINE_8_NATIVE_SOA_AVX512_STREAM(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist)

AVX2:
  RADIX2_PIPELINE_4_NATIVE_SOA_AVX2(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist)
  RADIX2_PIPELINE_4_NATIVE_SOA_AVX2_STREAM(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist)

SSE2:
  RADIX2_PIPELINE_2_NATIVE_SOA_SSE2(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist)
  RADIX2_PIPELINE_2_NATIVE_SOA_SSE2_STREAM(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist)


USAGE IN PROCESSING LOOPS:
--------------------------

In your radix2_process_range_native_soa() function:

    // Configure prefetch distance (compile-time tunable)
    const int prefetch_dist = RADIX2_PREFETCH_DISTANCE;  // Default: 24 elements
    
    // Use in loops:
    #ifdef __AVX512F__
        for (int k = k_start; k + 7 < k_end; k += 8) {
            RADIX2_PIPELINE_8_NATIVE_SOA_AVX512(k, in_re, in_im, out_re, out_im,
                                               stage_tw, half, prefetch_dist);
        }
    #endif


TUNING PREFETCH DISTANCE:
-------------------------

Compile with custom prefetch distance:
    gcc -DRADIX2_PREFETCH_DISTANCE=32 ...

Guidelines:
  - Memory-bound workloads: Increase to 32-48
  - Cache-resident workloads: Decrease to 16 or disable (0)
  - Small N (<1K): Set to 0 (disable)
  - Large N (>16K): Set to 24-32

Prefetch targets:
  - in_re[k+dist], in_im[k+dist]
  - in_re[k+half+dist], in_im[k+half+dist]
  - stage_tw->re[k+dist], stage_tw->im[k+dist]

All prefetch uses _MM_HINT_T0 (L1 cache).