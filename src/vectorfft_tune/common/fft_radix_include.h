/* fft_radix_include.h — compile-time selection of which ISA headers
   the validator pulls in. The driver sets VALIDATE_AVX2 and/or
   VALIDATE_AVX512 when compiling. */
#ifndef VFFT_VALIDATE_INCLUDE_H
#define VFFT_VALIDATE_INCLUDE_H

#if RADIX == 3
  #if defined(VALIDATE_AVX2)
    #include "fft_radix3_avx2.h"
    #include "vfft_r3_t1_dit_dispatch_avx2.h"
    #include "vfft_r3_t1s_dit_dispatch_avx2.h"
    #include "vfft_r3_t1_dit_log3_dispatch_avx2.h"
  #endif
  #if defined(VALIDATE_AVX512)
    #include "fft_radix3_avx512.h"
    #include "vfft_r3_t1_dit_dispatch_avx512.h"
    #include "vfft_r3_t1s_dit_dispatch_avx512.h"
    #include "vfft_r3_t1_dit_log3_dispatch_avx512.h"
  #endif
  #include "vfft_r3_plan_wisdom.h"
#elif RADIX == 4
  #if defined(VALIDATE_AVX2)
    #include "fft_radix4_avx2.h"
    #include "vfft_r4_t1_dit_dispatch_avx2.h"
    #include "vfft_r4_t1_dit_log3_dispatch_avx2.h"
    #include "vfft_r4_t1s_dit_dispatch_avx2.h"
  #endif
  #if defined(VALIDATE_AVX512)
    #include "fft_radix4_avx512.h"
    #include "vfft_r4_t1_dit_dispatch_avx512.h"
    #include "vfft_r4_t1_dit_log3_dispatch_avx512.h"
    #include "vfft_r4_t1s_dit_dispatch_avx512.h"
  #endif
  #include "vfft_r4_plan_wisdom.h"
#elif RADIX == 8
  #if defined(VALIDATE_AVX2)
    #include "fft_radix8_avx2.h"
    #include "vfft_r8_t1_dit_dispatch_avx2.h"
    #include "vfft_r8_t1_dif_dispatch_avx2.h"
    #include "vfft_r8_t1_dit_log3_dispatch_avx2.h"
    #include "vfft_r8_t1s_dit_dispatch_avx2.h"
  #endif
  #if defined(VALIDATE_AVX512)
    #include "fft_radix8_avx512.h"
    #include "vfft_r8_t1_dit_dispatch_avx512.h"
    #include "vfft_r8_t1_dif_dispatch_avx512.h"
    #include "vfft_r8_t1_dit_log3_dispatch_avx512.h"
    #include "vfft_r8_t1s_dit_dispatch_avx512.h"
  #endif
  #include "vfft_r8_plan_wisdom.h"
#elif RADIX == 16
  #if defined(VALIDATE_AVX2)
    #include "fft_radix16_avx2.h"
    #include "vfft_r16_t1_dit_dispatch_avx2.h"
    #include "vfft_r16_t1_dif_dispatch_avx2.h"
    #include "vfft_r16_t1_dit_log3_dispatch_avx2.h"
    #include "vfft_r16_t1s_dit_dispatch_avx2.h"
    #include "vfft_r16_t1_buf_dit_dispatch_avx2.h"
  #endif
  #if defined(VALIDATE_AVX512)
    #include "fft_radix16_avx512.h"
    #include "vfft_r16_t1_dit_dispatch_avx512.h"
    #include "vfft_r16_t1_dif_dispatch_avx512.h"
    #include "vfft_r16_t1_dit_log3_dispatch_avx512.h"
    #include "vfft_r16_t1s_dit_dispatch_avx512.h"
    #include "vfft_r16_t1_buf_dit_dispatch_avx512.h"
  #endif
  #include "vfft_r16_plan_wisdom.h"
#elif RADIX == 32
  #if defined(VALIDATE_AVX2)
    #include "fft_radix32_avx2.h"
    #include "vfft_r32_t1_dit_dispatch_avx2.h"
    #include "vfft_r32_t1_dif_dispatch_avx2.h"
    #include "vfft_r32_t1_dit_log3_dispatch_avx2.h"
    #include "vfft_r32_t1s_dit_dispatch_avx2.h"
    #include "vfft_r32_t1_buf_dit_dispatch_avx2.h"
  #endif
  #if defined(VALIDATE_AVX512)
    #include "fft_radix32_avx512.h"
    #include "vfft_r32_t1_dit_dispatch_avx512.h"
    #include "vfft_r32_t1_dif_dispatch_avx512.h"
    #include "vfft_r32_t1_dit_log3_dispatch_avx512.h"
    #include "vfft_r32_t1s_dit_dispatch_avx512.h"
    #include "vfft_r32_t1_buf_dit_dispatch_avx512.h"
  #endif
  #include "vfft_r32_plan_wisdom.h"
#elif RADIX == 64
  #if defined(VALIDATE_AVX2)
    #include "fft_radix64_avx2.h"
    #include "vfft_r64_t1_dit_dispatch_avx2.h"
    #include "vfft_r64_t1_dif_dispatch_avx2.h"
    #include "vfft_r64_t1_dit_log3_dispatch_avx2.h"
    #include "vfft_r64_t1s_dit_dispatch_avx2.h"
    #include "vfft_r64_t1_buf_dit_dispatch_avx2.h"
  #endif
  #if defined(VALIDATE_AVX512)
    #include "fft_radix64_avx512.h"
    #include "vfft_r64_t1_dit_dispatch_avx512.h"
    #include "vfft_r64_t1_dif_dispatch_avx512.h"
    #include "vfft_r64_t1_dit_log3_dispatch_avx512.h"
    #include "vfft_r64_t1s_dit_dispatch_avx512.h"
    #include "vfft_r64_t1_buf_dit_dispatch_avx512.h"
  #endif
  #include "vfft_r64_plan_wisdom.h"
#elif RADIX == 5
  #if defined(VALIDATE_AVX2)
    #include "fft_radix5_avx2.h"
    #include "vfft_r5_t1_dit_dispatch_avx2.h"
    #include "vfft_r5_t1s_dit_dispatch_avx2.h"
    #include "vfft_r5_t1_dit_log3_dispatch_avx2.h"
  #endif
  #if defined(VALIDATE_AVX512)
    #include "fft_radix5_avx512.h"
    #include "vfft_r5_t1_dit_dispatch_avx512.h"
    #include "vfft_r5_t1s_dit_dispatch_avx512.h"
    #include "vfft_r5_t1_dit_log3_dispatch_avx512.h"
  #endif
  #include "vfft_r5_plan_wisdom.h"
#elif RADIX == 7
  #if defined(VALIDATE_AVX2)
    #include "fft_radix7_avx2.h"
    #include "vfft_r7_t1_dit_dispatch_avx2.h"
    #include "vfft_r7_t1s_dit_dispatch_avx2.h"
    #include "vfft_r7_t1_dit_log3_dispatch_avx2.h"
  #endif
  #if defined(VALIDATE_AVX512)
    #include "fft_radix7_avx512.h"
    #include "vfft_r7_t1_dit_dispatch_avx512.h"
    #include "vfft_r7_t1s_dit_dispatch_avx512.h"
    #include "vfft_r7_t1_dit_log3_dispatch_avx512.h"
  #endif
  #include "vfft_r7_plan_wisdom.h"
#elif RADIX == 10
  #if defined(VALIDATE_AVX2)
    #include "fft_radix10_avx2.h"
    #include "vfft_r10_t1_dit_dispatch_avx2.h"
    #include "vfft_r10_t1s_dit_dispatch_avx2.h"
    #include "vfft_r10_t1_dit_log3_dispatch_avx2.h"
  #endif
  #if defined(VALIDATE_AVX512)
    #include "fft_radix10_avx512.h"
    #include "vfft_r10_t1_dit_dispatch_avx512.h"
    #include "vfft_r10_t1s_dit_dispatch_avx512.h"
    #include "vfft_r10_t1_dit_log3_dispatch_avx512.h"
  #endif
  #include "vfft_r10_plan_wisdom.h"
#elif RADIX == 20
  #if defined(VALIDATE_AVX2)
    #include "fft_radix20_avx2.h"
    #include "vfft_r20_t1_dit_dispatch_avx2.h"
    #include "vfft_r20_t1s_dit_dispatch_avx2.h"
    #include "vfft_r20_t1_dit_log3_dispatch_avx2.h"
  #endif
  #if defined(VALIDATE_AVX512)
    #include "fft_radix20_avx512.h"
    #include "vfft_r20_t1_dit_dispatch_avx512.h"
    #include "vfft_r20_t1s_dit_dispatch_avx512.h"
    #include "vfft_r20_t1_dit_log3_dispatch_avx512.h"
  #endif
  #include "vfft_r20_plan_wisdom.h"
#elif RADIX == 25
  #if defined(VALIDATE_AVX2)
    #include "fft_radix25_avx2.h"
    #include "vfft_r25_t1_dit_dispatch_avx2.h"
    #include "vfft_r25_t1s_dit_dispatch_avx2.h"
    #include "vfft_r25_t1_dit_log3_dispatch_avx2.h"
  #endif
  #if defined(VALIDATE_AVX512)
    #include "fft_radix25_avx512.h"
    #include "vfft_r25_t1_dit_dispatch_avx512.h"
    #include "vfft_r25_t1s_dit_dispatch_avx512.h"
    #include "vfft_r25_t1_dit_log3_dispatch_avx512.h"
  #endif
  #include "vfft_r25_plan_wisdom.h"
#else
  #error "validator shim does not yet cover this radix"
#endif

#endif
