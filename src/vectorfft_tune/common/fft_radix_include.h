/* fft_radix_include.h — compile-time selection of which ISA headers
   the validator pulls in. The driver sets VALIDATE_AVX2 and/or
   VALIDATE_AVX512 when compiling. */
#ifndef VFFT_VALIDATE_INCLUDE_H
#define VFFT_VALIDATE_INCLUDE_H

#if RADIX == 4
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
#else
  #error "validator shim does not yet cover this radix"
#endif

#endif
