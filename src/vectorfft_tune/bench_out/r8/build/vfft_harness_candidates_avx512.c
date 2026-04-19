/* Auto-generated per-ISA candidate fragment. DO NOT EDIT. */
#include <stddef.h>
#include "fft_radix8_avx512.h"

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

typedef struct {
  const char *variant, *isa, *protocol;
  t1_fn fwd, bwd;
  int requires_avx512;
} candidate_t;

const candidate_t CANDIDATES_AVX512[] = {
  {"ct_t1_dit", "avx512", "flat", radix8_t1_dit_fwd_avx512, radix8_t1_dit_bwd_avx512, 1},
  {"ct_t1_dit_prefetch", "avx512", "flat", radix8_t1_dit_prefetch_fwd_avx512, radix8_t1_dit_prefetch_bwd_avx512, 1},
  {"ct_t1_dif", "avx512", "flat", radix8_t1_dif_fwd_avx512, radix8_t1_dif_bwd_avx512, 1},
};
const size_t N_CANDIDATES_AVX512 = sizeof(CANDIDATES_AVX512) / sizeof(CANDIDATES_AVX512[0]);
