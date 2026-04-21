/* Auto-generated per-ISA candidate fragment. DO NOT EDIT. */
#include <stddef.h>
#include "fft_radix12_avx512.h"

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

typedef struct {
  const char *variant, *isa, *protocol;
  t1_fn fwd, bwd;
  int requires_avx512;
} candidate_t;

const candidate_t CANDIDATES_AVX512[] = {
  {"ct_t1_dit", "avx512", "flat", radix12_t1_dit_fwd_avx512, radix12_t1_dit_bwd_avx512, 1},
  {"ct_t1s_dit", "avx512", "t1s", radix12_t1s_dit_fwd_avx512, radix12_t1s_dit_bwd_avx512, 1},
  {"ct_t1_dit_log3", "avx512", "log3", radix12_t1_dit_log3_fwd_avx512, radix12_t1_dit_log3_bwd_avx512, 1},
  {"ct_t1_dit_u2a", "avx512", "flat", radix12_t1_dit_u2a_fwd_avx512, radix12_t1_dit_u2a_bwd_avx512, 1},
  {"ct_t1_dit_u2b", "avx512", "flat", radix12_t1_dit_u2b_fwd_avx512, radix12_t1_dit_u2b_bwd_avx512, 1},
  {"ct_t1_dit_log3_u2a", "avx512", "log3", radix12_t1_dit_log3_u2a_fwd_avx512, radix12_t1_dit_log3_u2a_bwd_avx512, 1},
  {"ct_t1_dit_log3_u2b", "avx512", "log3", radix12_t1_dit_log3_u2b_fwd_avx512, radix12_t1_dit_log3_u2b_bwd_avx512, 1},
};
const size_t N_CANDIDATES_AVX512 = sizeof(CANDIDATES_AVX512) / sizeof(CANDIDATES_AVX512[0]);
