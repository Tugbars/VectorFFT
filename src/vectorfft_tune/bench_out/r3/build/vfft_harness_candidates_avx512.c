/* Auto-generated per-ISA candidate fragment. DO NOT EDIT. */
#include <stddef.h>
#include "fft_radix3_avx512.h"

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

typedef struct {
  const char *variant, *isa, *protocol;
  t1_fn fwd, bwd;
  int requires_avx512;
} candidate_t;

const candidate_t CANDIDATES_AVX512[] = {
  {"ct_t1_dit_u1", "avx512", "flat", radix3_t1_dit_u1_fwd_avx512, radix3_t1_dit_u1_bwd_avx512, 1},
  {"ct_t1_dit_u2", "avx512", "flat", radix3_t1_dit_u2_fwd_avx512, radix3_t1_dit_u2_bwd_avx512, 1},
  {"ct_t1_dit_u3", "avx512", "flat", radix3_t1_dit_u3_fwd_avx512, radix3_t1_dit_u3_bwd_avx512, 1},
  {"ct_t1s_dit_u1", "avx512", "t1s", radix3_t1s_dit_u1_fwd_avx512, radix3_t1s_dit_u1_bwd_avx512, 1},
  {"ct_t1s_dit_u2", "avx512", "t1s", radix3_t1s_dit_u2_fwd_avx512, radix3_t1s_dit_u2_bwd_avx512, 1},
  {"ct_t1s_dit_u3", "avx512", "t1s", radix3_t1s_dit_u3_fwd_avx512, radix3_t1s_dit_u3_bwd_avx512, 1},
  {"ct_t1_dit_log3_u1", "avx512", "log3", radix3_t1_dit_log3_u1_fwd_avx512, radix3_t1_dit_log3_u1_bwd_avx512, 1},
  {"ct_t1_dit_log3_u2", "avx512", "log3", radix3_t1_dit_log3_u2_fwd_avx512, radix3_t1_dit_log3_u2_bwd_avx512, 1},
  {"ct_t1_dit_log3_u3", "avx512", "log3", radix3_t1_dit_log3_u3_fwd_avx512, radix3_t1_dit_log3_u3_bwd_avx512, 1},
};
const size_t N_CANDIDATES_AVX512 = sizeof(CANDIDATES_AVX512) / sizeof(CANDIDATES_AVX512[0]);
