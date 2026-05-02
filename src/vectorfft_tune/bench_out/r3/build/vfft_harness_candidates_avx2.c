/* Auto-generated per-ISA candidate fragment. DO NOT EDIT. */
#include <stddef.h>
#include "fft_radix3_avx2.h"

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

typedef struct {
  const char *variant, *isa, *protocol;
  t1_fn fwd, bwd;
  int requires_avx512;
} candidate_t;

const candidate_t CANDIDATES_AVX2[] = {
  {"ct_t1_dit_u1", "avx2", "flat", radix3_t1_dit_u1_fwd_avx2, radix3_t1_dit_u1_bwd_avx2, 0},
  {"ct_t1s_dit_u1", "avx2", "t1s", radix3_t1s_dit_u1_fwd_avx2, radix3_t1s_dit_u1_bwd_avx2, 0},
  {"ct_t1_dit_log3_u1", "avx2", "log3", radix3_t1_dit_log3_u1_fwd_avx2, radix3_t1_dit_log3_u1_bwd_avx2, 0},
};
const size_t N_CANDIDATES_AVX2 = sizeof(CANDIDATES_AVX2) / sizeof(CANDIDATES_AVX2[0]);
