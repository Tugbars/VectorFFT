/* Auto-generated per-ISA candidate fragment. DO NOT EDIT. */
#include <stddef.h>
#include "fft_radix16_avx2.h"

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

typedef struct {
  const char *variant, *isa, *protocol;
  t1_fn fwd, bwd;
  int requires_avx512;
} candidate_t;

const candidate_t CANDIDATES_AVX2[] = {
  {"ct_t1_dit", "avx2", "flat", radix16_t1_dit_fwd_avx2, radix16_t1_dit_bwd_avx2, 0},
  {"ct_t1_dif", "avx2", "flat", radix16_t1_dif_fwd_avx2, radix16_t1_dif_bwd_avx2, 0},
  {"ct_t1s_dit", "avx2", "t1s", radix16_t1s_dit_fwd_avx2, radix16_t1s_dit_bwd_avx2, 0},
  {"ct_t1_dit_log3", "avx2", "log3", radix16_t1_dit_log3_fwd_avx2, radix16_t1_dit_log3_bwd_avx2, 0},
  {"ct_t1_dif_log3", "avx2", "log3", radix16_t1_dif_log3_fwd_avx2, radix16_t1_dif_log3_bwd_avx2, 0},
  {"ct_t1_buf_dit_tile64_temporal", "avx2", "flat", radix16_t1_buf_dit_tile64_temporal_fwd_avx2, radix16_t1_buf_dit_tile64_temporal_bwd_avx2, 0},
  {"ct_t1_buf_dit_tile128_temporal", "avx2", "flat", radix16_t1_buf_dit_tile128_temporal_fwd_avx2, radix16_t1_buf_dit_tile128_temporal_bwd_avx2, 0},
};
const size_t N_CANDIDATES_AVX2 = sizeof(CANDIDATES_AVX2) / sizeof(CANDIDATES_AVX2[0]);
