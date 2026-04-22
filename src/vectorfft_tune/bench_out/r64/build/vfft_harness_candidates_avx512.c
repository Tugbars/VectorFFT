/* Auto-generated per-ISA candidate fragment. DO NOT EDIT. */
#include <stddef.h>
#include "fft_radix64_avx512.h"

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

typedef struct {
  const char *variant, *isa, *protocol;
  t1_fn fwd, bwd;
  int requires_avx512;
} candidate_t;

const candidate_t CANDIDATES_AVX512[] = {
  {"ct_t1_dit", "avx512", "flat", radix64_t1_dit_fwd_avx512, radix64_t1_dit_bwd_avx512, 1},
  {"ct_t1_dif", "avx512", "flat", radix64_t1_dif_fwd_avx512, radix64_t1_dif_bwd_avx512, 1},
  {"ct_t1s_dit", "avx512", "t1s", radix64_t1s_dit_fwd_avx512, radix64_t1s_dit_bwd_avx512, 1},
  {"ct_t1_dit_log3", "avx512", "log3", radix64_t1_dit_log3_fwd_avx512, radix64_t1_dit_log3_bwd_avx512, 1},
  {"ct_t1_dif_log3", "avx512", "log3", radix64_t1_dif_log3_fwd_avx512, radix64_t1_dif_log3_bwd_avx512, 1},
  {"ct_t1_dit_log3_isub2", "avx512", "log3", radix64_t1_dit_log3_isub2_fwd_avx512, radix64_t1_dit_log3_isub2_bwd_avx512, 1},
  {"ct_t1_buf_dit_tile64_temporal", "avx512", "flat", radix64_t1_buf_dit_tile64_temporal_fwd_avx512, radix64_t1_buf_dit_tile64_temporal_bwd_avx512, 1},
  {"ct_t1_buf_dit_tile128_temporal", "avx512", "flat", radix64_t1_buf_dit_tile128_temporal_fwd_avx512, radix64_t1_buf_dit_tile128_temporal_bwd_avx512, 1},
};
const size_t N_CANDIDATES_AVX512 = sizeof(CANDIDATES_AVX512) / sizeof(CANDIDATES_AVX512[0]);
