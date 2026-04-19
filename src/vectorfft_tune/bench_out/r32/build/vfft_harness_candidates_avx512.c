/* Auto-generated per-ISA candidate fragment. DO NOT EDIT. */
#include <stddef.h>
#include "fft_radix32_avx512.h"

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

typedef struct {
  const char *variant, *isa, *protocol;
  t1_fn fwd, bwd;
  int requires_avx512;
} candidate_t;

const candidate_t CANDIDATES_AVX512[] = {
  {"ct_t1_dit", "avx512", "flat", radix32_t1_dit_fwd_avx512, radix32_t1_dit_bwd_avx512, 1},
  {"ct_t1_dif", "avx512", "flat", radix32_t1_dif_fwd_avx512, radix32_t1_dif_bwd_avx512, 1},
  {"ct_t1s_dit", "avx512", "t1s", radix32_t1s_dit_fwd_avx512, radix32_t1s_dit_bwd_avx512, 1},
  {"ct_t1_dit_log3", "avx512", "log3", radix32_t1_dit_log3_fwd_avx512, radix32_t1_dit_log3_bwd_avx512, 1},
  {"ct_t1_buf_dit_tile64_temporal", "avx512", "flat", radix32_t1_buf_dit_tile64_temporal_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_bwd_avx512, 1},
  {"ct_t1_buf_dit_tile64_stream", "avx512", "flat", radix32_t1_buf_dit_tile64_stream_fwd_avx512, radix32_t1_buf_dit_tile64_stream_bwd_avx512, 1},
  {"ct_t1_buf_dit_tile128_temporal", "avx512", "flat", radix32_t1_buf_dit_tile128_temporal_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_bwd_avx512, 1},
  {"ct_t1_buf_dit_tile128_stream", "avx512", "flat", radix32_t1_buf_dit_tile128_stream_fwd_avx512, radix32_t1_buf_dit_tile128_stream_bwd_avx512, 1},
  {"ct_t1_buf_dit_tile256_temporal", "avx512", "flat", radix32_t1_buf_dit_tile256_temporal_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_bwd_avx512, 1},
  {"ct_t1_buf_dit_tile256_stream", "avx512", "flat", radix32_t1_buf_dit_tile256_stream_fwd_avx512, radix32_t1_buf_dit_tile256_stream_bwd_avx512, 1},
};
const size_t N_CANDIDATES_AVX512 = sizeof(CANDIDATES_AVX512) / sizeof(CANDIDATES_AVX512[0]);
