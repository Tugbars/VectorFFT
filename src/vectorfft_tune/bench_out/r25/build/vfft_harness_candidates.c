/* Auto-generated candidate aggregator. DO NOT EDIT. */
#include <stddef.h>

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

typedef struct {
  const char *variant, *isa, *protocol;
  t1_fn fwd, bwd;
  int requires_avx512;
} candidate_t;

extern const candidate_t CANDIDATES_AVX2[];
extern const size_t N_CANDIDATES_AVX2;
extern const candidate_t CANDIDATES_AVX512[];
extern const size_t N_CANDIDATES_AVX512;

/* Public view seen by the harness: concatenates every per-ISA array.
 * We build it at startup via a constructor-like initializer below, but
 * since C doesn't allow initializing an array from other arrays at file
 * scope we expose an indexer function instead. */
/* compile-time total: 14 */

static candidate_t _flat[14];
static size_t _flat_n = 0;
static int _flat_init = 0;

static void _init_flat(void) {
  if (_flat_init) return;
  _flat_init = 1;
  size_t j = 0;
  for (size_t i = 0; i < N_CANDIDATES_AVX2; i++) _flat[j++] = CANDIDATES_AVX2[i];
  for (size_t i = 0; i < N_CANDIDATES_AVX512; i++) _flat[j++] = CANDIDATES_AVX512[i];
  _flat_n = j;
}

const candidate_t *candidate_at(size_t i) {
  _init_flat();
  if (i >= _flat_n) return NULL;
  return &_flat[i];
}

size_t candidate_count(void) {
  _init_flat();
  return _flat_n;
}
