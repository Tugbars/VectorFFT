#include <stddef.h>
typedef void (*t1_fn)(double*, double*, const double*, const double*, size_t, size_t);
typedef struct { const char *variant, *isa, *protocol; t1_fn fwd, bwd; int requires_avx512; } candidate_t;
const candidate_t CANDIDATES_AVX512[1] = {{0}};
const size_t N_CANDIDATES_AVX512 = 0;
