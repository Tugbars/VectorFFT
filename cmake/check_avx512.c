#include <immintrin.h>
int main(void) {
    __m512d a = _mm512_setzero_pd();
    __m512d b = _mm512_set1_pd(1.0);
    __m512d c = _mm512_add_pd(a, b);
    (void)c;
    return 0;
}
