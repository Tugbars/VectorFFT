/* auto-generated K=1 JIT wrapper — k1_n1024_r2_32x32_avx2_ver2. Shape baked, buffers passed. */
#include "codelets/oop/avx2/radix32_n1_oop_ugul_avx2.c"
#include "codelets/oop/avx2/radix32_t1_oop_avx2.c"
__attribute__((flatten))
void vfft_k1_jit_exec(const double *sr, const double *si,
                      double *dr, double *di,
                      double *col_re, double *col_im,
                      const double *qr, const double *qi)
{
    radix32_n1_oop_fwd_avx2_UG_UL(sr, si, col_re, col_im, 0, 0, 32, 1, 1, 32, 32);
    radix32_t1_oop_fwd_avx2_UG_UG(col_re, col_im, dr, di, qr, qi, 32, 1, 32, 1, 32);
}
