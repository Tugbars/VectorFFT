/* auto-generated K=1 JIT wrapper — k1_n4096_r5_64x64_avx2_ver1. Shape baked, buffers passed. */
#include "codelets/oop/avx2/radix64_n1_oop_avx2.c"
#include "codelets/oop/avx2/radix64_t1_oop_ul_log3_avx2.c"
__attribute__((flatten))
void vfft_k1_jit_exec(const double *sr, const double *si,
                      double *dr, double *di,
                      double *col_re, double *col_im,
                      const double *qr, const double *qi)
{
    radix64_n1_oop_fwd_avx2_UG_UG(sr, si, col_re, col_im, 0, 0, 64, 1, 64, 1, 64);
    radix64_t1_oop_fwd_avx2_UL_UG_log3(col_re, col_im, dr, di, qr, qi, 1, 64, 64, 1, 64);
}
