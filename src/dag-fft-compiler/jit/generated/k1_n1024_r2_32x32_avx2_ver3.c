/* auto-generated K=1 JIT wrapper — k1_n1024_r2_32x32_avx2_ver3. Shape baked, buffers passed. */
#include "codelets/oop/avx2/radix32_n1_oop_ugul_avx2.c"
#include "codelets/oop/avx2/radix32_t1_oop_avx2.c"
__attribute__((flatten,noinline)) static void _k1_s1(
    const double *sr, const double *si, double *cr, double *ci)
{
    radix32_n1_oop_fwd_avx2_UG_UL(sr, si, cr, ci, 0, 0, 32, 1, 1, 32, 32);
}
__attribute__((flatten,noinline)) static void _k1_s2(
    const double *cr, const double *ci, double *dr, double *di,
    const double *qr, const double *qi)
{
    radix32_t1_oop_fwd_avx2_UG_UG(cr, ci, dr, di, qr, qi, 32, 1, 32, 1, 32);
}
void vfft_k1_jit_exec(const double *sr, const double *si,
                      double *dr, double *di,
                      double *col_re, double *col_im,
                      const double *qr, const double *qi)
{
    _k1_s1(sr, si, col_re, col_im);
    _k1_s2(col_re, col_im, dr, di, qr, qi);
}
