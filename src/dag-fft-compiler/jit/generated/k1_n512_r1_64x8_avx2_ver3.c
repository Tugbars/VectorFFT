/* auto-generated K=1 JIT wrapper — k1_n512_r1_64x8_avx2_ver3. Shape baked, buffers passed. */
#include "codelets/oop/avx2/radix8_n1_oop_avx2.c"
#include "codelets/oop/avx2/radix64_t1_oop_ul_avx2.c"
__attribute__((flatten,noinline)) static void _k1_s1(
    const double *sr, const double *si, double *cr, double *ci)
{
    radix8_n1_oop_fwd_avx2_UG_UG(sr, si, cr, ci, 0, 0, 64, 1, 64, 1, 64);
}
__attribute__((flatten,noinline)) static void _k1_s2(
    const double *cr, const double *ci, double *dr, double *di,
    const double *qr, const double *qi)
{
    radix64_t1_oop_fwd_avx2_UL_UG(cr, ci, dr, di, qr, qi, 1, 64, 8, 1, 8);
}
void vfft_k1_jit_exec(const double *sr, const double *si,
                      double *dr, double *di,
                      double *col_re, double *col_im,
                      const double *qr, const double *qi)
{
    _k1_s1(sr, si, col_re, col_im);
    _k1_s2(col_re, col_im, dr, di, qr, qi);
}
