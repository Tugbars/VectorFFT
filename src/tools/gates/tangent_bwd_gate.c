/* tangent_bwd_gate.c — the tangent BACKWARD twins (t2ttan mid: turned store, n1tan leaf) vs the classic backward kernels
 * they would replace: identical inputs, identical VTW2 table, identical
 * geometry; the classic kernel is the reference (it is gated elsewhere).
 * R8 must be BIT-IDENTICAL (the tangent rewrite at radix 8 is exact: only the
 * sqrt(1/2) folds change, by +-1 FMAs); R16 within 1e-13 relative.
 * Counts cover the 2-column wide path (2,4,6) and the odd tail (3,5).
 * The classic kernels are the build's ISA's (VFFT_IL_SYM, declared by the
 * registry); the tangent twins are built outside the corpus (design D3) and
 * exist at avx2 only: VFFT_IL_AVX2_ONLY names them (il_isa.h), 0 at avx512,
 * where each row is reported and not gated. The records below are therefore
 * avx2's VTW2.
 * Build: python gauntlet/build.py --compile --src src/tools/gates/tangent_bwd_gate.c   (the codelet library links both;
 *        VFFT_ISA=avx512 builds it at avx512) */
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <malloc.h>
#include "il_isa.h"   /* the IL registry at the build's ISA: VFFT_IL_SYM, VFFT_IL_AVX2_ONLY */
#define K(n) void n(const double*,const double*,double*,double*,const double*, \
                    const double*,size_t,size_t,size_t,size_t,size_t)
K(radix8_z_t2ttan_bwd_avx2);
K(radix8_z_n1tan_bwd_avx2);
K(radix16_z_t2ttan_bwd_avx2);
K(radix16_z_n1tan_bwd_avx2);
K(radix32_z_n1btan216_bwd_avx2);
typedef void (*krn)(const double*,const double*,double*,double*,const double*,
                    const double*,size_t,size_t,size_t,size_t,size_t);
typedef double _Complex cx;
static const double PI = 3.14159265358979323846;
static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static double rnd(void){ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
    return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }
/* VTW2 records, the fold [-s,+s]; sg = +1 for the backward */
static void genT(double *T, int R, int kc, double sg){
    for(int l=1;l<R;l++){
        double *r = T + (size_t)(l-1)*8;
        for(int c=0;c<2;c++){
            int kk = kc + c;
            cx d = cexp(sg*2.0*PI*I*(double)l*kk/512.0);
            r[c*2] = creal(d); r[c*2+1] = creal(d);
            r[4+c*2] = -cimag(d); r[4+c*2+1] = cimag(d);
        }
    }
}
/* run both kernels on the same input; return max |diff|, set *bit if bitwise */
static double cmp(krn a, krn b, int R, int cols, int mid, int *bit){
    int rec = (R-1)*8, N2 = R*cols*2, ncol2 = (cols+1)/2;
    double *zin = _aligned_malloc((size_t)N2*8, 64);
    double *za  = _aligned_malloc((size_t)N2*8, 64);
    double *zb  = _aligned_malloc((size_t)N2*8, 64);
    double *T   = _aligned_malloc((size_t)ncol2*rec*8, 64);
    for(int i=0;i<N2;i++) zin[i] = rnd();
    for(int g=0; g<ncol2; g++) genT(T + (size_t)g*rec, R, 2*g, +1.0);
    memset(za, 0, (size_t)N2*8); memset(zb, 0, (size_t)N2*8);
    if (mid) { a(zin,0,za,0,T,0,cols,0,R,0,cols); b(zin,0,zb,0,T,0,cols,0,R,0,cols); } /* t2t: turned store zout[k*OLs + o], OLs = R */
    else     { a(zin,0,za,0,(double*)0,0,cols,0,cols,0,cols); b(zin,0,zb,0,(double*)0,0,cols,0,cols,0,cols); } /* n1: zout[o*OLs + k], OLs = cols */
    double w = 0, scale = 0;
    for(int i=0;i<N2;i++){ double e = fabs(za[i]-zb[i]); if(e>w) w=e; if(fabs(za[i])>scale) scale=fabs(za[i]); }
    *bit = (memcmp(za, zb, (size_t)N2*8) == 0);
    _aligned_free(zin); _aligned_free(za); _aligned_free(zb); _aligned_free(T);
    return scale > 0 ? w/scale : w;
}
int main(void){
    struct { const char *nm; krn a, b; int R, mid; double tol; } t[] = {
        { "radix8  t2t bwd (mid)  ", VFFT_IL_SYM(radix8_z_t2t_bwd),  VFFT_IL_AVX2_ONLY(radix8_z_t2ttan_bwd),  8,  1, 0.0   },
        { "radix8  n1  bwd (leaf) ", VFFT_IL_SYM(radix8_z_n1_bwd),  VFFT_IL_AVX2_ONLY(radix8_z_n1tan_bwd),  8,  0, 0.0   },
        { "radix16 t2t bwd (mid)  ", VFFT_IL_SYM(radix16_z_t2t_bwd), VFFT_IL_AVX2_ONLY(radix16_z_t2ttan_bwd), 16, 1, 1e-13 },
        { "radix16 n1  bwd (leaf) ", VFFT_IL_SYM(radix16_z_n1_bwd), VFFT_IL_AVX2_ONLY(radix16_z_n1tan_bwd), 16, 0, 1e-13 },
        { "radix32 n1  bwd (leaf) ", VFFT_IL_SYM(radix32_z_n1b216_bwd), VFFT_IL_AVX2_ONLY(radix32_z_n1btan216_bwd), 32, 0, 1e-13 }, /* the tangent 2.16 vs the classic blocked 2.16 */
    };
    static const int counts[] = { 2, 4, 6, 3, 5, 32 };
    const int nt = (int)(sizeof t / sizeof t[0]);
    int fails = 0, nabsent = 0;
    for (int i = 0; i < nt; i++) {
        if (!t[i].b) {
            printf("%s absent at %s: not gated\n", t[i].nm, VFFT_IL_ISA_NAME);
            nabsent++;
            continue;
        }
        printf("%s", t[i].nm);
        for (size_t c = 0; c < sizeof counts / sizeof counts[0]; c++) {
            int bit; double e = cmp(t[i].a, t[i].b, t[i].R, counts[c], t[i].mid, &bit);
            int ok = t[i].tol == 0.0 ? bit : (e <= t[i].tol);
            printf(" c%-2d %s%.1e", counts[c], bit ? "bits " : "", e);
            if (!ok) { printf("(FAIL)"); fails++; }
        }
        printf("\n");
    }
    if (nabsent == nt) {
        printf("=== no tangent backward twin at %s: nothing gated ===\n", VFFT_IL_ISA_NAME);
        return 0;
    }
    printf("=== %s ===\n", fails ? "FAIL" : "ALL PASS");
    return fails != 0;
}
