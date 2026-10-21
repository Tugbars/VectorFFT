/* tangent_bwd_gate.c — the tangent BACKWARD twins (t2ttan mid: turned store, n1tan leaf) vs the classic backward kernels
 * they would replace: identical inputs, identical VTW2 table, identical
 * geometry; the classic kernel is the reference (it is gated elsewhere).
 * R8 must be BIT-IDENTICAL (the tangent rewrite at radix 8 is exact: only the
 * sqrt(1/2) folds change, by +-1 FMAs); R16 within 1e-13 relative.
 * Counts 1..9 cover the wide path and every leftover-column arm at both ISAs
 * (2 columns per 256-bit vector, 4 per 512-bit), and 32 the long loop.
 * Both kernels are the build's ISA's (VFFT_IL_SYM): the classic ones declared
 * by the registry, the tangent twins, built outside the corpus (design D3),
 * by il2p.h. The table is the library's own pair records at the build's ISA
 * (_vfft_il3p_vtw2, il2p.h). Every run is guarded: both outputs start as a
 * canary inside a canary band (every slot written, nothing outside it
 * stored), and the input and the table sit between NaN bands (a read past
 * either end poisons the result).
 * Build: python gauntlet/build.py --compile --src src/tools/gates/tangent_bwd_gate.c   (the codelet library links both;
 *        VFFT_ISA=avx512 builds it at avx512, run under Intel SDE where the host has no AVX-512) */
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <malloc.h>
#include "il_isa.h"   /* the IL registry at the build's ISA: VFFT_IL_SYM, VFFT_IL_TWREC */
#include "il2p.h"     /* the tangent twins' declarations; _vfft_il3p_vtw2, the library's pair records */
typedef void (*krn)(const double*,const double*,double*,double*,const double*,
                    const double*,size_t,size_t,size_t,size_t,size_t);
static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static double rnd(void){ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
    return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }
#define GUARD 64                   /* doubles of band on each side */
static double CANARY, POISON;      /* two quiet NaNs no arithmetic makes */
static int is(double v, double band){ return !memcmp(&v, &band, 8); }
static double *banded(size_t n, double band, double **base){
    double *b = _aligned_malloc((n + 2*GUARD)*sizeof(double), 64);
    for(size_t i=0;i<n+2*GUARD;i++) b[i] = band;
    *base = b;
    return b + GUARD;
}
static size_t band_hits(const double *base, size_t n, double band){
    size_t h = 0;
    for(size_t i=0;i<GUARD;i++){ h += !is(base[i], band); h += !is(base[GUARD+n+i], band); }
    return h;
}
/* the slots of one output that are unwritten (canary) or non-finite */
static size_t bad_slots(const double *z, size_t n){
    size_t b = 0;
    for(size_t i=0;i<n;i++) b += is(z[i], CANARY) || !isfinite(z[i]);
    return b;
}
/* run both kernels on the same input; return max |diff| relative, set *bit if
 * bitwise; NaN if a slot is unwritten or non-finite or a band was touched */
static double cmp(krn a, krn b, int R, int cols, int mid, int *bit){
    const size_t N2 = (size_t)2*R*cols;
    const size_t nt = _vfft_il3p_vtw2_recs(cols)*(size_t)(R-1)*VFFT_IL_TWREC;
    double *ib, *ab, *bb, *tb;
    double *zin = banded(N2, POISON, &ib);
    double *za  = banded(N2, CANARY, &ab);
    double *zb  = banded(N2, CANARY, &bb);
    double *T   = banded(nt, POISON, &tb);
    double *t   = _vfft_il3p_vtw2(R, 1, cols, 512, 1);   /* w = exp(+2 pi i l k / 512) */
    memcpy(T, t, nt*sizeof(double));
    vfft_aligned_free(t);
    for(size_t i=0;i<N2;i++) zin[i] = rnd();
    if (mid) { a(zin,0,za,0,T,0,cols,0,R,0,cols); b(zin,0,zb,0,T,0,cols,0,R,0,cols); } /* t2t: turned store zout[k*OLs + o], OLs = R */
    else     { a(zin,0,za,0,(double*)0,0,cols,0,cols,0,cols); b(zin,0,zb,0,(double*)0,0,cols,0,cols,0,cols); } /* n1: zout[o*OLs + k], OLs = cols */
    size_t bad = bad_slots(za, N2) + bad_slots(zb, N2)
               + band_hits(ab, N2, CANARY) + band_hits(bb, N2, CANARY)
               + band_hits(ib, N2, POISON) + band_hits(tb, nt, POISON);
    double w = 0, scale = 0;
    for(size_t i=0;i<N2;i++){ double e = fabs(za[i]-zb[i]); if(e>w) w=e; if(fabs(za[i])>scale) scale=fabs(za[i]); }
    *bit = !bad && (memcmp(za, zb, N2*8) == 0);
    _aligned_free(ib); _aligned_free(ab); _aligned_free(bb); _aligned_free(tb);
    if (bad) return NAN;
    return scale > 0 ? w/scale : w;
}
int main(void){
    uint64_t u = 0x7ff8dead0000beefull, v = 0x7ff8beef0000deadull;
    memcpy(&CANARY, &u, 8); memcpy(&POISON, &v, 8);
    struct { const char *nm; krn a, b; int R, mid; double tol; } t[] = {
        { "radix8  t2t bwd (mid)  ", VFFT_IL_SYM(radix8_z_t2t_bwd),  VFFT_IL_SYM(radix8_z_t2ttan_bwd),  8,  1, 0.0   },
        { "radix8  n1  bwd (leaf) ", VFFT_IL_SYM(radix8_z_n1_bwd),  VFFT_IL_SYM(radix8_z_n1tan_bwd),  8,  0, 0.0   },
        { "radix16 t2t bwd (mid)  ", VFFT_IL_SYM(radix16_z_t2t_bwd), VFFT_IL_SYM(radix16_z_t2ttan_bwd), 16, 1, 1e-13 },
        { "radix16 n1  bwd (leaf) ", VFFT_IL_SYM(radix16_z_n1_bwd), VFFT_IL_SYM(radix16_z_n1tan_bwd), 16, 0, 1e-13 },
        { "radix32 n1  bwd (leaf) ", VFFT_IL_SYM(radix32_z_n1b216_bwd), VFFT_IL_SYM(radix32_z_n1btan216_bwd), 32, 0, 1e-13 }, /* the tangent 2.16 vs the classic blocked 2.16 */
    };
    static const int counts[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 32 };
    const int nt = (int)(sizeof t / sizeof t[0]);
    int fails = 0;
    for (int i = 0; i < nt; i++) {
        printf("%s", t[i].nm);
        for (size_t c = 0; c < sizeof counts / sizeof counts[0]; c++) {
            int bit; double e = cmp(t[i].a, t[i].b, t[i].R, counts[c], t[i].mid, &bit);
            int ok = t[i].tol == 0.0 ? bit : (e <= t[i].tol);
            printf(" c%-2d %s%.1e", counts[c], bit ? "bits " : "", e);
            if (!ok) { printf("(FAIL)"); fails++; }
        }
        printf("\n");
    }
    printf("=== %s (%s) ===\n", fails ? "FAIL" : "ALL PASS", VFFT_IL_ISA_NAME);
    return fails != 0;
}
