/* tangent_gate.c — tolerance gate for every SHIPPED tangent codelet
 * (src/dag-fft-compiler/codelets/zil/<isa>/pair2p/tangent/). Conventions:
 *   MID fwd : Y[o] = sum_l W_R^{ol} . w512^{lk} . x_l      (pre-twiddle)
 *   LEAF    : Y[o] = sum_l W_R^{ol} x_l, stored corner-turned (no table)
 * The tangent codelets are built outside the corpus (design D3) at both ISAs:
 * VFFT_IL_SYM names them (il_isa.h) and il2p.h declares them. The twiddle
 * records are the library's own at the build's ISA (_vfft_il3p_vtw2, il2p.h:
 * VTW2 at avx2, vtw_avx512.h's records at avx512).
 * Counts 1..9 reach the wide body and every leftover-column arm at both ISAs
 * (2 columns per 256-bit vector, 4 per 512-bit), and one wide count (32 at
 * radix 8 and 16, 16 at radix 32) the long loop. Every run is guarded: the
 * output starts as a canary inside a canary band (every slot written, nothing
 * outside it stored), and the input and the table sit between NaN bands (a
 * read past either end poisons the result).
 * Build (from the repo root):
 *   python gauntlet/build.py --compile --src src/tools/gates/tangent_gate.c
 *   (VFFT_ISA=avx512 builds it at avx512; without AVX-512 on the host it runs
 *   under Intel SDE: sde.exe -spr -- src/tools/gates/tangent_gate.exe)
 */
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include "il_isa.h"   /* the build's ISA: VFFT_IL_SYM, VFFT_IL_ISA_NAME, VFFT_IL_TWREC */
#include "il2p.h"     /* the tangent declarations; _vfft_il3p_vtw2, the library's pair records */

/* the radix 6/10/12 tangent forms were deleted 2026-09-24 (no resolver arm) */
typedef void (*krn)(const double*,const double*,double*,double*,const double*,
                    const double*,size_t,size_t,size_t,size_t,size_t);

typedef double _Complex cx;
static const double PI = 3.14159265358979323846;
static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static double rnd(void){ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
    return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

#define GUARD 64                   /* doubles of band on each side */
static double CANARY, POISON;      /* two quiet NaNs no arithmetic makes */
static int is(double v, double band){ return !memcmp(&v, &band, 8); }

/* n doubles between two bands; *base is what to free */
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

/* one run at `cols` columns: the worst |got - ref|, NaN if a slot is
 * unwritten or non-finite or a band was touched (then *why says which) */
static double gate_one(krn fn, int R, int cols, int mid, char *why, size_t whyn){
    const size_t n = (size_t)2*R*cols;
    double *ib, *ob, *tb = 0;
    double *zin = banded(n, POISON, &ib);
    double *zo  = banded(n, CANARY, &ob);
    double *T = 0;
    size_t nt = 0;
    for(size_t i=0;i<n;i++) zin[i] = rnd();
    if(mid){
        double *t = _vfft_il3p_vtw2(R, 1, cols, 512, 0);   /* w = exp(-2 pi i l k / 512) */
        nt = _vfft_il3p_vtw2_recs(cols)*(size_t)(R-1)*VFFT_IL_TWREC;
        T = banded(nt, POISON, &tb);
        memcpy(T, t, nt*sizeof(double));
        vfft_aligned_free(t);
        fn(zin,0,zo,0,T,0,cols,0,cols,0,cols);
    } else {
        fn(zin,0,zo,0,(double*)0,0,cols,0,R,0,cols);
    }
    size_t unw = 0, nonf = 0, hits = band_hits(ob, n, CANARY) + band_hits(ib, n, POISON)
                                    + (mid ? band_hits(tb, nt, POISON) : 0);
    double w = 0;
    for(int k=0;k<cols;k++) for(int o=0;o<R;o++){
        cx ref = 0;
        for(int l=0;l<R;l++){
            cx xv = zin[2*((size_t)l*cols+k)] + I*zin[2*((size_t)l*cols+k)+1];
            ref += cexp(-2.0*PI*I*(double)o*l/(double)R)
                 * (mid ? cexp(-2.0*PI*I*(double)l*k/512.0) : 1.0) * xv;
        }
        size_t at = mid ? (size_t)o*cols+k : (size_t)k*R+o;
        double re = zo[2*at], im = zo[2*at+1];
        if(is(re, CANARY) || is(im, CANARY)) { unw++; continue; }
        if(!isfinite(re) || !isfinite(im)) { nonf++; continue; }
        double e = cabs(re + I*im - ref); if(e>w) w = e;
    }
    _aligned_free(ib); _aligned_free(ob); if(tb) _aligned_free(tb);
    if(unw || nonf || hits){
        snprintf(why, whyn, "%zu unwritten, %zu non-finite, %zu band doubles touched", unw, nonf, hits);
        return NAN;
    }
    return w;
}

int main(void){
    uint64_t u = 0x7ff8dead0000beefull, v = 0x7ff8beef0000deadull;
    memcpy(&CANARY, &u, 8); memcpy(&POISON, &v, 8);
    struct { const char *nm; krn k; int R, mid, wide; } r[5] = {
        { "radix8_z_t2tan_fwd       (R8  mid,  bit-identical)   ", VFFT_IL_SYM(radix8_z_t2tan_fwd),    8,  1, 32 },
        { "radix8_z_n1ttan_fwd      (R8  leaf, bit-identical)   ", VFFT_IL_SYM(radix8_z_n1ttan_fwd),   8,  0, 16 },
        { "radix16_z_t2tan_fwd      (R16 mid,  wing full-kernel)", VFFT_IL_SYM(radix16_z_t2tan_fwd),   16, 1, 32 },
        { "radix16_z_n1ttan_fwd     (R16 leaf, corner-turn)     ", VFFT_IL_SYM(radix16_z_n1ttan_fwd),  16, 0, 16 },
        { "radix32_z_t2bw32_fwd     (R32 mid,  wing32 2.16)     ", VFFT_IL_SYM(radix32_z_t2bw32_fwd),  32, 1, 16 },
    };
    int ok = 1;
    for(int i=0;i<5;i++){
        double worst = 0;
        int pass = 1;
        for(int c=1;c<=10;c++){
            const int cols = c <= 9 ? c : r[i].wide;
            char why[96] = "";
            double e = gate_one(r[i].k, r[i].R, cols, r[i].mid, why, sizeof why);
            if(!(e < 1e-10)){
                printf("%s  count %-2d %s  *** WRONG ***\n", r[i].nm, cols, why[0] ? why : "");
                pass = 0;
            }
            if(e > worst) worst = e;
        }
        printf("%s  %.3e  %s\n", r[i].nm, worst, pass ? "CORRECT" : "*** WRONG ***");
        ok &= pass;
    }
    printf("\n%s\n", ok ? "ALL 5 SHIPPED TANGENT CODELETS GATED CORRECT (counts 1..9 and wide, "
                          VFFT_IL_ISA_NAME ")"
                        : "GATE FAILURE - DO NOT SHIP");
    return ok ? 0 : 1;
}
