/* il_r8_m1_race.c — TIER-2 M1 GO/NO-GO: hand-written z-native radix-8 leaf vs
 * split-radix-8 + il boundary conversion, on interleaved (z) data.
 *
 * The question il_native_design.md M1 must answer BEFORE the emitter work:
 * for a twiddle-free radix-8 leaf on interleaved data, does computing
 * INTERLEAVED-NATIVE (pay the vaddsubpd/vshufpd ±i tax, no boundary pass) beat
 * computing SPLIT (cheap butterfly, but pay deinterleave-in / reinterleave-out)?
 * MKL/FFTW use interleaved-native; this is the direct measurement on OUR i9.
 *
 * Layout: point-major batch (the four-step column layout), K transforms of 8
 * complex, interleaved: z[p*(2K) + 2k + c], p=0..7 point, k transform, c re/im.
 * 2 complex per ymm (VL=2) = 2 transforms/iteration (FFTW n1fv shape).
 *
 * Arms (both on the SAME interleaved buffer, both gated bit vs naive):
 *   ZNAT  — z-native radix-8: load ymm, interleaved butterfly, store ymm.
 *   SPLIT — deinterleave each ymm to re/im, split radix-8, reinterleave, store
 *           (= our current architecture: split codelet behind il_in/il_out).
 * Also SPLITpure — split butterfly on already-split buffers (no conversion),
 *   the floor: isolates the boundary tax = SPLIT - SPLITpure.
 *
 * Methodology: pinned P-core (logical 2), HIGH prio, best-of-7, cachebust
 * between trials, arm order flipped per trial. Sweep K.
 *
 * Build: python build.py --src benches/il_r8_m1_race.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
static void cachebust(void)
{
    size_t s = 32u * 1024u * 1024u / 8u;
    double *j = (double *)malloc(s * 8);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a; free(j);
}
static double *ad(size_t n)
{
    void *p = _mm_malloc(n * sizeof(double), 64);
    if (!p) exit(1);
    return (double *)p;
}

/* ---------- interleaved (z-native) complex primitives, VL=2 ---------- */
/* ymm = [re0,im0,re1,im1] (2 complex). forward DFT sign e^{-i2pi..}. */
#define FLIP(x)  _mm256_permute_pd((x), 0x5)                 /* (re,im)->(im,re) */
static const __m256d SGN_RE = { -0.0, 0.0, -0.0, 0.0 };      /* negate re lanes  */
static const __m256d SGN_IM = { 0.0, -0.0, 0.0, -0.0 };      /* negate im lanes  */
/* x * (-i) = (im, -re)  [the forward radix-4 twiddle] */
static inline __m256d ZmulNI(__m256d x){ return _mm256_xor_pd(FLIP(x), SGN_IM); }
/* x * (+i) = (-im, re) */
static inline __m256d ZmulPI(__m256d x){ return _mm256_xor_pd(FLIP(x), SGN_RE); }

/* z-native radix-8 DIT, forward — EXACT interleaved translation of the verified
 * split dft8v (mono-64 bench, gates 1e-14). natural in/out order.
 * ZmulNI(z)=z*(-i)=(im,-re); every mixed re/im op below maps to one ±i helper. */
static inline void r8_z(const __m256d in[8], __m256d out[8])
{
    const __m256d C = _mm256_set1_pd(0.70710678118654752440);
    __m256d t0=_mm256_add_pd(in[0],in[4]), t1=_mm256_sub_pd(in[0],in[4]);
    __m256d t2=_mm256_add_pd(in[2],in[6]), t3=_mm256_sub_pd(in[2],in[6]);
    __m256d E0=_mm256_add_pd(t0,t2), E2=_mm256_sub_pd(t0,t2);
    __m256d E1=_mm256_add_pd(t1,ZmulNI(t3)), E3=_mm256_sub_pd(t1,ZmulNI(t3));
    __m256d s0=_mm256_add_pd(in[1],in[5]), s1=_mm256_sub_pd(in[1],in[5]);
    __m256d s2=_mm256_add_pd(in[3],in[7]), s3=_mm256_sub_pd(in[3],in[7]);
    __m256d O0=_mm256_add_pd(s0,s2), O2=_mm256_sub_pd(s0,s2);
    __m256d O1=_mm256_add_pd(s1,ZmulNI(s3)), O3=_mm256_sub_pd(s1,ZmulNI(s3));
    __m256d W1=_mm256_mul_pd(C,_mm256_add_pd(O1,ZmulNI(O1)));
    __m256d W2=ZmulNI(O2);
    __m256d W3=_mm256_mul_pd(C,_mm256_sub_pd(ZmulNI(O3),O3));
    out[0]=_mm256_add_pd(E0,O0); out[4]=_mm256_sub_pd(E0,O0);
    out[1]=_mm256_add_pd(E1,W1); out[5]=_mm256_sub_pd(E1,W1);
    out[2]=_mm256_add_pd(E2,W2); out[6]=_mm256_sub_pd(E2,W2);
    out[3]=_mm256_add_pd(E3,W3); out[7]=_mm256_sub_pd(E3,W3);
}

/* ---------- split radix-8 (our current-style compute) ---------- */
/* re[8],im[8] each __m256d (2 transforms). ±i is free (operand swap). */
static inline void r8_split(const __m256d re[8], const __m256d im[8],
                            __m256d ore[8], __m256d oim[8])
{
    const __m256d C = _mm256_set1_pd(0.70710678118654752440);
    const __m256d Z = _mm256_setzero_pd();
    __m256d t0r=_mm256_add_pd(re[0],re[4]), t0i=_mm256_add_pd(im[0],im[4]);
    __m256d t1r=_mm256_sub_pd(re[0],re[4]), t1i=_mm256_sub_pd(im[0],im[4]);
    __m256d t2r=_mm256_add_pd(re[2],re[6]), t2i=_mm256_add_pd(im[2],im[6]);
    __m256d t3r=_mm256_sub_pd(re[2],re[6]), t3i=_mm256_sub_pd(im[2],im[6]);
    __m256d E0r=_mm256_add_pd(t0r,t2r), E0i=_mm256_add_pd(t0i,t2i);
    __m256d E2r=_mm256_sub_pd(t0r,t2r), E2i=_mm256_sub_pd(t0i,t2i);
    __m256d E1r=_mm256_add_pd(t1r,t3i), E1i=_mm256_sub_pd(t1i,t3r);
    __m256d E3r=_mm256_sub_pd(t1r,t3i), E3i=_mm256_add_pd(t1i,t3r);
    __m256d s0r=_mm256_add_pd(re[1],re[5]), s0i=_mm256_add_pd(im[1],im[5]);
    __m256d s1r=_mm256_sub_pd(re[1],re[5]), s1i=_mm256_sub_pd(im[1],im[5]);
    __m256d s2r=_mm256_add_pd(re[3],re[7]), s2i=_mm256_add_pd(im[3],im[7]);
    __m256d s3r=_mm256_sub_pd(re[3],re[7]), s3i=_mm256_sub_pd(im[3],im[7]);
    __m256d O0r=_mm256_add_pd(s0r,s2r), O0i=_mm256_add_pd(s0i,s2i);
    __m256d O2r=_mm256_sub_pd(s0r,s2r), O2i=_mm256_sub_pd(s0i,s2i);
    __m256d O1r=_mm256_add_pd(s1r,s3i), O1i=_mm256_sub_pd(s1i,s3r);
    __m256d O3r=_mm256_sub_pd(s1r,s3i), O3i=_mm256_add_pd(s1i,s3r);
    __m256d W1r=_mm256_mul_pd(C,_mm256_add_pd(O1r,O1i)), W1i=_mm256_mul_pd(C,_mm256_sub_pd(O1i,O1r));
    __m256d W2r=O2i, W2i=_mm256_sub_pd(Z,O2r);
    __m256d W3r=_mm256_mul_pd(C,_mm256_sub_pd(O3i,O3r)), W3i=_mm256_sub_pd(Z,_mm256_mul_pd(C,_mm256_add_pd(O3r,O3i)));
    ore[0]=_mm256_add_pd(E0r,O0r); oim[0]=_mm256_add_pd(E0i,O0i);
    ore[4]=_mm256_sub_pd(E0r,O0r); oim[4]=_mm256_sub_pd(E0i,O0i);
    ore[1]=_mm256_add_pd(E1r,W1r); oim[1]=_mm256_add_pd(E1i,W1i);
    ore[5]=_mm256_sub_pd(E1r,W1r); oim[5]=_mm256_sub_pd(E1i,W1i);
    ore[2]=_mm256_add_pd(E2r,W2r); oim[2]=_mm256_add_pd(E2i,W2i);
    ore[6]=_mm256_sub_pd(E2r,W2r); oim[6]=_mm256_sub_pd(E2i,W2i);
    ore[3]=_mm256_add_pd(E3r,W3r); oim[3]=_mm256_add_pd(E3i,W3i);
    ore[7]=_mm256_sub_pd(E3r,W3r); oim[7]=_mm256_sub_pd(E3i,W3i);
}

/* il_in load lattice (the emitted radix8_n1_oop_il_in shape): 2 z-ymm (4
 * complex) -> re=[r0,r1,r2,r3], im=[i0,i1,i2,i3]. */
static inline void il_in(__m256d za, __m256d zb, __m256d *re, __m256d *im)
{
    /* za=[r0,i0,r1,i1] zb=[r2,i2,r3,i3] */
    __m256d lo = _mm256_unpacklo_pd(za, zb);  /* [r0,r2,r1,r3] */
    __m256d hi = _mm256_unpackhi_pd(za, zb);  /* [i0,i2,i1,i3] */
    *re = _mm256_permute4x64_pd(lo, 0xD8);    /* [r0,r1,r2,r3] */
    *im = _mm256_permute4x64_pd(hi, 0xD8);    /* [i0,i1,i2,i3] */
}
/* il_out store lattice: re/im -> 2 z-ymm (inverse of il_in). */
static inline void il_out(__m256d re, __m256d im, __m256d *za, __m256d *zb)
{
    __m256d r = _mm256_permute4x64_pd(re, 0xD8);  /* [r0,r2,r1,r3] */
    __m256d i = _mm256_permute4x64_pd(im, 0xD8);  /* [i0,i2,i1,i3] */
    *za = _mm256_unpacklo_pd(r, i);               /* [r0,i0,r1,i1] */
    *zb = _mm256_unpackhi_pd(r, i);               /* [r2,i2,r3,i3] */
}

/* ---------- arms over a K-batch (point-major z: z[p*2K + 2k + c]) ---------- */
/* ZNAT: z-native, 2 complex/ymm, 2 transforms/iter (2 radix-8 per 4 transforms). */
static void arm_znat(const double *zin, double *zout, int K)
{
    const size_t S = (size_t)2 * K;
    for (int k = 0; k < K; k += 2) {
        __m256d in[8], out[8];
        for (int p = 0; p < 8; p++) in[p] = _mm256_loadu_pd(zin + (size_t)p * S + 2 * k);
        r8_z(in, out);
        for (int p = 0; p < 8; p++) _mm256_storeu_pd(zout + (size_t)p * S + 2 * k, out[p]);
    }
}
/* SPLIT+IL: our current architecture — 4 transforms/iter, il_in convert,
 * 4-wide split radix-8, il_out convert. Same z buffer as ZNAT. */
static void arm_split_boundary(const double *zin, double *zout, int K)
{
    const size_t S = (size_t)2 * K;
    for (int k = 0; k < K; k += 4) {
        __m256d re[8], im[8], ore[8], oim[8];
        for (int p = 0; p < 8; p++) {
            __m256d za = _mm256_loadu_pd(zin + (size_t)p*S + 2*k);
            __m256d zb = _mm256_loadu_pd(zin + (size_t)p*S + 2*k + 4);
            il_in(za, zb, &re[p], &im[p]);
        }
        r8_split(re, im, ore, oim);
        for (int p = 0; p < 8; p++) {
            __m256d za, zb; il_out(ore[p], oim[p], &za, &zb);
            _mm256_storeu_pd(zout + (size_t)p*S + 2*k, za);
            _mm256_storeu_pd(zout + (size_t)p*S + 2*k + 4, zb);
        }
    }
}
/* SPLITpure: split radix-8 on separate re/im buffers (compute floor, 2 streams). */
static void arm_split_pure(const double *rin, const double *iin, double *rout, double *iout, int K)
{
    for (int k = 0; k < K; k += 4) {
        __m256d re[8], im[8], ore[8], oim[8];
        for (int p = 0; p < 8; p++) { re[p]=_mm256_loadu_pd(rin+(size_t)p*K+k); im[p]=_mm256_loadu_pd(iin+(size_t)p*K+k); }
        r8_split(re, im, ore, oim);
        for (int p = 0; p < 8; p++) { _mm256_storeu_pd(rout+(size_t)p*K+k, ore[p]); _mm256_storeu_pd(iout+(size_t)p*K+k, oim[p]); }
    }
}

/* EMITTED z-native codelet (codelets/zil/avx2/radix8_z_n1_avx2.c) — must be
 * BIT-identical to arm_znat (the hand oracle). */
extern void radix8_z_n1_fwd_avx2(const double *, const double *, double *,
    double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
static void arm_zemit(const double *zin, double *zout, int K)
{
    radix8_z_n1_fwd_avx2(zin, 0, zout, 0, 0, 0, (size_t)K, 0, (size_t)K, 0, (size_t)K);
}

static void naive_r8(const double zin[16], double zout[16])
{
    for (int m = 0; m < 8; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < 8; n++) {
            double a = -2.0 * M_PI * (double)(n * m) / 8.0, c = cos(a), s = sin(a);
            sr += zin[2*n]*c - zin[2*n+1]*s;
            si += zin[2*n]*s + zin[2*n+1]*c;
        }
        zout[2*m] = sr; zout[2*m+1] = si;
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    int Ks[] = { 8, 64, 256, 1024, 4096 };
    int nK = argc > 1 ? argc - 1 : 5;

    /* ---- gate at K=8 (point-major) ---- */
    {
        int K = 8; size_t S = 2*K;
        double *zin = ad(8*S), *zt = ad(8*S), *rin=ad(8*K),*iin=ad(8*K),*rt=ad(8*K),*it=ad(8*K);
        srand(1);
        for (int p=0;p<8;p++) for (int k=0;k<K;k++){ double re=(double)rand()/RAND_MAX-0.5, im=(double)rand()/RAND_MAX-0.5;
            zin[p*S+2*k]=re; zin[p*S+2*k+1]=im; rin[p*K+k]=re; iin[p*K+k]=im; }
        double ez=0, es=0, esp=0, eem=0;
        (void)zt;
        double *zt1=ad(8*S), *zt2=ad(8*S), *zt3=ad(8*S);
        arm_znat(zin, zt1, K); arm_split_boundary(zin, zt2, K); arm_split_pure(rin,iin,rt,it,K);
        arm_zemit(zin, zt3, K);
        for (size_t i=0;i<8*S;i++){ double d=fabs(zt3[i]-zt1[i]); if(d>eem)eem=d; }
        for (int k=0;k<K;k++){
            double nin[16], nout[16];
            for (int p=0;p<8;p++){ nin[2*p]=zin[p*S+2*k]; nin[2*p+1]=zin[p*S+2*k+1]; }
            naive_r8(nin, nout);
            for (int p=0;p<8;p++){
                double dz=fabs(zt1[p*S+2*k]-nout[2*p])+fabs(zt1[p*S+2*k+1]-nout[2*p+1]);
                double ds=fabs(zt2[p*S+2*k]-nout[2*p])+fabs(zt2[p*S+2*k+1]-nout[2*p+1]);
                double dp=fabs(rt[p*K+k]-nout[2*p])+fabs(it[p*K+k]-nout[2*p+1]);
                if(dz>ez)ez=dz; if(ds>es)es=ds; if(dp>esp)esp=dp;
            }
        }
        printf("GATE znat=%.2e  split=%.2e  splitpure=%.2e  EMIT-vs-hand=%.1f(bit)  %s\n",
               ez, es, esp, eem,
               (ez<1e-12&&es<1e-12&&esp<1e-12&&eem==0.0)?"PASS":"FAIL");
        if(!(ez<1e-12&&es<1e-12&&esp<1e-12&&eem==0.0)) return 1;
    }

    printf("# ns per radix-8 transform (lower=better); best-of-7, order-flipped, cachebust\n");
    printf("%-8s %10s %10s %12s %10s\n","K","ZNAT","SPLIT","SPLITpure","bndry-tax");
    for (int ki=0; ki<nK; ki++){
        int K = argc>1?atoi(argv[ki+1]):Ks[ki]; size_t S=2*K;
        double *zin=ad(8*S), *zo=ad(8*S), *rin=ad(8*K),*iin=ad(8*K),*ro=ad(8*K),*io=ad(8*K);
        srand(7+K);
        for (size_t i=0;i<8*S;i++) zin[i]=(double)rand()/RAND_MAX-0.5;
        for (size_t i=0;i<8*(size_t)K;i++){ rin[i]=(double)rand()/RAND_MAX-0.5; iin[i]=(double)rand()/RAND_MAX-0.5; }
        int reps = (int)(2e7/(double)K); if(reps<200)reps=200; if(reps>2000000)reps=2000000;
        double best[3]={1e18,1e18,1e18};
        for (int t=0;t<7;t++){
            if(t) cachebust();
            for (int a=0;a<3;a++){
                int arm=(t&1)?2-a:a;
                for(int w=0;w<10;w++){ if(arm==0)arm_znat(zin,zo,K); else if(arm==1)arm_split_boundary(zin,zo,K); else arm_split_pure(rin,iin,ro,io,K); }
                double t0=now_ms();
                for(int i=0;i<reps;i++){ if(arm==0)arm_znat(zin,zo,K); else if(arm==1)arm_split_boundary(zin,zo,K); else arm_split_pure(rin,iin,ro,io,K); }
                double ns=(now_ms()-t0)*1e6/((double)reps*K);
                if(ns<best[arm])best[arm]=ns;
            }
        }
        printf("%-8d %10.3f %10.3f %12.3f %+9.1f%%\n", K, best[0], best[1], best[2],
               100.0*(best[1]-best[2])/best[2]);
        _mm_free(zin);_mm_free(zo);_mm_free(rin);_mm_free(iin);_mm_free(ro);_mm_free(io);
    }
    printf("DONE\n");
    return 0;
}
