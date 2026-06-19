/* bench_t1p_rxr_vs_fftw.c — race the balanced R x R one-call OOP engine
 * (n1_oop column stage + t1p row stage, natural order, input preserved)
 * against FFTW via the GURU interface, OOP vs OOP.
 *
 * Compile with -DRADIX=7|8|13|32. N = RADIX*RADIX. K transforms, lane-blocked
 * split layout in blocks of V=8 (element e of lane l at [blk*N*V + e*V + l]).
 *
 * FFTW comparisons (both out-of-place, FFTW_PATIENT, plans printed):
 *   guru  : interleaved, FFTW's native best layout.
 *   guruS : guru64_split_dft on OUR exact lane-blocked split layout.
 * Correctness gate: engine output vs fftw_plan_dft_1d reference, relerr < 1e-9.
 * Method: rdtsc min-of-60 after warmup, FTZ/DAZ on. Relative ratios only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"

#ifndef RADIX
#define RADIX 8
#endif
#define N (RADIX*RADIX)
#define V 8

#define PASTE(a,b,c) a##b##c
#define MK(R) PASTE(radix,R,_n1_oop_fwd_avx512_UG_UG)
#define MKT(R) PASTE(radix,R,_t1p_oop_fwd_avx512_UG_UG)
#define MKL3(R) PASTE(radix,R,_t1p_oop_fwd_avx512_UG_UG_log3)
#define N1OOP MK(RADIX)
#define T1P   MKT(RADIX)
#define T1PL3 MKL3(RADIX)

typedef void oopk(const double*,const double*,double*,double*,
                  const double*,const double*,
                  size_t,size_t,size_t,size_t,size_t);
extern void N1OOP(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void T1P  (const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void T1PL3(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);

static double Qr[(RADIX-1)*RADIX], Qi[(RADIX-1)*RADIX];
static void initw(void){
  for (int l2 = 1; l2 < RADIX; l2++)
    for (int k2 = 0; k2 < RADIX; k2++) {
      double a = -2.0*M_PI*(double)(l2*k2)/(double)N;
      Qr[(l2-1)*RADIX+k2] = cos(a); Qi[(l2-1)*RADIX+k2] = sin(a);
    }
}

/* one V-lane block: stage 1 OOP src->dst, stage 2 t1p on dst (strides as in
 * the validated 32x32 engine, with 32 -> RADIX) */
static inline void blkF(const double*ir,const double*ii,double*o,double*oi){
  N1OOP(ir,ii,o,oi,0,0,(size_t)RADIX*V,1,(size_t)V,RADIX,(size_t)RADIX*V);
  T1P  (o,oi,o,oi,Qr,Qi,(size_t)RADIX*V,1,(size_t)RADIX*V,1,(size_t)RADIX*V);
}
static inline void blkL(const double*ir,const double*ii,double*o,double*oi){
  N1OOP(ir,ii,o,oi,0,0,(size_t)RADIX*V,1,(size_t)V,RADIX,(size_t)RADIX*V);
  T1PL3(o,oi,o,oi,Qr,Qi,(size_t)RADIX*V,1,(size_t)RADIX*V,1,(size_t)RADIX*V);
}
static void eF(const double*ir,const double*ii,double*o,double*oi,size_t K){
  for (size_t b = 0; b < K/V; b++) blkF(ir+b*(size_t)N*V, ii+b*(size_t)N*V, o+b*(size_t)N*V, oi+b*(size_t)N*V);
}
static void eL(const double*ir,const double*ii,double*o,double*oi,size_t K){
  for (size_t b = 0; b < K/V; b++) blkL(ir+b*(size_t)N*V, ii+b*(size_t)N*V, o+b*(size_t)N*V, oi+b*(size_t)N*V);
}

static unsigned long long mn(unsigned long long*a,int c){
  unsigned long long b=~0ULL; for(int i=0;i<c;i++) if(a[i]<b) b=a[i]; return b;
}

/* correctness gate vs scalar FFTW reference, sampled transforms */
static double verify(void(*eng)(const double*,const double*,double*,double*,size_t),
                     const double*ir,const double*ii,double*o,double*oi,
                     const double*xr,const double*xi,size_t K){
  eng(ir,ii,o,oi,K);
  fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*N), *fo=fftw_malloc(sizeof(fftw_complex)*N);
  fftw_plan p=fftw_plan_dft_1d(N,fi,fo,FFTW_FORWARD,FFTW_ESTIMATE);
  double mr=0;
  for (size_t t=0; t<K; t+=(K>3?K/3:1)) {
    size_t bk=t/V, l=t%V;
    for (int e=0;e<N;e++){ fi[e][0]=xr[t*(size_t)N+e]; fi[e][1]=xi[t*(size_t)N+e]; }
    fftw_execute(p);
    double me=0,mm=0;
    for (int k=0;k<N;k++){
      double dr=o [bk*(size_t)N*V+(size_t)k*V+l]-fo[k][0];
      double di=oi[bk*(size_t)N*V+(size_t)k*V+l]-fo[k][1];
      double e2=sqrt(dr*dr+di*di), m=hypot(fo[k][0],fo[k][1]);
      if(e2>me)me=e2; if(m>mm)mm=m;
    }
    if(mm>0 && me/mm>mr) mr=me/mm;
  }
  fftw_destroy_plan(p); fftw_free(fi); fftw_free(fo);
  return mr;
}

int main(int argc, char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  initw();
  size_t K = argc>1 ? (size_t)atol(argv[1]) : 1024;
  if (K % V) { fprintf(stderr,"K must be a multiple of %d\n",V); return 1; }
  size_t TOT=(size_t)N*K;

  /* engine buffers (lane-blocked split) + logical copies for the gate */
  double *ir=aligned_alloc(64,TOT*8), *ii=aligned_alloc(64,TOT*8);
  double *o =aligned_alloc(64,TOT*8), *oi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8), *xi=malloc(TOT*8);

  /* FFTW buffers: interleaved pair + split lane-blocked pair (own buffers,
   * planned BEFORE data fill since PATIENT clobbers) */
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT), *go=fftw_malloc(sizeof(fftw_complex)*TOT);
  double *sr=aligned_alloc(64,TOT*8), *si=aligned_alloc(64,TOT*8);
  double *sro=aligned_alloc(64,TOT*8), *sio=aligned_alloc(64,TOT*8);

  /* guru interleaved, FFTW's best: dims {N,1,1}, howmany {K,N,N}, OOP */
  fftw_iodim64 d1 = { N, 1, 1 };
  fftw_iodim64 h1 = { (ptrdiff_t)K, N, N };
  fftw_plan pg = fftw_plan_guru64_dft(1,&d1,1,&h1,gi,go,FFTW_FORWARD,FFTW_PATIENT);

  /* guru split on OUR lane-blocked layout: element stride V; howmany rank 2 =
   * {V lanes, stride 1} x {K/V blocks, stride N*V}. OOP, split. */
  fftw_iodim64 dS = { N, V, V };
  fftw_iodim64 hS[2] = { { V, 1, 1 }, { (ptrdiff_t)(K/V), (ptrdiff_t)N*V, (ptrdiff_t)N*V } };
  fftw_plan ps = fftw_plan_guru64_split_dft(1,&dS,2,hS,sr,si,sro,sio,FFTW_PATIENT);

  printf("== RADIX=%d  N=%d  K=%zu  (%.1f MB per split buffer set) ==\n",
         RADIX, N, K, TOT*8.0*2/1048576.0);
  printf("-- FFTW guru interleaved plan --\n");  fftw_print_plan(pg); printf("\n");
  printf("-- FFTW guru split (our layout) plan --\n"); fftw_print_plan(ps); printf("\n");

  /* fill identical data everywhere */
  srand(13);
  for (size_t t=0;t<K;t++) for (int e=0;e<N;e++){
    double vr=(double)rand()/RAND_MAX-0.5, vi=(double)rand()/RAND_MAX-0.5;
    size_t bk=t/V, l=t%V, off=bk*(size_t)N*V+(size_t)e*V+l;
    xr[t*(size_t)N+e]=vr;            xi[t*(size_t)N+e]=vi;
    ir[off]=vr;                      ii[off]=vi;
    sr[off]=vr;                      si[off]=vi;
    gi[t*(size_t)N+e][0]=vr;         gi[t*(size_t)N+e][1]=vi;
  }

  /* correctness gate */
  double rF=verify(eF,ir,ii,o,oi,xr,xi,K), rL=verify(eL,ir,ii,o,oi,xr,xi,K);
  printf("verify: flat relerr %.2e %s | log3 relerr %.2e %s\n",
         rF, rF<1e-9?"OK":"BAD", rL, rL<1e-9?"OK":"BAD");
  if (rF>=1e-9 || rL>=1e-9) { printf("ABORT: correctness gate failed\n"); return 2; }

  /* timing: warmup then rdtsc min-of-60 each */
  unsigned long long aF[64],aL[64],ag[64],as[64];
  for (int w=0;w<6;w++){ eF(ir,ii,o,oi,K); eL(ir,ii,o,oi,K); fftw_execute(pg); fftw_execute_split_dft(ps,sr,si,sro,sio); }
  for (int r=0;r<60;r++){ unsigned long long c=__rdtsc(); eF(ir,ii,o,oi,K); aF[r%64]=__rdtsc()-c; }
  for (int r=0;r<60;r++){ unsigned long long c=__rdtsc(); eL(ir,ii,o,oi,K); aL[r%64]=__rdtsc()-c; }
  for (int r=0;r<60;r++){ unsigned long long c=__rdtsc(); fftw_execute(pg); ag[r%64]=__rdtsc()-c; }
  for (int r=0;r<60;r++){ unsigned long long c=__rdtsc(); fftw_execute_split_dft(ps,sr,si,sro,sio); as[r%64]=__rdtsc()-c; }
  unsigned long long F=mn(aF,60), L=mn(aL,60), G=mn(ag,60), S=mn(as,60);

  printf("cycles: t1p_flat %llu | t1p_log3 %llu | fftw_guru %llu | fftw_guru_split %llu\n",F,L,G,S);
  printf("ratios: guru/flat %.3fx  guru/log3 %.3fx  | guruSplit/flat %.3fx  | log3/flat %.3f\n",
         (double)G/F,(double)G/L,(double)S/F,(double)L/F);
  return 0;
}
