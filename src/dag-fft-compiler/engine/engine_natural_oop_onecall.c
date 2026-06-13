/*
 * engine_natural_oop_onecall.c
 *
 * Fastest natural-order out-of-place engine for N=1024: balanced 32x32 one-call,
 * with a log3 twiddle stage. Each twiddle stage is a SINGLE codelet call over the
 * whole block (me = positions*lanes), enabled by t1p (per-position broadcast).
 *
 * blk32_log3 = 32x32, RECOMMENDED. Stage 2 (radix-32 twiddle) uses log3: it loads
 *   only the 5 base twiddles W^{1,2,4,8,16} per position (slots 0,1,3,7,15) and
 *   derives the other 26 by complex multiply, instead of broadcasting all 31.
 * blk32_flat = 32x32 with the flat (all-31-loads) twiddle stage, for contrast.
 * Both natural order, input preserved, machine precision (relerr ~8e-15). The
 * twiddle table is IDENTICAL for both (log3 reads a sparse subset of the same
 * slots), so only the stage-2 codelet differs.
 *
 * Measured (AVX-512, this VM, FFTW PATIENT, rdtsc min-of-60, interleaved):
 *   32x32 log3 : ~1.03 to 1.12x FFTW   <- best, and ~2 to 4% over flat (log3/flat
 *                ~0.94 to 0.99 across K=128/256/512/1024, both runs).
 *   32x32 flat : ~1.01 to 1.07x FFTW.
 * log3 wins because the radix-32 twiddle stage spends real time on 31 set1_pd
 * broadcast twiddle loads per position; trading 26 of them for FMAs (on otherwise
 * underused FMA ports) nets out faster, even though it adds ~52 mul/fma ops. The
 * compiler has special log3 inline-asm scheduling for R<=32 on AVX-512.
 *
 * 2-factor sweep context (all one-call, natural order): 32x32 (balanced) beats
 * 16x64 (1.01-1.04x) and 64x16 (~1.02x); see docs/OOP_DESIGN.md section 6. log3
 * then adds another ~2-4% on top of the balanced 32x32. Numbers are directional
 * on a noisy VM; the rankings (32x32 > 16x64 ~ 64x16 > FFTW, and log3 > flat) hold
 * across K. The log3 codelet is radix32_t1p_oop_fwd_avx512_UG_UG_log3, generated
 * with `gen_radix 32 --oop --oop-buffer-oop --oop-load UG --oop-store UG
 * --twiddled-pos --log3 --emit-c --isa avx512`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"
extern void radix32_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix32_t1p_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix32_t1p_oop_fwd_avx512_UG_UG_log3(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
#define N 1024
#define V 8
static double Q2r[31*32],Q2i[31*32];   /* same table for both: tw[(l2-1)*32+k2]=W_1024^{l2 k2} */
static void initw(void){ for(int l2=1;l2<32;l2++)for(int k2=0;k2<32;k2++){double a=-2.0*M_PI*(double)(l2*k2)/1024.0; Q2r[(l2-1)*32+k2]=cos(a);Q2i[(l2-1)*32+k2]=sin(a);} }
/* 32x32, flat t1p twiddle stage (current best) */
static inline void blkF(const double*ir,const double*ii,double*o,double*oi){
  radix32_n1_oop_fwd_avx512_UG_UG(ir,ii,o,oi,0,0,(size_t)32*V,1,(size_t)V,32,(size_t)32*V);
  radix32_t1p_oop_fwd_avx512_UG_UG(o,oi,o,oi,Q2r,Q2i,(size_t)32*V,1,(size_t)32*V,1,(size_t)32*V);
}
/* 32x32, log3 t1p twiddle stage (5 base loads + derive) */
static inline void blkL(const double*ir,const double*ii,double*o,double*oi){
  radix32_n1_oop_fwd_avx512_UG_UG(ir,ii,o,oi,0,0,(size_t)32*V,1,(size_t)V,32,(size_t)32*V);
  radix32_t1p_oop_fwd_avx512_UG_UG_log3(o,oi,o,oi,Q2r,Q2i,(size_t)32*V,1,(size_t)32*V,1,(size_t)32*V);
}
static void eF(const double*ir,const double*ii,double*o,double*oi,size_t K){for(size_t b=0;b<K/V;b++)blkF(ir+b*N*V,ii+b*N*V,o+b*N*V,oi+b*N*V);}
static void eL(const double*ir,const double*ii,double*o,double*oi,size_t K){for(size_t b=0;b<K/V;b++)blkL(ir+b*N*V,ii+b*N*V,o+b*N*V,oi+b*N*V);}
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
static double verify(void(*eng)(const double*,const double*,double*,double*,size_t),const double*ir,const double*ii,double*o,double*oi,const double*xr,const double*xi,size_t K){
  eng(ir,ii,o,oi,K); fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*N),*fo=fftw_malloc(sizeof(fftw_complex)*N);
  fftw_plan p=fftw_plan_dft_1d(N,fi,fo,FFTW_FORWARD,FFTW_ESTIMATE); double mr=0;
  for(size_t t=0;t<K;t+=(K>3?K/3:1)){size_t bk=t/V,l=t%V;for(int e=0;e<N;e++){fi[e][0]=xr[t*N+e];fi[e][1]=xi[t*N+e];}fftw_execute(p);
    double me=0,mm=0;for(int k=0;k<N;k++){double dr=o[bk*N*V+(size_t)k*V+l]-fo[k][0],di=oi[bk*N*V+(size_t)k*V+l]-fo[k][1];double e=sqrt(dr*dr+di*di),m=hypot(fo[k][0],fo[k][1]);if(e>me)me=e;if(m>mm)mm=m;}if(me/mm>mr)mr=me/mm;}
  fftw_destroy_plan(p);fftw_free(fi);fftw_free(fo);return mr;
}
int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  initw(); size_t K=argc>1?atol(argv[1]):512, TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8),*xi=malloc(TOT*8); srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;size_t bk=t/V,l=t%V;xr[t*N+e]=vr;xi[t*N+e]=vi;ir[bk*N*V+(size_t)e*V+l]=vr;ii[bk*N*V+(size_t)e*V+l]=vi;}
  double rF=verify(eF,ir,ii,o,oi,xr,xi,K), rL=verify(eL,ir,ii,o,oi,xr,xi,K);
  printf("K=%zu  flat relerr %.2e %s   log3 relerr %.2e %s\n",K,rF,rF<1e-9?"OK":"BAD",rL,rL<1e-9?"OK":"BAD");
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  for(size_t i=0;i<TOT;i++){gi[i][0]=sin(0.1*i);gi[i][1]=cos(0.07*i);}
  int nn[1]={N};fftw_plan pe=fftw_plan_many_dft(1,nn,(int)K,gi,NULL,1,N,go,NULL,1,N,FFTW_FORWARD,FFTW_PATIENT);
  unsigned long long aF[64],aL[64],ap[64];
  for(int w=0;w<6;w++){eF(ir,ii,o,oi,K);eL(ir,ii,o,oi,K);fftw_execute(pe);}
  for(int r=0;r<60;r++){unsigned long long c=__rdtsc();eF(ir,ii,o,oi,K);aF[r%64]=__rdtsc()-c;}
  for(int r=0;r<60;r++){unsigned long long c=__rdtsc();eL(ir,ii,o,oi,K);aL[r%64]=__rdtsc()-c;}
  for(int r=0;r<60;r++){unsigned long long c=__rdtsc();fftw_execute(pe);ap[r%64]=__rdtsc()-c;}
  unsigned long long F=mn(aF,60),L=mn(aL,60),P=mn(ap,60);
  printf("  32x32 flat %llu (%.2fx)  |  32x32 log3 %llu (%.2fx)  |  FFTW %llu  [log3/flat=%.3f]\n",F,(double)P/F,L,(double)P/L,P,(double)L/F);
  return 0;
}
