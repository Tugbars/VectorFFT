#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"
extern void radix64_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix16_t1s_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
#define N1 64
#define N2 16
#define N 1024
#define V 8
#define PADK 520
#define BLKPAD (PADK*N2)                 /* padded block size for padded-output engine */
static double twr[N1*(N2-1)],twi[N1*(N2-1)];
static double WR[N*V],WI[N*V];
static void initw(void){for(int k1=0;k1<N1;k1++)for(int n2=1;n2<N2;n2++){double a=-2.0*M_PI*(double)(n2*k1)/(double)N; twr[k1*(N2-1)+(n2-1)]=cos(a); twi[k1*(N2-1)+(n2-1)]=sin(a);}}
/* A: aliased direct write, contiguous natural order, no extra pass (the original blocked engine) */
static inline void eng_aliased(const double*ir,const double*ii,double*orr,double*oi,size_t K){
  for(size_t b=0;b<K/V;b++){const double*xr=ir+b*N*V,*xi=ii+b*N*V; double*yr=orr+b*N*V,*yi=oi+b*N*V;
    radix64_n1_oop_fwd_avx512_UG_UG(xr,xi,WR,WI,0,0,(size_t)N2*V,1,(size_t)N2*V,1,(size_t)N2*V);
    for(int k1=0;k1<N1;k1++) radix16_t1s_oop_fwd_avx512_UG_UG(WR+(size_t)N2*k1*V,WI+(size_t)N2*k1*V, yr+(size_t)k1*V,yi+(size_t)k1*V, twr+(size_t)k1*(N2-1),twi+(size_t)k1*(N2-1), (size_t)V,1,(size_t)N1*V,1,(size_t)V);
  }
}
/* B: padded direct write, NO de-pad, output in padded layout (block stride BLKPAD, k2 stride PADK) */
static inline void eng_padded(const double*ir,const double*ii,double*orr,double*oi,size_t K){
  for(size_t b=0;b<K/V;b++){const double*xr=ir+b*N*V,*xi=ii+b*N*V; double*yr=orr+b*BLKPAD,*yi=oi+b*BLKPAD;
    radix64_n1_oop_fwd_avx512_UG_UG(xr,xi,WR,WI,0,0,(size_t)N2*V,1,(size_t)N2*V,1,(size_t)N2*V);
    for(int k1=0;k1<N1;k1++) radix16_t1s_oop_fwd_avx512_UG_UG(WR+(size_t)N2*k1*V,WI+(size_t)N2*k1*V, yr+(size_t)k1*V,yi+(size_t)k1*V, twr+(size_t)k1*(N2-1),twi+(size_t)k1*(N2-1), (size_t)V,1,(size_t)PADK,1,(size_t)V);
  }
}
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  initw(); size_t K=argc>1?atol(argv[1]):512; size_t TOT=(size_t)N*K, TOTP=(size_t)BLKPAD*(K/V);
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*orr=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  double *pr=aligned_alloc(64,TOTP*8),*pi=aligned_alloc(64,TOTP*8),*xr=malloc(TOT*8),*xi=malloc(TOT*8); srand(7);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;
    size_t blk=t/V,lane=t%V; xr[t*N+e]=vr; xi[t*N+e]=vi; ir[blk*N*V+(size_t)e*V+lane]=vr; ii[blk*N*V+(size_t)e*V+lane]=vi;}
  eng_aliased(ir,ii,orr,oi,K); eng_padded(ir,ii,pr,pi,K);
  fftw_complex *fin=fftw_malloc(sizeof(fftw_complex)*N),*fout=fftw_malloc(sizeof(fftw_complex)*N);
  fftw_plan p=fftw_plan_dft_1d(N,fin,fout,FFTW_FORWARD,FFTW_ESTIMATE);
  double mA=0,mB=0;
  for(size_t t=0;t<K;t+=(K>3?K/3:1)){ size_t blk=t/V,lane=t%V;
    for(int e=0;e<N;e++){fin[e][0]=xr[t*N+e];fin[e][1]=xi[t*N+e];} fftw_execute(p);
    for(int e=0;e<N;e++){
      double a=hypot(orr[blk*N*V+(size_t)e*V+lane]-fout[e][0], oi[blk*N*V+(size_t)e*V+lane]-fout[e][1]);
      size_t pp=8*(size_t)(e%64)+PADK*(size_t)(e/64);
      double bb=hypot(pr[blk*BLKPAD+pp+lane]-fout[e][0], pi[blk*BLKPAD+pp+lane]-fout[e][1]);
      double m=hypot(fout[e][0],fout[e][1]); if(a/m>mA)mA=a/m; if(bb/m>mB)mB=bb/m;}}
  printf("K=%zu  eng_aliased err %.1e %s | eng_padded err %.1e %s\n",K,mA,mA<1e-9?"OK":"BAD",mB,mB<1e-9?"OK":"BAD");
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  for(size_t i=0;i<TOT;i++){gi[i][0]=sin(0.1*i); gi[i][1]=cos(0.07*i);}
  int nn[1]={N}; fftw_plan pe=fftw_plan_many_dft(1,nn,(int)K,gi,NULL,1,N,go,NULL,1,N,FFTW_FORWARD,FFTW_PATIENT);
  unsigned long long aa[64],ba[64],fa[64];
  for(int w=0;w<6;w++){eng_aliased(ir,ii,orr,oi,K);eng_padded(ir,ii,pr,pi,K);fftw_execute(pe);}
  for(int r=0;r<60;r++){unsigned long long c0=__rdtsc(); eng_aliased(ir,ii,orr,oi,K); aa[r%64]=__rdtsc()-c0;}
  for(int r=0;r<60;r++){unsigned long long c0=__rdtsc(); eng_padded(ir,ii,pr,pi,K); ba[r%64]=__rdtsc()-c0;}
  for(int r=0;r<60;r++){unsigned long long c0=__rdtsc(); fftw_execute(pe); fa[r%64]=__rdtsc()-c0;}
  unsigned long long A=mn(aa,60),B=mn(ba,60),F=mn(fa,60);
  printf("       aliased %llu (%.2fx) | padded %llu (%.2fx) | FFTW-oop PATIENT %llu\n",A,(double)F/A,B,(double)F/B,F);
  return 0;
}
