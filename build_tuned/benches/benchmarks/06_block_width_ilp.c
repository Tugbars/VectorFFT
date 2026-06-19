#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"
extern void radix64_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix16_t1s_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
#define N1 64
#define N2 16
#define N 1024
#ifndef V
#define V 8
#endif
static double twr[N1*(N2-1)],twi[N1*(N2-1)];
static double WR[N*V],WI[N*V];           /* block = V*N*16 bytes; V=8:128KB V=16:256KB V=32:512KB (all < L2 1MB) */
static void initw(void){for(int k1=0;k1<N1;k1++)for(int n2=1;n2<N2;n2++){double a=-2.0*M_PI*(double)(n2*k1)/(double)N; twr[k1*(N2-1)+(n2-1)]=cos(a); twi[k1*(N2-1)+(n2-1)]=sin(a);}}
static inline void eng(const double*ir,const double*ii,double*orr,double*oi,size_t K){
  for(size_t b=0;b<K/V;b++){const double*xr=ir+b*N*V,*xi=ii+b*N*V; double*yr=orr+b*N*V,*yi=oi+b*N*V;
    radix64_n1_oop_fwd_avx512_UG_UG(xr,xi,WR,WI,0,0,(size_t)N2*V,1,(size_t)N2*V,1,(size_t)N2*V);
    for(int k1=0;k1<N1;k1++) radix16_t1s_oop_fwd_avx512_UG_UG(WR+(size_t)N2*k1*V,WI+(size_t)N2*k1*V, yr+(size_t)k1*V,yi+(size_t)k1*V, twr+(size_t)k1*(N2-1),twi+(size_t)k1*(N2-1), (size_t)V,1,(size_t)N1*V,1,(size_t)V);
  }
}
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  initw(); size_t K=argc>1?atol(argv[1]):512; size_t TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*orr=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  for(size_t i=0;i<TOT;i++){ir[i]=sin(0.1*i);ii[i]=cos(0.07*i);}
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  for(size_t i=0;i<TOT;i++){gi[i][0]=sin(0.1*i);gi[i][1]=cos(0.07*i);}
  int nn[1]={N}; fftw_plan pe=fftw_plan_many_dft(1,nn,(int)K,gi,NULL,1,N,go,NULL,1,N,FFTW_FORWARD,FFTW_PATIENT);
  unsigned long long va[64],fa[64];
  for(int w=0;w<6;w++){eng(ir,ii,orr,oi,K);fftw_execute(pe);}
  for(int r=0;r<60;r++){unsigned long long c0=__rdtsc(); eng(ir,ii,orr,oi,K); va[r%64]=__rdtsc()-c0;}
  for(int r=0;r<60;r++){unsigned long long c0=__rdtsc(); fftw_execute(pe); fa[r%64]=__rdtsc()-c0;}
  unsigned long long Vc=mn(va,60),F=mn(fa,60);
  printf("V=%d  K=%zu  outer me=%d (%d ZMMs)  engine %llu  | FFTW %llu | %.2fx\n",V,K,V,V/8,Vc,F,(double)F/Vc);
  return 0;
}
