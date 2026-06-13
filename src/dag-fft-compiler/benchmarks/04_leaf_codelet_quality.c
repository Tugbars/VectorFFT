#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"
extern void radix16_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix64_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
static void bench(int R, void(*kern)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t)){
  size_t K=4096, sz=(size_t)R*K;
  double *ar=aligned_alloc(64,sz*8),*ai=aligned_alloc(64,sz*8),*cr=aligned_alloc(64,sz*8),*ci=aligned_alloc(64,sz*8);
  for(size_t i=0;i<sz;i++){ar[i]=(i%101)-50.0;ai[i]=(i%97)-48.0;}
  /* FFTW: K size-R transforms, out-of-place, interleaved (transform-major) */
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*sz),*go=fftw_malloc(sizeof(fftw_complex)*sz);
  for(size_t i=0;i<sz;i++){gi[i][0]=ar[i];gi[i][1]=ai[i];}
  int nn[1]={R}; fftw_plan p=fftw_plan_many_dft(1,nn,(int)K,gi,NULL,1,R,go,NULL,1,R,FFTW_FORWARD,FFTW_PATIENT);
  unsigned long long me[64],fe[64]; int it=2000;
  for(int w=0;w<8;w++) kern(ar,ai,cr,ci,0,0,K,1,K,1,K);
  for(int r=0;r<it;r++){unsigned long long t0=__rdtsc(); kern(ar,ai,cr,ci,0,0,K,1,K,1,K); me[r%64]=__rdtsc()-t0;}
  for(int w=0;w<8;w++) fftw_execute(p);
  for(int r=0;r<it;r++){unsigned long long t0=__rdtsc(); fftw_execute(p); fe[r%64]=__rdtsc()-t0;}
  unsigned long long M=mn(me,it<64?it:64),F=mn(fe,it<64?it:64);
  printf("size-%-2d no-twiddle, %zu transforms:  VFFT-OOP %llu  |  FFTW %llu  |  VFFT is %.2fx FFTW\n",R,K,M,F,(double)F/M);
}
int main(void){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  bench(16,radix16_n1_oop_fwd_avx512_UG_UG);
  bench(64,radix64_n1_oop_fwd_avx512_UG_UG);
  return 0;
}
