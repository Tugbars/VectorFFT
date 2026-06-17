#define _GNU_SOURCE 1
#define VFFT_RFFT_MAX_RADIX 32
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <mkl_dfti.h>
#define DF(r) extern void radix##r##_r2cf_avx2(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
#define DHF(r) extern void radix##r##_hc2hc_dit_fwd_avx2(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
DF(2)DF(4)DF(8)DF(16)DF(32) DHF(2)DHF(4)DHF(8)DHF(16)
#include "rfft.h"
static unsigned long long mn(unsigned long long*a,int n){unsigned long long m=~0ULL;for(int i=0;i<n;i++)if(a[i]<m)m=a[i];return m;}
int main(int argc,char**argv){
  int N=256; size_t K=argc>1?(size_t)atoi(argv[1]):8;
  int f[2]={8,32}, nf=2; size_t halfN=N/2;
  rfft_codelets_t reg; memset(&reg,0,sizeof reg);
  reg.r2cf[2]=radix2_r2cf_avx2;reg.r2cf[4]=radix4_r2cf_avx2;reg.r2cf[8]=radix8_r2cf_avx2;
  reg.r2cf[16]=radix16_r2cf_avx2;reg.r2cf[32]=radix32_r2cf_avx2;
  reg.hc2hc[2]=radix2_hc2hc_dit_fwd_avx2;reg.hc2hc[4]=radix4_hc2hc_dit_fwd_avx2;
  reg.hc2hc[8]=radix8_hc2hc_dit_fwd_avx2;reg.hc2hc[16]=radix16_hc2hc_dit_fwd_avx2;
  rfft_plan_t*pf=rfft_plan_create(N,K,f,nf,&reg);
  if(!pf){printf("plan NULL\n");return 1;}
  size_t NK=(size_t)N*K;
  double*x=calloc(NK,8),*hc=calloc(2*NK,8);
  srand(7);for(size_t i=0;i<NK;i++)x[i]=(double)rand()/RAND_MAX*2-1;
  // MKL forward real (r2c), transform-major
  DFTI_DESCRIPTOR_HANDLE h=0;
  DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
  DftiSetValue(h,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
  DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
  DftiSetValue(h,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
  DftiSetValue(h,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
  DftiSetValue(h,DFTI_OUTPUT_DISTANCE,(MKL_LONG)(halfN+1));
  DftiCommitDescriptor(h);
  double*xin=calloc(NK,8),*cce=calloc((halfN+1)*K*2,8);
  for(size_t t=0;t<K;t++)for(int n=0;n<N;n++)xin[t*N+n]=x[n*K+t];
  const int IT=120; unsigned long long ours[120],mkl[120];
  rfft_execute_fwd_packed(pf,x,hc); DftiComputeForward(h,xin,cce);
  for(int r=0;r<IT;r++){unsigned long long c;
    c=__rdtsc(); rfft_execute_fwd_packed(pf,x,hc);   ours[r]=__rdtsc()-c;
    c=__rdtsc(); DftiComputeForward(h,xin,cce);        mkl[r]=__rdtsc()-c;}
  unsigned long long o=mn(ours,IT),m=mn(mkl,IT);
  printf("FWD N=256 (8,32) K=%-3zu | ours=%llu mkl=%llu | mkl/ours=%.3f %s\n",
    K,o,m,(double)m/o,(double)m/o>1?"OURS faster":"mkl faster");
  DftiFreeDescriptor(&h); return 0;
}
