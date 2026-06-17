#define _GNU_SOURCE 1
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1     /* enable the ranged (one-call-walks-columns) terminator/stages */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <mkl_dfti.h>
#include <mkl_service.h>   /* mkl_set_num_threads — pin to 1 for a fair ST race */
/* Full ABI-typed registry (r2cf + hc2hc + RANGED variants), incl rfft.h. */
#include "rfft_registry_avx2.h"
static unsigned long long mn(unsigned long long*a,int n){unsigned long long m=~0ULL;for(int i=0;i<n;i++)if(a[i]<m)m=a[i];return m;}
int main(int argc,char**argv){
  mkl_set_num_threads(1);   /* fair single-thread race (ours is ST) */
  int N=256; size_t K=argc>1?(size_t)atoi(argv[1]):8;
  int f[2]={8,32}, nf=2; size_t halfN=N/2;
  rfft_codelets_t reg; memset(&reg,0,sizeof reg);
  rfft_register_all_avx2(&reg);   /* r2cf + hc2hc + RANGED variants (full set) */
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
