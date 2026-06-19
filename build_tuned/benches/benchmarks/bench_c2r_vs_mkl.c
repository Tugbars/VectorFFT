#define _GNU_SOURCE 1
#define VFFT_RFFT_MAX_RADIX 32
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <mkl_dfti.h>
#define DF(r) extern void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
#define DHF(r) extern void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
DF(2)DF(4)DF(8)DF(16)DF(32) DHF(2)DHF(4)DHF(8)DHF(16)
#include "rfft.h"
#include "c2r.h"
#include "c2r_registry_avx512.h"
static unsigned long long mn(unsigned long long*a,int n){unsigned long long m=~0ULL;for(int i=0;i<n;i++)if(a[i]<m)m=a[i];return m;}
int main(int argc,char**argv){
  int N=argc>1?atoi(argv[1]):256;
  size_t K=argc>2?(size_t)atoi(argv[2]):8;   // our batch (lane-interleaved)
  int f[4],nf;
  if(N==256){f[0]=8;f[1]=32;nf=2;} else if(N==64){f[0]=2;f[1]=2;f[2]=16;nf=3;}
  else if(N==32){f[0]=2;f[1]=16;nf=2;} else {f[0]=N;nf=1;}
  rfft_codelets_t reg; memset(&reg,0,sizeof reg);
  reg.r2cf[2]=radix2_r2cf_avx512;reg.r2cf[4]=radix4_r2cf_avx512;reg.r2cf[8]=radix8_r2cf_avx512;
  reg.r2cf[16]=radix16_r2cf_avx512;reg.r2cf[32]=radix32_r2cf_avx512;
  reg.hc2hc[2]=radix2_hc2hc_dit_fwd_avx512;reg.hc2hc[4]=radix4_hc2hc_dit_fwd_avx512;
  reg.hc2hc[8]=radix8_hc2hc_dit_fwd_avx512;reg.hc2hc[16]=radix16_hc2hc_dit_fwd_avx512;
  c2r_register_all_avx512(&reg);
  rfft_plan_t*pf=rfft_plan_create(N,K,f,nf,&reg);
  c2r_plan_t*pb=c2r_plan_create(N,K,f,nf,&reg);
  if(!pf||!pb){printf("plan NULL N=%d\n",N);return 1;}
  size_t NK=(size_t)N*K, halfN=N/2;
  double*x=calloc(NK,8),*hc=calloc(2*NK,8),*y=calloc(NK,8);
  srand(7);for(size_t i=0;i<NK;i++)x[i]=(double)rand()/RAND_MAX*2-1;
  rfft_execute_fwd_packed(pf,x,hc);

  // MKL: K transforms in TRANSFORM-MAJOR layout (its native batch).
  DFTI_DESCRIPTOR_HANDLE h=0;
  DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
  DftiSetValue(h,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
  DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
  DftiSetValue(h,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
  DftiSetValue(h,DFTI_INPUT_DISTANCE,(MKL_LONG)(halfN+1));
  DftiSetValue(h,DFTI_OUTPUT_DISTANCE,(MKL_LONG)N);
  DftiCommitDescriptor(h);
  // transform-major CCE: transform t at t*(halfN+1)
  double*cce=calloc((halfN+1)*K*2,8),*mout=calloc(NK,8);
  for(size_t t=0;t<K;t++) for(size_t kk=0;kk<=halfN;kk++){
    double sr=0,si=0; for(int n=0;n<N;n++){double th=-2*M_PI*kk*n/N; sr+=x[n*K+t]*cos(th); si+=x[n*K+t]*sin(th);}
    cce[(t*(halfN+1)+kk)*2]=sr; cce[(t*(halfN+1)+kk)*2+1]=si;
  }
  // correctness of MKL opponent (transform-major out: transform t at t*N)
  DftiComputeBackward(h,cce,mout);
  double emkl=0; for(size_t t=0;t<K;t++) for(int n=0;n<N;n++) emkl=fmax(emkl,fabs(mout[t*N+n]-N*x[n*K+t]));

  const int IT=120; unsigned long long ours[120],mkl[120];
  c2r_execute_packed(pb,hc,y); DftiComputeBackward(h,cce,mout);
  for(int r=0;r<IT;r++){unsigned long long c;
    c=__rdtsc(); c2r_execute_packed(pb,hc,y);     ours[r]=__rdtsc()-c;
    c=__rdtsc(); DftiComputeBackward(h,cce,mout);  mkl[r]=__rdtsc()-c;}
  unsigned long long o=mn(ours,IT),m=mn(mkl,IT);
  printf("N=%-4d K=%-3zu nf=%d | mkl_correct=%.1e | ours=%llu mkl=%llu | mkl/ours=%.3f %s\n",
    N,K,nf,emkl,o,m,(double)m/o,(double)m/o>1?"OURS faster":"mkl faster");
  DftiFreeDescriptor(&h);
  return 0;
}
