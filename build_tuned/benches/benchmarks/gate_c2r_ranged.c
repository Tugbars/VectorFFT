#define _GNU_SOURCE 1
#define VFFT_RFFT_MAX_RADIX 32
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define DF(r) extern void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
#define DB(r) extern void radix##r##_r2cb_avx512(const double*,const double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
#define DHF(r) extern void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
#define DHB(r) extern void radix##r##_hc2hc_dif_bwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
#define DHBR(r) extern void radix##r##_hc2hc_dif_rng_bwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,ptrdiff_t,int,size_t);
DF(2)DF(4)DF(8)DF(16)DF(32) DB(2)DB(4)DB(8)DB(16)DB(32)
DHF(2)DHF(4)DHF(8)DHF(16) DHB(2)DHB(4)DHB(8)DHB(16) DHBR(2)DHBR(4)DHBR(8)DHBR(16)
#include "rfft.h"
#include "c2r.h"
static double rt(int N,const int*f,int nf,size_t K,rfft_codelets_t*rf,rfft_codelets_t*rb){
  rfft_plan_t*pf=rfft_plan_create(N,K,f,nf,rf);
  c2r_plan_t*pb=c2r_plan_create(N,K,f,nf,rb);
  if(!pf||!pb)return -1;
  size_t NK=(size_t)N*K; double*x=calloc(NK,8),*hc=calloc(2*NK,8),*y=calloc(NK,8);
  srand(7);for(size_t i=0;i<NK;i++)x[i]=(double)rand()/RAND_MAX*2-1;
  rfft_execute_fwd_packed(pf,x,hc); c2r_execute_packed(pb,hc,y);
  double e=0;for(size_t i=0;i<NK;i++)e=fmax(e,fabs(y[i]-N*x[i]));
  rfft_plan_destroy(pf);c2r_plan_destroy(pb);free(x);free(hc);free(y);return e;
}
int main(void){
  rfft_codelets_t rf; memset(&rf,0,sizeof rf);
  rf.r2cf[2]=radix2_r2cf_avx512;rf.r2cf[4]=radix4_r2cf_avx512;rf.r2cf[8]=radix8_r2cf_avx512;
  rf.r2cf[16]=radix16_r2cf_avx512;rf.r2cf[32]=radix32_r2cf_avx512;
  rf.hc2hc[2]=radix2_hc2hc_dit_fwd_avx512;rf.hc2hc[4]=radix4_hc2hc_dit_fwd_avx512;
  rf.hc2hc[8]=radix8_hc2hc_dit_fwd_avx512;rf.hc2hc[16]=radix16_hc2hc_dit_fwd_avx512;
  rfft_codelets_t rb; memcpy(&rb,&rf,sizeof rb);
  rb.r2cb[2]=radix2_r2cb_avx512;rb.r2cb[4]=radix4_r2cb_avx512;rb.r2cb[8]=radix8_r2cb_avx512;
  rb.r2cb[16]=radix16_r2cb_avx512;rb.r2cb[32]=radix32_r2cb_avx512;
  rb.hc2hc_dif_bwd[2]=radix2_hc2hc_dif_bwd_avx512;rb.hc2hc_dif_bwd[4]=radix4_hc2hc_dif_bwd_avx512;
  rb.hc2hc_dif_bwd[8]=radix8_hc2hc_dif_bwd_avx512;rb.hc2hc_dif_bwd[16]=radix16_hc2hc_dif_bwd_avx512;
  /* WIRE THE RANGED CODELET — this is what we're testing */
  rb.hc2hc_dif_rng_bwd[2]=radix2_hc2hc_dif_rng_bwd_avx512;
  rb.hc2hc_dif_rng_bwd[4]=radix4_hc2hc_dif_rng_bwd_avx512;
  rb.hc2hc_dif_rng_bwd[8]=radix8_hc2hc_dif_rng_bwd_avx512;
  rb.hc2hc_dif_rng_bwd[16]=radix16_hc2hc_dif_rng_bwd_avx512;
  struct{int N,f[4],nf;size_t K;const char*t;}T[]={
    {16,{2,8,0,0},2,8,"(2,8)"},{32,{2,16,0,0},2,8,"(2,16)"},
    {64,{2,2,16,0},3,8,"(2,2,16)"},{256,{4,4,16,0},3,8,"(4,4,16)"},
    {256,{8,32,0,0},2,8,"(8,32)"},{256,{8,32,0,0},2,64,"(8,32)K64"},
  };
  int all=1;
  for(unsigned i=0;i<sizeof T/sizeof T[0];i++){
    double e=rt(T[i].N,T[i].f,T[i].nf,T[i].K,&rf,&rb);
    int p=(e>=0&&e<1e-9);if(!p)all=0;
    printf("RANGED  N=%-4d %-10s K=%-3zu  err=%.3e  %s\n",T[i].N,T[i].t,T[i].K,e,e<0?"PLAN-NULL":(p?"PASS":"FAIL"));
  }
  printf("%s\n",all?"== RANGED INTERIOR: ALL PASS ==":"== FAIL ==");
  return 0;
}
