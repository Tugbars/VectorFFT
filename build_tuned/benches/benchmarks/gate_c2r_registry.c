#define _GNU_SOURCE 1
#define VFFT_RFFT_MAX_RADIX 32
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define DF(r) extern void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
#define DHF(r) extern void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
DF(2)DF(4)DF(8)DF(16)DF(32) DHF(2)DHF(4)DHF(8)DHF(16)
#include "rfft.h"
#include "c2r.h"
#include "c2r_registry_avx512.h"
static double rt(int N,const int*ff,int nf,size_t K,rfft_codelets_t*regF,rfft_codelets_t*regB){
  rfft_plan_t*pf=rfft_plan_create(N,K,ff,nf,regF);
  if(!pf){fprintf(stderr,"  pf NULL N=%d nf=%d\n",N,nf);return -2.0;}
  c2r_plan_t*pb=c2r_plan_create(N,K,ff,nf,regB);
  if(!pb){fprintf(stderr,"  pb NULL N=%d nf=%d\n",N,nf);rfft_plan_destroy(pf);return -3.0;}
  size_t NK=(size_t)N*K;
  double*x=calloc(NK,8),*hc=calloc(2*NK,8),*y=calloc(NK,8);
  srand(7); for(size_t i=0;i<NK;i++) x[i]=(double)rand()/RAND_MAX*2-1;
  rfft_execute_fwd_packed(pf,x,hc);
  c2r_execute_packed(pb,hc,y);
  double e=0; for(size_t i=0;i<NK;i++) e=fmax(e,fabs(y[i]-N*x[i]));
  rfft_plan_destroy(pf); c2r_plan_destroy(pb); free(x);free(hc);free(y);
  return e;
}
int main(void){
  rfft_codelets_t regF; memset(&regF,0,sizeof regF);
  regF.r2cf[2]=radix2_r2cf_avx512; regF.r2cf[4]=radix4_r2cf_avx512;
  regF.r2cf[8]=radix8_r2cf_avx512; regF.r2cf[16]=radix16_r2cf_avx512;
  regF.r2cf[32]=radix32_r2cf_avx512;
  regF.hc2hc[2]=radix2_hc2hc_dit_fwd_avx512; regF.hc2hc[4]=radix4_hc2hc_dit_fwd_avx512;
  regF.hc2hc[8]=radix8_hc2hc_dit_fwd_avx512; regF.hc2hc[16]=radix16_hc2hc_dit_fwd_avx512;
  rfft_codelets_t regB; memset(&regB,0,sizeof regB);
  /* c2r plan reuses rfft_plan_create for geometry -> needs forward codelets
     present too; the AUTO registry adds the BACKWARD codelets on top. */
  regB.r2cf[2]=radix2_r2cf_avx512; regB.r2cf[4]=radix4_r2cf_avx512;
  regB.r2cf[8]=radix8_r2cf_avx512; regB.r2cf[16]=radix16_r2cf_avx512;
  regB.r2cf[32]=radix32_r2cf_avx512;
  regB.hc2hc[2]=radix2_hc2hc_dit_fwd_avx512; regB.hc2hc[4]=radix4_hc2hc_dit_fwd_avx512;
  regB.hc2hc[8]=radix8_hc2hc_dit_fwd_avx512; regB.hc2hc[16]=radix16_hc2hc_dit_fwd_avx512;
  c2r_register_all_avx512(&regB);   /* <-- AUTO registry under test fills backward slots */
  struct{int N;int f[4];int nf;size_t K;const char*tag;} T[]={
    {16,{16,0,0,0},1,8,"(16)"},
    {16,{2,8,0,0},2,8,"(2,8)"},
    {64,{2,2,16,0},3,8,"(2,2,16)"},
    {256,{4,4,16,0},3,8,"(4,4,16)"},
    {256,{8,32,0,0},2,8,"(8,32)"},
    {256,{8,32,0,0},2,64,"(8,32)K64"},
  };
  int allpass=1;
  for(unsigned i=0;i<sizeof T/sizeof T[0];i++){
    double e=rt(T[i].N,T[i].f,T[i].nf,T[i].K,&regF,&regB);
    int pass=(e>=0&&e<1e-9); if(!pass)allpass=0;
    printf("N=%-4d %-10s K=%-3zu  maxerr=%.3e  %s\n",T[i].N,T[i].tag,T[i].K,e,
           e<-1.5?(e<-2.5?"pb-NULL":"pf-NULL"):(pass?"PASS":"FAIL"));
  }
  printf("%s\n",allpass?"== AUTO-REGISTRY c2r: ALL PASS ==":"== FAIL ==");
  return 0;
}
