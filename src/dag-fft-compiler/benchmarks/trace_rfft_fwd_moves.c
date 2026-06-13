#define _GNU_SOURCE 1
#define VFFT_RFFT_TRACE 1
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
extern void radix8_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
extern void radix2_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
extern void radix2_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
#include "rfft_trace.h"
rfft_move_t g_rfft_moves[4096]; int g_rfft_nmoves=0;
int main(void){
  int N=16; size_t K=8;
  rfft_codelets_t reg; memset(&reg,0,sizeof reg);
  reg.r2cf[8]=radix8_r2cf_avx512; reg.r2cf[2]=radix2_r2cf_avx512;
  reg.hc2hc[2]=radix2_hc2hc_dit_fwd_avx512;
  int f[2]={2,8};
  rfft_plan_t*p=rfft_plan_create(N,K,f,2,&reg);
  if(!p){printf("plan NULL\n");return 1;}
  double x[16*8], outp[16*8];
  srand(7); for(int i=0;i<N*(int)K;i++) x[i]=(double)rand()/RAND_MAX*2-1;
  rfft_execute_fwd_packed(p, x, outp);
  printf("=== forward moves (N=16=(2,8)), offsets in K-units (/K) ===\n");
  for(int i=0;i<g_rfft_nmoves;i++){
    rfft_move_t*m=&g_rfft_moves[i];
    printf("%-4s d=%d r=%d m=%d k=%d Q=%zu | in_re=%ld in_im=%ld -> out_re=%ld out_im=%ld | is=%ld os=%ld\n",
      m->op,m->d,m->r,m->m,m->k,m->Q,
      m->ire/(long)K, m->iim<0?-1:m->iim/(long)K,
      m->ore/(long)K, m->oim<0?-1:m->oim/(long)K,
      m->is/(long)K, m->os/(long)K);
  }
  return 0;
}
