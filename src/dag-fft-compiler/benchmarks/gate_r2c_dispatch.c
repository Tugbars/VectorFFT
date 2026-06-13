#define _GNU_SOURCE 1
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#define VFFT_RFFT_MAX_RADIX 32
#include "r2c_dispatch.h"
#include "../prototype/generated/registry.h"
#define DECL(r) \
  void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
DECL(2) DECL(3) DECL(4) DECL(5) DECL(7) DECL(8) DECL(16) DECL(32)
int main(void){
  rfft_codelets_t reg; memset(&reg,0,sizeof reg);
#define R(r) reg.r2cf[r]=radix##r##_r2cf_avx512; reg.hc2hc[r]=radix##r##_hc2hc_dit_fwd_avx512;
  R(2) R(3) R(4) R(5) R(7) R(8) R(16) R(32)
  vfft_proto_registry_t c2c; vfft_proto_registry_init(&c2c);
  int N=256; size_t K=8; int H=N/2;
  double *x=aligned_alloc(64,(size_t)N*K*8), *o1=aligned_alloc(64,(size_t)N*K*8+8192), *o2=aligned_alloc(64,(size_t)N*K*8);
  srand(7); for(size_t i=0;i<(size_t)N*K;i++) x[i]=(double)rand()/RAND_MAX*2-1;

  /* (a) chooser NULL have */
  int f[VFFT_RFFT_MAX_STAGES]; int nf=vfft_r2c_choose_rfft_factors(256,NULL,f,VFFT_RFFT_MAX_STAGES);
  printf("chooser(256,NULL): nf=%d factors=",nf); for(int i=0;i<nf;i++)printf("%d ",f[i]); printf("\n");

  /* (b) SPLIT requested, rfft available -> rfft natural split */
  vfft_r2c_plan_t *ps=vfft_r2c_plan_create(N,K,VFFT_R2C_SPLIT,&reg,NULL,&c2c);
  printf("SPLIT+rfft: path=%s\n", ps && ps->path==VFFT_R2C_PATH_RFFT?"RFFT(natural)":(ps?"STRIDE":"NULL"));

  /* (c) PACKED requested, NO rfft reg -> must be NULL (stride can't pack) */
  vfft_r2c_plan_t *pp=vfft_r2c_plan_create(N,K,VFFT_R2C_PACKED,NULL,NULL,&c2c);
  printf("PACKED+no-rfft: %s (expect NULL)\n", pp?"NON-NULL(BUG)":"NULL");

  /* (d) SPLIT requested, NO rfft reg -> stride fallback */
  vfft_r2c_plan_t *pf=vfft_r2c_plan_create(N,K,VFFT_R2C_SPLIT,NULL,NULL,&c2c);
  printf("SPLIT+no-rfft: path=%s (expect STRIDE)\n", pf?(pf->path==VFFT_R2C_PATH_STRIDE?"STRIDE":"RFFT"):"NULL");
  /* gate the stride fallback */
  if(pf){
    vfft_r2c_execute_fwd(pf,x,o2,o2+ (size_t)H*K); /* split: out_re, out_im — use separate buffers */
    double *re=aligned_alloc(64,(size_t)H*K*8),*im=aligned_alloc(64,(size_t)H*K*8);
    vfft_r2c_execute_fwd(pf,x,re,im);
    double maxerr=0;
    for(int fb=0;fb<H;fb++)for(size_t v=0;v<K;v++){double sr=0,si=0;for(int n=0;n<N;n++){double th=-2*M_PI*fb*n/N;sr+=x[(size_t)n*K+v]*cos(th);si+=x[(size_t)n*K+v]*sin(th);}double e=fabs(re[(size_t)fb*K+v]-sr)+fabs(im[(size_t)fb*K+v]-si);if(e>maxerr)maxerr=e;}
    printf("stride fallback SPLIT vs brute: max err %.3e %s\n",maxerr,maxerr<1e-9?"PASS":"FAIL");
  }
  return 0;
}
