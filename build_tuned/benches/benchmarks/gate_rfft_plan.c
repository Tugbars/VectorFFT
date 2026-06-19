/* Gate for core/rfft.h: plan-driven execution vs brute packed,
 * harness cells + K variation (K=8 and K=64) + nf=1 edge. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#define VFFT_RFFT_MAX_RADIX 32
#include "rfft.h"
#define DECL(r) \
  void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
DECL(2) DECL(3) DECL(4) DECL(5) DECL(7) DECL(8) DECL(16)
void radix32_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
static void fill(rfft_codelets_t*c){
  memset(c,0,sizeof *c);
#define R(r) c->r2cf[r]=radix##r##_r2cf_avx512; c->hc2hc[r]=radix##r##_hc2hc_dit_fwd_avx512;
  R(2) R(3) R(4) R(5) R(7) R(8) R(16)
  c->r2cf[32]=radix32_r2cf_avx512;
}
static double x[512*64], out[512*64];
static int gate(const int*f,int nf,size_t K){
  rfft_codelets_t reg; fill(&reg);
  int N=1; for(int i=0;i<nf;i++) N*=f[i];
  rfft_plan_t*p=rfft_plan_create(N,K,f,nf,&reg);
  if(!p){ printf("  plan_create FAILED N=%d K=%zu\n",N,K); return 0; }
  for(size_t i=0;i<(size_t)N*K;i++) x[i]=sin(0.7*(double)i)+0.2*cos(1.9*(double)i+0.3);
  memset(out,0,sizeof out);
  rfft_execute_fwd_packed(p,x,out);
  double mx=0;
  for(size_t lane=0;lane<K;lane++)
    for(int pos=0;pos<N;pos++){
      int bin=(pos<=N/2)?pos:N-pos;
      double zr=0,zi=0;
      for(int t=0;t<N;t++){ double th=-2.0*M_PI*(double)bin*t/N;
        zr+=x[(size_t)t*K+lane]*cos(th); zi+=x[(size_t)t*K+lane]*sin(th); }
      double ex=(pos<=N/2)?zr:zi;
      double d=fabs(ex-out[(size_t)pos*K+lane]); if(d>mx)mx=d;
    }
  char fs[64]; int o=0;
  for(int i=0;i<nf;i++) o+=snprintf(fs+o,sizeof fs-o,"%d%s",f[i],i<nf-1?",":"");
  printf("  N=%-4d (%-10s) K=%-3zu: %.2e %s\n",N,fs,K,mx,mx<1e-12?"PASS":"FAIL");
  rfft_plan_destroy(p);
  return mx<1e-12;
}
int main(void){
  int ok=1;
  { int f[]={16};       ok&=gate(f,1,8);  }
  { int f[]={8,32};     ok&=gate(f,2,8);  }
  { int f[]={16,16};    ok&=gate(f,2,64); }  /* nf=1 pure leaf */
  { int f[]={4,4};      ok&=gate(f,2,8);  }
  { int f[]={2,4,4};    ok&=gate(f,3,8);  }
  { int f[]={4,4,4};    ok&=gate(f,3,8);  }
  { int f[]={2,4,4,4};  ok&=gate(f,4,8);  }
  { int f[]={4,4,16};   ok&=gate(f,3,8);  }
  { int f[]={2,4,4,8};  ok&=gate(f,4,8);  }
  { int f[]={4,5};      ok&=gate(f,2,8);  }
  { int f[]={7,3,5};    ok&=gate(f,3,8);  }
  { int f[]={4,4,16};   ok&=gate(f,3,64); }  /* K variation */
  { int f[]={2,4,4,8};  ok&=gate(f,4,64); }
  { int f[]={2,3,2};    ok&=gate(f,3,64); }
  printf(ok?"ALL PASS\n":"FAILURES\n");
  return ok?0:1;
}
