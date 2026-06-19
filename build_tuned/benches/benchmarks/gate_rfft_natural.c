#define _GNU_SOURCE 1
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#define VFFT_RFFT_MAX_RADIX 32
#include "rfft.h"
#define DECL(r) \
  void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2c_nat_fwd_avx512(const double*,const double*,double*,double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
DECL(2) DECL(3) DECL(4) DECL(5) DECL(7) DECL(8) DECL(16)
void radix32_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
static double x[256*64], pout[256*64], nre[129*64], nim[129*64];
static double rre[129*64], rim[129*64];
static int gate(const int*f,int nf,size_t K){
  rfft_codelets_t c; memset(&c,0,sizeof c);
#define R(r) c.r2cf[r]=radix##r##_r2cf_avx512; c.hc2hc[r]=radix##r##_hc2hc_dit_fwd_avx512; c.hc2c[r]=radix##r##_hc2c_nat_fwd_avx512;
  R(2) R(3) R(4) R(5) R(7) R(8) R(16)
  c.r2cf[32]=radix32_r2cf_avx512;
  int N=1; for(int i=0;i<nf;i++) N*=f[i];
  for(int i=0;i<N*(int)K;i++) x[i]=sin(0.37*i)+0.2*cos(1.1*i);
  rfft_plan_t*p=rfft_plan_create(N,K,f,nf,&c);
  if(!p){printf("plan FAIL\n");return 0;}
  rfft_execute_fwd_packed(p,x,pout);
  /* unpack packed -> natural reference */
  size_t nh=N/2;
  memcpy(rre,pout,K*8); memset(rim,0,K*8);
  for(size_t q=1;q<(size_t)((N+1)/2);q++){
    memcpy(rre+q*K,pout+q*K,K*8);
    memcpy(rim+q*K,pout+((size_t)N-q)*K,K*8);
  }
  if(N%2==0){ memcpy(rre+nh*K,pout+nh*K,K*8); memset(rim+nh*K,0,K*8); }
  memset(nre,0xAA,sizeof nre); memset(nim,0xAA,sizeof nim);
  rfft_execute_fwd_natural(p,x,nre,nim);
  double err=0;
  for(size_t q=0;q<=nh;q++)for(size_t v=0;v<K;v++){
    double e1=fabs(nre[q*K+v]-rre[q*K+v]);
    double e2=fabs(nim[q*K+v]-rim[q*K+v]);
    if(e1>err)err=e1; if(e2>err)err=e2;
  }
  rfft_plan_destroy(p);
  printf("  N=%-4d (",N);
  for(int i=0;i<nf;i++)printf("%d%s",f[i],i+1<nf?",":"");
  printf(") K=%-3zu nat: %.2e %s\n",K,err,err<5e-11?"PASS":"FAIL");
  return err<5e-11;
}
int main(void){
  int ok=1;
  { int f[]={16};       ok&=gate(f,1,8);  }
  { int f[]={16,16};    ok&=gate(f,2,64); }
  { int f[]={8,32};     ok&=gate(f,2,8);  }
  { int f[]={4,4};      ok&=gate(f,2,8);  }
  { int f[]={2,4,4};    ok&=gate(f,3,8);  }
  { int f[]={4,4,4};    ok&=gate(f,3,8);  }
  { int f[]={2,4,4,4};  ok&=gate(f,4,8);  }
  { int f[]={4,4,16};   ok&=gate(f,3,64); }
  { int f[]={2,4,4,8};  ok&=gate(f,4,8);  }
  { int f[]={4,5};      ok&=gate(f,2,8);  }
  { int f[]={7,3,5};    ok&=gate(f,3,8);  }
  { int f[]={2,3,2};    ok&=gate(f,3,64); }
  printf(ok?"ALL PASS\n":"FAILURES\n");
  return ok?0:1;
}
