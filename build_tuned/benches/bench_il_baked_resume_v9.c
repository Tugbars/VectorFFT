/* v9: item 3 — generic vs baked resume on the axis-0 pass, same binary. */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include <math.h>
#include "prime_dispatch.h"
#include "exhaustive_plan.h"
#include "il_layout.h"
#include "il_execute.h"
#include "generator/generated/registry.h"
static stride_plan_t *P0,*P1;
static const size_t K2=4096, NTOT=262144;
#define TIME(dst,stmt) { double _b=1e18; for(int _t=0;_t<7;_t++){ double _0=(double)__rdtsc(); for(int _r=0;_r<6;_r++){ stmt; } double _v=((double)__rdtsc()-_0)/6; if(_v<_b)_b=_v;} dst=_b; }
int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    P0=vfft_proto_exhaustive_plan(64,K2,&reg,0);
    P1=vfft_proto_exhaustive_plan(64,64,&reg,0);
    size_t n=NTOT;
    double *z=aligned_alloc(64,n*16),*z0=aligned_alloc(64,n*16),*z1=aligned_alloc(64,n*16);
    double *r0=aligned_alloc(64,n*8),*i0=aligned_alloc(64,n*8);
    double *r1=aligned_alloc(64,n*8),*i1=aligned_alloc(64,n*8);
    srand(3); for(size_t i=0;i<2*n;i++) z[i]=2.0*rand()/RAND_MAX-1;
    printf("baked lookup P0 f/b, P1 f/b: %s/%s %s/%s\n",
        _vfft_proto_lookup_fwd(P0)?"ok":"NULL", _vfft_proto_lookup_bwd(P0)?"ok":"NULL",
        _vfft_proto_lookup_fwd(P1)?"ok":"NULL", _vfft_proto_lookup_bwd(P1)?"ok":"NULL");
    vfft_proto_execute_fwd_ilin_ex(P0,z,r0,i0,K2,0);
    vfft_proto_execute_fwd_ilin_ex(P0,z,r1,i1,K2,1);
    printf("fwd ilin baked vs generic  : %s\n",
        (memcmp(r0,r1,n*8)||memcmp(i0,i1,n*8))?"**FAIL**":"BIT-EXACT");
    memcpy(r0,r1,n*8); memcpy(i0,i1,n*8);
    vfft_proto_execute_bwd_ilout_ex(P0,r0,i0,z0,K2,0);
    vfft_proto_execute_bwd_ilout_ex(P0,r1,i1,z1,K2,1);
    { double mx=0,sc=0;
      for(size_t i=0;i<2*n;i++){ double d=fabs(z0[i]-z1[i]); if(d>mx)mx=d;
          double a=fabs(z0[i]); if(a>sc)sc=a; }
      double rel = sc>0 ? mx/sc : mx;
      printf("bwd ilout baked vs generic : max rel %.2e %s (kernels differ: t1s vs t1)\n",
          rel, rel<1e-12?"OK":"**FAIL**"); }
    double tg,tb,bg,bb;
    TIME(tg, vfft_proto_execute_fwd_ilin_ex(P0,z,r0,i0,K2,0));
    TIME(tb, vfft_proto_execute_fwd_ilin_ex(P0,z,r0,i0,K2,1));
    TIME(bg, vfft_proto_execute_bwd_ilout_ex(P0,r1,i1,z1,K2,0));
    TIME(bb, vfft_proto_execute_bwd_ilout_ex(P0,r1,i1,z1,K2,1));
    printf("AXIS0 fwd  generic %9.0f | baked %9.0f | delta %+.0f (%.3fx)\n",tg,tb,tb-tg,tb/tg);
    printf("AXIS0 bwd  generic %9.0f | baked %9.0f | delta %+.0f (%.3fx)\n",bg,bb,bb-bg,bb/bg);
    return 0;
}
