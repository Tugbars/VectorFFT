/* dif_variant_map.c — map EXACTLY which twiddle-application executor macros are me-unsafe (partial-batch
 * wrong on asymmetric input). For each (orientation, variant-combo) run _c2c_mt's 4-slab split (me=8)
 * SEQUENTIALLY vs the full batch on RAND input. Which stage carries the twiddle:
 *   DIT: twiddle on stages 1..nf-1 (stage 0 needs_tw=0). variant of the twiddled stage = the tested macro.
 *   DIF: twiddle on stages 0..nf-2 (last stage needs_tw=0). variant of the twiddled stage = the tested macro.
 * Build: python build.py --src test/dif_variant_map.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "executor.h"
#include "planner.h"
#include "generator/generated/registry.h"

static double maxd(const double *a,const double *b,const double *c,const double *d,size_t n){
    double m=0; for(size_t i=0;i<n;i++){double e=fabs(a[i]-c[i])+fabs(b[i]-d[i]); if(e>m)m=e;} return m; }

static void probe(vfft_proto_registry_t *reg, const char *tag, int N, size_t K,
                  const int *fac, const int *var, int nf, int dif) {
    stride_plan_t *p = vfft_proto_plan_create_ex(N, K, fac, var, nf, dif, reg);
    if (!p) { printf("  %-22s plan NULL\n", tag); return; }
    size_t tot=(size_t)N*K;
    double *xr=malloc(tot*8),*xi=malloc(tot*8),*ar=malloc(tot*8),*ai=malloc(tot*8),*br=malloc(tot*8),*bi=malloc(tot*8);
    srand(11+N+(int)K); for(size_t i=0;i<tot;i++){ xr[i]=(double)rand()/RAND_MAX-0.5; xi[i]=(double)rand()/RAND_MAX-0.5; }
    memcpy(ar,xr,tot*8); memcpy(ai,xi,tot*8); vfft_proto_execute_fwd(p,ar,ai,K);          /* full */
    int T=8; size_t S=(((K+(size_t)T-1)/(size_t)T)+7)&~(size_t)7;
    memcpy(br,xr,tot*8); memcpy(bi,xi,tot*8);
    for(size_t k0=0;k0<K;k0+=S){ size_t me=(k0+S>K)?K-k0:S; vfft_proto_execute_fwd(p,br+k0,bi+k0,me); }
    double dfwd=maxd(ar,ai,br,bi,tot);
    /* roundtrip via split fwd + split bwd */
    memcpy(br,xr,tot*8); memcpy(bi,xi,tot*8);
    for(size_t k0=0;k0<K;k0+=S){ size_t me=(k0+S>K)?K-k0:S; vfft_proto_execute_fwd(p,br+k0,bi+k0,me); }
    for(size_t k0=0;k0<K;k0+=S){ size_t me=(k0+S>K)?K-k0:S; vfft_proto_execute_bwd(p,br+k0,bi+k0,me); }
    double drt=0,sc=0; for(size_t i=0;i<tot;i++){ if(fabs(xr[i])>sc)sc=fabs(xr[i]); double e=fabs(br[i]/N-xr[i])+fabs(bi[i]/N-xi[i]); if(e>drt)drt=e; } drt/=(sc>0?sc:1);
    printf("  %-22s fwd_split=%.1e  roundtrip_split=%.1e   %s\n", tag, dfwd, drt,
           (dfwd<1e-12&&drt<1e-9)?"ok":"*** UNSAFE");
    free(xr);free(xi);free(ar);free(ai);free(br);free(bi);
}
int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("# me-safety map: seq 4-slab(me=8) split vs full, rand input (generic path):\n");
    int f2[]={4,32};
    int vFF[]={0,0}, vLF[]={1,0}, vFL[]={0,1}, vLL[]={1,1};
    /* DIF: twiddle on stage 0 (radix 4). stage-0 variant = tested macro. */
    probe(&reg,"DIF 4.32 [FLAT,*]",128,32,f2,vFF,2,1);   /* DIF_FLAT */
    probe(&reg,"DIF 4.32 [LOG3,*]",128,32,f2,vLF,2,1);   /* DIF_LOG3 */
    /* DIT: twiddle on stage 1 (radix 32). stage-1 variant = tested macro. */
    probe(&reg,"DIT 4.32 [*,FLAT]",128,32,f2,vFF,2,0);   /* DIT_FLAT (known safe) */
    probe(&reg,"DIT 4.32 [*,LOG3]",128,32,f2,vFL,2,0);   /* DIT_LOG3 */
    /* 3-stage DIF: twiddle on stages 0,1. */
    int f3[]={4,4,32}; int v3LLF[]={1,1,0}, v3FFF[]={0,0,0};
    probe(&reg,"DIF 4.4.32 [F,F,*]",512,32,f3,v3FFF,3,1);
    probe(&reg,"DIF 4.4.32 [L,L,*]",512,32,f3,v3LLF,3,1);
    return 0;
}
