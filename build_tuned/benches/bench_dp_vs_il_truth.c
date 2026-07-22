/* DP planner vs IL-truth at (1024, 4), AVX2.
 * Arms: fresh DP MEASURE pick + forced DIT factorizations (WISDOM_ONLY).
 * Per arm: fully-folded IL roundtrip (fwd_ilin + bwd_ilout, jit executors)
 * and split roundtrip. Question: does the boundary-blind DP pick coincide
 * with the IL-truth winner, and if not, is the divergence stage-0 pricing? */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <x86intrin.h>
#include "prime_dispatch.h"
#include "plan_orchestrator.h"
#include "il_layout.h"
#include "il_execute.h"
#include "generator/generated/registry.h"
static const int N=1024; static const size_t K=4;
static int plan_forced(vfft_proto_handle_t *h, const char *wline,
                       const vfft_proto_registry_t *reg){
    char p[]="/tmp/wfXXXXXX"; int fd=mkstemp(p);
    dprintf(fd,"@version 6\n%s\n",wline); close(fd);
    vfft_proto_wisdom_t *w=calloc(1,sizeof *w);
    if(vfft_proto_wisdom_load(w,p)!=0){unlink(p);return -1;}
    unlink(p);
    return vfft_proto_plan(h,N,K,VFFT_PROTO_WISDOM_ONLY,reg,w,NULL);
}
static double maxrel(const double*a,const double*b,size_t n,double s){
    double mx=0,sc=0; for(size_t i=0;i<n;i++){double d=fabs(a[i]/s-b[i]);
        if(d>mx)mx=d; double v=fabs(b[i]); if(v>sc)sc=v;} return sc>0?mx/sc:mx; }
static void arm(const char*nm, vfft_proto_handle_t *h, const double *z0, int warm){
    size_t n=(size_t)N*K;
    double *z=aligned_alloc(64,n*16),*cr=aligned_alloc(64,n*8),*ci=aligned_alloc(64,n*8);
    memcpy(z,z0,n*16);
    int fi=1,fo=1;
    if(vfft_proto_execute_fwd_ilin_jit(h->plan,z,cr,ci,K,h->exec_fwd)<0){
        fi=0; vfft_il2sp(z,cr,ci,n); vfft_proto_plan_execute_fwd(h,cr,ci); }
    if(vfft_proto_execute_bwd_ilout_jit(h->plan,cr,ci,z,K,h->exec_bwd)<0){
        fo=0; vfft_proto_plan_execute_bwd(h,cr,ci); vfft_sp2il(cr,ci,z,n); }
    double e=maxrel(z,z0,2*n,(double)N);
    if(warm){ free(z);free(cr);free(ci); return; }
    int reps=400;
    double til=1e18,tsp=1e18;
    for(int t=0;t<9;t++){
        double t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){
            vfft_proto_execute_fwd_ilin_jit(h->plan,z,cr,ci,K,h->exec_fwd);
            vfft_proto_execute_bwd_ilout_jit(h->plan,cr,ci,z,K,h->exec_bwd); }
        double v=((double)__rdtsc()-t0)/reps; if(v<til)til=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){
            vfft_proto_plan_execute_fwd(h,cr,ci);
            vfft_proto_plan_execute_bwd(h,cr,ci); }
        v=((double)__rdtsc()-t0)/reps; if(v<tsp)tsp=v;
    }
    printf("%-14s fac=[",nm);
    for(int s=0;s<h->plan->num_stages;s++)
        printf("%d%s",h->plan->factors[s],s+1<h->plan->num_stages?",":"");
    printf("] %s %s/%s  IL-rt %8.0f  split-rt %8.0f  IL-tax %+5.1f%%  [err %.1e]\n",
        h->exec_fwd?"jit":"GEN", fi?"in":"IN!", fo?"out":"OUT!",
        til, tsp, 100.0*(til-tsp)/tsp, e);
    free(z);free(cr);free(ci);
}
typedef struct { const char *nm, *w; } farm_t;
static const farm_t FARMS[] = {
 {"[32,32]",      "1024 4 2 32 32 0.0 0 0 0 0 0 2 0"},
 {"[16,64]",      "1024 4 2 16 64 0.0 0 0 0 0 0 2 0"},
 {"[64,16]",      "1024 4 2 64 16 0.0 0 0 0 0 0 2 0"},
 {"[8,16,8]",     "1024 4 3 8 16 8 0.0 0 0 0 0 0 2 2 0"},
 {"[16,16,4]",    "1024 4 3 16 16 4 0.0 0 0 0 0 0 2 2 0"},
 {"[4,16,16]",    "1024 4 3 4 16 16 0.0 0 0 0 0 0 2 2 0"},
 {"[16,4,16]",    "1024 4 3 16 4 16 0.0 0 0 0 0 0 2 2 0"},
 {"[8,8,16]",     "1024 4 3 8 8 16 0.0 0 0 0 0 0 2 2 0"},
 {"[16,8,8]",     "1024 4 3 16 8 8 0.0 0 0 0 0 0 2 2 0"},
 {"[64,4,4]",     "1024 4 3 64 4 4 0.0 0 0 0 0 0 2 2 0"},
 {"[4,4,64]",     "1024 4 3 4 4 64 0.0 0 0 0 0 0 2 2 0"},
 {"[4x5]",        "1024 4 5 4 4 4 4 4 0.0 0 0 0 0 0 2 2 2 2 0"},
};
int main(int argc, char**argv){
    int warm = argc>1 && !strcmp(argv[1],"warm");
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    size_t n=(size_t)N*K;
    double *z0=aligned_alloc(64,n*16);
    srand(77); for(size_t i=0;i<2*n;i++) z0[i]=2.0*rand()/RAND_MAX-1;
    vfft_proto_handle_t hdp;
    static vfft_proto_wisdom_t wdp; memset(&wdp,0,sizeof wdp);
    if(vfft_proto_plan(&hdp,N,K,VFFT_PROTO_MEASURE,&reg,&wdp,NULL)==0)
        arm("DP-pick",&hdp,z0,warm);
    for(size_t a=0;a<sizeof FARMS/sizeof *FARMS;a++){
        vfft_proto_handle_t h;
        if(plan_forced(&h,FARMS[a].w,&reg)==0) arm(FARMS[a].nm,&h,z0,warm);
        else printf("%-14s plan-fail\n",FARMS[a].nm);
    }
    if(warm) fprintf(stderr,"warmed\n");
    return 0;
}
