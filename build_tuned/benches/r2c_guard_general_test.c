/* r2c_guard_general_test.c — is the stride_r2c guard STALE?
 *
 * The guard in stride_r2c_plan whitelists only inner shapes (8,16)/(16,8)/
 * single-stage, claiming _r2c_postprocess is shape-limited. But the current
 * postprocess reads each freq from its true slot (z_m = perm[mirror]*B), which
 * is general. This test builds the stride r2c path for MANY inner-128
 * factorizations and K values, runs it, and checks vs a reference DFT.
 *
 *   - Under the CURRENT guard: non-whitelisted shapes -> stride_r2c_plan NULL
 *     (reported BLOCKED). Whitelisted shapes -> PASS expected.
 *   - After LIFTING the guard: all shapes build; this reports PASS/FAIL,
 *     settling empirically whether the general recombine is correct.
 *
 * Build: cd build_tuned && python build.py --src benches/r2c_guard_general_test.c --compile
 * Run  : PATH += C:\mingw152\mingw64\bin, then run the .exe.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "executor.h"
#include "env.h"
#include "planner.h"
#include "proto_stride_compat.h"
#include "r2c.h"
#include "generator/generated/registry.h"

#define PIN_CORE 2

static double *alloc_d(size_t n){double*p=NULL;
    if(vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){fprintf(stderr,"alloc\n");exit(1);} return p;}

/* reference real DFT, lane `lane`, bins 0..N/2 -> compare against out_re/out_im */
static double check(const double *out_re,const double *out_im,const double *x,
                    int N,int halfN,size_t K,size_t lane){
    double me=0;
    for(int k=0;k<=halfN;k++){
        double rr=0,ri=0;
        for(int n=0;n<N;n++){double xn=x[(size_t)n*K+lane];double a=-2.0*M_PI*k*n/(double)N;rr+=xn*cos(a);ri+=xn*sin(a);}
        double er=fabs(out_re[(size_t)k*K+lane]-rr),ei=fabs(out_im[(size_t)k*K+lane]-ri);
        if(er>me)me=er; if(ei>me)me=ei;
    }
    return me;
}

typedef struct { int nf; int f[5]; const char *name; } shape_t;

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");

    const int N=256, halfN=N/2;
    const size_t Ks[]={8,32,256}; const int nK=3;
    /* inner-128 factorizations (product must = 128) */
    shape_t shapes[]={
        {1,{128},"128"},
        {2,{8,16},"8,16"},
        {2,{16,8},"16,8"},
        {2,{4,32},"4,32"},
        {2,{32,4},"32,4"},
        {2,{2,64},"2,64"},
        {2,{64,2},"64,2"},
        {3,{4,4,8},"4,4,8"},
        {3,{8,4,4},"8,4,4"},
        {3,{2,8,8},"2,8,8"},
        {4,{2,4,4,4},"2,4,4,4"},
    };
    int nsh=(int)(sizeof shapes/sizeof shapes[0]);

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    printf("=== stride r2c GENERAL-SHAPE correctness (N=256, inner=128) ===\n");
    printf("%-10s", "inner");
    for(int ki=0;ki<nK;ki++) printf("   K=%-5zu", Ks[ki]);
    printf("\n----------+----------+----------+----------\n");

    int total=0, blocked=0, passed=0, failed=0;
    for(int s=0;s<nsh;s++){
        /* verify product */
        int prod=1; for(int i=0;i<shapes[s].nf;i++) prod*=shapes[s].f[i];
        printf("%-10s", shapes[s].name);
        if(prod!=halfN){ printf("  (bad product %d!=%d)\n",prod,halfN); continue; }

        for(int ki=0;ki<nK;ki++){
            size_t K=Ks[ki]; size_t total_n=(size_t)N*K;
            total++;
            int variants[5]; for(int i=0;i<shapes[s].nf;i++) variants[i]=2; /* T1S */
            stride_plan_t *inner=vfft_proto_plan_create_ex(halfN,K,shapes[s].f,variants,shapes[s].nf,0,&reg);
            if(!inner){ printf("   %-7s","INNER!"); continue; }
            stride_plan_t *p=stride_r2c_plan(N,K,K,inner);  /* B=K */
            if(!p){ printf("   %-7s","BLOCKED"); blocked++; continue; }

            double *x=alloc_d(total_n);
            srand(31+(int)K+s);
            for(size_t i=0;i<total_n;i++) x[i]=(double)rand()/RAND_MAX*2-1;
            double *oor=alloc_d((size_t)(halfN+1)*K), *ooi=alloc_d((size_t)(halfN+1)*K);
            memset(oor,0,(size_t)(halfN+1)*K*8); memset(ooi,0,(size_t)(halfN+1)*K*8);
            stride_execute_r2c(p,x,oor,ooi);
            double e0=check(oor,ooi,x,N,halfN,K,0);
            double e1=check(oor,ooi,x,N,halfN,K,K-1);
            double e=e0>e1?e0:e1;
            if(e<1e-9){ printf("   %-7s","PASS"); passed++; }
            else      { printf("   FAIL%.0e",e); failed++; }
            vfft_proto_aligned_free(x); vfft_proto_aligned_free(oor); vfft_proto_aligned_free(ooi);
            stride_plan_destroy(p);   /* frees inner via override_destroy */
        }
        printf("\n");
    }
    printf("\n# total=%d  blocked(guard)=%d  passed=%d  failed=%d\n",total,blocked,passed,failed);
    printf("# (BLOCKED = guard refused; PASS/FAIL only for shapes the guard allowed.)\n");
    return 0;
}
