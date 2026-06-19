/* r2c_dif_inner_test.c — verify the DIF-aware recombine perm.
 *
 * Builds the stride r2c path with a DIF-forward inner (use_dif_forward=1, FLAT
 * variants — DIF supports FLAT/LOG3) for several inner-128 factorizations and K,
 * and checks the r2c output vs a reference DFT. This exercises
 * _r2c_compute_perm_dif (DIF output order = factor-reversed digit reversal).
 * For contrast it also builds the DIT version of each shape.
 *
 * Build: cd build_tuned && python build.py --src benches/r2c_dif_inner_test.c --compile
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
    if(vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){exit(1);} return p;}
static double check(const double *o_re,const double *o_im,const double *x,
                    int N,int halfN,size_t K,size_t lane){
    double me=0;
    for(int k=0;k<=halfN;k++){double rr=0,ri=0;
        for(int n=0;n<N;n++){double xn=x[(size_t)n*K+lane];double a=-2.0*M_PI*k*n/(double)N;rr+=xn*cos(a);ri+=xn*sin(a);}
        double er=fabs(o_re[(size_t)k*K+lane]-rr),ei=fabs(o_im[(size_t)k*K+lane]-ri);
        if(er>me)me=er; if(ei>me)me=ei;}
    return me;
}
typedef struct { int nf; int f[4]; const char *name; } shape_t;

/* build stride r2c with an inner of the given factorization + orientation, run, check */
static double run_one(int N,int halfN,size_t K,const shape_t*s,int use_dif,
                      vfft_proto_registry_t*reg){
    int variants[4]; for(int i=0;i<s->nf;i++) variants[i]=use_dif?0:2; /* DIF:FLAT, DIT:T1S */
    stride_plan_t *inner=vfft_proto_plan_create_ex(halfN,K,s->f,variants,s->nf,use_dif,reg);
    if(!inner) return -1.0;
    stride_plan_t *p=stride_r2c_plan(N,K,K,inner);
    if(!p) return -2.0;
    size_t NK=(size_t)N*K, OK=(size_t)(halfN+1)*K;
    double *x=alloc_d(NK); srand(202+(int)K+s->nf);
    for(size_t i=0;i<NK;i++) x[i]=(double)rand()/RAND_MAX*2-1;
    double *oor=alloc_d(OK),*ooi=alloc_d(OK);
    memset(oor,0,OK*8); memset(ooi,0,OK*8);
    stride_execute_r2c(p,x,oor,ooi);
    double e0=check(oor,ooi,x,N,halfN,K,0), e1=check(oor,ooi,x,N,halfN,K,K-1);
    double e=e0>e1?e0:e1;
    vfft_proto_aligned_free(x);vfft_proto_aligned_free(oor);vfft_proto_aligned_free(ooi);
    stride_plan_destroy(p);
    return e;
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    const int N=256, halfN=N/2;
    const size_t Ks[]={8,32,256}; const int nK=3;
    shape_t shapes[]={ {2,{8,16},"8,16"},{2,{16,8},"16,8"},{2,{4,32},"4,32"},
                       {3,{4,4,8},"4,4,8"},{3,{8,4,4},"8,4,4"},{3,{2,8,8},"2,8,8"} };
    int nsh=6;
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    printf("=== r2c with DIF inner (DIF-aware perm) vs DIT inner — both vs reference DFT ===\n");
    printf("%-8s %-6s  %-14s %-14s\n","inner","K","DIT_err","DIF_err");
    int fails=0;
    for(int s=0;s<nsh;s++) for(int ki=0;ki<nK;ki++){
        size_t K=Ks[ki];
        double edit=run_one(N,halfN,K,&shapes[s],0,&reg);
        double edif=run_one(N,halfN,K,&shapes[s],1,&reg);
        char db[32],fb[32];
        if(edit<0) snprintf(db,sizeof db,"%s",edit==-1?"inner NULL":"plan NULL");
        else snprintf(db,sizeof db,"%.2e%s",edit,edit<1e-9?"":" FAIL");
        if(edif<0) snprintf(fb,sizeof fb,"%s",edif==-1?"inner NULL":"plan NULL");
        else snprintf(fb,sizeof fb,"%.2e%s",edif,edif<1e-9?"":" FAIL");
        if((edit>=0&&edit>=1e-9)||(edif>=0&&edif>=1e-9)) fails++;
        printf("%-8s %-6zu  %-14s %-14s\n",shapes[s].name,K,db,fb);
    }
    printf("\n# fails=%d  (DIF_err must be <1e-9 — that validates _r2c_compute_perm_dif)\n",fails);
    return fails?1:0;
}
