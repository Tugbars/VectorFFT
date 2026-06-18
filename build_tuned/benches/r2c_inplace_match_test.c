/* r2c_inplace_match_test.c — verify the new in-place r2c entry
 * (stride_execute_r2c_inplace) matches the out-of-place path AND a reference
 * DFT, across general inner shapes + K. In-place: re (N*K) is overwritten with
 * out_re[0..N/2]; im ((N/2+1)*K) gets out_im.
 *
 * Build: cd build_tuned && python build.py --src benches/r2c_inplace_match_test.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core/executor.h"
#include "core/env.h"
#include "core/planner.h"
#include "core/proto_stride_compat.h"
#include "core/r2c.h"
#include "generator/generated/registry.h"

#define PIN_CORE 2
static double *alloc_d(size_t n){double*p=NULL;
    if(vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){exit(1);} return p;}

static double ref_err(const double *o_re,const double *o_im,const double *x,
                      int N,int halfN,size_t K,size_t lane){
    double me=0;
    for(int k=0;k<=halfN;k++){double rr=0,ri=0;
        for(int n=0;n<N;n++){double xn=x[(size_t)n*K+lane];double a=-2.0*M_PI*k*n/(double)N;rr+=xn*cos(a);ri+=xn*sin(a);}
        double er=fabs(o_re[(size_t)k*K+lane]-rr),ei=fabs(o_im[(size_t)k*K+lane]-ri);
        if(er>me)me=er; if(ei>me)me=ei;}
    return me;
}

typedef struct { int nf; int f[4]; const char *name; } shape_t;

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    const int N=256, halfN=N/2;
    const size_t Ks[]={8,64,256}; const int nK=3;
    shape_t shapes[]={ {2,{16,8},"16,8"}, {3,{4,4,8},"4,4,8"}, {2,{64,2},"64,2"} };
    int nsh=3;
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    printf("=== in-place vs out-of-place r2c match (and vs reference DFT) ===\n");
    printf("%-8s %-6s  %-12s %-12s %-12s\n","inner","K","oop_vs_DFT","inplace_vs_DFT","inplace_vs_oop");
    int fails=0;
    for(int s=0;s<nsh;s++) for(int ki=0;ki<nK;ki++){
        size_t K=Ks[ki]; size_t NK=(size_t)N*K; size_t OK=(size_t)(halfN+1)*K;
        int variants[4]; for(int i=0;i<shapes[s].nf;i++) variants[i]=2;
        stride_plan_t *inner=vfft_proto_plan_create_ex(halfN,K,shapes[s].f,variants,shapes[s].nf,0,&reg);
        if(!inner){ printf("%-8s %-6zu  inner NULL\n",shapes[s].name,K); continue; }
        stride_plan_t *p=stride_r2c_plan(N,K,K,inner);
        if(!p){ printf("%-8s %-6zu  plan NULL\n",shapes[s].name,K); continue; }

        double *x=alloc_d(NK); srand(101+(int)K+s);
        for(size_t i=0;i<NK;i++) x[i]=(double)rand()/RAND_MAX*2-1;

        /* out-of-place */
        double *oor=alloc_d(OK),*ooi=alloc_d(OK);
        memset(oor,0,OK*8); memset(ooi,0,OK*8);
        stride_execute_r2c(p,x,oor,ooi);

        /* in-place: load x into re(N*K), im=(N/2+1)*K */
        double *re=alloc_d(NK),*im=alloc_d(OK);
        memcpy(re,x,NK*8); memset(im,0,OK*8);
        stride_execute_r2c_inplace(p,re,im);

        double e_oop=ref_err(oor,ooi,x,N,halfN,K,0);
        double e_ip =ref_err(re, im, x,N,halfN,K,0);
        /* in-place vs oop: compare bins 0..N/2 across all lanes */
        double mm=0;
        for(int k=0;k<=halfN;k++) for(size_t l=0;l<K;l++){
            double dr=fabs(re[(size_t)k*K+l]-oor[(size_t)k*K+l]);
            double di=fabs(im[(size_t)k*K+l]-ooi[(size_t)k*K+l]);
            if(dr>mm)mm=dr; if(di>mm)mm=di;
        }
        const char *v=(e_oop<1e-9&&e_ip<1e-9&&mm<1e-12)?"OK":"FAIL";
        if(e_oop>=1e-9||e_ip>=1e-9||mm>=1e-12) fails++;
        printf("%-8s %-6zu  %-12.2e %-12.2e %-12.2e %s\n",shapes[s].name,K,e_oop,e_ip,mm,v);

        vfft_proto_aligned_free(x);vfft_proto_aligned_free(oor);vfft_proto_aligned_free(ooi);
        vfft_proto_aligned_free(re);vfft_proto_aligned_free(im);
        stride_plan_destroy(p);
    }
    printf("\n# fails=%d (need oop&inplace vs DFT <1e-9 AND inplace==oop <1e-12)\n",fails);
    return fails?1:0;
}
