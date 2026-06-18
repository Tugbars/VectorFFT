/* r2c_default_routing_check.c — validate the SHIPPING default: with no explicit
 * threshold set, vfft_r2c_plan_create must route K<=16 -> RFFT, K>=32 -> STRIDE
 * (default _vfft_r2c_decouple_min_k=32), and BOTH must be correct vs reference DFT.
 * Also exercises the in-place stride entry for the stride cells.
 *
 * Build: cd build_tuned && python build.py --src benches/r2c_default_routing_check.c --compile
 */
#define _GNU_SOURCE 1
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rfft_registry_avx2.h"
#include "core/r2c_dispatch.h"
#include "core/env.h"
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
int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    const int N=256, halfN=N/2;
    const size_t Ks[]={8,16,32,64,128,256}; const int nK=6;
    const char *expect[]={"RFFT","RFFT","STRIDE","STRIDE","STRIDE","STRIDE"};

    rfft_codelets_t rreg; memset(&rreg,0,sizeof rreg); rfft_register_all_avx2(&rreg);
    vfft_proto_registry_t creg; vfft_proto_registry_init(&creg);
    vfft_proto_wisdom_t wis; int hw=
        (vfft_proto_wisdom_load(&wis,"../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    if(!hw) hw=(vfft_proto_wisdom_load(&wis,"../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    if(hw) vfft_r2c_dispatch_set_c2c_wisdom(&wis);
    /* DO NOT set decouple_min_k -> use the shipping default (32). */

    printf("=== shipping-default r2c routing + correctness (N=256, SPLIT) ===\n");
    printf("%-5s %-8s %-8s %-12s %-12s %s\n","K","path","expect","oop_err","inplace_err","verdict");
    int fails=0;
    for(int ki=0;ki<nK;ki++){
        size_t K=Ks[ki]; size_t NK=(size_t)N*K, OK=(size_t)(halfN+1)*K;
        vfft_r2c_plan_t *p=vfft_r2c_plan_create(N,K,VFFT_R2C_SPLIT,&rreg,NULL,&creg);
        if(!p){printf("%-5zu plan NULL\n",K); fails++; continue;}
        const char *path=(p->path==VFFT_R2C_PATH_RFFT)?"RFFT":"STRIDE";

        double *x=alloc_d(NK); srand(55+(int)K);
        for(size_t i=0;i<NK;i++) x[i]=(double)rand()/RAND_MAX*2-1;
        double *oor=alloc_d(OK),*ooi=alloc_d(OK);
        memset(oor,0,OK*8); memset(ooi,0,OK*8);
        vfft_r2c_execute_fwd(p,x,oor,ooi);
        double e_oop=check(oor,ooi,x,N,halfN,K,0);

        /* in-place entry only meaningful for the stride path (override worker) */
        double e_ip=-1;
        if(p->path==VFFT_R2C_PATH_STRIDE){
            double *re=alloc_d(NK),*im=alloc_d(OK);
            memcpy(re,x,NK*8); memset(im,0,OK*8);
            stride_execute_r2c_inplace(p->stride,re,im);
            e_ip=check(re,im,x,N,halfN,K,0);
            vfft_proto_aligned_free(re); vfft_proto_aligned_free(im);
        }
        int path_ok=(strcmp(path,expect[ki])==0);
        int corr_ok=(e_oop<1e-9)&&(e_ip<0||e_ip<1e-9);
        const char *verdict=(path_ok&&corr_ok)?"OK":"FAIL";
        if(!path_ok||!corr_ok) fails++;
        if(e_ip<0) printf("%-5zu %-8s %-8s %-12.2e %-12s %s\n",K,path,expect[ki],e_oop,"(n/a)",verdict);
        else       printf("%-5zu %-8s %-8s %-12.2e %-12.2e %s\n",K,path,expect[ki],e_oop,e_ip,verdict);

        vfft_r2c_plan_destroy(p);
        vfft_proto_aligned_free(x); vfft_proto_aligned_free(oor); vfft_proto_aligned_free(ooi);
    }
    if(hw) vfft_proto_wisdom_free(&wis);
    printf("\n# fails=%d\n",fails);
    return fails?1:0;
}
