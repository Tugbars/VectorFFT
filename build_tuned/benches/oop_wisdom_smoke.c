/* oop_wisdom_smoke.c — validate the OOP wisdom round trip.
 *
 * Load oop_wisdom.txt, and for each entry rebuild the plan via the PURE-LOOKUP
 * runtime path (vfft_oop_plan_create_wisdom — no measurement), confirm it built
 * the kind the entry names, check correctness, and time vs MKL.
 *
 * Build: cd build_tuned && python build.py --src benches/oop_wisdom_smoke.c --mkl --compile
 * Run  : PATH += MKL bin + C:\mingw152\mingw64\bin ; reads ./oop_wisdom.txt
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "core/executor.h"
#include "core/env.h"
#include "core/oop_wisdom.h"

#define PIN_CORE 2
#define BEST_OF  15
#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif

#include <x86intrin.h>
static inline double now_ns(void){ return (double)__rdtsc(); }  /* cycles; vs-MKL ratio is dimensionless */
static int reps_for(size_t t){int r=(int)(8e6/(t+1)); if(r<20)r=20; if(r>50000)r=50000; return r;}
static const char *kn(int k){return k==VFFT_OOP_KIND_LEAF?"LEAF":k==VFFT_OOP_KIND_BAILEY2?"BAILEY2":"MODEB";}

/* LEAF/BAILEY2: roundtrip fwd+bwd==N*x. MODEB: bit-exact vs in-place dataflow. */
static double correctness(vfft_oop_plan_t*p,int N,size_t K){
    size_t T=(size_t)N*K;
    double *sr=AALLOC(T*8),*si=AALLOC(T*8),*dr=AALLOC(T*8),*di=AALLOC(T*8);
    srand(91+N);
    for(size_t i=0;i<T;i++){sr[i]=(double)rand()/RAND_MAX-0.5; si[i]=(double)rand()/RAND_MAX-0.5;}
    vfft_oop_execute_fwd(p,sr,si,dr,di);
    double me=0;
    if(p->kind==VFFT_OOP_KIND_MODEB && p->mb){
        double *tr=AALLOC(T*8),*ti=AALLOC(T*8); memcpy(tr,sr,T*8); memcpy(ti,si,T*8);
        vfft_proto_execute_fwd(p->mb,tr,ti,K);
        for(size_t i=0;i<T;i++){double a=fabs(dr[i]-tr[i]),b=fabs(di[i]-ti[i]); if(a>me)me=a; if(b>me)me=b;}
        AFREE(tr);AFREE(ti);
    } else {
        double *er=AALLOC(T*8),*ei=AALLOC(T*8); vfft_oop_execute_bwd(p,dr,di,er,ei);
        for(size_t i=0;i<T;i++){double a=fabs(er[i]/(double)N-sr[i]),b=fabs(ei[i]/(double)N-si[i]); if(a>me)me=a; if(b>me)me=b;}
        AFREE(er);AFREE(ei);
    }
    AFREE(sr);AFREE(si);AFREE(dr);AFREE(di);
    return me;
}
static double time_oop(vfft_oop_plan_t*p,const double*sr,const double*si,double*dr,double*di,size_t T){
    for(int w=0;w<5;w++) vfft_oop_execute_fwd(p,sr,si,dr,di);
    int reps=reps_for(T); double best=1e18;
    for(int t=0;t<BEST_OF;t++){double t0=now_ns();
        for(int i=0;i<reps;i++) vfft_oop_execute_fwd(p,sr,si,dr,di);
        double ns=(now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
static double time_mkl(DFTI_DESCRIPTOR_HANDLE d,const double*sr,const double*si,double*mr,double*mi,size_t T){
    for(int w=0;w<5;w++) DftiComputeForward(d,(void*)sr,(void*)si,mr,mi);
    int reps=reps_for(T); double best=1e18;
    for(int t=0;t<BEST_OF;t++){double t0=now_ns();
        for(int i=0;i<reps;i++) DftiComputeForward(d,(void*)sr,(void*)si,mr,mi);
        double ns=(now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    mkl_set_num_threads(1);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    vfft_oop_wisdom_t w;
    if(vfft_oop_wisdom_load(&w,"oop_wisdom.txt")!=0 && vfft_oop_wisdom_load(&w,"benches/oop_wisdom.txt")!=0){
        printf("could not load oop_wisdom.txt (run calibrate_oop first)\n"); return 1;
    }
    printf("== OOP wisdom smoke: %d entries, pure-lookup rebuild + correctness + vs MKL ==\n", w.count);
    printf("%-6s %-5s %-8s %-8s %-9s %-10s %s\n","N","K","entry","built","err","vs MKL","verdict");
    int fails=0;
    for(int i=0;i<w.count;i++){
        int N=w.e[i].N; size_t K=w.e[i].K; size_t T=(size_t)N*K;
        vfft_oop_plan_t *p = vfft_oop_plan_create_wisdom(N,K,&w,&reg);
        if(!p){ printf("%-6d %-5zu %-8s  BUILD NULL\n",N,K,kn(w.e[i].kind)); fails++; continue; }
        int kind_ok = (p->kind == w.e[i].kind);
        double err = correctness(p,N,K);

        double *sr=AALLOC(T*8),*si=AALLOC(T*8),*dr=AALLOC(T*8),*di=AALLOC(T*8),*mr=AALLOC(T*8),*mi=AALLOC(T*8);
        srand(53+N); for(size_t j=0;j<T;j++){sr[j]=(double)rand()/RAND_MAX-0.5; si[j]=(double)rand()/RAND_MAX-0.5;}
        DFTI_DESCRIPTOR_HANDLE d=0; MKL_LONG str[2]={0,(MKL_LONG)K};
        double m=0;
        if(DftiCreateDescriptor(&d,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N)==DFTI_NO_ERROR){
            DftiSetValue(d,DFTI_COMPLEX_STORAGE,DFTI_REAL_REAL); DftiSetValue(d,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
            DftiSetValue(d,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K); DftiSetValue(d,DFTI_INPUT_DISTANCE,1);
            DftiSetValue(d,DFTI_OUTPUT_DISTANCE,1); DftiSetValue(d,DFTI_INPUT_STRIDES,str); DftiSetValue(d,DFTI_OUTPUT_STRIDES,str);
            if(DftiCommitDescriptor(d)==DFTI_NO_ERROR) m=time_mkl(d,sr,si,mr,mi,T);
        }
        double v=time_oop(p,sr,si,dr,di,T);
        int corr_ok = err < (w.e[i].kind==VFFT_OOP_KIND_MODEB ? 1e-12 : 1e-9);
        const char *verdict = (kind_ok && corr_ok) ? "OK" : "FAIL";
        if(!kind_ok || !corr_ok) fails++;
        printf("%-6d %-5zu %-8s %-8s %-9.1e %-9.3f %s%s\n",
               N,K,kn(w.e[i].kind),kn(p->kind),err,m>0?m/v:0,verdict,kind_ok?"":" (kind mismatch)");
        if(d)DftiFreeDescriptor(&d);
        AFREE(sr);AFREE(si);AFREE(dr);AFREE(di);AFREE(mr);AFREE(mi);
        vfft_oop_plan_destroy(p);
    }
    printf("\n# fails=%d  (built kind must match entry; err < 1e-9 natural / 1e-12 MODEB)\n",fails);
    return fails?1:0;
}
