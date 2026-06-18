/* bench_r2c_inplace_vs_oop.c — clean A/B: r2c with IN-PLACE codelets only
 * (separate pack pass) vs OOP-fused pack (radix4_n1_oop). ORDER-NEUTRALIZED
 * timing (flip per trial) so the in-place-vs-OOP delta is unbiased — the
 * existing packfuse bench timed the in-place path 7th (thermally inflated).
 *
 * Question (from Tugbars): can we do r2c with NO OOP codelet — using only the
 * in-place c2c codelets that already win 238/238 — and what does it cost?
 *
 *   IN-PLACE path : separate de-interleave pack (x -> split z) + full in-place
 *                   c2c (vfft_proto_execute_fwd) + AVX2 Hermitian recombine.
 *                   Uses ONLY in-place codelets. No OOP. Productionizes with
 *                   the existing build.py codelet lib (inplace dir).
 *   OOP path      : radix4_n1_oop stage-0 reads x directly (pack fused into the
 *                   butterfly) + stages 1.. + AVX2 recombine.
 *
 * Both correctness-gated vs reference DFT (<1e-9). MKL r2c as the yardstick.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_r2c_inplace_vs_oop.c --mkl --compile
 * Run  : PATH += MKL bin + C:\mingw152\mingw64\bin, then run the .exe.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <mkl_dfti.h>
#include <mkl_service.h>

#include "core/executor.h"
#include "core/env.h"
#include "core/planner.h"
#include "core/dp_planner.h"
#include "core/proto_stride_compat.h"
#include "core/r2c.h"
#include "generator/generated/registry.h"
#include "../src/dag-fft-compiler/codelets/oop/avx2/radix4_n1_oop_avx2.c"

#define PIN_CORE 2
#define BEST_OF  21

static double *alloc_d(size_t n){double*p=NULL;
    if(vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){fprintf(stderr,"alloc\n");exit(1);} return p;}
static int reps_for(size_t total){int r=(int)(4e6/(total+1)); if(r<30)r=30; if(r>200000)r=200000; return r;}

static void compute_perm(const int *f,int nf,int N,int *perm){
    for(int n=0;n<N;n++){int idx=n,rev=0,rp=1;
        for(int s=0;s<nf;s++){int R=f[s];int d=idx%R;idx/=R;rev+=d*(N/(rp*R));rp*=R;} perm[n]=rev;}
}

/* ---- shared pieces (mirror packfuse, verified math) ---- */
static void prepack(const double *x,double *zre,double *zim,int halfN,size_t K){
    for(int j=0;j<halfN;j++){
        const double *xe=x+(size_t)(2*j)*K,*xo=x+(size_t)(2*j+1)*K;
        double *zr=zre+(size_t)j*K,*zi=zim+(size_t)j*K;
        for(size_t l=0;l<K;l++){zr[l]=xe[l];zi[l]=xo[l];}
    }
}
static int packfuse_s0(const stride_plan_t *p,const double *x,double *zre,double *zim,size_t K){
    const stride_stage_t *st0=&p->stages[0];
    if(p->use_dif_forward||st0->radix!=4) return 0;
    const size_t s0=st0->stride;
    for(int g=0;g<st0->num_groups;g++){
        size_t gb=st0->group_base[g];
        radix4_n1_oop_fwd_avx2_UG_UG(x+2*gb,x+K+2*gb,zre+gb,zim+gb,NULL,NULL,2*s0,1,s0,1,K);
    }
    return 1;
}
static void exec_from(const stride_plan_t *p,double *re,double *im,size_t sK,int s0){
    for(int s=s0;s<p->num_stages;s++){
        const stride_stage_t *st=&p->stages[s]; const int G=st->num_groups,R=st->radix;
        for(int g=0;g<G;g++){
            double *br=re+st->group_base[g],*bi=im+st->group_base[g];
            if(!st->needs_tw[g]){st->n1_fwd(br,bi,br,bi,st->stride,st->stride,sK);continue;}
            double cfr=st->cf0_re[g],cfi=st->cf0_im[g];
            if(st->use_log3){
                if(cfr!=1.0||cfi!=0.0) for(int j=0;j<R;j++){double*lr=br+(size_t)j*st->stride,*li=bi+(size_t)j*st->stride;_stride_cmul_scalar_inplace(lr,li,sK,cfr,cfi);}
                st->t1_fwd(br,bi,st->grp_tw_re[g],st->grp_tw_im[g],st->stride,sK);continue;}
            if(st->t1s_fwd){
                if(cfr!=1.0||cfi!=0.0)_stride_cmul_scalar_inplace(br,bi,sK,cfr,cfi);
                st->t1s_fwd(br,bi,st->tw_scalar_re[g],st->tw_scalar_im[g],st->stride,sK);continue;}
            if(cfr!=1.0||cfi!=0.0)_stride_cmul_scalar_inplace(br,bi,sK,cfr,cfi);
            const int Rm1=R-1; const double *sr=st->tw_scalar_re[g],*si=st->tw_scalar_im[g];
            double twr[63*VFFT_PROTO_TW_BLOCK_K],twi[63*VFFT_PROTO_TW_BLOCK_K];
            for(size_t kb=0;kb<sK;kb+=VFFT_PROTO_TW_BLOCK_K){size_t tk=sK-kb;if(tk>VFFT_PROTO_TW_BLOCK_K)tk=VFFT_PROTO_TW_BLOCK_K;
                for(int j=0;j<Rm1;j++){size_t o=(size_t)j*tk;_stride_broadcast_2(twr+o,twi+o,tk,sr[j],si[j]);}
                st->t1_fwd(br+kb,bi+kb,twr,twi,st->stride,tk);}
        }
    }
}
static void recombine_avx2(const double *zre,const double *zim,double *o_re,double *o_im,
                           const int *perm,const double *tw_re,const double *tw_im,int halfN,size_t K){
    {const double *Z0r=zre+(size_t)perm[0]*K,*Z0i=zim+(size_t)perm[0]*K;
     double *o0r=o_re,*o0i=o_im,*onr=o_re+(size_t)halfN*K,*oni=o_im+(size_t)halfN*K;
     const __m256d z=_mm256_setzero_pd();size_t l=0;
     for(;l+4<=K;l+=4){__m256d a=_mm256_load_pd(Z0r+l),b=_mm256_load_pd(Z0i+l);
        _mm256_store_pd(o0r+l,_mm256_add_pd(a,b));_mm256_store_pd(o0i+l,z);
        _mm256_store_pd(onr+l,_mm256_sub_pd(a,b));_mm256_store_pd(oni+l,z);}
     for(;l<K;l++){o0r[l]=Z0r[l]+Z0i[l];o0i[l]=0;onr[l]=Z0r[l]-Z0i[l];oni[l]=0;}}
    const __m256d hv=_mm256_set1_pd(0.5),sg=_mm256_set1_pd(-0.0);
    for(int k=1;k<halfN;k++){int mk=halfN-k;
        const double *Zk_r=zre+(size_t)perm[k]*K,*Zk_i=zim+(size_t)perm[k]*K;
        const double *Zm_r=zre+(size_t)perm[mk]*K,*Zm_i=zim+(size_t)perm[mk]*K;
        double *or_=o_re+(size_t)k*K,*oi_=o_im+(size_t)k*K;
        double c=tw_re[k],s=-tw_im[k]; const __m256d vc=_mm256_set1_pd(c),vs=_mm256_set1_pd(s);
        size_t l=0;
        for(;l+4<=K;l+=4){__m256d zr=_mm256_load_pd(Zk_r+l),zi=_mm256_load_pd(Zk_i+l),mr=_mm256_load_pd(Zm_r+l),mi=_mm256_load_pd(Zm_i+l);
            __m256d Er=_mm256_mul_pd(_mm256_add_pd(zr,mr),hv),Ei=_mm256_mul_pd(_mm256_sub_pd(zi,mi),hv);
            __m256d Or=_mm256_mul_pd(_mm256_sub_pd(zr,mr),hv),Oi=_mm256_mul_pd(_mm256_add_pd(zi,mi),hv);
            __m256d Tr=_mm256_fmsub_pd(vc,Oi,_mm256_mul_pd(vs,Or));
            __m256d Ti=_mm256_xor_pd(sg,_mm256_fmadd_pd(vc,Or,_mm256_mul_pd(vs,Oi)));
            _mm256_store_pd(or_+l,_mm256_add_pd(Er,Tr));_mm256_store_pd(oi_+l,_mm256_add_pd(Ei,Ti));}
        for(;l<K;l++){double zr=Zk_r[l],zi=Zk_i[l],mr=Zm_r[l],mi=Zm_i[l];
            double Er=0.5*(zr+mr),Ei=0.5*(zi-mi),Or=0.5*(zr-mr),Oi=0.5*(zi+mi);
            or_[l]=Er+(c*Oi-s*Or); oi_[l]=Ei+(-c*Or-s*Oi);}
    }
}

/* full paths */
static void run_inplace(const stride_plan_t *p,const double *x,double *zre,double *zim,
                        double *o_re,double *o_im,const int *perm,const double *tr,const double *ti,
                        int halfN,size_t K){
    prepack(x,zre,zim,halfN,K);
    vfft_proto_execute_fwd(p,zre,zim,K);
    recombine_avx2(zre,zim,o_re,o_im,perm,tr,ti,halfN,K);
}
static int run_oop(const stride_plan_t *p,const double *x,double *zre,double *zim,
                   double *o_re,double *o_im,const int *perm,const double *tr,const double *ti,
                   int halfN,size_t K){
    if(!packfuse_s0(p,x,zre,zim,K)){run_inplace(p,x,zre,zim,o_re,o_im,perm,tr,ti,halfN,K);return 0;}
    exec_from(p,zre,zim,K,1);
    recombine_avx2(zre,zim,o_re,o_im,perm,tr,ti,halfN,K);
    return 1;
}

static double gate(const double *o_re,const double *o_im,const double *x,int N,int halfN,size_t K){
    double me=0;
    for(int k=0;k<=halfN;k++){double rr=0,ri=0;
        for(int n=0;n<N;n++){double xn=x[(size_t)n*K+0];double a=-2.0*M_PI*k*n/(double)N;rr+=xn*cos(a);ri+=xn*sin(a);}
        double er=fabs(o_re[(size_t)k*K]-rr),ei=fabs(o_im[(size_t)k*K]-ri);if(er>me)me=er;if(ei>me)me=ei;}
    return me;
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn: pin cpu%d failed\n",PIN_CORE);
    mkl_set_num_threads(1);
    const int N=256,halfN=N/2;
    const size_t Ks[]={32,64,128,256}; const int nK=4;

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t wis; int hw=(vfft_proto_wisdom_load(&wis,"../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    if(!hw) hw=(vfft_proto_wisdom_load(&wis,"../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    printf("# c2c wisdom load: %s\n",hw?"OK":"FAILED");
    double *tr=alloc_d(halfN),*ti=alloc_d(halfN); _r2c_init_twiddles(N,tr,ti);

    printf("=== r2c IN-PLACE codelets (separate pack) vs OOP-fused pack vs MKL  ORDER-NEUTRALIZED\n");
    printf("    (N=256, ST, cpu%d, best-of-%d, flip per trial) ===\n",PIN_CORE,BEST_OF);
    printf("%-5s %12s %12s %12s %9s %9s %11s\n","K","inplace_ns","oop_ns","mkl_ns","inpl/mkl","oop/mkl","oop_vs_inpl");
    printf("------+------------+------------+------------+---------+---------+-----------\n");

    for(int ki=0;ki<nK;ki++){
        size_t K=Ks[ki]; size_t total=(size_t)N*K;
        double *x=alloc_d(total); srand(7+(int)K);
        for(size_t i=0;i<total;i++) x[i]=(double)rand()/RAND_MAX*2-1;
        stride_plan_t *inner=vfft_proto_auto_plan(halfN,K,&reg,hw?&wis:NULL);
        if(!inner){printf("%-5zu auto_plan NULL\n",K);vfft_proto_aligned_free(x);continue;}
        int perm[256]; compute_perm(inner->factors,inner->num_stages,halfN,perm);
        double *zre=alloc_d((size_t)halfN*K),*zim=alloc_d((size_t)halfN*K);
        double *oor=alloc_d((size_t)(halfN+1)*K),*ooi=alloc_d((size_t)(halfN+1)*K);

        char fs[64];size_t pp=0;
        for(int s=0;s<inner->num_stages;s++) pp+=(size_t)snprintf(fs+pp,sizeof fs-pp,"%s%d",s?",":"",inner->factors[s]);

        /* correctness */
        memset(oor,0,(size_t)(halfN+1)*K*8);memset(ooi,0,(size_t)(halfN+1)*K*8);
        run_inplace(inner,x,zre,zim,oor,ooi,perm,tr,ti,halfN,K); double e_ip=gate(oor,ooi,x,N,halfN,K);
        memset(oor,0,(size_t)(halfN+1)*K*8);memset(ooi,0,(size_t)(halfN+1)*K*8);
        int applied=run_oop(inner,x,zre,zim,oor,ooi,perm,tr,ti,halfN,K); double e_oop=gate(oor,ooi,x,N,halfN,K);
        printf("# K=%-4zu inner=(%s)  stage0_radix=%d  oop=%s\n",K,fs,inner->stages[0].radix,applied?"APPLIED":"FALLBACK(=inplace)");
        if(e_ip>=1e-9||e_oop>=1e-9){printf("%-5zu *** CORRECTNESS FAIL ip=%.2e oop=%.2e ***\n",K,e_ip,e_oop);
            vfft_proto_aligned_free(x);vfft_proto_aligned_free(zre);vfft_proto_aligned_free(zim);
            vfft_proto_aligned_free(oor);vfft_proto_aligned_free(ooi);stride_plan_destroy(inner);continue;}

        /* MKL */
        DFTI_DESCRIPTOR_HANDLE h=0;int mok=0;
        double *xin=alloc_d(total),*cce=alloc_d((size_t)(halfN+1)*K*2);
        for(size_t t=0;t<K;t++)for(int n=0;n<N;n++)xin[t*N+n]=x[(size_t)n*K+t];
        DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
        DftiSetValue(h,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
        DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
        DftiSetValue(h,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
        DftiSetValue(h,DFTI_OUTPUT_DISTANCE,(MKL_LONG)(halfN+1));
        mok=(DftiCommitDescriptor(h)==DFTI_NO_ERROR);

        int reps=reps_for(total);
        /* warm both */
        for(int w=0;w<10;w++){run_inplace(inner,x,zre,zim,oor,ooi,perm,tr,ti,halfN,K);run_oop(inner,x,zre,zim,oor,ooi,perm,tr,ti,halfN,K);}
        double best_ip=1e18,best_oop=1e18,best_m=1e18;
        for(int t=0;t<BEST_OF;t++){
            int flip=t&1;
            for(int side=0;side<2;side++){
                int do_oop=(side==0)?flip:!flip;
                double t0=vfft_proto_now_ns();
                if(do_oop) for(int i=0;i<reps;i++) run_oop(inner,x,zre,zim,oor,ooi,perm,tr,ti,halfN,K);
                else       for(int i=0;i<reps;i++) run_inplace(inner,x,zre,zim,oor,ooi,perm,tr,ti,halfN,K);
                double ns=(vfft_proto_now_ns()-t0)/reps;
                if(do_oop){if(ns<best_oop)best_oop=ns;} else {if(ns<best_ip)best_ip=ns;}
            }
            if(mok){double t0=vfft_proto_now_ns();for(int i=0;i<reps;i++)DftiComputeForward(h,(void*)xin,cce);
                double ns=(vfft_proto_now_ns()-t0)/reps;if(ns<best_m)best_m=ns;}
        }
        printf("%-5zu %12.1f %12.1f %12.1f %8.3fx %8.3fx %10.3fx\n",
               K,best_ip,best_oop,best_m, best_m>0?best_m/best_ip:0, best_m>0?best_m/best_oop:0, best_oop>0?best_ip/best_oop:0);

        if(h)DftiFreeDescriptor(&h);
        vfft_proto_aligned_free(x);vfft_proto_aligned_free(zre);vfft_proto_aligned_free(zim);
        vfft_proto_aligned_free(oor);vfft_proto_aligned_free(ooi);
        vfft_proto_aligned_free(xin);vfft_proto_aligned_free(cce);stride_plan_destroy(inner);
    }
    vfft_proto_aligned_free(tr);vfft_proto_aligned_free(ti);if(hw)vfft_proto_wisdom_free(&wis);
    printf("\n# inpl/mkl, oop/mkl = MKL_ns/ours (>1 = we beat MKL).\n");
    printf("# oop_vs_inpl = inplace_ns/oop_ns (>1 = OOP fusion faster; ~1.0 = in-place is free).\n");
    return 0;
}
