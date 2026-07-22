/* bench_fftnd.c — two modes:
 *
 *   fuse3d N          rank-3 fusion A/B: fft3d.h (unfused reference) vs
 *                     fftnd s=2 (structurally identical) vs fftnd s=1
 *                     (fused axis1+rows per plane — the 3D back-port),
 *                     x axis-0 lane-block {0, 512}. Identical auto inners
 *                     everywhere -> pure pass-structure comparison; fwd
 *                     output memcmp'd against fft3d as a live gate.
 *
 *   mkl4d N1 N2 N3 N4 rank-4 vs MKL DFTI (split, NOT_INPLACE): DP-planned
 *                     inners per axis, fftnd timed at s in {1,2,3} (the
 *                     split-point verdict data), sorted-|X| multiset vs MKL
 *                     + roundtrip as correctness.
 *
 * ST, pinned, cycles best-of. Build like bench_fft3d_vs_mkl.c.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "fft3d.h"
#include "fftnd.h"
#include "dp_planner.h"
#include "measure.h"
#include "env.h"
#include "generator/generated/registry.h"

#define PIN_CORE 0
#ifndef BEST_OF
#define BEST_OF 5
#endif
#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif
#include <x86intrin.h>
static inline double now_c(void){ return (double)__rdtsc(); }
static int reps_for(size_t t){int r=(int)(3e7/(t+1)); if(r<4)r=4; if(r>20000)r=20000; return r;}

static double bench_fwd(stride_plan_t *p, double *re, double *im, size_t n) {
    int reps = reps_for(n);
    for (int w=0; w<2; w++) stride_execute_fwd(p, re, im);
    double best = 1e18;
    for (int t=0; t<BEST_OF; t++) {
        double t0 = now_c();
        for (int i=0; i<reps; i++) stride_execute_fwd(p, re, im);
        double v = (now_c()-t0)/reps;
        if (v < best) best = v;
    }
    return best;
}

/* ── auto-inner builders (deterministic -> identical inners across plans) ── */
static stride_plan_t *nd3_auto(int N1,int N2,int N3,int split,size_t lb0,
                               const vfft_proto_registry_t *reg){
    int Nv[3]={N1,N2,N3};
    stride_fftnd_data_t tmp; memset(&tmp,0,sizeof tmp);
    tmp.rank=3; for(int m=0;m<3;m++) tmp.N[m]=Nv[m];
    _fftnd_fill_ok(&tmp);
    size_t B=_fftnd_choose_tile(N3,tmp.O[2]);
    stride_plan_t *pl[FFTND_MAX_RANK]={0};
    for(int m=0;m<3;m++){
        size_t Kp=(m==2)?B:tmp.K[m];
        pl[m]=vfft_proto_auto_plan_dispatch(Nv[m],Kp,reg,NULL);
        if(!pl[m]) return NULL;
    }
    size_t lb[3]={lb0,0,0};
    return stride_plan_nd_from(3,Nv,B,split,lb,pl);
}

static void mode_fuse3d(int N, const vfft_proto_registry_t *reg){
    size_t n=(size_t)N*N*N;
    printf("== rank-3 fusion A/B at %d^3 (%.0f MB split) — identical auto inners ==\n",
           N, n*16.0/1048576.0);
    double *xr=AALLOC(n*8),*xi=AALLOC(n*8),*re=AALLOC(n*8),*im=AALLOC(n*8);
    double *fr=AALLOC(n*8),*fi=AALLOC(n*8);
    srand(N);
    for(size_t i=0;i<n;i++){ xr[i]=(double)rand()/RAND_MAX-0.5;
                             xi[i]=(double)rand()/RAND_MAX-0.5; }

    /* fft3d reference (flat) + fwd snapshot for the bit-gate */
    size_t K0=(size_t)N*N, B=_fft3d_choose_tile(N,(size_t)N*N);
    stride_plan_t *r0=vfft_proto_auto_plan_dispatch(N,K0,reg,NULL);
    stride_plan_t *r1=vfft_proto_auto_plan_dispatch(N,(size_t)N,reg,NULL);
    stride_plan_t *rr=vfft_proto_auto_plan_dispatch(N,B,reg,NULL);
    stride_plan_t *p3d=stride_plan_3d_from(N,N,N,B,0,r0,r1,rr);
    memcpy(fr,xr,n*8); memcpy(fi,xi,n*8);
    stride_execute_fwd(p3d,fr,fi);

    memcpy(re,xr,n*8); memcpy(im,xi,n*8);
    double c3d = bench_fwd(p3d,re,im,n);
    printf("  fft3d  flat          : %12.0f cyc  (reference)\n", c3d);

    struct { int s; size_t lb0; const char *tag; } v[] = {
        { 2, 0,   "fftnd s=2 flat      " },
        { 1, 0,   "fftnd s=1 flat FUSED" },
        { 2, 512, "fftnd s=2 blkA=512  " },
        { 1, 512, "fftnd s=1 blkA FUSED" },
    };
    for (size_t k=0;k<4;k++){
        stride_plan_t *p=nd3_auto(N,N,N,v[k].s,v[k].lb0,reg);
        memcpy(re,xr,n*8); memcpy(im,xi,n*8);
        stride_execute_fwd(p,re,im);
        int eq = !memcmp(re,fr,n*8) && !memcmp(im,fi,n*8);
        memcpy(re,xr,n*8); memcpy(im,xi,n*8);
        double c = bench_fwd(p,re,im,n);
        printf("  %s : %12.0f cyc  vs fft3d %.3fx  bit=%s\n",
               v[k].tag, c, c3d/c, eq?"EXACT":"**MISMATCH**");
        stride_plan_destroy(p);
    }
    stride_plan_destroy(p3d);
    AFREE(xr);AFREE(xi);AFREE(re);AFREE(im);AFREE(fr);AFREE(fi);
}

/* ── DP-planned inner for mkl4d ── */
static stride_plan_t *dp_axis(int Nax, size_t K,
                              const vfft_proto_registry_t *reg, char *ds, size_t dsz){
    vfft_proto_dp_context_t ctx;
    vfft_proto_dp_init(&ctx, K, Nax);
    vfft_proto_plan_decision_t dec, pool[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int npool=0;
    double ns=vfft_proto_dp_plan_measure(&ctx,Nax,reg,&dec,pool,&npool,0);
    vfft_proto_dp_destroy(&ctx);
    if (ns>=1e17 || dec.nf<=0){
        snprintf(ds,dsz,"auto");
        return vfft_proto_auto_plan_dispatch(Nax,K,reg,NULL);
    }
    int off=snprintf(ds,dsz,"%s[",dec.use_dif_forward?"DIF ":"");
    for(int s=0;s<dec.nf&&off<(int)dsz-6;s++)
        off+=snprintf(ds+off,dsz-off,"%d%s",dec.factors[s],s+1<dec.nf?"x":"");
    snprintf(ds+off,dsz-off,"]");
    return vfft_proto_plan_create_ex(Nax,K,dec.factors,dec.variants,dec.nf,
                                     dec.use_dif_forward,reg);
}

static int cmp_d(const void*a,const void*b){
    double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y);
}

static void mode_mkl4d(const int *N, const vfft_proto_registry_t *reg){
    size_t n=(size_t)N[0]*N[1]*N[2]*N[3];
    double sc=(double)n;
    printf("== rank-4 %dx%dx%dx%d (%.0f MB) — DP inners, fftnd s-sweep vs MKL ==\n",
           N[0],N[1],N[2],N[3], n*16.0/1048576.0);

    stride_fftnd_data_t tmp; memset(&tmp,0,sizeof tmp);
    tmp.rank=4; for(int m=0;m<4;m++) tmp.N[m]=N[m];
    _fftnd_fill_ok(&tmp);
    size_t B=_fftnd_choose_tile(N[3],tmp.O[3]);

    /* DP once per axis; clone via decision is overkill here — build one plan
     * set per s (auto-planner determinism does not hold for DP under noise,
     * so we DP once and SHARE the plan objects across the s-sweep by
     * building all fftnd variants from freshly-created plans per s using
     * the SAME decisions). Simplest correct: DP once, keep decisions, create
     * plans per s from decisions. */
    char d0[96],d1[96],d2[96],d3[96];
    /* DP per axis, capturing plans directly for s-sweep reuse is unsafe
     * (ownership) — create 3 sets, one per s, from one DP pass each axis by
     * re-creating with create_ex on the SAME decision. */
    vfft_proto_plan_decision_t dec[4]; int have[4]={0,0,0,0};
    for(int m=0;m<4;m++){
        size_t Kp=(m==3)?B:tmp.K[m];
        vfft_proto_dp_context_t ctx;
        vfft_proto_dp_init(&ctx,Kp,N[m]);
        vfft_proto_plan_decision_t pool[VFFT_PROTO_MEASURE_DEPLOY_MAX]; int np=0;
        double ns=vfft_proto_dp_plan_measure(&ctx,N[m],reg,&dec[m],pool,&np,0);
        vfft_proto_dp_destroy(&ctx);
        have[m]=(ns<1e17&&dec[m].nf>0);
    }
    char *dsx[4]={d0,d1,d2,d3};
    for(int m=0;m<4;m++){
        if(have[m]){
            int off=snprintf(dsx[m],96,"%s[",dec[m].use_dif_forward?"DIF ":"");
            for(int s=0;s<dec[m].nf&&off<90;s++)
                off+=snprintf(dsx[m]+off,96-off,"%d%s",dec[m].factors[s],
                              s+1<dec[m].nf?"x":"");
            snprintf(dsx[m]+off,96-off,"]");
        } else snprintf(dsx[m],96,"auto");
    }
    printf("  DP: ax0 %-14s ax1 %-14s ax2 %-14s row %-12s\n",d0,d1,d2,d3);

    double *xr=AALLOC(n*8),*xi=AALLOC(n*8),*re=AALLOC(n*8),*im=AALLOC(n*8);
    srand(41+N[0]);
    for(size_t i=0;i<n;i++){ xr[i]=(double)rand()/RAND_MAX-0.5;
                             xi[i]=(double)rand()/RAND_MAX-0.5; }

    /* MKL rank-4, CCE INTERLEAVED (default/fastest; house methodology --
     * see the note in bench_fft3d_vs_mkl.c: the earlier split-storage
     * config understated MKL by 2.3-5.2x on the bench host). */
    double *zi=AALLOC(n*16),*zo=AALLOC(n*16);
    for(size_t q=0;q<n;q++){ zi[2*q]=xr[q]; zi[2*q+1]=xi[q]; }
    DFTI_DESCRIPTOR_HANDLE h=0;
    MKL_LONG dims[4]={N[0],N[1],N[2],N[3]};
    int mok=0;
    if(DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_COMPLEX,4,dims)==DFTI_NO_ERROR){
        DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
        mok=(DftiCommitDescriptor(h)==DFTI_NO_ERROR);
    }
    double bm=1e18;
    if(mok){
        DftiComputeForward(h,zi,zo);
        int reps=reps_for(n);
        for(int t=0;t<BEST_OF;t++){
            double t0=now_c();
            for(int i=0;i<reps;i++) DftiComputeForward(h,zi,zo);
            double v=(now_c()-t0)/reps; if(v<bm)bm=v;
        }
    }

    double best=1e18; int bests=0; double sme=-1, rt=1;
    for(int s=1;s<=3;s++){
        stride_plan_t *pl[FFTND_MAX_RANK]={0};
        int okp=1;
        for(int m=0;m<4;m++){
            size_t Kp=(m==3)?B:tmp.K[m];
            pl[m]=have[m]
                ? vfft_proto_plan_create_ex(N[m],Kp,dec[m].factors,dec[m].variants,
                                            dec[m].nf,dec[m].use_dif_forward,reg)
                : vfft_proto_auto_plan_dispatch(N[m],Kp,reg,NULL);
            if(!pl[m]) okp=0;
        }
        if(!okp){ printf("  s=%d plan FAIL\n",s); continue; }
        stride_plan_t *p=stride_plan_nd_from(4,N,B,s,NULL,pl);

        if(s==1){ /* correctness once */
            memcpy(re,xr,n*8); memcpy(im,xi,n*8);
            stride_execute_fwd(p,re,im);
            if(mok){
                double *sa=AALLOC(n*8),*sb=AALLOC(n*8),mm=0;
                for(size_t i=0;i<n;i++){ sa[i]=hypot(re[i],im[i]);
                    sb[i]=hypot(zo[2*i],zo[2*i+1]); if(sb[i]>mm)mm=sb[i]; }
                qsort(sa,n,8,cmp_d); qsort(sb,n,8,cmp_d);
                sme=0; for(size_t i=0;i<n;i++){ double e=fabs(sa[i]-sb[i]);
                    if(e>sme)sme=e; }
                if(mm>0) sme/=mm;
                AFREE(sa);AFREE(sb);
            }
            stride_execute_bwd(p,re,im);
            rt=0;
            for(size_t i=0;i<n;i++){
                double rel=(fabs(re[i]-sc*xr[i])+fabs(im[i]-sc*xi[i]))
                          /(fabs(sc*xr[i])+fabs(sc*xi[i])+1e-300);
                if(rel>rt)rt=rel;
            }
        }
        memcpy(re,xr,n*8); memcpy(im,xi,n*8);
        double c=bench_fwd(p,re,im,n);
        stride_fftnd_data_t *dd=(stride_fftnd_data_t*)p->override_data;
        printf("  fftnd s=%d (blk=[%zu,%zu,%zu]) : %12.0f cyc  vs MKL %.3fx\n",
               s, dd->lane_block[0],dd->lane_block[1],dd->lane_block[2],
               c, mok?bm/c:0);
        if(c<best){best=c;bests=s;}
        stride_plan_destroy(p);
    }
    printf("  MKL DFTI rank-4          : %12.0f cyc\n", mok?bm:0);
    printf("  verdict: best s=%d, %.3fx over MKL | rt=%.1e sortMKL=%.1e\n",
           bests, mok?bm/best:0, rt, sme);
    if(h)DftiFreeDescriptor(&h);
    AFREE(xr);AFREE(xi);AFREE(re);AFREE(im);AFREE(zi);AFREE(zo);
}

int main(int argc,char**argv){
    stride_env_init();
    if (stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn: pin failed\n");
    mkl_set_num_threads(1);
    stride_set_num_threads(1);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    if(argc>=3 && !strcmp(argv[1],"fuse3d")) mode_fuse3d(atoi(argv[2]),&reg);
    else if(argc>=6 && !strcmp(argv[1],"mkl4d")){
        int N[4]={atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atoi(argv[5])};
        mode_mkl4d(N,&reg);
    } else {
        mode_fuse3d(128,&reg);
        int N[4]={16,16,16,16};
        mode_mkl4d(N,&reg);
    }
    return 0;
}
