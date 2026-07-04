/* natorder_p0_scatter.c — Phase-0 GO/NO-GO for the natural-order SCRATCH scatter-terminator,
 * on the REAL calibrated plans (spike_wisdom.txt via wisdom-copy dir), per user direction:
 * N=4096, K in {4, 32} (the cells with calibrated entries; no new calibration).
 *   K=4 : nf=4, chain 4,4,8,32, use_dif_forward=1  (DIF! — natural mode's DIT-only clause matters here)
 *   K=32: nf=5, chain 4,4,4,8,8, DIT
 * Steps per cell:
 *   1. baseline: public-API in-place c2c fwd execute (calibrated plan, JIT-resolved), QPC best-of-5,
 *      chunked fwd-only (8 fwd per timed chunk, untimed rescale between chunks to avoid overflow).
 *   2. perm validation: compute mixed-radix digit-reversal perm from the chain (DIT and DIF-reversed
 *      variants); FFT random data once; find which mapping M gives natural[n]=scrambled[M[n]] == naive
 *      DFT (lane 0). Reports the matching orientation — the order-probe half of Phase 0.
 *   3. block/comb diagnostic: is {M[q+j*P]} a contiguous R-row block for each q? (design's comb algebra)
 *   4. pattern kernels (2 planes, row=K doubles): same-order copy pass | in-place seq read+write pass |
 *      scatter-q (sequential comb writes, perm reads) | scatter-b (sequential reads, perm writes) |
 *      PURE cycle-following in-place perm | PURE gather+copyback.
 *   5. derived: scatter bandwidth ratio vs same-order (the 0.6-0.85x question); est SCRATCH overhead
 *      = (scatter_best - inplace_pass)/fft; PURE overhead = perm_best/fft.
 * Build: python build.py --src test/natorder_p0_scatter.c --vfft
 * Run from build_tuned/ (wisdom copy in natorder_wis_p0/). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define NN 4096
static double qpc_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

/* mixed-radix digit reversal: little-endian digits of n over chain f[0..nf-1] become
 * big-endian slot digits (same order). perm[n] = slot of bin n. */
static void mk_perm(int N,const int *f,int nf,int *perm){
    for(int n=0;n<N;n++){ int t=n,slot=0,rem=N;
        for(int s=0;s<nf;s++){ int d=t%f[s]; t/=f[s]; rem/=f[s]; slot+=d*rem; }
        perm[n]=slot; }
}
static void inv_perm(int N,const int *p,int *ip){ for(int n=0;n<N;n++) ip[p[n]]=n; }

/* naive DFT of lane 0 (reference) */
static void naive_dft_lane0(const double *re,const double *im,size_t K,double *Xr,double *Xi){
    for(int k=0;k<NN;k++){ double sr=0,si=0;
        for(int n=0;n<NN;n++){ double a=-2.0*M_PI*(double)k*(double)n/(double)NN;
            double c=cos(a),s=sin(a),xr=re[(size_t)n*K],xi=im[(size_t)n*K];
            sr+=xr*c-xi*s; si+=xr*s+xi*c; }
        Xr[k]=sr; Xi[k]=si; }
}

/* ---- timing helpers: best-of-5 outer, inner reps sized for >=20ms per timed run ---- */
typedef void (*kfn)(void*);
static double time_kernel(kfn fn, void *ctx, int inner){
    double best=1e30;
    for(int o=0;o<5;o++){ double t0=qpc_ns();
        for(int i=0;i<inner;i++) fn(ctx);
        double dt=(qpc_ns()-t0)/inner; if(dt<best)best=dt; }
    return best;
}

typedef struct { double *dr,*di; const double *sr,*si; const int *map; size_t K; int R,P; } pat_t;
/* same-order streaming copy scratch->user (friendly terminator proxy) */
static void k_sameorder(void *v){ pat_t *p=v; memcpy(p->dr,p->sr,(size_t)NN*p->K*8); memcpy(p->di,p->si,(size_t)NN*p->K*8); }
/* in-place sequential read+write pass over dst (memory proxy of the current last in-place stage) */
static void k_inplacepass(void *v){ pat_t *p=v; size_t n=(size_t)NN*p->K;
    for(size_t i=0;i<n;i++){ p->dr[i]=p->dr[i]*1.000000001+1e-30; p->di[i]=p->di[i]*1.000000001+1e-30; } }
/* scatter-q: natural rows written in q-order => R sequential write streams; reads via map */
static void k_scatter_q(void *v){ pat_t *p=v; size_t K=p->K; int P=p->P,R=p->R;
    for(int q=0;q<P;q++) for(int j=0;j<R;j++){ int n=q+j*P; int m=p->map[n];
        memcpy(p->dr+(size_t)n*K,p->sr+(size_t)m*K,K*8); memcpy(p->di+(size_t)n*K,p->si+(size_t)m*K,K*8); } }
/* scatter-b: scrambled rows read sequentially; writes land via inverse map (comb-random) */
static void k_scatter_b(void *v){ pat_t *p=v; size_t K=p->K; const int *im_=p->map; /* map=IMAP here */
    for(int m=0;m<NN;m++){ int n=im_[m];
        memcpy(p->dr+(size_t)n*K,p->sr+(size_t)m*K,K*8); memcpy(p->di+(size_t)n*K,p->si+(size_t)m*K,K*8); } }
/* scatter-s: j OUTER, q inner — the design's TRUE "R sequential write streams": stream j writes the
 * contiguous region rows [j*P, (j+1)*P) natural... natural rows q+j*P for q ascending ARE contiguous
 * (row q+jP then q+1+jP). Reads scattered per q; adjacent j hit the same 64B line on later passes. */
static void k_scatter_s(void *v){ pat_t *p=v; size_t K=p->K; int P=p->P,R=p->R;
    for(int j=0;j<R;j++){ double *wr=p->dr+(size_t)j*P*K, *wi=p->di+(size_t)j*P*K;
        for(int q=0;q<P;q++){ int m=p->map[q+j*P];
            memcpy(wr+(size_t)q*K,p->sr+(size_t)m*K,K*8); memcpy(wi+(size_t)q*K,p->si+(size_t)m*K,K*8); } } }
/* PURE in-place cycle-following on dst using map (dst[n] <- dst[map[n]] along cycles) */
typedef struct { double *dr,*di,*tr,*ti; const int *map; size_t K; } cyc_t;
static void k_cycle(void *v){ cyc_t *c=v; size_t K=c->K; const int *M=c->map;
    for(int start=0;start<NN;start++){
        int m=M[start]; int mn=start;                       /* min-of-cycle rule */
        while(m!=start){ if(m<mn){mn=-1;break;} m=M[m]; }
        if(mn<0) continue;
        memcpy(c->tr,c->dr+(size_t)start*K,K*8); memcpy(c->ti,c->di+(size_t)start*K,K*8);
        int cur=start;
        for(;;){ int nxt=M[cur]; if(nxt==start) break;
            memcpy(c->dr+(size_t)cur*K,c->dr+(size_t)nxt*K,K*8);
            memcpy(c->di+(size_t)cur*K,c->di+(size_t)nxt*K,K*8); cur=nxt; }
        memcpy(c->dr+(size_t)cur*K,c->tr,K*8); memcpy(c->di+(size_t)cur*K,c->ti,K*8); } }
/* PURE gather into scratch then copy back (2-pass) */
typedef struct { double *dr,*di,*gr,*gi; const int *map; size_t K; } gat_t;
static void k_gather(void *v){ gat_t *g=v; size_t K=g->K; const int *M=g->map;
    for(int n=0;n<NN;n++){ memcpy(g->gr+(size_t)n*K,g->dr+(size_t)M[n]*K,K*8);
                           memcpy(g->gi+(size_t)n*K,g->di+(size_t)M[n]*K,K*8); }
    memcpy(g->dr,g->gr,(size_t)NN*K*8); memcpy(g->di,g->gi,(size_t)NN*K*8); }

/* FFT baseline: 8 fwd per timed chunk, rescale untimed */
typedef struct { vfft_plan p; double *re,*im; size_t n; } fftctx_t;
static double time_fft(fftctx_t *f){
    double best=1e30;
    for(int o=0;o<5;o++){
        for(size_t i=0;i<f->n;i++){ f->re[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; f->im[i]=(double)((i*40503u)&1023)/1024.0-0.5; }
        double acc=0; int chunks=4;
        for(int c=0;c<chunks;c++){
            double t0=qpc_ns();
            for(int r=0;r<8;r++) vfft_execute(f->p,VFFT_FORWARD,f->re,f->im,f->re,f->im);
            acc+=qpc_ns()-t0;
            double mx=0; for(size_t i=0;i<f->n;i+=97){ double a=fabs(f->re[i]); if(a>mx)mx=a; }
            if(mx>1e100||mx<1e-100){ double s=(mx>0)?1.0/mx:1.0; for(size_t i=0;i<f->n;i++){f->re[i]*=s;f->im[i]*=s;} }
        }
        double per=acc/(8.0*chunks); if(per<best)best=per; }
    return best;
}

static void run_cell(size_t K,const int *fac,int nf,int dif_hint){
    size_t n=(size_t)NN*K;
    printf("\n================ N=%d K=%zu chain=",NN,K);
    for(int i=0;i<nf;i++) printf("%d%s",fac[i],i<nf-1?"x":"");
    printf(" (%s-calibrated) ================\n",dif_hint?"DIF":"DIT");
    double *re=_aligned_malloc(n*8,64),*im=_aligned_malloc(n*8,64);
    double *sr=_aligned_malloc(n*8,64),*si=_aligned_malloc(n*8,64);
    double *dr=_aligned_malloc(n*8,64),*di=_aligned_malloc(n*8,64);
    double *Xr=malloc(NN*8),*Xi=malloc(NN*8);
    int *pf=malloc(NN*4),*ipf=malloc(NN*4),*pr=malloc(NN*4),*ipr=malloc(NN*4);
    int frev[16]; for(int i=0;i<nf;i++) frev[i]=fac[nf-1-i];
    mk_perm(NN,fac,nf,pf);  inv_perm(NN,pf,ipf);   /* forward-order digit reversal  */
    mk_perm(NN,frev,nf,pr); inv_perm(NN,pr,ipr);   /* reversed-order (DIF) variant  */

    /* plan (wisdom hit; JIT resolve may take a moment) */
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=NN; c.howmany=K; c.nthreads=1;
    vfft_plan p=vfft_create(&c);
    if(!p){ printf("plan NULL — wisdom copy missing the cell?\n"); return; }

    /* ---- 2. perm validation vs naive DFT (lane 0) ---- */
    srand(12345+(int)K);
    for(size_t i=0;i<n;i++){ re[i]=(double)rand()/RAND_MAX-0.5; im[i]=(double)rand()/RAND_MAX-0.5; }
    memcpy(sr,re,n*8); memcpy(si,im,n*8);
    naive_dft_lane0(re,im,K,Xr,Xi);
    vfft_execute(p,VFFT_FORWARD,re,im,re,im);      /* re/im now scrambled spectrum */
    const int *cand[4]={pf,ipf,pr,ipr}; const char *cname[4]={"perm(fwd-order)","iperm(fwd-order)","perm(rev-order)","iperm(rev-order)"};
    int best_c=-1; double best_e=1e30;
    for(int ci=0;ci<4;ci++){ double e=0;
        for(int k=0;k<NN;k++){ double d1=fabs(re[(size_t)cand[ci][k]*K]-Xr[k]),d2=fabs(im[(size_t)cand[ci][k]*K]-Xi[k]);
            if(d1>e)e=d1; if(d2>e)e=d2; }
        if(e<best_e){best_e=e;best_c=ci;} }
    printf("order probe: natural[k]=scrambled[M[k]] with M=%s  maxerr=%.2e %s\n",
           cname[best_c],best_e,best_e<1e-6?"(MATCH)":"(NO MATCH — all candidates failed!)");
    const int *M=cand[best_c]; int *IM=malloc(NN*4); inv_perm(NN,(int*)M,IM);

    /* ---- 3. comb/block diagnostic on the real chain ---- */
    int R=fac[nf-1], P=NN/R;
    int contig=1;
    for(int q=0;q<P&&contig;q+=(P/7)+1){ int b0=M[q]; /* block base? check run */
        for(int j=1;j<R;j++){ if(M[q+j*P]!=b0+j*0){ /* comb reads: rows M[q+jP] */ }
        }
        /* check the R source rows {M[q+j*P]} form one contiguous ascending block */
        int mn=M[q],mx=M[q];
        for(int j=1;j<R;j++){ int v=M[q+j*P]; if(v<mn)mn=v; if(v>mx)mx=v; }
        if(mx-mn!=R-1) contig=0; }
    printf("comb algebra: sources of natural comb {q+j*P} contiguous R-row block? %s (R=%d P=%d)\n",
           contig?"YES":"NO",R,P);

    /* ---- 1. FFT baseline ---- */
    fftctx_t fc={p,re,im,n};
    double t_fft=time_fft(&fc);
    printf("baseline FFT (public API, calibrated+JIT): %.0f ns  (wisdom best_ns anchor: K=4:20957 / K=32:142070)\n",t_fft);

    /* ---- 4. pattern kernels ---- */
    for(size_t i=0;i<n;i++){ sr[i]=1.0+(double)(i&255); si[i]=2.0+(double)(i&127); dr[i]=0; di[i]=0; }
    int inner = (int)(20e6/( (double)n*16.0 * 0.3 ))+1; /* aim >=20ms per timed run */
    pat_t so={dr,di,sr,si,NULL,K,R,P};             double t_so =time_kernel(k_sameorder,&so,inner);
    pat_t ip={dr,di,sr,si,NULL,K,R,P};             double t_ip =time_kernel(k_inplacepass,&ip,inner);
    pat_t sq={dr,di,sr,si,M,K,R,P};                double t_sq =time_kernel(k_scatter_q,&sq,inner);
    pat_t sb={dr,di,sr,si,IM,K,R,P};               double t_sb =time_kernel(k_scatter_b,&sb,inner);
    pat_t ss={dr,di,sr,si,M,K,R,P};                double t_ss =time_kernel(k_scatter_s,&ss,inner);
    double *tr=_aligned_malloc(K*8,64),*ti=_aligned_malloc(K*8,64);
    cyc_t cy={dr,di,tr,ti,M,K};                    double t_cy =time_kernel(k_cycle,&cy,inner);
    double *gr=_aligned_malloc(n*8,64),*gi=_aligned_malloc(n*8,64);
    gat_t ga={dr,di,gr,gi,M,K};                    double t_ga =time_kernel(k_gather,&ga,inner);

    double sc_best=t_sq; const char*sc_which="q-order";
    if(t_sb<sc_best){sc_best=t_sb;sc_which="b-order";}
    if(t_ss<sc_best){sc_best=t_ss;sc_which="s-stream";}
    double pu_best=t_cy<t_ga?t_cy:t_ga; const char*pu_which=t_cy<t_ga?"cycle":"gather";
    printf("\n  kernel                ns          GB/s(2pl r+w)\n");
    double bytes=(double)n*8*2*2; /* 2 planes, read+write */
    printf("  same-order copy   %9.0f   %6.1f\n",t_so,bytes/t_so);
    printf("  in-place pass     %9.0f   %6.1f   (last-stage memory proxy)\n",t_ip,bytes/t_ip);
    printf("  scatter-q         %9.0f   %6.1f   ratio vs same-order: %.2fx\n",t_sq,bytes/t_sq,t_so/t_sq);
    printf("  scatter-b         %9.0f   %6.1f   ratio vs same-order: %.2fx\n",t_sb,bytes/t_sb,t_so/t_sb);
    printf("  scatter-s(j-out)  %9.0f   %6.1f   ratio vs same-order: %.2fx\n",t_ss,bytes/t_ss,t_so/t_ss);
    printf("  PURE cycle        %9.0f   %6.1f\n",t_cy,bytes/t_cy);
    printf("  PURE gather(2p)   %9.0f   %6.1f\n",t_ga,bytes/t_ga);
    printf("\n  >> scatter bandwidth ratio (best=%s): %.2fx  [design assumed 0.60-0.85x]\n",sc_which,t_so/sc_best);
    printf("  >> est SCRATCH overhead = (scatter_best - inplace_pass)/FFT = (%.0f - %.0f)/%.0f = %+.1f%%\n",
           sc_best,t_ip,t_fft,100.0*(sc_best-t_ip)/t_fft);
    printf("  >> PURE overhead (best=%s) = %.0f/%.0f = +%.1f%%\n",pu_which,pu_best,t_fft,100.0*pu_best/t_fft);

    vfft_destroy(p);
    _aligned_free(re);_aligned_free(im);_aligned_free(sr);_aligned_free(si);_aligned_free(dr);_aligned_free(di);
    _aligned_free(tr);_aligned_free(ti);_aligned_free(gr);_aligned_free(gi);
    free(Xr);free(Xi);free(pf);free(ipf);free(pr);free(ipr);free(IM);
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1);   /* pin core 0 */
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    putenv("VFFT_WISDOM_DIR=natorder_wis_p0");
    printf("# Phase-0 scatter GO/NO-GO on REAL calibrated chains (N=4096; wisdom copy natorder_wis_p0)\n");
    int f4[4]={4,4,8,32};   run_cell(4, f4,4,1);   /* K=4  : DIF-calibrated */
    int f32[5]={4,4,4,8,8}; run_cell(32,f32,5,0);  /* K=32 : DIT */
    return 0;
}
