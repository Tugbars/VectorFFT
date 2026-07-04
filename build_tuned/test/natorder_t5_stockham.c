/* natorder_t5_stockham.c — T5: does STOCKHAM ping-pong deliver natural order at ~0% for
 * cache-resident cells?  COMPUTE-MATCHED design: ONE generic AVX2 radix-4 butterfly drives both
 *   (a) generic IN-PLACE DIF  (natural in -> digit-reversed out, zero data movement — our engine's
 *       access-pattern class), and
 *   (b) generic STOCKHAM      (natural in -> NATURAL out, ping-pong A<->B, regular strides).
 * Identical kernel, identical flop count (3 twiddle cmuls + DFT4 per group, nf stages) — the ratio
 * (b)/(a) isolates the MEMORY-PATTERN cost of Stockham's natural order. Tuned public-API time shown
 * as context only. Cells (all 4^n chains): 1024/4 (128KB w/ pingpong), 4096/4 (512KB), 1024/32 (1MB)
 * = in-region (P-core L2 = 2MB); 4096/32 (4MB) = out-of-region CONTROL (expect Stockham to lose).
 * Correctness: (b) vs naive DFT directly; (a) vs naive through the base-4 digit reversal (uniform
 * chain => involution, perm==iperm). Build: python build.py --src test/natorder_t5_stockham.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>
#include "vfft.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double qpc_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

/* ---- DFT4 on one 4-lane vector pair per leg (W = e^{-2pi i/4}) ---- */
#define DFT4(ar,ai,br,bi,cr,ci,dr,di, x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i) do{ \
    __m256d _u0r=_mm256_add_pd(ar,cr), _u0i=_mm256_add_pd(ai,ci); \
    __m256d _u1r=_mm256_sub_pd(ar,cr), _u1i=_mm256_sub_pd(ai,ci); \
    __m256d _u2r=_mm256_add_pd(br,dr), _u2i=_mm256_add_pd(bi,di); \
    __m256d _u3r=_mm256_sub_pd(br,dr), _u3i=_mm256_sub_pd(bi,di); \
    x0r=_mm256_add_pd(_u0r,_u2r); x0i=_mm256_add_pd(_u0i,_u2i); \
    x2r=_mm256_sub_pd(_u0r,_u2r); x2i=_mm256_sub_pd(_u0i,_u2i); \
    x1r=_mm256_add_pd(_u1r,_u3i); x1i=_mm256_sub_pd(_u1i,_u3r);   /* u1 - i*u3 */ \
    x3r=_mm256_sub_pd(_u1r,_u3i); x3i=_mm256_add_pd(_u1i,_u3r);   /* u1 + i*u3 */ \
}while(0)
#define CMUL(vr,vi,wr,wi, or_,oi_) do{ /* (vr+ivi)*(wr+iwi) */ \
    or_=_mm256_sub_pd(_mm256_mul_pd(vr,wr),_mm256_mul_pd(vi,wi)); \
    oi_=_mm256_add_pd(_mm256_mul_pd(vr,wi),_mm256_mul_pd(vi,wr)); \
}while(0)

/* twiddle tables: one per stage, tw[j][r-1], r=1..3, W_{P}^{j*r} with stage-specific P and j-range */
typedef struct { double *re, *im; int jn; } twtab_t;
static twtab_t mk_tw(int jn, int P){ twtab_t t; t.jn=jn;
    t.re=_aligned_malloc((size_t)jn*3*8,64); t.im=_aligned_malloc((size_t)jn*3*8,64);
    for(int j=0;j<jn;j++) for(int r=1;r<4;r++){ double a=-2.0*M_PI*(double)j*r/(double)P;
        t.re[j*3+r-1]=cos(a); t.im[j*3+r-1]=sin(a); }
    return t; }

/* ---- (a) generic IN-PLACE DIF: natural in -> base-4-digit-reversed out. nf stages, zero movement.
 * stage (L): blocks of L rows; M=L/4; legs base+j+r*M; DFT4; post-twiddle y_q *= W_L^{j*q}. ---- */
static void inplace_dif(double *re, double *im, int N, size_t K, twtab_t *tws, int nf)
{
    int L=N;
    for(int s=0;s<nf;s++){
        int M=L/4; twtab_t *tw=&tws[s];
        for(int base=0;base<N;base+=L)
          for(int j=0;j<M;j++){
            size_t i0=((size_t)base+j)*K;
            for(size_t c=0;c<K;c+=4){
                __m256d ar=_mm256_loadu_pd(re+i0+(size_t)0*M*K+c), ai=_mm256_loadu_pd(im+i0+(size_t)0*M*K+c);
                __m256d br=_mm256_loadu_pd(re+i0+(size_t)1*M*K+c), bi=_mm256_loadu_pd(im+i0+(size_t)1*M*K+c);
                __m256d cr=_mm256_loadu_pd(re+i0+(size_t)2*M*K+c), ci=_mm256_loadu_pd(im+i0+(size_t)2*M*K+c);
                __m256d dr=_mm256_loadu_pd(re+i0+(size_t)3*M*K+c), di=_mm256_loadu_pd(im+i0+(size_t)3*M*K+c);
                __m256d x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i;
                DFT4(ar,ai,br,bi,cr,ci,dr,di, x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i);
                __m256d w1r=_mm256_set1_pd(tw->re[j*3+0]), w1i=_mm256_set1_pd(tw->im[j*3+0]);
                __m256d w2r=_mm256_set1_pd(tw->re[j*3+1]), w2i=_mm256_set1_pd(tw->im[j*3+1]);
                __m256d w3r=_mm256_set1_pd(tw->re[j*3+2]), w3i=_mm256_set1_pd(tw->im[j*3+2]);
                __m256d y1r,y1i,y2r,y2i,y3r,y3i;
                CMUL(x1r,x1i,w1r,w1i,y1r,y1i); CMUL(x2r,x2i,w2r,w2i,y2r,y2i); CMUL(x3r,x3i,w3r,w3i,y3r,y3i);
                _mm256_storeu_pd(re+i0+(size_t)0*M*K+c,x0r); _mm256_storeu_pd(im+i0+(size_t)0*M*K+c,x0i);
                _mm256_storeu_pd(re+i0+(size_t)1*M*K+c,y1r); _mm256_storeu_pd(im+i0+(size_t)1*M*K+c,y1i);
                _mm256_storeu_pd(re+i0+(size_t)2*M*K+c,y2r); _mm256_storeu_pd(im+i0+(size_t)2*M*K+c,y2i);
                _mm256_storeu_pd(re+i0+(size_t)3*M*K+c,y3r); _mm256_storeu_pd(im+i0+(size_t)3*M*K+c,y3i);
            } }
        L=M;
    }
}

/* ---- (b) generic STOCKHAM: natural in -> natural out, ping-pong. stage (L, M=N/(4L)):
 * a_r = src[j+L(k+Mr)] * W_{4L}^{jr}; DFT4; dst[j+L(q+4k)] = b_q.  Returns which buffer holds out. */
static int stockham(double *Ar, double *Ai, double *Br, double *Bi, int N, size_t K, twtab_t *tws, int nf)
{
    double *sr=Ar,*si=Ai,*dr_=Br,*di_=Bi;
    int L=1;
    for(int s=0;s<nf;s++){
        int M=N/(4*L); twtab_t *tw=&tws[s];
        for(int k=0;k<M;k++)
          for(int j=0;j<L;j++){
            size_t is0=((size_t)j+(size_t)L*k)*K, stp=(size_t)L*M*K;
            size_t id0=((size_t)j+(size_t)L*4*k)*K;
            for(size_t c=0;c<K;c+=4){
                __m256d ar=_mm256_loadu_pd(sr+is0+0*stp+c), ai=_mm256_loadu_pd(si+is0+0*stp+c);
                __m256d br=_mm256_loadu_pd(sr+is0+1*stp+c), bi=_mm256_loadu_pd(si+is0+1*stp+c);
                __m256d cr=_mm256_loadu_pd(sr+is0+2*stp+c), ci=_mm256_loadu_pd(si+is0+2*stp+c);
                __m256d dr=_mm256_loadu_pd(sr+is0+3*stp+c), di=_mm256_loadu_pd(si+is0+3*stp+c);
                __m256d w1r=_mm256_set1_pd(tw->re[j*3+0]), w1i=_mm256_set1_pd(tw->im[j*3+0]);
                __m256d w2r=_mm256_set1_pd(tw->re[j*3+1]), w2i=_mm256_set1_pd(tw->im[j*3+1]);
                __m256d w3r=_mm256_set1_pd(tw->re[j*3+2]), w3i=_mm256_set1_pd(tw->im[j*3+2]);
                __m256d t1r,t1i,t2r,t2i,t3r,t3i;
                CMUL(br,bi,w1r,w1i,t1r,t1i); CMUL(cr,ci,w2r,w2i,t2r,t2i); CMUL(dr,di,w3r,w3i,t3r,t3i);
                __m256d x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i;
                DFT4(ar,ai,t1r,t1i,t2r,t2i,t3r,t3i, x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i);
                _mm256_storeu_pd(dr_+id0+(size_t)0*L*K+c,x0r); _mm256_storeu_pd(di_+id0+(size_t)0*L*K+c,x0i);
                _mm256_storeu_pd(dr_+id0+(size_t)1*L*K+c,x1r); _mm256_storeu_pd(di_+id0+(size_t)1*L*K+c,x1i);
                _mm256_storeu_pd(dr_+id0+(size_t)2*L*K+c,x2r); _mm256_storeu_pd(di_+id0+(size_t)2*L*K+c,x2i);
                _mm256_storeu_pd(dr_+id0+(size_t)3*L*K+c,x3r); _mm256_storeu_pd(di_+id0+(size_t)3*L*K+c,x3i);
            } }
        { double *t; t=sr;sr=dr_;dr_=t; t=si;si=di_;di_=t; }
        L*=4;
    }
    return sr==Ar ? 0 : 1;   /* which buffer holds the result */
}

static void naive_dft_lane0(const double *re,const double *im,int N,size_t K,double *Xr,double *Xi){
    for(int k=0;k<N;k++){ double sr=0,si=0;
        for(int n=0;n<N;n++){ double a=-2.0*M_PI*(double)k*n/N,c=cos(a),s=sin(a);
            double xr=re[(size_t)n*K],xi=im[(size_t)n*K];
            sr+=xr*c-xi*s; si+=xr*s+xi*c; }
        Xr[k]=sr; Xi[k]=si; } }

static void refill(double *re,double *im,size_t n){ for(size_t i=0;i<n;i++){
    re[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; im[i]=(double)((i*40503u)&1023)/1024.0-0.5; } }
static void rescale(double *re,double *im,size_t n){ double mx=0;
    for(size_t i=0;i<n;i+=13){ double a=fabs(re[i]); if(a>mx)mx=a; }
    if(mx>1e80||mx<1e-80){ double s=mx>0?1.0/mx:1.0; for(size_t i=0;i<n;i++){re[i]*=s;im[i]*=s;} } }

static void cell(int N, size_t K)
{
    int nf=0; for(int t=N;t>1;t/=4) nf++;
    size_t n=(size_t)N*K;
    double *Ar=_aligned_malloc(n*8,64),*Ai=_aligned_malloc(n*8,64);
    double *Br=_aligned_malloc(n*8,64),*Bi=_aligned_malloc(n*8,64);
    double *Xr=malloc((size_t)N*8),*Xi=malloc((size_t)N*8);
    /* twiddle tables (plan time) */
    twtab_t twd[8],tws[8];
    { int L=N; for(int s=0;s<nf;s++){ twd[s]=mk_tw(L/4,L); L/=4; } }         /* DIF: j<M=L/4, P=L    */
    { int L=1; for(int s=0;s<nf;s++){ tws[s]=mk_tw(L,4*L);  L*=4; } }        /* Stockham: j<L, P=4L  */

    /* correctness */
    srand(55+N+(int)K);
    for(size_t i=0;i<n;i++){ Ar[i]=(double)rand()/RAND_MAX-0.5; Ai[i]=(double)rand()/RAND_MAX-0.5; }
    naive_dft_lane0(Ar,Ai,N,K,Xr,Xi);
    memcpy(Br,Ar,n*8); memcpy(Bi,Ai,n*8);     /* keep pristine in B for the DIF run */
    /* Stockham correctness (fresh buffers) */
    double *Cr=_aligned_malloc(n*8,64),*Ci=_aligned_malloc(n*8,64);
    memcpy(Cr,Br,n*8); memcpy(Ci,Bi,n*8);
    double *Dr=_aligned_malloc(n*8,64),*Di=_aligned_malloc(n*8,64);
    int ob=stockham(Cr,Ci,Dr,Di,N,K,tws,nf);
    double *or_=ob?Dr:Cr, *oi_=ob?Di:Ci;
    double eS=0,sc=0;
    for(int k=0;k<N;k++){ double d1=fabs(or_[(size_t)k*K]-Xr[k]),d2=fabs(oi_[(size_t)k*K]-Xi[k]);
        if(d1>eS)eS=d1; if(d2>eS)eS=d2; if(fabs(Xr[k])>sc)sc=fabs(Xr[k]); }
    eS/= sc>0?sc:1;
    /* DIF correctness through the base-4 digit reversal (involution) */
    memcpy(Ar,Br,n*8); memcpy(Ai,Bi,n*8);
    inplace_dif(Ar,Ai,N,K,twd,nf);
    int *perm=malloc((size_t)N*4);
    for(int m=0;m<N;m++){ int t=m,p=0; for(int s=0;s<nf;s++){ p=p*4+(t&3); t>>=2; } perm[m]=p; }
    double eD=0;
    for(int k=0;k<N;k++){ double d1=fabs(Ar[(size_t)perm[k]*K]-Xr[k]),d2=fabs(Ai[(size_t)perm[k]*K]-Xi[k]);
        if(d1>eD)eD=d1; if(d2>eD)eD=d2; }
    eD/= sc>0?sc:1;

    /* tuned public-API context (wisdom-covered cells only; no calibration) */
    double t_api=-1;
    { vfft_config_t c; memset(&c,0,sizeof c);
      c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
      c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=1;
      vfft_plan p=vfft_create(&c);
      if(p){ double best=1e30; int inner=(int)(8e6/((double)n*nf))+8; inner&=~7; if(inner<8)inner=8;
        for(int o=0;o<5;o++){ refill(Ar,Ai,n); double acc=0; int done=0;
            while(done<inner){ double t0=qpc_ns();
                for(int r=0;r<8;r++) vfft_execute(p,VFFT_FORWARD,Ar,Ai,Ar,Ai);
                acc+=qpc_ns()-t0; done+=8; rescale(Ar,Ai,n); }
            double per=acc/done; if(per<best)best=per; }
        t_api=best; vfft_destroy(p); } }

    /* (a) generic in-place DIF timing */
    double t_dif; { double best=1e30; int inner=(int)(1.6e7/((double)n*nf))+8; inner&=~7; if(inner<8)inner=8;
        for(int o=0;o<5;o++){ refill(Ar,Ai,n); double acc=0; int done=0;
            while(done<inner){ double t0=qpc_ns();
                for(int r=0;r<8;r++) inplace_dif(Ar,Ai,N,K,twd,nf);
                acc+=qpc_ns()-t0; done+=8; rescale(Ar,Ai,n); }
            double per=acc/done; if(per<best)best=per; }
        t_dif=best; }
    /* (b) generic Stockham timing (ping-pong; feed next iteration from wherever output landed) */
    double t_st; { double best=1e30; int inner=(int)(1.6e7/((double)n*nf))+8; inner&=~7; if(inner<8)inner=8;
        for(int o=0;o<5;o++){ refill(Cr,Ci,n); double acc=0; int done=0;
            double *s0r=Cr,*s0i=Ci,*s1r=Dr,*s1i=Di;
            while(done<inner){ double t0=qpc_ns();
                for(int r=0;r<8;r++){ int b=stockham(s0r,s0i,s1r,s1i,N,K,tws,nf);
                    if(b){ double *t; t=s0r;s0r=s1r;s1r=t; t=s0i;s0i=s1i;s1i=t; } }
                acc+=qpc_ns()-t0; done+=8; rescale(s0r,s0i,n); }
            double per=acc/done; if(per<best)best=per; }
        t_st=best; }

    double ws=(double)n*16.0*2.0/1024.0;   /* both buffers, KB */
    printf("N=%-5d K=%-3zu nf=%d ws(pp)=%6.0fKB | DIF-generic %9.0f | STOCKHAM %9.0f | ratio %.3f | tuned-api %8.0f | err S=%.1e D=%.1e %s\n",
        N,K,nf,ws,t_dif,t_st,t_st/t_dif,t_api,eS,eD,(eS<1e-9&&eD<1e-9)?"ok":"<MATH FAIL>");
    _aligned_free(Ar);_aligned_free(Ai);_aligned_free(Br);_aligned_free(Bi);
    _aligned_free(Cr);_aligned_free(Ci);_aligned_free(Dr);_aligned_free(Di);
    free(Xr);free(Xi);free(perm);
    for(int s=0;s<nf;s++){ _aligned_free(twd[s].re);_aligned_free(twd[s].im);_aligned_free(tws[s].re);_aligned_free(tws[s].im); }
}

int main(void)
{
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    putenv("VFFT_WISDOM_DIR=natorder_wis_p0");
    printf("# T5: Stockham (natural) vs in-place DIF (scrambled), SAME generic radix-4 kernel. ratio=Stockham cost.\n");
    printf("# claim: ratio ~1.0 while ping-pong working set fits L2 (2MB); control cell should exceed 1.\n");
    cell(1024,4);    /* 128KB  in-region  */
    cell(4096,4);    /* 512KB  in-region  */
    cell(1024,32);   /*   1MB  in-region  */
    cell(4096,32);   /*   4MB  CONTROL    */
    return 0;
}
