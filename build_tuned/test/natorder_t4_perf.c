/* natorder_t4_perf.c — T4 perf triangle for LEAF-IP (natural-order in-place via aliased n1_oop).
 * Cells = the 6 wisdom-covered ones (user: no new calibration): N{16,32,64,128}xK4, N{64,128}xK64.
 * Per cell, QPC best-of-5, pinned core 0, adaptive inner reps (~10ms/sample), rescale between chunks:
 *   (i)  public-API in-place c2c fwd (calibrated plan)     — the scrambled baseline
 *   (ii) nr_ leaf ALIASED dst==src (no-restrict build)     — LEAF-IP natural
 *   (iii)rr_ leaf separate-dst (restrict, as shipped)      — the OOP-style call
 *   (iv) nr_ leaf separate-dst                             — restrict-removal OOP regression check
 * Ratios: (ii)/(i) natural-vs-scrambled cost; (ii)/(iii) aliased-vs-OOP; (iv)/(iii) Option-A tax.
 * Build: python build.py --src test/natorder_t4_perf.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>
#include <stddef.h>
#include "vfft.h"

/* restrict-intact renamed copies (BEFORE the neutralizing define) */
#define radix16_n1_oop_fwd_avx2_UG_UG  rr_radix16
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix16_n1_oop_avx2.c"
#undef  radix16_n1_oop_fwd_avx2_UG_UG
#define radix32_n1_oop_fwd_avx2_UG_UG  rr_radix32
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix32_n1_oop_avx2.c"
#undef  radix32_n1_oop_fwd_avx2_UG_UG
#define radix64_n1_oop_fwd_avx2_UG_UG  rr_radix64
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix64_n1_oop_avx2.c"
#undef  radix64_n1_oop_fwd_avx2_UG_UG
#define radix128_n1_oop_fwd_avx2_UG_UG rr_radix128
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix128_n1_oop_avx2.c"
#undef  radix128_n1_oop_fwd_avx2_UG_UG
/* no-restrict copies */
#define __restrict__
#define __restrict
#define radix16_n1_oop_fwd_avx2_UG_UG  nr_radix16
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix16_n1_oop_avx2.c"
#undef  radix16_n1_oop_fwd_avx2_UG_UG
#define radix32_n1_oop_fwd_avx2_UG_UG  nr_radix32
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix32_n1_oop_avx2.c"
#undef  radix32_n1_oop_fwd_avx2_UG_UG
#define radix64_n1_oop_fwd_avx2_UG_UG  nr_radix64
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix64_n1_oop_avx2.c"
#undef  radix64_n1_oop_fwd_avx2_UG_UG
#define radix128_n1_oop_fwd_avx2_UG_UG nr_radix128
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix128_n1_oop_avx2.c"
#undef  radix128_n1_oop_fwd_avx2_UG_UG

typedef void (*leaf_fn)(const double*,const double*,double*,double*,const double*,const double*,
                        size_t,size_t,size_t,size_t,size_t);
static leaf_fn RR(int N){ return N==16?rr_radix16:N==32?rr_radix32:N==64?rr_radix64:rr_radix128; }
static leaf_fn NR(int N){ return N==16?nr_radix16:N==32?nr_radix32:N==64?nr_radix64:nr_radix128; }

static double qpc_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

static void refill(double *re,double *im,size_t n){ for(size_t i=0;i<n;i++){
    re[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; im[i]=(double)((i*40503u)&1023)/1024.0-0.5; } }
static void rescale(double *re,double *im,size_t n){ double mx=0;
    for(size_t i=0;i<n;i+=17){ double a=fabs(re[i]); if(a>mx)mx=a; }
    if(mx>1e100||mx<1e-100){ double s=mx>0?1.0/mx:1.0; for(size_t i=0;i<n;i++){re[i]*=s;im[i]*=s;} } }

/* growing-value kernels (aliased / in-place API): time chunks of 8, rescale untimed */
#define TIME_GROWING(CALL) do{ best=1e30; \
    for(int o=0;o<5;o++){ refill(are,aim,n); double acc=0; int done=0; \
        while(done<inner){ double t0=qpc_ns(); for(int r=0;r<8;r++){ CALL; } acc+=qpc_ns()-t0; done+=8; rescale(are,aim,n);} \
        double per=acc/done; if(per<best)best=per; } }while(0)
/* stable kernels (separate dst): plain loop */
#define TIME_STABLE(CALL) do{ best=1e30; \
    for(int o=0;o<5;o++){ double t0=qpc_ns(); for(int r=0;r<inner;r++){ CALL; } double per=(qpc_ns()-t0)/inner; if(per<best)best=per; } }while(0)

static void cell(int N, size_t K)
{
    size_t n=(size_t)N*K;
    double *sre=_aligned_malloc(n*8,64),*sim=_aligned_malloc(n*8,64);
    double *dre=_aligned_malloc(n*8,64),*dim=_aligned_malloc(n*8,64);
    double *are=_aligned_malloc(n*8,64),*aim=_aligned_malloc(n*8,64);
    refill(sre,sim,n); memset(dre,0,n*8); memset(dim,0,n*8);

    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=1;
    vfft_plan p=vfft_create(&c);
    if(!p){ printf("N=%-4d K=%-3zu  plan NULL\n",N,K); return; }

    leaf_fn rr=RR(N), nrf=NR(N);
    /* adaptive inner: rough warmup on the OOP call */
    double t0=qpc_ns(); for(int r=0;r<64;r++) rr(sre,sim,dre,dim,NULL,NULL,K,1,K,1,K);
    double est=(qpc_ns()-t0)/64; int inner=(int)(1e7/(est>1?est:1)); if(inner<200)inner=200; if(inner>400000)inner=400000;
    inner=(inner+7)&~7;

    double best;
    TIME_GROWING( vfft_execute(p,VFFT_FORWARD,are,aim,are,aim) );            double t_ip=best;
    TIME_GROWING( nrf(are,aim,are,aim,NULL,NULL,K,1,K,1,K) );                double t_leafip=best;
    TIME_STABLE ( rr (sre,sim,dre,dim,NULL,NULL,K,1,K,1,K) );                double t_oop_rr=best;
    TIME_STABLE ( nrf(sre,sim,dre,dim,NULL,NULL,K,1,K,1,K) );                double t_oop_nr=best;

    printf("N=%-4d K=%-3zu inner=%-6d | inplace %8.1f | LEAF-IP %8.1f (%.2fx of inplace) | OOP-rr %8.1f (aliased=%.2fx) | OOP-nr %8.1f (optA tax %.2fx)\n",
           N,K,inner,t_ip,t_leafip,t_leafip/t_ip,t_oop_rr,t_leafip/t_oop_rr,t_oop_nr,t_oop_nr/t_oop_rr);
    vfft_destroy(p);
    _aligned_free(sre);_aligned_free(sim);_aligned_free(dre);_aligned_free(dim);_aligned_free(are);_aligned_free(aim);
}

int main(void)
{
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    putenv("VFFT_WISDOM_DIR=natorder_wis_p0");
    printf("# T4 perf triangle (ns/call, best-of-5). LEAF-IP = no-restrict aliased leaf = natural-order in-place.\n");
    printf("# K=4 N<=64 baselines are nf=1 monolithic (ALREADY natural = FREE mode) - LEAF-IP races FREE there.\n");
    cell(16,4); cell(32,4); cell(64,4); cell(128,4);
    cell(64,64); cell(128,64);
    return 0;
}
