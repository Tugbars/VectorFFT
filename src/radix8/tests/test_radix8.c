/**
 * test_r8_auto.c — Three-tier auto dispatch: packed table vs pack+walk
 * Also benchmarks packed vs pack+walk vs strided.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define R8_512_LD(p) _mm512_loadu_pd(p)
#define R8_512_ST(p,v) _mm512_storeu_pd((p),(v))
#define R8_256_LD(p) _mm256_loadu_pd(p)
#define R8_256_ST(p,v) _mm256_storeu_pd((p),(v))
#include "fft_radix8_dispatch.h"

static double now_ns(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1e9+ts.tv_nsec;
}
static double *alloc64(size_t n){
    double *p=NULL; posix_memalign((void**)&p,64,n*8); memset(p,0,n*8); return p;
}
static double maxerr(const double *ar,const double *ai,
                     const double *br,const double *bi,size_t n){
    double mx=0;
    for(size_t i=0;i<n;i++){
        double dr=fabs(ar[i]-br[i]),di=fabs(ai[i]-bi[i]);
        if(dr>mx)mx=dr; if(di>mx)mx=di;
    }
    return mx;
}

static void ref_tw_dft8(const double *ir, const double *ii,
                        double *or_, double *oi, size_t K, int fwd){
    const size_t NN=8*K;
    const double sign = fwd ? -1.0 : 1.0;
    for(size_t k=0;k<K;k++){
        double xr[8],xi[8];
        for(int n=0;n<8;n++){
            double dr=ir[n*K+k],di=ii[n*K+k];
            if(n>0){
                double a=sign*2.0*M_PI*(double)(n*k)/(double)NN;
                double wr=cos(a),wi=sin(a);
                double tr=dr*wr-di*wi; di=dr*wi+di*wr; dr=tr;
            }
            xr[n]=dr; xi[n]=di;
        }
        for(int m=0;m<8;m++){
            double sr=0,si=0;
            for(int n=0;n<8;n++){
                double a=sign*2.0*M_PI*(double)(m*n)/8.0;
                sr+=xr[n]*cos(a)-xi[n]*sin(a);
                si+=xr[n]*sin(a)+xi[n]*cos(a);
            }
            or_[m*K+k]=sr; oi[m*K+k]=si;
        }
    }
}

static void gen_flat_tw(double *re,double *im,size_t K){
    const size_t NN=8*K;
    for(int n=1;n<8;n++)
        for(size_t k=0;k<K;k++){
            double a=-2.0*M_PI*(double)(n*k)/(double)NN;
            re[(n-1)*K+k]=cos(a); im[(n-1)*K+k]=sin(a);
        }
}

__attribute__((target("avx512f")))
static void r8_pack(const double *sr,const double *si,double *dr,double *di,size_t K){
    for(size_t b=0;b<K/8;b++){
        size_t sk=b*8,dk=b*64;
        for(int n=0;n<8;n++){
            _mm512_storeu_pd(&dr[dk+n*8],_mm512_loadu_pd(&sr[n*K+sk]));
            _mm512_storeu_pd(&di[dk+n*8],_mm512_loadu_pd(&si[n*K+sk]));
        }
    }
}
__attribute__((target("avx512f")))
static void r8_unpack(const double *sr,const double *si,double *dr,double *di,size_t K){
    for(size_t b=0;b<K/8;b++){
        size_t sk=b*64,dk=b*8;
        for(int n=0;n<8;n++){
            _mm512_storeu_pd(&dr[n*K+dk],_mm512_loadu_pd(&sr[sk+n*8]));
            _mm512_storeu_pd(&di[n*K+dk],_mm512_loadu_pd(&si[sk+n*8]));
        }
    }
}
__attribute__((target("avx512f")))
static void r8_pack_tw(const double *sr,const double *si,double *dr,double *di,size_t K){
    for(size_t b=0;b<K/8;b++){
        size_t sk=b*8,dk=b*56;
        for(int n=0;n<7;n++){
            _mm512_storeu_pd(&dr[dk+n*8],_mm512_loadu_pd(&sr[n*K+sk]));
            _mm512_storeu_pd(&di[dk+n*8],_mm512_loadu_pd(&si[n*K+sk]));
        }
    }
}

int main(void){
    int total=0, passed=0;

    printf("=== Radix-8 Three-Tier Auto Dispatch Test ===\n");
    printf("  Walk threshold = %d\n\n", RADIX8_WALK_THRESHOLD);

    /* ── Correctness ── */
    printf("-- Correctness --\n");
    size_t Ks[]={8,64,256,512,1024,2048,4096};
    for(int ki=0;ki<7;ki++){
        size_t K=Ks[ki],NN=8*K,T=radix8_packed_optimal_T(K);
        double *ir=alloc64(NN),*ii=alloc64(NN);
        double *ref_r=alloc64(NN),*ref_i=alloc64(NN);
        double *out_r=alloc64(NN),*out_i=alloc64(NN);
        double *ftwr=alloc64(7*K),*ftwi=alloc64(7*K);
        double *pk_ir=alloc64(NN),*pk_ii=alloc64(NN);
        double *pk_or=alloc64(NN),*pk_oi=alloc64(NN);

        srand(42+(unsigned)K);
        for(size_t i=0;i<NN;i++){ir[i]=(double)rand()/RAND_MAX-.5;ii[i]=(double)rand()/RAND_MAX-.5;}
        gen_flat_tw(ftwr,ftwi,K);
        r8_pack(ir,ii,pk_ir,pk_ii,K);

        double *pk_twr=NULL,*pk_twi=NULL;
        if(!radix8_should_walk(K)){
            pk_twr=alloc64(7*K); pk_twi=alloc64(7*K);
            r8_pack_tw(ftwr,ftwi,pk_twr,pk_twi,K);
        }

        radix8_walk_plan_t wp512;
        void *wp=NULL;
        if(radix8_should_walk(K)){
            radix8_walk_plan_init(&wp512,K);
            wp=&wp512;
        }

        const char *mode = radix8_should_walk(K) ? "pk+walk" : "packed ";

        /* Forward */
        ref_tw_dft8(ir,ii,ref_r,ref_i,K,1);
        radix8_tw_packed_auto_fwd(pk_ir,pk_ii,pk_or,pk_oi,pk_twr,pk_twi,wp,K,T);
        r8_unpack(pk_or,pk_oi,out_r,out_i,K);
        double ef=maxerr(ref_r,ref_i,out_r,out_i,NN);

        /* Backward */
        ref_tw_dft8(ir,ii,ref_r,ref_i,K,0);
        r8_pack(ir,ii,pk_ir,pk_ii,K);
        if(radix8_should_walk(K)) radix8_walk_plan_init(&wp512,K);
        radix8_tw_packed_auto_bwd(pk_ir,pk_ii,pk_or,pk_oi,pk_twr,pk_twi,wp,K,T);
        r8_unpack(pk_or,pk_oi,out_r,out_i,K);
        double eb=maxerr(ref_r,ref_i,out_r,out_i,NN);

        int ok=(ef<1e-9 && eb<1e-9);
        total++; passed+=ok;
        printf("  K=%-5zu %s  fwd=%.1e  bwd=%.1e  %s\n", K, mode, ef, eb, ok?"PASS":"FAIL");

        free(ir);free(ii);free(ref_r);free(ref_i);free(out_r);free(out_i);
        free(pk_ir);free(pk_ii);free(pk_or);free(pk_oi);free(ftwr);free(ftwi);
        if(pk_twr)free(pk_twr); if(pk_twi)free(pk_twi);
    }

    printf("\n=== %d / %d passed ===\n\n", passed, total);

    /* ── Benchmark ── */
    printf("-- Benchmark: Packed-table vs Pack+Walk vs Strided (min-of-5, fwd only) --\n");
    printf("  %-5s | %8s %8s %8s | %7s %7s | winner\n",
           "K","packed","pk+walk","strided","pw/pk","pk/strd");
    printf("  ------+-----------------------------+-----------------+-------\n");

    size_t bKs[]={64,128,256,512,1024,2048,4096};
    for(int ki=0;ki<7;ki++){
        size_t K=bKs[ki],NN=8*K,T=8;
        double *ir=alloc64(NN),*ii=alloc64(NN);
        double *o1r=alloc64(NN),*o1i=alloc64(NN);
        double *ftwr=alloc64(7*K),*ftwi=alloc64(7*K);
        double *pk_ir=alloc64(NN),*pk_ii=alloc64(NN);
        double *pk_or=alloc64(NN),*pk_oi=alloc64(NN);
        double *pk_twr=alloc64(7*K),*pk_twi=alloc64(7*K);

        srand(42+(unsigned)K);
        for(size_t i=0;i<NN;i++){ir[i]=(double)rand()/RAND_MAX-.5;ii[i]=(double)rand()/RAND_MAX-.5;}
        gen_flat_tw(ftwr,ftwi,K);
        r8_pack(ir,ii,pk_ir,pk_ii,K);
        r8_pack_tw(ftwr,ftwi,pk_twr,pk_twi,K);

        radix8_walk_plan_t wp512;
        radix8_walk_plan_init(&wp512,K);

        int iters=(int)(1000000.0/K); if(iters<100)iters=100;
        int warmup=iters/10;
        double best_p=1e18,best_w=1e18,best_s=1e18;
        size_t nb=K/8;

        for(int run=0;run<5;run++){
            /* Packed table */
            for(int i=0;i<warmup;i++)
                for(size_t b=0;b<nb;b++)
                    radix8_tw_dit_kernel_fwd_avx512(pk_ir+b*64,pk_ii+b*64,pk_or+b*64,pk_oi+b*64,pk_twr+b*56,pk_twi+b*56,8);
            double t0=now_ns();
            for(int i=0;i<iters;i++)
                for(size_t b=0;b<nb;b++)
                    radix8_tw_dit_kernel_fwd_avx512(pk_ir+b*64,pk_ii+b*64,pk_or+b*64,pk_oi+b*64,pk_twr+b*56,pk_twi+b*56,8);
            double ns=(now_ns()-t0)/iters;
            if(ns<best_p)best_p=ns;

            /* Pack+walk */
            radix8_walk_plan_init(&wp512,K);
            for(int i=0;i<warmup;i++){
                radix8_walk_plan_init(&wp512,K);
                radix8_tw_pack_walk_fwd_avx512(pk_ir,pk_ii,pk_or,pk_oi,&wp512,K);
            }
            radix8_walk_plan_init(&wp512,K);
            t0=now_ns();
            for(int i=0;i<iters;i++){
                radix8_walk_plan_init(&wp512,K);
                radix8_tw_pack_walk_fwd_avx512(pk_ir,pk_ii,pk_or,pk_oi,&wp512,K);
            }
            ns=(now_ns()-t0)/iters;
            if(ns<best_w)best_w=ns;

            /* Strided tw */
            for(int i=0;i<warmup;i++)
                radix8_tw_dit_kernel_fwd_avx512(ir,ii,o1r,o1i,ftwr,ftwi,K);
            t0=now_ns();
            for(int i=0;i<iters;i++)
                radix8_tw_dit_kernel_fwd_avx512(ir,ii,o1r,o1i,ftwr,ftwi,K);
            ns=(now_ns()-t0)/iters;
            if(ns<best_s)best_s=ns;
        }

        double r_wp=best_p/best_w, r_ps=best_s/best_p;
        const char *win="PACKED"; double best=best_p;
        if(best_w<best){best=best_w;win="PK+WALK";}
        if(best_s<best){best=best_s;win="STRIDED";}

        printf("  K=%-4zu | %7.0f  %7.0f  %7.0f | %6.2fx %6.2fx | %s\n",
               K,best_p,best_w,best_s,r_wp,r_ps,win);

        free(ir);free(ii);free(o1r);free(o1i);free(ftwr);free(ftwi);
        free(pk_ir);free(pk_ii);free(pk_or);free(pk_oi);free(pk_twr);free(pk_twi);
    }

    return (passed==total)?0:1;
}