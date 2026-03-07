/**
 * test_r11_walk.c — Radix-11 pack+walk correctness + benchmark
 * Compares: packed-table vs pack+walk vs existing walk driver
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

#include "fft_radix11_genfft.h"
#include "fft_radix11_avx512_tw_pack_walk.h"

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

/* Reference DFT-11 with twiddles */
static void ref_tw_dft11(const double *ir,const double *ii,
                         double *or_,double *oi,size_t K,int fwd){
    const size_t NN=11*K;
    const double sign=fwd?-1.0:1.0;
    for(size_t k=0;k<K;k++){
        double xr[11],xi[11];
        for(int n=0;n<11;n++){
            double dr=ir[n*K+k],di=ii[n*K+k];
            if(n>0){
                double a=sign*2.0*M_PI*(double)(n*k)/(double)NN;
                double wr=cos(a),wi=sin(a);
                double tr=dr*wr-di*wi; di=dr*wi+di*wr; dr=tr;
            }
            xr[n]=dr; xi[n]=di;
        }
        for(int m=0;m<11;m++){
            double sr=0,si=0;
            for(int n=0;n<11;n++){
                double a=sign*2.0*M_PI*(double)(m*n)/11.0;
                sr+=xr[n]*cos(a)-xi[n]*sin(a);
                si+=xr[n]*sin(a)+xi[n]*cos(a);
            }
            or_[m*K+k]=sr; oi[m*K+k]=si;
        }
    }
}

__attribute__((target("avx512f")))
static void r11_pack_avx512(const double *sr,const double *si,
                            double *dr,double *di,size_t K){
    const size_t T=8;
    for(size_t b=0;b<K/T;b++)
        for(int n=0;n<11;n++)
            _mm512_store_pd(&dr[b*88+n*8],_mm512_loadu_pd(&sr[n*K+b*8])),
            _mm512_store_pd(&di[b*88+n*8],_mm512_loadu_pd(&si[n*K+b*8]));
}
__attribute__((target("avx512f")))
static void r11_unpack_avx512(const double *sr,const double *si,
                              double *dr,double *di,size_t K){
    const size_t T=8;
    for(size_t b=0;b<K/T;b++)
        for(int n=0;n<11;n++)
            _mm512_storeu_pd(&dr[n*K+b*8],_mm512_load_pd(&sr[b*88+n*8])),
            _mm512_storeu_pd(&di[n*K+b*8],_mm512_load_pd(&si[b*88+n*8]));
}

/* Pack twiddles: tw[(n-1)*K+k] → tw_pk[b*10*8 + n*8 + j] */
__attribute__((target("avx512f")))
static void r11_pack_tw(const double *sr,const double *si,
                        double *dr,double *di,size_t K){
    for(size_t b=0;b<K/8;b++)
        for(int n=0;n<10;n++)
            _mm512_store_pd(&dr[b*80+n*8],_mm512_loadu_pd(&sr[n*K+b*8])),
            _mm512_store_pd(&di[b*80+n*8],_mm512_loadu_pd(&si[n*K+b*8]));
}

/* Packed table driver: apply twiddles from packed table, call notw */
__attribute__((target("avx512f,fma")))
static void r11_tw_packed_table_fwd(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im,
    size_t K)
{
    const size_t T=8, dbs=11*T, tbs=10*T, nb=K/T;
    double __attribute__((aligned(64))) tmp_re[88], tmp_im[88];

    for(size_t blk=0;blk<nb;blk++){
        const double *bir=in_re+blk*dbs, *bii=in_im+blk*dbs;
        const double *btr=tw_re+blk*tbs, *bti=tw_im+blk*tbs;

        _mm512_store_pd(&tmp_re[0],_mm512_load_pd(&bir[0]));
        _mm512_store_pd(&tmp_im[0],_mm512_load_pd(&bii[0]));
        for(int n=0;n<10;n++){
            __m512d xr=_mm512_load_pd(&bir[(n+1)*T]);
            __m512d xi=_mm512_load_pd(&bii[(n+1)*T]);
            __m512d tr=_mm512_load_pd(&btr[n*T]);
            __m512d ti=_mm512_load_pd(&bti[n*T]);
            _mm512_store_pd(&tmp_re[(n+1)*T],
                _mm512_fnmadd_pd(xi,ti,_mm512_mul_pd(xr,tr)));
            _mm512_store_pd(&tmp_im[(n+1)*T],
                _mm512_fmadd_pd(xr,ti,_mm512_mul_pd(xi,tr)));
        }
        radix11_genfft_fwd_avx512(tmp_re,tmp_im,out_re+blk*dbs,out_im+blk*dbs,T);
    }
}

int main(void){
    printf("=== Radix-11 Pack+Walk Test ===\n\n");

    /* ── Correctness ── */
    printf("-- Correctness --\n");
    int total=0, passed=0;
    size_t cKs[]={8,64,256,1024,4096};
    for(int ki=0;ki<5;ki++){
        size_t K=cKs[ki], NN=11*K;
        double *ir=alloc64(NN),*ii=alloc64(NN);
        double *ref_r=alloc64(NN),*ref_i=alloc64(NN);
        double *out_r=alloc64(NN),*out_i=alloc64(NN);
        double *pk_ir=alloc64(NN),*pk_ii=alloc64(NN);
        double *pk_or=alloc64(NN),*pk_oi=alloc64(NN);

        srand(42+(unsigned)K);
        for(size_t i=0;i<NN;i++){ir[i]=(double)rand()/RAND_MAX-.5;ii[i]=(double)rand()/RAND_MAX-.5;}

        /* Reference */
        ref_tw_dft11(ir,ii,ref_r,ref_i,K,1);

        /* Pack+walk forward */
        r11_pack_avx512(ir,ii,pk_ir,pk_ii,K);
        radix11_walk_plan_t wp; radix11_walk_plan_init(&wp,K);
        radix11_tw_pack_walk_fwd_avx512(pk_ir,pk_ii,pk_or,pk_oi,&wp,K);
        r11_unpack_avx512(pk_or,pk_oi,out_r,out_i,K);
        double ef=maxerr(ref_r,ref_i,out_r,out_i,NN);

        /* Reference backward */
        ref_tw_dft11(ir,ii,ref_r,ref_i,K,0);
        r11_pack_avx512(ir,ii,pk_ir,pk_ii,K);
        radix11_walk_plan_init(&wp,K);
        radix11_tw_pack_walk_bwd_avx512(pk_ir,pk_ii,pk_or,pk_oi,&wp,K);
        r11_unpack_avx512(pk_or,pk_oi,out_r,out_i,K);
        double eb=maxerr(ref_r,ref_i,out_r,out_i,NN);

        int ok=(ef<1e-9 && eb<1e-9);
        total++; passed+=ok;
        printf("  K=%-5zu fwd=%.1e bwd=%.1e %s\n", K, ef, eb, ok?"PASS":"FAIL");

        free(ir);free(ii);free(ref_r);free(ref_i);free(out_r);free(out_i);
        free(pk_ir);free(pk_ii);free(pk_or);free(pk_oi);
    }
    printf("\n=== %d / %d passed ===\n\n", passed, total);

    /* ── Benchmark ── */
    printf("-- Benchmark: Packed-table vs Pack+Walk (min-of-5, fwd only) --\n");
    printf("  %-5s | %8s %8s | %7s | winner\n","K","pk-tbl","pk+walk","pw/pk");
    printf("  ------+-------------------+---------+-------\n");

    size_t bKs[]={64,256,512,1024,2048,4096};
    for(int ki=0;ki<6;ki++){
        size_t K=bKs[ki], NN=11*K;
        double *ir=alloc64(NN),*ii=alloc64(NN);
        double *pk_ir=alloc64(NN),*pk_ii=alloc64(NN);
        double *pk_or=alloc64(NN),*pk_oi=alloc64(NN);
        double *twr=alloc64(10*K),*twi=alloc64(10*K);
        double *pk_twr=alloc64(10*K),*pk_twi=alloc64(10*K);

        srand(42+(unsigned)K);
        for(size_t i=0;i<NN;i++){ir[i]=(double)rand()/RAND_MAX-.5;ii[i]=(double)rand()/RAND_MAX-.5;}

        r11_build_tw_table(K,twr,twi);
        r11_pack_avx512(ir,ii,pk_ir,pk_ii,K);
        r11_pack_tw(twr,twi,pk_twr,pk_twi,K);

        radix11_walk_plan_t wp;
        int iters=(int)(500000.0/K); if(iters<100)iters=100;
        int warmup=iters/10;
        double best_t=1e18, best_w=1e18;

        for(int run=0;run<5;run++){
            /* Packed table */
            for(int i=0;i<warmup;i++)
                r11_tw_packed_table_fwd(pk_ir,pk_ii,pk_or,pk_oi,pk_twr,pk_twi,K);
            double t0=now_ns();
            for(int i=0;i<iters;i++)
                r11_tw_packed_table_fwd(pk_ir,pk_ii,pk_or,pk_oi,pk_twr,pk_twi,K);
            double ns=(now_ns()-t0)/iters;
            if(ns<best_t)best_t=ns;

            /* Pack+walk */
            radix11_walk_plan_init(&wp,K);
            for(int i=0;i<warmup;i++){
                radix11_walk_plan_init(&wp,K);
                radix11_tw_pack_walk_fwd_avx512(pk_ir,pk_ii,pk_or,pk_oi,&wp,K);
            }
            radix11_walk_plan_init(&wp,K);
            t0=now_ns();
            for(int i=0;i<iters;i++){
                radix11_walk_plan_init(&wp,K);
                radix11_tw_pack_walk_fwd_avx512(pk_ir,pk_ii,pk_or,pk_oi,&wp,K);
            }
            ns=(now_ns()-t0)/iters;
            if(ns<best_w)best_w=ns;
        }

        double r=best_t/best_w;
        printf("  K=%-4zu | %7.0f  %7.0f | %6.2fx | %s\n",
               K, best_t, best_w, r, r>1.03?"PK+WALK":(r<0.97?"PK-TBL":"~tie"));

        free(ir);free(ii);free(pk_ir);free(pk_ii);free(pk_or);free(pk_oi);
        free(twr);free(twi);free(pk_twr);free(pk_twi);
    }

    return (passed==total)?0:1;
}
