/**
 * bench_sv.c — N=32768: sv codelets with executor-controlled transpose
 *
 * Three executors compared:
 *   OLD:  DIT + perm, t2 codelets, stage-by-stage
 *   BUF:  DIT + perm, t2 codelets, gather/scatter buffered outer stages
 *   SV:   DIT + perm, sv codelets for outer stages, sub-problem completion inner
 *   FFTW: FFTW_MEASURE reference
 *
 * N = 32768 = 16 × 16 × 8 × 16 (inner→outer)
 * Stage 0: R=16 K=1    (K=1 SIMD)
 * Stage 1: R=16 K=16   (t2, L1-resident)
 * Stage 2: R=8  K=256  (sv + transpose, 60KB > 48KB L1)
 * Stage 3: R=16 K=2048 (sv + transpose, 480KB >> L1)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#include "fft_n1_k1.h"
#include "fft_n1_k1_simd.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_notw.h"
#include "fft_radix16_avx2_dit_tw.h"
#include "fft_radix16_avx2_dif_tw.h"

static inline double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

#define N_FFT  32768
#define NSTAGES 4
#define K_BLOCK 16
#define VL 4

typedef void (*notw_fn)(const double*,const double*,double*,double*,size_t);
typedef void (*tw_fn)(const double*,const double*,double*,double*,
                      const double*,const double*,size_t);
typedef void (*sv_fn)(const double*,const double*,double*,double*,
                      const double*,const double*,size_t);

/* K=1 SIMD wrapper */
static void r16_k1_wrap(const double *ir, const double *ii,
                         double *or_, double *oi, size_t K) {
    (void)K; dft16_k1_fwd_avx2(ir, ii, or_, oi);
}

static void init_tw(double *twr, double *twi, size_t R, size_t K) {
    double RK = (double)(R * K);
    for (size_t n = 1; n < R; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * (double)(n * k) / RK;
            twr[(n-1)*K+k] = cos(a); twi[(n-1)*K+k] = sin(a);
        }
}

static size_t *build_perm(const size_t *rx, size_t ns, size_t N) {
    size_t *p = (size_t*)malloc(N * sizeof(size_t));
    for (size_t i = 0; i < N; i++) {
        size_t rem=i, rev=0, str=N;
        for (size_t s=0;s<ns;s++){str/=rx[s];rev+=(rem%rx[s])*str;rem/=rx[s];}
        p[i]=rev;
    }
    return p;
}

/* Pre-transpose twiddles into sv layout at plan time.
 * Original: tw[(n-1)*K + k]
 * SV:       tw_sv[block*(R-1)*K_BLOCK + (n-1)*K_BLOCK + k_local] */
static double *transpose_tw(const double *tw, size_t R, size_t K) {
    size_t n_blocks = K / K_BLOCK;
    double *sv = (double*)aligned_alloc(32, (R-1) * K * sizeof(double));
    for (size_t b = 0; b < n_blocks; b++)
        for (size_t n = 0; n < R-1; n++)
            memcpy(sv + b*(R-1)*K_BLOCK + n*K_BLOCK,
                   tw + n*K + b*K_BLOCK,
                   K_BLOCK * sizeof(double));
    return sv;
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTOR 1: OLD (DIT + perm, t2 codelets, stage-by-stage)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct { size_t R, K; notw_fn notw; tw_fn dit; double *twr, *twi; } stage_t;

static void exec_old(const double *ir, const double *ii,
                     double *or_, double *oi,
                     stage_t *st, const size_t *perm,
                     double *a, double *ai, double *b, double *bi)
{
    for (size_t i=0;i<N_FFT;i++){a[i]=ir[perm[i]];ai[i]=ii[perm[i]];}
    double *s=a,*si_=ai,*d=b,*di=bi;
    for (int i=0;i<NSTAGES;i++){
        size_t R=st[i].R,K=st[i].K,ng=N_FFT/(R*K);
        for(size_t g=0;g<ng;g++){size_t o=g*R*K;
            if(K>1&&st[i].dit&&st[i].twr)
                st[i].dit(s+o,si_+o,d+o,di+o,st[i].twr,st[i].twi,K);
            else st[i].notw(s+o,si_+o,d+o,di+o,K);}
        double*t;t=s;s=d;d=t;t=si_;si_=di;di=t;
    }
    memcpy(or_,s,N_FFT*8);memcpy(oi,si_,N_FFT*8);
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTOR 2: BUF (DIT + perm, t2 gather/scatter for outer stages)
 * ═══════════════════════════════════════════════════════════════ */

static double __attribute__((aligned(32))) g_sr[16*K_BLOCK], g_si[16*K_BLOCK];
static double __attribute__((aligned(32))) g_dr[16*K_BLOCK], g_di_[16*K_BLOCK];
static double __attribute__((aligned(32))) g_tr[15*K_BLOCK], g_ti[15*K_BLOCK];

static void exec_buf(const double *ir, const double *ii,
                     double *or_, double *oi,
                     stage_t *st, const size_t *perm,
                     double *a, double *ai, double *b, double *bi)
{
    for (size_t i=0;i<N_FFT;i++){a[i]=ir[perm[i]];ai[i]=ii[perm[i]];}
    double *s=a,*si_=ai,*d=b,*di=bi;

    /* Phase 1: inner stages 0,1 — sub-problem completion */
    size_t inner=256, nsubs=N_FFT/inner;
    for (size_t p=0;p<nsubs;p++){
        size_t sub=p*inner;
        for(size_t g=0;g<inner/16;g++){size_t o=sub+g*16;
            st[0].notw(s+o,si_+o,d+o,di+o,1);}
        st[1].dit(d+sub,di+sub,s+sub,si_+sub,st[1].twr,st[1].twi,16);
    }

    /* Phase 2: stage 2 R=8 K=256 — buffered t2 */
    {size_t R=8,K=256,ng=N_FFT/(R*K);
    for(size_t g=0;g<ng;g++){size_t go=g*R*K;
        for(size_t kb=0;kb<K;kb+=K_BLOCK){
            for(size_t n=0;n<R;n++){memcpy(g_sr+n*K_BLOCK,s+go+n*K+kb,K_BLOCK*8);
                                    memcpy(g_si+n*K_BLOCK,si_+go+n*K+kb,K_BLOCK*8);}
            for(size_t n=0;n<R-1;n++){memcpy(g_tr+n*K_BLOCK,st[2].twr+n*K+kb,K_BLOCK*8);
                                      memcpy(g_ti+n*K_BLOCK,st[2].twi+n*K+kb,K_BLOCK*8);}
            st[2].dit(g_sr,g_si,g_dr,g_di_,g_tr,g_ti,K_BLOCK);
            for(size_t n=0;n<R;n++){memcpy(d+go+n*K+kb,g_dr+n*K_BLOCK,K_BLOCK*8);
                                    memcpy(di+go+n*K+kb,g_di_+n*K_BLOCK,K_BLOCK*8);}
        }
    }}
    {double*t;t=s;s=d;d=t;t=si_;si_=di;di=t;}

    /* Phase 3: stage 3 R=16 K=2048 — unbuffered (1 group) */
    st[3].dit(s,si_,or_,oi,st[3].twr,st[3].twi,2048);
}

/* ═══════════════════════════════════════════════════════════════
 * EXECUTOR 3: SV (DIT + perm, sv codelets for outer stages)
 *
 * Outer stages: transpose K_BLOCK k-values into scratch at stride K_BLOCK,
 * call t1sv codelet K_BLOCK/VL times, transpose back.
 * Twiddles pre-transposed at plan time.
 * Inner stages: sub-problem completion with t2 codelets.
 * ═══════════════════════════════════════════════════════════════ */

/* Scratch for sv transpose — data */
static double __attribute__((aligned(32))) sv_in_re[64*K_BLOCK];
static double __attribute__((aligned(32))) sv_in_im[64*K_BLOCK];
static double __attribute__((aligned(32))) sv_out_re[64*K_BLOCK];
static double __attribute__((aligned(32))) sv_out_im[64*K_BLOCK];

typedef struct {
    size_t R, K, n_groups, n_blocks;
    sv_fn codelet;
    double *tw_sv_re, *tw_sv_im;  /* pre-transposed twiddles */
} sv_stage_t;

static void exec_sv(const double *ir, const double *ii,
                    double *or_, double *oi,
                    stage_t *st, sv_stage_t *sv_st,
                    const size_t *perm,
                    double *a, double *ai, double *b, double *bi)
{
    for (size_t i=0;i<N_FFT;i++){a[i]=ir[perm[i]];ai[i]=ii[perm[i]];}
    double *s=a,*si_=ai,*d=b,*di=bi;

    /* Phase 1: inner stages 0,1 — sub-problem completion */
    size_t inner=256, nsubs=N_FFT/inner;
    for (size_t p=0;p<nsubs;p++){
        size_t sub=p*inner;
        for(size_t g=0;g<inner/16;g++){size_t o=sub+g*16;
            st[0].notw(s+o,si_+o,d+o,di+o,1);}
        st[1].dit(d+sub,di+sub,s+sub,si_+sub,st[1].twr,st[1].twi,16);
    }

    /* Phase 2: stage 2 R=8 K=256 — sv codelet with transpose */
    {
        sv_stage_t *sv = &sv_st[0];
        size_t R=sv->R, K=sv->K;
        for (size_t g=0; g<sv->n_groups; g++) {
            size_t goff = g * R * K;
            for (size_t blk=0; blk<sv->n_blocks; blk++) {
                size_t kbase = blk * K_BLOCK;
                size_t tw_off = blk * (R-1) * K_BLOCK;
                /* Transpose data in */
                for (size_t n=0; n<R; n++) {
                    memcpy(sv_in_re + n*K_BLOCK, s + goff + n*K + kbase, K_BLOCK*8);
                    memcpy(sv_in_im + n*K_BLOCK, si_ + goff + n*K + kbase, K_BLOCK*8);
                }
                /* sv codelet calls: K_BLOCK/VL iterations */
                for (size_t j=0; j<K_BLOCK; j+=VL) {
                    sv->codelet(
                        sv_in_re + j, sv_in_im + j,
                        sv_out_re + j, sv_out_im + j,
                        sv->tw_sv_re + tw_off + j,
                        sv->tw_sv_im + tw_off + j,
                        K_BLOCK);
                }
                /* Transpose data out */
                for (size_t n=0; n<R; n++) {
                    memcpy(d + goff + n*K + kbase, sv_out_re + n*K_BLOCK, K_BLOCK*8);
                    memcpy(di + goff + n*K + kbase, sv_out_im + n*K_BLOCK, K_BLOCK*8);
                }
            }
        }
    }
    {double*t;t=s;s=d;d=t;t=si_;si_=di;di=t;}

    /* Phase 3: stage 3 R=16 K=2048 — sv codelet with transpose */
    {
        sv_stage_t *sv = &sv_st[1];
        size_t R=sv->R, K=sv->K;
        /* 1 group for K=2048 */
        for (size_t blk=0; blk<sv->n_blocks; blk++) {
            size_t kbase = blk * K_BLOCK;
            size_t tw_off = blk * (R-1) * K_BLOCK;
            for (size_t n=0; n<R; n++) {
                memcpy(sv_in_re + n*K_BLOCK, s + n*K + kbase, K_BLOCK*8);
                memcpy(sv_in_im + n*K_BLOCK, si_ + n*K + kbase, K_BLOCK*8);
            }
            for (size_t j=0; j<K_BLOCK; j+=VL) {
                sv->codelet(
                    sv_in_re + j, sv_in_im + j,
                    sv_out_re + j, sv_out_im + j,
                    sv->tw_sv_re + tw_off + j,
                    sv->tw_sv_im + tw_off + j,
                    K_BLOCK);
            }
            for (size_t n=0; n<R; n++) {
                memcpy(or_ + n*K + kbase, sv_out_re + n*K_BLOCK, K_BLOCK*8);
                memcpy(oi + n*K + kbase, sv_out_im + n*K_BLOCK, K_BLOCK*8);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════ */

typedef void (*exec_fn)(const double*,const double*,double*,double*);

static stage_t g_st[NSTAGES];
static sv_stage_t g_sv[2];
static const size_t *g_perm;
static double *g_a,*g_ai,*g_b,*g_bi;

static void wrap_old(const double*ir,const double*ii,double*or_,double*oi){
    exec_old(ir,ii,or_,oi,g_st,g_perm,g_a,g_ai,g_b,g_bi);}
static void wrap_buf(const double*ir,const double*ii,double*or_,double*oi){
    exec_buf(ir,ii,or_,oi,g_st,g_perm,g_a,g_ai,g_b,g_bi);}
static void wrap_sv(const double*ir,const double*ii,double*or_,double*oi){
    exec_sv(ir,ii,or_,oi,g_st,g_sv,g_perm,g_a,g_ai,g_b,g_bi);}

static double bench(exec_fn fn,const double*ir,const double*ii,
                    double*or_,double*oi,int reps){
    for(int i=0;i<20;i++)fn(ir,ii,or_,oi);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();
        for(int i=0;i<reps;i++)fn(ir,ii,or_,oi);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    return best;
}

static double bench_fftw(int reps){
    double*ri=fftw_malloc(N_FFT*8),*ii=fftw_malloc(N_FFT*8);
    double*ro=fftw_malloc(N_FFT*8),*io=fftw_malloc(N_FFT*8);
    for(size_t i=0;i<N_FFT;i++){ri[i]=(double)rand()/RAND_MAX;ii[i]=(double)rand()/RAND_MAX;}
    fftw_iodim dim={.n=N_FFT,.is=1,.os=1};
    fftw_iodim howm={.n=1,.is=N_FFT,.os=N_FFT};
    fftw_plan p=fftw_plan_guru_split_dft(1,&dim,1,&howm,ri,ii,ro,io,FFTW_MEASURE);
    for(int i=0;i<20;i++)fftw_execute(p);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();
        for(int i=0;i<reps;i++)fftw_execute_split_dft(p,ri,ii,ro,io);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    fftw_destroy_plan(p);fftw_free(ri);fftw_free(ii);fftw_free(ro);fftw_free(io);
    return best;
}

int main(void){
    srand(42);
    printf("================================================================\n");
    printf("  N=%d: SV codelets vs buffered t2 vs old t2 vs FFTW\n",N_FFT);
    printf("  16 x 16 x 8 x 16 | split=2 | K_BLOCK=%d | VL=%d\n",K_BLOCK,VL);
    printf("================================================================\n\n");

    /* Build stages */
    memset(g_st,0,sizeof(g_st));
    g_st[0]=(stage_t){16,1,r16_k1_wrap,NULL,NULL,NULL};
    g_st[1]=(stage_t){16,16,(notw_fn)radix16_n1_dit_kernel_fwd_avx2,
        (tw_fn)radix16_tw_flat_dit_kernel_fwd_avx2,
        aligned_alloc(32,15*16*8),aligned_alloc(32,15*16*8)};
    init_tw(g_st[1].twr,g_st[1].twi,16,16);
    g_st[2]=(stage_t){8,256,(notw_fn)radix8_notw_dit_kernel_fwd_avx2,
        (tw_fn)radix8_tw_dit_kernel_fwd_avx2,
        aligned_alloc(32,7*256*8),aligned_alloc(32,7*256*8)};
    init_tw(g_st[2].twr,g_st[2].twi,8,256);
    g_st[3]=(stage_t){16,2048,(notw_fn)radix16_n1_dit_kernel_fwd_avx2,
        (tw_fn)radix16_tw_flat_dit_kernel_fwd_avx2,
        aligned_alloc(32,15*2048*8),aligned_alloc(32,15*2048*8)};
    init_tw(g_st[3].twr,g_st[3].twi,16,2048);

    /* Build sv stages (pre-transposed twiddles) */
    g_sv[0]=(sv_stage_t){8,256, N_FFT/(8*256), 256/K_BLOCK,
        (sv_fn)radix8_t1sv_dit_kernel_fwd_avx2,
        transpose_tw(g_st[2].twr,8,256),
        transpose_tw(g_st[2].twi,8,256)};
    g_sv[1]=(sv_stage_t){16,2048, 1, 2048/K_BLOCK,
        (sv_fn)radix16_t1sv_dit_kernel_fwd_avx2,
        transpose_tw(g_st[3].twr,16,2048),
        transpose_tw(g_st[3].twi,16,2048)};

    size_t rx[]={16,16,8,16};
    g_perm=build_perm(rx,4,N_FFT);

    double *in_re=aligned_alloc(32,N_FFT*8),*in_im=aligned_alloc(32,N_FFT*8);
    double *out_re=aligned_alloc(32,N_FFT*8),*out_im=aligned_alloc(32,N_FFT*8);
    g_a=aligned_alloc(32,N_FFT*8); g_ai=aligned_alloc(32,N_FFT*8);
    g_b=aligned_alloc(32,N_FFT*8); g_bi=aligned_alloc(32,N_FFT*8);
    for(size_t i=0;i<N_FFT;i++){in_re[i]=(double)rand()/RAND_MAX-.5;
                                  in_im[i]=(double)rand()/RAND_MAX-.5;}

    int reps=200;

    /* Correctness first */
    double *fftw_re=fftw_malloc(N_FFT*8),*fftw_im=fftw_malloc(N_FFT*8);
    double *fftw_or=fftw_malloc(N_FFT*8),*fftw_oi=fftw_malloc(N_FFT*8);
    memcpy(fftw_re,in_re,N_FFT*8);memcpy(fftw_im,in_im,N_FFT*8);
    fftw_iodim dim={.n=N_FFT,.is=1,.os=1};
    fftw_iodim howm={.n=1,.is=N_FFT,.os=N_FFT};
    fftw_plan fp=fftw_plan_guru_split_dft(1,&dim,1,&howm,fftw_re,fftw_im,fftw_or,fftw_oi,FFTW_ESTIMATE);
    fftw_execute(fp);

    wrap_old(in_re,in_im,out_re,out_im);
    double e_old=0;for(size_t i=0;i<N_FFT;i++){double d=fabs(out_re[i]-fftw_or[i])+fabs(out_im[i]-fftw_oi[i]);if(d>e_old)e_old=d;}

    wrap_buf(in_re,in_im,out_re,out_im);
    double e_buf=0;for(size_t i=0;i<N_FFT;i++){double d=fabs(out_re[i]-fftw_or[i])+fabs(out_im[i]-fftw_oi[i]);if(d>e_buf)e_buf=d;}

    wrap_sv(in_re,in_im,out_re,out_im);
    double e_sv=0;for(size_t i=0;i<N_FFT;i++){double d=fabs(out_re[i]-fftw_or[i])+fabs(out_im[i]-fftw_oi[i]);if(d>e_sv)e_sv=d;}

    printf("Correctness vs FFTW:\n");
    printf("  Old: %.2e %s\n",e_old,e_old<1e-10?"OK":"FAIL");
    printf("  Buf: %.2e %s\n",e_buf,e_buf<1e-10?"OK":"FAIL");
    printf("  SV:  %.2e %s\n",e_sv, e_sv<1e-10?"OK":"FAIL");
    printf("\n");

    if(e_old>1e-10||e_buf>1e-10||e_sv>1e-10){
        printf("CORRECTNESS FAILURE — skipping perf.\n");
        return 1;
    }

    /* Performance */
    printf("Benchmarking (%d reps, 7 trials, best-of)...\n\n",reps);
    double fftw_ns=bench_fftw(reps);
    double old_ns=bench(wrap_old,in_re,in_im,out_re,out_im,reps);
    double buf_ns=bench(wrap_buf,in_re,in_im,out_re,out_im,reps);
    double sv_ns =bench(wrap_sv, in_re,in_im,out_re,out_im,reps);

    printf("  %-20s %8.0f ns\n","FFTW_MEASURE:",fftw_ns);
    printf("  %-20s %8.0f ns  (%.2fx vs FFTW)\n","Old (t2 stage-by-stage):",old_ns,fftw_ns/old_ns);
    printf("  %-20s %8.0f ns  (%.2fx vs FFTW)\n","Buf (t2 gather/scatter):",buf_ns,fftw_ns/buf_ns);
    printf("  %-20s %8.0f ns  (%.2fx vs FFTW)\n","SV  (sv + transpose):",sv_ns,fftw_ns/sv_ns);
    printf("\n");
    printf("  SV vs Old: %.2fx\n",old_ns/sv_ns);
    printf("  SV vs Buf: %.2fx\n",buf_ns/sv_ns);

    fftw_destroy_plan(fp);
    fftw_free(fftw_re);fftw_free(fftw_im);fftw_free(fftw_or);fftw_free(fftw_oi);
    return 0;
}
