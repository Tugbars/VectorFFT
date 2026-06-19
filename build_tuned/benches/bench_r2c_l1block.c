/* bench_r2c_l1block.c — THE LEVER: L1-block the K-batch in decoupled r2c.
 *
 * Premise (from the fresh breakdown): our decoupled r2c makes 3 separate
 * full-plane passes — pack / c2c-FFT / Hermitian-recombine — each streaming
 * the ~256KB N/2-by-K plane through L2. The recombine is bandwidth-bound
 * (AVX2 == scalar). MKL is faster because it keeps each transform L1-resident
 * across its whole pipeline. Vectorizing or reordering our recombine can't
 * win — only ELIMINATING the L2 round-trips can.
 *
 * Fix: process the K-batch in blocks of B lanes. For each block run the FULL
 * pipeline (pack -> FFT -> recombine). The block's working set is
 *   128 rows * B lanes * 8 bytes * 2 planes  (= 16KB at B=8)
 * which fits L1, so the intermediate z stays hot across all 3 steps instead
 * of round-tripping L2. Total L2/mem traffic drops from ~3-4x plane to ~1x.
 *
 * This is NOT the buffered-tiling that was ruled out (that was native rfft,
 * whose legs span all N rows -> L2-sized tile). The decoupled inner is only
 * N/2=128 -> one transform's column-block is L1-sized. The executor already
 * supports lane sub-blocking: execute_fwd(plan, re+l0, im+l0, B) processes
 * lanes [l0,l0+B) with the full plan->K row stride (executor_generic.h).
 *
 * Sweeps B in {4,8,16,32}. Correctness-gated per cell vs reference DFT.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_r2c_l1block.c --mkl --compile
 * Run  : PATH += MKL bin + C:\mingw152\mingw64\bin, then run the .exe.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>

#include "executor.h"
#include "env.h"
#include "planner.h"
#include "dp_planner.h"            /* vfft_proto_now_ns */
#include "proto_stride_compat.h"   /* posix_memalign / aligned_free, before r2c.h */
#include "r2c.h"                    /* _r2c_init_twiddles */
#include "generator/generated/registry.h"

#define PIN_CORE 2
#define BEST_OF  15

static double *alloc_d(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}
static int reps_for(size_t total) {
    int r = (int)(4e6 / (total + 1));
    if (r < 30) r = 30; if (r > 200000) r = 200000;
    return r;
}
static void compute_perm(const int *factors, int nf, int N, int *perm) {
    for (int n = 0; n < N; n++) {
        int idx = n, rev = 0, radix_product = 1;
        for (int s = 0; s < nf; s++) {
            int R = factors[s]; int digit = idx % R; idx /= R;
            rev += digit * (N / (radix_product * R)); radix_product *= R;
        }
        perm[n] = rev;
    }
}

/* ── UNBLOCKED reference: 3 full-plane passes (the current 0.81x path) ── */
static void unblocked_run(stride_plan_t *inner, const double *x,
                          double *zre, double *zim,
                          double *out_re, double *out_im,
                          const int *perm, const double *tw_re, const double *tw_im,
                          int halfN, size_t K) {
    for (int j = 0; j < halfN; j++) {
        const double *xe = x + (size_t)(2*j)*K, *xo = x + (size_t)(2*j+1)*K;
        double *zr = zre + (size_t)j*K, *zi = zim + (size_t)j*K;
        for (size_t l=0;l<K;l++){zr[l]=xe[l];zi[l]=xo[l];}
    }
    vfft_proto_execute_fwd(inner, zre, zim, K);
    {
        const double *Z0r = zre + (size_t)perm[0]*K, *Z0i = zim + (size_t)perm[0]*K;
        double *o0r = out_re, *o0i = out_im;
        double *onr = out_re + (size_t)halfN*K, *oni = out_im + (size_t)halfN*K;
        for (size_t l=0;l<K;l++){o0r[l]=Z0r[l]+Z0i[l];o0i[l]=0;onr[l]=Z0r[l]-Z0i[l];oni[l]=0;}
    }
    for (int k = 1; k < halfN; k++) {
        int mk = halfN - k;
        const double *Zk_r = zre + (size_t)perm[k]*K, *Zk_i = zim + (size_t)perm[k]*K;
        const double *Zm_r = zre + (size_t)perm[mk]*K, *Zm_i = zim + (size_t)perm[mk]*K;
        double *or_ = out_re + (size_t)k*K, *oi_ = out_im + (size_t)k*K;
        double c = tw_re[k], s = -tw_im[k];
        for (size_t l=0;l<K;l++){
            double zr=Zk_r[l],zi=Zk_i[l],mr=Zm_r[l],mi=Zm_i[l];
            double Er=0.5*(zr+mr),Ei=0.5*(zi-mi),Or=0.5*(zr-mr),Oi=0.5*(zi+mi);
            or_[l]=Er+(c*Oi-s*Or); oi_[l]=Ei+(-c*Or-s*Oi);
        }
    }
}

/* ── BLOCKED: per-K-block full pipeline (pack -> FFT -> recombine), L1-hot ── */
static void blocked_run(stride_plan_t *inner, const double *x,
                        double *zre, double *zim,
                        double *out_re, double *out_im,
                        const int *perm, const double *tw_re, const double *tw_im,
                        int halfN, size_t K, size_t B) {
    for (size_t l0 = 0; l0 < K; l0 += B) {
        size_t b = K - l0; if (b > B) b = B;

        /* (1) pack block: z[j] = x[2j] + i x[2j+1], lanes [l0,l0+b) */
        for (int j = 0; j < halfN; j++) {
            const double *xe = x + (size_t)(2*j)*K + l0;
            const double *xo = x + (size_t)(2*j+1)*K + l0;
            double *zr = zre + (size_t)j*K + l0;
            double *zi = zim + (size_t)j*K + l0;
            for (size_t l=0;l<b;l++){zr[l]=xe[l];zi[l]=xo[l];}
        }

        /* (2) FFT block — lanes [l0,l0+b), full plan->K row stride */
        vfft_proto_execute_fwd(inner, zre + l0, zim + l0, b);

        /* (3) Hermitian recombine block (perm-driven; rows now L1-hot) */
        {
            const double *Z0r = zre + (size_t)perm[0]*K + l0, *Z0i = zim + (size_t)perm[0]*K + l0;
            double *o0r = out_re + l0, *o0i = out_im + l0;
            double *onr = out_re + (size_t)halfN*K + l0, *oni = out_im + (size_t)halfN*K + l0;
            for (size_t l=0;l<b;l++){o0r[l]=Z0r[l]+Z0i[l];o0i[l]=0;onr[l]=Z0r[l]-Z0i[l];oni[l]=0;}
        }
        for (int k = 1; k < halfN; k++) {
            int mk = halfN - k;
            const double *Zk_r = zre + (size_t)perm[k]*K + l0, *Zk_i = zim + (size_t)perm[k]*K + l0;
            const double *Zm_r = zre + (size_t)perm[mk]*K + l0, *Zm_i = zim + (size_t)perm[mk]*K + l0;
            double *or_ = out_re + (size_t)k*K + l0, *oi_ = out_im + (size_t)k*K + l0;
            double c = tw_re[k], s = -tw_im[k];
            for (size_t l=0;l<b;l++){
                double zr=Zk_r[l],zi=Zk_i[l],mr=Zm_r[l],mi=Zm_i[l];
                double Er=0.5*(zr+mr),Ei=0.5*(zi-mi),Or=0.5*(zr-mr),Oi=0.5*(zi+mi);
                or_[l]=Er+(c*Oi-s*Or); oi_[l]=Ei+(-c*Or-s*Oi);
            }
        }
    }
}

static double bench_unblocked(stride_plan_t *inner, const double *x,
                              double *zre, double *zim, double *oor, double *ooi,
                              const int *perm, const double *tw_re, const double *tw_im,
                              int halfN, size_t K, size_t total) {
    for (int w=0;w<10;w++) unblocked_run(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K);
    int reps = reps_for(total); double best = 1e18;
    for (int t=0;t<BEST_OF;t++){
        double t0=vfft_proto_now_ns();
        for (int i=0;i<reps;i++) unblocked_run(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;
    }
    return best;
}
static double bench_blocked(stride_plan_t *inner, const double *x,
                            double *zre, double *zim, double *oor, double *ooi,
                            const int *perm, const double *tw_re, const double *tw_im,
                            int halfN, size_t K, size_t B, size_t total) {
    for (int w=0;w<10;w++) blocked_run(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K,B);
    int reps = reps_for(total); double best = 1e18;
    for (int t=0;t<BEST_OF;t++){
        double t0=vfft_proto_now_ns();
        for (int i=0;i<reps;i++) blocked_run(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K,B);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;
    }
    return best;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE h, const double *xin, double *cce, size_t total) {
    for (int w=0;w<10;w++) DftiComputeForward(h,(void*)xin,cce);
    int reps = reps_for(total); double best = 1e18;
    for (int t=0;t<BEST_OF;t++){
        double t0=vfft_proto_now_ns();
        for (int i=0;i<reps;i++) DftiComputeForward(h,(void*)xin,cce);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;
    }
    return best;
}

/* correctness: lane 0 (and one mid lane) vs reference DFT, k=0..N/2 */
static double gate(const double *oor, const double *ooi, const double *x,
                   int N, int halfN, size_t K, size_t lane) {
    double maxerr = 0.0;
    for (int k=0;k<=halfN;k++){
        double rr=0,ri=0;
        for (int n=0;n<N;n++){
            double xn=x[(size_t)n*K+lane];
            double ang=-2.0*M_PI*(double)k*(double)n/(double)N;
            rr+=xn*cos(ang); ri+=xn*sin(ang);
        }
        double er=fabs(oor[(size_t)k*K+lane]-rr), ei=fabs(ooi[(size_t)k*K+lane]-ri);
        if(er>maxerr)maxerr=er; if(ei>maxerr)maxerr=ei;
    }
    return maxerr;
}

int main(void) {
    stride_env_init();
    if (stride_pin_thread(PIN_CORE) != 0) fprintf(stderr,"warn: pin cpu%d failed\n",PIN_CORE);
    mkl_set_num_threads(1);

    const int N = 256, halfN = N/2;
    const size_t Ks[] = {32, 64, 128, 256};
    const int nK = (int)(sizeof Ks/sizeof Ks[0]);
    const size_t Bs[] = {4, 8, 16, 32};
    const int nB = (int)(sizeof Bs/sizeof Bs[0]);

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t wis; int have_wis =
        (vfft_proto_wisdom_load(&wis,"../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    if(!have_wis) have_wis=(vfft_proto_wisdom_load(&wis,"../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    printf("# c2c wisdom load: %s\n", have_wis?"OK":"FAILED");

    double *tw_re=alloc_d(halfN), *tw_im=alloc_d(halfN);
    _r2c_init_twiddles(N, tw_re, tw_im);

    printf("=== r2c L1-BLOCKED K-batch (pack+FFT+recombine per block, L1-hot) vs UNBLOCKED vs MKL\n");
    printf("    (N=256, ST, cpu%d, best-of-%d) ===\n", PIN_CORE, BEST_OF);
    printf("%-5s %12s", "K", "unblk_ns");
    for (int bi=0;bi<nB;bi++) printf("   blkB=%-3zu", Bs[bi]);
    printf(" %11s %10s %10s\n", "mkl_ns", "best_blk", "bestblk/mkl");
    printf("------+------------+----------+----------+----------+----------+------------+----------+----------\n");

    for (int ki=0; ki<nK; ki++) {
        size_t K = Ks[ki]; size_t total=(size_t)N*K;
        double *x = alloc_d(total);
        srand(7+(int)K);
        for (size_t i=0;i<total;i++) x[i]=(double)rand()/RAND_MAX*2-1;

        stride_plan_t *inner = vfft_proto_auto_plan(halfN, K, &reg, have_wis?&wis:NULL);
        if(!inner){printf("%-5zu auto_plan NULL\n",K); vfft_proto_aligned_free(x); continue;}
        int perm[256]; compute_perm(inner->factors, inner->num_stages, halfN, perm);

        double *zre=alloc_d((size_t)halfN*K), *zim=alloc_d((size_t)halfN*K);
        double *oor=alloc_d((size_t)(halfN+1)*K), *ooi=alloc_d((size_t)(halfN+1)*K);

        char fs[64]; size_t pp=0;
        for(int s=0;s<inner->num_stages;s++) pp+=(size_t)snprintf(fs+pp,sizeof fs-pp,"%s%d",s?",":"",inner->factors[s]);
        printf("# K=%-4zu inner=(%s)  blockKB(B=8)=%.0f KB\n", K, fs, (double)halfN*8*8*2/1024.0);

        /* correctness gate: unblocked + each B, lane 0 and lane K-1 */
        memset(oor,0,(size_t)(halfN+1)*K*sizeof(double)); memset(ooi,0,(size_t)(halfN+1)*K*sizeof(double));
        unblocked_run(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K);
        double eub = gate(oor,ooi,x,N,halfN,K,0);
        double worst_blk_err = 0.0;
        for (int bi=0;bi<nB;bi++){
            memset(oor,0,(size_t)(halfN+1)*K*sizeof(double)); memset(ooi,0,(size_t)(halfN+1)*K*sizeof(double));
            blocked_run(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K,Bs[bi]);
            double e0=gate(oor,ooi,x,N,halfN,K,0);
            double e1=gate(oor,ooi,x,N,halfN,K,K-1);
            if(e0>worst_blk_err)worst_blk_err=e0; if(e1>worst_blk_err)worst_blk_err=e1;
        }
        if (eub>=1e-9 || worst_blk_err>=1e-9) {
            printf("%-5zu *** CORRECTNESS FAIL unblk=%.2e blk=%.2e (ABORT) ***\n",K,eub,worst_blk_err);
            vfft_proto_aligned_free(x);vfft_proto_aligned_free(zre);vfft_proto_aligned_free(zim);
            vfft_proto_aligned_free(oor);vfft_proto_aligned_free(ooi); stride_plan_destroy(inner); continue;
        }

        /* MKL r2c transform-major */
        DFTI_DESCRIPTOR_HANDLE h=0; int mkl_ok=0;
        double *xin=alloc_d(total), *cce=alloc_d((size_t)(halfN+1)*K*2);
        for (size_t t=0;t<K;t++) for(int n=0;n<N;n++) xin[t*N+n]=x[(size_t)n*K+t];
        DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
        DftiSetValue(h,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
        DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
        DftiSetValue(h,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
        DftiSetValue(h,DFTI_OUTPUT_DISTANCE,(MKL_LONG)(halfN+1));
        mkl_ok=(DftiCommitDescriptor(h)==DFTI_NO_ERROR);

        double ub_ns = bench_unblocked(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K,total);
        double blk_ns[8]; double best_blk=1e18;
        for (int bi=0;bi<nB;bi++){
            blk_ns[bi]=bench_blocked(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K,Bs[bi],total);
            if(blk_ns[bi]<best_blk)best_blk=blk_ns[bi];
        }
        double m_ns = mkl_ok?bench_mkl(h,xin,cce,total):0;

        printf("%-5zu %12.1f", K, ub_ns);
        for (int bi=0;bi<nB;bi++) printf(" %10.1f", blk_ns[bi]);
        printf(" %11.1f %10.1f %9.3fx\n", m_ns, best_blk, m_ns>0?m_ns/best_blk:0);

        if(h)DftiFreeDescriptor(&h);
        vfft_proto_aligned_free(x);vfft_proto_aligned_free(zre);vfft_proto_aligned_free(zim);
        vfft_proto_aligned_free(oor);vfft_proto_aligned_free(ooi);
        vfft_proto_aligned_free(xin);vfft_proto_aligned_free(cce);
        stride_plan_destroy(inner);
    }

    vfft_proto_aligned_free(tw_re); vfft_proto_aligned_free(tw_im);
    if(have_wis) vfft_proto_wisdom_free(&wis);
    printf("\n# bestblk/mkl = MKL_ns / best_blocked_ns  (>1 = WE BEAT MKL).\n");
    printf("# block working set @B=8 = 128 rows * 8 lanes * 8B * 2 planes = 16 KB (fits L1).\n");
    return 0;
}
