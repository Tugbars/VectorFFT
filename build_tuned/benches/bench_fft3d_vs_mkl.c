/* bench_fft3d_vs_mkl.c — 3D c2c (core/fft3d.h) vs MKL DFTI 3D.
 *
 * Inner plans per axis via the DP PLANNER (dp_planner.h + measure.h):
 *   plan_axis0 = DP(N1, K = N2*N3)   plan_axis1 = DP(N2, K = N3)
 *   plan_row   = DP(N3, K = B)
 * Each DP decision (factor chain, variants, DIT/DIF) is printed. Non-smooth
 * axes fall back to auto_plan_dispatch (Rader/Bluestein).
 *
 * Correctness (order-agnostic — output is digit-scrambled per axis):
 *   roundtrip fwd+bwd == N1*N2*N3*x, and SORTED-|X| vs MKL (permutation-
 *   invariant elementwise cross-check of the spectrum multiset).
 *
 * Timing: rdtsc best-of, ST first (pinned, mkl_set_num_threads(1)); pass A
 * timed FLAT vs BLOCKED per size, best column vs MKL. Optional MT section.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_fft3d_vs_mkl.c --mkl --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "fft3d.h"
#include "dp_planner.h"
#include "measure.h"
#include "env.h"
#include "generator/generated/registry.h"

#define PIN_CORE 0
#ifndef BEST_OF
#define BEST_OF 7
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
static inline double now_ns(void){ return vfft_proto_now_ns(); }
static int reps_for(size_t t){int r=(int)(4e7/(t+1)); if(r<6)r=6; if(r>20000)r=20000; return r;}

/* ── DP-planned axis: decision -> plan_create_ex; prime/non-smooth -> auto ── */
static stride_plan_t *dp_plan_axis(int N, size_t K,
                                   const vfft_proto_registry_t *reg,
                                   char *desc, size_t desc_sz) {
    vfft_proto_dp_context_t ctx;
    vfft_proto_dp_init(&ctx, K, N);
    vfft_proto_plan_decision_t dec, pool[VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int npool = 0;
    double t0 = now_ns();
    double ns = vfft_proto_dp_plan_measure(&ctx, N, reg, &dec, pool, &npool, 0);
    double plan_ms = (now_ns() - t0) * 1e-6;
    vfft_proto_dp_destroy(&ctx);

    if (ns >= 1e17 || dec.nf <= 0) {
        snprintf(desc, desc_sz, "auto/dispatch (DP n/a)");
        return vfft_proto_auto_plan_dispatch(N, K, reg, NULL);
    }
    int off = snprintf(desc, desc_sz, "%s[", dec.use_dif_forward ? "DIF " : "");
    for (int s = 0; s < dec.nf && off < (int)desc_sz - 8; s++)
        off += snprintf(desc + off, desc_sz - off, "%d%s",
                        dec.factors[s], s + 1 < dec.nf ? "x" : "");
    snprintf(desc + off, desc_sz - off, "] %.0fms", plan_ms);
    return vfft_proto_plan_create_ex(N, K, dec.factors, dec.variants, dec.nf,
                                     dec.use_dif_forward, reg);
}

static stride_plan_t *make_3d(int N1, int N2, int N3, size_t a_block,
                              const vfft_proto_registry_t *reg, int verbose) {
    size_t K0 = (size_t)N2 * (size_t)N3;
    size_t NR = (size_t)N1 * (size_t)N2;
    size_t B  = _fft3d_choose_tile(N3, NR);
    char d0[128], d1[128], dr[128];
    stride_plan_t *p0 = dp_plan_axis(N1, K0,        reg, d0, sizeof d0);
    stride_plan_t *p1 = dp_plan_axis(N2, (size_t)N3, reg, d1, sizeof d1);
    stride_plan_t *pr = dp_plan_axis(N3, B,         reg, dr, sizeof dr);
    if (verbose)
        printf("    DP: ax0 %-28s ax1 %-24s row %-24s\n", d0, d1, dr);
    if (!p0 || !p1 || !pr) {
        if (p0) stride_plan_destroy(p0);
        if (p1) stride_plan_destroy(p1);
        if (pr) stride_plan_destroy(pr);
        return NULL;
    }
    return stride_plan_3d_from(N1, N2, N3, B, a_block, p0, p1, pr);
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void run_cell(int N1, int N2, int N3, const vfft_proto_registry_t *reg) {
    size_t T = (size_t)N1 * N2 * N3;
    double sc = (double)T;

    /* plans: FLAT and forced-BLOCKED share DP inners? DP is memoized per
     * context only, so just build twice (planning printed once). */
    stride_plan_t *pf = make_3d(N1, N2, N3, 0,          reg, 1);
    stride_plan_t *pb = make_3d(N1, N2, N3, (size_t)-1, reg, 0);  /* heuristic block */
    if (!pf || !pb) { printf("  %dx%dx%d plan NULL\n", N1, N2, N3);
        if (pf) stride_plan_destroy(pf); if (pb) stride_plan_destroy(pb); return; }
    size_t eff_blk = ((stride_fft3d_data_t *)pb->override_data)->a_block;

    double *re=AALLOC(T*8),*im=AALLOC(T*8),*xr=AALLOC(T*8),*xi=AALLOC(T*8);
    srand(11 + N1 + N2 + N3);
    for (size_t i=0;i<T;i++){ xr[i]=(double)rand()/RAND_MAX-0.5;
                              xi[i]=(double)rand()/RAND_MAX-0.5; }

    /* roundtrip (FLAT plan; BLOCKED validated bit-identical by the test) */
    memcpy(re,xr,T*8); memcpy(im,xi,T*8);
    stride_execute_fwd(pf,re,im);
    double *fr=AALLOC(T*8),*fi=AALLOC(T*8); memcpy(fr,re,T*8); memcpy(fi,im,T*8);
    stride_execute_bwd(pf,re,im);
    double rt=0;
    for (size_t i=0;i<T;i++){ double a=fabs(re[i]/sc-xr[i]),b=fabs(im[i]/sc-xi[i]);
                              if(a>rt)rt=a; if(b>rt)rt=b; }

    /* MKL 3D complex, CCE INTERLEAVED (its default and fastest storage) --
     * the v1.0 house methodology. NOTE: an earlier revision of this bench
     * used DFTI_REAL_REAL (split), inherited from a drifted bench file;
     * MKL's split multi-dim path measured 2.3-5.2x SLOWER than CCE on the
     * bench host, so vs-split ratios overstate the margin. Each library
     * runs its native layout on the same logical data. */
    double *zi=AALLOC(T*16),*zo=AALLOC(T*16);
    double *mr=zo,*mi=NULL; (void)mi;
    for (size_t q=0;q<T;q++){ zi[2*q]=xr[q]; zi[2*q+1]=xi[q]; }
    DFTI_DESCRIPTOR_HANDLE h=0; MKL_LONG dims[3]={N1,N2,N3};
    int mok=0; double sme=-1;
    if (DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_COMPLEX,3,dims)==DFTI_NO_ERROR){
        DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
        mok=(DftiCommitDescriptor(h)==DFTI_NO_ERROR);
    }
    if (mok){
        DftiComputeForward(h,zi,zo);
        /* sorted-|X| multiset compare: permutation-invariant, so it cross-
         * checks the scrambled spectrum elementwise against MKL. */
        double *sa=AALLOC(T*8),*sb=AALLOC(T*8); double mm=0;
        for (size_t i=0;i<T;i++){ sa[i]=hypot(fr[i],fi[i]);
                                  sb[i]=hypot(zo[2*i],zo[2*i+1]);
                                  if (sb[i]>mm) mm=sb[i]; }
        qsort(sa,T,8,cmp_d); qsort(sb,T,8,cmp_d);
        sme=0; for (size_t i=0;i<T;i++){ double e=fabs(sa[i]-sb[i]); if(e>sme)sme=e; }
        if (mm>0) sme/=mm;
        AFREE(sa); AFREE(sb);
    }

    /* timing: FLAT, BLOCKED, MKL — interleaved best-of */
    int reps=reps_for(T); double bf=1e18,bb=1e18,bm=1e18;
    memcpy(re,xr,T*8); memcpy(im,xi,T*8);
    for (int w=0;w<2;w++){ stride_execute_fwd(pf,re,im); stride_execute_fwd(pb,re,im);
                           if(mok)DftiComputeForward(h,zi,zo); }
    for (int t=0;t<BEST_OF;t++){
        double t0=now_c(); for(int i=0;i<reps;i++) stride_execute_fwd(pf,re,im);
        double v=(now_c()-t0)/reps; if(v<bf)bf=v;
        t0=now_c(); for(int i=0;i<reps;i++) stride_execute_fwd(pb,re,im);
        v=(now_c()-t0)/reps; if(v<bb)bb=v;
        if(mok){ t0=now_c(); for(int i=0;i<reps;i++) DftiComputeForward(h,zi,zo);
                 v=(now_c()-t0)/reps; if(v<bm)bm=v; }
    }
    double bv = bf<bb?bf:bb;
    printf("  %4dx%-4dx%-4d rt=%.1e sortMKL=%.1e | flat %11.0f | blk(%5zu) %11.0f | mkl %11.0f | A=%s | speed %.3f %s\n",
           N1,N2,N3, rt, sme, bf, eff_blk, bb, mok?bm:0,
           bb<bf?"BLOCKED":"FLAT", (mok&&bv>0)?bm/bv:0,
           rt<1e-9?"":"*** RT FAIL ***");
    fflush(stdout);

    if(h)DftiFreeDescriptor(&h);
    AFREE(re);AFREE(im);AFREE(xr);AFREE(xi);AFREE(fr);AFREE(fi);AFREE(zi);AFREE(zo);
    stride_plan_destroy(pf); stride_plan_destroy(pb);
}

int main(int argc, char **argv) {
    stride_env_init();
    if (stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn: pin failed\n");
    mkl_set_num_threads(1);
    stride_set_num_threads(1);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    printf("== 3D c2c (dag fft3d.h, DP-planned inners) vs MKL DFTI 3D "
           "(CCE interleaved, NOT_INPLACE, ST, cpu%d) ==\n", PIN_CORE);
    printf("# cycles/transform, best-of-%d. speed>1 = we win. "
           "sortMKL = sorted-|X| multiset vs MKL (order-agnostic elementwise).\n", BEST_OF);

    if (argc >= 4 && argv[1][0] >= '0' && argv[1][0] <= '9') {
        run_cell(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), &reg);
        return 0;
    }

    run_cell( 32, 32, 32,&reg);
    run_cell( 48, 48, 48,&reg);
    run_cell( 64, 64, 64,&reg);
    run_cell(128,128,128,&reg);
    run_cell(256, 64, 32,&reg);   /* aniso: big axis-0 (K0=2048) */
    run_cell( 32, 64,256,&reg);   /* aniso: big rows             */

    if (argc>1 && !strcmp(argv[1],"--mt")) {
        int T = argc>2 ? atoi(argv[2]) : 4;
        printf("== MT: dag T=%d vs MKL T=%d ==\n", T, T);
        stride_set_num_threads(T); mkl_set_num_threads(T);
        run_cell( 64, 64, 64,&reg);
        run_cell(128,128,128,&reg);
        stride_set_num_threads(1); mkl_set_num_threads(1);
    }
    return 0;
}
