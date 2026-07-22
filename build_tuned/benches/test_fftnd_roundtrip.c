/* test_fftnd_roundtrip.c — validate rank-general fftnd.h on dag.
 *
 * Gates:
 *   1. RANK-3 BIT-MATCH: fftnd (rank=3) fwd output must memcmp-equal
 *      fft3d.h's fwd output, for EVERY split s in {1,2} — fusion reorders
 *      block interleaving only; each element's op sequence is unchanged, so
 *      equality must be exact, not approximate. (Same auto inners on both
 *      sides: auto_plan_dispatch is deterministic per (N,K).)
 *   2. SPLIT EQUIVALENCE (rank 4): fwd(s=1) == fwd(s=2) == fwd(s=3), memcmp.
 *   3. ROUNDTRIP  bwd(fwd(x)) == (prod N) * x  at rank 3 and 4.
 *   4. PARSEVAL + DC on rank-4 cells.
 *   Matrix includes prime axes at every rank-4 position (Rader/Bluestein in
 *   unfused, fused-middle, and tiled positions), forced lane-blocks, and
 *   T in {1,2,4} (hierarchical outer x lane MT on anisotropic shapes).
 *
 * Build: python build.py --src benches/test_fftnd_roundtrip.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft3d.h"
#include "fftnd.h"
#include "generator/generated/registry.h"

#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif

static double frand(void) { return 2.0 * ((double)rand() / RAND_MAX) - 1.0; }
static int g_fail = 0;

static stride_plan_t *make_nd(int rank, const int *N, int split,
                              const size_t *lb,
                              const vfft_proto_registry_t *reg) {
    stride_fftnd_data_t tmp; memset(&tmp, 0, sizeof tmp);
    tmp.rank = rank;
    for (int m = 0; m < rank; m++) tmp.N[m] = N[m];
    _fftnd_fill_ok(&tmp);
    size_t B = _fftnd_choose_tile(N[rank-1], tmp.O[rank-1]);
    stride_plan_t *plans[FFTND_MAX_RANK] = {0};
    for (int m = 0; m < rank; m++) {
        size_t Kp = (m == rank-1) ? B : tmp.K[m];
        plans[m] = vfft_proto_auto_plan_dispatch(N[m], Kp, reg, NULL);
        if (!plans[m]) { for (int i=0;i<m;i++) stride_plan_destroy(plans[i]); return NULL; }
    }
    return stride_plan_nd_from(rank, N, B, split, lb, plans);
}

/* ── Gate 1: rank-3 bit-match vs fft3d, s in {1,2} ── */
static void gate_rank3_bitmatch(int N1, int N2, int N3,
                                const vfft_proto_registry_t *reg) {
    size_t n = (size_t)N1*N2*N3;
    int Nv[3] = { N1, N2, N3 };
    double *xr=AALLOC(n*8),*xi=AALLOC(n*8);
    double *ar=AALLOC(n*8),*ai=AALLOC(n*8),*br=AALLOC(n*8),*bi=AALLOC(n*8);
    srand(3 + N1);
    for (size_t i=0;i<n;i++){ xr[i]=frand(); xi[i]=frand(); }

    /* reference: fft3d (auto inners, FLAT) */
    size_t K0=(size_t)N2*N3, NR=(size_t)N1*N2, B=_fft3d_choose_tile(N3,NR);
    stride_plan_t *p0=vfft_proto_auto_plan_dispatch(N1,K0,reg,NULL);
    stride_plan_t *p1=vfft_proto_auto_plan_dispatch(N2,(size_t)N3,reg,NULL);
    stride_plan_t *pr=vfft_proto_auto_plan_dispatch(N3,B,reg,NULL);
    stride_plan_t *ref=stride_plan_3d_from(N1,N2,N3,B,0,p0,p1,pr);
    memcpy(ar,xr,n*8); memcpy(ai,xi,n*8);
    stride_execute_fwd(ref,ar,ai);

    for (int s=1;s<=2;s++){
        stride_plan_t *pn=make_nd(3,Nv,s,NULL,reg);
        memcpy(br,xr,n*8); memcpy(bi,xi,n*8);
        stride_execute_fwd(pn,br,bi);
        int eq = !memcmp(ar,br,n*8) && !memcmp(ai,bi,n*8);
        if (!eq) g_fail++;
        printf("  r3 bitmatch %3dx%-3dx%-3d s=%d vs fft3d: %s\n",
               N1,N2,N3,s, eq?"EXACT":"**MISMATCH**");
        stride_plan_destroy(pn);
    }
    stride_plan_destroy(ref);
    AFREE(xr);AFREE(xi);AFREE(ar);AFREE(ai);AFREE(br);AFREE(bi);
}

/* ── Gates 2-4: rank-4 cell ── */
static void run4(const int *N, int split, const size_t *lb, int T,
                 double *fwd_ref_re, double *fwd_ref_im, /* s-equiv check, may be NULL */
                 const vfft_proto_registry_t *reg) {
    size_t n = (size_t)N[0]*N[1]*N[2]*N[3];
    double sc = (double)n;
    stride_set_num_threads(T);
    stride_plan_t *p = make_nd(4, N, split, lb, reg);
    if (!p) { printf("  r4 %dx%dx%dx%d s=%d T=%d PLAN FAIL\n",
                     N[0],N[1],N[2],N[3],split,T); g_fail++; return; }
    stride_fftnd_data_t *d = (stride_fftnd_data_t *)p->override_data;

    double *re=AALLOC(n*8),*im=AALLOC(n*8),*rr=AALLOC(n*8),*ri=AALLOC(n*8);
    srand(17+N[0]+N[3]);
    double ien=0;
    for (size_t i=0;i<n;i++){ rr[i]=re[i]=frand(); ri[i]=im[i]=frand();
                              ien+=re[i]*re[i]+im[i]*im[i]; }
    stride_execute_fwd(p,re,im);

    int seq = 1;
    if (fwd_ref_re) seq = !memcmp(re,fwd_ref_re,n*8) && !memcmp(im,fwd_ref_im,n*8);
    else { fwd_ref_re=NULL; }

    double oen=0; for (size_t i=0;i<n;i++) oen+=re[i]*re[i]+im[i]*im[i];
    double pars=fabs(oen-sc*ien)/(sc*ien);

    stride_execute_bwd(p,re,im);
    double rt=0;
    for (size_t i=0;i<n;i++){
        double rel=(fabs(re[i]-sc*rr[i])+fabs(im[i]-sc*ri[i]))
                  /(fabs(sc*rr[i])+fabs(sc*ri[i])+1e-300);
        if (rel>rt) rt=rel;
    }
    for (size_t i=0;i<n;i++){ re[i]=1.0; im[i]=0.0; }
    stride_execute_fwd(p,re,im);
    int nz=0; for (size_t i=0;i<n;i++)
        if (fabs(re[i])+fabs(im[i]) > 1e-6*sc) nz++;

    int ok = rt<1e-11 && pars<1e-12 && nz==1 && seq;
    if (!ok) g_fail++;
    printf("  r4 %3dx%-3dx%-3dx%-3d s=%d lb=[%zu,%zu,%zu] T=%d rt=%.1e pars=%.1e dc=%d %s%s\n",
           N[0],N[1],N[2],N[3], d->split,
           d->lane_block[0],d->lane_block[1],d->lane_block[2], T,
           rt, pars, nz, seq?"":"S-MISMATCH ", ok?"OK":"**FAIL**");
    AFREE(re);AFREE(im);AFREE(rr);AFREE(ri);
    stride_plan_destroy(p);
}

/* fwd-only snapshot for split-equivalence reference */
static void snap_fwd(const int *N, int split, const vfft_proto_registry_t *reg,
                     double *or_, double *oi_) {
    size_t n=(size_t)N[0]*N[1]*N[2]*N[3];
    stride_set_num_threads(1);
    stride_plan_t *p = make_nd(4,N,split,NULL,reg);
    srand(17+N[0]+N[3]);
    for (size_t i=0;i<n;i++){ or_[i]=frand(); oi_[i]=frand(); }
    stride_execute_fwd(p,or_,oi_);
    stride_plan_destroy(p);
}

int main(void) {
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    printf("fftnd correctness matrix\n");

    /* Gate 1 — rank-3 bit-match vs fft3d (the 3D fusion back-port gate) */
    stride_set_num_threads(1);
    gate_rank3_bitmatch(32,32,32,&reg);
    gate_rank3_bitmatch(48,20,12,&reg);
    gate_rank3_bitmatch(16,61,16,&reg);   /* prime middle */

    /* Gate 2 — rank-4 split equivalence (bit-exact across s), T=1 */
    {
        int N[4]={16,16,16,16};
        size_t n=(size_t)16*16*16*16;
        double *fr=AALLOC(n*8),*fi=AALLOC(n*8);
        snap_fwd(N,3,&reg,fr,fi);            /* s=3 reference (unfused) */
        run4(N,1,NULL,1,fr,fi,&reg);
        run4(N,2,NULL,1,fr,fi,&reg);
        run4(N,3,NULL,1,fr,fi,&reg);
        AFREE(fr);AFREE(fi);
    }

    /* Gates 3-4 — rank-4 matrix: sizes x s x T (+ forced lane blocks) */
    int sizes[][4] = {
        { 16, 16, 16, 16 },
        {  8, 16, 32, 64 },   /* aniso ascending          */
        { 64, 32, 16,  8 },   /* aniso descending         */
        { 32, 32, 32, 64 },   /* QCD-ish 32^3 x 64 (32MB) */
        { 13, 16,  8,  8 },   /* prime axis 0             */
        {  8, 13,  8,  8 },   /* prime axis 1 (fusable)   */
        {  8,  8, 13,  8 },   /* prime axis 2 (fusable)   */
        {  8,  8,  8, 13 },   /* prime last (tiled)       */
        {  4,  8,  8,  8 },   /* O_s < T stress           */
    };
    size_t lbf[3] = { 64, 32, 0 };   /* forced lane blocks on axes 0,1 */
    for (size_t c = 0; c < sizeof(sizes)/sizeof(sizes[0]); c++)
        for (int s = 1; s <= 3; s++)
            for (int ti = 0; ti < 3; ti++) {
                int T = (int[]){1,2,4}[ti];
                run4(sizes[c], s, NULL, T, NULL, NULL, &reg);
            }
    /* forced lane-blocked variants, s=1 and s=2, T in {1,4} */
    run4(sizes[3],1,lbf,1,NULL,NULL,&reg);
    run4(sizes[3],1,lbf,4,NULL,NULL,&reg);
    run4(sizes[3],2,lbf,1,NULL,NULL,&reg);
    run4(sizes[3],2,lbf,4,NULL,NULL,&reg);

    /* rank-2 sanity through the same machinery (s forced to 1) */
    {
        int N2v[2]={64,48};
        stride_set_num_threads(1);
        stride_plan_t *p=make_nd(2,N2v,1,NULL,&reg);
        size_t n=64*48;
        double *re=AALLOC(n*8),*im=AALLOC(n*8),*rr=AALLOC(n*8),*ri=AALLOC(n*8);
        srand(5);
        for (size_t i=0;i<n;i++){ rr[i]=re[i]=frand(); ri[i]=im[i]=frand(); }
        stride_execute_fwd(p,re,im); stride_execute_bwd(p,re,im);
        double mx=0;
        for (size_t i=0;i<n;i++){
            double rel=(fabs(re[i]-n*rr[i])+fabs(im[i]-n*ri[i]))
                      /(fabs(n*rr[i])+fabs(n*ri[i])+1e-300);
            if (rel>mx) mx=rel;
        }
        int ok = mx<1e-11; if(!ok) g_fail++;
        printf("  r2 64x48 via fftnd rt=%.1e %s\n", mx, ok?"OK":"**FAIL**");
        AFREE(re);AFREE(im);AFREE(rr);AFREE(ri);
        stride_plan_destroy(p);
    }

    stride_set_num_threads(1);
    printf(g_fail ? "\n%d FAILURE(S)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
