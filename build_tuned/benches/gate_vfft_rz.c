/* §6a24 gate: R2C/C2R interleaved-z public contract (NULL-halves convention).
 * dim==NULL on R2C => dre = CCE spectrum; sim==NULL on C2R => sre = CCE in.
 * z arms must be BIT-identical to split arms (same math, different stores;
 * rfft convert-around copies the split values). Cells cover both dispatches. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int cell(vfft_wisdom *w, int N, size_t K) {
    size_t H = (size_t)N/2 + 1;
    int ok = 1;
    vfft_config_t cf; memset(&cf, 0, sizeof cf);
    cf.transform = VFFT_R2C; cf.placement = VFFT_OUTOFPLACE; cf.rigor = VFFT_MEASURE;
    cf.dims = 1; cf.n[0] = N; cf.howmany = K; cf.wisdom = w;
    vfft_plan pf = vfft_create(&cf);
    cf.transform = VFFT_C2R;
    vfft_plan pb = vfft_create(&cf);
    if (!pf || !pb) { printf("  [%d,%zu] create FAIL\n", N, K); return 0; }
    double *x  = aligned_alloc(64, (size_t)N*K*8);
    double *rr = aligned_alloc(64, H*K*8), *ri = aligned_alloc(64, H*K*8);
    double *z  = aligned_alloc(64, 2*H*K*8);
    double *yr = aligned_alloc(64, (size_t)N*K*8), *yz = aligned_alloc(64, (size_t)N*K*8);
    srand(7 + N + (int)K);
    for (size_t i = 0; i < (size_t)N*K; i++) x[i] = 2.0*rand()/RAND_MAX - 1;
    /* fwd: split ref vs z */
    vfft_execute(pf, VFFT_FORWARD, x, NULL, rr, ri);
    memset(z, 0x5A, 2*H*K*8);
    vfft_execute(pf, VFFT_FORWARD, x, NULL, z, NULL);
    size_t bad = 0;
    for (size_t i = 0; i < H*K; i++)
        if (z[2*i] != rr[i] || z[2*i+1] != ri[i]) bad++;
    printf("  [%4d,%4zu] fwd z-vs-split BIT: %s (%zu)\n", N, K, bad?"**FAIL**":"PASS", bad);
    ok &= (bad == 0);
    /* bwd: split-in ref vs z-in */
    vfft_execute(pb, VFFT_BACKWARD, rr, ri, yr, NULL);
    memset(yz, 0x5A, (size_t)N*K*8);
    vfft_execute(pb, VFFT_BACKWARD, z, NULL, yz, NULL);
    bad = 0;
    for (size_t i = 0; i < (size_t)N*K; i++) if (yz[i] != yr[i]) bad++;
    printf("  [%4d,%4zu] bwd z-vs-split BIT: %s (%zu)\n", N, K, bad?"**FAIL**":"PASS", bad);
    ok &= (bad == 0);
    /* roundtrip sanity: y/N ≈ x */
    double mx = 0;
    for (size_t i = 0; i < (size_t)N*K; i++) {
        double d = fabs(yz[i]/(double)N - x[i]); if (d > mx) mx = d;
    }
    printf("  [%4d,%4zu] roundtrip max|y/N - x| = %.3e %s\n", N, K, mx, mx<1e-11?"PASS":"**FAIL**");
    ok &= (mx < 1e-11);
    vfft_destroy(pf); vfft_destroy(pb);
    free(x); free(rr); free(ri); free(z); free(yr); free(yz);
    return ok;
}
static int cell2d(int N1, int N2)
{
    size_t H2 = (size_t)N2 / 2 + 1, M = (size_t)N1 * H2;
    int fails = 0;
    vfft_config_t cf; memset(&cf, 0, sizeof cf);
    cf.transform = VFFT_R2C; cf.placement = VFFT_OUTOFPLACE;
    cf.rigor = VFFT_MEASURE; cf.dims = 2; cf.n[0] = N1; cf.n[1] = N2; cf.howmany = 1;
    vfft_plan pf = vfft_create(&cf);
    cf.transform = VFFT_C2R;
    vfft_plan pb = vfft_create(&cf);
    if (!pf || !pb) { printf("  [2D %4d x%4d] create FAIL\n", N1, N2); return 0; }
    double *x = aligned_alloc(64, (size_t)N1 * N2 * 8);
    double *rr = aligned_alloc(64, M * 8), *ri = aligned_alloc(64, M * 8);
    double *z = aligned_alloc(64, 2 * M * 8);
    double *y = aligned_alloc(64, (size_t)N1 * N2 * 8);
    double *yz = aligned_alloc(64, (size_t)N1 * N2 * 8);
    srand(21 + N1 + N2);
    for (size_t i = 0; i < (size_t)N1 * N2; i++) x[i] = 2.0 * rand() / RAND_MAX - 1;
    vfft_execute(pf, VFFT_FORWARD, x, NULL, rr, ri);
    vfft_execute(pf, VFFT_FORWARD, x, NULL, z, NULL);
    size_t bad = 0;
    for (size_t i = 0; i < M; i++)
        if (z[2*i] != rr[i] || z[2*i+1] != ri[i]) bad++;
    printf("  [2D %4d x%4d] fwd z-vs-split BIT: %s (%zu)\n", N1, N2,
           bad ? "**FAIL**" : "PASS", bad);
    fails += bad ? 1 : 0;
    vfft_execute(pb, VFFT_BACKWARD, rr, ri, y, NULL);
    vfft_execute(pb, VFFT_BACKWARD, z, NULL, yz, NULL);
    bad = 0;
    for (size_t i = 0; i < (size_t)N1 * N2; i++) if (yz[i] != y[i]) bad++;
    printf("  [2D %4d x%4d] bwd z-vs-split BIT: %s (%zu)\n", N1, N2,
           bad ? "**FAIL**" : "PASS", bad);
    fails += bad ? 1 : 0;
    double sc = 1.0 / ((double)N1 * N2), mx = 0;
    for (size_t i = 0; i < (size_t)N1 * N2; i++) {
        double d = fabs(yz[i] * sc - x[i]);
        if (d > mx) mx = d;
    }
    printf("  [2D %4d x%4d] roundtrip max|y/(N1*N2) - x| = %.3e %s\n", N1, N2,
           mx, mx < 1e-12 ? "PASS" : "**FAIL**");
    fails += mx < 1e-12 ? 0 : 1;
    vfft_destroy(pf); vfft_destroy(pb);
    free(x); free(rr); free(ri); free(z); free(y); free(yz);
    return fails ? 0 : 1;
}

int main(void) {
    vfft_wisdom *w = vfft_wisdom_load("/tmp/wbr2c3");
    int ok = 1;
    ok &= cell(w, 512, 256);   /* stride dispatch, native z */
    ok &= cell(w, 2000, 4);    /* rfft dispatch, convert-around */
    ok &= cell(w, 200, 4);     /* rfft dispatch, odd radix set */
    ok &= cell(w, 128, 67);    /* odd B/K */
    ok &= cell2d(64, 64);   /* strided-r2c covered cell */
    ok &= cell2d(64, 128);
    ok &= cell2d(64, 256);  /* stw-covered cell */
    ok &= cell2d(64, 512);  /* r512 mono cell */
    ok &= cell2d(40, 32);   /* Q0-class: staged tail under avx512 now */
    ok &= cell2d(27, 32);   /* Q2: odd rows (lone-row two-for-one; 27=3^3 col-supported) */
    ok &= cell2d(44, 64);   /* Q2: ragged pairs both editions */
    ok &= cell2d(9, 128);   /* Q2: tiny + odd + large-N mono */
    {   /* Q4: dims==2 howmany!=1 must be rejected (silent-wrong guard) */
        vfft_config_t c4; memset(&c4,0,sizeof c4);
        c4.transform=VFFT_R2C; c4.placement=VFFT_OUTOFPLACE;
        c4.rigor=VFFT_MEASURE; c4.dims=2; c4.n[0]=32; c4.n[1]=32;
        c4.howmany=2;
        vfft_plan p4=vfft_create(&c4);
        printf("  [2D K=2] reject: %s\n", p4?"**FAIL**":"PASS");
        ok &= (p4==NULL);
    }
    {   /* §6a51: prime-N1 col path must be a clean NULL (was: silent-wrong) */
        vfft_config_t c5; memset(&c5,0,sizeof c5);
        c5.transform=VFFT_R2C; c5.placement=VFFT_OUTOFPLACE;
        c5.rigor=VFFT_MEASURE; c5.dims=2; c5.n[0]=41; c5.n[1]=32;
        c5.howmany=1;
        vfft_plan p5=vfft_create(&c5);
        printf("  [2D prime-N1] reject: %s\n", p5?"**FAIL**":"PASS");
        ok &= (p5==NULL);
    }
    ok &= cell2d(40, 64);   /* Q0: N1 % 16 != 0 — avx512 must fall to avx2 */
    ok &= cell2d(96, 200);
    if (w) vfft_wisdom_free(w);
    puts(ok ? "VFFT RZ GATE: ALL PASS" : "VFFT RZ GATE: FAILURES");
    return ok ? 0 : 1;
}
