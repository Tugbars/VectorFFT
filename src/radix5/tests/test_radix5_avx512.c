/**
 * @file  test_radix5_avx512.c
 * @brief Tests for radix-5 WFTA AVX-512 butterfly (U=2, masked tails)
 *
 * Tests:
 *   1. N1 forward vs naive DFT-5
 *   2. N1 fwd → bwd roundtrip
 *   3. Twiddled fwd vs naive (trivial twiddles)
 *   4. Twiddled fwd → bwd roundtrip (K=16)
 *   5. Masked tail K=11 (8+3 masked)
 *   6. Twiddled fwd vs manual twiddled DFT-5 (K=8)
 *   7. U=2 stress K=32 (2 full U=2 iters) N1 roundtrip
 *   8. U=2 twiddled K=64 (4 full U=2 iters) roundtrip
 *   9. Masked tail K=1 (edge case — single element)
 *  10. Masked tail K=3 (all-masked, no SIMD loop)
 *  11. K=17 (1 U=2 + masked tail of 1)
 *
 * Build:
 *   gcc -O2 -mavx512f -Wall -Wextra -I. -o test_radix5_avx512 test_radix5_avx512.c -lm
 *   icx -O2 -xCOMMON-AVX512 -Wall -I. -o test_radix5_avx512 test_radix5_avx512.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fft_radix5_avx512.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int g_pass = 0, g_fail = 0;
static const double TOL = 1e-12;

static void check(const char *tag, double got, double want) {
    double err = fabs(got - want);
    if (err < TOL) { g_pass++; }
    else { g_fail++; printf("  FAIL %s: got %.15e want %.15e err %.3e\n", tag, got, want, err); }
}

/* Naive DFT-5 reference (forward) */
static void naive_dft5_fwd(const double *xr, const double *xi,
                           double *yr, double *yi) {
    for (int m = 0; m < 5; m++) {
        yr[m] = 0; yi[m] = 0;
        for (int n = 0; n < 5; n++) {
            double angle = -2.0 * M_PI * m * n / 5.0;
            yr[m] += xr[n] * cos(angle) - xi[n] * sin(angle);
            yi[m] += xr[n] * sin(angle) + xi[n] * cos(angle);
        }
    }
}

/* ── Test 1: N1 forward vs naive DFT-5 ── */
static void test_n1_fwd_correctness(void) {
    printf("Test 1: N1 forward vs naive DFT-5\n");
    double a_re[1]={1}, a_im[1]={0.5}, b_re[1]={2}, b_im[1]={-1};
    double c_re[1]={-0.5}, c_im[1]={3}, d_re[1]={0.7}, d_im[1]={-2};
    double e_re[1]={1.3}, e_im[1]={0.8};
    double y0r[1],y0i[1],y1r[1],y1i[1],y2r[1],y2i[1],y3r[1],y3i[1],y4r[1],y4i[1];

    radix5_wfta_fwd_avx512_N1(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,e_re,e_im,
                              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i, 1);

    double xr[5]={1,2,-0.5,0.7,1.3}, xi[5]={0.5,-1,3,-2,0.8};
    double ref_r[5], ref_i[5];
    naive_dft5_fwd(xr, xi, ref_r, ref_i);

    double *yr[5]={y0r,y1r,y2r,y3r,y4r}, *yi[5]={y0i,y1i,y2i,y3i,y4i};
    for (int m = 0; m < 5; m++) {
        char tag[32]; sprintf(tag,"y%d_re",m); check(tag, yr[m][0], ref_r[m]);
        sprintf(tag,"y%d_im",m); check(tag, yi[m][0], ref_i[m]);
    }
}

/* ── Test 2: N1 fwd → bwd roundtrip ── */
static void test_n1_roundtrip(void) {
    printf("Test 2: N1 fwd -> bwd roundtrip\n");
    int K = 8;
    double a_re[8],a_im[8],b_re[8],b_im[8],c_re[8],c_im[8];
    double d_re[8],d_im[8],e_re[8],e_im[8];
    double y0r[8],y0i[8],y1r[8],y1i[8],y2r[8],y2i[8],y3r[8],y3i[8],y4r[8],y4i[8];
    double z0r[8],z0i[8],z1r[8],z1i[8],z2r[8],z2i[8],z3r[8],z3i[8],z4r[8],z4i[8];

    srand(42);
    for (int i = 0; i < K; i++) {
        a_re[i]=(rand()/(double)RAND_MAX-0.5)*10; a_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        b_re[i]=(rand()/(double)RAND_MAX-0.5)*10; b_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        c_re[i]=(rand()/(double)RAND_MAX-0.5)*10; c_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        d_re[i]=(rand()/(double)RAND_MAX-0.5)*10; d_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        e_re[i]=(rand()/(double)RAND_MAX-0.5)*10; e_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
    }
    radix5_wfta_fwd_avx512_N1(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,e_re,e_im,
                              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,K);
    radix5_wfta_bwd_avx512_N1(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                              z0r,z0i,z1r,z1i,z2r,z2i,z3r,z3i,z4r,z4i,K);
    const double *orig[5][2]={{a_re,a_im},{b_re,b_im},{c_re,c_im},{d_re,d_im},{e_re,e_im}};
    double *out[5][2]={{z0r,z0i},{z1r,z1i},{z2r,z2i},{z3r,z3i},{z4r,z4i}};
    for (int leg = 0; leg < 5; leg++)
        for (int i = 0; i < K; i++) {
            char t[32]; sprintf(t,"leg%d[%d]_re",leg,i); check(t,out[leg][0][i]/5.0,orig[leg][0][i]);
            sprintf(t,"leg%d[%d]_im",leg,i); check(t,out[leg][1][i]/5.0,orig[leg][1][i]);
        }
}

/* ── Test 3: Twiddled forward vs naive (K=1, trivial twiddles) ── */
static void test_twiddled_fwd_correctness(void) {
    printf("Test 3: Twiddled forward vs naive (K=1, stride=5)\n");
    double a_re[1]={1}, a_im[1]={0.5}, b_re[1]={2}, b_im[1]={-1};
    double c_re[1]={-0.5}, c_im[1]={3}, d_re[1]={0.7}, d_im[1]={-2};
    double e_re[1]={1.3}, e_im[1]={0.8};
    double tw1r[1]={1},tw1i[1]={0},tw2r[1]={1},tw2i[1]={0};
    double y0r[1],y0i[1],y1r[1],y1i[1],y2r[1],y2i[1],y3r[1],y3i[1],y4r[1],y4i[1];

    radix5_wfta_fwd_avx512(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,e_re,e_im,
                           y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                           tw1r,tw1i,tw2r,tw2i,1);

    double xr[5]={1,2,-0.5,0.7,1.3}, xi[5]={0.5,-1,3,-2,0.8};
    double ref_r[5], ref_i[5];
    naive_dft5_fwd(xr, xi, ref_r, ref_i);

    double *yr[5]={y0r,y1r,y2r,y3r,y4r}, *yi[5]={y0i,y1i,y2i,y3i,y4i};
    for (int m = 0; m < 5; m++) {
        char tag[32]; sprintf(tag,"y%d_re",m); check(tag, yr[m][0], ref_r[m]);
        sprintf(tag,"y%d_im",m); check(tag, yi[m][0], ref_i[m]);
    }
}

/* ── Test 4: Twiddled fwd → bwd roundtrip (K=16) ── */
static void test_twiddled_roundtrip(void) {
    printf("Test 4: Twiddled fwd -> bwd roundtrip (K=16)\n");
    int K = 16;
    double a_re[16],a_im[16],b_re[16],b_im[16],c_re[16],c_im[16];
    double d_re[16],d_im[16],e_re[16],e_im[16];
    double tw1r[16],tw1i[16],tw2r[16],tw2i[16];
    double y0r[16],y0i[16],y1r[16],y1i[16],y2r[16],y2i[16];
    double y3r[16],y3i[16],y4r[16],y4i[16];
    double z0r[16],z0i[16],z1r[16],z1i[16],z2r[16],z2i[16];
    double z3r[16],z3i[16],z4r[16],z4i[16];

    srand(777);
    int N = 5 * K;
    for (int i = 0; i < K; i++) {
        a_re[i]=(rand()/(double)RAND_MAX-0.5)*10; a_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        b_re[i]=(rand()/(double)RAND_MAX-0.5)*10; b_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        c_re[i]=(rand()/(double)RAND_MAX-0.5)*10; c_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        d_re[i]=(rand()/(double)RAND_MAX-0.5)*10; d_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        e_re[i]=(rand()/(double)RAND_MAX-0.5)*10; e_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        double angle = -2.0*M_PI*i/N;
        tw1r[i]=cos(angle);   tw1i[i]=sin(angle);
        tw2r[i]=cos(2*angle); tw2i[i]=sin(2*angle);
    }
    radix5_wfta_fwd_avx512(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,e_re,e_im,
                           y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                           tw1r,tw1i,tw2r,tw2i,K);
    radix5_wfta_bwd_avx512(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                           z0r,z0i,z1r,z1i,z2r,z2i,z3r,z3i,z4r,z4i,
                           tw1r,tw1i,tw2r,tw2i,K);
    const double *orig[5][2]={{a_re,a_im},{b_re,b_im},{c_re,c_im},{d_re,d_im},{e_re,e_im}};
    double *out[5][2]={{z0r,z0i},{z1r,z1i},{z2r,z2i},{z3r,z3i},{z4r,z4i}};
    for (int leg = 0; leg < 5; leg++)
        for (int i = 0; i < K; i++) {
            char t[32]; sprintf(t,"leg%d[%d]_re",leg,i); check(t,out[leg][0][i]/5.0,orig[leg][0][i]);
            sprintf(t,"leg%d[%d]_im",leg,i); check(t,out[leg][1][i]/5.0,orig[leg][1][i]);
        }
}

/* ── Test 5: Masked tail K=11 (8 + 3 masked) ── */
static void test_masked_tail(void) {
    printf("Test 5: Masked tail K=11 (8+3)\n");
    int K = 11;
    double a_re[11],a_im[11],b_re[11],b_im[11],c_re[11],c_im[11];
    double d_re[11],d_im[11],e_re[11],e_im[11];
    double y0r[11],y0i[11],y1r[11],y1i[11],y2r[11],y2i[11];
    double y3r[11],y3i[11],y4r[11],y4i[11];
    double z0r[11],z0i[11],z1r[11],z1i[11],z2r[11],z2i[11];
    double z3r[11],z3i[11],z4r[11],z4i[11];

    srand(314);
    for (int i = 0; i < K; i++) {
        a_re[i]=(rand()/(double)RAND_MAX-0.5)*10; a_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        b_re[i]=(rand()/(double)RAND_MAX-0.5)*10; b_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        c_re[i]=(rand()/(double)RAND_MAX-0.5)*10; c_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        d_re[i]=(rand()/(double)RAND_MAX-0.5)*10; d_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        e_re[i]=(rand()/(double)RAND_MAX-0.5)*10; e_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
    }
    radix5_wfta_fwd_avx512_N1(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,e_re,e_im,
                              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,K);
    radix5_wfta_bwd_avx512_N1(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                              z0r,z0i,z1r,z1i,z2r,z2i,z3r,z3i,z4r,z4i,K);
    const double *orig[5][2]={{a_re,a_im},{b_re,b_im},{c_re,c_im},{d_re,d_im},{e_re,e_im}};
    double *out[5][2]={{z0r,z0i},{z1r,z1i},{z2r,z2i},{z3r,z3i},{z4r,z4i}};
    for (int leg = 0; leg < 5; leg++)
        for (int i = 0; i < K; i++) {
            char t[32]; sprintf(t,"leg%d[%d]_re",leg,i); check(t,out[leg][0][i]/5.0,orig[leg][0][i]);
            sprintf(t,"leg%d[%d]_im",leg,i); check(t,out[leg][1][i]/5.0,orig[leg][1][i]);
        }
}

/* ── Test 6: Twiddled fwd vs manual twiddled DFT-5 (K=8) ── */
static void test_twiddled_vs_manual(void) {
    printf("Test 6: Twiddled fwd vs manual twiddled DFT-5 (K=8)\n");
    int K = 8;
    double a_re[8],a_im[8],b_re[8],b_im[8],c_re[8],c_im[8];
    double d_re[8],d_im[8],e_re[8],e_im[8];
    double tw1r[8],tw1i[8],tw2r[8],tw2i[8];
    double y0r[8],y0i[8],y1r[8],y1i[8],y2r[8],y2i[8],y3r[8],y3i[8],y4r[8],y4i[8];

    srand(9876);
    int N = 5 * K;
    for (int i = 0; i < K; i++) {
        a_re[i]=(rand()/(double)RAND_MAX-0.5)*10; a_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        b_re[i]=(rand()/(double)RAND_MAX-0.5)*10; b_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        c_re[i]=(rand()/(double)RAND_MAX-0.5)*10; c_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        d_re[i]=(rand()/(double)RAND_MAX-0.5)*10; d_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        e_re[i]=(rand()/(double)RAND_MAX-0.5)*10; e_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        double angle = -2.0*M_PI*i/N;
        tw1r[i]=cos(angle);   tw1i[i]=sin(angle);
        tw2r[i]=cos(2*angle); tw2i[i]=sin(2*angle);
    }
    radix5_wfta_fwd_avx512(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,e_re,e_im,
                           y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                           tw1r,tw1i,tw2r,tw2i,K);

    /* Reference: manual twiddle → naive DFT-5 per k */
    for (int i = 0; i < K; i++) {
        double angle = -2.0*M_PI*i/N;
        double w1r=cos(angle),w1i=sin(angle),w2r=cos(2*angle),w2i=sin(2*angle);
        double w3r=w1r*w2r-w1i*w2i, w3i=w1r*w2i+w1i*w2r;
        double w4r=w2r*w2r-w2i*w2i, w4i=2*w2r*w2i;
        double xr[5],xi[5];
        xr[0]=a_re[i]; xi[0]=a_im[i];
        xr[1]=b_re[i]*w1r-b_im[i]*w1i; xi[1]=b_re[i]*w1i+b_im[i]*w1r;
        xr[2]=c_re[i]*w2r-c_im[i]*w2i; xi[2]=c_re[i]*w2i+c_im[i]*w2r;
        xr[3]=d_re[i]*w3r-d_im[i]*w3i; xi[3]=d_re[i]*w3i+d_im[i]*w3r;
        xr[4]=e_re[i]*w4r-e_im[i]*w4i; xi[4]=e_re[i]*w4i+e_im[i]*w4r;
        double rr[5],ri[5];
        naive_dft5_fwd(xr,xi,rr,ri);
        double *yrs[5]={y0r,y1r,y2r,y3r,y4r}, *yis[5]={y0i,y1i,y2i,y3i,y4i};
        for (int m = 0; m < 5; m++) {
            char t[32]; sprintf(t,"k%d_y%d_re",i,m); check(t,yrs[m][i],rr[m]);
            sprintf(t,"k%d_y%d_im",i,m); check(t,yis[m][i],ri[m]);
        }
    }
}

/* ── Test 7: U=2 stress K=32 N1 roundtrip ── */
static int test7_u2_stress_k32(void) {
    printf("Test 7: U=2 stress K=32 fwd->bwd roundtrip\n");
    int K = 32, fails = 0;
    double a_re[32],a_im[32],b_re[32],b_im[32],c_re[32],c_im[32];
    double d_re[32],d_im[32],e_re[32],e_im[32];
    double y0r[32],y0i[32],y1r[32],y1i[32],y2r[32],y2i[32];
    double y3r[32],y3i[32],y4r[32],y4i[32];
    double z0r[32],z0i[32],z1r[32],z1i[32],z2r[32],z2i[32];
    double z3r[32],z3i[32],z4r[32],z4i[32];
    srand(12345);
    for (int i = 0; i < K; i++) {
        a_re[i]=(rand()/(double)RAND_MAX-0.5)*10; a_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        b_re[i]=(rand()/(double)RAND_MAX-0.5)*10; b_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        c_re[i]=(rand()/(double)RAND_MAX-0.5)*10; c_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        d_re[i]=(rand()/(double)RAND_MAX-0.5)*10; d_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        e_re[i]=(rand()/(double)RAND_MAX-0.5)*10; e_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
    }
    radix5_wfta_fwd_avx512_N1(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,e_re,e_im,
                              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,K);
    radix5_wfta_bwd_avx512_N1(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                              z0r,z0i,z1r,z1i,z2r,z2i,z3r,z3i,z4r,z4i,K);
    const double *orig[5][2]={{a_re,a_im},{b_re,b_im},{c_re,c_im},{d_re,d_im},{e_re,e_im}};
    double *out[5][2]={{z0r,z0i},{z1r,z1i},{z2r,z2i},{z3r,z3i},{z4r,z4i}};
    for (int leg=0;leg<5;leg++) for (int i=0;i<K;i++) {
        if (fabs(out[leg][0][i]/5.0-orig[leg][0][i])>1e-10) fails++;
        if (fabs(out[leg][1][i]/5.0-orig[leg][1][i])>1e-10) fails++;
    }
    return fails;
}

/* ── Test 8: U=2 twiddled K=64 roundtrip ── */
static int test8_u2_twiddled_k64(void) {
    printf("Test 8: U=2 twiddled K=64 fwd->bwd roundtrip\n");
    int K = 64, fails = 0;
    double a_re[64],a_im[64],b_re[64],b_im[64],c_re[64],c_im[64];
    double d_re[64],d_im[64],e_re[64],e_im[64];
    double tw1r[64],tw1i[64],tw2r[64],tw2i[64];
    double y0r[64],y0i[64],y1r[64],y1i[64],y2r[64],y2i[64];
    double y3r[64],y3i[64],y4r[64],y4i[64];
    double z0r[64],z0i[64],z1r[64],z1i[64],z2r[64],z2i[64];
    double z3r[64],z3i[64],z4r[64],z4i[64];
    srand(54321);
    int N = 5*K;
    for (int i = 0; i < K; i++) {
        a_re[i]=(rand()/(double)RAND_MAX-0.5)*10; a_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        b_re[i]=(rand()/(double)RAND_MAX-0.5)*10; b_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        c_re[i]=(rand()/(double)RAND_MAX-0.5)*10; c_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        d_re[i]=(rand()/(double)RAND_MAX-0.5)*10; d_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        e_re[i]=(rand()/(double)RAND_MAX-0.5)*10; e_im[i]=(rand()/(double)RAND_MAX-0.5)*10;
        double angle=-2.0*M_PI*i/N;
        tw1r[i]=cos(angle);   tw1i[i]=sin(angle);
        tw2r[i]=cos(2*angle); tw2i[i]=sin(2*angle);
    }
    radix5_wfta_fwd_avx512(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,e_re,e_im,
                           y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                           tw1r,tw1i,tw2r,tw2i,K);
    radix5_wfta_bwd_avx512(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                           z0r,z0i,z1r,z1i,z2r,z2i,z3r,z3i,z4r,z4i,
                           tw1r,tw1i,tw2r,tw2i,K);
    const double *orig[5][2]={{a_re,a_im},{b_re,b_im},{c_re,c_im},{d_re,d_im},{e_re,e_im}};
    double *out[5][2]={{z0r,z0i},{z1r,z1i},{z2r,z2i},{z3r,z3i},{z4r,z4i}};
    for (int leg=0;leg<5;leg++) for (int i=0;i<K;i++) {
        if (fabs(out[leg][0][i]/5.0-orig[leg][0][i])>1e-10) fails++;
        if (fabs(out[leg][1][i]/5.0-orig[leg][1][i])>1e-10) fails++;
    }
    return fails;
}

/* ── Test 9: Masked tail K=1 (edge case — single element, all masked) ── */
static int test9_masked_k1(void) {
    printf("Test 9: K=1 (all-masked, no SIMD loop)\n");
    int K = 1, fails = 0;
    double ar[1]={3.0},ai[1]={-1.5},br[1]={1.0},bi[1]={2.0};
    double cr[1]={-2.0},ci[1]={0.5},dr[1]={0.5},di[1]={-3.0};
    double er[1]={4.0},ei[1]={1.0};
    double y0r[1],y0i[1],y1r[1],y1i[1],y2r[1],y2i[1],y3r[1],y3i[1],y4r[1],y4i[1];
    double z0r[1],z0i[1],z1r[1],z1i[1],z2r[1],z2i[1],z3r[1],z3i[1],z4r[1],z4i[1];

    radix5_wfta_fwd_avx512_N1(ar,ai,br,bi,cr,ci,dr,di,er,ei,
                              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,K);
    radix5_wfta_bwd_avx512_N1(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                              z0r,z0i,z1r,z1i,z2r,z2i,z3r,z3i,z4r,z4i,K);
    const double *orig[5][2]={{ar,ai},{br,bi},{cr,ci},{dr,di},{er,ei}};
    double *out[5][2]={{z0r,z0i},{z1r,z1i},{z2r,z2i},{z3r,z3i},{z4r,z4i}};
    for (int leg=0;leg<5;leg++) {
        if (fabs(out[leg][0][0]/5.0-orig[leg][0][0])>1e-10) fails++;
        if (fabs(out[leg][1][0]/5.0-orig[leg][1][0])>1e-10) fails++;
    }
    return fails;
}

/* ── Test 10: K=3 (all masked, no SIMD loop) ── */
static int test10_masked_k3(void) {
    printf("Test 10: K=3 (all-masked) fwd->bwd roundtrip\n");
    int K = 3, fails = 0;
    double ar[3],ai[3],br[3],bi[3],cr[3],ci[3],dr[3],di[3],er[3],ei[3];
    double y0r[3],y0i[3],y1r[3],y1i[3],y2r[3],y2i[3],y3r[3],y3i[3],y4r[3],y4i[3];
    double z0r[3],z0i[3],z1r[3],z1i[3],z2r[3],z2i[3],z3r[3],z3i[3],z4r[3],z4i[3];
    srand(271828);
    for (int i=0;i<K;i++) {
        ar[i]=(rand()/(double)RAND_MAX-0.5)*10; ai[i]=(rand()/(double)RAND_MAX-0.5)*10;
        br[i]=(rand()/(double)RAND_MAX-0.5)*10; bi[i]=(rand()/(double)RAND_MAX-0.5)*10;
        cr[i]=(rand()/(double)RAND_MAX-0.5)*10; ci[i]=(rand()/(double)RAND_MAX-0.5)*10;
        dr[i]=(rand()/(double)RAND_MAX-0.5)*10; di[i]=(rand()/(double)RAND_MAX-0.5)*10;
        er[i]=(rand()/(double)RAND_MAX-0.5)*10; ei[i]=(rand()/(double)RAND_MAX-0.5)*10;
    }
    radix5_wfta_fwd_avx512_N1(ar,ai,br,bi,cr,ci,dr,di,er,ei,
                              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,K);
    radix5_wfta_bwd_avx512_N1(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                              z0r,z0i,z1r,z1i,z2r,z2i,z3r,z3i,z4r,z4i,K);
    const double *orig[5][2]={{ar,ai},{br,bi},{cr,ci},{dr,di},{er,ei}};
    double *out[5][2]={{z0r,z0i},{z1r,z1i},{z2r,z2i},{z3r,z3i},{z4r,z4i}};
    for (int leg=0;leg<5;leg++) for (int i=0;i<K;i++) {
        if (fabs(out[leg][0][i]/5.0-orig[leg][0][i])>1e-10) fails++;
        if (fabs(out[leg][1][i]/5.0-orig[leg][1][i])>1e-10) fails++;
    }
    return fails;
}

/* ── Test 11: K=17 (1 U=2 + masked tail of 1) twiddled roundtrip ── */
static int test11_u2_plus_masked(void) {
    printf("Test 11: K=17 (1 U=2 + masked 1) twiddled roundtrip\n");
    int K = 17, fails = 0;
    double ar[17],ai[17],br[17],bi[17],cr[17],ci[17],dr[17],di[17],er[17],ei[17];
    double tw1r[17],tw1i[17],tw2r[17],tw2i[17];
    double y0r[17],y0i[17],y1r[17],y1i[17],y2r[17],y2i[17];
    double y3r[17],y3i[17],y4r[17],y4i[17];
    double z0r[17],z0i[17],z1r[17],z1i[17],z2r[17],z2i[17];
    double z3r[17],z3i[17],z4r[17],z4i[17];
    srand(161803);
    int N = 5*K;
    for (int i=0;i<K;i++) {
        ar[i]=(rand()/(double)RAND_MAX-0.5)*10; ai[i]=(rand()/(double)RAND_MAX-0.5)*10;
        br[i]=(rand()/(double)RAND_MAX-0.5)*10; bi[i]=(rand()/(double)RAND_MAX-0.5)*10;
        cr[i]=(rand()/(double)RAND_MAX-0.5)*10; ci[i]=(rand()/(double)RAND_MAX-0.5)*10;
        dr[i]=(rand()/(double)RAND_MAX-0.5)*10; di[i]=(rand()/(double)RAND_MAX-0.5)*10;
        er[i]=(rand()/(double)RAND_MAX-0.5)*10; ei[i]=(rand()/(double)RAND_MAX-0.5)*10;
        double angle=-2.0*M_PI*i/N;
        tw1r[i]=cos(angle);   tw1i[i]=sin(angle);
        tw2r[i]=cos(2*angle); tw2i[i]=sin(2*angle);
    }
    radix5_wfta_fwd_avx512(ar,ai,br,bi,cr,ci,dr,di,er,ei,
                           y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                           tw1r,tw1i,tw2r,tw2i,K);
    radix5_wfta_bwd_avx512(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                           z0r,z0i,z1r,z1i,z2r,z2i,z3r,z3i,z4r,z4i,
                           tw1r,tw1i,tw2r,tw2i,K);
    const double *orig[5][2]={{ar,ai},{br,bi},{cr,ci},{dr,di},{er,ei}};
    double *out[5][2]={{z0r,z0i},{z1r,z1i},{z2r,z2i},{z3r,z3i},{z4r,z4i}};
    for (int leg=0;leg<5;leg++) for (int i=0;i<K;i++) {
        if (fabs(out[leg][0][i]/5.0-orig[leg][0][i])>1e-10) fails++;
        if (fabs(out[leg][1][i]/5.0-orig[leg][1][i])>1e-10) fails++;
    }
    return fails;
}

int main(void)
{
    printf("=== Radix-5 WFTA AVX-512 U=2 Tests ===\n\n");

    test_n1_fwd_correctness();           /* 10 checks */
    test_n1_roundtrip();                 /* 80 checks */
    test_twiddled_fwd_correctness();     /* 10 checks */
    test_twiddled_roundtrip();           /* 160 checks */
    test_masked_tail();                  /* 110 checks */
    test_twiddled_vs_manual();           /* 80 checks */

    int f7 = test7_u2_stress_k32();     /* 320 checks */
    int f8 = test8_u2_twiddled_k64();   /* 640 checks */
    int f9 = test9_masked_k1();         /* 10 checks */
    int f10 = test10_masked_k3();       /* 30 checks */
    int f11 = test11_u2_plus_masked();  /* 170 checks */

    int extra_checks = (32+64+1+3+17)*5*2;
    int extra_fail = f7+f8+f9+f10+f11;
    g_pass += (extra_checks - extra_fail);
    g_fail += extra_fail;

    printf("\n--- Results: %d passed, %d failed ---\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
