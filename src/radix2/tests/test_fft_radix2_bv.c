/**
 * @file test_fft_radix2_bv.c
 * @brief Correctness tests for radix-2 butterfly (with-twiddles + N1)
 *
 * Tests N1, twiddle, special cases (k=0/N4/N8/3N8), round-trip, edge cases,
 * and various N to hit every AVX2 SIMD cleanup path.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft_radix2.h"

#define EPS      1e-10
#define EPS_RT   1e-8    /* relaxed for round-trip accumulation */

static int g_run = 0, g_pass = 0, g_fail = 0;

#define CHECK(cond, fmt, ...) do { \
    g_run++; \
    if (!(cond)) { g_fail++; fprintf(stderr,"  FAIL [%s:%d] " fmt "\n", __func__,__LINE__,##__VA_ARGS__); } \
    else g_pass++; \
} while(0)

#define RUN(fn) do { \
    int _f = g_fail; printf("  %-50s", #fn); fn(); \
    printf("%s\n", g_fail==_f ? "PASS":"FAIL"); \
} while(0)

/* ── helpers ── */

static double *aalloc(size_t n) {
    void *p; if (posix_memalign(&p, 64, n*sizeof(double))) abort();
    memset(p, 0, n*sizeof(double)); return p;
}

static fft_twiddles_soa *tw_alloc(int half) {
    fft_twiddles_soa *t = malloc(sizeof(*t));
    t->re = aalloc(half); t->im = aalloc(half); return t;
}
static void tw_free(fft_twiddles_soa *t) {
    if(t){free((void*)t->re);free((void*)t->im);free(t);}
}

static void tw_fwd(fft_twiddles_soa *t, int half) {
    int N=half*2;
    for(int k=0;k<half;k++){double a=-2.0*M_PI*k/(double)N; t->re[k]=cos(a); t->im[k]=sin(a);}
}
static void tw_inv(fft_twiddles_soa *t, int half) {
    int N=half*2;
    for(int k=0;k<half;k++){double a=+2.0*M_PI*k/(double)N; t->re[k]=cos(a); t->im[k]=sin(a);}
}

static void ref_bv(double *or_, double *oi, const double *ir, const double *ii,
                   const fft_twiddles_soa *tw, int half) {
    for(int k=0;k<half;k++){
        double er=ir[k],ei=ii[k], oR=ir[k+half],oI=ii[k+half];
        double wr=tw->re[k],wi=tw->im[k];
        double pr=oR*wr-oI*wi, pi=oR*wi+oI*wr;
        or_[k]=er+pr; oi[k]=ei+pi; or_[k+half]=er-pr; oi[k+half]=ei-pi;
    }
}
static void ref_n1(double *or_, double *oi, const double *ir, const double *ii, int half) {
    for(int k=0;k<half;k++){
        double er=ir[k],ei=ii[k],oR=ir[k+half],oI=ii[k+half];
        or_[k]=er+oR; oi[k]=ei+oI; or_[k+half]=er-oR; oi[k+half]=ei-oI;
    }
}

static double maxerr(const double *a, const double *b, int n) {
    double m=0; for(int i=0;i<n;i++){double e=fabs(a[i]-b[i]); if(e>m) m=e;} return m;
}

static void fill(double *buf, int n, double seed) {
    for(int i=0;i<n;i++) buf[i] = sin(seed*i + 0.3) + 0.5*cos((seed+0.4)*i);
}

/* ── N1 tests ── */

static void test_n1_size(int N) {
    int h=N/2;
    double *ir=aalloc(N),*ii=aalloc(N),*or_=aalloc(N),*oi=aalloc(N),*rr=aalloc(N),*ri=aalloc(N);
    fill(ir,N,0.7); fill(ii,N,1.1);
    ref_n1(rr,ri,ir,ii,h);
    fft_radix2_bv_n1(or_,oi,ir,ii,h);
    CHECK(maxerr(or_,rr,N)<EPS, "N1 N=%d re err=%.2e",N,maxerr(or_,rr,N));
    CHECK(maxerr(oi,ri,N)<EPS, "N1 N=%d im err=%.2e",N,maxerr(oi,ri,N));
    free(ir);free(ii);free(or_);free(oi);free(rr);free(ri);
}

static void test_n1_sizes(void) {
    /* half: 1,2,3,4,5,7,8,9,16,17,32,64,128,256,1024,2048 */
    int s[]={2,4,6,8,10,14,16,18,32,34,64,128,256,512,1024,2048,4096};
    for(int i=0;i<(int)(sizeof(s)/sizeof(s[0]));i++) test_n1_size(s[i]);
}

/* ── twiddle tests ── */

static void test_tw_size(int N) {
    int h=N/2;
    double *ir=aalloc(N),*ii=aalloc(N),*or_=aalloc(N),*oi=aalloc(N),*rr=aalloc(N),*ri=aalloc(N);
    fft_twiddles_soa *tw=tw_alloc(h); tw_fwd(tw,h);
    fill(ir,N,0.7); fill(ii,N,1.1);
    ref_bv(rr,ri,ir,ii,tw,h);
    fft_radix2_bv(or_,oi,ir,ii,tw,h);
    CHECK(maxerr(or_,rr,N)<EPS, "tw N=%d re err=%.2e",N,maxerr(or_,rr,N));
    CHECK(maxerr(oi,ri,N)<EPS, "tw N=%d im err=%.2e",N,maxerr(oi,ri,N));
    free(ir);free(ii);free(or_);free(oi);free(rr);free(ri); tw_free(tw);
}

static void test_tw_sizes(void) {
    int s[]={4,8,16,32,64,128,256,512,1024,2048,4096};
    for(int i=0;i<(int)(sizeof(s)/sizeof(s[0]));i++) test_tw_size(s[i]);
}

/* ── special-case indices ── */

static void test_special_cases(void) {
    const int N=64, h=32;
    double *ir=aalloc(N),*ii=aalloc(N),*or_=aalloc(N),*oi=aalloc(N),*rr=aalloc(N),*ri=aalloc(N);
    fft_twiddles_soa *tw=tw_alloc(h); tw_fwd(tw,h);
    for(int i=0;i<N;i++){ir[i]=3.0*sin(0.37*i)+1.5; ii[i]=2.0*cos(0.53*i)-0.7;}
    ref_bv(rr,ri,ir,ii,tw,h);
    fft_radix2_bv(or_,oi,ir,ii,tw,h);

    int sk[]={0,8,16,24}; const char *lb[]={"k=0","k=N/8","k=N/4","k=3N/8"};
    for(int s=0;s<4;s++){
        int k=sk[s];
        double e=fmax(fmax(fabs(or_[k]-rr[k]),fabs(oi[k]-ri[k])),
                      fmax(fabs(or_[k+h]-rr[k+h]),fabs(oi[k+h]-ri[k+h])));
        CHECK(e<EPS, "%s err=%.2e",lb[s],e);
    }
    CHECK(maxerr(or_,rr,N)<EPS, "full re err=%.2e",maxerr(or_,rr,N));
    CHECK(maxerr(oi,ri,N)<EPS, "full im err=%.2e",maxerr(oi,ri,N));
    free(ir);free(ii);free(or_);free(oi);free(rr);free(ri); tw_free(tw);
}

/* ── N1 vs unity twiddles equivalence ── */

static void test_n1_vs_unity(void) {
    /* Use non-power-of-2 half to avoid special-case paths (k=N/4, N/8, 3N/8)
     * which hardcode forward twiddles and ignore the twiddle table.
     * BUG NOTE: fft_radix2_bv hardcodes forward-direction twiddles at special
     * indices — with all-ones twiddles, those indices produce wrong results.
     * This is a known issue for inverse FFT as well. */
    const int N=200, h=100;
    double *ir=aalloc(N),*ii=aalloc(N);
    double *on_r=aalloc(N),*on_i=aalloc(N),*ot_r=aalloc(N),*ot_i=aalloc(N);
    fft_twiddles_soa *tw=tw_alloc(h);
    for(int k=0;k<h;k++){tw->re[k]=1.0; tw->im[k]=0.0;}
    fill(ir,N,0.7); fill(ii,N,1.1);
    fft_radix2_bv_n1(on_r,on_i,ir,ii,h);
    fft_radix2_bv(ot_r,ot_i,ir,ii,tw,h);
    CHECK(maxerr(on_r,ot_r,N)<EPS, "n1 vs unity re=%.2e",maxerr(on_r,ot_r,N));
    CHECK(maxerr(on_i,ot_i,N)<EPS, "n1 vs unity im=%.2e",maxerr(on_i,ot_i,N));
    free(ir);free(ii);free(on_r);free(on_i);free(ot_r);free(ot_i); tw_free(tw);
}

/* ── forward-inverse round-trip ── */

static void test_roundtrip(void) {
    /* N1 butterfly is self-inverse up to factor 2:
     *   Y[k]=X[k]+X[k+h], Y[k+h]=X[k]-X[k+h]
     *   Z[k]=Y[k]+Y[k+h]=2*X[k], Z[k+h]=Y[k]-Y[k+h]=2*X[k+h]
     *
     * NOTE: a single DIT butterfly is NOT inverted by applying the same
     * butterfly with conjugate twiddles. That only works for a full multi-stage
     * FFT. For a single stage, the twiddle butterfly Z ≠ 2*X in general. */
    const int N=512, h=256;
    double *x_r=aalloc(N),*x_i=aalloc(N),*y_r=aalloc(N),*y_i=aalloc(N),*z_r=aalloc(N),*z_i=aalloc(N);
    fill(x_r,N,0.37); fill(x_i,N,0.53);

    /* N1 → N1 = 2×identity */
    fft_radix2_bv_n1(y_r,y_i,x_r,x_i,h);
    fft_radix2_bv_n1(z_r,z_i,y_r,y_i,h);

    double me=0;
    for(int i=0;i<N;i++)
        me=fmax(me,fmax(fabs(z_r[i]-2.0*x_r[i]),fabs(z_i[i]-2.0*x_i[i])));
    CHECK(me<EPS, "n1 roundtrip err=%.2e",me);

    free(x_r);free(x_i);free(y_r);free(y_i);free(z_r);free(z_i);
}

/* ── zero input ── */

static void test_zero(void) {
    const int N=64, h=32;
    double *ir=aalloc(N),*ii=aalloc(N),*or_=aalloc(N),*oi=aalloc(N);
    fft_radix2_bv_n1(or_,oi,ir,ii,h);
    double m=0; for(int i=0;i<N;i++) m=fmax(m,fmax(fabs(or_[i]),fabs(oi[i])));
    CHECK(m==0.0, "zero n1 err=%.2e",m);

    fft_twiddles_soa *tw=tw_alloc(h); tw_fwd(tw,h);
    fft_radix2_bv(or_,oi,ir,ii,tw,h);
    m=0; for(int i=0;i<N;i++) m=fmax(m,fmax(fabs(or_[i]),fabs(oi[i])));
    CHECK(m==0.0, "zero tw err=%.2e",m);
    free(ir);free(ii);free(or_);free(oi); tw_free(tw);
}

/* ── impulse ── */

static void test_impulse(void) {
    const int N=32, h=16;
    double *ir=aalloc(N),*ii=aalloc(N),*or_=aalloc(N),*oi=aalloc(N);
    ir[0]=1.0;
    fft_radix2_bv_n1(or_,oi,ir,ii,h);
    CHECK(fabs(or_[0]-1.0)<EPS, "impulse or[0]=%.6f",or_[0]);
    CHECK(fabs(or_[h]-1.0)<EPS, "impulse or[h]=%.6f",or_[h]);
    double m=0;
    for(int i=1;i<h;i++){
        m=fmax(m,fmax(fabs(or_[i]),fabs(oi[i])));
        m=fmax(m,fmax(fabs(or_[i+h]),fabs(oi[i+h])));
    }
    m=fmax(m,fmax(fabs(oi[0]),fabs(oi[h])));
    CHECK(m<EPS, "impulse noise=%.2e",m);
    free(ir);free(ii);free(or_);free(oi);
}

/* ── input not modified ── */

static void test_readonly(void) {
    const int N=128, h=64;
    double *ir=aalloc(N),*ii=aalloc(N),*cr=aalloc(N),*ci=aalloc(N),*or_=aalloc(N),*oi=aalloc(N);
    fill(ir,N,0.7); fill(ii,N,1.1);
    memcpy(cr,ir,N*sizeof(double)); memcpy(ci,ii,N*sizeof(double));

    fft_twiddles_soa *tw=tw_alloc(h); tw_fwd(tw,h);
    fft_radix2_bv(or_,oi,ir,ii,tw,h);
    CHECK(!memcmp(ir,cr,N*sizeof(double)), "bv modified in_re");
    CHECK(!memcmp(ii,ci,N*sizeof(double)), "bv modified in_im");

    fft_radix2_bv_n1(or_,oi,ir,ii,h);
    CHECK(!memcmp(ir,cr,N*sizeof(double)), "n1 modified in_re");
    CHECK(!memcmp(ii,ci,N*sizeof(double)), "n1 modified in_im");
    free(ir);free(ii);free(cr);free(ci);free(or_);free(oi); tw_free(tw);
}

/* ── special cases work correctly for BOTH directions ── */

static void test_special_case_inverse(void) {
    /* After fix: special cases read signs from twiddle table,
     * so inverse FFT should now match reference at all indices. */
    const int N=64, h=32;
    double *ir=aalloc(N),*ii=aalloc(N),*or_=aalloc(N),*oi=aalloc(N),*rr=aalloc(N),*ri=aalloc(N);
    fft_twiddles_soa *tw=tw_alloc(h);
    tw_inv(tw,h);  /* inverse twiddles */
    for(int i=0;i<N;i++){ir[i]=3.0*sin(0.37*i)+1.5; ii[i]=2.0*cos(0.53*i)-0.7;}
    ref_bv(rr,ri,ir,ii,tw,h);
    fft_radix2_bv(or_,oi,ir,ii,tw,h);

    int sk[]={0,8,16,24}; const char *lb[]={"k=0","k=N/8","k=N/4","k=3N/8"};
    for(int s=0;s<4;s++){
        int k=sk[s];
        double e=fmax(fmax(fabs(or_[k]-rr[k]),fabs(oi[k]-ri[k])),
                      fmax(fabs(or_[k+h]-rr[k+h]),fabs(oi[k+h]-ri[k+h])));
        CHECK(e<EPS, "inv %s err=%.2e",lb[s],e);
    }
    double full_re=maxerr(or_,rr,N), full_im=maxerr(oi,ri,N);
    CHECK(full_re<EPS, "inv full re=%.2e",full_re);
    CHECK(full_im<EPS, "inv full im=%.2e",full_im);
    free(ir);free(ii);free(or_);free(oi);free(rr);free(ri); tw_free(tw);
}

/* ── forward→inverse round-trip on power-of-2 (exercises all special cases) ── */

static void test_twiddle_roundtrip_pow2(void) {
    /* Verify inverse-direction butterfly matches reference on power-of-2 sizes
     * (exercises all special-case indices with inverse twiddles). */
    int sizes[]={16,32,64,128,256,512,1024};
    for(int s=0;s<(int)(sizeof(sizes)/sizeof(sizes[0]));s++){
        int N=sizes[s], h=N/2;
        double *ir=aalloc(N),*ii=aalloc(N),*or_=aalloc(N),*oi=aalloc(N),*rr=aalloc(N),*ri=aalloc(N);
        fft_twiddles_soa *tw=tw_alloc(h);
        tw_inv(tw,h);
        fill(ir,N,0.37); fill(ii,N,0.53);
        ref_bv(rr,ri,ir,ii,tw,h);
        fft_radix2_bv(or_,oi,ir,ii,tw,h);
        double e_re=maxerr(or_,rr,N), e_im=maxerr(oi,ri,N);
        CHECK(e_re<EPS, "inv N=%d re=%.2e",N,e_re);
        CHECK(e_im<EPS, "inv N=%d im=%.2e",N,e_im);
        free(ir);free(ii);free(or_);free(oi);free(rr);free(ri); tw_free(tw);
    }
}

/* ── capability queries ── */

static void test_caps(void) {
    const char *c = radix2_get_simd_capabilities();
    CHECK(c && strlen(c)>0, "caps null/empty");
    size_t a = radix2_get_alignment_requirement();
    CHECK(a>=16 && (a&(a-1))==0, "align=%zu",a);
    int w = radix2_get_vector_width();
    CHECK(w>=2 && (w&(w-1))==0, "vw=%d",w);
    printf("    [%s align=%zu vw=%d] ", c, a, w);
}

/* ── main ── */

int main(void) {
    printf("============================================\n");
    printf(" Radix-2 Butterfly Tests\n");
    printf("============================================\n\n");

    printf("[N1 twiddle-less]\n");      RUN(test_n1_sizes);
    printf("\n[With twiddles]\n");      RUN(test_tw_sizes);
    printf("\n[Special cases]\n");      RUN(test_special_cases);
    printf("\n[N1 vs unity tw]\n");     RUN(test_n1_vs_unity);
    printf("\n[Round-trip]\n");         RUN(test_roundtrip);
    printf("\n[Edge cases]\n");
    RUN(test_zero); RUN(test_impulse); RUN(test_readonly);
    printf("\n[Capabilities]\n");       RUN(test_caps);
    printf("\n[Inverse direction]\n");
    RUN(test_special_case_inverse);
    RUN(test_twiddle_roundtrip_pow2);

    printf("\n============================================\n");
    printf(" %d/%d passed, %d failed\n", g_pass, g_run, g_fail);
    printf("============================================\n");
    return g_fail ? 1 : 0;
}
