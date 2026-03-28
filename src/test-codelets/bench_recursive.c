/**
 * bench_recursive.c — N=32768: Six-step recursive executor vs flat vs FFTW
 *
 * Correct CT decomposition: n = n2*N1 + n1, k = k1*N2 + k2
 *
 * Six-step for DFT(N), N = N1 × N2:
 *   Input naturally N2×N1 matrix (n = n2*N1+n1).
 *   1. Transpose (N2×N1 → N1×N2)  — make columns contiguous
 *   2. N1 row DFTs of size N2      — inner DFT, each row fits L1
 *   3. Twiddle W_N^(n1·k2)         — sequential
 *   4. Transpose (N1×N2 → N2×N1)  — prepare for outer DFT
 *   5. N2 row DFTs of size N1      — outer DFT
 *   6. Transpose (N2×N1 → N1×N2)  — standard output order
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <fftw3.h>

#include "fft_n1_k1.h"
#include "fft_n1_k1_simd.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_notw.h"
#include "fft_radix16_avx2_dit_tw.h"

#define N_FFT 32768

static inline double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static void init_tw(double *twr, double *twi, size_t R, size_t K) {
    for (size_t n = 1; n < R; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * (double)(n * k) / (double)(R * K);
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

static void r16_k1_wrap(const double *ir, const double *ii,
                         double *or_, double *oi, size_t K) {
    (void)K; dft16_k1_fwd_avx2(ir, ii, or_, oi);
}

/* ═══════════════════════════════════════════════════════════════
 * Leaf DFT-256 = 16 × 16
 * ═══════════════════════════════════════════════════════════════ */
static double *tw256_re, *tw256_im;
static size_t *perm256;

static void leaf_init_256(void) {
    tw256_re = aligned_alloc(32, 15*16*8);
    tw256_im = aligned_alloc(32, 15*16*8);
    init_tw(tw256_re, tw256_im, 16, 16);
    size_t rx[] = {16, 16};
    perm256 = build_perm(rx, 2, 256);
}

static void leaf_dft256(const double *in_re, const double *in_im,
                         double *out_re, double *out_im,
                         double *buf_re, double *buf_im)
{
    for (size_t i = 0; i < 256; i++) {
        buf_re[i] = in_re[perm256[i]];
        buf_im[i] = in_im[perm256[i]];
    }
    for (size_t g = 0; g < 16; g++) {
        r16_k1_wrap(buf_re+g*16, buf_im+g*16, out_re+g*16, out_im+g*16, 1);
    }
    radix16_tw_flat_dit_kernel_fwd_avx2(out_re, out_im, buf_re, buf_im,
                                         tw256_re, tw256_im, 16);
    memcpy(out_re, buf_re, 256*8);
    memcpy(out_im, buf_im, 256*8);
}

/* ═══════════════════════════════════════════════════════════════
 * Leaf DFT-128 = 16 × 8
 * ═══════════════════════════════════════════════════════════════ */
static double *tw128_re, *tw128_im;
static size_t *perm128;

static void leaf_init_128(void) {
    tw128_re = aligned_alloc(32, 7*16*8);
    tw128_im = aligned_alloc(32, 7*16*8);
    init_tw(tw128_re, tw128_im, 8, 16);
    size_t rx[] = {16, 8};
    perm128 = build_perm(rx, 2, 128);
}

static void leaf_dft128(const double *in_re, const double *in_im,
                         double *out_re, double *out_im,
                         double *buf_re, double *buf_im)
{
    for (size_t i = 0; i < 128; i++) {
        buf_re[i] = in_re[perm128[i]];
        buf_im[i] = in_im[perm128[i]];
    }
    for (size_t g = 0; g < 8; g++) {
        r16_k1_wrap(buf_re+g*16, buf_im+g*16, out_re+g*16, out_im+g*16, 1);
    }
    radix8_tw_dit_kernel_fwd_avx2(out_re, out_im, buf_re, buf_im,
                                    tw128_re, tw128_im, 16);
    memcpy(out_re, buf_re, 128*8);
    memcpy(out_im, buf_im, 128*8);
}

/* ═══════════════════════════════════════════════════════════════
 * Tiled transpose
 * ═══════════════════════════════════════════════════════════════ */
#define TILE 32

static void do_transpose(const double *__restrict__ src,
                         double *__restrict__ dst,
                         size_t rows, size_t cols)
{
    for (size_t i0 = 0; i0 < rows; i0 += TILE) {
        size_t i1 = i0 + TILE; if (i1 > rows) i1 = rows;
        for (size_t j0 = 0; j0 < cols; j0 += TILE) {
            size_t j1 = j0 + TILE; if (j1 > cols) j1 = cols;
            for (size_t i = i0; i < i1; i++)
                for (size_t j = j0; j < j1; j++)
                    dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Twiddle multiply (AVX2)
 * ═══════════════════════════════════════════════════════════════ */
__attribute__((target("avx2,fma")))
static void twiddle_apply(double *__restrict__ re, double *__restrict__ im,
                           const double *__restrict__ twr, const double *__restrict__ twi,
                           size_t N)
{
    size_t i = 0;
    for (; i + 4 <= N; i += 4) {
        __m256d xr = _mm256_load_pd(&re[i]);
        __m256d xi = _mm256_load_pd(&im[i]);
        __m256d wr = _mm256_load_pd(&twr[i]);
        __m256d wi = _mm256_load_pd(&twi[i]);
        _mm256_store_pd(&re[i], _mm256_fmsub_pd(xr, wr, _mm256_mul_pd(xi, wi)));
        _mm256_store_pd(&im[i], _mm256_fmadd_pd(xr, wi, _mm256_mul_pd(xi, wr)));
    }
    for (; i < N; i++) {
        double xr=re[i],xi=im[i],wr=twr[i],wi=twi[i];
        re[i]=xr*wr-xi*wi; im[i]=xr*wi+xi*wr;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * SIX-STEP EXECUTOR
 *
 * CT decomposition: n = n2*N1+n1, k = k1*N2+k2
 * Input viewed as N2×N1 matrix.
 * ═══════════════════════════════════════════════════════════════ */

typedef void (*leaf_fn)(const double*, const double*,
                        double*, double*, double*, double*);

typedef struct {
    size_t N1, N2;
    leaf_fn dft_n2;    /* DFT of size N2 (step 2: inner, N1 calls) */
    leaf_fn dft_n1;    /* DFT of size N1 (step 5: outer, N2 calls) */
    double *tw_re, *tw_im;         /* tw[n1*N2+k2] = W_N^(n1*k2) */
    double *scr_re, *scr_im;       /* scratch N */
    double *scr2_re, *scr2_im;     /* scratch2 N (for step 6) */
    double *lbuf_re, *lbuf_im;     /* leaf scratch */
} sixstep_plan;

static sixstep_plan *sixstep_create(size_t N1, size_t N2,
                                      leaf_fn dft_n2, leaf_fn dft_n1)
{
    size_t N = N1 * N2;
    sixstep_plan *p = calloc(1, sizeof(*p));
    p->N1 = N1; p->N2 = N2;
    p->dft_n2 = dft_n2;  /* called N1 times in step 2 */
    p->dft_n1 = dft_n1;  /* called N2 times in step 5 */

    /* Twiddle: tw[n1*N2+k2] = W_N^(n1*k2), N1×N2 layout */
    p->tw_re = aligned_alloc(32, N*8);
    p->tw_im = aligned_alloc(32, N*8);
    for (size_t n1 = 0; n1 < N1; n1++)
        for (size_t k2 = 0; k2 < N2; k2++) {
            double a = -2.0*M_PI*(double)(n1*k2)/(double)N;
            p->tw_re[n1*N2+k2] = cos(a);
            p->tw_im[n1*N2+k2] = sin(a);
        }

    p->scr_re  = aligned_alloc(32, N*8);
    p->scr_im  = aligned_alloc(32, N*8);
    p->scr2_re = aligned_alloc(32, N*8);
    p->scr2_im = aligned_alloc(32, N*8);
    size_t mx = N1 > N2 ? N1 : N2;
    p->lbuf_re = aligned_alloc(32, mx*8);
    p->lbuf_im = aligned_alloc(32, mx*8);
    return p;
}

static void sixstep_execute(const sixstep_plan *p,
                             const double *in_re, const double *in_im,
                             double *out_re, double *out_im)
{
    const size_t N1 = p->N1, N2 = p->N2;

    /* Step 1: Transpose input (N2×N1) → scratch (N1×N2) */
    do_transpose(in_re, p->scr_re, N2, N1);
    do_transpose(in_im, p->scr_im, N2, N1);

    /* Step 2: N1 row DFTs of size N2.
     * scratch is N1×N2. Row n1 has N2 elements at scr[n1*N2..]. */
    for (size_t n1 = 0; n1 < N1; n1++)
        p->dft_n2(p->scr_re + n1*N2, p->scr_im + n1*N2,
                   out_re + n1*N2, out_im + n1*N2,
                   p->lbuf_re, p->lbuf_im);

    /* Step 3: Twiddle on out (N1×N2 layout).
     * out[n1*N2+k2] *= W_N^(n1*k2) */
    twiddle_apply(out_re, out_im, p->tw_re, p->tw_im, N1*N2);

    /* Step 4: Transpose out (N1×N2) → scratch (N2×N1) */
    do_transpose(out_re, p->scr_re, N1, N2);
    do_transpose(out_im, p->scr_im, N1, N2);

    /* Step 5: N2 row DFTs of size N1.
     * scratch is N2×N1. Row k2 has N1 elements at scr[k2*N1..]. */
    for (size_t k2 = 0; k2 < N2; k2++)
        p->dft_n1(p->scr_re + k2*N1, p->scr_im + k2*N1,
                   p->scr2_re + k2*N1, p->scr2_im + k2*N1,
                   p->lbuf_re, p->lbuf_im);

    /* Step 6: Transpose scr2 (N2×N1) → out (N1×N2) for standard order.
     * out[k1*N2+k2] = scr2[k2*N1+k1] = X[k1*N2+k2] */
    do_transpose(p->scr2_re, out_re, N2, N1);
    do_transpose(p->scr2_im, out_im, N2, N1);
}

static void sixstep_destroy(sixstep_plan *p) {
    free(p->tw_re); free(p->tw_im);
    free(p->scr_re); free(p->scr_im);
    free(p->scr2_re); free(p->scr2_im);
    free(p->lbuf_re); free(p->lbuf_im);
    free(p);
}

/* ═══════════════════════════════════════════════════════════════
 * FLAT EXECUTOR (16×16×8×16)
 * ═══════════════════════════════════════════════════════════════ */
static size_t *flat_perm;
static double *ftw1r,*ftw1i,*ftw2r,*ftw2i,*ftw3r,*ftw3i;

static void flat_init(void) {
    size_t rx[] = {16,16,8,16};
    flat_perm = build_perm(rx, 4, N_FFT);
    ftw1r=aligned_alloc(32,15*16*8);  ftw1i=aligned_alloc(32,15*16*8);
    ftw2r=aligned_alloc(32,7*256*8);  ftw2i=aligned_alloc(32,7*256*8);
    ftw3r=aligned_alloc(32,15*2048*8);ftw3i=aligned_alloc(32,15*2048*8);
    init_tw(ftw1r,ftw1i,16,16);
    init_tw(ftw2r,ftw2i,8,256);
    init_tw(ftw3r,ftw3i,16,2048);
}

static void exec_flat(const double *ir, const double *ii,
                      double *or_, double *oi,
                      double *a, double *ai, double *b, double *bi)
{
    for(size_t i=0;i<N_FFT;i++){a[i]=ir[flat_perm[i]];ai[i]=ii[flat_perm[i]];}
    for(size_t g=0;g<2048;g++){size_t o=g*16;
        r16_k1_wrap(a+o,ai+o,b+o,bi+o,1);}
    for(size_t g=0;g<128;g++){size_t o=g*256;
        radix16_tw_flat_dit_kernel_fwd_avx2(b+o,bi+o,a+o,ai+o,ftw1r,ftw1i,16);}
    for(size_t g=0;g<16;g++){size_t o=g*2048;
        radix8_tw_dit_kernel_fwd_avx2(a+o,ai+o,b+o,bi+o,ftw2r,ftw2i,256);}
    radix16_tw_flat_dit_kernel_fwd_avx2(b,bi,or_,oi,ftw3r,ftw3i,2048);
}

/* ═══════════════════════════════════════════════════════════════ */

static double bench_fftw(int reps) {
    double *ri=fftw_malloc(N_FFT*8),*ii_=fftw_malloc(N_FFT*8);
    double *ro=fftw_malloc(N_FFT*8),*io=fftw_malloc(N_FFT*8);
    for(size_t i=0;i<N_FFT;i++){ri[i]=(double)rand()/RAND_MAX;ii_[i]=(double)rand()/RAND_MAX;}
    fftw_iodim dim={.n=N_FFT,.is=1,.os=1};
    fftw_iodim howm={.n=1,.is=N_FFT,.os=N_FFT};
    fftw_plan p=fftw_plan_guru_split_dft(1,&dim,1,&howm,ri,ii_,ro,io,FFTW_MEASURE);
    for(int i=0;i<20;i++)fftw_execute(p);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();
        for(int i=0;i<reps;i++)fftw_execute_split_dft(p,ri,ii_,ro,io);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    fftw_destroy_plan(p);fftw_free(ri);fftw_free(ii_);fftw_free(ro);fftw_free(io);
    return best;
}

int main(void)
{
    srand(42);
    printf("================================================================\n");
    printf("  N=%d: Six-Step Recursive vs Flat vs FFTW\n", N_FFT);
    printf("================================================================\n\n");

    leaf_init_256();
    leaf_init_128();
    flat_init();

    double *in_re=aligned_alloc(32,N_FFT*8), *in_im=aligned_alloc(32,N_FFT*8);
    double *out_re=aligned_alloc(32,N_FFT*8),*out_im=aligned_alloc(32,N_FFT*8);
    double *a=aligned_alloc(32,N_FFT*8),*ai=aligned_alloc(32,N_FFT*8);
    double *b=aligned_alloc(32,N_FFT*8),*bi=aligned_alloc(32,N_FFT*8);
    for(size_t i=0;i<N_FFT;i++){in_re[i]=(double)rand()/RAND_MAX-.5;in_im[i]=(double)rand()/RAND_MAX-.5;}

    /* FFTW reference */
    double *fr=fftw_malloc(N_FFT*8),*fi=fftw_malloc(N_FFT*8);
    double *fo_r=fftw_malloc(N_FFT*8),*fo_i=fftw_malloc(N_FFT*8);
    memcpy(fr,in_re,N_FFT*8); memcpy(fi,in_im,N_FFT*8);
    fftw_iodim dim={.n=N_FFT,.is=1,.os=1};
    fftw_iodim howm={.n=1,.is=N_FFT,.os=N_FFT};
    fftw_plan fp=fftw_plan_guru_split_dft(1,&dim,1,&howm,fr,fi,fo_r,fo_i,FFTW_ESTIMATE);
    fftw_execute(fp);

    printf("Correctness vs FFTW:\n");

    /* Flat */
    exec_flat(in_re,in_im,out_re,out_im,a,ai,b,bi);
    double e=0; for(size_t i=0;i<N_FFT;i++){double d=fabs(out_re[i]-fo_r[i])+fabs(out_im[i]-fo_i[i]);if(d>e)e=d;}
    printf("  Flat:             %.2e %s\n", e, e<1e-10?"OK":"FAIL");

    /* Six-step: N1=256, N2=128
     * Step 2: 256 DFTs of size 128 (dft_n2 = leaf_dft128)
     * Step 5: 128 DFTs of size 256 (dft_n1 = leaf_dft256) */
    sixstep_plan *ss_a = sixstep_create(256, 128, leaf_dft128, leaf_dft256);
    sixstep_execute(ss_a, in_re, in_im, out_re, out_im);
    e=0; for(size_t i=0;i<N_FFT;i++){double d=fabs(out_re[i]-fo_r[i])+fabs(out_im[i]-fo_i[i]);if(d>e)e=d;}
    printf("  6-step (256×128):  %.2e %s\n", e, e<1e-10?"OK":"FAIL");
    int ok_a = e < 1e-10;

    /* Six-step: N1=128, N2=256 */
    sixstep_plan *ss_b = sixstep_create(128, 256, leaf_dft256, leaf_dft128);
    sixstep_execute(ss_b, in_re, in_im, out_re, out_im);
    e=0; for(size_t i=0;i<N_FFT;i++){double d=fabs(out_re[i]-fo_r[i])+fabs(out_im[i]-fo_i[i]);if(d>e)e=d;}
    printf("  6-step (128×256):  %.2e %s\n", e, e<1e-10?"OK":"FAIL");
    int ok_b = e < 1e-10;

    if (!ok_a && !ok_b) {
        printf("\nCORRECTNESS FAILURE — skipping perf.\n");
        return 1;
    }

    int reps = 200;
    printf("\nBenchmarking (%d reps, 7 trials, best-of)...\n\n", reps);

    double fftw_ns = bench_fftw(reps);

    /* Flat */
    for(int i=0;i<20;i++) exec_flat(in_re,in_im,out_re,out_im,a,ai,b,bi);
    double flat_best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();
        for(int i=0;i<reps;i++) exec_flat(in_re,in_im,out_re,out_im,a,ai,b,bi);
        double ns=(now_ns()-t0)/reps;if(ns<flat_best)flat_best=ns;}

    double ss_a_best=1e18, ss_b_best=1e18;

    if (ok_a) {
        for(int i=0;i<20;i++) sixstep_execute(ss_a,in_re,in_im,out_re,out_im);
        for(int t=0;t<7;t++){double t0=now_ns();
            for(int i=0;i<reps;i++) sixstep_execute(ss_a,in_re,in_im,out_re,out_im);
            double ns=(now_ns()-t0)/reps;if(ns<ss_a_best)ss_a_best=ns;}
    }

    if (ok_b) {
        for(int i=0;i<20;i++) sixstep_execute(ss_b,in_re,in_im,out_re,out_im);
        for(int t=0;t<7;t++){double t0=now_ns();
            for(int i=0;i<reps;i++) sixstep_execute(ss_b,in_re,in_im,out_re,out_im);
            double ns=(now_ns()-t0)/reps;if(ns<ss_b_best)ss_b_best=ns;}
    }

    printf("  %-28s %8.0f ns\n", "FFTW_MEASURE:", fftw_ns);
    printf("  %-28s %8.0f ns  (%.2fx vs FFTW)\n", "Flat (16x16x8x16):", flat_best, fftw_ns/flat_best);
    if (ok_a) printf("  %-28s %8.0f ns  (%.2fx vs FFTW)\n", "6-step (256x128):", ss_a_best, fftw_ns/ss_a_best);
    if (ok_b) printf("  %-28s %8.0f ns  (%.2fx vs FFTW)\n", "6-step (128x256):", ss_b_best, fftw_ns/ss_b_best);

    double best_ss = (ok_a && ss_a_best < ss_b_best) ? ss_a_best : ss_b_best;
    const char *best_name = (ok_a && ss_a_best < ss_b_best) ? "256x128" : "128x256";
    printf("\n  Best 6-step: %s\n", best_name);
    printf("  6-step vs Flat: %.2fx\n", flat_best / best_ss);
    printf("  6-step vs FFTW: %.2fx\n", fftw_ns / best_ss);

    sixstep_destroy(ss_a);
    sixstep_destroy(ss_b);
    fftw_destroy_plan(fp);
    fftw_free(fr);fftw_free(fi);fftw_free(fo_r);fftw_free(fo_i);
    return 0;
}
