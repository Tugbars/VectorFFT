/**
 * bench_t1_variants.c -- Compare t1 codelet variants at each K/M
 *
 * For each N = R*M, tests the 1-level CT executor with different t1 codelets:
 *   - t1_dit (flat twiddles, standard)
 *   - t1_dit_log3 (derived twiddles, fewer loads)
 *   - genfft DAG (if available)
 *
 * Shows which t1 variant wins at each M, revealing the crossover point
 * where log3/ladder beats flat.
 *
 * Also tests 2-level with different inner t1 variants.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

#include "fft_radix4_avx2.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_ct_n1.h"
#include "fft_radix16_avx2_ct_t1_dit.h"
#include "fft_radix32_avx2_ct_n1.h"
#include "fft_radix32_avx2_ct_t1_dit.h"
#include "r64_unified_avx2.h"

typedef void (*n1_ovs_fn)(const double*, const double*, double*, double*,
                          size_t, size_t, size_t, size_t);
typedef void (*t1_fn)(double*, double*, const double*, const double*,
                      size_t, size_t);

static void init_tw(double *W_re, double *W_im, size_t R, size_t me) {
    size_t N = R * me;
    for (size_t n = 1; n < R; n++)
        for (size_t m = 0; m < me; m++) {
            double a = -2.0 * M_PI * (double)(n * m) / (double)N;
            W_re[(n-1)*me + m] = cos(a);
            W_im[(n-1)*me + m] = sin(a);
        }
}

static void ct_1level(n1_ovs_fn n1, t1_fn t1,
    const double *ir, const double *ii, double *or_, double *oi,
    const double *W_re, const double *W_im, size_t R, size_t M)
{
    n1(ir, ii, or_, oi, R, 1, R, M);
    t1(or_, oi, W_re, W_im, M, M);
}

/* ================================================================
 * FFTW reference
 * ================================================================ */

static double bench_fftw(size_t N, int reps) {
    double *ri=fftw_malloc(N*8),*ii=fftw_malloc(N*8);
    double *ro=fftw_malloc(N*8),*io=fftw_malloc(N*8);
    for(size_t i=0;i<N;i++){ri[i]=(double)rand()/RAND_MAX;ii[i]=(double)rand()/RAND_MAX;}
    fftw_iodim d={.n=(int)N,.is=1,.os=1};
    fftw_iodim h={.n=1,.is=(int)N,.os=(int)N};
    fftw_plan p=fftw_plan_guru_split_dft(1,&d,1,&h,ri,ii,ro,io,FFTW_MEASURE);
    if(!p){fftw_free(ri);fftw_free(ii);fftw_free(ro);fftw_free(io);return -1;}
    for(int i=0;i<20;i++)fftw_execute(p);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();
        for(int i=0;i<reps;i++)fftw_execute_split_dft(p,ri,ii,ro,io);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    fftw_destroy_plan(p);fftw_free(ri);fftw_free(ii);fftw_free(ro);fftw_free(io);
    return best;
}

/* ================================================================
 * Test one R×M with a specific t1 variant
 * ================================================================ */

static double test_combo(size_t N, size_t R, size_t M,
                         n1_ovs_fn n1, t1_fn t1,
                         const double *in_re, const double *in_im,
                         const double *fftw_re, const double *fftw_im,
                         int reps)
{
    double *W_re = (double*)aligned_alloc(32, (R-1)*M*8);
    double *W_im = (double*)aligned_alloc(32, (R-1)*M*8);
    double *out_re = (double*)aligned_alloc(32, N*8);
    double *out_im = (double*)aligned_alloc(32, N*8);
    init_tw(W_re, W_im, R, M);

    ct_1level(n1, t1, in_re, in_im, out_re, out_im, W_re, W_im, R, M);
    double max_err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fabs(out_re[i] - fftw_re[i]) + fabs(out_im[i] - fftw_im[i]);
        if (e > max_err) max_err = e;
    }

    double ns = -1;
    if (max_err < 1e-10) {
        for (int i = 0; i < 20; i++)
            ct_1level(n1, t1, in_re, in_im, out_re, out_im, W_re, W_im, R, M);
        double best = 1e18;
        for (int t = 0; t < 7; t++) {
            double t0 = now_ns();
            for (int i = 0; i < reps; i++)
                ct_1level(n1, t1, in_re, in_im, out_re, out_im, W_re, W_im, R, M);
            double elapsed = (now_ns() - t0) / reps;
            if (elapsed < best) best = elapsed;
        }
        ns = best;
    }

    aligned_free(W_re); aligned_free(W_im);
    aligned_free(out_re); aligned_free(out_im);
    return ns;
}

/* ================================================================
 * Variant registry per radix
 * ================================================================ */

typedef struct {
    const char *name;
    t1_fn fn;
} t1_variant;

/* R=8 t1 variants */
static const t1_variant R8_T1[] = {
    {"flat",  (t1_fn)radix8_t1_dit_fwd_avx2},
    {"dif",   (t1_fn)radix8_t1_dif_fwd_avx2},
    {NULL, NULL}
};

/* R=16 t1 variants */
static const t1_variant R16_T1[] = {
    {"flat",  (t1_fn)radix16_t1_dit_fwd_avx2},
    {NULL, NULL}
};

/* R=32 t1 variants */
static const t1_variant R32_T1[] = {
    {"flat",  (t1_fn)radix32_t1_dit_fwd_avx2},
    {NULL, NULL}
};

/* R=64 t1 variants */
static const t1_variant R64_T1[] = {
    {"flat",  (t1_fn)radix64_t1_dit_fwd_avx2},
    {"log3",  (t1_fn)radix64_t1_dit_log3_fwd_avx2},
    {NULL, NULL}
};

typedef struct {
    size_t R;
    n1_ovs_fn n1;
    const t1_variant *t1_variants;
} radix_entry;

static const radix_entry RADIXES[] = {
    { 4,  (n1_ovs_fn)radix4_n1_ovs_fwd_avx2,  NULL},  /* R=4 has no t1 log3 */
    { 8,  (n1_ovs_fn)radix8_n1_ovs_fwd_avx2,  R8_T1},
    {16,  (n1_ovs_fn)radix16_n1_ovs_fwd_avx2, R16_T1},
    {32,  (n1_ovs_fn)radix32_n1_ovs_fwd_avx2, R32_T1},
    {64,  (n1_ovs_fn)radix64_n1_ovs_fwd_avx2, R64_T1},
    { 0, NULL, NULL}
};

static const radix_entry *find_radix(size_t R) {
    for (const radix_entry *r = RADIXES; r->R; r++)
        if (r->R == R) return r;
    return NULL;
}

/* ================================================================
 * Test all t1 variants for one N = R × M
 * ================================================================ */

static void test_N(size_t N) {
    int reps = (int)(2e6 / (N+1));
    if (reps < 200) reps = 200;
    if (reps > 200000) reps = 200000;

    double *in_re = (double*)aligned_alloc(32, N*8);
    double *in_im = (double*)aligned_alloc(32, N*8);
    srand(42);
    for (size_t i = 0; i < N; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    double *fre = fftw_malloc(N*8), *fim = fftw_malloc(N*8);
    double *fro = fftw_malloc(N*8), *fio = fftw_malloc(N*8);
    memcpy(fre, in_re, N*8); memcpy(fim, in_im, N*8);
    fftw_iodim d = {.n=(int)N, .is=1, .os=1};
    fftw_iodim h = {.n=1, .is=(int)N, .os=(int)N};
    fftw_plan fp = fftw_plan_guru_split_dft(1, &d, 1, &h, fre, fim, fro, fio, FFTW_ESTIMATE);
    fftw_execute(fp);
    double fftw_ns = bench_fftw(N, reps);

    printf("\n  N=%-6zu  FFTW=%.0f ns\n", N, fftw_ns);

    /* For each valid R*M=N with R<=M, test all t1 variants */
    for (const radix_entry *cr = RADIXES; cr->R; cr++) {
        size_t R = cr->R;
        if (N % R != 0) continue;
        size_t M = N / R;
        if (M < 4 || R > M) continue;

        /* Need child n1_ovs for M-point DFT */
        const radix_entry *cm = find_radix(M);
        if (!cm) continue;

        /* Twiddle table size for this R×M */
        size_t tw_bytes = (R-1) * M * 16;  /* re + im */
        double tw_kb = tw_bytes / 1024.0;

        printf("    %2zux%-3zu (tw=%.0fKB): ", R, M, tw_kb);
        fflush(stdout);

        /* Test with each available t1 variant for radix R */
        if (cr->t1_variants) {
            for (const t1_variant *v = cr->t1_variants; v->name; v++) {
                double ns = test_combo(N, R, M, cm->n1, v->fn,
                                       in_re, in_im, fro, fio, reps);
                if (ns > 0)
                    printf(" %s=%.0f(%.2fx)", v->name, ns, fftw_ns/ns);
                else
                    printf(" %s=FAIL", v->name);
                fflush(stdout);
            }
        } else {
            /* R=4 has only basic t1 */
            double ns = test_combo(N, R, M, cm->n1,
                                   (t1_fn)radix4_t1_dit_fwd_avx2,
                                   in_re, in_im, fro, fio, reps);
            if (ns > 0)
                printf(" flat=%.0f(%.2fx)", ns, fftw_ns/ns);
            else
                printf(" flat=FAIL");
        }
        printf("\n");
    }

    fftw_destroy_plan(fp);
    fftw_free(fre); fftw_free(fim); fftw_free(fro); fftw_free(fio);
    aligned_free(in_re); aligned_free(in_im);
}

/* ================================================================ */

int main(void) {
    printf("================================================================\n");
    printf("  t1 Variant Comparison: flat vs log3 vs genfft\n");
    printf("  Per R*M factorization, shows which t1 wins at each M\n");
    printf("  L1=48KB. Twiddle table = (R-1)*M*16 bytes\n");
    printf("================================================================\n");
    fflush(stdout);

    size_t sizes[] = {
        256, 512, 1024, 2048, 4096,
        /* Large N: only 2-level factorizations work, but we test
         * the 1-level R×M codelets at the M values that appear
         * in those factorizations. The key question: does log3 t1
         * beat flat t1 when the twiddle table exceeds L1? */
        8192, 16384, 32768,
        0
    };

    for (size_t *p = sizes; *p; p++) {
        test_N(*p);
        fflush(stdout);
    }

    printf("\n");
    return 0;
}
