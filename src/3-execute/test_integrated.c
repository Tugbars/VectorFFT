/*
 * test_integrated.c — VectorFFT planner with optimized codelets vs FFTW
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <fftw3.h>

/* Include planner */
#include "vfft_planner.h"

/* Include all genfft codelets */
#include "fft_radix11_genfft.h"
#include "fft_radix13_genfft.h"
#include "fft_radix17_genfft.h"
#include "fft_radix19_genfft.h"
#include "fft_radix23_genfft.h"

/* Wire them up */
#include "vfft_register_codelets.h"

/* ═══ Timing ═══ */
static double get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static double *aa64(size_t n) {
    double *p = (double *)aligned_alloc(64, n * sizeof(double));
    memset(p, 0, n * sizeof(double));
    return p;
}

static void fill_rand(double *p, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        p[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

/* ═══ Correctness test ═══ */
static void naive_dft(const double *ir, const double *ii,
                       double *nr, double *ni, size_t N) {
    for (size_t m = 0; m < N; m++) {
        double sr = 0, si = 0;
        for (size_t n = 0; n < N; n++) {
            double a = -2.0 * M_PI * m * n / (double)N;
            sr += ir[n]*cos(a) - ii[n]*sin(a);
            si += ir[n]*sin(a) + ii[n]*cos(a);
        }
        nr[m] = sr; ni[m] = si;
    }
}

static int test_correctness(size_t N, const vfft_codelet_registry *reg) {
    double *ir = aa64(N), *ii_ = aa64(N);
    double *gr = aa64(N), *gi = aa64(N);
    double *nr = aa64(N), *ni = aa64(N);
    fill_rand(ir, N, 1000+(unsigned)N);
    fill_rand(ii_, N, 2000+(unsigned)N);

    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) { printf("  N=%-6zu PLAN FAIL\n", N); return 0; }

    vfft_execute_fwd(plan, ir, ii_, gr, gi);
    naive_dft(ir, ii_, nr, ni, N);

    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
        if (e > err) err = e;
        double m = fmax(fabs(nr[i]), fabs(ni[i]));
        if (m > mag) mag = m;
    }
    double rel = mag > 0 ? err/mag : err;
    double tol = 1e-12 * (1.0 + log2((double)N));
    int pass = rel < tol;
    printf("  N=%-6zu  %zu stg  rel=%.2e  %s\n",
           N, plan->nstages, rel, pass ? "PASS" : "FAIL");

    vfft_plan_destroy(plan);
    free(ir); free(ii_); free(gr); free(gi); free(nr); free(ni);
    return pass;
}

/* ═══ Benchmark ═══ */
static void bench(size_t N, const vfft_codelet_registry *reg,
                   int warm, int trials)
{
    double *ir = aa64(N), *ii_ = aa64(N);
    double *or_ = aa64(N), *oi = aa64(N);
    fill_rand(ir, N, 9000+(unsigned)N);
    fill_rand(ii_, N, 9500+(unsigned)N);

    /* FFTW */
    fftw_complex *fin = fftw_alloc_complex(N);
    fftw_complex *fout = fftw_alloc_complex(N);
    for (size_t i = 0; i < N; i++) { fin[i][0] = ir[i]; fin[i][1] = ii_[i]; }
    fftw_plan fp = fftw_plan_dft_1d((int)N, fin, fout, FFTW_FORWARD, FFTW_MEASURE);
    for (int i = 0; i < warm; i++) fftw_execute(fp);
    double ns_fw = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns(); fftw_execute(fp);
        double dt = get_ns() - t0; if (dt < ns_fw) ns_fw = dt;
    }

    /* VectorFFT */
    vfft_plan *plan = vfft_plan_create(N, reg);
    for (int i = 0; i < warm; i++)
        vfft_execute_fwd(plan, ir, ii_, or_, oi);
    double ns_vf = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        vfft_execute_fwd(plan, ir, ii_, or_, oi);
        double dt = get_ns() - t0; if (dt < ns_vf) ns_vf = dt;
    }

    printf("  N=%-6zu  %zu stg  FW=%7.0f  VF=%7.0f  ratio=%5.2fx\n",
           N, plan->nstages, ns_fw, ns_vf, ns_fw / ns_vf);

    vfft_plan_destroy(plan);
    fftw_destroy_plan(fp); fftw_free(fin); fftw_free(fout);
    free(ir); free(ii_); free(or_); free(oi);
}

int main(void) {
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  VectorFFT Integrated — optimized codelets + planner vs FFTW\n");
    printf("════════════════════════════════════════════════════════════════\n\n");

    vfft_codelet_registry reg;
    vfft_register_all(&reg);
    vfft_print_registry(&reg);

    int p = 0, t = 0;

    /* ── Correctness with optimized codelets ── */
    printf("\n── Correctness (optimized primes + naive composites) ──\n");

    /* Single-stage primes (genfft optimized) */
    size_t primes[] = {11, 13, 17, 19, 23};
    for (int i = 0; i < 5; i++) { t++; p += test_correctness(primes[i], &reg); }

    /* Multi-stage: prime × composite */
    size_t mixed[] = {
        11*8, 11*64, 13*16, 13*128, 17*8, 17*64, 17*128,
        19*16, 19*32, 23*8, 23*16,
        11*13, 11*17, 13*19, 17*23,
        2*3*5*7*11, 3*5*7*13, 2*3*17*8,
    };
    for (int i = 0; i < (int)(sizeof(mixed)/sizeof(mixed[0])); i++) {
        t++; p += test_correctness(mixed[i], &reg);
    }

    /* Powers of 2 (naive codelets, multi-stage) */
    size_t pow2[] = {256, 512, 1024, 2048};
    for (int i = 0; i < 4; i++) { t++; p += test_correctness(pow2[i], &reg); }

    /* Roundtrip */
    printf("\n── Roundtrips ──\n");
    size_t rt_Ns[] = {11*64, 17*128, 2*3*5*7*11, 1024};
    for (int i = 0; i < 4; i++) {
        size_t N = rt_Ns[i];
        double *ir=aa64(N),*ii_=aa64(N),*fr=aa64(N),*fi=aa64(N),*br=aa64(N),*bi=aa64(N);
        fill_rand(ir,N,3000+(unsigned)N); fill_rand(ii_,N,4000+(unsigned)N);
        vfft_plan *plan = vfft_plan_create(N, &reg);
        vfft_execute_fwd(plan,ir,ii_,fr,fi);
        vfft_execute_bwd(plan,fr,fi,br,bi);
        double err=0,mag=0;
        for(size_t j=0;j<N;j++){br[j]/=(double)N;bi[j]/=(double)N;
            double e=fmax(fabs(ir[j]-br[j]),fabs(ii_[j]-bi[j]));if(e>err)err=e;
            double m=fmax(fabs(ir[j]),fabs(ii_[j]));if(m>mag)mag=m;}
        double rel=mag>0?err/mag:err;int ok=rel<1e-10;t++;p+=ok;
        printf("  N=%-6zu  rt rel=%.2e  %s\n",N,rel,ok?"PASS":"FAIL");
        vfft_plan_destroy(plan);
        free(ir);free(ii_);free(fr);free(fi);free(br);free(bi);
    }

    printf("\n  %d/%d %s\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    if (p != t) return 1;

    /* ── Benchmarks ── */
    printf("\n── Benchmark: VectorFFT (optimized primes) vs FFTW ──\n");
    printf("  Primes use genfft AVX-512. Composites still naive.\n");
    printf("  ratio > 1 = VectorFFT faster\n\n");

    printf("  --- Single primes (1 stage, N1 codelet only) ---\n");
    for (int i = 0; i < 5; i++)
        bench(primes[i], &reg, 500, 5000);

    printf("\n  --- Prime × power-of-2 (2 stages) ---\n");
    bench(11*8,   &reg, 500, 3000);
    bench(11*64,  &reg, 200, 2000);
    bench(13*16,  &reg, 500, 3000);
    bench(13*128, &reg, 200, 2000);
    bench(17*8,   &reg, 500, 3000);
    bench(17*64,  &reg, 200, 2000);
    bench(17*128, &reg, 100, 1000);
    bench(19*32,  &reg, 200, 2000);
    bench(23*16,  &reg, 500, 3000);

    printf("\n  --- Multi-factor (3-5 stages) ---\n");
    bench(2*3*5*7,    &reg, 200, 2000);
    bench(2*3*5*7*11, &reg, 100, 1000);
    bench(3*5*7*13,   &reg, 100, 1000);
    bench(2*3*17*8,   &reg, 200, 2000);

    printf("\n  --- Powers of 2 (naive codelets, baseline) ---\n");
    bench(256,  &reg, 500, 3000);
    bench(512,  &reg, 500, 3000);
    bench(1024, &reg, 200, 2000);
    bench(2048, &reg, 100, 1000);

    printf("\n════════════════════════════════════════════════════════════════\n");
    fftw_cleanup();
    return 0;
}
