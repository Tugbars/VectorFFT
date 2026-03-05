/*
 * bench_radix16_avx512_tw.c — Radix-16 twiddled AVX-512 vs FFTW
 */
#include "vfft_test_utils.h"
#include <fftw3.h>

#undef  R16L_LD
#undef  R16L_ST
#define R16L_LD(p)   _mm512_load_pd(p)
#define R16L_ST(p,v) _mm512_store_pd((p),(v))
#include "fft_radix16_avx512_tw.h"

/* Also include N1 for comparison */
#define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,fma")))
#define RESTRICT __restrict__
#define ALIGNAS_64 __attribute__((aligned(64)))
#include "fft_radix16_avx512_n1_gen.h"

static void build_flat_tw(size_t K, int dir, double *twr, double *twi) {
    size_t NN = 16 * K;
    for (int n = 1; n < 16; n++)
        for (size_t k = 0; k < K; k++) {
            double a = 2.0 * M_PI * (double)n * (double)k / (double)NN;
            twr[(n-1)*K + k] = cos(a);
            twi[(n-1)*K + k] = dir * sin(a);
        }
}

static void naive_tw_dft16_fwd(size_t K, size_t k,
    const double *ir, const double *ii,
    const double *twr, const double *twi,
    double *or_, double *oi) {
    double xr[16], xi[16];
    xr[0] = ir[k]; xi[0] = ii[k];
    for (int n = 1; n < 16; n++) {
        double wr = twr[(n-1)*K + k], wi = twi[(n-1)*K + k];
        xr[n] = ir[n*K + k]*wr - ii[n*K + k]*wi;
        xi[n] = ir[n*K + k]*wi + ii[n*K + k]*wr;
    }
    for (int m = 0; m < 16; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < 16; n++) {
            double a = -2.0 * M_PI * m * n / 16.0;
            sr += xr[n]*cos(a) - xi[n]*sin(a);
            si += xr[n]*sin(a) + xi[n]*cos(a);
        }
        or_[m*K + k] = sr; oi[m*K + k] = si;
    }
}

static int test_tw_fwd(size_t K) {
    size_t N = 16 * K;
    double *ir = aa64(N), *ii_ = aa64(N), *or_ = aa64(N), *oi = aa64(N);
    double *nr = aa64(N), *ni = aa64(N);
    double *ftwr = aa64(15*K), *ftwi = aa64(15*K);
    fill_rand(ir, N, 1000+(unsigned)K);
    fill_rand(ii_, N, 2000+(unsigned)K);
    build_flat_tw(K, -1, ftwr, ftwi);

    radix16_tw_flat_dit_kernel_fwd_avx512(ir, ii_, or_, oi, ftwr, ftwi, K);
    for (size_t k = 0; k < K; k++)
        naive_tw_dft16_fwd(K, k, ir, ii_, ftwr, ftwi, nr, ni);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(or_[i]-nr[i]), fabs(oi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr, N), max_abs(ni, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-13;
    printf("  tw fwd K=%-5zu rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(or_);r32_aligned_free(oi);r32_aligned_free(nr);r32_aligned_free(ni);r32_aligned_free(ftwr);r32_aligned_free(ftwi);
    return pass;
}

static int test_roundtrip(size_t K) {
    size_t N = 16 * K;
    double *ir = aa64(N), *ii_ = aa64(N);
    double *fr = aa64(N), *fi = aa64(N), *br = aa64(N), *bi = aa64(N);
    double *ftwr = aa64(15*K), *ftwi = aa64(15*K);
    double *btwr = aa64(15*K), *btwi = aa64(15*K);
    fill_rand(ir, N, 3000+(unsigned)K);
    fill_rand(ii_, N, 4000+(unsigned)K);
    build_flat_tw(K, -1, ftwr, ftwi);
    build_flat_tw(K, +1, btwr, btwi);

    radix16_tw_flat_dit_kernel_fwd_avx512(ir, ii_, fr, fi, ftwr, ftwi, K);
    radix16_tw_flat_dit_kernel_bwd_avx512(fr, fi, br, bi, btwr, btwi, K);

    /* Twiddles cancel in roundtrip: bwd(fwd(x)) = 16*x when tw are conjugates */
    double err = 0;
    for (size_t i = 0; i < N; i++) {
        br[i] /= 16.0; bi[i] /= 16.0;
        double e = fmax(fabs(ir[i]-br[i]), fabs(ii_[i]-bi[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(ir, N), max_abs(ii_, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-14;
    printf("  tw rt  K=%-5zu rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(fr);r32_aligned_free(fi);r32_aligned_free(br);r32_aligned_free(bi);
    r32_aligned_free(ftwr);r32_aligned_free(ftwi);r32_aligned_free(btwr);r32_aligned_free(btwi);
    return pass;
}

__attribute__((target("avx512f,avx512dq,fma")))
static void run_bench(size_t K, int warm, int trials) {
    size_t N = 16 * K;
    double *ir = aa64(N), *ii_ = aa64(N), *or_ = aa64(N), *oi = aa64(N);
    double *ftwr = aa64(15*K), *ftwi = aa64(15*K);
    fill_rand(ir, N, 9000+(unsigned)K);
    fill_rand(ii_, N, 9500+(unsigned)K);
    build_flat_tw(K, -1, ftwr, ftwi);

    /* FFTW */
    fftw_complex *fin = fftw_alloc_complex(N), *fout = fftw_alloc_complex(N);
    for (size_t k = 0; k < K; k++)
        for (int n = 0; n < 16; n++) {
            fin[k*16+n][0] = ir[n*K+k]; fin[k*16+n][1] = ii_[n*K+k];
        }
    int na[1] = {16};
    fftw_plan plan = fftw_plan_many_dft(1, na, (int)K,
        fin, NULL, 1, 16, fout, NULL, 1, 16, FFTW_FORWARD, FFTW_MEASURE);
    for (int i = 0; i < warm; i++) fftw_execute(plan);
    double bfw = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns(); fftw_execute(plan); double dt = get_ns()-t0;
        if (dt < bfw) bfw = dt;
    }

    /* N1 (no twiddles, baseline) */
    for (int i = 0; i < warm; i++)
        radix16_n1_dit_kernel_fwd_avx512(ir, ii_, or_, oi, K);
    double ns_n1 = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        radix16_n1_dit_kernel_fwd_avx512(ir, ii_, or_, oi, K);
        double dt = get_ns()-t0; if (dt < ns_n1) ns_n1 = dt;
    }

    /* Twiddled flat */
    for (int i = 0; i < warm; i++)
        radix16_tw_flat_dit_kernel_fwd_avx512(ir, ii_, or_, oi, ftwr, ftwi, K);
    double ns_tw = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        radix16_tw_flat_dit_kernel_fwd_avx512(ir, ii_, or_, oi, ftwr, ftwi, K);
        double dt = get_ns()-t0; if (dt < ns_tw) ns_tw = dt;
    }

    printf("  K=%-5zu  FFTW=%7.0f  N1=%7.0f(%5.2fx)  TW=%7.0f(%5.2fx)  TW/N1=%.2fx\n",
           K, bfw, ns_n1, bfw/ns_n1, ns_tw, bfw/ns_tw, ns_tw/ns_n1);

    fftw_destroy_plan(plan); fftw_free(fin); fftw_free(fout);
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(or_);r32_aligned_free(oi);r32_aligned_free(ftwr);r32_aligned_free(ftwi);
}

int main(void) {
    R32_REQUIRE_AVX512();
    printf("====================================================================\n");
    printf("  DFT-16 AVX-512: flat twiddled + N1 vs FFTW\n");
    printf("  15 ext twiddle loads/k-step, 32 L1 spill ops, 3 internal W₁₆\n");
    printf("====================================================================\n\n");

    int p = 0, t = 0;
    printf("-- Twiddled forward vs naive --\n");
    { size_t Ks[] = {8,16,32,64,128,256,512};
      for (int i = 0; i < 7; i++) { t++; p += test_tw_fwd(Ks[i]); } }

    printf("\n======================================\n");
    printf("  %d/%d passed  %s\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    printf("======================================\n");
    if (p != t) return 1;

    printf("\n-- BENCHMARK (ns, forward) --\n");
    printf("  FFTW=batched DFT-16 (no tw), N1=ours no tw, TW=ours+tw\n\n");
    run_bench(8,    500, 5000);
    run_bench(16,   500, 5000);
    run_bench(32,   500, 3000);
    run_bench(64,   500, 3000);
    run_bench(128,  200, 2000);
    run_bench(256,  200, 2000);
    run_bench(512,  100, 1000);
    run_bench(1024, 100, 1000);

    fftw_cleanup();
    return 0;
}