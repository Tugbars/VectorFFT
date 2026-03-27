/**
 * bench_r64.c — R=64 AVX2: our 8x8 CT codelets vs FFTW SIMD (stride-1)
 *
 * Build:
 *   gcc -O3 -march=native -mavx2 -mfma -o bench_r64 bench_r64.c \
 *       -I<hdr_dir> -I<fftw>/include -L<fftw>/lib -lfftw3 -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"
#include "r64_unified_avx2.h"
typedef void (*tw_fn)(const double*, const double*, double*, double*,
                      const double*, const double*, size_t);
typedef void (*notw_fn)(const double*, const double*, double*, double*, size_t);

static double bench_tw(tw_fn fn, const double *ir, const double *ii,
    double *or_, double *oi, const double *twr, const double *twi,
    size_t K, int reps) {
    for (int i = 0; i < 20; i++) fn(ir, ii, or_, oi, twr, twi, K);
    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) fn(ir, ii, or_, oi, twr, twi, K);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}
static double bench_nf(notw_fn fn, const double *ir, const double *ii,
    double *or_, double *oi, size_t K, int reps) {
    for (int i = 0; i < 20; i++) fn(ir, ii, or_, oi, K);
    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) fn(ir, ii, or_, oi, K);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}
static double bench_fftw(size_t K, int stride_k, int reps) {
    size_t N = 64 * K;
    double *ri = fftw_malloc(N*8), *ii = fftw_malloc(N*8);
    double *ro = fftw_malloc(N*8), *io = fftw_malloc(N*8);
    for (size_t i = 0; i < N; i++) { ri[i] = (double)rand()/RAND_MAX; ii[i] = (double)rand()/RAND_MAX; }
    fftw_iodim dim, howm;
    if (stride_k) { dim = (fftw_iodim){.n=64,.is=(int)K,.os=(int)K}; howm = (fftw_iodim){.n=(int)K,.is=1,.os=1}; }
    else { dim = (fftw_iodim){.n=64,.is=1,.os=1}; howm = (fftw_iodim){.n=(int)K,.is=64,.os=64}; }
    fftw_plan p = fftw_plan_guru_split_dft(1, &dim, 1, &howm, ri, ii, ro, io, FFTW_MEASURE);
    if (!p) { fftw_free(ri); fftw_free(ii); fftw_free(ro); fftw_free(io); return -1; }
    for (int i = 0; i < 20; i++) fftw_execute(p);
    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) fftw_execute_split_dft(p, ri, ii, ro, io);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    fftw_destroy_plan(p); fftw_free(ri); fftw_free(ii); fftw_free(ro); fftw_free(io);
    return best;
}
static void init_tw(double *twr, double *twi, size_t K) {
    for (int n = 1; n < 64; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * n * k / (64.0 * K);
            twr[(n-1)*K+k] = cos(a); twi[(n-1)*K+k] = sin(a);
        }
}

int main(void) {
    srand(42);
    size_t Ks[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int nK = sizeof(Ks) / sizeof(Ks[0]);

    printf("== R=64 AVX2: CT(8x8) notw / dit / dif vs FFTW SIMD ==\n\n");
    printf("%-5s %-7s %8s %8s  %14s %14s %14s\n",
        "K", "N", "FFTW_scl", "FFTW_sim", "CT_notw", "CT_dit_tw", "CT_dif_tw");

    for (int ki = 0; ki < nK; ki++) {
        size_t K = Ks[ki], N = 64 * K;
        double *ir  = aligned_alloc(32, N*8), *ii  = aligned_alloc(32, N*8);
        double *or_ = aligned_alloc(32, N*8), *oi  = aligned_alloc(32, N*8);
        double *twr = aligned_alloc(32, 63*K*8), *twi = aligned_alloc(32, 63*K*8);
        for (size_t i = 0; i < N; i++) { ir[i] = (double)rand()/RAND_MAX-.5; ii[i] = (double)rand()/RAND_MAX-.5; }
        init_tw(twr, twi, K);
        int reps = (int)(2e6/(N+1)); if(reps<200) reps=200; if(reps>2000000) reps=2000000;

        double fscl  = bench_fftw(K, 1, reps);
        double fsimd = bench_fftw(K, 0, reps);
        double ns_notw = bench_nf((notw_fn)radix64_n1_dit_kernel_fwd_avx2, ir, ii, or_, oi, K, reps);
        double ns_dit  = bench_tw((tw_fn)radix64_tw_flat_dit_kernel_fwd_avx2, ir, ii, or_, oi, twr, twi, K, reps);
        double ns_dif  = bench_tw((tw_fn)radix64_tw_flat_dif_kernel_fwd_avx2, ir, ii, or_, oi, twr, twi, K, reps);

        printf("%-5zu %-7zu %8.1f %8.1f  %5.0f(%5.1fx) %5.0f(%5.1fx) %5.0f(%5.1fx)\n",
            K, N, fscl, fsimd,
            ns_notw, fsimd/ns_notw, ns_dit, fsimd/ns_dit, ns_dif, fsimd/ns_dif);

        aligned_free(ir); aligned_free(ii); aligned_free(or_); aligned_free(oi); aligned_free(twr); aligned_free(twi);
    }
    printf("\nFormat: ns(ratio vs FFTW SIMD) -- higher = we win\n");
    return 0;
}