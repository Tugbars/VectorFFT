/* bench_r128_cascade_attribution.c — Time each stage of Path C
 * separately to attribute where the cascade overhead lives. Tells us
 * whether Phase 5's split-pointer I/O work is worth the codegen
 * complexity.
 *
 * Stages timed independently with warmup + min-of-N:
 *   S1: pre-pack       (memcpy of N reals into M strided streams)
 *   S2: rdft × M       (M codelet calls processing K_outer batches each)
 *   S3: hermitian-pack (memcpy of rdft output → hc2c-shaped buffer with conj)
 *   S4: hc2c × 1       (single codelet call with K_lanes = R · K_outer)
 *
 * Compare against the same Path A monolithic to see what budget the
 * cascade has to work with.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>

#ifdef _WIN32
  #include <windows.h>
  #include <malloc.h>
#else
  #include <time.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N 128
#define R 16
#define M 8

__attribute__((target("avx2,fma")))
void radix128_r2c_fwd_avx2_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

__attribute__((target("avx2,fma")))
void radix16_rdft_fwd_avx2_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

__attribute__((target("avx2,fma")))
void radix8_hc2c_dit_fwd_avx2_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im, size_t K);

static double *aa(size_t n) {
#ifdef _WIN32
    void *p = _aligned_malloc(n * sizeof(double), 32);
    if (!p) exit(1);
#else
    void *p = NULL;
    if (posix_memalign(&p, 32, n * sizeof(double)) != 0) exit(1);
#endif
    memset(p, 0, n * sizeof(double));
    return (double *)p;
}

static void af(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static double now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
#endif
}

__attribute__((target("avx2,fma")))
static void prepack(const double *in_re, double *streams, size_t K) {
    for (int g = 0; g < M; g++)
        for (int n_inner = 0; n_inner < R; n_inner++) {
            const double *src = in_re + (size_t)(g + M*n_inner) * K;
            double *dst = streams + (size_t)g*R*K + (size_t)n_inner*K;
            memcpy(dst, src, K * sizeof(double));
        }
}

__attribute__((target("avx2,fma")))
static void rdft_stage(const double *streams, double *E_re, double *E_im,
                       const double *dummy, size_t K) {
    for (int g = 0; g < M; g++) {
        const double *in_base = streams + (size_t)g*R*K;
        double *out_re_g = E_re + (size_t)g*R*K;
        double *out_im_g = E_im + (size_t)g*R*K;
        radix16_rdft_fwd_avx2_gen(in_base, dummy,
                                  out_re_g, out_im_g,
                                  dummy, dummy, K);
    }
}

__attribute__((target("avx2,fma")))
static void hermpack(const double *E_re, const double *E_im,
                     double *Z_re, double *Z_im,
                     size_t K, size_t K_lanes) {
    for (int g = 0; g < M; g++) {
        for (int k_inner = 0; k_inner < R; k_inner++) {
            int load_idx = (k_inner > R/2) ? (R - k_inner) : k_inner;
            int conjugate = (k_inner > R/2);
            const double *src_re = E_re + (size_t)g*R*K + (size_t)load_idx*K;
            const double *src_im = E_im + (size_t)g*R*K + (size_t)load_idx*K;
            double *dst_re = Z_re + (size_t)g*K_lanes + (size_t)k_inner*K;
            double *dst_im = Z_im + (size_t)g*K_lanes + (size_t)k_inner*K;
            if (!conjugate) {
                memcpy(dst_re, src_re, K * sizeof(double));
                memcpy(dst_im, src_im, K * sizeof(double));
            } else {
                memcpy(dst_re, src_re, K * sizeof(double));
                __m256d neg = _mm256_set1_pd(-0.0);
                for (size_t b = 0; b < K; b += 4) {
                    __m256d v = _mm256_loadu_pd(src_im + b);
                    _mm256_storeu_pd(dst_im + b, _mm256_xor_pd(v, neg));
                }
            }
        }
    }
}

static void hc2c_stage(const double *Z_re, const double *Z_im,
                       double *X_re, double *X_im,
                       const double *tw_re, const double *tw_im,
                       size_t K_lanes) {
    radix8_hc2c_dit_fwd_avx2_gen(Z_re, Z_im, X_re, X_im,
                                 tw_re, tw_im, K_lanes);
}

static double time_fn(void (*fn)(void), int reps, int trials) {
    double best = 1e18;
    for (int w = 0; w < 30; w++) fn();
    for (int t = 0; t < trials; t++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) fn();
        double dt = (now_ns() - t0) / (double)reps;
        if (dt < best) best = dt;
    }
    return best;
}

/* Globals so each stage wrapper has no args. */
static const double *g_in_re, *g_dummy, *g_E_re, *g_E_im;
static double *g_streams, *g_E_re_rw, *g_E_im_rw;
static double *g_Z_re, *g_Z_im, *g_X_re, *g_X_im;
static const double *g_tw_re, *g_tw_im;
static size_t g_K, g_K_lanes;
static double *g_outA_re, *g_outA_im;

static void w_pathA(void) {
    radix128_r2c_fwd_avx2_gen(g_in_re, g_dummy, g_outA_re, g_outA_im,
                              g_dummy, g_dummy, g_K);
}
static void w_prepack(void) { prepack(g_in_re, g_streams, g_K); }
static void w_rdft(void)    { rdft_stage(g_streams, g_E_re_rw, g_E_im_rw,
                                         g_dummy, g_K); }
static void w_hermpack(void){ hermpack(g_E_re, g_E_im, g_Z_re, g_Z_im,
                                       g_K, g_K_lanes); }
static void w_hc2c(void)    { hc2c_stage(g_Z_re, g_Z_im, g_X_re, g_X_im,
                                         g_tw_re, g_tw_im, g_K_lanes); }
static void w_cascade(void) {
    prepack(g_in_re, g_streams, g_K);
    rdft_stage(g_streams, g_E_re_rw, g_E_im_rw, g_dummy, g_K);
    hermpack(g_E_re_rw, g_E_im_rw, g_Z_re, g_Z_im, g_K, g_K_lanes);
    hc2c_stage(g_Z_re, g_Z_im, g_X_re, g_X_im, g_tw_re, g_tw_im, g_K_lanes);
}

static void init_twiddles(double *tw_re, double *tw_im,
                          size_t K, size_t K_lanes) {
    for (int g = 0; g < M; g++) {
        for (int k_inner = 0; k_inner < R; k_inner++) {
            double phi = -2.0 * M_PI * (double)(g * k_inner) / (double)N;
            double wr = cos(phi);
            double wi = sin(phi);
            for (size_t b = 0; b < K; b++) {
                size_t idx = (size_t)g*K_lanes + (size_t)k_inner*K + b;
                tw_re[idx] = wr;
                tw_im[idx] = wi;
            }
        }
    }
}

int main(void) {
    printf("================================================================\n");
    printf("  Path C cascade attribution at N=%d (rdft_%d + hc2c_%d)\n", N, R, M);
    printf("================================================================\n\n");

    size_t Ks[] = {32, 128, 512};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    printf("%-6s %-9s %-8s %-8s %-9s %-9s %-9s %-9s\n",
           "K", "Mono(ns)", "PrePack", "Rdft×M", "HermPack", "Hc2c×1",
           "Cascade", "Sum");
    printf("------ --------- -------- -------- --------- --------- "
           "--------- ---------\n");

    srand(42);
    for (int ki = 0; ki < nK; ki++) {
        size_t K = Ks[ki];
        size_t K_lanes = R * K;

        double *in_re   = aa(N * K);
        double *outA_re = aa(N * K), *outA_im = aa(N * K);
        double *streams = aa((size_t)M*R*K);
        double *E_re    = aa((size_t)M*R*K);
        double *E_im    = aa((size_t)M*R*K);
        double *Z_re    = aa((size_t)M*K_lanes);
        double *Z_im    = aa((size_t)M*K_lanes);
        double *X_re    = aa((size_t)M*K_lanes);
        double *X_im    = aa((size_t)M*K_lanes);
        double *tw_re   = aa((size_t)M*K_lanes);
        double *tw_im   = aa((size_t)M*K_lanes);
        double *dummy   = aa((size_t)M*K_lanes);
        for (size_t i = 0; i < N*K; i++)
            in_re[i] = (double)rand()/RAND_MAX - 0.5;
        init_twiddles(tw_re, tw_im, K, K_lanes);

        /* Wire globals. */
        g_in_re = in_re; g_dummy = dummy;
        g_streams = streams; g_E_re = E_re; g_E_im = E_im;
        g_E_re_rw = E_re; g_E_im_rw = E_im;
        g_Z_re = Z_re; g_Z_im = Z_im; g_X_re = X_re; g_X_im = X_im;
        g_tw_re = tw_re; g_tw_im = tw_im;
        g_K = K; g_K_lanes = K_lanes;
        g_outA_re = outA_re; g_outA_im = outA_im;

        /* Initialize intermediate buffers (some stages read from
         * E_re/E_im which is written by rdft — call once before timing
         * individual passes so we measure realistic state). */
        w_cascade();

        int reps  = (K <= 64) ? 1000 : (K <= 256 ? 200 : 50);
        int trials = 7;

        double tA   = time_fn(w_pathA,   reps, trials);
        double tPP  = time_fn(w_prepack, reps, trials);
        double tR   = time_fn(w_rdft,    reps, trials);
        double tHP  = time_fn(w_hermpack,reps, trials);
        double tHC  = time_fn(w_hc2c,    reps, trials);
        double tC   = time_fn(w_cascade, reps, trials);
        double sum  = tPP + tR + tHP + tHC;

        printf("%-6zu %-9.1f %-8.1f %-8.1f %-9.1f %-9.1f %-9.1f %-9.1f\n",
               K, tA, tPP, tR, tHP, tHC, tC, sum);

        af(in_re); af(outA_re); af(outA_im); af(streams);
        af(E_re); af(E_im); af(Z_re); af(Z_im);
        af(X_re); af(X_im); af(tw_re); af(tw_im); af(dummy);
    }

    printf("\nNotes:\n");
    printf("  Sum = independent timing of each stage.\n");
    printf("  Cascade = end-to-end timing of all 4 stages in sequence.\n");
    printf("  Sum < Cascade indicates pipelining/cache benefit from contiguity.\n");
    printf("  Sum > Cascade indicates re-warming costs between separate calls.\n");
    return 0;
}
