/* bench_vtune_cascade.c — VTune-instrumented Path C cascade attribution
 *                          at N=128 r2c, AVX2.
 *
 * Goal: VTune uarch-exploration of each cascade stage to determine why
 * hc2c dominates at ~50% of cascade time. Memory bound vs core bound
 * vs dependency chains?
 *
 * Stages with ITT task markers (per K value):
 *   PrePack    — copy N reals into M strided streams
 *   RdftXM     — M=8 calls of radix16_rdft_fwd_avx2_gen
 *   HermPack   — copy rdft outputs into hc2c-shaped buffer with conj
 *   Hc2c       — single radix8_hc2c_dit_fwd_avx2_gen call (K_lanes batch)
 *   Mono       — Path A baseline (radix128_r2c_fwd_avx2_gen)
 *
 * Each task runs ~1 second of work. VTune attributes retiring %, memory
 * bound %, port utilization, etc. per task.
 *
 * Build (via the existing build harness or manually with cl/icx):
 *   icx /O3 /Qstd:c11 /arch:AVX2 -I<vtune-sdk>/include
 *       bench_vtune_cascade.c
 *       <prototype>/codelets/avx2/large_pow2/r128_r2c_fwd.c
 *       <prototype>/codelets/avx2/mid_pow2/r16_rdft_fwd.c
 *       <prototype>/codelets/avx2/small_pow2/r8_hc2c_dit_fwd.c
 *       libittnotify.lib  /Fe:bench_vtune_cascade.exe
 *
 * Run:
 *   vtune -collect uarch-exploration -result-dir vt_cascade -- ^
 *         bench_vtune_cascade.exe
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

#include "ittnotify.h"

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

static __itt_domain *g_dom = NULL;

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
        radix16_rdft_fwd_avx2_gen(in_base, dummy, out_re_g, out_im_g,
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

static __itt_string_handle *make_task(const char *prefix, size_t K) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%s_K%zu", prefix, K);
    return __itt_string_handle_create(buf);
}

static void run_cell(size_t K, double target_seconds) {
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

    srand((unsigned)(0xC0DE ^ K));
    for (size_t i = 0; i < N*K; i++)
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
    init_twiddles(tw_re, tw_im, K, K_lanes);

    /* Warm up; also leaves E_re/E_im populated so HermPack reads
     * realistic data. */
    prepack(in_re, streams, K);
    rdft_stage(streams, E_re, E_im, dummy, K);

    /* Estimate reps to fill target_seconds for each task. Use the
     * cascade attribution numbers we already measured (roughly). */
    double ns_per_K[3] = {600+700+1100+2700, 2000+3600+3800+14600,
                          8000+37500+17600+70300};
    int k_idx = (K == 32) ? 0 : (K == 128 ? 1 : 2);
    double total_ns = ns_per_K[k_idx];
    int reps = (int)((target_seconds * 1e9) / total_ns);
    if (reps < 10) reps = 10;

    printf("=== Cell K=%zu, reps per task=%d (target %.2fs/task) ===\n",
           K, reps, target_seconds);

    __itt_string_handle *t_mono   = make_task("Mono",      K);
    __itt_string_handle *t_pp     = make_task("PrePack",   K);
    __itt_string_handle *t_rdft   = make_task("Rdft",      K);
    __itt_string_handle *t_hp     = make_task("HermPack",  K);
    __itt_string_handle *t_hc2c   = make_task("Hc2c",      K);

    /* Path A monolithic baseline. */
    __itt_task_begin(g_dom, __itt_null, __itt_null, t_mono);
    double tA0 = now_ns();
    for (int r = 0; r < reps; r++)
        radix128_r2c_fwd_avx2_gen(in_re, dummy, outA_re, outA_im,
                                  dummy, dummy, K);
    double tA = (now_ns() - tA0) / (double)reps;
    __itt_task_end(g_dom);

    __itt_task_begin(g_dom, __itt_null, __itt_null, t_pp);
    double tPP0 = now_ns();
    for (int r = 0; r < reps; r++)
        prepack(in_re, streams, K);
    double tPP = (now_ns() - tPP0) / (double)reps;
    __itt_task_end(g_dom);

    __itt_task_begin(g_dom, __itt_null, __itt_null, t_rdft);
    double tR0 = now_ns();
    for (int r = 0; r < reps; r++)
        rdft_stage(streams, E_re, E_im, dummy, K);
    double tR = (now_ns() - tR0) / (double)reps;
    __itt_task_end(g_dom);

    __itt_task_begin(g_dom, __itt_null, __itt_null, t_hp);
    double tHP0 = now_ns();
    for (int r = 0; r < reps; r++)
        hermpack(E_re, E_im, Z_re, Z_im, K, K_lanes);
    double tHP = (now_ns() - tHP0) / (double)reps;
    __itt_task_end(g_dom);

    __itt_task_begin(g_dom, __itt_null, __itt_null, t_hc2c);
    double tHC0 = now_ns();
    for (int r = 0; r < reps; r++)
        radix8_hc2c_dit_fwd_avx2_gen(Z_re, Z_im, X_re, X_im,
                                     tw_re, tw_im, K_lanes);
    double tHC = (now_ns() - tHC0) / (double)reps;
    __itt_task_end(g_dom);

    printf("  Mono     %8.1f ns\n", tA);
    printf("  PrePack  %8.1f ns\n", tPP);
    printf("  Rdft     %8.1f ns\n", tR);
    printf("  HermPack %8.1f ns\n", tHP);
    printf("  Hc2c     %8.1f ns\n", tHC);
    printf("  Cascade  %8.1f ns (sum)\n", tPP + tR + tHP + tHC);

    af(in_re); af(outA_re); af(outA_im); af(streams);
    af(E_re); af(E_im); af(Z_re); af(Z_im);
    af(X_re); af(X_im); af(tw_re); af(tw_im); af(dummy);
}

int main(void) {
    /* Pin to a P-core for clean profile. */
#ifdef _WIN32
    SetProcessAffinityMask(GetCurrentProcess(), 1ULL << 2);
    SetThreadAffinityMask(GetCurrentThread(),  1ULL << 2);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif

    g_dom = __itt_domain_create("VectorFFT.CascadeVTune");
    if (!g_dom) { fprintf(stderr, "ITT domain create failed\n"); return 1; }

    printf("================================================================\n");
    printf("  Cascade VTune attribution at N=%d (rdft_%d + hc2c_%d)\n",
           N, R, M);
    printf("  AVX2, ITT-marked tasks. P-core pinned, HIGH priority.\n");
    printf("================================================================\n\n");

    /* Run at three K values to span small/medium/large regimes.
     * Each task ~1 second so VTune gets enough samples per region. */
    run_cell(32,  1.0);
    run_cell(128, 1.0);
    run_cell(512, 1.0);

    return 0;
}
