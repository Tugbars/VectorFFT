/* score_and_time_plans.c — empirical test of cost-model V1 vs V2.
 *
 * For a curated set of multi-stage FFT factorizations, compute:
 *   - V1 score (current factorizer.h: per-stage independent, ops/SIMD * cache_factor)
 *   - V2 score (new: V1 + hot-set carry between stages + DTLB penalty)
 *   - measured plan cycles (run the codelets in sequence into a shared buffer)
 *
 * Goal: see if V2 predicts measured cycles better than V1. If V2 is
 * consistently closer to measured (smaller |V2-measured| / measured than
 * |V1-measured| / measured), the carry + DTLB additions earn their keep.
 *
 * NOT a correctness check — the codelets are called with arbitrary input
 * data and don't produce a real FFT. We measure cycles, not values.
 *
 * Test cells (hardcoded; edit MAIN to add more):
 *
 *   N=1024 K=128: buffer 256 KB, fits L2 but not L1 → hot-set carry matters
 *   N=4096 K=256: buffer 8 MB, exceeds all caches → DTLB pressure matters
 *
 * Plans within each cell are hardcoded to span different shapes
 * (radix mix, stage count, ordering).
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifdef _WIN32
  #include <windows.h>
  #include <intrin.h>
#else
  #include <unistd.h>
  #include <x86intrin.h>
#endif

#include "generated/radix_profile.h"
#include "factorizer.h"

#define BENCH_BATCHES 21
#define BENCH_TARGET_NS 50000000.0   /* 50 ms per batch — plan is much longer than codelet */

/* ─── codelet typedef + extern decls (AVX-2, mirrors measure_cpe.c) ─── */
typedef void (*codelet_fn)(double *, double *,
                           const double *, const double *,
                           size_t, size_t);

extern void radix2_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix2_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix3_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix3_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix4_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix4_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix5_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix5_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix6_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix6_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix7_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix7_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix8_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix8_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix10_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix10_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix11_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix11_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix12_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix12_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix13_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix13_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix16_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix16_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix17_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix17_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix19_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix19_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix20_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix20_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix25_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix25_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix64_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix64_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix128_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix128_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix256_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix256_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix512_n1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix512_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);

/* Look up the n1 codelet for radix R. Returns NULL if unsupported. */
static codelet_fn n1_for(int R) {
    switch (R) {
        case   2: return radix2_n1_fwd_avx2;
        case   3: return radix3_n1_fwd_avx2;
        case   4: return radix4_n1_fwd_avx2;
        case   5: return radix5_n1_fwd_avx2;
        case   6: return radix6_n1_fwd_avx2;
        case   7: return radix7_n1_fwd_avx2;
        case   8: return radix8_n1_fwd_avx2;
        case  10: return radix10_n1_fwd_avx2;
        case  11: return radix11_n1_fwd_avx2;
        case  12: return radix12_n1_fwd_avx2;
        case  13: return radix13_n1_fwd_avx2;
        case  16: return radix16_n1_fwd_avx2;
        case  17: return radix17_n1_fwd_avx2;
        case  19: return radix19_n1_fwd_avx2;
        case  20: return radix20_n1_fwd_avx2;
        case  25: return radix25_n1_fwd_avx2;
        case  32: return radix32_n1_fwd_avx2;
        case  64: return radix64_n1_fwd_avx2;
        case 128: return radix128_n1_fwd_avx2;
        case 256: return radix256_n1_fwd_avx2;
        case 512: return radix512_n1_fwd_avx2;
        default: return NULL;
    }
}
static codelet_fn t1_for(int R) {
    switch (R) {
        case   2: return radix2_t1_dit_fwd_avx2;
        case   3: return radix3_t1_dit_fwd_avx2;
        case   4: return radix4_t1_dit_fwd_avx2;
        case   5: return radix5_t1_dit_fwd_avx2;
        case   6: return radix6_t1_dit_fwd_avx2;
        case   7: return radix7_t1_dit_fwd_avx2;
        case   8: return radix8_t1_dit_fwd_avx2;
        case  10: return radix10_t1_dit_fwd_avx2;
        case  11: return radix11_t1_dit_fwd_avx2;
        case  12: return radix12_t1_dit_fwd_avx2;
        case  13: return radix13_t1_dit_fwd_avx2;
        case  16: return radix16_t1_dit_fwd_avx2;
        case  17: return radix17_t1_dit_fwd_avx2;
        case  19: return radix19_t1_dit_fwd_avx2;
        case  20: return radix20_t1_dit_fwd_avx2;
        case  25: return radix25_t1_dit_fwd_avx2;
        case  32: return radix32_t1_dit_fwd_avx2;
        case  64: return radix64_t1_dit_fwd_avx2;
        case 128: return radix128_t1_dit_fwd_avx2;
        case 256: return radix256_t1_dit_fwd_avx2;
        case 512: return radix512_t1_dit_fwd_avx2;
        default: return NULL;
    }
}

/* ─── timing helpers ─── */
#ifdef _WIN32
static double g_qpc_freq;
static double now_ns(void) {
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / g_qpc_freq;
}
static void timer_init(void) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); g_qpc_freq = (double)f.QuadPart; }
#else
static void timer_init(void) {}
static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
#endif

static double measure_freq_ghz(void) {
    (void)__rdtsc();
    double t0 = now_ns(); uint64_t r0 = __rdtsc();
    while (now_ns() - t0 < 50e6) {}
    uint64_t r1 = __rdtsc(); double t1 = now_ns();
    return (double)(r1 - r0) / (t1 - t0);
}

static void *xalloc(size_t bytes) {
    void *p;
#ifdef _WIN32
    p = _aligned_malloc(bytes, 64);
#else
    if (posix_memalign(&p, 64, bytes) != 0) p = NULL;
#endif
    if (!p) { fprintf(stderr, "alloc %zu failed\n", bytes); exit(1); }
    memset(p, 0, bytes);
    return p;
}
static void xfree(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ─── plan executor ───
 * For a DIT in-place factorization factors[0..nf-1] of N points with K
 * parallel batches, view the buffer as [r0, r1, ..., r_{nf-1}, k] with k
 * innermost. Stage s does R[s]-radix butterflies along the r_s axis with:
 *   - me  = N/R[s] × K  (total butterflies in the stage)
 *   - ios = K × ∏_{i>s} R[i]  (stride between butterfly legs in doubles)
 * The codelet processes 4 parallel butterflies per SIMD iteration with k
 * innermost; me MUST be a multiple of 4 for correctness (k always is). */
static void execute_plan(const int *factors, int nf, size_t K, int N,
                         double *rio_re, double *rio_im,
                         const double *tw_re, const double *tw_im)
{
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        size_t ios = K;
        for (int d = s + 1; d < nf; d++) ios *= factors[d];
        size_t me = (size_t)N / R * K;
        codelet_fn fn = (s == 0) ? n1_for(R) : t1_for(R);
        if (!fn) { fprintf(stderr, "no codelet for R=%d (stage %d)\n", R, s); exit(2); }
        fn(rio_re, rio_im, tw_re, tw_im, ios, me);
    }
}

/* Bench one plan: returns median (over BENCH_BATCHES) ns per call,
 * with auto-calibrated iters/batch to ≈ BENCH_TARGET_NS. */
static int dcmp(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}
static double bench_plan(const int *factors, int nf, size_t K, int N)
{
    /* Buffers sized for the whole plan: N*K doubles per channel. */
    size_t nelem = (size_t)N * K + 64;
    size_t ntw = (size_t)N * K + 64;  /* generous; max twiddle table is (R-1)×accK */
    double *rio_re = xalloc(nelem * sizeof(double));
    double *rio_im = xalloc(nelem * sizeof(double));
    double *tw_re  = xalloc(ntw   * sizeof(double));
    double *tw_im  = xalloc(ntw   * sizeof(double));
    for (size_t i = 0; i < nelem; i++) { rio_re[i] = 0.5; rio_im[i] = -0.5; }
    for (size_t i = 0; i < ntw;   i++) { tw_re[i]  = 0.5; tw_im[i]  = -0.5; }

    /* Calibrate */
    int n = 1;
    for (int trial = 0; trial < 12; trial++) {
        double t0 = now_ns();
        for (int i = 0; i < n; i++) execute_plan(factors, nf, K, N, rio_re, rio_im, tw_re, tw_im);
        double dt = now_ns() - t0;
        if (dt > 1e5) {
            n = (int)((double)n * BENCH_TARGET_NS / dt + 0.5);
            if (n < 1) n = 1;
            break;
        }
        n *= 4;
    }

    double samples[BENCH_BATCHES];
    for (int b = 0; b < BENCH_BATCHES; b++) {
        double t0 = now_ns();
        for (int i = 0; i < n; i++) execute_plan(factors, nf, K, N, rio_re, rio_im, tw_re, tw_im);
        samples[b] = (now_ns() - t0) / (double)n;
    }
    qsort(samples, BENCH_BATCHES, sizeof(double), dcmp);
    double median_ns = samples[BENCH_BATCHES / 2];

    xfree(rio_re); xfree(rio_im); xfree(tw_re); xfree(tw_im);
    return median_ns;
}

/* ─── plan list ─── */
typedef struct {
    int N;
    size_t K;
    int factors[8];
    int nf;
    const char *label;
} test_plan_t;

static const test_plan_t PLANS[] = {
    /* N=1024 K=128 — buffer 256 KB, fits L2 not L1, carry effect visible */
    {1024, 128, {32, 32},        2, "{32,32}      "},
    {1024, 128, {16, 64},        2, "{16,64}      "},
    {1024, 128, {64, 16},        2, "{64,16}      "},
    {1024, 128, {4, 16, 16},     3, "{4,16,16}    "},
    {1024, 128, {16, 16, 4},     3, "{16,16,4}    "},
    {1024, 128, {4, 4, 4, 4, 4}, 5, "{4,4,4,4,4}  "},

    /* N=4096 K=256 — buffer 16 MB, fits L3 (36 MB), exercises L3 tier */
    {4096, 256, {64, 64},        2, "{64,64}      "},
    {4096, 256, {16, 16, 16},    3, "{16,16,16}   "},
    {4096, 256, {4, 16, 64},     3, "{4,16,64}    "},
    {4096, 256, {64, 16, 4},     3, "{64,16,4}    "},
    {4096, 256, {8, 8, 8, 8},    4, "{8,8,8,8}    "},
    {4096, 256, {32, 128},       2, "{32,128}     "},
    {4096, 256, {128, 32},       2, "{128,32}     "},

    /* N=16384 K=512 — buffer 256 MB (>> L3 36 MB), DRAM-bound regime.
     * Tests whether the buffer-streaming floor correctly forces inner
     * stages to pay DRAM cost even when their per-group ws is tiny. */
    {16384, 512, {128, 128},     2, "{128,128}    "},
    {16384, 512, {64, 16, 16},   3, "{64,16,16}   "},
    {16384, 512, {32, 32, 16},   3, "{32,32,16}   "},
    {16384, 512, {16, 16, 16, 4}, 4, "{16,16,16,4}"},
    {16384, 512, {256, 64},      2, "{256,64}     "},
};
#define N_PLANS (sizeof(PLANS) / sizeof(PLANS[0]))

int main(void) {
    timer_init();
    double freq_ghz = measure_freq_ghz();
    printf("[score_and_time_plans] freq = %.3f GHz\n", freq_ghz);
    if (freq_ghz < 0.5 || freq_ghz > 10.0) {
        fprintf(stderr, "[error] freq out of range\n");
        return 2;
    }

    stride_cpu_info_t cpu = stride_detect_cpu();
    printf("[score_and_time_plans] L1=%zuKB L2=%zuKB dtlb=%d miss=%dcy\n\n",
           cpu.l1d_bytes/1024, cpu.l2_bytes/1024,
           cpu.dtlb_entries, cpu.dtlb_miss_cycles);

    printf("  %-14s %-18s | %12s %12s | %12s | %8s %8s\n",
           "plan", "(N,K)",
           "V1 score", "V2 score", "measured cyc",
           "V1/meas", "V2/meas");
    printf("  %s\n", "------------------------------------------------------------------------------------------------------------");

    int prev_N = -1; size_t prev_K = 0;
    int v2_better = 0, v1_better = 0;
    double sum_v1_err = 0, sum_v2_err = 0;

    for (size_t i = 0; i < N_PLANS; i++) {
        const test_plan_t *p = &PLANS[i];
        if (p->N != prev_N || p->K != prev_K) {
            printf("  --- N=%d K=%zu (buffer=%zuKB) ---\n",
                   p->N, p->K, (size_t)p->N * p->K * 16 / 1024);
            prev_N = p->N; prev_K = p->K;
        }
        double v1 = stride_score_factorization   (p->factors, p->nf, p->K, p->N, &cpu);
        double v2 = stride_score_factorization_v2(p->factors, p->nf, p->K, p->N, &cpu);
        double meas_ns = bench_plan(p->factors, p->nf, p->K, p->N);
        double meas_cyc = meas_ns * freq_ghz;
        double v1_err = fabs(v1 - meas_cyc) / meas_cyc;
        double v2_err = fabs(v2 - meas_cyc) / meas_cyc;
        sum_v1_err += v1_err;
        sum_v2_err += v2_err;
        if (v2_err < v1_err) v2_better++; else v1_better++;

        char nk[24]; snprintf(nk, sizeof(nk), "N=%d K=%zu", p->N, p->K);
        printf("  %-14s %-18s | %12.0f %12.0f | %12.0f | %7.2fx %7.2fx %s\n",
               p->label, nk, v1, v2, meas_cyc,
               v1 / meas_cyc, v2 / meas_cyc,
               (v2_err < v1_err) ? "V2 better" : "V1 better");
    }

    printf("\n[summary]\n");
    printf("  V2 closer to measured on %d/%zu plans\n", v2_better, N_PLANS);
    printf("  V1 closer to measured on %d/%zu plans\n", v1_better, N_PLANS);
    printf("  mean rel error: V1=%.3f V2=%.3f\n",
           sum_v1_err / N_PLANS, sum_v2_err / N_PLANS);
    return 0;
}
