/* bench_r512_avx2.c — Monolithic R=512 codelets vs multi-stage 16×32.
 *
 * Same shape as bench_r256_avx2.c but for R=512. Wisdom file picks 16×32
 * as the best multi-stage factorization for N=512 K=4 (T1S codelets),
 * matching doc 33's balanced-factorization finding. We use that single
 * factorization across K to keep the comparison clean.
 *
 * Contestants (per K ∈ {4, 32, 256}):
 *   1. mono_n1         radix512_n1_fwd
 *   2. mono_t1         radix512_t1_dit_fwd       (fed random twiddles)
 *   3. mono_t1s        radix512_t1s_dit_fwd      (fed random twiddles)
 *   4. mono_t1_log3    radix512_t1_dit_log3_fwd  (fed random twiddles)
 *   5. mono_t1s_log3   radix512_t1s_dit_log3_fwd (fed random twiddles)
 *   6. multi_16x32     r16_n1 × 32  →  twid mul  →  r32_n1 × 16
 *
 * Layout: rio[K*p + k] where p ∈ [0,512), p = 32*a + b, a∈[0,16), b∈[0,32).
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

/* ── Codelet externs ──────────────────────────────────────────────── */

__attribute__((target("avx2,fma")))
void radix512_n1_fwd_avx2_gen_inplace(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx2,fma")))
void radix512_t1_dit_fwd_avx2_gen_inplace_su_spill(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx2,fma")))
void radix512_t1s_dit_fwd_avx2_gen_inplace_su_spill(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx2,fma")))
void radix512_t1_dit_log3_fwd_avx2_gen_inplace_su_spill(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx2,fma")))
void radix512_t1s_dit_log3_fwd_avx2_gen_inplace_su_spill(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx2,fma")))
void radix16_n1_fwd_avx2_gen_inplace(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx2,fma")))
void radix32_n1_fwd_avx2_gen_inplace(
    double *, double *, const double *, const double *, size_t, size_t);

/* ── Helpers ──────────────────────────────────────────────────────── */

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

static void fr(double *p, size_t n, unsigned s) {
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (double)((int)(s >> 8) & 0x7fffff) / (double)0x800000 - 0.5;
    }
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

static double bench(void (*fn)(void), int repeat, int trials) {
    double best = 1e18;
    for (int i = 0; i < 30; i++) fn();
    for (int t = 0; t < trials; t++) {
        double t0 = now_ns();
        for (int i = 0; i < repeat; i++) fn();
        double dt = (now_ns() - t0) / (double)repeat;
        if (dt < best) best = dt;
    }
    return best;
}

/* ── Globals (set per K) ──────────────────────────────────────────── */

static size_t g_K;
static double *g_rio_re, *g_rio_im;
static double *g_tw_re, *g_tw_im;
static double *g_stage_tw_re, *g_stage_tw_im;  /* 16×32 inter-stage W */

/* ── Multi-stage 16×32 implementation ─────────────────────────────── */

/* W[a*32+b] = exp(-2πi * a*b / 512). */
static void precompute_stage_twiddles(void) {
    for (int a = 0; a < 16; a++) {
        for (int b = 0; b < 32; b++) {
            double phi = -2.0 * M_PI * (double)(a * b) / 512.0;
            g_stage_tw_re[a * 32 + b] = cos(phi);
            g_stage_tw_im[a * 32 + b] = sin(phi);
        }
    }
}

__attribute__((target("avx2,fma")))
static void multistage_twiddle_mul(double *re, double *im,
                                   const double *Wr, const double *Wi,
                                   size_t K) {
    for (int a = 0; a < 16; a++) {
        for (int b = 0; b < 32; b++) {
            int p = 32 * a + b;
            double wr = Wr[a * 32 + b];
            double wi = Wi[a * 32 + b];
            __m256d v_wr = _mm256_set1_pd(wr);
            __m256d v_wi = _mm256_set1_pd(wi);
            double *r = re + (size_t)p * K;
            double *m = im + (size_t)p * K;
            for (size_t k = 0; k < K; k += 4) {
                __m256d xr = _mm256_loadu_pd(r + k);
                __m256d xi = _mm256_loadu_pd(m + k);
                __m256d yr = _mm256_fnmadd_pd(xi, v_wi, _mm256_mul_pd(xr, v_wr));
                __m256d yi = _mm256_fmadd_pd(xr, v_wi, _mm256_mul_pd(xi, v_wr));
                _mm256_storeu_pd(r + k, yr);
                _mm256_storeu_pd(m + k, yi);
            }
        }
    }
}

static void call_multi_16x32(void) {
    /* Stage 1: 32 inner DFT-16, fixed b, stride 32*K over a. */
    for (int b = 0; b < 32; b++) {
        radix16_n1_fwd_avx2_gen_inplace(
            g_rio_re + (size_t)b * g_K,
            g_rio_im + (size_t)b * g_K,
            NULL, NULL,
            32 * g_K, g_K);
    }
    /* Inter-stage twiddle. */
    multistage_twiddle_mul(g_rio_re, g_rio_im,
                           g_stage_tw_re, g_stage_tw_im, g_K);
    /* Stage 2: 16 outer DFT-32, fixed a, stride K over b. */
    for (int a = 0; a < 16; a++) {
        radix32_n1_fwd_avx2_gen_inplace(
            g_rio_re + (size_t)a * 32 * g_K,
            g_rio_im + (size_t)a * 32 * g_K,
            NULL, NULL,
            g_K, g_K);
    }
}

/* ── Wrappers ────────────────────────────────────────────────────── */

static void call_mono_n1(void) {
    radix512_n1_fwd_avx2_gen_inplace(
        g_rio_re, g_rio_im, NULL, NULL, g_K, g_K);
}
static void call_mono_t1(void) {
    radix512_t1_dit_fwd_avx2_gen_inplace_su_spill(
        g_rio_re, g_rio_im, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_mono_t1s(void) {
    radix512_t1s_dit_fwd_avx2_gen_inplace_su_spill(
        g_rio_re, g_rio_im, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_mono_t1_log3(void) {
    radix512_t1_dit_log3_fwd_avx2_gen_inplace_su_spill(
        g_rio_re, g_rio_im, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_mono_t1s_log3(void) {
    radix512_t1s_dit_log3_fwd_avx2_gen_inplace_su_spill(
        g_rio_re, g_rio_im, g_tw_re, g_tw_im, g_K, g_K);
}

static int reps_for_K(size_t K) {
    if (K <= 4)   return 3000;
    if (K <= 32)  return 500;
    if (K <= 256) return 60;
    return 30;
}

/* ── Validate mono_n1 vs multi_16x32 by sorted-magnitudes ─────────── */

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static int validate_mono_vs_multi(void) {
    size_t N = 512 * g_K;
    double *in_re = aa(N), *in_im = aa(N);
    fr(in_re, N, 0x77);
    fr(in_im, N, 0x88);

    memcpy(g_rio_re, in_re, N * sizeof(double));
    memcpy(g_rio_im, in_im, N * sizeof(double));
    call_mono_n1();
    double *mag_mono = aa(N);
    for (size_t i = 0; i < N; i++)
        mag_mono[i] = g_rio_re[i] * g_rio_re[i] + g_rio_im[i] * g_rio_im[i];

    memcpy(g_rio_re, in_re, N * sizeof(double));
    memcpy(g_rio_im, in_im, N * sizeof(double));
    call_multi_16x32();
    double *mag_multi = aa(N);
    for (size_t i = 0; i < N; i++)
        mag_multi[i] = g_rio_re[i] * g_rio_re[i] + g_rio_im[i] * g_rio_im[i];

    qsort(mag_mono,  N, sizeof(double), cmp_double);
    qsort(mag_multi, N, sizeof(double), cmp_double);
    double maxrel = 0;
    for (size_t i = 0; i < N; i++) {
        double scale = fabs(mag_mono[i]) + 1e-12;
        double r = fabs(mag_mono[i] - mag_multi[i]) / scale;
        if (r > maxrel) maxrel = r;
    }

    af(in_re); af(in_im); af(mag_mono); af(mag_multi);
    return (maxrel < 1e-9) ? 0 : 1;
}

/* ── Main ─────────────────────────────────────────────────────────── */

static void run_K(size_t K) {
    g_K = K;
    size_t N = 512 * K;
    g_rio_re = aa(N);
    g_rio_im = aa(N);
    g_tw_re = aa((size_t)512 * K);
    g_tw_im = aa((size_t)512 * K);
    fr(g_tw_re, (size_t)512 * K, 0xc1);
    fr(g_tw_im, (size_t)512 * K, 0xc2);
    g_stage_tw_re = aa(16 * 32);
    g_stage_tw_im = aa(16 * 32);
    precompute_stage_twiddles();
    fr(g_rio_re, N, 0x11);
    fr(g_rio_im, N, 0x22);

    int rc = validate_mono_vs_multi();
    if (rc != 0)
        printf("  K=%-4zu  VALIDATION FAIL (mono_n1 vs multi_16x32 magnitudes differ)\n", K);

    int reps = reps_for_K(K);
    int trials = 7;

    fr(g_rio_re, N, 0x11); fr(g_rio_im, N, 0x22);
    double t_multi   = bench(call_multi_16x32,   reps, trials);
    fr(g_rio_re, N, 0x11); fr(g_rio_im, N, 0x22);
    double t_mono_n1 = bench(call_mono_n1,       reps, trials);
    fr(g_rio_re, N, 0x11); fr(g_rio_im, N, 0x22);
    double t_t1      = bench(call_mono_t1,       reps, trials);
    fr(g_rio_re, N, 0x11); fr(g_rio_im, N, 0x22);
    double t_t1s     = bench(call_mono_t1s,      reps, trials);
    fr(g_rio_re, N, 0x11); fr(g_rio_im, N, 0x22);
    double t_t1_log3 = bench(call_mono_t1_log3,  reps, trials);
    fr(g_rio_re, N, 0x11); fr(g_rio_im, N, 0x22);
    double t_t1s_log = bench(call_mono_t1s_log3, reps, trials);

    printf("\nK = %zu  (reps=%d, trials=%d)\n", K, reps, trials);
    printf("  %-20s  %10.1f ns   ratio=%5.3f  (baseline)\n",
           "multi_16x32",   t_multi, 1.0);
    printf("  %-20s  %10.1f ns   ratio=%5.3f%s\n",
           "mono_n1",       t_mono_n1, t_mono_n1 / t_multi,
           t_mono_n1 < t_multi ? "  WIN" : "");
    printf("  %-20s  %10.1f ns   ratio=%5.3f%s\n",
           "mono_t1",       t_t1, t_t1 / t_multi,
           t_t1 < t_multi ? "  WIN" : "");
    printf("  %-20s  %10.1f ns   ratio=%5.3f%s\n",
           "mono_t1s",      t_t1s, t_t1s / t_multi,
           t_t1s < t_multi ? "  WIN" : "");
    printf("  %-20s  %10.1f ns   ratio=%5.3f%s\n",
           "mono_t1_log3",  t_t1_log3, t_t1_log3 / t_multi,
           t_t1_log3 < t_multi ? "  WIN" : "");
    printf("  %-20s  %10.1f ns   ratio=%5.3f%s\n",
           "mono_t1s_log3", t_t1s_log, t_t1s_log / t_multi,
           t_t1s_log < t_multi ? "  WIN" : "");

    af(g_rio_re); af(g_rio_im);
    af(g_tw_re);  af(g_tw_im);
    af(g_stage_tw_re); af(g_stage_tw_im);
}

int main(int argc, char **argv) {
    printf("R=512 monolithic vs multi-stage 16x32 -- AVX2\n");
    printf("Layout: rio[K*p + k], p = 32*a + b\n");
    printf("\n");

    size_t Ks_default[] = {4, 32, 256, 0};
    size_t *Ks = Ks_default;
    size_t single[2] = {0, 0};
    if (argc > 1) {
        single[0] = (size_t)atoi(argv[1]);
        Ks = single;
    }
    for (int i = 0; Ks[i] != 0; i++) {
        if (Ks[i] % 4 != 0) continue;
        run_K(Ks[i]);
    }
    return 0;
}
