
/* Auto-generated bench harness. Do not edit. */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#ifdef _WIN32
  #include <windows.h>
  #include <malloc.h>
#endif

#define static /* flatten static decls so everything links */
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_dit__avx2.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_dit_log3__avx2.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1s_dit__avx2.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx2__tile16_draintemporal.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx2__tile16_drainstream.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx2__tile32_draintemporal.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx2__tile32_drainstream.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx2__tile64_draintemporal.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx2__tile64_drainstream.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx2__tile128_draintemporal.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx2__tile128_drainstream.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_dit__avx512.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_dit_log3__avx512.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1s_dit__avx512.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx512__tile16_draintemporal.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx512__tile16_drainstream.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx512__tile32_draintemporal.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx512__tile32_drainstream.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx512__tile64_draintemporal.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx512__tile64_drainstream.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx512__tile128_draintemporal.h"
#include "C:\Users\Tugbars\Desktop\highSpeedFFT\src\test-gen\R32\staging\ct_t1_buf_dit__avx512__tile128_drainstream.h"

/* ─── timing: portable high-resolution monotonic clock in ns ─── */
#ifdef _WIN32
static double now_ns(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    /* ns = ticks * 1e9 / freq */
    return (double)t.QuadPart * 1e9 / (double)freq.QuadPart;
}
#else
static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/* ─── aligned alloc / free ─── */
#ifdef _WIN32
static void *aalloc(size_t bytes) {
    void *p = _aligned_malloc(bytes, 64);
    if (p) memset(p, 0, bytes);
    return p;
}
static void afree(void *p) { _aligned_free(p); }
#else
static void *aalloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    memset(p, 0, bytes);
    return p;
}
static void afree(void *p) { free(p); }
#endif

/* ─── runtime CPU feature check for AVX-512 ─── */
static int have_avx512(void) {
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512dq");
}

/* ─── dispatch tables ─── */
typedef void (*t1_fn)(double *rio_re, double *rio_im,
                      const double *W_re, const double *W_im,
                      size_t ios, size_t me);

typedef struct {
    const char *id;
    const char *isa;
    t1_fn fwd;
    t1_fn bwd;
    int requires_avx512;
} candidate_entry_t;

/* Populated by emit_dispatch_table() below */
static const candidate_entry_t CANDIDATES[] = {
    {"ct_t1_dit__avx2", "avx2", radix16_t1_dit_fwd_avx2, radix16_t1_dit_bwd_avx2, 0},
    {"ct_t1_dit_log3__avx2", "avx2", radix16_t1_dit_log3_fwd_avx2, radix16_t1_dit_log3_bwd_avx2, 0},
    {"ct_t1s_dit__avx2", "avx2", radix16_t1s_dit_fwd_avx2, radix16_t1s_dit_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile16_draintemporal", "avx2", radix16_t1_buf_dit_tile16_temporal_fwd_avx2, radix16_t1_buf_dit_tile16_temporal_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile16_drainstream", "avx2", radix16_t1_buf_dit_tile16_stream_fwd_avx2, radix16_t1_buf_dit_tile16_stream_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32_draintemporal", "avx2", radix16_t1_buf_dit_tile32_temporal_fwd_avx2, radix16_t1_buf_dit_tile32_temporal_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32_drainstream", "avx2", radix16_t1_buf_dit_tile32_stream_fwd_avx2, radix16_t1_buf_dit_tile32_stream_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64_draintemporal", "avx2", radix16_t1_buf_dit_tile64_temporal_fwd_avx2, radix16_t1_buf_dit_tile64_temporal_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64_drainstream", "avx2", radix16_t1_buf_dit_tile64_stream_fwd_avx2, radix16_t1_buf_dit_tile64_stream_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128_draintemporal", "avx2", radix16_t1_buf_dit_tile128_temporal_fwd_avx2, radix16_t1_buf_dit_tile128_temporal_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128_drainstream", "avx2", radix16_t1_buf_dit_tile128_stream_fwd_avx2, radix16_t1_buf_dit_tile128_stream_bwd_avx2, 0},
    {"ct_t1_dit__avx512", "avx512", radix16_t1_dit_fwd_avx512, radix16_t1_dit_bwd_avx512, 1},
    {"ct_t1_dit_log3__avx512", "avx512", radix16_t1_dit_log3_fwd_avx512, radix16_t1_dit_log3_bwd_avx512, 1},
    {"ct_t1s_dit__avx512", "avx512", radix16_t1s_dit_fwd_avx512, radix16_t1s_dit_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile16_draintemporal", "avx512", radix16_t1_buf_dit_tile16_temporal_fwd_avx512, radix16_t1_buf_dit_tile16_temporal_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile16_drainstream", "avx512", radix16_t1_buf_dit_tile16_stream_fwd_avx512, radix16_t1_buf_dit_tile16_stream_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32_draintemporal", "avx512", radix16_t1_buf_dit_tile32_temporal_fwd_avx512, radix16_t1_buf_dit_tile32_temporal_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32_drainstream", "avx512", radix16_t1_buf_dit_tile32_stream_fwd_avx512, radix16_t1_buf_dit_tile32_stream_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64_draintemporal", "avx512", radix16_t1_buf_dit_tile64_temporal_fwd_avx512, radix16_t1_buf_dit_tile64_temporal_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64_drainstream", "avx512", radix16_t1_buf_dit_tile64_stream_fwd_avx512, radix16_t1_buf_dit_tile64_stream_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128_draintemporal", "avx512", radix16_t1_buf_dit_tile128_temporal_fwd_avx512, radix16_t1_buf_dit_tile128_temporal_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128_drainstream", "avx512", radix16_t1_buf_dit_tile128_stream_fwd_avx512, radix16_t1_buf_dit_tile128_stream_bwd_avx512, 1},

    {NULL, NULL, NULL, NULL, 0}
};

static int N_CANDIDATES =
    sizeof(CANDIDATES)/sizeof(CANDIDATES[0]) - 1;

/* ─── sweep points ─── */
typedef struct { size_t ios, me; } sweep_point_t;
static const sweep_point_t SWEEP[] = {
    {64, 64},
    {72, 64},
    {128, 64},
    {128, 128},
    {136, 128},
    {192, 128},
    {256, 256},
    {264, 256},
    {320, 256},
    {512, 512},
    {520, 512},
    {576, 512},
    {1024, 1024},
    {1032, 1024},
    {1088, 1024},
    {2048, 2048},
    {2056, 2048},
    {2112, 2048},

    {0, 0}
};
static int N_SWEEP = sizeof(SWEEP)/sizeof(SWEEP[0]) - 1;

/* ─── twiddle table setup ─── */
/* For R=16, W_re/im has 15 rows × me columns, in [(j-1)*me + m] layout */
#define R 16
#define PI 3.14159265358979323846
static void fill_twiddles(double *Wr, double *Wi, size_t me) {
    for (size_t n = 0; n < R - 1; n++) {
        for (size_t m = 0; m < me; m++) {
            double g = -2.0 * PI * (double)(n+1) * (double)m
                       / (double)(R * me);
            Wr[n*me + m] = cos(g);
            Wi[n*me + m] = sin(g);
        }
    }
}

/* ─── comparator for qsort (top-level; ICX/clang reject nested fns) ─── */
static int _dcmp(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/* ─── measurement of one (candidate, ios, me) ─── */
static double measure(t1_fn fn, size_t ios, size_t me, int bwd) {
    size_t alloc_N = R * (ios > me ? ios : me);
    double *rio_re = aalloc(alloc_N * sizeof(double));
    double *rio_im = aalloc(alloc_N * sizeof(double));
    double *src_re = aalloc(alloc_N * sizeof(double));
    double *src_im = aalloc(alloc_N * sizeof(double));
    double *Wr = aalloc((R-1) * me * sizeof(double));
    double *Wi = aalloc((R-1) * me * sizeof(double));
    (void)bwd;

    /* Random data — don't want zero inputs (FMA throughput differs) */
    unsigned s = 12345;
    for (size_t i = 0; i < alloc_N; i++) {
        s = s * 1103515245u + 12345u;
        src_re[i] = ((double)(s >> 16) / 32768.0) - 1.0;
        s = s * 1103515245u + 12345u;
        src_im[i] = ((double)(s >> 16) / 32768.0) - 1.0;
    }
    fill_twiddles(Wr, Wi, me);

    /* Reps scale so each timed region is ~1-5ms */
    size_t work = R * me;
    int reps = (int)(1000000.0 / ((double)work + 1));
    if (reps < 20) reps = 20;
    if (reps > 10000) reps = 10000;

    /* Warmup — get caches/TLB warm, branch predictor trained */
    for (int i = 0; i < 100; i++) {
        memcpy(rio_re, src_re, alloc_N * sizeof(double));
        memcpy(rio_im, src_im, alloc_N * sizeof(double));
        fn(rio_re, rio_im, Wr, Wi, ios, me);
    }

    /* Measure — median of 21 samples */
    double samples[21];
    for (int i = 0; i < 21; i++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(rio_re, src_re, alloc_N * sizeof(double));
            memcpy(rio_im, src_im, alloc_N * sizeof(double));
            fn(rio_re, rio_im, Wr, Wi, ios, me);
        }
        double ns_per_rep = (now_ns() - t0) / reps;
        samples[i] = ns_per_rep;
    }
    /* Also measure memcpy-only to subtract */
    double mc_samples[21];
    for (int i = 0; i < 21; i++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(rio_re, src_re, alloc_N * sizeof(double));
            memcpy(rio_im, src_im, alloc_N * sizeof(double));
        }
        mc_samples[i] = (now_ns() - t0) / reps;
    }

    /* Median */
    qsort(samples, 21, sizeof(double), _dcmp);
    qsort(mc_samples, 21, sizeof(double), _dcmp);
    double net = samples[10] - mc_samples[10];
    if (net < 0) net = samples[10];  /* memcpy subtraction guard */

    afree(rio_re); afree(rio_im); afree(src_re); afree(src_im);
    afree(Wr); afree(Wi);
    return net;
}

/* ─── main: sweep and emit JSON ─── */
int main(int argc, char **argv) {
    int skip_avx512 = !have_avx512();

    fprintf(stderr, "harness: %d candidates, %d sweep points, avx512=%s\n",
            N_CANDIDATES, N_SWEEP, skip_avx512 ? "MISSING (skipping)" : "OK");

    printf("{\n");
    printf("  \"avx512_available\": %s,\n", skip_avx512 ? "false" : "true");
    printf("  \"measurements\": [\n");

    int first = 1;
    for (int ci = 0; ci < N_CANDIDATES; ci++) {
        const candidate_entry_t *c = &CANDIDATES[ci];
        if (c->requires_avx512 && skip_avx512) {
            fprintf(stderr, "  [skip avx512] %s\n", c->id);
            continue;
        }
        fprintf(stderr, "  [%d/%d] %s\n", ci+1, N_CANDIDATES, c->id);

        for (int sp = 0; sp < N_SWEEP; sp++) {
            size_t ios = SWEEP[sp].ios;
            size_t me  = SWEEP[sp].me;

            double fwd_ns = measure(c->fwd, ios, me, 0);
            double bwd_ns = measure(c->bwd, ios, me, 1);

            if (!first) printf(",\n");
            first = 0;
            printf("    {\"id\": \"%s\", \"ios\": %zu, \"me\": %zu, "
                   "\"fwd_ns\": %.2f, \"bwd_ns\": %.2f}",
                   c->id, ios, me, fwd_ns, bwd_ns);
            fflush(stdout);
        }
    }
    printf("\n  ]\n}\n");
    return 0;
}
