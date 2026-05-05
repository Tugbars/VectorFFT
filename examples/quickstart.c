/**
 * quickstart.c — VectorFFT v1.0 public API tour
 *
 * One self-contained C program touching every public entry point in
 * <vfft.h>. Each section is a small function that:
 *   1. Creates a plan
 *   2. Runs it forward and backward (or just forward where applicable)
 *   3. Verifies correctness against a known input
 *   4. Reports timing
 *
 * Run from anywhere — wisdom is optional. With no wisdom file present the
 * library falls back to VFFT_ESTIMATE (closed-form cost-model). With a
 * loaded wisdom file (e.g. examples/14900KF/wisdom.txt) the same plans
 * become VFFT_MEASURE — calibrated, faster.
 *
 * Build (CMake):
 *     cmake --build build --target vfft_quickstart
 *
 * Build (manual, MSVC):
 *     cl /O2 /MD /arch:AVX2 /I include examples/quickstart.c \
 *         /link build_msvc/lib/Release/vfft.lib
 *
 * Build (manual, GCC/MinGW):
 *     gcc -O3 -mavx2 -mfma -I include examples/quickstart.c \
 *         -L build_gcc/lib -lvfft -lm -o quickstart.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vfft.h"

/* ───────────────────────────────────────────────────────────────────
 * Cross-platform high-resolution timer
 * ─────────────────────────────────────────────────────────────────── */

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static double now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/* ───────────────────────────────────────────────────────────────────
 * Helpers — fill, max-abs-diff, header printer
 * ─────────────────────────────────────────────────────────────────── */

static void fill_random(double *p, size_t n, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < n; i++)
        p[i] = (double)rand() / RAND_MAX - 0.5;
}

static double max_abs_diff(const double *a, const double *b, size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        double d = fabs(a[i] - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

static void section(const char *title)
{
    printf("\n=== %s ===\n", title);
}

/* ───────────────────────────────────────────────────────────────────
 * SECTION 1 — 1D Complex-to-Complex (C2C)
 *
 *   Plan:    vfft_plan_c2c(N, K, flags)
 *   Execute: vfft_execute_fwd / _bwd / _bwd_normalized
 *
 *   Layout:  re[n*K + k], im[n*K + k]   for n=0..N-1, k=0..K-1
 *
 *   Roundtrip:   bwd(fwd(x)) = N * x      (unnormalized)
 *                bwd_normalized(fwd(x)) = x
 * ─────────────────────────────────────────────────────────────────── */

static void demo_1d_c2c(unsigned flags)
{
    section("1D C2C  N=1024 K=4");

    const int N = 1024;
    const size_t K = 4;
    const size_t NK = (size_t)N * K;

    vfft_plan p = vfft_plan_c2c(N, K, flags);
    if (!p)
    {
        fprintf(stderr, "  vfft_plan_c2c failed\n");
        return;
    }

    double *re = (double *)vfft_alloc(NK * sizeof(double));
    double *im = (double *)vfft_alloc(NK * sizeof(double));
    double *re0 = (double *)vfft_alloc(NK * sizeof(double));
    double *im0 = (double *)vfft_alloc(NK * sizeof(double));

    fill_random(re, NK, 42);
    fill_random(im, NK, 43);
    memcpy(re0, re, NK * sizeof(double));
    memcpy(im0, im, NK * sizeof(double));

    double t0 = now_ns();
    vfft_execute_fwd(p, re, im);
    vfft_execute_bwd_normalized(p, re, im);
    double dt = now_ns() - t0;

    double err = fmax(max_abs_diff(re, re0, NK), max_abs_diff(im, im0, NK));
    printf("  fwd + bwd_normalized roundtrip:   err=%.2e   t=%.0f ns\n", err, dt);

    vfft_free(re);
    vfft_free(im);
    vfft_free(re0);
    vfft_free(im0);
    vfft_destroy(p);
}

/* ───────────────────────────────────────────────────────────────────
 * SECTION 2 — 1D Real-to-Complex / Complex-to-Real (R2C/C2R)
 *
 *   N reals → N/2+1 complex bins (Hermitian symmetry).
 *   Constraint: N even.
 *
 *   Output buffers:
 *     out_re sized N*K (workspace), valid output is first (N/2+1)*K
 *     out_im sized (N/2+1)*K
 *
 *   Roundtrip: c2r(r2c(x)) = N * x  (caller divides for normalization)
 * ─────────────────────────────────────────────────────────────────── */

static void demo_r2c(unsigned flags)
{
    section("1D R2C/C2R  N=512 K=8");

    const int N = 512;
    const size_t K = 8;
    const size_t Nfreq = (size_t)(N / 2 + 1);

    vfft_plan p = vfft_plan_r2c(N, K, flags);
    if (!p)
    {
        fprintf(stderr, "  vfft_plan_r2c failed\n");
        return;
    }

    double *real_in = (double *)vfft_alloc((size_t)N * K * sizeof(double));
    double *real_orig = (double *)vfft_alloc((size_t)N * K * sizeof(double));
    double *real_out = (double *)vfft_alloc((size_t)N * K * sizeof(double));
    double *bins_re = (double *)vfft_alloc((size_t)N * K * sizeof(double)); /* workspace */
    double *bins_im = (double *)vfft_alloc(Nfreq * K * sizeof(double));

    fill_random(real_in, (size_t)N * K, 100);
    memcpy(real_orig, real_in, (size_t)N * K * sizeof(double));

    vfft_execute_r2c(p, real_in, bins_re, bins_im);
    vfft_execute_c2r(p, bins_re, bins_im, real_out);

    double inv_N = 1.0 / (double)N;
    for (size_t i = 0; i < (size_t)N * K; i++)
        real_out[i] *= inv_N;

    double err = max_abs_diff(real_out, real_orig, (size_t)N * K);
    printf("  r2c + c2r/N roundtrip:            err=%.2e\n", err);

    vfft_free(real_in);
    vfft_free(real_orig);
    vfft_free(real_out);
    vfft_free(bins_re);
    vfft_free(bins_im);
    vfft_destroy(p);
}

/* ───────────────────────────────────────────────────────────────────
 * SECTION 3 — 2D Complex (C2C)
 *
 *   Plan:    vfft_plan_2d(N1, N2, flags)
 *   Layout:  re[i*N2 + j]   for i=0..N1-1, j=0..N2-1
 *   Roundtrip: bwd(fwd(x)) = N1*N2 * x
 * ─────────────────────────────────────────────────────────────────── */

static void demo_2d_c2c(unsigned flags)
{
    section("2D C2C  256x256");

    const int N1 = 256, N2 = 256;
    const size_t total = (size_t)N1 * N2;

    vfft_plan p = vfft_plan_2d(N1, N2, flags);
    if (!p)
    {
        fprintf(stderr, "  vfft_plan_2d failed\n");
        return;
    }

    double *re = (double *)vfft_alloc(total * sizeof(double));
    double *im = (double *)vfft_alloc(total * sizeof(double));
    double *re0 = (double *)vfft_alloc(total * sizeof(double));
    double *im0 = (double *)vfft_alloc(total * sizeof(double));

    fill_random(re, total, 7);
    fill_random(im, total, 8);
    memcpy(re0, re, total * sizeof(double));
    memcpy(im0, im, total * sizeof(double));

    vfft_execute_fwd(p, re, im);
    vfft_execute_bwd(p, re, im);

    double inv = 1.0 / ((double)N1 * (double)N2);
    for (size_t i = 0; i < total; i++)
    {
        re[i] *= inv;
        im[i] *= inv;
    }

    double err = fmax(max_abs_diff(re, re0, total), max_abs_diff(im, im0, total));
    printf("  fwd + bwd / (N1*N2) roundtrip:    err=%.2e\n", err);

    vfft_free(re);
    vfft_free(im);
    vfft_free(re0);
    vfft_free(im0);
    vfft_destroy(p);
}

/* ───────────────────────────────────────────────────────────────────
 * SECTION 4 — 2D Real-to-Complex / Complex-to-Real (R2C/C2R)
 *
 *   Plan:    vfft_plan_2d_r2c(N1, N2, flags)
 *   Layout:  real[i*N2 + j]                   for i=0..N1-1, j=0..N2-1
 *            bins_re[i*(N2/2+1) + f]          for i=0..N1-1, f=0..N2/2
 *   Constraint: N2 even.
 * ─────────────────────────────────────────────────────────────────── */

static void demo_2d_r2c(unsigned flags)
{
    section("2D R2C/C2R  256x256");

    const int N1 = 256, N2 = 256;
    const size_t freq2 = (size_t)(N2 / 2 + 1);
    const size_t total_real = (size_t)N1 * N2;
    const size_t total_freq = (size_t)N1 * freq2;

    vfft_plan p = vfft_plan_2d_r2c(N1, N2, flags);
    if (!p)
    {
        fprintf(stderr, "  vfft_plan_2d_r2c failed\n");
        return;
    }

    double *real_in = (double *)vfft_alloc(total_real * sizeof(double));
    double *real_orig = (double *)vfft_alloc(total_real * sizeof(double));
    double *real_out = (double *)vfft_alloc(total_real * sizeof(double));
    double *bins_re = (double *)vfft_alloc(total_freq * sizeof(double));
    double *bins_im = (double *)vfft_alloc(total_freq * sizeof(double));

    fill_random(real_in, total_real, 11);
    memcpy(real_orig, real_in, total_real * sizeof(double));

    vfft_execute_2d_r2c(p, real_in, bins_re, bins_im);
    vfft_execute_2d_c2r(p, bins_re, bins_im, real_out);

    double inv = 1.0 / ((double)N1 * (double)N2);
    for (size_t i = 0; i < total_real; i++)
        real_out[i] *= inv;

    double err = max_abs_diff(real_out, real_orig, total_real);
    printf("  2d_r2c + 2d_c2r / (N1*N2) error:  %.2e\n", err);

    vfft_free(real_in);
    vfft_free(real_orig);
    vfft_free(real_out);
    vfft_free(bins_re);
    vfft_free(bins_im);
    vfft_destroy(p);
}

/* ───────────────────────────────────────────────────────────────────
 * SECTION 5 — DCT-II / DCT-III (Makhoul, JPEG/MPEG/AAC)
 *
 *   FFTW conventions: REDFT10 / REDFT01.
 *   DCT-III is the inverse of DCT-II up to scale 2N:
 *       For y = vfft_execute_dct2(x), x = vfft_execute_dct3(y) / (2N).
 *   Constraint: N even.
 * ─────────────────────────────────────────────────────────────────── */

static void demo_dct2_dct3(unsigned flags)
{
    section("DCT-II / DCT-III  N=1024 K=4");

    const int N = 1024;
    const size_t K = 4;
    const size_t NK = (size_t)N * K;

    vfft_plan p = vfft_plan_dct2(N, K, flags);
    if (!p)
    {
        fprintf(stderr, "  vfft_plan_dct2 failed\n");
        return;
    }

    double *in = (double *)vfft_alloc(NK * sizeof(double));
    double *orig = (double *)vfft_alloc(NK * sizeof(double));
    double *mid = (double *)vfft_alloc(NK * sizeof(double));
    double *out = (double *)vfft_alloc(NK * sizeof(double));

    fill_random(in, NK, 21);
    memcpy(orig, in, NK * sizeof(double));

    vfft_execute_dct2(p, in, mid);  /* forward  */
    vfft_execute_dct3(p, mid, out); /* inverse  */

    double inv = 1.0 / (2.0 * N);
    for (size_t i = 0; i < NK; i++)
        out[i] *= inv;

    double err = max_abs_diff(out, orig, NK);
    printf("  dct3(dct2(x)) / (2N) error:       %.2e\n", err);

    vfft_free(in);
    vfft_free(orig);
    vfft_free(mid);
    vfft_free(out);
    vfft_destroy(p);
}

/* ───────────────────────────────────────────────────────────────────
 * SECTION 6 — DCT-IV (Lee 1984, MDCT for MP3/AAC/Vorbis/Opus)
 *
 *   FFTW convention: REDFT11. Involutory up to scale 2N:
 *       y = dct4(x);  x = dct4(y) / (2N).
 *   Constraint: N even.
 * ─────────────────────────────────────────────────────────────────── */

static void demo_dct4(unsigned flags)
{
    section("DCT-IV  N=1024 K=4");

    const int N = 1024;
    const size_t K = 4;
    const size_t NK = (size_t)N * K;

    vfft_plan p = vfft_plan_dct4(N, K, flags);
    if (!p)
    {
        fprintf(stderr, "  vfft_plan_dct4 failed\n");
        return;
    }

    double *in = (double *)vfft_alloc(NK * sizeof(double));
    double *orig = (double *)vfft_alloc(NK * sizeof(double));
    double *mid = (double *)vfft_alloc(NK * sizeof(double));
    double *out = (double *)vfft_alloc(NK * sizeof(double));

    fill_random(in, NK, 22);
    memcpy(orig, in, NK * sizeof(double));

    vfft_execute_dct4(p, in, mid);
    vfft_execute_dct4(p, mid, out);

    double inv = 1.0 / (2.0 * N);
    for (size_t i = 0; i < NK; i++)
        out[i] *= inv;

    double err = max_abs_diff(out, orig, NK);
    printf("  dct4(dct4(x)) / (2N) error:       %.2e\n", err);

    vfft_free(in);
    vfft_free(orig);
    vfft_free(mid);
    vfft_free(out);
    vfft_destroy(p);
}

/* ───────────────────────────────────────────────────────────────────
 * SECTION 7 — DST-II / DST-III
 *
 *   FFTW conventions: RODFT10 / RODFT01.
 *   DST-III inverts DST-II up to scale 2N. Constraint: N even.
 * ─────────────────────────────────────────────────────────────────── */

static void demo_dst2_dst3(unsigned flags)
{
    section("DST-II / DST-III  N=1024 K=4");

    const int N = 1024;
    const size_t K = 4;
    const size_t NK = (size_t)N * K;

    vfft_plan p = vfft_plan_dst2(N, K, flags);
    if (!p)
    {
        fprintf(stderr, "  vfft_plan_dst2 failed\n");
        return;
    }

    double *in = (double *)vfft_alloc(NK * sizeof(double));
    double *orig = (double *)vfft_alloc(NK * sizeof(double));
    double *mid = (double *)vfft_alloc(NK * sizeof(double));
    double *out = (double *)vfft_alloc(NK * sizeof(double));

    fill_random(in, NK, 23);
    memcpy(orig, in, NK * sizeof(double));

    vfft_execute_dst2(p, in, mid);
    vfft_execute_dst3(p, mid, out);

    double inv = 1.0 / (2.0 * N);
    for (size_t i = 0; i < NK; i++)
        out[i] *= inv;

    double err = max_abs_diff(out, orig, NK);
    printf("  dst3(dst2(x)) / (2N) error:       %.2e\n", err);

    vfft_free(in);
    vfft_free(orig);
    vfft_free(mid);
    vfft_free(out);
    vfft_destroy(p);
}

/* ───────────────────────────────────────────────────────────────────
 * SECTION 8 — DHT (Discrete Hartley Transform)
 *
 *   FFTW convention: FFTW_DHT. Self-inverse up to 1/N:
 *       y = dht(x);   x = dht(y) / N.
 *   Constraint: N even.
 * ─────────────────────────────────────────────────────────────────── */

static void demo_dht(unsigned flags)
{
    section("DHT  N=1024 K=4");

    const int N = 1024;
    const size_t K = 4;
    const size_t NK = (size_t)N * K;

    vfft_plan p = vfft_plan_dht(N, K, flags);
    if (!p)
    {
        fprintf(stderr, "  vfft_plan_dht failed\n");
        return;
    }

    double *in = (double *)vfft_alloc(NK * sizeof(double));
    double *orig = (double *)vfft_alloc(NK * sizeof(double));
    double *mid = (double *)vfft_alloc(NK * sizeof(double));
    double *out = (double *)vfft_alloc(NK * sizeof(double));

    fill_random(in, NK, 24);
    memcpy(orig, in, NK * sizeof(double));

    vfft_execute_dht(p, in, mid);
    vfft_execute_dht(p, mid, out);

    double inv = 1.0 / (double)N;
    for (size_t i = 0; i < NK; i++)
        out[i] *= inv;

    double err = max_abs_diff(out, orig, NK);
    printf("  dht(dht(x)) / N error:            %.2e\n", err);

    vfft_free(in);
    vfft_free(orig);
    vfft_free(mid);
    vfft_free(out);
    vfft_destroy(p);
}

/* ───────────────────────────────────────────────────────────────────
 * SECTION 9 — Threading and interleaved-array conversion
 *
 *   vfft_set_num_threads(n)      — global. Call before plan creation.
 *   vfft_get_num_threads()       — query current.
 *   vfft_deinterleave / reinterleave — bridge {re,im,re,im,...} <→ split.
 * ─────────────────────────────────────────────────────────────────── */

static void demo_threading_and_convert(void)
{
    section("Threading + interleave/deinterleave");

    int initial = vfft_get_num_threads();
    printf("  initial num_threads:              %d\n", initial);

    vfft_set_num_threads(4);
    printf("  after vfft_set_num_threads(4):    %d\n", vfft_get_num_threads());

    /* Roundtrip a small interleaved buffer through split-complex. */
    const size_t count = 8;
    double interleaved[16] = {
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    };
    double re[8], im[8], roundtrip[16];
    vfft_deinterleave(interleaved, re, im, count);
    vfft_reinterleave(re, im, roundtrip, count);
    int ok = memcmp(interleaved, roundtrip, sizeof(interleaved)) == 0;
    printf("  deinterleave + reinterleave:      %s\n", ok ? "ok" : "MISMATCH");

    /* Restore default (single-threaded) — important so later demos don't
     * inherit a thread count the user didn't set. */
    vfft_set_num_threads(initial > 0 ? initial : 1);
}

/* ───────────────────────────────────────────────────────────────────
 * SECTION 10 — Wisdom lifecycle
 *
 *   vfft_load_wisdom(path)   — merge wisdom from disk into the in-memory db.
 *   vfft_save_wisdom(path)   — write the in-memory db to disk.
 *   vfft_forget_wisdom()     — clear in-memory db (next MEASURE recalibrates).
 *
 *   The library does NOT auto-load any wisdom file. Users opt in.
 *   Bundled sample: examples/14900KF/wisdom.txt (Intel i9-14900KF).
 *
 *   Returns 0 on success.
 * ─────────────────────────────────────────────────────────────────── */

static int try_load_wisdom(void)
{
    /* Resolution order:
     *   1. $VFFT_WISDOM environment variable (user override)
     *   2. VFFT_WISDOM_DEFAULT_PATH compile-time define (set by CMake to
     *      the absolute path of build_tuned/vfft_wisdom_tuned.txt)
     *   3. Common cwd-relative spots, in case the binary is run from
     *      either the repo root or a build subdir without the define.
     */
    const char *candidates[] = {
        getenv("VFFT_WISDOM"),  /* may be NULL — skip, don't terminate */
#ifdef VFFT_WISDOM_DEFAULT_PATH
        VFFT_WISDOM_DEFAULT_PATH,
#endif
        /* Packaged sample wisdom (per-host, see examples/14900KF/README.md). */
        "examples/14900KF/wisdom.txt",
        "../examples/14900KF/wisdom.txt",
        "../../examples/14900KF/wisdom.txt",
        /* Live calibration output (developer location). */
        "build_tuned/vfft_wisdom_tuned.txt",
        "../build_tuned/vfft_wisdom_tuned.txt",
        "../../build_tuned/vfft_wisdom_tuned.txt",
    };
    const int n = (int)(sizeof(candidates) / sizeof(candidates[0]));
    for (int i = 0; i < n; i++)
    {
        if (!candidates[i]) continue;
        if (vfft_load_wisdom(candidates[i]) == 0)
        {
            printf("[wisdom] loaded from %s\n", candidates[i]);
            return 0;
        }
    }
    printf("[wisdom] none found — using VFFT_ESTIMATE for all plans\n");
    return -1;
}

/* ───────────────────────────────────────────────────────────────────
 * MAIN
 * ─────────────────────────────────────────────────────────────────── */

int main(void)
{
    /* Init: sets FTZ/DAZ, prepares the global codelet registry. Idempotent. */
    vfft_init();

    /* Pin the calling thread to a P-core for stable numbers in this demo.
     * Optional in user code; here it makes timing reproducible. */
    vfft_pin_thread(0);

    printf("VectorFFT %s  ISA: %s\n", vfft_version(), vfft_isa());

    /* All demos below use VFFT_ESTIMATE (the closed-form cost-model path).
     * Plans build in microseconds and land within ~1.3× of measured wisdom.
     * To switch to VFFT_MEASURE, load wisdom first (see SECTION 10 below)
     * and change `flags` here — but note that mixed MEASURE plans across
     * different transforms exposes a known v1.0 bug under MinGW GCC; use
     * MSVC or ICX if you need MEASURE end-to-end on Windows. */
    unsigned flags = VFFT_ESTIMATE;

    demo_1d_c2c(flags);
    demo_r2c(flags);
    demo_2d_c2c(flags);
    demo_2d_r2c(flags);
    demo_dct2_dct3(flags);
    demo_dct4(flags);
    demo_dst2_dst3(flags);
    demo_dht(flags);
    demo_threading_and_convert();

    /* SECTION 10 — Wisdom lifecycle (load / save / forget).
     * Run after the demos to show the API surface without affecting them. */
    section("Wisdom lifecycle");
    int has_wisdom = (try_load_wisdom() == 0);
    if (has_wisdom)
    {
        vfft_save_wisdom("vfft_wisdom_quickstart.txt");
        printf("[wisdom] saved current db to vfft_wisdom_quickstart.txt\n");
        vfft_forget_wisdom();
        printf("[wisdom] forgot in-memory db (next MEASURE recalibrates)\n");
    }

    printf("\nAll demos completed.\n");
    return 0;
}
