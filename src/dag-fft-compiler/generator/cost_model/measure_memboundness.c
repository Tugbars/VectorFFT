/* measure_memboundness.c — sibling to measure_cpe.c
 *
 * Per-radix cache-tier sensitivity for "large" codelets (R ≥ 16).
 *
 * The cost model's V4 wide_penalty term assumed a linear (R-4) load-port
 * penalty for wide codelets when their working set spills L1. That heuristic
 * is now wrong (post-c18b7c1 the wide codelets are spill-free; per-call CPE
 * is roughly flat across me-samples in radix_cpe.h). But it remained
 * load-bearing because it canceled overweighted buffer_pass_per_stage.
 *
 * This tool replaces the heuristic with a measurement: for each large
 * codelet R, bench it at five working-set sizes corresponding to L1 / L2 /
 * L3 / DRAM-resident data, and emit a per-(R, tier) inflation factor:
 *
 *     memboundness[R][tier] = cyc_tier / cyc_L1
 *
 * The cost model reads this table directly and multiplies the per-stage
 * data_cost by memboundness[R][stage_tier]. No more wide_penalty.
 *
 * Scope: ONLY large radixes are measured here. Small radixes (R ≤ 8) are
 * not the regime where wide_penalty fires; their existing CPE-table
 * sensitivity is fine.
 *
 * Output: src/prototype/cost_model/generated/radix_memboundness.h with the
 * inflation table. Same emit shape as radix_cpe.h.
 *
 * Run via:
 *   bash cost_model/build_measure_memboundness.sh
 *   build_tuned/measure_memboundness
 *
 * Recommended host state: same as measure_cpe (pinned core, performance
 * governor, no other significant load).
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

#include "generated/radix_dims.h"

/* ─────────────────────────── tunables ─────────────────────────── */

/* me-per-call is fixed at the K-axis size we want to model. In real plan
 * executors me = K_batch (the batch axis). 4 matches our pow2 K=4 test cells.
 * If you care about K=256 behavior, build with -DBENCH_ME=256. */
#ifndef BENCH_ME
#define BENCH_ME            4
#endif

/* ios is what we sweep — it's the codelet's stride between R legs, which is
 * what makes the working set land in different cache tiers. Working set per
 * call ≈ R × ios × 16 bytes. */
#define BENCH_N_RUNS       21
#define BENCH_TARGET_NS    10000000.0  /* ~10 ms per batch */
#define BENCH_CALIB_NS     100000.0
#define MAX_CV_DEFAULT     0.10        /* memboundness measurements at L3/DRAM
                                        * are inherently noisier than L1 CPE;
                                        * relax the CV gate accordingly. */

/* Cache-tier working-set TARGETS (bytes per call's data + twiddle footprint).
 * Each value is chosen well inside its tier on Raptor Lake to avoid boundary
 * effects:
 *   L1=48KB cap → measure at 16 KB
 *   L2=2MB cap  → measure at 512 KB
 *   L3=24MB cap → measure at 8 MB
 *   DRAM        → measure at 64 MB (clearly spilling L3) */
typedef enum {
    MB_TIER_L1   = 0,
    MB_TIER_L2   = 1,
    MB_TIER_L3   = 2,
    MB_TIER_DRAM = 3,
    MB_N_TIERS   = 4,
} mb_tier_t;

static const size_t MB_TIER_TARGET_BYTES[MB_N_TIERS] = {
    16ull   * 1024,           /* L1   */
    512ull  * 1024,           /* L2   */
    8ull    * 1024 * 1024,    /* L3   */
    64ull   * 1024 * 1024,    /* DRAM */
};

static const char *MB_TIER_NAME[MB_N_TIERS] = { "L1", "L2", "L3", "DRAM" };

/* Buffer cap (DRAM tier is largest, ~64 MB rio + 64 MB tw = ~256 MB worst
 * case for R=128 + me=131072 — too big. We cap per-tier and refuse to
 * measure cells that would exceed the cap, falling back to NaN). */
#define MB_BUFFER_CAP_BYTES (256ull * 1024 * 1024)

/* ─────────────────────── ISA selection ───────────────────────── */

#if defined(VFFT_ISA_AVX512)
  #define ISA_LABEL  "avx512"
  #define USE_AVX512 1
#else
  #define ISA_LABEL  "avx2"
  #define USE_AVX2   1
#endif

/* ─────────────── codelet dispatch (registry-based) ─────────────── */

typedef void (*codelet_fn)(double *, double *,
                           const double *, const double *,
                           size_t, size_t);

#include "../generated/registry.h"

typedef struct {
    int R;
    codelet_fn t1s_fn;
} radix_entry_t;

/* The "large" codelet set — where wide_penalty was meant to fire. R<16
 * uses the existing CPE table without inflation correction. */
static const int RADIX_LIST[] = { 16, 32, 64, 128, 256 };
#define N_RADIX (sizeof(RADIX_LIST) / sizeof(RADIX_LIST[0]))

static radix_entry_t   RADIX_TABLE[N_RADIX];
static vfft_proto_registry_t g_reg;

static void init_radix_table(void) {
#if defined(USE_AVX512)
    vfft_proto_registry_init_avx512(&g_reg);
#else
    vfft_proto_registry_init_avx2(&g_reg);
#endif
    for (size_t i = 0; i < N_RADIX; i++) {
        int R = RADIX_LIST[i];
        RADIX_TABLE[i].R       = R;
        RADIX_TABLE[i].t1s_fn  = (codelet_fn)g_reg.t1s_dit_fwd[R];
    }
}

/* ─────────────────────── timing plumbing ─────────────────────────── */

#ifdef _WIN32
static double g_qpc_freq;
static double now_ns(void) {
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / g_qpc_freq;
}
static void timer_init(void) {
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    g_qpc_freq = (double)f.QuadPart;
}
#else
static void timer_init(void) {}
static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
#endif

static double measure_freq_ghz(void) {
    (void)__rdtsc();
    double t0 = now_ns();
    uint64_t r0 = __rdtsc();
    while (now_ns() - t0 < 50e6) { }
    uint64_t r1 = __rdtsc();
    double t1 = now_ns();
    return (double)(r1 - r0) / (t1 - t0);
}

/* ───────────────────────── aligned alloc ───────────────────────── */

static void *xaligned_alloc(size_t bytes) {
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
static void xaligned_free(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ───────────────────────── stats helpers ───────────────────────── */

static int dcmp(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}
static double dmedian(double *xs, int n) {
    qsort(xs, n, sizeof(double), dcmp);
    return (n % 2) ? xs[n / 2] : 0.5 * (xs[n / 2 - 1] + xs[n / 2]);
}
static double dmean(const double *xs, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += xs[i];
    return s / (double)n;
}
static double dstddev(const double *xs, int n, double mean) {
    double s = 0;
    for (int i = 0; i < n; i++) { double d = xs[i] - mean; s += d * d; }
    return sqrt(s / (double)n);
}
static double dcv(double *xs, int n) {
    double m = dmean(xs, n);
    if (m <= 0) return 1e9;
    double tmp[64];
    if (n > 64) n = 64;
    memcpy(tmp, xs, n * sizeof(double));
    double sd = dstddev(tmp, n, m);
    return sd / m;
}

/* ───────────────── core bench loop (mirrors measure_cpe) ────────── */

static int calibrate_iters(codelet_fn fn,
                           double *rio_re, double *rio_im,
                           const double *tw_re, const double *tw_im,
                           size_t ios, size_t me, double target_ns)
{
    int n = 1;
    for (;;) {
        double t0 = now_ns();
        for (int i = 0; i < n; i++) fn(rio_re, rio_im, tw_re, tw_im, ios, me);
        double dt = now_ns() - t0;
        if (dt >= BENCH_CALIB_NS) {
            double per = dt / (double)n;
            int iters = (int)(target_ns / per);
            if (iters < 1) iters = 1;
            return iters;
        }
        n *= 2;
        if (n > (1 << 28)) return n;
    }
}

/* ios derivation: pick ios such that the per-call working set
 * (R × ios × 16 bytes) lands in the target tier. me stays fixed at BENCH_ME
 * (the plan-context K-axis size). Round to a multiple of 16 for alignment. */
static size_t ios_for_tier(int R, size_t target_bytes)
{
    /* Working set per call ≈ R × ios × 16 bytes (data; twiddle table is
     * R × me × 16 bytes = small constant). */
    size_t ios = target_bytes / (size_t)(R * 16);
    if (ios < 16) ios = 16;
    ios = (ios / 16) * 16;
    return ios;
}

/* Target total buffer size for the multi-group sweep. Should be large
 * enough to span the per-call working set many times, mimicking how the
 * plan executor sweeps groups across a multi-megabyte FFT dataset.
 *
 * 16 MB sits below typical L3 (24-36 MB on Raptor), so the sweep stays
 * L3-resident overall — matching the regime our test cells operate in.
 * Per-call working set varies by tier target; n_groups falls out of the
 * total/per_call ratio. */
#define MB_SWEEP_BUFFER_BYTES   (16ull * 1024 * 1024)
#define MB_SWEEP_MIN_GROUPS     4

static double bench_at_tier(codelet_fn fn, int R, size_t target_bytes,
                            size_t *out_ios, double *cv_out)
{
    *out_ios = 0;
    *cv_out = 0.0;
    if (fn == NULL) return 0.0;

    size_t me = BENCH_ME;
    size_t ios = ios_for_tier(R, target_bytes);
    if (ios == 0) return 0.0;
    *out_ios = ios;

    /* Per-group memory footprint, in doubles. Each group's rio access
     * spans (R-1)*ios + me doubles for input (and same for output, in-
     * place). Plus a small "stride between groups" so adjacent groups
     * don't overlap their working sets. */
    size_t per_group_doubles = (size_t)(R - 1) * ios + me + 64;

    /* How many groups fit in the sweep buffer? */
    size_t total_doubles_per_part =
        MB_SWEEP_BUFFER_BYTES / sizeof(double);
    size_t n_groups = total_doubles_per_part / per_group_doubles;
    if (n_groups < MB_SWEEP_MIN_GROUPS) n_groups = MB_SWEEP_MIN_GROUPS;
    /* But also clamp so total buffer doesn't blow the cap. */
    size_t alloc_doubles_per_part = n_groups * per_group_doubles + 64;
    if (alloc_doubles_per_part * sizeof(double) * 2 + 2 * (size_t)R * me * sizeof(double)
        > MB_BUFFER_CAP_BYTES)
    {
        /* Shrink to fit. */
        size_t max_doubles = (MB_BUFFER_CAP_BYTES - 2 * (size_t)R * me * sizeof(double))
                             / (2 * sizeof(double));
        n_groups = (max_doubles - 64) / per_group_doubles;
        if (n_groups < MB_SWEEP_MIN_GROUPS) return 0.0;
        alloc_doubles_per_part = n_groups * per_group_doubles + 64;
    }

    size_t tw_span = (size_t)R * me;

    double *rio_re = xaligned_alloc(alloc_doubles_per_part * sizeof(double));
    double *rio_im = xaligned_alloc(alloc_doubles_per_part * sizeof(double));
    double *tw_re  = xaligned_alloc((tw_span + 64) * sizeof(double));
    double *tw_im  = xaligned_alloc((tw_span + 64) * sizeof(double));

    /* Fill the rio buffer with non-trivial values so the compiler can't
     * elide the work. Same twiddle layout as the single-group bench. */
    for (size_t i = 0; i < alloc_doubles_per_part; i++) {
        rio_re[i] = 1.0;
        rio_im[i] = -1.0;
    }
    for (int j = 1; j < R; j++) {
        for (size_t k = 0; k < me; k++) {
            double th = -2.0 * 3.14159265358979323846 *
                        (double)((size_t)j * k) / (double)((size_t)R * me);
            tw_re[(size_t)(j - 1) * me + k] = cos(th);
            tw_im[(size_t)(j - 1) * me + k] = sin(th);
        }
    }

    /* Calibrate iters such that one OUTER iter (sweeping all n_groups)
     * takes ~10 ms. Probe with one sweep, scale from there. */
    double t_probe = now_ns();
    for (size_t g = 0; g < n_groups; g++) {
        size_t off = g * per_group_doubles;
        fn(rio_re + off, rio_im + off, tw_re, tw_im, ios, me);
    }
    double sweep_ns = now_ns() - t_probe;
    if (sweep_ns < 1.0) sweep_ns = 1.0;
    int outer_iters = (int)(BENCH_TARGET_NS / sweep_ns);
    if (outer_iters < 1) outer_iters = 1;
    if (outer_iters > 10000) outer_iters = 10000;

    double samples[BENCH_N_RUNS];
    for (int run = 0; run < BENCH_N_RUNS; run++) {
        double t0 = now_ns();
        for (int it = 0; it < outer_iters; it++) {
            for (size_t g = 0; g < n_groups; g++) {
                size_t off = g * per_group_doubles;
                fn(rio_re + off, rio_im + off, tw_re, tw_im, ios, me);
            }
        }
        double dt = now_ns() - t0;
        /* dt is wall-time for outer_iters × n_groups calls. */
        samples[run] = dt / ((double)outer_iters * (double)n_groups);
    }

    *cv_out = dcv(samples, BENCH_N_RUNS);
    double med = dmedian(samples, BENCH_N_RUNS);

    xaligned_free(rio_re); xaligned_free(rio_im);
    xaligned_free(tw_re);  xaligned_free(tw_im);
    return med;
}

/* ───────────────── memory bandwidth bench ──────────────────────────
 *
 * V4's `buffer_pass_per_stage` term assumed `2 × N×K×16 / bytes_per_cycle`
 * where bytes_per_cycle was hardcoded from spec sheets (L1=45, L2=14,
 * L3=7, DRAM=9). Real measured bandwidth differs — cache coherency
 * overhead, prefetcher behavior, and stride pattern all affect it.
 *
 * For each cache tier T, allocate a buffer of size T_target, run a tight
 * stride-1 read+write loop over it, measure cycles per byte. The result
 * is what V4 should use as its per-stage memory cost coefficient.
 *
 * Stride-1 is the simplest realistic access pattern — most FFT stages
 * read inputs and write outputs sequentially within their working set.
 * Strided patterns are slower and would yield a different (worse)
 * number; for V4's per-stage charge, stride-1 is the right approximation
 * since the stage's TOTAL byte traffic is what matters, not its access
 * pattern. */

static double bench_memory_pass(size_t target_bytes)
{
    /* Cap buffer at 32 MB. On Raptor Lake L3 ~36 MB, so 32 MB straddles
     * L3 capacity and gives a representative DRAM-ish measurement
     * without risking allocation pressure. */
    size_t nbytes = target_bytes;
    if (nbytes > 32ull * 1024 * 1024) nbytes = 32ull * 1024 * 1024;
    size_t nelem = nbytes / sizeof(double);
    if (nelem < 64) return 0.0;

    double *src = xaligned_alloc(nbytes);
    double *dst = xaligned_alloc(nbytes);
    for (size_t i = 0; i < nelem; i++) { src[i] = (double)i; dst[i] = 0.0; }

    /* Warmup with memcpy — well-optimized for all tiers. */
    for (int w = 0; w < 3; w++) memcpy(dst, src, nbytes);

    /* Pick iteration count from a single-call probe — way more reliable
     * than estimating bandwidth blindly. */
    double t_probe = now_ns();
    memcpy(dst, src, nbytes);
    double per_copy_ns = now_ns() - t_probe;
    if (per_copy_ns < 1.0) per_copy_ns = 1.0;
    int iters = (int)(5.0e6 / per_copy_ns);   /* aim for ~5 ms per run */
    if (iters < 2)     iters = 2;
    if (iters > 50000) iters = 50000;

    /* 7 runs — enough for a stable median, fast enough to stay sub-second. */
    enum { BANDWIDTH_RUNS = 7 };
    double samples[BANDWIDTH_RUNS];
    for (int run = 0; run < BANDWIDTH_RUNS; run++) {
        double t0 = now_ns();
        for (int k = 0; k < iters; k++) memcpy(dst, src, nbytes);
        double dt = now_ns() - t0;
        samples[run] = dt / (double)iters;
    }
    double med_ns = dmedian(samples, BANDWIDTH_RUNS);

    xaligned_free(src);
    xaligned_free(dst);

    /* med_ns is wall-time for ONE memcpy pass through `nbytes` bytes
     * of data. Bytes moved = 2 × nbytes (one read + one write). Return
     * ns per byte; the caller converts to cycles via the freq. */
    return med_ns / (2.0 * (double)nbytes);
}

/* ─────────────────────────── results ───────────────────────────── */

typedef struct {
    int    R;
    double ns_tier  [MB_N_TIERS];   /* ns per call at each tier */
    size_t ios_tier [MB_N_TIERS];   /* actual ios (stride) used at each tier */
    double cv_tier  [MB_N_TIERS];
    double cyc_tier [MB_N_TIERS];   /* derived: ns × freq */
    double factor   [MB_N_TIERS];   /* derived: cyc_tier / cyc_tier[L1] */
} row_t;

/* ───────────────────────── header emit ─────────────────────────── */

static void emit_header(const char *path, const row_t *rows, int nrows,
                        const double *tier_cyc_per_byte,
                        double freq_ghz, const char *isa_label)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "open %s failed\n", path); exit(1); }

    fprintf(f,
"/* radix_memboundness.h — auto-generated by measure_memboundness.\n"
" * DO NOT EDIT BY HAND. Regenerate via measure_memboundness on a quiet host.\n"
" *\n"
" * Per-radix per-cache-tier CPE inflation factor for large codelets.\n"
" * Replaces V4 cost-model's wide_penalty heuristic.\n"
" *\n"
" * Index: stride_radix_memboundness_{isa}[R][tier] = cyc_at_tier / cyc_at_L1\n"
" * Tiers: 0=L1, 1=L2, 2=L3, 3=DRAM.\n"
" *\n"
" * 1.0 means the codelet is unaffected by data tier (compute-bound).\n"
" * >1.0 means the codelet slows down when data is at that tier.\n"
" *\n"
" * Effective freq:  %.3f GHz (RDTSC / 50 ms wall)\n"
" * ISA tag:         %s\n"
" * Source samples:  bench_n_runs=%d, target=%.0f ms/batch\n"
" *\n"
" * Small radixes (R ≤ 8) are NOT in this table — they use the standard\n"
" * radix_cpe.h sensitivity (no separate memboundness correction needed).\n"
" */\n"
"#ifndef STRIDE_RADIX_MEMBOUNDNESS_H\n"
"#define STRIDE_RADIX_MEMBOUNDNESS_H\n\n"
"#include \"radix_dims.h\"\n\n"
"#define STRIDE_MEMBOUNDNESS_N_TIERS 4\n\n"
"/* Tier IDs match the enum in measure_memboundness.c. */\n"
"#define STRIDE_MB_TIER_L1   0\n"
"#define STRIDE_MB_TIER_L2   1\n"
"#define STRIDE_MB_TIER_L3   2\n"
"#define STRIDE_MB_TIER_DRAM 3\n\n"
"typedef struct {\n"
"    double factor[STRIDE_MEMBOUNDNESS_N_TIERS];  /* 1.0 = unaffected; >1 = slowdown */\n"
"} stride_radix_memboundness_t;\n\n",
        freq_ghz, isa_label,
        BENCH_N_RUNS, BENCH_TARGET_NS / 1.0e6);

    fprintf(f, "static const stride_radix_memboundness_t stride_radix_memboundness_%s[STRIDE_RADIX_PROFILE_MAX_R] = {\n",
            isa_label);
    /* All radixes not in our list default to factor=1.0 (no inflation). */
    for (int i = 0; i < nrows; i++) {
        fprintf(f, "    [%4d] = { .factor = { %.3f, %.3f, %.3f, %.3f } },\n",
                rows[i].R,
                rows[i].factor[MB_TIER_L1],
                rows[i].factor[MB_TIER_L2],
                rows[i].factor[MB_TIER_L3],
                rows[i].factor[MB_TIER_DRAM]);
    }
    fprintf(f, "};\n\n");

    /* Companion stub for the other ISA. */
    const char *other_isa = (strcmp(isa_label, "avx2") == 0) ? "avx512" : "avx2";
    fprintf(f, "/* Other-ISA companion (empty until measure_memboundness re-run with -DVFFT_ISA_AVX512). */\n");
    fprintf(f, "static const stride_radix_memboundness_t stride_radix_memboundness_%s[STRIDE_RADIX_PROFILE_MAX_R] = {0};\n\n",
            other_isa);

    /* ─── cache_bandwidth: measured cycles/byte per cache tier ────────
     * Replaces V4's hardcoded `45/14/7/9 bytes/cyc` constants in the
     * buffer_pass_per_stage computation. The cost model reads this
     * directly to convert (per-stage bytes touched) → cycles. */
    fprintf(f,
"/* Measured memory-pass cost per cache tier. Used by V4's\n"
" * buffer_pass_per_stage term: stage cycles = stage_bytes * cyc_per_byte[tier].\n"
" *\n"
" * Numbers are stride-1 read+write bandwidth on a buffer sized for the\n"
" * target tier. Reality may differ for strided access patterns; this is\n"
" * the lower-bound per-byte cost the codelet's data flow has to pay.\n"
" */\n"
"static const double stride_cache_cyc_per_byte_%s[STRIDE_MEMBOUNDNESS_N_TIERS] = {\n"
"    [STRIDE_MB_TIER_L1]   = %.5f,\n"
"    [STRIDE_MB_TIER_L2]   = %.5f,\n"
"    [STRIDE_MB_TIER_L3]   = %.5f,\n"
"    [STRIDE_MB_TIER_DRAM] = %.5f,\n"
"};\n\n",
        isa_label,
        tier_cyc_per_byte[MB_TIER_L1],
        tier_cyc_per_byte[MB_TIER_L2],
        tier_cyc_per_byte[MB_TIER_L3],
        tier_cyc_per_byte[MB_TIER_DRAM]);

    fprintf(f, "static const double stride_cache_cyc_per_byte_%s[STRIDE_MEMBOUNDNESS_N_TIERS] = {0};\n\n",
            other_isa);

    fprintf(f, "#endif /* STRIDE_RADIX_MEMBOUNDNESS_H */\n");
    fclose(f);
}

/* ───────────────────────────── main ───────────────────────────── */

int main(int argc, char **argv)
{
    int force = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--force") == 0) force = 1;
    }

    timer_init();
    init_radix_table();

    double freq_ghz = measure_freq_ghz();
    fprintf(stdout, "[measure_memboundness] effective freq = %.3f GHz, ISA=%s\n",
            freq_ghz, ISA_LABEL);
    fprintf(stdout, "[measure_memboundness] tier targets (per-call footprint, bytes):\n");
    for (int t = 0; t < MB_N_TIERS; t++)
        fprintf(stdout, "  %-4s = %zu (%.1f KB)\n",
                MB_TIER_NAME[t], MB_TIER_TARGET_BYTES[t],
                MB_TIER_TARGET_BYTES[t] / 1024.0);
    fprintf(stdout, "\n");

    row_t rows[N_RADIX];
    int max_cv_violation = 0;

    for (size_t i = 0; i < N_RADIX; i++) {
        int R = RADIX_TABLE[i].R;
        codelet_fn fn = RADIX_TABLE[i].t1s_fn;
        fprintf(stdout, "[R=%-3d] ", R);
        fflush(stdout);
        if (!fn) {
            fprintf(stdout, "  (no t1s codelet — skipping)\n");
            for (int t = 0; t < MB_N_TIERS; t++) {
                rows[i].ns_tier[t]  = 0;
                rows[i].cyc_tier[t] = 0;
                rows[i].ios_tier[t] = 0;
                rows[i].cv_tier[t]  = 0;
                rows[i].factor[t]   = 1.0;
            }
            rows[i].R = R;
            continue;
        }
        rows[i].R = R;
        for (int t = 0; t < MB_N_TIERS; t++) {
            size_t ios_used = 0;
            double cv = 0.0;
            double ns = bench_at_tier(fn, R, MB_TIER_TARGET_BYTES[t],
                                       &ios_used, &cv);
            rows[i].ns_tier[t]  = ns;
            rows[i].ios_tier[t] = ios_used;
            rows[i].cv_tier[t]  = cv;
            rows[i].cyc_tier[t] = ns * freq_ghz;
            if (cv > MAX_CV_DEFAULT) max_cv_violation = 1;
            fprintf(stdout, " %s:%6.1fns(ios=%zu,cv=%4.1f%%)",
                    MB_TIER_NAME[t], ns, ios_used, cv * 100.0);
            fflush(stdout);
        }
        double base = rows[i].cyc_tier[MB_TIER_L1];
        for (int t = 0; t < MB_N_TIERS; t++) {
            rows[i].factor[t] = (base > 0) ? rows[i].cyc_tier[t] / base : 1.0;
        }
        fprintf(stdout, "  factors:");
        for (int t = 0; t < MB_N_TIERS; t++)
            fprintf(stdout, " %s=%.2f", MB_TIER_NAME[t], rows[i].factor[t]);
        fprintf(stdout, "\n");
    }

    if (max_cv_violation && !force) {
        fprintf(stderr,
                "\n[measure_memboundness] FAIL: at least one cell exceeded "
                "CV threshold %.1f%%.\n  Re-run on a quieter host or pass "
                "--force to bypass.\n", MAX_CV_DEFAULT * 100.0);
        return 1;
    }

    /* ─── memory-bandwidth bench per cache tier ─────────────────────── */
    fprintf(stdout, "\n[memory-bandwidth] benching stride-1 memory passes:\n");
    fflush(stdout);
    double tier_cyc_per_byte[MB_N_TIERS];
    /* Skip DRAM tier: large-buffer alloc + first-touch zeroing is
     * unreliably slow on Windows. The cells we test all fit L3, so the
     * L3 measurement is the de-facto spill datum we need. DRAM defaults
     * to same value as L3 for cost-model purposes. */
    for (int t = 0; t < MB_N_TIERS - 1; t++) {
        fprintf(stdout, "  %-4s  starting (target=%zu bytes)...",
                MB_TIER_NAME[t], MB_TIER_TARGET_BYTES[t]);
        fflush(stdout);
        double ns_per_byte = bench_memory_pass(MB_TIER_TARGET_BYTES[t]);
        tier_cyc_per_byte[t] = ns_per_byte * freq_ghz;
        double bytes_per_cyc = (tier_cyc_per_byte[t] > 0)
                             ? 1.0 / tier_cyc_per_byte[t] : 0.0;
        double gbps = (ns_per_byte > 0)
                    ? 1.0 / (ns_per_byte) : 0.0;
        fprintf(stdout, "  %.4f cyc/byte  (%.2f B/cyc, %.2f GB/s)\n",
                tier_cyc_per_byte[t], bytes_per_cyc, gbps);
        fflush(stdout);
    }
    /* DRAM = L3 value (placeholder until we sort out the large-alloc path). */
    tier_cyc_per_byte[MB_TIER_DRAM] = tier_cyc_per_byte[MB_TIER_L3];
    fprintf(stdout, "  DRAM  (using L3 value — DRAM bench skipped for now)\n");
    fflush(stdout);

    const char *out_path = "cost_model/generated/radix_memboundness.h";
    emit_header(out_path, rows, (int)N_RADIX, tier_cyc_per_byte,
                freq_ghz, ISA_LABEL);
    fprintf(stdout, "\n[measure_memboundness] wrote %s\n", out_path);
    return 0;
}
