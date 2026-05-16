/* measure_cpe.c — prototype edition.
 *
 * Measure per-butterfly cycle cost for every available codelet variant at
 * K=256, emit cost_model/generated/radix_cpe.h. Concrete differences vs
 * production's tools/radix_profile/measure_cpe.c:
 *
 *   1. Uses the prototype's own auto-generated registry.h (sibling to
 *      production's src/core/registry.h). Codelet inventory comes from
 *      generated/registry.h via emit_registry_h.exe — no hand-coded
 *      externs. Codelet symbols are named radix{R}_{variant}_{isa}
 *      (post-rename, production-aligned naming). Pre-doc56 the OCaml
 *      emitter used decorated suffixes (_gen_inplace_su_spill etc.);
 *      those were stripped in the same session that introduced the
 *      registry, so naming is fully predictable from (R, variant, isa).
 *
 *   2. Codelet paths point at src/prototype/codelets/{isa}/{family}/*.c
 *      (linked separately via cost_model/build_measure_cpe.sh).
 *
 *   3. Four-column CPE struct matches production:
 *        cyc_n1     — no-twiddle first-stage codelet
 *        cyc_t1     — twiddled inner-stage, default rendering
 *        cyc_t1s    — twiddled inner-stage, set1-twiddle rendering
 *        cyc_log3   — twiddled inner-stage, log3 twiddle derivation
 *                     (0.0 for non-pow2 radixes where log3 doesn't apply)
 *      Each variant is bench'd standalone — no combining at measurement.
 *      The cost model picks which column to consult per its own logic.
 *
 *   4. Each row also reports predicted cycles from radix_profile.h
 *      (ops/SIMD-width fallback) and measured/predicted residuals.
 *      A regression summary at end tells us how the closed-form cost
 *      model compares against ground truth.
 *
 * Run via:
 *   bash cost_model/build_measure_cpe.sh
 *   build_tuned/measure_cpe                  [pinned core, perf governor]
 *
 * Recommended host state:
 *   - Single physical core, pinned (taskset -c 0 / SetProcessAffinityMask)
 *   - Performance governor active (no frequency scaling)
 *   - No other significant load
 *
 * The tool refuses to overwrite radix_cpe.h if any codelet's CV (across
 * BENCH_N_RUNS batches) exceeds MAX_CV_DEFAULT (5%). Pass --force to bypass.
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

#include "generated/radix_profile.h"   /* STRIDE_RADIX_PROFILE_MAX_R + profile tables */

/* ─────────────────────────── tunables ─────────────────────────── */

#define BENCH_K          256
#define BENCH_N_RUNS      51
#define BENCH_TARGET_NS   10000000.0  /* ~10 ms per batch — total per variant
                                       *   = 51 batches × 10 ms = 510 ms
                                       * × 4 variants per radix ≈ 2 s per radix.
                                       * R=512 t1 single call ≈ 268 µs, well
                                       * under per-batch target; no special-
                                       * casing needed at the high end. */
#define BENCH_CALIB_NS    100000.0    /* ~100 us calibration probe */
#define MAX_CV_DEFAULT    0.05        /* 5% — refuse threshold */

/* me-sweep: bench each (R, variant) at multiple inner-loop lengths so
 * the cost model can interpolate. Real plan executors use me values
 * far from the previous "me=K=256" calibration point; this lets us
 * close the amortization gap. ios stays fixed at K — stride-dependent
 * cache costs are handled separately via factorizer.h's cache_factor.
 *
 * Empirical me values in our test plans range 4K to 2M, so the sample
 * grid covers that range plus margins. log-spaced.
 */
#define CPE_N_ME_SAMPLES   3
static const size_t CPE_ME_VALUES[CPE_N_ME_SAMPLES] = {
    256, 4096, 65536
};

/* Per-cell memory cap (kept as a safety net for very large me at very
 * large R; the 1D sweep at these values is well within bounds for our
 * radix set, but the check is cheap). */
#define CPE_BUFFER_CAP     (8 * 1024 * 1024)

/* ─────────────────────── ISA selection ─────────────────────────
 * Default target is AVX-2. Compile with -DVFFT_ISA_AVX512=1 for AVX-512.
 */
#if defined(VFFT_ISA_AVX512)
  #define ISA_LABEL     "avx512"
  #define ISA_VEC_WIDTH 8
  #define USE_AVX512    1
#else
  #define ISA_LABEL     "avx2"
  #define ISA_VEC_WIDTH 4
  #define USE_AVX2      1
#endif

/* ──────────────────── codelet signature ─────────────────────────
 * All variants share the same 6-arg shape in the prototype:
 *   void radix{N}_{variant}_fwd_{isa}_gen_*(rio_re, rio_im,
 *                                           tw_re, tw_im, ios, me).
 * n1 codelets ignore the twiddle args but still take them — uniform
 * calling convention so the bench can dispatch generically.
 */
typedef void (*codelet_fn)(double *, double *,
                           const double *, const double *,
                           size_t, size_t);

/* ──────────────── registry-based codelet dispatch ───────────────
 *
 * Previously this file hand-coded ~150 extern declarations + a static
 * RADIX_TABLE per ISA, listing every codelet symbol the bench would
 * call. That inventory had to be kept in sync with the OCaml emitter
 * by hand — broke any time codelet naming changed.
 *
 * Now the OCaml side owns this inventory: emit_registry_h.exe emits
 * generated/registry.h with one extern per emitted codelet, plus an
 * init function that wires every codelet into a typed struct. We
 * #include that here and populate our RADIX_TABLE at runtime from the
 * registry slots.
 *
 * Benefits:
 *   - One source of truth (OCaml emit → header → consumer). Codelet
 *     renames/additions propagate via a single regen, no edits here.
 *   - Newly-emitted codelets (e.g. log3 on primes/composites added
 *     2026-05-16) automatically appear in the bench — non-NULL slots
 *     get measured.
 *   - Symbol naming is computable from (R, variant, isa) so no manual
 *     suffix tracking.
 */
#include "../generated/registry.h"

typedef struct {
    int R;
    codelet_fn n1_fn;
    codelet_fn t1_fn;
    codelet_fn t1s_fn;   /* NULL if codelet doesn't exist for this radix */
    codelet_fn log3_fn;  /* NULL if codelet doesn't exist for this radix */
} radix_entry_t;

/* Same radix set the prior hand-coded RADIX_TABLE had — R=1024 excluded
 * (research-only, out of cost-model scope). Order matters for stable
 * output across runs. */
static const int RADIX_LIST[] = {
    2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13,
    16, 17, 19, 20, 25, 32, 64, 128, 256, 512
};
#define N_RADIX (sizeof(RADIX_LIST) / sizeof(RADIX_LIST[0]))

static radix_entry_t RADIX_TABLE[N_RADIX];
static vfft_proto_registry_t g_reg;

/* Initialize RADIX_TABLE from the registry. Call once at startup. */
static void init_radix_table(void) {
#if defined(USE_AVX512)
    vfft_proto_registry_init_avx512(&g_reg);
#elif defined(USE_AVX2)
    vfft_proto_registry_init_avx2(&g_reg);
#else
#error "must define USE_AVX2 or USE_AVX512"
#endif
    for (size_t i = 0; i < N_RADIX; i++) {
        int R = RADIX_LIST[i];
        RADIX_TABLE[i].R       = R;
        /* Cast: vfft_proto_codelet_fn and codelet_fn have the same
         * signature but are nominally distinct C types. The cast is
         * always safe (identical underlying ABI). */
        RADIX_TABLE[i].n1_fn   = (codelet_fn)g_reg.n1_fwd[R];
        RADIX_TABLE[i].t1_fn   = (codelet_fn)g_reg.t1_dit_fwd[R];
        RADIX_TABLE[i].t1s_fn  = (codelet_fn)g_reg.t1s_dit_fwd[R];
        RADIX_TABLE[i].log3_fn = (codelet_fn)g_reg.t1_dit_log3_fwd[R];
    }
}

/* ─────────────────────── time / freq plumbing ─────────────────────── */

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

/* Calibrate the host's effective TSC/wall ratio via RDTSC over a known
 * 50ms wall window. Captures actual running freq under turbo / throttling. */
static double measure_freq_ghz(void) {
    (void)__rdtsc();
    double t0 = now_ns();
    uint64_t r0 = __rdtsc();
    while (now_ns() - t0 < 50e6) { /* spin ~50 ms */ }
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
    return dstddev(xs, n, m) / m;
}

/* ───────────────────────── bench kernel ───────────────────────── */

static int calibrate_iters(codelet_fn fn, double *rio_re, double *rio_im,
                           const double *W_re, const double *W_im,
                           size_t ios, size_t me, double target_ns)
{
    int n = 16;
    for (int trial = 0; trial < 12; trial++) {
        double t0 = now_ns();
        for (int i = 0; i < n; i++) fn(rio_re, rio_im, W_re, W_im, ios, me);
        double dt = now_ns() - t0;
        if (dt > BENCH_CALIB_NS) {
            int target = (int)((double)n * target_ns / dt + 0.5);
            return target < 1 ? 1 : target;
        }
        n *= 4;
    }
    return n;
}

/* Bench one codelet at the given inner-loop length me. Returns mean
 * ns/call across BENCH_N_RUNS batches; writes CV to *cv_out. ios stays
 * fixed at BENCH_K — only me varies (amortization sweep). Stride-cache
 * effects stay in factorizer.h's cache_factor. */
static double bench_one(codelet_fn fn, int R, size_t me, double *cv_out)
{
    if (fn == NULL) { *cv_out = 0.0; return 0.0; }
    size_t ios = BENCH_K;
    /* Skip cells that would exceed the buffer cap (safety; the 1D me-sweep
     * is well within bounds for our radix set). */
    size_t rio_span = (size_t)(R - 1) * ios + me;
    size_t tw_span  = (size_t)R * me;
    if (rio_span > CPE_BUFFER_CAP || tw_span > CPE_BUFFER_CAP) {
        *cv_out = 0.0;
        return 0.0;
    }
    /* rio buffer must span (R-1)*ios + me doubles (last leg's last access). */
    size_t nelem = rio_span + 64;
    /* Twiddle buffer: the codelet body accesses tw_re/tw_im at offsets
     * j*me + k (NOT j*ios + k) for legs j=1..R-1, k=0..me-1. So the
     * twiddle table needs (R-1)*me doubles, not (R-1)*ios. */
    size_t ntw   = tw_span + 64;
    double *rio_re = xaligned_alloc(nelem * sizeof(double));
    double *rio_im = xaligned_alloc(nelem * sizeof(double));
    double *tw_re  = xaligned_alloc(ntw   * sizeof(double));
    double *tw_im  = xaligned_alloc(ntw   * sizeof(double));
    for (size_t i = 0; i < nelem; i++) { rio_re[i] = 1.0; rio_im[i] = -1.0; }
    /* Fill the entire twiddle table with non-trivial values so the compiler
     * can't const-fold work. Twiddle column j is at offset (j-1)*me. */
    for (int j = 1; j < R; j++) {
        for (size_t k = 0; k < me; k++) {
            double th = -2.0 * 3.14159265358979323846 *
                        (double)((size_t)j * k) / (double)((size_t)R * me);
            tw_re[(size_t)(j - 1) * me + k] = cos(th);
            tw_im[(size_t)(j - 1) * me + k] = sin(th);
        }
    }

    int n_iters = calibrate_iters(fn, rio_re, rio_im, tw_re, tw_im, ios, me, BENCH_TARGET_NS);

    double medians[BENCH_N_RUNS];
    for (int run = 0; run < BENCH_N_RUNS; run++) {
        double t0 = now_ns();
        for (int i = 0; i < n_iters; i++)
            fn(rio_re, rio_im, tw_re, tw_im, ios, me);
        double dt = now_ns() - t0;
        medians[run] = dt / (double)n_iters;
    }
    *cv_out = dcv(medians, BENCH_N_RUNS);
    double avg = dmean(medians, BENCH_N_RUNS);

    xaligned_free(rio_re); xaligned_free(rio_im);
    xaligned_free(tw_re);  xaligned_free(tw_im);
    return avg;
}

/* ─────────────── predicted-cycle from radix_profile.h ────────────── */

static double predicted_cyc(int R, int is_n1)
{
    if (R <= 0 || R >= STRIDE_RADIX_PROFILE_MAX_R) return 0.0;
    const stride_radix_profile_t *p;
#if defined(USE_AVX2)
    p = is_n1 ? &stride_radix_profile_n1_avx2[R]
              : &stride_radix_profile_t1_avx2[R];
#else
    p = is_n1 ? &stride_radix_profile_n1_avx512[R]
              : &stride_radix_profile_t1_avx512[R];
#endif
    int ops = _stride_total_ops(p);
    if (ops == 0) return 0.0;
    return (double)ops / (double)ISA_VEC_WIDTH;
}

/* ──────────────────── per-radix result row ──────────────────────── */

typedef struct {
    int    R;
    /* 1D arrays indexed by me sample (CPE_N_ME_SAMPLES). */
    double cyc_n1   [CPE_N_ME_SAMPLES];
    double cyc_t1   [CPE_N_ME_SAMPLES];
    double cyc_t1s  [CPE_N_ME_SAMPLES];
    double cyc_log3 [CPE_N_ME_SAMPLES];
    double cv_n1    [CPE_N_ME_SAMPLES];
    double cv_t1    [CPE_N_ME_SAMPLES];
    double cv_t1s   [CPE_N_ME_SAMPLES];
    double cv_log3  [CPE_N_ME_SAMPLES];
    /* radix_profile.h ops/SIMD predicted cycles (single value, K=BENCH_K). */
    double pred_n1, pred_inner;
    int    has_t1s, has_log3;
} row_t;

/* ───────────────────────── header emit ─────────────────────────── */

static void emit_fingerprint(FILE *f, double freq_ghz, double max_cv)
{
    char brand[64] = "(unknown)";
#ifdef _WIN32
    int cpuinfo[4] = {0};
    __cpuid(cpuinfo, 0x80000000);
    if ((unsigned)cpuinfo[0] >= 0x80000004) {
        __cpuid((int *)&brand[0],  0x80000002);
        __cpuid((int *)&brand[16], 0x80000003);
        __cpuid((int *)&brand[32], 0x80000004);
    }
    fprintf(f, " * Host OS:    Windows\n");
#else
    fprintf(f, " * Host OS:    POSIX\n");
#endif
    fprintf(f, " * Host CPU:   %s\n", brand);
    fprintf(f, " * ISA tag:    %s\n", ISA_LABEL);
    fprintf(f, " * Eff. freq:  %.3f GHz (RDTSC / 50ms wall)\n", freq_ghz);
    fprintf(f, " * Max CV:     %.2f%% (refuse threshold %.2f%%)\n",
            max_cv * 100.0, MAX_CV_DEFAULT * 100.0);
    fprintf(f, " * Emission mode: M-active (VFFT_USE_REGALLOC=1 VFFT_USE_REGALLOC_M5=1)\n");
    fprintf(f, " * fma_lift gate: Direct + Cooley_Tukey (doc 56)\n");
    time_t t = time(NULL);
    struct tm *tm = gmtime(&t);
    fprintf(f, " * Date (UTC): %04d-%02d-%02d %02d:%02d\n",
            tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
            tm->tm_hour, tm->tm_min);
}

static void emit_header(const char *path, const row_t *rows, int n,
                        double freq_ghz, double max_cv)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }

    fprintf(f,
"/* radix_cpe.h — auto-generated by cost_model/measure_cpe.\n"
" * DO NOT EDIT BY HAND. Regenerate via measure_cpe on a quiet host.\n"
" *\n"
" * Per-radix cycles-per-butterfly at K=%d. Four columns:\n"
" *   cyc_n1    — no-twiddle first-stage codelet\n"
" *   cyc_t1    — twiddled inner-stage, default rendering\n"
" *   cyc_t1s   — twiddled inner-stage, set1-twiddle rendering\n"
" *   cyc_log3  — twiddled inner-stage, log3 twiddle derivation\n"
" *               (0.0 for non-pow2 radixes)\n"
" *\n"
" * Each column is the measured median (51 batches) of that specific\n"
" * codelet variant standalone — no combining across variants. The\n"
" * cost model picks which column to consult based on its own variant\n"
" * preference logic (or min over the available t-variants).\n"
" *\n"
" * Calibration fingerprint:\n",
        BENCH_K);
    emit_fingerprint(f, freq_ghz, max_cv);
    fprintf(f, " */\n");

    fprintf(f, "#ifndef STRIDE_RADIX_CPE_H\n");
    fprintf(f, "#define STRIDE_RADIX_CPE_H\n\n");
    fprintf(f, "#include \"radix_profile.h\"\n\n");

    fprintf(f, "#define CPE_N_ME_SAMPLES %d\n\n", CPE_N_ME_SAMPLES);
    fprintf(f, "static const size_t CPE_ME_VALUES[CPE_N_ME_SAMPLES] = {\n");
    for (int j = 0; j < CPE_N_ME_SAMPLES; j++) {
        fprintf(f, "    %zu,\n", CPE_ME_VALUES[j]);
    }
    fprintf(f, "};\n\n");

    fprintf(f, "typedef struct {\n");
    fprintf(f, "    double cyc_n1   [CPE_N_ME_SAMPLES];\n");
    fprintf(f, "    double cyc_t1   [CPE_N_ME_SAMPLES];\n");
    fprintf(f, "    double cyc_t1s  [CPE_N_ME_SAMPLES];   /* 0.0 = not measured */\n");
    fprintf(f, "    double cyc_log3 [CPE_N_ME_SAMPLES];   /* 0.0 = not applicable */\n");
    fprintf(f, "} stride_radix_cpe_t;\n\n");

    fprintf(f,
        "static const stride_radix_cpe_t stride_radix_cpe_%s"
        "[STRIDE_RADIX_PROFILE_MAX_R] = {\n", ISA_LABEL);
    for (int i = 0; i < n; i++) {
        const row_t *r = &rows[i];
        fprintf(f, "    [%4d] = {\n", r->R);
        fprintf(f, "        .cyc_n1   = { ");
        for (int j = 0; j < CPE_N_ME_SAMPLES; j++) fprintf(f, "%8.3f, ", r->cyc_n1[j]);
        fprintf(f, "},\n");
        fprintf(f, "        .cyc_t1   = { ");
        for (int j = 0; j < CPE_N_ME_SAMPLES; j++) fprintf(f, "%8.3f, ", r->cyc_t1[j]);
        fprintf(f, "},\n");
        fprintf(f, "        .cyc_t1s  = { ");
        for (int j = 0; j < CPE_N_ME_SAMPLES; j++) fprintf(f, "%8.3f, ", r->cyc_t1s[j]);
        fprintf(f, "},\n");
        fprintf(f, "        .cyc_log3 = { ");
        for (int j = 0; j < CPE_N_ME_SAMPLES; j++) fprintf(f, "%8.3f, ", r->cyc_log3[j]);
        fprintf(f, "},\n");
        fprintf(f, "    },\n");
    }
    fprintf(f, "};\n\n");

    const char *other = (strcmp(ISA_LABEL, "avx2") == 0) ? "avx512" : "avx2";
    fprintf(f, "/* Other-ISA companion (empty until measure_cpe re-run with %s). */\n", other);
    fprintf(f, "static const stride_radix_cpe_t stride_radix_cpe_%s"
               "[STRIDE_RADIX_PROFILE_MAX_R] = {0};\n\n", other);

    fprintf(f, "#endif /* STRIDE_RADIX_CPE_H */\n");
    fclose(f);
}

/* ───────────────────────── main driver ─────────────────────────── */

static int parse_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], flag)) return 1;
    return 0;
}
static const char *parse_kv(int argc, char **argv, const char *key) {
    size_t kl = strlen(key);
    for (int i = 1; i < argc; i++)
        if (!strncmp(argv[i], key, kl) && argv[i][kl] == '=')
            return argv[i] + kl + 1;
    return NULL;
}

int main(int argc, char **argv)
{
    int force = parse_flag(argc, argv, "--force");
    int no_emit = parse_flag(argc, argv, "--no-emit");
    const char *out_path = parse_kv(argc, argv, "--output");
    if (!out_path) out_path = "cost_model/generated/radix_cpe.h";

    timer_init();
    init_radix_table();  /* populate from registry — replaces hand-coded externs */
    printf("[measure_cpe] calibrating CPU frequency...\n");
    double freq_ghz = measure_freq_ghz();
    printf("[measure_cpe] effective freq = %.3f GHz\n", freq_ghz);
    if (freq_ghz < 0.5 || freq_ghz > 10.0) {
        fprintf(stderr, "[error] freq %.3f GHz out of plausible range\n", freq_ghz);
        return 2;
    }

    printf("[measure_cpe] %s, %d batches/(R,variant,me), me-sweep over %d values\n",
           ISA_LABEL, BENCH_N_RUNS, CPE_N_ME_SAMPLES);
    printf("\n  %4s %7s | %8s %8s %8s %8s | %5s %5s %5s %5s\n",
           "R", "me", "cyc_n1", "cyc_t1", "cyc_t1s", "cyc_log3",
           "cv_n1", "cv_t1", "cv_t1s", "cv_l3");
    printf("  -----+---------+-----------------------------------+-----------------------------\n");

    row_t *rows = calloc(N_RADIX, sizeof(row_t));
    double max_cv = 0.0;
    int max_cv_R = 0;
    const char *max_cv_variant = "";

    for (size_t i = 0; i < N_RADIX; i++) {
        const radix_entry_t *e = &RADIX_TABLE[i];
        row_t *r = &rows[i];
        r->R = e->R;
        r->has_t1s  = (e->t1s_fn  != NULL);
        r->has_log3 = (e->log3_fn != NULL);
        r->pred_n1    = predicted_cyc(e->R, 1);
        r->pred_inner = predicted_cyc(e->R, 0);

        /* 1-second cool-down between radixes for thermal fairness. */
        if (i > 0) {
#ifdef _WIN32
            Sleep(1000);
#else
            sleep(1);
#endif
        }

        for (int j = 0; j < CPE_N_ME_SAMPLES; j++) {
            size_t me = CPE_ME_VALUES[j];
            double ns_n1   = bench_one(e->n1_fn,   e->R, me, &r->cv_n1  [j]);
            double ns_t1   = bench_one(e->t1_fn,   e->R, me, &r->cv_t1  [j]);
            double ns_t1s  = bench_one(e->t1s_fn,  e->R, me, &r->cv_t1s [j]);
            double ns_log3 = bench_one(e->log3_fn, e->R, me, &r->cv_log3[j]);

            r->cyc_n1  [j] = ns_n1   * freq_ghz / (double)me;
            r->cyc_t1  [j] = ns_t1   * freq_ghz / (double)me;
            r->cyc_t1s [j] = r->has_t1s  ? ns_t1s  * freq_ghz / (double)me : 0.0;
            r->cyc_log3[j] = r->has_log3 ? ns_log3 * freq_ghz / (double)me : 0.0;

            if (r->cv_n1 [j] > max_cv) { max_cv = r->cv_n1 [j]; max_cv_R = r->R; max_cv_variant = "n1";  }
            if (r->cv_t1 [j] > max_cv) { max_cv = r->cv_t1 [j]; max_cv_R = r->R; max_cv_variant = "t1";  }
            if (r->has_t1s  && r->cv_t1s [j] > max_cv) { max_cv = r->cv_t1s [j]; max_cv_R = r->R; max_cv_variant = "t1s";  }
            if (r->has_log3 && r->cv_log3[j] > max_cv) { max_cv = r->cv_log3[j]; max_cv_R = r->R; max_cv_variant = "log3"; }

            printf("  %4d %7zu | %8.3f %8.3f %8.3f %8.3f | %4.1f%% %4.1f%% %4.1f%% %4.1f%%\n",
                   r->R, me,
                   r->cyc_n1[j], r->cyc_t1[j], r->cyc_t1s[j], r->cyc_log3[j],
                   r->cv_n1[j]*100, r->cv_t1[j]*100, r->cv_t1s[j]*100, r->cv_log3[j]*100);
        }
    }

    /* Regression summary per me cell across all radixes. */
    printf("\n[regression summary: measured cyc / predicted (ops/SIMD) per me sample]\n");
    for (int j = 0; j < CPE_N_ME_SAMPLES; j++) {
        double sum_n1 = 0, sum_t1 = 0, max_resN1 = 0, max_resT1 = 0;
        int nN1 = 0, nT1 = 0;
        for (size_t i = 0; i < N_RADIX; i++) {
            const row_t *r = &rows[i];
            if (r->pred_n1 > 0 && r->cyc_n1[j] > 0) {
                double res = r->cyc_n1[j] / r->pred_n1;
                sum_n1 += res; nN1++;
                if (res > max_resN1) max_resN1 = res;
            }
            if (r->pred_inner > 0 && r->cyc_t1[j] > 0) {
                double res = r->cyc_t1[j] / r->pred_inner;
                sum_t1 += res; nT1++;
                if (res > max_resT1) max_resT1 = res;
            }
        }
        printf("  me=%7zu:  n1 mean=%.3f max=%.3f (n=%d)   t1 mean=%.3f max=%.3f (n=%d)\n",
               CPE_ME_VALUES[j],
               sum_n1 / (nN1 ? nN1 : 1), max_resN1, nN1,
               sum_t1 / (nT1 ? nT1 : 1), max_resT1, nT1);
    }
    printf("  residual<1 = measured FASTER than predicted (ops/SIMD overstates)\n");
    printf("  residual>1 = measured SLOWER than predicted (ops/SIMD understates)\n");

    printf("\n[measure_cpe] max CV = %.2f%% (R=%d %s)\n",
           max_cv * 100.0, max_cv_R, max_cv_variant);

    if (max_cv > MAX_CV_DEFAULT && !force) {
        fprintf(stderr,
            "[error] CV threshold exceeded: %.2f%% > %.2f%%.\n"
            "        Likely causes: noisy machine, freq scaling, thermal\n"
            "        throttling, background load. Re-run on a calibration\n"
            "        host, or use --force to bypass.\n",
            max_cv * 100.0, MAX_CV_DEFAULT * 100.0);
        free(rows);
        return 3;
    }

    if (no_emit) {
        printf("[measure_cpe] --no-emit: skipping header write.\n");
    } else {
        emit_header(out_path, rows, (int)N_RADIX, freq_ghz, max_cv);
        printf("[measure_cpe] wrote %s\n", out_path);
    }

    free(rows);
    return 0;
}
