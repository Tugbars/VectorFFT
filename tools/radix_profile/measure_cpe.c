/* measure_cpe.c — measure per-butterfly cycle cost for every registered
 * codelet variant (n1, t1, t1s) at K=256, emit tools/radix_profile/radix_cpe.h.
 *
 * The cost model in src/core/factorizer.h reads this header to convert a
 * candidate factorization into a score. Hand-coded numbers (the previous
 * approach) are inherently per-CPU and per-build; this tool produces them
 * empirically on the host that built the codelets.
 *
 * Run via:
 *   python build_tuned/build.py --src tools/radix_profile/measure_cpe.c
 *   build_tuned/test/measure_cpe.exe              [run on calibration host]
 *
 * Recommended host state:
 *   - Single physical core, pinned (taskset -c 0 / SetProcessAffinityMask)
 *   - High Performance / performance governor active (orchestrator preflight)
 *   - No other significant load (close apps, disable indexing)
 *
 * The tool refuses to overwrite radix_cpe.h if any codelet's coefficient
 * of variation (CV across 21 medians) exceeds MAX_CV (default 5%); pass
 * --force to bypass. A "calibration fingerprint" comment block is baked
 * into the emitted header so reviewers can spot stale or wrong-platform
 * commits.
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

#include "registry.h"                  /* stride_registry_t, codelet typedefs */
#include "generated/radix_profile.h"   /* STRIDE_RADIX_PROFILE_MAX_R */

/* ─────────────────────────── tunables ─────────────────────────── */

#define BENCH_K          256          /* twiddle-stage column count */
#define BENCH_N_RUNS      21          /* batches; report median + CV */
#define BENCH_TARGET_NS   2000000.0   /* ~2 ms per batch (auto-iters) */
#define BENCH_CALIB_NS    100000.0    /* ~100 us calibration probe */
#define MAX_CV_DEFAULT    0.05        /* 5% — refuse if any codelet exceeds */

/* Radixes we expect to find codelets for. The registry can have NULLs
 * for unsupported variants; we skip those silently. */
static const int RADIXES[] = {
    2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 20, 25, 32, 64, 0
};

/* ───────────────────────── time / freq plumbing ───────────────────────── */

#ifdef _WIN32
static double g_qpc_freq;            /* QueryPerformanceFrequency, Hz */

static double now_ns(void) {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / g_qpc_freq;
}

static void timer_init(void) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    g_qpc_freq = (double)f.QuadPart;
}
#else
static void timer_init(void) {}
static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}
#endif

/* Measure the host's effective TSC / wall ratio over a known window. We use
 * RDTSC across a calibrated wall interval rather than reading nominal CPU
 * MHz — what matters is the rate the CPU is *actually* running at right now
 * (which differs from nominal under turbo / boost / thermal throttling). */
static double measure_freq_ghz(void) {
    /* warm */
    (void)__rdtsc();
    double t0 = now_ns();
    uint64_t r0 = __rdtsc();
    /* spin ~50 ms */
    while (now_ns() - t0 < 50e6) { /* nothing */ }
    uint64_t r1 = __rdtsc();
    double t1 = now_ns();
    double cycles = (double)(r1 - r0);
    double ns = t1 - t0;
    return cycles / ns;  /* GHz: cycles/ns */
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
    return (n % 2) ? xs[n / 2]
                   : 0.5 * (xs[n / 2 - 1] + xs[n / 2]);
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

/* Coefficient of variation. */
static double dcv(double *xs, int n) {
    double m = dmean(xs, n);
    if (m <= 0) return 1e9;
    return dstddev(xs, n, m) / m;
}

/* ───────────────────────── timing kernels ─────────────────────────
 *
 * For each variant we time `n_iters` codelet invocations, all with the same
 * arguments, in a tight loop. The compiler can't hoist the call (function
 * pointer through the registry struct), so each iteration is a real call.
 * Side-effect: rio data drifts over iterations, which is fine — we're not
 * checking correctness here, just throughput.
 */

/* Calibrate iteration count so one batch takes ~target_ns. Returns n_iters. */
static int calibrate_iters_n1(stride_n1_fn fn,
                              const double *in_re, const double *in_im,
                              double *out_re, double *out_im,
                              size_t is, size_t os, size_t vl,
                              double target_ns)
{
    int n = 16;
    for (int trial = 0; trial < 12; trial++) {
        double t0 = now_ns();
        for (int i = 0; i < n; i++) fn(in_re, in_im, out_re, out_im, is, os, vl);
        double dt = now_ns() - t0;
        if (dt > BENCH_CALIB_NS) {
            int target = (int)((double)n * target_ns / dt + 0.5);
            return target < 1 ? 1 : target;
        }
        n *= 4;
    }
    return n;
}

static int calibrate_iters_t1(stride_t1_fn fn,
                              double *rio_re, double *rio_im,
                              const double *W_re, const double *W_im,
                              size_t ios, size_t me,
                              double target_ns)
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

/* Time a codelet over BENCH_N_RUNS batches; return median ns/call,
 * write CV across batches into *cv_out. */
static double bench_n1(stride_n1_fn fn, int R)
{
    size_t K = BENCH_K;
    size_t is = K, os = K, vl = K;
    /* Buffers sized for R legs * K cols * 1 element each, plus padding. */
    size_t nelem = (size_t)R * K + 64;
    double *in_re  = xaligned_alloc(nelem * sizeof(double));
    double *in_im  = xaligned_alloc(nelem * sizeof(double));
    double *out_re = xaligned_alloc(nelem * sizeof(double));
    double *out_im = xaligned_alloc(nelem * sizeof(double));
    /* fill inputs with mild non-zero data so FMA chains aren't all-zero */
    for (size_t i = 0; i < nelem; i++) { in_re[i] = 1.0; in_im[i] = -1.0; }

    int n_iters = calibrate_iters_n1(fn, in_re, in_im, out_re, out_im,
                                     is, os, vl, BENCH_TARGET_NS);

    double medians[BENCH_N_RUNS];
    for (int run = 0; run < BENCH_N_RUNS; run++) {
        double t0 = now_ns();
        for (int i = 0; i < n_iters; i++)
            fn(in_re, in_im, out_re, out_im, is, os, vl);
        double dt = now_ns() - t0;
        medians[run] = dt / (double)n_iters;
    }

    xaligned_free(in_re); xaligned_free(in_im);
    xaligned_free(out_re); xaligned_free(out_im);

    return dmedian(medians, BENCH_N_RUNS);
}

static double bench_n1_with_cv(stride_n1_fn fn, int R, double *cv_out)
{
    size_t K = BENCH_K;
    size_t nelem = (size_t)R * K + 64;
    double *in_re  = xaligned_alloc(nelem * sizeof(double));
    double *in_im  = xaligned_alloc(nelem * sizeof(double));
    double *out_re = xaligned_alloc(nelem * sizeof(double));
    double *out_im = xaligned_alloc(nelem * sizeof(double));
    for (size_t i = 0; i < nelem; i++) { in_re[i] = 1.0; in_im[i] = -1.0; }

    int n_iters = calibrate_iters_n1(fn, in_re, in_im, out_re, out_im,
                                     K, K, K, BENCH_TARGET_NS);

    double medians[BENCH_N_RUNS];
    for (int run = 0; run < BENCH_N_RUNS; run++) {
        double t0 = now_ns();
        for (int i = 0; i < n_iters; i++)
            fn(in_re, in_im, out_re, out_im, K, K, K);
        double dt = now_ns() - t0;
        medians[run] = dt / (double)n_iters;
    }
    *cv_out = dcv(medians, BENCH_N_RUNS);

    xaligned_free(in_re); xaligned_free(in_im);
    xaligned_free(out_re); xaligned_free(out_im);

    return dmedian(medians, BENCH_N_RUNS);
}

/* For t1 variants we also need a twiddle table of size (R-1)*K complex.
 * Values themselves don't affect performance, but we compute realistic
 * cos/sin so the compiler can't hoist constant folding. */
static double bench_t1_with_cv(stride_t1_fn fn, int R, double *cv_out)
{
    size_t K = BENCH_K;
    size_t nelem = (size_t)R * K + 64;
    size_t ntw   = (size_t)(R - 1) * K + 64;
    double *rio_re = xaligned_alloc(nelem * sizeof(double));
    double *rio_im = xaligned_alloc(nelem * sizeof(double));
    double *W_re   = xaligned_alloc(ntw * sizeof(double));
    double *W_im   = xaligned_alloc(ntw * sizeof(double));
    for (size_t i = 0; i < nelem; i++) { rio_re[i] = 1.0; rio_im[i] = -1.0; }
    for (size_t j = 1; j < (size_t)R; j++) {
        for (size_t k = 0; k < K; k++) {
            double th = -2.0 * 3.14159265358979323846 *
                        (double)(j * k) / (double)(R * K);
            W_re[(j - 1) * K + k] = cos(th);
            W_im[(j - 1) * K + k] = sin(th);
        }
    }

    int n_iters = calibrate_iters_t1(fn, rio_re, rio_im, W_re, W_im,
                                     K, K, BENCH_TARGET_NS);

    double medians[BENCH_N_RUNS];
    for (int run = 0; run < BENCH_N_RUNS; run++) {
        double t0 = now_ns();
        for (int i = 0; i < n_iters; i++)
            fn(rio_re, rio_im, W_re, W_im, K, K);
        double dt = now_ns() - t0;
        medians[run] = dt / (double)n_iters;
    }
    *cv_out = dcv(medians, BENCH_N_RUNS);

    xaligned_free(rio_re); xaligned_free(rio_im);
    xaligned_free(W_re);   xaligned_free(W_im);

    return dmedian(medians, BENCH_N_RUNS);
}

/* ───────────────────────── per-radix result row ───────────────────────── */

typedef struct {
    int    R;
    int    have_n1, have_t1, have_t1s;
    double ns_n1, ns_t1, ns_t1s;     /* nanoseconds per codelet call */
    double cv_n1, cv_t1, cv_t1s;     /* coefficient of variation */
    double cyc_n1, cyc_t1, cyc_t1s;  /* cycles per butterfly */
} radix_row_t;

/* ───────────────────────── header emit ───────────────────────── */

static const char *isa_tag(void) {
#if defined(VFFT_ISA_AVX512)
    return "avx512";
#elif defined(VFFT_ISA_AVX2)
    return "avx2";
#else
    return "scalar";
#endif
}

static void emit_fingerprint(FILE *f, double freq_ghz, double max_cv) {
#ifdef _WIN32
    OSVERSIONINFOA os; memset(&os, 0, sizeof(os));
    os.dwOSVersionInfoSize = sizeof(os);
    /* GetVersionExA is deprecated but the values are still informational. */
    fprintf(f, " * Host OS:    Windows %lu.%lu (build %lu)\n",
            (unsigned long)os.dwMajorVersion, (unsigned long)os.dwMinorVersion,
            (unsigned long)os.dwBuildNumber);
#else
    fprintf(f, " * Host OS:    POSIX (uname omitted)\n");
#endif
    char *cpu_brand = NULL;
    int cpuinfo[4] = {0};
    char brand[64] = {0};
#ifdef _WIN32
    __cpuid(cpuinfo, 0x80000000);
    if ((unsigned)cpuinfo[0] >= 0x80000004) {
        __cpuid((int*)&brand[0],  0x80000002);
        __cpuid((int*)&brand[16], 0x80000003);
        __cpuid((int*)&brand[32], 0x80000004);
        cpu_brand = brand;
    }
#endif
    fprintf(f, " * Host CPU:   %s\n",
            cpu_brand ? cpu_brand : "(unknown)");
    fprintf(f, " * ISA tag:    %s\n", isa_tag());
    fprintf(f, " * Eff. freq:  %.3f GHz (RDTSC over 50ms wall)\n", freq_ghz);
    fprintf(f, " * Max CV:     %.2f%% (refuse threshold %.2f%%)\n",
            max_cv * 100.0, MAX_CV_DEFAULT * 100.0);
    time_t t = time(NULL);
    struct tm *tm = gmtime(&t);
    fprintf(f, " * Date (UTC): %04d-%02d-%02d %02d:%02d\n",
            tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
            tm->tm_hour, tm->tm_min);
}

static void emit_header(const char *path, const radix_row_t *rows, int n,
                        double freq_ghz, double max_cv)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }

    fprintf(f,
"/* radix_cpe.h — auto-generated by tools/radix_profile/measure_cpe.c.\n"
" * DO NOT EDIT BY HAND. Regenerate via measure_cpe.exe (preferably under\n"
" * the orchestrator's --auto-performance preflight on a calibration host).\n"
" *\n"
" * Per-radix per-variant cycles-per-butterfly at K=%d. Read by the cost\n"
" * model in src/core/factorizer.h to score candidate factorizations.\n"
" *\n", BENCH_K);
    fprintf(f, " * Calibration fingerprint:\n");
    emit_fingerprint(f, freq_ghz, max_cv);
    fprintf(f, " */\n");

    fprintf(f, "#ifndef STRIDE_RADIX_CPE_H\n");
    fprintf(f, "#define STRIDE_RADIX_CPE_H\n\n");
    fprintf(f, "#include \"radix_profile.h\"\n\n");
    fprintf(f, "typedef struct {\n");
    fprintf(f, "    double cyc_n1;\n");
    fprintf(f, "    double cyc_t1;\n");
    fprintf(f, "    double cyc_t1s;\n");
    fprintf(f, "} stride_radix_cpe_t;\n\n");

    /* The current build is one ISA at a time; emit only the matching table.
     * Other ISAs leave their table empty (factorizer falls back to ops). */
    fprintf(f,
        "static const stride_radix_cpe_t stride_radix_cpe_%s"
        "[STRIDE_RADIX_PROFILE_MAX_R] = {\n", isa_tag());
    for (int i = 0; i < n; i++) {
        const radix_row_t *r = &rows[i];
        if (!r->have_n1 && !r->have_t1 && !r->have_t1s) continue;
        fprintf(f, "    [%2d] = {", r->R);
        if (r->have_n1)  fprintf(f, " .cyc_n1 = %7.3f,", r->cyc_n1);
        if (r->have_t1)  fprintf(f, " .cyc_t1 = %7.3f,", r->cyc_t1);
        if (r->have_t1s) fprintf(f, " .cyc_t1s = %7.3f,", r->cyc_t1s);
        fprintf(f, " },\n");
    }
    fprintf(f, "};\n\n");

    /* Emit empty companion tables for the other ISAs so the factorizer
     * lookup compiles regardless of which ISA was used for measurement.
     * Lookup site falls back to ops/SIMD when the cyc_* slot is 0. */
    const char *all_isas[] = {"avx2", "avx512", "scalar", NULL};
    for (int i = 0; all_isas[i]; i++) {
        if (strcmp(all_isas[i], isa_tag()) == 0) continue;
        fprintf(f,
            "static const stride_radix_cpe_t stride_radix_cpe_%s"
            "[STRIDE_RADIX_PROFILE_MAX_R] = {0};\n", all_isas[i]);
    }
    fprintf(f, "\n#endif /* STRIDE_RADIX_CPE_H */\n");
    fclose(f);
}

/* ───────────────────────── main driver ───────────────────────── */

static int parse_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], flag)) return 1;
    return 0;
}

static const char *parse_kv(int argc, char **argv, const char *key) {
    size_t kl = strlen(key);
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], key, kl) && argv[i][kl] == '=')
            return argv[i] + kl + 1;
    }
    return NULL;
}

int main(int argc, char **argv) {
    int force = parse_flag(argc, argv, "--force");
    int no_emit = parse_flag(argc, argv, "--no-emit");
    int verbose = parse_flag(argc, argv, "--verbose") || parse_flag(argc, argv, "-v");
    const char *out_path = parse_kv(argc, argv, "--output");
    if (!out_path) out_path = "src/core/generated/radix_cpe.h";

    timer_init();
    printf("[measure_cpe] calibrating CPU frequency...\n");
    double freq_ghz = measure_freq_ghz();
    printf("[measure_cpe] effective freq = %.3f GHz (RDTSC/50ms wall)\n", freq_ghz);
    if (freq_ghz < 0.5 || freq_ghz > 10.0) {
        fprintf(stderr, "[error] freq %.3f GHz out of plausible range\n", freq_ghz);
        return 2;
    }

    stride_registry_t reg;
    stride_registry_init(&reg);

    int n_radixes = 0;
    for (const int *rp = RADIXES; *rp; rp++) n_radixes++;
    radix_row_t *rows = calloc(n_radixes, sizeof(radix_row_t));

    printf("[measure_cpe] %s, K=%d, %d runs/codelet, ~%.0fms/run\n",
           isa_tag(), BENCH_K, BENCH_N_RUNS, BENCH_TARGET_NS / 1e6);
    printf("\n  %3s  %-3s    %12s  %6s    %12s  %6s    %12s  %6s\n",
           "R", "isa", "n1 ns/call", "CV%", "t1 ns/call", "CV%", "t1s ns/call", "CV%");
    printf("  ---  ---    ------------  ------    ------------  ------    ------------  ------\n");

    int idx = 0;
    double max_cv = 0.0;
    int max_cv_R = 0;
    const char *max_cv_variant = "";

    for (const int *rp = RADIXES; *rp; rp++) {
        int R = *rp;
        radix_row_t *r = &rows[idx++];
        r->R = R;
        printf("  %3d  %-3s ", R, isa_tag());

        /* n1 forward (DIT stage 0 codelet) */
        if (R < STRIDE_REG_MAX_RADIX && reg.n1_fwd[R]) {
            double cv;
            double ns = bench_n1_with_cv(reg.n1_fwd[R], R, &cv);
            r->have_n1 = 1; r->ns_n1 = ns; r->cv_n1 = cv;
            r->cyc_n1 = ns * freq_ghz / (double)BENCH_K;
            if (cv > max_cv) { max_cv = cv; max_cv_R = R; max_cv_variant = "n1"; }
            printf("   %10.2f  %5.2f%%", ns, cv * 100.0);
        } else {
            printf("   %10s  %6s", "—", "—");
        }

        /* t1 forward (twiddled stage codelet) */
        if (R < STRIDE_REG_MAX_RADIX && reg.t1_fwd[R]) {
            double cv;
            double ns = bench_t1_with_cv(reg.t1_fwd[R], R, &cv);
            r->have_t1 = 1; r->ns_t1 = ns; r->cv_t1 = cv;
            r->cyc_t1 = ns * freq_ghz / (double)BENCH_K;
            if (cv > max_cv) { max_cv = cv; max_cv_R = R; max_cv_variant = "t1"; }
            printf("   %10.2f  %5.2f%%", ns, cv * 100.0);
        } else {
            printf("   %10s  %6s", "—", "—");
        }

        /* t1s forward (scalar-twiddle variant) */
        if (R < STRIDE_REG_MAX_RADIX && reg.t1s_fwd[R]) {
            double cv;
            double ns = bench_t1_with_cv(reg.t1s_fwd[R], R, &cv);
            r->have_t1s = 1; r->ns_t1s = ns; r->cv_t1s = cv;
            r->cyc_t1s = ns * freq_ghz / (double)BENCH_K;
            if (cv > max_cv) { max_cv = cv; max_cv_R = R; max_cv_variant = "t1s"; }
            printf("   %10.2f  %5.2f%%", ns, cv * 100.0);
        } else {
            printf("   %10s  %6s", "—", "—");
        }
        printf("\n");
    }

    printf("\n[measure_cpe] max CV = %.2f%% (R=%d %s)\n",
           max_cv * 100.0, max_cv_R, max_cv_variant);

    if (max_cv > MAX_CV_DEFAULT && !force) {
        fprintf(stderr,
            "[error] CV threshold exceeded: %.2f%% > %.2f%%.\n"
            "        Likely causes: noisy machine, frequency scaling, thermal\n"
            "        throttling, background load. Re-run on a calibration host\n"
            "        with --auto-performance, or use --force to bypass.\n",
            max_cv * 100.0, MAX_CV_DEFAULT * 100.0);
        free(rows);
        return 3;
    }

    if (no_emit) {
        printf("[measure_cpe] --no-emit: skipping header write.\n");
    } else {
        emit_header(out_path, rows, idx, freq_ghz, max_cv);
        printf("[measure_cpe] wrote %s\n", out_path);
    }

    if (verbose) {
        printf("\nCycles per butterfly (cyc_n1 / cyc_t1 / cyc_t1s):\n");
        for (int i = 0; i < idx; i++) {
            const radix_row_t *r = &rows[i];
            printf("  R=%2d:", r->R);
            if (r->have_n1)  printf("  n1=%7.2f", r->cyc_n1);  else printf("  n1=  ----");
            if (r->have_t1)  printf("  t1=%7.2f", r->cyc_t1);  else printf("  t1=  ----");
            if (r->have_t1s) printf("  t1s=%7.2f", r->cyc_t1s); else printf("  t1s=  ----");
            printf("\n");
        }
    }

    free(rows);
    return 0;
}
