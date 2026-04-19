/* harness.c — FFTW-style measurement harness for VectorFFT codelet bench.
 *
 * Timing algorithm (after fftw3/kernel/timer.c):
 *   1. Call codelet once with zeroed/warmed buffers ("train" cache).
 *   2. Start with iters=1. Time iters calls; if total < TIME_MIN, double iters.
 *   3. Once total >= TIME_MIN, repeat the (iters-call) block TIME_REPEAT times.
 *   4. Return min repeat / iters as ns-per-call.
 *   5. Hard cap total time via TIME_LIMIT.
 *
 * No memcpy between iterations — plan runs repeatedly on the same buffer
 * (matches FFTW's methodology exactly). The first call dominates cache fill;
 * subsequent calls are warm-cache steady state.
 *
 * Twiddle buffer allocation follows the protocol passed in on the command
 * line: 'flat' / 'log3' / 't1s' / 'log1_tight'. See protocols.py.
 *
 * CLI (driven by bench.py):
 *   harness --variant ct_t1s_dit --isa avx2 --protocol t1s --radix 4 \
 *           --me 64 --ios 64 --dir fwd
 * Outputs a single JSON line to stdout:
 *   {"variant":"ct_t1s_dit","isa":"avx2","protocol":"t1s",...,"ns":123.45}
 *
 * Build: one object per (candidate, isa); link all into one harness binary.
 *        bench.py generates a registration table in a separate .c file.
 */
#define _POSIX_C_SOURCE 200809L
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════
 * PLATFORM: timing + aligned_alloc
 * ═══════════════════════════════════════════════════════════════ */

#if defined(_WIN32)
  #include <windows.h>
  #include <malloc.h>
  static double now_ns(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER cnt;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1e9 / (double)freq.QuadPart;
  }
  static void *vfft_aligned_alloc(size_t align, size_t size) {
    return _aligned_malloc(size, align);
  }
  static void vfft_aligned_free(void *p) { _aligned_free(p); }
#else
  #include <time.h>
  #include <stdlib.h>
  static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
  }
  static void *vfft_aligned_alloc(size_t align, size_t size) {
    void *p = NULL;
    /* posix_memalign requires size be a multiple of align on some impls;
       round up. */
    size_t rounded = (size + align - 1) & ~(align - 1);
    if (posix_memalign(&p, align, rounded) != 0) return NULL;
    return p;
  }
  static void vfft_aligned_free(void *p) { free(p); }
#endif

/* ═══════════════════════════════════════════════════════════════
 * CODELET SIGNATURES (match registry.h)
 * ═══════════════════════════════════════════════════════════════ */

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

/* ═══════════════════════════════════════════════════════════════
 * CANDIDATE REGISTRATION
 *
 * bench.py emits vfft_harness_candidates.c which provides:
 *   extern const candidate_t CANDIDATES[];
 *   extern const size_t N_CANDIDATES;
 * Each candidate names the symbol and ISA at compile time; the harness
 * picks by --variant at runtime.
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
  const char *variant;   /* e.g. "ct_t1s_dit" */
  const char *isa;       /* "avx2" | "avx512" | "scalar" */
  const char *protocol;  /* "flat" | "log3" | "t1s" | "log1_tight" */
  t1_fn fwd, bwd;
  int requires_avx512;   /* 1 if this candidate needs AVX-512 */
} candidate_t;

/* Provided by vfft_harness_candidates.c (per-ISA files concatenated at runtime) */
extern const candidate_t *candidate_at(size_t i);
extern size_t candidate_count(void);

/* Runtime AVX-512 check (for auto-skip on hosts that can't run it). */
static int host_has_avx512(void) {
  #if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
    #if defined(__AVX512F__)
      return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
    #else
      /* Compiler didn't enable avx512 in this TU; do a cpuid probe.
         Easiest on gcc/clang/icx: still call __builtin_cpu_supports — it
         reads from CPUID, independent of compile-time ISA. */
      return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
    #endif
  #elif defined(_MSC_VER)
    int cpuInfo[4];
    __cpuidex(cpuInfo, 7, 0);
    int has_f  = (cpuInfo[1] >> 16) & 1;
    int has_dq = (cpuInfo[1] >> 17) & 1;
    return has_f && has_dq;
  #else
    return 0;
  #endif
}

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLE POPULATION
 *
 * Each protocol has a different buffer size and layout. The harness
 * populates them with deterministic unit-magnitude values (cos/sin of a
 * reproducible phase); exact values don't matter for timing, but they
 * should not be degenerate (0 / NaN / subnormal).
 * ═══════════════════════════════════════════════════════════════ */

static size_t twiddle_doubles(const char *protocol, int R, size_t me) {
  if (strcmp(protocol, "flat")       == 0) return (size_t)(R - 1) * me;
  if (strcmp(protocol, "log3")       == 0) return me;
  if (strcmp(protocol, "t1s")        == 0) return (size_t)(R - 1);
  if (strcmp(protocol, "log1_tight") == 0) return (size_t)2 * me;
  fprintf(stderr, "unknown protocol: %s\n", protocol);
  exit(2);
}

static void populate_twiddles(const char *protocol, int R, size_t me,
                              double *W_re, double *W_im)
{
  const size_t n = twiddle_doubles(protocol, R, me);
  /* Use a deterministic phase per index. Values must be unit-circle
     (|w|=1) for bit-exact cross-protocol comparisons if needed. */
  for (size_t i = 0; i < n; i++) {
    double phase = 0.0037 * (double)(i + 1);
    W_re[i] = cos(phase);
    W_im[i] = sin(phase);
  }
}

/* ═══════════════════════════════════════════════════════════════
 * FFTW-STYLE TIMING
 * ═══════════════════════════════════════════════════════════════ */

#define TIME_MIN     (10.0 * 1e6)   /* 10 ms in ns */
#define TIME_REPEAT  8
#define TIME_LIMIT   (2.0 * 1e9)    /* 2 s hard cap in ns */

static double time_block(t1_fn fn, double *rio_re, double *rio_im,
                         const double *W_re, const double *W_im,
                         size_t ios, size_t me, int iters)
{
  double t0 = now_ns();
  for (int i = 0; i < iters; i++) {
    fn(rio_re, rio_im, W_re, W_im, ios, me);
  }
  return now_ns() - t0;
}

static double measure(t1_fn fn, double *rio_re, double *rio_im,
                      const double *W_re, const double *W_im,
                      size_t ios, size_t me)
{
  /* Warm caches, train branch predictor. */
  fn(rio_re, rio_im, W_re, W_im, ios, me);

  /* Find iter count that produces at least TIME_MIN per block. */
  int iters = 1;
  double total_ns = 0.0;
  double dt_block = 0.0;
  double global_start = now_ns();

  for (;;) {
    dt_block = time_block(fn, rio_re, rio_im, W_re, W_im, ios, me, iters);
    total_ns = now_ns() - global_start;
    if (dt_block >= TIME_MIN) break;
    if (total_ns >= TIME_LIMIT) break;
    iters *= 2;
    if (iters > (1 << 22)) break;  /* sanity cap */
  }

  /* Run TIME_REPEAT blocks of `iters`, track minimum. */
  double best_block_ns = dt_block;
  for (int r = 1; r < TIME_REPEAT; r++) {
    total_ns = now_ns() - global_start;
    if (total_ns >= TIME_LIMIT) break;
    double t = time_block(fn, rio_re, rio_im, W_re, W_im, ios, me, iters);
    if (t < best_block_ns) best_block_ns = t;
  }

  return best_block_ns / (double)iters;
}

/* ═══════════════════════════════════════════════════════════════
 * ARG PARSING
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
  const char *variant;
  const char *isa;
  const char *dir;      /* "fwd" | "bwd" */
  const char *protocol; /* overrides candidate's reported protocol (for log1_tight) */
  int R;
  size_t me;
  size_t ios;
  int list_only;
} args_t;

static int parse_args(int argc, char **argv, args_t *a) {
  memset(a, 0, sizeof(*a));
  a->dir = "fwd";
  for (int i = 1; i < argc; i++) {
    #define TAKE(flag, dst) do { \
      if (strcmp(argv[i], flag) == 0 && i + 1 < argc) { dst = argv[++i]; continue; } \
    } while (0)
    TAKE("--variant",  a->variant);
    TAKE("--isa",      a->isa);
    TAKE("--dir",      a->dir);
    TAKE("--protocol", a->protocol);
    if (strcmp(argv[i], "--radix") == 0 && i + 1 < argc) { a->R   = atoi(argv[++i]); continue; }
    if (strcmp(argv[i], "--me")    == 0 && i + 1 < argc) { a->me  = (size_t)strtoull(argv[++i], NULL, 10); continue; }
    if (strcmp(argv[i], "--ios")   == 0 && i + 1 < argc) { a->ios = (size_t)strtoull(argv[++i], NULL, 10); continue; }
    if (strcmp(argv[i], "--list")  == 0) { a->list_only = 1; continue; }
    #undef TAKE
  }
  return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════ */

static void emit_skip_result(const args_t *a, const char *reason) {
  printf("{\"variant\":\"%s\",\"isa\":\"%s\",\"protocol\":\"%s\","
         "\"radix\":%d,\"me\":%zu,\"ios\":%zu,\"dir\":\"%s\","
         "\"skipped\":true,\"reason\":\"%s\"}\n",
         a->variant ? a->variant : "",
         a->isa ? a->isa : "",
         a->protocol ? a->protocol : "",
         a->R, a->me, a->ios, a->dir, reason);
}

int main(int argc, char **argv) {
  args_t a;
  parse_args(argc, argv, &a);

  if (a.list_only) {
    size_t n = candidate_count();
    for (size_t i = 0; i < n; i++) {
      const candidate_t *c = candidate_at(i);
      printf("%s\t%s\t%s\t%d\n",
             c->variant, c->isa, c->protocol, c->requires_avx512);
    }
    return 0;
  }

  if (!a.variant || !a.isa || a.R == 0 || a.me == 0 || a.ios == 0) {
    fprintf(stderr,
      "usage: harness --variant <v> --isa <avx2|avx512|scalar> --dir <fwd|bwd> "
      "--radix <R> --me <N> --ios <N> [--protocol <p>]\n"
      "       harness --list\n");
    return 2;
  }

  /* Locate candidate. */
  const candidate_t *cand = NULL;
  size_t n = candidate_count();
  for (size_t i = 0; i < n; i++) {
    const candidate_t *c = candidate_at(i);
    if (strcmp(c->variant, a.variant) == 0 &&
        strcmp(c->isa,     a.isa)     == 0) {
      cand = c;
      break;
    }
  }
  if (!cand) {
    emit_skip_result(&a, "candidate_not_found");
    return 0;
  }

  /* Auto-skip AVX-512 candidates on hosts without AVX-512. */
  if (cand->requires_avx512 && !host_has_avx512()) {
    if (!a.protocol) a.protocol = cand->protocol;
    emit_skip_result(&a, "host_lacks_avx512");
    return 0;
  }

  /* Protocol override (for log1_tight experiment). */
  const char *proto = a.protocol ? a.protocol : cand->protocol;

  t1_fn fn = (strcmp(a.dir, "bwd") == 0) ? cand->bwd : cand->fwd;
  if (!fn) {
    emit_skip_result(&a, "direction_not_available");
    return 0;
  }

  /* Allocate: R rows of ios doubles each, per re/im. */
  size_t nbuf = (size_t)a.R * a.ios;
  double *rio_re = (double*)vfft_aligned_alloc(64, nbuf * sizeof(double));
  double *rio_im = (double*)vfft_aligned_alloc(64, nbuf * sizeof(double));

  /* Twiddle table per protocol. */
  size_t tw_doubles = twiddle_doubles(proto, a.R, a.me);
  /* Pad allocation by SIMD width so accidental over-reads land in valid memory. */
  size_t tw_alloc = tw_doubles + 16;
  double *W_re = (double*)vfft_aligned_alloc(64, tw_alloc * sizeof(double));
  double *W_im = (double*)vfft_aligned_alloc(64, tw_alloc * sizeof(double));

  if (!rio_re || !rio_im || !W_re || !W_im) {
    emit_skip_result(&a, "alloc_failed");
    return 0;
  }

  /* Seed rio with deterministic non-trivial data. */
  for (size_t i = 0; i < nbuf; i++) {
    double p = 0.00123 * (double)i;
    rio_re[i] = cos(p);
    rio_im[i] = sin(p);
  }
  populate_twiddles(proto, a.R, a.me, W_re, W_im);
  /* Zero the padding so debug tools don't flag uninit reads. */
  for (size_t i = tw_doubles; i < tw_alloc; i++) { W_re[i] = 0.0; W_im[i] = 0.0; }

  /* Measure. */
  double ns = measure(fn, rio_re, rio_im, W_re, W_im, a.ios, a.me);

  /* Emit result. */
  printf("{\"variant\":\"%s\",\"isa\":\"%s\",\"protocol\":\"%s\","
         "\"radix\":%d,\"me\":%zu,\"ios\":%zu,\"dir\":\"%s\","
         "\"ns\":%.3f}\n",
         a.variant, a.isa, proto, a.R, a.me, a.ios, a.dir, ns);

  vfft_aligned_free(rio_re);
  vfft_aligned_free(rio_im);
  vfft_aligned_free(W_re);
  vfft_aligned_free(W_im);
  return 0;
}
