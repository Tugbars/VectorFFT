/* validate.c — correctness validator for generated dispatchers.
 *
 * For each protocol dispatcher at each sweep point, runs the dispatcher
 * and a known-good reference codelet on identical inputs, then compares
 * outputs element-wise. Asserts max diff < 1e-10 (machine-epsilon cushion).
 *
 * Reference codelets per protocol:
 *   flat:  radix{R}_t1_dit_{dir}_{isa}         (flat codelet, full twiddle table)
 *   log3:  radix{R}_t1_dit_log3_{dir}_{isa}    (only one log3 variant exists)
 *   t1s:   radix{R}_t1s_dit_{dir}_{isa}        (only one t1s variant exists)
 *
 * The validator is parameterized at compile time by -DRADIX=4 and by the
 * includes the compiling driver passes. See bench.py (validate phase).
 *
 * Exit 0 on full pass; exit 1 on any failure.
 */
#define _POSIX_C_SOURCE 200809L
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32)
  #include <malloc.h>
  static void *vfft_aligned_alloc(size_t align, size_t size) {
    return _aligned_malloc(size, align);
  }
  static void vfft_aligned_free(void *p) { _aligned_free(p); }
#else
  #include <stdlib.h>
  static void *vfft_aligned_alloc(size_t align, size_t size) {
    void *p = NULL;
    size_t rounded = (size + align - 1) & ~(align - 1);
    if (posix_memalign(&p, align, rounded) != 0) return NULL;
    return p;
  }
  static void vfft_aligned_free(void *p) { free(p); }
#endif

/* ═══════════════════════════════════════════════════════════════
 * Signatures
 * ═══════════════════════════════════════════════════════════════ */

#ifndef RADIX
#  error "define -DRADIX=<n>"
#endif

typedef void (*t1_fn)(double *rio_re, double *rio_im,
                     const double *W_re, const double *W_im,
                     size_t ios, size_t me);

/* ═══════════════════════════════════════════════════════════════
 * Twiddle layouts (must match harness.c)
 * ═══════════════════════════════════════════════════════════════ */

static size_t tw_doubles(const char *protocol, int R, size_t me) {
  /* log3 uses the same buffer shape as flat — sparse reads convention.
   * See harness.c for the rationale. */
  if (strcmp(protocol, "flat")       == 0) return (size_t)(R - 1) * me;
  if (strcmp(protocol, "log3")       == 0) return (size_t)(R - 1) * me;
  if (strcmp(protocol, "t1s")        == 0) return (size_t)(R - 1);
  return 0;
}

static void populate_tw(const char *protocol, int R, size_t me,
                        double *W_re, double *W_im)
{
  /* Must use ACTUAL DFT twiddles — W[(j-1)*me + m] = exp(-2πi*j*m/(R*me)).
     This way W^k = (W^1)^k, making flat/log1/log3 variants produce
     identical results from the same buffer. log3 reads only slots at
     positions (2^k - 1) for k = 0..log2(R)-1; those slots still need
     to contain the correct DFT twiddles. Simplest: populate the entire
     (R-1)*me buffer the same way flat does. log3 just reads a subset. */
  const double TWO_PI = 6.28318530717958647692;
  if (strcmp(protocol, "flat") == 0 || strcmp(protocol, "log3") == 0) {
    for (int j = 1; j < R; j++) {
      for (size_t m = 0; m < me; m++) {
        double phase = -TWO_PI * (double)j * (double)m / ((double)R * (double)me);
        W_re[(j - 1) * me + m] = cos(phase);
        W_im[(j - 1) * me + m] = sin(phase);
      }
    }
  } else if (strcmp(protocol, "t1s") == 0) {
    /* t1s: (R-1) scalars — one per leg, constant for this call.
       Use j-th scalar = exp(-2πi*j/R) (the natural "ω^j" values). */
    for (int j = 1; j < R; j++) {
      double phase = -TWO_PI * (double)j / (double)R;
      W_re[j - 1] = cos(phase);
      W_im[j - 1] = sin(phase);
    }
  }
}

/* ═══════════════════════════════════════════════════════════════
 * One validation pass
 * ═══════════════════════════════════════════════════════════════ */

static int compare_buffers(const double *a_re, const double *a_im,
                           const double *b_re, const double *b_im,
                           size_t n, double tol, double *out_max_diff)
{
  double max_diff = 0.0;
  for (size_t i = 0; i < n; i++) {
    double dr = a_re[i] - b_re[i];
    double di = a_im[i] - b_im[i];
    double d = sqrt(dr*dr + di*di);
    if (d > max_diff) max_diff = d;
  }
  *out_max_diff = max_diff;
  return max_diff <= tol;
}

typedef struct {
  const char *name;     /* descriptive name for output */
  t1_fn ref_fn;         /* reference codelet (e.g. the flat baseline) */
  t1_fn disp_fn;        /* dispatcher or variant to validate */
  const char *protocol; /* twiddle layout */
  size_t min_me;        /* skip points with me < min_me (for buf: tile size) */
} validate_case_t;

static int run_case(const validate_case_t *c, int R, size_t me, size_t ios,
                    double tol)
{
  size_t nbuf = (size_t)R * ios;
  double *rio_re_ref  = vfft_aligned_alloc(64, nbuf * sizeof(double));
  double *rio_im_ref  = vfft_aligned_alloc(64, nbuf * sizeof(double));
  double *rio_re_disp = vfft_aligned_alloc(64, nbuf * sizeof(double));
  double *rio_im_disp = vfft_aligned_alloc(64, nbuf * sizeof(double));

  size_t twn = tw_doubles(c->protocol, R, me);
  double *W_re = vfft_aligned_alloc(64, (twn + 16) * sizeof(double));
  double *W_im = vfft_aligned_alloc(64, (twn + 16) * sizeof(double));

  /* Reproducible input. */
  for (size_t i = 0; i < nbuf; i++) {
    double p = 0.00123 * (double)i;
    rio_re_ref[i]  = rio_re_disp[i]  = cos(p);
    rio_im_ref[i]  = rio_im_disp[i]  = sin(p);
  }
  populate_tw(c->protocol, R, me, W_re, W_im);

  c->ref_fn (rio_re_ref,  rio_im_ref,  W_re, W_im, ios, me);
  c->disp_fn(rio_re_disp, rio_im_disp, W_re, W_im, ios, me);

  double max_diff;
  int ok = compare_buffers(rio_re_ref, rio_im_ref,
                           rio_re_disp, rio_im_disp, nbuf, tol, &max_diff);
  if (!ok) {
    fprintf(stderr, "  [FAIL] %s  me=%zu ios=%zu  max_diff=%.3e (tol=%.1e)\n",
            c->name, me, ios, max_diff, tol);
  }

  vfft_aligned_free(rio_re_ref);
  vfft_aligned_free(rio_im_ref);
  vfft_aligned_free(rio_re_disp);
  vfft_aligned_free(rio_im_disp);
  vfft_aligned_free(W_re);
  vfft_aligned_free(W_im);
  return ok;
}

/* ═══════════════════════════════════════════════════════════════
 * Cases and drive
 * ═══════════════════════════════════════════════════════════════ */

/* The driver compile-time wiring (via -D flags from select_and_emit or
 * a small glue header) provides the concrete (ref_fn, disp_fn, protocol)
 * triples for each ISA we validate. We wire them explicitly below. */

/* The validate binary is built with both ISAs' headers available; each
 * case below uses the appropriate ISA function pointers. The validator's
 * caller compiles this file with -I to staging/ and -I to generated/. */

#include <fft_radix_include.h>  /* generated shim with all relevant #includes */

static const validate_case_t CASES[] = {
#define ADD_CASE(TAG, ISA, DIR, PROTO, REF, DISP) \
  { TAG " " #ISA " " #DIR, REF, DISP, PROTO, 0 }
#define ADD_CASE_MIN_ME(TAG, ISA, DIR, PROTO, REF, DISP, MINME) \
  { TAG " " #ISA " " #DIR, REF, DISP, PROTO, MINME }

#if RADIX == 3
  /* R=3 tuning matrix:
   *   AVX2:    U=1 only for each of {flat, t1s, log3}  (3 variants)
   *   AVX-512: U=1/U=2/U=3 for each of {flat, t1s, log3}  (9 variants)
   *
   * Reference selection note: because populate_tw produces different
   * buffer contents per protocol (flat/log3 fill (R-1)*me doubles;
   * t1s fills only (R-1) scalars), a cross-protocol reference would
   * read out-of-bounds or garbage. Each protocol self-references:
   *   flat cases  -> reference is radix3_t1_dit_u1      (flat reader)
   *   log3 cases  -> reference is radix3_t1_dit_log3_u1 (log3 reader)
   *   t1s cases   -> reference is radix3_t1s_dit_u1     (t1s reader)
   *
   * This is intra-protocol validation: U=2 and U=3 within a family
   * bit-match U=1 of the same family. It does NOT cross-validate
   * that flat and t1s compute semantically equivalent transforms
   * (they operate on different twiddle buffers). That's consistent
   * with how R=4..R=64 validators are structured. */
  #if defined(VALIDATE_AVX2)
    /* Per-variant (AVX2 = U=1 only) — U=1 self-validates (trivial) */
    ADD_CASE("t1_dit_u1",       avx2, fwd, "flat",
             radix3_t1_dit_u1_fwd_avx2,       radix3_t1_dit_u1_fwd_avx2),
    ADD_CASE("t1_dit_u1",       avx2, bwd, "flat",
             radix3_t1_dit_u1_bwd_avx2,       radix3_t1_dit_u1_bwd_avx2),
    ADD_CASE("t1s_dit_u1",      avx2, fwd, "t1s",
             radix3_t1s_dit_u1_fwd_avx2,      radix3_t1s_dit_u1_fwd_avx2),
    ADD_CASE("t1s_dit_u1",      avx2, bwd, "t1s",
             radix3_t1s_dit_u1_bwd_avx2,      radix3_t1s_dit_u1_bwd_avx2),
    ADD_CASE("t1_dit_log3_u1",  avx2, fwd, "log3",
             radix3_t1_dit_log3_u1_fwd_avx2,  radix3_t1_dit_log3_u1_fwd_avx2),
    ADD_CASE("t1_dit_log3_u1",  avx2, bwd, "log3",
             radix3_t1_dit_log3_u1_bwd_avx2,  radix3_t1_dit_log3_u1_bwd_avx2),
    /* Dispatcher-level */
    ADD_CASE("t1_dit",      avx2, fwd, "flat",
             radix3_t1_dit_u1_fwd_avx2,       vfft_r3_t1_dit_dispatch_fwd_avx2),
    ADD_CASE("t1_dit",      avx2, bwd, "flat",
             radix3_t1_dit_u1_bwd_avx2,       vfft_r3_t1_dit_dispatch_bwd_avx2),
    ADD_CASE("t1s_dit",     avx2, fwd, "t1s",
             radix3_t1s_dit_u1_fwd_avx2,      vfft_r3_t1s_dit_dispatch_fwd_avx2),
    ADD_CASE("t1s_dit",     avx2, bwd, "t1s",
             radix3_t1s_dit_u1_bwd_avx2,      vfft_r3_t1s_dit_dispatch_bwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, fwd, "log3",
             radix3_t1_dit_log3_u1_fwd_avx2,  vfft_r3_t1_dit_log3_dispatch_fwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, bwd, "log3",
             radix3_t1_dit_log3_u1_bwd_avx2,  vfft_r3_t1_dit_log3_dispatch_bwd_avx2),
  #endif
  #if defined(VALIDATE_AVX512)
    /* Per-variant: U=1 self, U=2/U=3 vs U=1 within same protocol */
    ADD_CASE("t1_dit_u1",       avx512, fwd, "flat",
             radix3_t1_dit_u1_fwd_avx512,       radix3_t1_dit_u1_fwd_avx512),
    ADD_CASE("t1_dit_u1",       avx512, bwd, "flat",
             radix3_t1_dit_u1_bwd_avx512,       radix3_t1_dit_u1_bwd_avx512),
    ADD_CASE("t1_dit_u2",       avx512, fwd, "flat",
             radix3_t1_dit_u1_fwd_avx512,       radix3_t1_dit_u2_fwd_avx512),
    ADD_CASE("t1_dit_u2",       avx512, bwd, "flat",
             radix3_t1_dit_u1_bwd_avx512,       radix3_t1_dit_u2_bwd_avx512),
    ADD_CASE("t1_dit_u3",       avx512, fwd, "flat",
             radix3_t1_dit_u1_fwd_avx512,       radix3_t1_dit_u3_fwd_avx512),
    ADD_CASE("t1_dit_u3",       avx512, bwd, "flat",
             radix3_t1_dit_u1_bwd_avx512,       radix3_t1_dit_u3_bwd_avx512),
    ADD_CASE("t1s_dit_u1",      avx512, fwd, "t1s",
             radix3_t1s_dit_u1_fwd_avx512,      radix3_t1s_dit_u1_fwd_avx512),
    ADD_CASE("t1s_dit_u1",      avx512, bwd, "t1s",
             radix3_t1s_dit_u1_bwd_avx512,      radix3_t1s_dit_u1_bwd_avx512),
    ADD_CASE("t1s_dit_u2",      avx512, fwd, "t1s",
             radix3_t1s_dit_u1_fwd_avx512,      radix3_t1s_dit_u2_fwd_avx512),
    ADD_CASE("t1s_dit_u2",      avx512, bwd, "t1s",
             radix3_t1s_dit_u1_bwd_avx512,      radix3_t1s_dit_u2_bwd_avx512),
    ADD_CASE("t1s_dit_u3",      avx512, fwd, "t1s",
             radix3_t1s_dit_u1_fwd_avx512,      radix3_t1s_dit_u3_fwd_avx512),
    ADD_CASE("t1s_dit_u3",      avx512, bwd, "t1s",
             radix3_t1s_dit_u1_bwd_avx512,      radix3_t1s_dit_u3_bwd_avx512),
    ADD_CASE("t1_dit_log3_u1",  avx512, fwd, "log3",
             radix3_t1_dit_log3_u1_fwd_avx512,  radix3_t1_dit_log3_u1_fwd_avx512),
    ADD_CASE("t1_dit_log3_u1",  avx512, bwd, "log3",
             radix3_t1_dit_log3_u1_bwd_avx512,  radix3_t1_dit_log3_u1_bwd_avx512),
    ADD_CASE("t1_dit_log3_u2",  avx512, fwd, "log3",
             radix3_t1_dit_log3_u1_fwd_avx512,  radix3_t1_dit_log3_u2_fwd_avx512),
    ADD_CASE("t1_dit_log3_u2",  avx512, bwd, "log3",
             radix3_t1_dit_log3_u1_bwd_avx512,  radix3_t1_dit_log3_u2_bwd_avx512),
    ADD_CASE("t1_dit_log3_u3",  avx512, fwd, "log3",
             radix3_t1_dit_log3_u1_fwd_avx512,  radix3_t1_dit_log3_u3_fwd_avx512),
    ADD_CASE("t1_dit_log3_u3",  avx512, bwd, "log3",
             radix3_t1_dit_log3_u1_bwd_avx512,  radix3_t1_dit_log3_u3_bwd_avx512),
    /* Dispatcher-level */
    ADD_CASE("t1_dit",      avx512, fwd, "flat",
             radix3_t1_dit_u1_fwd_avx512,       vfft_r3_t1_dit_dispatch_fwd_avx512),
    ADD_CASE("t1_dit",      avx512, bwd, "flat",
             radix3_t1_dit_u1_bwd_avx512,       vfft_r3_t1_dit_dispatch_bwd_avx512),
    ADD_CASE("t1s_dit",     avx512, fwd, "t1s",
             radix3_t1s_dit_u1_fwd_avx512,      vfft_r3_t1s_dit_dispatch_fwd_avx512),
    ADD_CASE("t1s_dit",     avx512, bwd, "t1s",
             radix3_t1s_dit_u1_bwd_avx512,      vfft_r3_t1s_dit_dispatch_bwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, fwd, "log3",
             radix3_t1_dit_log3_u1_fwd_avx512,  vfft_r3_t1_dit_log3_dispatch_fwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, bwd, "log3",
             radix3_t1_dit_log3_u1_bwd_avx512,  vfft_r3_t1_dit_log3_dispatch_bwd_avx512),
  #endif

#elif RADIX == 4
  #if defined(VALIDATE_AVX2)
    ADD_CASE("flat", avx2, fwd, "flat",
             radix4_t1_dit_fwd_avx2,        vfft_r4_t1_dit_dispatch_fwd_avx2),
    ADD_CASE("flat", avx2, bwd, "flat",
             radix4_t1_dit_bwd_avx2,        vfft_r4_t1_dit_dispatch_bwd_avx2),
    ADD_CASE("log3", avx2, fwd, "log3",
             radix4_t1_dit_log3_fwd_avx2,   vfft_r4_t1_dit_log3_dispatch_fwd_avx2),
    ADD_CASE("log3", avx2, bwd, "log3",
             radix4_t1_dit_log3_bwd_avx2,   vfft_r4_t1_dit_log3_dispatch_bwd_avx2),
    ADD_CASE("t1s",  avx2, fwd, "t1s",
             radix4_t1s_dit_fwd_avx2,       vfft_r4_t1s_dit_dispatch_fwd_avx2),
    ADD_CASE("t1s",  avx2, bwd, "t1s",
             radix4_t1s_dit_bwd_avx2,       vfft_r4_t1s_dit_dispatch_bwd_avx2),
  #endif
  #if defined(VALIDATE_AVX512)
    ADD_CASE("flat", avx512, fwd, "flat",
             radix4_t1_dit_fwd_avx512,      vfft_r4_t1_dit_dispatch_fwd_avx512),
    ADD_CASE("flat", avx512, bwd, "flat",
             radix4_t1_dit_bwd_avx512,      vfft_r4_t1_dit_dispatch_bwd_avx512),
    ADD_CASE("log3", avx512, fwd, "log3",
             radix4_t1_dit_log3_fwd_avx512, vfft_r4_t1_dit_log3_dispatch_fwd_avx512),
    ADD_CASE("log3", avx512, bwd, "log3",
             radix4_t1_dit_log3_bwd_avx512, vfft_r4_t1_dit_log3_dispatch_bwd_avx512),
    ADD_CASE("t1s",  avx512, fwd, "t1s",
             radix4_t1s_dit_fwd_avx512,     vfft_r4_t1s_dit_dispatch_fwd_avx512),
    ADD_CASE("t1s",  avx512, bwd, "t1s",
             radix4_t1s_dit_bwd_avx512,     vfft_r4_t1s_dit_dispatch_bwd_avx512),
  #endif

#elif RADIX == 8
  /* R=8 has four dispatchers:
   *   t1_dit      : variants dit / dit_prefetch / dit_log1 / dit_u2(AVX-512)
   *                 reference = radix8_t1_dit (flat protocol)
   *   t1_dif      : variants dif / dif_prefetch
   *                 reference = radix8_t1_dif (flat protocol)
   *   t1_dit_log3 : single variant t1_dit_log3
   *                 reference = self (log3 protocol — different twiddle layout)
   *   t1s_dit     : single variant t1s_dit
   *                 reference = self (t1s protocol — 7 scalars)
   * Each dispatcher validates against its own family's flat reference
   * codelet. log3 and t1s self-validate (dispatcher just wraps the sole
   * variant) since no other codelet reads their twiddle layout. */
  #if defined(VALIDATE_AVX2)
    ADD_CASE("t1_dit",     avx2, fwd, "flat",
             radix8_t1_dit_fwd_avx2,       vfft_r8_t1_dit_dispatch_fwd_avx2),
    ADD_CASE("t1_dit",     avx2, bwd, "flat",
             radix8_t1_dit_bwd_avx2,       vfft_r8_t1_dit_dispatch_bwd_avx2),
    ADD_CASE("t1_dif",     avx2, fwd, "flat",
             radix8_t1_dif_fwd_avx2,       vfft_r8_t1_dif_dispatch_fwd_avx2),
    ADD_CASE("t1_dif",     avx2, bwd, "flat",
             radix8_t1_dif_bwd_avx2,       vfft_r8_t1_dif_dispatch_bwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, fwd, "log3",
             radix8_t1_dit_log3_fwd_avx2,  vfft_r8_t1_dit_log3_dispatch_fwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, bwd, "log3",
             radix8_t1_dit_log3_bwd_avx2,  vfft_r8_t1_dit_log3_dispatch_bwd_avx2),
    ADD_CASE("t1s_dit",    avx2, fwd, "t1s",
             radix8_t1s_dit_fwd_avx2,      vfft_r8_t1s_dit_dispatch_fwd_avx2),
    ADD_CASE("t1s_dit",    avx2, bwd, "t1s",
             radix8_t1s_dit_bwd_avx2,      vfft_r8_t1s_dit_dispatch_bwd_avx2),
  #endif
  #if defined(VALIDATE_AVX512)
    ADD_CASE("t1_dit",     avx512, fwd, "flat",
             radix8_t1_dit_fwd_avx512,     vfft_r8_t1_dit_dispatch_fwd_avx512),
    ADD_CASE("t1_dit",     avx512, bwd, "flat",
             radix8_t1_dit_bwd_avx512,     vfft_r8_t1_dit_dispatch_bwd_avx512),
    ADD_CASE("t1_dif",     avx512, fwd, "flat",
             radix8_t1_dif_fwd_avx512,     vfft_r8_t1_dif_dispatch_fwd_avx512),
    ADD_CASE("t1_dif",     avx512, bwd, "flat",
             radix8_t1_dif_bwd_avx512,     vfft_r8_t1_dif_dispatch_bwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, fwd, "log3",
             radix8_t1_dit_log3_fwd_avx512, vfft_r8_t1_dit_log3_dispatch_fwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, bwd, "log3",
             radix8_t1_dit_log3_bwd_avx512, vfft_r8_t1_dit_log3_dispatch_bwd_avx512),
    ADD_CASE("t1s_dit",    avx512, fwd, "t1s",
             radix8_t1s_dit_fwd_avx512,    vfft_r8_t1s_dit_dispatch_fwd_avx512),
    ADD_CASE("t1s_dit",    avx512, bwd, "t1s",
             radix8_t1s_dit_bwd_avx512,    vfft_r8_t1s_dit_dispatch_bwd_avx512),
  #endif

#elif RADIX == 16
  /* R=16 Phase A has four dispatchers, one variant each:
   *   t1_dit      : ct_t1_dit        (flat)
   *   t1_dif      : ct_t1_dif        (flat) — different function from DIT
   *   t1_dit_log3 : ct_t1_dit_log3   (log3 protocol — sparse read)
   *   t1s_dit     : ct_t1s_dit       (t1s protocol — scalar broadcasts)
   * Each dispatcher self-validates since it wraps a single variant. */
  #if defined(VALIDATE_AVX2)
    ADD_CASE("t1_dit",     avx2, fwd, "flat",
             radix16_t1_dit_fwd_avx2,       vfft_r16_t1_dit_dispatch_fwd_avx2),
    ADD_CASE("t1_dit",     avx2, bwd, "flat",
             radix16_t1_dit_bwd_avx2,       vfft_r16_t1_dit_dispatch_bwd_avx2),
    ADD_CASE("t1_dif",     avx2, fwd, "flat",
             radix16_t1_dif_fwd_avx2,       vfft_r16_t1_dif_dispatch_fwd_avx2),
    ADD_CASE("t1_dif",     avx2, bwd, "flat",
             radix16_t1_dif_bwd_avx2,       vfft_r16_t1_dif_dispatch_bwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, fwd, "log3",
             radix16_t1_dit_log3_fwd_avx2,  vfft_r16_t1_dit_log3_dispatch_fwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, bwd, "log3",
             radix16_t1_dit_log3_bwd_avx2,  vfft_r16_t1_dit_log3_dispatch_bwd_avx2),
    ADD_CASE("t1s_dit",    avx2, fwd, "t1s",
             radix16_t1s_dit_fwd_avx2,      vfft_r16_t1s_dit_dispatch_fwd_avx2),
    ADD_CASE("t1s_dit",    avx2, bwd, "t1s",
             radix16_t1s_dit_bwd_avx2,      vfft_r16_t1s_dit_dispatch_bwd_avx2),
  #endif
  #if defined(VALIDATE_AVX512)
    ADD_CASE("t1_dit",     avx512, fwd, "flat",
             radix16_t1_dit_fwd_avx512,       vfft_r16_t1_dit_dispatch_fwd_avx512),
    ADD_CASE("t1_dit",     avx512, bwd, "flat",
             radix16_t1_dit_bwd_avx512,       vfft_r16_t1_dit_dispatch_bwd_avx512),
    ADD_CASE("t1_dif",     avx512, fwd, "flat",
             radix16_t1_dif_fwd_avx512,       vfft_r16_t1_dif_dispatch_fwd_avx512),
    ADD_CASE("t1_dif",     avx512, bwd, "flat",
             radix16_t1_dif_bwd_avx512,       vfft_r16_t1_dif_dispatch_bwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, fwd, "log3",
             radix16_t1_dit_log3_fwd_avx512,  vfft_r16_t1_dit_log3_dispatch_fwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, bwd, "log3",
             radix16_t1_dit_log3_bwd_avx512,  vfft_r16_t1_dit_log3_dispatch_bwd_avx512),
    ADD_CASE("t1s_dit",    avx512, fwd, "t1s",
             radix16_t1s_dit_fwd_avx512,      vfft_r16_t1s_dit_dispatch_fwd_avx512),
    ADD_CASE("t1s_dit",    avx512, bwd, "t1s",
             radix16_t1s_dit_bwd_avx512,      vfft_r16_t1s_dit_dispatch_bwd_avx512),
  #endif
  /* ── Phase B: buf variants (3 tiles × 2 drains × 2 ISAs × 2 dirs = 24 cases) ──
   * Each buf variant validates directly (not via dispatcher) against the
   * flat t1_dit reference. The dispatcher-level validator test at the top
   * only catches bugs in whatever variant the dispatcher happens to pick
   * at each me; testing each variant individually catches bugs in all six.
   * min_me = tile size: buf kernel requires me >= tile for a full iteration. */
  #if defined(VALIDATE_AVX2)
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx2, fwd, "flat",
                    radix16_t1_dit_fwd_avx2, radix16_t1_buf_dit_tile64_temporal_fwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx2, bwd, "flat",
                    radix16_t1_dit_bwd_avx2, radix16_t1_buf_dit_tile64_temporal_bwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx2, fwd, "flat",
                    radix16_t1_dit_fwd_avx2, radix16_t1_buf_dit_tile64_stream_fwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx2, bwd, "flat",
                    radix16_t1_dit_bwd_avx2, radix16_t1_buf_dit_tile64_stream_bwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx2, fwd, "flat",
                    radix16_t1_dit_fwd_avx2, radix16_t1_buf_dit_tile128_temporal_fwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx2, bwd, "flat",
                    radix16_t1_dit_bwd_avx2, radix16_t1_buf_dit_tile128_temporal_bwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx2, fwd, "flat",
                    radix16_t1_dit_fwd_avx2, radix16_t1_buf_dit_tile128_stream_fwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx2, bwd, "flat",
                    radix16_t1_dit_bwd_avx2, radix16_t1_buf_dit_tile128_stream_bwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx2, fwd, "flat",
                    radix16_t1_dit_fwd_avx2, radix16_t1_buf_dit_tile256_temporal_fwd_avx2, 256),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx2, bwd, "flat",
                    radix16_t1_dit_bwd_avx2, radix16_t1_buf_dit_tile256_temporal_bwd_avx2, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx2, fwd, "flat",
                    radix16_t1_dit_fwd_avx2, radix16_t1_buf_dit_tile256_stream_fwd_avx2, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx2, bwd, "flat",
                    radix16_t1_dit_bwd_avx2, radix16_t1_buf_dit_tile256_stream_bwd_avx2, 256),
    /* Dispatcher-level validation (checks whichever variant the dispatcher picks at each me) */
    ADD_CASE("t1_buf_dit", avx2, fwd, "flat",
             radix16_t1_dit_fwd_avx2, vfft_r16_t1_buf_dit_dispatch_fwd_avx2),
    ADD_CASE("t1_buf_dit", avx2, bwd, "flat",
             radix16_t1_dit_bwd_avx2, vfft_r16_t1_buf_dit_dispatch_bwd_avx2),
  #endif
  #if defined(VALIDATE_AVX512)
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx512, fwd, "flat",
                    radix16_t1_dit_fwd_avx512, radix16_t1_buf_dit_tile64_temporal_fwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx512, bwd, "flat",
                    radix16_t1_dit_bwd_avx512, radix16_t1_buf_dit_tile64_temporal_bwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx512, fwd, "flat",
                    radix16_t1_dit_fwd_avx512, radix16_t1_buf_dit_tile64_stream_fwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx512, bwd, "flat",
                    radix16_t1_dit_bwd_avx512, radix16_t1_buf_dit_tile64_stream_bwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx512, fwd, "flat",
                    radix16_t1_dit_fwd_avx512, radix16_t1_buf_dit_tile128_temporal_fwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx512, bwd, "flat",
                    radix16_t1_dit_bwd_avx512, radix16_t1_buf_dit_tile128_temporal_bwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx512, fwd, "flat",
                    radix16_t1_dit_fwd_avx512, radix16_t1_buf_dit_tile128_stream_fwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx512, bwd, "flat",
                    radix16_t1_dit_bwd_avx512, radix16_t1_buf_dit_tile128_stream_bwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx512, fwd, "flat",
                    radix16_t1_dit_fwd_avx512, radix16_t1_buf_dit_tile256_temporal_fwd_avx512, 256),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx512, bwd, "flat",
                    radix16_t1_dit_bwd_avx512, radix16_t1_buf_dit_tile256_temporal_bwd_avx512, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx512, fwd, "flat",
                    radix16_t1_dit_fwd_avx512, radix16_t1_buf_dit_tile256_stream_fwd_avx512, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx512, bwd, "flat",
                    radix16_t1_dit_bwd_avx512, radix16_t1_buf_dit_tile256_stream_bwd_avx512, 256),
    ADD_CASE("t1_buf_dit", avx512, fwd, "flat",
             radix16_t1_dit_fwd_avx512, vfft_r16_t1_buf_dit_dispatch_fwd_avx512),
    ADD_CASE("t1_buf_dit", avx512, bwd, "flat",
             radix16_t1_dit_bwd_avx512, vfft_r16_t1_buf_dit_dispatch_bwd_avx512),
  #endif

#elif RADIX == 32
  /* R=32 Phase A: same structure as R=16 — 4 dispatchers, one variant each. */
  #if defined(VALIDATE_AVX2)
    ADD_CASE("t1_dit",     avx2, fwd, "flat",
             radix32_t1_dit_fwd_avx2,       vfft_r32_t1_dit_dispatch_fwd_avx2),
    ADD_CASE("t1_dit",     avx2, bwd, "flat",
             radix32_t1_dit_bwd_avx2,       vfft_r32_t1_dit_dispatch_bwd_avx2),
    ADD_CASE("t1_dif",     avx2, fwd, "flat",
             radix32_t1_dif_fwd_avx2,       vfft_r32_t1_dif_dispatch_fwd_avx2),
    ADD_CASE("t1_dif",     avx2, bwd, "flat",
             radix32_t1_dif_bwd_avx2,       vfft_r32_t1_dif_dispatch_bwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, fwd, "log3",
             radix32_t1_dit_log3_fwd_avx2,  vfft_r32_t1_dit_log3_dispatch_fwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, bwd, "log3",
             radix32_t1_dit_log3_bwd_avx2,  vfft_r32_t1_dit_log3_dispatch_bwd_avx2),
    ADD_CASE("t1s_dit",    avx2, fwd, "t1s",
             radix32_t1s_dit_fwd_avx2,      vfft_r32_t1s_dit_dispatch_fwd_avx2),
    ADD_CASE("t1s_dit",    avx2, bwd, "t1s",
             radix32_t1s_dit_bwd_avx2,      vfft_r32_t1s_dit_dispatch_bwd_avx2),
  #endif
  #if defined(VALIDATE_AVX512)
    ADD_CASE("t1_dit",     avx512, fwd, "flat",
             radix32_t1_dit_fwd_avx512,       vfft_r32_t1_dit_dispatch_fwd_avx512),
    ADD_CASE("t1_dit",     avx512, bwd, "flat",
             radix32_t1_dit_bwd_avx512,       vfft_r32_t1_dit_dispatch_bwd_avx512),
    ADD_CASE("t1_dif",     avx512, fwd, "flat",
             radix32_t1_dif_fwd_avx512,       vfft_r32_t1_dif_dispatch_fwd_avx512),
    ADD_CASE("t1_dif",     avx512, bwd, "flat",
             radix32_t1_dif_bwd_avx512,       vfft_r32_t1_dif_dispatch_bwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, fwd, "log3",
             radix32_t1_dit_log3_fwd_avx512,  vfft_r32_t1_dit_log3_dispatch_fwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, bwd, "log3",
             radix32_t1_dit_log3_bwd_avx512,  vfft_r32_t1_dit_log3_dispatch_bwd_avx512),
    ADD_CASE("t1s_dit",    avx512, fwd, "t1s",
             radix32_t1s_dit_fwd_avx512,      vfft_r32_t1s_dit_dispatch_fwd_avx512),
    ADD_CASE("t1s_dit",    avx512, bwd, "t1s",
             radix32_t1s_dit_bwd_avx512,      vfft_r32_t1s_dit_dispatch_bwd_avx512),
  #endif
  /* ── R=32 Phase B: buf variants (3 tiles × 2 drains × 2 ISAs × 2 dirs = 24 cases) ── */
  #if defined(VALIDATE_AVX2)
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx2, fwd, "flat",
                    radix32_t1_dit_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_fwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx2, bwd, "flat",
                    radix32_t1_dit_bwd_avx2, radix32_t1_buf_dit_tile64_temporal_bwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx2, fwd, "flat",
                    radix32_t1_dit_fwd_avx2, radix32_t1_buf_dit_tile64_stream_fwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx2, bwd, "flat",
                    radix32_t1_dit_bwd_avx2, radix32_t1_buf_dit_tile64_stream_bwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx2, fwd, "flat",
                    radix32_t1_dit_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_fwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx2, bwd, "flat",
                    radix32_t1_dit_bwd_avx2, radix32_t1_buf_dit_tile128_temporal_bwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx2, fwd, "flat",
                    radix32_t1_dit_fwd_avx2, radix32_t1_buf_dit_tile128_stream_fwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx2, bwd, "flat",
                    radix32_t1_dit_bwd_avx2, radix32_t1_buf_dit_tile128_stream_bwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx2, fwd, "flat",
                    radix32_t1_dit_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_fwd_avx2, 256),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx2, bwd, "flat",
                    radix32_t1_dit_bwd_avx2, radix32_t1_buf_dit_tile256_temporal_bwd_avx2, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx2, fwd, "flat",
                    radix32_t1_dit_fwd_avx2, radix32_t1_buf_dit_tile256_stream_fwd_avx2, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx2, bwd, "flat",
                    radix32_t1_dit_bwd_avx2, radix32_t1_buf_dit_tile256_stream_bwd_avx2, 256),
    ADD_CASE("t1_buf_dit", avx2, fwd, "flat",
             radix32_t1_dit_fwd_avx2, vfft_r32_t1_buf_dit_dispatch_fwd_avx2),
    ADD_CASE("t1_buf_dit", avx2, bwd, "flat",
             radix32_t1_dit_bwd_avx2, vfft_r32_t1_buf_dit_dispatch_bwd_avx2),
  #endif
  #if defined(VALIDATE_AVX512)
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx512, fwd, "flat",
                    radix32_t1_dit_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_fwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx512, bwd, "flat",
                    radix32_t1_dit_bwd_avx512, radix32_t1_buf_dit_tile64_temporal_bwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx512, fwd, "flat",
                    radix32_t1_dit_fwd_avx512, radix32_t1_buf_dit_tile64_stream_fwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx512, bwd, "flat",
                    radix32_t1_dit_bwd_avx512, radix32_t1_buf_dit_tile64_stream_bwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx512, fwd, "flat",
                    radix32_t1_dit_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_fwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx512, bwd, "flat",
                    radix32_t1_dit_bwd_avx512, radix32_t1_buf_dit_tile128_temporal_bwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx512, fwd, "flat",
                    radix32_t1_dit_fwd_avx512, radix32_t1_buf_dit_tile128_stream_fwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx512, bwd, "flat",
                    radix32_t1_dit_bwd_avx512, radix32_t1_buf_dit_tile128_stream_bwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx512, fwd, "flat",
                    radix32_t1_dit_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_fwd_avx512, 256),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx512, bwd, "flat",
                    radix32_t1_dit_bwd_avx512, radix32_t1_buf_dit_tile256_temporal_bwd_avx512, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx512, fwd, "flat",
                    radix32_t1_dit_fwd_avx512, radix32_t1_buf_dit_tile256_stream_fwd_avx512, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx512, bwd, "flat",
                    radix32_t1_dit_bwd_avx512, radix32_t1_buf_dit_tile256_stream_bwd_avx512, 256),
    ADD_CASE("t1_buf_dit", avx512, fwd, "flat",
             radix32_t1_dit_fwd_avx512, vfft_r32_t1_buf_dit_dispatch_fwd_avx512),
    ADD_CASE("t1_buf_dit", avx512, bwd, "flat",
             radix32_t1_dit_bwd_avx512, vfft_r32_t1_buf_dit_dispatch_bwd_avx512),
  #endif

#elif RADIX == 64
  /* R=64 Phase A: same structure as R=16/R=32 — 4 dispatchers, one variant each. */
  #if defined(VALIDATE_AVX2)
    ADD_CASE("t1_dit",     avx2, fwd, "flat",
             radix64_t1_dit_fwd_avx2,       vfft_r64_t1_dit_dispatch_fwd_avx2),
    ADD_CASE("t1_dit",     avx2, bwd, "flat",
             radix64_t1_dit_bwd_avx2,       vfft_r64_t1_dit_dispatch_bwd_avx2),
    ADD_CASE("t1_dif",     avx2, fwd, "flat",
             radix64_t1_dif_fwd_avx2,       vfft_r64_t1_dif_dispatch_fwd_avx2),
    ADD_CASE("t1_dif",     avx2, bwd, "flat",
             radix64_t1_dif_bwd_avx2,       vfft_r64_t1_dif_dispatch_bwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, fwd, "log3",
             radix64_t1_dit_log3_fwd_avx2,  vfft_r64_t1_dit_log3_dispatch_fwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, bwd, "log3",
             radix64_t1_dit_log3_bwd_avx2,  vfft_r64_t1_dit_log3_dispatch_bwd_avx2),
    ADD_CASE("t1s_dit",    avx2, fwd, "t1s",
             radix64_t1s_dit_fwd_avx2,      vfft_r64_t1s_dit_dispatch_fwd_avx2),
    ADD_CASE("t1s_dit",    avx2, bwd, "t1s",
             radix64_t1s_dit_bwd_avx2,      vfft_r64_t1s_dit_dispatch_bwd_avx2),
  #endif
  #if defined(VALIDATE_AVX512)
    ADD_CASE("t1_dit",     avx512, fwd, "flat",
             radix64_t1_dit_fwd_avx512,       vfft_r64_t1_dit_dispatch_fwd_avx512),
    ADD_CASE("t1_dit",     avx512, bwd, "flat",
             radix64_t1_dit_bwd_avx512,       vfft_r64_t1_dit_dispatch_bwd_avx512),
    ADD_CASE("t1_dif",     avx512, fwd, "flat",
             radix64_t1_dif_fwd_avx512,       vfft_r64_t1_dif_dispatch_fwd_avx512),
    ADD_CASE("t1_dif",     avx512, bwd, "flat",
             radix64_t1_dif_bwd_avx512,       vfft_r64_t1_dif_dispatch_bwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, fwd, "log3",
             radix64_t1_dit_log3_fwd_avx512,  vfft_r64_t1_dit_log3_dispatch_fwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, bwd, "log3",
             radix64_t1_dit_log3_bwd_avx512,  vfft_r64_t1_dit_log3_dispatch_bwd_avx512),
    ADD_CASE("t1s_dit",    avx512, fwd, "t1s",
             radix64_t1s_dit_fwd_avx512,      vfft_r64_t1s_dit_dispatch_fwd_avx512),
    ADD_CASE("t1s_dit",    avx512, bwd, "t1s",
             radix64_t1s_dit_bwd_avx512,      vfft_r64_t1s_dit_dispatch_bwd_avx512),
  #endif
  /* ── R=64 Phase B: buf variants (3 tiles × 2 drains × 2 ISAs × 2 dirs = 24 cases) ── */
  #if defined(VALIDATE_AVX2)
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx2, fwd, "flat",
                    radix64_t1_dit_fwd_avx2, radix64_t1_buf_dit_tile64_temporal_fwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx2, bwd, "flat",
                    radix64_t1_dit_bwd_avx2, radix64_t1_buf_dit_tile64_temporal_bwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx2, fwd, "flat",
                    radix64_t1_dit_fwd_avx2, radix64_t1_buf_dit_tile64_stream_fwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx2, bwd, "flat",
                    radix64_t1_dit_bwd_avx2, radix64_t1_buf_dit_tile64_stream_bwd_avx2, 64),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx2, fwd, "flat",
                    radix64_t1_dit_fwd_avx2, radix64_t1_buf_dit_tile128_temporal_fwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx2, bwd, "flat",
                    radix64_t1_dit_bwd_avx2, radix64_t1_buf_dit_tile128_temporal_bwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx2, fwd, "flat",
                    radix64_t1_dit_fwd_avx2, radix64_t1_buf_dit_tile128_stream_fwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx2, bwd, "flat",
                    radix64_t1_dit_bwd_avx2, radix64_t1_buf_dit_tile128_stream_bwd_avx2, 128),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx2, fwd, "flat",
                    radix64_t1_dit_fwd_avx2, radix64_t1_buf_dit_tile256_temporal_fwd_avx2, 256),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx2, bwd, "flat",
                    radix64_t1_dit_bwd_avx2, radix64_t1_buf_dit_tile256_temporal_bwd_avx2, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx2, fwd, "flat",
                    radix64_t1_dit_fwd_avx2, radix64_t1_buf_dit_tile256_stream_fwd_avx2, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx2, bwd, "flat",
                    radix64_t1_dit_bwd_avx2, radix64_t1_buf_dit_tile256_stream_bwd_avx2, 256),
    ADD_CASE("t1_buf_dit", avx2, fwd, "flat",
             radix64_t1_dit_fwd_avx2, vfft_r64_t1_buf_dit_dispatch_fwd_avx2),
    ADD_CASE("t1_buf_dit", avx2, bwd, "flat",
             radix64_t1_dit_bwd_avx2, vfft_r64_t1_buf_dit_dispatch_bwd_avx2),
  #endif
  #if defined(VALIDATE_AVX512)
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx512, fwd, "flat",
                    radix64_t1_dit_fwd_avx512, radix64_t1_buf_dit_tile64_temporal_fwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile64_temporal",  avx512, bwd, "flat",
                    radix64_t1_dit_bwd_avx512, radix64_t1_buf_dit_tile64_temporal_bwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx512, fwd, "flat",
                    radix64_t1_dit_fwd_avx512, radix64_t1_buf_dit_tile64_stream_fwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile64_stream",    avx512, bwd, "flat",
                    radix64_t1_dit_bwd_avx512, radix64_t1_buf_dit_tile64_stream_bwd_avx512, 64),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx512, fwd, "flat",
                    radix64_t1_dit_fwd_avx512, radix64_t1_buf_dit_tile128_temporal_fwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile128_temporal", avx512, bwd, "flat",
                    radix64_t1_dit_bwd_avx512, radix64_t1_buf_dit_tile128_temporal_bwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx512, fwd, "flat",
                    radix64_t1_dit_fwd_avx512, radix64_t1_buf_dit_tile128_stream_fwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile128_stream",   avx512, bwd, "flat",
                    radix64_t1_dit_bwd_avx512, radix64_t1_buf_dit_tile128_stream_bwd_avx512, 128),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx512, fwd, "flat",
                    radix64_t1_dit_fwd_avx512, radix64_t1_buf_dit_tile256_temporal_fwd_avx512, 256),
    ADD_CASE_MIN_ME("buf_tile256_temporal", avx512, bwd, "flat",
                    radix64_t1_dit_bwd_avx512, radix64_t1_buf_dit_tile256_temporal_bwd_avx512, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx512, fwd, "flat",
                    radix64_t1_dit_fwd_avx512, radix64_t1_buf_dit_tile256_stream_fwd_avx512, 256),
    ADD_CASE_MIN_ME("buf_tile256_stream",   avx512, bwd, "flat",
                    radix64_t1_dit_bwd_avx512, radix64_t1_buf_dit_tile256_stream_bwd_avx512, 256),
    ADD_CASE("t1_buf_dit", avx512, fwd, "flat",
             radix64_t1_dit_fwd_avx512, vfft_r64_t1_buf_dit_dispatch_fwd_avx512),
    ADD_CASE("t1_buf_dit", avx512, bwd, "flat",
             radix64_t1_dit_bwd_avx512, vfft_r64_t1_buf_dit_dispatch_bwd_avx512),
  #endif

#elif RADIX == 5
  /* R=5 tuning matrix: 3 twiddle strategies × U=1 × 2 ISAs × 2 directions.
   * Per-protocol reference (same convention as R=3, R=7):
   *   flat  -> radix5_t1_dit
   *   t1s   -> radix5_t1s_dit
   *   log3  -> radix5_t1_dit_log3 */
  #if defined(VALIDATE_AVX2)
    ADD_CASE("t1_dit",      avx2, fwd, "flat",
             radix5_t1_dit_fwd_avx2,       vfft_r5_t1_dit_dispatch_fwd_avx2),
    ADD_CASE("t1_dit",      avx2, bwd, "flat",
             radix5_t1_dit_bwd_avx2,       vfft_r5_t1_dit_dispatch_bwd_avx2),
    ADD_CASE("t1s_dit",     avx2, fwd, "t1s",
             radix5_t1s_dit_fwd_avx2,      vfft_r5_t1s_dit_dispatch_fwd_avx2),
    ADD_CASE("t1s_dit",     avx2, bwd, "t1s",
             radix5_t1s_dit_bwd_avx2,      vfft_r5_t1s_dit_dispatch_bwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, fwd, "log3",
             radix5_t1_dit_log3_fwd_avx2,  vfft_r5_t1_dit_log3_dispatch_fwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, bwd, "log3",
             radix5_t1_dit_log3_bwd_avx2,  vfft_r5_t1_dit_log3_dispatch_bwd_avx2),
  #endif
  #if defined(VALIDATE_AVX512)
    ADD_CASE("t1_dit",      avx512, fwd, "flat",
             radix5_t1_dit_fwd_avx512,       vfft_r5_t1_dit_dispatch_fwd_avx512),
    ADD_CASE("t1_dit",      avx512, bwd, "flat",
             radix5_t1_dit_bwd_avx512,       vfft_r5_t1_dit_dispatch_bwd_avx512),
    ADD_CASE("t1s_dit",     avx512, fwd, "t1s",
             radix5_t1s_dit_fwd_avx512,      vfft_r5_t1s_dit_dispatch_fwd_avx512),
    ADD_CASE("t1s_dit",     avx512, bwd, "t1s",
             radix5_t1s_dit_bwd_avx512,      vfft_r5_t1s_dit_dispatch_bwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, fwd, "log3",
             radix5_t1_dit_log3_fwd_avx512,  vfft_r5_t1_dit_log3_dispatch_fwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, bwd, "log3",
             radix5_t1_dit_log3_bwd_avx512,  vfft_r5_t1_dit_log3_dispatch_bwd_avx512),
  #endif

#elif RADIX == 7
  /* R=7 tuning matrix: 3 twiddle strategies × U=1 × 2 ISAs × 2 directions.
   * Per-protocol reference (same convention as R=3, R=16, etc.):
   *   flat  -> radix7_t1_dit
   *   t1s   -> radix7_t1s_dit
   *   log3  -> radix7_t1_dit_log3
   * Each dispatcher self-validates against its protocol's reference. */
  #if defined(VALIDATE_AVX2)
    ADD_CASE("t1_dit",      avx2, fwd, "flat",
             radix7_t1_dit_fwd_avx2,       vfft_r7_t1_dit_dispatch_fwd_avx2),
    ADD_CASE("t1_dit",      avx2, bwd, "flat",
             radix7_t1_dit_bwd_avx2,       vfft_r7_t1_dit_dispatch_bwd_avx2),
    ADD_CASE("t1s_dit",     avx2, fwd, "t1s",
             radix7_t1s_dit_fwd_avx2,      vfft_r7_t1s_dit_dispatch_fwd_avx2),
    ADD_CASE("t1s_dit",     avx2, bwd, "t1s",
             radix7_t1s_dit_bwd_avx2,      vfft_r7_t1s_dit_dispatch_bwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, fwd, "log3",
             radix7_t1_dit_log3_fwd_avx2,  vfft_r7_t1_dit_log3_dispatch_fwd_avx2),
    ADD_CASE("t1_dit_log3", avx2, bwd, "log3",
             radix7_t1_dit_log3_bwd_avx2,  vfft_r7_t1_dit_log3_dispatch_bwd_avx2),
  #endif
  #if defined(VALIDATE_AVX512)
    ADD_CASE("t1_dit",      avx512, fwd, "flat",
             radix7_t1_dit_fwd_avx512,       vfft_r7_t1_dit_dispatch_fwd_avx512),
    ADD_CASE("t1_dit",      avx512, bwd, "flat",
             radix7_t1_dit_bwd_avx512,       vfft_r7_t1_dit_dispatch_bwd_avx512),
    ADD_CASE("t1s_dit",     avx512, fwd, "t1s",
             radix7_t1s_dit_fwd_avx512,      vfft_r7_t1s_dit_dispatch_fwd_avx512),
    ADD_CASE("t1s_dit",     avx512, bwd, "t1s",
             radix7_t1s_dit_bwd_avx512,      vfft_r7_t1s_dit_dispatch_bwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, fwd, "log3",
             radix7_t1_dit_log3_fwd_avx512,  vfft_r7_t1_dit_log3_dispatch_fwd_avx512),
    ADD_CASE("t1_dit_log3", avx512, bwd, "log3",
             radix7_t1_dit_log3_bwd_avx512,  vfft_r7_t1_dit_log3_dispatch_bwd_avx512),
  #endif

#else
  #error "validate.c: unsupported RADIX (extend CASES[])"
#endif
};
static const size_t N_CASES = sizeof(CASES) / sizeof(CASES[0]);

static const size_t ME_GRID[] = {64, 128, 256, 512, 1024, 2048};
static const size_t ME_N = sizeof(ME_GRID) / sizeof(ME_GRID[0]);

static const size_t ME_GRID_T1S[] = {64, 128};
static const size_t ME_N_T1S = sizeof(ME_GRID_T1S) / sizeof(ME_GRID_T1S[0]);

/* R=3 uses multiples of 24 so U=3 variants exercise the main unrolled
 * loop (step = 3*VL = 24) rather than falling entirely through to the
 * U=1 tail. These me values match the bench sweep in candidates.py. */
static const size_t ME_GRID_R3[] = {24, 48, 96, 192, 384, 768, 1536, 3072};
static const size_t ME_N_R3 = sizeof(ME_GRID_R3) / sizeof(ME_GRID_R3[0]);

/* R=5 uses multiples of lcm(5, 8) = 40 so both AVX2 k_step=4 and
 * AVX-512 k_step=8 divide me cleanly; R=5 codelet has no loop tail. */
static const size_t ME_GRID_R5[] = {40, 80, 160, 320, 640, 1280};
static const size_t ME_N_R5 = sizeof(ME_GRID_R5) / sizeof(ME_GRID_R5[0]);

/* R=7 uses multiples of 56 = lcm(7, 8) so both AVX2 k_step=4 and
 * AVX-512 k_step=8 divide me cleanly; R=7 codelet has no loop tail. */
static const size_t ME_GRID_R7[] = {56, 112, 224, 448, 896, 1792};
static const size_t ME_N_R7 = sizeof(ME_GRID_R7) / sizeof(ME_GRID_R7[0]);

int main(void) {
  int R = RADIX;
  const double TOL = 1e-10;

  int total = 0, failed = 0;

  /* Per-radix grid override for prime radixes. */
  const size_t *default_grid;
  size_t default_grid_n;
  if (R == 3) {
    default_grid = ME_GRID_R3;
    default_grid_n = ME_N_R3;
  } else if (R == 5) {
    default_grid = ME_GRID_R5;
    default_grid_n = ME_N_R5;
  } else if (R == 7) {
    default_grid = ME_GRID_R7;
    default_grid_n = ME_N_R7;
  } else {
    default_grid = ME_GRID;
    default_grid_n = ME_N;
  }

  for (size_t c = 0; c < N_CASES; c++) {
    const validate_case_t *cc = &CASES[c];
    /* Small-me t1s restriction only applies at power-of-two radixes;
     * prime radixes (R=3, R=5, R=7) use their own grid for all protocols. */
    int is_t1s_small_only = (R != 3 && R != 5 && R != 7 &&
                             strcmp(cc->protocol, "t1s") == 0);
    const size_t *grid = is_t1s_small_only ? ME_GRID_T1S : default_grid;
    size_t grid_n     = is_t1s_small_only ? ME_N_T1S    : default_grid_n;
    for (size_t i = 0; i < grid_n; i++) {
      size_t me = grid[i];
      if (cc->min_me && me < cc->min_me) continue;
      size_t ios_list[] = {me, me + 8, 8 * me};
      for (size_t j = 0; j < 3; j++) {
        total++;
        if (!run_case(cc, R, me, ios_list[j], TOL)) failed++;
      }
    }
  }

  printf("[validate] %d cases, %d failed\n", total, failed);
  return failed == 0 ? 0 : 1;
}
