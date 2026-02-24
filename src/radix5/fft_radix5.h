/**
 * @file  fft_radix5.h
 * @brief Radix-5 WFTA butterfly — dispatch vtable, types, and visitor declarations
 *
 * Provides:
 *   - Function pointer types for all butterfly variants
 *   - Dispatch vtable populated via runtime CPUID
 *   - Stage visitor declarations (implemented in fft_radix5_fv.c / bv.c)
 *
 * The centralized planner includes this header to:
 *   1. Call fft_radix5_init_vtable() once at plan creation
 *   2. Pass the vtable to fft_radix5_visit_forward/backward per stage
 *
 * Butterfly contract:
 *   All butterfly functions take 5 input legs (re/im), 5 output legs (re/im),
 *   optional twiddles (tw1,tw2 for BLOCKED2), and K (elements per leg).
 *   In-place (input == output) is supported.
 *   Legs need NOT be 64-byte aligned — unaligned SIMD loads are used.
 *   Twiddle arrays MUST be 64-byte aligned (planner responsibility).
 */

#ifndef FFT_RADIX5_H
#define FFT_RADIX5_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================== */
/*  Constants                                                          */
/* ================================================================== */

#define FFT_RADIX5  5

/* ================================================================== */
/*  Butterfly function pointer types                                   */
/* ================================================================== */

/**
 * Twiddled butterfly: applies W1,W2 twiddles (W3,W4 derived internally).
 * Used for stages s > 0 where K = 5^s > 1.
 * BLOCKED2 layout: tw1[k] = W^k, tw2[k] = W^{2k}.
 */
typedef void (*fft_r5_butterfly_tw_fn)(
    const double *restrict x0_re, const double *restrict x0_im,
    const double *restrict x1_re, const double *restrict x1_im,
    const double *restrict x2_re, const double *restrict x2_im,
    const double *restrict x3_re, const double *restrict x3_im,
    const double *restrict x4_re, const double *restrict x4_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    const double *restrict tw1_re, const double *restrict tw1_im,
    const double *restrict tw2_re, const double *restrict tw2_im,
    int K);

/**
 * N1 butterfly: no twiddles (all W = 1).
 * Used for stage 0 where K = 1.
 */
typedef void (*fft_r5_butterfly_n1_fn)(
    const double *restrict x0_re, const double *restrict x0_im,
    const double *restrict x1_re, const double *restrict x1_im,
    const double *restrict x2_re, const double *restrict x2_im,
    const double *restrict x3_re, const double *restrict x3_im,
    const double *restrict x4_re, const double *restrict x4_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    int K);

/* ================================================================== */
/*  Dispatch vtable                                                    */
/* ================================================================== */

typedef enum {
    FFT_R5_ISA_SCALAR = 0,
    FFT_R5_ISA_SSE2   = 1,
    FFT_R5_ISA_AVX2   = 2,
    FFT_R5_ISA_AVX512 = 3
} fft_r5_isa_t;

typedef struct {
    fft_r5_butterfly_tw_fn  fwd_tw;    /* forward twiddled  */
    fft_r5_butterfly_tw_fn  bwd_tw;    /* backward twiddled */
    fft_r5_butterfly_n1_fn  fwd_n1;    /* forward N1        */
    fft_r5_butterfly_n1_fn  bwd_n1;    /* backward N1       */
    fft_r5_isa_t            isa;       /* selected ISA      */
} fft_r5_vtable_t;

/* ================================================================== */
/*  CPUID detection                                                    */
/* ================================================================== */

#if defined(_MSC_VER) || (defined(_WIN32) && (defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)))
#include <intrin.h>

static inline int fft_r5_cpu_has_avx2(void) {
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1;
}
static inline int fft_r5_cpu_has_avx512f(void) {
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] >> 16) & 1;
}

#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>

static inline int fft_r5_cpu_has_avx2(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    return (b >> 5) & 1;
}
static inline int fft_r5_cpu_has_avx512f(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    return (b >> 16) & 1;
}

#else
static inline int fft_r5_cpu_has_avx2(void)   { return 0; }
static inline int fft_r5_cpu_has_avx512f(void) { return 0; }
#endif

/* ================================================================== */
/*  Forward declarations of all ISA-specific butterflies               */
/* ================================================================== */

/* ── Scalar (always available) ──────────────────────────────────── */
void radix5_wfta_fwd_scalar(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*,
    const double*, const double*, const double*, const double*, int);
void radix5_wfta_bwd_scalar(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*,
    const double*, const double*, const double*, const double*, int);
void radix5_wfta_fwd_scalar_N1(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*, int);
void radix5_wfta_bwd_scalar_N1(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*, int);

/* ── AVX2 ───────────────────────────────────────────────────────── */
#ifdef __AVX2__
void radix5_wfta_fwd_avx2(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*,
    const double*, const double*, const double*, const double*, int);
void radix5_wfta_bwd_avx2(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*,
    const double*, const double*, const double*, const double*, int);
void radix5_wfta_fwd_avx2_N1(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*, int);
void radix5_wfta_bwd_avx2_N1(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*, int);
#endif

/* ── AVX-512 ────────────────────────────────────────────────────── */
#ifdef __AVX512F__
void radix5_wfta_fwd_avx512(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*,
    const double*, const double*, const double*, const double*, int);
void radix5_wfta_bwd_avx512(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*,
    const double*, const double*, const double*, const double*, int);
void radix5_wfta_fwd_avx512_N1(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*, int);
void radix5_wfta_bwd_avx512_N1(
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*,
    double*, double*, double*, double*, double*, double*,
    double*, double*, double*, double*, int);
#endif

/* ================================================================== */
/*  Vtable initialization — call once at plan creation                 */
/* ================================================================== */

static inline void fft_radix5_init_vtable(fft_r5_vtable_t *vt) {
    /* Default: scalar */
    vt->fwd_tw = radix5_wfta_fwd_scalar;
    vt->bwd_tw = radix5_wfta_bwd_scalar;
    vt->fwd_n1 = radix5_wfta_fwd_scalar_N1;
    vt->bwd_n1 = radix5_wfta_bwd_scalar_N1;
    vt->isa    = FFT_R5_ISA_SCALAR;

#ifdef __AVX2__
    if (fft_r5_cpu_has_avx2()) {
        vt->fwd_tw = radix5_wfta_fwd_avx2;
        vt->bwd_tw = radix5_wfta_bwd_avx2;
        vt->fwd_n1 = radix5_wfta_fwd_avx2_N1;
        vt->bwd_n1 = radix5_wfta_bwd_avx2_N1;
        vt->isa    = FFT_R5_ISA_AVX2;
    }
#endif

#ifdef __AVX512F__
    if (fft_r5_cpu_has_avx512f()) {
        vt->fwd_tw = radix5_wfta_fwd_avx512;
        vt->bwd_tw = radix5_wfta_bwd_avx512;
        vt->fwd_n1 = radix5_wfta_fwd_avx512_N1;
        vt->bwd_n1 = radix5_wfta_bwd_avx512_N1;
        vt->isa    = FFT_R5_ISA_AVX512;
    }
#endif
}

static inline const char *fft_r5_isa_name(fft_r5_isa_t isa) {
    switch (isa) {
        case FFT_R5_ISA_AVX512: return "AVX-512";
        case FFT_R5_ISA_AVX2:   return "AVX2";
        case FFT_R5_ISA_SSE2:   return "SSE2";
        default:                 return "Scalar";
    }
}

/* ================================================================== */
/*  Stage visitor declarations                                         */
/*                                                                     */
/*  Implemented in fft_radix5_fv.c and fft_radix5_bv.c.               */
/*  The planner calls them once per DIT stage.                         */
/*                                                                     */
/*  Parameters:                                                        */
/*    vt         — dispatch vtable (from fft_radix5_init_vtable)       */
/*    re, im     — full data arrays (N elements, in-place)             */
/*    K          — elements per leg this stage (5^s)                   */
/*    num_groups — independent butterfly groups (N / (5*K))            */
/*    tw1/2      — BLOCKED2 twiddle arrays (K elements each), or NULL  */
/*                 for stage 0 (N1 butterfly)                          */
/*                                                                     */
/*  Forward DIT: stages s = 0, 1, ..., L-1  (digit-reversal BEFORE)   */
/*  Backward DIT: stages s = L-1, ..., 1, 0 (digit-reversal AFTER)    */
/*  Same twiddle tables for both directions.                           */
/* ================================================================== */

void fft_radix5_visit_forward(
    const fft_r5_vtable_t *vt,
    double *re, double *im,
    int K, int num_groups,
    const double *tw1_re, const double *tw1_im,
    const double *tw2_re, const double *tw2_im);

void fft_radix5_visit_backward(
    const fft_r5_vtable_t *vt,
    double *re, double *im,
    int K, int num_groups,
    const double *tw1_re, const double *tw1_im,
    const double *tw2_re, const double *tw2_im);

#ifdef __cplusplus
}
#endif

#endif /* FFT_RADIX5_H */