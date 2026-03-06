/**
 * @file vfft_register_codelets.h
 * @brief Wire all optimized VectorFFT codelets into the planner registry
 *
 * Include this AFTER vfft_planner.h and all codelet headers.
 * Detects ISA at runtime and registers the best available version.
 *
 * Usage:
 *   #include "vfft_planner.h"
 *   #include "fft_radix11_genfft.h"
 *   #include "fft_radix13_genfft.h"
 *   // ... all codelet headers ...
 *   #include "vfft_register_codelets.h"
 *
 *   vfft_codelet_registry reg;
 *   vfft_register_all(&reg);
 *   vfft_plan *plan = vfft_plan_create(N, &reg);
 */

#ifndef VFFT_REGISTER_CODELETS_H
#define VFFT_REGISTER_CODELETS_H

#include "vfft_planner.h"

/* ═══════════════════════════════════════════════════════════════
 * ISA DETECTION (shared across all radixes)
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static inline int vfft_has_avx512f(void) {
    unsigned int a,b,c,d;
    return __get_cpuid_count(7,0,&a,&b,&c,&d) ? (b>>16)&1 : 0; }
static inline int vfft_has_avx2(void) {
    unsigned int a,b,c,d;
    return __get_cpuid_count(7,0,&a,&b,&c,&d) ? (b>>5)&1 : 0; }
static inline int vfft_has_fma(void) {
    unsigned int a,b,c,d;
    return __get_cpuid(1,&a,&b,&c,&d) ? (c>>12)&1 : 0; }
#elif defined(_MSC_VER)
#include <intrin.h>
static inline int vfft_has_avx512f(void) { int i[4]; __cpuidex(i,7,0); return (i[1]>>16)&1; }
static inline int vfft_has_avx2(void)    { int i[4]; __cpuidex(i,7,0); return (i[1]>>5)&1; }
static inline int vfft_has_fma(void)     { int i[4]; __cpuid(i,1); return (i[2]>>12)&1; }
#else
static inline int vfft_has_avx512f(void) { return 0; }
static inline int vfft_has_avx2(void)    { return 0; }
static inline int vfft_has_fma(void)     { return 0; }
#endif

typedef enum {
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2   = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_t;

static inline vfft_isa_t vfft_detect_isa(void) {
    static vfft_isa_t cached = (vfft_isa_t)-1;
    if (cached != (vfft_isa_t)-1) return cached;
    if (vfft_has_avx512f() && vfft_has_fma()) cached = VFFT_ISA_AVX512;
    else if (vfft_has_avx2() && vfft_has_fma()) cached = VFFT_ISA_AVX2;
    else cached = VFFT_ISA_SCALAR;
    return cached;
}

/* ═══════════════════════════════════════════════════════════════
 * REGISTRATION MACROS
 *
 * For each radix with genfft codelets, register the best ISA:
 *   AVX-512 > AVX2 > scalar
 * ═══════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════
 * DISPATCHING WRAPPERS
 *
 * The planner calls codelets with arbitrary K (stride), which
 * may not be aligned to SIMD width. These wrappers check K
 * and dispatch to the best ISA that can handle it.
 * ═══════════════════════════════════════════════════════════════ */

#define VFFT_DISPATCH_WRAPPER(R, prefix) \
static void vfft_dispatch_r##R##_fwd( \
    const double *ri, const double *ii, double *ro, double *io, size_t K) \
{ \
    vfft_isa_t isa = vfft_detect_isa(); \
    (void)isa; \
    IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) \
        { prefix##_fwd_avx512(ri,ii,ro,io,K); return; }) \
    IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) \
        { prefix##_fwd_avx2(ri,ii,ro,io,K); return; }) \
    prefix##_fwd_scalar(ri,ii,ro,io,K); \
} \
static void vfft_dispatch_r##R##_bwd( \
    const double *ri, const double *ii, double *ro, double *io, size_t K) \
{ \
    vfft_isa_t isa = vfft_detect_isa(); \
    (void)isa; \
    IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) \
        { prefix##_bwd_avx512(ri,ii,ro,io,K); return; }) \
    IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) \
        { prefix##_bwd_avx2(ri,ii,ro,io,K); return; }) \
    prefix##_bwd_scalar(ri,ii,ro,io,K); \
}

/* Conditional ISA macros — disappear if ISA not compiled in */
#ifdef __AVX512F__
#define IF_AVX512(x) x
#else
#define IF_AVX512(x)
#endif
#ifdef __AVX2__
#define IF_AVX2(x) x
#else
#define IF_AVX2(x)
#endif

/* ═══════════════════════════════════════════════════════════════
 * REGISTER ALL OPTIMIZED CODELETS
 *
 * Starts with naive fallbacks, then overrides with optimized
 * versions for every radix that has a genfft codelet.
 * ═══════════════════════════════════════════════════════════════ */

/* ═══ Instantiate dispatch wrappers at file scope ═══ */

#ifdef FFT_RADIX11_GENFFT_H
VFFT_DISPATCH_WRAPPER(11, radix11_genfft)
#endif
#ifdef FFT_RADIX13_GENFFT_H
VFFT_DISPATCH_WRAPPER(13, radix13_genfft)
#endif
#ifdef FFT_RADIX17_GENFFT_H
VFFT_DISPATCH_WRAPPER(17, radix17_genfft)
#endif
#ifdef FFT_RADIX19_GENFFT_H
VFFT_DISPATCH_WRAPPER(19, radix19_genfft)
#endif
#ifdef FFT_RADIX23_GENFFT_H
VFFT_DISPATCH_WRAPPER(23, radix23_genfft)
#endif

static void vfft_register_all(vfft_codelet_registry *reg)
{
    vfft_registry_init_naive(reg);

#ifdef FFT_RADIX11_GENFFT_H
    vfft_registry_set(reg, 11, vfft_dispatch_r11_fwd, vfft_dispatch_r11_bwd);
#endif
#ifdef FFT_RADIX13_GENFFT_H
    vfft_registry_set(reg, 13, vfft_dispatch_r13_fwd, vfft_dispatch_r13_bwd);
#endif
#ifdef FFT_RADIX17_GENFFT_H
    vfft_registry_set(reg, 17, vfft_dispatch_r17_fwd, vfft_dispatch_r17_bwd);
#endif
#ifdef FFT_RADIX19_GENFFT_H
    vfft_registry_set(reg, 19, vfft_dispatch_r19_fwd, vfft_dispatch_r19_bwd);
#endif
#ifdef FFT_RADIX23_GENFFT_H
    vfft_registry_set(reg, 23, vfft_dispatch_r23_fwd, vfft_dispatch_r23_bwd);
#endif
}

/* ═══════════════════════════════════════════════════════════════
 * ISA REPORT (debug)
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_print_registry(const vfft_codelet_registry *reg)
{
    const char *isa_name[] = {"scalar", "AVX2", "AVX-512"};
    printf("  ISA: %s\n", isa_name[vfft_detect_isa()]);
    printf("  Registered codelets:\n");

    size_t radixes[] = {2,3,4,5,6,7,8,9,10,11,13,16,17,19,23,32,64,128};
    const char *optimized[] = {
        NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
        "genfft","genfft",NULL,"genfft","genfft","genfft",
        NULL,NULL,NULL
    };
    for (size_t i = 0; i < sizeof(radixes)/sizeof(radixes[0]); i++) {
        size_t r = radixes[i];
        if (reg->fwd[r]) {
            printf("    R=%-3zu  fwd=%s  bwd=%s  %s\n", r,
                   reg->fwd[r] ? "✓" : "✗",
                   reg->bwd[r] ? "✓" : "✗",
                   optimized[i] ? optimized[i] : "naive");
        }
    }
}

#endif /* VFFT_REGISTER_CODELETS_H */
