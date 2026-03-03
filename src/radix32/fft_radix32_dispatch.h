/**
 * @file fft_radix32_dispatch.h
 * @brief Runtime ISA dispatch for radix-32 FFT stage
 *
 * Provides two top-level entry points:
 *   radix32_forward()   — forward FFT radix-32 stage
 *   radix32_backward()  — backward (inverse) FFT radix-32 stage
 *
 * Dispatch hierarchy (first viable path wins):
 *   1. AVX-512  — K % 8 == 0, mode == BLOCKED8, CPU has AVX-512F+DQ+VL
 *   2. AVX2     — K % 4 == 0, K >= 8, CPU has AVX2+FMA
 *   3. Scalar   — any K ≥ 4 (ultimate fallback, supports all twiddle modes)
 *
 * SSE2 placeholder: not yet implemented, scalar covers this role for now.
 *
 * The dispatch decision is made once at first call (lazy init via
 * detect_isa_level), then cached. Subsequent calls go through a
 * direct branch on the cached level — no function pointer indirection,
 * so the compiler can still see the call targets for LTO/PGO.
 *
 * USAGE:
 *   // Allocate temp buffer (needed for AVX2/AVX-512 paths)
 *   double *temp_re = aligned_alloc(64, 32 * K * sizeof(double));
 *   double *temp_im = aligned_alloc(64, 32 * K * sizeof(double));
 *
 *   radix32_forward(K, in_re, in_im, out_re, out_im,
 *                   &pass1_tw, &pass2_tw, NULL, temp_re, temp_im);
 *
 *   // rec_tw can be non-NULL for scalar RECURRENCE mode
 *   // temp_re/temp_im can be NULL if you know only scalar path will run
 *
 * @author Tugbars
 * @date 2025
 */

#ifndef FFT_RADIX32_DISPATCH_H
#define FFT_RADIX32_DISPATCH_H

#include <stddef.h>
#include <stdint.h>
#include <assert.h>

/*==========================================================================
 * BACKEND HEADERS
 *
 * Include order matters: scalar includes AVX2 (for shared types),
 * AVX-512 driver includes core + AVX2 for type definitions.
 *=========================================================================*/
#include "fft_radix32_scalar.h"   /* scalar backend + shared type defs */
#include "fft_radix32_avx512.h"   /* AVX-512 backend (includes avx2 + core) */

/*==========================================================================
 * ISA LEVEL DETECTION
 *
 * Uses CPUID + XGETBV to detect:
 *   - AVX2 + FMA  (CPUID.7.0:EBX bit 5 + CPUID.1:ECX bit 12, OS XSAVE)
 *   - AVX-512F + AVX-512DQ + AVX-512VL  (CPUID.7.0:EBX bits 16,17,31)
 *   - OS support for ZMM state save  (XCR0 bits 5,6,7)
 *
 * Cached on first call — no overhead on subsequent queries.
 *=========================================================================*/

typedef enum
{
    ISA_SCALAR = 0,   /**< Scalar (no SIMD, or SSE2-only — future) */
    ISA_AVX2   = 1,   /**< AVX2 + FMA                              */
    ISA_AVX512 = 2    /**< AVX-512F + DQ + VL                      */
} radix32_isa_level_t;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _MSC_VER
#include <intrin.h>
static inline void cpuid_query(int leaf, int subleaf, uint32_t *eax, uint32_t *ebx,
                                uint32_t *ecx, uint32_t *edx)
{
    int regs[4];
    __cpuidex(regs, leaf, subleaf);
    *eax = (uint32_t)regs[0]; *ebx = (uint32_t)regs[1];
    *ecx = (uint32_t)regs[2]; *edx = (uint32_t)regs[3];
}
static inline uint64_t xgetbv_query(uint32_t idx)
{
    return _xgetbv(idx);
}
#else /* GCC / Clang */
#include <cpuid.h>
static inline void cpuid_query(int leaf, int subleaf, uint32_t *eax, uint32_t *ebx,
                                uint32_t *ecx, uint32_t *edx)
{
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
}
static inline uint64_t xgetbv_query(uint32_t idx)
{
    uint32_t lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(idx));
    return ((uint64_t)hi << 32) | lo;
}
#endif

/**
 * @brief Detect the highest supported ISA level for radix-32 FFT
 *
 * Performs CPUID + XGETBV checks once, caches the result.
 * Thread-safe on x86 (worst case: redundant detection, same result).
 */
static radix32_isa_level_t detect_isa_level(void)
{
    static radix32_isa_level_t cached = (radix32_isa_level_t)-1;

    if (cached != (radix32_isa_level_t)-1)
        return cached;

    uint32_t eax, ebx, ecx, edx;

    /*--------------------------------------------------------------
     * Step 1: Check CPUID leaf 1 for OSXSAVE + FMA + AVX
     *------------------------------------------------------------*/
    cpuid_query(1, 0, &eax, &ebx, &ecx, &edx);

    const int has_osxsave = (ecx >> 27) & 1;  /* CPUID.1:ECX.OSXSAVE */
    const int has_fma     = (ecx >> 12) & 1;  /* CPUID.1:ECX.FMA     */
    const int has_avx     = (ecx >> 28) & 1;  /* CPUID.1:ECX.AVX     */

    if (!has_osxsave || !has_avx || !has_fma) {
        cached = ISA_SCALAR;
        return cached;
    }

    /*--------------------------------------------------------------
     * Step 2: Check OS saves YMM state (XCR0 bits 1,2)
     *------------------------------------------------------------*/
    uint64_t xcr0 = xgetbv_query(0);
    const int os_saves_ymm = (xcr0 & 0x06) == 0x06;  /* bits 1+2: SSE+AVX */

    if (!os_saves_ymm) {
        cached = ISA_SCALAR;
        return cached;
    }

    /*--------------------------------------------------------------
     * Step 3: Check CPUID leaf 7 for AVX2
     *------------------------------------------------------------*/
    cpuid_query(7, 0, &eax, &ebx, &ecx, &edx);

    const int has_avx2 = (ebx >> 5) & 1;  /* CPUID.7.0:EBX.AVX2 */

    if (!has_avx2) {
        cached = ISA_SCALAR;
        return cached;
    }

    /* At this point: AVX2 + FMA confirmed */

    /*--------------------------------------------------------------
     * Step 4: Check AVX-512 feature flags
     *------------------------------------------------------------*/
    const int has_avx512f  = (ebx >> 16) & 1;  /* CPUID.7.0:EBX.AVX512F  */
    const int has_avx512dq = (ebx >> 17) & 1;  /* CPUID.7.0:EBX.AVX512DQ */
    const int has_avx512vl = (ebx >> 31) & 1;  /* CPUID.7.0:EBX.AVX512VL */

    /*--------------------------------------------------------------
     * Step 5: Check OS saves ZMM state (XCR0 bits 5,6,7)
     *------------------------------------------------------------*/
    const int os_saves_zmm = (xcr0 & 0xE0) == 0xE0;  /* bits 5+6+7 */

    if (has_avx512f && has_avx512dq && has_avx512vl && os_saves_zmm) {
        cached = ISA_AVX512;
        return cached;
    }

    cached = ISA_AVX2;
    return cached;
}

#else /* Non-x86 */

static radix32_isa_level_t detect_isa_level(void)
{
    return ISA_SCALAR;
}

#endif /* x86 detection */

/**
 * @brief Query the ISA level that the dispatch layer will use
 *
 * Useful for diagnostics, logging, or benchmark selection.
 */
static inline radix32_isa_level_t radix32_get_isa_level(void)
{
    return detect_isa_level();
}

/**
 * @brief Human-readable ISA level name
 */
static inline const char *radix32_isa_name(radix32_isa_level_t level)
{
    switch (level) {
        case ISA_AVX512: return "AVX-512";
        case ISA_AVX2:   return "AVX2+FMA";
        case ISA_SCALAR: return "Scalar";
        default:         return "Unknown";
    }
}

/*==========================================================================
 * DISPATCH LOGIC
 *
 * For each call, the dispatch selects the EFFECTIVE ISA level:
 *
 *   effective = min(hardware_isa, constraints_from_K_and_twiddle_mode)
 *
 * Constraint downgrades:
 *   AVX-512 → AVX2:  K % 8 != 0, or mode is RECURRENCE (AVX-512 supports
 *                     BLOCKED8 and BLOCKED4)
 *   AVX2 → Scalar:   K % 4 != 0, or K < 8
 *
 * This means the caller doesn't need to worry about alignment — the
 * dispatch handles it automatically. But for best performance, ensure
 * K is a multiple of 8 and use BLOCKED8 (K ≤ 256) or BLOCKED4 (K ≤ 4096).
 *=========================================================================*/

/**
 * @brief Compute the effective ISA level given K and twiddle mode
 */
static inline radix32_isa_level_t effective_isa(
    size_t K,
    tw_mode_t mode)
{
    radix32_isa_level_t hw = detect_isa_level();

    /* AVX-512 constraints: K%8==0, BLOCKED8 or BLOCKED4 */
    if (hw >= ISA_AVX512) {
        if ((K & 7) == 0 && K >= 16 &&
            (mode == TW_MODE_BLOCKED8 || mode == TW_MODE_BLOCKED4))
            return ISA_AVX512;
        /* Fall through to AVX2 check */
        hw = ISA_AVX2;
    }

    /* AVX2 constraints: K%4==0, K>=8 */
    if (hw >= ISA_AVX2) {
        if ((K & 3) == 0 && K >= 8)
            return ISA_AVX2;
        /* Fall through to scalar */
    }

    return ISA_SCALAR;
}

/*==========================================================================
 * FORWARD DISPATCH
 *
 * @param K         Samples per stripe
 * @param in_re     Input real [32 stripes][K]
 * @param in_im     Input imag [32 stripes][K]
 * @param out_re    Output real [32 stripes][K]
 * @param out_im    Output imag [32 stripes][K]
 * @param pass1_tw  Radix-4 DIT twiddles (BLOCKED2)
 * @param pass2_tw  Radix-8 DIF twiddles (multi-mode)
 * @param rec_tw    Scalar recurrence twiddles (NULL unless RECURRENCE)
 * @param temp_re   Temp buffer [32 stripes][K] (NULL ok if scalar-only)
 * @param temp_im   Temp buffer [32 stripes][K] (NULL ok if scalar-only)
 *
 * @return The ISA level actually used (for diagnostics)
 *=========================================================================*/

static radix32_isa_level_t radix32_forward(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    const radix32_isa_level_t isa = effective_isa(K, pass2_tw->mode);

    switch (isa) {

    case ISA_AVX512:
        assert(temp_re != NULL && temp_im != NULL &&
               "AVX-512 path requires temp buffers");
        radix32_stage_forward_avx512_multi(
            K, in_re, in_im, out_re, out_im,
            pass1_tw, pass2_tw, temp_re, temp_im);
        break;

    case ISA_AVX2:
        assert(temp_re != NULL && temp_im != NULL &&
               "AVX2 path requires temp buffers");
        radix32_stage_forward_avx2(
            K, in_re, in_im, out_re, out_im,
            pass1_tw, pass2_tw, temp_re, temp_im);
        break;

    case ISA_SCALAR:
    default:
        radix32_stage_forward_scalar(
            K, in_re, in_im, out_re, out_im,
            pass1_tw, pass2_tw, rec_tw);
        break;
    }

    return isa;
}

/*==========================================================================
 * BACKWARD DISPATCH
 *=========================================================================*/

static radix32_isa_level_t radix32_backward(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    const radix32_isa_level_t isa = effective_isa(K, pass2_tw->mode);

    switch (isa) {

    case ISA_AVX512:
        assert(temp_re != NULL && temp_im != NULL);
        radix32_stage_backward_avx512_multi(
            K, in_re, in_im, out_re, out_im,
            pass1_tw, pass2_tw, temp_re, temp_im);
        break;

    case ISA_AVX2:
        assert(temp_re != NULL && temp_im != NULL);
        radix32_stage_backward_avx2(
            K, in_re, in_im, out_re, out_im,
            pass1_tw, pass2_tw, temp_re, temp_im);
        break;

    case ISA_SCALAR:
    default:
        radix32_stage_backward_scalar(
            K, in_re, in_im, out_re, out_im,
            pass1_tw, pass2_tw, rec_tw);
        break;
    }

    return isa;
}

/*==========================================================================
 * FORCED-ISA VARIANTS
 *
 * For benchmarking: bypass auto-detection and force a specific backend.
 * Caller is responsible for meeting the ISA's constraints on K/mode.
 *=========================================================================*/

static inline void radix32_forward_avx512(
    size_t K,
    const double *RESTRICT in_re,  const double *RESTRICT in_im,
    double *RESTRICT out_re,        double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,       double *RESTRICT temp_im)
{
    assert((K & 7) == 0 && K >= 16 && pass2_tw->mode == TW_MODE_BLOCKED8);
    radix32_stage_forward_avx512(K, in_re, in_im, out_re, out_im,
                                  pass1_tw, &pass2_tw->b8, temp_re, temp_im);
}

static inline void radix32_backward_avx512(
    size_t K,
    const double *RESTRICT in_re,  const double *RESTRICT in_im,
    double *RESTRICT out_re,        double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,       double *RESTRICT temp_im)
{
    assert((K & 7) == 0 && K >= 16 && pass2_tw->mode == TW_MODE_BLOCKED8);
    radix32_stage_backward_avx512(K, in_re, in_im, out_re, out_im,
                                   pass1_tw, &pass2_tw->b8, temp_re, temp_im);
}

static inline void radix32_forward_avx2(
    size_t K,
    const double *RESTRICT in_re,  const double *RESTRICT in_im,
    double *RESTRICT out_re,        double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,       double *RESTRICT temp_im)
{
    assert((K & 3) == 0 && K >= 8);
    radix32_stage_forward_avx2(K, in_re, in_im, out_re, out_im,
                                pass1_tw, pass2_tw, temp_re, temp_im);
}

static inline void radix32_backward_avx2(
    size_t K,
    const double *RESTRICT in_re,  const double *RESTRICT in_im,
    double *RESTRICT out_re,        double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,       double *RESTRICT temp_im)
{
    assert((K & 3) == 0 && K >= 8);
    radix32_stage_backward_avx2(K, in_re, in_im, out_re, out_im,
                                 pass1_tw, pass2_tw, temp_re, temp_im);
}

static inline void radix32_forward_scalar(
    size_t K,
    const double *RESTRICT in_re,  const double *RESTRICT in_im,
    double *RESTRICT out_re,        double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw)
{
    radix32_stage_forward_scalar(K, in_re, in_im, out_re, out_im,
                                  pass1_tw, pass2_tw, rec_tw);
}

static inline void radix32_backward_scalar(
    size_t K,
    const double *RESTRICT in_re,  const double *RESTRICT in_im,
    double *RESTRICT out_re,        double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw)
{
    radix32_stage_backward_scalar(K, in_re, in_im, out_re, out_im,
                                   pass1_tw, pass2_tw, rec_tw);
}

#endif /* FFT_RADIX32_DISPATCH_H */
