/**
 * @file fft_radix32_uniform.h
 * @brief Shared infrastructure for radix-32 FFT ISA dispatch
 *
 * Provides:
 *   - Backend includes (scalar, AVX2, AVX-512)
 *   - ISA level enum and CPUID/XGETBV detection
 *   - effective_isa() — constraint-aware ISA selection
 *   - radix32_isa_name() — human-readable ISA string
 *
 * Included by fft_radix32_fv.c (forward) and fft_radix32_bv.c (backward).
 *
 * Dispatch hierarchy (first viable path wins):
 *   1. AVX-512  — K % 8 == 0, K >= 16, mode == BLOCKED8, HW has AVX-512F+DQ+VL
 *   2. AVX2     — K % 4 == 0, K >= 8, HW has AVX2+FMA
 *   3. Scalar   — any K >= 4 (ultimate fallback, all twiddle modes)
 *
 * SSE2 placeholder: not yet implemented, scalar covers this role for now.
 *
 * @author Tugbars
 * @date 2025
 */

#ifndef FFT_RADIX32_UNIFORM_H
#define FFT_RADIX32_UNIFORM_H

#include <stddef.h>
#include <stdint.h>
#include <assert.h>

/*==========================================================================
 * BACKEND HEADERS
 *
 * Include order: scalar (pulls in AVX2 for shared types) → AVX-512.
 *=========================================================================*/
#include "fft_radix32_scalar.h"
#include "fft_radix32_avx512.h"

/*==========================================================================
 * ISA LEVEL
 *=========================================================================*/

typedef enum
{
    ISA_SCALAR = 0,   /**< Scalar (no SIMD, or SSE2-only — future)  */
    ISA_AVX2   = 1,   /**< AVX2 + FMA                               */
    ISA_AVX512 = 2    /**< AVX-512F + DQ + VL                       */
} radix32_isa_level_t;

/*==========================================================================
 * CPUID + XGETBV DETECTION (x86 only)
 *=========================================================================*/

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _MSC_VER
#include <intrin.h>
static inline void cpuid_query(int leaf, int subleaf,
                                uint32_t *eax, uint32_t *ebx,
                                uint32_t *ecx, uint32_t *edx)
{
    int regs[4];
    __cpuidex(regs, leaf, subleaf);
    *eax = (uint32_t)regs[0]; *ebx = (uint32_t)regs[1];
    *ecx = (uint32_t)regs[2]; *edx = (uint32_t)regs[3];
}
static inline uint64_t xgetbv_query(uint32_t idx) { return _xgetbv(idx); }
#else /* GCC / Clang / ICX */
#include <cpuid.h>
static inline void cpuid_query(int leaf, int subleaf,
                                uint32_t *eax, uint32_t *ebx,
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
 * @brief Detect highest ISA level. Cached on first call.
 *
 * Checks:
 *   CPUID.1:ECX  → OSXSAVE, FMA, AVX
 *   XCR0         → OS saves YMM (bits 1,2) and ZMM (bits 5,6,7)
 *   CPUID.7.0:EBX → AVX2, AVX512F, AVX512DQ, AVX512VL
 *
 * Thread-safe: worst case is redundant detection yielding same result.
 */
static radix32_isa_level_t detect_isa_level(void)
{
    static radix32_isa_level_t cached = (radix32_isa_level_t)-1;
    if (cached != (radix32_isa_level_t)-1)
        return cached;

    uint32_t eax, ebx, ecx, edx;

    /* Leaf 1: OSXSAVE + FMA + AVX */
    cpuid_query(1, 0, &eax, &ebx, &ecx, &edx);
    if (!((ecx >> 27) & 1) || !((ecx >> 12) & 1) || !((ecx >> 28) & 1))
        return (cached = ISA_SCALAR);

    /* XCR0: OS must save YMM */
    uint64_t xcr0 = xgetbv_query(0);
    if ((xcr0 & 0x06) != 0x06)
        return (cached = ISA_SCALAR);

    /* Leaf 7: AVX2 */
    cpuid_query(7, 0, &eax, &ebx, &ecx, &edx);
    if (!((ebx >> 5) & 1))
        return (cached = ISA_SCALAR);

    /* AVX-512: F + DQ + VL + OS saves ZMM */
    if (((ebx >> 16) & 1) && ((ebx >> 17) & 1) && ((ebx >> 31) & 1) &&
        ((xcr0 & 0xE0) == 0xE0))
        return (cached = ISA_AVX512);

    return (cached = ISA_AVX2);
}

#else /* Non-x86: scalar only */

static radix32_isa_level_t detect_isa_level(void) { return ISA_SCALAR; }

#endif /* x86 detection */

/*==========================================================================
 * PUBLIC QUERIES
 *=========================================================================*/

/** @brief Query cached hardware ISA level */
static inline radix32_isa_level_t radix32_get_isa_level(void)
{
    return detect_isa_level();
}

/** @brief Human-readable ISA name */
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
 * EFFECTIVE ISA SELECTION
 *
 *   effective = min(hardware_isa, constraints(K, twiddle_mode))
 *
 * Downgrades:
 *   AVX-512 → AVX2:  K%8!=0 || K<16 || mode!=BLOCKED8
 *   AVX2 → Scalar:   K%4!=0 || K<8
 *=========================================================================*/

static inline radix32_isa_level_t effective_isa(size_t K, tw_mode_t mode)
{
    radix32_isa_level_t hw = detect_isa_level();

    if (hw >= ISA_AVX512) {
        if ((K & 7) == 0 && K >= 16 && mode == TW_MODE_BLOCKED8)
            return ISA_AVX512;
        hw = ISA_AVX2;
    }

    if (hw >= ISA_AVX2) {
        if ((K & 3) == 0 && K >= 8)
            return ISA_AVX2;
    }

    return ISA_SCALAR;
}

#endif /* FFT_RADIX32_UNIFORM_H */
