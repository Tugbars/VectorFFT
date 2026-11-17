/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* cpu_check.h -- THE CPU CHECK (owner 2026-11-14): the library's entry points refuse to run on a CPU that lacks an
 * instruction set this build was compiled for, where they would otherwise stop the program with an illegal
 * instruction.
 *
 * WHAT IT CHECKS: every instruction-set extension the compiler was told to use for this translation unit, read
 * from the compiler's own predefined macros (__AVX2__, __FMA__, __AVX512F__, __BMI2__, ...; -march=native defines
 * those of the building CPU), against the CPU's CPUID bits and, for the vector registers, the operating system's
 * XCR0 (the OS must save the YMM / ZMM state). So an AVX-512 build on an AVX2 CPU, an AVX2 build on an older CPU,
 * and a -march=native build moved to a CPU without one of the building CPU's extensions are all named.
 *
 * WHERE IT RUNS: vfft_create, vfft_info and vfft_malloc (vfft_api.h, vfft_memory.h) call vfft__cpu_lacks() first;
 * those entries and everything here are compiled for the baseline x86-64 instruction set (VFFT_BASELINE_FN), so
 * the check runs before any instruction it guards. Nothing here calls a libc function that a fortified header
 * turns into an always-inline wrapper (snprintf, fprintf, memcpy, strcpy: glibc's _FORTIFY_SOURCE cannot inline
 * them into a function of another target): the text is copied by hand, CPUID is read with the instruction, and
 * stderr gets fputs. The answer is computed once per process. GCC and Clang only (the library's compilers);
 * elsewhere it reports nothing missing. */
#ifndef VFFT_CPU_CHECK_H
#define VFFT_CPU_CHECK_H

#include <stddef.h>
#include <stdio.h>

#if defined(__GNUC__) && defined(__x86_64__)
#define VFFT_BASELINE_FN __attribute__((target("arch=x86-64"), noinline, unused))

/* s appended to dst[cap] at *len (always terminated) */
VFFT_BASELINE_FN static void vfft__cat(char *dst, size_t cap, size_t *len, const char *s)
{
    while (*s && *len + 1 < cap)
        dst[(*len)++] = *s++;
    if (cap)
        dst[*len < cap ? *len : cap - 1] = 0;
}

/* prefix and why into dst[cap]: the entry points' refusal for vfft_error() */
VFFT_BASELINE_FN static void vfft__errmsg(char *dst, size_t cap, const char *prefix, const char *why)
{
    size_t len = 0;
    vfft__cat(dst, cap, &len, prefix);
    vfft__cat(dst, cap, &len, why);
}

VFFT_BASELINE_FN static void vfft__cpuid(unsigned leaf, unsigned sub, unsigned r[4])
{
    __asm__ volatile("cpuid" : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3]) : "a"(leaf), "c"(sub));
}

/* the names of what this build needs and this CPU lacks, space-separated, into miss; their count */
VFFT_BASELINE_FN static int vfft__cpu_lacks_compute(char *miss, size_t cap)
{
    unsigned r[4], max = 0, maxe = 0, c1 = 0, b7 = 0, c7 = 0, a71 = 0, ce1 = 0, xlo = 0, xhi = 0;
    size_t len = 0;
    int n = 0;
    miss[0] = 0;
    vfft__cpuid(0, 0, r);
    max = r[0];
    if (max >= 1) { vfft__cpuid(1, 0, r); c1 = r[2]; }
    if (max >= 7) { vfft__cpuid(7, 0, r); b7 = r[1]; c7 = r[2]; vfft__cpuid(7, 1, r); a71 = r[0]; }
    vfft__cpuid(0x80000000u, 0, r);
    maxe = r[0];
    if (maxe >= 0x80000001u) { vfft__cpuid(0x80000001u, 0, r); ce1 = r[2]; }
    if (c1 & (1u << 27))   /* OSXSAVE: XCR0 says which register state the OS saves */
        __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    (void)xhi; (void)a71; (void)c7; (void)ce1; (void)b7;
#define VFFT__NEED(cond, name)                      \
    do                                              \
    {                                               \
        if (!(cond))                                \
        {                                           \
            if (n) vfft__cat(miss, cap, &len, " "); \
            vfft__cat(miss, cap, &len, name);       \
            n++;                                    \
        }                                           \
    } while (0)
#ifdef __SSE4_2__
    VFFT__NEED(c1 & (1u << 20), "SSE4.2");
#endif
#ifdef __POPCNT__
    VFFT__NEED(c1 & (1u << 23), "POPCNT");
#endif
#ifdef __MOVBE__
    VFFT__NEED(c1 & (1u << 22), "MOVBE");
#endif
#ifdef __F16C__
    VFFT__NEED(c1 & (1u << 29), "F16C");
#endif
#ifdef __AVX__
    VFFT__NEED((c1 & (1u << 28)) && (xlo & 0x6u) == 0x6u, "AVX");
#endif
#ifdef __FMA__
    VFFT__NEED(c1 & (1u << 12), "FMA");
#endif
#ifdef __AVX2__
    VFFT__NEED(b7 & (1u << 5), "AVX2");
#endif
#ifdef __BMI__
    VFFT__NEED(b7 & (1u << 3), "BMI1");
#endif
#ifdef __BMI2__
    VFFT__NEED(b7 & (1u << 8), "BMI2");
#endif
#ifdef __LZCNT__
    VFFT__NEED(ce1 & (1u << 5), "LZCNT");
#endif
#ifdef __AVXVNNI__
    VFFT__NEED(a71 & (1u << 4), "AVX-VNNI");
#endif
#ifdef __GFNI__
    VFFT__NEED(c7 & (1u << 8), "GFNI");
#endif
#ifdef __VAES__
    VFFT__NEED(c7 & (1u << 9), "VAES");
#endif
#ifdef __VPCLMULQDQ__
    VFFT__NEED(c7 & (1u << 10), "VPCLMULQDQ");
#endif
#ifdef __AVX512F__
    VFFT__NEED((b7 & (1u << 16)) && (xlo & 0xE6u) == 0xE6u, "AVX-512F");
#endif
#ifdef __AVX512DQ__
    VFFT__NEED(b7 & (1u << 17), "AVX-512DQ");
#endif
#ifdef __AVX512CD__
    VFFT__NEED(b7 & (1u << 28), "AVX-512CD");
#endif
#ifdef __AVX512BW__
    VFFT__NEED(b7 & (1u << 30), "AVX-512BW");
#endif
#ifdef __AVX512VL__
    VFFT__NEED(b7 & (1u << 31), "AVX-512VL");
#endif
#ifdef __AVX512IFMA__
    VFFT__NEED(b7 & (1u << 21), "AVX-512IFMA");
#endif
#ifdef __AVX512VBMI__
    VFFT__NEED(c7 & (1u << 1), "AVX-512VBMI");
#endif
#ifdef __AVX512VBMI2__
    VFFT__NEED(c7 & (1u << 6), "AVX-512VBMI2");
#endif
#ifdef __AVX512VNNI__
    VFFT__NEED(c7 & (1u << 11), "AVX-512VNNI");
#endif
#ifdef __AVX512BITALG__
    VFFT__NEED(c7 & (1u << 12), "AVX-512BITALG");
#endif
#ifdef __AVX512VPOPCNTDQ__
    VFFT__NEED(c7 & (1u << 14), "AVX-512VPOPCNTDQ");
#endif
#undef VFFT__NEED
    return n;
}

/* 1 = this CPU lacks what the build needs; why (for vfft_error) names it. Said once on standard error as well:
 * a program that does not test the NULL plan still learns why. */
VFFT_BASELINE_FN static int vfft__cpu_lacks(const char **why)
{
    static int state = -1;   /* -1 not asked yet, 0 fine, 1 lacking */
    static char msg[512];
    if (state < 0)
    {
        char miss[320];
        size_t len = 0;
        state = vfft__cpu_lacks_compute(miss, sizeof miss) > 0;
        msg[0] = 0;
        if (state)
        {
            char line[600];
            size_t ll = 0;
            vfft__cat(msg, sizeof msg, &len, "this CPU lacks instructions this build of VectorFFT was compiled for: ");
            vfft__cat(msg, sizeof msg, &len, miss);
            vfft__cat(msg, sizeof msg, &len, ". Build the library on this machine, or for any CPU with AVX2 and FMA "
                                             "(configure with -DVFFT_NATIVE=OFF -DVFFT_ISA=avx2)");
            vfft__cat(line, sizeof line, &ll, "vfft: ");
            vfft__cat(line, sizeof line, &ll, msg);
            vfft__cat(line, sizeof line, &ll, "\n");
            fputs(line, stderr);
        }
    }
    if (why) *why = msg;
    return state;
}
#else
#define VFFT_BASELINE_FN
static inline void vfft__errmsg(char *dst, size_t cap, const char *prefix, const char *why)
{
    snprintf(dst, cap, "%s%s", prefix, why);
}
static inline int vfft__cpu_lacks(const char **why)
{
    if (why) *why = "";
    return 0;
}
#endif

#endif /* VFFT_CPU_CHECK_H */
