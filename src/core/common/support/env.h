/* env.h — the runtime environment. Two parts:
 *
 *   PART 1 — CPU / runtime setup: denormal handling (FTZ/DAZ), SIMD-aligned
 *            + huge-page allocation, verbosity/version/ISA query, and thread
 *            affinity / core pinning.
 *
 *   PART 2 — the exhaustive search's tuning knobs — moved to
 *            split/planning/exhaustive_knobs.h (layout separation phase 5):
 *            only the split exhaustive planner reads them.
 */
#ifndef VFFT_COMMON_ENV_H
#define VFFT_COMMON_ENV_H

/* ===========================================================================
 * PART 1 — CPU / RUNTIME ENVIRONMENT
 *
 *   vfft_env_init();  // once per thread (FTZ/DAZ)
 *   double *re = vfft_aligned_alloc(N * K * sizeof(double));
 *   ...
 *   vfft_aligned_free(re);
 * ===========================================================================
 */

/* CPUID (vfft_print_info's brand string) is spelled differently per
 * toolchain, and the spellings collide:
 *
 *   MSVC/ICX  <intrin.h> declares __cpuid / __cpuidex as FUNCTIONS taking an
 *             int[4] output array.
 *   MinGW GCC <cpuid.h> defines __cpuid / __cpuid_count as 5-ARGUMENT MACROS
 *             writing four separate lvalues.
 *
 * From GCC 15.2 <immintrin.h> pulls in <cpuid.h>, so <intrin.h> included after
 * it fails ("macro '__cpuid' requires 5 arguments") depending only on include
 * order; undefining __cpuid does not help. So each toolchain gets its own
 * header and never both: GCC-family builds use <cpuid.h>, MSVC/ICX <intrin.h>.
 * _VFFT_CPUIDEX hides the shape difference. */
#if defined(_WIN32)
  #if defined(__GNUC__) && !defined(__INTEL_LLVM_COMPILER) && !defined(_MSC_VER)
    #include <cpuid.h>
    /* __cpuid_count is <cpuid.h>'s __cpuidex: same leaf/subleaf, but the four
     * output registers are separate lvalues instead of an int[4]. */
    #define _VFFT_CPUIDEX(regs, leaf, sub) \
        __cpuid_count((leaf), (sub), (regs)[0], (regs)[1], (regs)[2], (regs)[3])
  #else
    #include <intrin.h>
    #define _VFFT_CPUIDEX(regs, leaf, sub) \
        __cpuidex((int *)(regs), (leaf), (sub))
  #endif
#endif

/* ── MSVC compatibility shims ──────────────────────────────────────
 * Codelets/core use GCC/Clang/ICX __restrict__ and target attributes; MSVC
 * accepts neither. Map them so the same source builds on cl.exe too. */
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
  #ifndef __restrict__
    #define __restrict__ __restrict
  #endif
  #ifndef __attribute__
    #define __attribute__(x) /* no-op: MSVC uses /arch:AVX2 globally */
  #endif
#endif

#if defined(__linux__) && !defined(_GNU_SOURCE)
  #define _GNU_SOURCE 1   /* guarded: the driver may already define it */
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#endif

#ifndef _WIN32
#include <stdint.h> /* uintptr_t */
#endif

/* =====================================================================
 * DENORMAL HANDLING (FTZ/DAZ)
 *
 * Denormal arithmetic is 50-100x slower on x86 (microcode trap). In FFT they
 * appear from near-zero inputs, twiddle-product underflow, and inverse scaling.
 * FTZ flushes denormal RESULTS to zero, DAZ treats denormal INPUTS as zero.
 * Both are safe for FFT (below any real signal's noise floor) — HPC math
 * libs enable them. MXCSR is per-thread; call from each thread.
 * ===================================================================== */
static inline unsigned int vfft_env_init(void)
{
    unsigned int old_mxcsr = _mm_getcsr();
    /* FTZ = bit 15 (0x8000), DAZ = bit 6 (0x0040) */
    _mm_setcsr(old_mxcsr | 0x8040);
    return old_mxcsr;
}

static inline void vfft_env_restore(unsigned int saved_mxcsr)
{
    _mm_setcsr(saved_mxcsr);
}

/* Aligned memory: vfft_aligned_alloc / vfft_aligned_free (support/zalloc.h).
 * 2 MB pages for big data planes (the DTLB fix): support/hugepage.h, not wired
 * in yet. */
#include "common/support/zalloc.h" /* vfft_aligned_alloc / vfft_aligned_free: the one allocator */

/* =====================================================================
 * VERBOSITY + VERSION / ISA QUERY
 * ===================================================================== */

#define VFFT_VERSION_MAJOR 0
#define VFFT_VERSION_MINOR 1
#define VFFT_VERSION_PATCH 0
#define VFFT_VERSION_STRING "0.1.0"

#include "build_isa.h"   /* VFFT_ISA_NAME: the build's ISA, decided once */

static int _vfft_verbose = 0;

static inline void vfft_set_verbose(int level)
{
    _vfft_verbose = level;
}

static inline int vfft_get_verbose(void)
{
    return _vfft_verbose;
}

/* Prints version/ISA/CPU/FTZ info to stderr when verbose. Call after init. */
static inline void vfft_print_info(void)
{
    if (!_vfft_verbose)
        return;
    fprintf(stderr, "[VectorFFT] version %s  ISA: %s  sizeof(double)=%zu\n",
            VFFT_VERSION_STRING, VFFT_ISA_NAME, sizeof(double));
#if defined(_WIN32)
    {
        int cpuinfo[4] = {0};
        char brand[49] = {0};
        (void)cpuinfo;
        _VFFT_CPUIDEX((int *)&brand[0], 0x80000002, 0);
        _VFFT_CPUIDEX((int *)&brand[16], 0x80000003, 0);
        _VFFT_CPUIDEX((int *)&brand[32], 0x80000004, 0);
        fprintf(stderr, "[VectorFFT] CPU: %s\n", brand);
    }
#elif defined(__linux__)
    {
        FILE *f = fopen("/proc/cpuinfo", "r");
        if (f)
        {
            char line[256];
            while (fgets(line, sizeof(line), f))
            {
                if (strncmp(line, "model name", 10) == 0)
                {
                    char *p = strchr(line, ':');
                    if (p)
                        fprintf(stderr, "[VectorFFT] CPU:%s", p + 1);
                    break;
                }
            }
            fclose(f);
        }
    }
#endif
    fprintf(stderr, "[VectorFFT] FTZ+DAZ: %s\n",
            (_mm_getcsr() & 0x8040) == 0x8040 ? "enabled" : "disabled");
}

/* Thread count and the pool live in threads.h. */

/* =====================================================================
 * CPU AFFINITY / CORE PINNING
 *
 * Pinning prevents OS migration (L1/L2 invalidation, cross-CCX on Zen,
 * P/E-core migration on hybrid parts, NUMA). For benchmarking, pin to a P-core
 * to kill the biggest run-to-run variance source on hybrid CPUs.
 * ===================================================================== */

/* core_id: 0-based logical processor. Returns 0 / -1. */
static inline int vfft_pin_thread(int core_id)
{
    if (core_id < 0)
        return -1;
#if defined(_WIN32)
    DWORD_PTR mask = (DWORD_PTR)1 << core_id;
    return SetThreadAffinityMask(GetCurrentThread(), mask) ? 0 : -1;
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0 ? 0 : -1;
#else
    (void)core_id;
    return -1;
#endif
}

static inline int vfft_unpin_thread(void)
{
#if defined(_WIN32)
    DWORD_PTR all = ~(DWORD_PTR)0;
    return SetThreadAffinityMask(GetCurrentThread(), all) ? 0 : -1;
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    for (long i = 0; i < nprocs && i < CPU_SETSIZE; i++)
        CPU_SET((int)i, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0 ? 0 : -1;
#else
    return -1;
#endif
}

/* Total logical processors (incl. hyperthreads + E-cores). */
static inline int vfft_num_cores(void)
{
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#elif defined(__linux__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#else
    return 1;
#endif
}

#endif /* VFFT_COMMON_ENV_H */
