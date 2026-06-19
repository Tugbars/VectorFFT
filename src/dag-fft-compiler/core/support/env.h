/* env.h — dag-fft-compiler core environment header. Two parts:
 *
 *   PART 1 — CPU / runtime environment setup, carried over verbatim from the
 *            production src/core/env.h: denormal handling (FTZ/DAZ), SIMD-aligned
 *            + huge-page allocation, verbosity/version/ISA query, and thread
 *            affinity / core pinning. (stride_* names kept as-is.)
 *
 *   PART 2 — Exhaustive/joint SEARCH tuning knobs (stage-depth caps, variant
 *            pre-screen factor) with their VALIDATION-SCOPE note: these defaults
 *            were tuned on N=1024 only — see the banner in Part 2.
 */
#ifndef VFFT_PROTO_CORE_ENV_H
#define VFFT_PROTO_CORE_ENV_H

/* ===========================================================================
 * PART 1 — CPU / RUNTIME ENVIRONMENT  (carried over from src/core/env.h)
 *
 *   stride_env_init();  // call once at program start (FTZ/DAZ)
 *   double *re = stride_alloc(N * K * sizeof(double));
 *   ... stride_execute_fwd/bwd ...
 *   stride_free(re);
 * ===========================================================================
 */

/* On Windows, stride_print_info() calls __cpuidex. MSVC/ICX expose it via
 * <intrin.h>; MinGW GCC via <cpuid.h> (which <intrin.h> includes). Pull
 * <intrin.h> in so the declaration is visible before the call site. */
#if defined(_WIN32)
  #include <intrin.h>
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
 * Both are safe for FFT (below any real signal's noise floor) — MKL/IPP/HPC
 * libs enable them. MXCSR is per-thread; call from each thread.
 * ===================================================================== */
static inline unsigned int stride_env_init(void)
{
    unsigned int old_mxcsr = _mm_getcsr();
    /* FTZ = bit 15 (0x8000), DAZ = bit 6 (0x0040) */
    _mm_setcsr(old_mxcsr | 0x8040);
    return old_mxcsr;
}

static inline void stride_env_restore(unsigned int saved_mxcsr)
{
    _mm_setcsr(saved_mxcsr);
}

/* =====================================================================
 * ALIGNED + HUGE-PAGE MEMORY ALLOCATION
 *
 *  stride_alloc / stride_free            — 64-byte aligned, standard pages.
 *  stride_alloc_huge / stride_free_huge  — 2MB huge pages for FFT data buffers
 *      (re[]/im[]) when size > 64KB; eliminates DTLB misses from stride access
 *      (VTune: 23% DTLB Store Overhead at N=1000 K=256 with 4KB pages). Falls
 *      back to standard alloc if huge pages are unavailable.
 *  Huge-page setup: Windows "Lock pages in memory" privilege; Linux
 *      echo N > /proc/sys/vm/nr_hugepages (or THP).
 * ===================================================================== */

#define STRIDE_ALIGNMENT 64
#define STRIDE_HUGEPAGE_THRESHOLD (64 * 1024) /* use huge pages above 64KB */

static inline void *stride_alloc(size_t bytes)
{
    if (bytes == 0)
        return NULL;
#ifdef _WIN32
    return _aligned_malloc(bytes, STRIDE_ALIGNMENT);
#else
    void *p = NULL;
    if (posix_memalign(&p, STRIDE_ALIGNMENT, bytes) != 0)
        return NULL;
    return p;
#endif
}

static inline void *stride_alloc_huge(size_t bytes)
{
    if (bytes == 0)
        return NULL;

    /* Only use huge pages for large allocations */
    if (bytes < STRIDE_HUGEPAGE_THRESHOLD)
        return stride_alloc(bytes);

#ifdef _WIN32
    /* VirtualAlloc with MEM_LARGE_PAGES. Requires "Lock pages in memory"
     * privilege. Size must be a multiple of GetLargePageMinimum() (~2MB). */
    {
        SIZE_T page_size = GetLargePageMinimum();
        if (page_size == 0)
            goto fallback;

        SIZE_T alloc_size = (bytes + page_size - 1) & ~(page_size - 1);

        void *p = VirtualAlloc(NULL, alloc_size,
                               MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (p)
            return p;
    }
#elif defined(__linux__)
    /* mmap with MAP_HUGETLB; requires nr_hugepages > 0. THP as fallback. */
    {
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif
        size_t page_size = 2 * 1024 * 1024; /* 2MB */
        size_t alloc_size = (bytes + page_size - 1) & ~(page_size - 1);

        void *p = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED)
            return p;

        p = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED)
        {
            madvise(p, alloc_size, MADV_HUGEPAGE);
            return p;
        }
    }
#endif

#ifdef _WIN32
fallback:
#endif
    return stride_alloc(bytes);
}

static inline void stride_free(void *p)
{
    if (!p)
        return;
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* Detects huge vs fallback by 2MB-alignment heuristic. */
static inline void stride_free_huge(void *p, size_t bytes)
{
    if (!p)
        return;

    if (bytes < STRIDE_HUGEPAGE_THRESHOLD)
    {
        stride_free(p);
        return;
    }

#ifdef _WIN32
    if (((uintptr_t)p & (2 * 1024 * 1024 - 1)) == 0)
    {
        VirtualFree(p, 0, MEM_RELEASE);
        return;
    }
    _aligned_free(p);
#elif defined(__linux__)
    if (((uintptr_t)p & (2 * 1024 * 1024 - 1)) == 0)
    {
        size_t page_size = 2 * 1024 * 1024;
        size_t alloc_size = (bytes + page_size - 1) & ~(page_size - 1);
        munmap(p, alloc_size);
        return;
    }
    free(p);
#else
    stride_free(p);
#endif
}

/* =====================================================================
 * VERBOSITY + VERSION / ISA QUERY
 * ===================================================================== */

#define STRIDE_VERSION_MAJOR 0
#define STRIDE_VERSION_MINOR 1
#define STRIDE_VERSION_PATCH 0
#define STRIDE_VERSION_STRING "0.1.0"

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define STRIDE_ISA_NAME "avx512"
#elif defined(__AVX2__)
#define STRIDE_ISA_NAME "avx2"
#else
#define STRIDE_ISA_NAME "scalar"
#endif

static int _stride_verbose = 0;

static inline void stride_set_verbose(int level)
{
    _stride_verbose = level;
}

static inline int stride_get_verbose(void)
{
    return _stride_verbose;
}

/* Prints version/ISA/CPU/FTZ info to stderr when verbose. Call after init. */
static inline void stride_print_info(void)
{
    if (!_stride_verbose)
        return;
    fprintf(stderr, "[VectorFFT] version %s  ISA: %s  sizeof(double)=%zu\n",
            STRIDE_VERSION_STRING, STRIDE_ISA_NAME, sizeof(double));
#if defined(_WIN32)
    {
        int cpuinfo[4] = {0};
        char brand[49] = {0};
        (void)cpuinfo;
        __cpuidex((int *)&brand[0], 0x80000002, 0);
        __cpuidex((int *)&brand[16], 0x80000003, 0);
        __cpuidex((int *)&brand[32], 0x80000004, 0);
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

/* =====================================================================
 * THREAD CONTROL
 * Thread count / pool management live in threads.h (stride_set_num_threads /
 * stride_get_num_threads). Planned model: K-split parallelism (each thread a
 * contiguous K-chunk, full N-FFT, no inter-thread sync — stages independent
 * across K).
 * ===================================================================== */

/* =====================================================================
 * CPU AFFINITY / CORE PINNING
 *
 * Pinning prevents OS migration (L1/L2 invalidation, cross-CCX on Zen,
 * P/E-core migration on Intel hybrid, NUMA). For benchmarking, pin to a P-core
 * to kill the biggest run-to-run variance source on hybrid CPUs.
 * ===================================================================== */

/* core_id: 0-based logical processor. Returns 0 / -1. */
static inline int stride_pin_thread(int core_id)
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

static inline int stride_unpin_thread(void)
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
static inline int stride_get_num_cores(void)
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

/* ===========================================================================
 * PART 2 — EXHAUSTIVE / JOINT SEARCH TUNING KNOBS
 *
 *  ⚠ VALIDATION SCOPE — READ BEFORE TRUSTING THE DEFAULTS
 *  The default values below were tested/tuned on exactly **N=1024 K=4 (pow2)**
 *  on the i9-14900KF (2026-06-14). The absolutely-exhaustive validation run
 *  (262,428 candidates: all 35 decompositions x all orderings x all variants,
 *  NO pruning) established, for 1024:
 *    - latency is MONOTONICALLY worse with stage-count (2-stage 64x16 = 3.2us
 *      optimal; 10-stage = 13.6us, 4.2x); deeper brings ZERO benefit.
 *    - the depth-5 pow2 cap + the 2x variant pre-screen reach the IDENTICAL
 *      winner as no-pruning, at 32x fewer candidates / 56x faster.
 *  That is a **1024 result**, NOT validated at other sizes:
 *    - Larger pow2 (e.g. N=2^17): optimum may sit at 4-5 stages; depth-5 could
 *      clip a deeper win, and a different prune may matter.
 *    - Non-pow2 N (many small primes): needs deeper plans (hence nonpow2=9);
 *      the 2x prune may be too aggressive there.
 *  Before relying on these for a new size class, re-run the env-gated
 *  absolutely-exhaustive sweep and check the best-per-stage-count curve:
 *      VFFT_PROTO_EXH_MAX_DEPTH=16  VFFT_PROTO_EXH_PRUNE=1e9
 *  Every knob is env-overridable, so this needs no recompile.
 * ===========================================================================
 */

/* Stage-depth cap (exhaustive enumeration). Pow2 optima are shallow (2-stage
 * won at 1024); each extra stage = one more full memory pass => slower.
 * Non-pow2 needs deeper plans. TESTED-ON: 1024 (pow2 leg).
 * Override: VFFT_PROTO_EXH_MAX_DEPTH. */
#ifndef VFFT_PROTO_EXH_MAX_DEPTH_POW2
#define VFFT_PROTO_EXH_MAX_DEPTH_POW2    5
#endif
#ifndef VFFT_PROTO_EXH_MAX_DEPTH_NONPOW2
#define VFFT_PROTO_EXH_MAX_DEPTH_NONPOW2 9
#endif

/* Variant pre-screen factor: skip a factorization's variant cartesian when its
 * default-variant bench > FACTOR x the running global best. At 1024, 2x
 * recorded every depth's default while skipping hopeless deep cartesians,
 * giving the identical conclusion to no-prune. TESTED-ON: 1024.
 * Override: VFFT_PROTO_EXH_PRUNE (set huge, e.g. 1e9, to disable). */
#ifndef VFFT_PROTO_EXH_PRUNE_FACTOR
#define VFFT_PROTO_EXH_PRUNE_FACTOR 2.0
#endif

/* NOTE: a third cap, the per-decomposition permutation limit
 * VFFT_PROTO_DP_MAX_PERMS (720), lives in dp_planner.h. Not hit for 1024 pow2,
 * but it CAN clip orderings for non-pow2 with many distinct small primes —
 * also untested outside 1024. */

/* Accessors: default unless the matching env var overrides. hard_cap clamps to
 * the stage-array bound (pass STRIDE_MAX_STAGES); 0 = no clamp. */
static inline int vfft_proto_env_max_depth(int n_is_pow2, int hard_cap) {
    int d = n_is_pow2 ? VFFT_PROTO_EXH_MAX_DEPTH_POW2
                      : VFFT_PROTO_EXH_MAX_DEPTH_NONPOW2;
    const char *e = getenv("VFFT_PROTO_EXH_MAX_DEPTH");
    if (e) { int v = atoi(e); if (v > 0) d = v; }
    if (hard_cap > 0 && d > hard_cap) d = hard_cap;
    return d;
}

static inline double vfft_proto_env_prune_factor(void) {
    double p = VFFT_PROTO_EXH_PRUNE_FACTOR;
    const char *e = getenv("VFFT_PROTO_EXH_PRUNE");
    if (e) { double v = atof(e); if (v > 0) p = v; }
    return p;
}

/* Wisdom-write overwrite flag, shared by the calibrator (patient/measure) and
 * the planner write paths. 0 = preserve already-calibrated cells (skip them;
 * only fill in missing ones) — the safe incremental default. 1 = re-calibrate
 * and overwrite the cell with the new winner (vfft_proto_wisdom_add collapses to
 * one entry). Override: VFFT_PROTO_WISDOM_OVERWRITE=1. */
#ifndef VFFT_PROTO_WISDOM_OVERWRITE
#define VFFT_PROTO_WISDOM_OVERWRITE 0
#endif

static inline int vfft_proto_env_wisdom_overwrite(void) {
    int v = VFFT_PROTO_WISDOM_OVERWRITE;
    const char *e = getenv("VFFT_PROTO_WISDOM_OVERWRITE");
    if (e && *e) v = atoi(e);
    return v;
}

#endif /* VFFT_PROTO_CORE_ENV_H */
