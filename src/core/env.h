/**
 * stride_env.h -- CPU environment setup for stride-based FFT
 *
 * Controls hardware settings that affect FFT performance:
 *   - Denormal handling (FTZ/DAZ)
 *   - Memory allocation with SIMD alignment
 *
 * Usage:
 *   stride_env_init();  // call once at program start
 *
 *   // Allocate SIMD-aligned buffers for FFT data
 *   double *re = stride_alloc(N * K * sizeof(double));
 *   double *im = stride_alloc(N * K * sizeof(double));
 *   // ... use with stride_execute_fwd/bwd ...
 *   stride_free(re);
 *   stride_free(im);
 */
#ifndef STRIDE_ENV_H
#define STRIDE_ENV_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#elif defined(__linux__)
#  define _GNU_SOURCE
#  include <pthread.h>
#  include <sched.h>
#  include <unistd.h>
#  include <sys/mman.h>
#  include <stdint.h>
#endif

#ifndef _WIN32
#  include <stdint.h>  /* uintptr_t */
#endif

/* =====================================================================
 * DENORMAL HANDLING
 *
 * Denormal (subnormal) floating-point values occur when results are
 * very close to zero. On x86 CPUs, arithmetic with denormals is
 * 50-100x slower than normal floats because it traps to microcode.
 *
 * In FFT computation, denormals appear when:
 *   - Input data contains near-zero values
 *   - Twiddle factor products underflow during deep butterfly chains
 *   - Inverse FFT scaling produces small intermediate values
 *
 * FTZ (Flush To Zero): denormal RESULTS are flushed to zero
 * DAZ (Denormals Are Zero): denormal INPUTS are treated as zero
 *
 * Both are safe for FFT: denormals are below the noise floor of
 * any real signal. MKL, IPP, and most HPC libraries enable these.
 * ===================================================================== */

/**
 * stride_env_init -- Configure CPU for optimal FFT performance.
 *
 * Sets FTZ+DAZ flags on the calling thread's MXCSR register.
 * Call once at program start, before any FFT operations.
 *
 * Note: MXCSR is per-thread. If using multiple threads, call this
 * from each thread (child threads inherit the parent's MXCSR on
 * most platforms, but this is not guaranteed by all runtimes).
 *
 * Returns the previous MXCSR value (for restoring if needed).
 */
static inline unsigned int stride_env_init(void) {
    unsigned int old_mxcsr = _mm_getcsr();
    /* FTZ = bit 15 (0x8000), DAZ = bit 6 (0x0040) */
    _mm_setcsr(old_mxcsr | 0x8040);
    return old_mxcsr;
}

/**
 * stride_env_restore -- Restore previous MXCSR state.
 *
 * Use this if you need to return to the caller's original
 * denormal handling behavior (e.g., in a library context where
 * you shouldn't permanently alter global CPU state).
 */
static inline void stride_env_restore(unsigned int saved_mxcsr) {
    _mm_setcsr(saved_mxcsr);
}

/* =====================================================================
 * ALIGNED MEMORY ALLOCATION
 *
 * Two allocation strategies:
 *
 * 1. stride_alloc / stride_free -- 64-byte aligned, standard pages.
 *    Use for small buffers, twiddle tables, metadata.
 *
 * 2. stride_alloc_huge / stride_free_huge -- 2MB huge pages.
 *    Use for FFT data buffers (re[], im[]) when total size > 64KB.
 *    Eliminates DTLB misses caused by stride access patterns.
 *
 * VTune profiling shows 23% DTLB Store Overhead at N=1000 K=256
 * (4MB data across 1000 4KB pages, stride=2KB touches new page every
 * 2 elements). With 2MB huge pages, same data spans 2 pages -> zero
 * TLB misses.
 *
 * Huge page requirements:
 *   Windows: "Lock pages in memory" privilege (gpedit.msc -> User Rights)
 *   Linux:   echo 64 > /proc/sys/vm/nr_hugepages (or use THP)
 *
 * If huge pages are unavailable, falls back to standard allocation.
 * ===================================================================== */

#define STRIDE_ALIGNMENT 64
#define STRIDE_HUGEPAGE_THRESHOLD (64 * 1024)  /* use huge pages above 64KB */

/**
 * stride_alloc -- Allocate SIMD-aligned memory (standard pages).
 *
 * Returns a 64-byte aligned pointer suitable for FFT data buffers.
 * Free with stride_free(), NOT standard free().
 */
static inline void *stride_alloc(size_t bytes) {
    if (bytes == 0) return NULL;
#ifdef _WIN32
    return _aligned_malloc(bytes, STRIDE_ALIGNMENT);
#else
    void *p = NULL;
    if (posix_memalign(&p, STRIDE_ALIGNMENT, bytes) != 0) return NULL;
    return p;
#endif
}

/**
 * stride_alloc_huge -- Allocate with 2MB huge pages.
 *
 * For large FFT data buffers where stride access patterns cause
 * DTLB misses with 4KB pages. Falls back to stride_alloc() if
 * huge pages are unavailable.
 *
 * The returned pointer must be freed with stride_free_huge().
 */
static inline void *stride_alloc_huge(size_t bytes) {
    if (bytes == 0) return NULL;

    /* Only use huge pages for large allocations */
    if (bytes < STRIDE_HUGEPAGE_THRESHOLD)
        return stride_alloc(bytes);

#ifdef _WIN32
    /* VirtualAlloc with MEM_LARGE_PAGES.
     * Requires "Lock pages in memory" privilege.
     * Size must be a multiple of GetLargePageMinimum() (typically 2MB). */
    {
        SIZE_T page_size = GetLargePageMinimum();
        if (page_size == 0) goto fallback;

        /* Round up to huge page boundary */
        SIZE_T alloc_size = (bytes + page_size - 1) & ~(page_size - 1);

        void *p = VirtualAlloc(NULL, alloc_size,
                               MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (p) return p;
    }
#elif defined(__linux__)
    /* mmap with MAP_HUGETLB for explicit huge pages.
     * Requires nr_hugepages > 0 in /proc/sys/vm. */
    {
        #ifndef MAP_HUGETLB
        #define MAP_HUGETLB 0x40000
        #endif
        size_t page_size = 2 * 1024 * 1024; /* 2MB */
        size_t alloc_size = (bytes + page_size - 1) & ~(page_size - 1);

        void *p = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED) return p;

        /* Try transparent huge pages as second attempt */
        p = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) {
            madvise(p, alloc_size, MADV_HUGEPAGE);
            return p;
        }
    }
#endif

#ifdef _WIN32
fallback:
#endif
    /* Fallback to standard aligned allocation */
    return stride_alloc(bytes);
}

/**
 * stride_free -- Free memory allocated by stride_alloc.
 */
static inline void stride_free(void *p) {
    if (!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/**
 * stride_free_huge -- Free memory allocated by stride_alloc_huge.
 *
 * Must detect whether the pointer came from VirtualAlloc/mmap (huge)
 * or from stride_alloc (fallback). We use a simple heuristic:
 * huge-page allocations are always 2MB-aligned.
 */
static inline void stride_free_huge(void *p, size_t bytes) {
    if (!p) return;

    /* If below threshold, it was a regular stride_alloc */
    if (bytes < STRIDE_HUGEPAGE_THRESHOLD) {
        stride_free(p);
        return;
    }

#ifdef _WIN32
    /* Check if 2MB-aligned (came from VirtualAlloc with MEM_LARGE_PAGES) */
    if (((uintptr_t)p & (2 * 1024 * 1024 - 1)) == 0) {
        VirtualFree(p, 0, MEM_RELEASE);
        return;
    }
    /* Fallback path used stride_alloc */
    _aligned_free(p);
#elif defined(__linux__)
    /* Check if 2MB-aligned (came from mmap) */
    if (((uintptr_t)p & (2 * 1024 * 1024 - 1)) == 0) {
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
 *
 * Controlled by stride_set_verbose(). When enabled, library functions
 * print diagnostic info (ISA, cache sizes, plan choices) to stderr.
 * Default: silent.
 * ===================================================================== */

#define STRIDE_VERSION_MAJOR 0
#define STRIDE_VERSION_MINOR 1
#define STRIDE_VERSION_PATCH 0
#define STRIDE_VERSION_STRING "0.1.0"

/* Compile-time ISA detection — matches CMake ISA selection */
#if defined(__AVX512F__) && defined(__AVX512DQ__)
#  define STRIDE_ISA_NAME "avx512"
#elif defined(__AVX2__)
#  define STRIDE_ISA_NAME "avx2"
#else
#  define STRIDE_ISA_NAME "scalar"
#endif

static int _stride_verbose = 0;

/**
 * stride_set_verbose -- Enable/disable diagnostic output.
 *
 * level=0: silent (default)
 * level=1: print ISA, cache info, plan choices to stderr
 */
static inline void stride_set_verbose(int level) {
    _stride_verbose = level;
}

static inline int stride_get_verbose(void) {
    return _stride_verbose;
}

/**
 * stride_print_info -- Print library version, ISA, and detected CPU info.
 *
 * Only prints when verbose mode is active. Call after stride_env_init().
 */
static inline void stride_print_info(void) {
    if (!_stride_verbose) return;
    fprintf(stderr, "[VectorFFT] version %s  ISA: %s  sizeof(double)=%zu\n",
            STRIDE_VERSION_STRING, STRIDE_ISA_NAME, sizeof(double));
#if defined(_WIN32)
    {
        int cpuinfo[4] = {0};
        char brand[49] = {0};
        __cpuidex((int *)&brand[0],  0x80000002, 0);
        __cpuidex((int *)&brand[16], 0x80000003, 0);
        __cpuidex((int *)&brand[32], 0x80000004, 0);
        fprintf(stderr, "[VectorFFT] CPU: %s\n", brand);
    }
#elif defined(__linux__)
    /* /proc/cpuinfo first model name line */
    {
        FILE *f = fopen("/proc/cpuinfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "model name", 10) == 0) {
                    char *p = strchr(line, ':');
                    if (p) fprintf(stderr, "[VectorFFT] CPU:%s", p + 1);
                    break;
                }
            }
            fclose(f);
        }
    }
#endif
    /* Cache info (uses stride_detect_cpu from factorizer.h if available) */
    fprintf(stderr, "[VectorFFT] FTZ+DAZ: %s\n",
            (_mm_getcsr() & 0x8040) == 0x8040 ? "enabled" : "disabled");
}

/* =====================================================================
 * THREAD CONTROL
 *
 * Controls how many threads the FFT executor uses for parallel work.
 * Currently single-threaded — the API exists so user code can be
 * written against it now and benefit when multithreading is added.
 *
 * Planned approach: K-split parallelism. Each thread gets a contiguous
 * chunk of the K (batch) dimension and runs the full N-point FFT
 * pipeline on its chunk. No synchronization between threads during
 * execution — the stages are independent across K.
 * ===================================================================== */

/* Thread count and pool management live in threads.h.
 * Include threads.h to use stride_set_num_threads / stride_get_num_threads. */

/* =====================================================================
 * CPU AFFINITY
 *
 * Pin the calling thread to a specific logical CPU core. This prevents
 * the OS scheduler from migrating threads between cores, which causes:
 *   - L1/L2 cache invalidation (cold cache on new core)
 *   - Cross-CCX latency on AMD (Zen) processors
 *   - P-core / E-core migration on Intel hybrid (Alder Lake+)
 *   - NUMA penalties when crossing socket boundaries
 *
 * For FFT benchmarking, pinning to a P-core eliminates the biggest
 * source of run-to-run variance on hybrid architectures.
 *
 * For production multithreaded FFT, each worker thread should be pinned
 * to a distinct core to avoid contention and ensure local cache usage.
 * ===================================================================== */

/**
 * stride_pin_thread -- Pin the calling thread to a specific logical core.
 *
 * core_id: 0-based logical processor index.
 *          On hybrid CPUs, P-cores are typically the lower indices.
 *
 * Returns 0 on success, -1 on failure (invalid core_id or OS error).
 *
 * Platform notes:
 *   Windows: uses SetThreadAffinityMask (single core from group 0)
 *   Linux:   uses pthread_setaffinity_np with CPU_SET
 *   Other:   returns -1 (not implemented)
 */
static inline int stride_pin_thread(int core_id) {
    if (core_id < 0) return -1;
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

/**
 * stride_unpin_thread -- Remove CPU affinity restriction.
 *
 * Allows the OS scheduler to place the thread on any core again.
 * Returns 0 on success, -1 on failure.
 */
static inline int stride_unpin_thread(void) {
#if defined(_WIN32)
    /* Set affinity to all processors in group 0 */
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

/**
 * stride_get_num_cores -- Query number of logical processors available.
 *
 * Returns the total number of logical cores (including hyperthreads
 * and E-cores on hybrid CPUs). Use stride_pin_thread to select
 * specific cores for benchmarking.
 */
static inline int stride_get_num_cores(void) {
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

#endif /* STRIDE_ENV_H */
