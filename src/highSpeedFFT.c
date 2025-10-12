#include "highspeedFFT.h"
#include "time.h"
#include <immintrin.h>
#include <pthread.h>


//==============================================================================
// INLINE / ATTRIBUTES
//==============================================================================
#ifdef _MSC_VER
#define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

/**
 * @brief Build configuration option for twiddle factor computation.
 * Define USE_TWIDDLE_TABLES to use precomputed lookup tables for radices 2, 3, 4, 5, 7, 8, 11, and 13.
 * If undefined, all twiddle factors are computed dynamically at runtime using cos and sin.
 */
#define USE_TWIDDLE_TABLES

//==============================================================================
// FMA MACROS
//   - AVX: FMADD/FMSUB (256-bit)
//   - 128-bit: FMADD_SSE2_PD / FMSUB_SSE2_PD use FMA if available, else fallback
//   - Convenience aliases: FMADD_SSE2 / FMSUB_SSE2 == 128-bit versions
//==============================================================================

// 128-bit fallback helpers are always available (SSE2 has no FMA)
static ALWAYS_INLINE __m128d fmadd_sse2_fallback(__m128d a, __m128d b, __m128d c)
{
    return _mm_add_pd(_mm_mul_pd(a, b), c);
}
static ALWAYS_INLINE __m128d fmsub_sse2_fallback(__m128d a, __m128d b, __m128d c)
{
    return _mm_sub_pd(_mm_mul_pd(a, b), c);
}

#if defined(__FMA__) || defined(USE_FMA)

// --- 256-bit AVX FMA ---
#define FMADD(a, b, c) _mm256_fmadd_pd((a), (b), (c))
#define FMSUB(a, b, c) _mm256_fmsub_pd((a), (b), (c))

// --- 128-bit FMA (requires -mfma) ---
#define FMADD_SSE2_PD(a, b, c) _mm_fmadd_pd((a), (b), (c))
#define FMSUB_SSE2_PD(a, b, c) _mm_fmsub_pd((a), (b), (c))

#else

// --- 256-bit fallback ---
static ALWAYS_INLINE __m256d fmadd_fallback(__m256d a, __m256d b, __m256d c)
{
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
}
static ALWAYS_INLINE __m256d fmsub_fallback(__m256d a, __m256d b, __m256d c)
{
    return _mm256_sub_pd(_mm256_mul_pd(a, b), c);
}
#define FMADD(a, b, c) fmadd_fallback((a), (b), (c))
#define FMSUB(a, b, c) fmsub_fallback((a), (b), (c))

// --- 128-bit fallback ---
#define FMADD_SSE2_PD(a, b, c) fmadd_sse2_fallback((a), (b), (c))
#define FMSUB_SSE2_PD(a, b, c) fmsub_sse2_fallback((a), (b), (c))

#endif

// Convenience aliases to the 128-bit versions
#define FMADD_SSE2(a, b, c) FMADD_SSE2_PD((a), (b), (c))
#define FMSUB_SSE2(a, b, c) FMSUB_SSE2_PD((a), (b), (c))

//==============================================================================
// LOAD / STORE WRAPPERS
// If USE_ALIGNED_SIMD is defined, enforce alignment at the *access site*.
// On misalignment:
//   - If FFT_STRICT_ALIGNMENT is defined: print error and abort()
//   - Else: print error and FALL BACK to unaligned op
// If USE_ALIGNED_SIMD is NOT defined: always use unaligned ops (no checks).
//==============================================================================

//==============================================================================
// LOAD / STORE WRAPPERS
// Debug builds check alignment; release builds use fast paths
//==============================================================================

#ifdef FFT_DEBUG_ALIGNMENT
#define FFT_ALIGNMENT_CHECK
#endif

//=============================================================================
// ALIGNMENT HELPER FOR AVX-512
//==============================================================================
#ifdef HAS_AVX512
/**
 * @brief Check 64-byte alignment for AVX-512.
 */
static ALWAYS_INLINE int is_aligned_64(const void *p)
{
    return (((uintptr_t)p) & 63) == 0;
}
#endif

//==============================================================================
// LOAD / STORE WRAPPERS - AVX-512
//==============================================================================

/**
 * @brief Load 8 doubles (512 bits) with alignment handling.
 */
static ALWAYS_INLINE __m512d LOAD_PD512(const double *ptr)
{
#ifdef FFT_ALIGNMENT_CHECK
    if (!is_aligned_64(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX-512 load at %p (expected 64B)\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        return _mm512_loadu_pd(ptr);
#endif
    }
    return _mm512_load_pd(ptr);
#else
#ifdef USE_ALIGNED_SIMD
    return _mm512_load_pd(ptr);
#else
    return _mm512_loadu_pd(ptr);
#endif
#endif
}

/**
 * @brief Store 8 doubles (512 bits) with alignment handling.
 */
static ALWAYS_INLINE void STORE_PD512(double *ptr, __m512d v)
{
#ifdef FFT_ALIGNMENT_CHECK
    if (!is_aligned_64(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX-512 store at %p (expected 64B)\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        _mm512_storeu_pd(ptr, v);
        return;
#endif
    }
    _mm512_store_pd(ptr, v);
#else
#ifdef USE_ALIGNED_SIMD
    _mm512_store_pd(ptr, v);
#else
    _mm512_storeu_pd(ptr, v);
#endif
#endif
}

// Explicit unaligned versions
#define LOADU_PD512(ptr) _mm512_loadu_pd((const double *)(ptr))
#define STOREU_PD512(ptr, v) _mm512_storeu_pd((double *)(ptr), (v))

static ALWAYS_INLINE int is_aligned_32(const void *p)
{
    return (((uintptr_t)p) & 31) == 0;
}
static ALWAYS_INLINE int is_aligned_16(const void *p)
{
    return (((uintptr_t)p) & 15) == 0;
}

/**
 * @brief Load 4 doubles with optional alignment checking.
 *
 * Behavior depends on compile flags:
 * - FFT_DEBUG_ALIGNMENT: Check alignment at runtime, print warning if misaligned
 * - FFT_STRICT_ALIGNMENT: Abort on misalignment (debug only)
 * - USE_ALIGNED_SIMD: Use fast aligned load (requires 32-byte aligned ptr)
 * - Default: Use safe unaligned load
 *
 * @param ptr Pointer to 4 doubles (32 bytes). Must be 32-byte aligned if USE_ALIGNED_SIMD is set.
 * @return AVX register containing the 4 loaded doubles.
 */
static ALWAYS_INLINE __m256d LOAD_PD(const double *ptr)
{
#ifdef FFT_ALIGNMENT_CHECK
    if (!is_aligned_32(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX load at %p (expected 32B)\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        return _mm256_loadu_pd(ptr);
#endif
    }
    return _mm256_load_pd(ptr);
#else
#ifdef USE_ALIGNED_SIMD
    return _mm256_load_pd(ptr);
#else
    return _mm256_loadu_pd(ptr);
#endif
#endif
}

/**
 * @brief Store 4 doubles with optional alignment checking.
 *
 * Companion to LOAD_PD. Uses aligned/unaligned store based on same compile flags.
 *
 * @param ptr Pointer to write 4 doubles. Must be 32-byte aligned if USE_ALIGNED_SIMD is set.
 * @param v AVX register to store.
 */

static ALWAYS_INLINE void STORE_PD(double *ptr, __m256d v)
{
#ifdef FFT_ALIGNMENT_CHECK
    if (!is_aligned_32(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX store at %p (expected 32B)\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        _mm256_storeu_pd(ptr, v);
        return;
#endif
    }
    _mm256_store_pd(ptr, v);
#else
#ifdef USE_ALIGNED_SIMD
    _mm256_store_pd(ptr, v);
#else
    _mm256_storeu_pd(ptr, v);
#endif
#endif
}

/**
 * @brief Load 2 doubles with optional alignment checking (SSE2).
 *
 * SSE2 version for loading 1 complex number or tail elements.
 * Uses 16-byte alignment requirement instead of 32-byte.
 *
 * @param ptr Pointer to 2 doubles (16 bytes). Must be 16-byte aligned if USE_ALIGNED_SIMD is set.
 * @return SSE register containing the 2 loaded doubles.
 */
static ALWAYS_INLINE __m128d LOAD_SSE2(const double *ptr)
{
#ifdef FFT_ALIGNMENT_CHECK
    if (!is_aligned_16(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned SSE load at %p (expected 16B)\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        return _mm_loadu_pd(ptr);
#endif
    }
    return _mm_load_pd(ptr);
#else
#ifdef USE_ALIGNED_SIMD
    return _mm_load_pd(ptr);
#else
    return _mm_loadu_pd(ptr);
#endif
#endif
}

/**
 * @brief Store 2 doubles with optional alignment checking (SSE2).
 *
 * SSE2 version of STORE_PD for 1 complex number or tail elements.
 *
 * @param ptr Pointer to write 2 doubles. Must be 16-byte aligned if USE_ALIGNED_SIMD is set.
 * @param v SSE register to store.
 */
static ALWAYS_INLINE void STORE_SSE2(double *ptr, __m128d v)
{
#ifdef FFT_ALIGNMENT_CHECK
    if (!is_aligned_16(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned SSE store at %p (expected 16B)\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        _mm_storeu_pd(ptr, v);
        return;
#endif
    }
    _mm_store_pd(ptr, v);
#else
#ifdef USE_ALIGNED_SIMD
    _mm_store_pd(ptr, v);
#else
    _mm_storeu_pd(ptr, v);
#endif
#endif
}

// Explicit unaligned helpers (bypass checks)
#define LOADU_PD(ptr) _mm256_loadu_pd((const double *)(ptr))
#define STOREU_PD(ptr, v) _mm256_storeu_pd((double *)(ptr), (v))
#define LOADU_SSE2(ptr) _mm_loadu_pd((const double *)(ptr))
#define STOREU_SSE2(ptr, v) _mm_storeu_pd((double *)(ptr), (v))

//==============================================================================
// PREFETCH HELPERS
//==============================================================================
#ifndef FFT_PREFETCH_DISTANCE
#define FFT_PREFETCH_DISTANCE 8 // ~64B ahead for AoS complex<double>
#endif

//==============================================================================
// RADIX-SPECIFIC SCALAR CONSTANTS
//==============================================================================

// --- Radix-3 helper ---
static const double C3_SQRT3BY2 = 0.8660254037844386; // √3/2 for 120° rotation

// --- Radix-5 constants ---
static const double C5_1 = 0.30901699437;  // cos(72°)
static const double C5_2 = -0.80901699437; // cos(144°)
static const double S5_1 = 0.95105651629;  // sin(72°)
static const double S5_2 = 0.58778525229;  // sin(144°)

// --- Radix-7 constants ---
static const double C1 = 0.62348980185;  // cos( 51.43°)
static const double C2 = -0.22252093395; // cos(102.86°)
static const double C3 = -0.90096886790; // cos(154.29°)
static const double S1 = 0.78183148246;  // sin( 51.43°)
static const double S2 = 0.97492791218;  // sin(102.86°)
static const double S3 = 0.43388373911;  // sin(154.29°)

// --- Radix-8 constant ---
static const double C8_1 = 0.7071067811865476; // √2/2 for 45° rotation

// --- Radix-11 constants ---
static const double C11_1 = 0.8412535328311812;   // cos(2π/11)
static const double C11_2 = 0.4154150130018864;   // cos(4π/11)
static const double C11_3 = -0.14231483827328514; // cos(6π/11)
static const double C11_4 = -0.6548607339452850;  // cos(8π/11)
static const double C11_5 = -0.9594929736144974;  // cos(10π/11)
static const double S11_1 = 0.5406408174555976;   // sin(2π/11)
static const double S11_2 = 0.9096319953545184;   // sin(4π/11)
static const double S11_3 = 0.9898214418809327;   // sin(6π/11)
static const double S11_4 = 0.7557495743542583;   // sin(8π/11)
static const double S11_5 = 0.28173255684142967;  // sin(10π/11)

// Radix-16 constants (mainly just √2/2 variants)
static const double C16_1 = 0.9238795325112867;  // cos(π/8)
static const double C16_2 = 0.7071067811865476;  // cos(π/4) = √2/2
static const double C16_3 = 0.38268343236508984; // cos(3π/8)
static const double S16_1 = 0.38268343236508984; // sin(π/8)
static const double S16_2 = 0.7071067811865476;  // sin(π/4)
static const double S16_3 = 0.9238795325112867;  // sin(3π/8)

//==============================================================================
// DIVISIBILITY LOOKUP (for dividebyN up to 1024)
//==============================================================================
#define LOOKUP_MAX 1024

static const int primes[] = {
    2, 3, 4, 5, 7, 8, 11, 13, 17, 23, 29, 31, 37, 41, 43, 47, 53};
static const int num_primes = sizeof(primes) / sizeof(primes[0]);

static unsigned char dividebyN_lookup[LOOKUP_MAX]; // 0 = not divisible, 1 = divisible

static const int pre_sizes[] = {1, 2, 3, 4, 5, 7, 15, 20, 31, 64};
static const int num_pre = (int)(sizeof(pre_sizes) / sizeof(pre_sizes[0]));

static fft_data *all_chirps = NULL; // single contiguous block holding all precomputed sequences

// Initialize the lookup table at compile time or runtime
__attribute__((constructor)) static void init_dividebyN_lookup(void)
{
    // Set all to 0 initially (not divisible)
    for (int i = 0; i < LOOKUP_MAX; i++)
    {
        dividebyN_lookup[i] = 0;
    }
    dividebyN_lookup[1] = 1; // Special case: 1 is "divisible" (no factors needed)

    // Mark numbers divisible by the primes
    for (int n = 2; n < LOOKUP_MAX; n++)
    {
        int temp = n;
        int divisible = 1;
        while (temp > 1)
        {
            int factored = 0;
            for (int j = 0; j < num_primes; j++)
            {
                if (temp % primes[j] == 0)
                {
                    temp /= primes[j];
                    factored = 1;
                    break;
                }
            }
            if (!factored)
            {
                divisible = 0;
                break;
            }
        }
        if (divisible)
        {
            dividebyN_lookup[n] = 1;
        }
    }
}

//==============================================================================
// TWIDDLE FACTOR TABLES (per radix)
//==============================================================================

typedef struct
{
    double re;
    double im;
} complex_t;

// --- Radix-2 ---
static const complex_t twiddle_radix2[] = {
    {1.0, 0.0},
    {0.0, -1.0}};

// --- Radix-3 ---
static const complex_t twiddle_radix3[] = {
    {1.0, 0.0},
    {-0.5, -0.86602540378},
    {-0.5, 0.86602540378}};

// --- Radix-4 ---
static const complex_t twiddle_radix4[] = {
    {1.0, 0.0},
    {0.0, -1.0},
    {-1.0, 0.0},
    {0.0, 1.0}};

// --- Radix-5 ---
static const complex_t twiddle_radix5[] = {
    {1.0, 0.0},
    {0.30901699437, -0.95105651629},
    {-0.80901699437, -0.58778525229},
    {-0.80901699437, 0.58778525229},
    {0.30901699437, 0.95105651629}};

// --- Radix-7 ---
static const complex_t twiddle_radix7[] = {
    {1.0, 0.0},
    {0.62348980185, -0.78183148246},
    {-0.22252093395, -0.97492791218},
    {-0.90096886790, -0.43388373911},
    {-0.90096886790, 0.43388373911},
    {-0.22252093395, 0.97492791218},
    {0.62348980185, 0.78183148246}};

// --- Radix-8 ---
static const complex_t twiddle_radix8[] = {
    {1.0, 0.0},
    {0.70710678118, -0.70710678118},
    {0.0, -1.0},
    {-0.70710678118, -0.70710678118},
    {-1.0, 0.0},
    {-0.70710678118, 0.70710678118},
    {0.0, 1.0},
    {0.70710678118, 0.70710678118}};

// --- Radix-11 ---
static const complex_t twiddle_radix11[] = {
    {1.0, 0.0},                                  // k=0
    {0.8412535328311812, -0.5406408174555976},   // k=1
    {0.4154150130018864, -0.9096319953545184},   // k=2
    {-0.14231483827328514, -0.9898214418809327}, // k=3
    {-0.6548607339452850, -0.7557495743542583},  // k=4
    {-0.9594929736144974, -0.28173255684142967}, // k=5
    {-0.9594929736144974, 0.28173255684142967},  // k=6
    {-0.6548607339452850, 0.7557495743542583},   // k=7
    {-0.14231483827328514, 0.9898214418809327},  // k=8
    {0.4154150130018864, 0.9096319953545184},    // k=9
    {0.8412535328311812, 0.5406408174555976}     // k=10
};

// --- Radix-13 ---
static const complex_t twiddle_radix13[] = {
    {1.0, 0.0},                                 // k=0
    {0.8854560256532099, -0.46472317204376856}, // k=1
    {0.5680647467311558, -0.8229838658936564},  // k=2
    {0.12053668025532305, -0.992708874098054},  // k=3
    {-0.3546048870425356, -0.9350162426854148}, // k=4
    {-0.7485107481711011, -0.6631226582407952}, // k=5
    {-0.970941817426052, -0.23931566428755774}, // k=6
    {-0.970941817426052, 0.23931566428755774},  // k=7
    {-0.7485107481711011, 0.6631226582407952},  // k=8
    {-0.3546048870425356, 0.9350162426854148},  // k=9
    {0.12053668025532305, 0.992708874098054},   // k=10
    {0.5680647467311558, 0.8229838658936564},   // k=11
    {0.8854560256532099, 0.46472317204376856}   // k=12
};

//==============================================================================
// TWIDDLE DISPATCH TABLE
//==============================================================================
static const complex_t *twiddle_tables[14] = {
    [0] = NULL,
    [2] = twiddle_radix2,
    [3] = twiddle_radix3,
    [4] = twiddle_radix4,
    [5] = twiddle_radix5,
    [7] = twiddle_radix7,
    [8] = twiddle_radix8,
    [11] = twiddle_radix11,
    [13] = twiddle_radix13};

//==============================================================================
// BLUESTEin CHIRP: PRECOMPUTED SMALL-N TABLE
//==============================================================================

/**
 * @brief Maximum N for precomputed Bluestein chirp sequences.
 */
#define MAX_PRECOMPUTED_N 64

/**
 * @brief Precomputed chirp sequences for Bluestein’s algorithm.
 * Dynamically allocated array of pointers, each pointing to a chirp sequence for a specific N.
 */
static fft_data **bluestein_chirp = NULL;

/**
 * @brief Array of precomputed chirp sequence sizes.
 */
static int *chirp_sizes = NULL;

/**
 * @brief Number of precomputed chirp sequences (== num_pre).
 */
static int num_precomputed = 0;

/**
 * @brief Flag indicating whether the chirp table has been initialized.
 */
static int chirp_initialized = 0;

static pthread_once_t chirp_init_once = PTHREAD_ONCE_INIT;

/* Forward decls for portability (MSVC doesn’t support constructor attrs) */
static void init_bluestein_chirp_body(void);
static void cleanup_bluestein_chirp_body(void);

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
static void
init_bluestein_chirp(void)
{
    init_bluestein_chirp_body();
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((destructor))
#endif
static void
cleanup_bluestein_chirp(void)
{
    cleanup_bluestein_chirp_body();
}

/**
 * @brief Precomputes Bluestein chirp sequences for small signal lengths at program startup.
 *
 * This function runs once when the program starts, thanks to the `__attribute__((constructor))`
 * magic, and sets up a table of precomputed chirp sequences for Bluestein’s algorithm. These
 * sequences are used for signal lengths like 1, 2, 3, 4, 5, 7, 15, 20, 31, and 64, which are
 * common enough to be worth precomputing. Instead of a bunch of separate allocations, we stuff
 * all the chirps into one big, contiguous `all_chirps` array to save memory and keep things cache-friendly.
 *
 * **What’s a chirp sequence?** For Bluestein’s FFT, we need sequences like \( h(n) = e^{\pi i n^2 / N} \)
 * to turn the DFT into a convolution. Precomputing these for small N saves us from pricey sin/cos
 * calls during `bluestein_fft`. The sequences are stored globally, read-only, for all FFT objects to share.
 *
 *
 * @note Runs automatically at program startup. Exits with an error if memory allocation fails.
 * @warning The caller must ensure `cleanup_bluestein_chirp` is called at program exit (via
 *          `__attribute__((destructor))`) to free memory.
 */
static void init_bluestein_chirp_body(void)
{
    // total storage needed (rounded up per size to multiple of 4 for alignment-friendly access)
    int total_chirp = 0;
    for (int i = 0; i < num_pre; i++)
    {
        total_chirp += ((pre_sizes[i] + 3) & ~3);
    }

    // allocate descriptor arrays
    bluestein_chirp = (fft_data **)malloc((size_t)num_pre * sizeof(fft_data *));
    chirp_sizes = (int *)malloc((size_t)num_pre * sizeof(int));
    all_chirps = (fft_data *)_mm_malloc((size_t)total_chirp * sizeof(fft_data), 32);

    if (!bluestein_chirp || !chirp_sizes || !all_chirps)
    {
        fprintf(stderr, "Error: Memory allocation failed for Bluestein chirp table\n");
        if (all_chirps)
            _mm_free(all_chirps);
        if (bluestein_chirp)
            free(bluestein_chirp);
        if (chirp_sizes)
            free(chirp_sizes);
        all_chirps = NULL;
        bluestein_chirp = NULL;
        chirp_sizes = NULL;
        chirp_initialized = 0;
        num_precomputed = 0;
        return; // fail gracefully; caller can still run without precomputed chirps
    }

    // partition the big block and fill chirps
    int offset = 0;
    for (int idx = 0; idx < num_pre; idx++)
    {
        const int n = pre_sizes[idx];
        const int n_rounded = ((n + 3) & ~3);

        chirp_sizes[idx] = n; // logical size
        bluestein_chirp[idx] = all_chirps + offset;
        offset += n_rounded;

        // h(i) = e^{π i * i^2 / n}  (Bluestein chirp)
        // Use quadratic index l2 to accumulate 2*i+1 mod 2n; wrap must be ">= 2n"
        const fft_type theta = (fft_type)M_PI / (fft_type)n;
        int l2 = 0;
        const int len2 = 2 * n;

        for (int i = 0; i < n; i++)
        {
            const fft_type angle = theta * (fft_type)l2;
            bluestein_chirp[idx][i].re = cos(angle);
            bluestein_chirp[idx][i].im = sin(angle);

            l2 += 2 * i + 1;
            while (l2 >= len2)
                l2 -= len2; // wrap (>=, not >)
        }

        // If you prefer to zero the padded tail (n..n_rounded-1), uncomment:
        // for (int i = n; i < n_rounded; ++i) { bluestein_chirp[idx][i].re = 0.0; bluestein_chirp[idx][i].im = 0.0; }
    }

    num_precomputed = num_pre;
    chirp_initialized = 1;
}

static void cleanup_bluestein_chirp_body(void)
{
    if (all_chirps)
        _mm_free(all_chirps);
    if (bluestein_chirp)
        free(bluestein_chirp);
    if (chirp_sizes)
        free(chirp_sizes);

    all_chirps = NULL;
    bluestein_chirp = NULL;
    chirp_sizes = NULL;
    chirp_initialized = 0;
    num_precomputed = 0;
}

/**
 * @brief Checks if n is an exact power of prime p (n = p^k for some k ≥ 0).
 *
 * Returns true if n is divisible only by p, false otherwise. Used to detect
 * pure-power transforms (e.g., N = 3^k, 5^k).
 *
 * @param[in] n Number to check (n > 0).
 * @param[in] p Prime divisor (p > 1).
 * @return bool True if n = p^k, false otherwise.
 */
static bool is_exact_power(int n, int p)
{
    if (n <= 0 || p <= 1)
        return false;
    while (n % p == 0)
        n /= p;
    return n == 1;
}

// Build twiddles in linear order: tw[m] = e^{-2πi m / N}, m=0..N-1
static void build_twiddles_linear(fft_data *tw, int N)
{
    const double theta = -2.0 * M_PI / (double)N; // forward sign; inverse handled later by flipping imag
    for (int m = 0; m < N; ++m)
    {
        const double ang = theta * (double)m;
        tw[m].re = cos(ang);
        tw[m].im = sin(ang);
    }
}

/**
 * @brief Sets up an FFT object for computing Fast Fourier Transforms.
 *
 * This function lays the groundwork for FFT computations, supporting both mixed-radix transforms
 * for lengths that factor into small primes (2, 3, 5, 7, 11, 13, etc.) and Bluestein’s algorithm
 * for arbitrary lengths. It allocates memory, selects the appropriate algorithm, and precomputes
 * twiddle factors to keep `fft_exec` fast. Following FFTW’s lead, we use a two-buffer strategy:
 * one buffer for twiddle factors and a single large scratch buffer for all temporary data.
 * The global `all_chirps` array (initialized elsewhere) handles precomputed Bluestein chirp
 * sequences, so we focus on FFT setup here.
 *
 * **How it works**: It validates the input signal length and transform direction, chooses
 * mixed-radix or Bluestein based on factorization, allocates 32-byte aligned buffers for SIMD
 * performance, and configures the FFT object with lengths, factors, and buffers. For inverse
 * FFTs, it flips twiddle imaginary parts. For power-of-2, 3, 5, 7, 11, or 13 FFTs, it
 * precomputes twiddle factors and stage offsets to optimize `mixed_radix_dit_rec`.
 *
 * @param[in] signal_length The length of the input signal (N > 0). Number of complex samples.
 * @param[in] transform_direction +1 for forward FFT (e^{-2πi k n / N}), -1 for inverse
 *                               (e^{+2πi k n / N}).
 * @return fft_object A pointer to the initialized FFT object, or NULL if allocation fails.
 *
 * @warning Exits with an error if signal_length <= 0 or transform_direction isn’t ±1. Returns
 *          NULL if memory allocation fails.
 * @note Caller must free the object with `free_fft`. Buffers are 32-byte aligned for AVX2/SSE2.
 *       Scratch buffer is sized for worst-case mixed-radix or Bluestein needs.
 */
fft_object fft_init(int signal_length, int transform_direction)
{
    // Step 1: Validate inputs
    if (signal_length <= 0 || (transform_direction != 1 && transform_direction != -1))
    {
        fprintf(stderr, "Error: Signal length (%d) or direction (%d) is invalid\n",
                signal_length, transform_direction);
        return NULL;
    }

    // Step 2: Allocate fft_set structure
    fft_object fft_config = (fft_object)malloc(sizeof(struct fft_set));
    if (!fft_config)
    {
        fprintf(stderr, "Error: Failed to allocate fft_set structure\n");
        return NULL;
    }
    fft_config->num_precomputed_stages = 0;

    // Step 3: Initialize algorithm flags
    int is_factorable = dividebyN(signal_length);
    int is_power_of_2 = 0, is_power_of_3 = 0, is_power_of_5 = 0, is_power_of_7 = 0;
    int is_power_of_11 = 0, is_power_of_13 = 0;
    int is_power_of_16 = 0, is_power_of_32 = 0; // ← ADD THESE
    int twiddle_count = 0, max_scratch_size = 0, max_padded_length = 0;

    // Step 4: Set up buffer sizes and check power-of-radix
    if (is_factorable)
    {
        max_padded_length = signal_length;
        twiddle_count = signal_length;

        // Check for pure powers
        is_power_of_2 = (signal_length & (signal_length - 1)) == 0;
        is_power_of_3 = is_exact_power(signal_length, 3);
        is_power_of_5 = is_exact_power(signal_length, 5);
        is_power_of_7 = is_exact_power(signal_length, 7);
        is_power_of_11 = is_exact_power(signal_length, 11);
        is_power_of_13 = is_exact_power(signal_length, 13);
        is_power_of_16 = is_exact_power(signal_length, 16); // ← ADD
        is_power_of_32 = is_exact_power(signal_length, 32); // ← ADD
    }
    else
    {
        int want = 2 * signal_length - 1;
        unsigned v = (unsigned)want;
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        max_padded_length = (int)v;
        twiddle_count = max_padded_length;
    }

    // Step 5: Compute memory requirements
    int temp_factors[64];
    int num_factors = factors(is_factorable ? signal_length : max_padded_length, temp_factors);
    int twiddle_factors_size = 0;
    int scratch_needed = 0;

    if (is_factorable)
    {
        int temp_N = signal_length;

        // ========================================================================
        // UPDATED: Check for power-of-16 and power-of-32 FIRST (more optimal)
        // ========================================================================
        if (is_power_of_32 || is_power_of_16 || is_power_of_2 || is_power_of_3 ||
            is_power_of_5 || is_power_of_7 || is_power_of_11 || is_power_of_13)
        {
            // Determine radix (prefer larger radices for fewer stages)
            int radix;
            if (is_power_of_32)
            {
                radix = 32; // Best for large power-of-2 FFTs
            }
            else if (is_power_of_16)
            {
                radix = 16; // Good for medium power-of-2 FFTs
            }
            else if (is_power_of_2)
            {
                radix = 2; // Fallback for small power-of-2
            }
            else if (is_power_of_3)
            {
                radix = 3;
            }
            else if (is_power_of_5)
            {
                radix = 5;
            }
            else if (is_power_of_7)
            {
                radix = 7;
            }
            else if (is_power_of_11)
            {
                radix = 11;
            }
            else
            {
                radix = 13;
            }

            int stage = 0;
            for (int n = signal_length; n >= radix; n /= radix)
            {
                int sub_fft_size = n / radix;

                if (stage < MAX_STAGES)
                {
                    fft_config->stage_twiddle_offset[stage++] = twiddle_factors_size;
                }
                else
                {
                    fprintf(stderr, "Error: Exceeded MAX_STAGES (%d) for N=%d, radix=%d\n",
                            MAX_STAGES, signal_length, radix);
                    free_fft(fft_config);
                    return NULL;
                }

                twiddle_factors_size += (radix - 1) * sub_fft_size; // W_n^{j*k}, j=1..r-1
                scratch_needed += radix * sub_fft_size;             // Outputs
            }
            fft_config->num_precomputed_stages = stage;
        }
        else
        {
            // Mixed-radix: r*(N/r) outputs, (r-1)*(N/r) twiddles for radices ≤ 32
            for (int i = 0; i < num_factors; i++)
            {
                int radix = temp_factors[i];
                scratch_needed += radix * (temp_N / radix);
                if (radix <= 32) // ← UPDATED: Include radix-16 and radix-32
                {
                    scratch_needed += (radix - 1) * (temp_N / radix);
                }
                temp_N /= radix;
            }
        }

        max_scratch_size = scratch_needed;
        if (max_scratch_size < 4 * signal_length)
        {
            max_scratch_size = 4 * signal_length;
        }
    }
    else
    {
        max_scratch_size = 4 * max_padded_length;
    }

    // Step 6: Allocate twiddle and scratch buffers
    fft_config->twiddles = (fft_data *)_mm_malloc(twiddle_count * sizeof(fft_data), 32);
    fft_config->scratch = (fft_data *)_mm_malloc(max_scratch_size * sizeof(fft_data), 32);
    fft_config->twiddle_factors = NULL;

    if (!fft_config->twiddles || !fft_config->scratch)
    {
        fprintf(stderr, "Error: Failed to allocate twiddle or scratch buffers\n");
        free_fft(fft_config);
        return NULL;
    }

    // Step 7: Allocate twiddle_factors for pure-power FFTs
    // ========================================================================
    // UPDATED: Include radix-16 and radix-32
    // ========================================================================
    if (is_factorable && (is_power_of_32 || is_power_of_16 || is_power_of_2 ||
                          is_power_of_3 || is_power_of_5 || is_power_of_7 ||
                          is_power_of_11 || is_power_of_13))
    {
        fft_config->twiddle_factors =
            (fft_data *)_mm_malloc(twiddle_factors_size * sizeof(fft_data), 32);
        if (!fft_config->twiddle_factors)
        {
            fprintf(stderr, "Error: Failed to allocate twiddle_factors buffer\n");
            _mm_free(fft_config->twiddles);
            _mm_free(fft_config->scratch);
            free(fft_config);
            return NULL;
        }
    }

    // Step 8: Fill FFT config
    fft_config->n_input = signal_length;
    fft_config->n_fft = is_factorable ? signal_length : max_padded_length;
    fft_config->sgn = transform_direction;
    fft_config->max_scratch_size = max_scratch_size;
    fft_config->lt = is_factorable ? 0 : 1;

    // Step 9: Factorize n_fft
    fft_config->lf = factors(fft_config->n_fft, fft_config->factors);

    // Step 10: Compute twiddle factors
    build_twiddles_linear(fft_config->twiddles, fft_config->n_fft);

    // Step 11: Populate twiddle_factors for pure-power FFTs (k-major layout)
    // ========================================================================
    // UPDATED: Handle radix-16 and radix-32
    // ========================================================================

    if (fft_config->twiddle_factors)
    {
        int offset = 0;

        // Determine radix
        int radix;
        if (is_power_of_32)
        {
            radix = 32;
        }
        else if (is_power_of_16)
        {
            radix = 16;
        }
        else if (is_power_of_2)
        {
            radix = 2;
        }
        else if (is_power_of_3)
        {
            radix = 3;
        }
        else if (is_power_of_5)
        {
            radix = 5;
        }
        else if (is_power_of_7)
        {
            radix = 7;
        }
        else if (is_power_of_11)
        {
            radix = 11;
        }
        else
        {
            radix = 13;
        }

        // Walk pure-power stages from largest to smallest
        // For N = radix^s, we have s stages: N, N/radix, N/radix^2, ..., radix
        for (int N_stage = signal_length; N_stage >= radix; N_stage /= radix)
        {
            const int sub_len = N_stage / radix; // Size of sub-FFTs at this stage

            // Twiddle stride in the main twiddle table
            // W_{N_stage}^p corresponds to W_{n_fft}^{p * stride}
            const int stride = fft_config->n_fft / N_stage;

            // For each sub-FFT index k, compute twiddles W^{j*k} for j=1..radix-1
            // Store in k-major order: [(radix-1)*k + (j-1)]
            for (int k = 0; k < sub_len; ++k)
            {
                const int base = (radix - 1) * k;

                for (int j = 1; j < radix; ++j)
                {
                    // Compute the phase: (j * k) mod N_stage
                    const int p = (j * k) % N_stage;

                    // Map to main twiddle table index
                    const int idxN = (p * stride) % fft_config->n_fft;

                    // Store twiddle
                    fft_config->twiddle_factors[offset + base + (j - 1)] =
                        fft_config->twiddles[idxN];
                }
            }

            // Move to next stage's offset
            offset += (radix - 1) * sub_len;
        }

        // Verify we used exactly the space we allocated
        if (offset != twiddle_factors_size)
        {
            fprintf(stderr, "Error: Twiddle offset mismatch: computed %d, expected %d\n",
                    offset, twiddle_factors_size);
            free_fft(fft_config);
            return NULL;
        }
    }

    // Step 12: Adjust twiddles for inverse FFT
    if (transform_direction == -1)
    {
        for (int i = 0; i < twiddle_count; i++)
        {
            fft_config->twiddles[i].im = -fft_config->twiddles[i].im;
        }
        if (fft_config->twiddle_factors)
        {
            for (int i = 0; i < twiddle_factors_size; i++)
            {
                fft_config->twiddle_factors[i].im = -fft_config->twiddle_factors[i].im;
            }
        }
    }

    // Step 13: Return configured FFT object
    return fft_config;
}

#ifdef HAS_AVX512
/**
 * @brief Complex multiply (AoS) for 4 packed complex values using AVX-512.
 *
 * Input layout: a = [ar0, ai0, ar1, ai1, ar2, ai2, ar3, ai3]
 *               b = [br0, bi0, br1, bi1, br2, bi2, br3, bi3]
 *
 * Output: [ar0*br0 - ai0*bi0, ar0*bi0 + ai0*br0, ar1*br1 - ai1*bi1, ...]
 *
 * @param a First complex vector (4 complex numbers).
 * @param b Second complex vector (4 complex numbers).
 * @return Complex product in AoS layout.
 */
static ALWAYS_INLINE __m512d cmul_avx512_aos(__m512d a, __m512d b)
{
    __m512d a_re = _mm512_movedup_pd(a);        // [ar, ar, ar, ar, ...]
    __m512d a_im = _mm512_movedup_pd(a);        // [ai, ai, ai, ai, ...]
    __m512d b_flip = _mm512_permute_pd(b, 0x55); // [bi, br, bi, br, ...]

    return _mm512_fmaddsub_pd(a_re, b, _mm512_mul_pd(a_im, b_flip));
}
/**
 * @brief Load 4 consecutive complex numbers (8 doubles) into AVX-512 register.
 */
static ALWAYS_INLINE __m512d load4_aos(const fft_data *p)
{
    return LOADU_PD512(&p->re);
}

/**
 * @brief Store 4 complex numbers from AVX-512 register.
 *
 * @param p Pointer to destination.
 * @param v AVX-512 register containing 4 complex values.
 */
static ALWAYS_INLINE void store4_aos(fft_data *p, __m512d v)
{
    STOREU_PD512(&p->re, v);
}
#endif // HAS_AVX512

/**
 * @brief Complex multiply (AoS) for two packed complex vectors using AVX.
 *
 * Multiplies two vectors of complex numbers stored in **Array-of-Structures (AoS)** layout.
 * The inputs pack two complex values each:
 *
 *   - @p a = [ ar0, ai0, ar1, ai1 ]
 *   - @p b = [ br0, bi0, br1, bi1 ]
 *
 * The result is returned in the same AoS lane order:
 *
 *   - return = [ ar0*br0 - ai0*bi0,  ar0*bi0 + ai0*br0,  ar1*br1 - ai1*bi1,  ar1*bi1 + ai1*br1 ]
 *
 * This implementation uses lane-wise shuffles and `_mm256_addsub_pd` to perform the
 * (real, imag) pair computations without horizontal reductions. No FMA is required.
 *
 * @param a Packed complex vector in AoS layout: [ ar0, ai0, ar1, ai1 ].
 * @param b Packed complex vector in AoS layout: [ br0, bi0, br1, bi1 ].
 * @return __m256d AoS-packed complex product:
 *         [ re0, im0, re1, im1 ] = [ ar0*br0 - ai0*bi0,  ar0*bi0 + ai0*br0,
 *                                     ar1*br1 - ai1*bi1,  ar1*bi1 + ai1*br1 ].
 *
 * @note Requires AVX support (256-bit double vectors). Layout must be AoS.
 * @warning If your data is in SoA layout (all reals then all imags), use a different kernel.
 * @see cmul_sse2_aos for a 128-bit (single complex) SSE2 variant.
 */
static ALWAYS_INLINE __m256d cmul_avx2_aos(__m256d a, __m256d b)
{
    // a = [ar0, ai0, ar1, ai1], b = [br0, bi0, br1, bi1]
    __m256d brbr_bibi = b;                              // [br0, bi0, br1, bi1]
    __m256d b_swapped = _mm256_permute_pd(b, 0b0101);   // [bi0, br0, bi1, br1]
    __m256d ar_ai = a;                                  // [ar0, ai0, ar1, ai1]
    __m256d ai_ar = _mm256_permute_pd(a, 0b0101);       // [ai0, ar0, ai1, ar1]
    __m256d real_im1 = _mm256_mul_pd(ar_ai, brbr_bibi); // [ar0*br0, ai0*bi0, ar1*br1, ai1*bi1]
    __m256d im_real2 = _mm256_mul_pd(ai_ar, b_swapped); // [ai0*bi0, ar0*br0, ai1*bi1, ar1*br1]
    return _mm256_addsub_pd(real_im1, im_real2);        // [ar0*br0 - ai0*bi0, ar0*bi0 + ai0*br0, ...]
}

/**
 * @brief Complex multiply (AoS) for one packed complex value using SSE2.
 *
 * Multiplies two complex numbers stored in **Array-of-Structures (AoS)** layout:
 *
 *   - @p a = [ ar, ai ]
 *   - @p b = [ br, bi ]
 *
 * The result is returned as:
 *
 *   - return = [ ar*br - ai*bi,  ar*bi + ai*br ]
 *
 * Implemented with shuffles and `_mm_addsub_pd` to avoid horizontal sums.
 *
 * @param a Packed complex value in AoS layout: [ ar, ai ].
 * @param b Packed complex value in AoS layout: [ br, bi ].
 * @return __m128d AoS-packed complex product: [ re, im ].
 *
 * @note Requires SSE2 support. Layout must be AoS.
 * @see cmul_avx2_aos for a 256-bit, two-complex AVX variant.
 */
static ALWAYS_INLINE __m128d cmul_sse2_aos(__m128d a, __m128d b)
{
    // a = [ar, ai], b = [br, bi]
    __m128d real_im1 = _mm_mul_pd(a, b);         // [ar*br, ai*bi]
    __m128d ai_ar = _mm_shuffle_pd(a, a, 0b01);  // [ai, ar]
    __m128d bi_br = _mm_shuffle_pd(b, b, 0b01);  // [bi, br]
    __m128d im_real2 = _mm_mul_pd(ai_ar, bi_br); // [ai*bi, ar*br]
    return _mm_addsub_pd(real_im1, im_real2);    // [ar*br - ai*bi, ar*bi + ai*br]
}

/**
 * @brief Load two consecutive complex samples (AoS) into one AVX register.
 *
 * Loads:
 *   *p_k  = { re(k),   im(k)   }
 *   *p_k1 = { re(k+1), im(k+1) }
 *
 * and returns a single __m256d containing:
 *   [ re(k), im(k), re(k+1), im(k+1) ].
 *
 * Uses unaligned loads for robustness; if your data is guaranteed 16-byte
 * aligned for each complex sample, you may switch to aligned loads.
 *
 * @param p_k  Pointer to complex sample k (AoS: struct { double re, im; }).
 * @param p_k1 Pointer to complex sample k+1.
 * @return __m256d Packed vector [ re(k), im(k), re(k+1), im(k+1) ].
 *
 * @note Safe for unaligned input. Requires AVX (uses 2×128b -> 256b insert).
 */
static inline __m256d load2_aos(const fft_data *p_k, const fft_data *p_k1)
{
    __m128d lo = _mm_loadu_pd(&p_k->re);  // [re(k), im(k)]
    __m128d hi = _mm_loadu_pd(&p_k1->re); // [re(k+1), im(k+1)]
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1);
}

/**
 * @brief Deinterleave 4 AoS complex numbers (8 doubles) into SoA form (4-wide).
 *
 * Converts four adjacent complex values stored as Array-of-Structures (AoS)
 *  - src[0] = { r0, i0 }
 *  - src[1] = { r1, i1 }
 *  - src[2] = { r2, i2 }
 *  - src[3] = { r3, i3 }
 * into separate Structure-of-Arrays (SoA) vectors:
 *  - re = [ r0, r1, r2, r3 ]
 *  - im = [ i0, i1, i2, i3 ]
 *
 * AVX2 lane layout:
 *   Each __m256d holds four doubles (lanes 0..3). The routine uses
 *   permute2f128 + unpacklo/hi to transpose from AoS → SoA efficiently.
 *
 * @param[in]  src  Pointer to at least 4 consecutive @c fft_data in AoS layout.
 * @param[out] re   Pointer to an array of at least 4 doubles to receive real parts.
 * @param[out] im   Pointer to an array of at least 4 doubles to receive imaginary parts.
 *
 * @note Loads/stores are unaligned-safe via @c LOADU_PD / @c STOREU_PD.
 * @warning No bounds checking is performed. @p src, @p re, and @p im must not alias
 *          in ways that violate strict aliasing.
 * @see interleave4_soa_to_aos()
 */
static ALWAYS_INLINE void deinterleave4_aos_to_soa(const fft_data *src, double *re, double *im)
{
    __m256d v0 = LOADU_PD(&src[0].re); // [r0,i0, r1,i1]
    __m256d v1 = LOADU_PD(&src[2].re); // [r2,i2, r3,i3]

    __m256d lohi0 = _mm256_permute2f128_pd(v0, v1, 0x20); // [r0,i0, r2,i2]
    __m256d lohi1 = _mm256_permute2f128_pd(v0, v1, 0x31); // [r1,i1, r3,i3]

    __m256d re4 = _mm256_unpacklo_pd(lohi0, lohi1); // [r0,r1, r2,r3]
    __m256d im4 = _mm256_unpackhi_pd(lohi0, lohi1); // [i0,i1, i2,i3]

    STOREU_PD(re, re4);
    STOREU_PD(im, im4);
}

/**
 * @brief Interleave SoA re[4], im[4] back into AoS complex (4 values).
 *
 * Inverse of deinterleave4_aos_to_soa(). Takes:
 *  - re = [ r0, r1, r2, r3 ]
 *  - im = [ i0, i1, i2, i3 ]
 * and writes four adjacent complex values in AoS layout:
 *  - dst[0] = { r0, i0 }
 *  - dst[1] = { r1, i1 }
 *  - dst[2] = { r2, i2 }
 *  - dst[3] = { r3, i3 }
 *
 * AVX2 lane layout:
 *   Packs re and im with unpacklo/hi, then permutes 128-bit halves to
 *   reconstruct the AoS order.
 *
 * @param[in]  re   Pointer to 4 real components.
 * @param[in]  im   Pointer to 4 imaginary components.
 * @param[out] dst  Pointer to at least four @c fft_data outputs (AoS).
 *
 * @note Stores are unaligned-safe via @c STOREU_PD.
 * @warning No bounds checking is performed.
 * @see deinterleave4_aos_to_soa()
 */
static ALWAYS_INLINE void interleave4_soa_to_aos(const double *re, const double *im, fft_data *dst)
{
    __m256d re4 = LOADU_PD(re); // [r0,r1, r2,r3]
    __m256d im4 = LOADU_PD(im); // [i0,i1, i2,i3]

    __m256d ri0 = _mm256_unpacklo_pd(re4, im4); // [r0,i0, r2,i2]
    __m256d ri1 = _mm256_unpackhi_pd(re4, im4); // [r1,i1, r3,i3]

    __m256d v0 = _mm256_permute2f128_pd(ri0, ri1, 0x20); // [r0,i0, r1,i1]
    __m256d v1 = _mm256_permute2f128_pd(ri0, ri1, 0x31); // [r2,i2, r3,i3]

    STOREU_PD(&dst[0].re, v0);
    STOREU_PD(&dst[2].re, v1);
}

/**
 * @brief Complex multiply (pairwise) in SoA for AVX (4-wide).
 *
 * Computes, lane-wise, for four complex numbers in parallel:
 *   (ar + i*ai) * (br + i*bi)  →  rr + i*ri
 *
 * Using the standard identities:
 *   rr = ar*br − ai*bi
 *   ri = ar*bi + ai*br
 *
 * @param[in]  ar  Real parts of left operand (4-wide @c __m256d).
 * @param[in]  ai  Imag parts of left operand (4-wide @c __m256d).
 * @param[in]  br  Real parts of right operand (4-wide @c __m256d).
 * @param[in]  bi  Imag parts of right operand (4-wide @c __m256d).
 * @param[out] rr  Real parts of the result (4-wide).
 * @param[out] ri  Imag parts of the result (4-wide).
 *
 * @note Inputs/outputs are SoA (separate real/imag vectors). This avoids
 *       shuffle overhead compared to AoS SIMD paths.
 * @warning Outputs @p rr and @p ri are fully overwritten.
 */
static ALWAYS_INLINE void cmul_soa_avx(__m256d ar, __m256d ai,
                                       __m256d br, __m256d bi,
                                       __m256d *rr, __m256d *ri)
{
    // rr = ar*br - ai*bi
    // ri = ar*bi + ai*br
    *rr = FMSUB(ar, br, _mm256_mul_pd(ai, bi));
    *ri = FMADD(ar, bi, _mm256_mul_pd(ai, br));
}

/**
 * @brief Deinterleave two AoS complex numbers into SoA form (2-wide).
 *
 * Converts two adjacent complex values stored as Array-of-Structures (AoS)
 *  - src[0] = { r0, i0 }
 *  - src[1] = { r1, i1 }
 * into separate Structure-of-Arrays (SoA) vectors:
 *  - re2 = [ r0, r1 ]
 *  - im2 = [ i0, i1 ]
 *
 * Layout / lanes:
 *   - Uses SSE2: each __m128d carries two doubles (lane 0 and lane 1).
 *
 * @param[in]  src  Pointer to at least two @c fft_data elements in AoS layout.
 * @param[out] re2  Pointer to an array of at least 2 doubles to receive real parts.
 * @param[out] im2  Pointer to an array of at least 2 doubles to receive imaginary parts.
 *
 * @note Loads are unaligned-safe (@c _mm_loadu_pd). @p src, @p re2, and @p im2
 *       must not alias in a way that violates strict aliasing.
 * @warning No bounds checking is performed.
 * @see interleave2_soa_to_aos()
 */
static inline void deinterleave2_aos_to_soa(const fft_data *src, double *re2, double *im2)
{
    __m128d v = _mm_loadu_pd(&src[0].re); // [r0,i0]
    __m128d w = _mm_loadu_pd(&src[1].re); // [r1,i1]
    __m128d re = _mm_unpacklo_pd(v, w);   // [r0,r1]
    __m128d im = _mm_unpackhi_pd(v, w);   // [i0,i1]
    _mm_storeu_pd(re2, re);
    _mm_storeu_pd(im2, im);
}

/**
 * @brief Interleave SoA (2-wide) back to AoS complex numbers.
 *
 * Inverse of deinterleave2_aos_to_soa(). Takes:
 *  - re2 = [ r0, r1 ]
 *  - im2 = [ i0, i1 ]
 * and writes two adjacent complex values in AoS layout:
 *  - dst[0] = { r0, i0 }
 *  - dst[1] = { r1, i1 }
 *
 * @param[in]  re2  Pointer to 2 real components.
 * @param[in]  im2  Pointer to 2 imag components.
 * @param[out] dst  Pointer to at least two @c fft_data outputs (AoS).
 *
 * @note Stores are unaligned-safe (@c _mm_storeu_pd).
 * @warning No bounds checking is performed.
 * @see deinterleave2_aos_to_soa()
 */
static inline void interleave2_soa_to_aos(const double *re2, const double *im2, fft_data *dst)
{
    __m128d re = _mm_loadu_pd(re2);        // [r0,r1]
    __m128d im = _mm_loadu_pd(im2);        // [i0,i1]
    __m128d ri0 = _mm_unpacklo_pd(re, im); // [r0,i0]
    __m128d ri1 = _mm_unpackhi_pd(re, im); // [r1,i1]
    _mm_storeu_pd(&dst[0].re, ri0);
    _mm_storeu_pd(&dst[1].re, ri1);
}

/**
 * @brief 90° complex rotation (±i) in SoA for AVX (4-wide).
 *
 * Computes, lane-wise:
 *   out = (sign) * i * (re + i*im)
 *
 * Which is equivalent to:
 *   if sign == +1:  (out_re, out_im) = (-im,  re)   // multiply by +i
 *   if sign == -1:  (out_re, out_im) = ( im, -re)   // multiply by -i
 *
 * @param[in]  re       Real parts (4-wide, @c __m256d).
 * @param[in]  im       Imag parts (4-wide, @c __m256d).
 * @param[in]  sign     Direction: +1 for +i, -1 for -i. (Other values are undefined.)
 * @param[out] out_re   Real parts of rotated result (4-wide).
 * @param[out] out_im   Imag parts of rotated result (4-wide).
 *
 * @note This is branchy on @p sign to avoid extra shuffles. If you call with
 *       a compile-time constant sign, compilers typically fold this nicely.
 * @warning Requires AVX. The inputs/outputs are SoA (separate real/im vectors).
 */
static ALWAYS_INLINE void rot90_soa_avx(__m256d re, __m256d im, int sign,
                                        __m256d *out_re, __m256d *out_im)
{
    if (sign == 1)
    {
        *out_re = _mm256_sub_pd(_mm256_setzero_pd(), im); // -im
        *out_im = re;
    }
    else
    {
        *out_re = im;                                     // +im
        *out_im = _mm256_sub_pd(_mm256_setzero_pd(), re); // -re
    }
}

//==============================================================================
// RADIX-32 HELPER FUNCTIONS (place before mixed_radix_dit_rec)
//==============================================================================

#ifdef __AVX2__
/**
 * @brief Rotate two AoS-packed complex numbers by ±i using AVX2.
 *
 * Each __m256d contains [re0, im0, re1, im1].
 * Performs a 90° rotation in the complex plane.
 *
 * @param v      Two complex numbers packed as AoS in __m256d.
 * @param sign   +1 → multiply by +i; -1 → multiply by -i.
 * @return       Rotated complex pair.
 */
static ALWAYS_INLINE __m256d rot90_aos_avx2(__m256d v, int sign)
{
    __m256d swp = _mm256_permute_pd(v, 0b0101); // swap re↔im per complex
    if (sign == 1)
    {
        const __m256d m = _mm256_set_pd(0.0, -0.0, 0.0, -0.0); // negate im lanes
        return _mm256_xor_pd(swp, m);                          // +i rotation
    }
    else
    {
        const __m256d m = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); // negate re lanes
        return _mm256_xor_pd(swp, m);                          // -i rotation
    }
}

/**
 * @brief 4-point DIT FFT butterfly (AoS, two complex values per __m256d).
 *
 * Implements:
 *   y0 = a + b + c + d
 *   y1 = (a - c) + (-i)*(b - d)
 *   y2 = a + b - c - d
 *   y3 = (a - c) + (+i)*(b - d)
 *
 * @param a,b,c,d        Input/output vectors (in-place).
 * @param transform_sign +1 for forward FFT, -1 for inverse FFT.
 */
static ALWAYS_INLINE void radix4_butterfly_aos(__m256d *a, __m256d *b,
                                               __m256d *c, __m256d *d,
                                               int transform_sign)
{
    __m256d A = *a, B = *b, C = *c, D = *d;

    __m256d S0 = _mm256_add_pd(A, C);
    __m256d D0 = _mm256_sub_pd(A, C);
    __m256d S1 = _mm256_add_pd(B, D);
    __m256d D1 = _mm256_sub_pd(B, D);

    __m256d Y0 = _mm256_add_pd(S0, S1);
    __m256d Y2 = _mm256_sub_pd(S0, S1);

    // Use opposite signs for Y1 and Y3 according to transform direction
    __m256d rot_for_y1 = rot90_aos_avx2(D1, -transform_sign);
    __m256d rot_for_y3 = rot90_aos_avx2(D1, transform_sign);

    __m256d Y1 = _mm256_add_pd(D0, rot_for_y1);
    __m256d Y3 = _mm256_add_pd(D0, rot_for_y3);

    *a = Y0;
    *b = Y1;
    *c = Y2;
    *d = Y3;
}

/**
 * @brief 2-point DIT FFT butterfly (AoS, two complex per __m256d).
 *
 * Computes:
 *   y0 = a + b
 *   y1 = a - b
 *
 * @param a,b Input/output vectors (in-place).
 */
static ALWAYS_INLINE void radix2_butterfly_aos(__m256d *a, __m256d *b)
{
    __m256d A = *a, B = *b;
    *a = _mm256_add_pd(A, B);
    *b = _mm256_sub_pd(A, B);
}
#endif // __AVX2__

//==============================================================================
// Scalar fallbacks (C99-compatible)
//==============================================================================

/**
 * @brief Rotate a single complex number by ±i.
 *
 * @param re,im   Real and imaginary parts of input.
 * @param sign    +1 → multiply by +i, -1 → multiply by -i.
 * @param or_,oi  Output pointers for rotated real/imag parts.
 */
static inline void rot90_scalar(double re, double im, int sign,
                                double *or_, double *oi)
{
    if (sign == 1)
    {
        *or_ = -im;
        *oi = re;
    } // +i
    else
    {
        *or_ = im;
        *oi = -re;
    } // -i
}

/**
 * @brief 2-point scalar DIT butterfly.
 *
 * @param a,b Complex inputs/outputs (in-place).
 */
static inline void r2_butterfly(fft_data *a, fft_data *b)
{
    double tr = a->re + b->re, ti = a->im + b->im;
    double ur = a->re - b->re, ui = a->im - b->im;
    a->re = tr;
    a->im = ti;
    b->re = ur;
    b->im = ui;
}

/**
 * @brief 4-point scalar DIT FFT butterfly.
 *
 * Implements the same pattern as radix4_butterfly_aos.
 *
 * @param a,b,c,d        Complex inputs/outputs (in-place).
 * @param transform_sign +1 for forward FFT, -1 for inverse FFT.
 */
static inline void r4_butterfly(fft_data *a, fft_data *b,
                                fft_data *c, fft_data *d,
                                int transform_sign)
{
    double S0r = a->re + c->re, S0i = a->im + c->im;
    double D0r = a->re - c->re, D0i = a->im - c->im;
    double S1r = b->re + d->re, S1i = b->im + d->im;
    double D1r = b->re - d->re, D1i = b->im - d->im;

    fft_data y0 = {S0r + S1r, S0i + S1i};
    fft_data y2 = {S0r - S1r, S0i - S1i};

    double rposr, rposi, rnegr, rnegi;
    rot90_scalar(D1r, D1i, transform_sign, &rposr, &rposi);
    rot90_scalar(D1r, D1i, -transform_sign, &rnegr, &rnegi);

    fft_data y1 = {D0r + rnegr, D0i + rnegi};
    fft_data y3 = {D0r + rposr, D0i + rposi};

    *a = y0;
    *b = y1;
    *c = y2;
    *d = y3;
}

/**
 * @brief Performs recursive mixed-radix decimation-in-time (DIT) FFT on the input data.
 *
 * Computes the Fast Fourier Transform (FFT) using a recursive mixed-radix DIT approach,
 * supporting specific radices (2, 3, 4, 5, 7, 8) based on the prime factorization of the signal length N.
 * This method decomposes the FFT into smaller subproblems, applying butterfly operations for each radix,
 * and uses precomputed twiddle factors to efficiently combine results.
 *
 * @param[out] output_buffer Output buffer for FFT results (length N).
 *                          Stores the transformed complex values after applying the FFT.
 * @param[in] input_buffer Input signal data (length N).
 *                        The real and imaginary components of the input signal to be transformed.
 * @param[in] fft_obj FFT configuration object.
 *                   Contains the signal length, transform direction, prime factors, and precomputed twiddle factors.
 * @param[in] transform_sign Direction of the transform (+1 for forward, -1 for inverse).
 *                         Determines whether to perform a forward FFT (e^(-2πi k/N)) or inverse FFT (e^(+2πi k/N)).
 * @param[in] data_length Length of the data (N > 0).
 *                       The size of the input and output buffers, which must be positive.
 * @param[in] stride Stride length for accessing input data (l > 0).
 *                  The step size used to access elements in the input buffer, ensuring proper indexing for recursive decomposition.
 * @param[in] factor_index Index into the factors array for current radix (inc >= 0).
 *                        Indicates the current prime factor being processed from fft_obj->factors, used to determine the radix for this recursion level.
 * @warning If memory allocation fails, lengths are invalid (data_length <= 0, stride <= 0, or factor_index < 0),
 *          or radices are unsupported (radix > 8 and not handled by general radix decomposition), the function exits with an error.
 * @note Uses precomputed twiddle factors from the FFT object, which are complex exponentials of the form e^(-2πi k/N) or e^(+2πi k/N)
 *       depending on the transform_sign. The algorithm recursively divides the problem into smaller sub-FFTs,
 *       applies radix-specific butterfly operations (e.g., Radix-2, Radix-3), and combines results using twiddle factors.
 *       Mathematically, the FFT computes \(X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-2\pi i k n / N}\) for forward transforms,
 *       leveraging the divide-and-conquer strategy to reduce complexity from O(N^2) to O(N log N).
 */
static void mixed_radix_dit_rec(
    fft_data *output_buffer,
    fft_data *input_buffer,
    const fft_object fft_obj,
    int transform_sign,
    int data_length,
    int stride,
    int factor_index,
    int scratch_offset)
{
    //==========================================================================
    // 0) VALIDATION
    //==========================================================================
    if (data_length <= 0 || stride <= 0 || factor_index < 0)
    {
        fprintf(stderr,
                "Error: Invalid params - N=%d, stride=%d, factor_idx=%d\n",
                data_length, stride, factor_index);
        return;
    }

    //==========================================================================
    // 1) BASE CASE: length-1 copy
    //==========================================================================
    if (data_length == 1)
    {
        output_buffer[0] = input_buffer[0];
        return;
    }

    //==========================================================================
    // 2) CURRENT STAGE GEOMETRY
    //==========================================================================
    const int radix = fft_obj->factors[factor_index];
    const int sub_len = data_length / radix; // child FFT size
    const int next_stride = stride * radix;  // stride for children

    //==========================================================================
    // 3) SCRATCH PLANNING FOR THIS STAGE
    //
    // Layout in scratch[scratch_offset..]:
    //   - sub_outputs[radix * sub_len]: child FFT outputs (lane-major)
    //   - stage_tw[(radix-1) * sub_len]: twiddles if not precomputed (k-major)
    //
    // Child scratch starts AFTER this stage's frame (serial recursion).
    //==========================================================================
    fft_data *sub_outputs = fft_obj->scratch + scratch_offset;

    const int stage_outputs_size = radix * sub_len;
    const int stage_tw_size = (radix - 1) * sub_len;

    int need_this_stage = stage_outputs_size;
    int twiddle_in_scratch = 0;
    fft_data *stage_tw = NULL;

    // Check if we have precomputed twiddles for this stage
    if (fft_obj->twiddle_factors != NULL &&
        factor_index < fft_obj->num_precomputed_stages)
    {
        // Use precomputed table (k-major layout)
        stage_tw = fft_obj->twiddle_factors +
                   fft_obj->stage_twiddle_offset[factor_index];
    }
    else
    {
        // Generate twiddles dynamically into scratch
        twiddle_in_scratch = 1;
        need_this_stage += stage_tw_size;

        if (scratch_offset + need_this_stage > fft_obj->max_scratch_size)
        {
            fprintf(stderr,
                    "Error: Scratch overflow at radix=%d, N=%d. "
                    "Need %d, have %d (offset=%d)\n",
                    radix, data_length, need_this_stage,
                    fft_obj->max_scratch_size - scratch_offset, scratch_offset);
            return;
        }

        stage_tw = sub_outputs + stage_outputs_size;
    }

    //==========================================================================
    // 4) COMPUTE CHILD SCRATCH REQUIREMENTS (for parallel/optimized layout)
    //
    // Calculate maximum scratch needed by any child FFT
    //==========================================================================
    int child_scratch_per_fft = 0;

    if (sub_len > 1) // Children need recursion
    {
        // Estimate child scratch needs based on their factorization
        int child_factors[64];
        int child_num_factors = factors(sub_len, child_factors);

        // Pessimistic estimate: assume each child stage needs radix * (size/radix)
        int temp_size = sub_len;
        for (int i = 0; i < child_num_factors; ++i)
        {
            int child_radix = child_factors[i];
            int child_stage_need = child_radix * (temp_size / child_radix);

            // Also account for twiddles if not precomputed
            // For simplicity, assume worst case (no precomputation)
            child_stage_need += (child_radix - 1) * (temp_size / child_radix);

            if (child_stage_need > child_scratch_per_fft)
                child_scratch_per_fft = child_stage_need;

            temp_size /= child_radix;
        }
    }

    //==========================================================================
    // 5) ALLOCATE CHILD SCRATCH REGIONS (non-overlapping for better cache use)
    //
    // Two strategies:
    // A) Serial (original): All children share same scratch region
    // B) Parallel-ready: Each child gets dedicated scratch region
    //
    // We use strategy A for now (serial) but with better layout planning
    //==========================================================================

    // Child scratch starts AFTER this stage's data
    const int child_scratch_base = scratch_offset + need_this_stage;

    // Verify we have enough space for child recursion
    if (child_scratch_base + child_scratch_per_fft > fft_obj->max_scratch_size)
    {
        fprintf(stderr,
                "Error: Insufficient scratch for child FFTs at radix=%d, N=%d\n"
                "This stage needs %d, child needs %d, total available %d\n",
                radix, data_length, need_this_stage, child_scratch_per_fft,
                fft_obj->max_scratch_size);
        return;
    }

    //==========================================================================
    // 6) RECURSE INTO RADIX CHILDREN (serial execution, shared child scratch)
    //
    // All children share child_scratch_base since they execute serially
    //==========================================================================
    for (int i = 0; i < radix; ++i)
    {
        mixed_radix_dit_rec(
            sub_outputs + i * sub_len, // lane i destination
            input_buffer + i * stride, // lane i source (strided)
            fft_obj,
            transform_sign,
            sub_len,
            next_stride,
            factor_index + 1,
            child_scratch_base // ← All children share this offset
        );
    }

    //==========================================================================
    // 7) PREPARE TWIDDLES IF NOT PRECOMPUTED
    //==========================================================================
    if (twiddle_in_scratch)
    {
        const int nfft = fft_obj->n_fft;
        const int step = nfft / data_length;

        for (int k = 0; k < sub_len; ++k)
        {
            const int base = (radix - 1) * k;
            for (int j = 1; j < radix; ++j)
            {
                const int p = (j * k) % data_length;
                const int idxN = (p * step) % nfft;
                stage_tw[base + (j - 1)] = fft_obj->twiddles[idxN];
            }
        }
    }

    //==========================================================================
    // 8) RADIX DISPATCH
    //==========================================================================
    if (radix == 2)
    {
        const int half = sub_len;
        int k = 0;
        const int trivial_end = (half + 1) / 2; // First ~half of twiddles

#ifdef HAS_AVX512
        //======================================================================
        // AVX-512 PATH: 16x unrolling (process 16 butterflies at once)
        //======================================================================

        // Trivial twiddles (W^0 = 1): No complex multiply needed
        for (; k + 15 < trivial_end; k += 16)
        {
            // Prefetch ahead
            if (k + 32 < trivial_end)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 32].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 32 + half].re, _MM_HINT_T0);
            }

            // Load 16 even-indexed complex samples (4 loads × 4 complex each)
            __m512d e0 = load4_aos(&sub_outputs[k + 0]);
            __m512d e1 = load4_aos(&sub_outputs[k + 4]);
            __m512d e2 = load4_aos(&sub_outputs[k + 8]);
            __m512d e3 = load4_aos(&sub_outputs[k + 12]);

            // Load 16 odd samples
            __m512d o0 = load4_aos(&sub_outputs[k + 0 + half]);
            __m512d o1 = load4_aos(&sub_outputs[k + 4 + half]);
            __m512d o2 = load4_aos(&sub_outputs[k + 8 + half]);
            __m512d o3 = load4_aos(&sub_outputs[k + 12 + half]);

            // Radix-2 butterfly: X[k] = E[k] + O[k], X[k+N/2] = E[k] - O[k]
            __m512d x00 = _mm512_add_pd(e0, o0);
            __m512d x10 = _mm512_sub_pd(e0, o0);
            __m512d x01 = _mm512_add_pd(e1, o1);
            __m512d x11 = _mm512_sub_pd(e1, o1);
            __m512d x02 = _mm512_add_pd(e2, o2);
            __m512d x12 = _mm512_sub_pd(e2, o2);
            __m512d x03 = _mm512_add_pd(e3, o3);
            __m512d x13 = _mm512_sub_pd(e3, o3);

            // Store results
            STOREU_PD512(&output_buffer[k + 0].re, x00);
            STOREU_PD512(&output_buffer[k + 4].re, x01);
            STOREU_PD512(&output_buffer[k + 8].re, x02);
            STOREU_PD512(&output_buffer[k + 12].re, x03);
            STOREU_PD512(&output_buffer[k + 0 + half].re, x10);
            STOREU_PD512(&output_buffer[k + 4 + half].re, x11);
            STOREU_PD512(&output_buffer[k + 8 + half].re, x12);
            STOREU_PD512(&output_buffer[k + 12 + half].re, x13);
        }

        // Non-trivial twiddles: Need complex multiply (W^k * O[k])
        for (; k + 15 < half; k += 16)
        {
            if (k + 32 < half)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 32].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 32 + half].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[k + 32].re, _MM_HINT_T0);
            }

            // Load even/odd samples
            __m512d e0 = load4_aos(&sub_outputs[k + 0]);
            __m512d e1 = load4_aos(&sub_outputs[k + 4]);
            __m512d e2 = load4_aos(&sub_outputs[k + 8]);
            __m512d e3 = load4_aos(&sub_outputs[k + 12]);

            __m512d o0 = load4_aos(&sub_outputs[k + 0 + half]);
            __m512d o1 = load4_aos(&sub_outputs[k + 4 + half]);
            __m512d o2 = load4_aos(&sub_outputs[k + 8 + half]);
            __m512d o3 = load4_aos(&sub_outputs[k + 12 + half]);

            // Load twiddles
            __m512d w0 = load4_aos(&stage_tw[k + 0]);
            __m512d w1 = load4_aos(&stage_tw[k + 4]);
            __m512d w2 = load4_aos(&stage_tw[k + 8]);
            __m512d w3 = load4_aos(&stage_tw[k + 12]);

            // Twiddle multiply
            __m512d tw0 = cmul_avx512_aos(o0, w0);
            __m512d tw1 = cmul_avx512_aos(o1, w1);
            __m512d tw2 = cmul_avx512_aos(o2, w2);
            __m512d tw3 = cmul_avx512_aos(o3, w3);

            // Butterfly
            __m512d x00 = _mm512_add_pd(e0, tw0);
            __m512d x10 = _mm512_sub_pd(e0, tw0);
            __m512d x01 = _mm512_add_pd(e1, tw1);
            __m512d x11 = _mm512_sub_pd(e1, tw1);
            __m512d x02 = _mm512_add_pd(e2, tw2);
            __m512d x12 = _mm512_sub_pd(e2, tw2);
            __m512d x03 = _mm512_add_pd(e3, tw3);
            __m512d x13 = _mm512_sub_pd(e3, tw3);

            // Store
            STOREU_PD512(&output_buffer[k + 0].re, x00);
            STOREU_PD512(&output_buffer[k + 4].re, x01);
            STOREU_PD512(&output_buffer[k + 8].re, x02);
            STOREU_PD512(&output_buffer[k + 12].re, x03);
            STOREU_PD512(&output_buffer[k + 0 + half].re, x10);
            STOREU_PD512(&output_buffer[k + 4 + half].re, x11);
            STOREU_PD512(&output_buffer[k + 8 + half].re, x12);
            STOREU_PD512(&output_buffer[k + 12 + half].re, x13);
        }

        // Fall through to AVX2 cleanup for remaining elements
#endif // HAS_AVX512

#ifdef __AVX2__
        //======================================================================
        // OPTIMIZATION 1: First half has trivial twiddles (W^0 = 1)
        // No complex multiply needed! Saves ~50% of work for radix-2.
        //======================================================================

        // Process trivial twiddles with 4x unrolling
        for (; k + 7 < trivial_end; k += 8)
        {
            // Prefetch ahead
            if (k + 16 < trivial_end)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + half].re, _MM_HINT_T0);
            }

            // Load 8 even pairs (4 AVX2 loads)
            __m256d e0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
            __m256d e1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
            __m256d e2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
            __m256d e3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

            // Load 8 odd pairs (4 AVX2 loads)
            __m256d o0 = load2_aos(&sub_outputs[k + 0 + half], &sub_outputs[k + 1 + half]);
            __m256d o1 = load2_aos(&sub_outputs[k + 2 + half], &sub_outputs[k + 3 + half]);
            __m256d o2 = load2_aos(&sub_outputs[k + 4 + half], &sub_outputs[k + 5 + half]);
            __m256d o3 = load2_aos(&sub_outputs[k + 6 + half], &sub_outputs[k + 7 + half]);

            // Butterfly (no twiddle multiply!)
            __m256d x00 = _mm256_add_pd(e0, o0);
            __m256d x10 = _mm256_sub_pd(e0, o0);
            __m256d x01 = _mm256_add_pd(e1, o1);
            __m256d x11 = _mm256_sub_pd(e1, o1);
            __m256d x02 = _mm256_add_pd(e2, o2);
            __m256d x12 = _mm256_sub_pd(e2, o2);
            __m256d x03 = _mm256_add_pd(e3, o3);
            __m256d x13 = _mm256_sub_pd(e3, o3);

            // Store results
            STOREU_PD(&output_buffer[k + 0].re, x00);
            STOREU_PD(&output_buffer[k + 2].re, x01);
            STOREU_PD(&output_buffer[k + 4].re, x02);
            STOREU_PD(&output_buffer[k + 6].re, x03);
            STOREU_PD(&output_buffer[k + 0 + half].re, x10);
            STOREU_PD(&output_buffer[k + 2 + half].re, x11);
            STOREU_PD(&output_buffer[k + 4 + half].re, x12);
            STOREU_PD(&output_buffer[k + 6 + half].re, x13);
        }

        // Cleanup: 2x unrolling for remaining trivial twiddles
        for (; k + 1 < trivial_end; k += 2)
        {
            __m256d even = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
            __m256d odd = load2_aos(&sub_outputs[k + half], &sub_outputs[k + half + 1]);

            __m256d x0 = _mm256_add_pd(even, odd);
            __m256d x1 = _mm256_sub_pd(even, odd);

            STOREU_PD(&output_buffer[k].re, x0);
            STOREU_PD(&output_buffer[k + half].re, x1);
        }

        //======================================================================
        // OPTIMIZATION 2: Second half needs twiddle multiplies (4x unrolled)
        //======================================================================
        for (; k + 7 < half; k += 8)
        {
            // Prefetch ahead
            if (k + 16 < half)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + half].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[k + 16].re, _MM_HINT_T0);
            }

            // Load 8 even pairs
            __m256d e0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
            __m256d e1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
            __m256d e2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
            __m256d e3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

            // Load 8 odd pairs
            __m256d o0 = load2_aos(&sub_outputs[k + 0 + half], &sub_outputs[k + 1 + half]);
            __m256d o1 = load2_aos(&sub_outputs[k + 2 + half], &sub_outputs[k + 3 + half]);
            __m256d o2 = load2_aos(&sub_outputs[k + 4 + half], &sub_outputs[k + 5 + half]);
            __m256d o3 = load2_aos(&sub_outputs[k + 6 + half], &sub_outputs[k + 7 + half]);

            // Load 8 twiddles
            __m256d w0 = load2_aos(&stage_tw[k + 0], &stage_tw[k + 1]);
            __m256d w1 = load2_aos(&stage_tw[k + 2], &stage_tw[k + 3]);
            __m256d w2 = load2_aos(&stage_tw[k + 4], &stage_tw[k + 5]);
            __m256d w3 = load2_aos(&stage_tw[k + 6], &stage_tw[k + 7]);

            // Twiddle multiply
            __m256d tw0 = cmul_avx2_aos(o0, w0);
            __m256d tw1 = cmul_avx2_aos(o1, w1);
            __m256d tw2 = cmul_avx2_aos(o2, w2);
            __m256d tw3 = cmul_avx2_aos(o3, w3);

            // Butterfly
            __m256d x00 = _mm256_add_pd(e0, tw0);
            __m256d x10 = _mm256_sub_pd(e0, tw0);
            __m256d x01 = _mm256_add_pd(e1, tw1);
            __m256d x11 = _mm256_sub_pd(e1, tw1);
            __m256d x02 = _mm256_add_pd(e2, tw2);
            __m256d x12 = _mm256_sub_pd(e2, tw2);
            __m256d x03 = _mm256_add_pd(e3, tw3);
            __m256d x13 = _mm256_sub_pd(e3, tw3);

            // Store results
            STOREU_PD(&output_buffer[k + 0].re, x00);
            STOREU_PD(&output_buffer[k + 2].re, x01);
            STOREU_PD(&output_buffer[k + 4].re, x02);
            STOREU_PD(&output_buffer[k + 6].re, x03);
            STOREU_PD(&output_buffer[k + 0 + half].re, x10);
            STOREU_PD(&output_buffer[k + 2 + half].re, x11);
            STOREU_PD(&output_buffer[k + 4 + half].re, x12);
            STOREU_PD(&output_buffer[k + 6 + half].re, x13);
        }

        // Cleanup: 2x unrolling for remaining twiddle multiplies
        for (; k + 1 < half; k += 2)
        {
            if (k + 8 < half)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 8 + half].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[k + 8].re, _MM_HINT_T0);
            }

            __m256d even = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
            __m256d odd = load2_aos(&sub_outputs[k + half], &sub_outputs[k + half + 1]);
            __m256d w = load2_aos(&stage_tw[k], &stage_tw[k + 1]);

            __m256d tw = cmul_avx2_aos(odd, w);

            __m256d x0 = _mm256_add_pd(even, tw);
            __m256d x1 = _mm256_sub_pd(even, tw);

            STOREU_PD(&output_buffer[k].re, x0);
            STOREU_PD(&output_buffer[k + half].re, x1);
        }
#endif // __AVX2__

        //======================================================================
        // SSE2 TAIL: Handle remaining 0..1 complex numbers
        //======================================================================
        for (; k < half; ++k)
        {
            __m128d even = LOADU_SSE2(&sub_outputs[k].re);
            __m128d odd = LOADU_SSE2(&sub_outputs[k + half].re);

            if (k < trivial_end)
            {
                // Trivial twiddle (W^0 = 1)
                STOREU_SSE2(&output_buffer[k].re, _mm_add_pd(even, odd));
                STOREU_SSE2(&output_buffer[k + half].re, _mm_sub_pd(even, odd));
            }
            else
            {
                // Non-trivial twiddle
                __m128d w = LOADU_SSE2(&stage_tw[k].re);
                __m128d tw = cmul_sse2_aos(odd, w);
                STOREU_SSE2(&output_buffer[k].re, _mm_add_pd(even, tw));
                STOREU_SSE2(&output_buffer[k + half].re, _mm_sub_pd(even, tw));
            }
        }
    }
    else if (radix == 3)
    {
        //======================================================================
        // RADIX-3 BUTTERFLY (DIT) - FFTW-STYLE OPTIMIZED
        //
        // Stays in AoS throughout, uses 2x or 4x unrolling for better ILP.
        // Eliminates all AoS ↔ SoA conversions.
        //======================================================================

        const int third = sub_len;
        int k = 0;

#ifdef __AVX2__
        //------------------------------------------------------------------
        // AVX2 PATH: 4x unrolled, pure AoS, no conversions
        //------------------------------------------------------------------
        const __m256d vhalf = _mm256_set1_pd(0.5);
        const __m256d vsqrt3_2 = _mm256_set1_pd(C3_SQRT3BY2);

        // Precompute rotation masks for ±i multiplication
        const __m256d mask_plus_i = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);  // for +i
        const __m256d mask_minus_i = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); // for -i
        const __m256d rot_mask = (transform_sign == 1) ? mask_plus_i : mask_minus_i;

        for (; k + 7 < third; k += 8)
        {
            // Prefetch ahead
            if (k + 16 < third)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + third].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 2 * third].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[2 * (k + 16)].re, _MM_HINT_T0);
            }

            //==================================================================
            // Load inputs (8 butterflies = 4 AVX2 loads per lane)
            //==================================================================
            __m256d a0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
            __m256d a1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
            __m256d a2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
            __m256d a3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

            __m256d b0 = load2_aos(&sub_outputs[k + 0 + third], &sub_outputs[k + 1 + third]);
            __m256d b1 = load2_aos(&sub_outputs[k + 2 + third], &sub_outputs[k + 3 + third]);
            __m256d b2 = load2_aos(&sub_outputs[k + 4 + third], &sub_outputs[k + 5 + third]);
            __m256d b3 = load2_aos(&sub_outputs[k + 6 + third], &sub_outputs[k + 7 + third]);

            __m256d c0 = load2_aos(&sub_outputs[k + 0 + 2 * third], &sub_outputs[k + 1 + 2 * third]);
            __m256d c1 = load2_aos(&sub_outputs[k + 2 + 2 * third], &sub_outputs[k + 3 + 2 * third]);
            __m256d c2 = load2_aos(&sub_outputs[k + 4 + 2 * third], &sub_outputs[k + 5 + 2 * third]);
            __m256d c3 = load2_aos(&sub_outputs[k + 6 + 2 * third], &sub_outputs[k + 7 + 2 * third]);

            //==================================================================
            // Load twiddles W^k and W^{2k} (k-major: interleaved pairs)
            //==================================================================
            __m256d w1_0 = load2_aos(&stage_tw[2 * (k + 0)], &stage_tw[2 * (k + 1)]);
            __m256d w1_1 = load2_aos(&stage_tw[2 * (k + 2)], &stage_tw[2 * (k + 3)]);
            __m256d w1_2 = load2_aos(&stage_tw[2 * (k + 4)], &stage_tw[2 * (k + 5)]);
            __m256d w1_3 = load2_aos(&stage_tw[2 * (k + 6)], &stage_tw[2 * (k + 7)]);

            __m256d w2_0 = load2_aos(&stage_tw[2 * (k + 0) + 1], &stage_tw[2 * (k + 1) + 1]);
            __m256d w2_1 = load2_aos(&stage_tw[2 * (k + 2) + 1], &stage_tw[2 * (k + 3) + 1]);
            __m256d w2_2 = load2_aos(&stage_tw[2 * (k + 4) + 1], &stage_tw[2 * (k + 5) + 1]);
            __m256d w2_3 = load2_aos(&stage_tw[2 * (k + 6) + 1], &stage_tw[2 * (k + 7) + 1]);

            //==================================================================
            // Twiddle multiply: b2 = b * W^k, c2 = c * W^{2k}
            //==================================================================
            __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
            __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
            __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
            __m256d b2_3 = cmul_avx2_aos(b3, w1_3);

            __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
            __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
            __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
            __m256d c2_3 = cmul_avx2_aos(c3, w2_3);

//==================================================================
// Radix-3 butterfly computation (8 butterflies in parallel)
//==================================================================
#define RADIX3_BUTTERFLY_AVX2(a, b2, c2, y0, y1, y2)                \
    {                                                               \
        __m256d sum = _mm256_add_pd(b2, c2);                        \
        __m256d dif = _mm256_sub_pd(b2, c2);                        \
        y0 = _mm256_add_pd(a, sum);                                 \
        __m256d temp = _mm256_sub_pd(a, _mm256_mul_pd(sum, vhalf)); \
        __m256d dif_swp = _mm256_permute_pd(dif, 0b0101);           \
        __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask);           \
        __m256d rot = _mm256_mul_pd(rot90, vsqrt3_2);               \
        y1 = _mm256_add_pd(temp, rot);                              \
        y2 = _mm256_sub_pd(temp, rot);                              \
    }

            __m256d y0_0, y1_0, y2_0;
            __m256d y0_1, y1_1, y2_1;
            __m256d y0_2, y1_2, y2_2;
            __m256d y0_3, y1_3, y2_3;

            RADIX3_BUTTERFLY_AVX2(a0, b2_0, c2_0, y0_0, y1_0, y2_0);
            RADIX3_BUTTERFLY_AVX2(a1, b2_1, c2_1, y0_1, y1_1, y2_1);
            RADIX3_BUTTERFLY_AVX2(a2, b2_2, c2_2, y0_2, y1_2, y2_2);
            RADIX3_BUTTERFLY_AVX2(a3, b2_3, c2_3, y0_3, y1_3, y2_3);

#undef RADIX3_BUTTERFLY_AVX2

            //==================================================================
            // Store results (pure AoS, no conversions!)
            //==================================================================
            STOREU_PD(&output_buffer[k + 0].re, y0_0);
            STOREU_PD(&output_buffer[k + 2].re, y0_1);
            STOREU_PD(&output_buffer[k + 4].re, y0_2);
            STOREU_PD(&output_buffer[k + 6].re, y0_3);

            STOREU_PD(&output_buffer[k + 0 + third].re, y1_0);
            STOREU_PD(&output_buffer[k + 2 + third].re, y1_1);
            STOREU_PD(&output_buffer[k + 4 + third].re, y1_2);
            STOREU_PD(&output_buffer[k + 6 + third].re, y1_3);

            STOREU_PD(&output_buffer[k + 0 + 2 * third].re, y2_0);
            STOREU_PD(&output_buffer[k + 2 + 2 * third].re, y2_1);
            STOREU_PD(&output_buffer[k + 4 + 2 * third].re, y2_2);
            STOREU_PD(&output_buffer[k + 6 + 2 * third].re, y2_3);
        }

        //------------------------------------------------------------------
        // Cleanup: 2x unrolling for remaining butterflies
        //------------------------------------------------------------------
        const __m256d rot_mask_final = (transform_sign == 1)
                                           ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                                           : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

        for (; k + 1 < third; k += 2)
        {
            if (k + 8 < third)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 8 + third].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 8 + 2 * third].re, _MM_HINT_T0);
            }

            __m256d a = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
            __m256d b = load2_aos(&sub_outputs[k + third], &sub_outputs[k + third + 1]);
            __m256d c = load2_aos(&sub_outputs[k + 2 * third], &sub_outputs[k + 2 * third + 1]);

            __m256d w1 = load2_aos(&stage_tw[2 * k], &stage_tw[2 * (k + 1)]);
            __m256d w2 = load2_aos(&stage_tw[2 * k + 1], &stage_tw[2 * (k + 1) + 1]);

            __m256d b2 = cmul_avx2_aos(b, w1);
            __m256d c2 = cmul_avx2_aos(c, w2);

            __m256d sum = _mm256_add_pd(b2, c2);
            __m256d dif = _mm256_sub_pd(b2, c2);

            __m256d y0 = _mm256_add_pd(a, sum);
            __m256d temp = _mm256_sub_pd(a, _mm256_mul_pd(sum, vhalf));

            __m256d dif_swp = _mm256_permute_pd(dif, 0b0101);
            __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask_final);
            __m256d rot = _mm256_mul_pd(rot90, vsqrt3_2);

            __m256d y1 = _mm256_add_pd(temp, rot);
            __m256d y2 = _mm256_sub_pd(temp, rot);

            STOREU_PD(&output_buffer[k].re, y0);
            STOREU_PD(&output_buffer[k + third].re, y1);
            STOREU_PD(&output_buffer[k + 2 * third].re, y2);
        }
#endif // __AVX2__

        //------------------------------------------------------------------
        // SSE2 TAIL: Handle remaining 0..1 elements
        //------------------------------------------------------------------
        const __m128d vhalf128 = _mm_set1_pd(0.5);
        const __m128d vsqrt3_128 = _mm_set1_pd(C3_SQRT3BY2);

        for (; k < third; ++k)
        {
            __m128d a = LOADU_SSE2(&sub_outputs[k].re);
            __m128d b = LOADU_SSE2(&sub_outputs[k + third].re);
            __m128d c = LOADU_SSE2(&sub_outputs[k + 2 * third].re);

            __m128d w1 = LOADU_SSE2(&stage_tw[2 * k].re);
            __m128d w2 = LOADU_SSE2(&stage_tw[2 * k + 1].re);

            __m128d b2 = cmul_sse2_aos(b, w1);
            __m128d c2 = cmul_sse2_aos(c, w2);

            __m128d sum = _mm_add_pd(b2, c2);
            __m128d dif = _mm_sub_pd(b2, c2);

            __m128d y0 = _mm_add_pd(a, sum);
            STOREU_SSE2(&output_buffer[k].re, y0);

            __m128d temp = _mm_sub_pd(a, _mm_mul_pd(sum, vhalf128));

            __m128d swp = _mm_shuffle_pd(dif, dif, 0b01);
            __m128d rot90 = (transform_sign == 1)
                                ? _mm_xor_pd(swp, _mm_set_pd(-0.0, 0.0))
                                : _mm_xor_pd(swp, _mm_set_pd(0.0, -0.0));
            __m128d rot = _mm_mul_pd(rot90, vsqrt3_128);

            __m128d y1 = _mm_add_pd(temp, rot);
            __m128d y2 = _mm_sub_pd(temp, rot);

            STOREU_SSE2(&output_buffer[k + third].re, y1);
            STOREU_SSE2(&output_buffer[k + 2 * third].re, y2);
        }
    }
    else if (radix == 4)
    {
        //======================================================================
        // RADIX-4 BUTTERFLY (DIT) - FFTW-STYLE OPTIMIZED
        //
        // Pure AoS, no conversions, heavy unrolling for maximum performance.
        //======================================================================

        const int quarter = sub_len;
        int k = 0;

#ifdef HAS_AVX512
        //------------------------------------------------------------------
        // AVX-512 PATH: 16x unrolling (4 registers × 4 complex = 16 butterflies)
        //------------------------------------------------------------------

        // Precompute rotation masks (±i multiplication)
        const __m512d mask_plus_i_512 = _mm512_castsi512_pd(
            _mm512_set_epi64(0x0000000000000000, 0x8000000000000000,
                             0x0000000000000000, 0x8000000000000000,
                             0x0000000000000000, 0x8000000000000000,
                             0x0000000000000000, 0x8000000000000000));
        const __m512d mask_minus_i_512  = _mm512_castsi512_pd(
            _mm512_set_epi64(0x8000000000000000, 0x0000000000000000,
                             0x8000000000000000, 0x0000000000000000,
                             0x8000000000000000, 0x0000000000000000,
                             0x8000000000000000, 0x0000000000000000));
        const __m512d rot_mask_512  = (transform_sign == 1) ? mask_plus_i_512 : mask_minus_i_512 ;

           #define RADIX4_BUTTERFLY_AVX512(a, b2, c2, d2, y0, y1, y2, y3)    \
        {                                                                 \
        __m512d sumBD = _mm512_add_pd(b2, d2);                        \
        __m512d difBD = _mm512_sub_pd(b2, d2);                        \
        __m512d a_pc = _mm512_add_pd(a, c2);                          \
        __m512d a_mc = _mm512_sub_pd(a, c2);                          \
        y0 = _mm512_add_pd(a_pc, sumBD);                              \
        y2 = _mm512_sub_pd(a_pc, sumBD);                              \
        __m512d difBD_swp = _mm512_permute_pd(difBD, 0b01010101);     \
        __m512d rot = _mm512_xor_pd(difBD_swp, rot_mask_512);         \
        y1 = _mm512_sub_pd(a_mc, rot);                                \
        y3 = _mm512_add_pd(a_mc, rot);                                \
        }

        for (; k + 15 < quarter; k += 16)
        {
            // Prefetch ahead
            if (k + 32 < quarter)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 32].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[3 * (k + 32)].re, _MM_HINT_T0);
            }

            //==================================================================
            // Load inputs (16 butterflies = 4 loads per lane × 4 lanes)
            //==================================================================
            __m512d a0 = load4_aos(&sub_outputs[k + 0]);
            __m512d a1 = load4_aos(&sub_outputs[k + 4]);
            __m512d a2 = load4_aos(&sub_outputs[k + 8]);
            __m512d a3 = load4_aos(&sub_outputs[k + 12]);

            __m512d b0 = load4_aos(&sub_outputs[k + 0 + quarter]);
            __m512d b1 = load4_aos(&sub_outputs[k + 4 + quarter]);
            __m512d b2 = load4_aos(&sub_outputs[k + 8 + quarter]);
            __m512d b3 = load4_aos(&sub_outputs[k + 12 + quarter]);

            __m512d c0 = load4_aos(&sub_outputs[k + 0 + 2 * quarter]);
            __m512d c1 = load4_aos(&sub_outputs[k + 4 + 2 * quarter]);
            __m512d c2 = load4_aos(&sub_outputs[k + 8 + 2 * quarter]);
            __m512d c3 = load4_aos(&sub_outputs[k + 12 + 2 * quarter]);

            __m512d d0 = load4_aos(&sub_outputs[k + 0 + 3 * quarter]);
            __m512d d1 = load4_aos(&sub_outputs[k + 4 + 3 * quarter]);
            __m512d d2 = load4_aos(&sub_outputs[k + 8 + 3 * quarter]);
            __m512d d3 = load4_aos(&sub_outputs[k + 12 + 3 * quarter]);

            //==================================================================
            // Load twiddles W^k, W^{2k}, W^{3k} (k-major: 3 per butterfly)
            //==================================================================
            __m512d w1_0 = load4_aos(&stage_tw[3 * (k + 0)]);
            __m512d w1_1 = load4_aos(&stage_tw[3 * (k + 4)]);
            __m512d w1_2 = load4_aos(&stage_tw[3 * (k + 8)]);
            __m512d w1_3 = load4_aos(&stage_tw[3 * (k + 12)]);

            __m512d w2_0 = load4_aos(&stage_tw[3 * (k + 0) + 1]);
            __m512d w2_1 = load4_aos(&stage_tw[3 * (k + 4) + 1]);
            __m512d w2_2 = load4_aos(&stage_tw[3 * (k + 8) + 1]);
            __m512d w2_3 = load4_aos(&stage_tw[3 * (k + 12) + 1]);

            __m512d w3_0 = load4_aos(&stage_tw[3 * (k + 0) + 2]);
            __m512d w3_1 = load4_aos(&stage_tw[3 * (k + 4) + 2]);
            __m512d w3_2 = load4_aos(&stage_tw[3 * (k + 8) + 2]);
            __m512d w3_3 = load4_aos(&stage_tw[3 * (k + 12) + 2]);

            //==================================================================
            // Twiddle multiply
            //==================================================================
            __m512d b2_0 = cmul_avx512_aos(b0, w1_0);
            __m512d b2_1 = cmul_avx512_aos(b1, w1_1);
            __m512d b2_2 = cmul_avx512_aos(b2, w1_2);
            __m512d b2_3 = cmul_avx512_aos(b3, w1_3);

            __m512d c2_0 = cmul_avx512_aos(c0, w2_0);
            __m512d c2_1 = cmul_avx512_aos(c1, w2_1);
            __m512d c2_2 = cmul_avx512_aos(c2, w2_2);
            __m512d c2_3 = cmul_avx512_aos(c3, w2_3);

            __m512d d2_0 = cmul_avx512_aos(d0, w3_0);
            __m512d d2_1 = cmul_avx512_aos(d1, w3_1);
            __m512d d2_2 = cmul_avx512_aos(d2, w3_2);
            __m512d d2_3 = cmul_avx512_aos(d3, w3_3);

            __m512d y0_0, y1_0, y2_0, y3_0;
            __m512d y0_1, y1_1, y2_1, y3_1;
            __m512d y0_2, y1_2, y2_2, y3_2;
            __m512d y0_3, y1_3, y2_3, y3_3;

            RADIX4_BUTTERFLY_AVX512(a0, b2_0, c2_0, d2_0, y0_0, y1_0, y2_0, y3_0);
            RADIX4_BUTTERFLY_AVX512(a1, b2_1, c2_1, d2_1, y0_1, y1_1, y2_1, y3_1);
            RADIX4_BUTTERFLY_AVX512(a2, b2_2, c2_2, d2_2, y0_2, y1_2, y2_2, y3_2);
            RADIX4_BUTTERFLY_AVX512(a3, b2_3, c2_3, d2_3, y0_3, y1_3, y2_3, y3_3);

            //==================================================================
            // Store results
            //==================================================================
            STOREU_PD512(&output_buffer[k + 0].re, y0_0);
            STOREU_PD512(&output_buffer[k + 4].re, y0_1);
            STOREU_PD512(&output_buffer[k + 8].re, y0_2);
            STOREU_PD512(&output_buffer[k + 12].re, y0_3);

            STOREU_PD512(&output_buffer[k + 0 + quarter].re, y1_0);
            STOREU_PD512(&output_buffer[k + 4 + quarter].re, y1_1);
            STOREU_PD512(&output_buffer[k + 8 + quarter].re, y1_2);
            STOREU_PD512(&output_buffer[k + 12 + quarter].re, y1_3);

            STOREU_PD512(&output_buffer[k + 0 + 2 * quarter].re, y2_0);
            STOREU_PD512(&output_buffer[k + 4 + 2 * quarter].re, y2_1);
            STOREU_PD512(&output_buffer[k + 8 + 2 * quarter].re, y2_2);
            STOREU_PD512(&output_buffer[k + 12 + 2 * quarter].re, y2_3);

            STOREU_PD512(&output_buffer[k + 0 + 3 * quarter].re, y3_0);
            STOREU_PD512(&output_buffer[k + 4 + 3 * quarter].re, y3_1);
            STOREU_PD512(&output_buffer[k + 8 + 3 * quarter].re, y3_2);
            STOREU_PD512(&output_buffer[k + 12 + 3 * quarter].re, y3_3);
        }

    //==========================================================================
    // Cleanup: 8x unrolling (process 8 butterflies at once)
    //==========================================================================
    for (; k + 7 < quarter; k += 8)
    {
        // Load inputs (8 butterflies = 2 AVX-512 registers per lane)
        __m512d a0 = load4_aos(&sub_outputs[k + 0]);
        __m512d a1 = load4_aos(&sub_outputs[k + 4]);

        __m512d b0 = load4_aos(&sub_outputs[k + 0 + quarter]);
        __m512d b1 = load4_aos(&sub_outputs[k + 4 + quarter]);

        __m512d c0 = load4_aos(&sub_outputs[k + 0 + 2 * quarter]);
        __m512d c1 = load4_aos(&sub_outputs[k + 4 + 2 * quarter]);

        __m512d d0 = load4_aos(&sub_outputs[k + 0 + 3 * quarter]);
        __m512d d1 = load4_aos(&sub_outputs[k + 4 + 3 * quarter]);

        // Load twiddles
        __m512d w1_0 = load4_aos(&stage_tw[3 * (k + 0)]);
        __m512d w1_1 = load4_aos(&stage_tw[3 * (k + 4)]);

        __m512d w2_0 = load4_aos(&stage_tw[3 * (k + 0) + 1]);
        __m512d w2_1 = load4_aos(&stage_tw[3 * (k + 4) + 1]);

        __m512d w3_0 = load4_aos(&stage_tw[3 * (k + 0) + 2]);
        __m512d w3_1 = load4_aos(&stage_tw[3 * (k + 4) + 2]);

        // Twiddle multiply
        __m512d b2_0 = cmul_avx512_aos(b0, w1_0);
        __m512d b2_1 = cmul_avx512_aos(b1, w1_1);

        __m512d c2_0 = cmul_avx512_aos(c0, w2_0);
        __m512d c2_1 = cmul_avx512_aos(c1, w2_1);

        __m512d d2_0 = cmul_avx512_aos(d0, w3_0);
        __m512d d2_1 = cmul_avx512_aos(d1, w3_1);

        // Radix-4 butterfly (using the macro you defined earlier)
        __m512d y0_0, y1_0, y2_0, y3_0;
        __m512d y0_1, y1_1, y2_1, y3_1;

        RADIX4_BUTTERFLY_AVX512(a0, b2_0, c2_0, d2_0, y0_0, y1_0, y2_0, y3_0);
        RADIX4_BUTTERFLY_AVX512(a1, b2_1, c2_1, d2_1, y0_1, y1_1, y2_1, y3_1);

        // Store results
        STOREU_PD512(&output_buffer[k + 0].re, y0_0);
        STOREU_PD512(&output_buffer[k + 4].re, y0_1);

        STOREU_PD512(&output_buffer[k + 0 + quarter].re, y1_0);
        STOREU_PD512(&output_buffer[k + 4 + quarter].re, y1_1);

        STOREU_PD512(&output_buffer[k + 0 + 2 * quarter].re, y2_0);
        STOREU_PD512(&output_buffer[k + 4 + 2 * quarter].re, y2_1);

        STOREU_PD512(&output_buffer[k + 0 + 3 * quarter].re, y3_0);
        STOREU_PD512(&output_buffer[k + 4 + 3 * quarter].re, y3_1);
    }

    //==========================================================================
    // Cleanup: 4x unrolling (process 4 butterflies at once)
    //==========================================================================
    for (; k + 3 < quarter; k += 4)
    {
        // Load inputs (4 butterflies = 1 AVX-512 register per lane)
        __m512d a = load4_aos(&sub_outputs[k]);
        __m512d b = load4_aos(&sub_outputs[k + quarter]);
        __m512d c = load4_aos(&sub_outputs[k + 2 * quarter]);
        __m512d d = load4_aos(&sub_outputs[k + 3 * quarter]);

        // Load twiddles
        __m512d w1 = load4_aos(&stage_tw[3 * k]);
        __m512d w2 = load4_aos(&stage_tw[3 * k + 1]);
        __m512d w3 = load4_aos(&stage_tw[3 * k + 2]);

        // Twiddle multiply
        __m512d b2 = cmul_avx512_aos(b, w1);
        __m512d c2 = cmul_avx512_aos(c, w2);
        __m512d d2 = cmul_avx512_aos(d, w3);

        // Radix-4 butterfly
        __m512d y0, y1, y2, y3;
        RADIX4_BUTTERFLY_AVX512(a, b2, c2, d2, y0, y1, y2, y3);

        // Store results
        STOREU_PD512(&output_buffer[k].re, y0);
        STOREU_PD512(&output_buffer[k + quarter].re, y1);
        STOREU_PD512(&output_buffer[k + 2 * quarter].re, y2);
        STOREU_PD512(&output_buffer[k + 3 * quarter].re, y3);
    }

    #undef RADIX4_BUTTERFLY_AVX512
#endif // HAS_AVX512

#ifdef __AVX2__
        //------------------------------------------------------------------
        // AVX2 PATH: 8x unrolled, pure AoS
        //------------------------------------------------------------------

        // Precompute rotation masks
        const __m256d mask_plus_i = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
        const __m256d mask_minus_i = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
        const __m256d rot_mask = (transform_sign == 1) ? mask_plus_i : mask_minus_i;

         // DEFINE AVX2 MACRO
        #define RADIX4_BUTTERFLY_AVX2(a, b2, c2, d2, y0, y1, y2, y3)      \
        {                                                                 \
        __m256d sumBD = _mm256_add_pd(b2, d2);                        \
        __m256d difBD = _mm256_sub_pd(b2, d2);                        \
        __m256d a_pc = _mm256_add_pd(a, c2);                          \
        __m256d a_mc = _mm256_sub_pd(a, c2);                          \
        y0 = _mm256_add_pd(a_pc, sumBD);                              \
        y2 = _mm256_sub_pd(a_pc, sumBD);                              \
        __m256d difBD_swp = _mm256_permute_pd(difBD, 0b0101);         \
        __m256d rot = _mm256_xor_pd(difBD_swp, rot_mask);             \
        y1 = _mm256_sub_pd(a_mc, rot);                                \
        y3 = _mm256_add_pd(a_mc, rot);                                \
          }

        for (; k + 7 < quarter; k += 8)
        {
            // Prefetch ahead
            if (k + 16 < quarter)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + quarter].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 2 * quarter].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 3 * quarter].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[3 * (k + 16)].re, _MM_HINT_T0);
            }

            //==================================================================
            // Load inputs (8 butterflies = 4 AVX2 loads per lane)
            //==================================================================
            __m256d a0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
            __m256d a1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
            __m256d a2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
            __m256d a3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

            __m256d b0 = load2_aos(&sub_outputs[k + 0 + quarter], &sub_outputs[k + 1 + quarter]);
            __m256d b1 = load2_aos(&sub_outputs[k + 2 + quarter], &sub_outputs[k + 3 + quarter]);
            __m256d b2 = load2_aos(&sub_outputs[k + 4 + quarter], &sub_outputs[k + 5 + quarter]);
            __m256d b3 = load2_aos(&sub_outputs[k + 6 + quarter], &sub_outputs[k + 7 + quarter]);

            __m256d c0 = load2_aos(&sub_outputs[k + 0 + 2 * quarter], &sub_outputs[k + 1 + 2 * quarter]);
            __m256d c1 = load2_aos(&sub_outputs[k + 2 + 2 * quarter], &sub_outputs[k + 3 + 2 * quarter]);
            __m256d c2 = load2_aos(&sub_outputs[k + 4 + 2 * quarter], &sub_outputs[k + 5 + 2 * quarter]);
            __m256d c3 = load2_aos(&sub_outputs[k + 6 + 2 * quarter], &sub_outputs[k + 7 + 2 * quarter]);

            __m256d d0 = load2_aos(&sub_outputs[k + 0 + 3 * quarter], &sub_outputs[k + 1 + 3 * quarter]);
            __m256d d1 = load2_aos(&sub_outputs[k + 2 + 3 * quarter], &sub_outputs[k + 3 + 3 * quarter]);
            __m256d d2 = load2_aos(&sub_outputs[k + 4 + 3 * quarter], &sub_outputs[k + 5 + 3 * quarter]);
            __m256d d3 = load2_aos(&sub_outputs[k + 6 + 3 * quarter], &sub_outputs[k + 7 + 3 * quarter]);

            //==================================================================
            // Load twiddles W^k, W^{2k}, W^{3k} (k-major: 3 per butterfly)
            //==================================================================
            __m256d w1_0 = load2_aos(&stage_tw[3 * (k + 0)], &stage_tw[3 * (k + 1)]);
            __m256d w1_1 = load2_aos(&stage_tw[3 * (k + 2)], &stage_tw[3 * (k + 3)]);
            __m256d w1_2 = load2_aos(&stage_tw[3 * (k + 4)], &stage_tw[3 * (k + 5)]);
            __m256d w1_3 = load2_aos(&stage_tw[3 * (k + 6)], &stage_tw[3 * (k + 7)]);

            __m256d w2_0 = load2_aos(&stage_tw[3 * (k + 0) + 1], &stage_tw[3 * (k + 1) + 1]);
            __m256d w2_1 = load2_aos(&stage_tw[3 * (k + 2) + 1], &stage_tw[3 * (k + 3) + 1]);
            __m256d w2_2 = load2_aos(&stage_tw[3 * (k + 4) + 1], &stage_tw[3 * (k + 5) + 1]);
            __m256d w2_3 = load2_aos(&stage_tw[3 * (k + 6) + 1], &stage_tw[3 * (k + 7) + 1]);

            __m256d w3_0 = load2_aos(&stage_tw[3 * (k + 0) + 2], &stage_tw[3 * (k + 1) + 2]);
            __m256d w3_1 = load2_aos(&stage_tw[3 * (k + 2) + 2], &stage_tw[3 * (k + 3) + 2]);
            __m256d w3_2 = load2_aos(&stage_tw[3 * (k + 4) + 2], &stage_tw[3 * (k + 5) + 2]);
            __m256d w3_3 = load2_aos(&stage_tw[3 * (k + 6) + 2], &stage_tw[3 * (k + 7) + 2]);

            //==================================================================
            // Twiddle multiply
            //==================================================================
            __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
            __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
            __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
            __m256d b2_3 = cmul_avx2_aos(b3, w1_3);

            __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
            __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
            __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
            __m256d c2_3 = cmul_avx2_aos(c3, w2_3);

            __m256d d2_0 = cmul_avx2_aos(d0, w3_0);
            __m256d d2_1 = cmul_avx2_aos(d1, w3_1);
            __m256d d2_2 = cmul_avx2_aos(d2, w3_2);
            __m256d d2_3 = cmul_avx2_aos(d3, w3_3);
            __m256d y0_0, y1_0, y2_0, y3_0;
            __m256d y0_1, y1_1, y2_1, y3_1;
            __m256d y0_2, y1_2, y2_2, y3_2;
            __m256d y0_3, y1_3, y2_3, y3_3;

            RADIX4_BUTTERFLY_AVX2(a0, b2_0, c2_0, d2_0, y0_0, y1_0, y2_0, y3_0);
            RADIX4_BUTTERFLY_AVX2(a1, b2_1, c2_1, d2_1, y0_1, y1_1, y2_1, y3_1);
            RADIX4_BUTTERFLY_AVX2(a2, b2_2, c2_2, d2_2, y0_2, y1_2, y2_2, y3_2);
            RADIX4_BUTTERFLY_AVX2(a3, b2_3, c2_3, d2_3, y0_3, y1_3, y2_3, y3_3);

            //==================================================================
            // Store results (pure AoS, no conversions!)
            //==================================================================
            STOREU_PD(&output_buffer[k + 0].re, y0_0);
            STOREU_PD(&output_buffer[k + 2].re, y0_1);
            STOREU_PD(&output_buffer[k + 4].re, y0_2);
            STOREU_PD(&output_buffer[k + 6].re, y0_3);

            STOREU_PD(&output_buffer[k + 0 + quarter].re, y1_0);
            STOREU_PD(&output_buffer[k + 2 + quarter].re, y1_1);
            STOREU_PD(&output_buffer[k + 4 + quarter].re, y1_2);
            STOREU_PD(&output_buffer[k + 6 + quarter].re, y1_3);

            STOREU_PD(&output_buffer[k + 0 + 2 * quarter].re, y2_0);
            STOREU_PD(&output_buffer[k + 2 + 2 * quarter].re, y2_1);
            STOREU_PD(&output_buffer[k + 4 + 2 * quarter].re, y2_2);
            STOREU_PD(&output_buffer[k + 6 + 2 * quarter].re, y2_3);

            STOREU_PD(&output_buffer[k + 0 + 3 * quarter].re, y3_0);
            STOREU_PD(&output_buffer[k + 2 + 3 * quarter].re, y3_1);
            STOREU_PD(&output_buffer[k + 4 + 3 * quarter].re, y3_2);
            STOREU_PD(&output_buffer[k + 6 + 3 * quarter].re, y3_3);
        }

        //------------------------------------------------------------------
        // Cleanup: 2x unrolling
        //------------------------------------------------------------------
        const __m256d rot_mask_final = (transform_sign == 1)
                                           ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                                           : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

        for (; k + 1 < quarter; k += 2)
        {
            if (k + 8 < quarter)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 8 + quarter].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 8 + 2 * quarter].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 8 + 3 * quarter].re, _MM_HINT_T0);
            }

            __m256d a = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
            __m256d b = load2_aos(&sub_outputs[k + quarter], &sub_outputs[k + quarter + 1]);
            __m256d c = load2_aos(&sub_outputs[k + 2 * quarter], &sub_outputs[k + 2 * quarter + 1]);
            __m256d d = load2_aos(&sub_outputs[k + 3 * quarter], &sub_outputs[k + 3 * quarter + 1]);

            __m256d w1 = load2_aos(&stage_tw[3 * k], &stage_tw[3 * (k + 1)]);
            __m256d w2 = load2_aos(&stage_tw[3 * k + 1], &stage_tw[3 * (k + 1) + 1]);
            __m256d w3 = load2_aos(&stage_tw[3 * k + 2], &stage_tw[3 * (k + 1) + 2]);

            __m256d b2 = cmul_avx2_aos(b, w1);
            __m256d c2 = cmul_avx2_aos(c, w2);
            __m256d d2 = cmul_avx2_aos(d, w3);

            __m256d sumBD = _mm256_add_pd(b2, d2);
            __m256d difBD = _mm256_sub_pd(b2, d2);
            __m256d a_pc = _mm256_add_pd(a, c2);
            __m256d a_mc = _mm256_sub_pd(a, c2);

            __m256d y0 = _mm256_add_pd(a_pc, sumBD);
            __m256d y2 = _mm256_sub_pd(a_pc, sumBD);

            __m256d difBD_swp = _mm256_permute_pd(difBD, 0b0101);
            __m256d rot = _mm256_xor_pd(difBD_swp, rot_mask_final);

            __m256d y1 = _mm256_sub_pd(a_mc, rot);
            __m256d y3 = _mm256_add_pd(a_mc, rot);

            STOREU_PD(&output_buffer[k].re, y0);
            STOREU_PD(&output_buffer[k + quarter].re, y1);
            STOREU_PD(&output_buffer[k + 2 * quarter].re, y2);
            STOREU_PD(&output_buffer[k + 3 * quarter].re, y3);
        }
        #undef RADIX4_BUTTERFLY_AVX2
#endif // __AVX2__

        //------------------------------------------------------------------
        // SSE2 TAIL: Handle remaining 0..1 elements
        //------------------------------------------------------------------
        for (; k < quarter; ++k)
        {
            __m128d a = LOADU_SSE2(&sub_outputs[k].re);
            __m128d b = LOADU_SSE2(&sub_outputs[k + quarter].re);
            __m128d c = LOADU_SSE2(&sub_outputs[k + 2 * quarter].re);
            __m128d d = LOADU_SSE2(&sub_outputs[k + 3 * quarter].re);

            __m128d w1 = LOADU_SSE2(&stage_tw[3 * k].re);
            __m128d w2 = LOADU_SSE2(&stage_tw[3 * k + 1].re);
            __m128d w3 = LOADU_SSE2(&stage_tw[3 * k + 2].re);

            __m128d b2 = cmul_sse2_aos(b, w1);
            __m128d c2 = cmul_sse2_aos(c, w2);
            __m128d d2 = cmul_sse2_aos(d, w3);

            __m128d sumBD = _mm_add_pd(b2, d2);
            __m128d difBD = _mm_sub_pd(b2, d2);
            __m128d a_pc = _mm_add_pd(a, c2);
            __m128d a_mc = _mm_sub_pd(a, c2);

            __m128d y0 = _mm_add_pd(a_pc, sumBD);
            __m128d y2 = _mm_sub_pd(a_mc, sumBD);

            __m128d swp = _mm_shuffle_pd(difBD, difBD, 0b01);
            __m128d rot = (transform_sign == 1)
                              ? _mm_xor_pd(swp, _mm_set_pd(-0.0, 0.0))
                              : _mm_xor_pd(swp, _mm_set_pd(0.0, -0.0));

            __m128d y1 = _mm_sub_pd(a_mc, rot);
            __m128d y3 = _mm_add_pd(a_mc, rot);

            STOREU_SSE2(&output_buffer[k].re, y0);
            STOREU_SSE2(&output_buffer[k + quarter].re, y1);
            STOREU_SSE2(&output_buffer[k + 2 * quarter].re, y2);
            STOREU_SSE2(&output_buffer[k + 3 * quarter].re, y3);
        }
    }
    else if (radix == 5)
    {
        //======================================================================
        // RADIX-5 BUTTERFLY (Rader/Winograd DIT) - FFTW-STYLE OPTIMIZED
        //
        // Pure AoS, no conversions, 4x unrolling for maximum performance.
        //======================================================================

        const int fifth = sub_len;
        int k = 0;

#ifdef __AVX2__
        //------------------------------------------------------------------
        // AVX2 PATH: 4x unrolled, pure AoS
        //------------------------------------------------------------------
        const __m256d vc1 = _mm256_set1_pd(C5_1); // cos(2π/5)
        const __m256d vc2 = _mm256_set1_pd(C5_2); // cos(4π/5)
        const __m256d vs1 = _mm256_set1_pd(S5_1); // sin(2π/5)
        const __m256d vs2 = _mm256_set1_pd(S5_2); // sin(4π/5)

        // Precompute rotation masks
        const __m256d mask_plus_i = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
        const __m256d mask_minus_i = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
        const __m256d rot_mask = (transform_sign == 1) ? mask_plus_i : mask_minus_i;

        for (; k + 7 < fifth; k += 8)
        {
            // Prefetch ahead
            if (k + 16 < fifth)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + fifth].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 2 * fifth].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 3 * fifth].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 4 * fifth].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[4 * (k + 16)].re, _MM_HINT_T0);
            }

            //==================================================================
            // Load inputs (8 butterflies = 4 AVX2 loads per lane)
            //==================================================================
            __m256d a0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
            __m256d a1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
            __m256d a2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
            __m256d a3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

            __m256d b0 = load2_aos(&sub_outputs[k + 0 + fifth], &sub_outputs[k + 1 + fifth]);
            __m256d b1 = load2_aos(&sub_outputs[k + 2 + fifth], &sub_outputs[k + 3 + fifth]);
            __m256d b2 = load2_aos(&sub_outputs[k + 4 + fifth], &sub_outputs[k + 5 + fifth]);
            __m256d b3 = load2_aos(&sub_outputs[k + 6 + fifth], &sub_outputs[k + 7 + fifth]);

            __m256d c0 = load2_aos(&sub_outputs[k + 0 + 2 * fifth], &sub_outputs[k + 1 + 2 * fifth]);
            __m256d c1 = load2_aos(&sub_outputs[k + 2 + 2 * fifth], &sub_outputs[k + 3 + 2 * fifth]);
            __m256d c2 = load2_aos(&sub_outputs[k + 4 + 2 * fifth], &sub_outputs[k + 5 + 2 * fifth]);
            __m256d c3 = load2_aos(&sub_outputs[k + 6 + 2 * fifth], &sub_outputs[k + 7 + 2 * fifth]);

            __m256d d0 = load2_aos(&sub_outputs[k + 0 + 3 * fifth], &sub_outputs[k + 1 + 3 * fifth]);
            __m256d d1 = load2_aos(&sub_outputs[k + 2 + 3 * fifth], &sub_outputs[k + 3 + 3 * fifth]);
            __m256d d2 = load2_aos(&sub_outputs[k + 4 + 3 * fifth], &sub_outputs[k + 5 + 3 * fifth]);
            __m256d d3 = load2_aos(&sub_outputs[k + 6 + 3 * fifth], &sub_outputs[k + 7 + 3 * fifth]);

            __m256d e0 = load2_aos(&sub_outputs[k + 0 + 4 * fifth], &sub_outputs[k + 1 + 4 * fifth]);
            __m256d e1 = load2_aos(&sub_outputs[k + 2 + 4 * fifth], &sub_outputs[k + 3 + 4 * fifth]);
            __m256d e2 = load2_aos(&sub_outputs[k + 4 + 4 * fifth], &sub_outputs[k + 5 + 4 * fifth]);
            __m256d e3 = load2_aos(&sub_outputs[k + 6 + 4 * fifth], &sub_outputs[k + 7 + 4 * fifth]);

            //==================================================================
            // Load twiddles W^k, W^{2k}, W^{3k}, W^{4k} (k-major: 4 per butterfly)
            //==================================================================
            __m256d w1_0 = load2_aos(&stage_tw[4 * (k + 0)], &stage_tw[4 * (k + 1)]);
            __m256d w1_1 = load2_aos(&stage_tw[4 * (k + 2)], &stage_tw[4 * (k + 3)]);
            __m256d w1_2 = load2_aos(&stage_tw[4 * (k + 4)], &stage_tw[4 * (k + 5)]);
            __m256d w1_3 = load2_aos(&stage_tw[4 * (k + 6)], &stage_tw[4 * (k + 7)]);

            __m256d w2_0 = load2_aos(&stage_tw[4 * (k + 0) + 1], &stage_tw[4 * (k + 1) + 1]);
            __m256d w2_1 = load2_aos(&stage_tw[4 * (k + 2) + 1], &stage_tw[4 * (k + 3) + 1]);
            __m256d w2_2 = load2_aos(&stage_tw[4 * (k + 4) + 1], &stage_tw[4 * (k + 5) + 1]);
            __m256d w2_3 = load2_aos(&stage_tw[4 * (k + 6) + 1], &stage_tw[4 * (k + 7) + 1]);

            __m256d w3_0 = load2_aos(&stage_tw[4 * (k + 0) + 2], &stage_tw[4 * (k + 1) + 2]);
            __m256d w3_1 = load2_aos(&stage_tw[4 * (k + 2) + 2], &stage_tw[4 * (k + 3) + 2]);
            __m256d w3_2 = load2_aos(&stage_tw[4 * (k + 4) + 2], &stage_tw[4 * (k + 5) + 2]);
            __m256d w3_3 = load2_aos(&stage_tw[4 * (k + 6) + 2], &stage_tw[4 * (k + 7) + 2]);

            __m256d w4_0 = load2_aos(&stage_tw[4 * (k + 0) + 3], &stage_tw[4 * (k + 1) + 3]);
            __m256d w4_1 = load2_aos(&stage_tw[4 * (k + 2) + 3], &stage_tw[4 * (k + 3) + 3]);
            __m256d w4_2 = load2_aos(&stage_tw[4 * (k + 4) + 3], &stage_tw[4 * (k + 5) + 3]);
            __m256d w4_3 = load2_aos(&stage_tw[4 * (k + 6) + 3], &stage_tw[4 * (k + 7) + 3]);

            //==================================================================
            // Twiddle multiply
            //==================================================================
            __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
            __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
            __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
            __m256d b2_3 = cmul_avx2_aos(b3, w1_3);

            __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
            __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
            __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
            __m256d c2_3 = cmul_avx2_aos(c3, w2_3);

            __m256d d2_0 = cmul_avx2_aos(d0, w3_0);
            __m256d d2_1 = cmul_avx2_aos(d1, w3_1);
            __m256d d2_2 = cmul_avx2_aos(d2, w3_2);
            __m256d d2_3 = cmul_avx2_aos(d3, w3_3);

            __m256d e2_0 = cmul_avx2_aos(e0, w4_0);
            __m256d e2_1 = cmul_avx2_aos(e1, w4_1);
            __m256d e2_2 = cmul_avx2_aos(e2, w4_2);
            __m256d e2_3 = cmul_avx2_aos(e3, w4_3);

//==================================================================
// Radix-5 butterfly (8 butterflies in parallel)
//==================================================================
#define RADIX5_BUTTERFLY_AVX2(a, b2, c2, d2, e2, y0, y1, y2, y3, y4) \
    {                                                                \
        __m256d t0 = _mm256_add_pd(b2, e2);                          \
        __m256d t1 = _mm256_add_pd(c2, d2);                          \
        __m256d t2 = _mm256_sub_pd(b2, e2);                          \
        __m256d t3 = _mm256_sub_pd(c2, d2);                          \
        y0 = _mm256_add_pd(a, _mm256_add_pd(t0, t1));                \
        __m256d base1 = FMADD(vs1, t2, _mm256_mul_pd(vs2, t3));      \
        __m256d tmp1 = FMADD(vc1, t0, _mm256_mul_pd(vc2, t1));       \
        __m256d base1_swp = _mm256_permute_pd(base1, 0b0101);        \
        __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask);             \
        __m256d a1 = _mm256_add_pd(a, tmp1);                         \
        y1 = _mm256_add_pd(a1, r1);                                  \
        y4 = _mm256_sub_pd(a1, r1);                                  \
        __m256d base2 = FMSUB(vs2, t2, _mm256_mul_pd(vs1, t3));      \
        __m256d tmp2 = FMADD(vc2, t0, _mm256_mul_pd(vc1, t1));       \
        __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);        \
        __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask);             \
        __m256d a2 = _mm256_add_pd(a, tmp2);                         \
        y2 = _mm256_add_pd(a2, r2);                                  \
        y3 = _mm256_sub_pd(a2, r2);                                  \
    }

            __m256d y0_0, y1_0, y2_0, y3_0, y4_0;
            __m256d y0_1, y1_1, y2_1, y3_1, y4_1;
            __m256d y0_2, y1_2, y2_2, y3_2, y4_2;
            __m256d y0_3, y1_3, y2_3, y3_3, y4_3;

            RADIX5_BUTTERFLY_AVX2(a0, b2_0, c2_0, d2_0, e2_0, y0_0, y1_0, y2_0, y3_0, y4_0);
            RADIX5_BUTTERFLY_AVX2(a1, b2_1, c2_1, d2_1, e2_1, y0_1, y1_1, y2_1, y3_1, y4_1);
            RADIX5_BUTTERFLY_AVX2(a2, b2_2, c2_2, d2_2, e2_2, y0_2, y1_2, y2_2, y3_2, y4_2);
            RADIX5_BUTTERFLY_AVX2(a3, b2_3, c2_3, d2_3, e2_3, y0_3, y1_3, y2_3, y3_3, y4_3);

#undef RADIX5_BUTTERFLY_AVX2

            //==================================================================
            // Store results (pure AoS!)
            //==================================================================
            STOREU_PD(&output_buffer[k + 0].re, y0_0);
            STOREU_PD(&output_buffer[k + 2].re, y0_1);
            STOREU_PD(&output_buffer[k + 4].re, y0_2);
            STOREU_PD(&output_buffer[k + 6].re, y0_3);

            STOREU_PD(&output_buffer[k + 0 + fifth].re, y1_0);
            STOREU_PD(&output_buffer[k + 2 + fifth].re, y1_1);
            STOREU_PD(&output_buffer[k + 4 + fifth].re, y1_2);
            STOREU_PD(&output_buffer[k + 6 + fifth].re, y1_3);

            STOREU_PD(&output_buffer[k + 0 + 2 * fifth].re, y2_0);
            STOREU_PD(&output_buffer[k + 2 + 2 * fifth].re, y2_1);
            STOREU_PD(&output_buffer[k + 4 + 2 * fifth].re, y2_2);
            STOREU_PD(&output_buffer[k + 6 + 2 * fifth].re, y2_3);

            STOREU_PD(&output_buffer[k + 0 + 3 * fifth].re, y3_0);
            STOREU_PD(&output_buffer[k + 2 + 3 * fifth].re, y3_1);
            STOREU_PD(&output_buffer[k + 4 + 3 * fifth].re, y3_2);
            STOREU_PD(&output_buffer[k + 6 + 3 * fifth].re, y3_3);

            STOREU_PD(&output_buffer[k + 0 + 4 * fifth].re, y4_0);
            STOREU_PD(&output_buffer[k + 2 + 4 * fifth].re, y4_1);
            STOREU_PD(&output_buffer[k + 4 + 4 * fifth].re, y4_2);
            STOREU_PD(&output_buffer[k + 6 + 4 * fifth].re, y4_3);
        }

        //------------------------------------------------------------------
        // Cleanup: 2x unrolling
        //------------------------------------------------------------------
        const __m256d rot_mask_final = (transform_sign == 1)
                                           ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                                           : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

        for (; k + 1 < fifth; k += 2)
        {
            if (k + 8 < fifth)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 8 + fifth].re, _MM_HINT_T0);
            }

            __m256d a = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
            __m256d b = load2_aos(&sub_outputs[k + fifth], &sub_outputs[k + fifth + 1]);
            __m256d c = load2_aos(&sub_outputs[k + 2 * fifth], &sub_outputs[k + 2 * fifth + 1]);
            __m256d d = load2_aos(&sub_outputs[k + 3 * fifth], &sub_outputs[k + 3 * fifth + 1]);
            __m256d e = load2_aos(&sub_outputs[k + 4 * fifth], &sub_outputs[k + 4 * fifth + 1]);

            __m256d w1 = load2_aos(&stage_tw[4 * k], &stage_tw[4 * (k + 1)]);
            __m256d w2 = load2_aos(&stage_tw[4 * k + 1], &stage_tw[4 * (k + 1) + 1]);
            __m256d w3 = load2_aos(&stage_tw[4 * k + 2], &stage_tw[4 * (k + 1) + 2]);
            __m256d w4 = load2_aos(&stage_tw[4 * k + 3], &stage_tw[4 * (k + 1) + 3]);

            __m256d b2 = cmul_avx2_aos(b, w1);
            __m256d c2 = cmul_avx2_aos(c, w2);
            __m256d d2 = cmul_avx2_aos(d, w3);
            __m256d e2 = cmul_avx2_aos(e, w4);

            __m256d t0 = _mm256_add_pd(b2, e2);
            __m256d t1 = _mm256_add_pd(c2, d2);
            __m256d t2 = _mm256_sub_pd(b2, e2);
            __m256d t3 = _mm256_sub_pd(c2, d2);

            __m256d y0 = _mm256_add_pd(a, _mm256_add_pd(t0, t1));

            __m256d base1 = FMADD(vs1, t2, _mm256_mul_pd(vs2, t3));
            __m256d tmp1 = FMADD(vc1, t0, _mm256_mul_pd(vc2, t1));
            __m256d base1_swp = _mm256_permute_pd(base1, 0b0101);
            __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask_final);
            __m256d a1 = _mm256_add_pd(a, tmp1);
            __m256d y1 = _mm256_add_pd(a1, r1);
            __m256d y4 = _mm256_sub_pd(a1, r1);

            __m256d base2 = FMSUB(vs2, t2, _mm256_mul_pd(vs1, t3));
            __m256d tmp2 = FMADD(vc2, t0, _mm256_mul_pd(vc1, t1));
            __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);
            __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask_final);
            __m256d a2 = _mm256_add_pd(a, tmp2);
            __m256d y2 = _mm256_add_pd(a2, r2);
            __m256d y3 = _mm256_sub_pd(a2, r2);

            STOREU_PD(&output_buffer[k].re, y0);
            STOREU_PD(&output_buffer[k + fifth].re, y1);
            STOREU_PD(&output_buffer[k + 2 * fifth].re, y2);
            STOREU_PD(&output_buffer[k + 3 * fifth].re, y3);
            STOREU_PD(&output_buffer[k + 4 * fifth].re, y4);
        }
#endif // __AVX2__

        //------------------------------------------------------------------
        // SSE2 TAIL: Handle remaining 0..1 elements
        //------------------------------------------------------------------
        for (; k < fifth; ++k)
        {
            __m128d a = LOADU_SSE2(&sub_outputs[k].re);
            __m128d b = LOADU_SSE2(&sub_outputs[k + fifth].re);
            __m128d c = LOADU_SSE2(&sub_outputs[k + 2 * fifth].re);
            __m128d d = LOADU_SSE2(&sub_outputs[k + 3 * fifth].re);
            __m128d e = LOADU_SSE2(&sub_outputs[k + 4 * fifth].re);

            __m128d w1 = LOADU_SSE2(&stage_tw[4 * k].re);
            __m128d w2 = LOADU_SSE2(&stage_tw[4 * k + 1].re);
            __m128d w3 = LOADU_SSE2(&stage_tw[4 * k + 2].re);
            __m128d w4 = LOADU_SSE2(&stage_tw[4 * k + 3].re);

            __m128d b2 = cmul_sse2_aos(b, w1);
            __m128d c2 = cmul_sse2_aos(c, w2);
            __m128d d2 = cmul_sse2_aos(d, w3);
            __m128d e2 = cmul_sse2_aos(e, w4);

            __m128d t0 = _mm_add_pd(b2, e2);
            __m128d t1 = _mm_add_pd(c2, d2);
            __m128d t2 = _mm_sub_pd(b2, e2);
            __m128d t3 = _mm_sub_pd(c2, d2);

            __m128d y0 = _mm_add_pd(a, _mm_add_pd(t0, t1));
            STOREU_SSE2(&output_buffer[k].re, y0);

            const __m128d vc1_128 = _mm_set1_pd(C5_1);
            const __m128d vc2_128 = _mm_set1_pd(C5_2);
            const __m128d vs1_128 = _mm_set1_pd(S5_1);
            const __m128d vs2_128 = _mm_set1_pd(S5_2);

            __m128d base1 = FMADD_SSE2(vs1_128, t2, _mm_mul_pd(vs2_128, t3));
            __m128d tmp1 = FMADD_SSE2(vc1_128, t0, _mm_mul_pd(vc2_128, t1));
            __m128d base1_swp = _mm_shuffle_pd(base1, base1, 0b01);
            __m128d r1 = (transform_sign == 1)
                             ? _mm_xor_pd(base1_swp, _mm_set_pd(-0.0, 0.0))
                             : _mm_xor_pd(base1_swp, _mm_set_pd(0.0, -0.0));
            __m128d a1 = _mm_add_pd(a, tmp1);
            __m128d y1 = _mm_add_pd(a1, r1);
            __m128d y4 = _mm_sub_pd(a1, r1);

            __m128d base2 = FMSUB_SSE2(vs2_128, t2, _mm_mul_pd(vs1_128, t3));
            __m128d tmp2 = FMADD_SSE2(vc2_128, t0, _mm_mul_pd(vc1_128, t1));
            __m128d base2_swp = _mm_shuffle_pd(base2, base2, 0b01);
            __m128d r2 = (transform_sign == 1)
                             ? _mm_xor_pd(base2_swp, _mm_set_pd(-0.0, 0.0))
                             : _mm_xor_pd(base2_swp, _mm_set_pd(0.0, -0.0));
            __m128d a2 = _mm_add_pd(a, tmp2);
            __m128d y2 = _mm_add_pd(a2, r2);
            __m128d y3 = _mm_sub_pd(a2, r2);

            STOREU_SSE2(&output_buffer[k + fifth].re, y1);
            STOREU_SSE2(&output_buffer[k + 2 * fifth].re, y2);
            STOREU_SSE2(&output_buffer[k + 3 * fifth].re, y3);
            STOREU_SSE2(&output_buffer[k + 4 * fifth].re, y4);
        }
    }
    else if (radix == 7)
    {
        //==========================================================================
        // RADIX-7 BUTTERFLY (Rader DIT) - FFTW-STYLE OPTIMIZED
        //
        // Pure AoS, no conversions, heavy unrolling for maximum performance.
        //==========================================================================

        const int seventh = sub_len;
        int k = 0;

#ifdef __AVX2__
        //------------------------------------------------------------------
        // AVX2 PATH: 4x unrolled, pure AoS
        //------------------------------------------------------------------
        const __m256d vc1 = _mm256_set1_pd(C1);
        const __m256d vc2 = _mm256_set1_pd(C2);
        const __m256d vc3 = _mm256_set1_pd(C3);
        const __m256d vs1 = _mm256_set1_pd(S1);
        const __m256d vs2 = _mm256_set1_pd(S2);
        const __m256d vs3 = _mm256_set1_pd(S3);

        // Precompute rotation masks
        const __m256d mask_plus_i = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
        const __m256d mask_minus_i = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
        const __m256d rot_mask = (transform_sign == 1) ? mask_plus_i : mask_minus_i;

        for (; k + 7 < seventh; k += 8)
        {
            // Prefetch ahead
            if (k + 16 < seventh)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[6 * (k + 16)].re, _MM_HINT_T0);
            }

            //==================================================================
            // Load inputs (8 butterflies = 4 AVX2 loads per lane × 7 lanes)
            //==================================================================
            __m256d a0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
            __m256d a1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
            __m256d a2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
            __m256d a3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

            __m256d b0 = load2_aos(&sub_outputs[k + 0 + seventh], &sub_outputs[k + 1 + seventh]);
            __m256d b1 = load2_aos(&sub_outputs[k + 2 + seventh], &sub_outputs[k + 3 + seventh]);
            __m256d b2 = load2_aos(&sub_outputs[k + 4 + seventh], &sub_outputs[k + 5 + seventh]);
            __m256d b3 = load2_aos(&sub_outputs[k + 6 + seventh], &sub_outputs[k + 7 + seventh]);

            __m256d c0 = load2_aos(&sub_outputs[k + 0 + 2 * seventh], &sub_outputs[k + 1 + 2 * seventh]);
            __m256d c1 = load2_aos(&sub_outputs[k + 2 + 2 * seventh], &sub_outputs[k + 3 + 2 * seventh]);
            __m256d c2 = load2_aos(&sub_outputs[k + 4 + 2 * seventh], &sub_outputs[k + 5 + 2 * seventh]);
            __m256d c3 = load2_aos(&sub_outputs[k + 6 + 2 * seventh], &sub_outputs[k + 7 + 2 * seventh]);

            __m256d d0 = load2_aos(&sub_outputs[k + 0 + 3 * seventh], &sub_outputs[k + 1 + 3 * seventh]);
            __m256d d1 = load2_aos(&sub_outputs[k + 2 + 3 * seventh], &sub_outputs[k + 3 + 3 * seventh]);
            __m256d d2 = load2_aos(&sub_outputs[k + 4 + 3 * seventh], &sub_outputs[k + 5 + 3 * seventh]);
            __m256d d3 = load2_aos(&sub_outputs[k + 6 + 3 * seventh], &sub_outputs[k + 7 + 3 * seventh]);

            __m256d e0 = load2_aos(&sub_outputs[k + 0 + 4 * seventh], &sub_outputs[k + 1 + 4 * seventh]);
            __m256d e1 = load2_aos(&sub_outputs[k + 2 + 4 * seventh], &sub_outputs[k + 3 + 4 * seventh]);
            __m256d e2 = load2_aos(&sub_outputs[k + 4 + 4 * seventh], &sub_outputs[k + 5 + 4 * seventh]);
            __m256d e3 = load2_aos(&sub_outputs[k + 6 + 4 * seventh], &sub_outputs[k + 7 + 4 * seventh]);

            __m256d f0 = load2_aos(&sub_outputs[k + 0 + 5 * seventh], &sub_outputs[k + 1 + 5 * seventh]);
            __m256d f1 = load2_aos(&sub_outputs[k + 2 + 5 * seventh], &sub_outputs[k + 3 + 5 * seventh]);
            __m256d f2 = load2_aos(&sub_outputs[k + 4 + 5 * seventh], &sub_outputs[k + 5 + 5 * seventh]);
            __m256d f3 = load2_aos(&sub_outputs[k + 6 + 5 * seventh], &sub_outputs[k + 7 + 5 * seventh]);

            __m256d g0 = load2_aos(&sub_outputs[k + 0 + 6 * seventh], &sub_outputs[k + 1 + 6 * seventh]);
            __m256d g1 = load2_aos(&sub_outputs[k + 2 + 6 * seventh], &sub_outputs[k + 3 + 6 * seventh]);
            __m256d g2 = load2_aos(&sub_outputs[k + 4 + 6 * seventh], &sub_outputs[k + 5 + 6 * seventh]);
            __m256d g3 = load2_aos(&sub_outputs[k + 6 + 6 * seventh], &sub_outputs[k + 7 + 6 * seventh]);

            //==================================================================
            // Load twiddles (k-major: 6 per butterfly)
            //==================================================================
            __m256d w1_0 = load2_aos(&stage_tw[6 * (k + 0)], &stage_tw[6 * (k + 1)]);
            __m256d w1_1 = load2_aos(&stage_tw[6 * (k + 2)], &stage_tw[6 * (k + 3)]);
            __m256d w1_2 = load2_aos(&stage_tw[6 * (k + 4)], &stage_tw[6 * (k + 5)]);
            __m256d w1_3 = load2_aos(&stage_tw[6 * (k + 6)], &stage_tw[6 * (k + 7)]);

            __m256d w2_0 = load2_aos(&stage_tw[6 * (k + 0) + 1], &stage_tw[6 * (k + 1) + 1]);
            __m256d w2_1 = load2_aos(&stage_tw[6 * (k + 2) + 1], &stage_tw[6 * (k + 3) + 1]);
            __m256d w2_2 = load2_aos(&stage_tw[6 * (k + 4) + 1], &stage_tw[6 * (k + 5) + 1]);
            __m256d w2_3 = load2_aos(&stage_tw[6 * (k + 6) + 1], &stage_tw[6 * (k + 7) + 1]);

            __m256d w3_0 = load2_aos(&stage_tw[6 * (k + 0) + 2], &stage_tw[6 * (k + 1) + 2]);
            __m256d w3_1 = load2_aos(&stage_tw[6 * (k + 2) + 2], &stage_tw[6 * (k + 3) + 2]);
            __m256d w3_2 = load2_aos(&stage_tw[6 * (k + 4) + 2], &stage_tw[6 * (k + 5) + 2]);
            __m256d w3_3 = load2_aos(&stage_tw[6 * (k + 6) + 2], &stage_tw[6 * (k + 7) + 2]);

            __m256d w4_0 = load2_aos(&stage_tw[6 * (k + 0) + 3], &stage_tw[6 * (k + 1) + 3]);
            __m256d w4_1 = load2_aos(&stage_tw[6 * (k + 2) + 3], &stage_tw[6 * (k + 3) + 3]);
            __m256d w4_2 = load2_aos(&stage_tw[6 * (k + 4) + 3], &stage_tw[6 * (k + 5) + 3]);
            __m256d w4_3 = load2_aos(&stage_tw[6 * (k + 6) + 3], &stage_tw[6 * (k + 7) + 3]);

            __m256d w5_0 = load2_aos(&stage_tw[6 * (k + 0) + 4], &stage_tw[6 * (k + 1) + 4]);
            __m256d w5_1 = load2_aos(&stage_tw[6 * (k + 2) + 4], &stage_tw[6 * (k + 3) + 4]);
            __m256d w5_2 = load2_aos(&stage_tw[6 * (k + 4) + 4], &stage_tw[6 * (k + 5) + 4]);
            __m256d w5_3 = load2_aos(&stage_tw[6 * (k + 6) + 4], &stage_tw[6 * (k + 7) + 4]);

            __m256d w6_0 = load2_aos(&stage_tw[6 * (k + 0) + 5], &stage_tw[6 * (k + 1) + 5]);
            __m256d w6_1 = load2_aos(&stage_tw[6 * (k + 2) + 5], &stage_tw[6 * (k + 3) + 5]);
            __m256d w6_2 = load2_aos(&stage_tw[6 * (k + 4) + 5], &stage_tw[6 * (k + 5) + 5]);
            __m256d w6_3 = load2_aos(&stage_tw[6 * (k + 6) + 5], &stage_tw[6 * (k + 7) + 5]);

            //==================================================================
            // Twiddle multiply
            //==================================================================
            __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
            __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
            __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
            __m256d b2_3 = cmul_avx2_aos(b3, w1_3);

            __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
            __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
            __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
            __m256d c2_3 = cmul_avx2_aos(c3, w2_3);

            __m256d d2_0 = cmul_avx2_aos(d0, w3_0);
            __m256d d2_1 = cmul_avx2_aos(d1, w3_1);
            __m256d d2_2 = cmul_avx2_aos(d2, w3_2);
            __m256d d2_3 = cmul_avx2_aos(d3, w3_3);

            __m256d e2_0 = cmul_avx2_aos(e0, w4_0);
            __m256d e2_1 = cmul_avx2_aos(e1, w4_1);
            __m256d e2_2 = cmul_avx2_aos(e2, w4_2);
            __m256d e2_3 = cmul_avx2_aos(e3, w4_3);

            __m256d f2_0 = cmul_avx2_aos(f0, w5_0);
            __m256d f2_1 = cmul_avx2_aos(f1, w5_1);
            __m256d f2_2 = cmul_avx2_aos(f2, w5_2);
            __m256d f2_3 = cmul_avx2_aos(f3, w5_3);

            __m256d g2_0 = cmul_avx2_aos(g0, w6_0);
            __m256d g2_1 = cmul_avx2_aos(g1, w6_1);
            __m256d g2_2 = cmul_avx2_aos(g2, w6_2);
            __m256d g2_3 = cmul_avx2_aos(g3, w6_3);

//==================================================================
// Radix-7 butterfly (8 butterflies in parallel)
//==================================================================
#define RADIX7_BUTTERFLY_AVX2(a, b2, c2, d2, e2, f2, g2, y0, y1, y2, y3, y4, y5, y6)             \
    {                                                                                            \
        __m256d t0 = _mm256_add_pd(b2, g2);                                                      \
        __m256d t1 = _mm256_add_pd(c2, f2);                                                      \
        __m256d t2 = _mm256_add_pd(d2, e2);                                                      \
        __m256d t3 = _mm256_sub_pd(b2, g2);                                                      \
        __m256d t4 = _mm256_sub_pd(c2, f2);                                                      \
        __m256d t5 = _mm256_sub_pd(d2, e2);                                                      \
        y0 = _mm256_add_pd(a, _mm256_add_pd(_mm256_add_pd(t0, t1), t2));                         \
        __m256d base1 = FMADD(vs1, t3, FMADD(vs2, t4, _mm256_mul_pd(vs3, t5)));                  \
        __m256d base2 = FMADD(vs2, t3, FMADD(vs3, t4, _mm256_mul_pd(vs1, t5)));                  \
        __m256d base3 = FMADD(vs3, t3, FMADD(vs1, t4, _mm256_mul_pd(vs2, t5)));                  \
        __m256d base1_swp = _mm256_permute_pd(base1, 0b0101);                                    \
        __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);                                    \
        __m256d base3_swp = _mm256_permute_pd(base3, 0b0101);                                    \
        __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask);                                         \
        __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask);                                         \
        __m256d r3 = _mm256_xor_pd(base3_swp, rot_mask);                                         \
        __m256d tmp1 = _mm256_add_pd(a, FMADD(vc1, t0, FMADD(vc2, t1, _mm256_mul_pd(vc3, t2)))); \
        __m256d tmp2 = _mm256_add_pd(a, FMADD(vc2, t0, FMADD(vc3, t1, _mm256_mul_pd(vc1, t2)))); \
        __m256d tmp3 = _mm256_add_pd(a, FMADD(vc3, t0, FMADD(vc1, t1, _mm256_mul_pd(vc2, t2)))); \
        y1 = _mm256_add_pd(tmp1, r1);                                                            \
        y6 = _mm256_sub_pd(tmp1, r1);                                                            \
        y2 = _mm256_add_pd(tmp2, r2);                                                            \
        y5 = _mm256_sub_pd(tmp2, r2);                                                            \
        y3 = _mm256_add_pd(tmp3, r3);                                                            \
        y4 = _mm256_sub_pd(tmp3, r3);                                                            \
    }

            __m256d y0_0, y1_0, y2_0, y3_0, y4_0, y5_0, y6_0;
            __m256d y0_1, y1_1, y2_1, y3_1, y4_1, y5_1, y6_1;
            __m256d y0_2, y1_2, y2_2, y3_2, y4_2, y5_2, y6_2;
            __m256d y0_3, y1_3, y2_3, y3_3, y4_3, y5_3, y6_3;

            RADIX7_BUTTERFLY_AVX2(a0, b2_0, c2_0, d2_0, e2_0, f2_0, g2_0, y0_0, y1_0, y2_0, y3_0, y4_0, y5_0, y6_0);
            RADIX7_BUTTERFLY_AVX2(a1, b2_1, c2_1, d2_1, e2_1, f2_1, g2_1, y0_1, y1_1, y2_1, y3_1, y4_1, y5_1, y6_1);
            RADIX7_BUTTERFLY_AVX2(a2, b2_2, c2_2, d2_2, e2_2, f2_2, g2_2, y0_2, y1_2, y2_2, y3_2, y4_2, y5_2, y6_2);
            RADIX7_BUTTERFLY_AVX2(a3, b2_3, c2_3, d2_3, e2_3, f2_3, g2_3, y0_3, y1_3, y2_3, y3_3, y4_3, y5_3, y6_3);

#undef RADIX7_BUTTERFLY_AVX2

            //==================================================================
            // Store results (pure AoS!)
            //==================================================================
            STOREU_PD(&output_buffer[k + 0].re, y0_0);
            STOREU_PD(&output_buffer[k + 2].re, y0_1);
            STOREU_PD(&output_buffer[k + 4].re, y0_2);
            STOREU_PD(&output_buffer[k + 6].re, y0_3);

            STOREU_PD(&output_buffer[k + 0 + seventh].re, y1_0);
            STOREU_PD(&output_buffer[k + 2 + seventh].re, y1_1);
            STOREU_PD(&output_buffer[k + 4 + seventh].re, y1_2);
            STOREU_PD(&output_buffer[k + 6 + seventh].re, y1_3);

            STOREU_PD(&output_buffer[k + 0 + 2 * seventh].re, y2_0);
            STOREU_PD(&output_buffer[k + 2 + 2 * seventh].re, y2_1);
            STOREU_PD(&output_buffer[k + 4 + 2 * seventh].re, y2_2);
            STOREU_PD(&output_buffer[k + 6 + 2 * seventh].re, y2_3);

            STOREU_PD(&output_buffer[k + 0 + 3 * seventh].re, y3_0);
            STOREU_PD(&output_buffer[k + 2 + 3 * seventh].re, y3_1);
            STOREU_PD(&output_buffer[k + 4 + 3 * seventh].re, y3_2);
            STOREU_PD(&output_buffer[k + 6 + 3 * seventh].re, y3_3);

            STOREU_PD(&output_buffer[k + 0 + 4 * seventh].re, y4_0);
            STOREU_PD(&output_buffer[k + 2 + 4 * seventh].re, y4_1);
            STOREU_PD(&output_buffer[k + 4 + 4 * seventh].re, y4_2);
            STOREU_PD(&output_buffer[k + 6 + 4 * seventh].re, y4_3);

            STOREU_PD(&output_buffer[k + 0 + 5 * seventh].re, y5_0);
            STOREU_PD(&output_buffer[k + 2 + 5 * seventh].re, y5_1);
            STOREU_PD(&output_buffer[k + 4 + 5 * seventh].re, y5_2);
            STOREU_PD(&output_buffer[k + 6 + 5 * seventh].re, y5_3);

            STOREU_PD(&output_buffer[k + 0 + 6 * seventh].re, y6_0);
            STOREU_PD(&output_buffer[k + 2 + 6 * seventh].re, y6_1);
            STOREU_PD(&output_buffer[k + 4 + 6 * seventh].re, y6_2);
            STOREU_PD(&output_buffer[k + 6 + 6 * seventh].re, y6_3);
        }

        //------------------------------------------------------------------
        // Cleanup: 2x unrolling
        //------------------------------------------------------------------
        const __m256d rot_mask_final = (transform_sign == 1)
                                           ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                                           : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

        for (; k + 1 < seventh; k += 2)
        {
            if (k + 8 < seventh)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[6 * (k + 8)].re, _MM_HINT_T0);
            }

            // Load inputs (2 butterflies = 1 __m256d per lane)
            __m256d a = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
            __m256d b = load2_aos(&sub_outputs[k + seventh], &sub_outputs[k + seventh + 1]);
            __m256d c = load2_aos(&sub_outputs[k + 2 * seventh], &sub_outputs[k + 2 * seventh + 1]);
            __m256d d = load2_aos(&sub_outputs[k + 3 * seventh], &sub_outputs[k + 3 * seventh + 1]);
            __m256d e = load2_aos(&sub_outputs[k + 4 * seventh], &sub_outputs[k + 4 * seventh + 1]);
            __m256d f = load2_aos(&sub_outputs[k + 5 * seventh], &sub_outputs[k + 5 * seventh + 1]);
            __m256d g = load2_aos(&sub_outputs[k + 6 * seventh], &sub_outputs[k + 6 * seventh + 1]);

            // Load twiddles (2 butterflies)
            __m256d w1 = load2_aos(&stage_tw[6 * k], &stage_tw[6 * (k + 1)]);
            __m256d w2 = load2_aos(&stage_tw[6 * k + 1], &stage_tw[6 * (k + 1) + 1]);
            __m256d w3 = load2_aos(&stage_tw[6 * k + 2], &stage_tw[6 * (k + 1) + 2]);
            __m256d w4 = load2_aos(&stage_tw[6 * k + 3], &stage_tw[6 * (k + 1) + 3]);
            __m256d w5 = load2_aos(&stage_tw[6 * k + 4], &stage_tw[6 * (k + 1) + 4]);
            __m256d w6 = load2_aos(&stage_tw[6 * k + 5], &stage_tw[6 * (k + 1) + 5]);

            // Twiddle multiply
            __m256d b2 = cmul_avx2_aos(b, w1);
            __m256d c2 = cmul_avx2_aos(c, w2);
            __m256d d2 = cmul_avx2_aos(d, w3);
            __m256d e2 = cmul_avx2_aos(e, w4);
            __m256d f2 = cmul_avx2_aos(f, w5);
            __m256d g2 = cmul_avx2_aos(g, w6);

            // Radix-7 butterfly computation
            __m256d t0 = _mm256_add_pd(b2, g2);
            __m256d t1 = _mm256_add_pd(c2, f2);
            __m256d t2 = _mm256_add_pd(d2, e2);
            __m256d t3 = _mm256_sub_pd(b2, g2);
            __m256d t4 = _mm256_sub_pd(c2, f2);
            __m256d t5 = _mm256_sub_pd(d2, e2);

            // Y_0 = a + t0 + t1 + t2
            __m256d y0 = _mm256_add_pd(a, _mm256_add_pd(_mm256_add_pd(t0, t1), t2));

            // Base rotations using FMA
            __m256d base1 = FMADD(vs1, t3, FMADD(vs2, t4, _mm256_mul_pd(vs3, t5)));
            __m256d base2 = FMADD(vs2, t3, FMADD(vs3, t4, _mm256_mul_pd(vs1, t5)));
            __m256d base3 = FMADD(vs3, t3, FMADD(vs1, t4, _mm256_mul_pd(vs2, t5)));

            // Apply ±i rotations
            __m256d base1_swp = _mm256_permute_pd(base1, 0b0101);
            __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);
            __m256d base3_swp = _mm256_permute_pd(base3, 0b0101);

            __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask_final);
            __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask_final);
            __m256d r3 = _mm256_xor_pd(base3_swp, rot_mask_final);

            // Temporary values
            __m256d tmp1 = _mm256_add_pd(a, FMADD(vc1, t0, FMADD(vc2, t1, _mm256_mul_pd(vc3, t2))));
            __m256d tmp2 = _mm256_add_pd(a, FMADD(vc2, t0, FMADD(vc3, t1, _mm256_mul_pd(vc1, t2))));
            __m256d tmp3 = _mm256_add_pd(a, FMADD(vc3, t0, FMADD(vc1, t1, _mm256_mul_pd(vc2, t2))));

            // Final outputs (symmetric pairs)
            __m256d y1 = _mm256_add_pd(tmp1, r1);
            __m256d y6 = _mm256_sub_pd(tmp1, r1);

            __m256d y2 = _mm256_add_pd(tmp2, r2);
            __m256d y5 = _mm256_sub_pd(tmp2, r2);

            __m256d y3 = _mm256_add_pd(tmp3, r3);
            __m256d y4 = _mm256_sub_pd(tmp3, r3);

            // Store results
            STOREU_PD(&output_buffer[k].re, y0);
            STOREU_PD(&output_buffer[k + seventh].re, y1);
            STOREU_PD(&output_buffer[k + 2 * seventh].re, y2);
            STOREU_PD(&output_buffer[k + 3 * seventh].re, y3);
            STOREU_PD(&output_buffer[k + 4 * seventh].re, y4);
            STOREU_PD(&output_buffer[k + 5 * seventh].re, y5);
            STOREU_PD(&output_buffer[k + 6 * seventh].re, y6);
        }
#endif // __AVX2__

        //------------------------------------------------------------------
        // SSE2 TAIL: Single butterfly (keep your scalar code)
        //------------------------------------------------------------------
        for (; k < seventh; ++k)
        {
            const fft_data a = sub_outputs[k];
            const fft_data b = sub_outputs[k + seventh];
            const fft_data c = sub_outputs[k + 2 * seventh];
            const fft_data d = sub_outputs[k + 3 * seventh];
            const fft_data e = sub_outputs[k + 4 * seventh];
            const fft_data f = sub_outputs[k + 5 * seventh];
            const fft_data g = sub_outputs[k + 6 * seventh];

            const fft_data w1 = stage_tw[6 * k];
            const fft_data w2 = stage_tw[6 * k + 1];
            const fft_data w3 = stage_tw[6 * k + 2];
            const fft_data w4 = stage_tw[6 * k + 3];
            const fft_data w5 = stage_tw[6 * k + 4];
            const fft_data w6 = stage_tw[6 * k + 5];

            double b2r = b.re * w1.re - b.im * w1.im, b2i = b.re * w1.im + b.im * w1.re;
            double c2r = c.re * w2.re - c.im * w2.im, c2i = c.re * w2.im + c.im * w2.re;
            double d2r = d.re * w3.re - d.im * w3.im, d2i = d.re * w3.im + d.im * w3.re;
            double e2r = e.re * w4.re - e.im * w4.im, e2i = e.re * w4.im + e.im * w4.re;
            double f2r = f.re * w5.re - f.im * w5.im, f2i = f.re * w5.im + f.im * w5.re;
            double g2r = g.re * w6.re - g.im * w6.im, g2i = g.re * w6.im + g.im * w6.re;

            double t0r = b2r + g2r, t0i = b2i + g2i;
            double t1r = c2r + f2r, t1i = c2i + f2i;
            double t2r = d2r + e2r, t2i = d2i + e2i;
            double t3r = b2r - g2r, t3i = b2i - g2i;
            double t4r = c2r - f2r, t4i = c2i - f2i;
            double t5r = d2r - e2r, t5i = d2i - e2i;

            fft_data y0 = {a.re + (t0r + t1r + t2r), a.im + (t0i + t1i + t2i)};

            double r1r = (transform_sign == 1) ? -(S1 * t3i + S2 * t4i + S3 * t5i) : (S1 * t3i + S2 * t4i + S3 * t5i);
            double r1i = (transform_sign == 1) ? (S1 * t3r + S2 * t4r + S3 * t5r) : -(S1 * t3r + S2 * t4r + S3 * t5r);
            double r2r = (transform_sign == 1) ? -(S2 * t3i + S3 * t4i + S1 * t5i) : (S2 * t3i + S3 * t4i + S1 * t5i);
            double r2i = (transform_sign == 1) ? (S2 * t3r + S3 * t4r + S1 * t5r) : -(S2 * t3r + S3 * t4r + S1 * t5r);
            double r3r = (transform_sign == 1) ? -(S3 * t3i + S1 * t4i + S2 * t5i) : (S3 * t3i + S1 * t4i + S2 * t5i);
            double r3i = (transform_sign == 1) ? (S3 * t3r + S1 * t4r + S2 * t5r) : -(S3 * t3r + S1 * t4r + S2 * t5r);

            double tmp1r = a.re + (C1 * t0r + C2 * t1r + C3 * t2r);
            double tmp1i = a.im + (C1 * t0i + C2 * t1i + C3 * t2i);
            double tmp2r = a.re + (C2 * t0r + C3 * t1r + C1 * t2r);
            double tmp2i = a.im + (C2 * t0i + C3 * t1i + C1 * t2i);
            double tmp3r = a.re + (C3 * t0r + C1 * t1r + C2 * t2r);
            double tmp3i = a.im + (C3 * t0i + C1 * t1i + C2 * t2i);

            output_buffer[k] = y0;
            output_buffer[k + seventh] = (fft_data){tmp1r + r1r, tmp1i + r1i};
            output_buffer[k + 2 * seventh] = (fft_data){tmp2r + r2r, tmp2i + r2i};
            output_buffer[k + 3 * seventh] = (fft_data){tmp3r + r3r, tmp3i + r3i};
            output_buffer[k + 4 * seventh] = (fft_data){tmp3r - r3r, tmp3i - r3i};
            output_buffer[k + 5 * seventh] = (fft_data){tmp2r - r2r, tmp2i - r2i};
            output_buffer[k + 6 * seventh] = (fft_data){tmp1r - r1r, tmp1i - r1i};
        }
    }
    else if (radix == 8)
    {
        //==========================================================================
        // RADIX-8 BUTTERFLY (Split-radix DIT) - FFTW-STYLE OPTIMIZED
        //
        // Pure AoS, no conversions, 4x unrolling for maximum performance.
        //==========================================================================

        const int eighth = sub_len;
        int k = 0;

#ifdef HAS_AVX512
    //----------------------------------------------------------------------
    // AVX-512 PATH: 16x unrolling with cleanup loops
    //----------------------------------------------------------------------
    const __m512d vc_512 = _mm512_set1_pd(C8_1); // √2/2 = 0.707...

    // Precompute rotation masks for ±i
    const __m512d mask_plus_i_512 = _mm512_castsi512_pd(
        _mm512_set_epi64(0x0000000000000000, 0x8000000000000000,
                         0x0000000000000000, 0x8000000000000000,
                         0x0000000000000000, 0x8000000000000000,
                         0x0000000000000000, 0x8000000000000000));
    const __m512d mask_minus_i_512 = _mm512_castsi512_pd(
        _mm512_set_epi64(0x8000000000000000, 0x0000000000000000,
                         0x8000000000000000, 0x0000000000000000,
                         0x8000000000000000, 0x0000000000000000,
                         0x8000000000000000, 0x0000000000000000));
    const __m512d rot_mask_512 = (transform_sign == 1) ? mask_plus_i_512 : mask_minus_i_512;

    //==================================================================
    // DEFINE MACRO ONCE AT THE TOP
    //==================================================================
    #define RADIX8_BUTTERFLY_AVX512(a, b2, c2, d2, e2, f2, g2, h2, y0, y1, y2, y3, y4, y5, y6, y7) \
    {                                                                                              \
        __m512d s0 = _mm512_add_pd(b2, h2), d0 = _mm512_sub_pd(b2, h2);                            \
        __m512d s1 = _mm512_add_pd(c2, g2), d1 = _mm512_sub_pd(c2, g2);                            \
        __m512d s2 = _mm512_add_pd(d2, f2), d2_diff = _mm512_sub_pd(d2, f2);                       \
        __m512d t0 = _mm512_add_pd(a, e2), t4 = _mm512_sub_pd(a, e2);                              \
        y0 = _mm512_add_pd(t0, _mm512_add_pd(_mm512_add_pd(s0, s1), s2));                          \
        y4 = _mm512_add_pd(_mm512_sub_pd(t4, _mm512_add_pd(s0, s1)), s2);                          \
        __m512d base26 = _mm512_sub_pd(d2_diff, d0);                                               \
        __m512d base26_swp = _mm512_permute_pd(base26, 0b01010101);                                \
        __m512d rr26 = _mm512_xor_pd(base26_swp, rot_mask_512);                                    \
        __m512d t02 = _mm512_sub_pd(t0, s1);                                                       \
        y2 = _mm512_add_pd(t02, rr26);                                                             \
        y6 = _mm512_sub_pd(t02, rr26);                                                             \
        __m512d s0ms2 = _mm512_sub_pd(s0, s2);                                                     \
        __m512d real17 = _mm512_fmadd_pd(vc_512, s0ms2, t4);                                       \
        __m512d dd = _mm512_add_pd(d0, d2_diff);                                                   \
        __m512d V17 = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_fmadd_pd(vc_512, dd, d1));         \
        __m512d V17_swp = _mm512_permute_pd(V17, 0b01010101);                                      \
        __m512d rr17 = _mm512_xor_pd(V17_swp, rot_mask_512);                                       \
        y1 = _mm512_add_pd(real17, rr17);                                                          \
        y7 = _mm512_sub_pd(real17, rr17);                                                          \
        __m512d real35 = _mm512_fmsub_pd(vc_512, s0ms2, t4);                                       \
        __m512d dd2 = _mm512_sub_pd(d0, d2_diff);                                                  \
        __m512d V35 = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_fmadd_pd(vc_512, dd2, d1));        \
        __m512d V35_swp = _mm512_permute_pd(V35, 0b01010101);                                      \
        __m512d rr35 = _mm512_xor_pd(V35_swp, rot_mask_512);                                       \
        y3 = _mm512_add_pd(real35, rr35);                                                          \
        y5 = _mm512_sub_pd(real35, rr35);                                                          \
    }

    //==========================================================================
    // Main loop: 16x unrolling
    //==========================================================================
    for (; k + 15 < eighth; k += 16)
    {
        if (k + 32 < eighth)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[7 * (k + 32)].re, _MM_HINT_T0);
        }

        // Load inputs (16 butterflies = 4 loads × 8 lanes)
        __m512d a0 = load4_aos(&sub_outputs[k + 0]);
        __m512d a1 = load4_aos(&sub_outputs[k + 4]);
        __m512d a2 = load4_aos(&sub_outputs[k + 8]);
        __m512d a3 = load4_aos(&sub_outputs[k + 12]);

        __m512d b0 = load4_aos(&sub_outputs[k + 0 + eighth]);
        __m512d b1 = load4_aos(&sub_outputs[k + 4 + eighth]);
        __m512d b2 = load4_aos(&sub_outputs[k + 8 + eighth]);
        __m512d b3 = load4_aos(&sub_outputs[k + 12 + eighth]);

        __m512d c0 = load4_aos(&sub_outputs[k + 0 + 2 * eighth]);
        __m512d c1 = load4_aos(&sub_outputs[k + 4 + 2 * eighth]);
        __m512d c2 = load4_aos(&sub_outputs[k + 8 + 2 * eighth]);
        __m512d c3 = load4_aos(&sub_outputs[k + 12 + 2 * eighth]);

        __m512d d0 = load4_aos(&sub_outputs[k + 0 + 3 * eighth]);
        __m512d d1 = load4_aos(&sub_outputs[k + 4 + 3 * eighth]);
        __m512d d2 = load4_aos(&sub_outputs[k + 8 + 3 * eighth]);
        __m512d d3 = load4_aos(&sub_outputs[k + 12 + 3 * eighth]);

        __m512d e0 = load4_aos(&sub_outputs[k + 0 + 4 * eighth]);
        __m512d e1 = load4_aos(&sub_outputs[k + 4 + 4 * eighth]);
        __m512d e2 = load4_aos(&sub_outputs[k + 8 + 4 * eighth]);
        __m512d e3 = load4_aos(&sub_outputs[k + 12 + 4 * eighth]);

        __m512d f0 = load4_aos(&sub_outputs[k + 0 + 5 * eighth]);
        __m512d f1 = load4_aos(&sub_outputs[k + 4 + 5 * eighth]);
        __m512d f2 = load4_aos(&sub_outputs[k + 8 + 5 * eighth]);
        __m512d f3 = load4_aos(&sub_outputs[k + 12 + 5 * eighth]);

        __m512d g0 = load4_aos(&sub_outputs[k + 0 + 6 * eighth]);
        __m512d g1 = load4_aos(&sub_outputs[k + 4 + 6 * eighth]);
        __m512d g2 = load4_aos(&sub_outputs[k + 8 + 6 * eighth]);
        __m512d g3 = load4_aos(&sub_outputs[k + 12 + 6 * eighth]);

        __m512d h0 = load4_aos(&sub_outputs[k + 0 + 7 * eighth]);
        __m512d h1 = load4_aos(&sub_outputs[k + 4 + 7 * eighth]);
        __m512d h2 = load4_aos(&sub_outputs[k + 8 + 7 * eighth]);
        __m512d h3 = load4_aos(&sub_outputs[k + 12 + 7 * eighth]);

        // Load twiddles
        __m512d w1_0 = load4_aos(&stage_tw[7 * (k + 0)]);
        __m512d w1_1 = load4_aos(&stage_tw[7 * (k + 4)]);
        __m512d w1_2 = load4_aos(&stage_tw[7 * (k + 8)]);
        __m512d w1_3 = load4_aos(&stage_tw[7 * (k + 12)]);

        __m512d w2_0 = load4_aos(&stage_tw[7 * (k + 0) + 1]);
        __m512d w2_1 = load4_aos(&stage_tw[7 * (k + 4) + 1]);
        __m512d w2_2 = load4_aos(&stage_tw[7 * (k + 8) + 1]);
        __m512d w2_3 = load4_aos(&stage_tw[7 * (k + 12) + 1]);

        __m512d w3_0 = load4_aos(&stage_tw[7 * (k + 0) + 2]);
        __m512d w3_1 = load4_aos(&stage_tw[7 * (k + 4) + 2]);
        __m512d w3_2 = load4_aos(&stage_tw[7 * (k + 8) + 2]);
        __m512d w3_3 = load4_aos(&stage_tw[7 * (k + 12) + 2]);

        __m512d w4_0 = load4_aos(&stage_tw[7 * (k + 0) + 3]);
        __m512d w4_1 = load4_aos(&stage_tw[7 * (k + 4) + 3]);
        __m512d w4_2 = load4_aos(&stage_tw[7 * (k + 8) + 3]);
        __m512d w4_3 = load4_aos(&stage_tw[7 * (k + 12) + 3]);

        __m512d w5_0 = load4_aos(&stage_tw[7 * (k + 0) + 4]);
        __m512d w5_1 = load4_aos(&stage_tw[7 * (k + 4) + 4]);
        __m512d w5_2 = load4_aos(&stage_tw[7 * (k + 8) + 4]);
        __m512d w5_3 = load4_aos(&stage_tw[7 * (k + 12) + 4]);

        __m512d w6_0 = load4_aos(&stage_tw[7 * (k + 0) + 5]);
        __m512d w6_1 = load4_aos(&stage_tw[7 * (k + 4) + 5]);
        __m512d w6_2 = load4_aos(&stage_tw[7 * (k + 8) + 5]);
        __m512d w6_3 = load4_aos(&stage_tw[7 * (k + 12) + 5]);

        __m512d w7_0 = load4_aos(&stage_tw[7 * (k + 0) + 6]);
        __m512d w7_1 = load4_aos(&stage_tw[7 * (k + 4) + 6]);
        __m512d w7_2 = load4_aos(&stage_tw[7 * (k + 8) + 6]);
        __m512d w7_3 = load4_aos(&stage_tw[7 * (k + 12) + 6]);

        // Twiddle multiply
        __m512d b2_0 = cmul_avx512_aos(b0, w1_0), b2_1 = cmul_avx512_aos(b1, w1_1);
        __m512d b2_2 = cmul_avx512_aos(b2, w1_2), b2_3 = cmul_avx512_aos(b3, w1_3);

        __m512d c2_0 = cmul_avx512_aos(c0, w2_0), c2_1 = cmul_avx512_aos(c1, w2_1);
        __m512d c2_2 = cmul_avx512_aos(c2, w2_2), c2_3 = cmul_avx512_aos(c3, w2_3);

        __m512d d2_0 = cmul_avx512_aos(d0, w3_0), d2_1 = cmul_avx512_aos(d1, w3_1);
        __m512d d2_2 = cmul_avx512_aos(d2, w3_2), d2_3 = cmul_avx512_aos(d3, w3_3);

        __m512d e2_0 = cmul_avx512_aos(e0, w4_0), e2_1 = cmul_avx512_aos(e1, w4_1);
        __m512d e2_2 = cmul_avx512_aos(e2, w4_2), e2_3 = cmul_avx512_aos(e3, w4_3);

        __m512d f2_0 = cmul_avx512_aos(f0, w5_0), f2_1 = cmul_avx512_aos(f1, w5_1);
        __m512d f2_2 = cmul_avx512_aos(f2, w5_2), f2_3 = cmul_avx512_aos(f3, w5_3);

        __m512d g2_0 = cmul_avx512_aos(g0, w6_0), g2_1 = cmul_avx512_aos(g1, w6_1);
        __m512d g2_2 = cmul_avx512_aos(g2, w6_2), g2_3 = cmul_avx512_aos(g3, w6_3);

        __m512d h2_0 = cmul_avx512_aos(h0, w7_0), h2_1 = cmul_avx512_aos(h1, w7_1);
        __m512d h2_2 = cmul_avx512_aos(h2, w7_2), h2_3 = cmul_avx512_aos(h3, w7_3);

        // Butterfly
        __m512d y0_0, y1_0, y2_0, y3_0, y4_0, y5_0, y6_0, y7_0;
        __m512d y0_1, y1_1, y2_1, y3_1, y4_1, y5_1, y6_1, y7_1;
        __m512d y0_2, y1_2, y2_2, y3_2, y4_2, y5_2, y6_2, y7_2;
        __m512d y0_3, y1_3, y2_3, y3_3, y4_3, y5_3, y6_3, y7_3;

        RADIX8_BUTTERFLY_AVX512(a0, b2_0, c2_0, d2_0, e2_0, f2_0, g2_0, h2_0,
                                y0_0, y1_0, y2_0, y3_0, y4_0, y5_0, y6_0, y7_0);
        RADIX8_BUTTERFLY_AVX512(a1, b2_1, c2_1, d2_1, e2_1, f2_1, g2_1, h2_1,
                                y0_1, y1_1, y2_1, y3_1, y4_1, y5_1, y6_1, y7_1);
        RADIX8_BUTTERFLY_AVX512(a2, b2_2, c2_2, d2_2, e2_2, f2_2, g2_2, h2_2,
                                y0_2, y1_2, y2_2, y3_2, y4_2, y5_2, y6_2, y7_2);
        RADIX8_BUTTERFLY_AVX512(a3, b2_3, c2_3, d2_3, e2_3, f2_3, g2_3, h2_3,
                                y0_3, y1_3, y2_3, y3_3, y4_3, y5_3, y6_3, y7_3);

        // Store results
        STOREU_PD512(&output_buffer[k + 0].re, y0_0);
        STOREU_PD512(&output_buffer[k + 4].re, y0_1);
        STOREU_PD512(&output_buffer[k + 8].re, y0_2);
        STOREU_PD512(&output_buffer[k + 12].re, y0_3);

        STOREU_PD512(&output_buffer[k + 0 + eighth].re, y1_0);
        STOREU_PD512(&output_buffer[k + 4 + eighth].re, y1_1);
        STOREU_PD512(&output_buffer[k + 8 + eighth].re, y1_2);
        STOREU_PD512(&output_buffer[k + 12 + eighth].re, y1_3);

        STOREU_PD512(&output_buffer[k + 0 + 2 * eighth].re, y2_0);
        STOREU_PD512(&output_buffer[k + 4 + 2 * eighth].re, y2_1);
        STOREU_PD512(&output_buffer[k + 8 + 2 * eighth].re, y2_2);
        STOREU_PD512(&output_buffer[k + 12 + 2 * eighth].re, y2_3);

        STOREU_PD512(&output_buffer[k + 0 + 3 * eighth].re, y3_0);
        STOREU_PD512(&output_buffer[k + 4 + 3 * eighth].re, y3_1);
        STOREU_PD512(&output_buffer[k + 8 + 3 * eighth].re, y3_2);
        STOREU_PD512(&output_buffer[k + 12 + 3 * eighth].re, y3_3);

        STOREU_PD512(&output_buffer[k + 0 + 4 * eighth].re, y4_0);
        STOREU_PD512(&output_buffer[k + 4 + 4 * eighth].re, y4_1);
        STOREU_PD512(&output_buffer[k + 8 + 4 * eighth].re, y4_2);
        STOREU_PD512(&output_buffer[k + 12 + 4 * eighth].re, y4_3);

        STOREU_PD512(&output_buffer[k + 0 + 5 * eighth].re, y5_0);
        STOREU_PD512(&output_buffer[k + 4 + 5 * eighth].re, y5_1);
        STOREU_PD512(&output_buffer[k + 8 + 5 * eighth].re, y5_2);
        STOREU_PD512(&output_buffer[k + 12 + 5 * eighth].re, y5_3);

        STOREU_PD512(&output_buffer[k + 0 + 6 * eighth].re, y6_0);
        STOREU_PD512(&output_buffer[k + 4 + 6 * eighth].re, y6_1);
        STOREU_PD512(&output_buffer[k + 8 + 6 * eighth].re, y6_2);
        STOREU_PD512(&output_buffer[k + 12 + 6 * eighth].re, y6_3);

        STOREU_PD512(&output_buffer[k + 0 + 7 * eighth].re, y7_0);
        STOREU_PD512(&output_buffer[k + 4 + 7 * eighth].re, y7_1);
        STOREU_PD512(&output_buffer[k + 8 + 7 * eighth].re, y7_2);
        STOREU_PD512(&output_buffer[k + 12 + 7 * eighth].re, y7_3);
    }

    //==========================================================================
    // Cleanup: 8x unrolling (process 8 butterflies)
    //==========================================================================
    for (; k + 7 < eighth; k += 8)
    {
        // Load inputs (8 butterflies = 2 loads × 8 lanes)
        __m512d a0 = load4_aos(&sub_outputs[k + 0]);
        __m512d a1 = load4_aos(&sub_outputs[k + 4]);

        __m512d b0 = load4_aos(&sub_outputs[k + 0 + eighth]);
        __m512d b1 = load4_aos(&sub_outputs[k + 4 + eighth]);

        __m512d c0 = load4_aos(&sub_outputs[k + 0 + 2 * eighth]);
        __m512d c1 = load4_aos(&sub_outputs[k + 4 + 2 * eighth]);

        __m512d d0 = load4_aos(&sub_outputs[k + 0 + 3 * eighth]);
        __m512d d1 = load4_aos(&sub_outputs[k + 4 + 3 * eighth]);

        __m512d e0 = load4_aos(&sub_outputs[k + 0 + 4 * eighth]);
        __m512d e1 = load4_aos(&sub_outputs[k + 4 + 4 * eighth]);

        __m512d f0 = load4_aos(&sub_outputs[k + 0 + 5 * eighth]);
        __m512d f1 = load4_aos(&sub_outputs[k + 4 + 5 * eighth]);

        __m512d g0 = load4_aos(&sub_outputs[k + 0 + 6 * eighth]);
        __m512d g1 = load4_aos(&sub_outputs[k + 4 + 6 * eighth]);

        __m512d h0 = load4_aos(&sub_outputs[k + 0 + 7 * eighth]);
        __m512d h1 = load4_aos(&sub_outputs[k + 4 + 7 * eighth]);

        // Load twiddles
        __m512d w1_0 = load4_aos(&stage_tw[7 * (k + 0)]);
        __m512d w1_1 = load4_aos(&stage_tw[7 * (k + 4)]);

        __m512d w2_0 = load4_aos(&stage_tw[7 * (k + 0) + 1]);
        __m512d w2_1 = load4_aos(&stage_tw[7 * (k + 4) + 1]);

        __m512d w3_0 = load4_aos(&stage_tw[7 * (k + 0) + 2]);
        __m512d w3_1 = load4_aos(&stage_tw[7 * (k + 4) + 2]);

        __m512d w4_0 = load4_aos(&stage_tw[7 * (k + 0) + 3]);
        __m512d w4_1 = load4_aos(&stage_tw[7 * (k + 4) + 3]);

        __m512d w5_0 = load4_aos(&stage_tw[7 * (k + 0) + 4]);
        __m512d w5_1 = load4_aos(&stage_tw[7 * (k + 4) + 4]);

        __m512d w6_0 = load4_aos(&stage_tw[7 * (k + 0) + 5]);
        __m512d w6_1 = load4_aos(&stage_tw[7 * (k + 4) + 5]);

        __m512d w7_0 = load4_aos(&stage_tw[7 * (k + 0) + 6]);
        __m512d w7_1 = load4_aos(&stage_tw[7 * (k + 4) + 6]);

        // Twiddle multiply
        __m512d b2_0 = cmul_avx512_aos(b0, w1_0), b2_1 = cmul_avx512_aos(b1, w1_1);
        __m512d c2_0 = cmul_avx512_aos(c0, w2_0), c2_1 = cmul_avx512_aos(c1, w2_1);
        __m512d d2_0 = cmul_avx512_aos(d0, w3_0), d2_1 = cmul_avx512_aos(d1, w3_1);
        __m512d e2_0 = cmul_avx512_aos(e0, w4_0), e2_1 = cmul_avx512_aos(e1, w4_1);
        __m512d f2_0 = cmul_avx512_aos(f0, w5_0), f2_1 = cmul_avx512_aos(f1, w5_1);
        __m512d g2_0 = cmul_avx512_aos(g0, w6_0), g2_1 = cmul_avx512_aos(g1, w6_1);
        __m512d h2_0 = cmul_avx512_aos(h0, w7_0), h2_1 = cmul_avx512_aos(h1, w7_1);

        // Butterfly
        __m512d y0_0, y1_0, y2_0, y3_0, y4_0, y5_0, y6_0, y7_0;
        __m512d y0_1, y1_1, y2_1, y3_1, y4_1, y5_1, y6_1, y7_1;

        RADIX8_BUTTERFLY_AVX512(a0, b2_0, c2_0, d2_0, e2_0, f2_0, g2_0, h2_0,
                                y0_0, y1_0, y2_0, y3_0, y4_0, y5_0, y6_0, y7_0);
        RADIX8_BUTTERFLY_AVX512(a1, b2_1, c2_1, d2_1, e2_1, f2_1, g2_1, h2_1,
                                y0_1, y1_1, y2_1, y3_1, y4_1, y5_1, y6_1, y7_1);

        // Store results
        STOREU_PD512(&output_buffer[k + 0].re, y0_0);
        STOREU_PD512(&output_buffer[k + 4].re, y0_1);

        STOREU_PD512(&output_buffer[k + 0 + eighth].re, y1_0);
        STOREU_PD512(&output_buffer[k + 4 + eighth].re, y1_1);

        STOREU_PD512(&output_buffer[k + 0 + 2 * eighth].re, y2_0);
        STOREU_PD512(&output_buffer[k + 4 + 2 * eighth].re, y2_1);

        STOREU_PD512(&output_buffer[k + 0 + 3 * eighth].re, y3_0);
        STOREU_PD512(&output_buffer[k + 4 + 3 * eighth].re, y3_1);

        STOREU_PD512(&output_buffer[k + 0 + 4 * eighth].re, y4_0);
        STOREU_PD512(&output_buffer[k + 4 + 4 * eighth].re, y4_1);

        STOREU_PD512(&output_buffer[k + 0 + 5 * eighth].re, y5_0);
        STOREU_PD512(&output_buffer[k + 4 + 5 * eighth].re, y5_1);

        STOREU_PD512(&output_buffer[k + 0 + 6 * eighth].re, y6_0);
        STOREU_PD512(&output_buffer[k + 4 + 6 * eighth].re, y6_1);

        STOREU_PD512(&output_buffer[k + 0 + 7 * eighth].re, y7_0);
        STOREU_PD512(&output_buffer[k + 4 + 7 * eighth].re, y7_1);
    }

    //==========================================================================
    // Cleanup: 4x unrolling (process 4 butterflies)
    //==========================================================================
    for (; k + 3 < eighth; k += 4)
    {
        // Load inputs (4 butterflies = 1 load × 8 lanes)
        __m512d a = load4_aos(&sub_outputs[k]);
        __m512d b = load4_aos(&sub_outputs[k + eighth]);
        __m512d c = load4_aos(&sub_outputs[k + 2 * eighth]);
        __m512d d = load4_aos(&sub_outputs[k + 3 * eighth]);
        __m512d e = load4_aos(&sub_outputs[k + 4 * eighth]);
        __m512d f = load4_aos(&sub_outputs[k + 5 * eighth]);
        __m512d g = load4_aos(&sub_outputs[k + 6 * eighth]);
        __m512d h = load4_aos(&sub_outputs[k + 7 * eighth]);

        // Load twiddles
        __m512d w1 = load4_aos(&stage_tw[7 * k]);
        __m512d w2 = load4_aos(&stage_tw[7 * k + 1]);
        __m512d w3 = load4_aos(&stage_tw[7 * k + 2]);
        __m512d w4 = load4_aos(&stage_tw[7 * k + 3]);
        __m512d w5 = load4_aos(&stage_tw[7 * k + 4]);
        __m512d w6 = load4_aos(&stage_tw[7 * k + 5]);
        __m512d w7 = load4_aos(&stage_tw[7 * k + 6]);

        // Twiddle multiply
        __m512d b2 = cmul_avx512_aos(b, w1);
        __m512d c2 = cmul_avx512_aos(c, w2);
        __m512d d2 = cmul_avx512_aos(d, w3);
        __m512d e2 = cmul_avx512_aos(e, w4);
        __m512d f2 = cmul_avx512_aos(f, w5);
        __m512d g2 = cmul_avx512_aos(g, w6);
        __m512d h2 = cmul_avx512_aos(h, w7);

        // Butterfly
        __m512d y0, y1, y2, y3, y4, y5, y6, y7;
        RADIX8_BUTTERFLY_AVX512(a, b2, c2, d2, e2, f2, g2, h2,
                                y0, y1, y2, y3, y4, y5, y6, y7);

        // Store results
        STOREU_PD512(&output_buffer[k].re, y0);
        STOREU_PD512(&output_buffer[k + eighth].re, y1);
        STOREU_PD512(&output_buffer[k + 2 * eighth].re, y2);
        STOREU_PD512(&output_buffer[k + 3 * eighth].re, y3);
        STOREU_PD512(&output_buffer[k + 4 * eighth].re, y4);
        STOREU_PD512(&output_buffer[k + 5 * eighth].re, y5);
        STOREU_PD512(&output_buffer[k + 6 * eighth].re, y6);
        STOREU_PD512(&output_buffer[k + 7 * eighth].re, y7);
    }

    // UNDEFINE MACRO at the end
    #undef RADIX8_BUTTERFLY_AVX512
#endif // HAS_AVX512

#ifdef __AVX2__
        //------------------------------------------------------------------
        // AVX2 PATH: 4x unrolled, pure AoS
        //------------------------------------------------------------------
        const __m256d vc = _mm256_set1_pd(C8_1); // √2/2

        // Precompute rotation masks
        const __m256d mask_plus_i = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
        const __m256d mask_minus_i = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
        const __m256d rot_mask = (transform_sign == 1) ? mask_plus_i : mask_minus_i;

        for (; k + 7 < eighth; k += 8)
        {
            // Prefetch ahead
            if (k + 16 < eighth)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[7 * (k + 16)].re, _MM_HINT_T0);
            }

            //==================================================================
            // Load inputs (8 butterflies = 4 AVX2 loads per lane × 8 lanes)
            // This is a LOT of loads but unavoidable for radix-8
            //==================================================================
            __m256d a0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
            __m256d a1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
            __m256d a2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
            __m256d a3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

            __m256d b0 = load2_aos(&sub_outputs[k + 0 + eighth], &sub_outputs[k + 1 + eighth]);
            __m256d b1 = load2_aos(&sub_outputs[k + 2 + eighth], &sub_outputs[k + 3 + eighth]);
            __m256d b2 = load2_aos(&sub_outputs[k + 4 + eighth], &sub_outputs[k + 5 + eighth]);
            __m256d b3 = load2_aos(&sub_outputs[k + 6 + eighth], &sub_outputs[k + 7 + eighth]);

            __m256d c0 = load2_aos(&sub_outputs[k + 0 + 2 * eighth], &sub_outputs[k + 1 + 2 * eighth]);
            __m256d c1 = load2_aos(&sub_outputs[k + 2 + 2 * eighth], &sub_outputs[k + 3 + 2 * eighth]);
            __m256d c2 = load2_aos(&sub_outputs[k + 4 + 2 * eighth], &sub_outputs[k + 5 + 2 * eighth]);
            __m256d c3 = load2_aos(&sub_outputs[k + 6 + 2 * eighth], &sub_outputs[k + 7 + 2 * eighth]);

            __m256d d0 = load2_aos(&sub_outputs[k + 0 + 3 * eighth], &sub_outputs[k + 1 + 3 * eighth]);
            __m256d d1 = load2_aos(&sub_outputs[k + 2 + 3 * eighth], &sub_outputs[k + 3 + 3 * eighth]);
            __m256d d2 = load2_aos(&sub_outputs[k + 4 + 3 * eighth], &sub_outputs[k + 5 + 3 * eighth]);
            __m256d d3 = load2_aos(&sub_outputs[k + 6 + 3 * eighth], &sub_outputs[k + 7 + 3 * eighth]);

            __m256d e0 = load2_aos(&sub_outputs[k + 0 + 4 * eighth], &sub_outputs[k + 1 + 4 * eighth]);
            __m256d e1 = load2_aos(&sub_outputs[k + 2 + 4 * eighth], &sub_outputs[k + 3 + 4 * eighth]);
            __m256d e2 = load2_aos(&sub_outputs[k + 4 + 4 * eighth], &sub_outputs[k + 5 + 4 * eighth]);
            __m256d e3 = load2_aos(&sub_outputs[k + 6 + 4 * eighth], &sub_outputs[k + 7 + 4 * eighth]);

            __m256d f0 = load2_aos(&sub_outputs[k + 0 + 5 * eighth], &sub_outputs[k + 1 + 5 * eighth]);
            __m256d f1 = load2_aos(&sub_outputs[k + 2 + 5 * eighth], &sub_outputs[k + 3 + 5 * eighth]);
            __m256d f2 = load2_aos(&sub_outputs[k + 4 + 5 * eighth], &sub_outputs[k + 5 + 5 * eighth]);
            __m256d f3 = load2_aos(&sub_outputs[k + 6 + 5 * eighth], &sub_outputs[k + 7 + 5 * eighth]);

            __m256d g0 = load2_aos(&sub_outputs[k + 0 + 6 * eighth], &sub_outputs[k + 1 + 6 * eighth]);
            __m256d g1 = load2_aos(&sub_outputs[k + 2 + 6 * eighth], &sub_outputs[k + 3 + 6 * eighth]);
            __m256d g2 = load2_aos(&sub_outputs[k + 4 + 6 * eighth], &sub_outputs[k + 5 + 6 * eighth]);
            __m256d g3 = load2_aos(&sub_outputs[k + 6 + 6 * eighth], &sub_outputs[k + 7 + 6 * eighth]);

            __m256d h0 = load2_aos(&sub_outputs[k + 0 + 7 * eighth], &sub_outputs[k + 1 + 7 * eighth]);
            __m256d h1 = load2_aos(&sub_outputs[k + 2 + 7 * eighth], &sub_outputs[k + 3 + 7 * eighth]);
            __m256d h2 = load2_aos(&sub_outputs[k + 4 + 7 * eighth], &sub_outputs[k + 5 + 7 * eighth]);
            __m256d h3 = load2_aos(&sub_outputs[k + 6 + 7 * eighth], &sub_outputs[k + 7 + 7 * eighth]);

            //==================================================================
            // Load twiddles (k-major: 7 per butterfly)
            //==================================================================
            __m256d w1_0 = load2_aos(&stage_tw[7 * (k + 0)], &stage_tw[7 * (k + 1)]);
            __m256d w1_1 = load2_aos(&stage_tw[7 * (k + 2)], &stage_tw[7 * (k + 3)]);
            __m256d w1_2 = load2_aos(&stage_tw[7 * (k + 4)], &stage_tw[7 * (k + 5)]);
            __m256d w1_3 = load2_aos(&stage_tw[7 * (k + 6)], &stage_tw[7 * (k + 7)]);

            __m256d w2_0 = load2_aos(&stage_tw[7 * (k + 0) + 1], &stage_tw[7 * (k + 1) + 1]);
            __m256d w2_1 = load2_aos(&stage_tw[7 * (k + 2) + 1], &stage_tw[7 * (k + 3) + 1]);
            __m256d w2_2 = load2_aos(&stage_tw[7 * (k + 4) + 1], &stage_tw[7 * (k + 5) + 1]);
            __m256d w2_3 = load2_aos(&stage_tw[7 * (k + 6) + 1], &stage_tw[7 * (k + 7) + 1]);

            __m256d w3_0 = load2_aos(&stage_tw[7 * (k + 0) + 2], &stage_tw[7 * (k + 1) + 2]);
            __m256d w3_1 = load2_aos(&stage_tw[7 * (k + 2) + 2], &stage_tw[7 * (k + 3) + 2]);
            __m256d w3_2 = load2_aos(&stage_tw[7 * (k + 4) + 2], &stage_tw[7 * (k + 5) + 2]);
            __m256d w3_3 = load2_aos(&stage_tw[7 * (k + 6) + 2], &stage_tw[7 * (k + 7) + 2]);

            __m256d w4_0 = load2_aos(&stage_tw[7 * (k + 0) + 3], &stage_tw[7 * (k + 1) + 3]);
            __m256d w4_1 = load2_aos(&stage_tw[7 * (k + 2) + 3], &stage_tw[7 * (k + 3) + 3]);
            __m256d w4_2 = load2_aos(&stage_tw[7 * (k + 4) + 3], &stage_tw[7 * (k + 5) + 3]);
            __m256d w4_3 = load2_aos(&stage_tw[7 * (k + 6) + 3], &stage_tw[7 * (k + 7) + 3]);

            __m256d w5_0 = load2_aos(&stage_tw[7 * (k + 0) + 4], &stage_tw[7 * (k + 1) + 4]);
            __m256d w5_1 = load2_aos(&stage_tw[7 * (k + 2) + 4], &stage_tw[7 * (k + 3) + 4]);
            __m256d w5_2 = load2_aos(&stage_tw[7 * (k + 4) + 4], &stage_tw[7 * (k + 5) + 4]);
            __m256d w5_3 = load2_aos(&stage_tw[7 * (k + 6) + 4], &stage_tw[7 * (k + 7) + 4]);

            __m256d w6_0 = load2_aos(&stage_tw[7 * (k + 0) + 5], &stage_tw[7 * (k + 1) + 5]);
            __m256d w6_1 = load2_aos(&stage_tw[7 * (k + 2) + 5], &stage_tw[7 * (k + 3) + 5]);
            __m256d w6_2 = load2_aos(&stage_tw[7 * (k + 4) + 5], &stage_tw[7 * (k + 5) + 5]);
            __m256d w6_3 = load2_aos(&stage_tw[7 * (k + 6) + 5], &stage_tw[7 * (k + 7) + 5]);

            __m256d w7_0 = load2_aos(&stage_tw[7 * (k + 0) + 6], &stage_tw[7 * (k + 1) + 6]);
            __m256d w7_1 = load2_aos(&stage_tw[7 * (k + 2) + 6], &stage_tw[7 * (k + 3) + 6]);
            __m256d w7_2 = load2_aos(&stage_tw[7 * (k + 4) + 6], &stage_tw[7 * (k + 5) + 6]);
            __m256d w7_3 = load2_aos(&stage_tw[7 * (k + 6) + 6], &stage_tw[7 * (k + 7) + 6]);

            //==================================================================
            // Twiddle multiply
            //==================================================================
            __m256d b2_0 = cmul_avx2_aos(b0, w1_0), b2_1 = cmul_avx2_aos(b1, w1_1);
            __m256d b2_2 = cmul_avx2_aos(b2, w1_2), b2_3 = cmul_avx2_aos(b3, w1_3);

            __m256d c2_0 = cmul_avx2_aos(c0, w2_0), c2_1 = cmul_avx2_aos(c1, w2_1);
            __m256d c2_2 = cmul_avx2_aos(c2, w2_2), c2_3 = cmul_avx2_aos(c3, w2_3);

            __m256d d2_0 = cmul_avx2_aos(d0, w3_0), d2_1 = cmul_avx2_aos(d1, w3_1);
            __m256d d2_2 = cmul_avx2_aos(d2, w3_2), d2_3 = cmul_avx2_aos(d3, w3_3);

            __m256d e2_0 = cmul_avx2_aos(e0, w4_0), e2_1 = cmul_avx2_aos(e1, w4_1);
            __m256d e2_2 = cmul_avx2_aos(e2, w4_2), e2_3 = cmul_avx2_aos(e3, w4_3);

            __m256d f2_0 = cmul_avx2_aos(f0, w5_0), f2_1 = cmul_avx2_aos(f1, w5_1);
            __m256d f2_2 = cmul_avx2_aos(f2, w5_2), f2_3 = cmul_avx2_aos(f3, w5_3);

            __m256d g2_0 = cmul_avx2_aos(g0, w6_0), g2_1 = cmul_avx2_aos(g1, w6_1);
            __m256d g2_2 = cmul_avx2_aos(g2, w6_2), g2_3 = cmul_avx2_aos(g3, w6_3);

            __m256d h2_0 = cmul_avx2_aos(h0, w7_0), h2_1 = cmul_avx2_aos(h1, w7_1);
            __m256d h2_2 = cmul_avx2_aos(h2, w7_2), h2_3 = cmul_avx2_aos(h3, w7_3);

//==================================================================
// Radix-8 butterfly (8 butterflies in parallel)
// Using macro to reduce code duplication
//==================================================================
#define RADIX8_BUTTERFLY_AVX2(a, b2, c2, d2, e2, f2, g2, h2, y0, y1, y2, y3, y4, y5, y6, y7) \
    {                                                                                        \
        __m256d s0 = _mm256_add_pd(b2, h2), d0 = _mm256_sub_pd(b2, h2);                      \
        __m256d s1 = _mm256_add_pd(c2, g2), d1 = _mm256_sub_pd(c2, g2);                      \
        __m256d s2 = _mm256_add_pd(d2, f2), d2_diff = _mm256_sub_pd(d2, f2);                 \
        __m256d t0 = _mm256_add_pd(a, e2), t4 = _mm256_sub_pd(a, e2);                        \
        y0 = _mm256_add_pd(t0, _mm256_add_pd(_mm256_add_pd(s0, s1), s2));                    \
        y4 = _mm256_add_pd(_mm256_sub_pd(t4, _mm256_add_pd(s0, s1)), s2);                    \
        __m256d base26 = _mm256_sub_pd(d2_diff, d0);                                         \
        __m256d base26_swp = _mm256_permute_pd(base26, 0b0101);                              \
        __m256d rr26 = _mm256_xor_pd(base26_swp, rot_mask);                                  \
        __m256d t02 = _mm256_sub_pd(t0, s1);                                                 \
        y2 = _mm256_add_pd(t02, rr26);                                                       \
        y6 = _mm256_sub_pd(t02, rr26);                                                       \
        __m256d s0ms2 = _mm256_sub_pd(s0, s2);                                               \
        __m256d real17 = FMADD(vc, s0ms2, t4);                                               \
        __m256d dd = _mm256_add_pd(d0, d2_diff);                                             \
        __m256d V17 = _mm256_sub_pd(_mm256_setzero_pd(), FMADD(vc, dd, d1));                 \
        __m256d V17_swp = _mm256_permute_pd(V17, 0b0101);                                    \
        __m256d rr17 = _mm256_xor_pd(V17_swp, rot_mask);                                     \
        y1 = _mm256_add_pd(real17, rr17);                                                    \
        y7 = _mm256_sub_pd(real17, rr17);                                                    \
        __m256d real35 = FMSUB(vc, s0ms2, t4);                                               \
        __m256d dd2 = _mm256_sub_pd(d0, d2_diff);                                            \
        __m256d V35 = _mm256_sub_pd(_mm256_setzero_pd(), FMADD(vc, dd2, d1));                \
        __m256d V35_swp = _mm256_permute_pd(V35, 0b0101);                                    \
        __m256d rr35 = _mm256_xor_pd(V35_swp, rot_mask);                                     \
        y3 = _mm256_add_pd(real35, rr35);                                                    \
        y5 = _mm256_sub_pd(real35, rr35);                                                    \
    }

            __m256d y0_0, y1_0, y2_0, y3_0, y4_0, y5_0, y6_0, y7_0;
            __m256d y0_1, y1_1, y2_1, y3_1, y4_1, y5_1, y6_1, y7_1;
            __m256d y0_2, y1_2, y2_2, y3_2, y4_2, y5_2, y6_2, y7_2;
            __m256d y0_3, y1_3, y2_3, y3_3, y4_3, y5_3, y6_3, y7_3;

            RADIX8_BUTTERFLY_AVX2(a0, b2_0, c2_0, d2_0, e2_0, f2_0, g2_0, h2_0,
                                  y0_0, y1_0, y2_0, y3_0, y4_0, y5_0, y6_0, y7_0);
            RADIX8_BUTTERFLY_AVX2(a1, b2_1, c2_1, d2_1, e2_1, f2_1, g2_1, h2_1,
                                  y0_1, y1_1, y2_1, y3_1, y4_1, y5_1, y6_1, y7_1);
            RADIX8_BUTTERFLY_AVX2(a2, b2_2, c2_2, d2_2, e2_2, f2_2, g2_2, h2_2,
                                  y0_2, y1_2, y2_2, y3_2, y4_2, y5_2, y6_2, y7_2);
            RADIX8_BUTTERFLY_AVX2(a3, b2_3, c2_3, d2_3, e2_3, f2_3, g2_3, h2_3,
                                  y0_3, y1_3, y2_3, y3_3, y4_3, y5_3, y6_3, y7_3);

            //==================================================================
            // Store results (pure AoS!)
            //==================================================================
            STOREU_PD(&output_buffer[k + 0].re, y0_0);
            STOREU_PD(&output_buffer[k + 2].re, y0_1);
            STOREU_PD(&output_buffer[k + 4].re, y0_2);
            STOREU_PD(&output_buffer[k + 6].re, y0_3);

            STOREU_PD(&output_buffer[k + 0 + eighth].re, y1_0);
            STOREU_PD(&output_buffer[k + 2 + eighth].re, y1_1);
            STOREU_PD(&output_buffer[k + 4 + eighth].re, y1_2);
            STOREU_PD(&output_buffer[k + 6 + eighth].re, y1_3);

            STOREU_PD(&output_buffer[k + 0 + 2 * eighth].re, y2_0);
            STOREU_PD(&output_buffer[k + 2 + 2 * eighth].re, y2_1);
            STOREU_PD(&output_buffer[k + 4 + 2 * eighth].re, y2_2);
            STOREU_PD(&output_buffer[k + 6 + 2 * eighth].re, y2_3);

            STOREU_PD(&output_buffer[k + 0 + 3 * eighth].re, y3_0);
            STOREU_PD(&output_buffer[k + 2 + 3 * eighth].re, y3_1);
            STOREU_PD(&output_buffer[k + 4 + 3 * eighth].re, y3_2);
            STOREU_PD(&output_buffer[k + 6 + 3 * eighth].re, y3_3);

            STOREU_PD(&output_buffer[k + 0 + 4 * eighth].re, y4_0);
            STOREU_PD(&output_buffer[k + 2 + 4 * eighth].re, y4_1);
            STOREU_PD(&output_buffer[k + 4 + 4 * eighth].re, y4_2);
            STOREU_PD(&output_buffer[k + 6 + 4 * eighth].re, y4_3);

            STOREU_PD(&output_buffer[k + 0 + 5 * eighth].re, y5_0);
            STOREU_PD(&output_buffer[k + 2 + 5 * eighth].re, y5_1);
            STOREU_PD(&output_buffer[k + 4 + 5 * eighth].re, y5_2);
            STOREU_PD(&output_buffer[k + 6 + 5 * eighth].re, y5_3);

            STOREU_PD(&output_buffer[k + 0 + 6 * eighth].re, y6_0);
            STOREU_PD(&output_buffer[k + 2 + 6 * eighth].re, y6_1);
            STOREU_PD(&output_buffer[k + 4 + 6 * eighth].re, y6_2);
            STOREU_PD(&output_buffer[k + 6 + 6 * eighth].re, y6_3);

            STOREU_PD(&output_buffer[k + 0 + 7 * eighth].re, y7_0);
            STOREU_PD(&output_buffer[k + 2 + 7 * eighth].re, y7_1);
            STOREU_PD(&output_buffer[k + 4 + 7 * eighth].re, y7_2);
            STOREU_PD(&output_buffer[k + 6 + 7 * eighth].re, y7_3);
        }

        //------------------------------------------------------------------
        // Cleanup: 2x unrolling for remaining butterflies
        //------------------------------------------------------------------
        for (; k + 1 < eighth; k += 2)
        {
            if (k + 8 < eighth)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[7 * (k + 8)].re, _MM_HINT_T0);
            }

            // Load inputs (2 butterflies)
            __m256d a = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
            __m256d b = load2_aos(&sub_outputs[k + eighth], &sub_outputs[k + eighth + 1]);
            __m256d c = load2_aos(&sub_outputs[k + 2 * eighth], &sub_outputs[k + 2 * eighth + 1]);
            __m256d d = load2_aos(&sub_outputs[k + 3 * eighth], &sub_outputs[k + 3 * eighth + 1]);
            __m256d e = load2_aos(&sub_outputs[k + 4 * eighth], &sub_outputs[k + 4 * eighth + 1]);
            __m256d f = load2_aos(&sub_outputs[k + 5 * eighth], &sub_outputs[k + 5 * eighth + 1]);
            __m256d g = load2_aos(&sub_outputs[k + 6 * eighth], &sub_outputs[k + 6 * eighth + 1]);
            __m256d h = load2_aos(&sub_outputs[k + 7 * eighth], &sub_outputs[k + 7 * eighth + 1]);

            // Load twiddles (2 butterflies)
            __m256d w1 = load2_aos(&stage_tw[7 * k], &stage_tw[7 * (k + 1)]);
            __m256d w2 = load2_aos(&stage_tw[7 * k + 1], &stage_tw[7 * (k + 1) + 1]);
            __m256d w3 = load2_aos(&stage_tw[7 * k + 2], &stage_tw[7 * (k + 1) + 2]);
            __m256d w4 = load2_aos(&stage_tw[7 * k + 3], &stage_tw[7 * (k + 1) + 3]);
            __m256d w5 = load2_aos(&stage_tw[7 * k + 4], &stage_tw[7 * (k + 1) + 4]);
            __m256d w6 = load2_aos(&stage_tw[7 * k + 5], &stage_tw[7 * (k + 1) + 5]);
            __m256d w7 = load2_aos(&stage_tw[7 * k + 6], &stage_tw[7 * (k + 1) + 6]);

            // Twiddle multiply
            __m256d b2 = cmul_avx2_aos(b, w1);
            __m256d c2 = cmul_avx2_aos(c, w2);
            __m256d d2 = cmul_avx2_aos(d, w3);
            __m256d e2 = cmul_avx2_aos(e, w4);
            __m256d f2 = cmul_avx2_aos(f, w5);
            __m256d g2 = cmul_avx2_aos(g, w6);
            __m256d h2 = cmul_avx2_aos(h, w7);

            // Butterfly computation (same logic as 8x loop)
            __m256d s0 = _mm256_add_pd(b2, h2), d0 = _mm256_sub_pd(b2, h2);
            __m256d s1 = _mm256_add_pd(c2, g2), d1 = _mm256_sub_pd(c2, g2);
            __m256d s2 = _mm256_add_pd(d2, f2), d2_diff = _mm256_sub_pd(d2, f2);
            __m256d t0 = _mm256_add_pd(a, e2), t4 = _mm256_sub_pd(a, e2);

            __m256d y0 = _mm256_add_pd(t0, _mm256_add_pd(_mm256_add_pd(s0, s1), s2));
            __m256d y4 = _mm256_add_pd(_mm256_sub_pd(t4, _mm256_add_pd(s0, s1)), s2);

            __m256d base26 = _mm256_sub_pd(d2_diff, d0);
            __m256d base26_swp = _mm256_permute_pd(base26, 0b0101);
            __m256d rr26 = _mm256_xor_pd(base26_swp, rot_mask);
            __m256d t02 = _mm256_sub_pd(t0, s1);
            __m256d y2 = _mm256_add_pd(t02, rr26);
            __m256d y6 = _mm256_sub_pd(t02, rr26);

            __m256d s0ms2 = _mm256_sub_pd(s0, s2);
            __m256d real17 = FMADD(vc, s0ms2, t4);
            __m256d dd = _mm256_add_pd(d0, d2_diff);
            __m256d V17 = _mm256_sub_pd(_mm256_setzero_pd(), FMADD(vc, dd, d1));
            __m256d V17_swp = _mm256_permute_pd(V17, 0b0101);
            __m256d rr17 = _mm256_xor_pd(V17_swp, rot_mask);
            __m256d y1 = _mm256_add_pd(real17, rr17);
            __m256d y7 = _mm256_sub_pd(real17, rr17);

            __m256d real35 = FMSUB(vc, s0ms2, t4);
            __m256d dd2 = _mm256_sub_pd(d0, d2_diff);
            __m256d V35 = _mm256_sub_pd(_mm256_setzero_pd(), FMADD(vc, dd2, d1));
            __m256d V35_swp = _mm256_permute_pd(V35, 0b0101);
            __m256d rr35 = _mm256_xor_pd(V35_swp, rot_mask);
            __m256d y3 = _mm256_add_pd(real35, rr35);
            __m256d y5 = _mm256_sub_pd(real35, rr35);

            // Store results
            STOREU_PD(&output_buffer[k].re, y0);
            STOREU_PD(&output_buffer[k + eighth].re, y1);
            STOREU_PD(&output_buffer[k + 2 * eighth].re, y2);
            STOREU_PD(&output_buffer[k + 3 * eighth].re, y3);
            STOREU_PD(&output_buffer[k + 4 * eighth].re, y4);
            STOREU_PD(&output_buffer[k + 5 * eighth].re, y5);
            STOREU_PD(&output_buffer[k + 6 * eighth].re, y6);
            STOREU_PD(&output_buffer[k + 7 * eighth].re, y7);
        }
        #undef RADIX8_BUTTERFLY_AVX2
#endif // __AVX2__

        //------------------------------------------------------------------
        // SSE2 TAIL: Keep your scalar code
        //------------------------------------------------------------------
         for (; k < eighth; ++k)
        {
            __m128d a = LOADU_SSE2(&sub_outputs[k].re);
            __m128d b = LOADU_SSE2(&sub_outputs[k + eighth].re);
            __m128d c = LOADU_SSE2(&sub_outputs[k + 2 * eighth].re);
            __m128d d = LOADU_SSE2(&sub_outputs[k + 3 * eighth].re);
            __m128d e = LOADU_SSE2(&sub_outputs[k + 4 * eighth].re);
            __m128d f = LOADU_SSE2(&sub_outputs[k + 5 * eighth].re);
            __m128d g = LOADU_SSE2(&sub_outputs[k + 6 * eighth].re);
            __m128d h = LOADU_SSE2(&sub_outputs[k + 7 * eighth].re);

            __m128d w1 = LOADU_SSE2(&stage_tw[7 * k].re);
            __m128d w2 = LOADU_SSE2(&stage_tw[7 * k + 1].re);
            __m128d w3 = LOADU_SSE2(&stage_tw[7 * k + 2].re);
            __m128d w4 = LOADU_SSE2(&stage_tw[7 * k + 3].re);
            __m128d w5 = LOADU_SSE2(&stage_tw[7 * k + 4].re);
            __m128d w6 = LOADU_SSE2(&stage_tw[7 * k + 5].re);
            __m128d w7 = LOADU_SSE2(&stage_tw[7 * k + 6].re);

            __m128d b2 = cmul_sse2_aos(b, w1);
            __m128d c2 = cmul_sse2_aos(c, w2);
            __m128d d2 = cmul_sse2_aos(d, w3);
            __m128d e2 = cmul_sse2_aos(e, w4);
            __m128d f2 = cmul_sse2_aos(f, w5);
            __m128d g2 = cmul_sse2_aos(g, w6);
            __m128d h2 = cmul_sse2_aos(h, w7);

            // Butterfly (same logic as AVX2, using SSE2 instructions)
            __m128d vc_sse = _mm_set1_pd(C8_1);
            __m128d rot_mask_sse = (transform_sign == 1) 
                ? _mm_set_pd(-0.0, 0.0) 
                : _mm_set_pd(0.0, -0.0);

            __m128d s0 = _mm_add_pd(b2, h2), d0_v = _mm_sub_pd(b2, h2);
            __m128d s1 = _mm_add_pd(c2, g2), d1_v = _mm_sub_pd(c2, g2);
            __m128d s2 = _mm_add_pd(d2, f2), d2_diff = _mm_sub_pd(d2, f2);
            __m128d t0 = _mm_add_pd(a, e2), t4 = _mm_sub_pd(a, e2);

            __m128d y0 = _mm_add_pd(t0, _mm_add_pd(_mm_add_pd(s0, s1), s2));
            __m128d y4 = _mm_add_pd(_mm_sub_pd(t4, _mm_add_pd(s0, s1)), s2);

            __m128d base26 = _mm_sub_pd(d2_diff, d0_v);
            __m128d base26_swp = _mm_shuffle_pd(base26, base26, 0b01);
            __m128d rr26 = _mm_xor_pd(base26_swp, rot_mask_sse);
            __m128d t02 = _mm_sub_pd(t0, s1);
            __m128d y2 = _mm_add_pd(t02, rr26);
            __m128d y6 = _mm_sub_pd(t02, rr26);

            __m128d s0ms2 = _mm_sub_pd(s0, s2);
            __m128d real17 = FMADD_SSE2(vc_sse, s0ms2, t4);
            __m128d dd = _mm_add_pd(d0_v, d2_diff);
            __m128d V17 = _mm_sub_pd(_mm_setzero_pd(), FMADD_SSE2(vc_sse, dd, d1_v));
            __m128d V17_swp = _mm_shuffle_pd(V17, V17, 0b01);
            __m128d rr17 = _mm_xor_pd(V17_swp, rot_mask_sse);
            __m128d y1 = _mm_add_pd(real17, rr17);
            __m128d y7 = _mm_sub_pd(real17, rr17);

            __m128d real35 = FMSUB_SSE2(vc_sse, s0ms2, t4);
            __m128d dd2 = _mm_sub_pd(d0_v, d2_diff);
            __m128d V35 = _mm_sub_pd(_mm_setzero_pd(), FMADD_SSE2(vc_sse, dd2, d1_v));
            __m128d V35_swp = _mm_shuffle_pd(V35, V35, 0b01);
            __m128d rr35 = _mm_xor_pd(V35_swp, rot_mask_sse);
            __m128d y3 = _mm_add_pd(real35, rr35);
            __m128d y5 = _mm_sub_pd(real35, rr35);

            STOREU_SSE2(&output_buffer[k].re, y0);
            STOREU_SSE2(&output_buffer[k + eighth].re, y1);
            STOREU_SSE2(&output_buffer[k + 2 * eighth].re, y2);
            STOREU_SSE2(&output_buffer[k + 3 * eighth].re, y3);
            STOREU_SSE2(&output_buffer[k + 4 * eighth].re, y4);
            STOREU_SSE2(&output_buffer[k + 5 * eighth].re, y5);
            STOREU_SSE2(&output_buffer[k + 6 * eighth].re, y6);
            STOREU_SSE2(&output_buffer[k + 7 * eighth].re, y7);
        }
    }
    else if (radix == 11)
    {
        //==========================================================================
        // RADIX-11 BUTTERFLY (Rader DIT with 5 symmetric pairs)
        //
        // Uses Rader's algorithm: maps 11-point DFT to cyclic convolution
        // Exploits symmetry: 5 pairs (k, 11-k) for k=1..5
        //
        // Input:  sub_outputs[0..sub_len-1] through [10*sub_len..11*sub_len-1]
        //         stage_tw[10*k..10*k+9] (W^k through W^{10k}) k-major
        // Output: output_buffer in 11 lanes (Y_0..Y_10)
        //==========================================================================

        const int eleventh = sub_len;
        int k = 0;

#ifdef __AVX2__
        //----------------------------------------------------------------------
        // AVX2 PATH: Process 4 butterflies per iteration (SoA with FMA)
        //----------------------------------------------------------------------
        const __m256d vc1 = _mm256_set1_pd(C11_1); // cos(2π/11)
        const __m256d vc2 = _mm256_set1_pd(C11_2); // cos(4π/11)
        const __m256d vc3 = _mm256_set1_pd(C11_3); // cos(6π/11)
        const __m256d vc4 = _mm256_set1_pd(C11_4); // cos(8π/11)
        const __m256d vc5 = _mm256_set1_pd(C11_5); // cos(10π/11)
        const __m256d vs1 = _mm256_set1_pd(S11_1); // sin(2π/11)
        const __m256d vs2 = _mm256_set1_pd(S11_2); // sin(4π/11)
        const __m256d vs3 = _mm256_set1_pd(S11_3); // sin(6π/11)
        const __m256d vs4 = _mm256_set1_pd(S11_4); // sin(8π/11)
        const __m256d vs5 = _mm256_set1_pd(S11_5); // sin(10π/11)

        for (; k + 3 < eleventh; k += 4)
        {
            if (k + 8 < eleventh)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[10 * (k + 8)].re, _MM_HINT_T0);
            }

            // AoS -> SoA: Load 11 lanes × 4 points
            double aR[4], aI[4], bR[4], bI[4], cR[4], cI[4], dR[4], dI[4];
            double eR[4], eI[4], fR[4], fI[4], gR[4], gI[4], hR[4], hI[4];
            double iR[4], iI[4], jR[4], jI[4], kR[4], kI[4];

            deinterleave4_aos_to_soa(&sub_outputs[k], aR, aI);
            deinterleave4_aos_to_soa(&sub_outputs[k + eleventh], bR, bI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 2 * eleventh], cR, cI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 3 * eleventh], dR, dI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 4 * eleventh], eR, eI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 5 * eleventh], fR, fI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 6 * eleventh], gR, gI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 7 * eleventh], hR, hI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 8 * eleventh], iR, iI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 9 * eleventh], jR, jI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 10 * eleventh], kR, kI);

            __m256d Ar = _mm256_loadu_pd(aR), Ai = _mm256_loadu_pd(aI);
            __m256d Br = _mm256_loadu_pd(bR), Bi = _mm256_loadu_pd(bI);
            __m256d Cr = _mm256_loadu_pd(cR), Ci = _mm256_loadu_pd(cI);
            __m256d Dr = _mm256_loadu_pd(dR), Di = _mm256_loadu_pd(dI);
            __m256d Er = _mm256_loadu_pd(eR), Ei = _mm256_loadu_pd(eI);
            __m256d Fr = _mm256_loadu_pd(fR), Fi = _mm256_loadu_pd(fI);
            __m256d Gr = _mm256_loadu_pd(gR), Gi = _mm256_loadu_pd(gI);
            __m256d Hr = _mm256_loadu_pd(hR), Hi = _mm256_loadu_pd(hI);
            __m256d Ir = _mm256_loadu_pd(iR), Ii = _mm256_loadu_pd(iI);
            __m256d Jr = _mm256_loadu_pd(jR), Ji = _mm256_loadu_pd(jI);
            __m256d Kr = _mm256_loadu_pd(kR), Ki = _mm256_loadu_pd(kI);

            // Load twiddles (k-major: W^k..W^{10k} at 10*k)
            fft_data w1a[4], w2a[4], w3a[4], w4a[4], w5a[4];
            fft_data w6a[4], w7a[4], w8a[4], w9a[4], w10a[4];
            for (int p = 0; p < 4; ++p)
            {
                w1a[p] = stage_tw[10 * (k + p)];
                w2a[p] = stage_tw[10 * (k + p) + 1];
                w3a[p] = stage_tw[10 * (k + p) + 2];
                w4a[p] = stage_tw[10 * (k + p) + 3];
                w5a[p] = stage_tw[10 * (k + p) + 4];
                w6a[p] = stage_tw[10 * (k + p) + 5];
                w7a[p] = stage_tw[10 * (k + p) + 6];
                w8a[p] = stage_tw[10 * (k + p) + 7];
                w9a[p] = stage_tw[10 * (k + p) + 8];
                w10a[p] = stage_tw[10 * (k + p) + 9];
            }

            double w1R[4], w1I[4], w2R[4], w2I[4], w3R[4], w3I[4], w4R[4], w4I[4];
            double w5R[4], w5I[4], w6R[4], w6I[4], w7R[4], w7I[4], w8R[4], w8I[4];
            double w9R[4], w9I[4], w10R[4], w10I[4];

            deinterleave4_aos_to_soa(w1a, w1R, w1I);
            deinterleave4_aos_to_soa(w2a, w2R, w2I);
            deinterleave4_aos_to_soa(w3a, w3R, w3I);
            deinterleave4_aos_to_soa(w4a, w4R, w4I);
            deinterleave4_aos_to_soa(w5a, w5R, w5I);
            deinterleave4_aos_to_soa(w6a, w6R, w6I);
            deinterleave4_aos_to_soa(w7a, w7R, w7I);
            deinterleave4_aos_to_soa(w8a, w8R, w8I);
            deinterleave4_aos_to_soa(w9a, w9R, w9I);
            deinterleave4_aos_to_soa(w10a, w10R, w10I);

            __m256d W1r = _mm256_loadu_pd(w1R), W1i = _mm256_loadu_pd(w1I);
            __m256d W2r = _mm256_loadu_pd(w2R), W2i = _mm256_loadu_pd(w2I);
            __m256d W3r = _mm256_loadu_pd(w3R), W3i = _mm256_loadu_pd(w3I);
            __m256d W4r = _mm256_loadu_pd(w4R), W4i = _mm256_loadu_pd(w4I);
            __m256d W5r = _mm256_loadu_pd(w5R), W5i = _mm256_loadu_pd(w5I);
            __m256d W6r = _mm256_loadu_pd(w6R), W6i = _mm256_loadu_pd(w6I);
            __m256d W7r = _mm256_loadu_pd(w7R), W7i = _mm256_loadu_pd(w7I);
            __m256d W8r = _mm256_loadu_pd(w8R), W8i = _mm256_loadu_pd(w8I);
            __m256d W9r = _mm256_loadu_pd(w9R), W9i = _mm256_loadu_pd(w9I);
            __m256d W10r = _mm256_loadu_pd(w10R), W10i = _mm256_loadu_pd(w10I);

            // Twiddle multiply
            __m256d b2r, b2i, c2r, c2i, d2r, d2i, e2r, e2i, f2r, f2i;
            __m256d g2r, g2i, h2r, h2i, i2r, i2i, j2r, j2i, k2r, k2i;

            cmul_soa_avx(Br, Bi, W1r, W1i, &b2r, &b2i);
            cmul_soa_avx(Cr, Ci, W2r, W2i, &c2r, &c2i);
            cmul_soa_avx(Dr, Di, W3r, W3i, &d2r, &d2i);
            cmul_soa_avx(Er, Ei, W4r, W4i, &e2r, &e2i);
            cmul_soa_avx(Fr, Fi, W5r, W5i, &f2r, &f2i);
            cmul_soa_avx(Gr, Gi, W6r, W6i, &g2r, &g2i);
            cmul_soa_avx(Hr, Hi, W7r, W7i, &h2r, &h2i);
            cmul_soa_avx(Ir, Ii, W8r, W8i, &i2r, &i2i);
            cmul_soa_avx(Jr, Ji, W9r, W9i, &j2r, &j2i);
            cmul_soa_avx(Kr, Ki, W10r, W10i, &k2r, &k2i);

            // Form 5 symmetric pairs: (b,k), (c,j), (d,i), (e,h), (f,g)
            __m256d t0r = _mm256_add_pd(b2r, k2r), t0i = _mm256_add_pd(b2i, k2i);
            __m256d t1r = _mm256_add_pd(c2r, j2r), t1i = _mm256_add_pd(c2i, j2i);
            __m256d t2r = _mm256_add_pd(d2r, i2r), t2i = _mm256_add_pd(d2i, i2i);
            __m256d t3r = _mm256_add_pd(e2r, h2r), t3i = _mm256_add_pd(e2i, h2i);
            __m256d t4r = _mm256_add_pd(f2r, g2r), t4i = _mm256_add_pd(f2i, g2i);

            __m256d s0r = _mm256_sub_pd(b2r, k2r), s0i = _mm256_sub_pd(b2i, k2i);
            __m256d s1r = _mm256_sub_pd(c2r, j2r), s1i = _mm256_sub_pd(c2i, j2i);
            __m256d s2r = _mm256_sub_pd(d2r, i2r), s2i = _mm256_sub_pd(d2i, i2i);
            __m256d s3r = _mm256_sub_pd(e2r, h2r), s3i = _mm256_sub_pd(e2i, h2i);
            __m256d s4r = _mm256_sub_pd(f2r, g2r), s4i = _mm256_sub_pd(f2i, g2i);

            // Y_0 = a + t0 + t1 + t2 + t3 + t4
            __m256d sum_t_r = _mm256_add_pd(_mm256_add_pd(t0r, t1r),
                                            _mm256_add_pd(_mm256_add_pd(t2r, t3r), t4r));
            __m256d sum_t_i = _mm256_add_pd(_mm256_add_pd(t0i, t1i),
                                            _mm256_add_pd(_mm256_add_pd(t2i, t3i), t4i));
            __m256d y0r = _mm256_add_pd(Ar, sum_t_r);
            __m256d y0i = _mm256_add_pd(Ai, sum_t_i);

            // For pairs Y_m / Y_{11-m}, m=1..5:
            // Real part: a + c1*t0 + c2*t1 + c3*t2 + c4*t3 + c5*t4
            // Imag rot:  s1*s0 + s2*s1 + s3*s2 + s4*s3 + s5*s4

            // Pair 1: Y_1, Y_10
            __m256d tmp1r = _mm256_add_pd(Ar, FMADD(vc1, t0r, FMADD(vc2, t1r, FMADD(vc3, t2r, FMADD(vc4, t3r, _mm256_mul_pd(vc5, t4r))))));
            __m256d tmp1i = _mm256_add_pd(Ai, FMADD(vc1, t0i, FMADD(vc2, t1i, FMADD(vc3, t2i, FMADD(vc4, t3i, _mm256_mul_pd(vc5, t4i))))));

            __m256d base1r = FMADD(vs1, s0r, FMADD(vs2, s1r, FMADD(vs3, s2r, FMADD(vs4, s3r, _mm256_mul_pd(vs5, s4r)))));
            __m256d base1i = FMADD(vs1, s0i, FMADD(vs2, s1i, FMADD(vs3, s2i, FMADD(vs4, s3i, _mm256_mul_pd(vs5, s4i)))));

            __m256d r1r, r1i;
            rot90_soa_avx(base1r, base1i, transform_sign, &r1r, &r1i);

            __m256d y1r = _mm256_add_pd(tmp1r, r1r), y1i = _mm256_add_pd(tmp1i, r1i);
            __m256d y10r = _mm256_sub_pd(tmp1r, r1r), y10i = _mm256_sub_pd(tmp1i, r1i);

            // Pair 2: Y_2, Y_9 (permute coefficients cyclically)
            __m256d tmp2r = _mm256_add_pd(Ar, FMADD(vc2, t0r, FMADD(vc4, t1r, FMADD(vc5, t2r, FMADD(vc3, t3r, _mm256_mul_pd(vc1, t4r))))));
            __m256d tmp2i = _mm256_add_pd(Ai, FMADD(vc2, t0i, FMADD(vc4, t1i, FMADD(vc5, t2i, FMADD(vc3, t3i, _mm256_mul_pd(vc1, t4i))))));

            __m256d base2r = FMADD(vs2, s0r, FMADD(vs4, s1r, FMADD(vs5, s2r, FMADD(vs3, s3r, _mm256_mul_pd(vs1, s4r)))));
            __m256d base2i = FMADD(vs2, s0i, FMADD(vs4, s1i, FMADD(vs5, s2i, FMADD(vs3, s3i, _mm256_mul_pd(vs1, s4i)))));

            __m256d r2r, r2i;
            rot90_soa_avx(base2r, base2i, transform_sign, &r2r, &r2i);

            __m256d y2r = _mm256_add_pd(tmp2r, r2r), y2i = _mm256_add_pd(tmp2i, r2i);
            __m256d y9r = _mm256_sub_pd(tmp2r, r2r), y9i = _mm256_sub_pd(tmp2i, r2i);

            // Pair 3: Y_3, Y_8
            __m256d tmp3r = _mm256_add_pd(Ar, FMADD(vc3, t0r, FMADD(vc5, t1r, FMADD(vc2, t2r, FMADD(vc1, t3r, _mm256_mul_pd(vc4, t4r))))));
            __m256d tmp3i = _mm256_add_pd(Ai, FMADD(vc3, t0i, FMADD(vc5, t1i, FMADD(vc2, t2i, FMADD(vc1, t3i, _mm256_mul_pd(vc4, t4i))))));

            __m256d base3r = FMADD(vs3, s0r, FMADD(vs5, s1r, FMADD(vs2, s2r, FMADD(vs1, s3r, _mm256_mul_pd(vs4, s4r)))));
            __m256d base3i = FMADD(vs3, s0i, FMADD(vs5, s1i, FMADD(vs2, s2i, FMADD(vs1, s3i, _mm256_mul_pd(vs4, s4i)))));

            __m256d r3r, r3i;
            rot90_soa_avx(base3r, base3i, transform_sign, &r3r, &r3i);

            __m256d y3r = _mm256_add_pd(tmp3r, r3r), y3i = _mm256_add_pd(tmp3i, r3i);
            __m256d y8r = _mm256_sub_pd(tmp3r, r3r), y8i = _mm256_sub_pd(tmp3i, r3i);

            // Pair 4: Y_4, Y_7
            __m256d tmp4r = _mm256_add_pd(Ar, FMADD(vc4, t0r, FMADD(vc3, t1r, FMADD(vc1, t2r, FMADD(vc5, t3r, _mm256_mul_pd(vc2, t4r))))));
            __m256d tmp4i = _mm256_add_pd(Ai, FMADD(vc4, t0i, FMADD(vc3, t1i, FMADD(vc1, t2i, FMADD(vc5, t3i, _mm256_mul_pd(vc2, t4i))))));

            __m256d base4r = FMADD(vs4, s0r, FMADD(vs3, s1r, FMADD(vs1, s2r, FMADD(vs5, s3r, _mm256_mul_pd(vs2, s4r)))));
            __m256d base4i = FMADD(vs4, s0i, FMADD(vs3, s1i, FMADD(vs1, s2i, FMADD(vs5, s3i, _mm256_mul_pd(vs2, s4i)))));

            __m256d r4r, r4i;
            rot90_soa_avx(base4r, base4i, transform_sign, &r4r, &r4i);

            __m256d y4r = _mm256_add_pd(tmp4r, r4r), y4i = _mm256_add_pd(tmp4i, r4i);
            __m256d y7r = _mm256_sub_pd(tmp4r, r4r), y7i = _mm256_sub_pd(tmp4i, r4i);

            // Pair 5: Y_5, Y_6
            __m256d tmp5r = _mm256_add_pd(Ar, FMADD(vc5, t0r, FMADD(vc1, t1r, FMADD(vc4, t2r, FMADD(vc2, t3r, _mm256_mul_pd(vc3, t4r))))));
            __m256d tmp5i = _mm256_add_pd(Ai, FMADD(vc5, t0i, FMADD(vc1, t1i, FMADD(vc4, t2i, FMADD(vc2, t3i, _mm256_mul_pd(vc3, t4i))))));

            __m256d base5r = FMADD(vs5, s0r, FMADD(vs1, s1r, FMADD(vs4, s2r, FMADD(vs2, s3r, _mm256_mul_pd(vs3, s4r)))));
            __m256d base5i = FMADD(vs5, s0i, FMADD(vs1, s1i, FMADD(vs4, s2i, FMADD(vs2, s3i, _mm256_mul_pd(vs3, s4i)))));

            __m256d r5r, r5i;
            rot90_soa_avx(base5r, base5i, transform_sign, &r5r, &r5i);

            __m256d y5r = _mm256_add_pd(tmp5r, r5r), y5i = _mm256_add_pd(tmp5i, r5i);
            __m256d y6r = _mm256_sub_pd(tmp5r, r5r), y6i = _mm256_sub_pd(tmp5i, r5i);

            // SoA -> AoS stores
            double Y0R[4], Y0I[4], Y1R[4], Y1I[4], Y2R[4], Y2I[4], Y3R[4], Y3I[4];
            double Y4R[4], Y4I[4], Y5R[4], Y5I[4], Y6R[4], Y6I[4], Y7R[4], Y7I[4];
            double Y8R[4], Y8I[4], Y9R[4], Y9I[4], Y10R[4], Y10I[4];

            _mm256_storeu_pd(Y0R, y0r);
            _mm256_storeu_pd(Y0I, y0i);
            _mm256_storeu_pd(Y1R, y1r);
            _mm256_storeu_pd(Y1I, y1i);
            _mm256_storeu_pd(Y2R, y2r);
            _mm256_storeu_pd(Y2I, y2i);
            _mm256_storeu_pd(Y3R, y3r);
            _mm256_storeu_pd(Y3I, y3i);
            _mm256_storeu_pd(Y4R, y4r);
            _mm256_storeu_pd(Y4I, y4i);
            _mm256_storeu_pd(Y5R, y5r);
            _mm256_storeu_pd(Y5I, y5i);
            _mm256_storeu_pd(Y6R, y6r);
            _mm256_storeu_pd(Y6I, y6i);
            _mm256_storeu_pd(Y7R, y7r);
            _mm256_storeu_pd(Y7I, y7i);
            _mm256_storeu_pd(Y8R, y8r);
            _mm256_storeu_pd(Y8I, y8i);
            _mm256_storeu_pd(Y9R, y9r);
            _mm256_storeu_pd(Y9I, y9i);
            _mm256_storeu_pd(Y10R, y10r);
            _mm256_storeu_pd(Y10I, y10i);

            interleave4_soa_to_aos(Y0R, Y0I, &output_buffer[k]);
            interleave4_soa_to_aos(Y1R, Y1I, &output_buffer[k + eleventh]);
            interleave4_soa_to_aos(Y2R, Y2I, &output_buffer[k + 2 * eleventh]);
            interleave4_soa_to_aos(Y3R, Y3I, &output_buffer[k + 3 * eleventh]);
            interleave4_soa_to_aos(Y4R, Y4I, &output_buffer[k + 4 * eleventh]);
            interleave4_soa_to_aos(Y5R, Y5I, &output_buffer[k + 5 * eleventh]);
            interleave4_soa_to_aos(Y6R, Y6I, &output_buffer[k + 6 * eleventh]);
            interleave4_soa_to_aos(Y7R, Y7I, &output_buffer[k + 7 * eleventh]);
            interleave4_soa_to_aos(Y8R, Y8I, &output_buffer[k + 8 * eleventh]);
            interleave4_soa_to_aos(Y9R, Y9I, &output_buffer[k + 9 * eleventh]);
            interleave4_soa_to_aos(Y10R, Y10I, &output_buffer[k + 10 * eleventh]);
        }
#endif // __AVX2__

        //======================================================================
        // SCALAR TAIL: Handle remaining 0..3 elements
        //======================================================================
        for (; k < eleventh; ++k)
        {
            // Load 11 lanes
            const fft_data a = sub_outputs[k];
            const fft_data b = sub_outputs[k + eleventh];
            const fft_data c = sub_outputs[k + 2 * eleventh];
            const fft_data d = sub_outputs[k + 3 * eleventh];
            const fft_data e = sub_outputs[k + 4 * eleventh];
            const fft_data f = sub_outputs[k + 5 * eleventh];
            const fft_data g = sub_outputs[k + 6 * eleventh];
            const fft_data h = sub_outputs[k + 7 * eleventh];
            const fft_data i = sub_outputs[k + 8 * eleventh];
            const fft_data j = sub_outputs[k + 9 * eleventh];
            const fft_data kval = sub_outputs[k + 10 * eleventh]; // 'k' is loop var

            // Load twiddles (k-major)
            const fft_data w1 = stage_tw[10 * k];
            const fft_data w2 = stage_tw[10 * k + 1];
            const fft_data w3 = stage_tw[10 * k + 2];
            const fft_data w4 = stage_tw[10 * k + 3];
            const fft_data w5 = stage_tw[10 * k + 4];
            const fft_data w6 = stage_tw[10 * k + 5];
            const fft_data w7 = stage_tw[10 * k + 6];
            const fft_data w8 = stage_tw[10 * k + 7];
            const fft_data w9 = stage_tw[10 * k + 8];
            const fft_data w10 = stage_tw[10 * k + 9];

            // Twiddle multiply
            double b2r = b.re * w1.re - b.im * w1.im, b2i = b.re * w1.im + b.im * w1.re;
            double c2r = c.re * w2.re - c.im * w2.im, c2i = c.re * w2.im + c.im * w2.re;
            double d2r = d.re * w3.re - d.im * w3.im, d2i = d.re * w3.im + d.im * w3.re;
            double e2r = e.re * w4.re - e.im * w4.im, e2i = e.re * w4.im + e.im * w4.re;
            double f2r = f.re * w5.re - f.im * w5.im, f2i = f.re * w5.im + f.im * w5.re;
            double g2r = g.re * w6.re - g.im * w6.im, g2i = g.re * w6.im + g.im * w6.re;
            double h2r = h.re * w7.re - h.im * w7.im, h2i = h.re * w7.im + h.im * w7.re;
            double i2r = i.re * w8.re - i.im * w8.im, i2i = i.re * w8.im + i.im * w8.re;
            double j2r = j.re * w9.re - j.im * w9.im, j2i = j.re * w9.im + j.im * w9.re;
            double k2r = kval.re * w10.re - kval.im * w10.im, k2i = kval.re * w10.im + kval.im * w10.re;

            // Form 5 symmetric pairs
            double t0r = b2r + k2r, t0i = b2i + k2i;
            double t1r = c2r + j2r, t1i = c2i + j2i;
            double t2r = d2r + i2r, t2i = d2i + i2i;
            double t3r = e2r + h2r, t3i = e2i + h2i;
            double t4r = f2r + g2r, t4i = f2i + g2i;

            double s0r = b2r - k2r, s0i = b2i - k2i;
            double s1r = c2r - j2r, s1i = c2i - j2i;
            double s2r = d2r - i2r, s2i = d2i - i2i;
            double s3r = e2r - h2r, s3i = e2i - h2i;
            double s4r = f2r - g2r, s4i = f2i - g2i;

            // Y_0
            fft_data y0 = {
                a.re + (t0r + t1r + t2r + t3r + t4r),
                a.im + (t0i + t1i + t2i + t3i + t4i)};

            // Pair 1: Y_1, Y_10
            double tmp1r = a.re + (C11_1 * t0r + C11_2 * t1r + C11_3 * t2r + C11_4 * t3r + C11_5 * t4r);
            double tmp1i = a.im + (C11_1 * t0i + C11_2 * t1i + C11_3 * t2i + C11_4 * t3i + C11_5 * t4i);
            double base1r = S11_1 * s0r + S11_2 * s1r + S11_3 * s2r + S11_4 * s3r + S11_5 * s4r;
            double base1i = S11_1 * s0i + S11_2 * s1i + S11_3 * s2i + S11_4 * s3i + S11_5 * s4i;
            double r1r = (transform_sign == 1) ? -base1i : base1i;
            double r1i = (transform_sign == 1) ? base1r : -base1r;
            fft_data y1 = {tmp1r + r1r, tmp1i + r1i};
            fft_data y10 = {tmp1r - r1r, tmp1i - r1i};

            // Pair 2: Y_2, Y_9
            double tmp2r = a.re + (C11_2 * t0r + C11_4 * t1r + C11_5 * t2r + C11_3 * t3r + C11_1 * t4r);
            double tmp2i = a.im + (C11_2 * t0i + C11_4 * t1i + C11_5 * t2i + C11_3 * t3i + C11_1 * t4i);
            double base2r = S11_2 * s0r + S11_4 * s1r + S11_5 * s2r + S11_3 * s3r + S11_1 * s4r;
            double base2i = S11_2 * s0i + S11_4 * s1i + S11_5 * s2i + S11_3 * s3i + S11_1 * s4i;
            double r2r = (transform_sign == 1) ? -base2i : base2i;
            double r2i = (transform_sign == 1) ? base2r : -base2r;
            fft_data y2 = {tmp2r + r2r, tmp2i + r2i};
            fft_data y9 = {tmp2r - r2r, tmp2i - r2i};

            // Pair 3: Y_3, Y_8
            double tmp3r = a.re + (C11_3 * t0r + C11_5 * t1r + C11_2 * t2r + C11_1 * t3r + C11_4 * t4r);
            double tmp3i = a.im + (C11_3 * t0i + C11_5 * t1i + C11_2 * t2i + C11_1 * t3i + C11_4 * t4i);
            double base3r = S11_3 * s0r + S11_5 * s1r + S11_2 * s2r + S11_1 * s3r + S11_4 * s4r;
            double base3i = S11_3 * s0i + S11_5 * s1i + S11_2 * s2i + S11_1 * s3i + S11_4 * s4i;
            double r3r = (transform_sign == 1) ? -base3i : base3i;
            double r3i = (transform_sign == 1) ? base3r : -base3r;
            fft_data y3 = {tmp3r + r3r, tmp3i + r3i};
            fft_data y8 = {tmp3r - r3r, tmp3i - r3i};

            // Pair 4: Y_4, Y_7
            double tmp4r = a.re + (C11_4 * t0r + C11_3 * t1r + C11_1 * t2r + C11_5 * t3r + C11_2 * t4r);
            double tmp4i = a.im + (C11_4 * t0i + C11_3 * t1i + C11_1 * t2i + C11_5 * t3i + C11_2 * t4i);
            double base4r = S11_4 * s0r + S11_3 * s1r + S11_1 * s2r + S11_5 * s3r + S11_2 * s4r;
            double base4i = S11_4 * s0i + S11_3 * s1i + S11_1 * s2i + S11_5 * s3i + S11_2 * s4i;
            double r4r = (transform_sign == 1) ? -base4i : base4i;
            double r4i = (transform_sign == 1) ? base4r : -base4r;
            fft_data y4 = {tmp4r + r4r, tmp4i + r4i};
            fft_data y7 = {tmp4r - r4r, tmp4i - r4i};

            // Pair 5: Y_5, Y_6
            double tmp5r = a.re + (C11_5 * t0r + C11_1 * t1r + C11_4 * t2r + C11_2 * t3r + C11_3 * t4r);
            double tmp5i = a.im + (C11_5 * t0i + C11_1 * t1i + C11_4 * t2i + C11_2 * t3i + C11_3 * t4i);
            double base5r = S11_5 * s0r + S11_1 * s1r + S11_4 * s2r + S11_2 * s3r + S11_3 * s4r;
            double base5i = S11_5 * s0i + S11_1 * s1i + S11_4 * s2i + S11_2 * s3i + S11_3 * s4i;
            double r5r = (transform_sign == 1) ? -base5i : base5i;
            double r5i = (transform_sign == 1) ? base5r : -base5r;
            fft_data y5 = {tmp5r + r5r, tmp5i + r5i};
            fft_data y6 = {tmp5r - r5r, tmp5i - r5i};

            // Store
            output_buffer[k] = y0;
            output_buffer[k + eleventh] = y1;
            output_buffer[k + 2 * eleventh] = y2;
            output_buffer[k + 3 * eleventh] = y3;
            output_buffer[k + 4 * eleventh] = y4;
            output_buffer[k + 5 * eleventh] = y5;
            output_buffer[k + 6 * eleventh] = y6;
            output_buffer[k + 7 * eleventh] = y7;
            output_buffer[k + 8 * eleventh] = y8;
            output_buffer[k + 9 * eleventh] = y9;
            output_buffer[k + 10 * eleventh] = y10;
        }
    }
    else if (radix == 16)
    {
        //==========================================================================
        // RADIX-16 BUTTERFLY (2-stage radix-4 decomposition)
        //==========================================================================

        const int sixteenth = sub_len;
        int k = 0;

#ifdef __AVX2__
        //----------------------------------------------------------------------
        // Precompute W_4 intermediate twiddles (OUTSIDE loop)
        //----------------------------------------------------------------------

        // Convert twiddle_radix4 to AVX2 format
        __m256d W4_avx[4];
        for (int m = 0; m < 4; ++m)
        {
            const complex_t tw = twiddle_radix4[m];
            double tw_im = (transform_sign == 1) ? tw.im : -tw.im;
            W4_avx[m] = _mm256_set_pd(tw_im, tw.re, tw_im, tw.re);
        }

        // Precompute rotation masks
        const __m256d rot_mask = (transform_sign == 1)
                                     ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                                     : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

        //----------------------------------------------------------------------
        // Main loop: 8x unrolling
        //----------------------------------------------------------------------
        for (; k + 7 < sixteenth; k += 8)
        {
            if (k + 16 < sixteenth)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[15 * (k + 16)].re, _MM_HINT_T0);
            }

            //==================================================================
            // Load all 16 lanes (8 butterflies = 4 AVX2 loads per lane)
            //==================================================================
            __m256d x[16][4]; // [lane][butterfly_pair]

            for (int lane = 0; lane < 16; ++lane)
            {
                x[lane][0] = load2_aos(&sub_outputs[k + 0 + lane * sixteenth],
                                       &sub_outputs[k + 1 + lane * sixteenth]);
                x[lane][1] = load2_aos(&sub_outputs[k + 2 + lane * sixteenth],
                                       &sub_outputs[k + 3 + lane * sixteenth]);
                x[lane][2] = load2_aos(&sub_outputs[k + 4 + lane * sixteenth],
                                       &sub_outputs[k + 5 + lane * sixteenth]);
                x[lane][3] = load2_aos(&sub_outputs[k + 6 + lane * sixteenth],
                                       &sub_outputs[k + 7 + lane * sixteenth]);
            }

            //==================================================================
            // Stage 1: Apply input twiddles W^{jk}
            //==================================================================
            for (int lane = 1; lane < 16; ++lane)
            {
                __m256d tw0 = load2_aos(&stage_tw[15 * (k + 0) + (lane - 1)],
                                        &stage_tw[15 * (k + 1) + (lane - 1)]);
                __m256d tw1 = load2_aos(&stage_tw[15 * (k + 2) + (lane - 1)],
                                        &stage_tw[15 * (k + 3) + (lane - 1)]);
                __m256d tw2 = load2_aos(&stage_tw[15 * (k + 4) + (lane - 1)],
                                        &stage_tw[15 * (k + 5) + (lane - 1)]);
                __m256d tw3 = load2_aos(&stage_tw[15 * (k + 6) + (lane - 1)],
                                        &stage_tw[15 * (k + 7) + (lane - 1)]);

                x[lane][0] = cmul_avx2_aos(x[lane][0], tw0);
                x[lane][1] = cmul_avx2_aos(x[lane][1], tw1);
                x[lane][2] = cmul_avx2_aos(x[lane][2], tw2);
                x[lane][3] = cmul_avx2_aos(x[lane][3], tw3);
            }

            //==================================================================
            // Stage 2: First radix-4 (4 groups of 4)
            //==================================================================
            __m256d y[16][4];

            for (int group = 0; group < 4; ++group)
            {
                for (int b = 0; b < 4; ++b)
                {
                    __m256d a = x[group][b];
                    __m256d c = x[group + 4][b];
                    __m256d e = x[group + 8][b];
                    __m256d g = x[group + 12][b];

                    // Radix-4 butterfly
                    __m256d sumEG = _mm256_add_pd(c, g);
                    __m256d difEG = _mm256_sub_pd(c, g);
                    __m256d a_pe = _mm256_add_pd(a, e);
                    __m256d a_me = _mm256_sub_pd(a, e);

                    y[4 * group][b] = _mm256_add_pd(a_pe, sumEG);
                    y[4 * group + 2][b] = _mm256_sub_pd(a_pe, sumEG);

                    __m256d difEG_swp = _mm256_permute_pd(difEG, 0b0101);
                    __m256d rot = _mm256_xor_pd(difEG_swp, rot_mask);

                    y[4 * group + 1][b] = _mm256_sub_pd(a_me, rot);
                    y[4 * group + 3][b] = _mm256_add_pd(a_me, rot);
                }
            }

            //==================================================================
            // Stage 2.5: Apply intermediate twiddles W_4^{jm}
            //==================================================================

            // m=1: W_4^j for j=1,2,3
            for (int b = 0; b < 4; ++b)
            {
                y[5][b] = cmul_avx2_aos(y[5][b], W4_avx[1]);
                y[6][b] = cmul_avx2_aos(y[6][b], W4_avx[2]);
                y[7][b] = cmul_avx2_aos(y[7][b], W4_avx[3]);
            }

            // m=2: W_4^{2j} for j=1,2,3
            for (int b = 0; b < 4; ++b)
            {
                y[9][b] = cmul_avx2_aos(y[9][b], W4_avx[2]);
                // y[10][b] *= W4_avx[0] = 1 (skip)
                y[11][b] = cmul_avx2_aos(y[11][b], W4_avx[2]);
            }

            // m=3: W_4^{3j} for j=1,2,3
            for (int b = 0; b < 4; ++b)
            {
                y[13][b] = cmul_avx2_aos(y[13][b], W4_avx[3]);
                y[14][b] = cmul_avx2_aos(y[14][b], W4_avx[2]);
                y[15][b] = cmul_avx2_aos(y[15][b], W4_avx[1]);
            }

            //==================================================================
            // Stage 3: Second radix-4 (final)
            //==================================================================
            for (int m = 0; m < 4; ++m)
            {
                for (int b = 0; b < 4; ++b)
                {
                    __m256d a = y[m][b];
                    __m256d c = y[m + 4][b];
                    __m256d e = y[m + 8][b];
                    __m256d g = y[m + 12][b];

                    __m256d sumEG = _mm256_add_pd(c, g);
                    __m256d difEG = _mm256_sub_pd(c, g);
                    __m256d a_pe = _mm256_add_pd(a, e);
                    __m256d a_me = _mm256_sub_pd(a, e);

                    __m256d z0 = _mm256_add_pd(a_pe, sumEG);
                    __m256d z2 = _mm256_sub_pd(a_pe, sumEG);

                    __m256d difEG_swp = _mm256_permute_pd(difEG, 0b0101);
                    __m256d rot = _mm256_xor_pd(difEG_swp, rot_mask);

                    __m256d z1 = _mm256_sub_pd(a_me, rot);
                    __m256d z3 = _mm256_add_pd(a_me, rot);

                    // Store results
                    STOREU_PD(&output_buffer[k + 2 * b + m * sixteenth].re, z0);
                    STOREU_PD(&output_buffer[k + 2 * b + (m + 4) * sixteenth].re, z1);
                    STOREU_PD(&output_buffer[k + 2 * b + (m + 8) * sixteenth].re, z2);
                    STOREU_PD(&output_buffer[k + 2 * b + (m + 12) * sixteenth].re, z3);
                }
            }
        }

        //----------------------------------------------------------------------
        // Cleanup: 2x unrolling
        //----------------------------------------------------------------------
        for (; k + 1 < sixteenth; k += 2)
        {
            if (k + 8 < sixteenth)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[15 * (k + 8)].re, _MM_HINT_T0);
            }

            // Load 16 lanes
            __m256d x[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                x[lane] = load2_aos(&sub_outputs[k + lane * sixteenth],
                                    &sub_outputs[k + lane * sixteenth + 1]);
            }

            // Apply input twiddles
            for (int lane = 1; lane < 16; ++lane)
            {
                __m256d tw = load2_aos(&stage_tw[15 * k + (lane - 1)],
                                       &stage_tw[15 * (k + 1) + (lane - 1)]);
                x[lane] = cmul_avx2_aos(x[lane], tw);
            }

            // First radix-4 stage
            __m256d y[16];
            for (int group = 0; group < 4; ++group)
            {
                __m256d a = x[group];
                __m256d b = x[group + 4];
                __m256d c = x[group + 8];
                __m256d d = x[group + 12];

                __m256d sumBD = _mm256_add_pd(b, d);
                __m256d difBD = _mm256_sub_pd(b, d);
                __m256d a_pc = _mm256_add_pd(a, c);
                __m256d a_mc = _mm256_sub_pd(a, c);

                y[4 * group] = _mm256_add_pd(a_pc, sumBD);
                y[4 * group + 2] = _mm256_sub_pd(a_pc, sumBD);

                __m256d difBD_swp = _mm256_permute_pd(difBD, 0b0101);
                __m256d rot = _mm256_xor_pd(difBD_swp, rot_mask);

                y[4 * group + 1] = _mm256_sub_pd(a_mc, rot);
                y[4 * group + 3] = _mm256_add_pd(a_mc, rot);
            }

            // Apply intermediate twiddles W_4
            y[5] = cmul_avx2_aos(y[5], W4_avx[1]);
            y[6] = cmul_avx2_aos(y[6], W4_avx[2]);
            y[7] = cmul_avx2_aos(y[7], W4_avx[3]);

            y[9] = cmul_avx2_aos(y[9], W4_avx[2]);
            y[11] = cmul_avx2_aos(y[11], W4_avx[2]);

            y[13] = cmul_avx2_aos(y[13], W4_avx[3]);
            y[14] = cmul_avx2_aos(y[14], W4_avx[2]);
            y[15] = cmul_avx2_aos(y[15], W4_avx[1]);

            // Second radix-4 stage
            for (int m = 0; m < 4; ++m)
            {
                __m256d a = y[m];
                __m256d b = y[m + 4];
                __m256d c = y[m + 8];
                __m256d d = y[m + 12];

                __m256d sumBD = _mm256_add_pd(b, d);
                __m256d difBD = _mm256_sub_pd(b, d);
                __m256d a_pc = _mm256_add_pd(a, c);
                __m256d a_mc = _mm256_sub_pd(a, c);

                __m256d z0 = _mm256_add_pd(a_pc, sumBD);
                __m256d z2 = _mm256_sub_pd(a_pc, sumBD);

                __m256d difBD_swp = _mm256_permute_pd(difBD, 0b0101);
                __m256d rot = _mm256_xor_pd(difBD_swp, rot_mask);

                __m256d z1 = _mm256_sub_pd(a_mc, rot);
                __m256d z3 = _mm256_add_pd(a_mc, rot);

                STOREU_PD(&output_buffer[k + m * sixteenth].re, z0);
                STOREU_PD(&output_buffer[k + (m + 4) * sixteenth].re, z1);
                STOREU_PD(&output_buffer[k + (m + 8) * sixteenth].re, z2);
                STOREU_PD(&output_buffer[k + (m + 12) * sixteenth].re, z3);
            }
        }
#endif // __AVX2__

        //======================================================================
        // SCALAR TAIL
        //======================================================================
        for (; k < sixteenth; ++k)
        {
            // Load 16 lanes
            fft_data x[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                x[lane] = sub_outputs[k + lane * sixteenth];
            }

            // Apply twiddles W^{jk} for j=1..15
            for (int j = 1; j < 16; ++j)
            {
                fft_data tw = stage_tw[15 * k + (j - 1)];
                double xr = x[j].re, xi = x[j].im;
                x[j].re = xr * tw.re - xi * tw.im;
                x[j].im = xr * tw.im + xi * tw.re;
            }

            // First radix-4 stage (4 groups of 4)
            fft_data y[16];
            for (int group = 0; group < 4; ++group)
            {
                fft_data a = x[group];
                fft_data b = x[group + 4];
                fft_data c = x[group + 8];
                fft_data d = x[group + 12];

                // Radix-4 butterfly
                double sumBDr = b.re + d.re, sumBDi = b.im + d.im;
                double difBDr = b.re - d.re, difBDi = b.im - d.im;
                double a_pc_r = a.re + c.re, a_pc_i = a.im + c.im;
                double a_mc_r = a.re - c.re, a_mc_i = a.im - c.im;

                y[4 * group] = (fft_data){a_pc_r + sumBDr, a_pc_i + sumBDi};
                y[4 * group + 2] = (fft_data){a_pc_r - sumBDr, a_pc_i - sumBDi};

                double rotr = (transform_sign == 1) ? -difBDi : difBDi;
                double roti = (transform_sign == 1) ? difBDr : -difBDr;

                y[4 * group + 1] = (fft_data){a_mc_r - rotr, a_mc_i - roti};
                y[4 * group + 3] = (fft_data){a_mc_r + rotr, a_mc_i + roti};
            }

            // Apply intermediate twiddles W_4^{jm}
            for (int m = 0; m < 4; ++m)
            {
                // j=1: W_4^m
                if (m == 1) // -i
                {
                    double temp = y[4 * m + 1].re;
                    y[4 * m + 1].re = y[4 * m + 1].im * transform_sign;
                    y[4 * m + 1].im = -temp * transform_sign;
                }
                else if (m == 2) // -1
                {
                    y[4 * m + 1].re = -y[4 * m + 1].re;
                    y[4 * m + 1].im = -y[4 * m + 1].im;
                }
                else if (m == 3) // +i
                {
                    double temp = y[4 * m + 1].re;
                    y[4 * m + 1].re = -y[4 * m + 1].im * transform_sign;
                    y[4 * m + 1].im = temp * transform_sign;
                }

                // j=2: W_4^{2m}
                if (m == 1 || m == 3) // -1
                {
                    y[4 * m + 2].re = -y[4 * m + 2].re;
                    y[4 * m + 2].im = -y[4 * m + 2].im;
                }

                // j=3: W_4^{3m}
                if (m == 1) // +i
                {
                    double temp = y[4 * m + 3].re;
                    y[4 * m + 3].re = -y[4 * m + 3].im * transform_sign;
                    y[4 * m + 3].im = temp * transform_sign;
                }
                else if (m == 2) // -1
                {
                    y[4 * m + 3].re = -y[4 * m + 3].re;
                    y[4 * m + 3].im = -y[4 * m + 3].im;
                }
                else if (m == 3) // -i
                {
                    double temp = y[4 * m + 3].re;
                    y[4 * m + 3].re = y[4 * m + 3].im * transform_sign;
                    y[4 * m + 3].im = -temp * transform_sign;
                }
            }

            // Second radix-4 stage (final)
            for (int m = 0; m < 4; ++m)
            {
                fft_data a = y[m];
                fft_data b = y[m + 4];
                fft_data c = y[m + 8];
                fft_data d = y[m + 12];

                double sumBDr = b.re + d.re, sumBDi = b.im + d.im;
                double difBDr = b.re - d.re, difBDi = b.im - d.im;
                double a_pc_r = a.re + c.re, a_pc_i = a.im + c.im;
                double a_mc_r = a.re - c.re, a_mc_i = a.im - c.im;

                output_buffer[k + m * sixteenth] =
                    (fft_data){a_pc_r + sumBDr, a_pc_i + sumBDi};
                output_buffer[k + (m + 8) * sixteenth] =
                    (fft_data){a_pc_r - sumBDr, a_pc_i - sumBDi};

                double rotr = (transform_sign == 1) ? -difBDi : difBDi;
                double roti = (transform_sign == 1) ? difBDr : -difBDr;

                output_buffer[k + (m + 4) * sixteenth] =
                    (fft_data){a_mc_r - rotr, a_mc_i - roti};
                output_buffer[k + (m + 12) * sixteenth] =
                    (fft_data){a_mc_r + rotr, a_mc_i + roti};
            }
        }
    }

    else if (radix == 32)
    {
        //==========================================================================
        // RADIX-32 BUTTERFLY - HIGHLY OPTIMIZED
        //
        // Decomposition: 4 × 4 × 2 (radix-4, radix-4, radix-2)
        // With intermediate twiddles precomputed outside loops
        //==========================================================================

        const int thirtysecond = sub_len;
        int k = 0;

#ifdef __AVX2__
        //----------------------------------------------------------------------
        // Precompute ALL intermediate twiddles as constants (OUTSIDE loop)
        //----------------------------------------------------------------------

        // W_8 twiddles: exp(-2πim/8) for m=0..7
        // These are used between first and second radix-4 stages
        __m256d W8[8][3]; // [m][j-1] for j=1,2,3

        // W_8^0 = 1 (not needed, m=0 case skipped)
        // W_8^1 = cos(π/4) - i*sin(π/4) = (√2/2)(1 - i)
        const double c8 = 0.7071067811865476;
        W8[1][0] = _mm256_set_pd(-c8, c8, -c8, c8);     // j=1: W_8^1
        W8[1][1] = _mm256_set_pd(-1.0, 0.0, -1.0, 0.0); // j=2: W_8^2 = -i
        W8[1][2] = _mm256_set_pd(-c8, -c8, -c8, -c8);   // j=3: W_8^3 = (√2/2)(-1-i)

        // W_8^2 = -i
        W8[2][0] = _mm256_set_pd(-1.0, 0.0, -1.0, 0.0); // j=1: W_8^2 = -i
        W8[2][1] = _mm256_set_pd(0.0, -1.0, 0.0, -1.0); // j=2: W_8^4 = -1
        W8[2][2] = _mm256_set_pd(1.0, 0.0, 1.0, 0.0);   // j=3: W_8^6 = +i

        // W_8^3 = (√2/2)(-1 - i)
        W8[3][0] = _mm256_set_pd(-c8, -c8, -c8, -c8);
        W8[3][1] = _mm256_set_pd(1.0, 0.0, 1.0, 0.0); // j=2: W_8^6 = +i
        W8[3][2] = _mm256_set_pd(c8, -c8, c8, -c8);   // j=3: W_8^9 = (√2/2)(1-i)

        // W_8^4 = -1
        W8[4][0] = _mm256_set_pd(0.0, -1.0, 0.0, -1.0); // j=1: W_8^4 = -1
        W8[4][1] = _mm256_set_pd(0.0, 1.0, 0.0, 1.0);   // j=2: W_8^8 = 1
        W8[4][2] = _mm256_set_pd(0.0, -1.0, 0.0, -1.0); // j=3: W_8^12 = -1

        // W_8^5 = (√2/2)(-1 + i)
        W8[5][0] = _mm256_set_pd(c8, -c8, c8, -c8);
        W8[5][1] = _mm256_set_pd(1.0, 0.0, 1.0, 0.0); // j=2: W_8^10 = +i
        W8[5][2] = _mm256_set_pd(-c8, -c8, -c8, -c8);

        // W_8^6 = +i
        W8[6][0] = _mm256_set_pd(1.0, 0.0, 1.0, 0.0);
        W8[6][1] = _mm256_set_pd(0.0, 1.0, 0.0, 1.0);   // j=2: W_8^12 = 1
        W8[6][2] = _mm256_set_pd(-1.0, 0.0, -1.0, 0.0); // j=3: W_8^18 = -i

        // W_8^7 = (√2/2)(1 + i)
        W8[7][0] = _mm256_set_pd(c8, c8, c8, c8);
        W8[7][1] = _mm256_set_pd(-1.0, 0.0, -1.0, 0.0); // j=2: W_8^14 = -i
        W8[7][2] = _mm256_set_pd(-c8, c8, -c8, c8);

        // Sign adjustment for transform direction
        const __m256d sign_mask = (transform_sign == 1)
                                      ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)  // Forward: flip imaginary
                                      : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); // Inverse: flip real

        // Apply sign correction to all W8 twiddles
        __m256d W8_corrected[8][3];
        for (int m = 1; m < 8; ++m)
        {
            for (int j = 0; j < 3; ++j)
            {
                W8_corrected[m][j] = (transform_sign == -1)
                                         ? _mm256_xor_pd(W8[m][j], sign_mask)
                                         : W8[m][j];
            }
        }

        //----------------------------------------------------------------------
        // Main loop: 4x unrolling
        //----------------------------------------------------------------------
        for (; k + 7 < thirtysecond; k += 8)
        {
            if (k + 16 < thirtysecond)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[31 * (k + 16)].re, _MM_HINT_T0);
            }

            // Load all 32 lanes (4 butterflies × 32 lanes = 128 AVX2 loads)
            __m256d x[32][4]; // [lane][butterfly_pair]
            for (int lane = 0; lane < 32; ++lane)
            {
                x[lane][0] = load2_aos(&sub_outputs[k + 0 + lane * thirtysecond],
                                       &sub_outputs[k + 1 + lane * thirtysecond]);
                x[lane][1] = load2_aos(&sub_outputs[k + 2 + lane * thirtysecond],
                                       &sub_outputs[k + 3 + lane * thirtysecond]);
                x[lane][2] = load2_aos(&sub_outputs[k + 4 + lane * thirtysecond],
                                       &sub_outputs[k + 5 + lane * thirtysecond]);
                x[lane][3] = load2_aos(&sub_outputs[k + 6 + lane * thirtysecond],
                                       &sub_outputs[k + 7 + lane * thirtysecond]);
            }

            //==================================================================
            // Stage 1: Apply input twiddles W^{jk}
            //==================================================================
            for (int lane = 1; lane < 32; ++lane)
            {
                __m256d tw0 = load2_aos(&stage_tw[31 * (k + 0) + (lane - 1)],
                                        &stage_tw[31 * (k + 1) + (lane - 1)]);
                __m256d tw1 = load2_aos(&stage_tw[31 * (k + 2) + (lane - 1)],
                                        &stage_tw[31 * (k + 3) + (lane - 1)]);
                __m256d tw2 = load2_aos(&stage_tw[31 * (k + 4) + (lane - 1)],
                                        &stage_tw[31 * (k + 5) + (lane - 1)]);
                __m256d tw3 = load2_aos(&stage_tw[31 * (k + 6) + (lane - 1)],
                                        &stage_tw[31 * (k + 7) + (lane - 1)]);

                x[lane][0] = cmul_avx2_aos(x[lane][0], tw0);
                x[lane][1] = cmul_avx2_aos(x[lane][1], tw1);
                x[lane][2] = cmul_avx2_aos(x[lane][2], tw2);
                x[lane][3] = cmul_avx2_aos(x[lane][3], tw3);
            }

            //==================================================================
            // Stage 2: First radix-4 (8 groups, stride 8)
            //==================================================================
            for (int g = 0; g < 8; ++g)
            {
                for (int b = 0; b < 4; ++b)
                {
                    radix4_butterfly_aos(&x[g][b], &x[g + 8][b],
                                         &x[g + 16][b], &x[g + 24][b],
                                         transform_sign);
                }
            }

            //==================================================================
            // Stage 2.5: Apply intermediate twiddles W_8^{jm} (PRECOMPUTED!)
            //==================================================================
            for (int m = 1; m < 8; ++m) // m=0 has trivial twiddles
            {
                for (int j = 1; j <= 3; ++j)
                {
                    int idx = 4 * m + j;
                    for (int b = 0; b < 4; ++b)
                    {
                        x[idx][b] = cmul_avx2_aos(x[idx][b], W8_corrected[m][j - 1]);
                    }
                }
            }

            //==================================================================
            // Stage 3: Second radix-4 (8 groups, stride 1)
            //==================================================================
            for (int g = 0; g < 8; ++g)
            {
                int base = 4 * g;
                for (int b = 0; b < 4; ++b)
                {
                    radix4_butterfly_aos(&x[base][b], &x[base + 1][b],
                                         &x[base + 2][b], &x[base + 3][b],
                                         transform_sign);
                }
            }

            //==================================================================
            // Stage 3.5: Apply intermediate twiddles W_2^m = (-1)^m
            //==================================================================
            const __m256d neg_all = _mm256_set_pd(-0.0, -0.0, -0.0, -0.0);
            for (int m = 0; m < 16; ++m)
            {
                if (m % 2 == 1)
                {
                    int idx = 2 * m + 1;
                    for (int b = 0; b < 4; ++b)
                    {
                        x[idx][b] = _mm256_xor_pd(x[idx][b], neg_all);
                    }
                }
            }

            //==================================================================
            // Stage 4: Final radix-2 (16 groups)
            //==================================================================
            for (int g = 0; g < 16; ++g)
            {
                for (int b = 0; b < 4; ++b)
                {
                    radix2_butterfly_aos(&x[2 * g][b], &x[2 * g + 1][b]);
                }
            }

            //==================================================================
            // Store results
            //==================================================================
            for (int m = 0; m < 32; ++m)
            {
                STOREU_PD(&output_buffer[k + 0 + m * thirtysecond].re, x[m][0]);
                STOREU_PD(&output_buffer[k + 2 + m * thirtysecond].re, x[m][1]);
                STOREU_PD(&output_buffer[k + 4 + m * thirtysecond].re, x[m][2]);
                STOREU_PD(&output_buffer[k + 6 + m * thirtysecond].re, x[m][3]);
            }
        }

        //----------------------------------------------------------------------
        // Cleanup: 2x unrolling
        //----------------------------------------------------------------------
        for (; k + 1 < thirtysecond; k += 2)
        {
            if (k + 8 < thirtysecond)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[31 * (k + 8)].re, _MM_HINT_T0);
            }

            // Load all 32 lanes (2 butterflies per lane)
            __m256d x[32];
            for (int lane = 0; lane < 32; ++lane)
            {
                x[lane] = load2_aos(&sub_outputs[k + lane * thirtysecond],
                                    &sub_outputs[k + lane * thirtysecond + 1]);
            }

            //======================================================================
            // Stage 1: Apply input twiddles W^{jk}
            //======================================================================
            for (int lane = 1; lane < 32; ++lane)
            {
                __m256d tw = load2_aos(&stage_tw[31 * k + (lane - 1)],
                                       &stage_tw[31 * (k + 1) + (lane - 1)]);
                x[lane] = cmul_avx2_aos(x[lane], tw);
            }

            //======================================================================
            // Stage 2: First radix-4 decomposition (8 groups, stride 8)
            //======================================================================
            for (int g = 0; g < 8; ++g)
            {
                radix4_butterfly_aos(&x[g], &x[g + 8], &x[g + 16], &x[g + 24],
                                     transform_sign);
            }

            //======================================================================
            // Stage 2.5: Apply intermediate twiddles W_8^{jm} (PRECOMPUTED!)
            //======================================================================
            for (int m = 1; m < 8; ++m) // m=0 has trivial twiddles
            {
                for (int j = 1; j <= 3; ++j)
                {
                    int idx = 4 * m + j;
                    x[idx] = cmul_avx2_aos(x[idx], W8_corrected[m][j - 1]);
                }
            }

            //======================================================================
            // Stage 3: Second radix-4 decomposition (8 groups, stride 1)
            //======================================================================
            for (int g = 0; g < 8; ++g)
            {
                int base = 4 * g;
                radix4_butterfly_aos(&x[base], &x[base + 1],
                                     &x[base + 2], &x[base + 3],
                                     transform_sign);
            }

            //======================================================================
            // Stage 3.5: Apply intermediate twiddles W_2^m = (-1)^m
            //======================================================================
            const __m256d neg_all = _mm256_set_pd(-0.0, -0.0, -0.0, -0.0);
            for (int m = 0; m < 16; ++m)
            {
                if (m % 2 == 1)
                {
                    int idx = 2 * m + 1;
                    x[idx] = _mm256_xor_pd(x[idx], neg_all);
                }
            }

            //======================================================================
            // Stage 4: Final radix-2 decomposition (16 groups)
            //======================================================================
            for (int g = 0; g < 16; ++g)
            {
                radix2_butterfly_aos(&x[2 * g], &x[2 * g + 1]);
            }

            //======================================================================
            // Store results
            //======================================================================
            for (int m = 0; m < 32; ++m)
            {
                STOREU_PD(&output_buffer[k + m * thirtysecond].re, x[m]);
            }
        }

#endif // __AVX2__

        //----------------------------------------------------------------------
        // Scalar tail: Precompute twiddles ONCE
        //----------------------------------------------------------------------

        // Precompute W_8 twiddles for scalar path
        fft_data W8_scalar[8][3];
        for (int m = 0; m < 8; ++m)
        {
            for (int j = 1; j <= 3; ++j)
            {
                double angle = -2.0 * M_PI * j * m / 8.0;
                W8_scalar[m][j - 1].re = cos(angle);
                W8_scalar[m][j - 1].im = sin(angle) * transform_sign;
            }
        }

        for (; k < thirtysecond; ++k)
        {
            fft_data x[32];
            for (int lane = 0; lane < 32; ++lane)
            {
                x[lane] = sub_outputs[k + lane * thirtysecond];
            }

            // Stage 1: Input twiddles
            for (int lane = 1; lane < 32; ++lane)
            {
                const fft_data w = stage_tw[31 * k + (lane - 1)];
                const double rr = x[lane].re * w.re - x[lane].im * w.im;
                const double ri = x[lane].re * w.im + x[lane].im * w.re;
                x[lane].re = rr;
                x[lane].im = ri;
            }

            // Stage 2: First radix-4
            for (int g = 0; g < 8; ++g)
            {
                r4_butterfly(&x[g], &x[g + 8], &x[g + 16], &x[g + 24], transform_sign);
            }

            // Stage 2.5: Intermediate twiddles W_8 (PRECOMPUTED!)
            for (int m = 0; m < 8; ++m)
            {
                for (int j = 1; j <= 3; ++j)
                {
                    int idx = 4 * m + j;
                    const fft_data w = W8_scalar[m][j - 1];
                    const double xr = x[idx].re;
                    const double xi = x[idx].im;
                    x[idx].re = xr * w.re - xi * w.im;
                    x[idx].im = xr * w.im + xi * w.re;
                }
            }

            // Stage 3: Second radix-4
            for (int g = 0; g < 8; ++g)
            {
                int base = 4 * g;
                r4_butterfly(&x[base], &x[base + 1], &x[base + 2], &x[base + 3], transform_sign);
            }

            // Stage 3.5: Intermediate twiddles W_2
            for (int m = 0; m < 16; ++m)
            {
                if (m % 2 == 1)
                {
                    int idx = 2 * m + 1;
                    x[idx].re = -x[idx].re;
                    x[idx].im = -x[idx].im;
                }
            }

            // Stage 4: Final radix-2
            for (int g = 0; g < 16; ++g)
            {
                r2_butterfly(&x[2 * g], &x[2 * g + 1]);
            }

            // Store
            for (int m = 0; m < 32; ++m)
            {
                output_buffer[k + m * thirtysecond] = x[m];
            }
        }
    }
    else
    {
        //==========================================================================
        // GENERAL RADIX FALLBACK - HIGHLY OPTIMIZED
        //
        // Optimizations:
        // - Pure AoS throughout
        // - 4x/8x loop unrolling where possible
        // - Vectorized phase accumulation
        // - Special cases for m=0, m=r/2
        // - Prefetching and cache optimization
        //==========================================================================

        const int r = radix;
        const int K = data_length / r;
        const int next_stride = r * stride;
        const int nst = r - 1;

        // Scratch layout
        const int need_this = (fft_obj->twiddle_factors &&
                               factor_index < fft_obj->num_precomputed_stages)
                                  ? (r * K)
                                  : (r * K + nst * K);

        if (scratch_offset + need_this > fft_obj->max_scratch_size)
            return;

        fft_data *sub_outputs = fft_obj->scratch + scratch_offset;
        fft_data *stage_tw = NULL;
        int have_precomp = (fft_obj->twiddle_factors &&
                            factor_index < fft_obj->num_precomputed_stages);

        if (have_precomp)
            stage_tw = fft_obj->twiddle_factors +
                       fft_obj->stage_twiddle_offset[factor_index];
        else
            stage_tw = sub_outputs + r * K;

        // Recurse r children
        for (int j = 0; j < r; ++j)
        {
            mixed_radix_dit_rec(
                sub_outputs + j * K,
                input_buffer + j * stride,
                fft_obj, transform_sign,
                K, next_stride, factor_index + 1,
                scratch_offset);
        }

        // Build twiddles if needed
        if (!have_precomp)
        {
            const int N = r * K;
            for (int k = 0; k < K; ++k)
            {
                const int base = nst * k;
                for (int j = 1; j < r; ++j)
                {
                    const int idxN = (j * k) % N;
                    stage_tw[base + (j - 1)] = fft_obj->twiddles[idxN];
                }
            }
        }

        // Precompute W_r^m for m=0..r-1
        fft_data W_r[64];
        for (int m = 0; m < r; ++m)
        {
            double theta = 2.0 * M_PI * (double)m / (double)r;
            W_r[m].re = cos(theta);
            W_r[m].im = -(double)transform_sign * sin(theta);
        }

#ifdef __AVX2__
        //==========================================================================
        // AVX2: 4x unrolled with vectorized phase accumulation
        //==========================================================================
        int k = 0;

        // Main loop: process 4 butterflies at once (k, k+1, k+2, k+3)
        for (; k + 3 < K; k += 4)
        {
            // Prefetch
            if (k + 12 < K)
            {
                for (int j = 0; j < r; ++j)
                {
                    _mm_prefetch((const char *)&sub_outputs[j * K + k + 12].re, _MM_HINT_T0);
                    if (j < r - 1)
                        _mm_prefetch((const char *)&stage_tw[nst * (k + 12) + j].re, _MM_HINT_T0);
                }
            }

            // Apply stage twiddles to all lanes
            // T[j][i] holds butterfly i for lane j (i=0,1 in lower/upper halves of __m256d)
            __m256d T[64][2]; // [lane][pair_index]

            // Lane 0: no twiddle
            T[0][0] = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
            T[0][1] = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);

            // Lanes 1..r-1: apply stage twiddles
            for (int j = 1; j < r; ++j)
            {
                __m256d a0 = load2_aos(&sub_outputs[j * K + k],
                                       &sub_outputs[j * K + k + 1]);
                __m256d a1 = load2_aos(&sub_outputs[j * K + k + 2],
                                       &sub_outputs[j * K + k + 3]);

                __m256d w0 = load2_aos(&stage_tw[nst * k + (j - 1)],
                                       &stage_tw[nst * (k + 1) + (j - 1)]);
                __m256d w1 = load2_aos(&stage_tw[nst * (k + 2) + (j - 1)],
                                       &stage_tw[nst * (k + 3) + (j - 1)]);

                T[j][0] = cmul_avx2_aos(a0, w0);
                T[j][1] = cmul_avx2_aos(a1, w1);
            }

            // Compute outputs for each m
            for (int m = 0; m < r; ++m)
            {
                __m256d sum0, sum1;

                if (m == 0)
                {
                    //==============================================================
                    // FAST PATH m=0: W_r[0]^j = 1 for all j, just sum
                    //==============================================================
                    sum0 = T[0][0];
                    sum1 = T[0][1];
                    for (int j = 1; j < r; ++j)
                    {
                        sum0 = _mm256_add_pd(sum0, T[j][0]);
                        sum1 = _mm256_add_pd(sum1, T[j][1]);
                    }
                }
                else if (r % 2 == 0 && m == r / 2)
                {
                    //==============================================================
                    // FAST PATH m=r/2: W_r[r/2] = -1, so alternating signs
                    //==============================================================
                    sum0 = T[0][0];
                    sum1 = T[0][1];
                    for (int j = 1; j < r; ++j)
                    {
                        if (j % 2 == 1)
                        {
                            sum0 = _mm256_sub_pd(sum0, T[j][0]);
                            sum1 = _mm256_sub_pd(sum1, T[j][1]);
                        }
                        else
                        {
                            sum0 = _mm256_add_pd(sum0, T[j][0]);
                            sum1 = _mm256_add_pd(sum1, T[j][1]);
                        }
                    }
                }
                else
                {
                    //==============================================================
                    // GENERAL PATH: Compute phase powers and accumulate
                    //==============================================================
                    const fft_data step = W_r[m];
                    __m256d step_vec = _mm256_set_pd(step.im, step.re, step.im, step.re);

                    // Precompute first few powers
                    fft_data ph = {1.0, 0.0}; // ph^0
                    __m256d ph_vec = _mm256_set_pd(0.0, 1.0, 0.0, 1.0);

                    sum0 = T[0][0];
                    sum1 = T[0][1];

                    // Unroll inner loop by 2 for better ILP
                    int j = 1;
                    for (; j + 1 < r; j += 2)
                    {
                        // ph *= step (for j)
                        double new_re = ph.re * step.re - ph.im * step.im;
                        double new_im = ph.re * step.im + ph.im * step.re;
                        ph.re = new_re;
                        ph.im = new_im;
                        __m256d ph_vec1 = _mm256_set_pd(ph.im, ph.re, ph.im, ph.re);

                        // Accumulate j
                        __m256d term0 = cmul_avx2_aos(T[j][0], ph_vec1);
                        __m256d term1 = cmul_avx2_aos(T[j][1], ph_vec1);
                        sum0 = _mm256_add_pd(sum0, term0);
                        sum1 = _mm256_add_pd(sum1, term1);

                        // ph *= step (for j+1)
                        new_re = ph.re * step.re - ph.im * step.im;
                        new_im = ph.re * step.im + ph.im * step.re;
                        ph.re = new_re;
                        ph.im = new_im;
                        __m256d ph_vec2 = _mm256_set_pd(ph.im, ph.re, ph.im, ph.re);

                        // Accumulate j+1
                        term0 = cmul_avx2_aos(T[j + 1][0], ph_vec2);
                        term1 = cmul_avx2_aos(T[j + 1][1], ph_vec2);
                        sum0 = _mm256_add_pd(sum0, term0);
                        sum1 = _mm256_add_pd(sum1, term1);
                    }

                    // Handle remaining j (if r-1 is odd)
                    for (; j < r; ++j)
                    {
                        double new_re = ph.re * step.re - ph.im * step.im;
                        double new_im = ph.re * step.im + ph.im * step.re;
                        ph.re = new_re;
                        ph.im = new_im;
                        __m256d ph_vec_final = _mm256_set_pd(ph.im, ph.re, ph.im, ph.re);

                        __m256d term0 = cmul_avx2_aos(T[j][0], ph_vec_final);
                        __m256d term1 = cmul_avx2_aos(T[j][1], ph_vec_final);
                        sum0 = _mm256_add_pd(sum0, term0);
                        sum1 = _mm256_add_pd(sum1, term1);
                    }
                }

                // Store results
                STOREU_PD(&output_buffer[m * K + k].re, sum0);
                STOREU_PD(&output_buffer[m * K + k + 2].re, sum1);
            }
        }

        //==========================================================================
        // Cleanup: 2x unrolled
        //==========================================================================
        for (; k + 1 < K; k += 2)
        {
            if (k + 8 < K)
            {
                for (int j = 0; j < r; ++j)
                    _mm_prefetch((const char *)&sub_outputs[j * K + k + 8].re, _MM_HINT_T0);
            }

            __m256d T[64];
            T[0] = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);

            for (int j = 1; j < r; ++j)
            {
                __m256d a = load2_aos(&sub_outputs[j * K + k],
                                      &sub_outputs[j * K + k + 1]);
                __m256d w = load2_aos(&stage_tw[nst * k + (j - 1)],
                                      &stage_tw[nst * (k + 1) + (j - 1)]);
                T[j] = cmul_avx2_aos(a, w);
            }

            for (int m = 0; m < r; ++m)
            {
                __m256d sum = T[0];

                if (m == 0)
                {
                    for (int j = 1; j < r; ++j)
                        sum = _mm256_add_pd(sum, T[j]);
                }
                else if (r % 2 == 0 && m == r / 2)
                {
                    for (int j = 1; j < r; ++j)
                        sum = (j % 2 == 1) ? _mm256_sub_pd(sum, T[j])
                                           : _mm256_add_pd(sum, T[j]);
                }
                else
                {
                    fft_data ph = {1.0, 0.0};
                    const fft_data step = W_r[m];

                    for (int j = 1; j < r; ++j)
                    {
                        double new_re = ph.re * step.re - ph.im * step.im;
                        double new_im = ph.re * step.im + ph.im * step.re;
                        ph.re = new_re;
                        ph.im = new_im;

                        __m256d ph_vec = _mm256_set_pd(ph.im, ph.re, ph.im, ph.re);
                        __m256d term = cmul_avx2_aos(T[j], ph_vec);
                        sum = _mm256_add_pd(sum, term);
                    }
                }

                STOREU_PD(&output_buffer[m * K + k].re, sum);
            }
        }
#else
        int k = 0;
#endif

        //==========================================================================
        // SCALAR TAIL
        //==========================================================================
        for (; k < K; ++k)
        {
            if (k + 8 < K)
            {
                for (int j = 0; j < r; ++j)
                    _mm_prefetch((const char *)&sub_outputs[j * K + k + 8].re, _MM_HINT_T0);
            }

            fft_data T[64];
            T[0] = sub_outputs[k];

            for (int j = 1; j < r; ++j)
            {
                const fft_data a = sub_outputs[j * K + k];
                const fft_data w = stage_tw[nst * k + (j - 1)];
                T[j].re = a.re * w.re - a.im * w.im;
                T[j].im = a.re * w.im + a.im * w.re;
            }

            for (int m = 0; m < r; ++m)
            {
                double sum_re = T[0].re;
                double sum_im = T[0].im;

                if (m == 0)
                {
                    for (int j = 1; j < r; ++j)
                    {
                        sum_re += T[j].re;
                        sum_im += T[j].im;
                    }
                }
                else if (r % 2 == 0 && m == r / 2)
                {
                    for (int j = 1; j < r; ++j)
                    {
                        if (j % 2 == 1)
                        {
                            sum_re -= T[j].re;
                            sum_im -= T[j].im;
                        }
                        else
                        {
                            sum_re += T[j].re;
                            sum_im += T[j].im;
                        }
                    }
                }
                else
                {
                    fft_data ph = {1.0, 0.0};
                    const fft_data step = W_r[m];

                    for (int j = 1; j < r; ++j)
                    {
                        double new_re = ph.re * step.re - ph.im * step.im;
                        double new_im = ph.re * step.im + ph.im * step.re;
                        ph.re = new_re;
                        ph.im = new_im;

                        sum_re += T[j].re * ph.re - T[j].im * ph.im;
                        sum_im += T[j].re * ph.im + T[j].im * ph.re;
                    }
                }

                output_buffer[m * K + k].re = sum_re;
                output_buffer[m * K + k].im = sum_im;
            }
        }
    }
}

static inline int find_pre_idx(int N)
{
    for (int i = 0; i < num_pre; ++i)
        if (pre_sizes[i] == N)
            return i;
    return -1;
}

/**
 * @brief Computes the exponential terms for Bluestein’s FFT algorithm.
 *
 * Generates the exponential (chirp) sequence used in Bluestein’s algorithm to handle arbitrary-length FFTs.
 * Uses precomputed tables for small N (<= MAX_PRECOMPUTED_N) and dynamic computation for larger N.
 * Stores results in both `hl` (padded sequence) and `hlt` (temporary sequence).
 *
 * @param[out] hl Padded exponential sequence (length M).
 * @param[out] hlt Temporary exponential sequence (length len).
 * @param[in] input_length Length of the input signal (len > 0).
 * @param[in] padded_length Padded length for Bluestein’s algorithm (M > len).
 * @warning If lengths are invalid, the function exits with an error.
 * @note Uses PI = 3.1415926535897932384626433832795 for dynamic calculations.
 */
static inline void bluestein_exp(fft_data *temp_scratch, fft_data *chirp_out, int N, int M)
{
    pthread_once(&chirp_init_once, init_bluestein_chirp_body);

    // 1) Produce chirp_out[n] = exp(+i*pi*n^2/N), n = 0..N-1
    int pre = find_pre_idx(N);
    if (pre >= 0)
    {
        memcpy(chirp_out, bluestein_chirp[pre], (size_t)N * sizeof(fft_data));
    }
    else
    {
        const double theta = M_PI / (double)N;
        int l2 = 0, len2 = 2 * N;
        for (int n = 0; n < N; ++n)
        {
            double angle = theta * (double)l2;
            chirp_out[n].re = cos(angle);
            chirp_out[n].im = sin(angle);
            l2 += 2 * n + 1; // (n+1)^2 - n^2 = 2n+1
            while (l2 >= len2)
                l2 -= len2; // wrap mod 2N
        }
    }

    // 2) Build kernel b of length M:
    //    b[0] = 1
    //    for n=1..N-1: b[n] = conj(chirp_out[n]) = exp(-i*pi*n^2/N), b[M-n] = b[n]
    for (int i = 0; i < M; ++i)
    {
        temp_scratch[i].re = 0.0;
        temp_scratch[i].im = 0.0;
    }
    temp_scratch[0].re = 1.0;
    temp_scratch[0].im = 0.0;

    const int lim = (N > 1) ? (N - 1) : 0;
    for (int n = 1; n <= lim; ++n)
    {
        fft_data z;
        z.re = chirp_out[n].re;
        z.im = -chirp_out[n].im; // conjugate -> exp(-i*pi*n^2/N)

        temp_scratch[n] = z;
        temp_scratch[M - n] = z; // symmetric placement
    }
}

/**
 * @brief Runs Bluestein's FFT for signals of any length, with AVX2 speed-ups.
 *
 * This function handles FFTs for signal lengths that don’t nicely factor into small primes, using
 * Bluestein’s chirp z-transform trick. It turns the DFT into a convolution with a chirp sequence,
 * padded to a power-of-2 length so we can use fast FFTs. We’re leaning hard into AVX2 for speed and
 * using a memory setup inspired by FFTW—two big buffers (twiddles and scratch)—plus a single global
 * `all_chirps` array for precomputed chirp sequences.
 *
 * **What’s the math?** Basically, Bluestein rewrites the DFT \( X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-2\pi i k n / N} \)
 * using the identity \( k n = \frac{1}{2} [k^2 + n^2 - (k-n)^2] \). This gives us a convolution we
 * can compute efficiently in the frequency domain with FFTs of size M >= 2N-1 (rounded up to a power of 2).
 *
 * @param[in] input_signal The input data (length N), real and imaginary parts.
 * @param[out] output_signal Where we store the FFT results (length N).
 * @param[in,out] fft_config Our FFT setup, with signal length, direction, twiddles, and scratch space.
 * @param[in] transform_direction +1 for forward FFT, -1 for inverse (affects chirp phase).
 * @param[in] signal_length The input length (N > 0).
 *
 * @warning Crashes with an error if signal_length <= 0, scratch space is too small, or we can’t
 *          create a temporary FFT object.
 * @note Assumes input_signal, output_signal, and fft_config are valid. Scratch buffer must be
 *       pre-allocated in fft_init to hold at least 4 * padded_length elements.
 */
void bluestein_fft(
    const fft_data *input_signal,
    fft_data *output_signal,
    fft_object fft_config,
    int transform_direction,
    int signal_length)
{
    if (signal_length <= 0)
    {
        fprintf(stderr, "Error: Signal length (%d) is invalid\n", signal_length);
        return;
    }

    const int N = signal_length;

    // Choose M = next power of two >= 2N-1
    int M = 1;
    int need = 2 * N - 1;
    while (M < need)
        M <<= 1;

    // Scratch layout: 4*M complexes
    if (4 * M > fft_config->max_scratch_size)
    {
        fprintf(stderr, "Error: Scratch too small for Bluestein: need %d, have %d\n",
                4 * M, fft_config->max_scratch_size);
        return;
    }

    fft_data *S = fft_config->scratch;
    fft_data *B_time = S;
    fft_data *B_fft = S + M;
    fft_data *A_fft_or_time = S + 2 * M;
    fft_data *base_chirp = S + 3 * M;

    // Create FFT plans
    fft_object plan_fwd = fft_init(M, +1);
    fft_object plan_inv = fft_init(M, -1);
    if (!plan_fwd || !plan_inv)
    {
        fprintf(stderr, "Error: Couldn't create Bluestein FFT plans\n");
        if (plan_fwd)
            free_fft(plan_fwd);
        if (plan_inv)
            free_fft(plan_inv);
        return;
    }

    // Get base chirp: exp(+i*pi*n^2/N) for n=0..N-1
    bluestein_exp(A_fft_or_time, base_chirp, N, M);

    //==========================================================================
    // CORRECTED SIGN CONVENTION
    //==========================================================================
    // DFT: X[k] = sum_{n=0}^{N-1} x[n] * exp(-2πi*k*n/N)
    //
    // Bluestein identity: k*n = (k²+n²-(k-n)²)/2
    // So: exp(-2πi*k*n/N) = exp(-πi*k²/N) * exp(-πi*n²/N) * exp(+πi*(k-n)²/N)
    //
    // Forward FFT (transform_direction = +1):
    //   - Multiply input by exp(-πi*n²/N) = conj(base_chirp[n])
    //   - Convolve with kernel exp(+πi*m²/N) = base_chirp[m]
    //   - Multiply output by exp(-πi*k²/N) = conj(base_chirp[k])
    //
    // Inverse FFT (transform_direction = -1):
    //   - Multiply input by exp(+πi*n²/N) = base_chirp[n]
    //   - Convolve with kernel exp(-πi*m²/N) = conj(base_chirp[m])
    //   - Multiply output by exp(+πi*k²/N) = base_chirp[k]
    //==========================================================================

    const int use_conjugate_input = (transform_direction == +1);  // Forward uses conj
    const int use_conjugate_kernel = (transform_direction == -1); // Inverse uses conj

    //--------------------------------------------------------------------------
    // Build kernel B_time (mirrored) with pre-scaling by 1/M
    //--------------------------------------------------------------------------
    for (int i = 0; i < M; ++i)
    {
        B_time[i].re = 0.0;
        B_time[i].im = 0.0;
    }
    B_time[0].re = 1.0;

    if (use_conjugate_kernel)
    {
        // Inverse: kernel = conj(base_chirp) = exp(-πi*m²/N)
        for (int n = 1; n < N; ++n)
        {
            B_time[n].re = base_chirp[n].re;
            B_time[n].im = -base_chirp[n].im;
            B_time[M - n].re = base_chirp[n].re;
            B_time[M - n].im = -base_chirp[n].im;
        }
    }
    else
    {
        // Forward: kernel = base_chirp = exp(+πi*m²/N)
        for (int n = 1; n < N; ++n)
        {
            B_time[n] = base_chirp[n];
            B_time[M - n] = base_chirp[n];
        }
    }

    // Pre-scale kernel by 1/M
    const double invM = 1.0 / (double)M;
#if defined(__AVX2__)
    {
        int i = 0;
        const __m256d vscale = _mm256_set1_pd(invM);
        for (; i + 1 < M; i += 2)
        {
            __m256d v = LOADU_PD(&B_time[i].re);
            v = _mm256_mul_pd(v, vscale);
            STOREU_PD(&B_time[i].re, v);
        }
        if (i < M)
        {
            __m128d t = LOADU_SSE2(&B_time[i].re);
            t = _mm_mul_pd(t, _mm_set1_pd(invM));
            STOREU_SSE2(&B_time[i].re, t);
        }
    }
#else
    for (int i = 0; i < M; ++i)
    {
        B_time[i].re *= invM;
        B_time[i].im *= invM;
    }
#endif

    // FFT of kernel
    fft_exec(plan_fwd, B_time, B_fft);

    //--------------------------------------------------------------------------
    // Build A_time: input * chirp, zero-padded
    //--------------------------------------------------------------------------
#if defined(__AVX2__)
    {
        int n = 0;
        if (use_conjugate_input)
        {
            // Forward: multiply by conj(base_chirp) = exp(-πi*n²/N)
            const __m256d conj_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
            for (; n + 1 < N; n += 2)
            {
                __m256d x12 = LOADU_PD(&input_signal[n].re);
                __m256d c12 = LOADU_PD(&base_chirp[n].re);
                c12 = _mm256_xor_pd(c12, conj_mask); // Conjugate
                __m256d a12 = cmul_avx2_aos(x12, c12);
                STOREU_PD(&A_fft_or_time[n].re, a12);
            }
            if (n < N)
            {
                __m128d x1 = LOADU_SSE2(&input_signal[n].re);
                __m128d c1 = LOADU_SSE2(&base_chirp[n].re);
                c1 = _mm_xor_pd(c1, _mm_set_pd(-0.0, 0.0));
                __m128d a1 = cmul_sse2_aos(x1, c1);
                STOREU_SSE2(&A_fft_or_time[n].re, a1);
            }
        }
        else
        {
            // Inverse: multiply by base_chirp = exp(+πi*n²/N)
            for (; n + 1 < N; n += 2)
            {
                __m256d x12 = LOADU_PD(&input_signal[n].re);
                __m256d c12 = LOADU_PD(&base_chirp[n].re);
                __m256d a12 = cmul_avx2_aos(x12, c12);
                STOREU_PD(&A_fft_or_time[n].re, a12);
            }
            if (n < N)
            {
                __m128d x1 = LOADU_SSE2(&input_signal[n].re);
                __m128d c1 = LOADU_SSE2(&base_chirp[n].re);
                __m128d a1 = cmul_sse2_aos(x1, c1);
                STOREU_SSE2(&A_fft_or_time[n].re, a1);
            }
        }
    }
#else
    for (int n = 0; n < N; ++n)
    {
        double xr = input_signal[n].re, xi = input_signal[n].im;
        double cr = base_chirp[n].re, ci = base_chirp[n].im;
        if (use_conjugate_input)
            ci = -ci;
        A_fft_or_time[n].re = xr * cr - xi * ci;
        A_fft_or_time[n].im = xi * cr + xr * ci;
    }
#endif

    // Zero-pad
    for (int i = N; i < M; ++i)
    {
        A_fft_or_time[i].re = 0.0;
        A_fft_or_time[i].im = 0.0;
    }

    // FFT(A) -> store in B_time temporarily
    fft_exec(plan_fwd, A_fft_or_time, B_time);

    //--------------------------------------------------------------------------
    // Pointwise multiply: FFT(A) * FFT(B)
    //--------------------------------------------------------------------------
#if defined(__AVX2__)
    {
        int i = 0;
        for (; i + 1 < M; i += 2)
        {
            __m256d Af = LOADU_PD(&B_time[i].re);
            __m256d Bf = LOADU_PD(&B_fft[i].re);
            __m256d Cf = cmul_avx2_aos(Af, Bf);
            STOREU_PD(&A_fft_or_time[i].re, Cf);
        }
        if (i < M)
        {
            __m128d Af = LOADU_SSE2(&B_time[i].re);
            __m128d Bf = LOADU_SSE2(&B_fft[i].re);
            __m128d Cf = cmul_sse2_aos(Af, Bf);
            STOREU_SSE2(&A_fft_or_time[i].re, Cf);
        }
    }
#else
    for (int i = 0; i < M; ++i)
    {
        double ar = B_time[i].re, ai = B_time[i].im;
        double br = B_fft[i].re, bi = B_fft[i].im;
        A_fft_or_time[i].re = ar * br - ai * bi;
        A_fft_or_time[i].im = ai * br + ar * bi;
    }
#endif

    // IFFT (kernel was pre-scaled, so this gives true convolution)
    fft_exec(plan_inv, A_fft_or_time, B_time);

    //--------------------------------------------------------------------------
    // Final chirp multiply
    //--------------------------------------------------------------------------
#if defined(__AVX2__)
    {
        int k = 0;
        if (use_conjugate_input)
        {
            // Forward: multiply by conj(base_chirp) = exp(-πi*k²/N)
            const __m256d conj_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
            for (; k + 1 < N; k += 2)
            {
                __m256d y = LOADU_PD(&B_time[k].re);
                __m256d ck = LOADU_PD(&base_chirp[k].re);
                ck = _mm256_xor_pd(ck, conj_mask);
                __m256d out = cmul_avx2_aos(y, ck);
                STOREU_PD(&output_signal[k].re, out);
            }
            if (k < N)
            {
                __m128d y = LOADU_SSE2(&B_time[k].re);
                __m128d ck = LOADU_SSE2(&base_chirp[k].re);
                ck = _mm_xor_pd(ck, _mm_set_pd(-0.0, 0.0));
                __m128d out = cmul_sse2_aos(y, ck);
                STOREU_SSE2(&output_signal[k].re, out);
            }
        }
        else
        {
            // Inverse: multiply by base_chirp = exp(+πi*k²/N)
            for (; k + 1 < N; k += 2)
            {
                __m256d y = LOADU_PD(&B_time[k].re);
                __m256d ck = LOADU_PD(&base_chirp[k].re);
                __m256d out = cmul_avx2_aos(y, ck);
                STOREU_PD(&output_signal[k].re, out);
            }
            if (k < N)
            {
                __m128d y = LOADU_SSE2(&B_time[k].re);
                __m128d ck = LOADU_SSE2(&base_chirp[k].re);
                __m128d out = cmul_sse2_aos(y, ck);
                STOREU_SSE2(&output_signal[k].re, out);
            }
        }
    }
#else
    for (int k = 0; k < N; ++k)
    {
        double yr = B_time[k].re, yi = B_time[k].im;
        double cr = base_chirp[k].re, ci = base_chirp[k].im;
        if (use_conjugate_input)
            ci = -ci;
        output_signal[k].re = yr * cr - yi * ci;
        output_signal[k].im = yi * cr + yr * ci;
    }
#endif

    free_fft(plan_inv);
    free_fft(plan_fwd);
}

/**
 * @brief Executes the FFT on input data using the configured FFT object.
 *
 * Performs either a mixed-radix DIT FFT or Bluestein’s FFT based on the object’s configuration.
 *
 * @param[in] fft_obj FFT configuration object.
 * @param[in] input_data Input signal data (length N).
 * @param[out] output_data Output FFT results (length N).
 * @warning If the FFT object or data pointers are invalid, the function exits with an error.
 * @note Uses `mixed_radix_dit_rec` for power-of-N sizes or `bluestein_fft` for arbitrary sizes.
 */
void fft_exec(fft_object fft_obj, fft_data *inp, fft_data *oup)
{
    // Check for null pointers to avoid crashes
    // Ensure we have a valid FFT object and input/output buffers
    if (fft_obj == NULL || inp == NULL || oup == NULL)
    {
        fprintf(stderr, "Error: Invalid FFT object or data pointers\n");
        // exit
    }

    // Dispatch based on the FFT algorithm type
    // lt = 0 for mixed-radix (factorable lengths), lt = 1 for Bluestein (non-factorable)
    if (fft_obj->lt == 0)
    {
        // Set up for mixed-radix FFT
        // Start with stride=1, factor_index=0, and scratch_offset=0 for recursion
        int stride = 1;         // Initial stride for input indexing
        int factor_index = 0;   // Start at the first prime factor
        int scratch_offset = 0; // Initial offset for scratch buffer
        // Call mixed-radix FFT with transform length (n_fft)
        // Why n_fft? It’s the actual transform size (N for mixed-radix)
        mixed_radix_dit_rec(oup, inp, fft_obj, fft_obj->sgn,
                            fft_obj->n_fft, stride, factor_index, scratch_offset);
    }
    else if (fft_obj->lt == 1)
    {
        // Run Bluestein FFT for non-factorable lengths
        // Use n_input for input/output size, as Bluestein pads internally
        bluestein_fft(inp, oup, fft_obj, fft_obj->sgn, fft_obj->n_input);
    }
    else
    {
        // Handle invalid algorithm type
        // This shouldn’t happen unless fft_init is broken
        fprintf(stderr, "Error: Invalid FFT object type (lt = %d)\n", fft_obj->lt);
        // exit
    }
}

/**
 * @brief Checks if a number N is divisible by a series of small prime numbers.
 *
 * Uses a precomputed lookup table for small N (<= 1024) and falls back to division for larger N.
 *
 * @param[in] number Number to check for divisibility (N > 0).
 * @return int 1 if N is fully divisible by the primes, 0 otherwise.
 * @warning If N is invalid (<= 0), the function exits with an error.
 */
int dividebyN(int number)
{

    // Use lookup table for small N
    if (number < LOOKUP_MAX)
    {
        return dividebyN_lookup[number];
    }

    // Fallback for larger N
    int result = number;
    while (result % 53 == 0)
        result /= 53;
    while (result % 47 == 0)
        result /= 47;
    while (result % 43 == 0)
        result /= 43;
    while (result % 41 == 0)
        result /= 41;
    while (result % 37 == 0)
        result /= 37;
    while (result % 31 == 0)
        result /= 31;
    while (result % 29 == 0)
        result /= 29;
    while (result % 23 == 0)
        result /= 23;
    while (result % 17 == 0)
        result /= 17;
    while (result % 13 == 0)
        result /= 13;
    while (result % 11 == 0)
        result /= 11;
    while (result % 8 == 0)
        result /= 8;
    while (result % 7 == 0)
        result /= 7;
    while (result % 5 == 0)
        result /= 5;
    while (result % 4 == 0)
        result /= 4;
    while (result % 3 == 0)
        result /= 3;
    while (result % 2 == 0)
        result /= 2;
    return (result == 1) ? 1 : 0;
}

/**
 * @brief Computes the prime factorization of a number M into an array of factors.
 *
 * Decomposes M into its prime factors, storing them in the provided array up to a maximum of 32 factors.
 * Uses a list of primes and a heuristic for larger numbers.
 *
 * @param[in] number Number to factorize (M > 0).
 * @param[out] factors_array Array to store the prime factors (size at least 32).
 * @return int Number of factors found.
 * @warning If M is invalid (<= 0) or the array is NULL, the function exits with an error.
 */
int factors(int number, int *factors_array)
{
    if (factors_array == NULL)
    {
        fprintf(stderr, "Error: Invalid inputs for factors - number: %d, factors_array: %p\n",
                number, (void *)factors_array);
        // exit (left as-is in your codebase)
    }

    int index = 0, temp_number = number, prime, multiplier, factor1, factor2;

    // (unchanged) Large primes first
    while (temp_number % 53 == 0)
    {
        factors_array[index++] = 53;
        temp_number /= 53;
    }
    while (temp_number % 47 == 0)
    {
        factors_array[index++] = 47;
        temp_number /= 47;
    }
    while (temp_number % 43 == 0)
    {
        factors_array[index++] = 43;
        temp_number /= 43;
    }
    while (temp_number % 41 == 0)
    {
        factors_array[index++] = 41;
        temp_number /= 41;
    }
    while (temp_number % 37 == 0)
    {
        factors_array[index++] = 37;
        temp_number /= 37;
    }
    while (temp_number % 31 == 0)
    {
        factors_array[index++] = 31;
        temp_number /= 31;
    }
    while (temp_number % 29 == 0)
    {
        factors_array[index++] = 29;
        temp_number /= 29;
    }
    while (temp_number % 23 == 0)
    {
        factors_array[index++] = 23;
        temp_number /= 23;
    }
    while (temp_number % 19 == 0)
    {
        factors_array[index++] = 19;
        temp_number /= 19;
    }
    while (temp_number % 17 == 0)
    {
        factors_array[index++] = 17;
        temp_number /= 17;
    }
    while (temp_number % 13 == 0)
    {
        factors_array[index++] = 13;
        temp_number /= 13;
    }
    while (temp_number % 11 == 0)
    {
        factors_array[index++] = 11;
        temp_number /= 11;
    }

    while (temp_number % 32 == 0)
    {
        factors_array[index++] = 32;
        temp_number /= 32;
    } // NEW
    while (temp_number % 16 == 0)
    {
        factors_array[index++] = 16;
        temp_number /= 16;
    } // NEW
    // -------------------------------------------------------------------

    while (temp_number % 8 == 0)
    {
        factors_array[index++] = 8;
        temp_number /= 8;
    }
    while (temp_number % 7 == 0)
    {
        factors_array[index++] = 7;
        temp_number /= 7;
    }
    while (temp_number % 5 == 0)
    {
        factors_array[index++] = 5;
        temp_number /= 5;
    }
    while (temp_number % 4 == 0)
    {
        factors_array[index++] = 4;
        temp_number /= 4;
    }
    while (temp_number % 3 == 0)
    {
        factors_array[index++] = 3;
        temp_number /= 3;
    }
    while (temp_number % 2 == 0)
    {
        factors_array[index++] = 2;
        temp_number /= 2;
    }

    // (unchanged) heuristic 6k ± 1
    if (temp_number > 31)
    {
        prime = 2;
        while (temp_number > 1)
        {
            multiplier = prime * 6;
            factor1 = multiplier - 1;
            factor2 = multiplier + 1;
            while (temp_number % factor1 == 0)
            {
                factors_array[index++] = factor1;
                temp_number /= factor1;
            }
            while (temp_number % factor2 == 0)
            {
                factors_array[index++] = factor2;
                temp_number /= factor2;
            }
            prime++;
        }
    }

    return index;
}

/**
 * @brief Computes twiddle factors for a specific radix in the FFT using lookup tables.
 *
 * Generates complex exponential (twiddle) factors of the form \(e^{-2\pi i k / N}\) for a given signal length
 * and radix, used in butterfly operations of the FFT. For common radices (2, 3, 4, 5, 7, 8), precomputed lookup
 * tables are used to avoid runtime trigonometric calculations. For other radices or when the number of twiddles
 * exceeds the table size, the function falls back to computing factors dynamically using cosine and sine functions.
 *
 * @param[out] twiddle_factors Array to store twiddle factors (length N/radix).
 *                            Stores the real and imaginary components of the complex exponentials.
 * @param[in] signal_length Length of the signal (N > 0).
 *                         The total length of the signal for which twiddle factors are computed, determining the angle step.
 * @param[in] radix Radix for factorization (radix > 0).
 *                 The prime factor used to divide the signal length, determining the number of twiddle factors needed.
 * @return None (void function).
 * @warning If signal_length or radix is invalid (<= 0), the function exits with an error message to stderr.
 * @note For radices 2, 3, 4, 5, 7, and 8, precomputed tables are used up to radix-1 entries, with dynamic computation
 *       for additional twiddles if num_twiddles exceeds the table size. The first twiddle factor is always (1, 0),
 *       and subsequent factors follow the form \(e^{-2\pi i k / N}\). This optimization reduces computational overhead
 *       for repeated FFT calls with common sizes.
 */
void twiddle(fft_data *twiddle_factors, int signal_length, int radix)
{
    if (signal_length <= 0 || radix <= 0)
    {
        fprintf(stderr, "Error: Invalid inputs for twiddle - signal_length: %d, radix: %d\n", signal_length, radix);
        // exit
    }

    int num_twiddles = signal_length / radix; // Number of twiddle factors needed: N/radix

    if (radix <= 8 && twiddle_tables[radix])
    {
        // Use precomputed table for common radices (2, 3, 4, 5, 7, 8)
        const complex_t *table = twiddle_tables[radix]; // Pointer to static twiddle table for this radix
        for (int i = 0; i < num_twiddles && i < radix; i++)
        {
            // Copy precomputed values from table, cycling through radix-1 entries
            twiddle_factors[i].re = table[i % (radix - 1)].re; // Real part from table
            twiddle_factors[i].im = table[i % (radix - 1)].im; // Imaginary part from table
        }
        for (int i = radix - 1; i < num_twiddles; i++)
        {
            // Compute remaining twiddles dynamically if num_twiddles exceeds table size
            fft_type angle = PI2 * i / signal_length; // Angle for twiddle factor: 2πk/N
            twiddle_factors[i].re = cos(angle);       // Real part: cos(2πk/N)
            twiddle_factors[i].im = -sin(angle);      // Imaginary part: -sin(2πk/N) for forward FFT
        }
    }
    else
    {
        // Fallback for uncommon radices or when no table exists
        for (int i = 0; i < num_twiddles; i++)
        {
            fft_type angle = PI2 * i / signal_length; // Angle step: 2πk/N for k = 0 to N/radix - 1
            twiddle_factors[i].re = cos(angle);       // Real component of e^(-2πi k/N)
            twiddle_factors[i].im = -sin(angle);      // Imaginary component, negative for forward transform
        }
    }
}

/**
 * @brief Computes long twiddle factors for mixed-radix FFT using lookup tables where possible.
 *
 * Generates a sequence of complex exponential (twiddle) factors based on the prime factorization of the signal length,
 * used in mixed-radix FFT butterfly operations. Uses precomputed lookup tables for common radices (2, 3, 4, 5, 7, 8, 11, 13)
 * when USE_TWIDDLE_TABLES is defined, falling back to dynamic computation otherwise.
 *
 * @param[out] twiddle_sequence Array to store twiddle factors (length N-1).
 * @param[in] signal_length Length of the signal (N > 0).
 * @param[in] prime_factors Array of prime factors (size num_factors).
 * @param[in] num_factors Number of prime factors (num_factors > 0).
 */
void longvectorN(fft_data *twiddle_sequence, int signal_length, int *prime_factors, int num_factors)
{
    if (signal_length <= 0 || prime_factors == NULL || num_factors <= 0)
    {
        fprintf(stderr, "Error: Invalid inputs for longvectorN - signal_length: %d, prime_factors: %p, num_factors: %d\n",
                signal_length, (void *)prime_factors, num_factors);
        // exit
    }

    int cumulative_length = 1; // Tracks L as in original
    int counter = 0;

    for (int i = 0; i < num_factors; i++)
    {
        int radix = prime_factors[num_factors - 1 - i];
        int sub_length = cumulative_length; // Ls = L before update
        cumulative_length *= radix;         // L = L * radix

#ifdef USE_TWIDDLE_TABLES
        if (radix <= 13 && twiddle_tables[radix] != NULL)
        {
            const complex_t *table = twiddle_tables[radix];
            for (int j = 0; j < sub_length; j++)
            {
                for (int k = 0; k < radix - 1; k++)
                {
                    if (counter < signal_length - 1)
                    { // Prevent overflow
                        twiddle_sequence[counter].re = table[k].re;
                        twiddle_sequence[counter].im = table[k].im;
                        counter++;
                    }
                }
            }
        }
        else
        {
            fft_type theta = -PI2 / cumulative_length;
            for (int j = 0; j < sub_length; j++)
            {
                for (int k = 0; k < radix - 1; k++)
                {
                    if (counter < signal_length - 1)
                    {
                        fft_type angle = (k + 1) * j * theta;
                        twiddle_sequence[counter].re = cos(angle);
                        twiddle_sequence[counter].im = sin(angle);
                        counter++;
                    }
                }
            }
        }
#else
        fft_type theta = -PI2 / cumulative_length;
        for (int j = 0; j < sub_length; j++)
        {
            for (int k = 0; k < radix - 1; k++)
            {
                if (counter < signal_length - 1)
                {
                    fft_type angle = (k + 1) * j * theta;
                    twiddle_sequence[counter].re = cos(angle);
                    twiddle_sequence[counter].im = sin(angle);
                    counter++;
                }
            }
        }
#endif
    }

    if (counter != signal_length - 1)
    {
        fprintf(stderr, "Warning: Twiddle factor count (%d) does not match expected (%d) for N=%d\n",
                counter, signal_length - 1, signal_length);
    }
}

void free_fft(fft_object object)
{
    if (object)
    {
        if (object->twiddles)
            _mm_free(object->twiddles);
        if (object->scratch)
            _mm_free(object->scratch);
        if (object->twiddle_factors)
            _mm_free(object->twiddle_factors);
        free(object);
    }
}
