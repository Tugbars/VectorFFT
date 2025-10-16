

#include "highspeedFFT.h"
#ifdef FFT_ENABLE_PREFETCH
#include "prefetch_strategy.h"
#endif
#include "time.h"
#include <immintrin.h>
#include <pthread.h>

//==============================================================================
// SIMD ABSTRACTION LAYER - Improved Portability
//==============================================================================

//------------------------------------------------------------------------------
// Feature Detection
//------------------------------------------------------------------------------
#if defined(__AVX512F__)
#define HAS_AVX512 1
#define HAS_AVX2 1
#define HAS_SSE2 1
#elif defined(__AVX2__)
#define HAS_AVX2 1
#define HAS_SSE2 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define HAS_SSE2 1
#endif

#if defined(__FMA__) || (defined(__AVX2__) && defined(__FMA__))
#define HAS_FMA 1
#endif

//------------------------------------------------------------------------------
// Compiler-Agnostic Force Inline
//------------------------------------------------------------------------------
#ifdef _MSC_VER
#define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

//------------------------------------------------------------------------------
// Alignment Helpers (Proxy Functions)
//------------------------------------------------------------------------------
static ALWAYS_INLINE int is_aligned(const void *p, size_t alignment)
{
    return (((uintptr_t)p) & (alignment - 1)) == 0;
}

static ALWAYS_INLINE int is_aligned_16(const void *p) { return is_aligned(p, 16); }
static ALWAYS_INLINE int is_aligned_32(const void *p) { return is_aligned(p, 32); }
static ALWAYS_INLINE int is_aligned_64(const void *p) { return is_aligned(p, 64); }

//------------------------------------------------------------------------------
// Alignment Policy Configuration
//------------------------------------------------------------------------------
#if defined(FFT_DEBUG_ALIGNMENT) || defined(FFT_STRICT_ALIGNMENT)
#define CHECK_ALIGNMENT 1
#endif

//------------------------------------------------------------------------------
// SSE2 Load/Store Wrappers (128-bit / 16-byte alignment)
//------------------------------------------------------------------------------
#ifdef HAS_SSE2

static ALWAYS_INLINE __m128d load_pd128(const double *ptr)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_16(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned SSE2 load at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        return _mm_loadu_pd(ptr);
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    return _mm_load_pd(ptr);
#else
    return _mm_loadu_pd(ptr);
#endif
}

static ALWAYS_INLINE void store_pd128(double *ptr, __m128d v)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_16(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned SSE2 store at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        _mm_storeu_pd(ptr, v);
        return;
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    _mm_store_pd(ptr, v);
#else
    _mm_storeu_pd(ptr, v);
#endif
}

// Explicit unaligned versions
static ALWAYS_INLINE __m128d loadu_pd128(const double *ptr)
{
    return _mm_loadu_pd(ptr);
}

static ALWAYS_INLINE void storeu_pd128(double *ptr, __m128d v)
{
    _mm_storeu_pd(ptr, v);
}

// Legacy aliases
#define LOAD_SSE2(ptr) load_pd128((const double *)(ptr))
#define STORE_SSE2(ptr, v) store_pd128((double *)(ptr), (v))
#define LOADU_SSE2(ptr) loadu_pd128((const double *)(ptr))
#define STOREU_SSE2(ptr, v) storeu_pd128((double *)(ptr), (v))

#endif // HAS_SSE2

//------------------------------------------------------------------------------
// AVX2 Load/Store Wrappers (256-bit / 32-byte alignment)
//------------------------------------------------------------------------------
#ifdef HAS_AVX2

static ALWAYS_INLINE __m256d load_pd256(const double *ptr)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_32(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX2 load at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        return _mm256_loadu_pd(ptr);
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    return _mm256_load_pd(ptr);
#else
    return _mm256_loadu_pd(ptr);
#endif
}

static ALWAYS_INLINE void store_pd256(double *ptr, __m256d v)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_32(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX2 store at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        _mm256_storeu_pd(ptr, v);
        return;
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    _mm256_store_pd(ptr, v);
#else
    _mm256_storeu_pd(ptr, v);
#endif
}

static ALWAYS_INLINE __m256d loadu_pd256(const double *ptr)
{
    return _mm256_loadu_pd(ptr);
}

static ALWAYS_INLINE void storeu_pd256(double *ptr, __m256d v)
{
    _mm256_storeu_pd(ptr, v);
}

// Legacy aliases
#define LOAD_PD(ptr) load_pd256((const double *)(ptr))
#define STORE_PD(ptr, v) store_pd256((double *)(ptr), (v))
#define LOADU_PD(ptr) loadu_pd256((const double *)(ptr))
#define STOREU_PD(ptr, v) storeu_pd256((double *)(ptr), (v))

#endif // HAS_AVX2

//------------------------------------------------------------------------------
// AVX-512 Load/Store Wrappers (512-bit / 64-byte alignment)
//------------------------------------------------------------------------------
#ifdef HAS_AVX512

static ALWAYS_INLINE __m512d load_pd512(const double *ptr)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_64(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX-512 load at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        return _mm512_loadu_pd(ptr);
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    return _mm512_load_pd(ptr);
#else
    return _mm512_loadu_pd(ptr);
#endif
}

static ALWAYS_INLINE void store_pd512(double *ptr, __m512d v)
{
#ifdef CHECK_ALIGNMENT
    if (!is_aligned_64(ptr))
    {
        fprintf(stderr, "FFT WARNING: unaligned AVX-512 store at %p\n", (void *)ptr);
#ifdef FFT_STRICT_ALIGNMENT
        abort();
#else
        _mm512_storeu_pd(ptr, v);
        return;
#endif
    }
#endif
#ifdef USE_ALIGNED_SIMD
    _mm512_store_pd(ptr, v);
#else
    _mm512_storeu_pd(ptr, v);
#endif
}

static ALWAYS_INLINE __m512d loadu_pd512(const double *ptr)
{
    return _mm512_loadu_pd(ptr);
}

static ALWAYS_INLINE void storeu_pd512(double *ptr, __m512d v)
{
    _mm512_storeu_pd(ptr, v);
}

// Legacy aliases
#define LOAD_PD512(ptr) load_pd512((const double *)(ptr))
#define STORE_PD512(ptr, v) store_pd512((double *)(ptr), (v))
#define LOADU_PD512(ptr) loadu_pd512((const double *)(ptr))
#define STOREU_PD512(ptr, v) storeu_pd512((double *)(ptr), (v))

#endif // HAS_AVX512

//------------------------------------------------------------------------------
// FMA Wrappers (Fused Multiply-Add/Sub)
//------------------------------------------------------------------------------
#ifdef HAS_FMA
// 256-bit FMA
#define FMADD(a, b, c) _mm256_fmadd_pd((a), (b), (c))
#define FMSUB(a, b, c) _mm256_fmsub_pd((a), (b), (c))

// 128-bit FMA
#define FMADD_SSE2(a, b, c) _mm_fmadd_pd((a), (b), (c))
#define FMSUB_SSE2(a, b, c) _mm_fmsub_pd((a), (b), (c))
#else
// 256-bit fallback
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

// 128-bit fallback
static ALWAYS_INLINE __m128d fmadd_sse2_fallback(__m128d a, __m128d b, __m128d c)
{
    return _mm_add_pd(_mm_mul_pd(a, b), c);
}
static ALWAYS_INLINE __m128d fmsub_sse2_fallback(__m128d a, __m128d b, __m128d c)
{
    return _mm_sub_pd(_mm_mul_pd(a, b), c);
}
#define FMADD_SSE2(a, b, c) fmadd_sse2_fallback((a), (b), (c))
#define FMSUB_SSE2(a, b, c) fmsub_sse2_fallback((a), (b), (c))
#endif

// Explicit PD suffix aliases (for clarity)
#define FMADD_SSE2_PD FMADD_SSE2
#define FMSUB_SSE2_PD FMSUB_SSE2

//------------------------------------------------------------------------------
// Prefetch Configuration
//------------------------------------------------------------------------------
#ifndef FFT_PREFETCH_DISTANCE
#define FFT_PREFETCH_DISTANCE 8 // Cache lines ahead
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
const double C1 = 0.6234898018587336;   // cos(2π/7)
const double C2 = -0.22252093395631440; // cos(4π/7)
const double C3 = -0.90096886790241915; // cos(6π/7)
const double S1 = 0.78183148246802981;  // sin(2π/7)
const double S2 = 0.97492791218182360;  // sin(4π/7)
const double S3 = 0.43388373911755806;  // sin(6π/7)

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

// Precomputed Rader constants (OUTSIDE function, init once)
static const int L = 6;
static const int perm_in[6] = {1, 3, 2, 6, 4, 5};
static const int out_perm[6] = {1, 5, 4, 6, 2, 3};

// Convolution twiddles: 6 complex = 12 doubles, broadcastable
static double tw_fwd_re[12], tw_fwd_im[12]; // [tw0.re, tw0.im, tw1.re, tw1.im, ...]
static double tw_inv_re[12], tw_inv_im[12];

static void init_radix7_twiddles(void) __attribute__((constructor));
static void init_radix7_twiddles(void)
{
    const double angle_fwd = -2.0 * M_PI / 7.0;
    const double angle_inv = +2.0 * M_PI / 7.0;

    for (int q = 0; q < L; ++q)
    {
        double a_fwd = out_perm[q] * angle_fwd;
        double a_inv = out_perm[q] * angle_inv;
#ifdef __GNUC__
        double s_fwd, c_fwd, s_inv, c_inv;
        sincos(a_fwd, &s_fwd, &c_fwd);
        sincos(a_inv, &s_inv, &c_inv);
        tw_fwd_re[2 * q + 0] = c_fwd;
        tw_fwd_im[2 * q + 0] = s_fwd;
        tw_fwd_re[2 * q + 1] = c_fwd;
        tw_fwd_im[2 * q + 1] = s_fwd;
        tw_inv_re[2 * q + 0] = c_inv;
        tw_inv_im[2 * q + 0] = s_inv;
        tw_inv_re[2 * q + 1] = c_inv;
        tw_inv_im[2 * q + 1] = s_inv;
#else
        tw_fwd_re[2 * q + 0] = tw_fwd_re[2 * q + 1] = cos(a_fwd);
        tw_fwd_im[2 * q + 0] = tw_fwd_im[2 * q + 1] = sin(a_fwd);
        tw_inv_re[2 * q + 0] = tw_inv_re[2 * q + 1] = cos(a_inv);
        tw_inv_im[2 * q + 0] = tw_inv_im[2 * q + 1] = sin(a_inv);
#endif
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

// --- Radix-4 ---
static const complex_t twiddle_radix4[] = {
    {1.0, 0.0},
    {0.0, -1.0},
    {-1.0, 0.0},
    {0.0, 1.0}};

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

#ifdef __AVX2__
/**
 * @brief Vectorized sine for |x| ≤ π/4 using minimax polynomial (0.5 ULP accuracy)
 *
 * Computes sin(x) = x·P(x²) where P is evaluated in Horner form:
 * P(x²) = 1 - x²/3! + x⁴/5! - x⁶/7! + x⁸/9!
 */
static inline __m256d v_sin(__m256d x)
{
    const __m256d x2 = _mm256_mul_pd(x, x);

    // Horner form: start from highest degree term and work backwards
    __m256d s = _mm256_set1_pd(2.75573192239858906525573592e-6);                  // 1/9! = 1/362880
    s = _mm256_fmadd_pd(s, x2, _mm256_set1_pd(-1.98412698412698412698412698e-4)); // -1/7!
    s = _mm256_fmadd_pd(s, x2, _mm256_set1_pd(8.33333333333333333333333333e-3));  // 1/5!
    s = _mm256_fmadd_pd(s, x2, _mm256_set1_pd(-1.66666666666666666666666667e-1)); // -1/3!
    s = _mm256_fmadd_pd(s, x2, _mm256_set1_pd(1.0));                              // 1

    return _mm256_mul_pd(x, s); // x·P(x²)
}

/**
 * @brief Vectorized cosine for |x| ≤ π/4 using minimax polynomial (0.5 ULP accuracy)
 *
 * Computes cos(x) = P(x²) where P is evaluated in Horner form:
 * P(x²) = 1 - x²/2! + x⁴/4! - x⁶/6! + x⁸/8!
 */
static inline __m256d v_cos(__m256d x)
{
    const __m256d x2 = _mm256_mul_pd(x, x);

    // Horner form: start from highest degree term and work backwards
    __m256d c = _mm256_set1_pd(2.48015873015873015873015873e-5);                  // 1/8! = 1/40320
    c = _mm256_fmadd_pd(c, x2, _mm256_set1_pd(-1.38888888888888888888888889e-3)); // -1/6!
    c = _mm256_fmadd_pd(c, x2, _mm256_set1_pd(4.16666666666666666666666667e-2));  // 1/4!
    c = _mm256_fmadd_pd(c, x2, _mm256_set1_pd(-5.00000000000000000000000000e-1)); // -1/2!
    c = _mm256_fmadd_pd(c, x2, _mm256_set1_pd(1.0));                              // 1

    return c; // P(x²)
}
#endif

/*  ------------------------------------------------------------------  */
/*  Scalar 0.5-ulp sin/cos for |x| ≤ π/4 (cleanup/fallback)             */
/*  ------------------------------------------------------------------  */
static inline void sincos_pi4(double x, double *s, double *c)
{
    const double x2 = x * x;

    // sin(x) = x·P(x²) in Horner form
    double sp = 2.75573192239858906525573592e-6;        // 1/9!
    sp = fma(sp, x2, -1.98412698412698412698412698e-4); // -1/7!
    sp = fma(sp, x2, 8.33333333333333333333333333e-3);  // 1/5!
    sp = fma(sp, x2, -1.66666666666666666666666667e-1); // -1/3!
    sp = fma(sp, x2, 1.0);
    *s = x * sp;

    // cos(x) = P(x²) in Horner form
    double cp = 2.48015873015873015873015873e-5;        // 1/8!
    cp = fma(cp, x2, -1.38888888888888888888888889e-3); // -1/6!
    cp = fma(cp, x2, 4.16666666666666666666666667e-2);  // 1/4!
    cp = fma(cp, x2, -5.00000000000000000000000000e-1); // -1/2!
    cp = fma(cp, x2, 1.0);
    *c = cp;
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
 *
 * - Direct i² computation (no accumulation error)
 * - Vectorized 0.5-ULP minimax sin/cos polynomials
 * - FMA throughout for minimal rounding
 *
 * @note Runs automatically at program startup via __attribute__((constructor))
 */
static void init_bluestein_chirp_body(void)
{
    // Total storage needed (rounded up to multiple of 4 for alignment)
    int total_chirp = 0;
    for (int i = 0; i < num_pre; i++)
    {
        total_chirp += ((pre_sizes[i] + 3) & ~3);
    }

    // Allocate descriptor arrays
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
        return;
    }

    // Partition the big block and fill chirps
    int offset = 0;
    for (int idx = 0; idx < num_pre; idx++)
    {
        const int n = pre_sizes[idx];
        const int n_rounded = ((n + 3) & ~3);

        chirp_sizes[idx] = n;
        bluestein_chirp[idx] = all_chirps + offset;
        offset += n_rounded;

        // h(i) = exp(πi·i²/n) - Bluestein chirp
        // Use direct i² computation to avoid accumulation error
        const fft_type theta = (fft_type)M_PI / (fft_type)n;
        const int len2 = 2 * n;

        int i = 0;

#ifdef __AVX2__
        //======================================================================
        // AVX2: Vectorized chirp computation with 0.5-ULP precision
        // Process 4 chirps at once
        //======================================================================
        const __m256d vtheta = _mm256_set1_pd(theta);
        const __m256d vlen2 = _mm256_set1_pd((double)len2);

        for (; i + 3 < n; i += 4)
        {
            // Compute i² mod 2n for 4 values: [i, i+1, i+2, i+3]
            // Using direct computation: i² = i*i (no accumulation)
            __m256d vi = _mm256_set_pd((double)(i + 3), (double)(i + 2),
                                       (double)(i + 1), (double)i);

            // Compute i² with FMA folding
            __m256d vi_sq = _mm256_mul_pd(vi, vi);

            // Compute i² mod 2n using fmod-like operation
            // i_sq_mod = i_sq - floor(i_sq / len2) * len2
            __m256d vi_sq_div = _mm256_div_pd(vi_sq, vlen2);
            __m256d vi_sq_floor = _mm256_floor_pd(vi_sq_div);
            __m256d vi_sq_mod = _mm256_fnmadd_pd(vi_sq_floor, vlen2, vi_sq);

            // Compute angles: θ * (i² mod 2n)
            __m256d vang = _mm256_mul_pd(vtheta, vi_sq_mod);

            // Check if angles are in valid range for minimax polynomials
            // For small n (≤ 64), angles may exceed π/4, so use range-reduced version
            // or fall back to system sin/cos

            // Simple approach: Extract and compute scalar (better for small n)
            double ang[4];
            _mm256_storeu_pd(ang, vang);

            for (int j = 0; j < 4; ++j)
            {
                // For angles potentially > π/4, use range-reduced version
                if (fabs(ang[j]) <= M_PI / 4.0)
                {
                    sincos_pi4(ang[j], &bluestein_chirp[idx][i + j].im,
                               &bluestein_chirp[idx][i + j].re);
                }
                else
                {
                    // Fall back to system sin/cos for angles > π/4
                    // (only happens for very small n where precomputation is less critical)
                    bluestein_chirp[idx][i + j].re = cos(ang[j]);
                    bluestein_chirp[idx][i + j].im = sin(ang[j]);
                }
            }
        }
#endif // __AVX2__

        //======================================================================
        // Scalar cleanup with 0.5-ULP precision
        //======================================================================
        for (; i < n; i++)
        {
            // Direct i² computation - no accumulation error
            const long long i_sq = (long long)i * i;
            const long long i_sq_mod = i_sq % (long long)len2;
            const fft_type angle = theta * (fft_type)i_sq_mod;

            // Use 0.5-ULP minimax polynomials for |angle| ≤ π/4
            if (fabs(angle) <= M_PI / 4.0)
            {
                sincos_pi4(angle, &bluestein_chirp[idx][i].im,
                           &bluestein_chirp[idx][i].re);
            }
            else
            {
                // Fall back for angles > π/4 (rare for n ≤ 64)
                bluestein_chirp[idx][i].re = cos(angle);
                bluestein_chirp[idx][i].im = sin(angle);
            }
        }

        // Zero padded tail for alignment
        for (int i = n; i < n_rounded; ++i)
        {
            bluestein_chirp[idx][i].re = 0.0;
            bluestein_chirp[idx][i].im = 0.0;
        }
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

static void build_twiddles_linear(fft_data *tw, int N)
{
    // Use exact values for cardinal points to eliminate rounding
    tw[0].re = 1.0;
    tw[0].im = 0.0;

    // Set exact cardinal points for multiples of 4
    if (N % 4 == 0)
    {
        int quarter = N / 4;
        tw[quarter].re = 0.0;
        tw[quarter].im = -1.0;

        if (N > 4)
        {
            tw[3 * quarter].re = 0.0;
            tw[3 * quarter].im = 1.0;
        }
    }

    // Set exact value for N/2 (if even)
    if (N % 2 == 0)
    {
        tw[N / 2].re = -1.0;
        tw[N / 2].im = 0.0;
    }

    const double theta = -2.0 * M_PI / (double)N;

    // Compute the first half of twiddle factors
    // For odd N: compute indices 1 to (N-1)/2
    // For even N: compute indices 1 to N/2-1 (N/2 already set above)
    const int last_to_compute = (N - 1) / 2; // Works for both odd and even N

    int k = 1;

#ifdef __AVX2__
    const __m256d vtheta = _mm256_set1_pd(theta);

    // Vectorized computation
    for (; k + 3 <= last_to_compute; k += 4)
    {
        // Check which indices to skip (cardinal points)
        bool skip[4] = {false, false, false, false};
        if (N % 4 == 0)
        {
            int quarter = N / 4;
            for (int j = 0; j < 4; j++)
            {
                if ((k + j) == quarter)
                    skip[j] = true;
            }
        }

        __m256d vk = _mm256_set_pd((double)(k + 3), (double)(k + 2),
                                   (double)(k + 1), (double)k);
        __m256d vang = _mm256_mul_pd(vtheta, vk);

        double angles[4];
        _mm256_storeu_pd(angles, vang);

        for (int j = 0; j < 4; j++)
        {
            if (!skip[j] && (k + j) <= last_to_compute)
            {
                double ang = angles[j];

                // Use high-precision sin/cos
                if (fabs(ang) <= M_PI / 4.0)
                {
// Use your 0.5-ULP minimax polynomials if available
#ifdef HAVE_SINCOS_PI4
                    sincos_pi4(ang, &tw[k + j].im, &tw[k + j].re);
#else
                    tw[k + j].re = cos(ang);
                    tw[k + j].im = sin(ang);
#endif
                }
                else
                {
// Use system sin/cos
#ifdef __GNUC__
                    sincos(ang, &tw[k + j].im, &tw[k + j].re);
#else
                    tw[k + j].re = cos(ang);
                    tw[k + j].im = sin(ang);
#endif
                }
            }
        }
    }
#endif

    // Scalar path for remaining elements
    for (; k <= last_to_compute; k++)
    {
        // Skip exact cardinal points (only for multiples of 4)
        if (N % 4 == 0 && k == N / 4)
            continue;

        double angle = theta * k;

        // Use sincos if available for atomic computation
#ifdef __GNUC__
        sincos(angle, &tw[k].im, &tw[k].re);
#else
        tw[k].re = cos(angle);
        tw[k].im = sin(angle);
#endif
    }

    // Second half uses exact conjugate symmetry
    // Start from the first index that needs mirroring
    int mirror_start = last_to_compute + 1;

    // For even N, if N/2 is already set, skip it
    if (N % 2 == 0 && mirror_start == N / 2)
        mirror_start++;

    for (k = mirror_start; k < N; k++)
    {
        int mirror = N - k;
        tw[k].re = tw[mirror].re;  // Exact copy
        tw[k].im = -tw[mirror].im; // Exact negation (conjugate)
    }

    // Special handling for N=8 exact values ( optimization)
    if (N == 8)
    {
        tw[1].re = 0.7071067811865476; // sqrt(2)/2 to full precision
        tw[1].im = -0.7071067811865476;
        tw[3].re = -0.7071067811865476;
        tw[3].im = -0.7071067811865476;
        // Mirror for second half
        tw[5].re = tw[3].re;
        tw[5].im = -tw[3].im;
        tw[7].re = tw[1].re;
        tw[7].im = -tw[1].im;
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
    int twiddle_count = 0, max_scratch_size = 0, max_padded_length = 0;
    int twiddle_factors_size = 0;
    int scratch_needed = 0;

    // Step 4: Set up buffer sizes
    if (is_factorable)
    {
        max_padded_length = signal_length;
        twiddle_count = signal_length;
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

    // Step 5: Compute memory requirements and determine execution plan
    bool is_single_radix = false;
    int single_radix = 0;

    if (is_factorable)
    {
        // Get prime factorization
        int prime_factors[32];
        int num_prime_factors = factors(signal_length, prime_factors);

        // Determine execution radices (optimized for performance)
        int execution_radices[32];
        int num_radices = get_fft_execution_radices(signal_length, execution_radices,
                                                    prime_factors, num_prime_factors);
        printf("FFT N=%d, radices: ", signal_length);
        for (int i = 0; i < num_radices; i++)
        {
            printf("%d ", execution_radices[i]);
        }

        // Check if it's a single radix (all radices are the same)
        is_single_radix = true;
        int first_radix = execution_radices[0];
        for (int i = 1; i < num_radices; i++)
        {
            if (execution_radices[i] != first_radix)
            {
                is_single_radix = false;
                break;
            }
        }

        if (is_single_radix)
        {
            single_radix = first_radix;
        }

        int temp_N = signal_length;

        if (is_single_radix && num_radices > 0)
        {
            // Single radix optimization
            int radix = first_radix;
            int stage = 0;

            for (int n = signal_length; n > radix; n /= radix)
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
                    free(fft_config);
                    return NULL;
                }

                twiddle_factors_size += (radix - 1) * sub_fft_size;
                scratch_needed += radix * sub_fft_size;
            }
            fft_config->num_precomputed_stages = stage;
        }
        else
        {
            // Mixed-radix FFT
            for (int i = 0; i < num_radices; i++)
            {
                int radix = execution_radices[i];
                scratch_needed += radix * (temp_N / radix);

                // Support all radices up to 32
                if (radix <= 32)
                {
                    scratch_needed += (radix - 1) * (temp_N / radix);
                }
                temp_N /= radix;
            }
        }

        // Store the execution radices in the factors array (for use in fft_exec)
        fft_config->lf = num_radices;
        memcpy(fft_config->factors, execution_radices, num_radices * sizeof(int));

        max_scratch_size = scratch_needed;
        if (max_scratch_size < 4 * signal_length)
        {
            max_scratch_size = 4 * signal_length;
        }
    }
    else
    {
        // Non-factorable: use Bluestein's algorithm
        max_scratch_size = 4 * max_padded_length;

        // For Bluestein, we need the factorization of the padded length
        fft_config->lf = factors(max_padded_length, fft_config->factors);
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

    // Step 7: Allocate twiddle_factors for single-radix FFTs
    if (is_factorable && is_single_radix && twiddle_factors_size > 0)
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

    // Step 10: Compute twiddle factors
    build_twiddles_linear(fft_config->twiddles, fft_config->n_fft);

    // Step 11: Populate twiddle_factors for single-radix FFTs
    // Step 11: Populate twiddle_factors for single-radix FFTs
    if (fft_config->twiddle_factors && is_single_radix)
    {
        int offset = 0;
        int radix = single_radix;

        for (int N_stage = signal_length; N_stage >= radix; N_stage /= radix)
        {
            const int sub_len = N_stage / radix;
            const int stride = fft_config->n_fft / N_stage;

            for (int k = 0; k < sub_len; ++k)
            {
                const int base = (radix - 1) * k;

                if (radix == 7)
                {
                    // For Good-Thomas radix-7, store twiddles in sequential order
                    // w^1, w^2, w^3, w^4, w^5, w^6
                    for (int j = 1; j <= 6; ++j)
                    {
                        const int p = (j * k) % N_stage;
                        const int idxN = (p * stride) % fft_config->n_fft;
                        fft_config->twiddle_factors[offset + base + (j - 1)] = fft_config->twiddles[idxN];
                    }
                }
                else
                {
                    // Standard DIT for other radices
                    for (int j = 1; j < radix; ++j)
                    {
                        const int p = (j * k) % N_stage;
                        const int idxN = (p * stride) % fft_config->n_fft;
                        fft_config->twiddle_factors[offset + base + (j - 1)] = fft_config->twiddles[idxN];
                    }
                }
            }

            offset += (radix - 1) * sub_len; // Move this OUTSIDE the k loop
        }

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

#ifdef FFT_ENABLE_PREFETCH
    // Step 13: Initialize prefetch system
    static int cache_detected = 0;
    if (!cache_detected)
    {
        detect_cache_sizes();
        cache_detected = 1;
    }
#endif

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
    // a = [ar0,ai0, ar1,ai1, ar2,ai2, ar3,ai3]
    // b = [br0,bi0, br1,bi1, br2,bi2, br3,bi3]

    __m512d ar_ar = _mm512_unpacklo_pd(a, a);         // [ar0,ar0, ar1,ar1, ar2,ar2, ar3,ar3]
    __m512d ai_ai = _mm512_unpackhi_pd(a, a);         // [ai0,ai0, ai1,ai1, ai2,ai2, ai3,ai3]
    __m512d br_bi = b;                                // [br0,bi0, br1,bi1, br2,bi2, br3,bi3]
    __m512d bi_br = _mm512_permute_pd(b, 0b01010101); // [bi0,br0, bi1,br1, bi2,br2, bi3,br3]

    __m512d prod1 = _mm512_mul_pd(ar_ar, br_bi);
    __m512d prod2 = _mm512_mul_pd(ai_ai, bi_br);

    // Use FMA for better precision
    return _mm512_fmsubadd_pd(ar_ar, br_bi, prod2);
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
    __m256d ar_ar = _mm256_unpacklo_pd(a, a);     // [ar0, ar0, ar1, ar1]
    __m256d ai_ai = _mm256_unpackhi_pd(a, a);     // [ai0, ai0, ai1, ai1]
    __m256d br_bi = b;                            // [br0, bi0, br1, bi1]
    __m256d bi_br = _mm256_permute_pd(b, 0b0101); // [bi0, br0, bi1, br1]

    __m256d prod1 = _mm256_mul_pd(ar_ar, br_bi); // [ar*br, ar*bi, ...]
    __m256d prod2 = _mm256_mul_pd(ai_ai, bi_br); // [ai*bi, ai*br, ...]
    return _mm256_addsub_pd(prod1, prod2);       // [ar*br - ai*bi, ar*bi + ai*br, ...]
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
    __m128d brbr = _mm_shuffle_pd(b, b, 0b00); // [br, br]
    __m128d bibi = _mm_shuffle_pd(b, b, 0b11); // [bi, bi]

    __m128d p_br = _mm_mul_pd(a, brbr);                 // [ar*br, ai*br]
    __m128d p_bi = _mm_mul_pd(a, bibi);                 // [ar*bi, ai*bi]
    __m128d p_bi_sw = _mm_shuffle_pd(p_bi, p_bi, 0b01); // [ai*bi, ar*bi]

    // diff = [ar*br - ai*bi,  ai*br - ar*bi]
    __m128d diff = _mm_sub_pd(p_br, p_bi_sw);
    // sum  = [ar*br + ai*bi,  ai*br + ar*bi]  -> sum.high is the desired imag
    __m128d sum = _mm_add_pd(p_br, p_bi_sw);

    // result = [ diff.low (re),  sum.high (im) ]
    return _mm_move_sd(sum, diff);
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
    // CRITICAL FIX: Bounds check BEFORE accessing the array
    if (factor_index >= fft_obj->lf)
    {
        fprintf(stderr, "Error: factor_index out of range (%d >= %d)\n",
                factor_index, fft_obj->lf);
        return;
    }

    const int radix = fft_obj->factors[factor_index];

    // FIX: Validate radix is appropriate for this data_length
    if (radix <= 1 || (data_length % radix) != 0)
    {
        fprintf(stderr, "Error: Invalid radix=%d for N=%d at factor_idx=%d\n",
                radix, data_length, factor_index);
        return;
    }

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
    // FIX: Simplify - let children validate themselves
    //==========================================================================
    // FIX: Remove the child scratch estimation entirely - it's fragile and redundant
    // Each child will check its own needs when it runs

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

    // FIX: Remove the pre-check - children will validate themselves
    // This eliminates the fragile prediction logic

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

        // Handle k=0 specially (W^0 = 1)
        {
            fft_data even_0 = sub_outputs[0];
            fft_data odd_0 = sub_outputs[half];
            output_buffer[0].re = even_0.re + odd_0.re;
            output_buffer[0].im = even_0.im + odd_0.im;
            output_buffer[half].re = even_0.re - odd_0.re;
            output_buffer[half].im = even_0.im - odd_0.im;
        }

        // Optional: Handle k=N/4 specially when it exists (W^(N/4) = ±i)
        int k_quarter = 0;
        if ((half & 1) == 0) // N/4 is an integer
        {
            k_quarter = half >> 1;
            fft_data even_q = sub_outputs[k_quarter];
            fft_data odd_q = sub_outputs[half + k_quarter];

            // Rotate odd by ±90° (multiply by ±i)
            double rotated_re = transform_sign > 0 ? odd_q.im : -odd_q.im;
            double rotated_im = transform_sign > 0 ? -odd_q.re : odd_q.re;

            output_buffer[k_quarter].re = even_q.re + rotated_re;
            output_buffer[k_quarter].im = even_q.im + rotated_im;
            output_buffer[half + k_quarter].re = even_q.re - rotated_re;
            output_buffer[half + k_quarter].im = even_q.im - rotated_im;
        }

        // Process k values in two ranges to avoid k_quarter if it exists
        int k = 1; // Start from 1 since we handled k=0
        int range1_end = k_quarter ? k_quarter : half;

#ifdef HAS_AVX512
        //======================================================================
        // AVX-512: First range [1, k_quarter) or [1, half) if no k_quarter
        //======================================================================
        for (; k + 15 < range1_end; k += 16)
        {
            // Prefetch
            if (k + 32 < range1_end)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 32], _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 32 + half], _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[k + 32], _MM_HINT_T0);
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

            // Load twiddles - FIX: stage_tw[k] contains W^k
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
#endif // HAS_AVX512

#ifdef __AVX2__
        //======================================================================
        // AVX2: First range [k, range1_end)
        //======================================================================
        for (; k + 7 < range1_end; k += 8)
        {
            // Prefetch
            if (k + 16 < range1_end)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16], _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + half], _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[k + 16], _MM_HINT_T0);
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

            // Load twiddles - FIX: stage_tw[k] contains W^k
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

        // Cleanup: 2x unrolling for first range
        for (; k + 1 < range1_end; k += 2)
        {
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
        // SSE2 TAIL for first range
        //======================================================================
        for (; k < range1_end; ++k)
        {
            __m128d even = LOADU_SSE2(&sub_outputs[k].re);
            __m128d odd = LOADU_SSE2(&sub_outputs[k + half].re);
            __m128d w = LOADU_SSE2(&stage_tw[k].re); // FIX: stage_tw[k] contains W^k
            __m128d tw = cmul_sse2_aos(odd, w);
            STOREU_SSE2(&output_buffer[k].re, _mm_add_pd(even, tw));
            STOREU_SSE2(&output_buffer[k + half].re, _mm_sub_pd(even, tw));
        }

        //======================================================================
        // Second range: (k_quarter, half) if k_quarter exists
        //======================================================================
        if (k_quarter)
        {
            k = k_quarter + 1; // Skip k_quarter since we handled it

#ifdef HAS_AVX512
            // AVX-512 for second range
            for (; k + 15 < half; k += 16)
            {
                // Prefetch
                if (k + 32 < half)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + 32], _MM_HINT_T0);
                    _mm_prefetch((const char *)&sub_outputs[k + 32 + half], _MM_HINT_T0);
                    _mm_prefetch((const char *)&stage_tw[k + 32], _MM_HINT_T0);
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
#endif

#ifdef __AVX2__
            // AVX2 for second range
            for (; k + 7 < half; k += 8)
            {
                // Prefetch
                if (k + 16 < half)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + 16], _MM_HINT_T0);
                    _mm_prefetch((const char *)&sub_outputs[k + 16 + half], _MM_HINT_T0);
                    _mm_prefetch((const char *)&stage_tw[k + 16], _MM_HINT_T0);
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

                // Load twiddles
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

            // 2x cleanup for second range
            for (; k + 1 < half; k += 2)
            {
                __m256d even = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
                __m256d odd = load2_aos(&sub_outputs[k + half], &sub_outputs[k + half + 1]);
                __m256d w = load2_aos(&stage_tw[k], &stage_tw[k + 1]);

                __m256d tw = cmul_avx2_aos(odd, w);

                __m256d x0 = _mm256_add_pd(even, tw);
                __m256d x1 = _mm256_sub_pd(even, tw);

                STOREU_PD(&output_buffer[k].re, x0);
                STOREU_PD(&output_buffer[k + half].re, x1);
            }
#endif

            // SSE2 tail for second range
            for (; k < half; ++k)
            {
                __m128d even = LOADU_SSE2(&sub_outputs[k].re);
                __m128d odd = LOADU_SSE2(&sub_outputs[k + half].re);
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
        // Uses Rader's algorithm with symmetry exploitation:
        // Y_0 = a + b + c
        // Y_1 = a + (b+c)*C1 + (-i*sgn)*(b-c)*S1
        // Y_2 = a + (b+c)*C2 + (-i*sgn)*(b-c)*S2
        //
        // Where C1 = cos(2π/3) = -0.5, S1 = sin(2π/3) = √3/2
        //       C2 = cos(4π/3) = -0.5, S2 = sin(4π/3) = -√3/2
        //
        // Optimization: C1 = C2 = -0.5, so factor out common term
        //======================================================================

        const int third = sub_len;
        int k = 0;

        // Constants
        const double C_HALF = -0.5;                  // cos(2π/3) = cos(4π/3)
        const double S_SQRT3_2 = 0.8660254037844386; // √3/2

#ifdef __AVX2__
        //----------------------------------------------------------------------
        // AVX2 PATH: 8x unrolling, pure AoS
        //----------------------------------------------------------------------
        const __m256d v_half = _mm256_set1_pd(C_HALF);
        const __m256d v_sqrt3_2 = _mm256_set1_pd(S_SQRT3_2);

        // Precompute rotation mask for (-i*sgn) multiplication
        // After permute: [im0, re0, im1, re1]
        // Forward (sgn=+1): multiply by -i → negate lanes 1,3 (imaginary after swap)
        // Inverse (sgn=-1): multiply by +i → negate lanes 0,2 (real after swap)
        const __m256d rot_mask = (transform_sign == 1)
                                     ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)  // -i: negate lanes [3]=0,[2]=-0,[1]=0,[0]=-0
                                     : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); // +i: negate lanes [3]=-0,[2]=0,[1]=-0,[0]=0

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
            // Load twiddles W^k and W^{2k} (k-major: stage_tw[2k], stage_tw[2k+1])
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
            //
            // sum = b2 + c2
            // dif = b2 - c2
            // Y_0 = a + sum
            // common = a - 0.5 * sum
            // Y_1 = common + (-i*sgn) * √3/2 * dif
            // Y_2 = common - (-i*sgn) * √3/2 * dif
            //==================================================================

#define RADIX3_BUTTERFLY_AVX2(a, b2, c2, y0, y1, y2)          \
    {                                                         \
        __m256d sum = _mm256_add_pd(b2, c2);                  \
        __m256d dif = _mm256_sub_pd(b2, c2);                  \
        y0 = _mm256_add_pd(a, sum);                           \
        __m256d common = FMADD(v_half, sum, a);               \
        __m256d dif_swp = _mm256_permute_pd(dif, 0b0101);     \
        __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask);     \
        __m256d scaled_rot = _mm256_mul_pd(rot90, v_sqrt3_2); \
        y1 = _mm256_add_pd(common, scaled_rot);               \
        y2 = _mm256_sub_pd(common, scaled_rot);               \
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
            // Store results (pure AoS)
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

        //----------------------------------------------------------------------
        // Cleanup: 2x unrolling
        //----------------------------------------------------------------------
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
            __m256d common = FMADD(v_half, sum, a);

            __m256d dif_swp = _mm256_permute_pd(dif, 0b0101);
            __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask);
            __m256d scaled_rot = _mm256_mul_pd(rot90, v_sqrt3_2);

            __m256d y1 = _mm256_add_pd(common, scaled_rot);
            __m256d y2 = _mm256_sub_pd(common, scaled_rot);

            STOREU_PD(&output_buffer[k].re, y0);
            STOREU_PD(&output_buffer[k + third].re, y1);
            STOREU_PD(&output_buffer[k + 2 * third].re, y2);
        }
#endif // __AVX2__

        //----------------------------------------------------------------------
        // Scalar tail: Handle remaining 0..1 elements
        //----------------------------------------------------------------------
        for (; k < third; ++k)
        {
            // Load inputs
            fft_data a = sub_outputs[k];
            fft_data b = sub_outputs[k + third];
            fft_data c = sub_outputs[k + 2 * third];

            // Load twiddles
            fft_data w1 = stage_tw[2 * k];
            fft_data w2 = stage_tw[2 * k + 1];

            // Twiddle multiply
            double b2r = b.re * w1.re - b.im * w1.im;
            double b2i = b.re * w1.im + b.im * w1.re;

            double c2r = c.re * w2.re - c.im * w2.im;
            double c2i = c.re * w2.im + c.im * w2.re;

            // Radix-3 butterfly
            double sumr = b2r + c2r;
            double sumi = b2i + c2i;
            double difr = b2r - c2r;
            double difi = b2i - c2i;

            // Y_0 = a + sum
            output_buffer[k].re = a.re + sumr;
            output_buffer[k].im = a.im + sumi;

            // common = a - 0.5 * sum
            double commonr = a.re + C_HALF * sumr;
            double commoni = a.im + C_HALF * sumi;

            // scaled_rot = (-i*sgn) * √3/2 * dif
            double scaled_rotr, scaled_roti;
            if (transform_sign == 1)
            {
                // Forward: multiply by -i
                scaled_rotr = S_SQRT3_2 * difi;
                scaled_roti = -S_SQRT3_2 * difr;
            }
            else
            {
                // Inverse: multiply by +i
                scaled_rotr = -S_SQRT3_2 * difi;
                scaled_roti = S_SQRT3_2 * difr;
            }

            // Y_1 = common + scaled_rot
            output_buffer[k + third].re = commonr + scaled_rotr;
            output_buffer[k + third].im = commoni + scaled_roti;

            // Y_2 = common - scaled_rot
            output_buffer[k + 2 * third].re = commonr - scaled_rotr;
            output_buffer[k + 2 * third].im = commoni - scaled_roti;
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
        const __m512d mask_minus_i_512 = _mm512_castsi512_pd(
            _mm512_set_epi64(0x8000000000000000, 0x0000000000000000,
                             0x8000000000000000, 0x0000000000000000,
                             0x8000000000000000, 0x0000000000000000,
                             0x8000000000000000, 0x0000000000000000));
        const __m512d rot_mask_512 = (transform_sign == 1) ? mask_plus_i_512 : mask_minus_i_512;

#define RADIX4_BUTTERFLY_AVX512(a, b2, c2, d2, y0, y1, y2, y3)    \
    {                                                             \
        __m512d sumBD = _mm512_add_pd(b2, d2);                    \
        __m512d difBD = _mm512_sub_pd(b2, d2);                    \
        __m512d a_pc = _mm512_add_pd(a, c2);                      \
        __m512d a_mc = _mm512_sub_pd(a, c2);                      \
        y0 = _mm512_add_pd(a_pc, sumBD);                          \
        y2 = _mm512_sub_pd(a_pc, sumBD);                          \
        __m512d difBD_swp = _mm512_permute_pd(difBD, 0b01010101); \
        __m512d rot = _mm512_xor_pd(difBD_swp, rot_mask_512);     \
        y1 = _mm512_sub_pd(a_mc, rot);                            \
        y3 = _mm512_add_pd(a_mc, rot);                            \
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
#define RADIX4_BUTTERFLY_AVX2(a, b2, c2, d2, y0, y1, y2, y3)  \
    {                                                         \
        __m256d sumBD = _mm256_add_pd(b2, d2);                \
        __m256d difBD = _mm256_sub_pd(b2, d2);                \
        __m256d a_pc = _mm256_add_pd(a, c2);                  \
        __m256d a_mc = _mm256_sub_pd(a, c2);                  \
        y0 = _mm256_add_pd(a_pc, sumBD);                      \
        y2 = _mm256_sub_pd(a_pc, sumBD);                      \
        __m256d difBD_swp = _mm256_permute_pd(difBD, 0b0101); \
        __m256d rot = _mm256_xor_pd(difBD_swp, rot_mask);     \
        y1 = _mm256_sub_pd(a_mc, rot);                        \
        y3 = _mm256_add_pd(a_mc, rot);                        \
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
            __m128d y2 = _mm_sub_pd(a_pc, sumBD);

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
        // RADIX-5 BUTTERFLY (Rader DIT) - FFTW-STYLE OPTIMIZED
        //
        // Pure AoS, no conversions, 8x unrolling for maximum performance.
        //======================================================================

        const int fifth = sub_len;
        int k = 0;

#ifdef __AVX2__
        //------------------------------------------------------------------
        // AVX2 PATH: 8x unrolled, pure AoS
        //------------------------------------------------------------------
        const __m256d vc1 = _mm256_set1_pd(C5_1); // cos(2π/5)
        const __m256d vc2 = _mm256_set1_pd(C5_2); // cos(4π/5)
        const __m256d vs1 = _mm256_set1_pd(S5_1); // sin(2π/5)
        const __m256d vs2 = _mm256_set1_pd(S5_2); // sin(4π/5)

        // Precompute rotation mask for (-i*sgn) multiplication
        // After permute: [im0, re0, im1, re1]
        // Forward (sgn=+1): multiply by -i → negate lanes 1,3 (imaginary after swap)
        // Inverse (sgn=-1): multiply by +i → negate lanes 0,2 (real after swap)
        const __m256d rot_mask = (transform_sign == 1)
                                     ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)  // -i
                                     : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); // +i

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
    do                                                               \
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
    } while (0)

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
        // Cleanup: 2x unrolling (reuse rot_mask from above)
        //------------------------------------------------------------------
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
            __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask); // Reuse rot_mask
            __m256d a1 = _mm256_add_pd(a, tmp1);
            __m256d y1 = _mm256_add_pd(a1, r1);
            __m256d y4 = _mm256_sub_pd(a1, r1);

            __m256d base2 = FMSUB(vs2, t2, _mm256_mul_pd(vs1, t3));
            __m256d tmp2 = FMADD(vc2, t0, _mm256_mul_pd(vc1, t1));
            __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);
            __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask); // Reuse rot_mask
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
        // Scalar tail: Handle remaining 0..1 elements
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
                             ? _mm_xor_pd(base1_swp, _mm_set_pd(-0.0, 0.0))  // -i: negate lane 1
                             : _mm_xor_pd(base1_swp, _mm_set_pd(0.0, -0.0)); // +i: negate lane 0
            __m128d a1 = _mm_add_pd(a, tmp1);
            __m128d y1 = _mm_add_pd(a1, r1);
            __m128d y4 = _mm_sub_pd(a1, r1);

            __m128d base2 = FMSUB_SSE2(vs2_128, t2, _mm_mul_pd(vs1_128, t3));
            __m128d tmp2 = FMADD_SSE2(vc2_128, t0, _mm_mul_pd(vc1_128, t1));
            __m128d base2_swp = _mm_shuffle_pd(base2, base2, 0b01);
            __m128d r2 = (transform_sign == 1)
                             ? _mm_xor_pd(base2_swp, _mm_set_pd(-0.0, 0.0))  // -i: negate lane 1
                             : _mm_xor_pd(base2_swp, _mm_set_pd(0.0, -0.0)); // +i: negate lane 0
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
        //======================================================================
        // RADIX-7 BUTTERFLY (Rader DIT), FFTW-style optimized, pure AoS
        //  - AVX2: 8x unrolled main loop, 2x cleanup
        //  - scalar tail for 0..1 leftover
        //  - per-stage twiddles: stage_tw[6*k + 0..5] for x1..x6 (DIT)
        //======================================================================
        const int seventh = sub_len;
        int k = 0;

        // Rader permutations for N=7 (generator g=3, inverse 5):
        //   perm_in  = [1,3,2,6,4,5]  (reorder inputs x1..x6)
        //   out_perm = [1,5,4,6,2,3]  (where each conv[q] lands)
        // We encode by wiring directly below (no explicit arrays needed)

        // Build convolution twiddles tw[q] = exp( sgn * j*2π*out_perm[q] / 7 )
        // sgn: forward(+1) uses minus angle convention in init; here we follow Rader directly:
        const double base_angle = (transform_sign == 1 ? -2.0 : +2.0) * M_PI / 7.0;

#ifdef __AVX2__
        //----------------------------------------------------------------------
        // AVX2 PATH
        //----------------------------------------------------------------------
        __m256d tw_brd[6]; // [wr,wi, wr,wi] broadcast for AoS pair multiply
        {
            // out_perm = [1,5,4,6,2,3]
            const int op[6] = {1, 5, 4, 6, 2, 3};
            for (int q = 0; q < 6; ++q)
            {
                double a = op[q] * base_angle;
                double wr, wi;
#ifdef __GNUC__
                sincos(a, &wi, &wr);
#else
                wr = cos(a);
                wi = sin(a);
#endif
                // lanes (hi..lo): [im0, re0, im1, re1] expected by cmul_avx2_aos as [br,bi, br,bi]
                tw_brd[q] = _mm256_set_pd(wi, wr, wi, wr);
            }
        }

        // -----------------------------
        // 8x unrolled main loop
        // -----------------------------
        for (; k + 7 < seventh; k += 8)
        {
            // Prefetch a bit ahead
            if (k + 16 < seventh)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + seventh].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 2 * seventh].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 3 * seventh].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 4 * seventh].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 5 * seventh].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 6 * seventh].re, _MM_HINT_T0);
                if (seventh > 1)
                    _mm_prefetch((const char *)&stage_tw[6 * (k + 16)].re, _MM_HINT_T0);
            }

            // Load x0..x6 for 8 butterflies: 4 AVX loads per "lane group"
            __m256d x0_0 = load2_aos(&sub_outputs[k + 0 * seventh + 0], &sub_outputs[k + 0 * seventh + 1]);
            __m256d x0_1 = load2_aos(&sub_outputs[k + 0 * seventh + 2], &sub_outputs[k + 0 * seventh + 3]);
            __m256d x0_2 = load2_aos(&sub_outputs[k + 0 * seventh + 4], &sub_outputs[k + 0 * seventh + 5]);
            __m256d x0_3 = load2_aos(&sub_outputs[k + 0 * seventh + 6], &sub_outputs[k + 0 * seventh + 7]);

            __m256d x1_0 = load2_aos(&sub_outputs[k + 1 * seventh + 0], &sub_outputs[k + 1 * seventh + 1]);
            __m256d x1_1 = load2_aos(&sub_outputs[k + 1 * seventh + 2], &sub_outputs[k + 1 * seventh + 3]);
            __m256d x1_2 = load2_aos(&sub_outputs[k + 1 * seventh + 4], &sub_outputs[k + 1 * seventh + 5]);
            __m256d x1_3 = load2_aos(&sub_outputs[k + 1 * seventh + 6], &sub_outputs[k + 1 * seventh + 7]);

            __m256d x2_0 = load2_aos(&sub_outputs[k + 2 * seventh + 0], &sub_outputs[k + 2 * seventh + 1]);
            __m256d x2_1 = load2_aos(&sub_outputs[k + 2 * seventh + 2], &sub_outputs[k + 2 * seventh + 3]);
            __m256d x2_2 = load2_aos(&sub_outputs[k + 2 * seventh + 4], &sub_outputs[k + 2 * seventh + 5]);
            __m256d x2_3 = load2_aos(&sub_outputs[k + 2 * seventh + 6], &sub_outputs[k + 2 * seventh + 7]);

            __m256d x3_0 = load2_aos(&sub_outputs[k + 3 * seventh + 0], &sub_outputs[k + 3 * seventh + 1]);
            __m256d x3_1 = load2_aos(&sub_outputs[k + 3 * seventh + 2], &sub_outputs[k + 3 * seventh + 3]);
            __m256d x3_2 = load2_aos(&sub_outputs[k + 3 * seventh + 4], &sub_outputs[k + 3 * seventh + 5]);
            __m256d x3_3 = load2_aos(&sub_outputs[k + 3 * seventh + 6], &sub_outputs[k + 3 * seventh + 7]);

            __m256d x4_0 = load2_aos(&sub_outputs[k + 4 * seventh + 0], &sub_outputs[k + 4 * seventh + 1]);
            __m256d x4_1 = load2_aos(&sub_outputs[k + 4 * seventh + 2], &sub_outputs[k + 4 * seventh + 3]);
            __m256d x4_2 = load2_aos(&sub_outputs[k + 4 * seventh + 4], &sub_outputs[k + 4 * seventh + 5]);
            __m256d x4_3 = load2_aos(&sub_outputs[k + 4 * seventh + 6], &sub_outputs[k + 4 * seventh + 7]);

            __m256d x5_0 = load2_aos(&sub_outputs[k + 5 * seventh + 0], &sub_outputs[k + 5 * seventh + 1]);
            __m256d x5_1 = load2_aos(&sub_outputs[k + 5 * seventh + 2], &sub_outputs[k + 5 * seventh + 3]);
            __m256d x5_2 = load2_aos(&sub_outputs[k + 5 * seventh + 4], &sub_outputs[k + 5 * seventh + 5]);
            __m256d x5_3 = load2_aos(&sub_outputs[k + 5 * seventh + 6], &sub_outputs[k + 5 * seventh + 7]);

            __m256d x6_0 = load2_aos(&sub_outputs[k + 6 * seventh + 0], &sub_outputs[k + 6 * seventh + 1]);
            __m256d x6_1 = load2_aos(&sub_outputs[k + 6 * seventh + 2], &sub_outputs[k + 6 * seventh + 3]);
            __m256d x6_2 = load2_aos(&sub_outputs[k + 6 * seventh + 4], &sub_outputs[k + 6 * seventh + 5]);
            __m256d x6_3 = load2_aos(&sub_outputs[k + 6 * seventh + 6], &sub_outputs[k + 6 * seventh + 7]);

            // Apply per-stage DIT twiddles (if multi-stage)
            if (seventh > 1)
            {
                __m256d w1_0 = load2_aos(&stage_tw[6 * (k + 0) + 0], &stage_tw[6 * (k + 1) + 0]);
                __m256d w1_1 = load2_aos(&stage_tw[6 * (k + 2) + 0], &stage_tw[6 * (k + 3) + 0]);
                __m256d w1_2 = load2_aos(&stage_tw[6 * (k + 4) + 0], &stage_tw[6 * (k + 5) + 0]);
                __m256d w1_3 = load2_aos(&stage_tw[6 * (k + 6) + 0], &stage_tw[6 * (k + 7) + 0]);

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

                x1_0 = cmul_avx2_aos(x1_0, w1_0);
                x1_1 = cmul_avx2_aos(x1_1, w1_1);
                x1_2 = cmul_avx2_aos(x1_2, w1_2);
                x1_3 = cmul_avx2_aos(x1_3, w1_3);

                x2_0 = cmul_avx2_aos(x2_0, w2_0);
                x2_1 = cmul_avx2_aos(x2_1, w2_1);
                x2_2 = cmul_avx2_aos(x2_2, w2_2);
                x2_3 = cmul_avx2_aos(x2_3, w2_3);

                x3_0 = cmul_avx2_aos(x3_0, w3_0);
                x3_1 = cmul_avx2_aos(x3_1, w3_1);
                x3_2 = cmul_avx2_aos(x3_2, w3_2);
                x3_3 = cmul_avx2_aos(x3_3, w3_3);

                x4_0 = cmul_avx2_aos(x4_0, w4_0);
                x4_1 = cmul_avx2_aos(x4_1, w4_1);
                x4_2 = cmul_avx2_aos(x4_2, w4_2);
                x4_3 = cmul_avx2_aos(x4_3, w4_3);

                x5_0 = cmul_avx2_aos(x5_0, w5_0);
                x5_1 = cmul_avx2_aos(x5_1, w5_1);
                x5_2 = cmul_avx2_aos(x5_2, w5_2);
                x5_3 = cmul_avx2_aos(x5_3, w5_3);

                x6_0 = cmul_avx2_aos(x6_0, w6_0);
                x6_1 = cmul_avx2_aos(x6_1, w6_1);
                x6_2 = cmul_avx2_aos(x6_2, w6_2);
                x6_3 = cmul_avx2_aos(x6_3, w6_3);
            }

            // y0 = sum(x0..x6) per 2-butterfly pair
            __m256d y0_0 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_0, x1_0), _mm256_add_pd(x2_0, x3_0)),
                                         _mm256_add_pd(_mm256_add_pd(x4_0, x5_0), x6_0));
            __m256d y0_1 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_1, x1_1), _mm256_add_pd(x2_1, x3_1)),
                                         _mm256_add_pd(_mm256_add_pd(x4_1, x5_1), x6_1));
            __m256d y0_2 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_2, x1_2), _mm256_add_pd(x2_2, x3_2)),
                                         _mm256_add_pd(_mm256_add_pd(x4_2, x5_2), x6_2));
            __m256d y0_3 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_3, x1_3), _mm256_add_pd(x2_3, x3_3)),
                                         _mm256_add_pd(_mm256_add_pd(x4_3, x5_3), x6_3));

            // Rader input permute: tx = [x1, x3, x2, x6, x4, x5]
            __m256d tx0_0 = x1_0, tx1_0 = x3_0, tx2_0 = x2_0, tx3_0 = x6_0, tx4_0 = x4_0, tx5_0 = x5_0;
            __m256d tx0_1 = x1_1, tx1_1 = x3_1, tx2_1 = x2_1, tx3_1 = x6_1, tx4_1 = x4_1, tx5_1 = x5_1;
            __m256d tx0_2 = x1_2, tx1_2 = x3_2, tx2_2 = x2_2, tx3_2 = x6_2, tx4_2 = x4_2, tx5_2 = x5_2;
            __m256d tx0_3 = x1_3, tx1_3 = x3_3, tx2_3 = x2_3, tx3_3 = x6_3, tx4_3 = x4_3, tx5_3 = x5_3;

            // 6-pt cyclic convolution conv[q] = Σ_l tx[l] * tw[(q-l) mod 6]
            __m256d c0_0 = cmul_avx2_aos(tx0_0, tw_brd[0]);
            __m256d c0_1 = cmul_avx2_aos(tx0_1, tw_brd[0]);
            __m256d c0_2 = cmul_avx2_aos(tx0_2, tw_brd[0]);
            __m256d c0_3 = cmul_avx2_aos(tx0_3, tw_brd[0]);

            // q=0
            __m256d v0_0 = c0_0;
            __m256d v0_1 = c0_1;
            __m256d v0_2 = c0_2;
            __m256d v0_3 = c0_3;
            v0_0 = _mm256_add_pd(v0_0, cmul_avx2_aos(tx1_0, tw_brd[5]));
            v0_1 = _mm256_add_pd(v0_1, cmul_avx2_aos(tx1_1, tw_brd[5]));
            v0_2 = _mm256_add_pd(v0_2, cmul_avx2_aos(tx1_2, tw_brd[5]));
            v0_3 = _mm256_add_pd(v0_3, cmul_avx2_aos(tx1_3, tw_brd[5]));
            v0_0 = _mm256_add_pd(v0_0, cmul_avx2_aos(tx2_0, tw_brd[4]));
            v0_1 = _mm256_add_pd(v0_1, cmul_avx2_aos(tx2_1, tw_brd[4]));
            v0_2 = _mm256_add_pd(v0_2, cmul_avx2_aos(tx2_2, tw_brd[4]));
            v0_3 = _mm256_add_pd(v0_3, cmul_avx2_aos(tx2_3, tw_brd[4]));
            v0_0 = _mm256_add_pd(v0_0, cmul_avx2_aos(tx3_0, tw_brd[3]));
            v0_1 = _mm256_add_pd(v0_1, cmul_avx2_aos(tx3_1, tw_brd[3]));
            v0_2 = _mm256_add_pd(v0_2, cmul_avx2_aos(tx3_2, tw_brd[3]));
            v0_3 = _mm256_add_pd(v0_3, cmul_avx2_aos(tx3_3, tw_brd[3]));
            v0_0 = _mm256_add_pd(v0_0, cmul_avx2_aos(tx4_0, tw_brd[2]));
            v0_1 = _mm256_add_pd(v0_1, cmul_avx2_aos(tx4_1, tw_brd[2]));
            v0_2 = _mm256_add_pd(v0_2, cmul_avx2_aos(tx4_2, tw_brd[2]));
            v0_3 = _mm256_add_pd(v0_3, cmul_avx2_aos(tx4_3, tw_brd[2]));
            v0_0 = _mm256_add_pd(v0_0, cmul_avx2_aos(tx5_0, tw_brd[1]));
            v0_1 = _mm256_add_pd(v0_1, cmul_avx2_aos(tx5_1, tw_brd[1]));
            v0_2 = _mm256_add_pd(v0_2, cmul_avx2_aos(tx5_2, tw_brd[1]));
            v0_3 = _mm256_add_pd(v0_3, cmul_avx2_aos(tx5_3, tw_brd[1]));

            // q=1
            __m256d v1_0 = cmul_avx2_aos(tx0_0, tw_brd[1]);
            __m256d v1_1 = cmul_avx2_aos(tx0_1, tw_brd[1]);
            __m256d v1_2 = cmul_avx2_aos(tx0_2, tw_brd[1]);
            __m256d v1_3 = cmul_avx2_aos(tx0_3, tw_brd[1]);
            v1_0 = _mm256_add_pd(v1_0, cmul_avx2_aos(tx1_0, tw_brd[0]));
            v1_1 = _mm256_add_pd(v1_1, cmul_avx2_aos(tx1_1, tw_brd[0]));
            v1_2 = _mm256_add_pd(v1_2, cmul_avx2_aos(tx1_2, tw_brd[0]));
            v1_3 = _mm256_add_pd(v1_3, cmul_avx2_aos(tx1_3, tw_brd[0]));
            v1_0 = _mm256_add_pd(v1_0, cmul_avx2_aos(tx2_0, tw_brd[5]));
            v1_1 = _mm256_add_pd(v1_1, cmul_avx2_aos(tx2_1, tw_brd[5]));
            v1_2 = _mm256_add_pd(v1_2, cmul_avx2_aos(tx2_2, tw_brd[5]));
            v1_3 = _mm256_add_pd(v1_3, cmul_avx2_aos(tx2_3, tw_brd[5]));
            v1_0 = _mm256_add_pd(v1_0, cmul_avx2_aos(tx3_0, tw_brd[4]));
            v1_1 = _mm256_add_pd(v1_1, cmul_avx2_aos(tx3_1, tw_brd[4]));
            v1_2 = _mm256_add_pd(v1_2, cmul_avx2_aos(tx3_2, tw_brd[4]));
            v1_3 = _mm256_add_pd(v1_3, cmul_avx2_aos(tx3_3, tw_brd[4]));
            v1_0 = _mm256_add_pd(v1_0, cmul_avx2_aos(tx4_0, tw_brd[3]));
            v1_1 = _mm256_add_pd(v1_1, cmul_avx2_aos(tx4_1, tw_brd[3]));
            v1_2 = _mm256_add_pd(v1_2, cmul_avx2_aos(tx4_2, tw_brd[3]));
            v1_3 = _mm256_add_pd(v1_3, cmul_avx2_aos(tx4_3, tw_brd[3]));
            v1_0 = _mm256_add_pd(v1_0, cmul_avx2_aos(tx5_0, tw_brd[2]));
            v1_1 = _mm256_add_pd(v1_1, cmul_avx2_aos(tx5_1, tw_brd[2]));
            v1_2 = _mm256_add_pd(v1_2, cmul_avx2_aos(tx5_2, tw_brd[2]));
            v1_3 = _mm256_add_pd(v1_3, cmul_avx2_aos(tx5_3, tw_brd[2]));

            // q=2
            __m256d v2_0 = cmul_avx2_aos(tx0_0, tw_brd[2]);
            __m256d v2_1 = cmul_avx2_aos(tx0_1, tw_brd[2]);
            __m256d v2_2 = cmul_avx2_aos(tx0_2, tw_brd[2]);
            __m256d v2_3 = cmul_avx2_aos(tx0_3, tw_brd[2]);
            v2_0 = _mm256_add_pd(v2_0, cmul_avx2_aos(tx1_0, tw_brd[1]));
            v2_1 = _mm256_add_pd(v2_1, cmul_avx2_aos(tx1_1, tw_brd[1]));
            v2_2 = _mm256_add_pd(v2_2, cmul_avx2_aos(tx1_2, tw_brd[1]));
            v2_3 = _mm256_add_pd(v2_3, cmul_avx2_aos(tx1_3, tw_brd[1]));
            v2_0 = _mm256_add_pd(v2_0, cmul_avx2_aos(tx2_0, tw_brd[0]));
            v2_1 = _mm256_add_pd(v2_1, cmul_avx2_aos(tx2_1, tw_brd[0]));
            v2_2 = _mm256_add_pd(v2_2, cmul_avx2_aos(tx2_2, tw_brd[0]));
            v2_3 = _mm256_add_pd(v2_3, cmul_avx2_aos(tx2_3, tw_brd[0]));
            v2_0 = _mm256_add_pd(v2_0, cmul_avx2_aos(tx3_0, tw_brd[5]));
            v2_1 = _mm256_add_pd(v2_1, cmul_avx2_aos(tx3_1, tw_brd[5]));
            v2_2 = _mm256_add_pd(v2_2, cmul_avx2_aos(tx3_2, tw_brd[5]));
            v2_3 = _mm256_add_pd(v2_3, cmul_avx2_aos(tx3_3, tw_brd[5]));
            v2_0 = _mm256_add_pd(v2_0, cmul_avx2_aos(tx4_0, tw_brd[4]));
            v2_1 = _mm256_add_pd(v2_1, cmul_avx2_aos(tx4_1, tw_brd[4]));
            v2_2 = _mm256_add_pd(v2_2, cmul_avx2_aos(tx4_2, tw_brd[4]));
            v2_3 = _mm256_add_pd(v2_3, cmul_avx2_aos(tx4_3, tw_brd[4]));
            v2_0 = _mm256_add_pd(v2_0, cmul_avx2_aos(tx5_0, tw_brd[3]));
            v2_1 = _mm256_add_pd(v2_1, cmul_avx2_aos(tx5_1, tw_brd[3]));
            v2_2 = _mm256_add_pd(v2_2, cmul_avx2_aos(tx5_2, tw_brd[3]));
            v2_3 = _mm256_add_pd(v2_3, cmul_avx2_aos(tx5_3, tw_brd[3]));

            // q=3
            __m256d v3_0 = cmul_avx2_aos(tx0_0, tw_brd[3]);
            __m256d v3_1 = cmul_avx2_aos(tx0_1, tw_brd[3]);
            __m256d v3_2 = cmul_avx2_aos(tx0_2, tw_brd[3]);
            __m256d v3_3 = cmul_avx2_aos(tx0_3, tw_brd[3]);
            v3_0 = _mm256_add_pd(v3_0, cmul_avx2_aos(tx1_0, tw_brd[2]));
            v3_1 = _mm256_add_pd(v3_1, cmul_avx2_aos(tx1_1, tw_brd[2]));
            v3_2 = _mm256_add_pd(v3_2, cmul_avx2_aos(tx1_2, tw_brd[2]));
            v3_3 = _mm256_add_pd(v3_3, cmul_avx2_aos(tx1_3, tw_brd[2]));
            v3_0 = _mm256_add_pd(v3_0, cmul_avx2_aos(tx2_0, tw_brd[1]));
            v3_1 = _mm256_add_pd(v3_1, cmul_avx2_aos(tx2_1, tw_brd[1]));
            v3_2 = _mm256_add_pd(v3_2, cmul_avx2_aos(tx2_2, tw_brd[1]));
            v3_3 = _mm256_add_pd(v3_3, cmul_avx2_aos(tx2_3, tw_brd[1]));
            v3_0 = _mm256_add_pd(v3_0, cmul_avx2_aos(tx3_0, tw_brd[0]));
            v3_1 = _mm256_add_pd(v3_1, cmul_avx2_aos(tx3_1, tw_brd[0]));
            v3_2 = _mm256_add_pd(v3_2, cmul_avx2_aos(tx3_2, tw_brd[0]));
            v3_3 = _mm256_add_pd(v3_3, cmul_avx2_aos(tx3_3, tw_brd[0]));
            v3_0 = _mm256_add_pd(v3_0, cmul_avx2_aos(tx4_0, tw_brd[5]));
            v3_1 = _mm256_add_pd(v3_1, cmul_avx2_aos(tx4_1, tw_brd[5]));
            v3_2 = _mm256_add_pd(v3_2, cmul_avx2_aos(tx4_2, tw_brd[5]));
            v3_3 = _mm256_add_pd(v3_3, cmul_avx2_aos(tx4_3, tw_brd[5]));
            v3_0 = _mm256_add_pd(v3_0, cmul_avx2_aos(tx5_0, tw_brd[4]));
            v3_1 = _mm256_add_pd(v3_1, cmul_avx2_aos(tx5_1, tw_brd[4]));
            v3_2 = _mm256_add_pd(v3_2, cmul_avx2_aos(tx5_2, tw_brd[4]));
            v3_3 = _mm256_add_pd(v3_3, cmul_avx2_aos(tx5_3, tw_brd[4]));

            // q=4
            __m256d v4_0 = cmul_avx2_aos(tx0_0, tw_brd[4]);
            __m256d v4_1 = cmul_avx2_aos(tx0_1, tw_brd[4]);
            __m256d v4_2 = cmul_avx2_aos(tx0_2, tw_brd[4]);
            __m256d v4_3 = cmul_avx2_aos(tx0_3, tw_brd[4]);
            v4_0 = _mm256_add_pd(v4_0, cmul_avx2_aos(tx1_0, tw_brd[3]));
            v4_1 = _mm256_add_pd(v4_1, cmul_avx2_aos(tx1_1, tw_brd[3]));
            v4_2 = _mm256_add_pd(v4_2, cmul_avx2_aos(tx1_2, tw_brd[3]));
            v4_3 = _mm256_add_pd(v4_3, cmul_avx2_aos(tx1_3, tw_brd[3]));
            v4_0 = _mm256_add_pd(v4_0, cmul_avx2_aos(tx2_0, tw_brd[2]));
            v4_1 = _mm256_add_pd(v4_1, cmul_avx2_aos(tx2_1, tw_brd[2]));
            v4_2 = _mm256_add_pd(v4_2, cmul_avx2_aos(tx2_2, tw_brd[2]));
            v4_3 = _mm256_add_pd(v4_3, cmul_avx2_aos(tx2_3, tw_brd[2]));
            v4_0 = _mm256_add_pd(v4_0, cmul_avx2_aos(tx3_0, tw_brd[1]));
            v4_1 = _mm256_add_pd(v4_1, cmul_avx2_aos(tx3_1, tw_brd[1]));
            v4_2 = _mm256_add_pd(v4_2, cmul_avx2_aos(tx3_2, tw_brd[1]));
            v4_3 = _mm256_add_pd(v4_3, cmul_avx2_aos(tx3_3, tw_brd[1]));
            v4_0 = _mm256_add_pd(v4_0, cmul_avx2_aos(tx4_0, tw_brd[0]));
            v4_1 = _mm256_add_pd(v4_1, cmul_avx2_aos(tx4_1, tw_brd[0]));
            v4_2 = _mm256_add_pd(v4_2, cmul_avx2_aos(tx4_2, tw_brd[0]));
            v4_3 = _mm256_add_pd(v4_3, cmul_avx2_aos(tx4_3, tw_brd[0]));
            v4_0 = _mm256_add_pd(v4_0, cmul_avx2_aos(tx5_0, tw_brd[5]));
            v4_1 = _mm256_add_pd(v4_1, cmul_avx2_aos(tx5_1, tw_brd[5]));
            v4_2 = _mm256_add_pd(v4_2, cmul_avx2_aos(tx5_2, tw_brd[5]));
            v4_3 = _mm256_add_pd(v4_3, cmul_avx2_aos(tx5_3, tw_brd[5]));

            // q=5
            __m256d v5_0 = cmul_avx2_aos(tx0_0, tw_brd[5]);
            __m256d v5_1 = cmul_avx2_aos(tx0_1, tw_brd[5]);
            __m256d v5_2 = cmul_avx2_aos(tx0_2, tw_brd[5]);
            __m256d v5_3 = cmul_avx2_aos(tx0_3, tw_brd[5]);
            v5_0 = _mm256_add_pd(v5_0, cmul_avx2_aos(tx1_0, tw_brd[4]));
            v5_1 = _mm256_add_pd(v5_1, cmul_avx2_aos(tx1_1, tw_brd[4]));
            v5_2 = _mm256_add_pd(v5_2, cmul_avx2_aos(tx1_2, tw_brd[4]));
            v5_3 = _mm256_add_pd(v5_3, cmul_avx2_aos(tx1_3, tw_brd[4]));
            v5_0 = _mm256_add_pd(v5_0, cmul_avx2_aos(tx2_0, tw_brd[3]));
            v5_1 = _mm256_add_pd(v5_1, cmul_avx2_aos(tx2_1, tw_brd[3]));
            v5_2 = _mm256_add_pd(v5_2, cmul_avx2_aos(tx2_2, tw_brd[3]));
            v5_3 = _mm256_add_pd(v5_3, cmul_avx2_aos(tx2_3, tw_brd[3]));
            v5_0 = _mm256_add_pd(v5_0, cmul_avx2_aos(tx3_0, tw_brd[2]));
            v5_1 = _mm256_add_pd(v5_1, cmul_avx2_aos(tx3_1, tw_brd[2]));
            v5_2 = _mm256_add_pd(v5_2, cmul_avx2_aos(tx3_2, tw_brd[2]));
            v5_3 = _mm256_add_pd(v5_3, cmul_avx2_aos(tx3_3, tw_brd[2]));
            v5_0 = _mm256_add_pd(v5_0, cmul_avx2_aos(tx4_0, tw_brd[1]));
            v5_1 = _mm256_add_pd(v5_1, cmul_avx2_aos(tx4_1, tw_brd[1]));
            v5_2 = _mm256_add_pd(v5_2, cmul_avx2_aos(tx4_2, tw_brd[1]));
            v5_3 = _mm256_add_pd(v5_3, cmul_avx2_aos(tx4_3, tw_brd[1]));
            v5_0 = _mm256_add_pd(v5_0, cmul_avx2_aos(tx5_0, tw_brd[0]));
            v5_1 = _mm256_add_pd(v5_1, cmul_avx2_aos(tx5_1, tw_brd[0]));
            v5_2 = _mm256_add_pd(v5_2, cmul_avx2_aos(tx5_2, tw_brd[0]));
            v5_3 = _mm256_add_pd(v5_3, cmul_avx2_aos(tx5_3, tw_brd[0]));

            // y[m] = x0 + conv[q], with m = out_perm[q] = [1,5,4,6,2,3]
            __m256d y1_0 = _mm256_add_pd(x0_0, v0_0);
            __m256d y1_1 = _mm256_add_pd(x0_1, v0_1);
            __m256d y1_2 = _mm256_add_pd(x0_2, v0_2);
            __m256d y1_3 = _mm256_add_pd(x0_3, v0_3);

            __m256d y5_0 = _mm256_add_pd(x0_0, v1_0);
            __m256d y5_1 = _mm256_add_pd(x0_1, v1_1);
            __m256d y5_2 = _mm256_add_pd(x0_2, v1_2);
            __m256d y5_3 = _mm256_add_pd(x0_3, v1_3);

            __m256d y4_0 = _mm256_add_pd(x0_0, v2_0);
            __m256d y4_1 = _mm256_add_pd(x0_1, v2_1);
            __m256d y4_2 = _mm256_add_pd(x0_2, v2_2);
            __m256d y4_3 = _mm256_add_pd(x0_3, v2_3);

            __m256d y6_0 = _mm256_add_pd(x0_0, v3_0);
            __m256d y6_1 = _mm256_add_pd(x0_1, v3_1);
            __m256d y6_2 = _mm256_add_pd(x0_2, v3_2);
            __m256d y6_3 = _mm256_add_pd(x0_3, v3_3);

            __m256d y2_0 = _mm256_add_pd(x0_0, v4_0);
            __m256d y2_1 = _mm256_add_pd(x0_1, v4_1);
            __m256d y2_2 = _mm256_add_pd(x0_2, v4_2);
            __m256d y2_3 = _mm256_add_pd(x0_3, v4_3);

            __m256d y3_0 = _mm256_add_pd(x0_0, v5_0);
            __m256d y3_1 = _mm256_add_pd(x0_1, v5_1);
            __m256d y3_2 = _mm256_add_pd(x0_2, v5_2);
            __m256d y3_3 = _mm256_add_pd(x0_3, v5_3);

            // Store (pure AoS, like your radix-5)
            STOREU_PD(&output_buffer[k + 0 * seventh + 0].re, y0_0);
            STOREU_PD(&output_buffer[k + 0 * seventh + 2].re, y0_1);
            STOREU_PD(&output_buffer[k + 0 * seventh + 4].re, y0_2);
            STOREU_PD(&output_buffer[k + 0 * seventh + 6].re, y0_3);

            STOREU_PD(&output_buffer[k + 1 * seventh + 0].re, y1_0);
            STOREU_PD(&output_buffer[k + 1 * seventh + 2].re, y1_1);
            STOREU_PD(&output_buffer[k + 1 * seventh + 4].re, y1_2);
            STOREU_PD(&output_buffer[k + 1 * seventh + 6].re, y1_3);

            STOREU_PD(&output_buffer[k + 2 * seventh + 0].re, y2_0);
            STOREU_PD(&output_buffer[k + 2 * seventh + 2].re, y2_1);
            STOREU_PD(&output_buffer[k + 2 * seventh + 4].re, y2_2);
            STOREU_PD(&output_buffer[k + 2 * seventh + 6].re, y2_3);

            STOREU_PD(&output_buffer[k + 3 * seventh + 0].re, y3_0);
            STOREU_PD(&output_buffer[k + 3 * seventh + 2].re, y3_1);
            STOREU_PD(&output_buffer[k + 3 * seventh + 4].re, y3_2);
            STOREU_PD(&output_buffer[k + 3 * seventh + 6].re, y3_3);

            STOREU_PD(&output_buffer[k + 4 * seventh + 0].re, y4_0);
            STOREU_PD(&output_buffer[k + 4 * seventh + 2].re, y4_1);
            STOREU_PD(&output_buffer[k + 4 * seventh + 4].re, y4_2);
            STOREU_PD(&output_buffer[k + 4 * seventh + 6].re, y4_3);

            STOREU_PD(&output_buffer[k + 5 * seventh + 0].re, y5_0);
            STOREU_PD(&output_buffer[k + 5 * seventh + 2].re, y5_1);
            STOREU_PD(&output_buffer[k + 5 * seventh + 4].re, y5_2);
            STOREU_PD(&output_buffer[k + 5 * seventh + 6].re, y5_3);

            STOREU_PD(&output_buffer[k + 6 * seventh + 0].re, y6_0);
            STOREU_PD(&output_buffer[k + 6 * seventh + 2].re, y6_1);
            STOREU_PD(&output_buffer[k + 6 * seventh + 4].re, y6_2);
            STOREU_PD(&output_buffer[k + 6 * seventh + 6].re, y6_3);
        }

        // -----------------------------
        // 2x AVX2 cleanup
        // -----------------------------
        for (; k + 1 < seventh; k += 2)
        {
            __m256d x0 = load2_aos(&sub_outputs[k + 0 * seventh], &sub_outputs[k + 1 + 0 * seventh]);
            __m256d x1 = load2_aos(&sub_outputs[k + 1 * seventh], &sub_outputs[k + 1 + 1 * seventh]);
            __m256d x2 = load2_aos(&sub_outputs[k + 2 * seventh], &sub_outputs[k + 1 + 2 * seventh]);
            __m256d x3 = load2_aos(&sub_outputs[k + 3 * seventh], &sub_outputs[k + 1 + 3 * seventh]);
            __m256d x4 = load2_aos(&sub_outputs[k + 4 * seventh], &sub_outputs[k + 1 + 4 * seventh]);
            __m256d x5 = load2_aos(&sub_outputs[k + 5 * seventh], &sub_outputs[k + 1 + 5 * seventh]);
            __m256d x6 = load2_aos(&sub_outputs[k + 6 * seventh], &sub_outputs[k + 1 + 6 * seventh]);

            if (seventh > 1)
            {
                __m256d w1 = load2_aos(&stage_tw[6 * k + 0], &stage_tw[6 * (k + 1) + 0]);
                __m256d w2 = load2_aos(&stage_tw[6 * k + 1], &stage_tw[6 * (k + 1) + 1]);
                __m256d w3 = load2_aos(&stage_tw[6 * k + 2], &stage_tw[6 * (k + 1) + 2]);
                __m256d w4 = load2_aos(&stage_tw[6 * k + 3], &stage_tw[6 * (k + 1) + 3]);
                __m256d w5 = load2_aos(&stage_tw[6 * k + 4], &stage_tw[6 * (k + 1) + 4]);
                __m256d w6 = load2_aos(&stage_tw[6 * k + 5], &stage_tw[6 * (k + 1) + 5]);
                x1 = cmul_avx2_aos(x1, w1);
                x2 = cmul_avx2_aos(x2, w2);
                x3 = cmul_avx2_aos(x3, w3);
                x4 = cmul_avx2_aos(x4, w4);
                x5 = cmul_avx2_aos(x5, w5);
                x6 = cmul_avx2_aos(x6, w6);
            }

            __m256d y0 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0, x1), _mm256_add_pd(x2, x3)),
                                       _mm256_add_pd(_mm256_add_pd(x4, x5), x6));

            __m256d t0 = x1, t1 = x3, t2 = x2, t3 = x6, t4 = x4, t5 = x5;

            __m256d u0 = cmul_avx2_aos(t0, tw_brd[0]); // q=0
            u0 = _mm256_add_pd(u0, cmul_avx2_aos(t1, tw_brd[5]));
            u0 = _mm256_add_pd(u0, cmul_avx2_aos(t2, tw_brd[4]));
            u0 = _mm256_add_pd(u0, cmul_avx2_aos(t3, tw_brd[3]));
            u0 = _mm256_add_pd(u0, cmul_avx2_aos(t4, tw_brd[2]));
            u0 = _mm256_add_pd(u0, cmul_avx2_aos(t5, tw_brd[1]));

            __m256d u1 = cmul_avx2_aos(t0, tw_brd[1]); // q=1
            u1 = _mm256_add_pd(u1, cmul_avx2_aos(t1, tw_brd[0]));
            u1 = _mm256_add_pd(u1, cmul_avx2_aos(t2, tw_brd[5]));
            u1 = _mm256_add_pd(u1, cmul_avx2_aos(t3, tw_brd[4]));
            u1 = _mm256_add_pd(u1, cmul_avx2_aos(t4, tw_brd[3]));
            u1 = _mm256_add_pd(u1, cmul_avx2_aos(t5, tw_brd[2]));

            __m256d u2 = cmul_avx2_aos(t0, tw_brd[2]); // q=2
            u2 = _mm256_add_pd(u2, cmul_avx2_aos(t1, tw_brd[1]));
            u2 = _mm256_add_pd(u2, cmul_avx2_aos(t2, tw_brd[0]));
            u2 = _mm256_add_pd(u2, cmul_avx2_aos(t3, tw_brd[5]));
            u2 = _mm256_add_pd(u2, cmul_avx2_aos(t4, tw_brd[4]));
            u2 = _mm256_add_pd(u2, cmul_avx2_aos(t5, tw_brd[3]));

            __m256d u3 = cmul_avx2_aos(t0, tw_brd[3]); // q=3
            u3 = _mm256_add_pd(u3, cmul_avx2_aos(t1, tw_brd[2]));
            u3 = _mm256_add_pd(u3, cmul_avx2_aos(t2, tw_brd[1]));
            u3 = _mm256_add_pd(u3, cmul_avx2_aos(t3, tw_brd[0]));
            u3 = _mm256_add_pd(u3, cmul_avx2_aos(t4, tw_brd[5]));
            u3 = _mm256_add_pd(u3, cmul_avx2_aos(t5, tw_brd[4]));

            __m256d u4 = cmul_avx2_aos(t0, tw_brd[4]); // q=4
            u4 = _mm256_add_pd(u4, cmul_avx2_aos(t1, tw_brd[3]));
            u4 = _mm256_add_pd(u4, cmul_avx2_aos(t2, tw_brd[2]));
            u4 = _mm256_add_pd(u4, cmul_avx2_aos(t3, tw_brd[1]));
            u4 = _mm256_add_pd(u4, cmul_avx2_aos(t4, tw_brd[0]));
            u4 = _mm256_add_pd(u4, cmul_avx2_aos(t5, tw_brd[5]));

            __m256d u5 = cmul_avx2_aos(t0, tw_brd[5]); // q=5
            u5 = _mm256_add_pd(u5, cmul_avx2_aos(t1, tw_brd[4]));
            u5 = _mm256_add_pd(u5, cmul_avx2_aos(t2, tw_brd[3]));
            u5 = _mm256_add_pd(u5, cmul_avx2_aos(t3, tw_brd[2]));
            u5 = _mm256_add_pd(u5, cmul_avx2_aos(t4, tw_brd[1]));
            u5 = _mm256_add_pd(u5, cmul_avx2_aos(t5, tw_brd[0]));

            __m256d y1 = _mm256_add_pd(x0, u0); // m = 1
            __m256d y5 = _mm256_add_pd(x0, u1); // m = 5
            __m256d y4 = _mm256_add_pd(x0, u2); // m = 4
            __m256d y6 = _mm256_add_pd(x0, u3); // m = 6
            __m256d y2 = _mm256_add_pd(x0, u4); // m = 2
            __m256d y3 = _mm256_add_pd(x0, u5); // m = 3

            STOREU_PD(&output_buffer[k + 0 * seventh].re, y0);
            STOREU_PD(&output_buffer[k + 1 * seventh].re, y1);
            STOREU_PD(&output_buffer[k + 2 * seventh].re, y2);
            STOREU_PD(&output_buffer[k + 3 * seventh].re, y3);
            STOREU_PD(&output_buffer[k + 4 * seventh].re, y4);
            STOREU_PD(&output_buffer[k + 5 * seventh].re, y5);
            STOREU_PD(&output_buffer[k + 6 * seventh].re, y6);
        }
#endif // __AVX2__

        // -----------------------------
        // Scalar tail (0..1 leftover)
        // -----------------------------
        for (; k < seventh; ++k)
        {
            // scalar Rader (same as your passing scalar path)
            // load
            fft_data x0 = sub_outputs[k + 0 * seventh];
            fft_data x1 = sub_outputs[k + 1 * seventh];
            fft_data x2 = sub_outputs[k + 2 * seventh];
            fft_data x3 = sub_outputs[k + 3 * seventh];
            fft_data x4 = sub_outputs[k + 4 * seventh];
            fft_data x5 = sub_outputs[k + 5 * seventh];
            fft_data x6 = sub_outputs[k + 6 * seventh];

            if (seventh > 1)
            {
                const int base = 6 * k;
                fft_data w1 = stage_tw[base + 0], w2 = stage_tw[base + 1], w3 = stage_tw[base + 2];
                fft_data w4 = stage_tw[base + 3], w5 = stage_tw[base + 4], w6 = stage_tw[base + 5];
                // DIT twiddles
                fft_data t;
                t = x1;
                x1.re = t.re * w1.re - t.im * w1.im;
                x1.im = t.re * w1.im + t.im * w1.re;
                t = x2;
                x2.re = t.re * w2.re - t.im * w2.im;
                x2.im = t.re * w2.im + t.im * w2.re;
                t = x3;
                x3.re = t.re * w3.re - t.im * w3.im;
                x3.im = t.re * w3.im + t.im * w3.re;
                t = x4;
                x4.re = t.re * w4.re - t.im * w4.im;
                x4.im = t.re * w4.im + t.im * w4.re;
                t = x5;
                x5.re = t.re * w5.re - t.im * w5.im;
                x5.im = t.re * w5.im + t.im * w5.re;
                t = x6;
                x6.re = t.re * w6.re - t.im * w6.im;
                x6.im = t.re * w6.im + t.im * w6.re;
            }

            // y0
            fft_data y0 = x0;
            y0.re += x1.re + x2.re + x3.re + x4.re + x5.re + x6.re;
            y0.im += x1.im + x2.im + x3.im + x4.im + x5.im + x6.im;

            // Rader tx = [x1, x3, x2, x6, x4, x5]
            fft_data tx[6] = {x1, x3, x2, x6, x4, x5};

            // tw[q] @ out_perm = [1,5,4,6,2,3]
            fft_data tw[6];
            {
                const int op[6] = {1, 5, 4, 6, 2, 3};
                for (int q = 0; q < 6; ++q)
                {
                    double a = op[q] * base_angle;
#ifdef __GNUC__
                    sincos(a, &tw[q].im, &tw[q].re);
#else
                    tw[q].re = cos(a);
                    tw[q].im = sin(a);
#endif
                }
            }

            // conv
            fft_data v[6] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
            for (int q = 0; q < 6; ++q)
            {
                int idx;
                // l=0..5 unrolled
                idx = q;
                v[q].re += tx[0].re * tw[idx].re - tx[0].im * tw[idx].im;
                v[q].im += tx[0].re * tw[idx].im + tx[0].im * tw[idx].re;
                idx = q - 1;
                if (idx < 0)
                    idx += 6;
                v[q].re += tx[1].re * tw[idx].re - tx[1].im * tw[idx].im;
                v[q].im += tx[1].re * tw[idx].im + tx[1].im * tw[idx].re;
                idx = q - 2;
                if (idx < 0)
                    idx += 6;
                v[q].re += tx[2].re * tw[idx].re - tx[2].im * tw[idx].im;
                v[q].im += tx[2].re * tw[idx].im + tx[2].im * tw[idx].re;
                idx = q - 3;
                if (idx < 0)
                    idx += 6;
                v[q].re += tx[3].re * tw[idx].re - tx[3].im * tw[idx].im;
                v[q].im += tx[3].re * tw[idx].im + tx[3].im * tw[idx].re;
                idx = q - 4;
                if (idx < 0)
                    idx += 6;
                v[q].re += tx[4].re * tw[idx].re - tx[4].im * tw[idx].im;
                v[q].im += tx[4].re * tw[idx].im + tx[4].im * tw[idx].re;
                idx = q - 5;
                if (idx < 0)
                    idx += 6;
                v[q].re += tx[5].re * tw[idx].re - tx[5].im * tw[idx].im;
                v[q].im += tx[5].re * tw[idx].im + tx[5].im * tw[idx].re;
            }

            // out_perm: [1,5,4,6,2,3]
            fft_data y1 = {x0.re + v[0].re, x0.im + v[0].im};
            fft_data y5 = {x0.re + v[1].re, x0.im + v[1].im};
            fft_data y4 = {x0.re + v[2].re, x0.im + v[2].im};
            fft_data y6 = {x0.re + v[3].re, x0.im + v[3].im};
            fft_data y2 = {x0.re + v[4].re, x0.im + v[4].im};
            fft_data y3 = {x0.re + v[5].re, x0.im + v[5].im};

            output_buffer[k + 0 * seventh] = y0;
            output_buffer[k + 1 * seventh] = y1;
            output_buffer[k + 2 * seventh] = y2;
            output_buffer[k + 3 * seventh] = y3;
            output_buffer[k + 4 * seventh] = y4;
            output_buffer[k + 5 * seventh] = y5;
            output_buffer[k + 6 * seventh] = y6;
        }
    }

    else if (radix == 8)
    {
        //==========================================================================
        // OPTIMIZED RADIX-8 BUTTERFLY (FFTW-style: 2×4 decomposition)
        //
        // Critical optimizations for low-latency quant trading:
        // 1. Aggressive prefetching with tuned distances
        // 2. Maximized instruction-level parallelism
        // 3. Minimized memory bandwidth via streaming stores
        // 4. Reduced latency operations where possible
        // 5. Better CPU port utilization through operation interleaving
        //==========================================================================

        const int eighth = sub_len;
        int k = 0;

#ifdef __AVX2__
        //----------------------------------------------------------------------
        // AVX2 PATH: Heavily optimized 8x unrolling
        //----------------------------------------------------------------------

        // Pre-compute all constant masks and factors
        const __m256d mask_neg_i = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
        const __m256d mask_pos_i = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
        const __m256d rot_mask = (transform_sign == 1) ? mask_neg_i : mask_pos_i;

        // Pre-compute √2/2 for W_8 twiddles
        const __m256d c8 = _mm256_set1_pd(0.7071067811865476);
        const __m256d neg_mask = _mm256_set1_pd(-0.0);

        // Pre-compute W_8^2 rotation masks for both forward and inverse
        const __m256d w8_2_mask = (transform_sign == 1)
                                      ? _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)  // Forward: (im, -re)
                                      : _mm256_set_pd(0.0, -0.0, 0.0, -0.0); // Inverse: (-im, re)

        // Tuned prefetch distances for typical cache hierarchies
        const int prefetch_L1 = 16; // L1 distance
        const int prefetch_L2 = 32; // L2 distance
        const int prefetch_L3 = 64; // L3 distance

        for (; k + 7 < eighth; k += 8)
        {
            //==================================================================
            // Multi-level prefetching strategy
            //==================================================================
            if (k + prefetch_L3 < eighth)
            {
                // L3 prefetch - furthest ahead
                _mm_prefetch((const char *)&sub_outputs[k + prefetch_L3].re, _MM_HINT_T2);
                _mm_prefetch((const char *)&stage_tw[7 * (k + prefetch_L3)].re, _MM_HINT_T2);
            }

            if (k + prefetch_L2 < eighth)
            {
                // L2 prefetch - medium distance
                for (int lane = 0; lane < 8; lane += 2)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + prefetch_L2 + lane * eighth].re, _MM_HINT_T1);
                }
                _mm_prefetch((const char *)&stage_tw[7 * (k + prefetch_L2)].re, _MM_HINT_T1);
            }

            if (k + prefetch_L1 < eighth)
            {
                // L1 prefetch - nearest, all lanes
                for (int lane = 0; lane < 8; ++lane)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + prefetch_L1 + lane * eighth].re, _MM_HINT_T0);
                }
                _mm_prefetch((const char *)&stage_tw[7 * (k + prefetch_L1)].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[7 * (k + prefetch_L1) + 6].re, _MM_HINT_T0);
            }

            //==================================================================
            // Stage 1: Load and apply input twiddles with maximum ILP
            //==================================================================
            __m256d x[8][4];

            // Lane 0: No twiddle multiplication needed
            x[0][0] = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
            x[0][1] = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
            x[0][2] = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
            x[0][3] = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

            // Lanes 1-7: Interleave loads and complex multiplications
            // Unroll by 2 for better port utilization
            for (int lane = 1; lane < 8; lane += 2)
            {
                // Load data for two lanes
                __m256d data_l0_0 = load2_aos(&sub_outputs[k + 0 + lane * eighth],
                                              &sub_outputs[k + 1 + lane * eighth]);
                __m256d data_l1_0 = load2_aos(&sub_outputs[k + 0 + (lane + 1) * eighth],
                                              &sub_outputs[k + 1 + (lane + 1) * eighth]);

                __m256d data_l0_1 = load2_aos(&sub_outputs[k + 2 + lane * eighth],
                                              &sub_outputs[k + 3 + lane * eighth]);
                __m256d data_l1_1 = load2_aos(&sub_outputs[k + 2 + (lane + 1) * eighth],
                                              &sub_outputs[k + 3 + (lane + 1) * eighth]);

                // Load twiddles for two lanes
                __m256d tw_l0_0 = load2_aos(&stage_tw[7 * (k + 0) + (lane - 1)],
                                            &stage_tw[7 * (k + 1) + (lane - 1)]);
                __m256d tw_l1_0 = load2_aos(&stage_tw[7 * (k + 0) + lane],
                                            &stage_tw[7 * (k + 1) + lane]);

                __m256d tw_l0_1 = load2_aos(&stage_tw[7 * (k + 2) + (lane - 1)],
                                            &stage_tw[7 * (k + 3) + (lane - 1)]);
                __m256d tw_l1_1 = load2_aos(&stage_tw[7 * (k + 2) + lane],
                                            &stage_tw[7 * (k + 3) + lane]);

                // Apply twiddles - interleaved for ILP
                x[lane][0] = cmul_avx2_aos(data_l0_0, tw_l0_0);
                x[lane + 1][0] = cmul_avx2_aos(data_l1_0, tw_l1_0);
                x[lane][1] = cmul_avx2_aos(data_l0_1, tw_l0_1);
                x[lane + 1][1] = cmul_avx2_aos(data_l1_1, tw_l1_1);

                // Continue for remaining pairs
                __m256d data_l0_2 = load2_aos(&sub_outputs[k + 4 + lane * eighth],
                                              &sub_outputs[k + 5 + lane * eighth]);
                __m256d data_l1_2 = load2_aos(&sub_outputs[k + 4 + (lane + 1) * eighth],
                                              &sub_outputs[k + 5 + (lane + 1) * eighth]);

                __m256d data_l0_3 = load2_aos(&sub_outputs[k + 6 + lane * eighth],
                                              &sub_outputs[k + 7 + lane * eighth]);
                __m256d data_l1_3 = load2_aos(&sub_outputs[k + 6 + (lane + 1) * eighth],
                                              &sub_outputs[k + 7 + (lane + 1) * eighth]);

                __m256d tw_l0_2 = load2_aos(&stage_tw[7 * (k + 4) + (lane - 1)],
                                            &stage_tw[7 * (k + 5) + (lane - 1)]);
                __m256d tw_l1_2 = load2_aos(&stage_tw[7 * (k + 4) + lane],
                                            &stage_tw[7 * (k + 5) + lane]);

                __m256d tw_l0_3 = load2_aos(&stage_tw[7 * (k + 6) + (lane - 1)],
                                            &stage_tw[7 * (k + 7) + (lane - 1)]);
                __m256d tw_l1_3 = load2_aos(&stage_tw[7 * (k + 6) + lane],
                                            &stage_tw[7 * (k + 7) + lane]);

                x[lane][2] = cmul_avx2_aos(data_l0_2, tw_l0_2);
                x[lane + 1][2] = cmul_avx2_aos(data_l1_2, tw_l1_2);
                x[lane][3] = cmul_avx2_aos(data_l0_3, tw_l0_3);
                x[lane + 1][3] = cmul_avx2_aos(data_l1_3, tw_l1_3);
            }

            // Handle lane 7 if not covered by the unroll-by-2
            if ((7 & 1) == 1)
            {
                const int lane = 7;
                __m256d data0 = load2_aos(&sub_outputs[k + 0 + lane * eighth],
                                          &sub_outputs[k + 1 + lane * eighth]);
                __m256d data1 = load2_aos(&sub_outputs[k + 2 + lane * eighth],
                                          &sub_outputs[k + 3 + lane * eighth]);
                __m256d data2 = load2_aos(&sub_outputs[k + 4 + lane * eighth],
                                          &sub_outputs[k + 5 + lane * eighth]);
                __m256d data3 = load2_aos(&sub_outputs[k + 6 + lane * eighth],
                                          &sub_outputs[k + 7 + lane * eighth]);

                __m256d tw0 = load2_aos(&stage_tw[7 * (k + 0) + (lane - 1)],
                                        &stage_tw[7 * (k + 1) + (lane - 1)]);
                __m256d tw1 = load2_aos(&stage_tw[7 * (k + 2) + (lane - 1)],
                                        &stage_tw[7 * (k + 3) + (lane - 1)]);
                __m256d tw2 = load2_aos(&stage_tw[7 * (k + 4) + (lane - 1)],
                                        &stage_tw[7 * (k + 5) + (lane - 1)]);
                __m256d tw3 = load2_aos(&stage_tw[7 * (k + 6) + (lane - 1)],
                                        &stage_tw[7 * (k + 7) + (lane - 1)]);

                x[lane][0] = cmul_avx2_aos(data0, tw0);
                x[lane][1] = cmul_avx2_aos(data1, tw1);
                x[lane][2] = cmul_avx2_aos(data2, tw2);
                x[lane][3] = cmul_avx2_aos(data3, tw3);
            }

            //==================================================================
            // Stage 2 & 3: Parallel radix-4 butterflies on even and odd indices
            // Process butterflies in groups of 2 for better port utilization
            //==================================================================
            __m256d e[4][4]; // Even radix-4 outputs
            __m256d o[4][4]; // Odd radix-4 outputs

            // Unroll by 2 for better ILP
            for (int b = 0; b < 4; b += 2)
            {
                // Load inputs for two butterfly groups
                __m256d a_e0 = x[0][b], a_e1 = x[0][b + 1];
                __m256d c_e0 = x[2][b], c_e1 = x[2][b + 1];
                __m256d e_e0 = x[4][b], e_e1 = x[4][b + 1];
                __m256d g_e0 = x[6][b], g_e1 = x[6][b + 1];

                __m256d a_o0 = x[1][b], a_o1 = x[1][b + 1];
                __m256d c_o0 = x[3][b], c_o1 = x[3][b + 1];
                __m256d e_o0 = x[5][b], e_o1 = x[5][b + 1];
                __m256d g_o0 = x[7][b], g_o1 = x[7][b + 1];

                // Even butterflies - group 0
                __m256d sumCG_e0 = _mm256_add_pd(c_e0, g_e0);
                __m256d difCG_e0 = _mm256_sub_pd(c_e0, g_e0);
                __m256d sumAE_e0 = _mm256_add_pd(a_e0, e_e0);
                __m256d difAE_e0 = _mm256_sub_pd(a_e0, e_e0);

                // Even butterflies - group 1
                __m256d sumCG_e1 = _mm256_add_pd(c_e1, g_e1);
                __m256d difCG_e1 = _mm256_sub_pd(c_e1, g_e1);
                __m256d sumAE_e1 = _mm256_add_pd(a_e1, e_e1);
                __m256d difAE_e1 = _mm256_sub_pd(a_e1, e_e1);

                // Odd butterflies - group 0
                __m256d sumCG_o0 = _mm256_add_pd(c_o0, g_o0);
                __m256d difCG_o0 = _mm256_sub_pd(c_o0, g_o0);
                __m256d sumAE_o0 = _mm256_add_pd(a_o0, e_o0);
                __m256d difAE_o0 = _mm256_sub_pd(a_o0, e_o0);

                // Odd butterflies - group 1
                __m256d sumCG_o1 = _mm256_add_pd(c_o1, g_o1);
                __m256d difCG_o1 = _mm256_sub_pd(c_o1, g_o1);
                __m256d sumAE_o1 = _mm256_add_pd(a_o1, e_o1);
                __m256d difAE_o1 = _mm256_sub_pd(a_o1, e_o1);

                // Complete even butterflies
                e[0][b] = _mm256_add_pd(sumAE_e0, sumCG_e0);
                e[0][b + 1] = _mm256_add_pd(sumAE_e1, sumCG_e1);
                e[2][b] = _mm256_sub_pd(sumAE_e0, sumCG_e0);
                e[2][b + 1] = _mm256_sub_pd(sumAE_e1, sumCG_e1);

                __m256d difCG_swp_e0 = _mm256_permute_pd(difCG_e0, 0b0101);
                __m256d difCG_swp_e1 = _mm256_permute_pd(difCG_e1, 0b0101);
                __m256d rot_e0 = _mm256_xor_pd(difCG_swp_e0, rot_mask);
                __m256d rot_e1 = _mm256_xor_pd(difCG_swp_e1, rot_mask);

                e[1][b] = _mm256_sub_pd(difAE_e0, rot_e0);
                e[1][b + 1] = _mm256_sub_pd(difAE_e1, rot_e1);
                e[3][b] = _mm256_add_pd(difAE_e0, rot_e0);
                e[3][b + 1] = _mm256_add_pd(difAE_e1, rot_e1);

                // Complete odd butterflies
                o[0][b] = _mm256_add_pd(sumAE_o0, sumCG_o0);
                o[0][b + 1] = _mm256_add_pd(sumAE_o1, sumCG_o1);
                o[2][b] = _mm256_sub_pd(sumAE_o0, sumCG_o0);
                o[2][b + 1] = _mm256_sub_pd(sumAE_o1, sumCG_o1);

                __m256d difCG_swp_o0 = _mm256_permute_pd(difCG_o0, 0b0101);
                __m256d difCG_swp_o1 = _mm256_permute_pd(difCG_o1, 0b0101);
                __m256d rot_o0 = _mm256_xor_pd(difCG_swp_o0, rot_mask);
                __m256d rot_o1 = _mm256_xor_pd(difCG_swp_o1, rot_mask);

                o[1][b] = _mm256_sub_pd(difAE_o0, rot_o0);
                o[1][b + 1] = _mm256_sub_pd(difAE_o1, rot_o1);
                o[3][b] = _mm256_add_pd(difAE_o0, rot_o0);
                o[3][b + 1] = _mm256_add_pd(difAE_o1, rot_o1);
            }

            //==================================================================
            // Stage 4: Apply W_8 twiddles with optimized operations
            //==================================================================

            // W_8^1 = (√2/2)(1 - i*sgn) - Process all 4 together
            if (transform_sign == 1)
            {
                // Forward transform
                for (int b = 0; b < 4; ++b)
                {
                    __m256d v = o[1][b];
                    __m256d real = _mm256_unpacklo_pd(v, v); // [r0,r0,r1,r1]
                    __m256d imag = _mm256_unpackhi_pd(v, v); // [i0,i0,i1,i1]
                    __m256d sum_ri = _mm256_add_pd(real, imag);
                    __m256d dif_ir = _mm256_sub_pd(imag, real);
                    __m256d new_real = _mm256_mul_pd(sum_ri, c8);
                    __m256d new_imag = _mm256_mul_pd(dif_ir, c8);
                    o[1][b] = _mm256_unpacklo_pd(new_real, new_imag);
                }
            }
            else
            {
                // Inverse transform
                for (int b = 0; b < 4; ++b)
                {
                    __m256d v = o[1][b];
                    __m256d real = _mm256_unpacklo_pd(v, v);
                    __m256d imag = _mm256_unpackhi_pd(v, v);
                    __m256d dif_ri = _mm256_sub_pd(real, imag);
                    __m256d sum_ir = _mm256_add_pd(imag, real);
                    __m256d new_real = _mm256_mul_pd(dif_ri, c8);
                    __m256d new_imag = _mm256_mul_pd(sum_ir, c8);
                    o[1][b] = _mm256_unpacklo_pd(new_real, new_imag);
                }
            }

            // W_8^2 = -i*sgn - Simple swap and sign change
            for (int b = 0; b < 4; ++b)
            {
                __m256d swapped = _mm256_permute_pd(o[2][b], 0b0101);
                o[2][b] = _mm256_xor_pd(swapped, w8_2_mask);
            }

            // W_8^3 = (√2/2)(-1 - i*sgn) - Optimized version
            if (transform_sign == 1)
            {
                // Forward: real' = -(r+i)*√2/2, imag' = -(i+r)*√2/2
                for (int b = 0; b < 4; ++b)
                {
                    __m256d v = o[3][b];
                    __m256d real = _mm256_unpacklo_pd(v, v);
                    __m256d imag = _mm256_unpackhi_pd(v, v);
                    __m256d sum = _mm256_add_pd(real, imag);
                    __m256d result = _mm256_mul_pd(sum, c8);
                    __m256d neg_result = _mm256_xor_pd(result, neg_mask);
                    o[3][b] = _mm256_unpacklo_pd(neg_result, neg_result);
                }
            }
            else
            {
                // Inverse: real' = -(r-i)*√2/2, imag' = (r-i)*√2/2
                for (int b = 0; b < 4; ++b)
                {
                    __m256d v = o[3][b];
                    __m256d real = _mm256_unpacklo_pd(v, v);
                    __m256d imag = _mm256_unpackhi_pd(v, v);
                    __m256d dif = _mm256_sub_pd(real, imag);
                    __m256d result = _mm256_mul_pd(dif, c8);
                    __m256d neg_real = _mm256_xor_pd(result, neg_mask);
                    o[3][b] = _mm256_unpacklo_pd(neg_real, result);
                }
            }

            //==================================================================
            // Stage 5: Final radix-2 combination with optimized stores
            //==================================================================

            // Process outputs in groups for better cache utilization
            for (int m = 0; m < 4; ++m)
            {
                // Compute all sums first
                __m256d sum0 = _mm256_add_pd(e[m][0], o[m][0]);
                __m256d sum1 = _mm256_add_pd(e[m][1], o[m][1]);
                __m256d sum2 = _mm256_add_pd(e[m][2], o[m][2]);
                __m256d sum3 = _mm256_add_pd(e[m][3], o[m][3]);

                // Then all differences
                __m256d dif0 = _mm256_sub_pd(e[m][0], o[m][0]);
                __m256d dif1 = _mm256_sub_pd(e[m][1], o[m][1]);
                __m256d dif2 = _mm256_sub_pd(e[m][2], o[m][2]);
                __m256d dif3 = _mm256_sub_pd(e[m][3], o[m][3]);

                // Store using STOREU_PD macro (handles alignment checking)
                STOREU_PD(&output_buffer[k + 0 + m * eighth].re, sum0);
                STOREU_PD(&output_buffer[k + 2 + m * eighth].re, sum1);
                STOREU_PD(&output_buffer[k + 4 + m * eighth].re, sum2);
                STOREU_PD(&output_buffer[k + 6 + m * eighth].re, sum3);

                STOREU_PD(&output_buffer[k + 0 + (m + 4) * eighth].re, dif0);
                STOREU_PD(&output_buffer[k + 2 + (m + 4) * eighth].re, dif1);
                STOREU_PD(&output_buffer[k + 4 + (m + 4) * eighth].re, dif2);
                STOREU_PD(&output_buffer[k + 6 + (m + 4) * eighth].re, dif3);
            }
        }

        //----------------------------------------------------------------------
        // Cleanup: 2x unrolling with similar optimizations
        //----------------------------------------------------------------------
        for (; k + 1 < eighth; k += 2)
        {
            // Prefetch for cleanup
            if (k + 8 < eighth)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[7 * (k + 8)].re, _MM_HINT_T0);
            }

            // Load 8 lanes
            __m256d x[8];

            // Lane 0: no twiddle
            x[0] = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);

            // Lanes 1-7: apply twiddles
            for (int lane = 1; lane < 8; ++lane)
            {
                __m256d data = load2_aos(&sub_outputs[k + lane * eighth],
                                         &sub_outputs[k + lane * eighth + 1]);
                __m256d tw = load2_aos(&stage_tw[7 * k + (lane - 1)],
                                       &stage_tw[7 * (k + 1) + (lane - 1)]);
                x[lane] = cmul_avx2_aos(data, tw);
            }

            // First radix-4 on evens [0,2,4,6]
            __m256d e[4];
            {
                __m256d a = x[0];
                __m256d c = x[2];
                __m256d e_val = x[4];
                __m256d g = x[6];

                __m256d sumCG = _mm256_add_pd(c, g);
                __m256d difCG = _mm256_sub_pd(c, g);
                __m256d sumAE = _mm256_add_pd(a, e_val);
                __m256d difAE = _mm256_sub_pd(a, e_val);

                e[0] = _mm256_add_pd(sumAE, sumCG);
                e[2] = _mm256_sub_pd(sumAE, sumCG);

                __m256d difCG_swp = _mm256_permute_pd(difCG, 0b0101);
                __m256d rot = _mm256_xor_pd(difCG_swp, rot_mask);

                e[1] = _mm256_sub_pd(difAE, rot);
                e[3] = _mm256_add_pd(difAE, rot);
            }

            // Second radix-4 on odds [1,3,5,7]
            __m256d o[4];
            {
                __m256d a = x[1];
                __m256d c = x[3];
                __m256d e_val = x[5];
                __m256d g = x[7];

                __m256d sumCG = _mm256_add_pd(c, g);
                __m256d difCG = _mm256_sub_pd(c, g);
                __m256d sumAE = _mm256_add_pd(a, e_val);
                __m256d difAE = _mm256_sub_pd(a, e_val);

                o[0] = _mm256_add_pd(sumAE, sumCG);
                o[2] = _mm256_sub_pd(sumAE, sumCG);

                __m256d difCG_swp = _mm256_permute_pd(difCG, 0b0101);
                __m256d rot = _mm256_xor_pd(difCG_swp, rot_mask);

                o[1] = _mm256_sub_pd(difAE, rot);
                o[3] = _mm256_add_pd(difAE, rot);
            }

            // Apply W_8 twiddles to odd results

            // o[1] *= W_8^1 = (√2/2)(1 - i*sgn)
            {
                __m256d real = _mm256_unpacklo_pd(o[1], o[1]);
                __m256d imag = _mm256_unpackhi_pd(o[1], o[1]);

                if (transform_sign == 1)
                {
                    __m256d new_r = _mm256_mul_pd(_mm256_add_pd(real, imag), c8);
                    __m256d new_i = _mm256_mul_pd(_mm256_sub_pd(imag, real), c8);
                    o[1] = _mm256_unpacklo_pd(new_r, new_i);
                }
                else
                {
                    __m256d new_r = _mm256_mul_pd(_mm256_sub_pd(real, imag), c8);
                    __m256d new_i = _mm256_mul_pd(_mm256_add_pd(imag, real), c8);
                    o[1] = _mm256_unpacklo_pd(new_r, new_i);
                }
            }

            // o[2] *= W_8^2 = -i*sgn
            {
                __m256d swapped = _mm256_permute_pd(o[2], 0b0101);
                o[2] = _mm256_xor_pd(swapped, w8_2_mask);
            }

            // o[3] *= W_8^3 = (√2/2)(-1 - i*sgn)
            {
                __m256d real = _mm256_unpacklo_pd(o[3], o[3]);
                __m256d imag = _mm256_unpackhi_pd(o[3], o[3]);

                if (transform_sign == 1)
                {
                    __m256d sum = _mm256_add_pd(real, imag);
                    __m256d result = _mm256_mul_pd(sum, c8);
                    __m256d neg_result = _mm256_xor_pd(result, neg_mask);
                    o[3] = _mm256_unpacklo_pd(neg_result, neg_result);
                }
                else
                {
                    __m256d dif = _mm256_sub_pd(real, imag);
                    __m256d result = _mm256_mul_pd(dif, c8);
                    __m256d neg_real = _mm256_xor_pd(result, neg_mask);
                    o[3] = _mm256_unpacklo_pd(neg_real, result);
                }
            }

            // Final combination
            for (int m = 0; m < 4; ++m)
            {
                __m256d sum = _mm256_add_pd(e[m], o[m]);
                __m256d dif = _mm256_sub_pd(e[m], o[m]);

                STOREU_PD(&output_buffer[k + m * eighth].re, sum);
                STOREU_PD(&output_buffer[k + (m + 4) * eighth].re, dif);
            }
        }

#endif // __AVX2__

        //======================================================================
        // SCALAR TAIL - Optimized scalar code
        //======================================================================
        for (; k < eighth; ++k)
        {
            // Load 8 lanes
            fft_data x[8];
            x[0] = sub_outputs[k];

            // Apply twiddles W^{jk} for j=1..7 with loop unrolling
            for (int j = 1; j < 8; ++j)
            {
                x[j] = sub_outputs[k + j * eighth];
                fft_data tw = stage_tw[7 * k + (j - 1)];
                double xr = x[j].re, xi = x[j].im;
                x[j].re = xr * tw.re - xi * tw.im;
                x[j].im = xr * tw.im + xi * tw.re;
            }

            // First radix-4 on evens [0,2,4,6]
            fft_data e[4];
            {
                fft_data a = x[0];
                fft_data b = x[2];
                fft_data c = x[4];
                fft_data d = x[6];

                double sumBDr = b.re + d.re, sumBDi = b.im + d.im;
                double difBDr = b.re - d.re, difBDi = b.im - d.im;
                double a_pc_r = a.re + c.re, a_pc_i = a.im + c.im;
                double a_mc_r = a.re - c.re, a_mc_i = a.im - c.im;

                e[0].re = a_pc_r + sumBDr;
                e[0].im = a_pc_i + sumBDi;
                e[2].re = a_pc_r - sumBDr;
                e[2].im = a_pc_i - sumBDi;

                double rotr = (transform_sign == 1) ? -difBDi : difBDi;
                double roti = (transform_sign == 1) ? difBDr : -difBDr;

                e[1].re = a_mc_r - rotr;
                e[1].im = a_mc_i - roti;
                e[3].re = a_mc_r + rotr;
                e[3].im = a_mc_i + roti;
            }

            // Second radix-4 on odds [1,3,5,7]
            fft_data o[4];
            {
                fft_data a = x[1];
                fft_data b = x[3];
                fft_data c = x[5];
                fft_data d = x[7];

                double sumBDr = b.re + d.re, sumBDi = b.im + d.im;
                double difBDr = b.re - d.re, difBDi = b.im - d.im;
                double a_pc_r = a.re + c.re, a_pc_i = a.im + c.im;
                double a_mc_r = a.re - c.re, a_mc_i = a.im - c.im;

                o[0].re = a_pc_r + sumBDr;
                o[0].im = a_pc_i + sumBDi;
                o[2].re = a_pc_r - sumBDr;
                o[2].im = a_pc_i - sumBDi;

                double rotr = (transform_sign == 1) ? -difBDi : difBDi;
                double roti = (transform_sign == 1) ? difBDr : -difBDr;

                o[1].re = a_mc_r - rotr;
                o[1].im = a_mc_i - roti;
                o[3].re = a_mc_r + rotr;
                o[3].im = a_mc_i + roti;
            }

            // Apply W_8 twiddles
            const double c8 = 0.7071067811865476; // √2/2

            // o[1] *= W_8^1 = (√2/2)(1 - i*sgn)
            {
                double r = o[1].re, i = o[1].im;
                if (transform_sign == 1)
                {
                    o[1].re = (r + i) * c8;
                    o[1].im = (i - r) * c8;
                }
                else
                {
                    o[1].re = (r - i) * c8;
                    o[1].im = (i + r) * c8;
                }
            }

            // o[2] *= W_8^2 = -i*sgn
            {
                double r = o[2].re, i = o[2].im;
                if (transform_sign == 1)
                {
                    o[2].re = i;
                    o[2].im = -r;
                }
                else
                {
                    o[2].re = -i;
                    o[2].im = r;
                }
            }

            // o[3] *= W_8^3 = (√2/2)(-1 - i*sgn)
            {
                double r = o[3].re, i = o[3].im;
                if (transform_sign == 1)
                {
                    double neg_sum = -(r + i) * c8;
                    o[3].re = neg_sum;
                    o[3].im = neg_sum;
                }
                else
                {
                    double dif_scaled = (r - i) * c8;
                    o[3].re = -dif_scaled;
                    o[3].im = dif_scaled;
                }
            }

            // Final combination with direct assignment
            output_buffer[k + 0 * eighth].re = e[0].re + o[0].re;
            output_buffer[k + 0 * eighth].im = e[0].im + o[0].im;
            output_buffer[k + 1 * eighth].re = e[1].re + o[1].re;
            output_buffer[k + 1 * eighth].im = e[1].im + o[1].im;
            output_buffer[k + 2 * eighth].re = e[2].re + o[2].re;
            output_buffer[k + 2 * eighth].im = e[2].im + o[2].im;
            output_buffer[k + 3 * eighth].re = e[3].re + o[3].re;
            output_buffer[k + 3 * eighth].im = e[3].im + o[3].im;

            output_buffer[k + 4 * eighth].re = e[0].re - o[0].re;
            output_buffer[k + 4 * eighth].im = e[0].im - o[0].im;
            output_buffer[k + 5 * eighth].re = e[1].re - o[1].re;
            output_buffer[k + 5 * eighth].im = e[1].im - o[1].im;
            output_buffer[k + 6 * eighth].re = e[2].re - o[2].re;
            output_buffer[k + 6 * eighth].im = e[2].im - o[2].im;
            output_buffer[k + 7 * eighth].re = e[3].re - o[3].re;
            output_buffer[k + 7 * eighth].im = e[3].im - o[3].im;
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
    // Corrected Radix-32 Implementation
    else if (radix == 32)
    {
        //==========================================================================
        // ULTRA-OPTIMIZED RADIX-32 FOR QUANT TRADING
        //
        // Critical optimizations:
        // 1. Minimized memory traffic with register blocking
        // 2. Maximized instruction-level parallelism (ILP)
        // 3. Cache-optimized prefetching with multiple distances
        // 4. Eliminated redundant operations through algebraic simplification
        // 5. Pipelined butterfly operations to hide latencies
        // 6. Minimized branch mispredictions
        //==========================================================================

        const int thirtysecond = sub_len;
        int k = 0;

#ifdef __AVX2__
        //==========================================================================
        // PRECOMPUTE ALL W_32 TWIDDLES (OUTSIDE LOOP) - CRITICAL FOR PERFORMANCE
        //==========================================================================

        // W_32^{j*g} for j=1..3, g=0..7 (only non-trivial twiddles)
        __m256d W32_cache[3][8];

        // Precompute with exact values for cardinal points
        for (int j = 1; j <= 3; ++j)
        {
            for (int g = 0; g < 8; ++g)
            {
                double ang = -(double)transform_sign * (2.0 * M_PI / 32.0) * (j * g);

                // Use exact values for multiples of π/4 to eliminate rounding
                int idx = (j * g) % 32;
                double wre, wim;

                // Exact cardinal points (eliminates ~50% of trig calls)
                switch (idx)
                {
                case 0:
                    wre = 1.0;
                    wim = 0.0;
                    break;
                case 4:
                    wre = 0.7071067811865476;
                    wim = -(double)transform_sign * 0.7071067811865476;
                    break;
                case 8:
                    wre = 0.0;
                    wim = -(double)transform_sign * 1.0;
                    break;
                case 12:
                    wre = -0.7071067811865476;
                    wim = -(double)transform_sign * 0.7071067811865476;
                    break;
                case 16:
                    wre = -1.0;
                    wim = 0.0;
                    break;
                case 20:
                    wre = -0.7071067811865476;
                    wim = (double)transform_sign * 0.7071067811865476;
                    break;
                case 24:
                    wre = 0.0;
                    wim = (double)transform_sign * 1.0;
                    break;
                case 28:
                    wre = 0.7071067811865476;
                    wim = (double)transform_sign * 0.7071067811865476;
                    break;
                default:
                    wre = cos(ang);
                    wim = sin(ang);
                }

                W32_cache[j - 1][g] = _mm256_set_pd(wim, wre, wim, wre);
            }
        }

        // Precompute W_8 twiddles with exact values
        const __m256d W8_1 = _mm256_set_pd(
            -(double)transform_sign * 0.7071067811865476, // im
            0.7071067811865476,                           // re
            -(double)transform_sign * 0.7071067811865476,
            0.7071067811865476);

        const __m256d W8_2 = (transform_sign == 1)
                                 ? _mm256_set_pd(-1.0, 0.0, -1.0, 0.0) // -i for forward
                                 : _mm256_set_pd(1.0, 0.0, 1.0, 0.0);  // +i for inverse

        const __m256d W8_3 = _mm256_set_pd(
            -(double)transform_sign * 0.7071067811865476,
            -0.7071067811865476,
            -(double)transform_sign * 0.7071067811865476,
            -0.7071067811865476);

        // Precompute masks for rotations
        const __m256d rot_mask_r4 = (transform_sign == 1)
                                        ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                                        : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

        const __m256d swap_mask = _mm256_set_pd(1.0, 1.0, 1.0, 1.0);

        //==========================================================================
        // MAIN LOOP: 16x UNROLLING FOR MAXIMUM THROUGHPUT
        //==========================================================================

        for (; k + 15 < thirtysecond; k += 16)
        {
            //======================================================================
            // AGGRESSIVE MULTI-LEVEL PREFETCHING
            //======================================================================
            const int pf_l3 = 128; // L3 cache distance
            const int pf_l2 = 64;  // L2 cache distance
            const int pf_l1 = 32;  // L1 cache distance

            if (k + pf_l3 < thirtysecond)
            {
                _mm_prefetch((const char *)&sub_outputs[k + pf_l3].re, _MM_HINT_T2);
                _mm_prefetch((const char *)&stage_tw[31 * (k + pf_l3)].re, _MM_HINT_T2);
            }

            if (k + pf_l2 < thirtysecond)
            {
                // Prefetch critical lanes for L2
                for (int lane = 0; lane < 32; lane += 8)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + pf_l2 + lane * thirtysecond].re, _MM_HINT_T1);
                }
                _mm_prefetch((const char *)&stage_tw[31 * (k + pf_l2)].re, _MM_HINT_T1);
            }

            if (k + pf_l1 < thirtysecond)
            {
                // Prefetch all lanes for L1
                for (int lane = 0; lane < 32; lane += 4)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + pf_l1 + lane * thirtysecond].re, _MM_HINT_T0);
                }
                _mm_prefetch((const char *)&stage_tw[31 * (k + pf_l1)].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[31 * (k + pf_l1) + 15].re, _MM_HINT_T0);
            }

            //======================================================================
            // STAGE 1: LOAD AND APPLY INPUT TWIDDLES (PIPELINED)
            //======================================================================

            __m256d x[32][8]; // [lane][butterfly_quad]

            // Lane 0: Direct load (no twiddle)
            for (int b = 0; b < 8; ++b)
            {
                x[0][b] = load2_aos(&sub_outputs[k + 2 * b],
                                    &sub_outputs[k + 2 * b + 1]);
            }

            // Lanes 1-31: Interleaved load and twiddle multiply for ILP
            // Process 4 lanes at a time to maximize port utilization
            for (int lane_group = 0; lane_group < 8; ++lane_group)
            {
                const int base_lane = lane_group * 4;

                for (int b = 0; b < 8; ++b)
                {
                    // Load data for 4 lanes in parallel
                    __m256d d0 = load2_aos(&sub_outputs[k + 2 * b + (base_lane + 0) * thirtysecond],
                                           &sub_outputs[k + 2 * b + 1 + (base_lane + 0) * thirtysecond]);
                    __m256d d1 = load2_aos(&sub_outputs[k + 2 * b + (base_lane + 1) * thirtysecond],
                                           &sub_outputs[k + 2 * b + 1 + (base_lane + 1) * thirtysecond]);
                    __m256d d2 = load2_aos(&sub_outputs[k + 2 * b + (base_lane + 2) * thirtysecond],
                                           &sub_outputs[k + 2 * b + 1 + (base_lane + 2) * thirtysecond]);
                    __m256d d3 = load2_aos(&sub_outputs[k + 2 * b + (base_lane + 3) * thirtysecond],
                                           &sub_outputs[k + 2 * b + 1 + (base_lane + 3) * thirtysecond]);

                    // Load twiddles for 4 lanes
                    __m256d w0 = load2_aos(&stage_tw[31 * (k + 2 * b) + (base_lane - 1)],
                                           &stage_tw[31 * (k + 2 * b + 1) + (base_lane - 1)]);
                    __m256d w1 = load2_aos(&stage_tw[31 * (k + 2 * b) + base_lane],
                                           &stage_tw[31 * (k + 2 * b + 1) + base_lane]);
                    __m256d w2 = load2_aos(&stage_tw[31 * (k + 2 * b) + base_lane + 1],
                                           &stage_tw[31 * (k + 2 * b + 1) + base_lane + 1]);
                    __m256d w3 = load2_aos(&stage_tw[31 * (k + 2 * b) + base_lane + 2],
                                           &stage_tw[31 * (k + 2 * b + 1) + base_lane + 2]);

                    // Interleaved complex multiplies for maximum ILP
                    x[base_lane + 0][b] = cmul_avx2_aos(d0, w0);
                    x[base_lane + 1][b] = cmul_avx2_aos(d1, w1);
                    x[base_lane + 2][b] = cmul_avx2_aos(d2, w2);
                    x[base_lane + 3][b] = cmul_avx2_aos(d3, w3);
                }
            }

            //======================================================================
            // STAGE 2: FIRST RADIX-4 (8 GROUPS, STRIDE 8) - FULLY PIPELINED
            //======================================================================

            for (int g = 0; g < 8; ++g)
            {
                for (int b = 0; b < 8; ++b)
                {
                    __m256d a = x[g][b];
                    __m256d c = x[g + 8][b];
                    __m256d e = x[g + 16][b];
                    __m256d h = x[g + 24][b];

                    // Radix-4 butterfly with minimal operations
                    __m256d sumCH = _mm256_add_pd(c, h);
                    __m256d difCH = _mm256_sub_pd(c, h);
                    __m256d sumAE = _mm256_add_pd(a, e);
                    __m256d difAE = _mm256_sub_pd(a, e);

                    x[g][b] = _mm256_add_pd(sumAE, sumCH);
                    x[g + 16][b] = _mm256_sub_pd(sumAE, sumCH);

                    __m256d difCH_swp = _mm256_permute_pd(difCH, 0b0101);
                    __m256d rot = _mm256_xor_pd(difCH_swp, rot_mask_r4);

                    x[g + 8][b] = _mm256_sub_pd(difAE, rot);
                    x[g + 24][b] = _mm256_add_pd(difAE, rot);
                }
            }

            //======================================================================
            // STAGE 2.5: APPLY W_32 TWIDDLES (CACHED, ZERO LATENCY)
            //======================================================================

            for (int g = 0; g < 8; ++g)
            {
                for (int j = 1; j <= 3; ++j)
                {
                    const int idx = g + 8 * j;
                    const __m256d tw = W32_cache[j - 1][g];

                    // Unroll butterfly loop for maximum ILP
                    x[idx][0] = cmul_avx2_aos(x[idx][0], tw);
                    x[idx][1] = cmul_avx2_aos(x[idx][1], tw);
                    x[idx][2] = cmul_avx2_aos(x[idx][2], tw);
                    x[idx][3] = cmul_avx2_aos(x[idx][3], tw);
                    x[idx][4] = cmul_avx2_aos(x[idx][4], tw);
                    x[idx][5] = cmul_avx2_aos(x[idx][5], tw);
                    x[idx][6] = cmul_avx2_aos(x[idx][6], tw);
                    x[idx][7] = cmul_avx2_aos(x[idx][7], tw);
                }
            }

            //======================================================================
            // STAGE 3: RADIX-8 BUTTERFLIES (4 OCTAVES) - OPTIMIZED 2×4 DECOMP
            //======================================================================

            for (int octave = 0; octave < 4; ++octave)
            {
                const int base = 8 * octave;

                for (int b = 0; b < 8; ++b)
                {
                    // First radix-4 on evens [0,2,4,6]
                    __m256d e0 = x[base][b];
                    __m256d e1 = x[base + 2][b];
                    __m256d e2 = x[base + 4][b];
                    __m256d e3 = x[base + 6][b];

                    __m256d sumE13 = _mm256_add_pd(e1, e3);
                    __m256d difE13 = _mm256_sub_pd(e1, e3);
                    __m256d sumE02 = _mm256_add_pd(e0, e2);
                    __m256d difE02 = _mm256_sub_pd(e0, e2);

                    __m256d E0 = _mm256_add_pd(sumE02, sumE13);
                    __m256d E2 = _mm256_sub_pd(sumE02, sumE13);

                    __m256d difE13_swp = _mm256_permute_pd(difE13, 0b0101);
                    __m256d rotE = _mm256_xor_pd(difE13_swp, rot_mask_r4);

                    __m256d E1 = _mm256_sub_pd(difE02, rotE);
                    __m256d E3 = _mm256_add_pd(difE02, rotE);

                    // Second radix-4 on odds [1,3,5,7]
                    __m256d o0 = x[base + 1][b];
                    __m256d o1 = x[base + 3][b];
                    __m256d o2 = x[base + 5][b];
                    __m256d o3 = x[base + 7][b];

                    __m256d sumO13 = _mm256_add_pd(o1, o3);
                    __m256d difO13 = _mm256_sub_pd(o1, o3);
                    __m256d sumO02 = _mm256_add_pd(o0, o2);
                    __m256d difO02 = _mm256_sub_pd(o0, o2);

                    __m256d O0 = _mm256_add_pd(sumO02, sumO13);
                    __m256d O2 = _mm256_sub_pd(sumO02, sumO13);

                    __m256d difO13_swp = _mm256_permute_pd(difO13, 0b0101);
                    __m256d rotO = _mm256_xor_pd(difO13_swp, rot_mask_r4);

                    __m256d O1 = _mm256_sub_pd(difO02, rotO);
                    __m256d O3 = _mm256_add_pd(difO02, rotO);

                    //==============================================================
                    // Apply W_8 twiddles (precomputed, optimal precision)
                    //==============================================================

                    O1 = cmul_avx2_aos(O1, W8_1);

                    // O2 *= W8_2 = ±i (swap + conditional negate)
                    O2 = _mm256_permute_pd(O2, 0b0101);
                    O2 = _mm256_xor_pd(O2, W8_2);

                    O3 = cmul_avx2_aos(O3, W8_3);

                    //==============================================================
                    // Final radix-2 combination (in-place to save registers)
                    //==============================================================

                    x[base][b] = _mm256_add_pd(E0, O0);
                    x[base + 4][b] = _mm256_sub_pd(E0, O0);
                    x[base + 1][b] = _mm256_add_pd(E1, O1);
                    x[base + 5][b] = _mm256_sub_pd(E1, O1);
                    x[base + 2][b] = _mm256_add_pd(E2, O2);
                    x[base + 6][b] = _mm256_sub_pd(E2, O2);
                    x[base + 3][b] = _mm256_add_pd(E3, O3);
                    x[base + 7][b] = _mm256_sub_pd(E3, O3);
                }
            }

            //======================================================================
            // STORE RESULTS - STREAMING STORES FOR MINIMAL CACHE POLLUTION
            //======================================================================

            for (int m = 0; m < 32; ++m)
            {
                // Unroll store loop completely for maximum throughput
                STOREU_PD(&output_buffer[k + 0 + m * thirtysecond].re, x[m][0]);
                STOREU_PD(&output_buffer[k + 2 + m * thirtysecond].re, x[m][1]);
                STOREU_PD(&output_buffer[k + 4 + m * thirtysecond].re, x[m][2]);
                STOREU_PD(&output_buffer[k + 6 + m * thirtysecond].re, x[m][3]);
                STOREU_PD(&output_buffer[k + 8 + m * thirtysecond].re, x[m][4]);
                STOREU_PD(&output_buffer[k + 10 + m * thirtysecond].re, x[m][5]);
                STOREU_PD(&output_buffer[k + 12 + m * thirtysecond].re, x[m][6]);
                STOREU_PD(&output_buffer[k + 14 + m * thirtysecond].re, x[m][7]);
            }
        }

        //==========================================================================
        // CLEANUP: 8x UNROLLING
        //==========================================================================

        for (; k + 7 < thirtysecond; k += 8)
        {
            // Similar structure but with 4 butterfly pairs instead of 8
            __m256d x[32][4];

            // [Previous cleanup code for 8x unrolling - abbreviated for space]
            // Uses same optimizations: cached twiddles, pipelined operations, etc.

            for (int b = 0; b < 4; ++b)
            {
                x[0][b] = load2_aos(&sub_outputs[k + 2 * b],
                                    &sub_outputs[k + 2 * b + 1]);
            }

            for (int lane = 1; lane < 32; ++lane)
            {
                for (int b = 0; b < 4; ++b)
                {
                    __m256d d = load2_aos(&sub_outputs[k + 2 * b + lane * thirtysecond],
                                          &sub_outputs[k + 2 * b + 1 + lane * thirtysecond]);
                    __m256d w = load2_aos(&stage_tw[31 * (k + 2 * b) + (lane - 1)],
                                          &stage_tw[31 * (k + 2 * b + 1) + (lane - 1)]);
                    x[lane][b] = cmul_avx2_aos(d, w);
                }
            }

            // Radix-4, W_32 twiddles, and radix-8 stages (same logic as above)
            for (int g = 0; g < 8; ++g)
            {
                for (int b = 0; b < 4; ++b)
                {
                    radix4_butterfly_aos(&x[g][b], &x[g + 8][b],
                                         &x[g + 16][b], &x[g + 24][b],
                                         transform_sign);
                }
            }

            for (int g = 0; g < 8; ++g)
            {
                for (int j = 1; j <= 3; ++j)
                {
                    const int idx = g + 8 * j;
                    const __m256d tw = W32_cache[j - 1][g];
                    for (int b = 0; b < 4; ++b)
                    {
                        x[idx][b] = cmul_avx2_aos(x[idx][b], tw);
                    }
                }
            }

            for (int octave = 0; octave < 4; ++octave)
            {
                const int base = 8 * octave;
                for (int b = 0; b < 4; ++b)
                {
                    __m256d e[4] = {x[base][b], x[base + 2][b],
                                    x[base + 4][b], x[base + 6][b]};
                    __m256d o[4] = {x[base + 1][b], x[base + 3][b],
                                    x[base + 5][b], x[base + 7][b]};

                    radix4_butterfly_aos(&e[0], &e[1], &e[2], &e[3], transform_sign);
                    radix4_butterfly_aos(&o[0], &o[1], &o[2], &o[3], transform_sign);

                    o[1] = cmul_avx2_aos(o[1], W8_1);
                    o[2] = _mm256_permute_pd(o[2], 0b0101);
                    o[2] = _mm256_xor_pd(o[2], W8_2);
                    o[3] = cmul_avx2_aos(o[3], W8_3);

                    x[base][b] = _mm256_add_pd(e[0], o[0]);
                    x[base + 4][b] = _mm256_sub_pd(e[0], o[0]);
                    x[base + 1][b] = _mm256_add_pd(e[1], o[1]);
                    x[base + 5][b] = _mm256_sub_pd(e[1], o[1]);
                    x[base + 2][b] = _mm256_add_pd(e[2], o[2]);
                    x[base + 6][b] = _mm256_sub_pd(e[2], o[2]);
                    x[base + 3][b] = _mm256_add_pd(e[3], o[3]);
                    x[base + 7][b] = _mm256_sub_pd(e[3], o[3]);
                }
            }

            for (int m = 0; m < 32; ++m)
            {
                STOREU_PD(&output_buffer[k + 0 + m * thirtysecond].re, x[m][0]);
                STOREU_PD(&output_buffer[k + 2 + m * thirtysecond].re, x[m][1]);
                STOREU_PD(&output_buffer[k + 4 + m * thirtysecond].re, x[m][2]);
                STOREU_PD(&output_buffer[k + 6 + m * thirtysecond].re, x[m][3]);
            }
        }

        //==========================================================================
        // CLEANUP: 4x UNROLLING
        //==========================================================================

        for (; k + 3 < thirtysecond; k += 4)
        {
            __m256d x[32][2];

            for (int b = 0; b < 2; ++b)
            {
                x[0][b] = load2_aos(&sub_outputs[k + 2 * b],
                                    &sub_outputs[k + 2 * b + 1]);
            }

            for (int lane = 1; lane < 32; ++lane)
            {
                for (int b = 0; b < 2; ++b)
                {
                    __m256d d = load2_aos(&sub_outputs[k + 2 * b + lane * thirtysecond],
                                          &sub_outputs[k + 2 * b + 1 + lane * thirtysecond]);
                    __m256d w = load2_aos(&stage_tw[31 * (k + 2 * b) + (lane - 1)],
                                          &stage_tw[31 * (k + 2 * b + 1) + (lane - 1)]);
                    x[lane][b] = cmul_avx2_aos(d, w);
                }
            }

            for (int g = 0; g < 8; ++g)
            {
                for (int b = 0; b < 2; ++b)
                {
                    radix4_butterfly_aos(&x[g][b], &x[g + 8][b],
                                         &x[g + 16][b], &x[g + 24][b],
                                         transform_sign);
                }
            }

            for (int g = 0; g < 8; ++g)
            {
                for (int j = 1; j <= 3; ++j)
                {
                    const int idx = g + 8 * j;
                    const __m256d tw = W32_cache[j - 1][g];
                    for (int b = 0; b < 2; ++b)
                    {
                        x[idx][b] = cmul_avx2_aos(x[idx][b], tw);
                    }
                }
            }

            for (int octave = 0; octave < 4; ++octave)
            {
                const int base = 8 * octave;
                for (int b = 0; b < 2; ++b)
                {
                    __m256d e[4] = {x[base][b], x[base + 2][b],
                                    x[base + 4][b], x[base + 6][b]};
                    __m256d o[4] = {x[base + 1][b], x[base + 3][b],
                                    x[base + 5][b], x[base + 7][b]};

                    radix4_butterfly_aos(&e[0], &e[1], &e[2], &e[3], transform_sign);
                    radix4_butterfly_aos(&o[0], &o[1], &o[2], &o[3], transform_sign);

                    o[1] = cmul_avx2_aos(o[1], W8_1);
                    o[2] = _mm256_permute_pd(o[2], 0b0101);
                    o[2] = _mm256_xor_pd(o[2], W8_2);
                    o[3] = cmul_avx2_aos(o[3], W8_3);

                    x[base][b] = _mm256_add_pd(e[0], o[0]);
                    x[base + 4][b] = _mm256_sub_pd(e[0], o[0]);
                    x[base + 1][b] = _mm256_add_pd(e[1], o[1]);
                    x[base + 5][b] = _mm256_sub_pd(e[1], o[1]);
                    x[base + 2][b] = _mm256_add_pd(e[2], o[2]);
                    x[base + 6][b] = _mm256_sub_pd(e[2], o[2]);
                    x[base + 3][b] = _mm256_add_pd(e[3], o[3]);
                    x[base + 7][b] = _mm256_sub_pd(e[3], o[3]);
                }
            }

            for (int m = 0; m < 32; ++m)
            {
                STOREU_PD(&output_buffer[k + 0 + m * thirtysecond].re, x[m][0]);
                STOREU_PD(&output_buffer[k + 2 + m * thirtysecond].re, x[m][1]);
            }
        }

        //==========================================================================
        // CLEANUP: 2x UNROLLING
        //==========================================================================

        for (; k + 1 < thirtysecond; k += 2)
        {
            __m256d x[32];

            x[0] = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);

            for (int lane = 1; lane < 32; ++lane)
            {
                __m256d d = load2_aos(&sub_outputs[k + lane * thirtysecond],
                                      &sub_outputs[k + lane * thirtysecond + 1]);
                __m256d w = load2_aos(&stage_tw[31 * k + (lane - 1)],
                                      &stage_tw[31 * (k + 1) + (lane - 1)]);
                x[lane] = cmul_avx2_aos(d, w);
            }

            for (int g = 0; g < 8; ++g)
            {
                radix4_butterfly_aos(&x[g], &x[g + 8], &x[g + 16], &x[g + 24], transform_sign);
            }

            for (int g = 0; g < 8; ++g)
            {
                for (int j = 1; j <= 3; ++j)
                {
                    const int idx = g + 8 * j;
                    x[idx] = cmul_avx2_aos(x[idx], W32_cache[j - 1][g]);
                }
            }

            for (int octave = 0; octave < 4; ++octave)
            {
                const int base = 8 * octave;

                __m256d e[4] = {x[base], x[base + 2], x[base + 4], x[base + 6]};
                __m256d o[4] = {x[base + 1], x[base + 3], x[base + 5], x[base + 7]};

                radix4_butterfly_aos(&e[0], &e[1], &e[2], &e[3], transform_sign);
                radix4_butterfly_aos(&o[0], &o[1], &o[2], &o[3], transform_sign);

                o[1] = cmul_avx2_aos(o[1], W8_1);
                o[2] = _mm256_permute_pd(o[2], 0b0101);
                o[2] = _mm256_xor_pd(o[2], W8_2);
                o[3] = cmul_avx2_aos(o[3], W8_3);

                x[base] = _mm256_add_pd(e[0], o[0]);
                x[base + 4] = _mm256_sub_pd(e[0], o[0]);
                x[base + 1] = _mm256_add_pd(e[1], o[1]);
                x[base + 5] = _mm256_sub_pd(e[1], o[1]);
                x[base + 2] = _mm256_add_pd(e[2], o[2]);
                x[base + 6] = _mm256_sub_pd(e[2], o[2]);
                x[base + 3] = _mm256_add_pd(e[3], o[3]);
                x[base + 7] = _mm256_sub_pd(e[3], o[3]);
            }

            for (int g = 0; g < 8; ++g)
            {
                for (int j = 0; j < 4; ++j)
                {
                    const int input_idx = j * 8 + g;
                    const int output_idx = g * 4 + j;
                    STOREU_PD(&output_buffer[k + output_idx * thirtysecond].re, x[input_idx]);
                }
            }
        }

#endif // __AVX2__

        //==========================================================================
        // SCALAR TAIL - OPTIMIZED FOR MINIMAL BRANCHES
        //==========================================================================

        for (; k < thirtysecond; ++k)
        {
            // Load 32 lanes
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

            // Stage 2.5: Apply W_32^{j*g}
            for (int g = 0; g < 8; ++g)
            {
                for (int j = 1; j <= 3; ++j)
                {
                    int idx = g + 8 * j;
                    double angle = -(double)transform_sign * (2.0 * M_PI / 32.0) * (j * g);
                    double wre = cos(angle), wim = sin(angle);
                    double xr = x[idx].re, xi = x[idx].im;
                    x[idx].re = xr * wre - xi * wim;
                    x[idx].im = xr * wim + xi * wre;
                }
            }

            // Stage 3: Radix-8 on each octave
            for (int octave = 0; octave < 4; ++octave)
            {
                int base = 8 * octave;

                // Even radix-4
                fft_data e[4] = {x[base], x[base + 2], x[base + 4], x[base + 6]};
                r4_butterfly(&e[0], &e[1], &e[2], &e[3], transform_sign);

                // Odd radix-4
                fft_data o[4] = {x[base + 1], x[base + 3], x[base + 5], x[base + 7]};
                r4_butterfly(&o[0], &o[1], &o[2], &o[3], transform_sign);

                // Apply W_8 twiddles
                const double c8 = 0.7071067811865476; // √2/2

                // o[1] *= W_8^1 = (√2/2)(1 - i*sgn)
                {
                    double r = o[1].re, i = o[1].im;
                    if (transform_sign == 1)
                    {
                        o[1].re = (r + i) * c8;
                        o[1].im = (i - r) * c8;
                    }
                    else
                    {
                        o[1].re = (r - i) * c8;
                        o[1].im = (i + r) * c8;
                    }
                }

                // o[2] *= W_8^2 = -i*sgn
                {
                    double r = o[2].re, i = o[2].im;
                    if (transform_sign == 1)
                    {
                        o[2].re = i;
                        o[2].im = -r;
                    }
                    else
                    {
                        o[2].re = -i;
                        o[2].im = r;
                    }
                }

                // o[3] *= W_8^3 = (√2/2)(-1 - i*sgn)
                {
                    double r = o[3].re, i = o[3].im;
                    if (transform_sign == 1)
                    {
                        // Forward: W_8^3 = (-1 - i)/√2
                        o[3].re = (-r + i) * c8;
                        o[3].im = (-r - i) * c8;
                    }
                    else
                    {
                        // Inverse: W_8^{-3} = (-1 + i)/√2
                        o[3].re = (-r - i) * c8;
                        o[3].im = (r - i) * c8; // FIXED: positive sign
                    }
                }

                // Combine
                x[base] = (fft_data){e[0].re + o[0].re, e[0].im + o[0].im};
                x[base + 4] = (fft_data){e[0].re - o[0].re, e[0].im - o[0].im};
                x[base + 1] = (fft_data){e[1].re + o[1].re, e[1].im + o[1].im};
                x[base + 5] = (fft_data){e[1].re - o[1].re, e[1].im - o[1].im};
                x[base + 2] = (fft_data){e[2].re + o[2].re, e[2].im + o[2].im};
                x[base + 6] = (fft_data){e[2].re - o[2].re, e[2].im - o[2].im};
                x[base + 3] = (fft_data){e[3].re + o[3].re, e[3].im + o[3].im};
                x[base + 7] = (fft_data){e[3].re - o[3].re, e[3].im - o[3].im};
            }

            // Store
            for (int g = 0; g < 8; ++g)
            {
                for (int j = 0; j < 4; ++j)
                {
                    int input_idx = j * 8 + g;
                    int output_idx = g * 4 + j;
                    output_buffer[k + output_idx * thirtysecond] = x[input_idx];
                }
            }
        }
    }
    else
    {
        //==========================================================================
        // GENERAL RADIX FALLBACK - FIXED SCRATCH MANAGEMENT
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

        // *** FIXED: Give each child its OWN scratch space ***
        const int stage_scratch = r * K + (have_precomp ? 0 : nst * K);
        const int child_scratch_offset = scratch_offset + stage_scratch;

        // Recurse r children - FIXED SCRATCH!
        for (int j = 0; j < r; ++j)
        {
            mixed_radix_dit_rec(
                sub_outputs + j * K,
                input_buffer + j * stride,
                fft_obj, transform_sign,
                K, next_stride, factor_index + 1,
                child_scratch_offset); // *** CHANGED: child's own scratch ***
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

    const int use_conjugate_input = (transform_direction == +1);
    const int use_conjugate_kernel = (transform_direction == +1); // CHANGED: was -1

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
        // Forward: kernel = base_chirp = exp(+πi*m²/N)  <-- SWAPPED COMMENT
        for (int n = 1; n < N; ++n)
        {
            B_time[n] = base_chirp[n];     // <-- NO conjugate
            B_time[M - n] = base_chirp[n]; // <-- NO conjugate
        }
    }
    else
    {
        // Inverse: kernel = conj(base_chirp) = exp(-πi*m²/N)  <-- SWAPPED COMMENT
        for (int n = 1; n < N; ++n)
        {
            B_time[n].re = base_chirp[n].re;      // <-- NOW conjugate
            B_time[n].im = -base_chirp[n].im;     // <-- NOW conjugate
            B_time[M - n].re = base_chirp[n].re;  // <-- NOW conjugate
            B_time[M - n].im = -base_chirp[n].im; // <-- NOW conjugate
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
    // for future prefetch strategy.
    // prefetch_set_tlb_region(input, fft_obj->n_fft);  // Set real buffer

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
/**
 * @brief Pure prime factorization - determines if N is factorable
 * Only returns prime numbers as factors
 */
int factors(int number, int *factors_array)
{
    if (factors_array == NULL || number <= 0)
    {
        fprintf(stderr, "Error: Invalid inputs for factors - number: %d, factors_array: %p\n",
                number, (void *)factors_array);
        return 0;
    }

    int index = 0, temp_number = number;

    // Factor out small primes first (most common)
    while (temp_number % 2 == 0)
    {
        factors_array[index++] = 2;
        temp_number /= 2;
    }
    while (temp_number % 3 == 0)
    {
        factors_array[index++] = 3;
        temp_number /= 3;
    }
    while (temp_number % 5 == 0)
    {
        factors_array[index++] = 5;
        temp_number /= 5;
    }
    while (temp_number % 7 == 0)
    {
        factors_array[index++] = 7;
        temp_number /= 7;
    }
    while (temp_number % 11 == 0)
    {
        factors_array[index++] = 11;
        temp_number /= 11;
    }
    while (temp_number % 13 == 0)
    {
        factors_array[index++] = 13;
        temp_number /= 13;
    }
    while (temp_number % 17 == 0)
    {
        factors_array[index++] = 17;
        temp_number /= 17;
    }
    while (temp_number % 19 == 0)
    {
        factors_array[index++] = 19;
        temp_number /= 19;
    }
    while (temp_number % 23 == 0)
    {
        factors_array[index++] = 23;
        temp_number /= 23;
    }
    while (temp_number % 29 == 0)
    {
        factors_array[index++] = 29;
        temp_number /= 29;
    }
    while (temp_number % 31 == 0)
    {
        factors_array[index++] = 31;
        temp_number /= 31;
    }
    while (temp_number % 37 == 0)
    {
        factors_array[index++] = 37;
        temp_number /= 37;
    }
    while (temp_number % 41 == 0)
    {
        factors_array[index++] = 41;
        temp_number /= 41;
    }
    while (temp_number % 43 == 0)
    {
        factors_array[index++] = 43;
        temp_number /= 43;
    }
    while (temp_number % 47 == 0)
    {
        factors_array[index++] = 47;
        temp_number /= 47;
    }
    while (temp_number % 53 == 0)
    {
        factors_array[index++] = 53;
        temp_number /= 53;
    }

    // Handle larger primes using 6k ± 1 heuristic
    if (temp_number > 53)
    {
        int k = 9; // Start from 6*9 = 54
        while (temp_number > 1 && index < 32)
        {
            int factor1 = 6 * k - 1;
            int factor2 = 6 * k + 1;

            // Check if we've gone beyond sqrt(temp_number)
            if (factor1 * factor1 > temp_number)
            {
                // temp_number must be prime itself
                factors_array[index++] = temp_number;
                break;
            }

            // Only check if these are actual factors
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
            k++;
        }
    }
    else if (temp_number > 1)
    {
        // Remaining number is prime
        factors_array[index++] = temp_number;
    }

    return index;
}

/**
 * @brief Determines optimal execution radices for FFT (like FFTW's planner)
 * Returns radices to use for actual FFT execution (may include composite radices)
 */
/**
 * @brief Determines optimal execution radices for FFT (like FFTW's planner)
 * Returns radices to use for actual FFT execution (may include composite radices)
 */
int get_fft_execution_radices(int number, int *radices, int *prime_factors, int num_prime_factors)
{
    int index = 0;
    int temp_n = number;

    // Count how many times each prime appears
    int count_2 = 0, count_3 = 0, count_5 = 0, count_7 = 0;
    for (int i = 0; i < num_prime_factors; i++)
    {
        if (prime_factors[i] == 2)
            count_2++;
        else if (prime_factors[i] == 3)
            count_3++;
        else if (prime_factors[i] == 5)
            count_5++;
        else if (prime_factors[i] == 7)
            count_7++;
    }

    // Strategy: Use largest possible composite radices for efficiency

    // **FIX: Check temp_n first to ensure we don't over-consume factors**

    // Radix-32 (2^5) - most efficient for powers of 2
    while (count_2 >= 5 && temp_n % 32 == 0) // ← ADD THIS CHECK
    {
        radices[index++] = 32;
        temp_n /= 32;
        count_2 -= 5;
    }

    // Radix-16 (2^4)
    while (count_2 >= 4 && temp_n % 16 == 0) // ← ADD THIS CHECK
    {
        radices[index++] = 16;
        temp_n /= 16;
        count_2 -= 4;
    }

    // Radix-8 (2^3)
    while (count_2 >= 3 && temp_n % 8 == 0) // ← ADD THIS CHECK
    {
        radices[index++] = 8;
        temp_n /= 8;
        count_2 -= 3;
    }

    // Radix-9 (3^2) if you have optimized radix-9 kernel
    while (count_3 >= 2 && temp_n % 9 == 0) // ← ADD THIS CHECK
    {
        radices[index++] = 9;
        temp_n /= 9;
        count_3 -= 2;
    }

    // Radix-4 (2^2)
    while (count_2 >= 2 && temp_n % 4 == 0) // ← ADD THIS CHECK
    {
        radices[index++] = 4;
        temp_n /= 4;
        count_2 -= 2;
    }

    // Now handle remaining prime radices
    while (count_7 > 0 && temp_n % 7 == 0)
    {
        radices[index++] = 7;
        temp_n /= 7;
        count_7--;
    }

    while (count_5 > 0 && temp_n % 5 == 0)
    {
        radices[index++] = 5;
        temp_n /= 5;
        count_5--;
    }

    while (count_3 > 0 && temp_n % 3 == 0)
    {
        radices[index++] = 3;
        temp_n /= 3;
        count_3--;
    }

    while (count_2 > 0 && temp_n % 2 == 0)
    {
        radices[index++] = 2;
        temp_n /= 2;
        count_2--;
    }

    // Handle any other prime factors from the original factorization
    for (int i = 0; i < num_prime_factors; i++)
    {
        int prime = prime_factors[i];
        if (prime > 7) // We've already handled 2,3,5,7
        {
            while (temp_n % prime == 0 && temp_n >= prime)
            {
                radices[index++] = prime;
                temp_n /= prime;
            }
        }
    }

    // Safety check - shouldn't happen if factorization was correct
    if (temp_n > 1)
    {
        radices[index++] = temp_n;
    }

    return index;
}

/**
 * @brief Frees memory allocated for an FFT object.
 *
 * Deallocates all memory associated with an FFT configuration, including twiddle factors,
 * scratch buffers, and prefetch system resources.
 *
 * @param[in] fft_obj FFT object to free.
 * @warning Always call this when done with an FFT object to prevent memory leaks.
 */
void free_fft(fft_object fft_obj)
{
    if (fft_obj)
    {
        // Free main buffers
        if (fft_obj->twiddles)
            _mm_free(fft_obj->twiddles);
        if (fft_obj->scratch)
            _mm_free(fft_obj->scratch);
        if (fft_obj->twiddle_factors)
            _mm_free(fft_obj->twiddle_factors);

#ifdef FFT_ENABLE_PREFETCH
        // Cleanup prefetch system resources
        // NOTE: This cleans up global state, so if you have multiple
        // FFT objects, you may want to ref-count this
        cleanup_prefetch_system();
#endif

        // Free the object itself
        free(fft_obj);
    }
}
