#include "highspeedFFT.h"
#include "time.h"
#include <immintrin.h>

//==============================================================================
// INLINE‑FORCE MACRO
//==============================================================================
#ifdef _MSC_VER
#define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

#ifndef ADDSUB_ROT
#define ADDSUB_ROT 0
#endif

//==============================================================================
// CONSTANT VECTORS
//==============================================================================
#define AVX_ONE _mm256_set1_pd(1.0) // 256‑bit vector of all 1.0
#define SSE2_ONE _mm_set1_pd(1.0)   // 128‑bit vector of all 1.0

//==============================================================================
// FMA MACROS (256‑bit)
//==============================================================================
#if defined(__FMA__) || defined(USE_FMA)
#define FMADD(a, b, c) _mm256_fmadd_pd((a), (b), (c))
#define FMSUB(a, b, c) _mm256_fmsub_pd((a), (b), (c))
#else
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
#endif

//==============================================================================
// FMA‑FALLBACK FOR SSE2 (always fallback: SSE2 has no FMA3 opcodes)
//==============================================================================
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

//==============================================================================
// LOAD / STORE MACROS
//==============================================================================
#ifdef USE_ALIGNED_SIMD
// AVX2 aligned
#define LOAD_PD(ptr) _mm256_load_pd((double const *)(ptr))
#define STORE_PD(ptr, v) _mm256_store_pd((double *)(ptr), (v))
// SSE2 aligned
#define LOAD_SSE2(ptr) _mm_load_pd((double const *)(ptr))
#define STORE_SSE2(ptr, v) _mm_store_pd((double *)(ptr), (v))
#else
// AVX2 unaligned
#define LOAD_PD(ptr) _mm256_loadu_pd((double const *)(ptr))
#define STORE_PD(ptr, v) _mm256_storeu_pd((double *)(ptr), (v))
// SSE2 unaligned
#define LOAD_SSE2(ptr) _mm_loadu_pd((double const *)(ptr))
#define STORE_SSE2(ptr, v) _mm_storeu_pd((double *)(ptr), (v))
#endif

#define LOADU_SSE2(ptr) _mm_loadu_pd((const double *)(ptr))

//==============================================================================
// OPTIONAL: Unaligned AVX2 loads for mixed‑radix codepaths
//==============================================================================
#define LOADU_PD(ptr) _mm256_loadu_pd((double const *)(ptr))
#define STOREU_PD(ptr, v) _mm256_storeu_pd((double *)(ptr), (v))

// File-scope scalar constants for radix-7
static const double C1 = 0.62348980185, C2 = -0.22252093395, C3 = -0.9009688679; // cos(51.43°), cos(102.86°), cos(154.29°)
static const double S1 = 0.78183148246, S2 = 0.97492791218, S3 = 0.43388373911;  // sin(51.43°), sin(102.86°), sin(154.29°)

static const double C3_SQRT3BY2 = 0.8660254037844386; // √3/2 for 120° rotation

// File-scope constants for radix-5
static const double C5_1 = 0.30901699437;  // cos(72°)
static const double C5_2 = -0.80901699437; // cos(144°)
static const double S5_1 = 0.95105651629;  // sin(72°)
static const double S5_2 = 0.58778525229;  // sin(144°)

static const double C8_1 = 0.7071067811865476; // √2/2 for 45° rotation

// File-scope constants for radix-11
static const double C11_1 = 0.8412535328311812;   // cos(2π/11)
static const double C11_2 = 0.4154150130018864;   // cos(4π/11)
static const double C11_3 = -0.14231483827328514; // cos(6π/11)
static const double C11_4 = -0.654860733945285;   // cos(8π/11)
static const double C11_5 = -0.9594929736144974;  // cos(10π/11)
static const double S11_1 = 0.5406408174555976;   // sin(2π/11)
static const double S11_2 = 0.9096319953545184;   // sin(4π/11)
static const double S11_3 = 0.9898214418809327;   // sin(6π/11)
static const double S11_4 = 0.7557495743542583;   // sin(8π/11)
static const double S11_5 = 0.28173255684142967;  // sin(10π/11)

/**
 * @brief Build configuration option for twiddle factor computation.
 * Define USE_TWIDDLE_TABLES to use precomputed lookup tables for radices 2, 3, 4, 5, 7, 8, 11, and 13.
 * If undefined, all twiddle factors are computed dynamically at runtime using cos and sin.
 */
#define USE_TWIDDLE_TABLES

// Precomputed lookup table for dividebyN up to 1024
#define LOOKUP_MAX 1024
static const int primes[] = {2, 3, 4, 5, 7, 8, 11, 13, 17, 23, 29, 31, 37, 41, 43, 47, 53};
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

// Precomputed twiddle factor tables for common radices
typedef struct
{
    double re;
    double im;
} complex_t;

static const complex_t twiddle_radix2[] = {{1.0, 0.0}, {0.0, -1.0}};
static const complex_t twiddle_radix3[] = {{1.0, 0.0}, {-0.5, -0.86602540378}, {-0.5, 0.86602540378}};
static const complex_t twiddle_radix4[] = {{1.0, 0.0}, {0.0, -1.0}, {-1.0, 0.0}, {0.0, 1.0}};
static const complex_t twiddle_radix5[] = {{1.0, 0.0}, {0.30901699437, -0.95105651629}, {-0.80901699437, -0.58778525229}, {-0.80901699437, 0.58778525229}, {0.30901699437, 0.95105651629}};
static const complex_t twiddle_radix7[] = {{1.0, 0.0}, {0.62348980185, -0.78183148246}, {-0.22252093395, -0.97492791218}, {-0.9009688679, -0.43388373911}, {-0.9009688679, 0.43388373911}, {-0.22252093395, 0.97492791218}, {0.62348980185, 0.78183148246}};
static const complex_t twiddle_radix8[] = {{1.0, 0.0}, {0.70710678118, -0.70710678118}, {0.0, -1.0}, {-0.70710678118, -0.70710678118}, {-1.0, 0.0}, {-0.70710678118, 0.70710678118}, {0.0, 1.0}, {0.70710678118, 0.70710678118}};

static const complex_t twiddle_radix11[] = {
    {1.0, 0.0},                                  // k=0
    {0.8412535328311812, -0.5406408174555976},   // k=1
    {0.4154150130018864, -0.9096319953545184},   // k=2
    {-0.14231483827328514, -0.9898214418809327}, // k=3
    {-0.654860733945285, -0.7557495743542583},   // k=4
    {-0.9594929736144974, -0.28173255684142967}, // k=5
    {-0.9594929736144974, 0.28173255684142967},  // k=6
    {-0.654860733945285, 0.7557495743542583},    // k=7
    {-0.14231483827328514, 0.9898214418809327},  // k=8
    {0.4154150130018864, 0.9096319953545184},    // k=9
    {0.8412535328311812, 0.5406408174555976}     // k=10
};

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

// Update the twiddle_tables array to include new radices
static const complex_t *twiddle_tables[14] = {
    /* 0 */ [0] = NULL,
    /* 2 */[2] = twiddle_radix2,
    /* 3 */[3] = twiddle_radix3,
    /* 4 */[4] = twiddle_radix4,
    /* 5 */[5] = twiddle_radix5,
    /* 7 */[7] = twiddle_radix7,
    /* 8 */[8] = twiddle_radix8,
    /*11 */[11] = twiddle_radix11,
    /*13 */[13] = twiddle_radix13};

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
 * @brief Number of precomputed chirp sequences.
 */
static int num_precomputed = 0;

/**
 * @brief Flag indicating whether the chirp table has been initialized.
 */
static int chirp_initialized = 0;

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
__attribute__((constructor)) static void init_bluestein_chirp(void)
{
    if (chirp_initialized)
        return;

    // Figure out how much space we need for all chirp sequences
    // We’re using pre_sizes[] = {1, 2, 3, 4, 5, 7, 15, 20, 31, 64}
    int total_chirp = 0;
    for (int i = 0; i < num_pre; i++)
    {
        total_chirp += ((pre_sizes[i] + 3) & ~3); // Round up to next multiple of 4
    }

    // Allocate our three arrays:
    // - bluestein_chirp: array of pointers to each chirp sequence
    // - chirp_sizes: stores the length of each chirp (matches pre_sizes)
    // - all_chirps: one big block for all chirp data
    bluestein_chirp = (fft_data **)malloc(num_pre * sizeof(fft_data *));
    chirp_sizes = (int *)malloc(num_pre * sizeof(int));
    all_chirps = (fft_data *)_mm_malloc(total_chirp * sizeof(fft_data), 32);
    if (!bluestein_chirp || !chirp_sizes || !all_chirps)
    {
        // If any allocation fails, clean up what we got and bail
        // Trade-off: Exiting is harsh, but it’s a startup error, so recovery is tricky
        fprintf(stderr, "Error: Memory allocation failed for Bluestein chirp table\n");
        _mm_free(all_chirps);
        free(bluestein_chirp);
        free(chirp_sizes);
        // exit
    }

    // Set up the pointers and fill the chirp sequences
    // We walk through each size, point bluestein_chirp[i] to the right spot in all_chirps,
    // and compute the chirp values (e^{\pi i n^2 / N})
    int offset = 0;
    for (int idx = 0; idx < num_pre; idx++)
    {
        int n = pre_sizes[idx];
        chirp_sizes[idx] = n;                       // Store the size for lookup in bluestein_exp
        bluestein_chirp[idx] = all_chirps + offset; // Point to the current chunk
        offset += ((n + 3) & ~3);                   // Move offset to next aligned boundary

        // Compute the chirp sequence for this length
        // The angle is π n^2 / N, with a quadratic index (l2) to avoid floating-point drift
        // Why l2? It’s a clever trick to compute n^2 mod 2N without big numbers
        fft_type theta = M_PI / n;
        int l2 = 0, len2 = 2 * n;
        for (int i = 0; i < n; i++)
        {
            fft_type angle = theta * l2;
            bluestein_chirp[idx][i].re = cos(angle);
            bluestein_chirp[idx][i].im = sin(angle);
            l2 += 2 * i + 1; // Quadratic term: n^2 = (2i+1) mod 2N
            while (l2 > len2)
                l2 -= len2; // Wrap around to keep indices in bounds
        }
    }

    chirp_initialized = 1;
}

__attribute__((destructor)) static void cleanup_bluestein_chirp(void)
{
    _mm_free(all_chirps);
    free(bluestein_chirp);
    free(chirp_sizes);
    all_chirps = NULL;
    bluestein_chirp = NULL;
    chirp_sizes = NULL;
    chirp_initialized = 0;
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
    // Ensure signal length is positive and direction is +1 or -1
    // Invalid inputs are a user error, so we exit with a clear message
    if (signal_length <= 0 || (transform_direction != 1 && transform_direction != -1))
    {
        fprintf(stderr, "Error: Signal length (%d) or direction (%d) is invalid\n",
                signal_length, transform_direction);
        return NULL;
    }

    // Step 2: Allocate fft_set structure
    // Allocate early to safely write stage_twiddle_offset in Step 5
    fft_object fft_config = (fft_object)malloc(sizeof(struct fft_set));
    if (!fft_config)
    {
        fprintf(stderr, "Error: Failed to allocate fft_set structure\n");
        return NULL;
    }
    fft_config->num_precomputed_stages = 0; // Initialize stage count

    // Step 3: Initialize algorithm flags
    // Check if length factors into small primes and if it’s a power of 2, 3, 5, 7, 11, or 13
    int is_factorable = dividebyN(signal_length);
    int is_power_of_2 = 0, is_power_of_3 = 0, is_power_of_5 = 0, is_power_of_7 = 0;
    int is_power_of_11 = 0, is_power_of_13 = 0;
    int twiddle_count = 0, max_scratch_size = 0, max_padded_length = 0;

    // Step 4: Set up buffer sizes and check power-of-radix
    // Mixed-radix uses N directly; Bluestein pads to next power of 2 ≥ 2N-1
    if (is_factorable)
    {
        max_padded_length = signal_length; // No padding for mixed-radix
        twiddle_count = signal_length;     // N twiddles for all stages
        // Check for pure powers using is_exact_power
        is_power_of_2 = (signal_length & (signal_length - 1)) == 0; // Fast power-of-2 check
        is_power_of_3 = is_exact_power(signal_length, 3);
        is_power_of_5 = is_exact_power(signal_length, 5);
        is_power_of_7 = is_exact_power(signal_length, 7);
        is_power_of_11 = is_exact_power(signal_length, 11);
        is_power_of_13 = is_exact_power(signal_length, 13);
    }
    else
    {
        // Bluestein needs M ≥ 2N-1, rounded up to power of 2.
        // Use an integer next_pow2 to avoid FP edge cases.
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
        twiddle_count = max_padded_length; // M twiddles for Bluestein
    }

    // Step 5: Compute memory requirements
    // Estimate scratch and twiddle_factors sizes based on factorization
    int temp_factors[64];
    int num_factors = factors(is_factorable ? signal_length : max_padded_length, temp_factors);
    int twiddle_factors_size = 0; // Size for precomputed twiddles
    int scratch_needed = 0;       // Scratch for recursion or Bluestein

    if (is_factorable)
    {
        int temp_N = signal_length;
        if (is_power_of_2 || is_power_of_3 || is_power_of_5 || is_power_of_7 ||
            is_power_of_11 || is_power_of_13)
        {
            // For pure-power FFTs, sum needs per recursion level
            // Radix-r needs (r-1)*(N/r) twiddles, r*(N/r) scratch
            int radix = is_power_of_2 ? 2 : is_power_of_3 ? 3
                                        : is_power_of_5   ? 5
                                        : is_power_of_7   ? 7
                                        : is_power_of_11  ? 11
                                                          : 13;
            int stage = 0;
            for (int n = signal_length; n >= radix; n /= radix)
            {
                int sub_fft_size = n / radix;
                // Store stage offset for twiddle_factors
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
            fft_config->num_precomputed_stages = stage; // Record number of stages
        }
        else
        {
            // Mixed-radix: r*(N/r) outputs, (r-1)*(N/r) twiddles for radices ≤ 13
            for (int i = 0; i < num_factors; i++)
            {
                int radix = temp_factors[i];
                scratch_needed += radix * (temp_N / radix); // Outputs
                if (radix <= 13)
                {
                    scratch_needed += (radix - 1) * (temp_N / radix); // Twiddles
                }
                temp_N /= radix;
            }
        }

        // Ensure scratch size covers worst-case, fallback to 4*N
        // Note: Mixed-radix may need more scratch for twiddles in radix-11/13
        max_scratch_size = scratch_needed;
        if (max_scratch_size < 4 * signal_length)
        {
            max_scratch_size = 4 * signal_length;
        }
    }
    else
    {
        // Bluestein: 4*M for chirped_signal, temp_chirp, ifft_result, chirp_sequence
        max_scratch_size = 4 * max_padded_length;
    }

    // Step 6: Allocate twiddle and scratch buffers
    // 32-byte aligned for AVX2/SSE2 SIMD performance
    fft_config->twiddles = (fft_data *)_mm_malloc(twiddle_count * sizeof(fft_data), 32);
    fft_config->scratch = (fft_data *)_mm_malloc(max_scratch_size * sizeof(fft_data), 32);
    fft_config->twiddle_factors = NULL;

    // Check allocation failures and clean up
    if (!fft_config->twiddles || !fft_config->scratch)
    {
        fprintf(stderr, "Error: Failed to allocate twiddle or scratch buffers\n");
        free_fft(fft_config);
        return NULL;
    }

    // Step 7: Allocate twiddle_factors for pure-power FFTs
    // Precompute twiddles to skip copying in mixed_radix_dit_rec
    if (is_factorable && (is_power_of_2 || is_power_of_3 || is_power_of_5 || is_power_of_7 ||
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
    // Set lengths, direction, algorithm type
    fft_config->n_input = signal_length;
    fft_config->n_fft = is_factorable ? signal_length : max_padded_length;
    fft_config->sgn = transform_direction;
    fft_config->max_scratch_size = max_scratch_size;
    fft_config->lt = is_factorable ? 0 : 1; // 0 = mixed-radix, 1 = Bluestein

    // Step 9: Factorize n_fft
    // Store prime factors for recursion in mixed_radix_dit_rec
    fft_config->lf = factors(fft_config->n_fft, fft_config->factors);

    // Step 10: Compute twiddle factors
    // Populate twiddles with e^{-2πi k / N} (or e^{-2πi k / M})
    build_twiddles_linear(fft_config->twiddles, fft_config->n_fft);

    // Step 11: Populate twiddle_factors for pure-power FFTs
    // Store W_{N_stage}^{j*k} (j=1..radix-1) for each level, *mapped* into the global W_N table.
    // Mapping: W_{N_stage}^{p} == W_{N}^{p * (N / N_stage)}.
    // Step 11: Populate twiddle_factors for pure-power FFTs (k-major layout)
    // Requires fft_config->twiddles[m] == W_{n_fft}^m  (linear table; see #2 below)
    if (fft_config->twiddle_factors)
    {
        int offset = 0;
        const int radix = (is_power_of_2 ? 2 : (is_power_of_3 ? 3 : (is_power_of_5 ? 5 : (is_power_of_7 ? 7 : (is_power_of_11 ? 11 : 13)))));

        // Walk pure-power stages: N_stage = signal_length, signal_length/radix, ...
        for (int N_stage = signal_length; N_stage >= radix; N_stage /= radix)
        {
            const int sub_len = N_stage / radix;
            const int stride = fft_config->n_fft / N_stage; // exact by construction

            for (int k = 0; k < sub_len; ++k)
            {
                const int base = (radix - 1) * k; // k-major
                for (int j = 1; j < radix; ++j)
                {
                    // Map W_{N_stage}^{j*k} => W_{n_fft}^{ (j*k) * (n_fft/N_stage) }
                    const int p = (j * k) % N_stage;
                    const int idxN = (p * stride) % fft_config->n_fft;
                    fft_config->twiddle_factors[offset + base + (j - 1)] =
                        fft_config->twiddles[idxN];
                }
            }
            offset += (radix - 1) * sub_len;
        }
    }

    // Step 12: Adjust twiddles for inverse FFT
    // Flip imaginary parts for e^{+2πi k / N}
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
    // Ready for fft_exec with all buffers and factors set
    return fft_config;
}

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
 * @brief Conjugate two AoS-packed complex doubles with AVX (no cross-lane ops).
 *
 * Treats @p z as two complex numbers packed in Array-of-Structures (AoS) layout:
 *   [ re0, im0, re1, im1 ]  (lane 0..3).
 *
 * Conjugation flips the sign of the imaginary parts only:
 *   [ re0, -im0, re1, -im1 ].
 *
 * This uses an XOR with a sign-bit mask built from ±0.0 to toggle the sign
 * of lanes 1 and 3 (the imaginary lanes) without changing the reals.
 *
 * @param z AoS-packed vector [ re0, im0, re1, im1 ].
 * @return __m256d AoS-packed conjugate [ re0, -im0, re1, -im1 ].
 *
 * @note Requires AVX. Data layout must be AoS (interleaved re,im).
 */
static ALWAYS_INLINE __m256d conj_avx2_aos(__m256d z)
{
    const __m256d mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); // lanes: [re1, im1, re0, im0]? NO.
    // Correct mask is [0.0, -0.0, 0.0, -0.0] for [re0, im0, re1, im1]:
    const __m256d corr = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
    (void)mask; // avoid unused if you keep both lines for reference
    return _mm256_xor_pd(z, corr);
}

/**
 * @brief Conjugate one AoS-packed complex double with SSE2.
 *
 * Treats @p z as a single complex number in AoS layout:
 *   [ re, im ]  (lane 0..1).
 *
 * Conjugation flips the sign of the imaginary part only:
 *   [ re, -im ].
 *
 * @param z AoS-packed vector [ re, im ].
 * @return __m128d AoS-packed conjugate [ re, -im ].
 *
 * @note Requires SSE2. Data layout must be AoS (interleaved re,im).
 */
static ALWAYS_INLINE __m128d conj_sse2_aos(__m128d z)
{
    const __m128d mask = _mm_set_pd(-0.0, 0.0);
    return _mm_xor_pd(z, mask);
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
    __m256d ar_br = _mm256_mul_pd(ar, br);
    __m256d ai_bi = _mm256_mul_pd(ai, bi);
    __m256d ar_bi = _mm256_mul_pd(ar, bi);
    __m256d ai_br = _mm256_mul_pd(ai, br);
    *rr = _mm256_sub_pd(ar_br, ai_bi);
    *ri = _mm256_add_pd(ar_bi, ai_br);
}

#ifndef FFT_PREFETCH_DISTANCE
#define FFT_PREFETCH_DISTANCE 8 // ~64B ahead for AoS complex<double>
#endif

#define FFT_PREFETCH_AOS(ptr)                                \
    do                                                       \
    {                                                        \
        _mm_prefetch((const char *)&(ptr)->re, _MM_HINT_T0); \
        _mm_prefetch((const char *)&(ptr)->im, _MM_HINT_T0); \
    } while (0)

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
 * @brief Complex multiply (pairwise) in SoA for SSE2 (2-wide).
 *
 * Computes, lane-wise:
 *   (ar + i*ai) * (br + i*bi)  ->  rr + i*ri
 *
 * Where each @c __m128d packs two doubles (lane 0 / lane 1).
 *
 * @param[in]  ar  Real parts of the left operand (2-wide).
 * @param[in]  ai  Imag parts of the left operand (2-wide).
 * @param[in]  br  Real parts of the right operand (2-wide).
 * @param[in]  bi  Imag parts of the right operand (2-wide).
 * @param[out] rr  Real parts of the result (2-wide).
 * @param[out] ri  Imag parts of the result (2-wide).
 *
 * @note Uses standard complex multiply:
 *       rr = ar*br - ai*bi,  ri = ar*bi + ai*br.
 * @warning Outputs @p rr and @p ri are fully overwritten.
 */
static inline void cmul_soa_sse2(__m128d ar, __m128d ai, __m128d br, __m128d bi, __m128d *rr, __m128d *ri)
{
    *rr = _mm_sub_pd(_mm_mul_pd(ar, br), _mm_mul_pd(ai, bi));
    *ri = _mm_add_pd(_mm_mul_pd(ar, bi), _mm_mul_pd(ai, br));
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

/**
 * Build per-stage twiddles in SoA (k-major):
 * For radix=4: we need W^{k}, W^{2k}, W^{3k} into (w1r,w1i, w2r,w2i, w3r,w3i).
 * Mapping: W_{L}^{p} = W_{N}^{ p * (N/L) }.
 */
static ALWAYS_INLINE void stage_twiddles_soa_fill(int L, int sub_len,
                                                  const fft_object obj,
                                                  double *w1r, double *w1i,
                                                  double *w2r, double *w2i,
                                                  double *w3r, double *w3i)
{
    const int N = obj->n_fft;
    const int step = N / L;
    for (int k = 0; k < sub_len; ++k)
    {
        int p1 = (1 * k) % L;
        int p2 = (2 * k) % L;
        int p3 = (3 * k) % L;

        const fft_data t1 = obj->twiddles[(p1 * step) % N];
        const fft_data t2 = obj->twiddles[(p2 * step) % N];
        const fft_data t3 = obj->twiddles[(p3 * step) % N];

        w1r[k] = t1.re;
        w1i[k] = t1.im;
        w2r[k] = t2.re;
        w2i[k] = t2.im;
        w3r[k] = t3.re;
        w3i[k] = t3.im;
    }
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
    //--------------------------------------------------------------------------
    // 0) Basic validation
    //--------------------------------------------------------------------------
    if (data_length <= 0 || stride <= 0 || factor_index < 0)
    {
        fprintf(stderr,
                "Error: Invalid mixed-radix FFT inputs - data_length: %d, stride: %d, factor_index: %d\n",
                data_length, stride, factor_index);
        // exit
    }

    //--------------------------------------------------------------------------
    // 1) Base case: length-1 copy (only leaf)
    //    All other sizes recurse + stage-combine by the current radix.
    //--------------------------------------------------------------------------
    if (data_length == 1)
    {
        /**
         * @brief Base case for recursion: copy a single element.
         *
         * When the data length is 1, there’s no transformation needed—just copy the input
         * complex value to the output buffer. This marks the termination of the recursive
         * decomposition, as no further division is possible.
         */
        output_buffer[0] = input_buffer[0];
        return;
    }

    //--------------------------------------------------------------------------
    // 2) Current stage radix and subproblem geometry
    //--------------------------------------------------------------------------
    const int radix = fft_obj->factors[factor_index];
    const int sub_len = data_length / radix; // child FFT size
    const int next_stride = stride * radix;  // stride inside children

    //--------------------------------------------------------------------------
    // 3) Scratch planning for this stage
    //
    // We store the children outputs (lane-major) contiguously in scratch:
    //   sub_outputs[ i*sub_len + k ] = child i's output at index k  (i=0..radix-1, k=0..sub_len-1)
    //
    // If we don't have precomputed per-stage twiddles, we also need space for:
    //   stage_twiddles[(radix-1)*sub_len], laid out k-major:
    //     for each k, pack j=1..radix-1: tw[(radix-1)*k + (j-1)] = W_{data_length}^{j*k}
    //
    // NOTE: We pass the *same* scratch_offset to each child serially (no overlap in time),
    //       so siblings can reuse deeper scratch safely.
    //--------------------------------------------------------------------------
    fft_data *sub_outputs = fft_obj->scratch + scratch_offset;

    // Required scratch for this stage "frame":
    const int stage_outputs = radix * sub_len;
    const int stage_tw_count = (radix - 1) * sub_len;
    int need_this_stage = stage_outputs; // always need child outputs
    int twiddle_in_scratch = 0;

    fft_data *stage_tw = NULL;

    if (fft_obj->twiddle_factors != NULL && factor_index < fft_obj->num_precomputed_stages)
    {
        // Use precomputed table (k-major layout): points at this stage’s block.
        stage_tw = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
    }
    else
    {
        // We’ll generate twiddles into scratch right after sub_outputs.
        twiddle_in_scratch = 1;
        need_this_stage += stage_tw_count;
        if (scratch_offset + need_this_stage > fft_obj->max_scratch_size)
        {
            fprintf(stderr,
                    "Error: Scratch too small at stage (radix=%d, N=%d). Need %d, have %d @off=%d\n",
                    radix, data_length, need_this_stage,
                    fft_obj->max_scratch_size - scratch_offset, scratch_offset);
            // exit
        }
        stage_tw = sub_outputs + stage_outputs;
    }

    //--------------------------------------------------------------------------
    // 4) Recurse into the radix children (serially; reusing deeper scratch)
    //    Child i writes its sub-FFT into sub_outputs + i*sub_len
    //--------------------------------------------------------------------------
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
            scratch_offset /* reuse deeper scratch serially */);
    }

    //--------------------------------------------------------------------------
    // 5) Prepare per-stage twiddles if not precomputed
    //
    // Mapping from W_{data_length} to the global W_{fft_obj->n_fft} table:
    //   W_{data_length}^{p} == W_{n_fft}^{ p * (n_fft / data_length) }  (exact when data_length | n_fft)
    //
    // We fill k-major for SIMD-friendly access:
    //   for k in [0..sub_len)
    //     for j in [1..radix-1]
    //       stage_tw[ (radix-1)*k + (j-1) ] = W^{j*k}
    //--------------------------------------------------------------------------
    // 5) Prepare per-stage twiddles if not precomputed (k-major layout)
    if (twiddle_in_scratch)
    {
        const int nfft = fft_obj->n_fft;
        const int step = nfft / data_length; // exact for our construction
        for (int k = 0; k < sub_len; ++k)
        {
            const int base = (radix - 1) * k;
            for (int j = 1; j < radix; ++j)
            {
                const int p = (j * k) % data_length; // exponent in W_{data_length}
                const int idxN = (p * step) % nfft;  // map to global W table
                stage_tw[base + (j - 1)] = fft_obj->twiddles[idxN];
            }
        }
    }

    //--------------------------------------------------------------------------
    // 6) Stage combine per radix
    //    We write stage outputs to 'output_buffer' in canonical DIT order:
    //      output_buffer[m*sub_len + k] = X_m(k),  m=0..radix-1, k=0..sub_len-1
    //--------------------------------------------------------------------------
    else if (radix == 2)
    {
        const int sub_fft_length = data_length / 2; // N/2
        const int next_stride = 2 * stride;
        const int sub_fft_size = sub_fft_length;

        // Scratch requirement: outputs (2*N/2) + local twiddles (N/2) if not precomputed
        const int required_size = (fft_obj->twiddle_factors != NULL) ? (2 * sub_fft_size) : (3 * sub_fft_size);
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            /*
            fprintf(stderr, "Error: Scratch too small for radix-2 at off %d (need %d, have %d)
                            ",
                    scratch_offset,
                    required_size, fft_obj->max_scratch_size - scratch_offset);
            */
            return;
        }

        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors;

        if (fft_obj->twiddle_factors != NULL)
        {
            if (factor_index >= fft_obj->num_precomputed_stages)
            {
                /*
                fprintf(stderr, "Error: factor_index %d >= num_precomputed_stages %d (radix-2)
                                ",
                        factor_index,
                        fft_obj->num_precomputed_stages);
                */
                return;
            }
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + scratch_offset + 2 * sub_fft_size; // tail of this slice
        }

        // Child scratch slices (conservative split)
        const int child_scratch_per_branch = (fft_obj->twiddle_factors != NULL) ? (2 * (sub_fft_size / 2)) : (3 * (sub_fft_size / 2));
        if (child_scratch_per_branch * 2 > required_size)
        {
            return;
        }
        const int child_offset1 = scratch_offset;
        const int child_offset2 = scratch_offset + required_size / 2; // may be unaligned -> use unaligned loads/stores

        // Recurse (even, odd)
        mixed_radix_dit_rec(sub_fft_outputs, input_buffer, fft_obj, transform_sign,
                            sub_fft_length, next_stride, factor_index + 1, child_offset1);
        mixed_radix_dit_rec(sub_fft_outputs + sub_fft_size, input_buffer + stride, fft_obj, transform_sign,
                            sub_fft_length, next_stride, factor_index + 1, child_offset2);

        // Prepare local twiddles if not precomputed: require twiddles[m] = W_N^m, m=0..N-1
        if (fft_obj->twiddle_factors == NULL)
        {
            const int N = 2 * sub_fft_size;
            if (fft_obj->n_fft < N)
            {
                fprintf(stderr, "Error: twiddle array too small: need %d have %d", N, fft_obj->n_fft);
                return;
            }
            for (int k = 0; k < sub_fft_size; ++k)
            {
                const int idx = k; // W_N^k
                twiddle_factors[k].re = fft_obj->twiddles[idx].re;
                twiddle_factors[k].im = fft_obj->twiddles[idx].im;
            }
        }

        // AVX2 body: 2 complex numbers per iter
        // Negation masks for AoS lanes [re0, im0, re1, im1]
        const __m256d FLIP_RE = _mm256_set_pd(+0.0, -0.0, +0.0, -0.0); // negate real lanes (0,2)
        const __m256d FLIP_IM = _mm256_set_pd(-0.0, +0.0, -0.0, +0.0); // negate imag lanes (1,3)

        int k = 0;
        for (; k + 1 < sub_fft_size; k += 2)
        {
            // (Optional) prefetch a bit ahead – both re/im
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + sub_fft_size].im, _MM_HINT_T0);

            __m256d even = LOADU_PD(&sub_fft_outputs[k].re);               // [er0, ei0, er1, ei1]
            __m256d odd = LOADU_PD(&sub_fft_outputs[k + sub_fft_size].re); // [or0, oi0, or1, oi1]
            __m256d w = LOADU_PD(&twiddle_factors[k].re);                  // [wr0, wi0, wr1, wi1]

            __m256d tw = cmul_avx2_aos(odd, w); // odd * W^k

            // X0 = even + tw  (use addsub + flip real lanes of tw)
            __m256d tw_flip_re = _mm256_xor_pd(tw, FLIP_RE);
            __m256d x0 = _mm256_addsub_pd(even, tw_flip_re);

            // X1 = even - tw  (use addsub + flip imag lanes of tw == conj(tw))
            __m256d tw_conj = _mm256_xor_pd(tw, FLIP_IM);
            __m256d x1 = _mm256_addsub_pd(even, tw_conj);

            STOREU_PD(&output_buffer[k].re, x0);
            STOREU_PD(&output_buffer[k + sub_fft_size].re, x1);
        }

        // SSE2 tail: last single complex if odd length
        if (k < sub_fft_size)
        {
            const __m128d FLIP_RE_128 = _mm_set_pd(+0.0, -0.0); // negate real lane (low)
            const __m128d FLIP_IM_128 = _mm_set_pd(-0.0, +0.0); // negate imag lane (high)

            __m128d e = LOADU_SSE2(&sub_fft_outputs[k].re);                // [er, ei]
            __m128d o = LOADU_SSE2(&sub_fft_outputs[k + sub_fft_size].re); // [or, oi]
            __m128d w1 = LOADU_SSE2(&twiddle_factors[k].re);               // [wr, wi]

            __m128d tw = cmul_sse2_aos(o, w1);

            // X0 = e + tw
            __m128d tw_flip_re = _mm_xor_pd(tw, FLIP_RE_128);
            __m128d x0 = _mm_addsub_pd(e, tw_flip_re);

            // X1 = e - tw
            __m128d tw_conj = _mm_xor_pd(tw, FLIP_IM_128);
            __m128d x1 = _mm_addsub_pd(e, tw_conj);

            STOREU_SSE2(&output_buffer[k].re, x0);
            STOREU_SSE2(&output_buffer[k + sub_fft_size].re, x1);
        }
    }
    else if (radix == 3)
    {
        // --- sizes & stride ---
        const int sub_fft_length = data_length / 3; // N/3
        const int next_stride = 3 * stride;
        const int sub_fft_size = sub_fft_length;

        // --- scratch sizing: outputs + local twiddles if needed ---
        const int required_size =
            (fft_obj->twiddle_factors != NULL) ? (3 * sub_fft_size) : (5 * sub_fft_size);
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            /* fprintf(stderr, "..."); */ return;
        }

        // sub-FFT outputs live here: X0 | X1 | X2 (each sub_fft_size)
        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors;

        if (fft_obj->twiddle_factors != NULL)
        {
            if (factor_index >= fft_obj->num_precomputed_stages)
            { /* fprintf... */
                return;
            }
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            // local twiddles occupy tail of this slice: 2*sub_fft_size entries (W^k, W^{2k})
            twiddle_factors = fft_obj->scratch + scratch_offset + 3 * sub_fft_size;
        }

        // --- child scratch partition (conservative) ---
        const int child_need =
            (fft_obj->twiddle_factors != NULL) ? (3 * (sub_fft_size / 3)) : (5 * (sub_fft_size / 3));
        if (3 * child_need > required_size)
        { /* fprintf... */
            return;
        }

        // --- recurse lanes 0,1,2 ---
        for (int lane = 0; lane < 3; ++lane)
        {
            mixed_radix_dit_rec(
                sub_fft_outputs + lane * sub_fft_size,
                input_buffer + lane * stride,
                fft_obj, transform_sign,
                sub_fft_length, next_stride,
                factor_index + 1,
                scratch_offset + lane * (required_size / 3));
        }

        // --- prepare local twiddles if not precomputed ---
        if (fft_obj->twiddle_factors == NULL)
        {
            const int N = 3 * sub_fft_size; // stage length
            if (fft_obj->n_fft < N)
            { /* fprintf... */
                return;
            }
            for (int k = 0; k < sub_fft_size; ++k)
            {
                // w1 = W_N^k, w2 = W_N^{2k} (assumes twiddles[m] = W_N^m)
                twiddle_factors[2 * k + 0] = fft_obj->twiddles[k];
                twiddle_factors[2 * k + 1] = fft_obj->twiddles[(2 * k) % N];
            }
        }

        // --- constants ---
        const __m256d vhalf = _mm256_set1_pd(0.5);
        const __m256d vsign_s = _mm256_set1_pd((double)transform_sign * C3_SQRT3BY2);

        // --- AVX2 core: 2 complex per iter ---
        int k = 0;
        for (; k + 1 < sub_fft_size; k += 2)
        {
            _mm_prefetch((const char *)&output_buffer[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&output_buffer[k + 16 + sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&output_buffer[k + 16 + 2 * sub_fft_size].re, _MM_HINT_T0);

            // a,b,c: sub-FFTs X0,X1,X2 at k and k+1
            __m256d a = LOADU_PD(&sub_fft_outputs[k].re);
            __m256d b = LOADU_PD(&sub_fft_outputs[k + sub_fft_size].re);
            __m256d c = LOADU_PD(&sub_fft_outputs[k + 2 * sub_fft_size].re);

            // was: LOADU_PD(&twiddle_factors[2*k + 0].re)
            __m256d w1 = load2_aos(&twiddle_factors[2 * k + 0], &twiddle_factors[2 * (k + 1) + 0]);
            // was: LOADU_PD(&twiddle_factors[2*k + 1].re)
            __m256d w2 = load2_aos(&twiddle_factors[2 * k + 1], &twiddle_factors[2 * (k + 1) + 1]);

            __m256d b2 = cmul_avx2_aos(b, w1);
            __m256d c2 = cmul_avx2_aos(c, w2);

            __m256d sum = _mm256_add_pd(b2, c2);
            __m256d dif = _mm256_sub_pd(b2, c2);

            // X0 = a + sum
            __m256d x0 = _mm256_add_pd(a, sum);
            STOREU_PD(&output_buffer[k].re, x0);

            // t = a - 1/2*sum
            __m256d t = _mm256_sub_pd(a, _mm256_mul_pd(sum, vhalf));

            // rot = (sign*sqrt3/2) * rotate90(dif) ; rotate90(z) = (-Im, Re)
            __m256d swapped = _mm256_permute_pd(dif, 0b0101);        // [im0,re0, im1,re1]
            __m256d negmask = _mm256_set_pd(+0.0, -0.0, +0.0, -0.0); // negate lanes 0 and 2 of 'swapped'
            __m256d rot90 = _mm256_xor_pd(swapped, negmask);         // (-im, re)
            __m256d rot = _mm256_mul_pd(rot90, vsign_s);

            // X1 = t + rot ; X2 = t - rot
            __m256d x1 = _mm256_add_pd(t, rot);
            __m256d x2 = _mm256_sub_pd(t, rot);
            STOREU_PD(&output_buffer[k + sub_fft_size].re, x1);
            STOREU_PD(&output_buffer[k + 2 * sub_fft_size].re, x2);
        }

        // --- SSE2 tail: 1 complex ---
        if (k < sub_fft_size)
        {
            const __m128d vhalf128 = _mm_set1_pd(0.5);
            const __m128d vsign_s128 = _mm_set1_pd((double)transform_sign * C3_SQRT3BY2);

            __m128d a = LOADU_SSE2(&sub_fft_outputs[k].re);
            __m128d b = LOADU_SSE2(&sub_fft_outputs[k + sub_fft_size].re);
            __m128d c = LOADU_SSE2(&sub_fft_outputs[k + 2 * sub_fft_size].re);

            __m128d w1 = LOADU_SSE2(&twiddle_factors[2 * k + 0].re);
            __m128d w2 = LOADU_SSE2(&twiddle_factors[2 * k + 1].re);

            __m128d b2 = cmul_sse2_aos(b, w1);
            __m128d c2 = cmul_sse2_aos(c, w2);

            __m128d sum = _mm_add_pd(b2, c2);
            __m128d dif = _mm_sub_pd(b2, c2);

            __m128d x0 = _mm_add_pd(a, sum);
            STOREU_SSE2(&output_buffer[k].re, x0);

            __m128d t = _mm_sub_pd(a, _mm_mul_pd(sum, vhalf128));
            __m128d swp = _mm_shuffle_pd(dif, dif, 0b01);           // [im,re]
            __m128d rot90 = _mm_xor_pd(swp, _mm_set_pd(-0.0, 0.0)); // (-im, re)
            __m128d rot = _mm_mul_pd(rot90, vsign_s128);

            __m128d x1 = _mm_add_pd(t, rot);
            __m128d x2 = _mm_sub_pd(t, rot);
            STOREU_SSE2(&output_buffer[k + sub_fft_size].re, x1);
            STOREU_SSE2(&output_buffer[k + 2 * sub_fft_size].re, x2);
        }
    }
    else if (radix == 4)
    {
        // --- sizes & stride ---
        const int sub_fft_length = data_length / 4; // N/4
        const int next_stride = 4 * stride;
        const int sub_fft_size = sub_fft_length;

        // --- scratch sizing: outputs + local twiddles (if not precomputed) ---
        // outputs: 4 * sub_fft_size
        // local twiddles (W^k,W^{2k},W^{3k}): 3 * sub_fft_size (only if not precomputed)
        const int required_size =
            (fft_obj->twiddle_factors != NULL) ? (4 * sub_fft_size) : (7 * sub_fft_size);
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            /* fprintf(stderr, "radix-4: scratch too small\n"); */
            return;
        }

        // sub-FFT outputs: 4 contiguous blocks of size sub_fft_size (AoS)
        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;

        // per-stage twiddles (either precomputed or carved from scratch tail)
        fft_data *twiddle_factors = NULL;
        if (fft_obj->twiddle_factors != NULL)
        {
            if (factor_index >= fft_obj->num_precomputed_stages)
            { /* fprintf... */
                return;
            }
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + (scratch_offset + 4 * sub_fft_size); // after outputs
        }

        // --- child scratch partition (conservative, non-overlapping) ---
        const int child_need =
            (fft_obj->twiddle_factors != NULL) ? (4 * (sub_fft_size / 4)) : (7 * (sub_fft_size / 4));
        if (4 * child_need > required_size)
        {
            /* fprintf(stderr, "radix-4: child scratch exceeds parent allocation\n"); */
            return;
        }

        // --- recurse lanes 0..3 ---
        for (int lane = 0; lane < 4; ++lane)
        {
            mixed_radix_dit_rec(
                sub_fft_outputs + lane * sub_fft_size,
                input_buffer + lane * stride,
                fft_obj, transform_sign,
                sub_fft_length, next_stride,
                factor_index + 1,
                scratch_offset + lane * (required_size / 4));
        }

        // --- prepare local twiddles if not precomputed ---
        if (fft_obj->twiddle_factors == NULL)
        {
            const int N = 4 * sub_fft_size; // stage length
            if (fft_obj->n_fft < N)
            { /* fprintf... */
                return;
            }
            // Layout per k: [W^k, W^{2k}, W^{3k}]
            for (int k = 0; k < sub_fft_size; ++k)
            {
                twiddle_factors[3 * k + 0] = fft_obj->twiddles[(1 * k) % N];
                twiddle_factors[3 * k + 1] = fft_obj->twiddles[(2 * k) % N];
                twiddle_factors[3 * k + 2] = fft_obj->twiddles[(3 * k) % N];
            }
        }

        // --- constants ---
        const __m256d vsign = _mm256_set1_pd((double)transform_sign); // +1 forward, -1 inverse

        _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 0 * sub_fft_size].re, _MM_HINT_T0);
        _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 0 * sub_fft_size].im, _MM_HINT_T0);
        _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 1 * sub_fft_size].re, _MM_HINT_T0);
        _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 1 * sub_fft_size].im, _MM_HINT_T0);
        _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 2 * sub_fft_size].re, _MM_HINT_T0);
        _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 2 * sub_fft_size].im, _MM_HINT_T0);
        _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 3 * sub_fft_size].re, _MM_HINT_T0);
        _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 3 * sub_fft_size].im, _MM_HINT_T0);

        // --- AVX2 core: 2 complex per iter (AoS = [re0,im0,re1,im1]) ---
        // sub-FFT outputs are AoS blocks laid out as 4 contiguous lanes:
        //   lane0: sub_fft_outputs[0 .. sub_fft_size-1]
        //   lane1: sub_fft_outputs[sub_fft_size .. 2*sub_fft_size-1], etc.
        // twiddle_factors AoS per-k layout: [W^k, W^{2k}, W^{3k}] (each fft_data)
        int k = 0;
        for (; k + 3 < sub_fft_size; k += 4)
        {
            //----- A: gather 4 AoS -> SoA (re[4], im[4]) for each lane -----
            double aR[4], aI[4], bR[4], bI[4], cR[4], cI[4], dR[4], dI[4];

            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 0 * sub_fft_size], aR, aI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 1 * sub_fft_size], bR, bI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 2 * sub_fft_size], cR, cI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 3 * sub_fft_size], dR, dI);

            __m256d Ar = _mm256_loadu_pd(aR), Ai = _mm256_loadu_pd(aI);
            __m256d Br = _mm256_loadu_pd(bR), Bi = _mm256_loadu_pd(bI);
            __m256d Cr = _mm256_loadu_pd(cR), Ci = _mm256_loadu_pd(cI);
            __m256d Dr = _mm256_loadu_pd(dR), Di = _mm256_loadu_pd(dI);

            //----- B: make 4-wide SoA twiddles for k..k+3 (gather from AoS table) -----
            fft_data w1a[4], w2a[4], w3a[4];
            w1a[0] = twiddle_factors[3 * (k + 0) + 0];
            w1a[1] = twiddle_factors[3 * (k + 1) + 0];
            w1a[2] = twiddle_factors[3 * (k + 2) + 0];
            w1a[3] = twiddle_factors[3 * (k + 3) + 0];

            w2a[0] = twiddle_factors[3 * (k + 0) + 1];
            w2a[1] = twiddle_factors[3 * (k + 1) + 1];
            w2a[2] = twiddle_factors[3 * (k + 2) + 1];
            w2a[3] = twiddle_factors[3 * (k + 3) + 1];

            w3a[0] = twiddle_factors[3 * (k + 0) + 2];
            w3a[1] = twiddle_factors[3 * (k + 1) + 2];
            w3a[2] = twiddle_factors[3 * (k + 2) + 2];
            w3a[3] = twiddle_factors[3 * (k + 3) + 2];

            double w1R[4], w1I[4], w2R[4], w2I[4], w3R[4], w3I[4];
            deinterleave4_aos_to_soa(w1a, w1R, w1I);
            deinterleave4_aos_to_soa(w2a, w2R, w2I);
            deinterleave4_aos_to_soa(w3a, w3R, w3I);

            __m256d W1r = _mm256_loadu_pd(w1R), W1i = _mm256_loadu_pd(w1I);
            __m256d W2r = _mm256_loadu_pd(w2R), W2i = _mm256_loadu_pd(w2I);
            __m256d W3r = _mm256_loadu_pd(w3R), W3i = _mm256_loadu_pd(w3I);

            //----- C: twiddle application in SoA using cmul_soa_avx -----
            __m256d b2r, b2i, c2r, c2i, d2r, d2i;
            cmul_soa_avx(Br, Bi, W1r, W1i, &b2r, &b2i);
            cmul_soa_avx(Cr, Ci, W2r, W2i, &c2r, &c2i);
            cmul_soa_avx(Dr, Di, W3r, W3i, &d2r, &d2i);

            //----- D: radix-4 identities (SoA) -----
            __m256d sumBD_r = _mm256_add_pd(b2r, d2r);
            __m256d sumBD_i = _mm256_add_pd(b2i, d2i);
            __m256d difBD_r = _mm256_sub_pd(b2r, d2r);
            __m256d difBD_i = _mm256_sub_pd(b2i, d2i);

            __m256d a_plus_c_r = _mm256_add_pd(Ar, c2r);
            __m256d a_plus_c_i = _mm256_add_pd(Ai, c2i);
            __m256d a_minus_c_r = _mm256_sub_pd(Ar, c2r);
            __m256d a_minus_c_i = _mm256_sub_pd(Ai, c2i);

            // X0 = a + b2 + c2 + d2
            __m256d x0r = _mm256_add_pd(a_plus_c_r, sumBD_r);
            __m256d x0i = _mm256_add_pd(a_plus_c_i, sumBD_i);

            // X2 = a - b2 + c2 - d2
            __m256d x2r = _mm256_sub_pd(a_minus_c_r, sumBD_r);
            __m256d x2i = _mm256_sub_pd(a_minus_c_i, sumBD_i);

            // X1/X3: t = a - c2, u = b2 - d2, rot = sign * i * u
            __m256d tr = a_minus_c_r, ti = a_minus_c_i;
            __m256d rr, ri;                                            // rot
            rot90_soa_avx(difBD_r, difBD_i, transform_sign, &rr, &ri); // rr+ i*ri = sign * i * u

            __m256d x1r = _mm256_sub_pd(tr, rr);
            __m256d x1i = _mm256_sub_pd(ti, ri);
            __m256d x3r = _mm256_add_pd(tr, rr);
            __m256d x3i = _mm256_add_pd(ti, ri);

            //----- E: SoA -> AoS and store to output (4 results) -----
            double X0R[4], X0I[4], X1R[4], X1I[4], X2R[4], X2I[4], X3R[4], X3I[4];
            _mm256_storeu_pd(X0R, x0r);
            _mm256_storeu_pd(X0I, x0i);
            _mm256_storeu_pd(X1R, x1r);
            _mm256_storeu_pd(X1I, x1i);
            _mm256_storeu_pd(X2R, x2r);
            _mm256_storeu_pd(X2I, x2i);
            _mm256_storeu_pd(X3R, x3r);
            _mm256_storeu_pd(X3I, x3i);

            interleave4_soa_to_aos(X0R, X0I, &output_buffer[k + 0 * sub_fft_size]);
            interleave4_soa_to_aos(X1R, X1I, &output_buffer[k + 1 * sub_fft_size]);
            interleave4_soa_to_aos(X2R, X2I, &output_buffer[k + 2 * sub_fft_size]);
            interleave4_soa_to_aos(X3R, X3I, &output_buffer[k + 3 * sub_fft_size]);
        }

        // --- scalar tail for leftover 1..3 elements ---
        for (; k < sub_fft_size; ++k)
        {
            // Load AoS
            const fft_data a = sub_fft_outputs[k + 0 * sub_fft_size];
            const fft_data b = sub_fft_outputs[k + 1 * sub_fft_size];
            const fft_data c = sub_fft_outputs[k + 2 * sub_fft_size];
            const fft_data d = sub_fft_outputs[k + 3 * sub_fft_size];

            const fft_data w1 = twiddle_factors[3 * k + 0];
            const fft_data w2 = twiddle_factors[3 * k + 1];
            const fft_data w3 = twiddle_factors[3 * k + 2];

            // Twiddle
            double b2r = b.re * w1.re - b.im * w1.im;
            double b2i = b.re * w1.im + b.im * w1.re;
            double c2r = c.re * w2.re - c.im * w2.im;
            double c2i = c.re * w2.im + c.im * w2.re;
            double d2r = d.re * w3.re - d.im * w3.im;
            double d2i = d.re * w3.im + d.im * w3.re;

            // Combine
            double sumBD_r = b2r + d2r, sumBD_i = b2i + d2i;
            double difBD_r = b2r - d2r, difBD_i = b2i - d2i;

            double a_pc_r = a.re + c2r, a_pc_i = a.im + c2i;
            double a_mc_r = a.re - c2r, a_mc_i = a.im - c2i;

            fft_data x0 = {a_pc_r + sumBD_r, a_pc_i + sumBD_i};
            fft_data x2 = {a_mc_r - sumBD_r, a_mc_i - sumBD_i};

            // rot = sign * i * (difBD_r + i*difBD_i)
            double rr = (transform_sign == 1) ? -difBD_i : difBD_i;
            double ri = (transform_sign == 1) ? difBD_r : -difBD_r;

            fft_data x1 = {a_mc_r - rr, a_mc_i - ri};
            fft_data x3 = {a_mc_r + rr, a_mc_i + ri};

            output_buffer[k + 0 * sub_fft_size] = x0;
            output_buffer[k + 1 * sub_fft_size] = x1;
            output_buffer[k + 2 * sub_fft_size] = x2;
            output_buffer[k + 3 * sub_fft_size] = x3;
        }
    }
    else if (radix == 5)
    {
        // --- stage sizing ---
        const int sub_fft_length = data_length / 5; // N/5
        const int next_stride = 5 * stride;
        const int sub_fft_size = sub_fft_length;

        // outputs (5*N/5) + local twiddles (4*N/5) if not precomputed
        const int required_size =
            (fft_obj->twiddle_factors != NULL) ? (5 * sub_fft_size) : (9 * sub_fft_size);
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            return; // scratch too small
        }

        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors;
        if (fft_obj->twiddle_factors != NULL)
        {
            if (factor_index >= fft_obj->num_precomputed_stages)
                return;
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + scratch_offset + 5 * sub_fft_size; // tail
        }

        // --- recurse 5 children ---
        for (int i = 0; i < 5; ++i)
        {
            mixed_radix_dit_rec(sub_fft_outputs + i * sub_fft_size,
                                input_buffer + i * stride,
                                fft_obj, transform_sign,
                                sub_fft_length, next_stride,
                                factor_index + 1,
                                scratch_offset + i * (required_size / 5));
        }

        // --- local twiddle prep if not precomputed ---
        if (fft_obj->twiddle_factors == NULL)
        {
            const int N = 5 * sub_fft_size;
            if (fft_obj->n_fft < N)
                return;
            for (int k = 0; k < sub_fft_size; ++k)
            {
                twiddle_factors[4 * k + 0] = fft_obj->twiddles[(1 * k) % N]; // W_N^{k}
                twiddle_factors[4 * k + 1] = fft_obj->twiddles[(2 * k) % N]; // W_N^{2k}
                twiddle_factors[4 * k + 2] = fft_obj->twiddles[(3 * k) % N]; // W_N^{3k}
                twiddle_factors[4 * k + 3] = fft_obj->twiddles[(4 * k) % N]; // W_N^{4k}
            }
        }

        // --- SoA AVX2 core: 4 k's per iter ---
#if defined(__AVX2__)
        const __m256d vc1 = _mm256_set1_pd(C5_1);
        const __m256d vc2 = _mm256_set1_pd(C5_2);
        const __m256d vs1 = _mm256_set1_pd(S5_1);
        const __m256d vs2 = _mm256_set1_pd(S5_2);

        int k = 0;
        for (; k + 3 < sub_fft_size; k += 4)
        {
            // modest prefetch of AoS lanes ahead
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8].im, _MM_HINT_T0);

            // A) AoS -> SoA for 5 lanes, 4 points (k..k+3)
            double aR[4], aI[4], bR[4], bI[4], cR[4], cI[4], dR[4], dI[4], eR[4], eI[4];
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 0 * sub_fft_size], aR, aI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 1 * sub_fft_size], bR, bI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 2 * sub_fft_size], cR, cI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 3 * sub_fft_size], dR, dI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 4 * sub_fft_size], eR, eI);

            __m256d Ar = _mm256_loadu_pd(aR), Ai = _mm256_loadu_pd(aI);
            __m256d Br = _mm256_loadu_pd(bR), Bi = _mm256_loadu_pd(bI);
            __m256d Cr = _mm256_loadu_pd(cR), Ci = _mm256_loadu_pd(cI);
            __m256d Dr = _mm256_loadu_pd(dR), Di = _mm256_loadu_pd(dI);
            __m256d Er = _mm256_loadu_pd(eR), Ei = _mm256_loadu_pd(eI);

            // B) Gather 4 twiddles per j=1..4 across k..k+3, AoS->SoA
            fft_data w1a[4], w2a[4], w3a[4], w4a[4];
            w1a[0] = twiddle_factors[4 * (k + 0) + 0];
            w1a[1] = twiddle_factors[4 * (k + 1) + 0];
            w1a[2] = twiddle_factors[4 * (k + 2) + 0];
            w1a[3] = twiddle_factors[4 * (k + 3) + 0];

            w2a[0] = twiddle_factors[4 * (k + 0) + 1];
            w2a[1] = twiddle_factors[4 * (k + 1) + 1];
            w2a[2] = twiddle_factors[4 * (k + 2) + 1];
            w2a[3] = twiddle_factors[4 * (k + 3) + 1];

            w3a[0] = twiddle_factors[4 * (k + 0) + 2];
            w3a[1] = twiddle_factors[4 * (k + 1) + 2];
            w3a[2] = twiddle_factors[4 * (k + 2) + 2];
            w3a[3] = twiddle_factors[4 * (k + 3) + 2];

            w4a[0] = twiddle_factors[4 * (k + 0) + 3];
            w4a[1] = twiddle_factors[4 * (k + 1) + 3];
            w4a[2] = twiddle_factors[4 * (k + 2) + 3];
            w4a[3] = twiddle_factors[4 * (k + 3) + 3];

            double w1R[4], w1I[4], w2R[4], w2I[4], w3R[4], w3I[4], w4R[4], w4I[4];
            deinterleave4_aos_to_soa(w1a, w1R, w1I);
            deinterleave4_aos_to_soa(w2a, w2R, w2I);
            deinterleave4_aos_to_soa(w3a, w3R, w3I);
            deinterleave4_aos_to_soa(w4a, w4R, w4I);

            __m256d W1r = _mm256_loadu_pd(w1R), W1i = _mm256_loadu_pd(w1I);
            __m256d W2r = _mm256_loadu_pd(w2R), W2i = _mm256_loadu_pd(w2I);
            __m256d W3r = _mm256_loadu_pd(w3R), W3i = _mm256_loadu_pd(w3I);
            __m256d W4r = _mm256_loadu_pd(w4R), W4i = _mm256_loadu_pd(w4I);

            // C) Apply twiddles in SoA
            __m256d b2r, b2i, c2r, c2i, d2r, d2i, e2r, e2i;
            cmul_soa_avx(Br, Bi, W1r, W1i, &b2r, &b2i);
            cmul_soa_avx(Cr, Ci, W2r, W2i, &c2r, &c2i);
            cmul_soa_avx(Dr, Di, W3r, W3i, &d2r, &d2i);
            cmul_soa_avx(Er, Ei, W4r, W4i, &e2r, &e2i);

            // D) Radix-5 SoA identities
            __m256d t0r = _mm256_add_pd(b2r, e2r); // b+e
            __m256d t0i = _mm256_add_pd(b2i, e2i);
            __m256d t1r = _mm256_add_pd(c2r, d2r); // c+d
            __m256d t1i = _mm256_add_pd(c2i, d2i);
            __m256d t2r = _mm256_sub_pd(b2r, e2r); // b-e
            __m256d t2i = _mm256_sub_pd(b2i, e2i);
            __m256d t3r = _mm256_sub_pd(c2r, d2r); // c-d
            __m256d t3i = _mm256_sub_pd(c2i, d2i);

            // X0 = a + t0 + t1
            __m256d x0r = _mm256_add_pd(Ar, _mm256_add_pd(t0r, t1r));
            __m256d x0i = _mm256_add_pd(Ai, _mm256_add_pd(t0i, t1i));

            // base1 = s1*(b-e) + s2*(c-d)
            __m256d b1r = _mm256_add_pd(_mm256_mul_pd(vs1, t2r), _mm256_mul_pd(vs2, t3r));
            __m256d b1i = _mm256_add_pd(_mm256_mul_pd(vs1, t2i), _mm256_mul_pd(vs2, t3i));
            // tmp1  = c1*(b+e) + c2*(c+d)
            __m256d tmp1r = _mm256_add_pd(_mm256_mul_pd(vc1, t0r), _mm256_mul_pd(vc2, t1r));
            __m256d tmp1i = _mm256_add_pd(_mm256_mul_pd(vc1, t0i), _mm256_mul_pd(vc2, t1i));
            // rot1 = sign * i * base1
            __m256d r1r, r1i;
            rot90_soa_avx(b1r, b1i, transform_sign, &r1r, &r1i);
            // a_pt1 = a + tmp1
            __m256d a1r = _mm256_add_pd(Ar, tmp1r);
            __m256d a1i = _mm256_add_pd(Ai, tmp1i);
            // X1 = a_pt1 + rot1 ; X4 = a_pt1 - rot1
            __m256d x1r = _mm256_add_pd(a1r, r1r);
            __m256d x1i = _mm256_add_pd(a1i, r1i);
            __m256d x4r = _mm256_sub_pd(a1r, r1r);
            __m256d x4i = _mm256_sub_pd(a1i, r1i);

            // base2 = s2*(b-e) - s1*(c-d)
            __m256d b2r_ = _mm256_sub_pd(_mm256_mul_pd(vs2, t2r), _mm256_mul_pd(vs1, t3r));
            __m256d b2i_ = _mm256_sub_pd(_mm256_mul_pd(vs2, t2i), _mm256_mul_pd(vs1, t3i));
            // tmp2  = c2*(b+e) + c1*(c+d)
            __m256d tmp2r = _mm256_add_pd(_mm256_mul_pd(vc2, t0r), _mm256_mul_pd(vc1, t1r));
            __m256d tmp2i = _mm256_add_pd(_mm256_mul_pd(vc2, t0i), _mm256_mul_pd(vc1, t1i));
            // rot2 = sign * i * base2
            __m256d r2r, r2i;
            rot90_soa_avx(b2r_, b2i_, transform_sign, &r2r, &r2i);
            // a_pt2 = a + tmp2
            __m256d a2r = _mm256_add_pd(Ar, tmp2r);
            __m256d a2i = _mm256_add_pd(Ai, tmp2i);
            // X2 = a_pt2 + rot2 ; X3 = a_pt2 - rot2
            __m256d x2r = _mm256_add_pd(a2r, r2r);
            __m256d x2i = _mm256_add_pd(a2i, r2i);
            __m256d x3r = _mm256_sub_pd(a2r, r2r);
            __m256d x3i = _mm256_sub_pd(a2i, r2i);

            // E) SoA -> AoS stores (4 points for each lane)
            double X0R[4], X0I[4], X1R[4], X1I[4], X2R[4], X2I[4], X3R[4], X3I[4], X4R[4], X4I[4];
            _mm256_storeu_pd(X0R, x0r);
            _mm256_storeu_pd(X0I, x0i);
            _mm256_storeu_pd(X1R, x1r);
            _mm256_storeu_pd(X1I, x1i);
            _mm256_storeu_pd(X2R, x2r);
            _mm256_storeu_pd(X2I, x2i);
            _mm256_storeu_pd(X3R, x3r);
            _mm256_storeu_pd(X3I, x3i);
            _mm256_storeu_pd(X4R, x4r);
            _mm256_storeu_pd(X4I, x4i);

            interleave4_soa_to_aos(X0R, X0I, &output_buffer[k + 0 * sub_fft_size]);
            interleave4_soa_to_aos(X1R, X1I, &output_buffer[k + 1 * sub_fft_size]);
            interleave4_soa_to_aos(X2R, X2I, &output_buffer[k + 2 * sub_fft_size]);
            interleave4_soa_to_aos(X3R, X3I, &output_buffer[k + 3 * sub_fft_size]);
            interleave4_soa_to_aos(X4R, X4I, &output_buffer[k + 4 * sub_fft_size]);
        }
#endif // __AVX2__

        // --- scalar tail: leftover 0..3 elements (and whole path if no AVX2) ---
#if defined(__AVX2__)
        int tail_k = k;
#else
        int tail_k = 0;
#endif
        for (int kk = tail_k; kk < sub_fft_size; ++kk)
        {
            // Load AoS values for this k
            const fft_data a = sub_fft_outputs[kk + 0 * sub_fft_size];
            const fft_data b = sub_fft_outputs[kk + 1 * sub_fft_size];
            const fft_data c = sub_fft_outputs[kk + 2 * sub_fft_size];
            const fft_data d = sub_fft_outputs[kk + 3 * sub_fft_size];
            const fft_data e = sub_fft_outputs[kk + 4 * sub_fft_size];

            const fft_data w1 = twiddle_factors[4 * kk + 0];
            const fft_data w2 = twiddle_factors[4 * kk + 1];
            const fft_data w3 = twiddle_factors[4 * kk + 2];
            const fft_data w4 = twiddle_factors[4 * kk + 3];

            // Twiddle
            double b2r = b.re * w1.re - b.im * w1.im;
            double b2i = b.re * w1.im + b.im * w1.re;
            double c2r = c.re * w2.re - c.im * w2.im;
            double c2i = c.re * w2.im + c.im * w2.re;
            double d2r = d.re * w3.re - d.im * w3.im;
            double d2i = d.re * w3.im + d.im * w3.re;
            double e2r = e.re * w4.re - e.im * w4.im;
            double e2i = e.re * w4.im + e.im * w4.re;

            double t0r = b2r + e2r, t0i = b2i + e2i;
            double t1r = c2r + d2r, t1i = c2i + d2i;
            double t2r = b2r - e2r, t2i = b2i - e2i;
            double t3r = c2r - d2r, t3i = c2i - d2i;

            // X0
            fft_data X0 = {a.re + t0r + t1r, a.im + t0i + t1i};

            // base1, tmp1
            double base1r = S5_1 * t2r + S5_2 * t3r;
            double base1i = S5_1 * t2i + S5_2 * t3i;
            double tmp1r = C5_1 * t0r + C5_2 * t1r;
            double tmp1i = C5_1 * t0i + C5_2 * t1i;
            // rot1 = sign * i * base1
            double r1r = (transform_sign == 1) ? -base1i : base1i;
            double r1i = (transform_sign == 1) ? base1r : -base1r;
            fft_data a1 = {a.re + tmp1r, a.im + tmp1i};
            fft_data X1 = {a1.re + r1r, a1.im + r1i};
            fft_data X4 = {a1.re - r1r, a1.im - r1i};

            // base2, tmp2
            double base2r = S5_2 * t2r - S5_1 * t3r;
            double base2i = S5_2 * t2i - S5_1 * t3i;
            double tmp2r = C5_2 * t0r + C5_1 * t1r;
            double tmp2i = C5_2 * t0i + C5_1 * t1i;
            // rot2 = sign * i * base2
            double r2r = (transform_sign == 1) ? -base2i : base2i;
            double r2i = (transform_sign == 1) ? base2r : -base2r;
            fft_data a2 = {a.re + tmp2r, a.im + tmp2i};
            fft_data X2 = {a2.re + r2r, a2.im + r2i};
            fft_data X3 = {a2.re - r2r, a2.im - r2i};

            output_buffer[kk + 0 * sub_fft_size] = X0;
            output_buffer[kk + 1 * sub_fft_size] = X1;
            output_buffer[kk + 2 * sub_fft_size] = X2;
            output_buffer[kk + 3 * sub_fft_size] = X3;
            output_buffer[kk + 4 * sub_fft_size] = X4;
        }
    }
    else if (radix == 7)
    {
        // --- stage sizing ---
        const int sub_fft_length = data_length / 7; // N/7
        const int next_stride = 7 * stride;
        const int sub_fft_size = sub_fft_length;

        // outputs (7 * N/7) + local twiddles (6 * N/7) if not precomputed
        const int required_size =
            (fft_obj->twiddle_factors != NULL) ? (7 * sub_fft_size) : (13 * sub_fft_size);
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
            return;

        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors = NULL;
        if (fft_obj->twiddle_factors != NULL)
        {
            if (factor_index >= fft_obj->num_precomputed_stages)
                return;
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + scratch_offset + 7 * sub_fft_size; // tail
        }

        // --- recurse lanes 0..6 ---
        for (int lane = 0; lane < 7; ++lane)
        {
            mixed_radix_dit_rec(
                sub_fft_outputs + lane * sub_fft_size,
                input_buffer + lane * stride,
                fft_obj, transform_sign,
                sub_fft_length, next_stride,
                factor_index + 1,
                scratch_offset + lane * (required_size / 7));
        }

        // --- local twiddle prep if not precomputed ---
        if (fft_obj->twiddle_factors == NULL)
        {
            const int N = 7 * sub_fft_size; // stage length
            if (fft_obj->n_fft < N)
                return;

            // Layout per k: [W^{1k}, W^{2k}, ..., W^{6k}]
            for (int k = 0; k < sub_fft_size; ++k)
            {
                twiddle_factors[6 * k + 0] = fft_obj->twiddles[(1 * k) % N];
                twiddle_factors[6 * k + 1] = fft_obj->twiddles[(2 * k) % N];
                twiddle_factors[6 * k + 2] = fft_obj->twiddles[(3 * k) % N];
                twiddle_factors[6 * k + 3] = fft_obj->twiddles[(4 * k) % N];
                twiddle_factors[6 * k + 4] = fft_obj->twiddles[(5 * k) % N];
                twiddle_factors[6 * k + 5] = fft_obj->twiddles[(6 * k) % N];
            }
        }

        // =========================
        // SoA AVX2 core (k += 4)
        // =========================
#if defined(__AVX2__)
        const __m256d vc1 = _mm256_set1_pd(C1);
        const __m256d vc2 = _mm256_set1_pd(C2);
        const __m256d vc3 = _mm256_set1_pd(C3);
        const __m256d vs1 = _mm256_set1_pd(S1);
        const __m256d vs2 = _mm256_set1_pd(S2);
        const __m256d vs3 = _mm256_set1_pd(S3);

        int k = 0;
        for (; k + 3 < sub_fft_size; k += 4)
        {
            // modest prefetch (~64B ahead) for all 7 lanes, both re/im
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 0 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 0 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 1 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 1 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 2 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 2 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 3 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 3 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 4 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 4 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 5 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 5 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 6 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 6 * sub_fft_size].im, _MM_HINT_T0);

            // ----- A: AoS -> SoA for 7 lanes, 4 points each -----
            double aR[4], aI[4], bR[4], bI[4], cR[4], cI[4], dR[4], dI[4];
            double eR[4], eI[4], fR[4], fI[4], gR[4], gI[4];

            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 0 * sub_fft_size], aR, aI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 1 * sub_fft_size], bR, bI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 2 * sub_fft_size], cR, cI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 3 * sub_fft_size], dR, dI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 4 * sub_fft_size], eR, eI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 5 * sub_fft_size], fR, fI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 6 * sub_fft_size], gR, gI);

            __m256d Ar = _mm256_loadu_pd(aR), Ai = _mm256_loadu_pd(aI);
            __m256d Br = _mm256_loadu_pd(bR), Bi = _mm256_loadu_pd(bI);
            __m256d Cr = _mm256_loadu_pd(cR), Ci = _mm256_loadu_pd(cI);
            __m256d Dr = _mm256_loadu_pd(dR), Di = _mm256_loadu_pd(dI);
            __m256d Er = _mm256_loadu_pd(eR), Ei = _mm256_loadu_pd(eI);
            __m256d Fr = _mm256_loadu_pd(fR), Fi = _mm256_loadu_pd(fI);
            __m256d Gr = _mm256_loadu_pd(gR), Gi = _mm256_loadu_pd(gI);

            // ----- B: build SoA twiddles for k..k+3 (powers 1..6) -----
            fft_data w1a[4], w2a[4], w3a[4], w4a[4], w5a[4], w6a[4];
            for (int p = 0; p < 4; ++p)
            {
                w1a[p] = twiddle_factors[6 * (k + p) + 0];
                w2a[p] = twiddle_factors[6 * (k + p) + 1];
                w3a[p] = twiddle_factors[6 * (k + p) + 2];
                w4a[p] = twiddle_factors[6 * (k + p) + 3];
                w5a[p] = twiddle_factors[6 * (k + p) + 4];
                w6a[p] = twiddle_factors[6 * (k + p) + 5];
            }

            double w1R[4], w1I[4], w2R[4], w2I[4], w3R[4], w3I[4];
            double w4R[4], w4I[4], w5R[4], w5I[4], w6R[4], w6I[4];
            deinterleave4_aos_to_soa(w1a, w1R, w1I);
            deinterleave4_aos_to_soa(w2a, w2R, w2I);
            deinterleave4_aos_to_soa(w3a, w3R, w3I);
            deinterleave4_aos_to_soa(w4a, w4R, w4I);
            deinterleave4_aos_to_soa(w5a, w5R, w5I);
            deinterleave4_aos_to_soa(w6a, w6R, w6I);

            __m256d W1r = _mm256_loadu_pd(w1R), W1i = _mm256_loadu_pd(w1I);
            __m256d W2r = _mm256_loadu_pd(w2R), W2i = _mm256_loadu_pd(w2I);
            __m256d W3r = _mm256_loadu_pd(w3R), W3i = _mm256_loadu_pd(w3I);
            __m256d W4r = _mm256_loadu_pd(w4R), W4i = _mm256_loadu_pd(w4I);
            __m256d W5r = _mm256_loadu_pd(w5R), W5i = _mm256_loadu_pd(w5I);
            __m256d W6r = _mm256_loadu_pd(w6R), W6i = _mm256_loadu_pd(w6I);

            // ----- C: twiddle application in SoA -----
            __m256d b2r, b2i, c2r, c2i, d2r, d2i, e2r, e2i, f2r, f2i, g2r, g2i;
            cmul_soa_avx(Br, Bi, W1r, W1i, &b2r, &b2i);
            cmul_soa_avx(Cr, Ci, W2r, W2i, &c2r, &c2i);
            cmul_soa_avx(Dr, Di, W3r, W3i, &d2r, &d2i);
            cmul_soa_avx(Er, Ei, W4r, W4i, &e2r, &e2i);
            cmul_soa_avx(Fr, Fi, W5r, W5i, &f2r, &f2i);
            cmul_soa_avx(Gr, Gi, W6r, W6i, &g2r, &g2i);

            // ----- D: radix-7 identities (SoA) -----
            // t0 = b+g, t1 = c+f, t2 = d+e
            __m256d t0r = _mm256_add_pd(b2r, g2r), t0i = _mm256_add_pd(b2i, g2i);
            __m256d t1r = _mm256_add_pd(c2r, f2r), t1i = _mm256_add_pd(c2i, f2i);
            __m256d t2r = _mm256_add_pd(d2r, e2r), t2i = _mm256_add_pd(d2i, e2i);
            // t3 = b-g, t4 = c-f, t5 = d-e
            __m256d t3r = _mm256_sub_pd(b2r, g2r), t3i = _mm256_sub_pd(b2i, g2i);
            __m256d t4r = _mm256_sub_pd(c2r, f2r), t4i = _mm256_sub_pd(c2i, f2i);
            __m256d t5r = _mm256_sub_pd(d2r, e2r), t5i = _mm256_sub_pd(d2i, e2i);

            // X0 = a + t0 + t1 + t2
            __m256d x0r = _mm256_add_pd(Ar, _mm256_add_pd(_mm256_add_pd(t0r, t1r), t2r));
            __m256d x0i = _mm256_add_pd(Ai, _mm256_add_pd(_mm256_add_pd(t0i, t1i), t2i));

            // helper: rot90_soa_avx(base, sign) -> (rr,ri) = sign * i * base
            __m256d base1r = _mm256_add_pd(_mm256_mul_pd(vs1, t3r),
                                           _mm256_add_pd(_mm256_mul_pd(vs2, t4r), _mm256_mul_pd(vs3, t5r)));
            __m256d base1i = _mm256_add_pd(_mm256_mul_pd(vs1, t3i),
                                           _mm256_add_pd(_mm256_mul_pd(vs2, t4i), _mm256_mul_pd(vs3, t5i)));

            __m256d base2r = _mm256_add_pd(_mm256_mul_pd(vs2, t3r),
                                           _mm256_add_pd(_mm256_mul_pd(vs3, t4r), _mm256_mul_pd(vs1, t5r)));
            __m256d base2i = _mm256_add_pd(_mm256_mul_pd(vs2, t3i),
                                           _mm256_add_pd(_mm256_mul_pd(vs3, t4i), _mm256_mul_pd(vs1, t5i)));

            __m256d base3r = _mm256_add_pd(_mm256_mul_pd(vs3, t3r),
                                           _mm256_add_pd(_mm256_mul_pd(vs1, t4r), _mm256_mul_pd(vs2, t5r)));
            __m256d base3i = _mm256_add_pd(_mm256_mul_pd(vs3, t3i),
                                           _mm256_add_pd(_mm256_mul_pd(vs1, t4i), _mm256_mul_pd(vs2, t5i)));

            __m256d rr1, ri1, rr2, ri2, rr3, ri3;
            rot90_soa_avx(base1r, base1i, transform_sign, &rr1, &ri1);
            rot90_soa_avx(base2r, base2i, transform_sign, &rr2, &ri2);
            rot90_soa_avx(base3r, base3i, transform_sign, &rr3, &ri3);

            // tmp for real parts: a + (c1 t0 + c2 t1 + c3 t2)
            __m256d tmp1r = _mm256_add_pd(Ar, _mm256_add_pd(_mm256_mul_pd(vc1, t0r),
                                                            _mm256_add_pd(_mm256_mul_pd(vc2, t1r), _mm256_mul_pd(vc3, t2r))));
            __m256d tmp1i = _mm256_add_pd(Ai, _mm256_add_pd(_mm256_mul_pd(vc1, t0i),
                                                            _mm256_add_pd(_mm256_mul_pd(vc2, t1i), _mm256_mul_pd(vc3, t2i))));

            __m256d tmp2r = _mm256_add_pd(Ar, _mm256_add_pd(_mm256_mul_pd(vc2, t0r),
                                                            _mm256_add_pd(_mm256_mul_pd(vc3, t1r), _mm256_mul_pd(vc1, t2r))));
            __m256d tmp2i = _mm256_add_pd(Ai, _mm256_add_pd(_mm256_mul_pd(vc2, t0i),
                                                            _mm256_add_pd(_mm256_mul_pd(vc3, t1i), _mm256_mul_pd(vc1, t2i))));

            __m256d tmp3r = _mm256_add_pd(Ar, _mm256_add_pd(_mm256_mul_pd(vc3, t0r),
                                                            _mm256_add_pd(_mm256_mul_pd(vc1, t1r), _mm256_mul_pd(vc2, t2r))));
            __m256d tmp3i = _mm256_add_pd(Ai, _mm256_add_pd(_mm256_mul_pd(vc3, t0i),
                                                            _mm256_add_pd(_mm256_mul_pd(vc1, t1i), _mm256_mul_pd(vc2, t2i))));

            // X1/X6, X2/X5, X3/X4
            __m256d x1r = _mm256_add_pd(tmp1r, rr1), x1i = _mm256_add_pd(tmp1i, ri1);
            __m256d x6r = _mm256_sub_pd(tmp1r, rr1), x6i = _mm256_sub_pd(tmp1i, ri1);

            __m256d x2r = _mm256_add_pd(tmp2r, rr2), x2i = _mm256_add_pd(tmp2i, ri2);
            __m256d x5r = _mm256_sub_pd(tmp2r, rr2), x5i = _mm256_sub_pd(tmp2i, ri2);

            __m256d x3r = _mm256_add_pd(tmp3r, rr3), x3i = _mm256_add_pd(tmp3i, ri3);
            __m256d x4r = _mm256_sub_pd(tmp3r, rr3), x4i = _mm256_sub_pd(tmp3i, ri3);

            // ----- E: SoA -> AoS stores (4 results per lane) -----
            double X0R[4], X0I[4], X1R[4], X1I[4], X2R[4], X2I[4];
            double X3R[4], X3I[4], X4R[4], X4I[4], X5R[4], X5I[4], X6R[4], X6I[4];

            _mm256_storeu_pd(X0R, x0r);
            _mm256_storeu_pd(X0I, x0i);
            _mm256_storeu_pd(X1R, x1r);
            _mm256_storeu_pd(X1I, x1i);
            _mm256_storeu_pd(X2R, x2r);
            _mm256_storeu_pd(X2I, x2i);
            _mm256_storeu_pd(X3R, x3r);
            _mm256_storeu_pd(X3I, x3i);
            _mm256_storeu_pd(X4R, x4r);
            _mm256_storeu_pd(X4I, x4i);
            _mm256_storeu_pd(X5R, x5r);
            _mm256_storeu_pd(X5I, x5i);
            _mm256_storeu_pd(X6R, x6r);
            _mm256_storeu_pd(X6I, x6i);

            interleave4_soa_to_aos(X0R, X0I, &output_buffer[k + 0 * sub_fft_size]);
            interleave4_soa_to_aos(X1R, X1I, &output_buffer[k + 1 * sub_fft_size]);
            interleave4_soa_to_aos(X2R, X2I, &output_buffer[k + 2 * sub_fft_size]);
            interleave4_soa_to_aos(X3R, X3I, &output_buffer[k + 3 * sub_fft_size]);
            interleave4_soa_to_aos(X4R, X4I, &output_buffer[k + 4 * sub_fft_size]);
            interleave4_soa_to_aos(X5R, X5I, &output_buffer[k + 5 * sub_fft_size]);
            interleave4_soa_to_aos(X6R, X6I, &output_buffer[k + 6 * sub_fft_size]);
        }
#else
        int k = 0;
#endif // __AVX2__

        // =========================
        // scalar AoS tail (1..3)
        // =========================
        for (; k < sub_fft_size; ++k)
        {
            const fft_data a = sub_fft_outputs[k + 0 * sub_fft_size];
            const fft_data b = sub_fft_outputs[k + 1 * sub_fft_size];
            const fft_data c = sub_fft_outputs[k + 2 * sub_fft_size];
            const fft_data d = sub_fft_outputs[k + 3 * sub_fft_size];
            const fft_data e = sub_fft_outputs[k + 4 * sub_fft_size];
            const fft_data f = sub_fft_outputs[k + 5 * sub_fft_size];
            const fft_data g = sub_fft_outputs[k + 6 * sub_fft_size];

            const fft_data w1 = twiddle_factors[6 * k + 0];
            const fft_data w2 = twiddle_factors[6 * k + 1];
            const fft_data w3 = twiddle_factors[6 * k + 2];
            const fft_data w4 = twiddle_factors[6 * k + 3];
            const fft_data w5 = twiddle_factors[6 * k + 4];
            const fft_data w6 = twiddle_factors[6 * k + 5];

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

            // X0
            fft_data X0 = {a.re + (t0r + t1r + t2r), a.im + (t0i + t1i + t2i)};

            // base rotations: sign * i * base
            double r1r = (transform_sign == 1 ? -(S1 * t3i + S2 * t4i + S3 * t5i)
                                              : (S1 * t3i + S2 * t4i + S3 * t5i));
            double r1i = (transform_sign == 1 ? (S1 * t3r + S2 * t4r + S3 * t5r)
                                              : -(S1 * t3r + S2 * t4r + S3 * t5r));

            double r2r = (transform_sign == 1 ? -(S2 * t3i + S3 * t4i + S1 * t5i)
                                              : (S2 * t3i + S3 * t4i + S1 * t5i));
            double r2i = (transform_sign == 1 ? (S2 * t3r + S3 * t4r + S1 * t5r)
                                              : -(S2 * t3r + S3 * t4r + S1 * t5r));

            double r3r = (transform_sign == 1 ? -(S3 * t3i + S1 * t4i + S2 * t5i)
                                              : (S3 * t3i + S1 * t4i + S2 * t5i));
            double r3i = (transform_sign == 1 ? (S3 * t3r + S1 * t4r + S2 * t5r)
                                              : -(S3 * t3r + S1 * t4r + S2 * t5r));

            double tmp1r = a.re + (C1 * t0r + C2 * t1r + C3 * t2r);
            double tmp1i = a.im + (C1 * t0i + C2 * t1i + C3 * t2i);

            double tmp2r = a.re + (C2 * t0r + C3 * t1r + C1 * t2r);
            double tmp2i = a.im + (C2 * t0i + C3 * t1i + C1 * t2i);

            double tmp3r = a.re + (C3 * t0r + C1 * t1r + C2 * t2r);
            double tmp3i = a.im + (C3 * t0i + C1 * t1i + C2 * t2i);

            fft_data X1 = {tmp1r + r1r, tmp1i + r1i};
            fft_data X6 = {tmp1r - r1r, tmp1i - r1i};

            fft_data X2 = {tmp2r + r2r, tmp2i + r2i};
            fft_data X5 = {tmp2r - r2r, tmp2i - r2i};

            fft_data X3 = {tmp3r + r3r, tmp3i + r3i};
            fft_data X4 = {tmp3r - r3r, tmp3i - r3i};

            output_buffer[k + 0 * sub_fft_size] = X0;
            output_buffer[k + 1 * sub_fft_size] = X1;
            output_buffer[k + 2 * sub_fft_size] = X2;
            output_buffer[k + 3 * sub_fft_size] = X3;
            output_buffer[k + 4 * sub_fft_size] = X4;
            output_buffer[k + 5 * sub_fft_size] = X5;
            output_buffer[k + 6 * sub_fft_size] = X6;
        }
    }
    else if (radix == 8)
    {
        // --- stage sizing ---
        const int sub_fft_length = data_length / 8; // N/8
        const int next_stride = 8 * stride;
        const int sub_fft_size = sub_fft_length;

        // outputs (8*N/8) + local twiddles (7*N/8) if not precomputed
        const int required_size =
            (fft_obj->twiddle_factors != NULL) ? (8 * sub_fft_size) : (15 * sub_fft_size);
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
            return;

        // sub-FFT outputs: 8 contiguous AoS blocks (X0..X7), each length sub_fft_size
        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;

        // per-stage twiddles
        fft_data *twiddle_factors = NULL;
        if (fft_obj->twiddle_factors != NULL)
        {
            if (factor_index >= fft_obj->num_precomputed_stages)
                return;
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + (scratch_offset + 8 * sub_fft_size); // after outputs
        }

        // --- recurse lanes 0..7 ---
        for (int lane = 0; lane < 8; ++lane)
        {
            mixed_radix_dit_rec(
                sub_fft_outputs + lane * sub_fft_size,
                input_buffer + lane * stride,
                fft_obj, transform_sign,
                sub_fft_length, next_stride,
                factor_index + 1,
                scratch_offset + lane * (required_size / 8));
        }

        // --- local twiddle prep if not precomputed ---
        if (fft_obj->twiddle_factors == NULL)
        {
            const int N = 8 * sub_fft_size; // stage length
            if (fft_obj->n_fft < N)
                return;

            // per k: [W^{1k}, W^{2k}, ..., W^{7k}]
            for (int k = 0; k < sub_fft_size; ++k)
            {
                twiddle_factors[7 * k + 0] = fft_obj->twiddles[(1 * k) % N];
                twiddle_factors[7 * k + 1] = fft_obj->twiddles[(2 * k) % N];
                twiddle_factors[7 * k + 2] = fft_obj->twiddles[(3 * k) % N];
                twiddle_factors[7 * k + 3] = fft_obj->twiddles[(4 * k) % N];
                twiddle_factors[7 * k + 4] = fft_obj->twiddles[(5 * k) % N];
                twiddle_factors[7 * k + 5] = fft_obj->twiddles[(6 * k) % N];
                twiddle_factors[7 * k + 6] = fft_obj->twiddles[(7 * k) % N];
            }
        }

        // =========================
        // SoA AVX2 core (k += 4)
        // =========================
#if defined(__AVX2__)
        const __m256d vc = _mm256_set1_pd(C8_1); // √2/2

        int k = 0;
        for (; k + 3 < sub_fft_size; k += 4)
        {
            // modest prefetch (~64B ahead), both re/im for 8 lanes
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 0 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 0 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 1 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 1 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 2 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 2 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 3 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 3 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 4 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 4 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 5 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 5 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 6 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 6 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 7 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 8 + 7 * sub_fft_size].im, _MM_HINT_T0);

            // ----- A: AoS -> SoA for 8 lanes, 4 points each -----
            double aR[4], aI[4], bR[4], bI[4], cR[4], cI[4], dR[4], dI[4];
            double eR[4], eI[4], fR[4], fI[4], gR[4], gI[4], hR[4], hI[4];

            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 0 * sub_fft_size], aR, aI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 1 * sub_fft_size], bR, bI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 2 * sub_fft_size], cR, cI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 3 * sub_fft_size], dR, dI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 4 * sub_fft_size], eR, eI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 5 * sub_fft_size], fR, fI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 6 * sub_fft_size], gR, gI);
            deinterleave4_aos_to_soa(&sub_fft_outputs[k + 7 * sub_fft_size], hR, hI);

            __m256d Ar = _mm256_loadu_pd(aR), Ai = _mm256_loadu_pd(aI);
            __m256d Br = _mm256_loadu_pd(bR), Bi = _mm256_loadu_pd(bI);
            __m256d Cr = _mm256_loadu_pd(cR), Ci = _mm256_loadu_pd(cI);
            __m256d Dr = _mm256_loadu_pd(dR), Di = _mm256_loadu_pd(dI);
            __m256d Er = _mm256_loadu_pd(eR), Ei = _mm256_loadu_pd(eI);
            __m256d Fr = _mm256_loadu_pd(fR), Fi = _mm256_loadu_pd(fI);
            __m256d Gr = _mm256_loadu_pd(gR), Gi = _mm256_loadu_pd(gI);
            __m256d Hr = _mm256_loadu_pd(hR), Hi = _mm256_loadu_pd(hI);

            // ----- B: build SoA twiddles for k..k+3 (powers 1..7) -----
            fft_data w1a[4], w2a[4], w3a[4], w4a[4], w5a[4], w6a[4], w7a[4];
            for (int p = 0; p < 4; ++p)
            {
                w1a[p] = twiddle_factors[7 * (k + p) + 0];
                w2a[p] = twiddle_factors[7 * (k + p) + 1];
                w3a[p] = twiddle_factors[7 * (k + p) + 2];
                w4a[p] = twiddle_factors[7 * (k + p) + 3];
                w5a[p] = twiddle_factors[7 * (k + p) + 4];
                w6a[p] = twiddle_factors[7 * (k + p) + 5];
                w7a[p] = twiddle_factors[7 * (k + p) + 6];
            }

            double w1R[4], w1I[4], w2R[4], w2I[4], w3R[4], w3I[4];
            double w4R[4], w4I[4], w5R[4], w5I[4], w6R[4], w6I[4], w7R[4], w7I[4];
            deinterleave4_aos_to_soa(w1a, w1R, w1I);
            deinterleave4_aos_to_soa(w2a, w2R, w2I);
            deinterleave4_aos_to_soa(w3a, w3R, w3I);
            deinterleave4_aos_to_soa(w4a, w4R, w4I);
            deinterleave4_aos_to_soa(w5a, w5R, w5I);
            deinterleave4_aos_to_soa(w6a, w6R, w6I);
            deinterleave4_aos_to_soa(w7a, w7R, w7I);

            __m256d W1r = _mm256_loadu_pd(w1R), W1i = _mm256_loadu_pd(w1I);
            __m256d W2r = _mm256_loadu_pd(w2R), W2i = _mm256_loadu_pd(w2I);
            __m256d W3r = _mm256_loadu_pd(w3R), W3i = _mm256_loadu_pd(w3I);
            __m256d W4r = _mm256_loadu_pd(w4R), W4i = _mm256_loadu_pd(w4I);
            __m256d W5r = _mm256_loadu_pd(w5R), W5i = _mm256_loadu_pd(w5I);
            __m256d W6r = _mm256_loadu_pd(w6R), W6i = _mm256_loadu_pd(w6I);
            __m256d W7r = _mm256_loadu_pd(w7R), W7i = _mm256_loadu_pd(w7I);

            // ----- C: apply twiddles in SoA -----
            __m256d b2r, b2i, c2r, c2i, d2r, d2i, e2r, e2i, f2r, f2i, g2r, g2i, h2r, h2i;
            cmul_soa_avx(Br, Bi, W1r, W1i, &b2r, &b2i);
            cmul_soa_avx(Cr, Ci, W2r, W2i, &c2r, &c2i);
            cmul_soa_avx(Dr, Di, W3r, W3i, &d2r, &d2i);
            cmul_soa_avx(Er, Ei, W4r, W4i, &e2r, &e2i);
            cmul_soa_avx(Fr, Fi, W5r, W5i, &f2r, &f2i);
            cmul_soa_avx(Gr, Gi, W6r, W6i, &g2r, &g2i);
            cmul_soa_avx(Hr, Hi, W7r, W7i, &h2r, &h2i);

            // ----- D: radix-8 identities (SoA) -----
            // pair sums/diffs
            __m256d s0r = _mm256_add_pd(b2r, h2r), s0i = _mm256_add_pd(b2i, h2i); // b+h
            __m256d d0r = _mm256_sub_pd(b2r, h2r), d0i = _mm256_sub_pd(b2i, h2i); // b-h
            __m256d s1r = _mm256_add_pd(c2r, g2r), s1i = _mm256_add_pd(c2i, g2i); // c+g
            __m256d d1r = _mm256_sub_pd(c2r, g2r), d1i = _mm256_sub_pd(c2i, g2i); // c-g
            __m256d s2r = _mm256_add_pd(d2r, f2r), s2i = _mm256_add_pd(d2i, f2i); // d+f
            __m256d d2r = _mm256_sub_pd(d2r, f2r), d2i = _mm256_sub_pd(d2i, f2i); // d-f

            __m256d t0r = _mm256_add_pd(Ar, e2r), t0i = _mm256_add_pd(Ai, e2i); // a+e2
            __m256d t4r = _mm256_sub_pd(Ar, e2r), t4i = _mm256_sub_pd(Ai, e2i); // a-e2

            // X0 = t0 + s0 + s1 + s2
            __m256d x0r = _mm256_add_pd(t0r, _mm256_add_pd(_mm256_add_pd(s0r, s1r), s2r));
            __m256d x0i = _mm256_add_pd(t0i, _mm256_add_pd(_mm256_add_pd(s0i, s1i), s2i));

            // X4 = t4 - s0 - s1 + s2
            __m256d x4r = _mm256_add_pd(_mm256_sub_pd(t4r, _mm256_add_pd(s0r, s1r)), s2r);
            __m256d x4i = _mm256_add_pd(_mm256_sub_pd(t4i, _mm256_add_pd(s0i, s1i)), s2i);

            // X2 / X6:
            // base26 = (d2 - f2) - (b - h) = d2 - f2 - b + h
            __m256d base26r = _mm256_sub_pd(d2r, d0r);
            __m256d base26i = _mm256_sub_pd(d2i, d0i);
            __m256d rr26, ri26;
            rot90_soa_avx(base26r, base26i, transform_sign, &rr26, &ri26); // sign * i * base26
            __m256d t02r = _mm256_sub_pd(t0r, s1r);
            __m256d t02i = _mm256_sub_pd(t0i, s1i);
            __m256d x2r = _mm256_add_pd(t02r, rr26), x2i = _mm256_add_pd(t02i, ri26);
            __m256d x6r = _mm256_sub_pd(t02r, rr26), x6i = _mm256_sub_pd(t02i, ri26);

            // X1 / X7:
            // real17 = t4 + c*(s0 - s2)
            __m256d s0ms2r = _mm256_sub_pd(s0r, s2r);
            __m256d s0ms2i = _mm256_sub_pd(s0i, s2i);
            __m256d real17r = _mm256_add_pd(t4r, _mm256_mul_pd(vc, s0ms2r));
            __m256d real17i = _mm256_add_pd(t4i, _mm256_mul_pd(vc, s0ms2i));
            // V17 = -( d1 + c*(d0 + d2) )
            __m256d dd_r = _mm256_add_pd(d0r, d2r);
            __m256d dd_i = _mm256_add_pd(d0i, d2i);
            __m256d V17r = _mm256_add_pd(_mm256_mul_pd(vc, dd_r), d1r);
            __m256d V17i = _mm256_add_pd(_mm256_mul_pd(vc, dd_i), d1i);
            V17r = _mm256_sub_pd(_mm256_setzero_pd(), V17r);
            V17i = _mm256_sub_pd(_mm256_setzero_pd(), V17i);
            __m256d rr17, ri17;
            rot90_soa_avx(V17r, V17i, transform_sign, &rr17, &ri17);
            __m256d x1r = _mm256_add_pd(real17r, rr17), x1i = _mm256_add_pd(real17i, ri17);
            __m256d x7r = _mm256_sub_pd(real17r, rr17), x7i = _mm256_sub_pd(real17i, ri17);

            // X3 / X5:
            // real35 = t4 - c*(s0 - s2)
            __m256d real35r = _mm256_sub_pd(t4r, _mm256_mul_pd(vc, s0ms2r));
            __m256d real35i = _mm256_sub_pd(t4i, _mm256_mul_pd(vc, s0ms2i));
            // V35 = -( d1 + c*(d0 - d2) )
            __m256d dd2_r = _mm256_sub_pd(d0r, d2r);
            __m256d dd2_i = _mm256_sub_pd(d0i, d2i);
            __m256d V35r = _mm256_add_pd(_mm256_mul_pd(vc, dd2_r), d1r);
            __m256d V35i = _mm256_add_pd(_mm256_mul_pd(vc, dd2_i), d1i);
            V35r = _mm256_sub_pd(_mm256_setzero_pd(), V35r);
            V35i = _mm256_sub_pd(_mm256_setzero_pd(), V35i);
            __m256d rr35, ri35;
            rot90_soa_avx(V35r, V35i, transform_sign, &rr35, &ri35);
            __m256d x3r = _mm256_add_pd(real35r, rr35), x3i = _mm256_add_pd(real35i, ri35);
            __m256d x5r = _mm256_sub_pd(real35r, rr35), x5i = _mm256_sub_pd(real35i, ri35);

            // ----- E: SoA -> AoS stores -----
            double X0R[4], X0I[4], X1R[4], X1I[4], X2R[4], X2I[4], X3R[4], X3I[4];
            double X4R[4], X4I[4], X5R[4], X5I[4], X6R[4], X6I[4], X7R[4], X7I[4];

            _mm256_storeu_pd(X0R, x0r);
            _mm256_storeu_pd(X0I, x0i);
            _mm256_storeu_pd(X1R, x1r);
            _mm256_storeu_pd(X1I, x1i);
            _mm256_storeu_pd(X2R, x2r);
            _mm256_storeu_pd(X2I, x2i);
            _mm256_storeu_pd(X3R, x3r);
            _mm256_storeu_pd(X3I, x3i);
            _mm256_storeu_pd(X4R, x4r);
            _mm256_storeu_pd(X4I, x4i);
            _mm256_storeu_pd(X5R, x5r);
            _mm256_storeu_pd(X5I, x5i);
            _mm256_storeu_pd(X6R, x6r);
            _mm256_storeu_pd(X6I, x6i);
            _mm256_storeu_pd(X7R, x7r);
            _mm256_storeu_pd(X7I, x7i);

            interleave4_soa_to_aos(X0R, X0I, &output_buffer[k + 0 * sub_fft_size]);
            interleave4_soa_to_aos(X1R, X1I, &output_buffer[k + 1 * sub_fft_size]);
            interleave4_soa_to_aos(X2R, X2I, &output_buffer[k + 2 * sub_fft_size]);
            interleave4_soa_to_aos(X3R, X3I, &output_buffer[k + 3 * sub_fft_size]);
            interleave4_soa_to_aos(X4R, X4I, &output_buffer[k + 4 * sub_fft_size]);
            interleave4_soa_to_aos(X5R, X5I, &output_buffer[k + 5 * sub_fft_size]);
            interleave4_soa_to_aos(X6R, X6I, &output_buffer[k + 6 * sub_fft_size]);
            interleave4_soa_to_aos(X7R, X7I, &output_buffer[k + 7 * sub_fft_size]);
        }
#else
        int k = 0;
#endif // __AVX2__

        // =========================
        // scalar AoS tail (1..3)
        // =========================
        for (; k < sub_fft_size; ++k)
        {
            const fft_data a = sub_fft_outputs[k + 0 * sub_fft_size];
            const fft_data b = sub_fft_outputs[k + 1 * sub_fft_size];
            const fft_data c = sub_fft_outputs[k + 2 * sub_fft_size];
            const fft_data d = sub_fft_outputs[k + 3 * sub_fft_size];
            const fft_data e = sub_fft_outputs[k + 4 * sub_fft_size];
            const fft_data f = sub_fft_outputs[k + 5 * sub_fft_size];
            const fft_data g = sub_fft_outputs[k + 6 * sub_fft_size];
            const fft_data h = sub_fft_outputs[k + 7 * sub_fft_size];

            const fft_data w1 = twiddle_factors[7 * k + 0];
            const fft_data w2 = twiddle_factors[7 * k + 1];
            const fft_data w3 = twiddle_factors[7 * k + 2];
            const fft_data w4 = twiddle_factors[7 * k + 3];
            const fft_data w5 = twiddle_factors[7 * k + 4];
            const fft_data w6 = twiddle_factors[7 * k + 5];
            const fft_data w7 = twiddle_factors[7 * k + 6];

            // twiddle
            double b2r = b.re * w1.re - b.im * w1.im, b2i = b.re * w1.im + b.im * w1.re;
            double c2r = c.re * w2.re - c.im * w2.im, c2i = c.re * w2.im + c.im * w2.re;
            double d2r = d.re * w3.re - d.im * w3.im, d2i = d.re * w3.im + d.im * w3.re;
            double e2r = e.re * w4.re - e.im * w4.im, e2i = e.re * w4.im + e.im * w4.re;
            double f2r = f.re * w5.re - f.im * w5.im, f2i = f.re * w5.im + f.im * w5.re;
            double g2r = g.re * w6.re - g.im * w6.im, g2i = g.re * w6.im + g.im * w6.re;
            double h2r = h.re * w7.re - h.im * w7.im, h2i = h.re * w7.im + h.im * w7.re;

            // sums/diffs
            double s0r = b2r + h2r, s0i = b2i + h2i;
            double d0r = b2r - h2r, d0i = b2i - h2i;
            double s1r = c2r + g2r, s1i = c2i + g2i;
            double d1r = c2r - g2r, d1i = c2i - g2i;
            double s2r = d2r + f2r, s2i = d2i + f2i;
            double d2mr = d2r - f2r, d2mi = d2i - f2i;

            double t0r = a.re + e2r, t0i = a.im + e2i;
            double t4r = a.re - e2r, t4i = a.im - e2i;

            // X0
            fft_data X0 = {t0r + (s0r + s1r + s2r), t0i + (s0i + s1i + s2i)};

            // X4
            fft_data X4 = {t4r - s0r - s1r + s2r, t4i - s0i - s1i + s2i};

            // X2 / X6
            double base26r = d2mr - d0r, base26i = d2mi - d0i;
            double rr26 = (transform_sign == 1 ? -base26i : base26i);
            double ri26 = (transform_sign == 1 ? base26r : -base26r);
            double t02r = t0r - s1r, t02i = t0i - s1i;
            fft_data X2 = {t02r + rr26, t02i + ri26};
            fft_data X6 = {t02r - rr26, t02i - ri26};

            // X1 / X7
            double s0ms2r = s0r - s2r, s0ms2i = s0i - s2i;
            double real17r = t4r + C8_1 * s0ms2r;
            double real17i = t4i + C8_1 * s0ms2i;
            double dd_r = d0r + d2mr, dd_i = d0i + d2mi;
            double V17r = -(C8_1 * dd_r + d1r);
            double V17i = -(C8_1 * dd_i + d1i);
            double rr17 = (transform_sign == 1 ? -V17i : V17i);
            double ri17 = (transform_sign == 1 ? V17r : -V17r);
            fft_data X1 = {real17r + rr17, real17i + ri17};
            fft_data X7 = {real17r - rr17, real17i - ri17};

            // X3 / X5
            double real35r = t4r - C8_1 * s0ms2r;
            double real35i = t4i - C8_1 * s0ms2i;
            double dd2_r = d0r - d2mr, dd2_i = d0i - d2mi;
            double V35r = -(C8_1 * dd2_r + d1r);
            double V35i = -(C8_1 * dd2_i + d1i);
            double rr35 = (transform_sign == 1 ? -V35i : V35i);
            double ri35 = (transform_sign == 1 ? V35r : -V35r);
            fft_data X3 = {real35r + rr35, real35i + ri35};
            fft_data X5 = {real35r - rr35, real35i - ri35};

            output_buffer[k + 0 * sub_fft_size] = X0;
            output_buffer[k + 1 * sub_fft_size] = X1;
            output_buffer[k + 2 * sub_fft_size] = X2;
            output_buffer[k + 3 * sub_fft_size] = X3;
            output_buffer[k + 4 * sub_fft_size] = X4;
            output_buffer[k + 5 * sub_fft_size] = X5;
            output_buffer[k + 6 * sub_fft_size] = X6;
            output_buffer[k + 7 * sub_fft_size] = X7;
        }
    }
    else if (radix == 11)
    {
        /**
         * @brief Radix-11 decomposition for eleven-point sub-FFTs with AVX2 and SSE2 vectorization and FMA support.
         *
         * Computes the FFT for data lengths divisible by 11 by splitting into eleven sub-FFTs (indices n mod 11 = 0 to 10),
         * applying recursive FFTs, and combining results with twiddle factors. Optimized for power-of-11 (N=11^r) and
         * mixed-radix FFTs using pre-allocated scratch and stage-specific twiddle offsets.
         *
         * Mathematically: Computes:
         *   X(k) = X_0(k) + W_N^k * X_1(k) + ... + W_N^{10k} * X_10(k),
         * where X_0, ..., X_10 are sub-FFTs of size N/11, W_N^k = e^{-2πi k / N}.
         * Uses rotations at multiples of 360°/11 ≈ 32.727° with constants C11_1, ..., S11_5.
         *
         * @warning Assumes fft_obj->twiddles has n_fft ≥ N elements, scratch is 32-byte aligned,
         *          and factor_index is valid for twiddle_factors.
         */
        // Step 1: Compute subproblem size and stride
        int sub_fft_length = data_length / 11; // Size of each sub-FFT (N/11)
        int next_stride = 11 * stride;         // Stride increases elevenfold
        int sub_fft_size = sub_fft_length;     // Sub-FFT size for indexing

        // Step 2: Validate scratch buffer
        int required_size = fft_obj->twiddle_factors != NULL ? 11 * sub_fft_size : 21 * sub_fft_size;
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Scratch buffer too small for radix-11 at offset %d (need %d, have %d)\n",
                    scratch_offset, required_size, fft_obj->max_scratch_size - scratch_offset);
            // exit
        }

        // Step 3: Assign scratch slices
        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors;
        if (fft_obj->twiddle_factors != NULL)
        {
            if (factor_index >= fft_obj->num_precomputed_stages)
            {
                fprintf(stderr, "Error: Invalid factor_index (%d) exceeds num_precomputed_stages (%d) for radix-11\n",
                        factor_index, fft_obj->num_precomputed_stages);
                // exit
            }
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + scratch_offset + 11 * sub_fft_size;
        }

        // Step 4: Compute child scratch offsets
        int child_scratch_per_branch = required_size;
        int total_child_scratch = 11 * child_scratch_per_branch;
        if (scratch_offset + total_child_scratch > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Total child scratch size (%d) exceeds available scratch at offset %d (have %d) for radix-11\n",
                    total_child_scratch, scratch_offset, fft_obj->max_scratch_size - scratch_offset);
            // exit
        }
        int child_offsets[11];
        child_offsets[0] = scratch_offset;
        for (int i = 1; i < 11; i++)
            child_offsets[i] = child_offsets[i - 1] + child_scratch_per_branch;

        // Step 5: Recurse on eleven sub-FFTs
        for (int i = 0; i < 11; i++)
            mixed_radix_dit_rec(sub_fft_outputs + i * sub_fft_size, input_buffer + i * stride, fft_obj,
                                transform_sign, sub_fft_length, next_stride, factor_index + 1,
                                child_offsets[i]);

        // Step 6: Prepare twiddle factors (mixed-radix only)
        if (fft_obj->twiddle_factors == NULL)
        {
            if (fft_obj->n_fft < sub_fft_length - 1 + 10 * sub_fft_size)
            {
                fprintf(stderr, "Error: Twiddle array too small (need at least %d elements, have %d)\n",
                        sub_fft_length - 1 + 10 * sub_fft_size, fft_obj->n_fft);
                // exit
            }
#ifdef USE_TWIDDLE_TABLES
            if (sub_fft_size <= 11 && twiddle_tables[11] != NULL)
            {
                const complex_t *table = twiddle_tables[11];
                for (int k = 0; k < sub_fft_size; k++)
                    for (int n = 0; n < 10; n++)
                    {
                        twiddle_factors[10 * k + n].re = table[n + 1].re;
                        twiddle_factors[10 * k + n].im = table[n + 1].im;
                    }
            }
            else
#endif
            {
                for (int k = 0; k < sub_fft_size; k++)
                    for (int n = 0; n < 10; n++)
                    {
                        int idx = n * sub_fft_size + k;
                        twiddle_factors[10 * k + n].re = fft_obj->twiddles[idx].re;
                        twiddle_factors[10 * k + n].im = fft_obj->twiddles[idx].im;
                    }
            }
        }

        // Step 7: Flatten outputs into separate real/imag arrays using union
        union
        {
            fft_data data[11 * sub_fft_size];
            struct
            {
                double re[11 * sub_fft_size];
                double im[11 * sub_fft_size];
            } flat;
        } *scratch_union = (void *)(fft_obj->scratch + scratch_offset);
        for (int lane = 0; lane < 11; lane++)
        {
            fft_data *base = sub_fft_outputs + lane * sub_fft_size;
            for (int k = 0; k < sub_fft_size; k++)
            {
                scratch_union->flat.re[lane * sub_fft_size + k] = base[k].re;
                scratch_union->flat.im[lane * sub_fft_size + k] = base[k].im;
            }
        }
        double *out_re = scratch_union->flat.re;
        double *out_im = scratch_union->flat.im;

        {
            // Step 8
            // k=0 (no twiddle multiplications)
            {
                fft_data *X[11];
                for (int i = 0; i < 11; i++)
                    X[i] = &output_buffer[i * sub_fft_size];

                fft_type a_r = out_re[0], a_i = out_im[0];
                fft_type b_r = out_re[sub_fft_size], b_i = out_im[sub_fft_size];
                fft_type c_r = out_re[2 * sub_fft_size], c_i = out_im[2 * sub_fft_size];
                fft_type d_r = out_re[3 * sub_fft_size], d_i = out_im[3 * sub_fft_size];
                fft_type e_r = out_re[4 * sub_fft_size], e_i = out_im[4 * sub_fft_size];
                fft_type f_r = out_re[5 * sub_fft_size], f_i = out_im[5 * sub_fft_size];
                fft_type g_r = out_re[6 * sub_fft_size], g_i = out_im[6 * sub_fft_size];
                fft_type h_r = out_re[7 * sub_fft_size], h_i = out_im[7 * sub_fft_size];
                fft_type i_r = out_re[8 * sub_fft_size], i_i = out_im[8 * sub_fft_size];
                fft_type j_r = out_re[9 * sub_fft_size], j_i = out_im[9 * sub_fft_size];
                fft_type k_r = out_re[10 * sub_fft_size], k_i = out_im[10 * sub_fft_size];

                fft_type t0_r = b_r + k_r, t0_i = b_i + k_i; // B + K
                fft_type t1_r = c_r + j_r, t1_i = c_i + j_i; // C + J
                fft_type t2_r = d_r + i_r, t2_i = d_i + i_i; // D + I
                fft_type t3_r = e_r + h_r, t3_i = e_i + h_i; // E + H
                fft_type t4_r = f_r + g_r, t4_i = f_i + g_i; // F + G
                fft_type t5_r = b_r - k_r, t5_i = b_i - k_i; // B - K
                fft_type t6_r = c_r - j_r, t6_i = c_i - j_i; // C - J
                fft_type t7_r = d_r - i_r, t7_i = d_i - i_i; // D - I
                fft_type t8_r = e_r - h_r, t8_i = e_i - h_i; // E - H
                fft_type t9_r = f_r - g_r, t9_i = f_i - g_i; // F - G

                X[0]->re = a_r + t0_r + t1_r + t2_r + t3_r + t4_r;
                X[0]->im = a_i + t0_i + t1_i + t2_i + t3_i + t4_i;

                fft_type tmp_r = a_r + C11_1 * t0_r + C11_2 * t1_r + C11_3 * t2_r + C11_4 * t3_r + C11_5 * t4_r;
                fft_type tmp_i = a_i + C11_1 * t0_i + C11_2 * t1_i + C11_3 * t2_i + C11_4 * t3_i + C11_5 * t4_i;
                fft_type rot_r = transform_sign * (S11_1 * t5_i + S11_2 * t6_i + S11_3 * t7_i + S11_4 * t8_i + S11_5 * t9_i);
                fft_type rot_i = transform_sign * (-S11_1 * t5_r - S11_2 * t6_r - S11_3 * t7_r - S11_4 * t8_r - S11_5 * t9_r);
                X[1]->re = tmp_r + rot_r;
                X[1]->im = tmp_i + rot_i;
                X[10]->re = tmp_r - rot_r;
                X[10]->im = tmp_i - rot_i;

                tmp_r = a_r + C11_2 * t0_r + C11_4 * t1_r + C11_5 * t2_r + C11_3 * t3_r + C11_1 * t4_r;
                tmp_i = a_i + C11_2 * t0_i + C11_4 * t1_i + C11_5 * t2_i + C11_3 * t3_i + C11_1 * t4_i;
                rot_r = transform_sign * (S11_2 * t5_i + S11_4 * t6_i + S11_5 * t7_i + S11_3 * t8_i + S11_1 * t9_i);
                rot_i = transform_sign * (-S11_2 * t5_r - S11_4 * t6_r - S11_5 * t7_r - S11_3 * t8_r - S11_1 * t9_r);
                X[2]->re = tmp_r + rot_r;
                X[2]->im = tmp_i + rot_i;
                X[9]->re = tmp_r - rot_r;
                X[9]->im = tmp_i - rot_i;

                tmp_r = a_r + C11_3 * t0_r + C11_5 * t1_r + C11_2 * t2_r + C11_1 * t3_r + C11_4 * t4_r;
                tmp_i = a_i + C11_3 * t0_i + C11_5 * t1_i + C11_2 * t2_i + C11_1 * t3_i + C11_4 * t4_i;
                rot_r = transform_sign * (S11_3 * t5_i + S11_5 * t6_i + S11_2 * t7_i + S11_1 * t8_i + S11_4 * t9_i);
                rot_i = transform_sign * (-S11_3 * t5_r - S11_5 * t6_r - S11_2 * t7_r - S11_1 * t8_r - S11_4 * t9_r);
                X[3]->re = tmp_r + rot_r;
                X[3]->im = tmp_i + rot_i;
                X[8]->re = tmp_r - rot_r;
                X[8]->im = tmp_i - rot_i;

                tmp_r = a_r + C11_4 * t0_r + C11_3 * t1_r + C11_1 * t2_r + C11_5 * t3_r + C11_2 * t4_r;
                tmp_i = a_i + C11_4 * t0_i + C11_3 * t1_i + C11_1 * t2_i + C11_5 * t3_i + C11_2 * t4_i;
                rot_r = transform_sign * (S11_4 * t5_i + S11_3 * t6_i + S11_1 * t7_i + S11_5 * t8_i + S11_2 * t9_i);
                rot_i = transform_sign * (-S11_4 * t5_r - S11_3 * t6_r - S11_1 * t7_r - S11_5 * t8_r - S11_2 * t9_r);
                X[4]->re = tmp_r + rot_r;
                X[4]->im = tmp_i + rot_i;
                X[7]->re = tmp_r - rot_r;
                X[7]->im = tmp_i - rot_i;

                tmp_r = a_r + C11_5 * t0_r + C11_1 * t1_r + C11_4 * t2_r + C11_2 * t3_r + C11_3 * t4_r;
                tmp_i = a_i + C11_5 * t0_i + C11_1 * t1_i + C11_4 * t2_i + C11_2 * t3_i + C11_3 * t4_i;
                rot_r = transform_sign * (S11_5 * t5_i + S11_1 * t6_i + S11_4 * t7_i + S11_2 * t8_i + S11_3 * t9_i);
                rot_i = transform_sign * (-S11_5 * t5_r - S11_1 * t6_r - S11_4 * t7_r - S11_2 * t8_r - S11_3 * t9_r);
                X[5]->re = tmp_r + rot_r;
                X[5]->im = tmp_i + rot_i;
                X[6]->re = tmp_r - rot_r;
                X[6]->im = tmp_i - rot_i;
            }

            // k=1 (with twiddle multiplications)
            if (sub_fft_size > 1)
            {
                fft_data *X[11];
                for (int i = 0; i < 11; i++)
                    X[i] = &output_buffer[i * sub_fft_size + 1];

                fft_type a_r = out_re[1], a_i = out_im[1];
                fft_type b_r = out_re[sub_fft_size + 1], b_i = out_im[sub_fft_size + 1];
                fft_type c_r = out_re[2 * sub_fft_size + 1], c_i = out_im[2 * sub_fft_size + 1];
                fft_type d_r = out_re[3 * sub_fft_size + 1], d_i = out_im[3 * sub_fft_size + 1];
                fft_type e_r = out_re[4 * sub_fft_size + 1], e_i = out_im[4 * sub_fft_size + 1];
                fft_type f_r = out_re[5 * sub_fft_size + 1], f_i = out_im[5 * sub_fft_size + 1];
                fft_type g_r = out_re[6 * sub_fft_size + 1], g_i = out_im[6 * sub_fft_size + 1];
                fft_type h_r = out_re[7 * sub_fft_size + 1], h_i = out_im[7 * sub_fft_size + 1];
                fft_type i_r = out_re[8 * sub_fft_size + 1], i_i = out_im[8 * sub_fft_size + 1];
                fft_type j_r = out_re[9 * sub_fft_size + 1], j_i = out_im[9 * sub_fft_size + 1];
                fft_type k_r = out_re[10 * sub_fft_size + 1], k_i = out_im[10 * sub_fft_size + 1];

                int idx = 10 * 1;
                fft_type w1r = twiddle_factors[idx + 0].re, w1i = twiddle_factors[idx + 0].im;
                fft_type w2r = twiddle_factors[idx + 1].re, w2i = twiddle_factors[idx + 1].im;
                fft_type w3r = twiddle_factors[idx + 2].re, w3i = twiddle_factors[idx + 2].im;
                fft_type w4r = twiddle_factors[idx + 3].re, w4i = twiddle_factors[idx + 3].im;
                fft_type w5r = twiddle_factors[idx + 4].re, w5i = twiddle_factors[idx + 4].im;
                fft_type w6r = twiddle_factors[idx + 5].re, w6i = twiddle_factors[idx + 5].im;
                fft_type w7r = twiddle_factors[idx + 6].re, w7i = twiddle_factors[idx + 6].im;
                fft_type w8r = twiddle_factors[idx + 7].re, w8i = twiddle_factors[idx + 7].im;
                fft_type w9r = twiddle_factors[idx + 8].re, w9i = twiddle_factors[idx + 8].im;
                fft_type w10r = twiddle_factors[idx + 9].re, w10i = twiddle_factors[idx + 9].im;

                fft_type b2_r = b_r * w1r - b_i * w1i, b2_i = b_i * w1r + b_r * w1i;
                fft_type c2_r = c_r * w2r - c_i * w2i, c2_i = c_i * w2r + c_r * w2i;
                fft_type d2_r = d_r * w3r - d_i * w3i, d2_i = d_i * w3r + d_r * w3i;
                fft_type e2_r = e_r * w4r - e_i * w4i, e2_i = e_i * w4r + e_r * w4i;
                fft_type f2_r = f_r * w5r - f_i * w5i, f2_i = f_i * w5r + f_r * w5i;
                fft_type g2_r = g_r * w6r - g_i * w6i, g2_i = g_i * w6r + g_r * w6i;
                fft_type h2_r = h_r * w7r - h_i * w7i, h2_i = h_i * w7r + h_r * w7i;
                fft_type i2_r = i_r * w8r - i_i * w8i, i2_i = i_i * w8r + i_r * w8i;
                fft_type j2_r = j_r * w9r - j_i * w9i, j2_i = j_i * w9r + j_r * w9i;
                fft_type k2_r = k_r * w10r - k_i * w10i, k2_i = k_i * w10r + k_r * w10i;

                fft_type t0_r = b2_r + k2_r, t0_i = b2_i + k2_i;
                fft_type t1_r = c2_r + j2_r, t1_i = c2_i + j2_i;
                fft_type t2_r = d2_r + i2_r, t2_i = d2_i + i2_i;
                fft_type t3_r = e2_r + h2_r, t3_i = e2_i + h2_i;
                fft_type t4_r = f2_r + g2_r, t4_i = f2_i + g2_i;
                fft_type t5_r = b2_r - k2_r, t5_i = b2_i - k2_i;
                fft_type t6_r = c2_r - j2_r, t6_i = c2_i - j2_i;
                fft_type t7_r = d2_r - i2_r, t7_i = d2_i - i2_i;
                fft_type t8_r = e2_r - h2_r, t8_i = e2_i - h2_i;
                fft_type t9_r = f2_r - g2_r, t9_i = f2_i - g2_i;

                X[0]->re = a_r + t0_r + t1_r + t2_r + t3_r + t4_r;
                X[0]->im = a_i + t0_i + t1_i + t2_i + t3_i + t4_i;

                fft_type tmp_r = a_r + C11_1 * t0_r + C11_2 * t1_r + C11_3 * t2_r + C11_4 * t3_r + C11_5 * t4_r;
                fft_type tmp_i = a_i + C11_1 * t0_i + C11_2 * t1_i + C11_3 * t2_i + C11_4 * t3_i + C11_5 * t4_i;
                fft_type rot_r = transform_sign * (S11_1 * t5_i + S11_2 * t6_i + S11_3 * t7_i + S11_4 * t8_i + S11_5 * t9_i);
                fft_type rot_i = transform_sign * (-S11_1 * t5_r - S11_2 * t6_r - S11_3 * t7_r - S11_4 * t8_r - S11_5 * t9_r);
                X[1]->re = tmp_r + rot_r;
                X[1]->im = tmp_i + rot_i;
                X[10]->re = tmp_r - rot_r;
                X[10]->im = tmp_i - rot_i;

                tmp_r = a_r + C11_2 * t0_r + C11_4 * t1_r + C11_5 * t2_r + C11_3 * t3_r + C11_1 * t4_r;
                tmp_i = a_i + C11_2 * t0_i + C11_4 * t1_i + C11_5 * t2_i + C11_3 * t3_i + C11_1 * t4_i;
                rot_r = transform_sign * (S11_2 * t5_i + S11_4 * t6_i + S11_5 * t7_i + S11_3 * t8_i + S11_1 * t9_i);
                rot_i = transform_sign * (-S11_2 * t5_r - S11_4 * t6_r - S11_5 * t7_r - S11_3 * t8_r - S11_1 * t9_r);
                X[2]->re = tmp_r + rot_r;
                X[2]->im = tmp_i + rot_i;
                X[9]->re = tmp_r - rot_r;
                X[9]->im = tmp_i - rot_i;

                tmp_r = a_r + C11_3 * t0_r + C11_5 * t1_r + C11_2 * t2_r + C11_1 * t3_r + C11_4 * t4_r;
                tmp_i = a_i + C11_3 * t0_i + C11_5 * t1_i + C11_2 * t2_i + C11_1 * t3_i + C11_4 * t4_i;
                rot_r = transform_sign * (S11_3 * t5_i + S11_5 * t6_i + S11_2 * t7_i + S11_1 * t8_i + S11_4 * t9_i);
                rot_i = transform_sign * (-S11_3 * t5_r - S11_5 * t6_r - S11_2 * t7_r - S11_1 * t8_r - S11_4 * t9_r);
                X[3]->re = tmp_r + rot_r;
                X[3]->im = tmp_i + rot_i;
                X[8]->re = tmp_r - rot_r;
                X[8]->im = tmp_i - rot_i;

                tmp_r = a_r + C11_4 * t0_r + C11_3 * t1_r + C11_1 * t2_r + C11_5 * t3_r + C11_2 * t4_r;
                tmp_i = a_i + C11_4 * t0_i + C11_3 * t1_i + C11_1 * t2_i + C11_5 * t3_i + C11_2 * t4_i;
                rot_r = transform_sign * (S11_4 * t5_i + S11_3 * t6_i + S11_1 * t7_i + S11_5 * t8_i + S11_2 * t9_i);
                rot_i = transform_sign * (-S11_4 * t5_r - S11_3 * t6_r - S11_1 * t7_r - S11_5 * t8_r - S11_2 * t9_r);
                X[4]->re = tmp_r + rot_r;
                X[4]->im = tmp_i + rot_i;
                X[7]->re = tmp_r - rot_r;
                X[7]->im = tmp_i - rot_i;

                tmp_r = a_r + C11_5 * t0_r + C11_1 * t1_r + C11_4 * t2_r + C11_2 * t3_r + C11_3 * t4_r;
                tmp_i = a_i + C11_5 * t0_i + C11_1 * t1_i + C11_4 * t2_i + C11_2 * t3_i + C11_3 * t4_i;
                rot_r = transform_sign * (S11_5 * t5_i + S11_1 * t6_i + S11_4 * t7_i + S11_2 * t8_i + S11_3 * t9_i);
                rot_i = transform_sign * (-S11_5 * t5_r - S11_1 * t6_r - S11_4 * t7_r - S11_2 * t8_r - S11_3 * t9_r);
                X[5]->re = tmp_r + rot_r;
                X[5]->im = tmp_i + rot_i;
                X[6]->re = tmp_r - rot_r;
                X[6]->im = tmp_i - rot_i;
            }

            // Copy k=0,1 results from output_buffer to out_re/out_im for consistency
            for (int i = 0; i < 11; i++)
            {
                out_re[i * sub_fft_size] = output_buffer[i * sub_fft_size].re;
                out_im[i * sub_fft_size] = output_buffer[i * sub_fft_size].im;
                if (sub_fft_size > 1)
                {
                    out_re[i * sub_fft_size + 1] = output_buffer[i * sub_fft_size + 1].re;
                    out_im[i * sub_fft_size + 1] = output_buffer[i * sub_fft_size + 1].im;
                }
            }
        }

        // Step 8b: SSE2 computation for k=2,3 using out_re/out_im
        if (sub_fft_size > 3)
        {
            int k0 = 2;
            __m128d vsign_sse2 = _mm_set1_pd((double)transform_sign);
            __m128d vc11_1_sse2 = _mm_set1_pd(C11_1);
            __m128d vc11_2_sse2 = _mm_set1_pd(C11_2);
            __m128d vc11_3_sse2 = _mm_set1_pd(C11_3);
            __m128d vc11_4_sse2 = _mm_set1_pd(C11_4);
            __m128d vc11_5_sse2 = _mm_set1_pd(C11_5);
            __m128d vs11_1_sse2 = _mm_set1_pd(S11_1);
            __m128d vs11_2_sse2 = _mm_set1_pd(S11_2);
            __m128d vs11_3_sse2 = _mm_set1_pd(S11_3);
            __m128d vs11_4_sse2 = _mm_set1_pd(S11_4);
            __m128d vs11_5_sse2 = _mm_set1_pd(S11_5);

            __m128d a_r = _mm_loadu_pd(out_re + k0);
            __m128d a_i = _mm_loadu_pd(out_im + k0);
            __m128d b_r = _mm_loadu_pd(out_re + k0 + sub_fft_size);
            __m128d b_i = _mm_loadu_pd(out_im + k0 + sub_fft_size);
            __m128d c_r = _mm_loadu_pd(out_re + k0 + 2 * sub_fft_size);
            __m128d c_i = _mm_loadu_pd(out_im + k0 + 2 * sub_fft_size);
            __m128d d_r = _mm_loadu_pd(out_re + k0 + 3 * sub_fft_size);
            __m128d d_i = _mm_loadu_pd(out_im + k0 + 3 * sub_fft_size);
            __m128d e_r = _mm_loadu_pd(out_re + k0 + 4 * sub_fft_size);
            __m128d e_i = _mm_loadu_pd(out_im + k0 + 4 * sub_fft_size);
            __m128d f_r = _mm_loadu_pd(out_re + k0 + 5 * sub_fft_size);
            __m128d f_i = _mm_loadu_pd(out_im + k0 + 5 * sub_fft_size);
            __m128d g_r = _mm_loadu_pd(out_re + k0 + 6 * sub_fft_size);
            __m128d g_i = _mm_loadu_pd(out_im + k0 + 6 * sub_fft_size);
            __m128d h_r = _mm_loadu_pd(out_re + k0 + 7 * sub_fft_size);
            __m128d h_i = _mm_loadu_pd(out_im + k0 + 7 * sub_fft_size);
            __m128d i_r = _mm_loadu_pd(out_re + k0 + 8 * sub_fft_size);
            __m128d i_i = _mm_loadu_pd(out_im + k0 + 8 * sub_fft_size);
            __m128d j_r = _mm_loadu_pd(out_re + k0 + 9 * sub_fft_size);
            __m128d j_i = _mm_loadu_pd(out_im + k0 + 9 * sub_fft_size);
            __m128d k_r = _mm_loadu_pd(out_re + k0 + 10 * sub_fft_size);
            __m128d k_i = _mm_loadu_pd(out_im + k0 + 10 * sub_fft_size);

            int idx_k0 = 10 * 2, idx_k1 = 10 * 3;
            __m128d w1r = _mm_set_pd(twiddle_factors[idx_k1 + 0].re, twiddle_factors[idx_k0 + 0].re);
            __m128d w1i = _mm_set_pd(twiddle_factors[idx_k1 + 0].im, twiddle_factors[idx_k0 + 0].im);
            __m128d w2r = _mm_set_pd(twiddle_factors[idx_k1 + 1].re, twiddle_factors[idx_k0 + 1].re);
            __m128d w2i = _mm_set_pd(twiddle_factors[idx_k1 + 1].im, twiddle_factors[idx_k0 + 1].im);
            __m128d w3r = _mm_set_pd(twiddle_factors[idx_k1 + 2].re, twiddle_factors[idx_k0 + 2].re);
            __m128d w3i = _mm_set_pd(twiddle_factors[idx_k1 + 2].im, twiddle_factors[idx_k0 + 2].im);
            __m128d w4r = _mm_set_pd(twiddle_factors[idx_k1 + 3].re, twiddle_factors[idx_k0 + 3].re);
            __m128d w4i = _mm_set_pd(twiddle_factors[idx_k1 + 3].im, twiddle_factors[idx_k0 + 3].im);
            __m128d w5r = _mm_set_pd(twiddle_factors[idx_k1 + 4].re, twiddle_factors[idx_k0 + 4].re);
            __m128d w5i = _mm_set_pd(twiddle_factors[idx_k1 + 4].im, twiddle_factors[idx_k0 + 4].im);
            __m128d w6r = _mm_set_pd(twiddle_factors[idx_k1 + 5].re, twiddle_factors[idx_k0 + 5].re);
            __m128d w6i = _mm_set_pd(twiddle_factors[idx_k1 + 5].im, twiddle_factors[idx_k0 + 5].im);
            __m128d w7r = _mm_set_pd(twiddle_factors[idx_k1 + 6].re, twiddle_factors[idx_k0 + 6].re);
            __m128d w7i = _mm_set_pd(twiddle_factors[idx_k1 + 6].im, twiddle_factors[idx_k0 + 6].im);
            __m128d w8r = _mm_set_pd(twiddle_factors[idx_k1 + 7].re, twiddle_factors[idx_k0 + 7].re);
            __m128d w8i = _mm_set_pd(twiddle_factors[idx_k1 + 7].im, twiddle_factors[idx_k0 + 7].im);
            __m128d w9r = _mm_set_pd(twiddle_factors[idx_k1 + 8].re, twiddle_factors[idx_k0 + 8].re);
            __m128d w9i = _mm_set_pd(twiddle_factors[idx_k1 + 8].im, twiddle_factors[idx_k0 + 8].im);
            __m128d w10r = _mm_set_pd(twiddle_factors[idx_k1 + 9].re, twiddle_factors[idx_k0 + 9].re);
            __m128d w10i = _mm_set_pd(twiddle_factors[idx_k1 + 9].im, twiddle_factors[idx_k0 + 9].im);

            __m128d b2r = FMSUB_SSE2(b_r, w1r, _mm_mul_pd(b_i, w1i));
            __m128d b2i = FMADD_SSE2(b_i, w1r, _mm_mul_pd(b_r, w1i));
            __m128d c2r = FMSUB_SSE2(c_r, w2r, _mm_mul_pd(c_i, w2i));
            __m128d c2i = FMADD_SSE2(c_i, w2r, _mm_mul_pd(c_r, w2i));
            __m128d d2r = FMSUB_SSE2(d_r, w3r, _mm_mul_pd(d_i, w3i));
            __m128d d2i = FMADD_SSE2(d_i, w3r, _mm_mul_pd(d_r, w3i));
            __m128d e2r = FMSUB_SSE2(e_r, w4r, _mm_mul_pd(e_i, w4i));
            __m128d e2i = FMADD_SSE2(e_i, w4r, _mm_mul_pd(e_r, w4i));
            __m128d f2r = FMSUB_SSE2(f_r, w5r, _mm_mul_pd(f_i, w5i));
            __m128d f2i = FMADD_SSE2(f_i, w5r, _mm_mul_pd(f_r, w5i));
            __m128d g2r = FMSUB_SSE2(g_r, w6r, _mm_mul_pd(g_i, w6i));
            __m128d g2i = FMADD_SSE2(g_i, w6r, _mm_mul_pd(g_r, w6i));
            __m128d h2r = FMSUB_SSE2(h_r, w7r, _mm_mul_pd(h_i, w7i));
            __m128d h2i = FMADD_SSE2(h_i, w7r, _mm_mul_pd(h_r, w7i));
            __m128d i2r = FMSUB_SSE2(i_r, w8r, _mm_mul_pd(i_i, w8i));
            __m128d i2i = FMADD_SSE2(i_i, w8r, _mm_mul_pd(i_r, w8i));
            __m128d j2r = FMSUB_SSE2(j_r, w9r, _mm_mul_pd(j_i, w9i));
            __m128d j2i = FMADD_SSE2(j_i, w9r, _mm_mul_pd(j_r, w9i));
            __m128d k2r = FMSUB_SSE2(k_r, w10r, _mm_mul_pd(k_i, w10i));
            __m128d k2i = FMADD_SSE2(k_i, w10r, _mm_mul_pd(k_r, w10i));

            __m128d t0r = _mm_add_pd(b2r, k2r);
            __m128d t0i = _mm_add_pd(b2i, k2i);
            __m128d t1r = _mm_add_pd(c2r, j2r);
            __m128d t1i = _mm_add_pd(c2i, j2i);
            __m128d t2r = _mm_add_pd(d2r, i2r);
            __m128d t2i = _mm_add_pd(d2i, i2i);
            __m128d t3r = _mm_add_pd(e2r, h2r);
            __m128d t3i = _mm_add_pd(e2i, h2i);
            __m128d t4r = _mm_add_pd(f2r, g2r);
            __m128d t4i = _mm_add_pd(f2i, g2i);
            __m128d t5r = _mm_sub_pd(b2r, k2r);
            __m128d t5i = _mm_sub_pd(b2i, k2i);
            __m128d t6r = _mm_sub_pd(c2r, j2r);
            __m128d t6i = _mm_sub_pd(c2i, j2i);
            __m128d t7r = _mm_sub_pd(d2r, i2r);
            __m128d t7i = _mm_sub_pd(d2i, i2i);
            __m128d t8r = _mm_sub_pd(e2r, h2r);
            __m128d t8i = _mm_sub_pd(e2i, h2i);
            __m128d t9r = _mm_sub_pd(f2r, g2r);
            __m128d t9i = _mm_sub_pd(f2i, g2i);

            __m128d sum_r = _mm_add_pd(t0r, _mm_add_pd(t1r, _mm_add_pd(t2r, _mm_add_pd(t3r, t4r))));
            __m128d sum_i = _mm_add_pd(t0i, _mm_add_pd(t1i, _mm_add_pd(t2i, _mm_add_pd(t3i, t4i))));
            _mm_storeu_pd(&out_re[k0], _mm_add_pd(a_r, sum_r));
            _mm_storeu_pd(&out_im[k0], _mm_add_pd(a_i, sum_i));

            __m128d tmp1r = FMADD_SSE2(vc11_5_sse2, t4r,
                                       FMADD_SSE2(vc11_4_sse2, t3r,
                                                  FMADD_SSE2(vc11_3_sse2, t2r,
                                                             FMADD_SSE2(vc11_2_sse2, t1r,
                                                                        FMADD_SSE2(vc11_1_sse2, t0r, a_r)))));
            __m128d tmp1i = FMADD_SSE2(vc11_5_sse2, t4i,
                                       FMADD_SSE2(vc11_4_sse2, t3i,
                                                  FMADD_SSE2(vc11_3_sse2, t2i,
                                                             FMADD_SSE2(vc11_2_sse2, t1i,
                                                                        FMADD_SSE2(vc11_1_sse2, t0i, a_i))))); // +1 closing paren

            __m128d rot1r = FMADD_SSE2(vs11_5_sse2, t9i,
                                       FMADD_SSE2(vs11_4_sse2, t8i,
                                                  FMADD_SSE2(vs11_3_sse2, t7i,
                                                             FMADD_SSE2(vs11_2_sse2, t6i,
                                                                        _mm_mul_pd(vs11_1_sse2, t5i))))); // +1 closing paren

            __m128d rot1i = FMADD_SSE2(vs11_5_sse2, t9r,
                                       FMADD_SSE2(vs11_4_sse2, t8r,
                                                  FMADD_SSE2(vs11_3_sse2, t7r,
                                                             FMADD_SSE2(vs11_2_sse2, t6r,
                                                                        _mm_mul_pd(vs11_1_sse2, t5r))))); // +1 closing paren
            rot1r = _mm_mul_pd(vsign_sse2, rot1r);
            rot1i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), vsign_sse2), rot1i);
            _mm_storeu_pd(&out_re[k0 + sub_fft_size], _mm_add_pd(tmp1r, rot1r));
            _mm_storeu_pd(&out_im[k0 + sub_fft_size], _mm_add_pd(tmp1i, rot1i));
            _mm_storeu_pd(&out_re[k0 + 10 * sub_fft_size], _mm_sub_pd(tmp1r, rot1r));
            _mm_storeu_pd(&out_im[k0 + 10 * sub_fft_size], _mm_sub_pd(tmp1i, rot1i));

            __m128d tmp2r = FMADD_SSE2(vc11_1_sse2, t4r,
                                       FMADD_SSE2(vc11_3_sse2, t3r,
                                                  FMADD_SSE2(vc11_5_sse2, t2r,
                                                             FMADD_SSE2(vc11_4_sse2, t1r,
                                                                        FMADD_SSE2(vc11_2_sse2, t0r, a_r))))); // +1 )

            __m128d tmp2i = FMADD_SSE2(vc11_1_sse2, t4i,
                                       FMADD_SSE2(vc11_3_sse2, t3i,
                                                  FMADD_SSE2(vc11_5_sse2, t2i,
                                                             FMADD_SSE2(vc11_4_sse2, t1i,
                                                                        FMADD_SSE2(vc11_2_sse2, t0i, a_i))))); // +1 )

            __m128d rot2r = FMADD_SSE2(vs11_1_sse2, t9i,
                                       FMADD_SSE2(vs11_3_sse2, t8i,
                                                  FMADD_SSE2(vs11_5_sse2, t7i,
                                                             FMADD_SSE2(vs11_4_sse2, t6i,
                                                                        _mm_mul_pd(vs11_2_sse2, t5i))))); // +1 )

            __m128d rot2i = FMADD_SSE2(vs11_1_sse2, t9r,
                                       FMADD_SSE2(vs11_3_sse2, t8r,
                                                  FMADD_SSE2(vs11_5_sse2, t7r,
                                                             FMADD_SSE2(vs11_4_sse2, t6r,
                                                                        _mm_mul_pd(vs11_2_sse2, t5r))))); // +1 )
            rot2r = _mm_mul_pd(vsign_sse2, rot2r);
            rot2i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), vsign_sse2), rot2i);
            _mm_storeu_pd(&out_re[k0 + 2 * sub_fft_size], _mm_add_pd(tmp2r, rot2r));
            _mm_storeu_pd(&out_im[k0 + 2 * sub_fft_size], _mm_add_pd(tmp2i, rot2i));
            _mm_storeu_pd(&out_re[k0 + 9 * sub_fft_size], _mm_sub_pd(tmp2r, rot2r));
            _mm_storeu_pd(&out_im[k0 + 9 * sub_fft_size], _mm_sub_pd(tmp2i, rot2i));

            __m128d tmp3r = FMADD_SSE2(vc11_4_sse2, t4r,
                                       FMADD_SSE2(vc11_1_sse2, t3r,
                                                  FMADD_SSE2(vc11_2_sse2, t2r,
                                                             FMADD_SSE2(vc11_5_sse2, t1r,
                                                                        FMADD_SSE2(vc11_3_sse2, t0r, a_r))))); // +1 )

            __m128d tmp3i = FMADD_SSE2(vc11_4_sse2, t4i,
                                       FMADD_SSE2(vc11_1_sse2, t3i,
                                                  FMADD_SSE2(vc11_2_sse2, t2i,
                                                             FMADD_SSE2(vc11_5_sse2, t1i,
                                                                        FMADD_SSE2(vc11_3_sse2, t0i, a_i))))); // +1 )

            __m128d rot3r = FMADD_SSE2(vs11_4_sse2, t9i,
                                       FMADD_SSE2(vs11_1_sse2, t8i,
                                                  FMADD_SSE2(vs11_2_sse2, t7i,
                                                             FMADD_SSE2(vs11_5_sse2, t6i,
                                                                        _mm_mul_pd(vs11_3_sse2, t5i))))); // +1 )

            __m128d rot3i = FMADD_SSE2(vs11_4_sse2, t9r,
                                       FMADD_SSE2(vs11_1_sse2, t8r,
                                                  FMADD_SSE2(vs11_2_sse2, t7r,
                                                             FMADD_SSE2(vs11_5_sse2, t6r,
                                                                        _mm_mul_pd(vs11_3_sse2, t5r))))); // +1 )
            rot3r = _mm_mul_pd(vsign_sse2, rot3r);
            rot3i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), vsign_sse2), rot3i);
            _mm_storeu_pd(&out_re[k0 + 3 * sub_fft_size], _mm_add_pd(tmp3r, rot3r));
            _mm_storeu_pd(&out_im[k0 + 3 * sub_fft_size], _mm_add_pd(tmp3i, rot3i));
            _mm_storeu_pd(&out_re[k0 + 8 * sub_fft_size], _mm_sub_pd(tmp3r, rot3r));
            _mm_storeu_pd(&out_im[k0 + 8 * sub_fft_size], _mm_sub_pd(tmp3i, rot3i));

            __m128d tmp4r = FMADD_SSE2(vc11_2_sse2, t4r,
                                       FMADD_SSE2(vc11_5_sse2, t3r,
                                                  FMADD_SSE2(vc11_1_sse2, t2r,
                                                             FMADD_SSE2(vc11_3_sse2, t1r,
                                                                        FMADD_SSE2(vc11_4_sse2, t0r, a_r))))); // +1 )

            __m128d tmp4i = FMADD_SSE2(vc11_2_sse2, t4i,
                                       FMADD_SSE2(vc11_5_sse2, t3i,
                                                  FMADD_SSE2(vc11_1_sse2, t2i,
                                                             FMADD_SSE2(vc11_3_sse2, t1i,
                                                                        FMADD_SSE2(vc11_4_sse2, t0i, a_i))))); // +1 )

            __m128d rot4r = FMADD_SSE2(vs11_2_sse2, t9i,
                                       FMADD_SSE2(vs11_5_sse2, t8i,
                                                  FMADD_SSE2(vs11_1_sse2, t7i,
                                                             FMADD_SSE2(vs11_3_sse2, t6i,
                                                                        _mm_mul_pd(vs11_4_sse2, t5i))))); // +1 )

            __m128d rot4i = FMADD_SSE2(vs11_2_sse2, t9r,
                                       FMADD_SSE2(vs11_5_sse2, t8r,
                                                  FMADD_SSE2(vs11_1_sse2, t7r,
                                                             FMADD_SSE2(vs11_3_sse2, t6r,
                                                                        _mm_mul_pd(vs11_4_sse2, t5r))))); // +1 )
            rot4r = _mm_mul_pd(vsign_sse2, rot4r);
            rot4i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), vsign_sse2), rot4i);
            _mm_storeu_pd(&out_re[k0 + 4 * sub_fft_size], _mm_add_pd(tmp4r, rot4r));
            _mm_storeu_pd(&out_im[k0 + 4 * sub_fft_size], _mm_add_pd(tmp4i, rot4i));
            _mm_storeu_pd(&out_re[k0 + 7 * sub_fft_size], _mm_sub_pd(tmp4r, rot4r));
            _mm_storeu_pd(&out_im[k0 + 7 * sub_fft_size], _mm_sub_pd(tmp4i, rot4i));

            __m128d tmp5r = FMADD_SSE2(vc11_3_sse2, t4r,
                                       FMADD_SSE2(vc11_2_sse2, t3r,
                                                  FMADD_SSE2(vc11_4_sse2, t2r,
                                                             FMADD_SSE2(vc11_1_sse2, t1r,
                                                                        FMADD_SSE2(vc11_5_sse2, t0r, a_r))))); // +1 )

            __m128d tmp5i = FMADD_SSE2(vc11_3_sse2, t4i,
                                       FMADD_SSE2(vc11_2_sse2, t3i,
                                                  FMADD_SSE2(vc11_4_sse2, t2i,
                                                             FMADD_SSE2(vc11_1_sse2, t1i,
                                                                        FMADD_SSE2(vc11_5_sse2, t0i, a_i))))); // +1 )

            __m128d rot5r = FMADD_SSE2(vs11_3_sse2, t9i,
                                       FMADD_SSE2(vs11_2_sse2, t8i,
                                                  FMADD_SSE2(vs11_4_sse2, t7i,
                                                             FMADD_SSE2(vs11_1_sse2, t6i,
                                                                        _mm_mul_pd(vs11_5_sse2, t5i))))); // +1 )

            __m128d rot5i = FMADD_SSE2(vs11_3_sse2, t9r,
                                       FMADD_SSE2(vs11_2_sse2, t8r,
                                                  FMADD_SSE2(vs11_4_sse2, t7r,
                                                             FMADD_SSE2(vs11_1_sse2, t6r,
                                                                        _mm_mul_pd(vs11_5_sse2, t5r))))); // +1 )
            rot5r = _mm_mul_pd(vsign_sse2, rot5r);
            rot5i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), vsign_sse2), rot5i);
            _mm_storeu_pd(&out_re[k0 + 5 * sub_fft_size], _mm_add_pd(tmp5r, rot5r));
            _mm_storeu_pd(&out_im[k0 + 5 * sub_fft_size], _mm_add_pd(tmp5i, rot5i));
            _mm_storeu_pd(&out_re[k0 + 6 * sub_fft_size], _mm_sub_pd(tmp5r, rot5r));
            _mm_storeu_pd(&out_im[k0 + 6 * sub_fft_size], _mm_sub_pd(tmp5i, rot5i));

            // Copy k=2,3 results from out_re/out_im to output_buffer
            for (int i = 0; i < 11; i++)
            {
                output_buffer[i * sub_fft_size + k0].re = out_re[i * sub_fft_size + k0];
                output_buffer[i * sub_fft_size + k0].im = out_im[i * sub_fft_size + k0];
                output_buffer[i * sub_fft_size + k0 + 1].re = out_re[i * sub_fft_size + k0 + 1];
                output_buffer[i * sub_fft_size + k0 + 1].im = out_im[i * sub_fft_size + k0 + 1];
            }
        }
        // Step 8c: Scalar computation for k=2 when sub_fft_size == 3
        else if (sub_fft_size == 3)
        {
            int k = 2;
            fft_data *X[11];
            for (int i = 0; i < 11; i++)
                X[i] = &output_buffer[i * sub_fft_size + k];

            fft_type a_r = out_re[k], a_i = out_im[k];
            fft_type b_r = out_re[k + sub_fft_size], b_i = out_im[k + sub_fft_size];
            fft_type c_r = out_re[k + 2 * sub_fft_size], c_i = out_im[k + 2 * sub_fft_size];
            fft_type d_r = out_re[k + 3 * sub_fft_size], d_i = out_im[k + 3 * sub_fft_size];
            fft_type e_r = out_re[k + 4 * sub_fft_size], e_i = out_im[k + 4 * sub_fft_size];
            fft_type f_r = out_re[k + 5 * sub_fft_size], f_i = out_im[k + 5 * sub_fft_size];
            fft_type g_r = out_re[k + 6 * sub_fft_size], g_i = out_im[k + 6 * sub_fft_size];
            fft_type h_r = out_re[k + 7 * sub_fft_size], h_i = out_im[k + 7 * sub_fft_size];
            fft_type i_r = out_re[k + 8 * sub_fft_size], i_i = out_im[k + 8 * sub_fft_size];
            fft_type j_r = out_re[k + 9 * sub_fft_size], j_i = out_im[k + 9 * sub_fft_size];
            fft_type k_r = out_re[k + 10 * sub_fft_size], k_i = out_im[k + 10 * sub_fft_size];

            int idx = 10 * k;
            fft_type w1r = twiddle_factors[idx + 0].re, w1i = twiddle_factors[idx + 0].im;
            fft_type w2r = twiddle_factors[idx + 1].re, w2i = twiddle_factors[idx + 1].im;
            fft_type w3r = twiddle_factors[idx + 2].re, w3i = twiddle_factors[idx + 2].im;
            fft_type w4r = twiddle_factors[idx + 3].re, w4i = twiddle_factors[idx + 3].im;
            fft_type w5r = twiddle_factors[idx + 4].re, w5i = twiddle_factors[idx + 4].im;
            fft_type w6r = twiddle_factors[idx + 5].re, w6i = twiddle_factors[idx + 5].im;
            fft_type w7r = twiddle_factors[idx + 6].re, w7i = twiddle_factors[idx + 6].im;
            fft_type w8r = twiddle_factors[idx + 7].re, w8i = twiddle_factors[idx + 7].im;
            fft_type w9r = twiddle_factors[idx + 8].re, w9i = twiddle_factors[idx + 8].im;
            fft_type w10r = twiddle_factors[idx + 9].re, w10i = twiddle_factors[idx + 9].im;

            fft_type b2_r = b_r * w1r - b_i * w1i, b2_i = b_i * w1r + b_r * w1i;
            fft_type c2_r = c_r * w2r - c_i * w2i, c2_i = c_i * w2r + c_r * w2i;
            fft_type d2_r = d_r * w3r - d_i * w3i, d2_i = d_i * w3r + d_r * w3i;
            fft_type e2_r = e_r * w4r - e_i * w4i, e2_i = e_i * w4r + e_r * w4i;
            fft_type f2_r = f_r * w5r - f_i * w5i, f2_i = f_i * w5r + f_r * w5i;
            fft_type g2_r = g_r * w6r - g_i * w6i, g2_i = g_i * w6r + g_r * w6i;
            fft_type h2_r = h_r * w7r - h_i * w7i, h2_i = h_i * w7r + h_r * w7i;
            fft_type i2_r = i_r * w8r - i_i * w8i, i2_i = i_i * w8r + i_r * w8i;
            fft_type j2_r = j_r * w9r - j_i * w9i, j2_i = j_i * w9r + j_r * w9i;
            fft_type k2_r = k_r * w10r - k_i * w10i, k2_i = k_i * w10r + k_r * w10i;

            fft_type t0_r = b2_r + k2_r, t0_i = b2_i + k2_i;
            fft_type t1_r = c2_r + j2_r, t1_i = c2_i + j2_i;
            fft_type t2_r = d2_r + i2_r, t2_i = d2_i + i2_i;
            fft_type t3_r = e2_r + h2_r, t3_i = e2_i + h2_i;
            fft_type t4_r = f2_r + g2_r, t4_i = f2_i + g2_i;
            fft_type t5_r = b2_r - k2_r, t5_i = b2_i - k2_i;
            fft_type t6_r = c2_r - j2_r, t6_i = c2_i - j2_i;
            fft_type t7_r = d2_r - i2_r, t7_i = d2_i - i2_i;
            fft_type t8_r = e2_r - h2_r, t8_i = e2_i - h2_i;
            fft_type t9_r = f2_r - g2_r, t9_i = f2_i - g2_i;

            out_re[k] = a_r + t0_r + t1_r + t2_r + t3_r + t4_r;
            out_im[k] = a_i + t0_i + t1_i + t2_i + t3_i + t4_i;

            fft_type tmp_r = a_r + C11_1 * t0_r + C11_2 * t1_r + C11_3 * t2_r + C11_4 * t3_r + C11_5 * t4_r;
            fft_type tmp_i = a_i + C11_1 * t0_i + C11_2 * t1_i + C11_3 * t2_i + C11_4 * t3_i + C11_5 * t4_i;
            fft_type rot_r = transform_sign * (S11_1 * t5_i + S11_2 * t6_i + S11_3 * t7_i + S11_4 * t8_i + S11_5 * t9_i);
            fft_type rot_i = transform_sign * (-S11_1 * t5_r - S11_2 * t6_r - S11_3 * t7_r - S11_4 * t8_r - S11_5 * t9_r);
            out_re[k + sub_fft_size] = tmp_r + rot_r;
            out_im[k + sub_fft_size] = tmp_i + rot_i;
            out_re[k + 10 * sub_fft_size] = tmp_r - rot_r;
            out_im[k + 10 * sub_fft_size] = tmp_i - rot_i;

            tmp_r = a_r + C11_2 * t0_r + C11_4 * t1_r + C11_5 * t2_r + C11_3 * t3_r + C11_1 * t4_r;
            tmp_i = a_i + C11_2 * t0_i + C11_4 * t1_i + C11_5 * t2_i + C11_3 * t3_i + C11_1 * t4_i;
            rot_r = transform_sign * (S11_2 * t5_i + S11_4 * t6_i + S11_5 * t7_i + S11_3 * t8_i + S11_1 * t9_i);
            rot_i = transform_sign * (-S11_2 * t5_r - S11_4 * t6_r - S11_5 * t7_r - S11_3 * t8_r - S11_1 * t9_r);
            out_re[k + 2 * sub_fft_size] = tmp_r + rot_r;
            out_im[k + 2 * sub_fft_size] = tmp_i + rot_i;
            out_re[k + 9 * sub_fft_size] = tmp_r - rot_r;
            out_im[k + 9 * sub_fft_size] = tmp_i - rot_i;

            tmp_r = a_r + C11_3 * t0_r + C11_5 * t1_r + C11_2 * t2_r + C11_1 * t3_r + C11_4 * t4_r;
            tmp_i = a_i + C11_3 * t0_i + C11_5 * t1_i + C11_2 * t2_i + C11_1 * t3_i + C11_4 * t4_i;
            rot_r = transform_sign * (S11_3 * t5_i + S11_5 * t6_i + S11_2 * t7_i + S11_1 * t8_i + S11_4 * t9_i);
            rot_i = transform_sign * (-S11_3 * t5_r - S11_5 * t6_r - S11_2 * t7_r - S11_1 * t8_r - S11_4 * t9_r);
            out_re[k + 3 * sub_fft_size] = tmp_r + rot_r;
            out_im[k + 3 * sub_fft_size] = tmp_i + rot_i;
            out_re[k + 8 * sub_fft_size] = tmp_r - rot_r;
            out_im[k + 8 * sub_fft_size] = tmp_i - rot_i;

            tmp_r = a_r + C11_4 * t0_r + C11_3 * t1_r + C11_1 * t2_r + C11_5 * t3_r + C11_2 * t4_r;
            tmp_i = a_i + C11_4 * t0_i + C11_3 * t1_i + C11_1 * t2_i + C11_5 * t3_i + C11_2 * t4_i;
            rot_r = transform_sign * (S11_4 * t5_i + S11_3 * t6_i + S11_1 * t7_i + S11_5 * t8_i + S11_2 * t9_i);
            rot_i = transform_sign * (-S11_4 * t5_r - S11_3 * t6_r - S11_1 * t7_r - S11_5 * t8_r - S11_2 * t9_r);
            out_re[k + 4 * sub_fft_size] = tmp_r + rot_r;
            out_im[k + 4 * sub_fft_size] = tmp_i + rot_i;
            out_re[k + 7 * sub_fft_size] = tmp_r - rot_r;
            out_im[k + 7 * sub_fft_size] = tmp_i - rot_i;

            tmp_r = a_r + C11_5 * t0_r + C11_1 * t1_r + C11_4 * t2_r + C11_2 * t3_r + C11_3 * t4_r;
            tmp_i = a_i + C11_5 * t0_i + C11_1 * t1_i + C11_4 * t2_i + C11_2 * t3_i + C11_3 * t4_i;
            rot_r = transform_sign * (S11_5 * t5_i + S11_1 * t6_i + S11_4 * t7_i + S11_2 * t8_i + S11_3 * t9_i);
            rot_i = transform_sign * (-S11_5 * t5_r - S11_1 * t6_r - S11_4 * t7_r - S11_2 * t8_r - S11_3 * t9_r);
            out_re[k + 5 * sub_fft_size] = tmp_r + rot_r;
            out_im[k + 5 * sub_fft_size] = tmp_i + rot_i;
            out_re[k + 6 * sub_fft_size] = tmp_r - rot_r;
            out_im[k + 6 * sub_fft_size] = tmp_i - rot_i;

            // Copy k=2 results from out_re/out_im to output_buffer
            for (int i = 0; i < 11; i++)
            {
                output_buffer[i * sub_fft_size + k].re = out_re[i * sub_fft_size + k];
                output_buffer[i * sub_fft_size + k].im = out_im[i * sub_fft_size + k];
            }
        }

        // Step 9: AVX2 vectorized butterfly for k=4 to sub_fft_size-1
        __m256d vsign = _mm256_set1_pd((double)transform_sign);
        __m256d vc11_1 = _mm256_set1_pd(C11_1);
        __m256d vc11_2 = _mm256_set1_pd(C11_2);
        __m256d vc11_3 = _mm256_set1_pd(C11_3);
        __m256d vc11_4 = _mm256_set1_pd(C11_4);
        __m256d vc11_5 = _mm256_set1_pd(C11_5);
        __m256d vs11_1 = _mm256_set1_pd(S11_1);
        __m256d vs11_2 = _mm256_set1_pd(S11_2);
        __m256d vs11_3 = _mm256_set1_pd(S11_3);
        __m256d vs11_4 = _mm256_set1_pd(S11_4);
        __m256d vs11_5 = _mm256_set1_pd(S11_5);
        int k = 4;
        for (; k + 3 < sub_fft_size; k += 4)
        {
            __m256d a_r = LOAD_PD(out_re + k);
            __m256d a_i = LOAD_PD(out_im + k);
            __m256d b_r = LOAD_PD(out_re + k + sub_fft_size);
            __m256d b_i = LOAD_PD(out_im + k + sub_fft_size);
            __m256d c_r = LOAD_PD(out_re + k + 2 * sub_fft_size);
            __m256d c_i = LOAD_PD(out_im + k + 2 * sub_fft_size);
            __m256d d_r = LOAD_PD(out_re + k + 3 * sub_fft_size);
            __m256d d_i = LOAD_PD(out_im + k + 3 * sub_fft_size);
            __m256d e_r = LOAD_PD(out_re + k + 4 * sub_fft_size);
            __m256d e_i = LOAD_PD(out_im + k + 4 * sub_fft_size);
            __m256d f_r = LOAD_PD(out_re + k + 5 * sub_fft_size);
            __m256d f_i = LOAD_PD(out_im + k + 5 * sub_fft_size);
            __m256d g_r = LOAD_PD(out_re + k + 6 * sub_fft_size);
            __m256d g_i = LOAD_PD(out_im + k + 6 * sub_fft_size);
            __m256d h_r = LOAD_PD(out_re + k + 7 * sub_fft_size);
            __m256d h_i = LOAD_PD(out_im + k + 7 * sub_fft_size);
            __m256d i_r = LOAD_PD(out_re + k + 8 * sub_fft_size);
            __m256d i_i = LOAD_PD(out_im + k + 8 * sub_fft_size);
            __m256d j_r = LOAD_PD(out_re + k + 9 * sub_fft_size);
            __m256d j_i = LOAD_PD(out_im + k + 9 * sub_fft_size);
            __m256d k_r = LOAD_PD(out_re + k + 10 * sub_fft_size);
            __m256d k_i = LOAD_PD(out_im + k + 10 * sub_fft_size);

            int idx = 10 * k;
            __m256d w1r = LOADU_PD(&twiddle_factors[idx + 0].re);
            __m256d w1i = LOADU_PD(&twiddle_factors[idx + 0].im);
            __m256d w2r = LOADU_PD(&twiddle_factors[idx + 1].re);
            __m256d w2i = LOADU_PD(&twiddle_factors[idx + 1].im);
            __m256d w3r = LOADU_PD(&twiddle_factors[idx + 2].re);
            __m256d w3i = LOADU_PD(&twiddle_factors[idx + 2].im);
            __m256d w4r = LOADU_PD(&twiddle_factors[idx + 3].re);
            __m256d w4i = LOADU_PD(&twiddle_factors[idx + 3].im);
            __m256d w5r = LOADU_PD(&twiddle_factors[idx + 4].re);
            __m256d w5i = LOADU_PD(&twiddle_factors[idx + 4].im);
            __m256d w6r = LOADU_PD(&twiddle_factors[idx + 5].re);
            __m256d w6i = LOADU_PD(&twiddle_factors[idx + 5].im);
            __m256d w7r = LOADU_PD(&twiddle_factors[idx + 6].re);
            __m256d w7i = LOADU_PD(&twiddle_factors[idx + 6].im);
            __m256d w8r = LOADU_PD(&twiddle_factors[idx + 7].re);
            __m256d w8i = LOADU_PD(&twiddle_factors[idx + 7].im);
            __m256d w9r = LOADU_PD(&twiddle_factors[idx + 8].re);
            __m256d w9i = LOADU_PD(&twiddle_factors[idx + 8].im);
            __m256d w10r = LOADU_PD(&twiddle_factors[idx + 9].re);
            __m256d w10i = LOADU_PD(&twiddle_factors[idx + 9].im);

            __m256d b2r = FMSUB(b_r, w1r, _mm256_mul_pd(b_i, w1i));
            __m256d b2i = FMADD(b_i, w1r, _mm256_mul_pd(b_r, w1i));
            __m256d c2r = FMSUB(c_r, w2r, _mm256_mul_pd(c_i, w2i));
            __m256d c2i = FMADD(c_i, w2r, _mm256_mul_pd(c_r, w2i));
            __m256d d2r = FMSUB(d_r, w3r, _mm256_mul_pd(d_i, w3i));
            __m256d d2i = FMADD(d_i, w3r, _mm256_mul_pd(d_r, w3i));
            __m256d e2r = FMSUB(e_r, w4r, _mm256_mul_pd(e_i, w4i));
            __m256d e2i = FMADD(e_i, w4r, _mm256_mul_pd(e_r, w4i));
            __m256d f2r = FMSUB(f_r, w5r, _mm256_mul_pd(f_i, w5i));
            __m256d f2i = FMADD(f_i, w5r, _mm256_mul_pd(f_r, w5i));
            __m256d g2r = FMSUB(g_r, w6r, _mm256_mul_pd(g_i, w6i));
            __m256d g2i = FMADD(g_i, w6r, _mm256_mul_pd(g_r, w6i));
            __m256d h2r = FMSUB(h_r, w7r, _mm256_mul_pd(h_i, w7i));
            __m256d h2i = FMADD(h_i, w7r, _mm256_mul_pd(h_r, w7i));
            __m256d i2r = FMSUB(i_r, w8r, _mm256_mul_pd(i_i, w8i));
            __m256d i2i = FMADD(i_i, w8r, _mm256_mul_pd(i_r, w8i));
            __m256d j2r = FMSUB(j_r, w9r, _mm256_mul_pd(j_i, w9i));
            __m256d j2i = FMADD(j_i, w9r, _mm256_mul_pd(j_r, w9i));
            __m256d k2r = FMSUB(k_r, w10r, _mm256_mul_pd(k_i, w10i));
            __m256d k2i = FMADD(k_i, w10r, _mm256_mul_pd(k_r, w10i));

            __m256d t0r = _mm256_add_pd(b2r, k2r);
            __m256d t0i = _mm256_add_pd(b2i, k2i);
            __m256d t1r = _mm256_add_pd(c2r, j2r);
            __m256d t1i = _mm256_add_pd(c2i, j2i);
            __m256d t2r = _mm256_add_pd(d2r, i2r);
            __m256d t2i = _mm256_add_pd(d2i, i2i);
            __m256d t3r = _mm256_add_pd(e2r, h2r);
            __m256d t3i = _mm256_add_pd(e2i, h2i);
            __m256d t4r = _mm256_add_pd(f2r, g2r);
            __m256d t4i = _mm256_add_pd(f2i, g2i);
            __m256d t5r = _mm256_sub_pd(b2r, k2r);
            __m256d t5i = _mm256_sub_pd(b2i, k2i);
            __m256d t6r = _mm256_sub_pd(c2r, j2r);
            __m256d t6i = _mm256_sub_pd(c2i, j2i);
            __m256d t7r = _mm256_sub_pd(d2r, i2r);
            __m256d t7i = _mm256_sub_pd(d2i, i2i);
            __m256d t8r = _mm256_sub_pd(e2r, h2r);
            __m256d t8i = _mm256_sub_pd(e2i, h2i);
            __m256d t9r = _mm256_sub_pd(f2r, g2r);
            __m256d t9i = _mm256_sub_pd(f2i, g2i);

            __m256d sum_r = _mm256_add_pd(t0r, _mm256_add_pd(t1r, _mm256_add_pd(t2r, _mm256_add_pd(t3r, t4r))));
            __m256d sum_i = _mm256_add_pd(t0i, _mm256_add_pd(t1i, _mm256_add_pd(t2i, _mm256_add_pd(t3i, t4i))));
            STORE_PD(&output_buffer[k].re, _mm256_add_pd(a_r, sum_r));
            STORE_PD(&output_buffer[k].im, _mm256_add_pd(a_i, sum_i));

            __m256d tmp1r = FMADD(vc11_5, t4r,
                                  FMADD(vc11_4, t3r,
                                        FMADD(vc11_3, t2r,
                                              FMADD(vc11_2, t1r,
                                                    FMADD(vc11_1, t0r, a_r)))));

            __m256d tmp1i = FMADD(vc11_5, t4i,
                                  FMADD(vc11_4, t3i,
                                        FMADD(vc11_3, t2i,
                                              FMADD(vc11_2, t1i,
                                                    FMADD(vc11_1, t0i, a_i)))));

            __m256d rot1r = FMADD(vs11_5, t9i,
                                  FMADD(vs11_4, t8i,
                                        FMADD(vs11_3, t7i,
                                              FMADD(vs11_2, t6i,
                                                    _mm256_mul_pd(vs11_1, t5i)))));

            __m256d rot1i = FMADD(vs11_5, t9r,
                                  FMADD(vs11_4, t8r,
                                        FMADD(vs11_3, t7r,
                                              FMADD(vs11_2, t6r,
                                                    _mm256_mul_pd(vs11_1, t5r)))));
            rot1r = _mm256_mul_pd(vsign, rot1r);
            rot1i = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), vsign), rot1i);
            STORE_PD(&output_buffer[k + sub_fft_size].re, _mm256_add_pd(tmp1r, rot1r));
            STORE_PD(&output_buffer[k + sub_fft_size].im, _mm256_add_pd(tmp1i, rot1i));
            STORE_PD(&output_buffer[k + 10 * sub_fft_size].re, _mm256_sub_pd(tmp1r, rot1r));
            STORE_PD(&output_buffer[k + 10 * sub_fft_size].im, _mm256_sub_pd(tmp1i, rot1i));

            __m256d tmp2r = FMADD(vc11_1, t4r,
                                  FMADD(vc11_3, t3r,
                                        FMADD(vc11_5, t2r,
                                              FMADD(vc11_4, t1r,
                                                    FMADD(vc11_2, t0r, a_r)))));

            __m256d tmp2i = FMADD(vc11_1, t4i,
                                  FMADD(vc11_3, t3i,
                                        FMADD(vc11_5, t2i,
                                              FMADD(vc11_4, t1i,
                                                    FMADD(vc11_2, t0i, a_i)))));

            __m256d rot2r = FMADD(vs11_1, t9i,
                                  FMADD(vs11_3, t8i,
                                        FMADD(vs11_5, t7i,
                                              FMADD(vs11_4, t6i,
                                                    _mm256_mul_pd(vs11_2, t5i)))));

            __m256d rot2i = FMADD(vs11_1, t9r,
                                  FMADD(vs11_3, t8r,
                                        FMADD(vs11_5, t7r,
                                              FMADD(vs11_4, t6r,
                                                    _mm256_mul_pd(vs11_2, t5r)))));
            rot2r = _mm256_mul_pd(vsign, rot2r);
            rot2i = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), vsign), rot2i);
            STORE_PD(&output_buffer[k + 2 * sub_fft_size].re, _mm256_add_pd(tmp2r, rot2r));
            STORE_PD(&output_buffer[k + 2 * sub_fft_size].im, _mm256_add_pd(tmp2i, rot2i));
            STORE_PD(&output_buffer[k + 9 * sub_fft_size].re, _mm256_sub_pd(tmp2r, rot2r));
            STORE_PD(&output_buffer[k + 9 * sub_fft_size].im, _mm256_sub_pd(tmp2i, rot2i));

            __m256d tmp3r = FMADD(vc11_4, t4r,
                                  FMADD(vc11_1, t3r,
                                        FMADD(vc11_2, t2r,
                                              FMADD(vc11_5, t1r,
                                                    FMADD(vc11_3, t0r, a_r)))));

            __m256d tmp3i = FMADD(vc11_4, t4i,
                                  FMADD(vc11_1, t3i,
                                        FMADD(vc11_2, t2i,
                                              FMADD(vc11_5, t1i,
                                                    FMADD(vc11_3, t0i, a_i)))));

            __m256d rot3r = FMADD(vs11_4, t9i,
                                  FMADD(vs11_1, t8i,
                                        FMADD(vs11_2, t7i,
                                              FMADD(vs11_5, t6i,
                                                    _mm256_mul_pd(vs11_3, t5i)))));

            __m256d rot3i = FMADD(vs11_4, t9r,
                                  FMADD(vs11_1, t8r,
                                        FMADD(vs11_2, t7r,
                                              FMADD(vs11_5, t6r,
                                                    _mm256_mul_pd(vs11_3, t5r)))));
            rot3r = _mm256_mul_pd(vsign, rot3r);
            rot3i = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), vsign), rot3i);
            STORE_PD(&output_buffer[k + 3 * sub_fft_size].re, _mm256_add_pd(tmp3r, rot3r));
            STORE_PD(&output_buffer[k + 3 * sub_fft_size].im, _mm256_add_pd(tmp3i, rot3i));
            STORE_PD(&output_buffer[k + 8 * sub_fft_size].re, _mm256_sub_pd(tmp3r, rot3r));
            STORE_PD(&output_buffer[k + 8 * sub_fft_size].im, _mm256_sub_pd(tmp3i, rot3i));

            __m256d tmp4r = FMADD(vc11_2, t4r,
                                  FMADD(vc11_5, t3r,
                                        FMADD(vc11_1, t2r,
                                              FMADD(vc11_3, t1r,
                                                    FMADD(vc11_4, t0r, a_r)))));

            __m256d tmp4i = FMADD(vc11_2, t4i,
                                  FMADD(vc11_5, t3i,
                                        FMADD(vc11_1, t2i,
                                              FMADD(vc11_3, t1i,
                                                    FMADD(vc11_4, t0i, a_i)))));

            __m256d rot4r = FMADD(vs11_2, t9i,
                                  FMADD(vs11_5, t8i,
                                        FMADD(vs11_1, t7i,
                                              FMADD(vs11_3, t6i,
                                                    _mm256_mul_pd(vs11_4, t5i)))));

            __m256d rot4i = FMADD(vs11_2, t9r,
                                  FMADD(vs11_5, t8r,
                                        FMADD(vs11_1, t7r,
                                              FMADD(vs11_3, t6r,
                                                    _mm256_mul_pd(vs11_4, t5r)))));
            rot4r = _mm256_mul_pd(vsign, rot4r);
            rot4i = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), vsign), rot4i);
            STORE_PD(&output_buffer[k + 4 * sub_fft_size].re, _mm256_add_pd(tmp4r, rot4r));
            STORE_PD(&output_buffer[k + 4 * sub_fft_size].im, _mm256_add_pd(tmp4i, rot4i));
            STORE_PD(&output_buffer[k + 7 * sub_fft_size].re, _mm256_sub_pd(tmp4r, rot4r));
            STORE_PD(&output_buffer[k + 7 * sub_fft_size].im, _mm256_sub_pd(tmp4i, rot4i));

            __m256d tmp5r = FMADD(vc11_3, t4r,
                                  FMADD(vc11_2, t3r,
                                        FMADD(vc11_4, t2r,
                                              FMADD(vc11_1, t1r,
                                                    FMADD(vc11_5, t0r, a_r)))));

            __m256d tmp5i = FMADD(vc11_3, t4i,
                                  FMADD(vc11_2, t3i,
                                        FMADD(vc11_4, t2i,
                                              FMADD(vc11_1, t1i,
                                                    FMADD(vc11_5, t0i, a_i)))));

            __m256d rot5r = FMADD(vs11_3, t9i,
                                  FMADD(vs11_2, t8i,
                                        FMADD(vs11_4, t7i,
                                              FMADD(vs11_1, t6i,
                                                    _mm256_mul_pd(vs11_5, t5i)))));

            __m256d rot5i = FMADD(vs11_3, t9r,
                                  FMADD(vs11_2, t8r,
                                        FMADD(vs11_4, t7r,
                                              FMADD(vs11_1, t6r,
                                                    _mm256_mul_pd(vs11_5, t5r)))));
            rot5r = _mm256_mul_pd(vsign, rot5r);
            rot5i = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), vsign), rot5i);
            STORE_PD(&output_buffer[k + 5 * sub_fft_size].re, _mm256_add_pd(tmp5r, rot5r));
            STORE_PD(&output_buffer[k + 5 * sub_fft_size].im, _mm256_add_pd(tmp5i, rot5i));
            STORE_PD(&output_buffer[k + 6 * sub_fft_size].re, _mm256_sub_pd(tmp5r, rot5r));
            STORE_PD(&output_buffer[k + 6 * sub_fft_size].im, _mm256_sub_pd(tmp5i, rot5i));
        }

        // Step 10: SSE2 vectorized tail for remaining k
        __m128d vsign_sse2 = _mm_set1_pd((double)transform_sign);
        __m128d vc11_1_sse2 = _mm_set1_pd(C11_1);
        __m128d vc11_2_sse2 = _mm_set1_pd(C11_2);
        __m128d vc11_3_sse2 = _mm_set1_pd(C11_3);
        __m128d vc11_4_sse2 = _mm_set1_pd(C11_4);
        __m128d vc11_5_sse2 = _mm_set1_pd(C11_5);
        __m128d vs11_1_sse2 = _mm_set1_pd(S11_1);
        __m128d vs11_2_sse2 = _mm_set1_pd(S11_2);
        __m128d vs11_3_sse2 = _mm_set1_pd(S11_3);
        __m128d vs11_4_sse2 = _mm_set1_pd(S11_4);
        __m128d vs11_5_sse2 = _mm_set1_pd(S11_5);
        for (; k + 1 < sub_fft_size; k += 2)
        {
            __m128d a_r = LOAD_SSE2(out_re + k);
            __m128d a_i = LOAD_SSE2(out_im + k);
            __m128d b_r = LOAD_SSE2(out_re + k + sub_fft_size);
            __m128d b_i = LOAD_SSE2(out_im + k + sub_fft_size);
            __m128d c_r = LOAD_SSE2(out_re + k + 2 * sub_fft_size);
            __m128d c_i = LOAD_SSE2(out_im + k + 2 * sub_fft_size);
            __m128d d_r = LOAD_SSE2(out_re + k + 3 * sub_fft_size);
            __m128d d_i = LOAD_SSE2(out_im + k + 3 * sub_fft_size);
            __m128d e_r = LOAD_SSE2(out_re + k + 4 * sub_fft_size);
            __m128d e_i = LOAD_SSE2(out_im + k + 4 * sub_fft_size);
            __m128d f_r = LOAD_SSE2(out_re + k + 5 * sub_fft_size);
            __m128d f_i = LOAD_SSE2(out_im + k + 5 * sub_fft_size);
            __m128d g_r = LOAD_SSE2(out_re + k + 6 * sub_fft_size);
            __m128d g_i = LOAD_SSE2(out_im + k + 6 * sub_fft_size);
            __m128d h_r = LOAD_SSE2(out_re + k + 7 * sub_fft_size);
            __m128d h_i = LOAD_SSE2(out_im + k + 7 * sub_fft_size);
            __m128d i_r = LOAD_SSE2(out_re + k + 8 * sub_fft_size);
            __m128d i_i = LOAD_SSE2(out_im + k + 8 * sub_fft_size);
            __m128d j_r = LOAD_SSE2(out_re + k + 9 * sub_fft_size);
            __m128d j_i = LOAD_SSE2(out_im + k + 9 * sub_fft_size);
            __m128d k_r = LOAD_SSE2(out_re + k + 10 * sub_fft_size);
            __m128d k_i = LOAD_SSE2(out_im + k + 10 * sub_fft_size);

            int idx = 10 * k;
            __m128d w1r = LOADU_SSE2(&twiddle_factors[idx + 0].re);
            __m128d w1i = LOADU_SSE2(&twiddle_factors[idx + 0].im);
            __m128d w2r = LOADU_SSE2(&twiddle_factors[idx + 1].re);
            __m128d w2i = LOADU_SSE2(&twiddle_factors[idx + 1].im);
            __m128d w3r = LOADU_SSE2(&twiddle_factors[idx + 2].re);
            __m128d w3i = LOADU_SSE2(&twiddle_factors[idx + 2].im);
            __m128d w4r = LOADU_SSE2(&twiddle_factors[idx + 3].re);
            __m128d w4i = LOADU_SSE2(&twiddle_factors[idx + 3].im);
            __m128d w5r = LOADU_SSE2(&twiddle_factors[idx + 4].re);
            __m128d w5i = LOADU_SSE2(&twiddle_factors[idx + 4].im);
            __m128d w6r = LOADU_SSE2(&twiddle_factors[idx + 5].re);
            __m128d w6i = LOADU_SSE2(&twiddle_factors[idx + 5].im);
            __m128d w7r = LOADU_SSE2(&twiddle_factors[idx + 6].re);
            __m128d w7i = LOADU_SSE2(&twiddle_factors[idx + 6].im);
            __m128d w8r = LOADU_SSE2(&twiddle_factors[idx + 7].re);
            __m128d w8i = LOADU_SSE2(&twiddle_factors[idx + 7].im);
            __m128d w9r = LOADU_SSE2(&twiddle_factors[idx + 8].re);
            __m128d w9i = LOADU_SSE2(&twiddle_factors[idx + 8].im);
            __m128d w10r = LOADU_SSE2(&twiddle_factors[idx + 9].re);
            __m128d w10i = LOADU_SSE2(&twiddle_factors[idx + 9].im);

            __m128d b2r = FMSUB_SSE2(b_r, w1r, _mm_mul_pd(b_i, w1i));
            __m128d b2i = FMADD_SSE2(b_i, w1r, _mm_mul_pd(b_r, w1i));
            __m128d c2r = FMSUB_SSE2(c_r, w2r, _mm_mul_pd(c_i, w2i));
            __m128d c2i = FMADD_SSE2(c_i, w2r, _mm_mul_pd(c_r, w2i));
            __m128d d2r = FMSUB_SSE2(d_r, w3r, _mm_mul_pd(d_i, w3i));
            __m128d d2i = FMADD_SSE2(d_i, w3r, _mm_mul_pd(d_r, w3i));
            __m128d e2r = FMSUB_SSE2(e_r, w4r, _mm_mul_pd(e_i, w4i));
            __m128d e2i = FMADD_SSE2(e_i, w4r, _mm_mul_pd(e_r, w4i));
            __m128d f2r = FMSUB_SSE2(f_r, w5r, _mm_mul_pd(f_i, w5i));
            __m128d f2i = FMADD_SSE2(f_i, w5r, _mm_mul_pd(f_r, w5i));
            __m128d g2r = FMSUB_SSE2(g_r, w6r, _mm_mul_pd(g_i, w6i));
            __m128d g2i = FMADD_SSE2(g_i, w6r, _mm_mul_pd(g_r, w6i));
            __m128d h2r = FMSUB_SSE2(h_r, w7r, _mm_mul_pd(h_i, w7i));
            __m128d h2i = FMADD_SSE2(h_i, w7r, _mm_mul_pd(h_r, w7i));
            __m128d i2r = FMSUB_SSE2(i_r, w8r, _mm_mul_pd(i_i, w8i));
            __m128d i2i = FMADD_SSE2(i_i, w8r, _mm_mul_pd(i_r, w8i));
            __m128d j2r = FMSUB_SSE2(j_r, w9r, _mm_mul_pd(j_i, w9i));
            __m128d j2i = FMADD_SSE2(j_i, w9r, _mm_mul_pd(j_r, w9i));
            __m128d k2r = FMSUB_SSE2(k_r, w10r, _mm_mul_pd(k_i, w10i));
            __m128d k2i = FMADD_SSE2(k_i, w10r, _mm_mul_pd(k_r, w10i));

            __m128d t0r = _mm_add_pd(b2r, k2r);
            __m128d t0i = _mm_add_pd(b2i, k2i);
            __m128d t1r = _mm_add_pd(c2r, j2r);
            __m128d t1i = _mm_add_pd(c2i, j2i);
            __m128d t2r = _mm_add_pd(d2r, i2r);
            __m128d t2i = _mm_add_pd(d2i, i2i);
            __m128d t3r = _mm_add_pd(e2r, h2r);
            __m128d t3i = _mm_add_pd(e2i, h2i);
            __m128d t4r = _mm_add_pd(f2r, g2r);
            __m128d t4i = _mm_add_pd(f2i, g2i);
            __m128d t5r = _mm_sub_pd(b2r, k2r);
            __m128d t5i = _mm_sub_pd(b2i, k2i);
            __m128d t6r = _mm_sub_pd(c2r, j2r);
            __m128d t6i = _mm_sub_pd(c2i, j2i);
            __m128d t7r = _mm_sub_pd(d2r, i2r);
            __m128d t7i = _mm_sub_pd(d2i, i2i);
            __m128d t8r = _mm_sub_pd(e2r, h2r);
            __m128d t8i = _mm_sub_pd(e2i, h2i);
            __m128d t9r = _mm_sub_pd(f2r, g2r);
            __m128d t9i = _mm_sub_pd(f2i, g2i);

            __m128d sum_r = _mm_add_pd(t0r, _mm_add_pd(t1r, _mm_add_pd(t2r, _mm_add_pd(t3r, t4r))));
            __m128d sum_i = _mm_add_pd(t0i, _mm_add_pd(t1i, _mm_add_pd(t2i, _mm_add_pd(t3i, t4i))));
            STORE_SSE2(&output_buffer[k].re, _mm_add_pd(a_r, sum_r));
            STORE_SSE2(&output_buffer[k].im, _mm_add_pd(a_i, sum_i));

            __m128d tmp1r = FMADD_SSE2(vc11_1_sse2, t0r,
                                       FMADD_SSE2(vc11_2_sse2, t1r,
                                                  FMADD_SSE2(vc11_3_sse2, t2r,
                                                             FMADD_SSE2(vc11_4_sse2, t3r,
                                                                        FMADD_SSE2(vc11_5_sse2, t4r, a_r)))));

            __m128d tmp1i = FMADD_SSE2(vc11_1_sse2, t0i,
                                       FMADD_SSE2(vc11_2_sse2, t1i,
                                                  FMADD_SSE2(vc11_3_sse2, t2i,
                                                             FMADD_SSE2(vc11_4_sse2, t3i,
                                                                        FMADD_SSE2(vc11_5_sse2, t4i, a_i)))));

            __m128d rot1r = FMADD_SSE2(vs11_1_sse2, t5i,
                                       FMADD_SSE2(vs11_2_sse2, t6i,
                                                  FMADD_SSE2(vs11_3_sse2, t7i,
                                                             FMADD_SSE2(vs11_4_sse2, t8i,
                                                                        _mm_mul_pd(vs11_5_sse2, t9i)))));

            __m128d rot1i = FMADD_SSE2(vs11_1_sse2, t5r,
                                       FMADD_SSE2(vs11_2_sse2, t6r,
                                                  FMADD_SSE2(vs11_3_sse2, t7r,
                                                             FMADD_SSE2(vs11_4_sse2, t8r,
                                                                        _mm_mul_pd(vs11_5_sse2, t9r)))));
            rot1r = _mm_mul_pd(vsign_sse2, rot1r);
            rot1i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), vsign_sse2), rot1i);
            STORE_SSE2(&output_buffer[k + sub_fft_size].re, _mm_add_pd(tmp1r, rot1r));
            STORE_SSE2(&output_buffer[k + sub_fft_size].im, _mm_add_pd(tmp1i, rot1i));
            STORE_SSE2(&output_buffer[k + 10 * sub_fft_size].re, _mm_sub_pd(tmp1r, rot1r));
            STORE_SSE2(&output_buffer[k + 10 * sub_fft_size].im, _mm_sub_pd(tmp1i, rot1i));

            __m128d tmp2r = FMADD_SSE2(vc11_1_sse2, t0r,
                                       FMADD_SSE2(vc11_2_sse2, t4r,
                                                  FMADD_SSE2(vc11_3_sse2, t3r,
                                                             FMADD_SSE2(vc11_4_sse2, t1r,
                                                                        FMADD_SSE2(vc11_5_sse2, t2r, a_r)))));

            __m128d tmp2i = FMADD_SSE2(vc11_1_sse2, t0i,
                                       FMADD_SSE2(vc11_2_sse2, t4i,
                                                  FMADD_SSE2(vc11_3_sse2, t3i,
                                                             FMADD_SSE2(vc11_4_sse2, t1i,
                                                                        FMADD_SSE2(vc11_5_sse2, t2i, a_i)))));

            __m128d rot2r = FMADD_SSE2(vs11_1_sse2, t5i,
                                       FMADD_SSE2(vs11_2_sse2, t9i,
                                                  FMADD_SSE2(vs11_3_sse2, t8i,
                                                             FMADD_SSE2(vs11_4_sse2, t6i,
                                                                        _mm_mul_pd(vs11_5_sse2, t7i)))));

            __m128d rot2i = FMADD_SSE2(vs11_1_sse2, t5r,
                                       FMADD_SSE2(vs11_2_sse2, t9r,
                                                  FMADD_SSE2(vs11_3_sse2, t8r,
                                                             FMADD_SSE2(vs11_4_sse2, t6r,
                                                                        _mm_mul_pd(vs11_5_sse2, t7r)))));
            rot2r = _mm_mul_pd(vsign_sse2, rot2r);
            rot2i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), vsign_sse2), rot2i);
            STORE_SSE2(&output_buffer[k + 2 * sub_fft_size].re, _mm_add_pd(tmp2r, rot2r));
            STORE_SSE2(&output_buffer[k + 2 * sub_fft_size].im, _mm_add_pd(tmp2i, rot2i));
            STORE_SSE2(&output_buffer[k + 9 * sub_fft_size].re, _mm_sub_pd(tmp2r, rot2r));
            STORE_SSE2(&output_buffer[k + 9 * sub_fft_size].im, _mm_sub_pd(tmp2i, rot2i));

            __m128d tmp3r = FMADD_SSE2(vc11_1_sse2, t3r,
                                       FMADD_SSE2(vc11_2_sse2, t2r,
                                                  FMADD_SSE2(vc11_3_sse2, t0r,
                                                             FMADD_SSE2(vc11_4_sse2, t4r,
                                                                        FMADD_SSE2(vc11_5_sse2, t1r, a_r)))));

            __m128d tmp3i = FMADD_SSE2(vc11_1_sse2, t3i,
                                       FMADD_SSE2(vc11_2_sse2, t2i,
                                                  FMADD_SSE2(vc11_3_sse2, t0i,
                                                             FMADD_SSE2(vc11_4_sse2, t4i,
                                                                        FMADD_SSE2(vc11_5_sse2, t1i, a_i)))));

            __m128d rot3r = FMADD_SSE2(vs11_1_sse2, t8i,
                                       FMADD_SSE2(vs11_2_sse2, t7i,
                                                  FMADD_SSE2(vs11_3_sse2, t5i,
                                                             FMADD_SSE2(vs11_4_sse2, t9i,
                                                                        _mm_mul_pd(vs11_5_sse2, t6i)))));

            __m128d rot3i = FMADD_SSE2(vs11_1_sse2, t8r,
                                       FMADD_SSE2(vs11_2_sse2, t7r,
                                                  FMADD_SSE2(vs11_3_sse2, t5r,
                                                             FMADD_SSE2(vs11_4_sse2, t9r,
                                                                        _mm_mul_pd(vs11_5_sse2, t6r)))));
            rot3r = _mm_mul_pd(vsign_sse2, rot3r);
            rot3i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), vsign_sse2), rot3i);
            STORE_SSE2(&output_buffer[k + 3 * sub_fft_size].re, _mm_add_pd(tmp3r, rot3r));
            STORE_SSE2(&output_buffer[k + 3 * sub_fft_size].im, _mm_add_pd(tmp3i, rot3i));
            STORE_SSE2(&output_buffer[k + 8 * sub_fft_size].re, _mm_sub_pd(tmp3r, rot3r));
            STORE_SSE2(&output_buffer[k + 8 * sub_fft_size].im, _mm_sub_pd(tmp3i, rot3i));

            __m128d tmp4r = FMADD_SSE2(vc11_1_sse2, t2r,
                                       FMADD_SSE2(vc11_2_sse2, t4r,
                                                  FMADD_SSE2(vc11_3_sse2, t1r,
                                                             FMADD_SSE2(vc11_4_sse2, t0r,
                                                                        FMADD_SSE2(vc11_5_sse2, t3r, a_r)))));

            __m128d tmp4i = FMADD_SSE2(vc11_1_sse2, t2i,
                                       FMADD_SSE2(vc11_2_sse2, t4i,
                                                  FMADD_SSE2(vc11_3_sse2, t1i,
                                                             FMADD_SSE2(vc11_4_sse2, t0i,
                                                                        FMADD_SSE2(vc11_5_sse2, t3i, a_i)))));

            __m128d rot4r = FMADD_SSE2(vs11_1_sse2, t7i,
                                       FMADD_SSE2(vs11_2_sse2, t9i,
                                                  FMADD_SSE2(vs11_3_sse2, t6i,
                                                             FMADD_SSE2(vs11_4_sse2, t5i,
                                                                        _mm_mul_pd(vs11_5_sse2, t8i)))));

            __m128d rot4i = FMADD_SSE2(vs11_1_sse2, t7r,
                                       FMADD_SSE2(vs11_2_sse2, t9r,
                                                  FMADD_SSE2(vs11_3_sse2, t6r,
                                                             FMADD_SSE2(vs11_4_sse2, t5r,
                                                                        _mm_mul_pd(vs11_5_sse2, t8r)))));
            rot4r = _mm_mul_pd(vsign_sse2, rot4r);
            rot4i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), vsign_sse2), rot4i);
            STORE_SSE2(&output_buffer[k + 4 * sub_fft_size].re, _mm_add_pd(tmp4r, rot4r));
            STORE_SSE2(&output_buffer[k + 4 * sub_fft_size].im, _mm_add_pd(tmp4i, rot4i));
            STORE_SSE2(&output_buffer[k + 7 * sub_fft_size].re, _mm_sub_pd(tmp4r, rot4r));
            STORE_SSE2(&output_buffer[k + 7 * sub_fft_size].im, _mm_sub_pd(tmp4i, rot4i));

            __m128d tmp5r = FMADD_SSE2(vc11_1_sse2, t1r,
                                       FMADD_SSE2(vc11_2_sse2, t3r,
                                                  FMADD_SSE2(vc11_3_sse2, t4r,
                                                             FMADD_SSE2(vc11_4_sse2, t2r,
                                                                        FMADD_SSE2(vc11_5_sse2, t0r, a_r)))));

            __m128d tmp5i = FMADD_SSE2(vc11_1_sse2, t1i,
                                       FMADD_SSE2(vc11_2_sse2, t3i,
                                                  FMADD_SSE2(vc11_3_sse2, t4i,
                                                             FMADD_SSE2(vc11_4_sse2, t2i,
                                                                        FMADD_SSE2(vc11_5_sse2, t0i, a_i)))));

            __m128d rot5r = FMADD_SSE2(vs11_1_sse2, t6i,
                                       FMADD_SSE2(vs11_2_sse2, t8i,
                                                  FMADD_SSE2(vs11_3_sse2, t9i,
                                                             FMADD_SSE2(vs11_4_sse2, t7i,
                                                                        _mm_mul_pd(vs11_5_sse2, t5i)))));

            __m128d rot5i = FMADD_SSE2(vs11_1_sse2, t6r,
                                       FMADD_SSE2(vs11_2_sse2, t8r,
                                                  FMADD_SSE2(vs11_3_sse2, t9r,
                                                             FMADD_SSE2(vs11_4_sse2, t7r,
                                                                        _mm_mul_pd(vs11_5_sse2, t5r)))));
            rot5r = _mm_mul_pd(vsign_sse2, rot5r);
            rot5i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), vsign_sse2), rot5i);
            STORE_SSE2(&output_buffer[k + 5 * sub_fft_size].re, _mm_add_pd(tmp5r, rot5r));
            STORE_SSE2(&output_buffer[k + 5 * sub_fft_size].im, _mm_add_pd(tmp5i, rot5i));
            STORE_SSE2(&output_buffer[k + 6 * sub_fft_size].re, _mm_sub_pd(tmp5r, rot5r));
            STORE_SSE2(&output_buffer[k + 6 * sub_fft_size].im, _mm_sub_pd(tmp5i, rot5i));
        }

        for (; k < sub_fft_size; k++)
        {
            fft_data *X[11];
            for (int i = 0; i < 11; i++)
                X[i] = &output_buffer[i * sub_fft_size + k];

            fft_type a_r = out_re[k], a_i = out_im[k];
            fft_type b_r = out_re[k + sub_fft_size], b_i = out_im[k + sub_fft_size];
            fft_type c_r = out_re[k + 2 * sub_fft_size], c_i = out_im[k + 2 * sub_fft_size];
            fft_type d_r = out_re[k + 3 * sub_fft_size], d_i = out_im[k + 3 * sub_fft_size];
            fft_type e_r = out_re[k + 4 * sub_fft_size], e_i = out_im[k + 4 * sub_fft_size];
            fft_type f_r = out_re[k + 5 * sub_fft_size], f_i = out_im[k + 5 * sub_fft_size];
            fft_type g_r = out_re[k + 6 * sub_fft_size], g_i = out_im[k + 6 * sub_fft_size];
            fft_type h_r = out_re[k + 7 * sub_fft_size], h_i = out_im[k + 7 * sub_fft_size];
            fft_type i_r = out_re[k + 8 * sub_fft_size], i_i = out_im[k + 8 * sub_fft_size];
            fft_type j_r = out_re[k + 9 * sub_fft_size], j_i = out_im[k + 9 * sub_fft_size];
            fft_type k_r = out_re[k + 10 * sub_fft_size], k_i = out_im[k + 10 * sub_fft_size];

            int idx = 10 * k;
            fft_type w1r = twiddle_factors[idx + 0].re, w1i = twiddle_factors[idx + 0].im;
            fft_type w2r = twiddle_factors[idx + 1].re, w2i = twiddle_factors[idx + 1].im;
            fft_type w3r = twiddle_factors[idx + 2].re, w3i = twiddle_factors[idx + 2].im;
            fft_type w4r = twiddle_factors[idx + 3].re, w4i = twiddle_factors[idx + 3].im;
            fft_type w5r = twiddle_factors[idx + 4].re, w5i = twiddle_factors[idx + 4].im;
            fft_type w6r = twiddle_factors[idx + 5].re, w6i = twiddle_factors[idx + 5].im;
            fft_type w7r = twiddle_factors[idx + 6].re, w7i = twiddle_factors[idx + 6].im;
            fft_type w8r = twiddle_factors[idx + 7].re, w8i = twiddle_factors[idx + 7].im;
            fft_type w9r = twiddle_factors[idx + 8].re, w9i = twiddle_factors[idx + 8].im;
            fft_type w10r = twiddle_factors[idx + 9].re, w10i = twiddle_factors[idx + 9].im;

            fft_type b2_r = b_r * w1r - b_i * w1i, b2_i = b_i * w1r + b_r * w1i;
            fft_type c2_r = c_r * w2r - c_i * w2i, c2_i = c_i * w2r + c_r * w2i;
            fft_type d2_r = d_r * w3r - d_i * w3i, d2_i = d_i * w3r + d_r * w3i;
            fft_type e2_r = e_r * w4r - e_i * w4i, e2_i = e_i * w4r + e_r * w4i;
            fft_type f2_r = f_r * w5r - f_i * w5i, f2_i = f_i * w5r + f_r * w5i;
            fft_type g2_r = g_r * w6r - g_i * w6i, g2_i = g_i * w6r + g_r * w6i;
            fft_type h2_r = h_r * w7r - h_i * w7i, h2_i = h_i * w7r + h_r * w7i;
            fft_type i2_r = i_r * w8r - i_i * w8i, i2_i = i_i * w8r + i_r * w8i;
            fft_type j2_r = j_r * w9r - j_i * w9i, j2_i = j_i * w9r + j_r * w9i;
            fft_type k2_r = k_r * w10r - k_i * w10i, k2_i = k_i * w10r + k_r * w10i;

            fft_type t0_r = b2_r + k2_r, t0_i = b2_i + k2_i;
            fft_type t1_r = c2_r + j2_r, t1_i = c2_i + j2_i;
            fft_type t2_r = d2_r + i2_r, t2_i = d2_i + i2_i;
            fft_type t3_r = e2_r + h2_r, t3_i = e2_i + h2_i;
            fft_type t4_r = f2_r + g2_r, t4_i = f2_i + g2_i;
            fft_type t5_r = b2_r - k2_r, t5_i = b2_i - k2_i;
            fft_type t6_r = c2_r - j2_r, t6_i = c2_i - j2_i;
            fft_type t7_r = d2_r - i2_r, t7_i = d2_i - i2_i;
            fft_type t8_r = e2_r - h2_r, t8_i = e2_i - h2_i;
            fft_type t9_r = f2_r - g2_r, t9_i = f2_i - g2_i;

            X[0]->re = a_r + t0_r + t1_r + t2_r + t3_r + t4_r;
            X[0]->im = a_i + t0_i + t1_i + t2_i + t3_i + t4_i;

            fft_type tmp_r = a_r + C11_1 * t0_r + C11_2 * t1_r + C11_3 * t2_r + C11_4 * t3_r + C11_5 * t4_r;
            fft_type tmp_i = a_i + C11_1 * t0_i + C11_2 * t1_i + C11_3 * t2_i + C11_4 * t3_i + C11_5 * t4_i;
            fft_type rot_r = transform_sign * (S11_1 * t5_i + S11_2 * t6_i + S11_3 * t7_i + S11_4 * t8_i + S11_5 * t9_i);
            fft_type rot_i = transform_sign * (-S11_1 * t5_r - S11_2 * t6_r - S11_3 * t7_r - S11_4 * t8_r - S11_5 * t9_r);
            X[1]->re = tmp_r + rot_r;
            X[1]->im = tmp_i + rot_i;
            X[10]->re = tmp_r - rot_r;
            X[10]->im = tmp_i - rot_i;

            tmp_r = a_r + C11_2 * t0_r + C11_4 * t1_r + C11_5 * t2_r + C11_3 * t3_r + C11_1 * t4_r;
            tmp_i = a_i + C11_2 * t0_i + C11_4 * t1_i + C11_5 * t2_i + C11_3 * t3_i + C11_1 * t4_i;
            rot_r = transform_sign * (S11_2 * t5_i + S11_4 * t6_i + S11_5 * t7_i + S11_3 * t8_i + S11_1 * t9_i);
            rot_i = transform_sign * (-S11_2 * t5_r - S11_4 * t6_r - S11_5 * t7_r - S11_3 * t8_r - S11_1 * t9_r);
            X[2]->re = tmp_r + rot_r;
            X[2]->im = tmp_i + rot_i;
            X[9]->re = tmp_r - rot_r;
            X[9]->im = tmp_i - rot_i;

            tmp_r = a_r + C11_3 * t0_r + C11_5 * t1_r + C11_2 * t2_r + C11_1 * t3_r + C11_4 * t4_r;
            tmp_i = a_i + C11_3 * t0_i + C11_5 * t1_i + C11_2 * t2_i + C11_1 * t3_i + C11_4 * t4_i;
            rot_r = transform_sign * (S11_3 * t5_i + S11_5 * t6_i + S11_2 * t7_i + S11_1 * t8_i + S11_4 * t9_i);
            rot_i = transform_sign * (-S11_3 * t5_r - S11_5 * t6_r - S11_2 * t7_r - S11_1 * t8_r - S11_4 * t9_r);
            X[3]->re = tmp_r + rot_r;
            X[3]->im = tmp_i + rot_i;
            X[8]->re = tmp_r - rot_r;
            X[8]->im = tmp_i - rot_i;

            tmp_r = a_r + C11_4 * t0_r + C11_3 * t1_r + C11_1 * t2_r + C11_5 * t3_r + C11_2 * t4_r;
            tmp_i = a_i + C11_4 * t0_i + C11_3 * t1_i + C11_1 * t2_i + C11_5 * t3_i + C11_2 * t4_i;
            rot_r = transform_sign * (S11_4 * t5_i + S11_3 * t6_i + S11_1 * t7_i + S11_5 * t8_i + S11_2 * t9_i);
            rot_i = transform_sign * (-S11_4 * t5_r - S11_3 * t6_r - S11_1 * t7_r - S11_5 * t8_r - S11_2 * t9_r);
            X[4]->re = tmp_r + rot_r;
            X[4]->im = tmp_i + rot_i;
            X[7]->re = tmp_r - rot_r;
            X[7]->im = tmp_i - rot_i;

            tmp_r = a_r + C11_5 * t0_r + C11_1 * t1_r + C11_4 * t2_r + C11_2 * t3_r + C11_3 * t4_r;
            tmp_i = a_i + C11_5 * t0_i + C11_1 * t1_i + C11_4 * t2_i + C11_2 * t3_i + C11_3 * t4_i;
            rot_r = transform_sign * (S11_5 * t5_i + S11_1 * t6_i + S11_4 * t7_i + S11_2 * t8_i + S11_3 * t9_i);
            rot_i = transform_sign * (-S11_5 * t5_r - S11_1 * t6_r - S11_4 * t7_r - S11_2 * t8_r - S11_3 * t9_r);
            X[5]->re = tmp_r + rot_r;
            X[5]->im = tmp_i + rot_i;
            X[6]->re = tmp_r - rot_r;
            X[6]->im = tmp_i - rot_i;
        }
    }
    else
    {
        const int r = radix;           // current radix
        const int K = data_length / r; // sub-FFT length
        const int next_stride = r * stride;
        const int nst = r - 1; // #twiddle lanes (j=1..r-1)

        // Scratch layout: always need r*K outputs; plus stage twiddles if not precomputed
        const int need_this = (fft_obj->twiddle_factors && factor_index < fft_obj->num_precomputed_stages)
                                  ? (r * K)
                                  : (r * K + nst * K);

        if (scratch_offset + need_this > fft_obj->max_scratch_size)
        {
            /* fprintf(stderr,"general radix: scratch too small\n"); */
            return;
        }

        // Child outputs live here (lane-major, AoS)
        fft_data *sub_outputs = fft_obj->scratch + scratch_offset;

        // Twiddle block for this stage: k-major (for each k pack j=1..r-1)
        fft_data *stage_tw = NULL;
        int have_precomp = (fft_obj->twiddle_factors && factor_index < fft_obj->num_precomputed_stages);
        if (have_precomp)
        {
            stage_tw = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            stage_tw = sub_outputs + r * K; // carve after outputs
        }

        // Recurse r children (each writes K AoS complexes)
        for (int j = 0; j < r; ++j)
        {
            mixed_radix_dit_rec(
                sub_outputs + j * K,       // lane j destination
                input_buffer + j * stride, // lane j source
                fft_obj, transform_sign,
                K, next_stride, factor_index + 1,
                scratch_offset /* reuse deeper scratch serially */);
        }

        // Build per-stage twiddles if not precomputed:
        // stage_tw[nst*k + (j-1)] = W_N^{j*k}, with N = r*K and fft_obj->twiddles[m] = W_N^m
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

        // Optional: precompute W_r^m once (phasor steps for the size-r DFT across j)
        // step(m) = cos(2π m / r) - i * (sign) * sin(2π m / r)
        double *Wr = (double *)alloca(sizeof(double) * r);
        double *Wi = (double *)alloca(sizeof(double) * r);
        for (int m = 0; m < r; ++m)
        {
            double th = 2.0 * M_PI * (double)m / (double)r;
            Wr[m] = cos(th);
            Wi[m] = -(double)transform_sign * sin(th);
        }

        // -------- AVX2 4-wide SoA core --------
#if defined(__AVX2__)
        int k = 0;
        for (; k + 3 < K; k += 4)
        {
            // modest prefetch
            const int pfk = k + FFT_PREFETCH_DISTANCE;
            if (pfk < K)
            {
                for (int j = 0; j < r; ++j)
                    FFT_PREFETCH_AOS(&sub_outputs[j * K + pfk]);
            }

            // Gather A_j(k..k+3) into SoA, apply stage twiddle for j>=1
            // Store into Tr[j], Ti[j]
            // Use small static arrays; for very large r this still works fine (you can heap if you like).
            __m256d Tr_buf[64], Ti_buf[64];
            __m256d *Tr = Tr_buf, *Ti = Ti_buf;

            // j = 0 (no stage twiddle)
            {
                double aR[4], aI[4];
                deinterleave4_aos_to_soa(&sub_outputs[0 * K + k], aR, aI);
                Tr[0] = _mm256_loadu_pd(aR);
                Ti[0] = _mm256_loadu_pd(aI);
            }

            // j = 1..r-1 with twiddles
            for (int j = 1; j < r; ++j)
            {
                // A_j
                double aR[4], aI[4];
                deinterleave4_aos_to_soa(&sub_outputs[j * K + k], aR, aI);
                __m256d Ajr = _mm256_loadu_pd(aR);
                __m256d Aji = _mm256_loadu_pd(aI);

                // W_N^{j*(k..k+3)} (gather 4 AoS twiddles -> SoA)
                fft_data wAos[4] = {
                    stage_tw[nst * (k + 0) + (j - 1)],
                    stage_tw[nst * (k + 1) + (j - 1)],
                    stage_tw[nst * (k + 2) + (j - 1)],
                    stage_tw[nst * (k + 3) + (j - 1)]};
                double wR[4], wI[4];
                deinterleave4_aos_to_soa(wAos, wR, wI);
                __m256d Wjr = _mm256_loadu_pd(wR);
                __m256d Wji = _mm256_loadu_pd(wI);

                // T_j = A_j * W
                cmul_soa_avx(Ajr, Aji, Wjr, Wji, &Tr[j], &Ti[j]);
            }

            // For each output m=0..r-1: sum_j T_j * (step_m)^j   (T_0 = A_0)
            for (int m = 0; m < r; ++m)
            {
                __m256d sumr = Tr[0];
                __m256d sumi = Ti[0];

                if (m != 0)
                {
                    const double step_r = Wr[m];
                    const double step_i = Wi[m];
                    double ph_r = 1.0, ph_i = 0.0; // (step)^0

                    for (int j = 1; j < r; ++j)
                    {
                        // ph *= step
                        const double npr = ph_r * step_r - ph_i * step_i;
                        const double npi = ph_r * step_i + ph_i * step_r;
                        ph_r = npr;
                        ph_i = npi;

                        __m256d PHr = _mm256_set1_pd(ph_r);
                        __m256d PHi = _mm256_set1_pd(ph_i);
                        __m256d ar, ai;
                        cmul_soa_avx(Tr[j], Ti[j], PHr, PHi, &ar, &ai);
                        sumr = _mm256_add_pd(sumr, ar);
                        sumi = _mm256_add_pd(sumi, ai);
                    }
                }

                // SoA -> AoS, store 4 results into output lane m
                double XR[4], XI[4];
                _mm256_storeu_pd(XR, sumr);
                _mm256_storeu_pd(XI, sumi);
                interleave4_soa_to_aos(XR, XI, &output_buffer[m * K + k]);
            }
        }
#else
        int k = 0;
#endif

        // -------- SSE2 2-wide tail (k, k+1) --------
        for (; k + 1 < K; k += 2)
        {
            const int pfk = k + FFT_PREFETCH_DISTANCE;
            if (pfk < K)
            {
                for (int j = 0; j < r; ++j)
                    FFT_PREFETCH_AOS(&sub_outputs[j * K + pfk]);
            }

            // j=0
            double r2[2], i2[2];
            deinterleave2_aos_to_soa(&sub_outputs[0 * K + k], r2, i2);
            __m128d Tr0 = _mm_loadu_pd(r2);
            __m128d Ti0 = _mm_loadu_pd(i2);

            // Keep T_j arrays in SSE2
            // (If you want heap for very large r, you can switch, but stack handles typical r like 11, 13.)
            __m128d Tr2_buf[64], Ti2_buf[64];
            Tr2_buf[0] = Tr0;
            Ti2_buf[0] = Ti0;

            // j >= 1
            for (int j = 1; j < r; ++j)
            {
                deinterleave2_aos_to_soa(&sub_outputs[j * K + k], r2, i2);
                __m128d Ajr = _mm_loadu_pd(r2);
                __m128d Aji = _mm_loadu_pd(i2);

                fft_data wA[2] = {
                    stage_tw[nst * (k + 0) + (j - 1)],
                    stage_tw[nst * (k + 1) + (j - 1)]};
                __m128d Wjr = _mm_set_pd(wA[1].re, wA[0].re);
                __m128d Wji = _mm_set_pd(wA[1].im, wA[0].im);

                cmul_soa_sse2(Ajr, Aji, Wjr, Wji, &Tr2_buf[j], &Ti2_buf[j]);
            }

            // outputs m = 0..r-1
            for (int m = 0; m < r; ++m)
            {
                __m128d sumr = Tr2_buf[0];
                __m128d sumi = Ti2_buf[0];

                if (m != 0)
                {
                    const double step_r = Wr[m];
                    const double step_i = Wi[m];
                    double ph_r = 1.0, ph_i = 0.0;

                    for (int j = 1; j < r; ++j)
                    {
                        const double npr = ph_r * step_r - ph_i * step_i;
                        const double npi = ph_r * step_i + ph_i * step_r;
                        ph_r = npr;
                        ph_i = npi;

                        __m128d PHr = _mm_set1_pd(ph_r);
                        __m128d PHi = _mm_set1_pd(ph_i);
                        __m128d rr, ri;
                        cmul_soa_sse2(Tr2_buf[j], Ti2_buf[j], PHr, PHi, &rr, &ri);
                        sumr = _mm_add_pd(sumr, rr);
                        sumi = _mm_add_pd(sumi, ri);
                    }
                }

                double XR[2], XI[2];
                _mm_storeu_pd(XR, sumr);
                _mm_storeu_pd(XI, sumi);
                interleave2_soa_to_aos(XR, XI, &output_buffer[m * K + k]);
            }
        }

        // -------- scalar 1-wide tail --------
        for (; k < K; ++k)
        {
            const int pfk = k + FFT_PREFETCH_DISTANCE;
            if (pfk < K)
            {
                for (int j = 0; j < r; ++j)
                    FFT_PREFETCH_AOS(&sub_outputs[j * K + pfk]);
            }

            // T_j = A_j * W^{j*k}, with T_0 = A_0
            fft_data Tj0 = sub_outputs[0 * K + k];
            // temp arrays on stack for clarity
            double Trs[64], Tis[64];
            Trs[0] = Tj0.re;
            Tis[0] = Tj0.im;

            for (int j = 1; j < r; ++j)
            {
                const fft_data a = sub_outputs[j * K + k];
                const fft_data w = stage_tw[nst * k + (j - 1)];
                Trs[j] = a.re * w.re - a.im * w.im;
                Tis[j] = a.re * w.im + a.im * w.re;
            }

            // outputs m = 0..r-1
            for (int m = 0; m < r; ++m)
            {
                double sumr = Trs[0], sumi = Tis[0];
                if (m != 0)
                {
                    const double step_r = Wr[m];
                    const double step_i = Wi[m];
                    double ph_r = 1.0, ph_i = 0.0;
                    for (int j = 1; j < r; ++j)
                    {
                        const double npr = ph_r * step_r - ph_i * step_i;
                        const double npi = ph_r * step_i + ph_i * step_r;
                        ph_r = npr;
                        ph_i = npi;
                        sumr += Trs[j] * ph_r - Tis[j] * ph_i;
                        sumi += Trs[j] * ph_i + Tis[j] * ph_r;
                    }
                }
                output_buffer[m * K + k].re = sumr;
                output_buffer[m * K + k].im = sumi;
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
    if (!chirp_initialized)
        init_bluestein_chirp();

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
    int transform_direction, // +1 forward (negative exponential), -1 inverse (positive)
    int signal_length)       // N
{
    if (signal_length <= 0)
    {
        fprintf(stderr, "Error: Signal length (%d) is invalid\n", signal_length);
        return;
    }

    const int N = signal_length;

    // --- choose M = next power of two >= 2N-1 ---
    int M = 1;
    int need = 2 * N - 1;
    while (M < need)
        M <<= 1;

    // --- scratch layout: 4*M complexes ---
    if (4 * M > fft_config->max_scratch_size)
    {
        fprintf(stderr, "Error: Scratch too small for Bluestein: need %d, have %d\n",
                4 * M, fft_config->max_scratch_size);
        return;
    }
    fft_data *S = fft_config->scratch;
    fft_data *B_time = S;                // length M: mirrored chirp kernel, pre-scaled by 1/M
    fft_data *B_fft = S + M;             // length M: FFT(B_time)
    fft_data *A_fft_or_time = S + 2 * M; // length M: reused buffer
    fft_data *base_chirp = S + 3 * M;    // length N: bluestein_exp() writes base = exp(+i*pi*n^2/N)

    // --- plans (do NOT mutate a plan to invert) ---
    fft_object plan_fwd = fft_init(M, +1);
    fft_object plan_inv = fft_init(M, -1);
    if (!plan_fwd || !plan_inv)
    {
        fprintf(stderr, "Error: Couldn’t create Bluestein FFT plans\n");
        if (plan_fwd)
            free_fft(plan_fwd);
        if (plan_inv)
            free_fft(plan_inv);
        return;
    }

    // --- get base chirp from your helper ---
    // Convention used here: bluestein_exp(tmp, chirp_out, N, M) fills chirp_out[n] = exp(+i*pi*n^2/N), n=0..N-1
    // We pass an unused temp (A_fft_or_time) to satisfy the API.
    bluestein_exp(A_fft_or_time /*unused temp*/, base_chirp, N, M);

    // --- set up signs (see note below) ---
    // Forward (+1):  chirpA = conj(base), B uses +sign (base)
    // Inverse (-1):  chirpA = base,       B uses -sign (conj(base))
    const int is_forward = (transform_direction == +1);

    // --- build B_time (mirrored kernel) and pre-scale by 1/M ---
    // zero
    for (int i = 0; i < M; ++i)
    {
        B_time[i].re = 0.0;
        B_time[i].im = 0.0;
    }

    // head
    if (is_forward)
    {
        // B[n] = base[n]
        for (int n = 0; n < N; ++n)
        {
            B_time[n] = base_chirp[n];
        }
        // tail mirror
        for (int n = 1; n < N; ++n)
        {
            B_time[M - n] = base_chirp[n];
        }
    }
    else
    {
        // B[n] = conj(base[n])
        for (int n = 0; n < N; ++n)
        {
            B_time[n].re = base_chirp[n].re;
            B_time[n].im = -base_chirp[n].im;
        }
        for (int n = 1; n < N; ++n)
        {
            B_time[M - n].re = base_chirp[n].re;
            B_time[M - n].im = -base_chirp[n].im;
        }
    }

    // pre-scale kernel by 1/M (so IFFT(FFT(A)*FFT(B)) = linear conv)
    const double invM = 1.0 / (double)M;
#if defined(__AVX2__)
    {
        int i = 0;
        const __m256d vscale = _mm256_set1_pd(invM);
        for (; i + 1 < M; i += 2)
        {
            __m256d v = LOADU_PD(&B_time[i].re); // AoS: [re_i,im_i,re_{i+1},im_{i+1}]
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

    // --- FFT of kernel ---
    fft_exec(plan_fwd, B_time, B_fft);

    // --- A_time = x[n] * chirpA[n], zero-padded to M ---
#if defined(__AVX2__)
    {
        int n = 0;
        if (is_forward)
        {
            // chirpA = conj(base)
            for (; n + 1 < N; n += 2)
            {
                __m256d x12 = LOADU_PD(&((fft_data *)input_signal)[n].re);
                __m256d c12 = LOADU_PD(&base_chirp[n].re);
                // conj(base): [re, -im]
                __m256d conj_mask = _mm256_set_pd(-0.0, +0.0, -0.0, +0.0);
                c12 = _mm256_xor_pd(c12, conj_mask);
                __m256d a12 = cmul_avx2_aos(x12, c12);
                STOREU_PD(&A_fft_or_time[n].re, a12);
            }
        }
        else
        {
            // chirpA = base
            for (; n + 1 < N; n += 2)
            {
                __m256d x12 = LOADU_PD(&((fft_data *)input_signal)[n].re);
                __m256d c12 = LOADU_PD(&base_chirp[n].re);
                __m256d a12 = cmul_avx2_aos(x12, c12);
                STOREU_PD(&A_fft_or_time[n].re, a12);
            }
        }
        if (n < N)
        {
            __m128d x1 = LOADU_SSE2(&((fft_data *)input_signal)[n].re);
            __m128d c1 = LOADU_SSE2(&base_chirp[n].re);
            if (is_forward)
            {
                // conj(base)
                c1 = _mm_xor_pd(c1, _mm_set_pd(-0.0, +0.0));
            }
            __m128d a1 = cmul_sse2_aos(x1, c1);
            STOREU_SSE2(&A_fft_or_time[n].re, a1);
        }
    }
#else
    for (int n = 0; n < N; ++n)
    {
        double xr = input_signal[n].re, xi = input_signal[n].im;
        double cr = base_chirp[n].re, ci = base_chirp[n].im;
        if (is_forward)
            ci = -ci; // conj(base)
        A_fft_or_time[n].re = xr * cr - xi * ci;
        A_fft_or_time[n].im = xi * cr + xr * ci;
    }
#endif
    for (int i = N; i < M; ++i)
    {
        A_fft_or_time[i].re = 0.0;
        A_fft_or_time[i].im = 0.0;
    }

    // --- FFT(A_time) -> stash into B_time to reuse bandwidth ---
    fft_exec(plan_fwd, A_fft_or_time, B_time); // B_time now holds FFT(A)

    // --- Pointwise multiply: FFT(A) *= FFT(B) ---
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

    // --- IFFT to time domain (linear convolution, because kernel was pre-scaled by 1/M) ---
    fft_exec(plan_inv, A_fft_or_time, B_time); // B_time now holds conv(A,B)

    // --- final multiply by chirpA[k] (same sign rule as the first step) ---
#if defined(__AVX2__)
    {
        int k = 0;
        if (is_forward)
        {
            // chirpA = conj(base)
            for (; k + 1 < N; k += 2)
            {
                __m256d y = LOADU_PD(&B_time[k].re);
                __m256d ck = LOADU_PD(&base_chirp[k].re);
                ck = _mm256_xor_pd(ck, _mm256_set_pd(-0.0, +0.0, -0.0, +0.0)); // conj
                __m256d out = cmul_avx2_aos(y, ck);
                STOREU_PD(&output_signal[k].re, out);
            }
        }
        else
        {
            // chirpA = base
            for (; k + 1 < N; k += 2)
            {
                __m256d y = LOADU_PD(&B_time[k].re);
                __m256d ck = LOADU_PD(&base_chirp[k].re);
                __m256d out = cmul_avx2_aos(y, ck);
                STOREU_PD(&output_signal[k].re, out);
            }
        }
        if (k < N)
        {
            __m128d y = LOADU_SSE2(&B_time[k].re);
            __m128d ck = LOADU_SSE2(&base_chirp[k].re);
            if (is_forward)
                ck = _mm_xor_pd(ck, _mm_set_pd(-0.0, +0.0)); // conj
            __m128d out = cmul_sse2_aos(y, ck);
            STOREU_SSE2(&output_signal[k].re, out);
        }
    }
#else
    for (int k = 0; k < N; ++k)
    {
        double yr = B_time[k].re, yi = B_time[k].im;
        double cr = base_chirp[k].re, ci = base_chirp[k].im;
        if (is_forward)
            ci = -ci; // conj(base)
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
 * @brief Checks if a number M is divisible by a divisor d, reducing M repeatedly.
 *
 * Determines if M can be fully divided by d, returning 1 if M becomes 1, 0 otherwise.
 *
 * @param[in] number Number to check for divisibility (M > 0).
 * @param[in] divisor Divisor to divide by (d > 0).
 * @return int 1 if M is fully divisible by d, 0 otherwise.
 * @warning If M or d is invalid (<= 0), the function exits with an error.
 */
int divideby(int number, int divisor)
{
    if (number <= 0 || divisor <= 0)
    {
        fprintf(stderr, "Error: Invalid inputs for divideby - number: %d, divisor: %d\n", number, divisor);
        // exit
    }

    int result = number;
    while (result % divisor == 0)
    {
        result /= divisor;
    }
    return (result == 1) ? 1 : 0;
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
        fprintf(stderr, "Error: Invalid inputs for factors - number: %d, factors_array: %p\n", number, (void *)factors_array);
        // exit
    }

    int index = 0, temp_number = number, prime, multiplier, factor1, factor2;
    // Check divisibility by a list of primes
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

    // Handle larger numbers using a heuristic (6k ± 1 method)
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
