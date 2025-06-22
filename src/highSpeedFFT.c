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

#define LOADU_SSE2(ptr) _mm_loadu_pd((const double*)(ptr))

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
static const complex_t *twiddle_tables[] = {
    NULL,            // Radix 0 (unused)
    twiddle_radix2,  // Radix 2
    twiddle_radix3,  // Radix 3
    twiddle_radix4,  // Radix 4
    twiddle_radix5,  // Radix 5
    NULL,            // Radix 6 (no table)
    twiddle_radix7,  // Radix 7
    twiddle_radix8,  // Radix 8
    NULL,            // Radix 9 (no table)
    NULL,            // Radix 10 (no table)
    twiddle_radix11, // Radix 11
    NULL,            // Radix 12 (no table)
    twiddle_radix13  // Radix 13
};

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
    int twiddle_count, max_scratch_size, max_padded_length;

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
        // Bluestein needs M ≥ 2N-1, rounded up to power of 2
        max_padded_length = (int)pow(2.0, ceil(log2((double)(2 * signal_length - 1))));
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
        fft_config->twiddle_factors = (fft_data *)_mm_malloc(twiddle_factors_size * sizeof(fft_data), 32);
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
    longvectorN(fft_config->twiddles, fft_config->n_fft, fft_config->factors, fft_config->lf);

    // Step 11: Populate twiddle_factors for pure-power FFTs
    // Store W_n^{j*k} (j=1..radix-1) for each level
    // Note: Future extension could support mixed-radix special cases by storing offsets per factor
    if (fft_config->twiddle_factors)
    {
        int offset = 0;
        int radix = is_power_of_2 ? 2 : is_power_of_3 ? 3
                                    : is_power_of_5   ? 5
                                    : is_power_of_7   ? 7
                                    : is_power_of_11  ? 11
                                                      : 13;
        for (int n = signal_length; n >= radix; n /= radix)
        {
            int sub_fft_size = n / radix;
            for (int j = 1; j < radix; j++)
            {
                for (int k = 0; k < sub_fft_size; k++)
                {
                    int idx = sub_fft_size - 1 + j * k; // W_n^{j*k}
                    fft_config->twiddle_factors[offset + (j - 1) * sub_fft_size + k].re = fft_config->twiddles[idx].re;
                    fft_config->twiddle_factors[offset + (j - 1) * sub_fft_size + k].im = fft_config->twiddles[idx].im;
                }
            }
            offset += (radix - 1) * sub_fft_size;
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
    if (data_length <= 0 || stride <= 0 || factor_index < 0)
    {
        fprintf(stderr, "Error: Invalid mixed-radix FFT inputs - data_length: %d, stride: %d, factor_index: %d\n", data_length, stride, factor_index);
        // exit
    }

    int radix, sub_length, new_stride;
    if (data_length > 1)
    {
        radix = fft_obj->factors[factor_index];
    }

    if (data_length == 1)
    {
        /**
         * @brief Base case for recursion: copy a single element.
         *
         * When the data length is 1, there’s no transformation needed—just copy the input
         * complex value to the output buffer. This marks the termination of the recursive
         * decomposition, as no further division is possible.
         */
        output_buffer[0].re = input_buffer[0].re;
        output_buffer[0].im = input_buffer[0].im;
    }
    else if (data_length == 2)
    {
        /**
         * @brief Radix-2 butterfly operation for two-point FFT.
         *
         * Implements a two-point butterfly, the simplest form of DIT FFT, combining two input points
         * using addition and subtraction. This corresponds to the FFT formula for N=2,
         * where \(X(0) = x(0) + x(1)\) and \(X(1) = x(0) - x(1)\) for forward transforms,
         * adjusted by transform_sign for inverse transforms.
         */
        fft_type tau1r = output_buffer[0].re, tau1i = output_buffer[0].im;

        output_buffer[0].re = input_buffer[0].re;
        output_buffer[0].im = input_buffer[0].im;
        output_buffer[1].re = input_buffer[stride].re;
        output_buffer[1].im = input_buffer[stride].im;

        output_buffer[0].re = tau1r + output_buffer[1].re;
        output_buffer[0].im = tau1i + output_buffer[1].im;
        output_buffer[1].re = tau1r - output_buffer[1].re;
        output_buffer[1].im = tau1i - output_buffer[1].im;
    }
    else if (data_length == 3)
    {
        /**
         * @brief Radix-3 butterfly operation for three-point FFT.
         *
         * Implements a three-point butterfly using precomputed constants (e.g., \(\sqrt{3}/2\)),
         * combining three input points to compute the FFT. This uses the DIT formula for N=3,
         * where the transform involves rotations by 120° and 240° in the complex plane,
         * adjusted by transform_sign. Mathematically, it applies \(X(k) = \sum_{n=0}^{2} x(n) \cdot e^{-2\pi i k n / 3}\),
         * leveraging symmetry and precomputed roots of unity for efficiency.
         */
        fft_type tau0r, tau0i, tau1r, tau1i, tau2r, tau2i;
        const fft_type sqrt3_by_2 = 0.86602540378; // sqrt(3)/2, approximately 0.866, used for 120° rotation

        output_buffer[0].re = input_buffer[0].re;
        output_buffer[0].im = input_buffer[0].im;
        output_buffer[1].re = input_buffer[stride].re;
        output_buffer[1].im = input_buffer[stride].im;
        output_buffer[2].re = input_buffer[2 * stride].re;
        output_buffer[2].im = input_buffer[2 * stride].im;

        tau0r = output_buffer[1].re + output_buffer[2].re;                                 // Sum of second and third points (real)
        tau0i = output_buffer[1].im + output_buffer[2].im;                                 // Sum of second and third points (imag)
        tau1r = transform_sign * sqrt3_by_2 * (output_buffer[1].re - output_buffer[2].re); // Rotated difference (real)
        tau1i = transform_sign * sqrt3_by_2 * (output_buffer[1].im - output_buffer[2].im); // Rotated difference (imag)
        tau2r = output_buffer[0].re - tau0r * 0.5;                                         // Center point adjustment (real)
        tau2i = output_buffer[0].im - tau0i * 0.5;                                         // Center point adjustment (imag)

        output_buffer[0].re = tau0r + output_buffer[0].re; // Combine with first point (real)
        output_buffer[0].im = tau0i + output_buffer[0].im; // Combine with first point (imag)
        output_buffer[1].re = tau2r + tau1i;               // Apply rotation for second output (real)
        output_buffer[1].im = tau2i - tau1r;               // Apply rotation for second output (imag)
        output_buffer[2].re = tau2r - tau1i;               // Apply rotation for third output (real)
        output_buffer[2].im = tau2i + tau1r;               // Apply rotation for third output (imag)
    }
    else if (data_length == 4)
    {
        /**
         * @brief Radix-4 butterfly operation for four-point FFT.
         *
         * Implements a four-point butterfly, combining four input points using additions, subtractions,
         * and a 90° rotation (via \(\sqrt{2}/2\)). This corresponds to the DIT formula for N=4,
         * where the transform involves pairwise combinations and rotations, adjusted by transform_sign.
         * Mathematically, it computes \(X(k) = \sum_{n=0}^{3} x(n) \cdot e^{-2\pi i k n / 4}\),
         * leveraging symmetry for efficiency.
         */
        fft_type tau0r, tau0i, tau1r, tau1i, tau2r, tau2i, tau3r, tau3i;

        output_buffer[0].re = input_buffer[0].re;
        output_buffer[0].im = input_buffer[0].im;
        output_buffer[1].re = input_buffer[stride].re;
        output_buffer[1].im = input_buffer[stride].im;
        output_buffer[2].re = input_buffer[2 * stride].re;
        output_buffer[2].im = input_buffer[2 * stride].im;
        output_buffer[3].re = input_buffer[3 * stride].re;
        output_buffer[3].im = input_buffer[3 * stride].im;

        tau0r = output_buffer[0].re + output_buffer[2].re;                    // Sum of first and third points (real)
        tau0i = output_buffer[0].im + output_buffer[2].im;                    // Sum of first and third points (imag)
        tau1r = output_buffer[0].re - output_buffer[2].re;                    // Difference of first and third points (real)
        tau1i = output_buffer[0].im - output_buffer[2].im;                    // Difference of first and third points (imag)
        tau2r = output_buffer[1].re + output_buffer[3].re;                    // Sum of second and fourth points (real)
        tau2i = output_buffer[1].im + output_buffer[3].im;                    // Sum of second and fourth points (imag)
        tau3r = transform_sign * (output_buffer[1].re - output_buffer[3].re); // Rotated difference (real)
        tau3i = transform_sign * (output_buffer[1].im - output_buffer[3].im); // Rotated difference (imag)

        output_buffer[0].re = tau0r + tau2r; // Combine sums for first output (real)
        output_buffer[0].im = tau0i + tau2i; // Combine sums for first output (imag)
        output_buffer[1].re = tau1r + tau3i; // Apply rotation for second output (real)
        output_buffer[1].im = tau1i - tau3r; // Apply rotation for second output (imag)
        output_buffer[2].re = tau0r - tau2r; // Combine differences for third output (real)
        output_buffer[2].im = tau0i - tau2i; // Combine differences for third output (imag)
        output_buffer[3].re = tau1r - tau3i; // Apply rotation for fourth output (real)
        output_buffer[3].im = tau1i + tau3r; // Apply rotation for fourth output (imag)
    }
    else if (data_length == 5)
    {
        /**
         * @brief Radix-5 butterfly operation for five-point FFT.
         *
         * Implements a five-point butterfly using precomputed constants (e.g., cos/sin of 72° and 144°),
         * combining five input points to compute the FFT. This uses the DIT formula for N=5,
         * involving rotations by 72°, 144°, 216°, and 288° in the complex plane, adjusted by transform_sign.
         * Mathematically, it computes \(X(k) = \sum_{n=0}^{4} x(n) \cdot e^{-2\pi i k n / 5}\),
         * leveraging symmetry and precomputed roots of unity for efficiency.
         */
        fft_type tau0r, tau0i, tau1r, tau1i, tau2r, tau2i, tau3r, tau3i, tau4r, tau4i, tau5r, tau5i, tau6r, tau6i;
        const fft_type c1 = 0.30901699437, c2 = -0.80901699437, s1 = 0.95105651629, s2 = 0.58778525229; // cos/sin of 72° and 144°

        output_buffer[0].re = input_buffer[0].re;
        output_buffer[0].im = input_buffer[0].im;
        output_buffer[1].re = input_buffer[stride].re;
        output_buffer[1].im = input_buffer[stride].im;
        output_buffer[2].re = input_buffer[2 * stride].re;
        output_buffer[2].im = input_buffer[2 * stride].im;
        output_buffer[3].re = input_buffer[3 * stride].re;
        output_buffer[3].im = input_buffer[3 * stride].im;
        output_buffer[4].re = input_buffer[4 * stride].re;
        output_buffer[4].im = input_buffer[4 * stride].im;

        tau0r = output_buffer[1].re + output_buffer[4].re; // Sum of first and fifth points (real)
        tau0i = output_buffer[1].im + output_buffer[4].im; // Sum of first and fifth points (imag)
        tau2r = output_buffer[1].re - output_buffer[4].re; // Difference of first and fifth points (real)
        tau2i = output_buffer[1].im - output_buffer[4].im; // Difference of first and fifth points (imag)
        tau1r = output_buffer[2].re + output_buffer[3].re; // Sum of second and fourth points (real)
        tau1i = output_buffer[2].im + output_buffer[3].im; // Sum of second and fourth points (imag)
        tau3r = output_buffer[2].re - output_buffer[3].re; // Difference of second and fourth points (real)
        tau3i = output_buffer[2].im - output_buffer[3].im; // Difference of second and fourth points (imag)

        tau4r = c1 * tau0r + c2 * tau1r; // Apply 72° rotation (real)
        tau4i = c1 * tau0i + c2 * tau1i; // Apply 72° rotation (imag)
        if (transform_sign == 1)
        {
            tau5r = s1 * tau2r + s2 * tau3r; // Apply 144° rotation (real, forward)
            tau5i = s1 * tau2i + s2 * tau3i; // Apply 144° rotation (imag, forward)
        }
        else
        {
            tau5r = -s1 * tau2r - s2 * tau3r; // Apply 144° rotation (real, inverse)
            tau5i = -s1 * tau2i - s2 * tau3i; // Apply 144° rotation (imag, inverse)
        }

        tau6r = output_buffer[0].re + tau4r; // Combine with center point (real)
        tau6i = output_buffer[0].im + tau4i; // Combine with center point (imag)
        output_buffer[1].re = tau6r + tau5i; // First rotated output (real)
        output_buffer[1].im = tau6i - tau5r; // First rotated output (imag)
        output_buffer[4].re = tau6r - tau5i; // Fifth rotated output (real)
        output_buffer[4].im = tau6i + tau5r; // Fifth rotated output (imag)

        tau4r = c2 * tau0r + c1 * tau1r; // Apply 144° rotation (real)
        tau4i = c2 * tau0i + c1 * tau1i; // Apply 144° rotation (imag)
        if (transform_sign == 1)
        {
            tau5r = s2 * tau2r - s1 * tau3r; // Apply 72° rotation (real, forward)
            tau5i = s2 * tau2i - s1 * tau3i; // Apply 72° rotation (imag, forward)
        }
        else
        {
            tau5r = -s2 * tau2r + s1 * tau3r; // Apply 72° rotation (real, inverse)
            tau5i = -s2 * tau2i + s1 * tau3i; // Apply 72° rotation (imag, inverse)
        }

        tau6r = output_buffer[0].re + tau4r; // Combine with center point (real)
        tau6i = output_buffer[0].im + tau4i; // Combine with center point (imag)
        output_buffer[2].re = tau6r + tau5i; // Second rotated output (real)
        output_buffer[2].im = tau6i - tau5r; // Second rotated output (imag)
        output_buffer[3].re = tau6r - tau5i; // Fourth rotated output (real)
        output_buffer[3].im = tau6i + tau5r; // Fourth rotated output (imag)

        output_buffer[0].re += tau0r + tau1r; // Update center point (real)
        output_buffer[0].im += tau0i + tau1i; // Update center point (imag)
    }
    else if (data_length == 7)
    {
        /**
         * @brief Radix-7 butterfly operation for seven-point FFT.
         *
         * Implements a seven-point butterfly using precomputed constants (e.g., cos/sin of 51.43°, 102.86°, 154.29°),
         * combining seven input points to compute the FFT. This uses the DIT formula for N=7,
         * involving rotations by specific angles in the complex plane, adjusted by transform_sign.
         * Mathematically, it computes \(X(k) = \sum_{n=0}^{6} x(n) \cdot e^{-2\pi i k n / 7}\),
         * leveraging symmetry and precomputed roots of unity for efficiency.
         */
        fft_type tau0r, tau0i, tau1r, tau1i, tau2r, tau2i, tau3r, tau3i, tau4r, tau4i, tau5r, tau5i, tau6r, tau6i, tau7r, tau7i;
        const fft_type c1 = 0.62348980185, c2 = -0.22252093395, c3 = -0.9009688679, s1 = 0.78183148246, s2 = 0.97492791218, s3 = 0.43388373911; // cos/sin of 51.43°, 102.86°, 154.29°

        // Copy input to output (7 points), ensuring all input values are loaded for processing
        for (int i = 0; i < 7; i++)
        {
            output_buffer[i].re = input_buffer[i * stride].re; // Load real part
            output_buffer[i].im = input_buffer[i * stride].im; // Load imaginary part
        }

        tau0r = output_buffer[1].re + output_buffer[6].re; // Sum of first and seventh points (real)
        tau3r = output_buffer[1].re - output_buffer[6].re; // Difference of first and seventh points (real)
        tau0i = output_buffer[1].im + output_buffer[6].im; // Sum of first and seventh points (imag)
        tau3i = output_buffer[1].im - output_buffer[6].im; // Difference of first and seventh points (imag)

        tau1r = output_buffer[2].re + output_buffer[5].re; // Sum of second and sixth points (real)
        tau4r = output_buffer[2].re - output_buffer[5].re; // Difference of second and sixth points (real)
        tau1i = output_buffer[2].im + output_buffer[5].im; // Sum of second and sixth points (imag)
        tau4i = output_buffer[2].im - output_buffer[5].im; // Difference of second and sixth points (imag)

        tau2r = output_buffer[3].re + output_buffer[4].re; // Sum of third and fifth points (real)
        tau5r = output_buffer[3].re - output_buffer[4].re; // Difference of third and fifth points (real)
        tau2i = output_buffer[3].im + output_buffer[4].im; // Sum of third and fifth points (imag)
        tau5i = output_buffer[3].im - output_buffer[4].im; // Difference of third and fifth points (imag)

        tau6r = output_buffer[0].re + c1 * tau0r + c2 * tau1r + c3 * tau2r; // Combine with center, apply 51.43° rotation (real)
        tau6i = output_buffer[0].im + c1 * tau0i + c2 * tau1i + c3 * tau2i; // Combine with center, apply 51.43° rotation (imag)
        if (transform_sign == 1)
        {
            tau7r = -s1 * tau3r - s2 * tau4r - s3 * tau5r; // Apply rotations for forward transform (real)
            tau7i = -s1 * tau3i - s2 * tau4i - s3 * tau5i; // Apply rotations for forward transform (imag)
        }
        else
        {
            tau7r = s1 * tau3r + s2 * tau4r + s3 * tau5r; // Apply rotations for inverse transform (real)
            tau7i = s1 * tau3i + s2 * tau4i + s3 * tau5i; // Apply rotations for inverse transform (imag)
        }

        output_buffer[1].re = tau6r - tau7i; // First rotated output (real)
        output_buffer[6].re = tau6r + tau7i; // Seventh rotated output (real)
        output_buffer[1].im = tau6i + tau7r; // First rotated output (imag)
        output_buffer[6].im = tau6i - tau7r; // Seventh rotated output (imag)

        tau6r = output_buffer[0].re + c2 * tau0r + c3 * tau1r + c1 * tau2r; // Combine with center, apply 102.86° rotation (real)
        tau6i = output_buffer[0].im + c2 * tau0i + c3 * tau1i + c1 * tau2i; // Combine with center, apply 102.86° rotation (imag)
        if (transform_sign == 1)
        {
            tau7r = -s2 * tau3r + s3 * tau4r + s1 * tau5r; // Apply rotations for forward transform (real)
            tau7i = -s2 * tau3i + s3 * tau4i + s1 * tau5i; // Apply rotations for forward transform (imag)
        }
        else
        {
            tau7r = s2 * tau3r - s3 * tau4r - s1 * tau5r; // Apply rotations for inverse transform (real)
            tau7i = s2 * tau3i - s3 * tau4i - s1 * tau5i; // Apply rotations for inverse transform (imag)
        }

        output_buffer[2].re = tau6r - tau7i; // Second rotated output (real)
        output_buffer[5].re = tau6r + tau7i; // Sixth rotated output (real)
        output_buffer[2].im = tau6i + tau7r; // Second rotated output (imag)
        output_buffer[5].im = tau6i - tau7r; // Sixth rotated output (imag)

        tau6r = output_buffer[0].re + c3 * tau0r + c1 * tau1r + c2 * tau2r; // Combine with center, apply 154.29° rotation (real)
        tau6i = output_buffer[0].im + c3 * tau0i + c1 * tau1i + c2 * tau2i; // Combine with center, apply 154.29° rotation (imag)
        if (transform_sign == 1)
        {
            tau7r = -s3 * tau3r + s1 * tau4r - s2 * tau5r; // Apply rotations for forward transform (real)
            tau7i = -s3 * tau3i + s1 * tau4i - s2 * tau5i; // Apply rotations for forward transform (imag)
        }
        else
        {
            tau7r = s3 * tau3r - s1 * tau4r + s2 * tau5r; // Apply rotations for inverse transform (real)
            tau7i = s3 * tau3i - s1 * tau4i + s2 * tau5i; // Apply rotations for inverse transform (imag)
        }

        output_buffer[3].re = tau6r - tau7i; // Third rotated output (real)
        output_buffer[4].re = tau6r + tau7i; // Fifth rotated output (real)
        output_buffer[3].im = tau6i + tau7r; // Third rotated output (imag)
        output_buffer[4].im = tau6i - tau7r; // Fifth rotated output (imag)

        output_buffer[0].re += tau0r + tau1r + tau2r; // Update center point (real)
        output_buffer[0].im += tau0i + tau1i + tau2i; // Update center point (imag)
    }
    else if (data_length == 8)
    {
        /**
         * @brief Radix-8 butterfly operation for eight-point FFT.
         *
         * Implements an eight-point butterfly using precomputed constants (e.g., \(\sqrt{2}/2\)),
         * combining eight input points to compute the FFT. This uses the DIT formula for N=8,
         * involving rotations by 45°, 90°, 135°, etc., in the complex plane, adjusted by transform_sign.
         * Mathematically, it computes \(X(k) = \sum_{n=0}^{7} x(n) \cdot e^{-2\pi i k n / 8}\),
         * leveraging symmetry and precomputed roots of unity for efficiency.
         */
        fft_type tau0r, tau0i, tau1r, tau1i, tau2r, tau2i, tau3r, tau3i, tau4r, tau4i, tau5r, tau5i, tau6r, tau6i, tau7r, tau7i, tau8r, tau8i, tau9r, tau9i;
        const fft_type c1 = 0.70710678118654752440084436210485, s1 = c1; // sqrt(2)/2, approximately 0.707, used for 45° rotation

        // Copy input to output (8 points), ensuring all input values are loaded for processing
        for (int i = 0; i < 8; i++)
        {
            output_buffer[i].re = input_buffer[i * stride].re; // Load real part
            output_buffer[i].im = input_buffer[i * stride].im; // Load imaginary part
        }

        tau0r = output_buffer[0].re + output_buffer[4].re; // Sum of first and fifth points (real)
        tau4r = output_buffer[0].re - output_buffer[4].re; // Difference of first and fifth points (real)
        tau0i = output_buffer[0].im + output_buffer[4].im; // Sum of first and fifth points (imag)
        tau4i = output_buffer[0].im - output_buffer[4].im; // Difference of first and fifth points (imag)

        tau1r = output_buffer[1].re + output_buffer[7].re; // Sum of second and eighth points (real)
        tau5r = output_buffer[1].re - output_buffer[7].re; // Difference of second and eighth points (real)
        tau1i = output_buffer[1].im + output_buffer[7].im; // Sum of second and eighth points (imag)
        tau5i = output_buffer[1].im - output_buffer[7].im; // Difference of second and eighth points (imag)

        tau2r = output_buffer[3].re + output_buffer[5].re; // Sum of fourth and sixth points (real)
        tau6r = output_buffer[3].re - output_buffer[5].re; // Difference of fourth and sixth points (real)
        tau2i = output_buffer[3].im + output_buffer[5].im; // Sum of fourth and sixth points (imag)
        tau6i = output_buffer[3].im - output_buffer[5].im; // Difference of fourth and sixth points (imag)

        tau3r = output_buffer[2].re + output_buffer[6].re; // Sum of third and seventh points (real)
        tau7r = output_buffer[2].re - output_buffer[6].re; // Difference of third and seventh points (real)
        tau3i = output_buffer[2].im + output_buffer[6].im; // Sum of third and seventh points (imag)
        tau7i = output_buffer[2].im - output_buffer[6].im; // Difference of third and seventh points (imag)

        output_buffer[0].re = tau0r + tau1r + tau2r + tau3r; // Combine all sums for first output (real)
        output_buffer[0].im = tau0i + tau1i + tau2i + tau3i; // Combine all sums for first output (imag)
        output_buffer[4].re = tau0r - tau1r - tau2r + tau3r; // Combine differences for fifth output (real)
        output_buffer[4].im = tau0i - tau1i - tau2i + tau3i; // Combine differences for fifth output (imag)

        fft_type temp1r = tau1r - tau2r, temp1i = tau1i - tau2i; // Intermediate difference for rotation
        fft_type temp2r = tau5r + tau6r, temp2i = tau5i + tau6i; // Intermediate sum for rotation

        tau8r = tau4r + c1 * temp1r; // Apply 45° rotation (real)
        tau8i = tau4i + c1 * temp1i; // Apply 45° rotation (imag)
        if (transform_sign == 1)
        {
            tau9r = -s1 * temp2r - tau7r; // Apply 45° rotation for forward transform (real)
            tau9i = -s1 * temp2i - tau7i; // Apply 45° rotation for forward transform (imag)
        }
        else
        {
            tau9r = s1 * temp2r + tau7r; // Apply 45° rotation for inverse transform (real)
            tau9i = s1 * temp2i + tau7i; // Apply 45° rotation for inverse transform (imag)
        }

        output_buffer[1].re = tau8r - tau9i; // Second rotated output (real)
        output_buffer[1].im = tau8i + tau9r; // Second rotated output (imag)
        output_buffer[7].re = tau8r + tau9i; // Eighth rotated output (real)
        output_buffer[7].im = tau8i - tau9r; // Eighth rotated output (imag)

        tau8r = tau0r - tau3r; // Difference for third output (real)
        tau8i = tau0i - tau3i; // Difference for third output (imag)
        if (transform_sign == 1)
        {
            tau9r = -tau5r + tau6r; // Apply rotation for forward transform (real)
            tau9i = -tau5i + tau6i; // Apply rotation for forward transform (imag)
        }
        else
        {
            tau9r = tau5r - tau6r; // Apply rotation for inverse transform (real)
            tau9i = tau5i - tau6i; // Apply rotation for inverse transform (imag)
        }

        output_buffer[2].re = tau8r - tau9i; // Third rotated output (real)
        output_buffer[2].im = tau8i + tau9r; // Third rotated output (imag)
        output_buffer[6].re = tau8r + tau9i; // Seventh rotated output (real)
        output_buffer[6].im = tau8i - tau9r; // Seventh rotated output (imag)

        tau8r = tau4r - c1 * temp1r; // Apply -45° rotation (real)
        tau8i = tau4i - c1 * temp1i; // Apply -45° rotation (imag)
        if (transform_sign == 1)
        {
            tau9r = -s1 * temp2r + tau7r; // Apply 45° rotation for forward transform (real)
            tau9i = -s1 * temp2i + tau7i; // Apply 45° rotation for forward transform (imag)
        }
        else
        {
            tau9r = s1 * temp2r - tau7r; // Apply 45° rotation for inverse transform (real)
            tau9i = s1 * temp2i - tau7i; // Apply 45° rotation for inverse transform (imag)
        }

        output_buffer[3].re = tau8r - tau9i; // Fourth rotated output (real)
        output_buffer[3].im = tau8i + tau9r; // Fourth rotated output (imag)
        output_buffer[5].re = tau8r + tau9i; // Sixth rotated output (real)
        output_buffer[5].im = tau8i - tau9r; // Sixth rotated output (imag)
    }
    else if (radix == 2)
    {
        /**
         * @brief Optimized Radix-2 decomposition with AVX2, SSE2 tail, direct output writes,
         * and 2x unrolled loop with prefetching.
         *
         * Intention: Compute FFT for data lengths divisible by 2, splitting into two sub-FFTs
         * (even/odd indices), applying recursive FFTs, and combining with twiddle factors.
         * Optimized for power-of-2 (N=2^r) and mixed-radix FFTs.
         *
         * Changes from original:
         * 1. SSE2 tail: Processes two points at a time, reducing scalar tail to at most 1.
         * 2. Direct writes: Butterflies write to output_buffer, eliminating copy loop.
         * 3. 2x unroll: Processes 8 points per AVX2 iteration with prefetching.
         *
         * Mathematically: Computes:
         *   X(k) = X_even(k) + W_N^k * X_odd(k),
         *   X(k+N/2) = X_even(k) - W_N^k * X_odd(k),
         * where X_even, X_odd are sub-FFTs of size N/2, W_N^k = e^{-2πi k / N}.
         *
         * Assumptions:
         * - output_buffer, sub_fft_outputs, twiddle_factors are 32-byte aligned.
         * - fft_obj->twiddles has n_fft ≥ data_length elements.
         * - Scratch buffer is pre-allocated with sufficient size.
         * - transform_sign and stride are correctly set by caller.
         */
        // Step 1: Compute subproblem size and stride
        int sub_fft_length = data_length / 2; // Size of each sub-FFT (N/2)
        int next_stride = 2 * stride;         // Double stride for next level
        int sub_fft_size = sub_fft_length;    // Sub-FFT size for indexing

        // Step 2: Validate scratch buffer
        // Power-of-2: 2*(N/2) for outputs; mixed-radix: (2+1)*(N/2) = 3*(N/2) for outputs + twiddles
        int required_size = fft_obj->twiddle_factors != NULL ? 2 * sub_fft_size : 3 * sub_fft_size;
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Scratch buffer too small for radix-2 at offset %d (need %d, have %d)\n",
                    scratch_offset, required_size, fft_obj->max_scratch_size - scratch_offset);
            return;
        }

        // Step 3: Assign scratch slices
        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors;
        if (fft_obj->twiddle_factors != NULL)
        {
            if (factor_index >= fft_obj->num_precomputed_stages)
            {
                fprintf(stderr, "Error: Invalid factor_index (%d) exceeds num_precomputed_stages (%d) for radix-2\n",
                        factor_index, fft_obj->num_precomputed_stages);
                return;
            }
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + scratch_offset + 2 * sub_fft_size;
        }

        // Step 4: Compute offsets for child recursions
        int child_scratch_per_branch = fft_obj->twiddle_factors != NULL ? 2 * (sub_fft_size / 2) : 3 * (sub_fft_size / 2);
        if (child_scratch_per_branch * 2 > required_size)
        {
            fprintf(stderr, "Error: Child scratch size (%d) exceeds parent allocation (%d) for radix-2\n",
                    child_scratch_per_branch * 2, required_size);
            return;
        }
        int child_offset1 = scratch_offset;
        int child_offset2 = scratch_offset + required_size / 2;

        // Step 5: Recurse on even and odd indices
        mixed_radix_dit_rec(sub_fft_outputs, input_buffer, fft_obj, transform_sign, sub_fft_length,
                            next_stride, factor_index + 1, child_offset1); // Even indices
        mixed_radix_dit_rec(sub_fft_outputs + sub_fft_size, input_buffer + stride, fft_obj, transform_sign,
                            sub_fft_length, next_stride, factor_index + 1, child_offset2); // Odd indices

        // Step 6: Prepare twiddle factors (mixed-radix only)
        if (fft_obj->twiddle_factors == NULL)
        {
            if (fft_obj->n_fft < 2 * sub_fft_size)
            {
                fprintf(stderr, "Error: Twiddle array too small (need at least %d elements, have %d)\n",
                        2 * sub_fft_size, fft_obj->n_fft);
                return;
            }
            for (int k = 0; k < sub_fft_size; k++)
            {
                int idx = sub_fft_length - 1 + k;
                twiddle_factors[k].re = fft_obj->twiddles[idx].re;
                twiddle_factors[k].im = fft_obj->twiddles[idx].im;
            }
        }

        // Step 7: Run AVX2 butterfly operations with 2x unroll and prefetching
        int k = 0;
        for (; k + 7 < sub_fft_size; k += 8)
        {
            // Prefetch data 128 bytes (16 doubles) ahead
            _mm_prefetch((const char *)&sub_fft_outputs[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 16].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 16 + sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_fft_outputs[k + 16 + sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&twiddle_factors[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&twiddle_factors[k + 16].im, _MM_HINT_T0);

            // First 4 points (k to k+3)
            __m256d even_re = LOAD_PD(&sub_fft_outputs[k].re);
            __m256d even_im = LOAD_PD(&sub_fft_outputs[k].im);
            __m256d odd_re = LOAD_PD(&sub_fft_outputs[k + sub_fft_size].re);
            __m256d odd_im = LOAD_PD(&sub_fft_outputs[k + sub_fft_size].im);
            __m256d twiddle_re = LOAD_PD(&twiddle_factors[k].re);
            __m256d twiddle_im = LOAD_PD(&twiddle_factors[k].im);

            __m256d twiddled_re = FMSUB(odd_re, twiddle_re, _mm256_mul_pd(odd_im, twiddle_im));
            __m256d twiddled_im = FMADD(odd_im, twiddle_re, _mm256_mul_pd(odd_re, twiddle_im));

            STORE_PD(&output_buffer[k].re, FMADD(twiddled_re, AVX_ONE, even_re)); // X(k)
            STORE_PD(&output_buffer[k].im, FMADD(twiddled_im, AVX_ONE, even_im));
            STORE_PD(&output_buffer[k + sub_fft_size].re, FMSUB(even_re, AVX_ONE, twiddled_re)); // X(k+N/2)
            STORE_PD(&output_buffer[k + sub_fft_size].im, FMSUB(even_im, AVX_ONE, twiddled_im));

            // Second 4 points (k+4 to k+7)
            __m256d even_re2 = LOAD_PD(&sub_fft_outputs[k + 4].re);
            __m256d even_im2 = LOAD_PD(&sub_fft_outputs[k + 4].im);
            __m256d odd_re2 = LOAD_PD(&sub_fft_outputs[k + 4 + sub_fft_size].re);
            __m256d odd_im2 = LOAD_PD(&sub_fft_outputs[k + 4 + sub_fft_size].im);
            __m256d twiddle_re2 = LOAD_PD(&twiddle_factors[k + 4].re);
            __m256d twiddle_im2 = LOAD_PD(&twiddle_factors[k + 4].im);

            __m256d twiddled_re2 = FMSUB(odd_re2, twiddle_re2, _mm256_mul_pd(odd_im2, twiddle_im2));
            __m256d twiddled_im2 = FMADD(odd_im2, twiddle_re2, _mm256_mul_pd(odd_re2, twiddle_im2));

            STORE_PD(&output_buffer[k + 4].re, FMADD(twiddled_re2, AVX_ONE, even_re2));
            STORE_PD(&output_buffer[k + 4].im, FMADD(twiddled_im2, AVX_ONE, even_im2));
            STORE_PD(&output_buffer[k + 4 + sub_fft_size].re, FMSUB(even_re2, AVX_ONE, twiddled_re2));
            STORE_PD(&output_buffer[k + 4 + sub_fft_size].im, FMSUB(even_im2, AVX_ONE, twiddled_im2));
        }

        // Step 8: SSE2 tail for remaining 2 points
        if (k + 1 < sub_fft_size)
        {
            __m128d even_re = LOAD_SSE2(&sub_fft_outputs[k].re);
            __m128d even_im = LOAD_SSE2(&sub_fft_outputs[k].im);
            __m128d odd_re = LOAD_SSE2(&sub_fft_outputs[k + sub_fft_size].re);
            __m128d odd_im = LOAD_SSE2(&sub_fft_outputs[k + sub_fft_size].im);
            __m128d twiddle_re = LOAD_SSE2(&twiddle_factors[k].re);
            __m128d twiddle_im = LOAD_SSE2(&twiddle_factors[k].im);

            __m128d twiddled_re = FMSUB_SSE2(odd_re, twiddle_re, _mm_mul_pd(odd_im, twiddle_im));
            __m128d twiddled_im = FMADD_SSE2(odd_im, twiddle_re, _mm_mul_pd(odd_re, twiddle_im));

            STORE_SSE2(&output_buffer[k].re, FMADD_SSE2(twiddled_re, SSE2_ONE, even_re));
            STORE_SSE2(&output_buffer[k].im, FMADD_SSE2(twiddled_im, SSE2_ONE, even_im));
            STORE_SSE2(&output_buffer[k + sub_fft_size].re, FMSUB_SSE2(even_re, SSE2_ONE, twiddled_re));
            STORE_SSE2(&output_buffer[k + sub_fft_size].im, FMSUB_SSE2(even_im, SSE2_ONE, twiddled_im));
            k += 2;
        }

        // Step 9: Scalar tail for one remaining point
        if (k < sub_fft_size)
        {
            double even_re = sub_fft_outputs[k].re, even_im = sub_fft_outputs[k].im;
            double odd_re = sub_fft_outputs[k + sub_fft_size].re, odd_im = sub_fft_outputs[k + sub_fft_size].im;
            double twiddled_re = odd_re * twiddle_factors[k].re - odd_im * twiddle_factors[k].im;
            double twiddled_im = odd_im * twiddle_factors[k].re + odd_re * twiddle_factors[k].im;

            output_buffer[k].re = even_re + twiddled_re;
            output_buffer[k].im = even_im + twiddled_im;
            output_buffer[k + sub_fft_size].re = even_re - twiddled_re;
            output_buffer[k + sub_fft_size].im = even_im - twiddled_im;
        }
    }
    else if (radix == 3)
    {
        /**
         * @brief Optimized Radix-3 decomposition for three-point sub-FFTs with AVX2 vectorization,
         *        SSE2 tail, direct output writes, and 2x unrolled loop with prefetching.
         *
         * Intention: Efficiently compute the FFT for data lengths divisible by 3 by splitting into
         * three sub-FFTs (indices n mod 3 = 0, 1, 2), applying recursive FFTs, and combining results
         * with twiddle factors. Optimized for power-of-3 (N=3^r) and mixed-radix FFTs using
         * pre-allocated scratch and stage-specific twiddle offsets. Follows FFTW’s two-buffer strategy
         * for thread safety and performance.
         *
         * Mathematically: Computes:
         *   X(k) = X_0(k) + W_N^k * X_1(k) + W_N^{2k} * X_2(k),
         *   X(k+N/3) = X_0(k) - ½(X_1(k) + X_2(k)) + i * sign * (√3/2) * (X_1(k) - X_2(k)),
         *   X(k+2N/3) = X_0(k) - ½(X_1(k) + X_2(k)) - i * sign * (√3/2) * (X_1(k) - X_2(k)),
         * where X_0, X_1, X_2 are sub-FFTs of size N/3, W_N^k = e^{-2πi k / N}.
         * The butterfly uses 120° and 240° roots of unity (√3/2 for rotation).
         *
         * Optimization:
         * - AVX2 processes 4 k-values per iteration (256-bit vectors), doubling SSE2 throughput.
         * - 2x unrolling handles 8 k-values per loop, reducing branch overhead.
         * - Prefetching targets output_buffer 256 bytes ahead to hide memory latency.
         * - Direct loads from sub_fft_outputs avoid O(N) flattening to out_re/out_im.
         * - Direct writes to output_buffer eliminate O(N) copy loop.
         * - SSE2 tail minimizes scalar operations for non-4-aligned sub_fft_size.
         * - FMA (FMADD/FMSUB) computes (a*b ± c) for accuracy and speed.
         * - Pre-allocated scratch eliminates malloc/free overhead.
         * - 32-byte aligned buffers enable aligned SIMD loads/stores.
         *
         * Changes from original:
         * 1. Replaced SSE2 loop with AVX2, processing 4 k-values per iteration.
         * 2. Added SSE2 tail to handle 2 k-values, reducing scalar tail to at most 1.
         * 3. Eliminated flattening copy by loading directly from sub_fft_outputs.
         * 4. Eliminated final copy by writing directly to output_buffer.
         * 5. Unrolled AVX2 loop 2x with prefetching to output_buffer at 256 bytes.
         *
         * @warning Assumes fft_obj->twiddles has n_fft ≥ N elements, scratch and output_buffer
         *          are 32-byte aligned, and factor_index is valid for twiddle_factors.
         */
        // Step 1: Compute subproblem size and stride
        int sub_fft_length = data_length / 3; // Size of each sub-FFT (N/3)
        int next_stride = 3 * stride;         // Triple stride for next level
        int sub_fft_size = sub_fft_length;    // Sub-FFT size for indexing

        // Step 2: Validate scratch buffer
        // Power-of-3: 3*(N/3) for outputs; mixed-radix: (3+2)*(N/3) = 5*(N/3) for outputs + twiddles
        int required_size = fft_obj->twiddle_factors != NULL ? 3 * sub_fft_size : 5 * sub_fft_size;
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Scratch buffer too small for radix-3 at offset %d (need %d, have %d)\n",
                    scratch_offset, required_size, fft_obj->max_scratch_size - scratch_offset);
            return;
        }

        // Step 3: Assign scratch slices
        // sub_fft_outputs: X_0, X_1, X_2 (3*sub_fft_size fft_data)
        // twiddle_factors: W_N^k, W_N^{2k} (2*sub_fft_size fft_data, mixed-radix only)
        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors;
        if (fft_obj->twiddle_factors != NULL)
        {
            // Validate factor_index for power-of-3 FFTs
            if (factor_index >= fft_obj->num_precomputed_stages)
            {
                fprintf(stderr, "Error: Invalid factor_index (%d) exceeds num_precomputed_stages (%d) for radix-3\n",
                        factor_index, fft_obj->num_precomputed_stages);
                return;
            }
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + scratch_offset + 3 * sub_fft_size;
        }

        // Step 4: Compute child scratch offsets
        // Each child needs 3*(N/9) (power-of-3) or 5*(N/9) (mixed-radix), tiled within parent block
        int child_scratch_per_branch = fft_obj->twiddle_factors != NULL ? 3 * (sub_fft_size / 3) : 5 * (sub_fft_size / 3);
        if (child_scratch_per_branch * 3 > required_size)
        {
            fprintf(stderr, "Error: Child scratch size (%d) exceeds parent allocation (%d) for radix-3\n",
                    child_scratch_per_branch * 3, required_size);
            return;
        }
        int child_offset1 = scratch_offset;                         // First child starts at parent offset
        int child_offset2 = scratch_offset + required_size / 3;     // Second child uses second third
        int child_offset3 = scratch_offset + 2 * required_size / 3; // Third child uses final third

        // Step 5: Recurse on three sub-FFTs
        // Use distinct scratch offsets to prevent overlap
        for (int i = 0; i < 3; i++)
        {
            mixed_radix_dit_rec(sub_fft_outputs + i * sub_fft_size, input_buffer + i * stride, fft_obj,
                                transform_sign, sub_fft_length, next_stride, factor_index + 1,
                                scratch_offset + i * required_size / 3);
        }

        // Step 6: Prepare twiddle factors (mixed-radix only)
        // For power-of-3 FFTs, twiddle_factors points to precomputed W_N^k, W_N^{2k} via stage_twiddle_offset
        if (fft_obj->twiddle_factors == NULL)
        {
            if (fft_obj->n_fft < 3 * sub_fft_size)
            {
                fprintf(stderr, "Error: Twiddle array too small (need at least %d elements, have %d)\n",
                        3 * sub_fft_size, fft_obj->n_fft);
                return;
            }
            for (int k = 0; k < sub_fft_size; k++)
            {
                int idx = sub_fft_length - 1 + 2 * k;                          // Index for W_N^k, W_N^{2k}
                twiddle_factors[2 * k + 0].re = fft_obj->twiddles[idx + 0].re; // W_N^k real
                twiddle_factors[2 * k + 0].im = fft_obj->twiddles[idx + 0].im; // W_N^k imag
                twiddle_factors[2 * k + 1].re = fft_obj->twiddles[idx + 1].re; // W_N^{2k} real
                twiddle_factors[2 * k + 1].im = fft_obj->twiddles[idx + 1].im; // W_N^{2k} imag
            }
        }

        // Step 7: AVX2 vectorized butterfly with 2x unroll
        // Combine: X(k), X(k+N/3), X(k+2N/3) with 120°/240° rotations
        // Load directly from sub_fft_outputs, write to output_buffer
        __m256d v_sign_factor = _mm256_set1_pd((double)transform_sign); // ±1 for transform direction
        __m256d vhalf = _mm256_set1_pd(0.5);                            // Constant for -½(b + c)
        __m256d vsqrt3by2 = _mm256_set1_pd(C3_SQRT3BY2);                // √3/2 for 120° rotation
        int k = 0;
        for (; k + 7 < sub_fft_size; k += 8)
        {
            // Prefetch output_buffer and twiddle_factors 256 bytes (32 doubles) ahead
            _mm_prefetch((const char *)&output_buffer[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&output_buffer[k + 32].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&output_buffer[k + 32 + sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&output_buffer[k + 32 + sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&output_buffer[k + 32 + 2 * sub_fft_size].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&output_buffer[k + 32 + 2 * sub_fft_size].im, _MM_HINT_T0);
            _mm_prefetch((const char *)&twiddle_factors[2 * (k + 32)].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&twiddle_factors[2 * (k + 32)].im, _MM_HINT_T0);

            // First 4 points (k to k+3)
            // Load directly from sub_fft_outputs with stride sub_fft_size
            __m256d a_r = LOAD_PD(&sub_fft_outputs[k].re);                    // X_0[k:k+3] real
            __m256d a_i = LOAD_PD(&sub_fft_outputs[k].im);                    // X_0[k:k+3] imag
            __m256d b_r = LOAD_PD(&sub_fft_outputs[k + sub_fft_size].re);     // X_1[k:k+3] real
            __m256d b_i = LOAD_PD(&sub_fft_outputs[k + sub_fft_size].im);     // X_1[k:k+3] imag
            __m256d c_r = LOAD_PD(&sub_fft_outputs[k + 2 * sub_fft_size].re); // X_2[k:k+3] real
            __m256d c_i = LOAD_PD(&sub_fft_outputs[k + 2 * sub_fft_size].im); // X_2[k:k+3] imag

            // Load twiddle factors (W_N^k, W_N^{2k})
            int idx = 2 * k;
            __m256d w1r = LOAD_PD(&twiddle_factors[idx + 0].re); // W_N^k real
            __m256d w1i = LOAD_PD(&twiddle_factors[idx + 0].im); // W_N^k imag
            __m256d w2r = LOAD_PD(&twiddle_factors[idx + 1].re); // W_N^{2k} real
            __m256d w2i = LOAD_PD(&twiddle_factors[idx + 1].im); // W_N^{2k} imag

            // Apply twiddle factors: (b/c) * W_N^k
            __m256d b2r = FMSUB(b_r, w1r, _mm256_mul_pd(b_i, w1i)); // b_r * w1r - b_i * w1i
            __m256d b2i = FMADD(b_i, w1r, _mm256_mul_pd(b_r, w1i)); // b_i * w1r + b_r * w1i
            __m256d c2r = FMSUB(c_r, w2r, _mm256_mul_pd(c_i, w2i)); // c_r * w2r - c_i * w2i
            __m256d c2i = FMADD(c_i, w2r, _mm256_mul_pd(c_r, w2i)); // c_i * w2r + c_r * w2i

            // Compute sums and differences
            __m256d sum_r = _mm256_add_pd(b2r, c2r);  // b + c real
            __m256d sum_i = _mm256_add_pd(b2i, c2i);  // b + c imag
            __m256d diff_r = _mm256_sub_pd(b2r, c2r); // b - c real
            __m256d diff_i = _mm256_sub_pd(b2i, c2i); // b - c imag

            // X_0 = a + (b + c)
            __m256d x0r = _mm256_add_pd(a_r, sum_r); // X(k) real
            __m256d x0i = _mm256_add_pd(a_i, sum_i); // X(k) imag
            STORE_PD(&output_buffer[k].re, x0r);     // Write to output_buffer
            STORE_PD(&output_buffer[k].im, x0i);

            // X_1 = a - ½(b + c) + i * (sign * (√3/2) * (b - c))
            __m256d t_r = _mm256_sub_pd(a_r, _mm256_mul_pd(sum_r, vhalf));                                                      // a - ½(b + c) real
            __m256d t_i = _mm256_sub_pd(a_i, _mm256_mul_pd(sum_i, vhalf));                                                      // a - ½(b + c) imag
            __m256d rot_r = _mm256_mul_pd(diff_i, _mm256_mul_pd(v_sign_factor, vsqrt3by2));                                     // sign * (√3/2) * (b - c) real
            __m256d rot_i = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), diff_r), _mm256_mul_pd(v_sign_factor, vsqrt3by2)); // -sign * (√3/2) * (b - c) imag
            __m256d x1r = _mm256_add_pd(t_r, rot_r);                                                                            // X(k + N/3) real
            __m256d x1i = _mm256_add_pd(t_i, rot_i);                                                                            // X(k + N/3) imag
            STORE_PD(&output_buffer[k + sub_fft_size].re, x1r);                                                                 // Write to output_buffer
            STORE_PD(&output_buffer[k + sub_fft_size].im, x1i);

            // X_2 = a - ½(b + c) - i * (sign * (√3/2) * (b - c))
            __m256d x2r = _mm256_sub_pd(t_r, rot_r);                // X(k + 2N/3) real
            __m256d x2i = _mm256_sub_pd(t_i, rot_i);                // X(k + 2N/3) imag
            STORE_PD(&output_buffer[k + 2 * sub_fft_size].re, x2r); // Write to output_buffer
            STORE_PD(&output_buffer[k + 2 * sub_fft_size].im, x2i);

            // Second 4 points (k+4 to k+7)
            __m256d a_r2 = LOAD_PD(&sub_fft_outputs[k + 4].re);
            __m256d a_i2 = LOAD_PD(&sub_fft_outputs[k + 4].im);
            __m256d b_r2 = LOAD_PD(&sub_fft_outputs[k + 4 + sub_fft_size].re);
            __m256d b_i2 = LOAD_PD(&sub_fft_outputs[k + 4 + sub_fft_size].im);
            __m256d c_r2 = LOAD_PD(&sub_fft_outputs[k + 4 + 2 * sub_fft_size].re);
            __m256d c_i2 = LOAD_PD(&sub_fft_outputs[k + 4 + 2 * sub_fft_size].im);

            idx = 2 * (k + 4);
            __m256d w1r2 = LOAD_PD(&twiddle_factors[idx + 0].re);
            __m256d w1i2 = LOAD_PD(&twiddle_factors[idx + 0].im);
            __m256d w2r2 = LOAD_PD(&twiddle_factors[idx + 1].re);
            __m256d w2i2 = LOAD_PD(&twiddle_factors[idx + 1].im);

            __m256d b2r2 = FMSUB(b_r2, w1r2, _mm256_mul_pd(b_i2, w1i2));
            __m256d b2i2 = FMADD(b_i2, w1r2, _mm256_mul_pd(b_r2, w1i2));
            __m256d c2r2 = FMSUB(c_r2, w2r2, _mm256_mul_pd(c_i2, w2i2));
            __m256d c2i2 = FMADD(c_i2, w2r2, _mm256_mul_pd(c_r2, w2i2));

            __m256d sum_r2 = _mm256_add_pd(b2r2, c2r2);
            __m256d sum_i2 = _mm256_add_pd(b2i2, c2i2);
            __m256d diff_r2 = _mm256_sub_pd(b2r2, c2r2);
            __m256d diff_i2 = _mm256_sub_pd(b2i2, c2i2);

            __m256d x0r2 = _mm256_add_pd(a_r2, sum_r2);
            __m256d x0i2 = _mm256_add_pd(a_i2, sum_i2);
            STORE_PD(&output_buffer[k + 4].re, x0r2);
            STORE_PD(&output_buffer[k + 4].im, x0i2);

            __m256d t_r2 = _mm256_sub_pd(a_r2, _mm256_mul_pd(sum_r2, vhalf));
            __m256d t_i2 = _mm256_sub_pd(a_i2, _mm256_mul_pd(sum_i2, vhalf));
            __m256d rot_r2 = _mm256_mul_pd(diff_i2, _mm256_mul_pd(v_sign_factor, vsqrt3by2));
            __m256d rot_i2 = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), diff_r2), _mm256_mul_pd(v_sign_factor, vsqrt3by2));
            __m256d x1r2 = _mm256_add_pd(t_r2, rot_r2);
            __m256d x1i2 = _mm256_add_pd(t_i2, rot_i2);
            STORE_PD(&output_buffer[k + 4 + sub_fft_size].re, x1r2);
            STORE_PD(&output_buffer[k + 4 + sub_fft_size].im, x1i2);

            __m256d x2r2 = _mm256_sub_pd(t_r2, rot_r2);
            __m256d x2i2 = _mm256_sub_pd(t_i2, rot_i2);
            STORE_PD(&output_buffer[k + 4 + 2 * sub_fft_size].re, x2r2);
            STORE_PD(&output_buffer[k + 4 + 2 * sub_fft_size].im, x2i2);
        }

        // Step 8: SSE2 tail for remaining 2 points
        // Handle k-values not covered by AVX2 loop
        if (k + 1 < sub_fft_size)
        {
            // Preload constants for SSE2
            __m128d v_sign_factor_sse2 = _mm_set1_pd((double)transform_sign); // ±1 for transform direction
            __m128d vhalf_sse2 = _mm_set1_pd(0.5);                            // Constant for -½(b + c)
            __m128d vsqrt3by2_sse2 = _mm_set1_pd(C3_SQRT3BY2);                // √3/2 for 120° rotation

            // Load directly from sub_fft_outputs
            __m128d a_r = LOAD_SSE2(&sub_fft_outputs[k].re);                    // X_0[k:k+1] real
            __m128d a_i = LOAD_SSE2(&sub_fft_outputs[k].im);                    // X_0[k:k+1] imag
            __m128d b_r = LOAD_SSE2(&sub_fft_outputs[k + sub_fft_size].re);     // X_1[k:k+1] real
            __m128d b_i = LOAD_SSE2(&sub_fft_outputs[k + sub_fft_size].im);     // X_1[k:k+1] imag
            __m128d c_r = LOAD_SSE2(&sub_fft_outputs[k + 2 * sub_fft_size].re); // X_2[k:k+1] real
            __m128d c_i = LOAD_SSE2(&sub_fft_outputs[k + 2 * sub_fft_size].im); // X_2[k:k+1] imag

            // Load twiddle factors
            int idx = 2 * k;
            __m128d w1r = LOAD_SSE2(&twiddle_factors[idx + 0].re); // W_N^k real
            __m128d w1i = LOAD_SSE2(&twiddle_factors[idx + 0].im); // W_N^k imag
            __m128d w2r = LOAD_SSE2(&twiddle_factors[idx + 1].re); // W_N^{2k} real
            __m128d w2i = LOAD_SSE2(&twiddle_factors[idx + 1].im); // W_N^{2k} imag

            // Apply twiddle factors
            __m128d b2r = FMSUB_SSE2(b_r, w1r, _mm_mul_pd(b_i, w1i)); // b_r * w1r - b_i * w1i
            __m128d b2i = FMADD_SSE2(b_i, w1r, _mm_mul_pd(b_r, w1i)); // b_i * w1r + b_r * w1i
            __m128d c2r = FMSUB_SSE2(c_r, w2r, _mm_mul_pd(c_i, w2i)); // c_r * w2r - c_i * w2i
            __m128d c2i = FMADD_SSE2(c_i, w2r, _mm_mul_pd(c_r, w2i)); // c_i * w2r + c_r * w2i

            // Compute sums and differences
            __m128d sum_r = _mm_add_pd(b2r, c2r);  // b + c real
            __m128d sum_i = _mm_add_pd(b2i, c2i);  // b + c imag
            __m128d diff_r = _mm_sub_pd(b2r, c2r); // b - c real
            __m128d diff_i = _mm_sub_pd(b2i, c2i); // b - c imag

            // X_0 = a + (b + c)
            __m128d x0r = _mm_add_pd(a_r, sum_r);  // X(k) real
            __m128d x0i = _mm_add_pd(a_i, sum_i);  // X(k) imag
            STORE_SSE2(&output_buffer[k].re, x0r); // Write to output_buffer
            STORE_SSE2(&output_buffer[k].im, x0i);

            // X_1 = a - ½(b + c) + i * (sign * (√3/2) * (b - c))
            __m128d t_r = _mm_sub_pd(a_r, _mm_mul_pd(sum_r, vhalf_sse2));                                                     // a - ½(b + c) real
            __m128d t_i = _mm_sub_pd(a_i, _mm_mul_pd(sum_i, vhalf_sse2));                                                     // a - ½(b + c) imag
            __m128d rot_r = _mm_mul_pd(diff_i, _mm_mul_pd(v_sign_factor_sse2, vsqrt3by2_sse2));                               // sign * (√3/2) * (b - c) real
            __m128d rot_i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), diff_r), _mm_mul_pd(v_sign_factor_sse2, vsqrt3by2_sse2)); // -sign * (√3/2) * (b - c) imag
            __m128d x1r = _mm_add_pd(t_r, rot_r);                                                                             // X(k + N/3) real
            __m128d x1i = _mm_add_pd(t_i, rot_i);                                                                             // X(k + N/3) imag
            STORE_SSE2(&output_buffer[k + sub_fft_size].re, x1r);                                                             // Write to output_buffer
            STORE_SSE2(&output_buffer[k + sub_fft_size].im, x1i);

            // X_2 = a - ½(b + c) - i * (sign * (√3/2) * (b - c))
            __m128d x2r = _mm_sub_pd(t_r, rot_r);                     // X(k + 2N/3) real
            __m128d x2i = _mm_sub_pd(t_i, rot_i);                     // X(k + 2N/3) imag
            STORE_SSE2(&output_buffer[k + 2 * sub_fft_size].re, x2r); // Write to output_buffer
            STORE_SSE2(&output_buffer[k + 2 * sub_fft_size].im, x2i);
            k += 2;
        }

        // Step 9: Scalar tail for one remaining point
        // Handle any k not covered by AVX2/SSE2 loops (e.g., sub_fft_size odd)
        if (k < sub_fft_size)
        {
            // Load twiddle factors
            fft_type w1r = twiddle_factors[2 * k + 0].re; // W_N^k real
            fft_type w1i = twiddle_factors[2 * k + 0].im; // W_N^k imag
            fft_type w2r = twiddle_factors[2 * k + 1].re; // W_N^{2k} real
            fft_type w2i = twiddle_factors[2 * k + 1].im; // W_N^{2k} imag;

            // Load sub-FFT points directly from sub_fft_outputs
            fft_type a_re = sub_fft_outputs[k].re;                    // X_0[k] real
            fft_type a_im = sub_fft_outputs[k].im;                    // X_0[k] imag
            fft_type b_re = sub_fft_outputs[k + sub_fft_size].re;     // X_1[k] real
            fft_type b_im = sub_fft_outputs[k + sub_fft_size].im;     // X_1[k] imag
            fft_type c_re = sub_fft_outputs[k + 2 * sub_fft_size].re; // X_2[k] real
            fft_type c_im = sub_fft_outputs[k + 2 * sub_fft_size].im; // X_2[k] imag

            // Apply twiddle factors
            fft_type b2_re = b_re * w1r - b_im * w1i; // W_N^k * X_1 real
            fft_type b2_im = b_im * w1r + b_re * w1i; // W_N^k * X_1 imag
            fft_type c2_re = c_re * w2r - c_im * w2i; // W_N^{2k} * X_2 real
            fft_type c2_im = c_im * w2r + c_re * w2i; // W_N^{2k} * X_2 imag

            // Compute sums and differences
            fft_type sum_re = b2_re + c2_re;  // b + c real
            fft_type sum_im = b2_im + c2_im;  // b + c imag
            fft_type diff_re = b2_re - c2_re; // b - c real
            fft_type diff_im = b2_im - c2_im; // b - c imag

            // X_0 = a + (b + c)
            output_buffer[k].re = a_re + sum_re; // X(k) real
            output_buffer[k].im = a_im + sum_im; // X(k) imag

            // X_1 = a - ½(b + c) + i * (sign * (√3/2) * (b - c))
            fft_type t_re = a_re - (sum_re * 0.5);                       // a - ½(b + c) real
            fft_type t_im = a_im - (sum_im * 0.5);                       // a - ½(b + c) imag
            fft_type rot_re = diff_im * (transform_sign * C3_SQRT3BY2);  // sign * (√3/2) * (b - c) real
            fft_type rot_im = -diff_re * (transform_sign * C3_SQRT3BY2); // -sign * (√3/2) * (b - c) imag
            output_buffer[k + sub_fft_size].re = t_re + rot_re;          // X(k + N/3) real
            output_buffer[k + sub_fft_size].im = t_im + rot_im;          // X(k + N/3) imag

            // X_2 = a - ½(b + c) - i * (sign * (√3/2) * (b - c))
            output_buffer[k + 2 * sub_fft_size].re = t_re - rot_re; // X(k + 2N/3) real
            output_buffer[k + 2 * sub_fft_size].im = t_im - rot_im; // X(k + 2N/3) imag
        }
    }
    else if (radix == 4)
    {
        /**
         * @brief Radix-4 decomposition for four-point sub-FFTs with SSE2 vectorization and FMA support.
         *
         * Intention: Optimize FFT computation for data lengths divisible by 4 by splitting into
         * four sub-FFTs, corresponding to indices n mod 4 = 0, 1, 2, 3, and combining results
         * with twiddle factors. Efficient for power-of-4 (N=4^r) and mixed-radix FFTs using
         * pre-allocated scratch and stage-specific twiddle offsets. Follows FFTW’s two-buffer
         * strategy for thread safety and performance.
         *
         * Mathematically: Computes:
         *   X(k) = X_0(k) + W_N^k * X_1(k) + W_N^{2k} * X_2(k) + W_N^{3k} * X_3(k),
         * where X_0, X_1, X_2, X_3 are sub-FFTs of size N/4, W_N^k = e^{-2πi k / N}.
         * The radix-4 butterfly uses 90°, 180°, and 270° rotations
         */
        // Step 1: Compute subproblem size and stride
        int sub_fft_length = data_length / 4; // Size of each sub-FFT (N/4)
        int next_stride = 4 * stride;         // Stride quadruples for next level
        int sub_fft_size = sub_fft_length;    // Sub-FFT size for indexing

        // Step 2: Validate scratch buffer
        // Power-of-4: 4*(N/4) for outputs; mixed-radix: (4+3)*(N/4) = 7*(N/4) for outputs + twiddles
        int required_size = fft_obj->twiddle_factors != NULL ? 4 * sub_fft_size : 7 * sub_fft_size;
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Scratch buffer too small for radix-4 at offset %d (need %d, have %d)\n",
                    scratch_offset, required_size, fft_obj->max_scratch_size - scratch_offset);
            // exit
        }

        // Step 3: Assign scratch slices
        // sub_fft_outputs: X_0, X_1, X_2, X_3 (4*sub_fft_size fft_data)
        // twiddle_factors: W_N^k, W_N^{2k}, W_N^{3k} (3*sub_fft_size fft_data, mixed-radix only)
        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors;
        if (fft_obj->twiddle_factors != NULL)
        {
            // Validate factor_index for power-of-4 FFTs
            if (factor_index >= fft_obj->num_precomputed_stages)
            {
                fprintf(stderr, "Error: Invalid factor_index (%d) exceeds num_precomputed_stages (%d) for radix-4\n",
                        factor_index, fft_obj->num_precomputed_stages);
                // exit
            }
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + scratch_offset + 4 * sub_fft_size;
        }

        // Step 4: Compute child scratch offsets
        // Each child needs 4*(N/16) (power-of-4) or 7*(N/16) (mixed-radix), tiled within parent block
        int child_scratch_per_branch = fft_obj->twiddle_factors != NULL ? 4 * (sub_fft_size / 4) : 7 * (sub_fft_size / 4);
        if (child_scratch_per_branch * 4 > required_size)
        {
            fprintf(stderr, "Error: Child scratch size (%d) exceeds parent allocation (%d) for radix-4\n",
                    child_scratch_per_branch * 4, required_size);
            // exit
        }
        int child_offsets[4];
        for (int i = 0; i < 4; i++)
        {
            child_offsets[i] = scratch_offset + i * (required_size / 4); // Divide scratch equally
        }

        // Step 5: Recurse on four sub-FFTs
        for (int i = 0; i < 4; i++)
        {
            mixed_radix_dit_rec(sub_fft_outputs + i * sub_fft_size, input_buffer + i * stride, fft_obj,
                                transform_sign, sub_fft_length, next_stride, factor_index + 1,
                                child_offsets[i]);
        }

        // Step 6: Prepare twiddle factors (mixed-radix only)
        if (fft_obj->twiddle_factors == NULL)
        {
            if (fft_obj->n_fft < sub_fft_length - 1 + 3 * sub_fft_size)
            {
                fprintf(stderr, "Error: Twiddle array too small (need at least %d elements, have %d)\n",
                        sub_fft_length - 1 + 3 * sub_fft_size, fft_obj->n_fft);
                // exit
            }
#ifdef USE_TWIDDLE_TABLES
            if (sub_fft_size <= 4 && twiddle_tables[4] != NULL)
            {
                const complex_t *table = twiddle_tables[4];
                for (int k = 0; k < sub_fft_size; k++)
                {
                    for (int n = 0; n < 3; n++)
                    {
                        twiddle_factors[3 * k + n].re = table[n + 1].re; // W_N^{n*k} real
                        twiddle_factors[3 * k + n].im = table[n + 1].im; // W_N^{n*k} imag
                    }
                }
            }
            else
#endif
            {
                for (int k = 0; k < sub_fft_size; k++)
                {
                    int idx = sub_fft_length - 1 + 3 * k; // Index for W_N^k, W_N^{2k}, W_N^{3k}
                    for (int n = 0; n < 3; n++)
                    {
                        twiddle_factors[3 * k + n].re = fft_obj->twiddles[idx + n].re; // W_N^{n*k} real
                        twiddle_factors[3 * k + n].im = fft_obj->twiddles[idx + n].im; // W_N^{n*k} imag
                    }
                }
            }
        }

        // Step 7: Flatten outputs into contiguous arrays in scratch
        // Use sub_fft_outputs for 4*sub_fft_size fft_data (8*sub_fft_size doubles)
        // out_re: first 4*sub_fft_size doubles; out_im: next 4*sub_fft_size doubles
        double *out_re = (double *)sub_fft_outputs;
        double *out_im = (double *)sub_fft_outputs + 4 * sub_fft_size;
        for (int lane = 0; lane < 4; lane++)
        {
            fft_data *base = sub_fft_outputs + lane * sub_fft_size;
            for (int k = 0; k < sub_fft_size; k++)
            {
                out_re[lane * sub_fft_size + k] = base[k].re;
                out_im[lane * sub_fft_size + k] = base[k].im;
            }
        }

        // Step 8: SSE2 vectorized butterfly for k=0 to sub_fft_size-1
        __m128d vsign = _mm_set1_pd((double)transform_sign); // Vectorized transform sign
        int k = 0;
        for (; k + 1 < sub_fft_size; k += 2)
        {
            // Handle k=0 separately if k=0 is in this loop
            if (k == 0)
            {
                fft_data *X0 = &output_buffer[0];                // First sub-FFT point
                fft_data *X1 = &output_buffer[sub_fft_size];     // Second sub-FFT point
                fft_data *X2 = &output_buffer[2 * sub_fft_size]; // Third sub-FFT point
                fft_data *X3 = &output_buffer[3 * sub_fft_size]; // Fourth sub-FFT point

                fft_type a_re = X0->re + X2->re;                    // Sum of first and third (real)
                fft_type a_im = X0->im + X2->im;                    // Sum of first and third (imag)
                fft_type b_re = X0->re - X2->re;                    // Difference of first and third (real)
                fft_type b_im = X0->im - X2->im;                    // Difference of first and third (imag)
                fft_type c_re = X1->re + X3->re;                    // Sum of second and fourth (real)
                fft_type c_im = X1->im + X3->im;                    // Sum of second and fourth (imag)
                fft_type d_re = transform_sign * (X1->re - X3->re); // Rotated difference (real, 90°)
                fft_type d_im = transform_sign * (X1->im - X3->im); // Rotated difference (imag, 90°)

                X0->re = a_re + c_re; // X(0) real
                X0->im = a_im + c_im; // X(0) imag
                X2->re = a_re - c_re; // X(2) real
                X2->im = a_im - c_im; // X(2) imag
                X1->re = b_re + d_im; // X(1) real
                X1->im = b_im - d_re; // X(1) imag
                X3->re = b_re - d_im; // X(3) real
                X3->im = b_im + d_re; // X(3) imag
                continue;
            }

            // Load four sub-FFT points for two k values (32-byte aligned with USE_ALIGNED_SIMD)
            __m128d a_r = LOAD_SSE2(out_re + k);                    // X_0[k:k+1] real
            __m128d a_i = LOAD_SSE2(out_im + k);                    // X_0[k:k+1] imag
            __m128d b_r = LOAD_SSE2(out_re + k + sub_fft_size);     // X_1[k:k+1] real
            __m128d b_i = LOAD_SSE2(out_im + k + sub_fft_size);     // X_1[k:k+1] imag
            __m128d c_r = LOAD_SSE2(out_re + k + 2 * sub_fft_size); // X_2[k:k+1] real
            __m128d c_i = LOAD_SSE2(out_im + k + 2 * sub_fft_size); // X_2[k:k+1] imag
            __m128d d_r = LOAD_SSE2(out_re + k + 3 * sub_fft_size); // X_3[k:k+1] real
            __m128d d_i = LOAD_SSE2(out_im + k + 3 * sub_fft_size); // X_3[k:k+1] imag

            // Load twiddle factors for two k values (aligned for power-of-4, scratch for mixed-radix)
            int idx = 3 * k;                                       // Base index for W_N^k, W_N^{2k}, W_N^{3k}
            __m128d w1r = LOAD_SSE2(&twiddle_factors[idx + 0].re); // W_N^k real
            __m128d w1i = LOAD_SSE2(&twiddle_factors[idx + 0].im); // W_N^k imag
            __m128d w2r = LOAD_SSE2(&twiddle_factors[idx + 1].re); // W_N^{2k} real
            __m128d w2i = LOAD_SSE2(&twiddle_factors[idx + 1].im); // W_N^{2k} imag
            __m128d w3r = LOAD_SSE2(&twiddle_factors[idx + 2].re); // W_N^{3k} real
            __m128d w3i = LOAD_SSE2(&twiddle_factors[idx + 2].im); // W_N^{3k} imag

            // Apply twiddle factors to X_1, X_2, X_3
            // FMSUB_SSE2(a, b, c) computes a*b - c, FMADD_SSE2(a, b, c) computes a*b + c
            __m128d b2r = FMSUB_SSE2(b_r, w1r, _mm_mul_pd(b_i, w1i)); // W_N^k * X_1 real
            __m128d b2i = FMADD_SSE2(b_i, w1r, _mm_mul_pd(b_r, w1i)); // W_N^k * X_1 imag
            __m128d c2r = FMSUB_SSE2(c_r, w2r, _mm_mul_pd(c_i, w2i)); // W_N^{2k} * X_2 real
            __m128d c2i = FMADD_SSE2(c_i, w2r, _mm_mul_pd(c_r, w2i)); // W_N^{2k} * X_2 imag
            __m128d d2r = FMSUB_SSE2(d_r, w3r, _mm_mul_pd(d_i, w3i)); // W_N^{3k} * X_3 real
            __m128d d2i = FMADD_SSE2(d_i, w3r, _mm_mul_pd(d_r, w3i)); // W_N^{3k} * X_3 imag

            // Radix-4 butterfly
            __m128d a2r = _mm_add_pd(a_r, c2r);                    // X_0 + W_N^{2k} * X_2 real
            __m128d a2i = _mm_add_pd(a_i, c2i);                    // X_0 + W_N^{2k} * X_2 imag
            __m128d e_r = _mm_sub_pd(a_r, c2r);                    // X_0 - W_N^{2k} * X_2 real
            __m128d e_i = _mm_sub_pd(a_i, c2i);                    // X_0 - W_N^{2k} * X_2 imag
            __m128d f_r = _mm_add_pd(b2r, d2r);                    // W_N^k * X_1 + W_N^{3k} * X_3 real
            __m128d f_i = _mm_add_pd(b2i, d2i);                    // W_N^k * X_1 + W_N^{3k} * X_3 imag
            __m128d g_r = _mm_mul_pd(_mm_sub_pd(b2r, d2r), vsign); // (W_N^k * X_1 - W_N^{3k} * X_3) * sign real
            __m128d g_i = _mm_mul_pd(_mm_sub_pd(b2i, d2i), vsign); // (W_N^k * X_1 - W_N^{3k} * X_3) * sign imag

            // Compute outputs
            STORE_SSE2(out_re + k, _mm_add_pd(a2r, f_r));                    // X(k) real
            STORE_SSE2(out_im + k, _mm_add_pd(a2i, f_i));                    // X(k) imag
            STORE_SSE2(out_re + k + 2 * sub_fft_size, _mm_sub_pd(a2r, f_r)); // X(k + N/4) real
            STORE_SSE2(out_im + k + 2 * sub_fft_size, _mm_sub_pd(a2i, f_i)); // X(k + N/4) imag
            STORE_SSE2(out_re + k + sub_fft_size, _mm_add_pd(e_r, g_i));     // X(k + N/2) real
            STORE_SSE2(out_im + k + sub_fft_size, _mm_sub_pd(e_i, g_r));     // X(k + N/2) imag
            STORE_SSE2(out_re + k + 3 * sub_fft_size, _mm_sub_pd(e_r, g_i)); // X(k + 3N/4) real
            STORE_SSE2(out_im + k + 3 * sub_fft_size, _mm_add_pd(e_i, g_r)); // X(k + 3N/4) imag
        }

        // Step 9: Scalar tail for remaining k
        for (; k < sub_fft_size; k++)
        {
            // Handle k=0 separately if not covered in vectorized loop
            if (k == 0)
            {
                fft_data *X0 = &output_buffer[0];                // First sub-FFT point
                fft_data *X1 = &output_buffer[sub_fft_size];     // Second sub-FFT point
                fft_data *X2 = &output_buffer[2 * sub_fft_size]; // Third sub-FFT point
                fft_data *X3 = &output_buffer[3 * sub_fft_size]; // Fourth sub-FFT point

                fft_type a_re = X0->re + X2->re;                    // Sum of first and third (real)
                fft_type a_im = X0->im + X2->im;                    // Sum of first and third (imag)
                fft_type b_re = X0->re - X2->re;                    // Difference of first and third (real)
                fft_type b_im = X0->im - X2->im;                    // Difference of first and third (imag)
                fft_type c_re = X1->re + X3->re;                    // Sum of second and fourth (real)
                fft_type c_im = X1->im + X3->im;                    // Sum of second and fourth (imag)
                fft_type d_re = transform_sign * (X1->re - X3->re); // Rotated difference (real, 90°)
                fft_type d_im = transform_sign * (X1->im - X3->im); // Rotated difference (imag, 90°)

                X0->re = a_re + c_re; // X(0) real
                X0->im = a_im + c_im; // X(0) imag
                X2->re = a_re - c_re; // X(2) real
                X2->im = a_im - c_im; // X(2) imag
                X1->re = b_re + d_im; // X(1) real
                X1->im = b_im - d_re; // X(1) imag
                X3->re = b_re - d_im; // X(3) real
                X3->im = b_im + d_re; // X(3) imag
                continue;
            }

            // Load twiddle factors
            int idx = 3 * k;                            // Index into twiddle array
            fft_type w1r = twiddle_factors[idx + 0].re; // W_N^k real
            fft_type w1i = twiddle_factors[idx + 0].im; // W_N^k imag
            fft_type w2r = twiddle_factors[idx + 1].re; // W_N^{2k} real
            fft_type w2i = twiddle_factors[idx + 1].im; // W_N^{2k} imag
            fft_type w3r = twiddle_factors[idx + 2].re; // W_N^{3k} real
            fft_type w3i = twiddle_factors[idx + 2].im; // W_N^{3k} imag

            // Load the four sub-FFT points
            fft_type a_re = out_re[k];                    // X_0[k] real
            fft_type a_im = out_im[k];                    // X_0[k] imag
            fft_type b_re = out_re[k + sub_fft_size];     // X_1[k] real
            fft_type b_im = out_im[k + sub_fft_size];     // X_1[k] imag
            fft_type c_re = out_re[k + 2 * sub_fft_size]; // X_2[k] real
            fft_type c_im = out_im[k + 2 * sub_fft_size]; // X_2[k] imag
            fft_type d_re = out_re[k + 3 * sub_fft_size]; // X_3[k] real
            fft_type d_im = out_im[k + 3 * sub_fft_size]; // X_3[k] imag

            // Apply twiddle factors to X_1, X_2, X_3
            fft_type b2_re = b_re * w1r - b_im * w1i; // W_N^k * X_1 real
            fft_type b2_im = b_im * w1r + b_re * w1i; // W_N^k * X_1 imag
            fft_type c2_re = c_re * w2r - c_im * w2i; // W_N^{2k} * X_2 real
            fft_type c2_im = c_im * w2r + c_re * w2i; // W_N^{2k} * X_2 imag
            fft_type d2_re = d_re * w3r - d_im * w3i; // W_N^{3k} * X_3 real
            fft_type d2_im = d_im * w3r + d_re * w3i; // W_N^{3k} * X_3 imag

            // Radix-4 butterfly
            fft_type a2_re = a_re + c2_re;                    // X_0 + W_N^{2k} * X_2 real
            fft_type a2_im = a_im + c2_im;                    // X_0 + W_N^{2k} * X_2 imag
            fft_type e_re = a_re - c2_re;                     // X_0 - W_N^{2k} * X_2 real
            fft_type e_im = a_im - c2_im;                     // X_0 - W_N^{2k} * X_2 imag
            fft_type f_re = b2_re + d2_re;                    // W_N^k * X_1 + W_N^{3k} * X_3 real
            fft_type f_im = b2_im + d2_im;                    // W_N^k * X_1 + W_N^{3k} * X_3 imag
            fft_type g_re = transform_sign * (b2_re - d2_re); // (W_N^k * X_1 - W_N^{3k} * X_3) * sign real
            fft_type g_im = transform_sign * (b2_im - d2_im); // (W_N^k * X_1 - W_N^{3k} * X_3) * sign imag

            // Store results
            out_re[k] = a2_re + f_re;                    // X(k) real
            out_im[k] = a2_im + f_im;                    // X(k) imag
            out_re[k + 2 * sub_fft_size] = a2_re - f_re; // X(k + N/4) real
            out_im[k + 2 * sub_fft_size] = a2_im - f_im; // X(k + N/4) imag
            out_re[k + sub_fft_size] = e_re + g_im;      // X(k + N/2) real
            out_im[k + sub_fft_size] = e_im - g_re;      // X(k + N/2) imag
            out_re[k + 3 * sub_fft_size] = e_re - g_im;  // X(k + 3N/4) real
            out_im[k + 3 * sub_fft_size] = e_im + g_re;  // X(k + 3N/4) imag
        }

        // Step 10: Copy flattened results back to output_buffer
        for (int lane = 0; lane < 4; lane++)
        {
            fft_data *base = output_buffer + lane * sub_fft_size;
            for (int k = 0; k < sub_fft_size; k++)
            {
                base[k].re = out_re[lane * sub_fft_size + k];
                base[k].im = out_im[lane * sub_fft_size + k];
            }
        }
    }
    else if (radix == 5)
    {
        /**
         * @brief Radix-5 decomposition for five-point sub-FFTs with SSE2 vectorization and FMA support.
         *
         * Intention: Compute the FFT for data lengths divisible by 5 by splitting into five sub-FFTs,
         * corresponding to indices n mod 5 = 0, 1, 2, 3, 4, and combining results with twiddle factors.
         * Optimized for power-of-5 (N=5^r) and mixed-radix FFTs using pre-allocated scratch and
         * stage-specific twiddle offsets. Follows FFTW’s two-buffer strategy for thread safety and
         * performance.
         *
         * Mathematically: Computes:
         *   X(k) = X_0(k) + W_N^k * X_1(k) + W_N^{2k} * X_2(k) + W_N^{3k} * X_3(k) + W_N^{4k} * X_4(k),
         * where X_0, ..., X_4 are sub-FFTs of size N/5, W_N^k = e^{-2πi k / N}.
         * The butterfly uses rotations at 72°, 144°, 216°, and 288° with constants C5_1, C5_2, S5_1, S5_2.
         *
         */
        // Step 1: Compute subproblem size and stride
        int sub_fft_length = data_length / 5; // Size of each sub-FFT (N/5)
        int next_stride = 5 * stride;         // Stride increases fivefold
        int sub_fft_size = sub_fft_length;    // Sub-FFT size for indexing

        // Step 2: Validate scratch buffer
        // Power-of-5: 5*(N/5) for outputs; mixed-radix: (5+4)*(N/5) = 9*(N/5) for outputs + twiddles
        int required_size = fft_obj->twiddle_factors != NULL ? 5 * sub_fft_size : 9 * sub_fft_size;
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Scratch buffer too small for radix-5 at offset %d (need %d, have %d)\n",
                    scratch_offset, required_size, fft_obj->max_scratch_size - scratch_offset);
            // exit
        }

        // Step 3: Assign scratch slices
        // sub_fft_outputs: X_0, X_1, X_2, X_3, X_4 (5*sub_fft_size fft_data)
        // twiddle_factors: W_N^k, W_N^{2k}, W_N^{3k}, W_N^{4k} (4*sub_fft_size fft_data, mixed-radix only)
        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors;
        if (fft_obj->twiddle_factors != NULL)
        {
            // Validate factor_index for power-of-5 FFTs
            if (factor_index >= fft_obj->num_precomputed_stages)
            {
                fprintf(stderr, "Error: Invalid factor_index (%d) exceeds num_precomputed_stages (%d) for radix-5\n",
                        factor_index, fft_obj->num_precomputed_stages);
                // exit
            }
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + scratch_offset + 5 * sub_fft_size;
        }

        // Step 4: Compute child scratch offsets
        // Each child needs ceil(sub_fft_size/5)*5 (power-of-5) or ceil(sub_fft_size/5)*9 (mixed-radix)
        int child_sub_fft_size = (sub_fft_size + 4) / 5; // Ceiling division for next recursion level
        int child_scratch_per_branch = fft_obj->twiddle_factors != NULL ? 5 * child_sub_fft_size : 9 * child_sub_fft_size;
        int total_child_scratch = 5 * child_scratch_per_branch; // Total for all 5 children
        if (scratch_offset + total_child_scratch + (fft_obj->twiddle_factors ? 0 : 4 * sub_fft_size) > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Total child scratch size (%d) exceeds available scratch at offset %d (have %d) for radix-5\n",
                    total_child_scratch + (fft_obj->twiddle_factors ? 0 : 4 * sub_fft_size),
                    scratch_offset, fft_obj->max_scratch_size - scratch_offset);
            // exit
        }
        int child_offsets[5];
        child_offsets[0] = scratch_offset;
        for (int i = 1; i < 5; i++)
        {
            child_offsets[i] = child_offsets[i - 1] + child_scratch_per_branch; // Cumulative offsets
        }

        // Step 5: Recurse on five sub-FFTs
        for (int i = 0; i < 5; i++)
        {
            mixed_radix_dit_rec(sub_fft_outputs + i * sub_fft_size, input_buffer + i * stride, fft_obj,
                                transform_sign, sub_fft_length, next_stride, factor_index + 1,
                                child_offsets[i]);
        }

        // Step 6: Prepare twiddle factors (mixed-radix only)
        if (fft_obj->twiddle_factors == NULL)
        {
            if (fft_obj->n_fft < sub_fft_length - 1 + 4 * sub_fft_size)
            {
                fprintf(stderr, "Error: Twiddle array too small (need at least %d elements, have %d)\n",
                        sub_fft_length - 1 + 4 * sub_fft_size, fft_obj->n_fft);
                // exit
            }
#ifdef USE_TWIDDLE_TABLES
            if (sub_fft_size <= 5 && twiddle_tables[5] != NULL)
            {
                const complex_t *table = twiddle_tables[5];
                for (int k = 0; k < sub_fft_size; k++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        twiddle_factors[4 * k + n].re = table[n + 1].re; // W_N^{n*k} real
                        twiddle_factors[4 * k + n].im = table[n + 1].im; // W_N^{n*k} imag
                    }
                }
            }
            else
#endif
            {
                for (int k = 0; k < sub_fft_size; k++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        int idx = n * sub_fft_size + k;                            // Correct indexing for W_N^{n*k}
                        twiddle_factors[4 * k + n].re = fft_obj->twiddles[idx].re; // W_N^{n*k} real
                        twiddle_factors[4 * k + n].im = fft_obj->twiddles[idx].im; // W_N^{n*k} imag
                    }
                }
            }
        }

        // Step 7: Flatten outputs into separate real/imag arrays using union
        union
        {
            fft_data data[5 * sub_fft_size];
            struct
            {
                double re[5 * sub_fft_size];
                double im[5 * sub_fft_size];
            } flat;
        } *scratch_union = (void *)(fft_obj->scratch + scratch_offset);
        for (int lane = 0; lane < 5; lane++)
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

        // Step 8: Scalar prologue for k=0 and k=1
        {
            // k=0 (no twiddle multiplications)
            {
                fft_data *X0 = &output_buffer[0];                // First sub-FFT point
                fft_data *X1 = &output_buffer[sub_fft_size];     // Second sub-FFT point
                fft_data *X2 = &output_buffer[2 * sub_fft_size]; // Third sub-FFT point
                fft_data *X3 = &output_buffer[3 * sub_fft_size]; // Fourth sub-FFT point
                fft_data *X4 = &output_buffer[4 * sub_fft_size]; // Fifth sub-FFT point

                fft_type a_re = out_re[0], a_im = out_im[0];                               // First point
                fft_type b_re = out_re[sub_fft_size], b_im = out_im[sub_fft_size];         // Second point
                fft_type c_re = out_re[2 * sub_fft_size], c_im = out_im[2 * sub_fft_size]; // Third point
                fft_type d_re = out_re[3 * sub_fft_size], d_im = out_im[3 * sub_fft_size]; // Fourth point
                fft_type e_re = out_re[4 * sub_fft_size], e_im = out_im[4 * sub_fft_size]; // Fifth point

                // X0 = a + b + c + d + e
                X0->re = a_re + b_re + c_re + d_re + e_re; // Sum all points (real)
                X0->im = a_im + b_im + c_im + d_im + e_im; // Sum all points (imag)

                // X1 = a + c1*(b+e) + c2*(c+d) + i*transform_sign*( s1*(b-e) + s2*(c-d) )
                X1->re = a_re + C5_1 * (b_re + e_re) + C5_2 * (c_re + d_re) +
                         transform_sign * (S5_1 * (b_im - e_im) + S5_2 * (c_im - d_im)); // X1 real
                X1->im = a_im + C5_1 * (b_im + e_im) + C5_2 * (c_im + d_im) -
                         transform_sign * (S5_1 * (b_re - e_re) + S5_2 * (c_re - d_re)); // X1 imag

                // X4 = a + c1*(b+e) + c2*(c+d) - i*transform_sign*( s1*(b-e) + s2*(c-d) )
                X4->re = a_re + C5_1 * (b_re + e_re) + C5_2 * (c_re + d_re) -
                         transform_sign * (S5_1 * (b_im - e_im) + S5_2 * (c_im - d_im)); // X4 real
                X4->im = a_im + C5_1 * (b_im + e_im) + C5_2 * (c_im + d_im) +
                         transform_sign * (S5_1 * (b_re - e_re) + S5_2 * (c_re - d_re)); // X4 imag

                // X2 = a + c2*(b+e) + c1*(c+d) + i*transform_sign*( s2*(b-e) - s1*(c-d) )
                X2->re = a_re + C5_2 * (b_re + e_re) + C5_1 * (c_re + d_re) +
                         transform_sign * (S5_2 * (b_im - e_im) - S5_1 * (c_im - d_im)); // X2 real
                X2->im = a_im + C5_2 * (b_im + e_im) + C5_1 * (c_im + d_im) -
                         transform_sign * (S5_2 * (b_re - e_re) - S5_1 * (c_re - d_re)); // X2 imag

                // X3 = a + c2*(b+e) + c1*(c+d) - i*transform_sign*( s2*(b-e) - s1*(c-d) )
                X3->re = a_re + C5_2 * (b_re + e_re) + C5_1 * (c_re + d_re) -
                         transform_sign * (S5_2 * (b_im - e_im) - S5_1 * (c_im - d_im)); // X3 real
                X3->im = a_im + C5_2 * (b_im + e_im) + C5_1 * (c_im + d_im) +
                         transform_sign * (S5_2 * (b_re - e_re) - S5_1 * (c_re - d_re)); // X3 imag
            }

            // k=1 (with twiddle multiplications)
            if (sub_fft_size > 1)
            {
                fft_data *X0 = &output_buffer[1];                    // X(1)
                fft_data *X1 = &output_buffer[sub_fft_size + 1];     // X(1 + N/5)
                fft_data *X2 = &output_buffer[2 * sub_fft_size + 1]; // X(1 + 2N/5)
                fft_data *X3 = &output_buffer[3 * sub_fft_size + 1]; // X(1 + 3N/5)
                fft_data *X4 = &output_buffer[4 * sub_fft_size + 1]; // X(1 + 4N/5)

                fft_type a_re = out_re[1], a_im = out_im[1];                                       // X_0[1]
                fft_type b_re = out_re[sub_fft_size + 1], b_im = out_im[sub_fft_size + 1];         // X_1[1]
                fft_type c_re = out_re[2 * sub_fft_size + 1], c_im = out_im[2 * sub_fft_size + 1]; // X_2[1]
                fft_type d_re = out_re[3 * sub_fft_size + 1], d_im = out_im[3 * sub_fft_size + 1]; // X_3[1]
                fft_type e_re = out_re[4 * sub_fft_size + 1], e_im = out_im[4 * sub_fft_size + 1]; // X_4[1]

                // Load twiddle factors for k=1
                int idx = 4 * 1;                                                               // Index for k=1
                fft_type w1r = twiddle_factors[idx + 0].re, w1i = twiddle_factors[idx + 0].im; // W_N^1
                fft_type w2r = twiddle_factors[idx + 1].re, w2i = twiddle_factors[idx + 1].im; // W_N^2
                fft_type w3r = twiddle_factors[idx + 2].re, w3i = twiddle_factors[idx + 2].im; // W_N^3
                fft_type w4r = twiddle_factors[idx + 3].re, w4i = twiddle_factors[idx + 3].im; // W_N^4

                // Apply twiddle factors
                fft_type b2_re = b_re * w1r - b_im * w1i; // W_N^1 * X_1 real
                fft_type b2_im = b_im * w1r + b_re * w1i; // W_N^1 * X_1 imag
                fft_type c2_re = c_re * w2r - c_im * w2i; // W_N^2 * X_2 real
                fft_type c2_im = c_im * w2r + c_re * w2i; // W_N^2 * X_2 imag
                fft_type d2_re = d_re * w3r - d_im * w3i; // W_N^3 * X_3 real
                fft_type d2_im = d_im * w3r + d_re * w3i; // W_N^3 * X_3 imag
                fft_type e2_re = e_re * w4r - e_im * w4i; // W_N^4 * X_4 real
                fft_type e2_im = e_im * w4r + e_re * w4i; // W_N^4 * X_4 imag

                // Compute pairwise sums and differences
                fft_type t0_re = b2_re + e2_re; // b + e real
                fft_type t0_im = b2_im + e2_im; // b + e imag
                fft_type t1_re = c2_re + d2_re; // c + d real
                fft_type t1_im = c2_im + d2_im; // c + d imag
                fft_type t2_re = b2_re - e2_re; // b - e real
                fft_type t2_im = b2_im - e2_im; // b - e imag
                fft_type t3_re = c2_re - d2_re; // c - d real
                fft_type t3_im = c2_im - d2_im; // c - d imag

                // X_0 = a + (b+e) + (c+d)
                X0->re = a_re + t0_re + t1_re; // X(1) real
                X0->im = a_im + t0_im + t1_im; // X(1) imag

                // X_1 = a + c1*(b+e) + c2*(c+d) + i*transform_sign*( s1*(b-e) + s2*(c-d) )
                fft_type tmp1_re = C5_1 * t0_re + C5_2 * t1_re;                    // c1*(b+e) + c2*(c+d) real
                fft_type tmp1_im = C5_1 * t0_im + C5_2 * t1_im;                    // c1*(b+e) + c2*(c+d) imag
                fft_type rot1_re = transform_sign * (S5_1 * t2_im + S5_2 * t3_im); // s1*(b-e) + s2*(c-d) real
                fft_type rot1_im = transform_sign * (S5_2 * t3_re - S5_1 * t2_re); // s2*(c-d) - s1*(b-e) imag
                X1->re = a_re + tmp1_re + rot1_re;                                 // X(1 + N/5) real
                X1->im = a_im + tmp1_im - rot1_im;                                 // X(1 + N/5) imag

                // X_4 = a + c1*(b+e) + c2*(c+d) - i*transform_sign*( s1*(b-e) + s2*(c-d) )
                X4->re = a_re + tmp1_re - rot1_re; // X(1 + 4N/5) real
                X4->im = a_im + tmp1_im + rot1_im; // X(1 + 4N/5) imag

                // X_2 = a + c2*(b+e) + c1*(c+d) + i*transform_sign*( s2*(b-e) - s1*(c-d) )
                fft_type tmp2_re = C5_2 * t0_re + C5_1 * t1_re;                    // c2*(b+e) + c1*(c+d) real
                fft_type tmp2_im = C5_2 * t0_im + C5_1 * t1_im;                    // c2*(b+e) + c1*(c+d) imag
                fft_type rot2_re = transform_sign * (S5_2 * t2_im - S5_1 * t3_im); // s2*(b-e) - s1*(c-d) real
                fft_type rot2_im = transform_sign * (S5_1 * t2_re + S5_2 * t3_re); // s1*(b-e) + s2*(c-d) imag
                X2->re = a_re + tmp2_re + rot2_re;                                 // X(1 + 2N/5) real
                X2->im = a_im + tmp2_im - rot2_im;                                 // X(1 + 2N/5) imag

                // X_3 = a + c2*(b+e) + c1*(c+d) - i*transform_sign*( s2*(b-e) - s1*(c-d) )
                X3->re = a_re + tmp2_re - rot2_re; // X(1 + 3N/5) real
                X3->im = a_im + tmp2_im + rot2_im; // X(1 + 3N/5) imag
            }
        }

        // Step 9: SSE2 vectorized butterfly for k=2 to sub_fft_size-1
        __m128d vsign = _mm_set1_pd((double)transform_sign); // Vectorized transform sign
        __m128d vc5_1 = _mm_set1_pd(C5_1);                   // cos(72°)
        __m128d vc5_2 = _mm_set1_pd(C5_2);                   // cos(144°)
        __m128d vs5_1 = _mm_set1_pd(S5_1);                   // sin(72°)
        __m128d vs5_2 = _mm_set1_pd(S5_2);                   // sin(144°)
        int k = 2;
        for (; k + 1 < sub_fft_size; k += 2)
        {
            // Load five sub-FFT points for two k values (32-byte aligned with USE_ALIGNED_SIMD)
            __m128d a_r = LOAD_SSE2(out_re + k);                    // X_0[k:k+1] real
            __m128d a_i = LOAD_SSE2(out_im + k);                    // X_0[k:k+1] imag
            __m128d b_r = LOAD_SSE2(out_re + k + sub_fft_size);     // X_1[k:k+1] real
            __m128d b_i = LOAD_SSE2(out_im + k + sub_fft_size);     // X_1[k:k+1] imag
            __m128d c_r = LOAD_SSE2(out_re + k + 2 * sub_fft_size); // X_2[k:k+1] real
            __m128d c_i = LOAD_SSE2(out_im + k + 2 * sub_fft_size); // X_2[k:k+1] imag
            __m128d d_r = LOAD_SSE2(out_re + k + 3 * sub_fft_size); // X_3[k:k+1] real
            __m128d d_i = LOAD_SSE2(out_im + k + 3 * sub_fft_size); // X_3[k:k+1] imag
            __m128d e_r = LOAD_SSE2(out_re + k + 4 * sub_fft_size); // X_4[k:k+1] real
            __m128d e_i = LOAD_SSE2(out_im + k + 4 * sub_fft_size); // X_4[k:k+1] imag

            // Load twiddle factors for two k values (aligned for power-of-5, scratch for mixed-radix)
            int idx = 4 * k;                                       // Base index for W_N^k, ..., W_N^{4k}
            __m128d w1r = LOAD_SSE2(&twiddle_factors[idx + 0].re); // W_N^k real
            __m128d w1i = LOAD_SSE2(&twiddle_factors[idx + 0].im); // W_N^k imag
            __m128d w2r = LOAD_SSE2(&twiddle_factors[idx + 1].re); // W_N^{2k} real
            __m128d w2i = LOAD_SSE2(&twiddle_factors[idx + 1].im); // W_N^{2k} imag
            __m128d w3r = LOAD_SSE2(&twiddle_factors[idx + 2].re); // W_N^{3k} real
            __m128d w3i = LOAD_SSE2(&twiddle_factors[idx + 2].im); // W_N^{3k} imag
            __m128d w4r = LOAD_SSE2(&twiddle_factors[idx + 3].re); // W_N^{4k} real
            __m128d w4i = LOAD_SSE2(&twiddle_factors[idx + 3].im); // W_N^{4k} imag

            // Apply twiddle factors
            // FMSUB_SSE2(a, b, c) computes a*b - c, FMADD_SSE2(a, b, c) computes a*b + c
            __m128d b2r = FMSUB_SSE2(b_r, w1r, _mm_mul_pd(b_i, w1i)); // W_N^k * X_1 real
            __m128d b2i = FMADD_SSE2(b_i, w1r, _mm_mul_pd(b_r, w1i)); // W_N^k * X_1 imag
            __m128d c2r = FMSUB_SSE2(c_r, w2r, _mm_mul_pd(c_i, w2i)); // W_N^{2k} * X_2 real
            __m128d c2i = FMADD_SSE2(c_i, w2r, _mm_mul_pd(c_r, w2i)); // W_N^{2k} * X_2 imag
            __m128d d2r = FMSUB_SSE2(d_r, w3r, _mm_mul_pd(d_i, w3i)); // W_N^{3k} * X_3 real
            __m128d d2i = FMADD_SSE2(d_i, w3r, _mm_mul_pd(d_r, w3i)); // W_N^{3k} * X_3 imag
            __m128d e2r = FMSUB_SSE2(e_r, w4r, _mm_mul_pd(e_i, w4i)); // W_N^{4k} * X_4 real
            __m128d e2i = FMADD_SSE2(e_i, w4r, _mm_mul_pd(e_r, w4i)); // W_N^{4k} * X_4 imag

            // Compute pairwise sums and differences
            __m128d t0r = _mm_add_pd(b2r, e2r); // b + e real
            __m128d t0i = _mm_add_pd(b2i, e2i); // b + e imag
            __m128d t1r = _mm_add_pd(c2r, d2r); // c + d real
            __m128d t1i = _mm_add_pd(c2i, d2i); // c + d imag
            __m128d t2r = _mm_sub_pd(b2r, e2r); // b - e real
            __m128d t2i = _mm_sub_pd(b2i, e2i); // b - e imag
            __m128d t3r = _mm_sub_pd(c2r, d2r); // c - d real
            __m128d t3i = _mm_sub_pd(c2i, d2i); // c - d imag

            // X_0 = a + (b+e) + (c+d)
            __m128d x0r = _mm_add_pd(a_r, _mm_add_pd(t0r, t1r)); // X(k) real
            __m128d x0i = _mm_add_pd(a_i, _mm_add_pd(t0i, t1i)); // X(k) imag
            STORE_SSE2(out_re + k, x0r);
            STORE_SSE2(out_im + k, x0i);

            // Compute X_1 and X_4
            __m128d tmp1r = _mm_add_pd(_mm_mul_pd(vc5_1, t0r), _mm_mul_pd(vc5_2, t1r)); // c1*(b+e) + c2*(c+d) real
            __m128d tmp1i = _mm_add_pd(_mm_mul_pd(vc5_1, t0i), _mm_mul_pd(vc5_2, t1i)); // c1*(b+e) + c2*(c+d) imag
            __m128d rot1r = _mm_add_pd(_mm_mul_pd(vs5_1, t2i), _mm_mul_pd(vs5_2, t3i)); // s1*(b-e) + s2*(c-d) real
            __m128d rot1i = _mm_sub_pd(_mm_mul_pd(vs5_2, t3r), _mm_mul_pd(vs5_1, t2r)); // s2*(c-d) - s1*(b-e) imag
            rot1r = _mm_mul_pd(rot1r, vsign);                                           // Apply transform_sign
            rot1i = _mm_mul_pd(rot1i, vsign);
            __m128d x1r = _mm_add_pd(_mm_add_pd(a_r, tmp1r), rot1r); // X(k + N/5) real
            __m128d x1i = _mm_sub_pd(_mm_add_pd(a_i, tmp1i), rot1i); // X(k + N/5) imag
            __m128d x4r = _mm_sub_pd(_mm_add_pd(a_r, tmp1r), rot1r); // X(k + 4N/5) real
            __m128d x4i = _mm_add_pd(_mm_add_pd(a_i, tmp1i), rot1i); // X(k + 4N/5) imag
            STORE_SSE2(out_re + k + sub_fft_size, x1r);
            STORE_SSE2(out_im + k + sub_fft_size, x1i);
            STORE_SSE2(out_re + k + 4 * sub_fft_size, x4r);
            STORE_SSE2(out_im + k + 4 * sub_fft_size, x4i);

            // Compute X_2 and X_3
            __m128d tmp2r = _mm_add_pd(_mm_mul_pd(vc5_2, t0r), _mm_mul_pd(vc5_1, t1r)); // c2*(b+e) + c1*(c+d) real
            __m128d tmp2i = _mm_add_pd(_mm_mul_pd(vc5_2, t0i), _mm_mul_pd(vc5_1, t1i)); // c2*(b+e) + c1*(c+d) imag
            __m128d rot2r = _mm_sub_pd(_mm_mul_pd(vs5_2, t2i), _mm_mul_pd(vs5_1, t3i)); // s2*(b-e) - s1*(c-d) real
            __m128d rot2i = _mm_add_pd(_mm_mul_pd(vs5_1, t2r), _mm_mul_pd(vs5_2, t3r)); // s1*(b-e) + s2*(c-d) imag
            rot2r = _mm_mul_pd(rot2r, vsign);                                           // Apply transform_sign
            rot2i = _mm_mul_pd(rot2i, vsign);
            __m128d x2r = _mm_add_pd(_mm_add_pd(a_r, tmp2r), rot2r); // X(k + 2N/5) real
            __m128d x2i = _mm_sub_pd(_mm_add_pd(a_i, tmp2i), rot2i); // X(k + 2N/5) imag
            __m128d x3r = _mm_sub_pd(_mm_add_pd(a_r, tmp2r), rot2r); // X(k + 3N/5) real
            __m128d x3i = _mm_add_pd(_mm_add_pd(a_i, tmp2i), rot2i); // X(k + 3N/5) imag
            STORE_SSE2(out_re + k + 2 * sub_fft_size, x2r);
            STORE_SSE2(out_im + k + 2 * sub_fft_size, x2i);
            STORE_SSE2(out_re + k + 3 * sub_fft_size, x3r);
            STORE_SSE2(out_im + k + 3 * sub_fft_size, x3i);
        }

        // Step 10: Scalar tail for remaining k
        for (; k < sub_fft_size; k++)
        {
            // Load twiddle factors
            int idx = 4 * k;                            // Index into twiddle array
            fft_type w1r = twiddle_factors[idx + 0].re; // W_N^k real
            fft_type w1i = twiddle_factors[idx + 0].im; // W_N^k imag
            fft_type w2r = twiddle_factors[idx + 1].re; // W_N^{2k} real
            fft_type w2i = twiddle_factors[idx + 1].im; // W_N^{2k} imag
            fft_type w3r = twiddle_factors[idx + 2].re; // W_N^{3k} real
            fft_type w3i = twiddle_factors[idx + 2].im; // W_N^{3k} imag
            fft_type w4r = twiddle_factors[idx + 3].re; // W_N^{4k} real
            fft_type w4i = twiddle_factors[idx + 3].im; // W_N^{4k} imag

            // Load sub-FFT points
            fft_type a_re = out_re[k], a_im = out_im[k];                                       // X_0[k]
            fft_type b_re = out_re[k + sub_fft_size], b_im = out_im[k + sub_fft_size];         // X_1[k]
            fft_type c_re = out_re[k + 2 * sub_fft_size], c_im = out_im[k + 2 * sub_fft_size]; // X_2[k]
            fft_type d_re = out_re[k + 3 * sub_fft_size], d_im = out_im[k + 3 * sub_fft_size]; // X_3[k]
            fft_type e_re = out_re[k + 4 * sub_fft_size], e_im = out_im[k + 4 * sub_fft_size]; // X_4[k]

            // Apply twiddle factors
            fft_type b2_re = b_re * w1r - b_im * w1i; // W_N^k * X_1 real
            fft_type b2_im = b_im * w1r + b_re * w1i; // W_N^k * X_1 imag
            fft_type c2_re = c_re * w2r - c_im * w2i; // W_N^{2k} * X_2 real
            fft_type c2_im = c_im * w2r + c_re * w2i; // W_N^{2k} * X_2 imag
            fft_type d2_re = d_re * w3r - d_im * w3i; // W_N^{3k} * X_3 real
            fft_type d2_im = d_im * w3r + d_re * w3i; // W_N^{3k} * X_3 imag
            fft_type e2_re = e_re * w4r - e_im * w4i; // W_N^{4k} * X_4 real
            fft_type e2_im = e_im * w4r + e_re * w4i; // W_N^{4k} * X_4 imag

            // Compute pairwise sums and differences
            fft_type t0_re = b2_re + e2_re; // b + e real
            fft_type t0_im = b2_im + e2_im; // b + e imag
            fft_type t1_re = c2_re + d2_re; // c + d real
            fft_type t1_im = c2_im + d2_im; // c + d imag
            fft_type t2_re = b2_re - e2_re; // b - e real
            fft_type t2_im = b2_im - e2_im; // b - e imag
            fft_type t3_re = c2_re - d2_re; // c - d real
            fft_type t3_im = c2_im - d2_im; // c - d imag

            // X_0 = a + (b+e) + (c+d)
            out_re[k] = a_re + t0_re + t1_re; // X(k) real
            out_im[k] = a_im + t0_im + t1_im; // X(k) imag

            // X_1 = a + c1*(b+e) + c2*(c+d) + i*transform_sign*( s1*(b-e) + s2*(c-d) )
            fft_type tmp1_re = C5_1 * t0_re + C5_2 * t1_re;                    // c1*(b+e) + c2*(c+d) real
            fft_type tmp1_im = C5_1 * t0_im + C5_2 * t1_im;                    // c1*(b+e) + c2*(c+d) imag
            fft_type rot1_re = transform_sign * (S5_1 * t2_im + S5_2 * t3_im); // s1*(b-e) + s2*(c-d) real
            fft_type rot1_im = transform_sign * (S5_2 * t3_re - S5_1 * t2_re); // s2*(c-d) - s1*(b-e) imag
            out_re[k + sub_fft_size] = a_re + tmp1_re + rot1_re;               // X(k + N/5) real
            out_im[k + sub_fft_size] = a_im + tmp1_im - rot1_im;               // X(k + N/5) imag

            // X_4 = a + c1*(b+e) + c2*(c+d) - i*transform_sign*( s1*(b-e) + s2*(c-d) )
            out_re[k + 4 * sub_fft_size] = a_re + tmp1_re - rot1_re; // X(k + 4N/5) real
            out_im[k + 4 * sub_fft_size] = a_im + tmp1_im + rot1_im; // X(k + 4N/5) imag

            // X_2 = a + c2*(b+e) + c1*(c+d) + i*transform_sign*( s2*(b-e) - s1*(c-d) )
            fft_type tmp2_re = C5_2 * t0_re + C5_1 * t1_re;                    // c2*(b+e) + c1*(c+d) real
            fft_type tmp2_im = C5_2 * t0_im + C5_1 * t1_im;                    // c2*(b+e) + c1*(c+d) imag
            fft_type rot2_re = transform_sign * (S5_2 * t2_im - S5_1 * t3_im); // s2*(b-e) - s1*(c-d) real
            fft_type rot2_im = transform_sign * (S5_1 * t2_re + S5_2 * t3_re); // s1*(b-e) + s2*(c-d) imag
            out_re[k + 2 * sub_fft_size] = a_re + tmp2_re + rot2_re;           // X(k + 2N/5) real
            out_im[k + 2 * sub_fft_size] = a_im + tmp2_im - rot2_im;           // X(k + 2N/5) imag

            // X_3 = a + c2*(b+e) + c1*(c+d) - i*transform_sign*( s2*(b-e) - s1*(c-d) )
            out_re[k + 3 * sub_fft_size] = a_re + tmp2_re - rot2_re; // X(k + 3N/5) real
            out_im[k + 3 * sub_fft_size] = a_im + tmp2_im + rot2_im; // X(k + 3N/5) imag
        }

        // Step 11: Copy flattened results back to output_buffer
        for (int lane = 0; lane < 5; lane++)
        {
            fft_data *base = output_buffer + lane * sub_fft_size;
            for (int k = 0; k < sub_fft_size; k++)
            {
                base[k].re = out_re[lane * sub_fft_size + k];
                base[k].im = out_im[lane * sub_fft_size + k];
            }
        }
    }
    else if (radix == 7)
    {
        /**
         * @brief Radix-7 decomposition for seven-point sub-FFTs with SSE2 vectorization and FMA support.
         *
         * Intention: Compute the FFT for data lengths divisible by 7 by splitting into seven sub-FFTs,
         * corresponding to indices n mod 7 = 0 to 6, and combining results with twiddle factors.
         * Optimized for power-of-7 (N=7^r) and mixed-radix FFTs using pre-allocated scratch and
         * stage-specific twiddle offsets. Follows FFTW’s two-buffer strategy for thread safety and
         * performance.
         *
         * Mathematically: Computes:
         *   X(k) = X_0(k) + W_N^k * X_1(k) + ... + W_N^{6k} * X_6(k),
         * where X_0, ..., X_6 are sub-FFTs of size N/7, W_N^k = e^{-2πi k / N}.
         * The butterfly uses rotations at multiples of 360°/7 ≈ 51.43° with constants
         * C1, C2, C3, S1, S2, S3.
         *
         */
        // Step 1: Compute subproblem size and stride
        int sub_fft_length = data_length / 7; // Size of each sub-FFT (N/7)
        int next_stride = 7 * stride;         // Stride increases sevenfold
        int sub_fft_size = sub_fft_length;    // Sub-FFT size for indexing

        // Step 2: Validate scratch buffer
        // Power-of-7: 7*(N/7) for outputs; mixed-radix: (7+6)*(N/7) = 13*(N/7) for outputs + twiddles
        int required_size = fft_obj->twiddle_factors != NULL ? 7 * sub_fft_size : 13 * sub_fft_size;
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Scratch buffer too small for radix-7 at offset %d (need %d, have %d)\n",
                    scratch_offset, required_size, fft_obj->max_scratch_size - scratch_offset);
            // exit
        }

        // Step 3: Assign scratch slices
        // sub_fft_outputs: X_0, ..., X_6 (7*sub_fft_size fft_data)
        // twiddle_factors: W_N^k, ..., W_N^{6k} (6*sub_fft_size fft_data, mixed-radix only)
        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors;
        if (fft_obj->twiddle_factors != NULL)
        {
            // Validate factor_index for power-of-7 FFTs
            if (factor_index >= fft_obj->num_precomputed_stages)
            {
                fprintf(stderr, "Error: Invalid factor_index (%d) exceeds num_precomputed_stages (%d) for radix-7\n",
                        factor_index, fft_obj->num_precomputed_stages);
                // exit
            }
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + scratch_offset + 7 * sub_fft_size;
        }

        // Step 4: Compute child scratch offsets
        // Each child needs ceil(sub_fft_size/7)*7 (power-of-7) or ceil(sub_fft_size/7)*13 (mixed-radix)
        int child_sub_fft_size = (sub_fft_size + 6) / 7; // Ceiling division for next recursion level
        int child_scratch_per_branch = fft_obj->twiddle_factors != NULL ? 7 * child_sub_fft_size : 13 * child_sub_fft_size;
        int total_child_scratch = 7 * child_scratch_per_branch; // Total for all 7 children
        if (scratch_offset + total_child_scratch + (fft_obj->twiddle_factors ? 0 : 6 * sub_fft_size) > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Total child scratch size (%d) exceeds available scratch at offset %d (have %d) for radix-7\n",
                    total_child_scratch + (fft_obj->twiddle_factors ? 0 : 6 * sub_fft_size),
                    scratch_offset, fft_obj->max_scratch_size - scratch_offset);
            // exit
        }
        int child_offsets[7];
        child_offsets[0] = scratch_offset;
        for (int i = 1; i < 7; i++)
        {
            child_offsets[i] = child_offsets[i - 1] + child_scratch_per_branch; // Cumulative offsets
        }

        // Step 5: Recurse on seven sub-FFTs
        for (int i = 0; i < 7; i++)
        {
            mixed_radix_dit_rec(sub_fft_outputs + i * sub_fft_size, input_buffer + i * stride, fft_obj,
                                transform_sign, sub_fft_length, next_stride, factor_index + 1,
                                child_offsets[i]);
        }

        // Step 6: Prepare twiddle factors (mixed-radix only)
        if (fft_obj->twiddle_factors == NULL)
        {
            if (fft_obj->n_fft < sub_fft_length - 1 + 6 * sub_fft_size)
            {
                fprintf(stderr, "Error: Twiddle array too small (need at least %d elements, have %d)\n",
                        sub_fft_length - 1 + 6 * sub_fft_size, fft_obj->n_fft);
                // exit
            }
#ifdef USE_TWIDDLE_TABLES
            if (sub_fft_size <= 7 && twiddle_tables[7] != NULL)
            {
                const complex_t *table = twiddle_tables[7];
                for (int k = 0; k < sub_fft_size; k++)
                {
                    for (int n = 0; n < 6; n++)
                    {
                        twiddle_factors[6 * k + n].re = table[n + 1].re; // W_N^{n*k} real
                        twiddle_factors[6 * k + n].im = table[n + 1].im; // W_N^{n*k} imag
                    }
                }
            }
            else
#endif
            {
                for (int k = 0; k < sub_fft_size; k++)
                {
                    for (int n = 0; n < 6; n++)
                    {
                        // Correct indexing: W_N^{n*k} for stage stride
                        int idx = n * sub_fft_size + k;
                        twiddle_factors[6 * k + n].re = fft_obj->twiddles[idx].re; // W_N^{n*k} real
                        twiddle_factors[6 * k + n].im = fft_obj->twiddles[idx].im; // W_N^{n*k} imag
                    }
                }
            }
        }

        // Step 7: Flatten outputs into separate real/imag arrays using union
        union
        {
            fft_data data[7 * sub_fft_size];
            struct
            {
                double re[7 * sub_fft_size];
                double im[7 * sub_fft_size];
            } flat;
        } *scratch_union = (void *)(fft_obj->scratch + scratch_offset);
        for (int lane = 0; lane < 7; lane++)
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

        // Step 8: Scalar prologue for k=0 (no twiddle multiplications)
        {
            fft_data *X0 = &output_buffer[0];                // X(0)
            fft_data *X1 = &output_buffer[sub_fft_size];     // X(1)
            fft_data *X2 = &output_buffer[2 * sub_fft_size]; // X(2)
            fft_data *X3 = &output_buffer[3 * sub_fft_size]; // X(3)
            fft_data *X4 = &output_buffer[4 * sub_fft_size]; // X(4)
            fft_data *X5 = &output_buffer[5 * sub_fft_size]; // X(5)
            fft_data *X6 = &output_buffer[6 * sub_fft_size]; // X(6)

            fft_type a_r = out_re[0], a_i = out_im[0];                               // First point
            fft_type b_r = out_re[sub_fft_size], b_i = out_im[sub_fft_size];         // Second point
            fft_type c_r = out_re[2 * sub_fft_size], c_i = out_im[2 * sub_fft_size]; // Third point
            fft_type d_r = out_re[3 * sub_fft_size], d_i = out_im[3 * sub_fft_size]; // Fourth point
            fft_type e_r = out_re[4 * sub_fft_size], e_i = out_im[4 * sub_fft_size]; // Fifth point
            fft_type f_r = out_re[5 * sub_fft_size], f_i = out_im[5 * sub_fft_size]; // Sixth point
            fft_type g_r = out_re[6 * sub_fft_size], g_i = out_im[6 * sub_fft_size]; // Seventh point

            fft_type t0r = b_r + g_r, t0i = b_i + g_i; // B + G
            fft_type t3r = b_r - g_r, t3i = b_i - g_i; // B - G
            fft_type t1r = c_r + f_r, t1i = c_i + f_i; // C + F
            fft_type t4r = c_r - f_r, t4i = c_i - f_i; // C - F
            fft_type t2r = d_r + e_r, t2i = d_i + e_i; // D + E
            fft_type t5r = d_r - e_r, t5i = d_i - e_i; // D - E

            X0->re = a_r + t0r + t1r + t2r; // X(0) real
            X0->im = a_i + t0i + t1i + t2i; // X(0) imag

            fft_type tmp_r = a_r + C1 * t0r + C2 * t1r + C3 * t2r; // X(1) and X(6)
            fft_type tmp_i = a_i + C1 * t0i + C2 * t1i + C3 * t2i;
            fft_type rot_r = -(S1 * t3r + S2 * t4r + S3 * t5r) * transform_sign;
            fft_type rot_i = -(S1 * t3i + S2 * t4i + S3 * t5i) * transform_sign;
            X1->re = tmp_r - rot_i; // X(1) real
            X1->im = tmp_i + rot_r; // X(1) imag
            X6->re = tmp_r + rot_i; // X(6) real
            X6->im = tmp_i - rot_r; // X(6) imag

            tmp_r = a_r + C2 * t0r + C3 * t1r + C1 * t2r; // X(2) and X(5)
            tmp_i = a_i + C2 * t0i + C3 * t1i + C1 * t2i;
            rot_r = -(S2 * t3r + S3 * t4r + S1 * t5r) * transform_sign;
            rot_i = -(S2 * t3i + S3 * t4i + S1 * t5i) * transform_sign;
            X2->re = tmp_r - rot_i; // X(2) real
            X2->im = tmp_i + rot_r; // X(2) imag
            X5->re = tmp_r + rot_i; // X(5) real
            X5->im = tmp_i - rot_r; // X(5) imag

            tmp_r = a_r + C3 * t0r + C1 * t1r + C2 * t2r; // X(3) and X(4)
            tmp_i = a_i + C3 * t0i + C1 * t1i + C2 * t2i;
            rot_r = -(S3 * t3r + S1 * t4r + S2 * t5r) * transform_sign;
            rot_i = -(S3 * t3i + S1 * t4i + S2 * t5i) * transform_sign;
            X3->re = tmp_r - rot_i; // X(3) real
            X3->im = tmp_i + rot_r; // X(3) imag
            X4->re = tmp_r + rot_i; // X(4) real
            X4->im = tmp_i - rot_r; // X(4) imag
        }

        // Step 9: SSE2 vectorized butterfly for k=2 to sub_fft_size-1
        __m128d vsign = _mm_set1_pd((double)transform_sign); // Vectorized transform sign
        __m128d vc1 = _mm_set1_pd(C1);                       // cos(51.43°)
        __m128d vc2 = _mm_set1_pd(C2);                       // cos(102.86°)
        __m128d vc3 = _mm_set1_pd(C3);                       // cos(154.29°)
        __m128d vs1 = _mm_set1_pd(S1);                       // sin(51.43°)
        __m128d vs2 = _mm_set1_pd(S2);                       // sin(102.86°)
        __m128d vs3 = _mm_set1_pd(S3);                       // sin(154.29°)
        int k = 2;
        for (; k + 1 < sub_fft_size; k += 2)
        {
            // Load seven sub-FFT points for two k values (32-byte aligned with USE_ALIGNED_SIMD)
            __m128d a_r = LOAD_SSE2(out_re + k);                    // X_0[k:k+1] real
            __m128d a_i = LOAD_SSE2(out_im + k);                    // X_0[k:k+1] imag
            __m128d b_r = LOAD_SSE2(out_re + k + sub_fft_size);     // X_1[k:k+1] real
            __m128d b_i = LOAD_SSE2(out_im + k + sub_fft_size);     // X_1[k:k+1] imag
            __m128d c_r = LOAD_SSE2(out_re + k + 2 * sub_fft_size); // X_2[k:k+1] real
            __m128d c_i = LOAD_SSE2(out_im + k + 2 * sub_fft_size); // X_2[k:k+1] imag
            __m128d d_r = LOAD_SSE2(out_re + k + 3 * sub_fft_size); // X_3[k:k+1] real
            __m128d d_i = LOAD_SSE2(out_im + k + 3 * sub_fft_size); // X_3[k:k+1] imag
            __m128d e_r = LOAD_SSE2(out_re + k + 4 * sub_fft_size); // X_4[k:k+1] real
            __m128d e_i = LOAD_SSE2(out_im + k + 4 * sub_fft_size); // X_4[k:k+1] imag
            __m128d f_r = LOAD_SSE2(out_re + k + 5 * sub_fft_size); // X_5[k:k+1] real
            __m128d f_i = LOAD_SSE2(out_im + k + 5 * sub_fft_size); // X_5[k:k+1] imag
            __m128d g_r = LOAD_SSE2(out_re + k + 6 * sub_fft_size); // X_6[k:k+1] real
            __m128d g_i = LOAD_SSE2(out_im + k + 6 * sub_fft_size); // X_6[k:k+1] imag

            // Load six twiddle factors for two k values (aligned for power-of-7, scratch for mixed-radix)
            int idx = 6 * k;                                       // Base index for W_N^k, ..., W_N^{6k}
            __m128d w1r = LOAD_SSE2(&twiddle_factors[idx + 0].re); // W_N^k real
            __m128d w1i = LOAD_SSE2(&twiddle_factors[idx + 0].im); // W_N^k imag
            __m128d w2r = LOAD_SSE2(&twiddle_factors[idx + 1].re); // W_N^{2k} real
            __m128d w2i = LOAD_SSE2(&twiddle_factors[idx + 1].im); // W_N^{2k} imag
            __m128d w3r = LOAD_SSE2(&twiddle_factors[idx + 2].re); // W_N^{3k} real
            __m128d w3i = LOAD_SSE2(&twiddle_factors[idx + 2].im); // W_N^{3k} imag
            __m128d w4r = LOAD_SSE2(&twiddle_factors[idx + 3].re); // W_N^{4k} real
            __m128d w4i = LOAD_SSE2(&twiddle_factors[idx + 3].im); // W_N^{4k} imag
            __m128d w5r = LOAD_SSE2(&twiddle_factors[idx + 4].re); // W_N^{5k} real
            __m128d w5i = LOAD_SSE2(&twiddle_factors[idx + 4].im); // W_N^{5k} imag
            __m128d w6r = LOAD_SSE2(&twiddle_factors[idx + 5].re); // W_N^{6k} real
            __m128d w6i = LOAD_SSE2(&twiddle_factors[idx + 5].im); // W_N^{6k} imag

            // Apply twiddle factors to points B to G
            // FMSUB_SSE2(a, b, c) computes a*b - c, FMADD_SSE2(a, b, c) computes a*b + c
            __m128d b2r = FMSUB_SSE2(b_r, w1r, _mm_mul_pd(b_i, w1i)); // W_N^k * B real
            __m128d b2i = FMADD_SSE2(b_i, w1r, _mm_mul_pd(b_r, w1i)); // W_N^k * B imag
            __m128d c2r = FMSUB_SSE2(c_r, w2r, _mm_mul_pd(c_i, w2i)); // W_N^{2k} * C real
            __m128d c2i = FMADD_SSE2(c_i, w2r, _mm_mul_pd(c_r, w2i)); // W_N^{2k} * C imag
            __m128d d2r = FMSUB_SSE2(d_r, w3r, _mm_mul_pd(d_i, w3i)); // W_N^{3k} * D real
            __m128d d2i = FMADD_SSE2(d_i, w3r, _mm_mul_pd(d_r, w3i)); // W_N^{3k} * D imag
            __m128d e2r = FMSUB_SSE2(e_r, w4r, _mm_mul_pd(e_i, w4i)); // W_N^{4k} * E real
            __m128d e2i = FMADD_SSE2(e_i, w4r, _mm_mul_pd(e_r, w4i)); // W_N^{4k} * E imag
            __m128d f2r = FMSUB_SSE2(f_r, w5r, _mm_mul_pd(f_i, w5i)); // W_N^{5k} * F real
            __m128d f2i = FMADD_SSE2(f_i, w5r, _mm_mul_pd(f_r, w5i)); // W_N^{5k} * F imag
            __m128d g2r = FMSUB_SSE2(g_r, w6r, _mm_mul_pd(g_i, w6i)); // W_N^{6k} * G real
            __m128d g2i = FMADD_SSE2(g_i, w6r, _mm_mul_pd(g_r, w6i)); // W_N^{6k} * G imag

            // Compute pairwise sums and differences
            __m128d t0r = _mm_add_pd(b2r, g2r); // B + G real
            __m128d t0i = _mm_add_pd(b2i, g2i); // B + G imag
            __m128d t3r = _mm_sub_pd(b2r, g2r); // B - G real
            __m128d t3i = _mm_sub_pd(b2i, g2i); // B - G imag
            __m128d t1r = _mm_add_pd(c2r, f2r); // C + F real
            __m128d t1i = _mm_add_pd(c2i, f2i); // C + F imag
            __m128d t4r = _mm_sub_pd(c2r, f2r); // C - F real
            __m128d t4i = _mm_sub_pd(c2i, f2i); // C - F imag
            __m128d t2r = _mm_add_pd(d2r, e2r); // D + E real
            __m128d t2i = _mm_add_pd(d2i, e2i); // D + E imag
            __m128d t5r = _mm_sub_pd(d2r, e2r); // D - E real
            __m128d t5i = _mm_sub_pd(d2i, e2i); // D - E imag

            // Compute X(k) = a + t0 + t1 + t2
            __m128d sum01r = _mm_add_pd(t0r, t1r);    // (B+G) + (C+F) real
            __m128d sum01i = _mm_add_pd(t0i, t1i);    // (B+G) + (C+F) imag
            __m128d sum2ar = _mm_add_pd(sum01r, t2r); // + (D+E) real
            __m128d sum2ai = _mm_add_pd(sum01i, t2i); // + (D+E) imag
            __m128d out0r = _mm_add_pd(a_r, sum2ar);  // X(k) real
            __m128d out0i = _mm_add_pd(a_i, sum2ai);  // X(k) imag
            STORE_SSE2(out_re + k, out0r);            // Store X(k) real
            STORE_SSE2(out_im + k, out0i);            // Store X(k) imag

            // Compute rotated center for X(k+1, k+6): a + c1*t0 + c2*t1 + c3*t2
            __m128d tmp1r = FMADD_SSE2(vc3, t2r, FMADD_SSE2(vc2, t1r, FMADD_SSE2(vc1, t0r, a_r))); // Real
            __m128d tmp1i = FMADD_SSE2(vc3, t2i, FMADD_SSE2(vc2, t1i, FMADD_SSE2(vc1, t0i, a_i))); // Imag
            __m128d rot1r = FMADD_SSE2(vs3, t5r, FMADD_SSE2(vs2, t4r, _mm_mul_pd(vs1, t3r)));      // Rotation real
            __m128d rot1i = FMADD_SSE2(vs3, t5i, FMADD_SSE2(vs2, t4i, _mm_mul_pd(vs1, t3i)));      // Rotation imag
            rot1r = _mm_mul_pd(rot1r, vsign);                                                      // Apply sign flip
            rot1i = _mm_mul_pd(rot1i, vsign);
            STORE_SSE2(out_re + k + sub_fft_size, _mm_sub_pd(tmp1r, rot1i));     // X(k+1) real
            STORE_SSE2(out_im + k + sub_fft_size, _mm_add_pd(tmp1i, rot1r));     // X(k+1) imag
            STORE_SSE2(out_re + k + 6 * sub_fft_size, _mm_add_pd(tmp1r, rot1i)); // X(k+6) real
            STORE_SSE2(out_im + k + 6 * sub_fft_size, _mm_sub_pd(tmp1i, rot1r)); // X(k+6) imag

            // Compute rotated center for X(k+2, k+5): a + c2*t0 + c3*t1 + c1*t2
            __m128d tmp2r = FMADD_SSE2(vc1, t2r, FMADD_SSE2(vc3, t1r, FMADD_SSE2(vc2, t0r, a_r))); // Real
            __m128d tmp2i = FMADD_SSE2(vc1, t2i, FMADD_SSE2(vc3, t1i, FMADD_SSE2(vc2, t0i, a_i))); // Imag
            __m128d rot2r = FMADD_SSE2(vs1, t5r, FMADD_SSE2(vs3, t4r, _mm_mul_pd(vs2, t3r)));      // Rotation real
            __m128d rot2i = FMADD_SSE2(vs1, t5i, FMADD_SSE2(vs3, t4i, _mm_mul_pd(vs2, t3i)));      // Rotation imag
            rot2r = _mm_mul_pd(rot2r, vsign);                                                      // Apply sign flip
            rot2i = _mm_mul_pd(rot2i, vsign);
            STORE_SSE2(out_re + k + 2 * sub_fft_size, _mm_sub_pd(tmp2r, rot2i)); // X(k+2) real
            STORE_SSE2(out_im + k + 2 * sub_fft_size, _mm_add_pd(tmp2i, rot2r)); // X(k+2) imag
            STORE_SSE2(out_re + k + 5 * sub_fft_size, _mm_add_pd(tmp2r, rot2i)); // X(k+5) real
            STORE_SSE2(out_im + k + 5 * sub_fft_size, _mm_sub_pd(tmp2i, rot2r)); // X(k+5) imag

            // Compute rotated center for X(k+3, k+4): a + c3*t0 + c1*t1 + c2*t2
            __m128d tmp3r = FMADD_SSE2(vc2, t2r, FMADD_SSE2(vc1, t1r, FMADD_SSE2(vc3, t0r, a_r))); // Real
            __m128d tmp3i = FMADD_SSE2(vc2, t2i, FMADD_SSE2(vc1, t1i, FMADD_SSE2(vc3, t0i, a_i))); // Imag
            __m128d rot3r = FMADD_SSE2(vs2, t5r, FMADD_SSE2(vs1, t4r, _mm_mul_pd(vs3, t3r)));      // Rotation real
            __m128d rot3i = FMADD_SSE2(vs2, t5i, FMADD_SSE2(vs1, t4i, _mm_mul_pd(vs3, t3i)));      // Rotation imag
            rot3r = _mm_mul_pd(rot3r, vsign);                                                      // Apply sign flip
            rot3i = _mm_mul_pd(rot3i, vsign);
            STORE_SSE2(out_re + k + 3 * sub_fft_size, _mm_sub_pd(tmp3r, rot3i)); // X(k+3) real
            STORE_SSE2(out_im + k + 3 * sub_fft_size, _mm_add_pd(tmp3i, rot3r)); // X(k+3) imag
            STORE_SSE2(out_re + k + 4 * sub_fft_size, _mm_add_pd(tmp3r, rot3i)); // X(k+4) real
            STORE_SSE2(out_im + k + 4 * sub_fft_size, _mm_sub_pd(tmp3i, rot3r)); // X(k+4) imag
        }

        // Step 10: Scalar tail for remaining k
        for (; k < sub_fft_size; k++)
        {
            // Load twiddle factors
            int idx = 6 * k;                            // Index into twiddle array
            fft_type w1r = twiddle_factors[idx + 0].re; // W_N^k real
            fft_type w1i = twiddle_factors[idx + 0].im; // W_N^k imag
            fft_type w2r = twiddle_factors[idx + 1].re; // W_N^{2k} real
            fft_type w2i = twiddle_factors[idx + 1].im; // W_N^{2k} imag
            fft_type w3r = twiddle_factors[idx + 2].re; // W_N^{3k} real
            fft_type w3i = twiddle_factors[idx + 2].im; // W_N^{3k} imag
            fft_type w4r = twiddle_factors[idx + 3].re; // W_N^{4k} real
            fft_type w4i = twiddle_factors[idx + 3].im; // W_N^{4k} imag
            fft_type w5r = twiddle_factors[idx + 4].re; // W_N^{5k} real
            fft_type w5i = twiddle_factors[idx + 4].im; // W_N^{5k} imag
            fft_type w6r = twiddle_factors[idx + 5].re; // W_N^{6k} real
            fft_type w6i = twiddle_factors[idx + 5].im; // W_N^{6k} imag

            // Load sub-FFT points
            fft_type a_r = out_re[k], a_i = out_im[k];                                       // X_0[k]
            fft_type b_r = out_re[k + sub_fft_size], b_i = out_im[k + sub_fft_size];         // X_1[k]
            fft_type c_r = out_re[k + 2 * sub_fft_size], c_i = out_im[k + 2 * sub_fft_size]; // X_2[k]
            fft_type d_r = out_re[k + 3 * sub_fft_size], d_i = out_im[k + 3 * sub_fft_size]; // X_3[k]
            fft_type e_r = out_re[k + 4 * sub_fft_size], e_i = out_im[k + 4 * sub_fft_size]; // X_4[k]
            fft_type f_r = out_re[k + 5 * sub_fft_size], f_i = out_im[k + 5 * sub_fft_size]; // X_5[k]
            fft_type g_r = out_re[k + 6 * sub_fft_size], g_i = out_im[k + 6 * sub_fft_size]; // X_6[k]

            // Apply twiddle factors
            fft_type b2_r = b_r * w1r - b_i * w1i; // W_N^k * B real
            fft_type b2_i = b_i * w1r + b_r * w1i; // W_N^k * B imag
            fft_type c2_r = c_r * w2r - c_i * w2i; // W_N^{2k} * C real
            fft_type c2_i = c_i * w2r + c_r * w2i; // W_N^{2k} * C imag
            fft_type d2_r = d_r * w3r - d_i * w3i; // W_N^{3k} * D real
            fft_type d2_i = d_i * w3r + d_r * w3i; // W_N^{3k} * D imag
            fft_type e2_r = e_r * w4r - e_i * w4i; // W_N^{4k} * E real
            fft_type e2_i = e_i * w4r + e_r * w4i; // W_N^{4k} * E imag
            fft_type f2_r = f_r * w5r - f_i * w5i; // W_N^{5k} * F real
            fft_type f2_i = f_i * w5r + f_r * w5i; // W_N^{5k} * F imag
            fft_type g2_r = g_r * w6r - g_i * w6i; // W_N^{6k} * G real
            fft_type g2_i = g_i * w6r + g_r * w6i; // W_N^{6k} * G imag

            // Compute pairwise sums and differences
            fft_type t0_r = b2_r + g2_r; // B + G real
            fft_type t0_i = b2_i + g2_i; // B + G imag
            fft_type t3_r = b2_r - g2_r; // B - G real
            fft_type t3_i = b2_i - g2_i; // B - G imag
            fft_type t1_r = c2_r + f2_r; // C + F real
            fft_type t1_i = c2_i + f2_i; // C + F imag
            fft_type t4_r = c2_r - f2_r; // C - F real
            fft_type t4_i = c2_i - f2_i; // C - F imag
            fft_type t2_r = d2_r + e2_r; // D + E real
            fft_type t2_i = d2_i + e2_i; // D + E imag
            fft_type t5_r = d2_r - e2_r; // D - E real
            fft_type t5_i = d2_i - e2_i; // D - E imag

            // Compute X(k) = a + t0 + t1 + t2
            fft_type sum01_r = t0_r + t1_r;    // (B+G) + (C+F) real
            fft_type sum01_i = t0_i + t1_i;    // (B+G) + (C+F) imag
            fft_type sum2a_r = sum01_r + t2_r; // + (D+E) real
            fft_type sum2a_i = sum01_i + t2_i; // + (D+E) imag
            out_re[k] = a_r + sum2a_r;         // X(k) real
            out_im[k] = a_i + sum2a_i;         // X(k) imag

            // Compute X(k+1, k+6): a + c1*t0 + c2*t1 + c3*t2 ± i*transform_sign*(s1*t3 + s2*t4 + s3*t5)
            fft_type tmp1_r = a_r + C1 * t0_r + C2 * t1_r + C3 * t2_r;
            fft_type tmp1_i = a_i + C1 * t0_i + C2 * t1_i + C3 * t2_i;
            fft_type rot1_r = -(S1 * t3_r + S2 * t4_r + S3 * t5_r) * transform_sign;
            fft_type rot1_i = -(S1 * t3_i + S2 * t4_i + S3 * t5_i) * transform_sign;
            out_re[k + sub_fft_size] = tmp1_r - rot1_i;     // X(k+1) real
            out_im[k + sub_fft_size] = tmp1_i + rot1_r;     // X(k+1) imag
            out_re[k + 6 * sub_fft_size] = tmp1_r + rot1_i; // X(k+6) real
            out_im[k + 6 * sub_fft_size] = tmp1_i - rot1_r; // X(k+6) imag

            // Compute X(k+2, k+5): a + c2*t0 + c3*t1 + c1*t2 ± i*transform_sign*(s2*t3 + s3*t4 + s1*t5)
            fft_type tmp2_r = a_r + C2 * t0_r + C3 * t1_r + C1 * t2_r;
            fft_type tmp2_i = a_i + C2 * t0_i + C3 * t1_i + C1 * t2_i;
            fft_type rot2_r = -(S2 * t3_r + S3 * t4_r + S1 * t5_r) * transform_sign;
            fft_type rot2_i = -(S2 * t3_i + S3 * t4_i + S1 * t5_i) * transform_sign;
            out_re[k + 2 * sub_fft_size] = tmp2_r - rot2_i; // X(k+2) real
            out_im[k + 2 * sub_fft_size] = tmp2_i + rot2_r; // X(k+2) imag
            out_re[k + 5 * sub_fft_size] = tmp2_r + rot2_i; // X(k+5) real
            out_im[k + 5 * sub_fft_size] = tmp2_i - rot2_r; // X(k+5) imag

            // Compute X(k+3, k+4): a + c3*t0 + c1*t1 + c2*t2 ± i*transform_sign*(s3*t3 + s1*t4 + s2*t5)
            fft_type tmp3_r = a_r + C3 * t0_r + C1 * t1_r + C2 * t2_r;
            fft_type tmp3_i = a_i + C3 * t0_i + C1 * t1_i + C2 * t2_i;
            fft_type rot3_r = -(S3 * t3_r + S1 * t4_r + S2 * t5_r) * transform_sign;
            fft_type rot3_i = -(S3 * t3_i + S1 * t4_i + S2 * t5_i) * transform_sign;
            out_re[k + 3 * sub_fft_size] = tmp3_r - rot3_i; // X(k+3) real
            out_im[k + 3 * sub_fft_size] = tmp3_i + rot3_r; // X(k+3) imag
            out_re[k + 4 * sub_fft_size] = tmp3_r + rot3_i; // X(k+4) real
            out_im[k + 4 * sub_fft_size] = tmp3_i - rot3_r; // X(k+4) imag
        }

        // Step 11: Copy flattened results back to output_buffer
        for (int lane = 0; lane < 7; lane++)
        {
            fft_data *base = output_buffer + lane * sub_fft_size;
            for (int k = 0; k < sub_fft_size; k++)
            {
                base[k].re = out_re[lane * sub_fft_size + k];
                base[k].im = out_im[lane * sub_fft_size + k];
            }
        }
    }
    else if (radix == 8)
    {
        /**
         * @brief Radix-8 decomposition for eight-point sub-FFTs with SSE2 vectorization and FMA support.
         *
         * Intention: Compute the FFT for data lengths divisible by 8 by splitting into eight sub-FFTs,
         * corresponding to indices n mod 8 = 0 to 7, and combining results with twiddle factors.
         * Optimized for power-of-8 (N=8^r) and mixed-radix FFTs using pre-allocated scratch and
         * stage-specific twiddle offsets. Follows FFTW’s two-buffer strategy for thread safety and
         * performance.
         *
         * Mathematically: Computes:
         *   X(k) = X_0(k) + W_N^k * X_1(k) + ... + W_N^{7k} * X_7(k),
         * where X_0, ..., X_7 are sub-FFTs of size N/8, W_N^k = e^{-2πi k / N}.
         * The butterfly uses rotations at multiples of 45° with constant C8_1 = √2/2.
         */
        // Step 1: Compute subproblem size and stride
        int sub_fft_length = data_length / 8; // Size of each sub-FFT (N/8)
        int next_stride = 8 * stride;         // Stride increases eightfold
        int sub_fft_size = sub_fft_length;    // Sub-FFT size for indexing

        // Step 2: Validate scratch buffer
        // Power-of-8: 8*(N/8) for outputs; mixed-radix: (8+7)*(N/8) = 15*(N/8) for outputs + twiddles
        int required_size = fft_obj->twiddle_factors != NULL ? 8 * sub_fft_size : 15 * sub_fft_size;
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Scratch buffer too small for radix-8 at offset %d (need %d, have %d)\n",
                    scratch_offset, required_size, fft_obj->max_scratch_size - scratch_offset);
            // exit
        }

        // Step 3: Assign scratch slices
        // sub_fft_outputs: X_0, ..., X_7 (8*sub_fft_size fft_data)
        // twiddle_factors: W_N^k, ..., W_N^{7k} (7*sub_fft_size fft_data, mixed-radix only)
        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors;
        if (fft_obj->twiddle_factors != NULL)
        {
            // Validate factor_index for power-of-8 FFTs
            if (factor_index >= fft_obj->num_precomputed_stages)
            {
                fprintf(stderr, "Error: Invalid factor_index (%d) exceeds num_precomputed_stages (%d) for radix-8\n",
                        factor_index, fft_obj->num_precomputed_stages);
                // exit
            }
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            twiddle_factors = fft_obj->scratch + scratch_offset + 8 * sub_fft_size;
        }

        // Step 4: Compute child scratch offsets
        // Each child needs ceil(sub_fft_size/8)*8 (power-of-8) or ceil(sub_fft_size/8)*15 (mixed-radix)
        int child_sub_fft_size = (sub_fft_size + 7) / 8; // Ceiling division for next recursion level
        int child_scratch_per_branch = fft_obj->twiddle_factors != NULL ? 8 * child_sub_fft_size : 15 * child_sub_fft_size;
        int total_child_scratch = 8 * child_scratch_per_branch; // Total for all 8 children
        if (scratch_offset + total_child_scratch + (fft_obj->twiddle_factors ? 0 : 7 * sub_fft_size) > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Total child scratch size (%d) exceeds available scratch at offset %d (have %d) for radix-8\n",
                    total_child_scratch + (fft_obj->twiddle_factors ? 0 : 7 * sub_fft_size),
                    scratch_offset, fft_obj->max_scratch_size - scratch_offset);
            // exit
        }
        int child_offsets[8];
        child_offsets[0] = scratch_offset;
        for (int i = 1; i < 8; i++)
        {
            child_offsets[i] = child_offsets[i - 1] + child_scratch_per_branch; // Cumulative offsets
        }

        // Step 5: Recurse on eight sub-FFTs
        for (int i = 0; i < 8; i++)
        {
            mixed_radix_dit_rec(sub_fft_outputs + i * sub_fft_size, input_buffer + i * stride, fft_obj,
                                transform_sign, sub_fft_length, next_stride, factor_index + 1,
                                child_offsets[i]);
        }

        // Step 6: Prepare twiddle factors (mixed-radix only)
        if (fft_obj->twiddle_factors == NULL)
        {
            if (fft_obj->n_fft < sub_fft_length - 1 + 7 * sub_fft_size)
            {
                fprintf(stderr, "Error: Twiddle array too small (need at least %d elements, have %d)\n",
                        sub_fft_length - 1 + 7 * sub_fft_size, fft_obj->n_fft);
                // exit
            }
#ifdef USE_TWIDDLE_TABLES
            if (sub_fft_size <= 8 && twiddle_tables[8] != NULL)
            {
                const complex_t *table = twiddle_tables[8];
                for (int k = 0; k < sub_fft_size; k++)
                {
                    for (int n = 0; n < 7; n++)
                    {
                        twiddle_factors[7 * k + n].re = table[n + 1].re; // W_N^{n*k} real
                        twiddle_factors[7 * k + n].im = table[n + 1].im; // W_N^{n*k} imag
                    }
                }
            }
            else
#endif
            {
                for (int k = 0; k < sub_fft_size; k++)
                {
                    for (int n = 0; n < 7; n++)
                    {
                        int idx = n * sub_fft_size + k;                            // Correct indexing for W_N^{n*k}
                        twiddle_factors[7 * k + n].re = fft_obj->twiddles[idx].re; // W_N^{n*k} real
                        twiddle_factors[7 * k + n].im = fft_obj->twiddles[idx].im; // W_N^{n*k} imag
                    }
                }
            }
        }

        // Step 7: Flatten outputs into separate real/imag arrays using union
        union
        {
            fft_data data[8 * sub_fft_size];
            struct
            {
                double re[8 * sub_fft_size];
                double im[8 * sub_fft_size];
            } flat;
        } *scratch_union = (void *)(fft_obj->scratch + scratch_offset);
        for (int lane = 0; lane < 8; lane++)
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

        // Step 8: Scalar prologue for k=0 to k=3
        {
            // k=0 (no twiddle multiplications)
            {
                fft_data *X0 = &output_buffer[0];                // X(0)
                fft_data *X1 = &output_buffer[sub_fft_size];     // X(N/8)
                fft_data *X2 = &output_buffer[2 * sub_fft_size]; // X(2N/8)
                fft_data *X3 = &output_buffer[3 * sub_fft_size]; // X(3N/8)
                fft_data *X4 = &output_buffer[4 * sub_fft_size]; // X(4N/8)
                fft_data *X5 = &output_buffer[5 * sub_fft_size]; // X(5N/8)
                fft_data *X6 = &output_buffer[6 * sub_fft_size]; // X(6N/8)
                fft_data *X7 = &output_buffer[7 * sub_fft_size]; // X(7N/8)

                fft_type a_r = out_re[0], a_i = out_im[0];                               // X_0[0]
                fft_type b_r = out_re[sub_fft_size], b_i = out_im[sub_fft_size];         // X_1[0]
                fft_type c_r = out_re[2 * sub_fft_size], c_i = out_im[2 * sub_fft_size]; // X_2[0]
                fft_type d_r = out_re[3 * sub_fft_size], d_i = out_im[3 * sub_fft_size]; // X_3[0]
                fft_type e_r = out_re[4 * sub_fft_size], e_i = out_im[4 * sub_fft_size]; // X_4[0]
                fft_type f_r = out_re[5 * sub_fft_size], f_i = out_im[5 * sub_fft_size]; // X_5[0]
                fft_type g_r = out_re[6 * sub_fft_size], g_i = out_im[6 * sub_fft_size]; // X_6[0]
                fft_type h_r = out_re[7 * sub_fft_size], h_i = out_im[7 * sub_fft_size]; // X_7[0]

                fft_type t0_r = b_r + h_r, t0_i = b_i + h_i; // B + H
                fft_type t1_r = c_r + g_r, t1_i = c_i + g_i; // C + G
                fft_type t2_r = d_r + f_r, t2_i = d_i + f_i; // D + F
                fft_type t3_r = b_r - h_r, t3_i = b_i - h_i; // B - H
                fft_type t4_r = c_r - g_r, t4_i = c_i - g_i; // C - G
                fft_type t5_r = d_r - f_r, t5_i = d_i - f_i; // D - F

                fft_type sum_r = a_r + t0_r + t1_r + t2_r; // X_0 real
                fft_type sum_i = a_i + t0_i + t1_i + t2_i; // X_0 imag
                X0->re = sum_r;                            // X(0) real
                X0->im = sum_i;                            // X(0) imag

                X4->re = a_r - (t0_r + t1_r) + t2_r; // X(4N/8) real
                X4->im = a_i - (t0_i + t1_i) + t2_i; // X(4N/8) imag

                fft_type tmp1_r = a_r - t2_r, tmp1_i = a_i - t2_i;   // a - (d+f)
                fft_type tmp2_r = t3_r - t5_r, tmp2_i = t3_i - t5_i; // (b-h) - (d-f)

                fft_type u1_r = tmp1_r + t1_r, u1_i = tmp1_i + t1_i;               // Base for X_1, X_7
                fft_type rot1_r = C8_1 * (t0_i - t2_i) + transform_sign * tmp2_i;  // Rotation real
                fft_type rot1_i = -C8_1 * (t0_r - t2_r) + transform_sign * tmp2_r; // Rotation imag
                X1->re = u1_r - rot1_r;                                            // X(N/8) real
                X1->im = u1_i - rot1_i;                                            // X(N/8) imag
                X7->re = u1_r + rot1_r;                                            // X(7N/8) real
                X7->im = u1_i + rot1_i;                                            // X(7N/8) imag

                fft_type u2_r = tmp1_r + t0_r, u2_i = tmp1_i + t0_i;                      // Base for X_2, X_6
                fft_type rot2_r = C8_1 * (t1_i - t2_i) + transform_sign * (t4_i - t5_i);  // Rotation real
                fft_type rot2_i = -C8_1 * (t1_r - t2_r) + transform_sign * (t4_r - t5_r); // Rotation imag
                X2->re = u2_r - rot2_r;                                                   // X(2N/8) real
                X2->im = u2_i - rot2_i;                                                   // X(2N/8) imag
                X6->re = u2_r + rot2_r;                                                   // X(6N/8) real
                X6->im = u2_i + rot2_i;                                                   // X(6N/8) imag

                fft_type u3_r = tmp1_r + t5_r, u3_i = tmp1_i + t5_i;                      // Base for X_3, X_5
                fft_type rot3_r = C8_1 * (t3_i + t4_i) + transform_sign * (t0_i - t2_i);  // Rotation real
                fft_type rot3_i = -C8_1 * (t3_r + t4_r) + transform_sign * (t0_r - t2_r); // Rotation imag
                X3->re = u3_r - rot3_r;                                                   // X(3N/8) real
                X3->im = u3_i - rot3_i;                                                   // X(3N/8) imag
                X5->re = u3_r + rot3_r;                                                   // X(5N/8) real
                X5->im = u3_i + rot3_i;                                                   // X(5N/8) imag
            }

            // k=1 to k=3 (with twiddle multiplications)
            for (int k = 1; k < 4 && k < sub_fft_size; k++)
            {
                fft_data *X0 = &output_buffer[k];                    // X(k)
                fft_data *X1 = &output_buffer[sub_fft_size + k];     // X(k + N/8)
                fft_data *X2 = &output_buffer[2 * sub_fft_size + k]; // X(k + 2N/8)
                fft_data *X3 = &output_buffer[3 * sub_fft_size + k]; // X(k + 3N/8)
                fft_data *X4 = &output_buffer[4 * sub_fft_size + k]; // X(k + 4N/8)
                fft_data *X5 = &output_buffer[5 * sub_fft_size + k]; // X(k + 5N/8)
                fft_data *X6 = &output_buffer[6 * sub_fft_size + k]; // X(k + 6N/8)
                fft_data *X7 = &output_buffer[7 * sub_fft_size + k]; // X(k + 7N/8)

                fft_type a_r = out_re[k], a_i = out_im[k];                                       // X_0[k]
                fft_type b_r = out_re[sub_fft_size + k], b_i = out_im[sub_fft_size + k];         // X_1[k]
                fft_type c_r = out_re[2 * sub_fft_size + k], c_i = out_im[2 * sub_fft_size + k]; // X_2[k]
                fft_type d_r = out_re[3 * sub_fft_size + k], d_i = out_im[3 * sub_fft_size + k]; // X_3[k]
                fft_type e_r = out_re[4 * sub_fft_size + k], e_i = out_im[4 * sub_fft_size + k]; // X_4[k]
                fft_type f_r = out_re[5 * sub_fft_size + k], f_i = out_im[5 * sub_fft_size + k]; // X_5[k]
                fft_type g_r = out_re[6 * sub_fft_size + k], g_i = out_im[6 * sub_fft_size + k]; // X_6[k]
                fft_type h_r = out_re[7 * sub_fft_size + k], h_i = out_im[7 * sub_fft_size + k]; // X_7[k]

                // Load twiddle factors for k
                int idx = 7 * k;                                                               // Index for k
                fft_type w1r = twiddle_factors[idx + 0].re, w1i = twiddle_factors[idx + 0].im; // W_N^k
                fft_type w2r = twiddle_factors[idx + 1].re, w2i = twiddle_factors[idx + 1].im; // W_N^{2k}
                fft_type w3r = twiddle_factors[idx + 2].re, w3i = twiddle_factors[idx + 2].im; // W_N^{3k}
                fft_type w4r = twiddle_factors[idx + 3].re, w4i = twiddle_factors[idx + 3].im; // W_N^{4k}
                fft_type w5r = twiddle_factors[idx + 4].re, w5i = twiddle_factors[idx + 4].im; // W_N^{5k}
                fft_type w6r = twiddle_factors[idx + 5].re, w6i = twiddle_factors[idx + 5].im; // W_N^{6k}
                fft_type w7r = twiddle_factors[idx + 6].re, w7i = twiddle_factors[idx + 6].im; // W_N^{7k}

                // Apply twiddle factors
                fft_type b2_r = b_r * w1r - b_i * w1i, b2_i = b_i * w1r + b_r * w1i; // W_N^k * B
                fft_type c2_r = c_r * w2r - c_i * w2i, c2_i = c_i * w2r + c_r * w2i; // W_N^{2k} * C
                fft_type d2_r = d_r * w3r - d_i * w3i, d2_i = d_i * w3r + d_r * w3i; // W_N^{3k} * D
                fft_type e2_r = e_r * w4r - e_i * w4i, e2_i = e_i * w4r + e_r * w4i; // W_N^{4k} * E
                fft_type f2_r = f_r * w5r - f_i * w5i, f2_i = f_i * w5r + f_r * w5i; // W_N^{5k} * F
                fft_type g2_r = g_r * w6r - g_i * w6i, g2_i = g_i * w6r + g_r * w6i; // W_N^{6k} * G
                fft_type h2_r = h_r * w7r - h_i * w7i, h2_i = h_i * w7r + h_r * w7i; // W_N^{7k} * H

                // Compute pairwise sums and differences
                fft_type t0_r = b2_r + h2_r, t0_i = b2_i + h2_i; // B + H
                fft_type t1_r = c2_r + g2_r, t1_i = c2_i + g2_i; // C + G
                fft_type t2_r = d2_r + f2_r, t2_i = d2_i + f2_i; // D + F
                fft_type t3_r = b2_r - h2_r, t3_i = b2_i - h2_i; // B - H
                fft_type t4_r = c2_r - g2_r, t4_i = c2_i - g2_i; // C - G
                fft_type t5_r = d2_r - f2_r, t5_i = d2_i - f2_i; // D - F

                // Compute X_0 and X_4
                X0->re = a_r + t0_r + t1_r + t2_r;   // X(k) real
                X0->im = a_i + t0_i + t1_i + t2_i;   // X(k) imag
                X4->re = a_r - (t0_r + t1_r) + t2_r; // X(k + 4N/8) real
                X4->im = a_i - (t0_i + t1_i) + t2_i; // X(k + 4N/8) imag

                // Compute helpers
                fft_type tmp1_r = a_r - t2_r, tmp1_i = a_i - t2_i;   // a - (d+f)
                fft_type tmp2_r = t3_r - t5_r, tmp2_i = t3_i - t5_i; // (b-h) - (d-f)

                // Compute X_1 and X_7
                fft_type u1_r = tmp1_r + t1_r, u1_i = tmp1_i + t1_i;               // Base term
                fft_type rot1_r = C8_1 * (t0_i - t2_i) + transform_sign * tmp2_i;  // Rotation real
                fft_type rot1_i = -C8_1 * (t0_r - t2_r) + transform_sign * tmp2_r; // Rotation imag
                X1->re = u1_r - rot1_r;                                            // X(k + N/8) real
                X1->im = u1_i - rot1_i;                                            // X(k + N/8) imag
                X7->re = u1_r + rot1_r;                                            // X(k + 7N/8) real
                X7->im = u1_i + rot1_i;                                            // X(k + 7N/8) imag

                // Compute X_2 and X_6
                fft_type u2_r = tmp1_r + t0_r, u2_i = tmp1_i + t0_i;                      // Base term
                fft_type rot2_r = C8_1 * (t1_i - t2_i) + transform_sign * (t4_i - t5_i);  // Rotation real
                fft_type rot2_i = -C8_1 * (t1_r - t2_r) + transform_sign * (t4_r - t5_r); // Rotation imag
                X2->re = u2_r - rot2_r;                                                   // X(k + 2N/8) real
                X2->im = u2_i - rot2_i;                                                   // X(k + 2N/8) imag
                X6->re = u2_r + rot2_r;                                                   // X(k + 6N/8) real
                X6->im = u2_i + rot2_i;                                                   // X(k + 6N/8) imag

                // Compute X_3 and X_5
                fft_type u3_r = tmp1_r + t5_r, u3_i = tmp1_i + t5_i;                      // Base term
                fft_type rot3_r = C8_1 * (t3_i + t4_i) + transform_sign * (t0_i - t2_i);  // Rotation real
                fft_type rot3_i = -C8_1 * (t3_r + t4_r) + transform_sign * (t0_r - t2_r); // Rotation imag
                X3->re = u3_r - rot3_r;                                                   // X(k + 3N/8) real
                X3->im = u3_i - rot3_i;                                                   // X(k + 3N/8) imag
                X5->re = u3_r + rot3_r;                                                   // X(k + 5N/8) real
                X5->im = u3_i + rot3_i;                                                   // X(k + 5N/8) imag
            }
        }

        // Step 9: SSE2 vectorized butterfly for k=4 to sub_fft_size-1
        __m128d vsign = _mm_set1_pd((double)transform_sign); // Vectorized transform sign
        __m128d vc8_1 = _mm_set1_pd(C8_1);                   // √2/2
        int k = 4;
        for (; k + 1 < sub_fft_size; k += 2)
        {
            // Load eight sub-FFT points for two k values (32-byte aligned with USE_ALIGNED_SIMD)
            __m128d a_r = LOAD_SSE2(out_re + k);                    // X_0[k:k+1] real
            __m128d a_i = LOAD_SSE2(out_im + k);                    // X_0[k:k+1] imag
            __m128d b_r = LOAD_SSE2(out_re + k + sub_fft_size);     // X_1[k:k+1] real
            __m128d b_i = LOAD_SSE2(out_im + k + sub_fft_size);     // X_1[k:k+1] imag
            __m128d c_r = LOAD_SSE2(out_re + k + 2 * sub_fft_size); // X_2[k:k+1] real
            __m128d c_i = LOAD_SSE2(out_im + k + 2 * sub_fft_size); // X_2[k:k+1] imag
            __m128d d_r = LOAD_SSE2(out_re + k + 3 * sub_fft_size); // X_3[k:k+1] real
            __m128d d_i = LOAD_SSE2(out_im + k + 3 * sub_fft_size); // X_3[k:k+1] imag
            __m128d e_r = LOAD_SSE2(out_re + k + 4 * sub_fft_size); // X_4[k:k+1] real
            __m128d e_i = LOAD_SSE2(out_im + k + 4 * sub_fft_size); // X_4[k:k+1] imag
            __m128d f_r = LOAD_SSE2(out_re + k + 5 * sub_fft_size); // X_5[k:k+1] real
            __m128d f_i = LOAD_SSE2(out_im + k + 5 * sub_fft_size); // X_5[k:k+1] imag
            __m128d g_r = LOAD_SSE2(out_re + k + 6 * sub_fft_size); // X_6[k:k+1] real
            __m128d g_i = LOAD_SSE2(out_im + k + 6 * sub_fft_size); // X_6[k:k+1] imag
            __m128d h_r = LOAD_SSE2(out_re + k + 7 * sub_fft_size); // X_7[k:k+1] real
            __m128d h_i = LOAD_SSE2(out_im + k + 7 * sub_fft_size); // X_7[k:k+1] imag

            // Load seven twiddle factors for two k values (aligned for power-of-8, scratch for mixed-radix)
            int idx = 7 * k;                                       // Base index for W_N^k, ..., W_N^{7k}
            __m128d w1r = LOAD_SSE2(&twiddle_factors[idx + 0].re); // W_N^k real
            __m128d w1i = LOAD_SSE2(&twiddle_factors[idx + 0].im); // W_N^k imag
            __m128d w2r = LOAD_SSE2(&twiddle_factors[idx + 1].re); // W_N^{2k} real
            __m128d w2i = LOAD_SSE2(&twiddle_factors[idx + 1].im); // W_N^{2k} imag
            __m128d w3r = LOAD_SSE2(&twiddle_factors[idx + 2].re); // W_N^{3k} real
            __m128d w3i = LOAD_SSE2(&twiddle_factors[idx + 2].im); // W_N^{3k} imag
            __m128d w4r = LOAD_SSE2(&twiddle_factors[idx + 3].re); // W_N^{4k} real
            __m128d w4i = LOAD_SSE2(&twiddle_factors[idx + 3].im); // W_N^{4k} imag
            __m128d w5r = LOAD_SSE2(&twiddle_factors[idx + 4].re); // W_N^{5k} real
            __m128d w5i = LOAD_SSE2(&twiddle_factors[idx + 4].im); // W_N^{5k} imag
            __m128d w6r = LOAD_SSE2(&twiddle_factors[idx + 5].re); // W_N^{6k} real
            __m128d w6i = LOAD_SSE2(&twiddle_factors[idx + 5].im); // W_N^{6k} imag
            __m128d w7r = LOAD_SSE2(&twiddle_factors[idx + 6].re); // W_N^{7k} real
            __m128d w7i = LOAD_SSE2(&twiddle_factors[idx + 6].im); // W_N^{7k} imag

            // Apply twiddle factors
            __m128d b2r = FMSUB_SSE2(b_r, w1r, _mm_mul_pd(b_i, w1i)); // W_N^k * B real
            __m128d b2i = FMADD_SSE2(b_i, w1r, _mm_mul_pd(b_r, w1i)); // W_N^k * B imag
            __m128d c2r = FMSUB_SSE2(c_r, w2r, _mm_mul_pd(c_i, w2i)); // W_N^{2k} * C real
            __m128d c2i = FMADD_SSE2(c_i, w2r, _mm_mul_pd(c_r, w2i)); // W_N^{2k} * C imag
            __m128d d2r = FMSUB_SSE2(d_r, w3r, _mm_mul_pd(d_i, w3i)); // W_N^{3k} * D real
            __m128d d2i = FMADD_SSE2(d_i, w3r, _mm_mul_pd(d_r, w3i)); // W_N^{3k} * D imag
            __m128d e2r = FMSUB_SSE2(e_r, w4r, _mm_mul_pd(e_i, w4i)); // W_N^{4k} * E real
            __m128d e2i = FMADD_SSE2(e_i, w4r, _mm_mul_pd(e_r, w4i)); // W_N^{4k} * E imag
            __m128d f2r = FMSUB_SSE2(f_r, w5r, _mm_mul_pd(f_i, w5i)); // W_N^{5k} * F real
            __m128d f2i = FMADD_SSE2(f_i, w5r, _mm_mul_pd(f_r, w5i)); // W_N^{5k} * F imag
            __m128d g2r = FMSUB_SSE2(g_r, w6r, _mm_mul_pd(g_i, w6i)); // W_N^{6k} * G real
            __m128d g2i = FMADD_SSE2(g_i, w6r, _mm_mul_pd(g_r, w6i)); // W_N^{6k} * G imag
            __m128d h2r = FMSUB_SSE2(h_r, w7r, _mm_mul_pd(h_i, w7i)); // W_N^{7k} * H real
            __m128d h2i = FMADD_SSE2(h_i, w7r, _mm_mul_pd(h_r, w7i)); // W_N^{7k} * H imag

            // Compute pairwise sums and differences
            __m128d t0r = _mm_add_pd(b2r, h2r); // B + H real
            __m128d t0i = _mm_add_pd(b2i, h2i); // B + H imag
            __m128d t1r = _mm_add_pd(c2r, g2r); // C + G real
            __m128d t1i = _mm_add_pd(c2i, g2i); // C + G imag
            __m128d t2r = _mm_add_pd(d2r, f2r); // D + F real
            __m128d t2i = _mm_add_pd(d2i, f2i); // D + F imag
            __m128d t3r = _mm_sub_pd(b2r, h2r); // B - H real
            __m128d t3i = _mm_sub_pd(b2i, h2i); // B - H imag
            __m128d t4r = _mm_sub_pd(c2r, g2r); // C - G real
            __m128d t4i = _mm_sub_pd(c2i, g2i); // C - G imag
            __m128d t5r = _mm_sub_pd(d2r, f2r); // D - F real
            __m128d t5i = _mm_sub_pd(d2i, f2i); // D - F imag

            // Compute X_0: a + (b+h) + (c+g) + (d+f)
            __m128d sumr = _mm_add_pd(a_r, _mm_add_pd(t0r, _mm_add_pd(t1r, t2r)));
            __m128d sumi = _mm_add_pd(a_i, _mm_add_pd(t0i, _mm_add_pd(t1i, t2i)));
            STORE_SSE2(out_re + k, sumr); // X(k) real
            STORE_SSE2(out_im + k, sumi); // X(k) imag

            // Compute X_4: a - (b+h) - (c+g) + (d+f)
            __m128d x4r = _mm_add_pd(_mm_sub_pd(a_r, _mm_add_pd(t0r, t1r)), t2r);
            __m128d x4i = _mm_add_pd(_mm_sub_pd(a_i, _mm_add_pd(t0i, t1i)), t2i);
            STORE_SSE2(out_re + k + 4 * sub_fft_size, x4r); // X(k + 4N/8) real
            STORE_SSE2(out_im + k + 4 * sub_fft_size, x4i); // X(k + 4N/8) imag

            // Compute helpers
            __m128d tmp1r = _mm_sub_pd(a_r, t2r), tmp1i = _mm_sub_pd(a_i, t2i); // a - (d+f)
            __m128d tmp2r = _mm_sub_pd(t3r, t5r), tmp2i = _mm_sub_pd(t3i, t5i); // (b-h) - (d-f)

            // Compute X_1 and X_7
            __m128d u1r = _mm_add_pd(tmp1r, t1r), u1i = _mm_add_pd(tmp1i, t1i);                             // Base term
            __m128d rot1r = FMADD_SSE2(_mm_sub_pd(t0i, t2i), vc8_1, _mm_mul_pd(vsign, tmp2i));              // Rotation real
            __m128d rot1i = FMADD_SSE2(_mm_sub_pd(t0r, t2r), _mm_set1_pd(-C8_1), _mm_mul_pd(vsign, tmp2r)); // Rotation imag
            STORE_SSE2(out_re + k + sub_fft_size, _mm_sub_pd(u1r, rot1r));                                  // X(k + N/8) real
            STORE_SSE2(out_im + k + sub_fft_size, _mm_sub_pd(u1i, rot1i));                                  // X(k + N/8) imag
            STORE_SSE2(out_re + k + 7 * sub_fft_size, _mm_add_pd(u1r, rot1r));                              // X(k + 7N/8) real
            STORE_SSE2(out_im + k + 7 * sub_fft_size, _mm_add_pd(u1i, rot1i));                              // X(k + 7N/8) imag

            // Compute X_2 and X_6
            __m128d u2r = _mm_add_pd(tmp1r, t0r), u2i = _mm_add_pd(tmp1i, t0i);                                            // Base term
            __m128d rot2r = FMADD_SSE2(_mm_sub_pd(t1i, t2i), vc8_1, _mm_mul_pd(vsign, _mm_sub_pd(t4i, t5i)));              // Rotation real
            __m128d rot2i = FMADD_SSE2(_mm_sub_pd(t1r, t2r), _mm_set1_pd(-C8_1), _mm_mul_pd(vsign, _mm_sub_pd(t4r, t5r))); // Rotation imag
            STORE_SSE2(out_re + k + 2 * sub_fft_size, _mm_sub_pd(u2r, rot2r));                                             // X(k + 2N/8) real
            STORE_SSE2(out_im + k + 2 * sub_fft_size, _mm_sub_pd(u2i, rot2i));                                             // X(k + 2N/8) imag
            STORE_SSE2(out_re + k + 6 * sub_fft_size, _mm_add_pd(u2r, rot2r));                                             // X(k + 6N/8) real
            STORE_SSE2(out_im + k + 6 * sub_fft_size, _mm_add_pd(u2i, rot2i));                                             // X(k + 6N/8) imag

            // Compute X_3 and X_5
            __m128d u3r = _mm_add_pd(tmp1r, t5r), u3i = _mm_add_pd(tmp1i, t5i);                                            // Base term
            __m128d rot3r = FMADD_SSE2(_mm_add_pd(t3i, t4i), vc8_1, _mm_mul_pd(vsign, _mm_sub_pd(t0i, t2i)));              // Rotation real
            __m128d rot3i = FMADD_SSE2(_mm_add_pd(t3r, t4r), _mm_set1_pd(-C8_1), _mm_mul_pd(vsign, _mm_sub_pd(t0r, t2r))); // Rotation imag
            STORE_SSE2(out_re + k + 3 * sub_fft_size, _mm_sub_pd(u3r, rot3r));                                             // X(k + 3N/8) real
            STORE_SSE2(out_im + k + 3 * sub_fft_size, _mm_sub_pd(u3i, rot3i));                                             // X(k + 3N/8) imag
            STORE_SSE2(out_re + k + 5 * sub_fft_size, _mm_add_pd(u3r, rot3r));                                             // X(k + 5N/8) real
            STORE_SSE2(out_im + k + 5 * sub_fft_size, _mm_add_pd(u3i, rot3i));                                             // X(k + 5N/8) imag
        }

        // Step 10: Scalar tail for remaining k
        for (; k < sub_fft_size; k++)
        {
            // Load twiddle factors
            int idx = 7 * k;                            // Index into twiddle array
            fft_type w1r = twiddle_factors[idx + 0].re; // W_N^k real
            fft_type w1i = twiddle_factors[idx + 0].im; // W_N^k imag
            fft_type w2r = twiddle_factors[idx + 1].re; // W_N^{2k} real
            fft_type w2i = twiddle_factors[idx + 1].im; // W_N^{2k} imag
            fft_type w3r = twiddle_factors[idx + 2].re; // W_N^{3k} real
            fft_type w3i = twiddle_factors[idx + 2].im; // W_N^{3k} imag
            fft_type w4r = twiddle_factors[idx + 3].re; // W_N^{4k} real
            fft_type w4i = twiddle_factors[idx + 3].im; // W_N^{4k} imag
            fft_type w5r = twiddle_factors[idx + 4].re; // W_N^{5k} real
            fft_type w5i = twiddle_factors[idx + 4].im; // W_N^{5k} imag
            fft_type w6r = twiddle_factors[idx + 5].re; // W_N^{6k} real
            fft_type w6i = twiddle_factors[idx + 5].im; // W_N^{6k} imag
            fft_type w7r = twiddle_factors[idx + 6].re; // W_N^{7k} real
            fft_type w7i = twiddle_factors[idx + 6].im; // W_N^{7k} imag

            // Load sub-FFT points
            fft_type a_r = out_re[k], a_i = out_im[k];                                       // X_0[k]
            fft_type b_r = out_re[k + sub_fft_size], b_i = out_im[k + sub_fft_size];         // X_1[k]
            fft_type c_r = out_re[k + 2 * sub_fft_size], c_i = out_im[k + 2 * sub_fft_size]; // X_2[k]
            fft_type d_r = out_re[k + 3 * sub_fft_size], d_i = out_im[k + 3 * sub_fft_size]; // X_3[k]
            fft_type e_r = out_re[k + 4 * sub_fft_size], e_i = out_im[k + 4 * sub_fft_size]; // X_4[k]
            fft_type f_r = out_re[k + 5 * sub_fft_size], f_i = out_im[k + 5 * sub_fft_size]; // X_5[k]
            fft_type g_r = out_re[k + 6 * sub_fft_size], g_i = out_im[k + 6 * sub_fft_size]; // X_6[k]
            fft_type h_r = out_re[k + 7 * sub_fft_size], h_i = out_im[k + 7 * sub_fft_size]; // X_7[k]

            // Apply twiddle factors
            fft_type b2_r = b_r * w1r - b_i * w1i, b2_i = b_i * w1r + b_r * w1i; // W_N^k * B
            fft_type c2_r = c_r * w2r - c_i * w2i, c2_i = c_i * w2r + c_r * w2i; // W_N^{2k} * C
            fft_type d2_r = d_r * w3r - d_i * w3i, d2_i = d_i * w3r + d_r * w3i; // W_N^{3k} * D
            fft_type e2_r = e_r * w4r - e_i * w4i, e2_i = e_i * w4r + e_r * w4i; // W_N^{4k} * E
            fft_type f2_r = f_r * w5r - f_i * w5i, f2_i = f_i * w5r + f_r * w5i; // W_N^{5k} * F
            fft_type g2_r = g_r * w6r - g_i * w6i, g2_i = g_i * w6r + g_r * w6i; // W_N^{6k} * G
            fft_type h2_r = h_r * w7r - h_i * w7i, h2_i = h_i * w7r + h_r * w7i; // W_N^{7k} * H

            // Compute pairwise sums and differences
            fft_type t0_r = b2_r + h2_r, t0_i = b2_i + h2_i; // B + H
            fft_type t1_r = c2_r + g2_r, t1_i = c2_i + g2_i; // C + G
            fft_type t2_r = d2_r + f2_r, t2_i = d2_i + f2_i; // D + F
            fft_type t3_r = b2_r - h2_r, t3_i = b2_i - h2_i; // B - H
            fft_type t4_r = c2_r - g2_r, t4_i = c2_i - g2_i; // C - G
            fft_type t5_r = d2_r - f2_r, t5_i = d2_i - f2_i; // D - F

            // Compute X_0 and X_4
            out_re[k] = a_r + t0_r + t1_r + t2_r;                      // X(k) real
            out_im[k] = a_i + t0_i + t1_i + t2_i;                      // X(k) imag
            out_re[k + 4 * sub_fft_size] = a_r - (t0_r + t1_r) + t2_r; // X(k + 4N/8) real
            out_im[k + 4 * sub_fft_size] = a_i - (t0_i + t1_i) + t2_i; // X(k + 4N/8) imag

            // Compute helpers
            fft_type tmp1_r = a_r - t2_r, tmp1_i = a_i - t2_i;   // a - (d+f)
            fft_type tmp2_r = t3_r - t5_r, tmp2_i = t3_i - t5_i; // (b-h) - (d-f)

            // Compute X_1 and X_7
            fft_type u1_r = tmp1_r + t1_r, u1_i = tmp1_i + t1_i;               // Base term
            fft_type rot1_r = C8_1 * (t0_i - t2_i) + transform_sign * tmp2_i;  // Rotation real
            fft_type rot1_i = -C8_1 * (t0_r - t2_r) + transform_sign * tmp2_r; // Rotation imag
            out_re[k + sub_fft_size] = u1_r - rot1_r;                          // X(k + N/8) real
            out_im[k + sub_fft_size] = u1_i - rot1_i;                          // X(k + N/8) imag
            out_re[k + 7 * sub_fft_size] = u1_r + rot1_r;                      // X(k + 7N/8) real
            out_im[k + 7 * sub_fft_size] = u1_i + rot1_i;                      // X(k + 7N/8) imag

            // Compute X_2 and X_6
            fft_type u2_r = tmp1_r + t0_r, u2_i = tmp1_i + t0_i;                      // Base term
            fft_type rot2_r = C8_1 * (t1_i - t2_i) + transform_sign * (t4_i - t5_i);  // Rotation real
            fft_type rot2_i = -C8_1 * (t1_r - t2_r) + transform_sign * (t4_r - t5_r); // Rotation imag
            out_re[k + 2 * sub_fft_size] = u2_r - rot2_r;                             // X(k + 2N/8) real
            out_im[k + 2 * sub_fft_size] = u2_i - rot2_i;                             // X(k + 2N/8) imag
            out_re[k + 6 * sub_fft_size] = u2_r + rot2_r;                             // X(k + 6N/8) real
            out_im[k + 6 * sub_fft_size] = u2_i + rot2_i;                             // X(k + 6N/8) imag

            // Compute X_3 and X_5
            fft_type u3_r = tmp1_r + t5_r, u3_i = tmp1_i + t5_i;                      // Base term
            fft_type rot3_r = C8_1 * (t3_i + t4_i) + transform_sign * (t0_i - t2_i);  // Rotation real
            fft_type rot3_i = -C8_1 * (t3_r + t4_r) + transform_sign * (t0_r - t2_r); // Rotation imag
            out_re[k + 3 * sub_fft_size] = u3_r - rot3_r;                             // X(k + 3N/8) real
            out_im[k + 3 * sub_fft_size] = u3_i - rot3_i;                             // X(k + 3N/8) imag
            out_re[k + 5 * sub_fft_size] = u3_r + rot3_r;                             // X(k + 5N/8) real
            out_im[k + 5 * sub_fft_size] = u3_i + rot3_i;                             // X(k + 5N/8) imag
        }

        // Step 11: Copy flattened results back to output_buffer
        for (int lane = 0; lane < 8; lane++)
        {
            fft_data *base = output_buffer + lane * sub_fft_size;
            for (int k = 0; k < sub_fft_size; k++)
            {
                base[k].re = out_re[lane * sub_fft_size + k];
                base[k].im = out_im[lane * sub_fft_size + k];
            }
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
        /**
         * @brief General radix decomposition for prime factors greater than 8 with SSE2 vectorization.
         *
         * Intention: Compute the FFT for data lengths divisible by arbitrary radices (> 8) by splitting into
         * radix sub-FFTs of size N/radix, applying recursive FFTs, and combining results with twiddle factors.
         * Optimized for power-of-radix and mixed-radix FFTs using pre-allocated scratch and stage-specific
         * twiddle offsets. Follows FFTW’s two-buffer strategy for thread safety and performance.
         *
         * Mathematically: Computes:
         *   X(k) = \sum_{n=0}^{radix-1} X_n(k) \cdot W_N^{n*k},
         * where X_0, ..., X_{radix-1} are sub-FFTs of size N/radix, W_N^k = e^{-2πi k / N}.
         * Uses trigonometric symmetry to reduce computations for conjugate pairs.
         *
         * Process:
         * 1. Divide data into radix sub-FFTs of size N/radix, increasing stride by radix.
         * 2. Validate scratch size: (6*radix-2)*(N/radix) for power-of-radix, (7*radix-3)*(N/radix) for mixed-radix.
         * 3. Assign scratch slices for sub-FFT outputs, twiddles, cos/sin tables, y_real/y_imag, and butterfly outputs.
         * 4. Compute child scratch offsets with ceiling division to prevent overlap.
         * 5. Recursively compute sub-FFTs for indices n mod radix = 0 to radix-1.
         * 6. Load twiddle factors from stage_twiddle_offset (power-of-radix) or copy to scratch (mixed-radix).
         * 7. Precompute cosine/sine values for twiddle synthesis.
         * 8. Flatten sub-FFT outputs into dedicated real/imag arrays to avoid aliasing.
         * 9. Compute k=0 to k=1 scalarly, then vectorized butterfly with SSE2 for k=2 to sub_fft_size-1, scalar tail.
         * 10. Store results in a dedicated scratch buffer and copy to output_buffer.
         *
         * Optimization:
         * - SSE2 processes 2 complex points per iteration, using aligned loads/stores with USE_ALIGNED_SIMD.
         * - Pre-allocated scratch eliminates malloc/free overhead for most buffers.
         * - Single allocation for y_r/y_i arrays reduces heap churn.
         * - stage_twiddle_offset skips ad-hoc indexing for power-of-radix FFTs.
         * - twiddle_tables used for small N with USE_TWIDDLE_TABLES.
         * - Dedicated y_real/y_imag avoids strict-aliasing violations.
         * - k=0 to k=1 handled in scalar prologue to ensure all indices are computed.
         * - Scratch is 32-byte aligned, supporting aligned SIMD.
         *
         * @warning Assumes fft_obj->twiddles has n_fft ≥ N elements, scratch is 32-byte aligned,
         *          radix <= MAX_STAGES, and factor_index is valid for twiddle_factors.
         */
        // Step 1: Compute subproblem size and stride
        int sub_fft_length = data_length / radix; // Size of each sub-FFT: N/radix
        int next_stride = radix * stride;         // Stride for next recursion: stride * radix
        int sub_fft_size = sub_fft_length;        // Alias for sub-FFT size
        int mid_radix = (radix - 1) / 2;          // Half of radix-1 for conjugate pair symmetry

        // Step 2: Validate radix
        if (radix > MAX_STAGES)
        {
            fprintf(stderr, "Error: Radix %d exceeds maximum supported radix (%d)\n", radix, MAX_STAGES);
            // exit
        }

        // Step 3: Validate scratch buffer
        // Power-of-radix: (radix + 2*(radix-1) + 2*radix + radix) = (6*radix-2)*(N/radix)
        // Mixed-radix: (radix + (radix-1) + 2*(radix-1) + 2*radix + radix) = (7*radix-3)*(N/radix)
        // for sub_fft_outputs + twiddles + cos/sin + y_real/y_imag + butterfly_outputs
        int required_size = fft_obj->twiddle_factors != NULL ? (6 * radix - 2) * sub_fft_size : (7 * radix - 3) * sub_fft_size;
        if (scratch_offset + required_size > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Scratch buffer too small for radix-%d at offset %d (need %d, have %d)\n",
                    radix, scratch_offset, required_size, fft_obj->max_scratch_size - scratch_offset);
            // exit
        }

        // Step 4: Assign scratch slices
        // sub_fft_outputs: radix*sub_fft_size fft_data for sub-FFT results
        // twiddle_factors: (radix-1)*sub_fft_size fft_data for W_N^{n*k} (mixed-radix only)
        // cos_values/sin_values: 2*(radix-1)*sub_fft_size doubles for cos/sin(i*2π/radix)
        // y_real/y_imag: 2*radix*sub_fft_size doubles for flattened real/imag arrays
        // butterfly_outputs: radix*sub_fft_size fft_data for final FFT outputs
        fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
        fft_data *twiddle_factors;
        double *cos_values, *sin_values, *y_real, *y_imag;
        fft_data *butterfly_outputs;
        int twiddle_offset, cos_sin_offset, y_offset, butterfly_offset;
        if (fft_obj->twiddle_factors != NULL)
        {
            // Validate factor_index for power-of-radix FFTs
            if (factor_index >= fft_obj->num_precomputed_stages)
            {
                fprintf(stderr, "Error: Invalid factor_index (%d) exceeds num_precomputed_stages (%d) for radix-%d\n",
                        factor_index, fft_obj->num_precomputed_stages, radix);
                // exit
            }
            twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index]; // Precomputed twiddles
            twiddle_offset = scratch_offset + radix * sub_fft_size;
            cos_sin_offset = twiddle_offset;
            y_offset = cos_sin_offset + 2 * (radix - 1) * sub_fft_size;
            butterfly_offset = y_offset + 2 * radix * sub_fft_size;
        }
        else
        {
            twiddle_offset = scratch_offset + radix * sub_fft_size;
            twiddle_factors = fft_obj->scratch + twiddle_offset; // Scratch space for twiddles
            cos_sin_offset = twiddle_offset + (radix - 1) * sub_fft_size;
            y_offset = cos_sin_offset + 2 * (radix - 1) * sub_fft_size;
            butterfly_offset = y_offset + 2 * radix * sub_fft_size;
        }
        cos_values = (double *)(fft_obj->scratch + cos_sin_offset);                              // cos(i*2π/radix) table
        sin_values = (double *)(fft_obj->scratch + cos_sin_offset + (radix - 1) * sub_fft_size); // sin(i*2π/radix) table
        y_real = (double *)(fft_obj->scratch + y_offset);                                        // Dedicated block for real components
        y_imag = (double *)(fft_obj->scratch + y_offset + radix * sub_fft_size);                 // Dedicated block for imag components
        butterfly_outputs = fft_obj->scratch + butterfly_offset;                                 // Final FFT outputs

        // Step 5: Compute child scratch offsets
        // Each child needs ceil(sub_fft_size/radix)*(6*radix-2) (power-of-radix) or ceil(sub_fft_size/radix)*(7*radix-3) (mixed-radix)
        int child_sub_fft_size = (sub_fft_size + radix - 1) / radix; // Ceiling division for child FFT size
        int child_scratch_per_branch = fft_obj->twiddle_factors != NULL ? (6 * radix - 2) * child_sub_fft_size : (7 * radix - 3) * child_sub_fft_size;
        int total_child_scratch = radix * child_scratch_per_branch; // Total scratch for all children
        if (scratch_offset + total_child_scratch + (fft_obj->twiddle_factors ? 0 : (radix - 1) * sub_fft_size) > fft_obj->max_scratch_size)
        {
            fprintf(stderr, "Error: Total child scratch size (%d) exceeds available scratch at offset %d (have %d) for radix-%d\n",
                    total_child_scratch + (fft_obj->twiddle_factors ? 0 : (radix - 1) * sub_fft_size),
                    scratch_offset, fft_obj->max_scratch_size - scratch_offset, radix);
            // exit
        }
        int *child_offsets = (int *)malloc(radix * sizeof(int));
        if (!child_offsets)
        {
            fprintf(stderr, "Error: Memory allocation failed for child_offsets\n");
            // exit
        }
        child_offsets[0] = scratch_offset;
        for (int i = 1; i < radix; i++)
        {
            child_offsets[i] = child_offsets[i - 1] + child_scratch_per_branch; // Cumulative scratch offsets
        }

        // Step 6: Recurse on radix sub-FFTs
        for (int i = 0; i < radix; i++)
        {
            mixed_radix_dit_rec(sub_fft_outputs + i * sub_fft_size, input_buffer + i * stride, fft_obj,
                                transform_sign, sub_fft_length, next_stride, factor_index + 1,
                                child_offsets[i]); // Compute X_i(k) for i=0 to radix-1
        }

        // Step 7: Prepare twiddle factors (mixed-radix only)
        if (fft_obj->twiddle_factors == NULL)
        {
            if (fft_obj->n_fft < sub_fft_length - 1 + (radix - 1) * sub_fft_size)
            {
                fprintf(stderr, "Error: Twiddle array too small (need at least %d elements, have %d)\n",
                        sub_fft_length - 1 + (radix - 1) * sub_fft_size, fft_obj->n_fft);
                // exit
            }
#ifdef USE_TWIDDLE_TABLES
            if (sub_fft_size <= radix && twiddle_tables[radix] != NULL)
            {
                const complex_t *table = twiddle_tables[radix];
                for (int k = 0; k < sub_fft_size; k++)
                {
                    for (int n = 0; n < radix - 1; n++)
                    {
                        twiddle_factors[(radix - 1) * k + n].re = table[n + 1].re; // Copy W_N^{n*k} real from table
                        twiddle_factors[(radix - 1) * k + n].im = table[n + 1].im; // Copy W_N^{n*k} imag from table
                    }
                }
            }
            else
#endif
            {
                for (int k = 0; k < sub_fft_size; k++)
                {
                    for (int n = 0; n < radix - 1; n++)
                    {
                        int idx = n * sub_fft_size + k;                                      // Index for W_N^{n*k}
                        twiddle_factors[(radix - 1) * k + n].re = fft_obj->twiddles[idx].re; // Store W_N^{n*k} real
                        twiddle_factors[(radix - 1) * k + n].im = fft_obj->twiddles[idx].im; // Store W_N^{n*k} imag
                    }
                }
            }
        }

        // Step 8: Precompute cosine and sine values
        for (int i = 1; i <= mid_radix; i++)
        {
            cos_values[i - 1] = cos(i * PI2 / radix); // Compute cos(i*2π/radix) for twiddle synthesis
            sin_values[i - 1] = sin(i * PI2 / radix); // Compute sin(i*2π/radix) for twiddle synthesis
        }
        for (int i = 0; i < mid_radix; i++)
        {
            sin_values[i + mid_radix] = -sin_values[mid_radix - 1 - i]; // Mirror: sin((radix-i)*2π/radix) = -sin(i*2π/radix)
            cos_values[i + mid_radix] = cos_values[mid_radix - 1 - i];  // Mirror: cos((radix-i)*2π/radix) = cos(i*2π/radix)
        }

        // Step 9: Flatten outputs into dedicated real/imag arrays
        for (int lane = 0; lane < radix; lane++)
        {
            fft_data *base = sub_fft_outputs + lane * sub_fft_size;
            for (int k = 0; k < sub_fft_size; k++)
            {
                y_real[lane * sub_fft_size + k] = base[k].re; // Copy real part of X_lane(k) to dedicated block
                y_imag[lane * sub_fft_size + k] = base[k].im; // Copy imag part of X_lane(k) to dedicated block
            }
        }

        // Step 10: Scalar prologue for k=0 and k=1
        {
            // k=0 (no twiddle multiplications)
            {
                fft_data *X0 = &butterfly_outputs[0];
                fft_type sum_r = 0.0, sum_i = 0.0;
                for (int i = 0; i < radix; i++)
                {
                    sum_r += y_real[i * sub_fft_size]; // Sum X_i(0) real parts for X(0)
                    sum_i += y_imag[i * sub_fft_size]; // Sum X_i(0) imag parts for X(0)
                }
                X0->re = sum_r; // X(0) real = \sum_{i=0}^{radix-1} X_i(0) real
                X0->im = sum_i; // X(0) imag = \sum_{i=0}^{radix-1} X_i(0) imag

                for (int u = 0; u < mid_radix; u++)
                {
                    fft_type temp1_r = y_real[0], temp1_i = y_imag[0]; // Initialize with X_0(0)
                    fft_type temp2_r = 0.0, temp2_i = 0.0;             // Initialize rotation terms
                    for (int v = 0; v < mid_radix; v++)
                    {
                        int temp = (u + 1) * (v + 1); // Compute index (u+1)(v+1) mod radix
                        while (temp >= radix)
                            temp -= radix;                                                                              // Reduce mod radix
                        int temp_temp = temp - 1;                                                                       // Index for cos/sin tables
                        fft_type tau_r = y_real[(v + 1) * sub_fft_size] + y_real[(radix - 1 - v) * sub_fft_size];       // X_{v+1}(0) + X_{radix-1-v}(0) real
                        fft_type tau_i = y_imag[(v + 1) * sub_fft_size] + y_imag[(radix - 1 - v) * sub_fft_size];       // X_{v+1}(0) + X_{radix-1-v}(0) imag
                        fft_type tau_r_minus = y_real[(v + 1) * sub_fft_size] - y_real[(radix - 1 - v) * sub_fft_size]; // X_{v+1}(0) - X_{radix-1-v}(0) real
                        fft_type tau_i_minus = y_imag[(v + 1) * sub_fft_size] - y_imag[(radix - 1 - v) * sub_fft_size]; // X_{v+1}(0) - X_{radix-1-v}(0) imag
                        temp1_r += cos_values[temp_temp] * tau_r;                                                       // Add cos((u+1)(v+1)*2π/radix) * (X_{v+1} + X_{radix-1-v}) real
                        temp1_i += cos_values[temp_temp] * tau_i;                                                       // Add cos((u+1)(v+1)*2π/radix) * (X_{v+1} + X_{radix-1-v}) imag
                        temp2_r -= sin_values[temp_temp] * tau_r_minus;                                                 // Subtract sin((u+1)(v+1)*2π/radix) * (X_{v+1} - X_{radix-1-v}) real
                        temp2_i -= sin_values[temp_temp] * tau_i_minus;                                                 // Subtract sin((u+1)(v+1)*2π/radix) * (X_{v+1} - X_{radix-1-v}) imag
                    }
                    temp2_r = transform_sign * temp2_r;                                       // Apply sign (±1) for forward/inverse transform
                    temp2_i = transform_sign * temp2_i;                                       // Apply sign (±1) for forward/inverse transform
                    butterfly_outputs[(u + 1) * sub_fft_size].re = temp1_r - temp2_i;         // X((u+1)N/radix) real
                    butterfly_outputs[(u + 1) * sub_fft_size].im = temp1_i + temp2_r;         // X((u+1)N/radix) imag
                    butterfly_outputs[(radix - u - 1) * sub_fft_size].re = temp1_r + temp2_i; // X((radix-u-1)N/radix) real (conjugate pair)
                    butterfly_outputs[(radix - u - 1) * sub_fft_size].im = temp1_i - temp2_r; // X((radix-u-1)N/radix) imag (conjugate pair)
                }
            }

            // k=1 (with twiddle multiplications)
            if (sub_fft_size > 1)
            {
                fft_type *y_r = (fft_type *)malloc(radix * sizeof(fft_type));
                fft_type *y_i = (fft_type *)malloc(radix * sizeof(fft_type));
                if (!y_r || !y_i)
                {
                    fprintf(stderr, "Error: Memory allocation failed for y_r/y_i\n");
                    free(y_r);
                    free(y_i);
                    free(child_offsets);
                    // exit
                }
                fft_data *X0 = &butterfly_outputs[1];
                y_r[0] = y_real[1]; // X_0(1) real
                y_i[0] = y_imag[1]; // X_0(1) imag
                for (int i = 0; i < radix - 1; i++)
                {
                    int idx = (radix - 1) * 1 + i;                                         // Index for W_N^{i*1}
                    fft_type w_r = twiddle_factors[idx].re, w_i = twiddle_factors[idx].im; // Twiddle W_N^{i}
                    fft_type x_r = y_real[(i + 1) * sub_fft_size + 1];                     // X_{i+1}(1) real
                    fft_type x_i = y_imag[(i + 1) * sub_fft_size + 1];                     // X_{i+1}(1) imag
                    y_r[i + 1] = x_r * w_r - x_i * w_i;                                    // Real part of X_{i+1}(1) * W_N^{i}
                    y_i[i + 1] = x_i * w_r + x_r * w_i;                                    // Imag part of X_{i+1}(1) * W_N^{i}
                }
                fft_type sum_r = y_r[0], sum_i = y_i[0]; // Initialize sum with X_0(1)
                for (int i = 0; i < mid_radix; i++)
                {
                    fft_type tau_r = y_r[i + 1] + y_r[radix - 1 - i]; // Sum X_{i+1}(1)*W_N^{i+1} + X_{radix-1-i}(1)*W_N^{radix-1-i} real
                    fft_type tau_i = y_i[i + 1] + y_i[radix - 1 - i]; // Sum X_{i+1}(1)*W_N^{i+1} + X_{radix-1-i}(1)*W_N^{radix-1-i} imag
                    sum_r += tau_r;                                   // Add to X(1) real
                    sum_i += tau_i;                                   // Add to X(1) imag
                }
                X0->re = sum_r; // X(1) real = \sum_{i=0}^{radix-1} X_i(1) * W_N^{i}
                X0->im = sum_i; // X(1) imag = \sum_{i=0}^{radix-1} X_i(1) * W_N^{i}

                for (int u = 0; u < mid_radix; u++)
                {
                    fft_type temp1_r = y_r[0], temp1_i = y_i[0]; // Initialize with X_0(1)
                    fft_type temp2_r = 0.0, temp2_i = 0.0;       // Initialize rotation terms
                    for (int v = 0; v < mid_radix; v++)
                    {
                        int temp = (u + 1) * (v + 1); // Compute index (u+1)(v+1) mod radix
                        while (temp >= radix)
                            temp -= radix;                                      // Reduce mod radix
                        int temp_temp = temp - 1;                               // Index for cos/sin tables
                        fft_type tau_r = y_r[v + 1] + y_r[radix - 1 - v];       // Sum X_{v+1}(1)*W_N^{v+1} + X_{radix-1-v}(1)*W_N^{radix-1-v} real
                        fft_type tau_i = y_i[v + 1] + y_i[radix - 1 - v];       // Sum X_{v+1}(1)*W_N^{v+1} + X_{radix-1-v}(1)*W_N^{radix-1-v} imag
                        fft_type tau_r_minus = y_r[v + 1] - y_r[radix - 1 - v]; // Diff X_{v+1}(1)*W_N^{v+1} - X_{radix-1-v}(1)*W_N^{radix-1-v} real
                        fft_type tau_i_minus = y_i[v + 1] - y_i[radix - 1 - v]; // Diff X_{v+1}(1)*W_N^{v+1} - X_{radix-1-v}(1)*W_N^{radix-1-v} imag
                        temp1_r += cos_values[temp_temp] * tau_r;               // Add cos((u+1)(v+1)*2π/radix) * sum real
                        temp1_i += cos_values[temp_temp] * tau_i;               // Add cos((u+1)(v+1)*2π/radix) * sum imag
                        temp2_r -= sin_values[temp_temp] * tau_r_minus;         // Subtract sin((u+1)(v+1)*2π/radix) * diff real
                        temp2_i -= sin_values[temp_temp] * tau_i_minus;         // Subtract sin((u+1)(v+1)*2π/radix) * diff imag
                    }
                    temp2_r = transform_sign * temp2_r;                                           // Apply sign for transform direction
                    temp2_i = transform_sign * temp2_i;                                           // Apply sign for transform direction
                    butterfly_outputs[(u + 1) * sub_fft_size + 1].re = temp1_r - temp2_i;         // X(1 + (u+1)N/radix) real
                    butterfly_outputs[(u + 1) * sub_fft_size + 1].im = temp1_i + temp2_r;         // X(1 + (u+1)N/radix) imag
                    butterfly_outputs[(radix - u - 1) * sub_fft_size + 1].re = temp1_r + temp2_i; // X(1 + (radix-u-1)N/radix) real
                    butterfly_outputs[(radix - u - 1) * sub_fft_size + 1].im = temp1_i - temp2_r; // X(1 + (radix-u-1)N/radix) imag
                }
                free(y_r);
                free(y_i);
            }
        }

        // Step 11: SSE2 vectorized butterfly for k=2 to sub_fft_size-1
        fft_type *y_r = (fft_type *)malloc(radix * sizeof(fft_type) * 2); // Allocate for k and k+1
        fft_type *y_i = (fft_type *)malloc(radix * sizeof(fft_type) * 2);
        if (!y_r || !y_i)
        {
            fprintf(stderr, "Error: Memory allocation failed for y_r/y_i\n");
            free(y_r);
            free(y_i);
            free(child_offsets);
            // exit
        }
        __m128d vsign = _mm_set1_pd((double)transform_sign); // Vectorized transform sign (±1)
        int k = 2;
        for (; k + 1 < sub_fft_size; k += 2)
        {
            fft_data *X0 = &butterfly_outputs[k];
            y_r[0] = y_real[k];     // X_0(k) real
            y_i[0] = y_imag[k];     // X_0(k) imag
            y_r[1] = y_real[k + 1]; // X_0(k+1) real
            y_i[1] = y_imag[k + 1]; // X_0(k+1) imag
            for (int i = 0; i < radix - 1; i++)
            {
                int idx = (radix - 1) * k + i;                                         // Index for W_N^{i*k}
                fft_type w_r = twiddle_factors[idx].re, w_i = twiddle_factors[idx].im; // Twiddle W_N^{i*k}
                fft_type x_r = y_real[(i + 1) * sub_fft_size + k];                     // X_{i+1}(k) real
                fft_type x_i = y_imag[(i + 1) * sub_fft_size + k];                     // X_{i+1}(k) imag
                y_r[2 * (i + 1)] = x_r * w_r - x_i * w_i;                              // Real part of X_{i+1}(k) * W_N^{i*k}
                y_i[2 * (i + 1)] = x_i * w_r + x_r * w_i;                              // Imag part of X_{i+1}(k) * W_N^{i*k}
                idx = (radix - 1) * (k + 1) + i;                                       // Index for W_N^{i*(k+1)}
                w_r = twiddle_factors[idx].re;
                w_i = twiddle_factors[idx].im;                // Twiddle W_N^{i*(k+1)}
                x_r = y_real[(i + 1) * sub_fft_size + k + 1]; // X_{i+1}(k+1) real
                x_i = y_imag[(i + 1) * sub_fft_size + k + 1]; // X_{i+1}(k+1) imag
                y_r[2 * (i + 1) + 1] = x_r * w_r - x_i * w_i; // Real part of X_{i+1}(k+1) * W_N^{i*(k+1)}
                y_i[2 * (i + 1) + 1] = x_i * w_r + x_r * w_i; // Imag part of X_{i+1}(k+1) * W_N^{i*(k+1)}
            }
            __m128d sum_r = LOAD_SSE2(&y_r[0]), sum_i = LOAD_SSE2(&y_i[0]); // Load X_0(k:k+1)
            for (int i = 0; i < mid_radix; i++)
            {
                __m128d tau_r = _mm_add_pd(LOAD_SSE2(&y_r[2 * (i + 1)]), LOAD_SSE2(&y_r[2 * (radix - 1 - i)])); // Sum X_{i+1}*W_N^{i+1*k} + X_{radix-1-i}*W_N^{(radix-1-i)*k} real
                __m128d tau_i = _mm_add_pd(LOAD_SSE2(&y_i[2 * (i + 1)]), LOAD_SSE2(&y_i[2 * (radix - 1 - i)])); // Sum X_{i+1}*W_N^{i+1*k} + X_{radix-1-i}*W_N^{(radix-1-i)*k} imag
                sum_r = _mm_add_pd(sum_r, tau_r);                                                               // Add to X(k:k+1) real
                sum_i = _mm_add_pd(sum_i, tau_i);                                                               // Add to X(k:k+1) imag
            }
            STORE_SSE2(&X0->re, sum_r); // Store X(k:k+1) real = \sum_{i=0}^{radix-1} X_i(k:k+1) * W_N^{i*k}
            STORE_SSE2(&X0->im, sum_i); // Store X(k:k+1) imag = \sum_{i=0}^{radix-1} X_i(k:k+1) * W_N^{i*k}

            for (int u = 0; u < mid_radix; u++)
            {
                __m128d temp1_r = LOAD_SSE2(&y_r[0]), temp1_i = LOAD_SSE2(&y_i[0]); // Initialize with X_0(k:k+1)
                __m128d temp2_r = _mm_setzero_pd(), temp2_i = _mm_setzero_pd();     // Initialize rotation terms
                for (int v = 0; v < mid_radix; v++)
                {
                    int temp = (u + 1) * (v + 1); // Compute index (u+1)(v+1) mod radix
                    while (temp >= radix)
                        temp -= radix;                                                                                    // Reduce mod radix
                    int temp_temp = temp - 1;                                                                             // Index for cos/sin tables
                    __m128d cos_v = _mm_set1_pd(cos_values[temp_temp]);                                                   // cos((u+1)(v+1)*2π/radix)
                    __m128d sin_v = _mm_set1_pd(sin_values[temp_temp]);                                                   // sin((u+1)(v+1)*2π/radix)
                    __m128d tau_r = _mm_add_pd(LOAD_SSE2(&y_r[2 * (v + 1)]), LOAD_SSE2(&y_r[2 * (radix - 1 - v)]));       // Sum X_{v+1}*W_N^{v+1*k} + X_{radix-1-v}*W_N^{(radix-1-v)*k} real
                    __m128d tau_i = _mm_add_pd(LOAD_SSE2(&y_i[2 * (v + 1)]), LOAD_SSE2(&y_i[2 * (radix - 1 - v)]));       // Sum X_{v+1}*W_N^{v+1*k} + X_{radix-1-v}*W_N^{(radix-1-v)*k} imag
                    __m128d tau_r_minus = _mm_sub_pd(LOAD_SSE2(&y_r[2 * (v + 1)]), LOAD_SSE2(&y_r[2 * (radix - 1 - v)])); // Diff X_{v+1}*W_N^{v+1*k} - X_{radix-1-v}*W_N^{(radix-1-v)*k} real
                    __m128d tau_i_minus = _mm_sub_pd(LOAD_SSE2(&y_i[2 * (v + 1)]), LOAD_SSE2(&y_i[2 * (radix - 1 - v)])); // Diff X_{v+1}*W_N^{v+1*k} - X_{radix-1-v}*W_N^{(radix-1-v)*k} imag
                    temp1_r = _mm_add_pd(temp1_r, _mm_mul_pd(cos_v, tau_r));                                              // Add cos * sum real
                    temp1_i = _mm_add_pd(temp1_i, _mm_mul_pd(cos_v, tau_i));                                              // Add cos * sum imag
                    temp2_r = _mm_sub_pd(temp2_r, _mm_mul_pd(sin_v, tau_r_minus));                                        // Subtract sin * diff real
                    temp2_i = _mm_sub_pd(temp2_i, _mm_mul_pd(sin_v, tau_i_minus));                                        // Subtract sin * diff imag
                }
                temp2_r = _mm_mul_pd(vsign, temp2_r); // Apply sign for transform direction
                temp2_i = _mm_mul_pd(vsign, temp2_i); // Apply sign for transform direction
                fft_data *Xu = &butterfly_outputs[(u + 1) * sub_fft_size + k];
                fft_data *Xradix_u = &butterfly_outputs[(radix - u - 1) * sub_fft_size + k];
                STORE_SSE2(&Xu->re, _mm_sub_pd(temp1_r, temp2_i));       // X(k:k+1 + (u+1)N/radix) real
                STORE_SSE2(&Xu->im, _mm_add_pd(temp1_i, temp2_r));       // X(k:k+1 + (u+1)N/radix) imag
                STORE_SSE2(&Xradix_u->re, _mm_add_pd(temp1_r, temp2_i)); // X(k:k+1 + (radix-u-1)N/radix) real
                STORE_SSE2(&Xradix_u->im, _mm_sub_pd(temp1_i, temp2_r)); // X(k:k+1 + (radix-u-1)N/radix) imag
            }
        }
        free(y_r);
        free(y_i);

        // Step 12: Scalar tail for remaining k
        for (; k < sub_fft_size; k++)
        {
            fft_type *y_r = (fft_type *)malloc(radix * sizeof(fft_type));
            fft_type *y_i = (fft_type *)malloc(radix * sizeof(fft_type));
            if (!y_r || !y_i)
            {
                fprintf(stderr, "Error: Memory allocation failed for y_r/y_i\n");
                free(y_r);
                free(y_i);
                free(child_offsets);
                // exit
            }
            fft_data *X0 = &butterfly_outputs[k];
            y_r[0] = y_real[k]; // X_0(k) real
            y_i[0] = y_imag[k]; // X_0(k) imag
            for (int i = 0; i < radix - 1; i++)
            {
                int idx = (radix - 1) * k + i;                                         // Index for W_N^{i*k}
                fft_type w_r = twiddle_factors[idx].re, w_i = twiddle_factors[idx].im; // Twiddle W_N^{i*k}
                fft_type x_r = y_real[(i + 1) * sub_fft_size + k];                     // X_{i+1}(k) real
                fft_type x_i = y_imag[(i + 1) * sub_fft_size + k];                     // X_{i+1}(k) imag
                y_r[i + 1] = x_r * w_r - x_i * w_i;                                    // Real part of X_{i+1}(k) * W_N^{i*k}
                y_i[i + 1] = x_i * w_r + x_r * w_i;                                    // Imag part of X_{i+1}(k) * W_N^{i*k}
            }
            fft_type sum_r = y_r[0], sum_i = y_i[0]; // Initialize sum with X_0(k)
            for (int i = 0; i < mid_radix; i++)
            {
                fft_type tau_r = y_r[i + 1] + y_r[radix - 1 - i]; // Sum X_{i+1}(k)*W_N^{i+1*k} + X_{radix-1-i}(k)*W_N^{(radix-1-i)*k} real
                fft_type tau_i = y_i[i + 1] + y_i[radix - 1 - i]; // Sum X_{i+1}(k)*W_N^{i+1*k} + X_{radix-1-i}(k)*W_N^{(radix-1-i)*k} imag
                sum_r += tau_r;                                   // Add to X(k) real
                sum_i += tau_i;                                   // Add to X(k) imag
            }
            X0->re = sum_r; // X(k) real = \sum_{i=0}^{radix-1} X_i(k) * W_N^{i*k}
            X0->im = sum_i; // X(k) imag = \sum_{i=0}^{radix-1} X_i(k) * W_N^{i*k}

            for (int u = 0; u < mid_radix; u++)
            {
                fft_type temp1_r = y_r[0], temp1_i = y_i[0]; // Initialize with X_0(k)
                fft_type temp2_r = 0.0, temp2_i = 0.0;       // Initialize rotation terms
                for (int v = 0; v < mid_radix; v++)
                {
                    int temp = (u + 1) * (v + 1); // Compute index (u+1)(v+1) mod radix
                    while (temp >= radix)
                        temp -= radix;                                      // Reduce mod radix
                    int temp_temp = temp - 1;                               // Index for cos/sin tables
                    fft_type tau_r = y_r[v + 1] + y_r[radix - 1 - v];       // Sum X_{v+1}(k)*W_N^{v+1*k} + X_{radix-1-v}(k)*W_N^{(radix-1-v)*k} real
                    fft_type tau_i = y_i[v + 1] + y_i[radix - 1 - v];       // Sum X_{v+1}(k)*W_N^{v+1*k} + X_{radix-1-v}(k)*W_N^{(radix-1-v)*k} imag
                    fft_type tau_r_minus = y_r[v + 1] - y_r[radix - 1 - v]; // Diff X_{v+1}(k)*W_N^{v+1*k} - X_{radix-1-v}(k)*W_N^{(radix-1-v)*k} real
                    fft_type tau_i_minus = y_i[v + 1] - y_i[radix - 1 - v]; // Diff X_{v+1}(k)*W_N^{v+1*k} - X_{radix-1-v}(k)*W_N^{(radix-1-v)*k} imag
                    temp1_r += cos_values[temp_temp] * tau_r;               // Add cos((u+1)(v+1)*2π/radix) * sum real
                    temp1_i += cos_values[temp_temp] * tau_i;               // Add cos((u+1)(v+1)*2π/radix) * sum imag
                    temp2_r -= sin_values[temp_temp] * tau_r_minus;         // Subtract sin((u+1)(v+1)*2π/radix) * diff real
                    temp2_i -= sin_values[temp_temp] * tau_i_minus;         // Subtract sin((u+1)(v+1)*2π/radix) * diff imag
                }
                temp2_r = transform_sign * temp2_r;                                           // Apply sign for transform direction
                temp2_i = transform_sign * temp2_i;                                           // Apply sign for transform direction
                butterfly_outputs[(u + 1) * sub_fft_size + k].re = temp1_r - temp2_i;         // X(k + (u+1)N/radix) real
                butterfly_outputs[(u + 1) * sub_fft_size + k].im = temp1_i + temp2_r;         // X(k + (u+1)N/radix) imag
                butterfly_outputs[(radix - u - 1) * sub_fft_size + k].re = temp1_r + temp2_i; // X(k + (radix-u-1)N/radix) real
                butterfly_outputs[(radix - u - 1) * sub_fft_size + k].im = temp1_i - temp2_r; // X(k + (radix-u-1)N/radix) imag
            }
            free(y_r);
            free(y_i);
        }

        // Step 13: Copy butterfly results to output_buffer
        for (int lane = 0; lane < radix; lane++)
        {
            fft_data *base = output_buffer + lane * sub_fft_size;
            fft_data *src = butterfly_outputs + lane * sub_fft_size;
            for (int k = 0; k < sub_fft_size; k++)
            {
                base[k].re = src[k].re; // Copy X(lane*N/radix + k) real
                base[k].im = src[k].im; // Copy X(lane*N/radix + k) imag
            }
        }

        // Step 14: Free allocated memory
        free(child_offsets);
    }
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
static void bluestein_exp(fft_data *hl, fft_data *hlt, int input_length, int padded_length)
{
    if (input_length <= 0 || padded_length <= input_length)
    {
        fprintf(stderr, "Error: Invalid lengths for Bluestein’s exponential - input_length: %d, padded_length: %d\n",
                input_length, padded_length);
        // exit
    }

    int index;
    int found = 0;

    if (input_length < MAX_PRECOMPUTED_N)
    {
        for (int idx = 0; idx < num_precomputed; idx++)
        {
            if (chirp_sizes[idx] == input_length)
            {
                for (index = 0; index < input_length; index++)
                {
                    hlt[index] = bluestein_chirp[idx][index];
                    hl[index] = hlt[index];
                }
                found = 1;
                break;
            }
        }
    }

    if (!found)
    { // Fallback to dynamic computation if not precomputed or N >= MAX_PRECOMPUTED_N
        const fft_type PI = 3.1415926535897932384626433832795;
        fft_type theta = PI / input_length;
        int l2 = 0, len2 = 2 * input_length;
        for (index = 0; index < input_length; ++index)
        {
            fft_type angle = theta * l2;
            hlt[index].re = cos(angle);
            hlt[index].im = sin(angle);
            hl[index].re = hlt[index].re;
            hl[index].im = hlt[index].im;
            l2 += 2 * index + 1;
            while (l2 > len2)
                l2 -= len2;
        }
    }

    for (index = input_length; index < padded_length - input_length + 1; index++)
    {
        hl[index].re = 0.0;
        hl[index].im = 0.0;
    }

    for (index = padded_length - input_length + 1; index < padded_length; index++)
    {
        int mirrored_index = padded_length - index;
        hl[index].re = hlt[mirrored_index].re;
        hl[index].im = hlt[mirrored_index].im;
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
    // Quick check to make sure we’re not working with nonsense
    if (signal_length <= 0)
    {
        fprintf(stderr, "Error: Signal length (%d) is invalid\n", signal_length);
        // exit
    }

    // Figure out the padded length (M), which is the next power of 2 >= 2N-1
    // We pull this from max_scratch_size / 4, set up in fft_init to ensure we have enough room
    int padded_length = fft_config->max_scratch_size / 4;

    // Grab our scratch buffer—our one-stop shop for all temporary data
    // Why? FFTW’s approach of one big workspace cuts down on mallocs and keeps data cozy in cache
    fft_data *S = fft_config->scratch;

    // Split the scratch into chunks for our needs
    // - chirped_signal: holds the FFT of the chirp and later the padded signal (M elements)
    // - temp_chirp: temp storage for chirp and FFT inputs (M elements)
    // - ifft_result: stores IFFT output (M elements)
    // - chirp_sequence: the chirp sequence itself (N <= M elements)
    // Choice: Sequential layout keeps memory access tight; 4*M covers everything we need
    fft_data *chirped_signal = S;
    fft_data *temp_chirp = S + padded_length;
    fft_data *ifft_result = S + 2 * padded_length;
    fft_data *chirp_sequence = S + 3 * padded_length;

    // Double-check we’ve got enough scratch space
    // If not, bail out with a clear error—better than segfaulting later
    if (4 * padded_length > fft_config->max_scratch_size)
    {
        fprintf(stderr, "Error: Scratch buffer too small for Bluestein (need %d, have %d)\n",
                4 * padded_length, fft_config->max_scratch_size);
        // exit
    }

    // Set up a temporary FFT object for the padded length (M)
    // Why not reuse fft_config? Modifying it could mess things up in multi-threaded code,
    // so we play it safe with a fresh object, even if it’s a bit of extra work
    fft_object temp_config = fft_init(padded_length, transform_direction);
    if (!temp_config)
    {
        fprintf(stderr, "Error: Couldn’t create temporary FFT object\n");
        // exit
    }

    // Generate the chirp sequence (h(n) = e^{πi n^2 / N}) and store it
    // We use bluestein_exp to pull from our global `all_chirps` array for small N,
    // or compute it dynamically for larger N. Output goes into temp_chirp and chirp_sequence
    // Why global all_chirps? One big allocation for all chirps saves us from multiple mallocs
    // and keeps data close for better cache hits
    bluestein_exp(temp_chirp, chirp_sequence, signal_length, padded_length);

    // Scale the chirp by 1/M for normalization
    // Choice: Do this upfront to avoid scaling later, keeping the FFT steps clean
    fft_type scale = 1.0 / padded_length;
    for (int i = 0; i < padded_length; ++i)
    {
        temp_chirp[i].re *= scale;
        temp_chirp[i].im *= scale;
    }

    // Compute the FFT of the scaled chirp sequence
    // We store it in chirped_signal, which we’ll reuse later for the padded signal
    // Why reuse? Saves scratch space and simplifies buffer management
    fft_exec(temp_config, temp_chirp, chirped_signal);

    // Multiply the input signal by the chirp sequence (x(n) * h(n) for forward, x(n) * h^*(n) for inverse)
    // We’re going full AVX2 here to crunch 4 complex numbers at once
    // Note: chirp_sequence is aligned (from scratch), so we use aligned loads for speed
    int n = 0;
    for (; n + 3 < signal_length; n += 4)
    {
        // Load 4 input points (reversed order for _mm256_set_pd)
        __m256d input_re = _mm256_set_pd(input_signal[n + 3].re, input_signal[n + 2].re,
                                         input_signal[n + 1].re, input_signal[n + 0].re);
        __m256d input_im = _mm256_set_pd(input_signal[n + 3].im, input_signal[n + 2].im,
                                         input_signal[n + 1].im, input_signal[n + 0].im);
        // Load 4 chirp points (aligned, thanks to scratch allocation)
        __m256d chirp_re = _mm256_load_pd(&chirp_sequence[n].re);
        __m256d chirp_im = _mm256_load_pd(&chirp_sequence[n].im);

        // For inverse FFT, conjugate the chirp (negate imaginary part)
        // Why here? It’s cheaper than storing separate conjugated chirps
        if (transform_direction == -1)
        {
            chirp_im = _mm256_sub_pd(_mm256_setzero_pd(), chirp_im);
        }

        // Complex multiply: (a + bi)(c + di) = (ac - bd) + i(ad + bc)
        // Using FMA for better precision and speed on modern CPUs
        __m256d result_re = _mm256_fmsub_pd(input_re, chirp_re, _mm256_mul_pd(input_im, chirp_im));
        __m256d result_im = _mm256_fmadd_pd(input_im, chirp_re, _mm256_mul_pd(input_re, chirp_im));

        // Store results in temp_chirp (unaligned, as we can’t guarantee alignment)
        _mm256_storeu_pd(&temp_chirp[n].re, result_re);
        _mm256_storeu_pd(&temp_chirp[n].im, result_im);
    }
    // Handle any leftover points with scalar code
    // Choice: SIMD for the bulk, scalar for the tail—keeps it simple and robust
    for (; n < signal_length; ++n)
    {
        double input_re = input_signal[n].re, input_im = input_signal[n].im;
        double chirp_re = chirp_sequence[n].re, chirp_im = chirp_sequence[n].im;
        if (transform_direction == -1)
            chirp_im = -chirp_im;
        temp_chirp[n].re = input_re * chirp_re - input_im * chirp_im;
        temp_chirp[n].im = input_im * chirp_re + input_re * chirp_im;
    }

    // Zero-pad the chirped signal to length M
    // This sets up the convolution by ensuring no aliasing in the frequency domain
    // Choice: Explicit loop is clear and fast enough; could’ve used memset but this is more readable
    for (int i = signal_length; i < padded_length; ++i)
    {
        temp_chirp[i].re = 0.0;
        temp_chirp[i].im = 0.0;
    }

    // Run FFT on the padded chirped signal
    // We’re storing the result in ifft_result to keep things organized
    fft_exec(temp_config, temp_chirp, ifft_result);

    // Pointwise multiplication in the frequency domain (y(n) * h_k(n) for forward, y(n) * h_k^*(n) for inverse)
    // Again, AVX2 for speed, processing 4 complex points at a time
    n = 0;
    for (; n + 3 < padded_length; n += 4)
    {
        // Load FFT results and chirp FFT
        // Using aligned loads since ifft_result and chirped_signal are in our aligned scratch buffer
        __m256d fft_re = _mm256_load_pd(&ifft_result[n].re);
        __m256d fft_im = _mm256_load_pd(&ifft_result[n].im);
        __m256d chirp_fft_re = _mm256_load_pd(&chirped_signal[n].re);
        __m256d chirp_fft_im = _mm256_load_pd(&chirped_signal[n].im);

        // Conjugate chirp FFT for inverse transform
        // Doing it on-the-fly saves memory and keeps things flexible
        if (transform_direction == -1)
        {
            chirp_fft_im = _mm256_sub_pd(_mm256_setzero_pd(), chirp_fft_im);
        }

        // Complex multiply with FMA for efficiency
        __m256d result_re = _mm256_fmsub_pd(fft_re, chirp_fft_re, _mm256_mul_pd(fft_im, chirp_fft_im));
        __m256d result_im = _mm256_fmadd_pd(fft_im, chirp_fft_re, _mm256_mul_pd(fft_re, chirp_fft_im));

        // Store back in temp_chirp (unaligned store for safety)
        _mm256_storeu_pd(&temp_chirp[n].re, result_re);
        _mm256_storeu_pd(&temp_chirp[n].im, result_im);
    }
    // Scalar tail for any remaining points
    // Keeps the code robust for non-multiples of 4
    for (; n < padded_length; ++n)
    {
        double fft_re = ifft_result[n].re, fft_im = ifft_result[n].im;
        double chirp_fft_re = chirped_signal[n].re, chirp_fft_im = chirped_signal[n].im;
        if (transform_direction == -1)
            chirp_fft_im = -chirp_fft_im;
        temp_chirp[n].re = fft_re * chirp_fft_re - fft_im * chirp_fft_im;
        temp_chirp[n].im = fft_im * chirp_fft_re + fft_re * chirp_fft_im;
    }

    // Flip the twiddle factors for the inverse FFT
    // We’re modifying temp_config’s twiddles to avoid messing with the original fft_config
    for (int i = 0; i < padded_length; ++i)
    {
        temp_config->twiddles[i].im = -temp_config->twiddles[i].im;
    }
    temp_config->sgn = -transform_direction;

    // Run the inverse FFT to get the convolution result
    fft_exec(temp_config, temp_chirp, ifft_result);

    // Final step: multiply by the chirp sequence again to extract the DFT
    // Same AVX2 approach as before, with aligned loads for chirp_sequence
    n = 0;
    for (; n + 3 < signal_length; n += 4)
    {
        __m256d ifft_re = _mm256_load_pd(&ifft_result[n].re);
        __m256d ifft_im = _mm256_load_pd(&ifft_result[n].im);
        __m256d chirp_re = _mm256_load_pd(&chirp_sequence[n].re);
        __m256d chirp_im = _mm256_load_pd(&chirp_sequence[n].im);

        if (transform_direction == -1)
        {
            chirp_im = _mm256_sub_pd(_mm256_setzero_pd(), chirp_im);
        }

        __m256d result_re = _mm256_fmsub_pd(ifft_re, chirp_re, _mm256_mul_pd(ifft_im, chirp_im));
        __m256d result_im = _mm256_fmadd_pd(ifft_im, chirp_re, _mm256_mul_pd(ifft_re, chirp_im));

        // Store final results in output_signal (unaligned, as we don’t control its alignment)
        _mm256_storeu_pd(&output_signal[n].re, result_re);
        _mm256_storeu_pd(&output_signal[n].im, result_im);
    }
    // Scalar tail for the last few points
    for (; n < signal_length; ++n)
    {
        double ifft_re = ifft_result[n].re, ifft_im = ifft_result[n].im;
        double chirp_re = chirp_sequence[n].re, chirp_im = chirp_sequence[n].im;
        if (transform_direction == -1)
            chirp_im = -chirp_im;
        output_signal[n].re = ifft_re * chirp_re - ifft_im * chirp_im;
        output_signal[n].im = ifft_im * chirp_re + ifft_re * chirp_im;
    }

    // Clean up the temporary FFT object
    // Gotta free it since we created it, keeping memory tidy
    free_fft(temp_config);
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
