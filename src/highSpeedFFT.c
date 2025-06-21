#include "highspeedFFT.h"
#include "time.h"
#include <immintrin.h>

// Inlining macro for MSVC and GCC/Clang
#ifdef _MSC_VER
  #define ALWAYS_INLINE __forceinline
#else
  #define ALWAYS_INLINE __attribute__((always_inline))
#endif

// SIMD constant macro for butterfly operations
#define AVX_ONE _mm256_set1_pd(1.0)
#define SSE2_ONE _mm_set1_pd(1.0)   // SSE2: 128-bit vector of 1.0

// FMA macro definitions for portable AVX2/FMA support
#if defined(__FMA__) || defined(USE_FMA)
  #define FMADD(a,b,c) _mm256_fmadd_pd((a),(b),(c))
  #define FMSUB(a,b,c) _mm256_fmsub_pd((a),(b),(c))
#else
  static inline ALWAYS_INLINE __m256d fmadd_fallback(__m256d a, __m256d b, __m256d c) {
      // a*b + c
      return _mm256_add_pd(_mm256_mul_pd(a,b), c);
  }
  static inline ALWAYS_INLINE __m256d fmsub_fallback(__m256d a, __m256d b, __m256d c) {
      // a*b - c
      return _mm256_sub_pd(_mm256_mul_pd(a,b), c);
  }
  #define FMADD(a,b,c) fmadd_fallback((a),(b),(c))
  #define FMSUB(a,b,c) fmsub_fallback((a),(b),(c))
#endif

// SSE2 FMA fallbacks for radix-4 (128-bit vectors)
static inline ALWAYS_INLINE __m128d fmadd_sse2_fallback(__m128d a, __m128d b, __m128d c) {
    // a*b + c
    return _mm_add_pd(_mm_mul_pd(a,b), c);
}
static inline ALWAYS_INLINE __m128d fmsub_sse2_fallback(__m128d a, __m128d b, __m128d c) {
    // a*b - c
    return _mm_sub_pd(_mm_mul_pd(a,b), c);
}
#if defined(__FMA__) || defined(USE_FMA)
  #define FMADD_SSE2(a,b,c) _mm_fmadd_pd((a),(b),(c))
  #define FMSUB_SSE2(a,b,c) _mm_fmsub_pd((a),(b),(c))
#else
  #define FMADD_SSE2(a,b,c) fmadd_sse2_fallback((a),(b),(c))
  #define FMSUB_SSE2(a,b,c) fmsub_sse2_fallback((a),(b),(c))
#endif

// SIMD load/store macros
// AVX2 for radix-2, 7, 8 (256-bit vectors)
#ifdef USE_ALIGNED_SIMD
  #define LOADU_PD _mm256_load_pd
  #define STOREU_PD _mm256_store_pd
#else
  #define LOADU_PD _mm256_loadu_pd
  #define STOREU_PD _mm256_storeu_pd
#endif

// SSE2 for radix-4 (128-bit vectors)
#ifdef USE_ALIGNED_SIMD
  #define LOADU_SSE2 _mm_load_pd
  #define STOREU_SSE2 _mm_store_pd
#else
  #define LOADU_SSE2 _mm_loadu_pd
  #define STOREU_SSE2 _mm_storeu_pd
#endif

// File-scope scalar constants for radix-7
static const double C1 = 0.62348980185, C2 = -0.22252093395, C3 = -0.9009688679; // cos(51.43°), cos(102.86°), cos(154.29°)
static const double S1 = 0.78183148246, S2 = 0.97492791218, S3 = 0.43388373911; // sin(51.43°), sin(102.86°), sin(154.29°)

static const double C3_SQRT3BY2 = 0.8660254037844386; // √3/2 for 120° rotation

// File-scope constants for radix-5
static const double C5_1 = 0.30901699437;  // cos(72°)
static const double C5_2 = -0.80901699437; // cos(144°)
static const double S5_1 = 0.95105651629;  // sin(72°)
static const double S5_2 = 0.58778525229;  // sin(144°)

static const double C8_1 = 0.7071067811865476; // √2/2 for 45° rotation

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
__attribute__((constructor)) static void init_bluestein_chirp(void) {
    if (chirp_initialized) return;

    // Figure out how much space we need for all chirp sequences
    // We’re using pre_sizes[] = {1, 2, 3, 4, 5, 7, 15, 20, 31, 64}
    int total_chirp = 0;
    for (int i = 0; i < num_pre; i++) {
        total_chirp += ((pre_sizes[i] + 3) & ~3); // Round up to next multiple of 4
    }

    // Allocate our three arrays:
    // - bluestein_chirp: array of pointers to each chirp sequence
    // - chirp_sizes: stores the length of each chirp (matches pre_sizes)
    // - all_chirps: one big block for all chirp data
    bluestein_chirp = (fft_data **)malloc(num_pre * sizeof(fft_data *));
    chirp_sizes = (int *)malloc(num_pre * sizeof(int));
    all_chirps = (fft_data *)_mm_malloc(total_chirp * sizeof(fft_data), 32);
    if (!bluestein_chirp || !chirp_sizes || !all_chirps) {
        // If any allocation fails, clean up what we got and bail
        // Trade-off: Exiting is harsh, but it’s a startup error, so recovery is tricky
        fprintf(stderr, "Error: Memory allocation failed for Bluestein chirp table\n");
        _mm_free(all_chirps);
        free(bluestein_chirp);
        free(chirp_sizes);
        exit(EXIT_FAILURE);
    }

    // Set up the pointers and fill the chirp sequences
    // We walk through each size, point bluestein_chirp[i] to the right spot in all_chirps,
    // and compute the chirp values (e^{\pi i n^2 / N})
    int offset = 0;
    for (int idx = 0; idx < num_pre; idx++) {
        int n = pre_sizes[idx];
        chirp_sizes[idx] = n; // Store the size for lookup in bluestein_exp
        bluestein_chirp[idx] = all_chirps + offset; // Point to the current chunk
        offset += ((n + 3) & ~3); // Move offset to next aligned boundary

        // Compute the chirp sequence for this length
        // The angle is π n^2 / N, with a quadratic index (l2) to avoid floating-point drift
        // Why l2? It’s a clever trick to compute n^2 mod 2N without big numbers
        fft_type theta = M_PI / n;
        int l2 = 0, len2 = 2 * n;
        for (int i = 0; i < n; i++) {
            fft_type angle = theta * l2;
            bluestein_chirp[idx][i].re = cos(angle);
            bluestein_chirp[idx][i].im = sin(angle);
            l2 += 2 * i + 1; // Quadratic term: n^2 = (2i+1) mod 2N
            while (l2 > len2) l2 -= len2; // Wrap around to keep indices in bounds
        }
    }

    chirp_initialized = 1;
}

__attribute__((destructor)) static void cleanup_bluestein_chirp(void) {
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
static bool is_exact_power(int n, int p) {
    if (n <= 0 || p <= 1) return false;
    while (n % p == 0) n /= p;
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
fft_object fft_init(int signal_length, int transform_direction) {
    // Step 1: Validate inputs
    // Ensure signal length is positive and direction is +1 or -1
    // Invalid inputs are a user error, so we exit with a clear message
    if (signal_length <= 0 || (transform_direction != 1 && transform_direction != -1)) {
        fprintf(stderr, "Error: Signal length (%d) or direction (%d) is invalid\n",
                signal_length, transform_direction);
        return NULL;
    }

    // Step 2: Allocate fft_set structure
    // Allocate early to safely write stage_twiddle_offset in Step 5
    fft_object fft_config = (fft_object)malloc(sizeof(struct fft_set));
    if (!fft_config) {
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
    if (is_factorable) {
        max_padded_length = signal_length; // No padding for mixed-radix
        twiddle_count = signal_length;    // N twiddles for all stages
        // Check for pure powers using is_exact_power
        is_power_of_2 = (signal_length & (signal_length - 1)) == 0; // Fast power-of-2 check
        is_power_of_3 = is_exact_power(signal_length, 3);
        is_power_of_5 = is_exact_power(signal_length, 5);
        is_power_of_7 = is_exact_power(signal_length, 7);
        is_power_of_11 = is_exact_power(signal_length, 11);
        is_power_of_13 = is_exact_power(signal_length, 13);
    } else {
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
    if (is_factorable) {
        int temp_N = signal_length;
        if (is_power_of_2 || is_power_of_3 || is_power_of_5 || is_power_of_7 ||
            is_power_of_11 || is_power_of_13) {
            // For pure-power FFTs, sum needs per recursion level
            // Radix-r needs (r-1)*(N/r) twiddles, r*(N/r) scratch
            int radix = is_power_of_2 ? 2 : is_power_of_3 ? 3 : is_power_of_5 ? 5 :
                        is_power_of_7 ? 7 : is_power_of_11 ? 11 : 13;
            int stage = 0;
            for (int n = signal_length; n >= radix; n /= radix) {
                int sub_fft_size = n / radix;
                // Store stage offset for twiddle_factors
                if (stage < MAX_STAGES) {
                    fft_config->stage_twiddle_offset[stage++] = twiddle_factors_size;
                } else {
                    fprintf(stderr, "Error: Exceeded MAX_STAGES (%d) for N=%d, radix=%d\n",
                            MAX_STAGES, signal_length, radix);
                    free_fft(fft_config);
                    return NULL;
                }
                twiddle_factors_size += (radix - 1) * sub_fft_size; // W_n^{j*k}, j=1..r-1
                scratch_needed += radix * sub_fft_size;              // Outputs
            }
            fft_config->num_precomputed_stages = stage; // Record number of stages
        } else {
            // Mixed-radix: r*(N/r) outputs, (r-1)*(N/r) twiddles for radices ≤ 13
            for (int i = 0; i < num_factors; i++) {
                int radix = temp_factors[i];
                scratch_needed += radix * (temp_N / radix); // Outputs
                if (radix <= 13) {
                    scratch_needed += (radix - 1) * (temp_N / radix); // Twiddles
                }
                temp_N /= radix;
            }
        }
        // Ensure scratch size covers worst-case, fallback to 4*N
        // Note: Mixed-radix may need more scratch for twiddles in radix-11/13
        max_scratch_size = scratch_needed;
        if (max_scratch_size < 4 * signal_length) {
            max_scratch_size = 4 * signal_length;
        }
    } else {
        // Bluestein: 4*M for chirped_signal, temp_chirp, ifft_result, chirp_sequence
        max_scratch_size = 4 * max_padded_length;
    }

    // Step 6: Allocate twiddle and scratch buffers
    // 32-byte aligned for AVX2/SSE2 SIMD performance
    fft_config->twiddles = (fft_data *)_mm_malloc(twiddle_count * sizeof(fft_data), 32);
    fft_config->scratch = (fft_data *)_mm_malloc(max_scratch_size * sizeof(fft_data), 32);
    fft_config->twiddle_factors = NULL;

    // Check allocation failures and clean up
    if (!fft_config->twiddles || !fft_config->scratch) {
        fprintf(stderr, "Error: Failed to allocate twiddle or scratch buffers\n");
        free_fft(fft_config);
        return NULL;
    }

    // Step 7: Allocate twiddle_factors for pure-power FFTs
    // Precompute twiddles to skip copying in mixed_radix_dit_rec
    if (is_factorable && (is_power_of_2 || is_power_of_3 || is_power_of_5 || is_power_of_7 ||
                          is_power_of_11 || is_power_of_13)) {
        fft_config->twiddle_factors = (fft_data *)_mm_malloc(twiddle_factors_size * sizeof(fft_data), 32);
        if (!fft_config->twiddle_factors) {
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
    if (fft_config->twiddle_factors) {
        int offset = 0;
        int radix = is_power_of_2 ? 2 : is_power_of_3 ? 3 : is_power_of_5 ? 5 :
                    is_power_of_7 ? 7 : is_power_of_11 ? 11 : 13;
        for (int n = signal_length; n >= radix; n /= radix) {
            int sub_fft_size = n / radix;
            for (int j = 1; j < radix; j++) {
                for (int k = 0; k < sub_fft_size; k++) {
                    int idx = sub_fft_size - 1 + j * k; // W_n^{j*k}
                    fft_config->twiddle_factors[offset + (j-1) * sub_fft_size + k].re = fft_config->twiddles[idx].re;
                    fft_config->twiddle_factors[offset + (j-1) * sub_fft_size + k].im = fft_config->twiddles[idx].im;
                }
            }
            offset += (radix - 1) * sub_fft_size;
        }
    }

    // Step 12: Adjust twiddles for inverse FFT
    // Flip imaginary parts for e^{+2πi k / N}
    if (transform_direction == -1) {
        for (int i = 0; i < twiddle_count; i++) {
            fft_config->twiddles[i].im = -fft_config->twiddles[i].im;
        }
        if (fft_config->twiddle_factors) {
            for (int i = 0; i < twiddle_factors_size; i++) {
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
        exit(EXIT_FAILURE);
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
    else if (radix == 2) {
    /**
     * @brief Radix-2 decomposition for two-point sub-FFTs with AVX2 vectorization and FMA support.
     *
     * Intention: Efficiently compute the FFT for data lengths divisible by 2 by splitting into
     * two sub-FFTs (even and odd indices), applying recursive FFTs, and combining results with
     * twiddle factors. Optimized for power-of-2 (N=2^r) and mixed-radix FFTs using pre-allocated
     * scratch and stage-specific twiddle offsets. Follows FFTW’s two-buffer strategy for thread
     * safety and performance.
     *
     * Mathematically: Computes:
     *   X(k) = X_even(k) + W_N^k * X_odd(k),
     *   X(k+N/2) = X_even(k) - W_N^k * X_odd(k),
     * where X_even, X_odd are sub-FFTs of size N/2, W_N^k = e^{-2πi k / N}.
     *
     * Process:
     * 1. Divide data into two sub-FFTs of size N/2, doubling stride.
     * 2. Validate scratch size: 2*(N/2) for outputs (power-of-2), 3*(N/2) for outputs + twiddles (mixed-radix).
     * 3. Assign scratch slices for sub-FFT outputs and twiddles (mixed-radix only).
     * 4. Compute child scratch offsets to prevent overlap, using parent’s allocated scratch.
     * 5. Recursively compute even and odd sub-FFTs.
     * 6. Load twiddle factors from stage_twiddle_offset[factor_index] (power-of-2) or copy to scratch (mixed-radix).
     * 7. Perform butterfly operations with AVX2 for k divisible by 4, scalar tail.
     * 8. Copy results to output_buffer.
     *
     * Optimization:
     * - AVX2 processes 4 complex points per iteration, using aligned loads/stores when USE_ALIGNED_SIMD is defined.
     * - FMA (FMADD/FMSUB) computes (a*b ± c) for accuracy and speed, enabled by __FMA__ or USE_FMA.
     * - Pre-allocated scratch eliminates malloc/free overhead.
     * - stage_twiddle_offset skips ad-hoc indexing for power-of-2 FFTs.
     * - Scratch and twiddle_factors are 32-byte aligned, supporting aligned SIMD.
     *
     * @warning Assumes fft_obj->twiddles has n_fft ≥ N elements, scratch is 32-byte aligned,
     *          and factor_index is valid for twiddle_factors.
     */
    // Step 1: Compute subproblem size and stride
    int sub_fft_length = data_length / 2; // Size of each sub-FFT (N/2)
    int next_stride = 2 * stride;         // Double stride for next level
    int sub_fft_size = sub_fft_length;    // Sub-FFT size for indexing

    // Step 2: Validate scratch buffer
    // Power-of-2: 2*(N/2) for outputs; mixed-radix: (2+1)*(N/2) = 3*(N/2) for outputs + twiddles
    int required_size = fft_obj->twiddle_factors != NULL ? 2 * sub_fft_size : 3 * sub_fft_size;
    if (scratch_offset + required_size > fft_obj->max_scratch_size) {
        fprintf(stderr, "Error: Scratch buffer too small for radix-2 at offset %d (need %d, have %d)\n",
                scratch_offset, required_size, fft_obj->max_scratch_size - scratch_offset);
        exit(EXIT_FAILURE);
    }

    // Step 3: Assign scratch slices
    // sub_fft_outputs: even/odd sub-FFT results (2*sub_fft_size fft_data)
    // twiddle_factors: W_N^k (sub_fft_size fft_data, mixed-radix only)
    fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
    fft_data *twiddle_factors;
    if (fft_obj->twiddle_factors != NULL) {
        // Validate factor_index for power-of-2 FFTs
        if (factor_index >= fft_obj->num_precomputed_stages) {
            fprintf(stderr, "Error: Invalid factor_index (%d) exceeds num_precomputed_stages (%d) for radix-2\n",
                    factor_index, fft_obj->num_precomputed_stages);
            exit(EXIT_FAILURE);
        }
        twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
    } else {
        twiddle_factors = fft_obj->scratch + scratch_offset + 2 * sub_fft_size;
    }

    // Step 4: Compute offsets for child recursions
    // Each child needs 2*(N/4) (power-of-2) or 3*(N/4) (mixed-radix)
    // Use parent’s required_size to ensure non-overlapping regions
    int child_scratch_per_branch = fft_obj->twiddle_factors != NULL ? 2 * (sub_fft_size / 2) : 3 * (sub_fft_size / 2);
    if (child_scratch_per_branch * 2 > required_size) {
        fprintf(stderr, "Error: Child scratch size (%d) exceeds parent allocation (%d) for radix-2\n",
                child_scratch_per_branch * 2, required_size);
        exit(EXIT_FAILURE);
    }
    int child_offset1 = scratch_offset;                    // First child starts at parent offset
    int child_offset2 = scratch_offset + required_size / 2; // Second child uses second half

    // Step 5: Recurse on even and odd indices
    // Use distinct scratch offsets to prevent overlap
    mixed_radix_dit_rec(sub_fft_outputs, input_buffer, fft_obj, transform_sign, sub_fft_length,
                        next_stride, factor_index + 1, child_offset1); // Even indices
    mixed_radix_dit_rec(sub_fft_outputs + sub_fft_size, input_buffer + stride, fft_obj, transform_sign,
                        sub_fft_length, next_stride, factor_index + 1, child_offset2); // Odd indices

    // Step 6: Prepare twiddle factors (mixed-radix only)
    // For power-of-2 FFTs, twiddle_factors points to precomputed W_N^k via stage_twiddle_offset
    if (fft_obj->twiddle_factors == NULL) {
        if (fft_obj->n_fft < 2 * sub_fft_size) {
            fprintf(stderr, "Error: Twiddle array too small (need at least %d elements, have %d)\n",
                    2 * sub_fft_size, fft_obj->n_fft);
            exit(EXIT_FAILURE);
        }
        for (int k = 0; k < sub_fft_size; k++) {
            int idx = sub_fft_length - 1 + k; // Index for W_N^k
            twiddle_factors[k].re = fft_obj->twiddles[idx].re;
            twiddle_factors[k].im = fft_obj->twiddles[idx].im;
        }
    }

    // Step 7: Run AVX2 butterfly operations
    // Combine: X(k) = even + W_N^k * odd, X(k+N/2) = even - W_N^k * odd
    // Scratch and twiddle_factors are 32-byte aligned (via _mm_malloc), enabling aligned loads/stores with USE_ALIGNED_SIMD
    int k = 0;
    for (; k + 3 < sub_fft_size; k += 4) {
        // Load even/odd sub-FFT points (32-byte aligned with USE_ALIGNED_SIMD)
        __m256d even_re = LOADU_PD(&sub_fft_outputs[k].re);     // Even real parts
        __m256d even_im = LOADU_PD(&sub_fft_outputs[k].im);     // Even imag parts
        __m256d odd_re = LOADU_PD(&sub_fft_outputs[k + sub_fft_size].re); // Odd real
        __m256d odd_im = LOADU_PD(&sub_fft_outputs[k + sub_fft_size].im); // Odd imag
        // Load twiddle factors (aligned for power-of-2, scratch for mixed-radix)
        __m256d twiddle_re = LOADU_PD(&twiddle_factors[k].re);   // W_N^k real
        __m256d twiddle_im = LOADU_PD(&twiddle_factors[k].im);   // W_N^k imag

        // Twiddle multiply: (odd_re + i*odd_im) * (twiddle_re + i*twiddle_im)
        // FMSUB(a, b, c) computes a*b - c, FMADD(a, b, c) computes a*b + c
        __m256d twiddled_re = FMSUB(odd_re, twiddle_re, _mm256_mul_pd(odd_im, twiddle_im)); // odd_re * twiddle_re - odd_im * twiddle_im
        __m256d twiddled_im = FMADD(odd_im, twiddle_re, _mm256_mul_pd(odd_re, twiddle_im)); // odd_im * twiddle_re + odd_re * twiddle_im

        // Butterfly operations
        STOREU_PD(&sub_fft_outputs[k].re, FMADD(twiddled_re, AVX_ONE, even_re)); // X(k) real
        STOREU_PD(&sub_fft_outputs[k].im, FMADD(twiddled_im, AVX_ONE, even_im)); // X(k) imag
        STOREU_PD(&sub_fft_outputs[k + sub_fft_size].re, FMSUB(even_re, AVX_ONE, twiddled_re)); // X(k+N/2) real
        STOREU_PD(&sub_fft_outputs[k + sub_fft_size].im, FMSUB(even_im, AVX_ONE, twiddled_im)); // X(k+N/2) imag
    }

    // Scalar tail: Handle remaining points
    for (; k < sub_fft_size; ++k) {
        double even_re = sub_fft_outputs[k].re, even_im = sub_fft_outputs[k].im;           // Even point
        double odd_re = sub_fft_outputs[k + sub_fft_size].re, odd_im = sub_fft_outputs[k + sub_fft_size].im; // Odd point
        // Twiddle multiply
        double twiddled_re = odd_re * twiddle_factors[k].re - odd_im * twiddle_factors[k].im;   // Real part
        double twiddled_im = odd_im * twiddle_factors[k].re + odd_re * twiddle_factors[k].im;   // Imag part
        // Butterfly
        sub_fft_outputs[k].re = even_re + twiddled_re;                                 // X(k) real
        sub_fft_outputs[k].im = even_im + twiddled_im;                                 // X(k) imag
        sub_fft_outputs[k + sub_fft_size].re = even_re - twiddled_re;                  // X(k+N/2) real
        sub_fft_outputs[k + sub_fft_size].im = even_im - twiddled_im;                  // X(k+N/2) imag
    }

    // Step 8: Copy results to output_buffer
    // Move X(k) and X(k+N/2) to final positions
    for (int lane = 0; lane < 2; lane++) {
        fft_data *base = output_buffer + lane * sub_fft_size;
        for (int k = 0; k < sub_fft_size; k++) {
            base[k].re = sub_fft_outputs[lane * sub_fft_size + k].re;
            base[k].im = sub_fft_outputs[lane * sub_fft_size + k].im;
        }
    }
    }
    else if (radix == 3) {
    /**
     * @brief Radix-3 decomposition for three-point sub-FFTs with SSE2 vectorization and FMA support.
     *
     * Intention: Efficiently compute the FFT for data lengths divisible by 3 by splitting into
     * three sub-FFTs (indices n mod 3 = 0, 1, 2), applying recursive FFTs, and combining results
     * with twiddle factors. Optimized for power-of-3 (N=3^r) and mixed-radix FFTs using
     * pre-allocated scratch and stage-specific twiddle offsets. Follows FFTW’s two-buffer strategy
     * for thread safety and performance.
     *
     * Mathematically: Computes:
     *   X(k) = X_0(k) + W_N^k * X_1(k) + W_N^{2k} * X_2(k),
     * where X_0, X_1, X_2 are sub-FFTs of size N/3, W_N^k = e^{-2πi k / N}.
     * The butterfly uses 120° and 240° roots of unity (√3/2 for rotation).
     *
     * Process:
     * 1. Divide data into three sub-FFTs of size N/3, tripling stride.
     * 2. Validate scratch size: 3*(N/3) for outputs (power-of-3), 5*(N/3) for outputs + twiddles (mixed-radix).
     * 3. Assign scratch slices for sub-FFT outputs and twiddles (mixed-radix only).
     * 4. Compute child scratch offsets to prevent overlap, using parent’s allocated scratch.
     * 5. Recursively compute sub-FFTs for indices n mod 3 = 0, 1, 2.
     * 6. Load twiddle factors from stage_twiddle_offset[factor_index] (power-of-3) or copy to scratch (mixed-radix).
     * 7. Flatten sub-FFT outputs into contiguous real/imag arrays.
     * 8. Perform butterfly operations with SSE2 for k divisible by 2, scalar tail.
     * 9. Copy results to output_buffer.
     *
     * Optimization:
     * - SSE2 processes 2 complex points per iteration, using aligned loads/stores when USE_ALIGNED_SIMD is defined.
     * - FMA (FMADD_SSE2/FMSUB_SSE2) computes (a*b ± c) for accuracy and speed, enabled by __FMA__ or USE_FMA.
     * - Pre-allocated scratch eliminates malloc/free overhead.
     * - stage_twiddle_offset skips ad-hoc indexing for power-of-3 FFTs.
     * - Scratch and twiddle_factors are 32-byte aligned, supporting aligned SIMD.
     *
     * @warning Assumes fft_obj->twiddles has n_fft ≥ N elements, scratch is 32-byte aligned,
     *          and factor_index is valid for twiddle_factors.
     */
    // Step 1: Compute subproblem size and stride
    int sub_fft_length = data_length / 3; // Size of each sub-FFT (N/3)
    int next_stride = 3 * stride;         // Triple stride for next level
    int sub_fft_size = sub_fft_length;    // Sub-FFT size for indexing

    // Step 2: Validate scratch buffer
    // Power-of-3: 3*(N/3) for outputs; mixed-radix: (3+2)*(N/3) = 5*(N/3) for outputs + twiddles
    int required_size = fft_obj->twiddle_factors != NULL ? 3 * sub_fft_size : 5 * sub_fft_size;
    if (scratch_offset + required_size > fft_obj->max_scratch_size) {
        fprintf(stderr, "Error: Scratch buffer too small for radix-3 at offset %d (need %d, have %d)\n",
                scratch_offset, required_size, fft_obj->max_scratch_size - scratch_offset);
        exit(EXIT_FAILURE);
    }

    // Step 3: Assign scratch slices
    // sub_fft_outputs: X_0, X_1, X_2 (3*sub_fft_size fft_data)
    // twiddle_factors: W_N^k, W_N^{2k} (2*sub_fft_size fft_data, mixed-radix only)
    fft_data *sub_fft_outputs = fft_obj->scratch + scratch_offset;
    fft_data *twiddle_factors;
    if (fft_obj->twiddle_factors != NULL) {
        // Validate factor_index for power-of-3 FFTs
        if (factor_index >= fft_obj->num_precomputed_stages) {
            fprintf(stderr, "Error: Invalid factor_index (%d) exceeds num_precomputed_stages (%d) for radix-3\n",
                    factor_index, fft_obj->num_precomputed_stages);
            exit(EXIT_FAILURE);
        }
        twiddle_factors = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
    } else {
        twiddle_factors = fft_obj->scratch + scratch_offset + 3 * sub_fft_size;
    }

    // Step 4: Compute child scratch offsets
    // Each child needs 3*(N/9) (power-of-3) or 5*(N/9) (mixed-radix), tiled within parent block
    // Use parent’s required_size to ensure non-overlapping regions
    int child_scratch_per_branch = fft_obj->twiddle_factors != NULL ? 3 * (sub_fft_size / 3) : 5 * (sub_fft_size / 3);
    if (child_scratch_per_branch * 3 > required_size) {
        fprintf(stderr, "Error: Child scratch size (%d) exceeds parent allocation (%d) for radix-3\n",
                child_scratch_per_branch * 3, required_size);
        exit(EXIT_FAILURE);
    }
    int child_offset1 = scratch_offset;                    // First child starts at parent offset
    int child_offset2 = scratch_offset + required_size / 3; // Second child uses second third
    int child_offset3 = scratch_offset + 2 * required_size / 3; // Third child uses final third

    // Step 5: Recurse on three sub-FFTs
    // Use distinct scratch offsets to prevent overlap
    for (int i = 0; i < 3; i++) {
        mixed_radix_dit_rec(sub_fft_outputs + i * sub_fft_size, input_buffer + i * stride, fft_obj,
                            transform_sign, sub_fft_length, next_stride, factor_index + 1,
                            scratch_offset + i * required_size / 3);
    }

    // Step 6: Prepare twiddle factors (mixed-radix only)
    // For power-of-3 FFTs, twiddle_factors points to precomputed W_N^k, W_N^{2k} via stage_twiddle_offset
    if (fft_obj->twiddle_factors == NULL) {
        if (fft_obj->n_fft < 3 * sub_fft_size) {
            fprintf(stderr, "Error: Twiddle array too small (need at least %d elements, have %d)\n",
                    3 * sub_fft_size, fft_obj->n_fft);
            exit(EXIT_FAILURE);
        }
        for (int k = 0; k < sub_fft_size; k++) {
            int idx = sub_fft_length - 1 + 2 * k; // Index for W_N^k, W_N^{2k}
            twiddle_factors[2 * k + 0].re = fft_obj->twiddles[idx + 0].re; // W_N^k real
            twiddle_factors[2 * k + 0].im = fft_obj->twiddles[idx + 0].im; // W_N^k imag
            twiddle_factors[2 * k + 1].re = fft_obj->twiddles[idx + 1].re; // W_N^{2k} real
            twiddle_factors[2 * k + 1].im = fft_obj->twiddles[idx + 1].im; // W_N^{2k} imag
        }
    }

    // Step 7: Flatten outputs into contiguous arrays in scratch
    // Use sub_fft_outputs for 3*sub_fft_size fft_data (6*sub_fft_size doubles)
    // out_re: first 3*sub_fft_size doubles; out_im: next 3*sub_fft_size doubles
    // Scratch is 32-byte aligned, ensuring safe aligned loads/stores with USE_ALIGNED_SIMD
    double *out_re = (double *)sub_fft_outputs;
    double *out_im = (double *)sub_fft_outputs + 3 * sub_fft_size;
    for (int lane = 0; lane < 3; lane++) {
        fft_data *base = sub_fft_outputs + lane * sub_fft_size;
        for (int k = 0; k < sub_fft_size; k++) {
            out_re[lane * sub_fft_size + k] = base[k].re;
            out_im[lane * sub_fft_size + k] = base[k].im;
        }
    }

    // Step 8: SSE2 vectorized butterfly for k=0 to sub_fft_size-1
    // Combine: X(k), X(k+N/3), X(k+2N/3) with 120°/240° rotations
    // Use aligned loads/stores with USE_ALIGNED_SIMD since scratch and twiddle_factors are 32-byte aligned
    __m128d v_sign_factor = _mm_set1_pd((double)transform_sign); // ±1 for transform direction
    __m128d vhalf = _mm_set1_pd(0.5);                           // Constant for -½(b + c)
    __m128d vsqrt3by2 = _mm_set1_pd(C3_SQRT3BY2);               // √3/2 for 120° rotation
    int k = 0;
    for (; k + 1 < sub_fft_size; k += 2) {
        // Load three sub-FFT points for two k values (32-byte aligned with USE_ALIGNED_SIMD)
        __m128d a_r = LOADU_SSE2(out_re + k);                     // X_0[k:k+1] real
        __m128d a_i = LOADU_SSE2(out_im + k);                     // X_0[k:k+1] imag
        __m128d b_r = LOADU_SSE2(out_re + k + sub_fft_size);      // X_1[k:k+1] real
        __m128d b_i = LOADU_SSE2(out_im + k + sub_fft_size);      // X_1[k:k+1] imag
        __m128d c_r = LOADU_SSE2(out_re + k + 2 * sub_fft_size);  // X_2[k:k+1] real
        __m128d c_i = LOADU_SSE2(out_im + k + 2 * sub_fft_size);  // X_2[k:k+1] imag

        // Load twiddle factors (aligned for power-of-3, scratch for mixed-radix)
        int idx = 2 * k; // Base index for W_N^k, W_N^{2k}
        __m128d w1r = LOADU_SSE2(&twiddle_factors[idx + 0].re);   // W_N^k real
        __m128d w1i = LOADU_SSE2(&twiddle_factors[idx + 0].im);   // W_N^k imag
        __m128d w2r = LOADU_SSE2(&twiddle_factors[idx + 1].re);   // W_N^{2k} real
        __m128d w2i = LOADU_SSE2(&twiddle_factors[idx + 1].im);   // W_N^{2k} imag

        // Apply twiddle factors: (b/c) * W_N^k
        // FMSUB_SSE2(a, b, c) computes a*b - c, FMADD_SSE2(a, b, c) computes a*b + c
        __m128d b2r = FMSUB_SSE2(b_r, w1r, _mm_mul_pd(b_i, w1i)); // b_r * w1r - b_i * w1i
        __m128d b2i = FMADD_SSE2(b_i, w1r, _mm_mul_pd(b_r, w1i)); // b_i * w1r + b_r * w1i
        __m128d c2r = FMSUB_SSE2(c_r, w2r, _mm_mul_pd(c_i, w2i)); // c_r * w2r - c_i * w2i
        __m128d c2i = FMADD_SSE2(c_i, w2r, _mm_mul_pd(c_r, w2i)); // c_i * w2r + c_r * w2i

        // Compute sums and differences
        __m128d sum_r = _mm_add_pd(b2r, c2r); // b + c real
        __m128d sum_i = _mm_add_pd(b2i, c2i); // b + c imag
        __m128d diff_r = _mm_sub_pd(b2r, c2r); // b - c real
        __m128d diff_i = _mm_sub_pd(b2i, c2i); // b - c imag

        // X_0 = a + (b + c)
        __m128d x0r = _mm_add_pd(a_r, sum_r); // X(k) real
        __m128d x0i = _mm_add_pd(a_i, sum_i); // X(k) imag
        STOREU_SSE2(out_re + k, x0r);
        STOREU_SSE2(out_im + k, x0i);

        // X_1 = a - ½(b + c) + i * (sign * (√3/2) * (b - c))
        __m128d t_r = _mm_sub_pd(a_r, _mm_mul_pd(sum_r, vhalf)); // a - ½(b + c) real
        __m128d t_i = _mm_sub_pd(a_i, _mm_mul_pd(sum_i, vhalf)); // a - ½(b + c) imag
        __m128d rot_r = _mm_mul_pd(diff_i, _mm_mul_pd(v_sign_factor, vsqrt3by2)); // sign * (√3/2) * (b - c) real
        __m128d rot_i = _mm_mul_pd(_mm_sub_pd(_mm_setzero_pd(), diff_r), _mm_mul_pd(v_sign_factor, vsqrt3by2)); // -sign * (√3/2) * (b - c) imag
        __m128d x1r = _mm_add_pd(t_r, rot_r); // X(k + N/3) real
        __m128d x1i = _mm_add_pd(t_i, rot_i); // X(k + N/3) imag
        STOREU_SSE2(out_re + k + sub_fft_size, x1r);
        STOREU_SSE2(out_im + k + sub_fft_size, x1i);

        // X_2 = a - ½(b + c) - i * (sign * (√3/2) * (b - c))
        __m128d x2r = _mm_sub_pd(t_r, rot_r); // X(k + 2N/3) real
        __m128d x2i = _mm_sub_pd(t_i, rot_i); // X(k + 2N/3) imag
        STOREU_SSE2(out_re + k + 2 * sub_fft_size, x2r);
        STOREU_SSE2(out_im + k + 2 * sub_fft_size, x2i);
    }

    // Step 9: Scalar tail for remaining k
    // Handle any k not covered by SSE2 loop (e.g., sub_fft_size odd)
    for (; k < sub_fft_size; k++) {
        // Load twiddle factors
        fft_type w1r = twiddle_factors[2 * k + 0].re; // W_N^k real
        fft_type w1i = twiddle_factors[2 * k + 0].im; // W_N^k imag
        fft_type w2r = twiddle_factors[2 * k + 1].re; // W_N^{2k} real
        fft_type w2i = twiddle_factors[2 * k + 1].im; // W_N^{2k} imag

        // Load sub-FFT points
        fft_type a_re = out_re[k], a_im = out_im[k];                     // X_0
        fft_type b_re = out_re[k + sub_fft_size], b_im = out_im[k + sub_fft_size]; // X_1
        fft_type c_re = out_re[k + 2 * sub_fft_size], c_im = out_im[k + 2 * sub_fft_size]; // X_2

        // Apply twiddle factors
        fft_type b2_re = b_re * w1r - b_im * w1i; // W_N^k * X_1 real
        fft_type b2_im = b_im * w1r + b_re * w1i; // W_N^k * X_1 imag
        fft_type c2_re = c_re * w2r - c_im * w2i; // W_N^{2k} * X_2 real
        fft_type c2_im = c_im * w2r + c_re * w2i; // W_N^{2k} * X_2 imag

        // Compute sums and differences
        fft_type sum_re = b2_re + c2_re; // b + c real
        fft_type sum_im = b2_im + c2_im; // b + c imag
        fft_type diff_re = b2_re - c2_re; // b - c real
        fft_type diff_im = b2_im - c2_im; // b - c imag

        // X_0 = a + (b + c)
        out_re[k] = a_re + sum_re; // X(k) real
        out_im[k] = a_im + sum_im; // X(k) imag

        // X_1 = a - ½(b + c) + i * (sign * (√3/2) * (b - c))
        fft_type t_re = a_re - (sum_re * 0.5); // a - ½(b + c) real
        fft_type t_im = a_im - (sum_im * 0.5); // a - ½(b + c) imag
        fft_type rot_re = diff_im * (transform_sign * C3_SQRT3BY2); // sign * (√3/2) * (b - c) real
        fft_type rot_im = -diff_re * (transform_sign * C3_SQRT3BY2); // -sign * (√3/2) * (b - c) imag
        out_re[k + sub_fft_size] = t_re + rot_re; // X(k + N/3) real
        out_im[k + sub_fft_size] = t_im + rot_im; // X(k + N/3) imag

        // X_2 = a - ½(b + c) - i * (sign * (√3/2) * (b - c))
        out_re[k + 2 * sub_fft_size] = t_re - rot_re; // X(k + 2N/3) real
        out_im[k + 2 * sub_fft_size] = t_im - rot_im; // X(k + 2N/3) imag
    }

    // Step 10: Copy results back to output_buffer
    // Move X(k), X(k+N/3), X(k+2N/3) to final positions
    for (int lane = 0; lane < 3; lane++) {
        fft_data *base = output_buffer + lane * sub_fft_size;
        for (int k = 0; k < sub_fft_size; k++) {
            base[k].re = out_re[lane * sub_fft_size + k];
            base[k].im = out_im[lane * sub_fft_size + k];
        }
    }
}
    else if (radix == 4)
    {
    /**
     * @brief Radix-4 decomposition for four-point sub-FFTs with SSE2 vectorization and FMA support.
     *
     * Intention: Optimize FFT computation for data lengths divisible by 4 by splitting into
     * four sub-FFTs, corresponding to indices n mod 4 = 0, 1, 2, 3, and combining results
     * with twiddle factors. This is efficient for N=4^r or mixed-radix cases, using SSE2 for
     * broader compatibility compared to AVX2.
     *
     * Mathematically: The FFT is computed as:
     *   \( X(k) = X_0(k) + W_N^k \cdot X_1(k) + W_N^{2k} \cdot X_2(k) + W_N^{3k} \cdot X_3(k) \),
     * where \( X_0, X_1, X_2, X_3 \) are sub-FFTs of size N/4, and \( W_N^k = e^{-2\pi i k / N} \).
     * The radix-4 butterfly uses 90°, 180°, and 270° rotations.
     *
     */
    // Step 1: Compute subproblem size and new stride
    int sub_length = data_length / 4; // Size of each sub-FFT (N/4)
    int new_stride = 4 * stride;      // Stride quadruples to access every fourth element

    // Step 2: Recurse on the four decimated sub-FFTs
    mixed_radix_dit_rec(output_buffer, input_buffer, fft_obj, transform_sign, sub_length,
                        new_stride, factor_index + 1); // n mod 4 = 0
    mixed_radix_dit_rec(output_buffer + sub_length, input_buffer + stride, fft_obj,
                        transform_sign, sub_length, new_stride, factor_index + 1); // n mod 4 = 1
    mixed_radix_dit_rec(output_buffer + 2 * sub_length, input_buffer + 2 * stride, fft_obj,
                        transform_sign, sub_length, new_stride, factor_index + 1); // n mod 4 = 2
    mixed_radix_dit_rec(output_buffer + 3 * sub_length, input_buffer + 3 * stride, fft_obj,
                        transform_sign, sub_length, new_stride, factor_index + 1); // n mod 4 = 3

    // Step 3: Flatten twiddle factors into contiguous arrays for SSE2
    double *tw_re_contig = malloc(3 * sub_length * sizeof(double)); // Contiguous real twiddles
    double *tw_im_contig = malloc(3 * sub_length * sizeof(double)); // Contiguous imag twiddles
    if (!tw_re_contig || !tw_im_contig)
    {
        fprintf(stderr, "Error: Memory allocation failed for twiddle arrays\n");
        free(tw_re_contig);
        free(tw_im_contig);
        exit(EXIT_FAILURE);
    }
    if (fft_obj->N - 1 < sub_length - 1 + 3 * (sub_length - 1))
    {
        fprintf(stderr, "Error: Twiddle array too small (need %d elements, have %d)\n",
                sub_length - 1 + 3 * (sub_length - 1), fft_obj->N - 1);
        free(tw_re_contig);
        free(tw_im_contig);
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k < sub_length; k++)
    {
        int idx = (sub_length - 1) + 3 * k; // Base index for W_N^k, W_N^{2k}, W_N^{3k}
        tw_re_contig[3 * k + 0] = fft_obj->twiddle[idx + 0].re; // W_N^k real
        tw_im_contig[3 * k + 0] = fft_obj->twiddle[idx + 0].im; // W_N^k imag
        tw_re_contig[3 * k + 1] = fft_obj->twiddle[idx + 1].re; // W_N^{2k} real
        tw_im_contig[3 * k + 1] = fft_obj->twiddle[idx + 1].im; // W_N^{2k} imag
        tw_re_contig[3 * k + 2] = fft_obj->twiddle[idx + 2].re; // W_N^{3k} real
        tw_im_contig[3 * k + 2] = fft_obj->twiddle[idx + 2].im; // W_N^{3k} imag
    }

    // Step 4: Flatten output_buffer into contiguous arrays for SSE2
    double *out_re = malloc(4 * sub_length * sizeof(double)); // Contiguous real outputs
    double *out_im = malloc(4 * sub_length * sizeof(double)); // Contiguous imag outputs
    if (!out_re || !out_im)
    {
        fprintf(stderr, "Error: Memory allocation failed for output arrays\n");
        free(tw_re_contig);
        free(tw_im_contig);
        free(out_re);
        free(out_im);
        exit(EXIT_FAILURE);
    }
    for (int lane = 0; lane < 4; lane++)
    {
        fft_data *base = output_buffer + lane * sub_length;
        for (int k = 0; k < sub_length; k++)
        {
            out_re[lane * sub_length + k] = base[k].re;
            out_im[lane * sub_length + k] = base[k].im;
        }
    }

    // Step 5: Handle k=0 separately (no twiddles needed)
    {
        fft_data *X0 = &output_buffer[0];              // First sub-FFT point
        fft_data *X1 = &output_buffer[sub_length];     // Second sub-FFT point
        fft_data *X2 = &output_buffer[2 * sub_length]; // Third sub-FFT point
        fft_data *X3 = &output_buffer[3 * sub_length]; // Fourth sub-FFT point

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
    }

    // Step 6: SSE2 vectorized loop for k=1 to sub_length-1 with twiddle factors
    __m128d vsign = _mm_set1_pd((double)transform_sign); // Vectorized transform sign
    int k = 1;
    for (; k + 1 < sub_length; k += 2)
    {
        // Load four sub-FFT points for two k values
        __m128d a_r = LOADU_SSE2(out_re + k);               // X0[k:k+1] real
        __m128d a_i = LOADU_SSE2(out_im + k);               // X0[k:k+1] imag
        __m128d b_r = LOADU_SSE2(out_re + k + sub_length);  // X1[k:k+1] real
        __m128d b_i = LOADU_SSE2(out_im + k + sub_length);  // X1[k:k+1] imag
        __m128d c_r = LOADU_SSE2(out_re + k + 2 * sub_length); // X2[k:k+1] real
        __m128d c_i = LOADU_SSE2(out_im + k + 2 * sub_length); // X2[k:k+1] imag
        __m128d d_r = LOADU_SSE2(out_re + k + 3 * sub_length); // X3[k:k+1] real
        __m128d d_i = LOADU_SSE2(out_im + k + 3 * sub_length); // X3[k:k+1] imag

        // Load twiddle factors for two k values
        int idx = 3 * k; // Base index for W_N^k, W_N^{2k}, W_N^{3k}
        __m128d w1r = LOADU_SSE2(tw_re_contig + idx + 0); // W_N^k real
        __m128d w1i = LOADU_SSE2(tw_im_contig + idx + 0); // W_N^k imag
        __m128d w2r = LOADU_SSE2(tw_re_contig + idx + 1); // W_N^{2k} real
        __m128d w2i = LOADU_SSE2(tw_im_contig + idx + 1); // W_N^{2k} imag
        __m128d w3r = LOADU_SSE2(tw_re_contig + idx + 2); // W_N^{3k} real
        __m128d w3i = LOADU_SSE2(tw_im_contig + idx + 2); // W_N^{3k} imag

        // Apply twiddle factors to X1, X2, X3
        __m128d b2r = FMSUB_SSE2(b_r, w1r, _mm_mul_pd(b_i, w1i)); // W_N^k * X1 real
        __m128d b2i = FMADD_SSE2(b_i, w1r, _mm_mul_pd(b_r, w1i)); // W_N^k * X1 imag
        __m128d c2r = FMSUB_SSE2(c_r, w2r, _mm_mul_pd(c_i, w2i)); // W_N^{2k} * X2 real
        __m128d c2i = FMADD_SSE2(c_i, w2r, _mm_mul_pd(c_r, w2i)); // W_N^{2k} * X2 imag
        __m128d d2r = FMSUB_SSE2(d_r, w3r, _mm_mul_pd(d_i, w3i)); // W_N^{3k} * X3 real
        __m128d d2i = FMADD_SSE2(d_i, w3r, _mm_mul_pd(d_r, w3i)); // W_N^{3k} * X3 imag

        // Radix-4 butterfly
        __m128d a2r = _mm_add_pd(a_r, c2r); // X0 + W_N^{2k} * X2 real
        __m128d a2i = _mm_add_pd(a_i, c2i); // X0 + W_N^{2k} * X2 imag
        __m128d e_r = _mm_sub_pd(a_r, c2r); // X0 - W_N^{2k} * X2 real
        __m128d e_i = _mm_sub_pd(a_i, c2i); // X0 - W_N^{2k} * X2 imag
        __m128d f_r = _mm_add_pd(b2r, d2r); // W_N^k * X1 + W_N^{3k} * X3 real
        __m128d f_i = _mm_add_pd(b2i, d2i); // W_N^k * X1 + W_N^{3k} * X3 imag
        __m128d g_r = _mm_mul_pd(_mm_sub_pd(b2r, d2r), vsign); // (W_N^k * X1 - W_N^{3k} * X3) * sign real
        __m128d g_i = _mm_mul_pd(_mm_sub_pd(b2i, d2i), vsign); // (W_N^k * X1 - W_N^{3k} * X3) * sign imag

        // Compute outputs
        STOREU_SSE2(out_re + k, _mm_add_pd(a2r, f_r)); // X(k) real
        STOREU_SSE2(out_im + k, _mm_add_pd(a2i, f_i)); // X(k) imag
        STOREU_SSE2(out_re + k + 2 * sub_length, _mm_sub_pd(a2r, f_r)); // X(k + N/4) real
        STOREU_SSE2(out_im + k + 2 * sub_length, _mm_sub_pd(a2i, f_i)); // X(k + N/4) imag
        STOREU_SSE2(out_re + k + sub_length, _mm_add_pd(e_r, g_i)); // X(k + N/2) real
        STOREU_SSE2(out_im + k + sub_length, _mm_sub_pd(e_i, g_r)); // X(k + N/2) imag
        STOREU_SSE2(out_re + k + 3 * sub_length, _mm_sub_pd(e_r, g_i)); // X(k + 3N/4) real
        STOREU_SSE2(out_im + k + 3 * sub_length, _mm_add_pd(e_i, g_r)); // X(k + 3N/4) imag
    }

    // Step 7: Scalar tail for remaining k
    for (; k < sub_length; k++)
    {
        int idx = (sub_length - 1) + 3 * k;          // Index into twiddle array
        fft_type w1r = fft_obj->twiddle[idx + 0].re; // W_N^k real
        fft_type w1i = fft_obj->twiddle[idx + 0].im; // W_N^k imag
        fft_type w2r = fft_obj->twiddle[idx + 1].re; // W_N^{2k} real
        fft_type w2i = fft_obj->twiddle[idx + 1].im; // W_N^{2k} imag
        fft_type w3r = fft_obj->twiddle[idx + 2].re; // W_N^{3k} real
        fft_type w3i = fft_obj->twiddle[idx + 2].im; // W_N^{3k} imag

        // Load the four sub-FFT points
        fft_data *X0 = &output_buffer[k];                  // First point
        fft_data *X1 = &output_buffer[k + sub_length];     // Second point
        fft_data *X2 = &output_buffer[k + 2 * sub_length]; // Third point
        fft_data *X3 = &output_buffer[k + 3 * sub_length]; // Fourth point

        // Apply twiddle factors to the "odd" branches
        fft_type b_re = X1->re * w1r - X1->im * w1i; // W_N^k * X_1 real
        fft_type b_im = X1->im * w1r + X1->re * w1i; // W_N^k * X_1 imag
        fft_type c_re = X2->re * w2r - X2->im * w2i; // W_N^{2k} * X_2 real
        fft_type c_im = X2->im * w2r + X2->re * w2i; // W_N^{2k} * X_2 imag
        fft_type d_re = X3->re * w3r - X3->im * w3i; // W_N^{3k} * X_3 real
        fft_type d_im = X3->im * w3r + X3->re * w3i; // W_N^{3k} * X_3 imag

        // Radix-4 butterfly on (X0, b, c, d)
        fft_type a_re = X0->re + c_re;                  // Sum with third point (real)
        fft_type a_im = X0->im + c_im;                  // Sum with third point (imag)
        fft_type e_re = X0->re - c_re;                  // Difference with third point (real)
        fft_type e_im = X0->im - c_im;                  // Difference with third point (imag)
        fft_type f_re = b_re + d_re;                    // Sum of twiddled second and fourth (real)
        fft_type f_im = b_im + d_im;                    // Sum of twiddled second and fourth (imag)
        fft_type g_re = transform_sign * (b_re - d_re); // Rotated difference (real)
        fft_type g_im = transform_sign * (b_im - d_im); // Rotated difference (imag)

        // Store results
        X0->re = a_re + f_re; // X(k) real
        X0->im = a_im + f_im; // X(k) imag
        X2->re = a_re - f_re; // X(k + N/4) real
        X2->im = a_im - f_im; // X(k + N/4) imag
        X1->re = e_re + g_im; // X(k + N/2) real
        X1->im = e_im - g_re; // X(k + N/2) imag
        X3->re = e_re - g_im; // X(k + 3N/4) real
        X3->im = e_im + g_re; // X(k + 3N/4) imag
    }

    // Step 8: Copy flattened results back to output_buffer
    for (int lane = 0; lane < 4; lane++)
    {
        fft_data *base = output_buffer + lane * sub_length;
        for (int k = 0; k < sub_length; k++)
        {
            base[k].re = out_re[lane * sub_length + k];
            base[k].im = out_im[lane * sub_length + k];
        }
    }

    // Step 9: Clean up allocated memory
    free(tw_re_contig);
    free(tw_im_contig);
    free(out_re);
    free(out_im);
    }
    else if (radix == 5)
    {
    /**
     * @brief Radix-5 decomposition for five-point sub-FFTs with SSE2 vectorization and FMA support.
     *
     * Intention: Compute the FFT for data lengths divisible by 5 by splitting into five sub-FFTs,
     * corresponding to indices n mod 5 = 0, 1, 2, 3, 4, and combining results with twiddle factors.
     * This supports N=5^r or mixed-radix cases, using SSE2 for broader compatibility.
     *
     * Mathematically: The FFT is computed as:
     *   \( X(k) = X_0(k) + W_N^k \cdot X_1(k) + W_N^{2k} \cdot X_2(k) + W_N^{3k} \cdot X_3(k) + W_N^{4k} \cdot X_4(k) \),
     * where \( X_0, ..., X_4 \) are sub-FFTs of size N/5, and \( W_N^k = e^{-2\pi i k / N} \).
     * The radix-5 butterfly uses rotations at 72°, 144°, 216°, and 288°.
     *
     */
    int sub_length = data_length / 5; // Size of each sub-FFT (N/5)
    int new_stride = 5 * stride;      // Stride increases fivefold
    int M = sub_length;               // Sub-FFT size for indexing

    // Step 1: Recurse into each of the 5 decimated sub-FFTs
    for (int i = 0; i < 5; i++)
    {
        mixed_radix_dit_rec(output_buffer + i * sub_length, input_buffer + i * stride, fft_obj,
                            transform_sign, sub_length, new_stride, factor_index + 1);
    }

    // Step 2: Flatten twiddle factors into contiguous arrays for SSE2
    double *tw_re_contig = malloc(4 * M * sizeof(double)); // Contiguous real twiddles
    double *tw_im_contig = malloc(4 * M * sizeof(double)); // Contiguous imag twiddles
    if (!tw_re_contig || !tw_im_contig)
    {
        fprintf(stderr, "Error: Memory allocation failed for twiddle arrays\n");
        free(tw_re_contig);
        free(tw_im_contig);
        exit(EXIT_FAILURE);
    }
    if (fft_obj->N - 1 < sub_length - 1 + 4 * (sub_length - 1))
    {
        fprintf(stderr, "Error: Twiddle array too small (need %d elements, have %d)\n",
                sub_length - 1 + 4 * (sub_length - 1), fft_obj->N - 1);
        free(tw_re_contig);
        free(tw_im_contig);
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k < M; k++)
    {
        int idx = (sub_length - 1) + 4 * k; // twiddle[sub_length-1 + n*k] = W_N^{n*k}
        for (int n = 0; n < 4; n++)
        {
            tw_re_contig[4 * k + n] = fft_obj->twiddle[idx + n].re; // W_N^{n*k} real
            tw_im_contig[4 * k + n] = fft_obj->twiddle[idx + n].im; // W_N^{n*k} imag
        }
    }

    // Step 3: Flatten output_buffer into contiguous arrays for SSE2
    double *out_re = malloc(5 * M * sizeof(double)); // Contiguous real outputs
    double *out_im = malloc(5 * M * sizeof(double)); // Contiguous imag outputs
    if (!out_re || !out_im)
    {
        fprintf(stderr, "Error: Memory allocation failed for output arrays\n");
        free(tw_re_contig);
        free(tw_im_contig);
        free(out_re);
        free(out_im);
        exit(EXIT_FAILURE);
    }
    for (int lane = 0; lane < 5; lane++)
    {
        fft_data *base = output_buffer + lane * M;
        for (int k = 0; k < M; k++)
        {
            out_re[lane * M + k] = base[k].re;
            out_im[lane * M + k] = base[k].im;
        }
    }

    // Step 4: k=0 (no twiddle multiplications)
    {
        fft_data *X0 = &output_buffer[0];              // First sub-FFT point
        fft_data *X1 = &output_buffer[sub_length];     // Second sub-FFT point
        fft_data *X2 = &output_buffer[2 * sub_length]; // Third sub-FFT point
        fft_data *X3 = &output_buffer[3 * sub_length]; // Fourth sub-FFT point
        fft_data *X4 = &output_buffer[4 * sub_length]; // Fifth sub-FFT point

        fft_type a_re = X0->re, a_im = X0->im; // First point
        fft_type b_re = X1->re, b_im = X1->im; // Second point
        fft_type c_re = X2->re, c_im = X2->im; // Third point
        fft_type d_re = X3->re, d_im = X3->im; // Fourth point
        fft_type e_re = X4->re, e_im = X4->im; // Fifth point

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

    // Step 5: SSE2 vectorized loop for k=1 to sub_length-1
    __m128d vsign = _mm_set1_pd((double)transform_sign); // Vectorized transform sign
    int k = 1;
    for (; k + 1 < M; k += 2)
    {
        // Load five sub-FFT points for two k values
        __m128d a_r = LOADU_SSE2(out_re + k);               // X0[k:k+1] real
        __m128d a_i = LOADU_SSE2(out_im + k);               // X0[k:k+1] imag
        __m128d b_r = LOADU_SSE2(out_re + k + M);           // X1[k:k+1] real
        __m128d b_i = LOADU_SSE2(out_im + k + M);           // X1[k:k+1] imag
        __m128d c_r = LOADU_SSE2(out_re + k + 2 * M);       // X2[k:k+1] real
        __m128d c_i = LOADU_SSE2(out_im + k + 2 * M);       // X2[k:k+1] imag
        __m128d d_r = LOADU_SSE2(out_re + k + 3 * M);       // X3[k:k+1] real
        __m128d d_i = LOADU_SSE2(out_im + k + 3 * M);       // X3[k:k+1] imag
        __m128d e_r = LOADU_SSE2(out_re + k + 4 * M);       // X4[k:k+1] real
        __m128d e_i = LOADU_SSE2(out_im + k + 4 * M);       // X4[k:k+1] imag

        // Load twiddle factors for two k values
        int idx = 4 * k; // Base index for W_N^k, ..., W_N^{4k}
        __m128d w1r = LOADU_SSE2(tw_re_contig + idx + 0); // W_N^k real
        __m128d w1i = LOADU_SSE2(tw_im_contig + idx + 0); // W_N^k imag
        __m128d w2r = LOADU_SSE2(tw_re_contig + idx + 1); // W_N^{2k} real
        __m128d w2i = LOADU_SSE2(tw_im_contig + idx + 1); // W_N^{2k} imag
        __m128d w3r = LOADU_SSE2(tw_re_contig + idx + 2); // W_N^{3k} real
        __m128d w3i = LOADU_SSE2(tw_im_contig + idx + 2); // W_N^{3k} imag
        __m128d w4r = LOADU_SSE2(tw_re_contig + idx + 3); // W_N^{4k} real
        __m128d w4i = LOADU_SSE2(tw_im_contig + idx + 3); // W_N^{4k} imag

        // Apply twiddle factors
        __m128d b2r = FMSUB_SSE2(b_r, w1r, _mm_mul_pd(b_i, w1i)); // W_N^k * X1 real
        __m128d b2i = FMADD_SSE2(b_i, w1r, _mm_mul_pd(b_r, w1i)); // W_N^k * X1 imag
        __m128d c2r = FMSUB_SSE2(c_r, w2r, _mm_mul_pd(c_i, w2i)); // W_N^{2k} * X2 real
        __m128d c2i = FMADD_SSE2(c_i, w2r, _mm_mul_pd(c_r, w2i)); // W_N^{2k} * X2 imag
        __m128d d2r = FMSUB_SSE2(d_r, w3r, _mm_mul_pd(d_i, w3i)); // W_N^{3k} * X3 real
        __m128d d2i = FMADD_SSE2(d_i, w3r, _mm_mul_pd(d_r, w3i)); // W_N^{3k} * X3 imag
        __m128d e2r = FMSUB_SSE2(e_r, w4r, _mm_mul_pd(e_i, w4i)); // W_N^{4k} * X4 real
        __m128d e2i = FMADD_SSE2(e_i, w4r, _mm_mul_pd(e_r, w4i)); // W_N^{4k} * X4 imag

        // Compute pairwise sums and differences
        __m128d t0r = _mm_add_pd(b2r, e2r); // b + e real
        __m128d t0i = _mm_add_pd(b2i, e2i); // b + e imag
        __m128d t1r = _mm_add_pd(c2r, d2r); // c + d real
        __m128d t1i = _mm_add_pd(c2i, d2i); // c + d imag
        __m128d t2r = _mm_sub_pd(b2r, e2r); // b - e real
        __m128d t2i = _mm_sub_pd(b2i, e2i); // b - e imag
        __m128d t3r = _mm_sub_pd(c2r, d2r); // c - d real
        __m128d t3i = _mm_sub_pd(c2i, d2i); // c - d imag

        // X0 = a + (b+e) + (c+d)
        __m128d x0r = _mm_add_pd(a_r, _mm_add_pd(t0r, t1r)); // X(k) real
        __m128d x0i = _mm_add_pd(a_i, _mm_add_pd(t0i, t1i)); // X(k) imag
        STOREU_SSE2(out_re + k, x0r);
        STOREU_SSE2(out_im + k, x0i);

        // Compute X1 and X4
        __m128d tmp1r = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(C5_1), t0r), _mm_mul_pd(_mm_set1_pd(C5_2), t1r)); // c1*(b+e) + c2*(c+d) real
        __m128d tmp1i = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(C5_1), t0i), _mm_mul_pd(_mm_set1_pd(C5_2), t1i)); // c1*(b+e) + c2*(c+d) imag
        __m128d rot1r = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(S5_1), t2i), _mm_mul_pd(_mm_set1_pd(S5_2), t3i)); // s1*(b-e) + s2*(c-d) real
        __m128d rot1i = _mm_sub_pd(_mm_mul_pd(_mm_set1_pd(S5_2), t3r), _mm_mul_pd(_mm_set1_pd(S5_1), t2r)); // s2*(c-d) - s1*(b-e) imag
        rot1r = _mm_mul_pd(rot1r, vsign); // Apply transform_sign
        rot1i = _mm_mul_pd(rot1i, vsign);
        __m128d x1r = _mm_add_pd(_mm_add_pd(a_r, tmp1r), rot1r); // X(k + N/5) real
        __m128d x1i = _mm_sub_pd(_mm_add_pd(a_i, tmp1i), rot1i); // X(k + N/5) imag
        __m128d x4r = _mm_sub_pd(_mm_add_pd(a_r, tmp1r), rot1r); // X(k + 4N/5) real
        __m128d x4i = _mm_add_pd(_mm_add_pd(a_i, tmp1i), rot1i); // X(k + 4N/5) imag
        STOREU_SSE2(out_re + k + M, x1r);
        STOREU_SSE2(out_im + k + M, x1i);
        STOREU_SSE2(out_re + k + 4 * M, x4r);
        STOREU_SSE2(out_im + k + 4 * M, x4i);

        // Compute X2 and X3
        __m128d tmp2r = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(C5_2), t0r), _mm_mul_pd(_mm_set1_pd(C5_1), t1r)); // c2*(b+e) + c1*(c+d) real
        __m128d tmp2i = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(C5_2), t0i), _mm_mul_pd(_mm_set1_pd(C5_1), t1i)); // c2*(b+e) + c1*(c+d) imag
        __m128d rot2r = _mm_sub_pd(_mm_mul_pd(_mm_set1_pd(S5_2), t2i), _mm_mul_pd(_mm_set1_pd(S5_1), t3i)); // s2*(b-e) - s1*(c-d) real
        __m128d rot2i = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(S5_1), t2r), _mm_mul_pd(_mm_set1_pd(S5_2), t3r)); // s1*(b-e) + s2*(c-d) imag
        rot2r = _mm_mul_pd(rot2r, vsign); // Apply transform_sign
        rot2i = _mm_mul_pd(rot2i, vsign);
        __m128d x2r = _mm_add_pd(_mm_add_pd(a_r, tmp2r), rot2r); // X(k + 2N/5) real
        __m128d x2i = _mm_sub_pd(_mm_add_pd(a_i, tmp2i), rot2i); // X(k + 2N/5) imag
        __m128d x3r = _mm_sub_pd(_mm_add_pd(a_r, tmp2r), rot2r); // X(k + 3N/5) real
        __m128d x3i = _mm_add_pd(_mm_add_pd(a_i, tmp2i), rot2i); // X(k + 3N/5) imag
        STOREU_SSE2(out_re + k + 2 * M, x2r);
        STOREU_SSE2(out_im + k + 2 * M, x2i);
        STOREU_SSE2(out_re + k + 3 * M, x3r);
        STOREU_SSE2(out_im + k + 3 * M, x3i);
    }

    // Step 6: Scalar tail for remaining k
    for (; k < M; k++)
    {
        int idx = (sub_length - 1) + 4 * k;          // Index into twiddle array
        fft_type w1r = fft_obj->twiddle[idx + 0].re; // W_N^k real
        fft_type w1i = fft_obj->twiddle[idx + 0].im; // W_N^k imag
        fft_type w2r = fft_obj->twiddle[idx + 1].re; // W_N^{2k} real
        fft_type w2i = fft_obj->twiddle[idx + 1].im; // W_N^{2k} imag
        fft_type w3r = fft_obj->twiddle[idx + 2].re; // W_N^{3k} real
        fft_type w3i = fft_obj->twiddle[idx + 2].im; // W_N^{3k} imag
        fft_type w4r = fft_obj->twiddle[idx + 3].re; // W_N^{4k} real
        fft_type w4i = fft_obj->twiddle[idx + 3].im; // W_N^{4k} imag

        fft_type a_re = out_re[k], a_im = out_im[k];                     // First point
        fft_type b_re = out_re[k + M] * w1r - out_im[k + M] * w1i;       // W_N^k * X_1 real
        fft_type b_im = out_im[k + M] * w1r + out_re[k + M] * w1i;       // W_N^k * X_1 imag
        fft_type c_re = out_re[k + 2 * M] * w2r - out_im[k + 2 * M] * w2i; // W_N^{2k} * X_2 real
        fft_type c_im = out_im[k + 2 * M] * w2r + out_re[k + 2 * M] * w2i; // W_N^{2k} * X_2 imag
        fft_type d_re = out_re[k + 3 * M] * w3r - out_im[k + 3 * M] * w3i; // W_N^{3k} * X_3 real
        fft_type d_im = out_im[k + 3 * M] * w3r + out_re[k + 3 * M] * w3i; // W_N^{3k} * X_3 imag
        fft_type e_re = out_re[k + 4 * M] * w4r - out_im[k + 4 * M] * w4i; // W_N^{4k} * X_4 real
        fft_type e_im = out_im[k + 4 * M] * w4r + out_re[k + 4 * M] * w4i; // W_N^{4k} * X_4 imag

        out_re[k] = a_re + b_re + c_re + d_re + e_re; // X(k) real
        out_im[k] = a_im + b_im + c_im + d_im + e_im; // X(k) imag
        out_re[k + M] = a_re + C5_1 * (b_re + e_re) + C5_2 * (c_re + d_re) +
                        transform_sign * (S5_1 * (b_im - e_im) + S5_2 * (c_im - d_im)); // X(k + N/5) real
        out_im[k + M] = a_im + C5_1 * (b_im + e_im) + C5_2 * (c_im + d_im) -
                        transform_sign * (S5_1 * (b_re - e_re) + S5_2 * (c_re - d_re)); // X(k + N/5) imag
        out_re[k + 4 * M] = a_re + C5_1 * (b_re + e_re) + C5_2 * (c_re + d_re) -
                            transform_sign * (S5_1 * (b_im - e_im) + S5_2 * (c_im - d_im)); // X(k + 4N/5) real
        out_im[k + 4 * M] = a_im + C5_1 * (b_im + e_im) + C5_2 * (c_im + d_im) +
                            transform_sign * (S5_1 * (b_re - e_re) + S5_2 * (c_re - d_re)); // X(k + 4N/5) imag
        out_re[k + 2 * M] = a_re + C5_2 * (b_re + e_re) + C5_1 * (c_re + d_re) +
                            transform_sign * (S5_2 * (b_im - e_im) - S5_1 * (c_im - d_im)); // X(k + 2N/5) real
        out_im[k + 2 * M] = a_im + C5_2 * (b_im + e_im) + C5_1 * (c_im + d_im) -
                            transform_sign * (S5_2 * (b_re - e_re) - S5_1 * (c_re - d_re)); // X(k + 2N/5) imag
        out_re[k + 3 * M] = a_re + C5_2 * (b_re + e_re) + C5_1 * (c_re + d_re) -
                            transform_sign * (S5_2 * (b_im - e_im) - S5_1 * (c_im - d_im)); // X(k + 3N/5) real
        out_im[k + 3 * M] = a_im + C5_2 * (b_im + e_im) + C5_1 * (c_im + d_im) +
                            transform_sign * (S5_2 * (b_re - e_re) - S5_1 * (c_re - d_re)); // X(k + 3N/5) imag
    }

    // Step 7: Copy flattened results back to output_buffer
    for (int lane = 0; lane < 5; lane++)
    {
        fft_data *base = output_buffer + lane * M;
        for (int k = 0; k < M; k++)
        {
            base[k].re = out_re[lane * M + k];
            base[k].im = out_im[lane * M + k];
        }
    }

    // Step 8: Clean up allocated memory
    free(tw_re_contig);
    free(tw_im_contig);
    free(out_re);
    free(out_im);
    }
    else if (radix == 7)
    {
    /**
     * @brief Radix-7 decomposition for seven-point sub-FFTs with AVX2 vectorization and FMA support.
     *
     * Intention: Optimize FFT computation for data lengths divisible by 7 by splitting into seven sub-FFTs,
     * corresponding to indices n mod 7 = 0 to 6, and combining results with twiddle factors. This supports
     * N=7^r or mixed-radix cases, leveraging AVX2 for parallel processing and FMA for efficiency.
     *
     * Mathematically: The FFT is computed as:
     *   \( X(k) = X_0(k) + W_N^k \cdot X_1(k) + ... + W_N^{6k} \cdot X_6(k) \),
     * where \( X_0, ..., X_6 \) are sub-FFTs of size N/7, and \( W_N^k = e^{-2\pi i k / N} \).
     * The radix-7 butterfly uses rotations at multiples of 360°/7 ≈ 51.43°.
     *
     * Process:
     * 1. Divide data into seven subproblems of size N/7, adjusting stride.
     * 2. Recursively compute FFTs for each subproblem.
     * 3. Pre-build contiguous twiddle arrays for efficient SIMD access.
     * 4. Flatten output_buffer into contiguous real/imag arrays to avoid interleaved memory issues.
     * 5. Compute k=0 scalarly without twiddle factors.
     * 6. Combine results for k≥1 using AVX2 with FMA (via FMADD/FMSUB) for k divisible by 4.
     * 7. Handle remaining k with scalar operations.
     * 8. Copy results back to output_buffer.
     * 9. Clean up allocated memory.
     *
     * Optimization:
     * - AVX2 vectorization processes four complex points simultaneously.
     * - FMA instructions (FMADD/FMSUB) reduce instruction count and improve numerical accuracy
     *   when compiled with -mfma or USE_FMA defined (requires Intel Haswell+ or AMD Zen+).
     * - Without FMA, inlined fallback functions (ALWAYS_INLINE) avoid call overhead.
     * - AVX_ONE macro inlines _mm256_set1_pd(1.0) for constant vectors.
     * - File-scope scalar constants (C1, S1, etc.) and function-scope SIMD vectors (CC1, SS1) avoid
     *   per-call splat overhead.
     * - Branch-free sign flip with SIGN_MASK improves loop performance.
     * - AVX_ZERO constant avoids _mm256_setzero_pd() calls.
     * - Contiguous twiddle and output arrays ensure efficient SIMD loads/stores.
     * - Symmetry in outputs (e.g., X_1/X_6, X_2/X_5, X_3/X_4 are conjugates) reduces computations.
     *
     */
    int sub_length = data_length / 7; // Size of each sub-FFT (N/7)
    int new_stride = 7 * stride;      // Stride increases sevenfold
    int M = sub_length;               // Sub-FFT size for indexing

    // Step 1: Recurse on each of the 7 sub-FFTs
    for (int i = 0; i < 7; i++)
    {
        mixed_radix_dit_rec(output_buffer + i * sub_length, input_buffer + i * stride,
                            fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
    }

    // Step 2: Pre-build contiguous twiddle arrays
    double *tw_re_contig = malloc(6 * M * sizeof(double)); // Contiguous real twiddles
    double *tw_im_contig = malloc(6 * M * sizeof(double)); // Contiguous imag twiddles
    if (!tw_re_contig || !tw_im_contig)
    {
        fprintf(stderr, "Error: Memory allocation failed for twiddle arrays\n");
        free(tw_re_contig);
        free(tw_im_contig);
        exit(EXIT_FAILURE);
    }
    if (fft_obj->N - 1 < 7 * M - 1)
    {
        fprintf(stderr, "Error: Twiddle array too small (need %d elements, have %d)\n", 7 * M - 1, fft_obj->N - 1);
        free(tw_re_contig);
        free(tw_im_contig);
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k < M; k++)
    {
        for (int n = 1; n <= 6; n++)
        {
            int idx = sub_length - 1 + n * k; // twiddle[idx] = W_N^{n*k}; validate in fft_init
            tw_re_contig[6 * k + (n - 1)] = fft_obj->twiddle[idx].re;
            tw_im_contig[6 * k + (n - 1)] = fft_obj->twiddle[idx].im;
        }
    }

    // Step 3: Flatten output_buffer into contiguous real/imag arrays
    double *out_re = malloc(7 * M * sizeof(double)); // Contiguous real outputs
    double *out_im = malloc(7 * M * sizeof(double)); // Contiguous imag outputs
    if (!out_re || !out_im)
    {
        fprintf(stderr, "Error: Memory allocation failed for output arrays\n");
        free(tw_re_contig);
        free(tw_im_contig);
        free(out_re);
        free(out_im);
        exit(EXIT_FAILURE);
    }
    for (int lane = 0; lane < 7; lane++)
    {
        fft_data *base = output_buffer + lane * M;
        for (int k = 0; k < M; k++)
        {
            out_re[lane * M + k] = base[k].re;
            out_im[lane * M + k] = base[k].im;
        }
    }

    // Step 4: Define SIMD constants
    __m256d CC1       = _mm256_set1_pd(C1);
    __m256d CC2       = _mm256_set1_pd(C2);
    __m256d CC3       = _mm256_set1_pd(C3);
    __m256d SS1       = _mm256_set1_pd(S1);
    __m256d SS2       = _mm256_set1_pd(S2);
    __m256d SS3       = _mm256_set1_pd(S3);
    __m256d AVX_ZERO  = _mm256_setzero_pd();
    __m256d SIGN_MASK = _mm256_set1_pd(transform_sign == 1 ? -1.0 : +1.0);

    // Step 5: Scalar prologue for k=0 (no twiddle factors)
    {
        double a_r = out_re[0], a_i = out_im[0];
        double b_r = out_re[M], b_i = out_im[M];
        double c_r = out_re[2 * M], c_i = out_im[2 * M];
        double d_r = out_re[3 * M], d_i = out_im[3 * M];
        double e_r = out_re[4 * M], e_i = out_im[4 * M];
        double f_r = out_re[5 * M], f_i = out_im[5 * M];
        double g_r = out_re[6 * M], g_i = out_im[6 * M];

        double t0r = b_r + g_r, t0i = b_i + g_i; // B + G
        double t3r = b_r - g_r, t3i = b_i - g_i; // B - G
        double t1r = c_r + f_r, t1i = c_i + f_i; // C + F
        double t4r = c_r - f_r, t4i = c_i - f_i; // C - F
        double t2r = d_r + e_r, t2i = d_i + e_i; // D + E
        double t5r = d_r - e_r, t5i = d_i - e_i; // D - E

        out_re[0] = a_r + t0r + t1r + t2r; // X(0)
        out_im[0] = a_i + t0i + t1i + t2i;

        double tmp_r = a_r + C1 * t0r + C2 * t1r + C3 * t2r; // X(1) and X(6)
        double tmp_i = a_i + C1 * t0i + C2 * t1i + C3 * t2i;
        double rot_r = -(S1 * t3r + S2 * t4r + S3 * t5r) * transform_sign;
        double rot_i = -(S1 * t3i + S2 * t4i + S3 * t5i) * transform_sign;
        out_re[1 * M] = tmp_r - rot_i;
        out_im[1 * M] = tmp_i + rot_r;
        out_re[6 * M] = tmp_r + rot_i;
        out_im[6 * M] = tmp_i - rot_r;

        tmp_r = a_r + C2 * t0r + C3 * t1r + C1 * t2r; // X(2) and X(5)
        tmp_i = a_i + C2 * t0i + C3 * t1i + C1 * t2i;
        rot_r = -(S2 * t3r + S3 * t4r + S1 * t5r) * transform_sign;
        rot_i = -(S2 * t3i + S3 * t4i + S1 * t5i) * transform_sign;
        out_re[2 * M] = tmp_r - rot_i;
        out_im[2 * M] = tmp_i + rot_r;
        out_re[5 * M] = tmp_r + rot_i;
        out_im[5 * M] = tmp_i - rot_r;

        tmp_r = a_r + C3 * t0r + C1 * t1r + C2 * t2r; // X(3) and X(4)
        tmp_i = a_i + C3 * t0i + C1 * t1i + C2 * t2i;
        rot_r = -(S3 * t3r + S1 * t4r + S2 * t5r) * transform_sign;
        rot_i = -(S3 * t3i + S1 * t4i + S2 * t5i) * transform_sign;
        out_re[3 * M] = tmp_r - rot_i;
        out_im[3 * M] = tmp_i + rot_r;
        out_re[4 * M] = tmp_r + rot_i;
        out_im[4 * M] = tmp_i - rot_r;
    }

    // Step 6: Vectorized loop for k=1 and beyond
    int k = 1;
    for (; k + 3 < M; k += 4)
    {
        // Load seven sub-FFT points (A to G)
        __m256d a_r = LOADU_PD(out_re + k + 0 * M); // First point real
        __m256d a_i = LOADU_PD(out_im + k + 0 * M); // First point imag
        __m256d b_r = LOADU_PD(out_re + k + 1 * M); // Second point real
        __m256d b_i = LOADU_PD(out_im + k + 1 * M); // Second point imag
        __m256d c_r = LOADU_PD(out_re + k + 2 * M); // Third point real
        __m256d c_i = LOADU_PD(out_im + k + 2 * M); // Third point imag
        __m256d d_r = LOADU_PD(out_re + k + 3 * M); // Fourth point real
        __m256d d_i = LOADU_PD(out_im + k + 3 * M); // Fourth point imag
        __m256d e_r = LOADU_PD(out_re + k + 4 * M); // Fifth point real
        __m256d e_i = LOADU_PD(out_im + k + 4 * M); // Fifth point imag
        __m256d f_r = LOADU_PD(out_re + k + 5 * M); // Sixth point real
        __m256d f_i = LOADU_PD(out_im + k + 5 * M); // Sixth point imag
        __m256d g_r = LOADU_PD(out_re + k + 6 * M); // Seventh point real
        __m256d g_i = LOADU_PD(out_im + k + 6 * M); // Seventh point imag

        // Load six twiddle factors W_N^k to W_N^{6k}
        int base = 6 * k;
        __m256d w1_r = LOADU_PD(tw_re_contig + base + 0); // W_N^k real
        __m256d w1_i = LOADU_PD(tw_im_contig + base + 0); // W_N^k imag
        __m256d w2_r = LOADU_PD(tw_re_contig + base + 1); // W_N^{2k} real
        __m256d w2_i = LOADU_PD(tw_im_contig + base + 1); // W_N^{2k} imag
        __m256d w3_r = LOADU_PD(tw_re_contig + base + 2); // W_N^{3k} real
        __m256d w3_i = LOADU_PD(tw_im_contig + base + 2); // W_N^{3k} imag
        __m256d w4_r = LOADU_PD(tw_re_contig + base + 3); // W_N^{4k} real
        __m256d w4_i = LOADU_PD(tw_im_contig + base + 3); // W_N^{4k} imag
        __m256d w5_r = LOADU_PD(tw_re_contig + base + 4); // W_N^{5k} real
        __m256d w5_i = LOADU_PD(tw_im_contig + base + 4); // W_N^{5k} imag
        __m256d w6_r = LOADU_PD(tw_re_contig + base + 5); // W_N^{6k} real
        __m256d w6_i = LOADU_PD(tw_im_contig + base + 5); // W_N^{6k} imag

        // Apply twiddle factors to points B to G
        __m256d b2_r = FMSUB(b_r, w1_r, _mm256_mul_pd(b_i, w1_i)); // W_N^k * B real
        __m256d b2_i = FMADD(b_i, w1_r, _mm256_mul_pd(b_r, w1_i)); // W_N^k * B imag
        __m256d c2_r = FMSUB(c_r, w2_r, _mm256_mul_pd(c_i, w2_i)); // W_N^{2k} * C real
        __m256d c2_i = FMADD(c_i, w2_r, _mm256_mul_pd(c_r, w2_i)); // W_N^{2k} * C imag
        __m256d d2_r = FMSUB(d_r, w3_r, _mm256_mul_pd(d_i, w3_i)); // W_N^{3k} * D real
        __m256d d2_i = FMADD(d_i, w3_r, _mm256_mul_pd(d_r, w3_i)); // W_N^{3k} * D imag
        __m256d e2_r = FMSUB(e_r, w4_r, _mm256_mul_pd(e_i, w4_i)); // W_N^{4k} * E real
        __m256d e2_i = FMADD(e_i, w4_r, _mm256_mul_pd(e_r, w4_i)); // W_N^{4k} * E imag
        __m256d f2_r = FMSUB(f_r, w5_r, _mm256_mul_pd(f_i, w5_i)); // W_N^{5k} * F real
        __m256d f2_i = FMADD(f_i, w5_r, _mm256_mul_pd(f_r, w5_i)); // W_N^{5k} * F imag
        __m256d g2_r = FMSUB(g_r, w6_r, _mm256_mul_pd(g_i, w6_i)); // W_N^{6k} * G real
        __m256d g2_i = FMADD(g_i, w6_r, _mm256_mul_pd(g_r, w6_i)); // W_N^{6k} * G imag

        // Compute pairwise sums and differences
        __m256d t0r = _mm256_add_pd(b2_r, g2_r); // B + G real
        __m256d t0i = _mm256_add_pd(b2_i, g2_i); // B + G imag
        __m256d t3r = _mm256_sub_pd(b2_r, g2_r); // B - G real
        __m256d t3i = _mm256_sub_pd(b2_i, g2_i); // B - G imag
        __m256d t1r = _mm256_add_pd(c2_r, f2_r); // C + F real
        __m256d t1i = _mm256_add_pd(c2_i, f2_i); // C + F imag
        __m256d t4r = _mm256_sub_pd(c2_r, f2_r); // C - F real
        __m256d t4i = _mm256_sub_pd(c2_i, f2_i); // C - F imag
        __m256d t2r = _mm256_add_pd(d2_r, e2_r); // D + E real
        __m256d t2i = _mm256_add_pd(d2_i, e2_i); // D + E imag
        __m256d t5r = _mm256_sub_pd(d2_r, e2_r); // D - E real
        __m256d t5i = _mm256_sub_pd(d2_i, e2_i); // D - E imag

        // Compute X(k) = a + t0 + t1 + t2
        __m256d sum01r = _mm256_add_pd(t0r, t1r);    // (B+G) + (C+F) real
        __m256d sum01i = _mm256_add_pd(t0i, t1i);    // (B+G) + (C+F) imag
        __m256d sum2ar = _mm256_add_pd(sum01r, t2r); // + (D+E) real
        __m256d sum2ai = _mm256_add_pd(sum01i, t2i); // + (D+E) imag
        __m256d out0r = _mm256_add_pd(a_r, sum2ar);  // X(k) real
        __m256d out0i = _mm256_add_pd(a_i, sum2ai);  // X(k) imag
        STOREU_PD(out_re + k + 0 * M, out0r); // Store X(k) real
        STOREU_PD(out_im + k + 0 * M, out0i); // Store X(k) imag

        // Compute rotated center for X(k+1, k+6): a + c1*t0 + c2*t1 + c3*t2
        __m256d tmp_r = FMADD(CC3, t2r, FMADD(CC2, t1r, FMADD(CC1, t0r, a_r))); // Real
        __m256d tmp_i = FMADD(CC3, t2i, FMADD(CC2, t1i, FMADD(CC1, t0i, a_i))); // Imag

        // Compute rotations for X(k+1) and X(k+6)
        __m256d rot1 = FMADD(SS3, t5r, FMADD(SS2, t4r, _mm256_mul_pd(SS1, t3r)));  // Rotation real
        __m256d rot1i = FMADD(SS3, t5i, FMADD(SS2, t4i, _mm256_mul_pd(SS1, t3i))); // Rotation imag
        rot1 = _mm256_mul_pd(rot1, SIGN_MASK);  // Apply sign flip
        rot1i = _mm256_mul_pd(rot1i, SIGN_MASK);
        STOREU_PD(out_re + k + 1 * M, _mm256_sub_pd(tmp_r, rot1i)); // X(k+1) real
        STOREU_PD(out_im + k + 1 * M, _mm256_add_pd(tmp_i, rot1));  // X(k+1) imag
        STOREU_PD(out_re + k + 6 * M, _mm256_add_pd(tmp_r, rot1i)); // X(k+6) real
        STOREU_PD(out_im + k + 6 * M, _mm256_sub_pd(tmp_i, rot1));  // X(k+6) imag

        // Compute rotations for X(k+2) and X(k+5)
        __m256d tmp2_r = FMADD(CC1, t2r, FMADD(CC3, t1r, FMADD(CC2, t0r, a_r))); // Real
        __m256d tmp2_i = FMADD(CC1, t2i, FMADD(CC3, t1i, FMADD(CC2, t0i, a_i))); // Imag
        __m256d rot2 = FMADD(SS1, t5r, FMADD(SS3, t4r, _mm256_mul_pd(SS2, t3r))); // Rotation real
        __m256d rot2i = FMADD(SS1, t5i, FMADD(SS3, t4i, _mm256_mul_pd(SS2, t3i))); // Rotation imag
        rot2 = _mm256_mul_pd(rot2, SIGN_MASK);  // Apply sign flip
        rot2i = _mm256_mul_pd(rot2i, SIGN_MASK);
        STOREU_PD(out_re + k + 2 * M, _mm256_sub_pd(tmp2_r, rot2i)); // X(k+2) real
        STOREU_PD(out_im + k + 2 * M, _mm256_add_pd(tmp2_i, rot2));  // X(k+2) imag
        STOREU_PD(out_re + k + 5 * M, _mm256_add_pd(tmp2_r, rot2i)); // X(k+5) real
        STOREU_PD(out_im + k + 5 * M, _mm256_sub_pd(tmp2_i, rot2));  // X(k+5) imag

        // Compute rotations for X(k+3) and X(k+4)
        __m256d tmp3_r = FMADD(CC2, t2r, FMADD(CC1, t1r, FMADD(CC3, t0r, a_r))); // Real
        __m256d tmp3_i = FMADD(CC2, t2i, FMADD(CC1, t1i, FMADD(CC3, t0i, a_i))); // Imag
        __m256d rot3 = FMADD(SS2, t5r, FMADD(SS1, t4r, _mm256_mul_pd(SS3, t3r))); // Rotation real
        __m256d rot3i = FMADD(SS2, t5i, FMADD(SS1, t4i, _mm256_mul_pd(SS3, t3i))); // Rotation imag
        rot3 = _mm256_mul_pd(rot3, SIGN_MASK);  // Apply sign flip
        rot3i = _mm256_mul_pd(rot3i, SIGN_MASK);
        STOREU_PD(out_re + k + 3 * M, _mm256_sub_pd(tmp3_r, rot3i)); // X(k+3) real
        STOREU_PD(out_im + k + 3 * M, _mm256_add_pd(tmp3_i, rot3));  // X(k+3) imag
        STOREU_PD(out_re + k + 4 * M, _mm256_add_pd(tmp3_r, rot3i)); // X(k+4) real
        STOREU_PD(out_im + k + 4 * M, _mm256_sub_pd(tmp3_i, rot3));  // X(k+4) imag
    }

    // Step 7: Scalar tail for remaining k
    for (; k < M; ++k)
    {
        /**
         * @brief Combine sub-FFT results for higher indices using twiddle factors (scalar).
         *
         * For frequency indices k not divisible by 4, applies twiddle factors to rotate and
         * combine the sub-FFT results from seven recursive calls. This ensures correct phase
         * alignment of frequency components. FMA emulation not used, as scalar benefits are minimal.
         *
         * Mathematically: Computes:
         *   \( X(k) = \sum_{n=0}^{6} X_n(k) \cdot W_N^{kn} \),
         * where \( W_N = e^{-2\pi i / N} \) for forward transforms, adjusted by transform_sign.
         */
        double a_r = out_re[k + 0 * M], a_i = out_im[k + 0 * M];
        double b_r = out_re[k + 1 * M], b_i = out_im[k + 1 * M];
        double c_r = out_re[k + 2 * M], c_i = out_im[k + 2 * M];
        double d_r = out_re[k + 3 * M], d_i = out_im[k + 3 * M];
        double e_r = out_re[k + 4 * M], e_i = out_im[k + 4 * M];
        double f_r = out_re[k + 5 * M], f_i = out_im[k + 5 * M];
        double g_r = out_re[k + 6 * M], g_i = out_im[k + 6 * M];

        int base = 6 * k;
        double w1r = tw_re_contig[base + 0], w1i = tw_im_contig[base + 0];
        double w2r = tw_re_contig[base + 1], w2i = tw_im_contig[base + 1];
        double w3r = tw_re_contig[base + 2], w3i = tw_im_contig[base + 2];
        double w4r = tw_re_contig[base + 3], w4i = tw_im_contig[base + 3];
        double w5r = tw_re_contig[base + 4], w5i = tw_im_contig[base + 4];
        double w6r = tw_re_contig[base + 5], w6i = tw_im_contig[base + 5];

        double b2r = b_r * w1r - b_i * w1i;
        double b2i = b_i * w1r + b_r * w1i;
        double c2r = c_r * w2r - c_i * w2i;
        double c2i = c_i * w2r + c_r * w2i;
        double d2r = d_r * w3r - d_i * w3i;
        double d2i = d_i * w3r + d_r * w3i;
        double e2r = e_r * w4r - e_i * w4i;
        double e2i = e_i * w4r + e_r * w4i;
        double f2r = f_r * w5r - f_i * w5i;
        double f2i = f_i * w5r + f_r * w5i;
        double g2r = g_r * w6r - g_i * w6i;
        double g2i = g_i * w6r + g_r * w6i;

        double t0r = b2r + g2r, t0i = b2i + g2i;
        double t3r = b2r - g2r, t3i = b2i - g2i;
        double t1r = c2r + f2r, t1i = c2i + f2i;
        double t4r = c2r - f2r, t4i = c2i - f2i;
        double t2r = d2r + e2r, t2i = d2i + e2i;
        double t5r = d2r - e2r, t5i = d2i - e2i;

        double sumr = a_r + t0r + t1r + t2r;
        double sumi = a_i + t0i + t1i + t2i;
        out_re[k + 0 * M] = sumr;
        out_im[k + 0 * M] = sumi;

        double tmp_r, tmp_i, rot_r, rot_i;

        tmp_r = a_r + C1 * t0r + C2 * t1r + C3 * t2r;
        tmp_i = a_i + C1 * t0i + C2 * t1i + C3 * t2i;
        rot_r = -(S1 * t3r + S2 * t4r + S3 * t5r) * transform_sign;
        rot_i = -(S1 * t3i + S2 * t4i + S3 * t5i) * transform_sign;
        out_re[k + 1 * M] = tmp_r - rot_i;
        out_im[k + 1 * M] = tmp_i + rot_r;
        out_re[k + 6 * M] = tmp_r + rot_i;
        out_im[k + 6 * M] = tmp_i - rot_r;

        tmp_r = a_r + C2 * t0r + C3 * t1r + C1 * t2r;
        tmp_i = a_i + C2 * t0i + C3 * t1i + C1 * t2i;
        rot_r = -(S2 * t3r + S3 * t4r + S1 * t5r) * transform_sign;
        rot_i = -(S2 * t3i + S3 * t4i + S1 * t5i) * transform_sign;
        out_re[k + 2 * M] = tmp_r - rot_i;
        out_im[k + 2 * M] = tmp_i + rot_r;
        out_re[k + 5 * M] = tmp_r + rot_i;
        out_im[k + 5 * M] = tmp_i - rot_r;

        tmp_r = a_r + C3 * t0r + C1 * t1r + C2 * t2r;
        tmp_i = a_i + C3 * t0i + C1 * t1i + C2 * t2i;
        rot_r = -(S3 * t3r + S1 * t4r + S2 * t5r) * transform_sign;
        rot_i = -(S3 * t3i + S1 * t4i + S2 * t5i) * transform_sign;
        out_re[k + 3 * M] = tmp_r - rot_i;
        out_im[k + 3 * M] = tmp_i + rot_r;
        out_re[k + 4 * M] = tmp_r + rot_i;
        out_im[k + 4 * M] = tmp_i - rot_r;
    }

    // Step 8: Copy results back to output_buffer
    for (int lane = 0; lane < 7; lane++)
    {
        fft_data *base = output_buffer + lane * M;
        for (int k = 0; k < M; k++)
        {
            base[k].re = out_re[lane * M + k];
            base[k].im = out_im[lane * M + k];
        }
    }

    // Step 9: Clean up allocated memory
    free(tw_re_contig);
    free(tw_im_contig);
    free(out_re);
    free(out_im);
    }
    else if (radix == 8)
    {
    /**
     * @brief Radix-8 decomposition for eight-point sub-FFTs with AVX2 vectorization and FMA support.
     *
     * Intention: Optimize FFT computation for data lengths divisible by 8 by splitting into eight sub-FFTs,
     * corresponding to indices n mod 8 = 0 to 7, and combining results with twiddle factors. This is highly
     * efficient for N=8^r or mixed-radix cases, leveraging AVX2 for parallel processing and FMA for efficiency.
     *
     * Mathematically: The FFT is computed as:
     *   \( X(k) = X_0(k) + W_N^k \cdot X_1(k) + ... + W_N^{7k} \cdot X_7(k) \),
     * where \( X_0, ..., X_7 \) are sub-FFTs of size N/8, and \( W_N^k = e^{-2\pi i k / N} \).
     * The radix-8 butterfly uses rotations at multiples of 360°/8 = 45°, with √2/2 for 45° rotations.
     *
     */
    int sub_length = data_length / 8; // Size of each sub-FFT (N/8)
    int M = sub_length;               // Sub-FFT size for indexing
    int new_stride = 8 * stride;      // Stride increases eightfold

    // Step 1: Flatten twiddle factors into contiguous arrays for AVX2
    int N = fft_obj->N;                            // Original signal length
    int Tlen = N - 1;                              // Number of twiddle factors
    double *tw_re = malloc(Tlen * sizeof(double)); // Contiguous real twiddles
    double *tw_im = malloc(Tlen * sizeof(double)); // Contiguous imag twiddles
    if (!tw_re || !tw_im)
    {
        fprintf(stderr, "Error: Memory allocation failed for twiddle arrays\n");
        free(tw_re);
        free(tw_im);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < Tlen; ++i)
    {
        tw_re[i] = fft_obj->twiddle[i].re; // Copy real part
        tw_im[i] = fft_obj->twiddle[i].im; // Copy imag part
    }

    // Step 2: Recurse on 8 sub-FFTs
    for (int i = 0; i < 8; ++i)
    {
        mixed_radix_dit_rec(output_buffer + i * sub_length, input_buffer + i * stride,
                            fft_obj, transform_sign, sub_length, new_stride,
                            factor_index + 1); // Compute sub-FFT for each lane
    }

    // Step 3: Flatten interleaved fft_data outputs into contiguous arrays
    double *out_re = malloc(8 * M * sizeof(double)); // Contiguous real outputs
    double *out_im = malloc(8 * M * sizeof(double)); // Contiguous imag outputs
    if (!out_re || !out_im)
    {
        fprintf(stderr, "Error: Memory allocation failed for output arrays\n");
        free(tw_re);
        free(tw_im);
        free(out_re);
        free(out_im);
        exit(EXIT_FAILURE);
    }
    for (int lane = 0; lane < 8; lane++)
    {
        fft_data *base = output_buffer + lane * M; // Base pointer for lane
        for (int k = 0; k < M; k++)
        {
            out_re[lane * M + k] = base[k].re; // Copy real part
            out_im[lane * M + k] = base[k].im; // Copy imag part
        }
    }

    // Step 4: AVX2 vectorized radix-8 butterfly
    int k = 0;                                  // Frequency index
    __m256d vsign = _mm256_set1_pd((double)transform_sign); // Vectorized transform sign

    // Vectorized loop: Process 4 complex points at a time
    for (; k + 3 < M; k += 4)
    {
        // Load eight sub-FFT points (A to H)
        __m256d ar = LOADU_PD(out_re + k + 0 * M); // First point real
        __m256d ai = LOADU_PD(out_im + k + 0 * M); // First point imag
        __m256d br = LOADU_PD(out_re + k + 1 * M); // Second point real
        __m256d bi = LOADU_PD(out_im + k + 1 * M); // Second point imag
        __m256d cr = LOADU_PD(out_re + k + 2 * M); // Third point real
        __m256d ci = LOADU_PD(out_im + k + 2 * M); // Third point imag
        __m256d dr = LOADU_PD(out_re + k + 3 * M); // Fourth point real
        __m256d di = LOADU_PD(out_im + k + 3 * M); // Fourth point imag
        __m256d er = LOADU_PD(out_re + k + 4 * M); // Fifth point real
        __m256d ei = LOADU_PD(out_im + k + 4 * M); // Fifth point imag
        __m256d fr = LOADU_PD(out_re + k + 5 * M); // Sixth point real
        __m256d fi = LOADU_PD(out_im + k + 5 * M); // Sixth point imag
        __m256d gr = LOADU_PD(out_re + k + 6 * M); // Seventh point real
        __m256d gi = LOADU_PD(out_im + k + 6 * M); // Seventh point imag
        __m256d hr = LOADU_PD(out_re + k + 7 * M); // Eighth point real
        __m256d hi = LOADU_PD(out_im + k + 7 * M); // Eighth point imag

        // Load seven twiddle factors W_N^k to W_N^{7k}
        int idx = (M - 1) + 7 * k;                       // Twiddle index
        __m256d w1r = LOADU_PD(tw_re + idx + 0);  // W_N^k real
        __m256d w1i = LOADU_PD(tw_im + idx + 0);  // W_N^k imag
        __m256d w2r = LOADU_PD(tw_re + idx + 1);  // W_N^{2k} real
        __m256d w2i = LOADU_PD(tw_im + idx + 1);  // W_N^{2k} imag
        __m256d w3r = LOADU_PD(tw_re + idx + 2);  // W_N^{3k} real
        __m256d w3i = LOADU_PD(tw_im + idx + 2);  // W_N^{3k} imag
        __m256d w4r = LOADU_PD(tw_re + idx + 3);  // W_N^{4k} real
        __m256d w4i = LOADU_PD(tw_im + idx + 3);  // W_N^{4k} imag
        __m256d w5r = LOADU_PD(tw_re + idx + 4);  // W_N^{5k} real
        __m256d w5i = LOADU_PD(tw_im + idx + 4);  // W_N^{5k} imag
        __m256d w6r = LOADU_PD(tw_re + idx + 5);  // W_N^{6k} real
        __m256d w6i = LOADU_PD(tw_im + idx + 5);  // W_N^{6k} imag
        __m256d w7r = LOADU_PD(tw_re + idx + 6);  // W_N^{7k} real
        __m256d w7i = LOADU_PD(tw_im + idx + 6);  // W_N^{7k} imag

        // Apply twiddle factors to points B to H
        __m256d b2r = FMSUB(br, w1r, _mm256_mul_pd(bi, w1i)); // W_N^k * B real
        __m256d b2i = FMADD(bi, w1r, _mm256_mul_pd(br, w1i)); // W_N^k * B imag
        __m256d c2r = FMSUB(cr, w2r, _mm256_mul_pd(ci, w2i)); // W_N^{2k} * C real
        __m256d c2i = FMADD(ci, w2r, _mm256_mul_pd(cr, w2i)); // W_N^{2k} * C imag
        __m256d d2r = FMSUB(dr, w3r, _mm256_mul_pd(di, w3i)); // W_N^{3k} * D real
        __m256d d2i = FMADD(di, w3r, _mm256_mul_pd(dr, w3i)); // W_N^{3k} * D imag
        __m256d e2r = FMSUB(er, w4r, _mm256_mul_pd(ei, w4i)); // W_N^{4k} * E real
        __m256d e2i = FMADD(ei, w4r, _mm256_mul_pd(er, w4i)); // W_N^{4k} * E imag
        __m256d f2r = FMSUB(fr, w5r, _mm256_mul_pd(fi, w5i)); // W_N^{5k} * F real
        __m256d f2i = FMADD(fi, w5r, _mm256_mul_pd(fr, w5i)); // W_N^{5k} * F imag
        __m256d g2r = FMSUB(gr, w6r, _mm256_mul_pd(gi, w6i)); // W_N^{6k} * G real
        __m256d g2i = FMADD(gi, w6r, _mm256_mul_pd(gr, w6i)); // W_N^{6k} * G imag
        __m256d h2r = FMSUB(hr, w7r, _mm256_mul_pd(hi, w7i)); // W_N^{7k} * H real
        __m256d h2i = FMADD(hi, w7r, _mm256_mul_pd(hr, w7i)); // W_N^{7k} * H imag

        // Compute pairwise sums and differences
        __m256d t0r = _mm256_add_pd(b2r, h2r), t0i = _mm256_add_pd(b2i, h2i); // B + H
        __m256d t1r = _mm256_add_pd(c2r, g2r), t1i = _mm256_add_pd(c2i, g2i); // C + G
        __m256d t2r = _mm256_add_pd(d2r, f2r), t2i = _mm256_add_pd(d2i, f2i); // D + F
        __m256d t3r = _mm256_sub_pd(b2r, h2r), t3i = _mm256_sub_pd(b2i, h2i); // B - H
        __m256d t4r = _mm256_sub_pd(c2r, g2r), t4i = _mm256_sub_pd(c2i, g2i); // C - G
        __m256d t5r = _mm256_sub_pd(d2r, f2r), t5i = _mm256_sub_pd(d2i, f2i); // D - F

        // Compute X_0: a + (b+h) + (c+g) + (d+f)
        __m256d sumr = _mm256_add_pd(ar, _mm256_add_pd(t0r, _mm256_add_pd(t1r, t2r)));
        __m256d sumi = _mm256_add_pd(ai, _mm256_add_pd(t0i, _mm256_add_pd(t1i, t2i)));
        STOREU_PD(out_re + k + 0 * M, sumr); // Store X(k) real
        STOREU_PD(out_im + k + 0 * M, sumi); // Store X(k) imag

        // Compute X_4: a - (b+h) - (c+g) + (d+f)
        __m256d x4r = _mm256_add_pd(_mm256_sub_pd(ar, _mm256_add_pd(t0r, t1r)), t2r);
        __m256d x4i = _mm256_add_pd(_mm256_sub_pd(ai, _mm256_add_pd(t0i, t1i)), t2i);
        STOREU_PD(out_re + k + 4 * M, x4r); // Store X(k + N/2) real
        STOREU_PD(out_im + k + 4 * M, x4i); // Store X(k + N/2) imag

        // Compute helpers for rotations
        __m256d tmp1r = _mm256_sub_pd(ar, t2r), tmp1i = _mm256_sub_pd(ai, t2i); // a - (d+f)
        __m256d diff0r = FMADD(_mm256_sub_pd(t0r, t2r), _mm256_set1_pd(C8_1), _mm256_mul_pd(vsign, _mm256_sub_pd(t3r, t5r))); // Rotation for X_1/X_7 real
        __m256d diff0i = FMADD(_mm256_sub_pd(t0i, t2i), _mm256_set1_pd(C8_1), _mm256_mul_pd(vsign, _mm256_sub_pd(t3i, t5i))); // Rotation for X_1/X_7 imag

        // Compute X_1 and X_7
        __m256d u1r = _mm256_add_pd(tmp1r, t1r), u1i = _mm256_add_pd(tmp1i, t1i); // Base term
        STOREU_PD(out_re + k + 1 * M, _mm256_sub_pd(u1r, diff0r));         // X(k + N/8) real
        STOREU_PD(out_im + k + 1 * M, _mm256_add_pd(u1i, diff0i));         // X(k + N/8) imag
        STOREU_PD(out_re + k + 7 * M, _mm256_add_pd(u1r, diff0r));         // X(k + 7N/8) real
        STOREU_PD(out_im + k + 7 * M, _mm256_sub_pd(u1i, diff0i));         // X(k + 7N/8) imag

        // Compute X_2 and X_6
        __m256d u2r = _mm256_add_pd(tmp1r, t0r), u2i = _mm256_add_pd(tmp1i, t0i); // Base term
        __m256d rot2r = FMADD(_mm256_sub_pd(t1i, t2i), _mm256_set1_pd(C8_1), _mm256_mul_pd(vsign, _mm256_sub_pd(t4i, t5i))); // Rotation real
        __m256d rot2i = FMADD(_mm256_sub_pd(t1r, t2r), _mm256_set1_pd(-C8_1), _mm256_mul_pd(vsign, _mm256_sub_pd(t4r, t5r))); // Rotation imag
        STOREU_PD(out_re + k + 2 * M, _mm256_sub_pd(u2r, rot2r));          // X(k + 2N/8) real
        STOREU_PD(out_im + k + 2 * M, _mm256_sub_pd(u2i, rot2i));          // X(k + 2N/8) imag
        STOREU_PD(out_re + k + 6 * M, _mm256_add_pd(u2r, rot2r));          // X(k + 6N/8) real
        STOREU_PD(out_im + k + 6 * M, _mm256_add_pd(u2i, rot2i));          // X(k + 6N/8) imag

        // Compute X_3 and X_5
        __m256d u3r = _mm256_add_pd(tmp1r, t5r), u3i = _mm256_add_pd(tmp1i, t5i); // Base term
        __m256d rot3r = FMADD(_mm256_add_pd(t3i, t4i), _mm256_set1_pd(C8_1), _mm256_mul_pd(vsign, _mm256_sub_pd(t0i, t2i))); // Rotation real
        __m256d rot3i = FMADD(_mm256_add_pd(t3r, t4r), _mm256_set1_pd(-C8_1), _mm256_mul_pd(vsign, _mm256_sub_pd(t0r, t2r))); // Rotation imag
        STOREU_PD(out_re + k + 3 * M, _mm256_sub_pd(u3r, rot3r));          // X(k + 3N/8) real
        STOREU_PD(out_im + k + 3 * M, _mm256_sub_pd(u3i, rot3i));          // X(k + 3N/8) imag
        STOREU_PD(out_re + k + 5 * M, _mm256_add_pd(u3r, rot3r));          // X(k + 5N/8) real
        STOREU_PD(out_im + k + 5 * M, _mm256_add_pd(u3i, rot3i));          // X(k + 5N/8) imag
    }

    // Step 5: Scalar tail for remaining k < M
    for (; k < M; k++)
    {
        /**
         * @brief Combine sub-FFT results for higher indices using twiddle factors (scalar).
         *
         * For frequency indices k not divisible by 4, applies twiddle factors to rotate and
         * combine the sub-FFT results from eight recursive calls. This mirrors the vectorized
         * logic but operates on single points. FMA emulation not used, as scalar benefits are minimal.
         *
         * Mathematically: Computes:
         *   \( X(k) = \sum_{n=0}^{7} X_n(k) \cdot W_N^{kn} \),
         * where \( W_N = e^{-2\pi i / N} \) for forward transforms, adjusted by transform_sign.
         */
        // Load eight sub-FFT points
        double a_r = out_re[k + 0 * M], a_i = out_im[k + 0 * M]; // First point
        double b_r = out_re[k + 1 * M], b_i = out_im[k + 1 * M]; // Second point
        double c_r = out_re[k + 2 * M], c_i = out_im[k + 2 * M]; // Third point
        double d_r = out_re[k + 3 * M], d_i = out_im[k + 3 * M]; // Fourth point
        double e_r = out_re[k + 4 * M], e_i = out_im[k + 4 * M]; // Fifth point
        double f_r = out_re[k + 5 * M], f_i = out_im[k + 5 * M]; // Sixth point
        double g_r = out_re[k + 6 * M], g_i = out_im[k + 6 * M]; // Seventh point
        double h_r = out_re[k + 7 * M], h_i = out_im[k + 7 * M]; // Eighth point

        // Load twiddle factors
        int base = (M - 1) + 7 * k;                          // Adjust index for scalar
        double w1r = tw_re[base + 0], w1i = tw_im[base + 0]; // W_N^k
        double w2r = tw_re[base + 1], w2i = tw_im[base + 1]; // W_N^{2k}
        double w3r = tw_re[base + 2], w3i = tw_im[base + 2]; // W_N^{3k}
        double w4r = tw_re[base + 3], w4i = tw_im[base + 3]; // W_N^{4k}
        double w5r = tw_re[base + 4], w5i = tw_im[base + 4]; // W_N^{5k}
        double w6r = tw_re[base + 5], w6i = tw_im[base + 5]; // W_N^{6k}
        double w7r = tw_re[base + 6], w7i = tw_im[base + 6]; // W_N^{7k}

        // Apply twiddle factors
        double b2r = b_r * w1r - b_i * w1i, b2i = b_i * w1r + b_r * w1i; // W_N^k * B
        double c2r = c_r * w2r - c_i * w2i, c2i = c_i * w2r + c_r * w2i; // W_N^{2k} * C
        double d2r = d_r * w3r - d_i * w3i, d2i = d_i * w3r + d_r * w3i; // W_N^{3k} * D
        double e2r = e_r * w4r - e_i * w4i, e2i = e_i * w4r + e_r * w4i; // W_N^{4k} * E
        double f2r = f_r * w5r - f_i * w5i, f2i = f_i * w5r + f_r * w5i; // W_N^{5k} * F
        double g2r = g_r * w6r - g_i * w6i, g2i = g_i * w6r + g_r * w6i; // W_N^{6k} * G
        double h2r = h_r * w7r - h_i * w7i, h2i = h_i * w7r + h_r * w7i; // W_N^{7k} * H

        // Compute pairwise sums and differences
        double t0r = b2r + h2r, t0i = b2i + h2i; // B + H
        double t1r = c2r + g2r, t1i = c2i + g2i; // C + G
        double t2r = d2r + f2r, t2i = d2i + f2i; // D + F
        double t3r = b2r - h2r, t3i = b2i - h2i; // B - H
        double t4r = c2r - g2r, t4i = c2i - g2i; // C - G
        double t5r = d2r - f2r, t5i = d2i - f2i; // D - F

        // Compute X_0 and X_4
        out_re[k + 0 * M] = a_r + t0r + t1r + t2r;   // X(k) real
        out_im[k + 0 * M] = a_i + t0i + t1i + t2i;   // X(k) imag
        out_re[k + 4 * M] = a_r - (t0r + t1r) + t2r; // X(k + N/2) real
        out_im[k + 4 * M] = a_i - (t0i + t1i) + t2i; // X(k + N/2) imag

        // Compute helpers
        double tmp1r = a_r - t2r, tmp1i = a_i - t2i; // a - (d+f)
        double tmp2r = t3r - t5r, tmp2i = t3i - t5i; // (b-h) - (d-f)

        // Compute X_1 and X_7
        {
            double u_r = tmp1r + t1r, u_i = tmp1i + t1i;               // Base term
            double rot_r = C8_1 * (t0i - t2i) + transform_sign * tmp2i;  // Rotation real
            double rot_i = -C8_1 * (t0r - t2r) + transform_sign * tmp2r; // Rotation imag
            out_re[k + 1 * M] = u_r - rot_r;                           // X(k + N/8) real
            out_im[k + 1 * M] = u_i - rot_i;                           // X(k + N/8) imag
            out_re[k + 7 * M] = u_r + rot_r;                           // X(k + 7N/8) real
            out_im[k + 7 * M] = u_i + rot_i;                           // X(k + 7N/8) imag
        }

        // Compute X_2 and X_6
        {
            double u_r = tmp1r + t0r, u_i = tmp1i + t0i;                     // Base term
            double rot_r = C8_1 * (t1i - t2i) + transform_sign * (t4i - t5i);  // Rotation real
            double rot_i = -C8_1 * (t1r - t2r) + transform_sign * (t4r - t5r); // Rotation imag
            out_re[k + 2 * M] = u_r - rot_r;                                 // X(k + 2N/8) real
            out_im[k + 2 * M] = u_i - rot_i;                                 // X(k + 2N/8) imag
            out_re[k + 6 * M] = u_r + rot_r;                                 // X(k + 6N/8) real
            out_im[k + 6 * M] = u_i + rot_i;                                 // X(k + 6N/8) imag
        }

        // Compute X_3 and X_5
        {
            double u_r = tmp1r + t5r, u_i = tmp1i + t5i;                     // Base term
            double rot_r = C8_1 * (t3i + t4i) + transform_sign * (t0i - t2i);  // Rotation real
            double rot_i = -C8_1 * (t3r + t4r) + transform_sign * (t0r - t2r); // Rotation imag
            out_re[k + 3 * M] = u_r - rot_r;                                 // X(k + 3N/8) real
            out_im[k + 3 * M] = u_i - rot_i;                                 // X(k + 3N/8) imag
            out_re[k + 5 * M] = u_r + rot_r;                                 // X(k + 5N/8) real
            out_im[k + 5 * M] = u_i + rot_i;                                 // X(k + 5N/8) imag
        }
    }

    // Step 6: Copy results back to output_buffer
    for (int lane = 0; lane < 8; lane++)
    {
        fft_data *base = output_buffer + lane * M; // Base pointer for lane
        for (int k = 0; k < M; k++)
        {
            base[k].re = out_re[lane * M + k]; // Copy real part back
            base[k].im = out_im[lane * M + k]; // Copy imag part back
        }
    }

    // Step 7: Clean up allocated memory
    free(tw_re);  // Free twiddle real array
    free(tw_im);  // Free twiddle imag array
    free(out_re); // Free output real array
    free(out_im); // Free output imag array
    }
    else
    {
        /**
         * @brief General radix decomposition for prime factors greater than 8.
         *
         * Handles arbitrary radices (prime factors > 8) using a generic decomposition strategy,
         * recursively dividing the data into radix sub-FFTs of length N/radix, applying the FFT to each,
         * and combining results using dynamically computed twiddle factors and trigonometric functions.
         * This extends the DIT strategy for non-standard radices, where \(X(k) = \sum_{n=0}^{radix-1} x(n) \cdot W_N^{kn}\),
         * with \(W_N = e^{-2\pi i / N}\) for forward transforms, adjusted by transform_sign.
         */
        int sub_length = data_length / radix, index;
        int mid_radix = (radix - 1) / 2, target, u, v, temp, temp_temp;
        fft_type temp1r, temp1i, temp2r, temp2i;
        fft_type *twiddle_real = (fft_type *)malloc((radix - 1) * sizeof(fft_type));
        fft_type *twiddle_imag = (fft_type *)malloc((radix - 1) * sizeof(fft_type));
        fft_type *tau_real = (fft_type *)malloc((radix - 1) * sizeof(fft_type));
        fft_type *tau_imag = (fft_type *)malloc((radix - 1) * sizeof(fft_type));
        fft_type *cos_values = (fft_type *)malloc((radix - 1) * sizeof(fft_type));
        fft_type *sin_values = (fft_type *)malloc((radix - 1) * sizeof(fft_type));
        fft_type *y_real = (fft_type *)malloc(radix * sizeof(fft_type));
        fft_type *y_imag = (fft_type *)malloc(radix * sizeof(fft_type));

        if (twiddle_real == NULL || twiddle_imag == NULL || tau_real == NULL || tau_imag == NULL ||
            cos_values == NULL || sin_values == NULL || y_real == NULL || y_imag == NULL)
        {
            fprintf(stderr, "Error: Memory allocation failed for general radix FFT arrays\n");
            // Free already allocated memory before exiting
            free(twiddle_real);
            free(twiddle_imag);
            free(tau_real);
            free(tau_imag);
            free(cos_values);
            free(sin_values);
            free(y_real);
            free(y_imag);
            exit(EXIT_FAILURE);
        }

        new_stride = radix * stride;
        for (int i = 0; i < radix; ++i)
        {
            /**
             * @brief Recursively compute sub-FFTs for each radix group.
             *
             * Performs recursive FFT on each of the radix sub-problems, dividing the data into
             * `radix` groups of length `sub_length`. This step reduces the problem size,
             * enabling efficient computation via divide-and-conquer.
             */
            mixed_radix_dit_rec(output_buffer + i * sub_length, input_buffer + i * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        }

        // Precompute cosine and sine values for the general radix
        for (int i = 1; i < mid_radix + 1; ++i)
        {
            /**
             * @brief Precompute trigonometric values for twiddle factor synthesis.
             *
             * Calculates cosine and sine values for angles corresponding to \(k \cdot 2\pi / radix\),
             * used to dynamically generate twiddle factors for the general radix butterfly.
             * These values represent the real and imaginary components of \(e^{-2\pi i k / radix}\).
             */
            cos_values[i - 1] = cos(i * PI2 / radix); // Real part of twiddle factor
            sin_values[i - 1] = sin(i * PI2 / radix); // Imaginary part of twiddle factor
        }
        for (int i = 0; i < mid_radix; ++i)
        {
            sin_values[i + mid_radix] = -sin_values[mid_radix - 1 - i]; // Mirror sine values for symmetry
            cos_values[i + mid_radix] = cos_values[mid_radix - 1 - i];  // Mirror cosine values for symmetry
        }

        for (int k = 0; k < sub_length; ++k)
        {
            /**
             * @brief Combine sub-FFT results for each frequency index using general radix butterflies.
             *
             * For each frequency index k, loads sub-FFT results, applies twiddle factors, and performs
             * a general radix butterfly to combine the results. This uses dynamically computed twiddle
             * factors and trigonometric functions to handle arbitrary radices, ensuring correct phase
             * alignment of frequency components. Mathematically, this implements \(X(k) = \sum_{n=0}^{radix-1} x(n) \cdot W_N^{kn}\),
             * where \(W_N = e^{-2\pi i / N}\) for forward transforms, adjusted by transform_sign.
             */
            index = sub_length - 1 + (radix - 1) * k; // Index into twiddle factors for current frequency
            y_real[0] = output_buffer[k].re;          // Save first sub-FFT real part
            y_imag[0] = output_buffer[k].im;          // Save first sub-FFT imaginary part
            for (int i = 0; i < radix - 1; ++i)
            {
                twiddle_real[i] = (fft_obj->twiddle + index)->re;                                                        // Real part of twiddle factor
                twiddle_imag[i] = (fft_obj->twiddle + index)->im;                                                        // Imaginary part of twiddle factor
                target = k + (i + 1) * sub_length;                                                                       // Target for subsequent sub-FFT results
                y_real[i + 1] = output_buffer[target].re * twiddle_real[i] - output_buffer[target].im * twiddle_imag[i]; // Apply twiddle (real)
                y_imag[i + 1] = output_buffer[target].im * twiddle_real[i] + output_buffer[target].re * twiddle_imag[i]; // Apply twiddle (imag)
                index++;
            }

            for (int i = 0; i < mid_radix; ++i)
            {
                tau_real[i] = y_real[i + 1] + y_real[radix - 1 - i];             // Sum pairs of rotated outputs (real)
                tau_imag[i + mid_radix] = y_imag[i + 1] - y_imag[radix - 1 - i]; // Difference for higher indices (imag)
                tau_imag[i] = y_imag[i + 1] + y_imag[radix - 1 - i];             // Sum pairs of rotated outputs (imag)
                tau_real[i + mid_radix] = y_real[i + 1] - y_real[radix - 1 - i]; // Difference for higher indices (real)
            }

            temp1r = y_real[0]; // Initialize with first real component
            temp1i = y_imag[0]; // Initialize with first imaginary component
            for (int i = 0; i < mid_radix; ++i)
            {
                temp1r += tau_real[i]; // Sum real components
                temp1i += tau_imag[i]; // Sum imaginary components
            }

            output_buffer[k].re = temp1r; // Store combined real part for current frequency
            output_buffer[k].im = temp1i; // Store combined imaginary part for current frequency

            for (int u = 0; u < mid_radix; u++)
            {
                temp1r = y_real[0]; // Reset real component
                temp1i = y_imag[0]; // Reset imaginary component
                temp2r = 0.0;       // Initialize real rotation component
                temp2i = 0.0;       // Initialize imaginary rotation component
                for (int v = 0; v < mid_radix; v++)
                {
                    temp = (u + 1) * (v + 1); // Compute index for twiddle factor
                    while (temp >= radix)
                    {
                        temp -= radix; // Normalize index modulo radix
                    }
                    temp_temp = temp - 1; // Adjust for zero-based indexing

                    temp1r += cos_values[temp_temp] * tau_real[v];             // Apply cosine for real part
                    temp1i += cos_values[temp_temp] * tau_imag[v];             // Apply cosine for imaginary part
                    temp2r -= sin_values[temp_temp] * tau_real[v + mid_radix]; // Apply negative sine for real rotation
                    temp2i -= sin_values[temp_temp] * tau_imag[v + mid_radix]; // Apply negative sine for imaginary rotation
                }
                temp2r = transform_sign * temp2r; // Adjust rotation by transform direction (real)
                temp2i = transform_sign * temp2i; // Adjust rotation by transform direction (imag)

                output_buffer[k + (u + 1) * sub_length].re = temp1r - temp2i;         // Store rotated real part for positive index
                output_buffer[k + (u + 1) * sub_length].im = temp1i + temp2r;         // Store rotated imaginary part for positive index
                output_buffer[k + (radix - u - 1) * sub_length].re = temp1r + temp2i; // Store rotated real part for negative index
                output_buffer[k + (radix - u - 1) * sub_length].im = temp1i - temp2r; // Store rotated imaginary part for negative index
            }
        }

        // Free allocated memory
        free(twiddle_real);
        free(twiddle_imag);
        free(tau_real);
        free(tau_imag);
        free(cos_values);
        free(sin_values);
        free(y_real);
        free(y_imag);
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
        exit(EXIT_FAILURE);
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
    if (signal_length <= 0) {
        fprintf(stderr, "Error: Signal length (%d) is invalid\n", signal_length);
        exit(EXIT_FAILURE);
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
    if (4 * padded_length > fft_config->max_scratch_size) {
        fprintf(stderr, "Error: Scratch buffer too small for Bluestein (need %d, have %d)\n",
                4 * padded_length, fft_config->max_scratch_size);
        exit(EXIT_FAILURE);
    }

    // Set up a temporary FFT object for the padded length (M)
    // Why not reuse fft_config? Modifying it could mess things up in multi-threaded code,
    // so we play it safe with a fresh object, even if it’s a bit of extra work
    fft_object temp_config = fft_init(padded_length, transform_direction);
    if (!temp_config) {
        fprintf(stderr, "Error: Couldn’t create temporary FFT object\n");
        exit(EXIT_FAILURE);
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
    for (int i = 0; i < padded_length; ++i) {
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
    for (; n + 3 < signal_length; n += 4) {
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
        if (transform_direction == -1) {
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
    for (; n < signal_length; ++n) {
        double input_re = input_signal[n].re, input_im = input_signal[n].im;
        double chirp_re = chirp_sequence[n].re, chirp_im = chirp_sequence[n].im;
        if (transform_direction == -1) chirp_im = -chirp_im;
        temp_chirp[n].re = input_re * chirp_re - input_im * chirp_im;
        temp_chirp[n].im = input_im * chirp_re + input_re * chirp_im;
    }

    // Zero-pad the chirped signal to length M
    // This sets up the convolution by ensuring no aliasing in the frequency domain
    // Choice: Explicit loop is clear and fast enough; could’ve used memset but this is more readable
    for (int i = signal_length; i < padded_length; ++i) {
        temp_chirp[i].re = 0.0;
        temp_chirp[i].im = 0.0;
    }

    // Run FFT on the padded chirped signal
    // We’re storing the result in ifft_result to keep things organized
    fft_exec(temp_config, temp_chirp, ifft_result);

    // Pointwise multiplication in the frequency domain (y(n) * h_k(n) for forward, y(n) * h_k^*(n) for inverse)
    // Again, AVX2 for speed, processing 4 complex points at a time
    n = 0;
    for (; n + 3 < padded_length; n += 4) {
        // Load FFT results and chirp FFT
        // Using aligned loads since ifft_result and chirped_signal are in our aligned scratch buffer
        __m256d fft_re = _mm256_load_pd(&ifft_result[n].re);
        __m256d fft_im = _mm256_load_pd(&ifft_result[n].im);
        __m256d chirp_fft_re = _mm256_load_pd(&chirped_signal[n].re);
        __m256d chirp_fft_im = _mm256_load_pd(&chirped_signal[n].im);

        // Conjugate chirp FFT for inverse transform
        // Doing it on-the-fly saves memory and keeps things flexible
        if (transform_direction == -1) {
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
    for (; n < padded_length; ++n) {
        double fft_re = ifft_result[n].re, fft_im = ifft_result[n].im;
        double chirp_fft_re = chirped_signal[n].re, chirp_fft_im = chirped_signal[n].im;
        if (transform_direction == -1) chirp_fft_im = -chirp_fft_im;
        temp_chirp[n].re = fft_re * chirp_fft_re - fft_im * chirp_fft_im;
        temp_chirp[n].im = fft_im * chirp_fft_re + fft_re * chirp_fft_im;
    }

    // Flip the twiddle factors for the inverse FFT
    // We’re modifying temp_config’s twiddles to avoid messing with the original fft_config
    for (int i = 0; i < padded_length; ++i) {
        temp_config->twiddles[i].im = -temp_config->twiddles[i].im;
    }
    temp_config->sgn = -transform_direction;

    // Run the inverse FFT to get the convolution result
    fft_exec(temp_config, temp_chirp, ifft_result);

    // Final step: multiply by the chirp sequence again to extract the DFT
    // Same AVX2 approach as before, with aligned loads for chirp_sequence
    n = 0;
    for (; n + 3 < signal_length; n += 4) {
        __m256d ifft_re = _mm256_load_pd(&ifft_result[n].re);
        __m256d ifft_im = _mm256_load_pd(&ifft_result[n].im);
        __m256d chirp_re = _mm256_load_pd(&chirp_sequence[n].re);
        __m256d chirp_im = _mm256_load_pd(&chirp_sequence[n].im);

        if (transform_direction == -1) {
            chirp_im = _mm256_sub_pd(_mm256_setzero_pd(), chirp_im);
        }

        __m256d result_re = _mm256_fmsub_pd(ifft_re, chirp_re, _mm256_mul_pd(ifft_im, chirp_im));
        __m256d result_im = _mm256_fmadd_pd(ifft_im, chirp_re, _mm256_mul_pd(ifft_re, chirp_im));

        // Store final results in output_signal (unaligned, as we don’t control its alignment)
        _mm256_storeu_pd(&output_signal[n].re, result_re);
        _mm256_storeu_pd(&output_signal[n].im, result_im);
    }
    // Scalar tail for the last few points
    for (; n < signal_length; ++n) {
        double ifft_re = ifft_result[n].re, ifft_im = ifft_result[n].im;
        double chirp_re = chirp_sequence[n].re, chirp_im = chirp_sequence[n].im;
        if (transform_direction == -1) chirp_im = -chirp_im;
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
void fft_exec(fft_object fft_obj, fft_data *inp, fft_data *oup) {
    // Check for null pointers to avoid crashes
    // Ensure we have a valid FFT object and input/output buffers
    if (fft_obj == NULL || inp == NULL || oup == NULL) {
        fprintf(stderr, "Error: Invalid FFT object or data pointers\n");
        exit(EXIT_FAILURE);
    }

    // Dispatch based on the FFT algorithm type
    // lt = 0 for mixed-radix (factorable lengths), lt = 1 for Bluestein (non-factorable)
    if (fft_obj->lt == 0) {
        // Set up for mixed-radix FFT
        // Start with stride=1, factor_index=0, and scratch_offset=0 for recursion
        int stride = 1;          // Initial stride for input indexing
        int factor_index = 0;    // Start at the first prime factor
        int scratch_offset = 0;  // Initial offset for scratch buffer
        // Call mixed-radix FFT with transform length (n_fft)
        // Why n_fft? It’s the actual transform size (N for mixed-radix)
        mixed_radix_dit_rec(oup, inp, fft_obj, fft_obj->sgn,
                            fft_obj->n_fft, stride, factor_index, scratch_offset);
    } else if (fft_obj->lt == 1) {
        // Run Bluestein FFT for non-factorable lengths
        // Use n_input for input/output size, as Bluestein pads internally
        bluestein_fft(inp, oup, fft_obj, fft_obj->sgn, fft_obj->n_input);
    } else {
        // Handle invalid algorithm type
        // This shouldn’t happen unless fft_init is broken
        fprintf(stderr, "Error: Invalid FFT object type (lt = %d)\n", fft_obj->lt);
        exit(EXIT_FAILURE);
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
        exit(EXIT_FAILURE);
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
        exit(EXIT_FAILURE);
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
        exit(EXIT_FAILURE);
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
        exit(EXIT_FAILURE);
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

void free_fft(fft_object object) {
    if (object) {
        if (object->twiddles) _mm_free(object->twiddles);
        if (object->scratch) _mm_free(object->scratch);
        if (object->twiddle_factors) _mm_free(object->twiddle_factors);
        free(object);
    }
}