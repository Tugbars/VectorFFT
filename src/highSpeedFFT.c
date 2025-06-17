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
 * @brief Initializes precomputed chirp sequences at program startup.
 *
 * Computes chirp sequences h(n) = e^(πi n^2 / N) for a select set of N values using high-precision math.
 * Runs automatically before main() due to the constructor attribute.
 */
__attribute__((constructor)) static void init_bluestein_chirp(void)
{
    if (chirp_initialized)
        return; // Prevent re-initialization

    int sizes[] = {1, 2, 3, 4, 5, 7, 15, 20, 31, 64}; // Precompute sizes including 64 for test case
    num_precomputed = sizeof(sizes) / sizeof(sizes[0]);

    bluestein_chirp = malloc(num_precomputed * sizeof(fft_data *));
    chirp_sizes = malloc(num_precomputed * sizeof(int));
    if (!bluestein_chirp || !chirp_sizes)
    {
        fprintf(stderr, "Error: Memory allocation failed for Bluestein chirp table\n");
        exit(EXIT_FAILURE);
    }

    for (int idx = 0; idx < num_precomputed; idx++)
    {
        int n = sizes[idx];
        chirp_sizes[idx] = n;
        bluestein_chirp[idx] = malloc(n * sizeof(fft_data));
        if (!bluestein_chirp[idx])
        {
            fprintf(stderr, "Error: Memory allocation failed for chirp size %d\n", n);
            exit(EXIT_FAILURE);
        }
        fft_type theta = M_PI / n; // Base angle step: π/N
        int l2 = 0, len2 = 2 * n;  // Quadratic index and wrap-around limit
        for (int i = 0; i < n; i++)
        {
            fft_type angle = theta * l2;             // Angle: π * n^2 / N
            bluestein_chirp[idx][i].re = cos(angle); // Real part
            bluestein_chirp[idx][i].im = sin(angle); // Imaginary part
            l2 += 2 * i + 1;                         // Quadratic term: n^2 mod 2N
            while (l2 > len2)
                l2 -= len2; // Wrap around modulo 2N
        }
    }
    chirp_initialized = 1;
}

/**
 * @brief Initializes an FFT object for computing the Fast Fourier Transform (FFT).
 *
 * Allocates and configures an FFT object based on the signal length and transform direction.
 * Chooses between a mixed-radix FFT for signal lengths divisible by small primes or Bluestein’s
 * algorithm for arbitrary lengths, precomputing twiddle factors accordingly.
 *
 * @param[in] signal_length Length of the input signal (must be positive, \(N > 0\)).
 * @param[in] transform_direction Direction of the transform (\(+1\) for forward FFT, \(-1\) for inverse FFT).
 * @return fft_object Pointer to the initialized FFT object, or NULL if allocation fails or inputs are invalid.
 * @warning No explicit validation of inputs; assumes \(N > 0\) and \(sgn = \pm 1\). Invalid inputs may lead to undefined behavior.
 * @note The function allocates memory for the FFT object and twiddle factors using `malloc`. The caller is responsible
 *       for freeing this memory with `free_fft`. Twiddle factors are adjusted for inverse transforms by negating their
 *       imaginary components.
 */
fft_object fft_init(int signal_length, int transform_direction)
{
    fft_object fft_config = NULL; // Pointer to the FFT configuration object
    int twiddle_count;            // Number of twiddle factors to compute
    int is_factorable;            // Flag indicating if signal_length is factorable by supported primes

    // Check if the signal length can be factored into supported primes (mixed-radix case)
    is_factorable = dividebyN(signal_length);

    if (is_factorable)
    {
        // Mixed-radix FFT: Signal length is a product of supported primes (e.g., 2, 3, 4, 5, 7, 8)
        // Allocate memory for the fft_set structure plus (N-1) twiddle factors
        fft_config = (fft_object)malloc(sizeof(struct fft_set) + sizeof(fft_data) * (signal_length - 1));
        if (fft_config == NULL)
        {
            return NULL; // Memory allocation failed
        }

        // Factor the signal length into primes and store in factors array
        fft_config->lf = factors(signal_length, fft_config->factors);

        // Compute twiddle factors for mixed-radix decomposition
        longvectorN(fft_config->twiddle, signal_length, fft_config->factors, fft_config->lf);

        // Set twiddle count and algorithm type
        twiddle_count = signal_length;
        fft_config->lt = 0; // 0 indicates mixed-radix algorithm
    }
    else
    {
        // Bluestein’s algorithm: Signal length is not factorable, requires padding to a power of 2
        int next_power_of_2; // Smallest power of 2 >= N for initial estimation
        int padded_length;   // Final padded length for Bluestein’s convolution

        // Estimate the next power of 2 using log10 for simplicity (log2 would be more precise)
        next_power_of_2 = (int)pow(2.0, ceil(log10(signal_length) / log10(2.0)));

        // Ensure padded length is sufficient for Bluestein’s convolution (at least 2N-1)
        if (next_power_of_2 < 2 * signal_length - 2)
        {
            padded_length = next_power_of_2 * 2; // Double to ensure enough space
        }
        else
        {
            padded_length = next_power_of_2; // Already sufficient
        }

        // Allocate memory for the fft_set structure plus (M-1) twiddle factors
        fft_config = (fft_object)malloc(sizeof(struct fft_set) + sizeof(fft_data) * (padded_length - 1));
        if (fft_config == NULL)
        {
            return NULL; // Memory allocation failed
        }

        // Factor the padded length into primes (typically powers of 2)
        fft_config->lf = factors(padded_length, fft_config->factors);

        // Compute twiddle factors for the padded length
        longvectorN(fft_config->twiddle, padded_length, fft_config->factors, fft_config->lf);

        // Set twiddle count and algorithm type
        twiddle_count = padded_length;
        fft_config->lt = 1; // 1 indicates Bluestein’s algorithm
    }

    // Store the original signal length and transform direction
    fft_config->N = signal_length;
    fft_config->sgn = transform_direction;

    // Adjust twiddle factors for inverse FFT by negating imaginary parts
    if (transform_direction == -1)
    {
        for (int i = 0; i < twiddle_count; i++)
        {
            fft_config->twiddle[i].im = -fft_config->twiddle[i].im;
        }
    }

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
static void mixed_radix_dit_rec(fft_data *output_buffer, fft_data *input_buffer, const fft_object fft_obj, int transform_sign, int data_length, int stride, int factor_index)
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
         * @brief Radix-2 decomposition for two-point sub-FFTs with AVX2 vectorization and optional FMA support.
         *
         * Intention: Efficiently compute the FFT for data lengths divisible by 2 by splitting the problem
         * into two smaller sub-FFTs (even and odd indices), applying the FFT recursively, and combining
         * results with twiddle factors. This is the core of the decimation-in-time (DIT) strategy for
         * powers of 2 (N=2^r). Optimized with FMA instructions when available.
         *
         * Process:
         * 1. Divide the data into two subproblems of size N/2, adjusting stride for indexing.
         * 2. Recursively compute FFTs for even and odd indices.
         * 3. Pre-build contiguous twiddle arrays for efficient SIMD access.
         * 4. Flatten output_buffer into contiguous real/imag arrays to avoid interleaved memory issues.
         * 5. Perform butterfly operations using AVX2 with FMA (via FMADD/FMSUB macros) for k divisible by 4,
         *    with a scalar tail for remaining k.
         * 6. Copy results back to output_buffer.
         * 7. Clean up allocated memory.
         *
         * Optimization:
         * - AVX2 vectorization processes four complex numbers simultaneously.
         * - FMA instructions (FMADD/FMSUB) reduce instruction count and improve numerical accuracy
         *   when compiled with -mfma or USE_FMA defined (requires Intel Haswell+ or AMD Zen+).
         * - Without FMA, inlined fallback functions (ALWAYS_INLINE) avoid call overhead.
         * - AVX_ONE macro inlines _mm256_set1_pd(1.0), avoiding storage and initializer issues; relies on
         *   compiler optimization (CSE) to avoid redundant vector creation.
         * - Precomputed twiddle factors avoid runtime trigonometric calls.
         * - Contiguous twiddle and output arrays ensure efficient SIMD loads/stores.
         * - Butterfly operations exploit symmetry to minimize computations.
         *
         * Transform sign: Determines the twiddle factor phase (forward: negative exponent, inverse:
         * positive exponent), handled via precomputed twiddle factors adjusted during fft_init.
         *
         * Note: Uses unaligned AVX2 loads/stores (LOADU_PD/STOREU_PD) as malloc doesn’t guarantee
         * 32-byte alignment. Future optimization may use posix_memalign with USE_ALIGNED_SIMD to enable
         * _mm256_load_pd/_mm256_store_pd. Twiddle and output arrays are allocated per call; pre-allocation
         * in fft_obj is planned to avoid malloc/free overhead. Runtime FMA detection (e.g.,
         * __builtin_cpu_supports or _cpuid) could be added for dynamic dispatch but is omitted for performance.
         *
         *
         * @warning Assumes fft_obj->twiddle has at least N-1 elements, where N = data_length,
         * to support indices up to sub_length-1 + (M-1). If undersized, undefined behavior may occur.
         */
        // Step 1: Compute subproblem size and new stride
        int sub_length = data_length / 2; // Size of each sub-FFT (N/2)
        int new_stride = 2 * stride;      // Stride doubles to access every other element
        int M = sub_length;               // Sub-FFT size for indexing

        // Step 2: Recurse on even- and odd-indexed halves
        mixed_radix_dit_rec(output_buffer, input_buffer,
                            fft_obj, transform_sign, sub_length,
                            new_stride, factor_index + 1); // Even indices
        mixed_radix_dit_rec(output_buffer + sub_length, input_buffer + stride,
                            fft_obj, transform_sign, sub_length,
                            new_stride, factor_index + 1); // Odd indices

        // Step 3: Pre-build contiguous twiddle arrays
        double *tw_re_contig = malloc(M * sizeof(double)); // Contiguous real twiddles
        double *tw_im_contig = malloc(M * sizeof(double)); // Contiguous imag twiddles
        if (!tw_re_contig || !tw_im_contig) {
            fprintf(stderr, "Error: Memory allocation failed for twiddle arrays\n");
            free(tw_re_contig);
            free(tw_im_contig);
            exit(EXIT_FAILURE);
        }
        if (fft_obj->N - 1 < 2 * M - 1) {
            fprintf(stderr, "Error: Twiddle array too small (need %d elements, have %d)\n", 2 * M - 1, fft_obj->N - 1);
            free(tw_re_contig);
            free(tw_im_contig);
            exit(EXIT_FAILURE);
        }
        for (int k = 0; k < M; k++) {
            int idx = sub_length - 1 + k; // fft_obj->twiddle[idx] = W_N^k
            tw_re_contig[k] = fft_obj->twiddle[idx].re;
            tw_im_contig[k] = fft_obj->twiddle[idx].im;
        }

        // Step 4: Flatten output_buffer into contiguous real/imag arrays
        double *out_re = malloc(2 * M * sizeof(double)); // Contiguous real outputs
        double *out_im = malloc(2 * M * sizeof(double)); // Contiguous imag outputs
        if (!out_re || !out_im) {
            fprintf(stderr, "Error: Memory allocation failed for output arrays\n");
            free(tw_re_contig);
            free(tw_im_contig);
            free(out_re);
            free(out_im);
            exit(EXIT_FAILURE);
        }
        for (int lane = 0; lane < 2; lane++) {
            fft_data *base = output_buffer + lane * M;
            for (int k = 0; k < M; k++) {
                out_re[lane * M + k] = base[k].re;
                out_im[lane * M + k] = base[k].im;
            }
        }

        // Step 5: Perform AVX2-accelerated butterfly operations with FMA
        int k = 0;

        // Vectorized loop: Process 4 complex numbers at a time using AVX2
        for (; k + 3 < M; k += 4) {
            __m256d er = LOADU_PD(out_re + k);     // Load even sub-FFT real parts
            __m256d ei = LOADU_PD(out_im + k);     // Load even sub-FFT imag parts
            __m256d br = LOADU_PD(out_re + k + M); // Load odd sub-FFT real parts
            __m256d bi = LOADU_PD(out_im + k + M); // Load odd sub-FFT real parts
            __m256d wre = LOADU_PD(tw_re_contig + k); // Load twiddle real parts
            __m256d wim = LOADU_PD(tw_im_contig + k); // Load twiddle imag parts

            // Twiddle multiply
            __m256d tr = FMSUB(br, wre, _mm256_mul_pd(bi, wim)); // tr = br*wre - bi*wim
            __m256d ti = FMADD(bi, wre, _mm256_mul_pd(br, wim)); // ti = bi*wre + br*wim

            // Butterfly: X(k) = even + twiddled_odd, X(k+N/2) = even - twiddled_odd
            STOREU_PD(out_re + k, FMADD(tr, AVX_ONE, er)); // X(k) real = er + tr
            STOREU_PD(out_im + k, FMADD(ti, AVX_ONE, ei)); // X(k) imag = ei + ti
            STOREU_PD(out_re + k + M, FMSUB(er, AVX_ONE, tr)); // X(k+N/2) real = er - tr
            STOREU_PD(out_im + k + M, FMSUB(ei, AVX_ONE, ti)); // X(k+N/2) imag = ei - ti
        }

        // Scalar tail: Handle remaining elements not divisible by 4
        for (; k < M; ++k) {
            double ar = out_re[k], ai = out_im[k];           // Even sub-FFT point
            double br_ = out_re[k + M], bi_ = out_im[k + M]; // Odd sub-FFT point
            double tr = br_ * tw_re_contig[k] - bi_ * tw_im_contig[k]; // Twiddle mult real
            double ti = bi_ * tw_re_contig[k] + br_ * tw_im_contig[k]; // Twiddle mult imag
            out_re[k] = ar + tr;                             // X(k) real
            out_im[k] = ai + ti;                             // X(k) imag
            out_re[k + M] = ar - tr;                         // X(k+N/2) real
            out_im[k + M] = ai - ti;                         // X(k+N/2) imag
        }

        // Step 6: Copy results back to output_buffer
        for (int lane = 0; lane < 2; lane++) {
            fft_data *base = output_buffer + lane * M;
            for (int k = 0; k < M; k++) {
                base[k].re = out_re[lane * M + k];
                base[k].im = out_im[lane * M + k];
            }
        }

        // Step 7: Clean up allocated memory
        free(tw_re_contig);
        free(tw_im_contig);
        free(out_re);
        free(out_im);
    }
    else if (radix == 3)
    {
        /**
         * @brief Radix-3 decomposition for three-point sub-FFTs.
         *
         * Intention: Compute the FFT for data lengths divisible by 3 by splitting into three sub-FFTs,
         * corresponding to indices n mod 3 = 0, 1, 2, and combining results with twiddle factors.
         * This extends the DIT strategy for N=3^r or mixed-radix cases.
         *
         * Mathematically: The FFT is computed as:
         *   \( X(k) = X_0(k) + W_N^k \cdot X_1(k) + W_N^{2k} \cdot X_2(k) \),
         * where \( X_0, X_1, X_2 \) are sub-FFTs of size N/3, and \( W_N^k = e^{-2\pi i k / N} \).
         * The butterfly operation uses the 120° and 240° roots of unity, with √3/2 for rotations.
         *
         * Process:
         * 1. Divide data into three subproblems of size N/3, adjusting stride.
         * 2. Recursively compute FFTs for each subproblem.
         * 3. Combine results using a radix-3 butterfly, applying twiddle factors W_N^k and W_N^{2k}.
         * 4. For each k, compute sums and differences, apply 120° rotations using √3/2, and store results.
         *
         * Optimization:
         * - Precomputed √3/2 ≈ 0.8660254037844386 avoids runtime trigonometric calculations.
         * - Twiddle factors are fetched from fft_obj->twiddle, precomputed during initialization.
         * - Symmetry in outputs (e.g., X_1 and X_2 share computations) reduces operations.
         *
         * Transform sign: Affects the rotation direction (forward: negative phase, inverse: positive phase),
         * applied via transform_sign * √3/2 in the butterfly.
         */
        // Step 1: Compute subproblem size and new stride
        int sub_length = data_length / 3; // Size of each sub-FFT (N/3)
        new_stride = 3 * stride;          // Stride triples to access every third element

        // Step 2: Recurse on the three decimated sub-FFTs
        mixed_radix_dit_rec(output_buffer, input_buffer, fft_obj, transform_sign, sub_length,
                            new_stride, factor_index + 1); // n mod 3 = 0
        mixed_radix_dit_rec(output_buffer + sub_length, input_buffer + stride, fft_obj,
                            transform_sign, sub_length, new_stride, factor_index + 1); // n mod 3 = 1
        mixed_radix_dit_rec(output_buffer + 2 * sub_length, input_buffer + 2 * stride, fft_obj,
                            transform_sign, sub_length, new_stride, factor_index + 1); // n mod 3 = 2

        // Step 3: Perform radix-3 butterfly operations
        const fft_type sqrt3by2 = (fft_type)0.8660254037844386; // √3/2 for 120° rotation
        for (int k = 0; k < sub_length; k++)
        {
            // Fetch twiddle factors W_N^k and W_N^{2k}
            int idx1 = sub_length - 1 + 2 * k;            // Index into twiddle array
            fft_type w1r = fft_obj->twiddle[idx1 + 0].re; // W_N^k real
            fft_type w1i = fft_obj->twiddle[idx1 + 0].im; // W_N^k imag
            fft_type w2r = fft_obj->twiddle[idx1 + 1].re; // W_N^{2k} real
            fft_type w2i = fft_obj->twiddle[idx1 + 1].im; // W_N^{2k} imag

            // Load the three partial FFT results
            fft_data *X0 = &output_buffer[k];                  // First sub-FFT point
            fft_data *X1 = &output_buffer[k + sub_length];     // Second sub-FFT point
            fft_data *X2 = &output_buffer[k + 2 * sub_length]; // Third sub-FFT point

            fft_type a_re = X0->re, a_im = X0->im; // Store first point
            // Apply twiddle factors to second and third points
            fft_type b_re = X1->re * w1r - X1->im * w1i; // W_N^k * X_1 real
            fft_type b_im = X1->im * w1r + X1->re * w1i; // W_N^k * X_1 imag
            fft_type c_re = X2->re * w2r - X2->im * w2i; // W_N^{2k} * X_2 real
            fft_type c_im = X2->im * w2r + X2->re * w2i; // W_N^{2k} * X_2 imag

            // Compute the radix-3 butterfly
            fft_type sum_re = b_re + c_re;  // Sum of twiddled points (real)
            fft_type sum_im = b_im + c_im;  // Sum of twiddled points (imag)
            fft_type diff_re = b_re - c_re; // Difference (real)
            fft_type diff_im = b_im - c_im; // Difference (imag)

            // X_0 = a + (b + c)
            X0->re = a_re + sum_re; // First output real
            X0->im = a_im + sum_im; // First output imag

            // X_1 = a - ½(b + c) + i * (sign * (√3/2) * (b - c))
            fft_type t_re = a_re - (sum_re * (fft_type)0.5);          // Center term real
            fft_type t_im = a_im - (sum_im * (fft_type)0.5);          // Center term imag
            fft_type rot_re = diff_im * (transform_sign * sqrt3by2);  // Rotation real
            fft_type rot_im = -diff_re * (transform_sign * sqrt3by2); // Rotation imag
            X1->re = t_re + rot_re;                                   // Second output real
            X1->im = t_im + rot_im;                                   // Second output imag

            // X_2 = a - ½(b + c) - i * (sign * (√3/2) * (b - c))
            X2->re = t_re - rot_re; // Third output real
            X2->im = t_im - rot_im; // Third output imag
        }
    }
    else if (radix == 4)
    {
        /**
         * @brief Radix-4 decomposition for four-point sub-FFTs.
         *
         * Intention: Optimize FFT computation for data lengths divisible by 4 by splitting into
         * four sub-FFTs, corresponding to indices n mod 4 = 0, 1, 2, 3, and combining results
         * with twiddle factors. This is highly efficient for N=4^r or mixed-radix cases.
         *
         * Mathematically: The FFT is computed as:
         *   \( X(k) = X_0(k) + W_N^k \cdot X_1(k) + W_N^{2k} \cdot X_2(k) + W_N^{3k} \cdot X_3(k) \),
         * where \( X_0, X_1, X_2, X_3 \) are sub-FFTs of size N/4, and \( W_N^k = e^{-2\pi i k / N} \).
         * The radix-4 butterfly uses 90°, 180°, and 270° rotations.
         *
         * Process:
         * 1. Divide data into four subproblems of size N/4, adjusting stride.
         * 2. Recursively compute FFTs for each subproblem.
         * 3. For k=0, perform a butterfly without twiddles (simplified operations).
         * 4. For k>0, apply twiddle factors W_N^k, W_N^{2k}, W_N^{3k}, and compute the butterfly.
         *
         * Optimization:
         * - Handles k=0 separately to avoid unnecessary twiddle multiplications.
         * - Uses precomputed twiddle factors to minimize trigonometric calculations.
         * - Exploits symmetry in the butterfly (e.g., 180° rotation is negation).
         * - Reduces memory accesses by operating in-place.
         *
         * Transform sign: Affects the phase of rotations for odd branches (forward: negative,
         * inverse: positive), applied via transform_sign in the butterfly.
         */
        // Step 1: Compute subproblem size and new stride
        int sub_length = data_length / 4; // Size of each sub-FFT (N/4)
        new_stride = 4 * stride;          // Stride quadruples to access every fourth element

        // Step 2: Recurse on the four decimated sub-FFTs
        mixed_radix_dit_rec(output_buffer, input_buffer, fft_obj, transform_sign, sub_length,
                            new_stride, factor_index + 1); // n mod 4 = 0
        mixed_radix_dit_rec(output_buffer + sub_length, input_buffer + stride, fft_obj,
                            transform_sign, sub_length, new_stride, factor_index + 1); // n mod 4 = 1
        mixed_radix_dit_rec(output_buffer + 2 * sub_length, input_buffer + 2 * stride, fft_obj,
                            transform_sign, sub_length, new_stride, factor_index + 1); // n mod 4 = 2
        mixed_radix_dit_rec(output_buffer + 3 * sub_length, input_buffer + 3 * stride, fft_obj,
                            transform_sign, sub_length, new_stride, factor_index + 1); // n mod 4 = 3

        // Step 3: Handle k=0 separately (no twiddles needed)
        {
            fft_data *X0 = &output_buffer[0];              // First sub-FFT point
            fft_data *X1 = &output_buffer[sub_length];     // Second sub-FFT point
            fft_data *X2 = &output_buffer[2 * sub_length]; // Third sub-FFT point
            fft_data *X3 = &output_buffer[3 * sub_length]; // Fourth sub-FFT point

            // Simple radix-4 butterfly without twiddles
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

        // Step 4: Handle k = 1 to sub_length-1 with twiddle factors
        for (int k = 1; k < sub_length; k++)
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
    }
    else if (radix == 5)
    {
        /**
         * @brief Radix-5 decomposition for five-point sub-FFTs.
         *
         * Intention: Compute the FFT for data lengths divisible by 5 by splitting into five sub-FFTs,
         * corresponding to indices n mod 5 = 0, 1, 2, 3, 4, and combining results with twiddle factors.
         * This supports N=5^r or mixed-radix cases.
         *
         * Mathematically: The FFT is computed as:
         *   \( X(k) = X_0(k) + W_N^k \cdot X_1(k) + W_N^{2k} \cdot X_2(k) + W_N^{3k} \cdot X_3(k) + W_N^{4k} \cdot X_4(k) \),
         * where \( X_0, ..., X_4 \) are sub-FFTs of size N/5, and \( W_N^k = e^{-2\pi i k / N} \).
         * The radix-5 butterfly uses rotations at 72°, 144°, 216°, and 288°.
         *
         * Process:
         * 1. Divide data into five subproblems of size N/5, adjusting stride.
         * 2. Recursively compute FFTs for each subproblem.
         * 3. For k=0, perform a butterfly without twiddles using precomputed cos/sin constants.
         * 4. For k>0, apply twiddle factors W_N^k to W_N^{4k}, and compute the butterfly.
         *
         * Optimization:
         * - Precomputed constants for cos/sin(72°, 144°) avoid runtime trigonometric calls.
         * - Symmetry in outputs (e.g., X_1/X_4, X_2/X_3 are conjugates) reduces computations.
         * - Separate k=0 case avoids unnecessary twiddle multiplications.
         *
         * Transform sign: Adjusts the rotation signs for forward/inverse transforms, applied in the
         * butterfly computations.
         */
        int sub_length = data_length / 5; // Size of each sub-FFT (N/5)
        new_stride = 5 * stride;          // Stride increases fivefold

        // Step 1: Recurse into each of the 5 decimated sub-FFTs
        mixed_radix_dit_rec(output_buffer, input_buffer, fft_obj, transform_sign, sub_length,
                            new_stride, factor_index + 1); // n mod 5 = 0
        mixed_radix_dit_rec(output_buffer + sub_length, input_buffer + stride, fft_obj,
                            transform_sign, sub_length, new_stride, factor_index + 1); // n mod 5 = 1
        mixed_radix_dit_rec(output_buffer + 2 * sub_length, input_buffer + 2 * stride, fft_obj,
                            transform_sign, sub_length, new_stride, factor_index + 1); // n mod 5 = 2
        mixed_radix_dit_rec(output_buffer + 3 * sub_length, input_buffer + 3 * stride, fft_obj,
                            transform_sign, sub_length, new_stride, factor_index + 1); // n mod 5 = 3
        mixed_radix_dit_rec(output_buffer + 4 * sub_length, input_buffer + 4 * stride, fft_obj,
                            transform_sign, sub_length, new_stride, factor_index + 1); // n mod 5 = 4

        // Constants for radix-5
        const fft_type c1 = 0.30901699437;  // cos(72°)
        const fft_type c2 = -0.80901699437; // cos(144°)
        const fft_type s1 = 0.95105651629;  // sin(72°)
        const fft_type s2 = 0.58778525229;  // sin(144°)

        // Step 2: k = 0 (no twiddle multiplications)
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
            X1->re = a_re + c1 * (b_re + e_re) + c2 * (c_re + d_re) +
                     transform_sign * (s1 * (b_im - e_im) + s2 * (c_im - d_im)); // X1 real
            X1->im = a_im + c1 * (b_im + e_im) + c2 * (c_im + d_im) -
                     transform_sign * (s1 * (b_re - e_re) + s2 * (c_re - d_re)); // X1 imag

            // X4 = a + c1*(b+e) + c2*(c+d) - i*transform_sign*( s1*(b-e) + s2*(c-d) )
            X4->re = a_re + c1 * (b_re + e_re) + c2 * (c_re + d_re) -
                     transform_sign * (s1 * (b_im - e_im) + s2 * (c_im - d_im)); // X4 real
            X4->im = a_im + c1 * (b_im + e_im) + c2 * (c_im + d_im) +
                     transform_sign * (s1 * (b_re - e_re) + s2 * (c_re - d_re)); // X4 imag

            // X2 = a + c2*(b+e) + c1*(c+d) + i*transform_sign*( s2*(b-e) - s1*(c-d) )
            X2->re = a_re + c2 * (b_re + e_re) + c1 * (c_re + d_re) +
                     transform_sign * (s2 * (b_im - e_im) - s1 * (c_im - d_im)); // X2 real
            X2->im = a_im + c2 * (b_im + e_im) + c1 * (c_im + d_im) -
                     transform_sign * (s2 * (b_re - e_re) - s1 * (c_re - d_re)); // X2 imag

            // X3 = a + c2*(b+e) + c1*(c+d) - i*transform_sign*( s2*(b-e) - s1*(c-d) )
            X3->re = a_re + c2 * (b_re + e_re) + c1 * (c_re + d_re) -
                     transform_sign * (s2 * (b_im - e_im) - s1 * (c_im - d_im)); // X3 real
            X3->im = a_im + c2 * (b_im + e_im) + c1 * (c_im + d_im) +
                     transform_sign * (s2 * (b_re - e_re) - s1 * (c_re - d_re)); // X3 imag
        }

        // Step 3: k = 1 to sub_length-1: Apply twiddles and perform butterfly
        for (int k = 1; k < sub_length; k++)
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

            fft_data *X0 = &output_buffer[k];                  // First point
            fft_data *X1 = &output_buffer[k + sub_length];     // Second point
            fft_data *X2 = &output_buffer[k + 2 * sub_length]; // Third point
            fft_data *X3 = &output_buffer[k + 3 * sub_length]; // Fourth point
            fft_data *X4 = &output_buffer[k + 4 * sub_length]; // Fifth point

            // Apply twiddle factors
            fft_type a_re = X0->re, a_im = X0->im;       // First point
            fft_type b_re = X1->re * w1r - X1->im * w1i; // W_N^k * X_1 real
            fft_type b_im = X1->im * w1r + X1->re * w1i; // W_N^k * X_1 imag
            fft_type c_re = X2->re * w2r - X2->im * w2i; // W_N^{2k} * X_2 real
            fft_type c_im = X2->im * w2r + X2->re * w2i; // W_N^{2k} * X_2 imag
            fft_type d_re = X3->re * w3r - X3->im * w3i; // W_N^{3k} * X_3 real
            fft_type d_im = X3->im * w3r + X3->re * w3i; // W_N^{3k} * X_3 imag
            fft_type e_re = X4->re * w4r - X4->im * w4i; // W_N^{4k} * X_4 real
            fft_type e_im = X4->im * w4r + X4->re * w4i; // W_N^{4k} * X_4 imag

            // Perform the same five-way butterfly as k=0
            X0->re = a_re + b_re + c_re + d_re + e_re; // X(k) real
            X0->im = a_im + b_im + c_im + d_im + e_im; // X(k) imag
            X1->re = a_re + c1 * (b_re + e_re) + c2 * (c_re + d_re) +
                     transform_sign * (s1 * (b_im - e_im) + s2 * (c_im - d_im)); // X(k + N/5) real
            X1->im = a_im + c1 * (b_im + e_im) + c2 * (c_im + d_im) -
                     transform_sign * (s1 * (b_re - e_re) + s2 * (c_re - d_re)); // X(k + N/5) imag
            X4->re = a_re + c1 * (b_re + e_re) + c2 * (c_re + d_re) -
                     transform_sign * (s1 * (b_im - e_im) + s2 * (c_im - d_im)); // X(k + 4N/5) real
            X4->im = a_im + c1 * (b_im + e_im) + c2 * (c_im + d_im) +
                     transform_sign * (s1 * (b_re - e_re) + s2 * (c_re - d_re)); // X(k + 4N/5) imag
            X2->re = a_re + c2 * (b_re + e_re) + c1 * (c_re + d_re) +
                     transform_sign * (s2 * (b_im - e_im) - s1 * (c_im - d_im)); // X(k + 2N/5) real
            X2->im = a_im + c2 * (b_im + e_im) + c1 * (c_im + d_im) -
                     transform_sign * (s2 * (b_re - e_re) - s1 * (c_re - d_re)); // X(k + 2N/5) imag
            X3->re = a_re + c2 * (b_re + e_re) + c1 * (c_re + d_re) -
                     transform_sign * (s2 * (b_im - e_im) - s1 * (c_im - d_im)); // X(k + 3N/5) real
            X3->im = a_im + c2 * (b_im + e_im) + c1 * (c_im + d_im) +
                     transform_sign * (s2 * (b_re - e_re) - s1 * (c_re - d_re)); // X(k + 3N/5) imag
        }
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
     * Transform sign: Adjusts rotation signs for forward/inverse transforms using a branch-free
     * SIGN_MASK multiplication.
     *
     * Note: Uses unaligned AVX2 loads/stores (LOADU_PD/STOREU_PD) as malloc doesn’t guarantee
     * 32-byte alignment. Twiddle and output arrays are allocated per call; pre-allocation in fft_obj
     * is planned to avoid malloc/free overhead. Scalar tail uses standard operations, as FMA emulation
     * offers minimal benefit. Twiddle index idx = sub_length - 1 + n*k assumes twiddle[sub_length-1 + n*k] = W_N^{n*k};
     * validate against fft_init to ensure correct phases. Runtime FMA detection (e.g., _cpuid for MSVC)
     * could be added but is omitted for performance. Micro-unrolling (e.g., #pragma unroll) not used,
     * as MSVC relies on /O2 auto-unrolling.
     *
     * @warning Assumes fft_obj->twiddle has at least 7M-1 elements, where M = data_length/7,
     * to support indices up to sub_length-1 + 6*(M-1). If undersized, undefined behavior may occur.
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
     * Process:
     * 1. Divide data into eight subproblems of size N/8, adjusting stride.
     * 2. Flatten twiddle factors into contiguous arrays for efficient AVX2 access.
     * 3. Recursively compute FFTs for each subproblem.
     * 4. Flatten interleaved fft_data outputs into contiguous real/imag arrays to avoid struct stride issues.
     * 5. Perform vectorized radix-8 butterfly operations using AVX2 with FMA (via FMADD/FMSUB) for k divisible by 4.
     * 6. Handle remaining k with scalar operations, mirroring the vectorized logic.
     * 7. Copy results back to the output buffer and clean up allocated memory.
     *
     * Optimization:
     * - AVX2 vectorization processes four complex points simultaneously.
     * - FMA instructions (FMADD/FMSUB) reduce instruction count and improve numerical accuracy
     *   when compiled with /fp:fast or -mfma and USE_FMA defined (requires Intel Haswell+ or AMD Zen+).
     * - Without FMA, inlined fallback functions (ALWAYS_INLINE) avoid call overhead.
     * - AVX_ONE macro inlines _mm256_set1_pd(1.0) for constant vectors.
     * - File-scope scalar constant C8_1 avoids per-call overhead and naming conflicts with radix-7; SIMD vectors
     *   inlined to avoid MSVC initializer errors.
     * - Contiguous twiddle and output arrays ensure efficient SIMD loads/stores.
     * - Symmetry in outputs (e.g., X_1/X_7, X_2/X_6, X_3/X_5 are conjugates) reduces computations.
     *
     * Transform sign: Adjusts rotation directions for forward/inverse transforms, applied via
     * vsign in the butterfly computations.
     *
     * Note: Uses unaligned AVX2 loads/stores (LOADU_PD/STOREU_PD) as malloc doesn’t guarantee
     * 32-byte alignment. Twiddle and output arrays are allocated per call; pre-allocation in fft_obj
     * is planned to avoid malloc/free overhead. Scalar tail uses standard operations, as FMA emulation
     * offers minimal benefit. Twiddle index idx = (M - 1) + 7*k assumes twiddle[M-1 + n*k] = W_N^{n*k};
     * validate against fft_init to ensure correct phases. Runtime FMA detection (e.g., _cpuid for MSVC)
     * could be added but is omitted for performance.
     *
     * Requirements: Compile with /arch:AVX2 (MSVC) or -mavx2 (GCC) for AVX2 support. Use /fp:fast or
     * -mfma for FMA optimization. Assumes FMADD, FMSUB, AVX_ONE, LOADU_PD, STOREU_PD macros are defined.
     *
     * @warning Assumes fft_obj->twiddle has at least N-1 elements, where N = data_length,
     * to support indices up to M-1 + 7*(M-1). If undersized, undefined behavior may occur.
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
 * @brief Performs Bluestein’s FFT algorithm for arbitrary-length signals using SIMD vectorization.
 *
 * Intention: Compute the Fast Fourier Transform (FFT) for non-power-of-2 signal lengths using
 * Bluestein’s chirp z-transform, optimized with AVX2 SIMD instructions. This algorithm transforms
 * the DFT of length N into a convolution with a chirp sequence, padded to a power-of-2 length
 * for efficient computation using a standard FFT. It supports both forward and inverse transforms.
 *
 * Mathematically: Bluestein’s algorithm computes the DFT:
 *   \( X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-2\pi i k n / N} \)
 * by rewriting it as a convolution using the identity:
 *   \( k n = \frac{1}{2} [k^2 + n^2 - (k-n)^2] \).
 * This yields:
 *   \( X(k) = e^{-\pi i k^2 / N} \cdot \sum_{n=0}^{N-1} [x(n) \cdot e^{-\pi i n^2 / N}] \cdot e^{\pi i (k-n)^2 / N} \),
 * which is computed as a circular convolution of length M >= 2N-1 (padded to a power of 2) between
 * the chirped input \( x(n) \cdot e^{-\pi i n^2 / N} \) and the chirp sequence \( e^{\pi i n^2 / N} \).
 * The convolution is performed efficiently in the frequency domain using FFTs.
 *
 * Process:
 * 1. Validate the signal length and compute the padded length (smallest power of 2 >= 2N-1).
 * 2. Allocate temporary buffers for computations.
 * 3. Generate and scale the chirp sequence h(n) = e^{\pi i n^2 / N}, using precomputed values if available.
 * 4. Compute the FFT of the scaled chirp sequence.
 * 5. Multiply the input signal by the chirp (or its conjugate for inverse FFT) using SIMD.
 * 6. Zero-pad the chirped signal to the padded length.
 * 7. Compute the FFT of the padded chirped signal.
 * 8. Perform pointwise multiplication with the chirp FFT (or its conjugate for inverse) using SIMD.
 * 9. Compute the inverse FFT of the result, adjusting the FFT object for the inverse transform.
 * 10. Multiply the result by the chirp again to extract the final DFT, using SIMD.
 * 11. Restore the FFT object state and free temporary buffers.
 *
 * Optimization:
 * - AVX2 SIMD vectorization processes four complex numbers simultaneously in key multiplication steps.
 * - Precomputed chirp sequences (via bluestein_exp) reduce runtime trigonometric calculations.
 * - Padded length is a power of 2, enabling efficient mixed-radix FFTs.
 * - Chirp FFT is precomputed and reused, avoiding redundant calculations.
 * - Memory alignment is assumed for SIMD efficiency (caller must ensure aligned allocations).
 *
 * Transform Direction:
 * - Forward FFT (transform_direction == 1): Uses chirp h(n) = e^{\pi i n^2 / N} and its FFT.
 * - Inverse FFT (transform_direction == -1): Uses conjugate chirp h^*(n) and conjugate FFT.
 *
 * @param[in] input_signal Input signal data (length N).
 *                        Real and imaginary components of the input to be transformed.
 * @param[out] output_signal Output FFT results (length N).
 *                          Stores the transformed complex values after Bluestein’s FFT.
 * @param[in,out] fft_config FFT configuration object.
 *                       Contains signal length, transform direction, factors, and twiddle factors,
 *                       temporarily modified for padded length computations.
 * @param[in] transform_direction Direction of transform (+1 for forward, -1 for inverse).
 *                               Determines forward FFT (e^{-2\pi i k n / N}) or inverse FFT (e^{+2\pi i k n / N}).
 * @param[in] signal_length Length of the input signal (N > 0).
 *                         Size of input and output buffers, must be positive.
 *
 * @warning If memory allocation fails or signal_length is invalid (<= 0), the function exits with an error.
 * @note Temporarily modifies fft_config (N, lt, sgn, twiddle factors) and restores it afterward.
 *       Assumes input_signal, output_signal, and fft_config are valid pointers.
 *       Caller must ensure memory buffers are aligned for SIMD operations to maximize performance.
 */
void bluestein_fft_simd(
    const fft_data *input_signal,
    fft_data *output_signal,
    fft_object fft_config,
    int transform_direction,
    int signal_length)
{
    // Step 1: Validate signal length
    if (signal_length <= 0) {
        fprintf(stderr, "Error: Signal length (%d) must be positive for Bluestein’s FFT\n", signal_length);
        exit(EXIT_FAILURE);
    }

    // Step 2: Calculate padded length for convolution
    // Ensure M >= 2N-1 to avoid aliasing in circular convolution, rounded to the next power of 2
    int min_padded_length = 2 * signal_length - 1;
    int padded_length = (int)pow(2.0, ceil(log2((double)min_padded_length)));
    int original_N = fft_config->N;          // Save original FFT length
    int original_lt = fft_config->lt;        // Save original algorithm type
    int original_sgn = fft_config->sgn;      // Save original transform direction

    // Step 3: Temporarily reconfigure FFT object for padded length
    fft_config->N = padded_length;           // Set to padded length for FFT computations
    fft_config->lt = 0;                      // Use mixed-radix algorithm

    // Step 4: Allocate temporary buffers
    fft_data *chirped_signal = malloc(sizeof(fft_data) * padded_length); // Padded chirped input
    fft_data *chirp_fft = malloc(sizeof(fft_data) * padded_length);      // FFT of chirp sequence
    fft_data *temp_chirp = malloc(sizeof(fft_data) * padded_length);     // Temporary chirp storage
    fft_data *ifft_result = malloc(sizeof(fft_data) * padded_length);    // Inverse FFT output
    fft_data *chirp_sequence = malloc(sizeof(fft_data) * signal_length); // Chirp sequence h(n)
    if (!chirped_signal || !chirp_fft || !temp_chirp || !ifft_result || !chirp_sequence) {
        fprintf(stderr, "Error: Memory allocation failed for Bluestein’s FFT arrays\n");
        free(chirped_signal); free(chirp_fft); free(temp_chirp); free(ifft_result); free(chirp_sequence);
        exit(EXIT_FAILURE);
    }

    // Step 5: Generate and scale chirp sequence
    // Compute h(n) = e^{\pi i n^2 / N} using bluestein_exp, scale by 1/M for normalization
    bluestein_exp(temp_chirp, chirp_sequence, signal_length, padded_length);
    fft_type scale = 1.0 / padded_length;
    for (int i = 0; i < padded_length; ++i) {
        temp_chirp[i].re *= scale;           // Scale real part
        temp_chirp[i].im *= scale;           // Scale imaginary part
    }

    // Step 6: Compute FFT of the scaled chirp sequence
    fft_exec(fft_config, temp_chirp, chirp_fft);

    // Step 7: Multiply input by chirp sequence using AVX2
    // For forward FFT: x(n) * h(n); for inverse FFT: x(n) * h^*(n)
    int n = 0;
    for (; n + 3 < signal_length; n += 4) {
        // Load four input signal points
        __m256d input_re = _mm256_set_pd(
            input_signal[n + 3].re, input_signal[n + 2].re,
            input_signal[n + 1].re, input_signal[n + 0].re);
        __m256d input_im = _mm256_set_pd(
            input_signal[n + 3].im, input_signal[n + 2].im,
            input_signal[n + 1].im, input_signal[n + 0].im);
        // Load four chirp sequence points
        __m256d chirp_re = _mm256_set_pd(
            chirp_sequence[n + 3].re, chirp_sequence[n + 2].re,
            chirp_sequence[n + 1].re, chirp_sequence[n + 0].re);
        __m256d chirp_im = _mm256_set_pd(
            chirp_sequence[n + 3].im, chirp_sequence[n + 2].im,
            chirp_sequence[n + 1].im, chirp_sequence[n + 0].im);

        // Adjust for inverse FFT: conjugate chirp (negate imaginary part)
        if (transform_direction == -1) {
            chirp_im = _mm256_sub_pd(_mm256_setzero_pd(), chirp_im);
        }

        // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
        __m256d result_re = _mm256_fmsub_pd(input_re, chirp_re, _mm256_mul_pd(input_im, chirp_im));
        __m256d result_im = _mm256_fmadd_pd(input_im, chirp_re, _mm256_mul_pd(input_re, chirp_im));

        // Store results
        _mm256_storeu_pd(&chirped_signal[n].re, result_re);
        _mm256_storeu_pd(&chirped_signal[n].im, result_im);
    }
    // Scalar tail for remaining points
    for (; n < signal_length; ++n) {
        double input_re = input_signal[n].re, input_im = input_signal[n].im;
        double chirp_re = chirp_sequence[n].re, chirp_im = chirp_sequence[n].im;
        if (transform_direction == -1) {
            chirp_im = -chirp_im;            // Conjugate for inverse FFT
        }
        chirped_signal[n].re = input_re * chirp_re - input_im * chirp_im;
        chirped_signal[n].im = input_im * chirp_re + input_re * chirp_im;
    }

    // Step 8: Zero-pad the chirped signal
    for (int i = signal_length; i < padded_length; ++i) {
        chirped_signal[i].re = 0.0;
        chirped_signal[i].im = 0.0;
    }

    // Step 9: Compute FFT of the padded chirped signal
    fft_exec(fft_config, chirped_signal, temp_chirp); // Reuse temp_chirp for FFT result

    // Step 10: Pointwise multiplication in frequency domain using AVX2
    // Forward: y(n) * h_k(n); Inverse: y(n) * h_k^*(n)
    n = 0;
    for (; n + 3 < padded_length; n += 4) {
        // Load four points from FFT result
        __m256d fft_re = _mm256_set_pd(
            temp_chirp[n + 3].re, temp_chirp[n + 2].re,
            temp_chirp[n + 1].re, temp_chirp[n + 0].re);
        __m256d fft_im = _mm256_set_pd(
            temp_chirp[n + 3].im, temp_chirp[n + 2].im,
            temp_chirp[n + 1].im, temp_chirp[n + 0].im);
        // Load four points from chirp FFT
        __m256d chirp_fft_re = _mm256_set_pd(
            chirp_fft[n + 3].re, chirp_fft[n + 2].re,
            chirp_fft[n + 1].re, chirp_fft[n + 0].re);
        __m256d chirp_fft_im = _mm256_set_pd(
            chirp_fft[n + 3].im, chirp_fft[n + 2].im,
            chirp_fft[n + 1].im, chirp_fft[n + 0].im);

        // Adjust for inverse FFT: conjugate chirp FFT
        if (transform_direction == -1) {
            chirp_fft_im = _mm256_sub_pd(_mm256_setzero_pd(), chirp_fft_im);
        }

        // Complex multiplication
        __m256d result_re = _mm256_fmsub_pd(fft_re, chirp_fft_re, _mm256_mul_pd(fft_im, chirp_fft_im));
        __m256d result_im = _mm256_fmadd_pd(fft_im, chirp_fft_re, _mm256_mul_pd(fft_re, chirp_fft_im));

        // Store results
        _mm256_storeu_pd(&temp_chirp[n].re, result_re);
        _mm256_storeu_pd(&temp_chirp[n].im, result_im);
    }
    // Scalar tail
    for (; n < padded_length; ++n) {
        double fft_re = temp_chirp[n].re, fft_im = temp_chirp[n].im;
        double chirp_fft_re = chirp_fft[n].re, chirp_fft_im = chirp_fft[n].im;
        if (transform_direction == -1) {
            chirp_fft_im = -chirp_fft_im;    // Conjugate for inverse FFT
        }
        temp_chirp[n].re = fft_re * chirp_fft_re - fft_im * chirp_fft_im;
        temp_chirp[n].im = fft_im * chirp_fft_re + fft_re * chirp_fft_im;
    }

    // Step 11: Compute inverse FFT
    // Adjust twiddle factors and sign for inverse transform
    for (int i = 0; i < padded_length; ++i) {
        fft_config->twiddle[i].im = -fft_config->twiddle[i].im;
    }
    fft_config->sgn = -transform_direction;
    fft_exec(fft_config, temp_chirp, ifft_result);

    // Step 12: Apply chirp sequence again to extract final results using AVX2
    // Forward: y(n) * h(n); Inverse: y(n) * h^*(n)
    n = 0;
    for (; n + 3 < signal_length; n += 4) {
        // Load four IFFT result points
        __m256d ifft_re = _mm256_set_pd(
            ifft_result[n + 3].re, ifft_result[n + 2].re,
            ifft_result[n + 1].re, ifft_result[n + 0].re);
        __m256d ifft_im = _mm256_set_pd(
            ifft_result[n + 3].im, ifft_result[n + 2].im,
            ifft_result[n + 1].im, ifft_result[n + 0].im);
        // Load four chirp sequence points
        __m256d chirp_re = _mm256_set_pd(
            chirp_sequence[n + 3].re, chirp_sequence[n + 2].re,
            chirp_sequence[n + 1].re, chirp_sequence[n + 0].re);
        __m256d chirp_im = _mm256_set_pd(
            chirp_sequence[n + 3].im, chirp_sequence[n + 2].im,
            chirp_sequence[n + 1].im, chirp_sequence[n + 0].im);

        // Adjust for inverse FFT: conjugate chirp
        if (transform_direction == -1) {
            chirp_im = _mm256_sub_pd(_mm256_setzero_pd(), chirp_im);
        }

        // Complex multiplication
        __m256d result_re = _mm256_fmsub_pd(ifft_re, chirp_re, _mm256_mul_pd(ifft_im, chirp_im));
        __m256d result_im = _mm256_fmadd_pd(ifft_im, chirp_re, _mm256_mul_pd(ifft_re, chirp_im));

        // Store results
        _mm256_storeu_pd(&output_signal[n].re, result_re);
        _mm256_storeu_pd(&output_signal[n].im, result_im);
    }
    // Scalar tail
    for (; n < signal_length; ++n) {
        double ifft_re = ifft_result[n].re, ifft_im = ifft_result[n].im;
        double chirp_re = chirp_sequence[n].re, chirp_im = chirp_sequence[n].im;
        if (transform_direction == -1) {
            chirp_im = -chirp_im;            // Conjugate for inverse FFT
        }
        output_signal[n].re = ifft_re * chirp_re - ifft_im * chirp_im;
        output_signal[n].im = ifft_im * chirp_re + ifft_re * chirp_im;
    }

    // Step 13: Restore original FFT object state
    fft_config->sgn = original_sgn;
    fft_config->N = original_N;
    fft_config->lt = original_lt;
    for (int i = 0; i < padded_length; ++i) {
        fft_config->twiddle[i].im = -fft_config->twiddle[i].im; // Restore twiddle factors
    }

    // Step 14: Free temporary buffers
    free(chirped_signal);
    free(chirp_fft);
    free(temp_chirp);
    free(ifft_result);
    free(chirp_sequence);
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
void fft_exec(fft_object fft_obj, fft_data *input_data, fft_data *output_data)
{
    if (fft_obj == NULL || input_data == NULL || output_data == NULL)
    {
        fprintf(stderr, "Error: Invalid FFT object or data pointers\n");
        exit(EXIT_FAILURE);
    }

    if (fft_obj->lt == 0)
    {
        int stride = 1, factor_index = 0;
        mixed_radix_dit_rec(output_data, input_data, fft_obj, fft_obj->sgn, fft_obj->N, stride, factor_index);
    }
    else if (fft_obj->lt == 1)
    {
        bluestein_fft(input_data, output_data, fft_obj, fft_obj->sgn, fft_obj->N);
    }
    else
    {
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

void free_fft(fft_object object)
{
    free(object);
}
