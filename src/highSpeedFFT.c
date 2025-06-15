#include "highspeedFFT.h"
#include "time.h"
#include <immintrin.h>


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
    else if (radix == 2)
    {
        /**
         * @brief Radix-2 decomposition for two-point sub-FFTs.
         *
         * Recursively divides the data into two sub-FFTs of length N/2, applies the FFT to each,
         * and combines the results using twiddle factors. This is the core DIT strategy for N=2^r,
         * where \(X(k) = X_e(k) + W_N^k \cdot X_o(k)\), with \(W_N = e^{-2\pi i / N}\) and
         * \(X_e(k)\), \(X_o(k)\) being even/odd indexed sub-FFTs, adjusted by transform_sign.
         */
        // Step 1: compute subproblem size and new stride
        int sub_length = data_length / 2;
        new_stride = 2 * stride;

        // Step 2: recurse on even- and odd-indexed halves
        mixed_radix_dit_rec(output_buffer,
                            input_buffer,
                            fft_obj,
                            transform_sign,
                            sub_length,
                            new_stride,
                            factor_index + 1);

        mixed_radix_dit_rec(output_buffer + sub_length,
                            input_buffer + stride,
                            fft_obj,
                            transform_sign,
                            sub_length,
                            new_stride,
                            factor_index + 1);

        // Step 3: point at your real and imag arrays and twiddles
        double *out_re = &output_buffer[0].re;
        double *out_im = &output_buffer[0].im;
        double *tw_re  = &fft_obj->twiddle[sub_length - 1].re;
        double *tw_im  = &fft_obj->twiddle[sub_length - 1].im;

        // Step 4: do the AVX2‑accelerated butterfly
        int k = 0, M = sub_length;

        // Vectorized loop (4 lanes at a time)
        for (; k + 3 < M; k += 4) {
            __m256d er  = _mm256_load_pd(out_re + k);
            __m256d ei  = _mm256_load_pd(out_im + k);
            __m256d br  = _mm256_load_pd(out_re + k + M);
            __m256d bi  = _mm256_load_pd(out_im + k + M);
            __m256d wre = _mm256_load_pd(tw_re  + k);
            __m256d wim = _mm256_load_pd(tw_im  + k);

            // tr = br*wre - bi*wim
            __m256d tr = _mm256_fmsub_pd(br, wre,
                            _mm256_mul_pd(bi, wim));
            // ti = bi*wre + br*wim
            __m256d ti = _mm256_fmadd_pd(bi, wre,
                            _mm256_mul_pd(br, wim));

            _mm256_store_pd(out_re   + k,     _mm256_add_pd(er, tr));
            _mm256_store_pd(out_im   + k,     _mm256_add_pd(ei, ti));
            _mm256_store_pd(out_re   + k + M, _mm256_sub_pd(er, tr));
            _mm256_store_pd(out_im   + k + M, _mm256_sub_pd(ei, ti));
        }

        // Scalar tail for k not divisible by 4
        for (; k < M; ++k) {
            double ar = out_re[k],  ai = out_im[k];
            double br_ = out_re[k+M], bi_ = out_im[k+M];
            double tr  =  br_ * tw_re[k] - bi_ * tw_im[k];
            double ti  =  bi_ * tw_re[k] + br_ * tw_im[k];
            out_re[k]   = ar + tr;
            out_im[k]   = ai + ti;
            out_re[k+M] = ar - tr;
            out_im[k+M] = ai - ti;
        }
    }
    else if (radix == 3)
    {
        /**
         * @brief Radix-3 decomposition for three-point sub-FFTs.
         *
         * Recursively divides the data into three sub-FFTs of length N/3, applies the FFT to each,
         * and combines the results using twiddle factors. This extends the DIT strategy for N=3^r,
         * where \(X(k) = X_0(k) + W_N^k \cdot X_1(k) + W_N^{2k} \cdot X_2(k)\), with \(W_N = e^{-2\pi i / N}\)
         * and \(X_0(k)\), \(X_1(k)\), \(X_2(k)\) being the three sub-FFTs, adjusted by transform_sign.
         */
          // 1) compute subproblem size and new stride
        int sub_length = data_length / 3;
        new_stride = 3 * stride;

        // 2) recurse on the three decimated sub‑FFTs
        mixed_radix_dit_rec(output_buffer,
                            input_buffer,
                            fft_obj,
                            transform_sign,
                            sub_length,
                            new_stride,
                            factor_index + 1);

        mixed_radix_dit_rec(output_buffer + sub_length,
                            input_buffer + stride,
                            fft_obj,
                            transform_sign,
                            sub_length,
                            new_stride,
                            factor_index + 1);

        mixed_radix_dit_rec(output_buffer + 2*sub_length,
                            input_buffer + 2*stride,
                            fft_obj,
                            transform_sign,
                            sub_length,
                            new_stride,
                            factor_index + 1);

        // 3) do the radix-3 butterflies
        const fft_type sqrt3by2 = (fft_type)0.8660254037844386;  // √3/2
        for (int k = 0; k < sub_length; k++) {
            // fetch twiddles W(N)^{k} and W(N)^{2k}
            int idx1 = sub_length - 1 + 2*k;
            fft_type w1r = fft_obj->twiddle[idx1 + 0].re;
            fft_type w1i = fft_obj->twiddle[idx1 + 0].im;
            fft_type w2r = fft_obj->twiddle[idx1 + 1].re;
            fft_type w2i = fft_obj->twiddle[idx1 + 1].im;

            // load the three partial FFT results
            fft_data *X0 = &output_buffer[k];
            fft_data *X1 = &output_buffer[k + sub_length];
            fft_data *X2 = &output_buffer[k + 2*sub_length];

            fft_type a_re = X0->re, a_im = X0->im;
            // apply twiddle to the odd branches
            fft_type b_re =  X1->re * w1r - X1->im * w1i;
            fft_type b_im =  X1->im * w1r + X1->re * w1i;
            fft_type c_re =  X2->re * w2r - X2->im * w2i;
            fft_type c_im =  X2->im * w2r + X2->re * w2i;

            // compute the three-way butterfly
            fft_type sum_re = b_re + c_re;
            fft_type sum_im = b_im + c_im;
            fft_type diff_re = b_re - c_re;
            fft_type diff_im = b_im - c_im;

            // X0 ← a + (b+c)
            X0->re = a_re + sum_re;
            X0->im = a_im + sum_im;

            // X1 ← a − ½(b+c)  +  i·(sign·(√3/2)·(b−c))
            fft_type t_re = a_re - (sum_re * (fft_type)0.5);
            fft_type t_im = a_im - (sum_im * (fft_type)0.5);
            fft_type rot_re =  diff_im * (transform_sign * sqrt3by2);
            fft_type rot_im = -diff_re * (transform_sign * sqrt3by2);

            X1->re = t_re + rot_re;
            X1->im = t_im + rot_im;

            // X2 ← a − ½(b+c)  −  i·(sign·(√3/2)·(b−c))
            X2->re = t_re - rot_re;
            X2->im = t_im - rot_im;
        }
    }
    else if (radix == 4)
    {
        /**
         * @brief Radix-4 decomposition for four-point sub-FFTs.
         *
         * Recursively divides the data into four sub-FFTs of length N/4, applies the FFT to each,
         * and combines the results using twiddle factors. This extends the DIT strategy for N=4^r,
         * where \(X(k) = X_0(k) + W_N^k \cdot X_1(k) + W_N^{2k} \cdot X_2(k) + W_N^{3k} \cdot X_3(k)\),
         * with \(W_N = e^{-2\pi i / N}\) and \(X_0(k)\), \(X_1(k)\), \(X_2(k)\), \(X_3(k)\) being the four sub-FFTs,
         * adjusted by transform_sign.
         */
        // 1) compute subproblem size and new stride
        int sub_length = data_length / 4;
        new_stride      = 4 * stride;

        // 2) recurse on the four decimated sub‑FFTs
        mixed_radix_dit_rec(output_buffer,
                            input_buffer,
                            fft_obj,
                            transform_sign,
                            sub_length,
                            new_stride,
                            factor_index + 1);

        mixed_radix_dit_rec(output_buffer +     sub_length,
                            input_buffer  +     stride,
                            fft_obj,
                            transform_sign,
                            sub_length,
                            new_stride,
                            factor_index + 1);

        mixed_radix_dit_rec(output_buffer + 2 * sub_length,
                            input_buffer  + 2 * stride,
                            fft_obj,
                            transform_sign,
                            sub_length,
                            new_stride,
                            factor_index + 1);

        mixed_radix_dit_rec(output_buffer + 3 * sub_length,
                            input_buffer  + 3 * stride,
                            fft_obj,
                            transform_sign,
                            sub_length,
                            new_stride,
                            factor_index + 1);

        // 3) handle k=0 separately (no twiddles needed)
        {
            fft_data *X0 = &output_buffer[0];
            fft_data *X1 = &output_buffer[    sub_length];
            fft_data *X2 = &output_buffer[2 * sub_length];
            fft_data *X3 = &output_buffer[3 * sub_length];

            // simple radix‑4 butterfly
            fft_type a_re = X0->re + X2->re;
            fft_type a_im = X0->im + X2->im;
            fft_type b_re = X0->re - X2->re;
            fft_type b_im = X0->im - X2->im;
            fft_type c_re = X1->re + X3->re;
            fft_type c_im = X1->im + X3->im;
            fft_type d_re = transform_sign * (X1->re - X3->re);
            fft_type d_im = transform_sign * (X1->im - X3->im);

            X0->re = a_re + c_re;
            X0->im = a_im + c_im;
            X2->re = a_re - c_re;
            X2->im = a_im - c_im;

            X1->re = b_re + d_im;
            X1->im = b_im - d_re;
            X3->re = b_re - d_im;
            X3->im = b_im + d_re;
        }

        // 4) k = 1 .. sub_length-1, with twiddles W_N^{k}, W_N^{2k}, W_N^{3k}
        for (int k = 1; k < sub_length; k++)
        {
            int idx = (sub_length - 1) + 3*k;
            fft_type w1r = fft_obj->twiddle[idx + 0].re;
            fft_type w1i = fft_obj->twiddle[idx + 0].im;
            fft_type w2r = fft_obj->twiddle[idx + 1].re;
            fft_type w2i = fft_obj->twiddle[idx + 1].im;
            fft_type w3r = fft_obj->twiddle[idx + 2].re;
            fft_type w3i = fft_obj->twiddle[idx + 2].im;

            // load the four branches
            fft_data *X0 = &output_buffer[k];
            fft_data *X1 = &output_buffer[k +     sub_length];
            fft_data *X2 = &output_buffer[k + 2 * sub_length];
            fft_data *X3 = &output_buffer[k + 3 * sub_length];

            // apply twiddles to the “odd” branches
            fft_type b_re =  X1->re * w1r - X1->im * w1i;
            fft_type b_im =  X1->im * w1r + X1->re * w1i;
            fft_type c_re =  X2->re * w2r - X2->im * w2i;
            fft_type c_im =  X2->im * w2r + X2->re * w2i;
            fft_type d_re =  X3->re * w3r - X3->im * w3i;
            fft_type d_im =  X3->im * w3r + X3->re * w3i;

            // radix‑4 butterfly on (X0, b, c, d)
            fft_type a_re = X0->re + c_re;
            fft_type a_im = X0->im + c_im;
            fft_type e_re = X0->re - c_re;
            fft_type e_im = X0->im - c_im;
            fft_type f_re =       b_re + d_re;
            fft_type f_im =       b_im + d_im;
            fft_type g_re = transform_sign * (b_re - d_re);
            fft_type g_im = transform_sign * (b_im - d_im);

            // store back
            X0->re = a_re + f_re;
            X0->im = a_im + f_im;

            X2->re = a_re - f_re;
            X2->im = a_im - f_im;

            X1->re = e_re + g_im;
            X1->im = e_im - g_re;

            X3->re = e_re - g_im;
            X3->im = e_im + g_re;
        }
    }
    else if (radix == 5)
    {
        /**
         * @brief Radix-5 decomposition for five-point sub-FFTs.
         *
         * Recursively divides the data into five sub-FFTs of length N/5, applies the FFT to each,
         * and combines the results using twiddle factors. This extends the DIT strategy for N=5^r,
         * where \(X(k) = X_0(k) + W_N^k \cdot X_1(k) + W_N^{2k} \cdot X_2(k) + W_N^{3k} \cdot X_3(k) + W_N^{4k} \cdot X_4(k)\),
         * with \(W_N = e^{-2\pi i / N}\) and \(X_0(k)\), \(X_1(k)\), ..., \(X_4(k)\) being the five sub-FFTs,
         * adjusted by transform_sign.
         */
        int sub_length = data_length / 5, index, target1, target2, target3, target4;
        fft_type twiddle_real, twiddle_imag, twiddle2_real, twiddle2_imag, twiddle3_real, twiddle3_imag, twiddle4_real, twiddle4_imag;
        fft_type tau0r, tau0i, tau1r, tau1i, tau2r, tau2i, tau3r, tau3i, tau4r, tau4i, tau5r, tau5i, tau6r, tau6i;
        fft_type a_real, a_imag, b_real, b_imag, c_real, c_imag, d_real, d_imag, e_real, e_imag;
        const fft_type c1 = 0.30901699437, c2 = -0.80901699437, s1 = 0.95105651629, s2 = 0.58778525229; // cos/sin of 72° and 144°
        new_stride = 5 * stride;

        mixed_radix_dit_rec(output_buffer, input_buffer, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + sub_length, input_buffer + stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 2 * sub_length, input_buffer + 2 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 3 * sub_length, input_buffer + 3 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 4 * sub_length, input_buffer + 4 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);

        target1 = sub_length;
        target2 = target1 + sub_length;
        target3 = target2 + sub_length;
        target4 = target3 + sub_length;

        a_real = output_buffer[0].re;       // Save first sub-FFT real part
        a_imag = output_buffer[0].im;       // Save first sub-FFT imaginary part
        b_real = output_buffer[target1].re; // Save second sub-FFT real part
        b_imag = output_buffer[target1].im; // Save second sub-FFT imaginary part
        c_real = output_buffer[target2].re; // Save third sub-FFT real part
        c_imag = output_buffer[target2].im; // Save third sub-FFT imaginary part
        d_real = output_buffer[target3].re; // Save fourth sub-FFT real part
        d_imag = output_buffer[target3].im; // Save fourth sub-FFT imaginary part
        e_real = output_buffer[target4].re; // Save fifth sub-FFT real part
        e_imag = output_buffer[target4].im; // Save fifth sub-FFT imaginary part

        tau0r = b_real + e_real; // Sum of second and fifth sub-FFTs (real)
        tau0i = b_imag + e_imag; // Sum of second and fifth sub-FFTs (imag)
        tau1r = c_real + d_real; // Sum of third and fourth sub-FFTs (real)
        tau1i = c_imag + d_imag; // Sum of third and fourth sub-FFTs (imag)
        tau2r = b_real - e_real; // Difference of second and fifth sub-FFTs (real)
        tau2i = b_imag - e_imag; // Difference of second and fifth sub-FFTs (imag)
        tau3r = c_real - d_real; // Difference of third and fourth sub-FFTs (real)
        tau3i = c_imag - d_imag; // Difference of third and fourth sub-FFTs (imag)

        output_buffer[0].re = a_real + tau0r + tau1r; // Combine all sums for first output (real)
        output_buffer[0].im = a_imag + tau0i + tau1i; // Combine all sums for first output (imag)
        tau4r = c1 * tau0r + c2 * tau1r;              // Apply 72° rotation (real)
        tau4i = c1 * tau0i + c2 * tau1i;              // Apply 72° rotation (imag)
        if (transform_sign == 1)
        {
            tau5r = s1 * tau2r + s2 * tau3r; // Apply 144° rotation for forward transform (real)
            tau5i = s1 * tau2i + s2 * tau3i; // Apply 144° rotation for forward transform (imag)
        }
        else
        {
            tau5r = -s1 * tau2r - s2 * tau3r; // Apply 144° rotation for inverse transform (real)
            tau5i = -s1 * tau2i - s2 * tau3i; // Apply 144° rotation for inverse transform (imag)
        }
        tau6r = a_real + tau4r;                    // Combine with center (real)
        tau6i = a_imag + tau4i;                    // Combine with center (imag)
        output_buffer[target1].re = tau6r + tau5i; // Second rotated output (real)
        output_buffer[target1].im = tau6i - tau5r; // Second rotated output (imag)
        output_buffer[target4].re = tau6r - tau5i; // Fifth rotated output (real)
        output_buffer[target4].im = tau6i + tau5r; // Fifth rotated output (imag)

        tau4r = c2 * tau0r + c1 * tau1r; // Apply 144° rotation (real)
        tau4i = c2 * tau0i + c1 * tau1i; // Apply 144° rotation (imag)
        if (transform_sign == 1)
        {
            tau5r = s2 * tau2r - s1 * tau3r; // Apply 72° rotation for forward transform (real)
            tau5i = s2 * tau2i - s1 * tau3i; // Apply 72° rotation for forward transform (imag)
        }
        else
        {
            tau5r = -s2 * tau2r + s1 * tau3r; // Apply 72° rotation for inverse transform (real)
            tau5i = -s2 * tau2i + s1 * tau3i; // Apply 72° rotation for inverse transform (imag)
        }
        tau6r = a_real + tau4r;                    // Combine with center (real)
        tau6i = a_imag + tau4i;                    // Combine with center (imag)
        output_buffer[target2].re = tau6r + tau5i; // Third rotated output (real)
        output_buffer[target2].im = tau6i - tau5r; // Third rotated output (imag)
        output_buffer[target3].re = tau6r - tau5i; // Fourth rotated output (real)
        output_buffer[target3].im = tau6i + tau5r; // Fourth rotated output (imag)

        for (int k = 1; k < sub_length; k++)
        {
            index = sub_length - 1 + 4 * k;                // Index into twiddle factors for current frequency
            twiddle_real = (fft_obj->twiddle + index)->re; // Real part of first twiddle factor
            twiddle_imag = (fft_obj->twiddle + index)->im; // Imaginary part of first twiddle factor
            index++;
            twiddle2_real = (fft_obj->twiddle + index)->re; // Real part of second twiddle factor
            twiddle2_imag = (fft_obj->twiddle + index)->im; // Imaginary part of second twiddle factor
            index++;
            twiddle3_real = (fft_obj->twiddle + index)->re; // Real part of third twiddle factor
            twiddle3_imag = (fft_obj->twiddle + index)->im; // Imaginary part of third twiddle factor
            index++;
            twiddle4_real = (fft_obj->twiddle + index)->re; // Real part of fourth twiddle factor
            twiddle4_imag = (fft_obj->twiddle + index)->im; // Imaginary part of fourth twiddle factor

            target1 = k + sub_length;       // Target for second sub-FFT result
            target2 = target1 + sub_length; // Target for third sub-FFT result
            target3 = target2 + sub_length; // Target for fourth sub-FFT result
            target4 = target3 + sub_length; // Target for fifth sub-FFT result

            a_real = output_buffer[k].re;                                                                   // Save current sub-FFT real part
            a_imag = output_buffer[k].im;                                                                   // Save current sub-FFT imaginary part
            b_real = output_buffer[target1].re * twiddle_real - output_buffer[target1].im * twiddle_imag;   // Apply first twiddle (real)
            b_imag = output_buffer[target1].im * twiddle_real + output_buffer[target1].re * twiddle_imag;   // Apply first twiddle (imag)
            c_real = output_buffer[target2].re * twiddle2_real - output_buffer[target2].im * twiddle2_imag; // Apply second twiddle (real)
            c_imag = output_buffer[target2].im * twiddle2_real + output_buffer[target2].re * twiddle2_imag; // Apply second twiddle (imag)
            d_real = output_buffer[target3].re * twiddle3_real - output_buffer[target3].im * twiddle3_imag; // Apply third twiddle (real)
            d_imag = output_buffer[target3].im * twiddle3_real + output_buffer[target3].re * twiddle3_imag; // Apply third twiddle (imag)
            e_real = output_buffer[target4].re * twiddle4_real - output_buffer[target4].im * twiddle4_imag; // Apply fourth twiddle (real)
            e_imag = output_buffer[target4].im * twiddle4_real + output_buffer[target4].re * twiddle4_imag; // Apply fourth twiddle (imag)

            tau0r = b_real + e_real; // Sum of second and fifth sub-FFTs (real)
            tau0i = b_imag + e_imag; // Sum of second and fifth sub-FFTs (imag)
            tau1r = c_real + d_real; // Sum of third and fourth sub-FFTs (real)
            tau1i = c_imag + d_imag; // Sum of third and fourth sub-FFTs (imag)
            tau2r = b_real - e_real; // Difference of second and fifth sub-FFTs (real)
            tau2i = b_imag - e_imag; // Difference of second and fifth sub-FFTs (imag)
            tau3r = c_real - d_real; // Difference of third and fourth sub-FFTs (real)
            tau3i = c_imag - d_imag; // Difference of third and fourth sub-FFTs (imag)

            output_buffer[k].re = a_real + tau0r + tau1r; // Combine all sums for current output (real)
            output_buffer[k].im = a_imag + tau0i + tau1i; // Combine all sums for current output (imag)
            tau4r = c1 * tau0r + c2 * tau1r;              // Apply 72° rotation (real)
            tau4i = c1 * tau0i + c2 * tau1i;              // Apply 72° rotation (imag)
            if (transform_sign == 1)
            {
                tau5r = s1 * tau2r + s2 * tau3r; // Apply 144° rotation for forward transform (real)
                tau5i = s1 * tau2i + s2 * tau3i; // Apply 144° rotation for forward transform (imag)
            }
            else
            {
                tau5r = -s1 * tau2r - s2 * tau3r; // Apply 144° rotation for inverse transform (real)
                tau5i = -s1 * tau2i - s2 * tau3i; // Apply 144° rotation for inverse transform (imag)
            }
            tau6r = a_real + tau4r;                    // Combine with center (real)
            tau6i = a_imag + tau4i;                    // Combine with center (imag)
            output_buffer[target1].re = tau6r + tau5i; // Second rotated output (real)
            output_buffer[target1].im = tau6i - tau5r; // Second rotated output (imag)
            output_buffer[target4].re = tau6r - tau5i; // Fifth rotated output (real)
            output_buffer[target4].im = tau6i + tau5r; // Fifth rotated output (imag)

            tau4r = c2 * tau0r + c1 * tau1r; // Apply 144° rotation (real)
            tau4i = c2 * tau0i + c1 * tau1i; // Apply 144° rotation (imag)
            if (transform_sign == 1)
            {
                tau5r = s2 * tau2r - s1 * tau3r; // Apply 72° rotation for forward transform (real)
                tau5i = s2 * tau2i - s1 * tau3i; // Apply 72° rotation for forward transform (imag)
            }
            else
            {
                tau5r = -s2 * tau2r + s1 * tau3r; // Apply 72° rotation for inverse transform (real)
                tau5i = -s2 * tau2i + s1 * tau3i; // Apply 72° rotation for inverse transform (imag)
            }
            tau6r = a_real + tau4r;                    // Combine with center (real)
            tau6i = a_imag + tau4i;                    // Combine with center (imag)
            output_buffer[target2].re = tau6r + tau5i; // Third rotated output (real)
            output_buffer[target2].im = tau6i - tau5r; // Third rotated output (imag)
            output_buffer[target3].re = tau6r - tau5i; // Fourth rotated output (real)
            output_buffer[target3].im = tau6i + tau5r; // Fourth rotated output (imag)
        }
    }
    else if (radix == 7)
    {
        /**
         * @brief Radix-7 decomposition for seven-point sub-FFTs.
         *
         * Recursively divides the data into seven sub-FFTs of length N/7, applies the FFT to each,
         * and combines the results using twiddle factors. This extends the DIT strategy for N=7^r,
         * where \(X(k) = X_0(k) + W_N^k \cdot X_1(k) + ... + W_N^{6k} \cdot X_6(k)\),
         * with \(W_N = e^{-2\pi i / N}\) and \(X_0(k)\), ..., \(X_6(k)\) being the seven sub-FFTs,
         * adjusted by transform_sign.
         */
        int sub_length = data_length / 7, index, target1, target2, target3, target4, target5, target6;
        fft_type twiddle_real, twiddle_imag, twiddle2_real, twiddle2_imag, twiddle3_real, twiddle3_imag, twiddle4_real, twiddle4_imag, twiddle5_real, twiddle5_imag, twiddle6_real, twiddle6_imag;
        fft_type tau0r, tau0i, tau1r, tau1i, tau2r, tau2i, tau3r, tau3i, tau4r, tau4i, tau5r, tau5i, tau6r, tau6i, tau7r, tau7i;
        fft_type a_real, a_imag, b_real, b_imag, c_real, c_imag, d_real, d_imag, e_real, e_imag, f_real, f_imag, g_real, g_imag;
        const fft_type c1 = 0.62348980185, c2 = -0.22252093395, c3 = -0.9009688679, s1 = 0.78183148246, s2 = 0.97492791218, s3 = 0.43388373911; // cos/sin of 51.43°, 102.86°, 154.29°
        new_stride = 7 * stride;

        mixed_radix_dit_rec(output_buffer, input_buffer, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + sub_length, input_buffer + stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 2 * sub_length, input_buffer + 2 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 3 * sub_length, input_buffer + 3 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 4 * sub_length, input_buffer + 4 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 5 * sub_length, input_buffer + 5 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 6 * sub_length, input_buffer + 6 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);

        target1 = sub_length;
        target2 = target1 + sub_length;
        target3 = target2 + sub_length;
        target4 = target3 + sub_length;
        target5 = target4 + sub_length;
        target6 = target5 + sub_length;

        a_real = output_buffer[0].re;       // Save first sub-FFT real part
        a_imag = output_buffer[0].im;       // Save first sub-FFT imaginary part
        b_real = output_buffer[target1].re; // Save second sub-FFT real part
        b_imag = output_buffer[target1].im; // Save second sub-FFT imaginary part
        c_real = output_buffer[target2].re; // Save third sub-FFT real part
        c_imag = output_buffer[target2].im; // Save third sub-FFT imaginary part
        d_real = output_buffer[target3].re; // Save fourth sub-FFT real part
        d_imag = output_buffer[target3].im; // Save fourth sub-FFT imaginary part
        e_real = output_buffer[target4].re; // Save fifth sub-FFT real part
        e_imag = output_buffer[target4].im; // Save fifth sub-FFT imaginary part
        f_real = output_buffer[target5].re; // Save sixth sub-FFT real part
        f_imag = output_buffer[target5].im; // Save sixth sub-FFT imaginary part
        g_real = output_buffer[target6].re; // Save seventh sub-FFT real part
        g_imag = output_buffer[target6].im; // Save seventh sub-FFT imaginary part

        tau0r = b_real + g_real; // Sum of second and seventh sub-FFTs (real)
        tau3r = b_real - g_real; // Difference of second and seventh sub-FFTs (real)
        tau0i = b_imag + g_imag; // Sum of second and seventh sub-FFTs (imag)
        tau3i = b_imag - g_imag; // Difference of second and seventh sub-FFTs (imag)
        tau1r = c_real + f_real; // Sum of third and sixth sub-FFTs (real)
        tau4r = c_real - f_real; // Difference of third and sixth sub-FFTs (real)
        tau1i = c_imag + f_imag; // Sum of third and sixth sub-FFTs (imag)
        tau4i = c_imag - f_imag; // Difference of third and sixth sub-FFTs (imag)
        tau2r = d_real + e_real; // Sum of fourth and fifth sub-FFTs (real)
        tau5r = d_real - e_real; // Difference of fourth and fifth sub-FFTs (real)
        tau2i = d_imag + e_imag; // Sum of fourth and fifth sub-FFTs (imag)
        tau5i = d_imag - e_imag; // Difference of fourth and fifth sub-FFTs (imag)

        output_buffer[0].re = a_real + tau0r + tau1r + tau2r;  // Combine all sums for first output (real)
        output_buffer[0].im = a_imag + tau0i + tau1i + tau2i;  // Combine all sums for first output (imag)
        tau6r = a_real + c1 * tau0r + c2 * tau1r + c3 * tau2r; // Apply 51.43° rotation (real)
        tau6i = a_imag + c1 * tau0i + c2 * tau1i + c3 * tau2i; // Apply 51.43° rotation (imag)
        if (transform_sign == 1)
        {
            tau7r = -s1 * tau3r - s2 * tau4r - s3 * tau5r; // Apply rotations for forward transform (real, -51.43°, -102.86°, -154.29°)
            tau7i = -s1 * tau3i - s2 * tau4i - s3 * tau5i; // Apply rotations for forward transform (imag, -51.43°, -102.86°, -154.29°)
        }
        else
        {
            tau7r = s1 * tau3r + s2 * tau4r + s3 * tau5r; // Apply rotations for inverse transform (real, +51.43°, +102.86°, +154.29°)
            tau7i = s1 * tau3i + s2 * tau4i + s3 * tau5i; // Apply rotations for inverse transform (imag, +51.43°, +102.86°, +154.29°)
        }

        output_buffer[target1].re = tau6r - tau7i; // Second rotated output (real)
        output_buffer[target1].im = tau6i + tau7r; // Second rotated output (imag)
        output_buffer[target6].re = tau6r + tau7i; // Seventh rotated output (real)
        output_buffer[target6].im = tau6i - tau7r; // Seventh rotated output (imag)

        tau6r = a_real + c2 * tau0r + c3 * tau1r + c1 * tau2r; // Apply 102.86° rotation (real)
        tau6i = a_imag + c2 * tau0i + c3 * tau1i + c1 * tau2i; // Apply 102.86° rotation (imag)
        if (transform_sign == 1)
        {
            tau7r = -s2 * tau3r + s3 * tau4r + s1 * tau5r; // Apply rotations for forward transform (real, -102.86°, +154.29°, +51.43°)
            tau7i = -s2 * tau3i + s3 * tau4i + s1 * tau5i; // Apply rotations for forward transform (imag, -102.86°, +154.29°, +51.43°)
        }
        else
        {
            tau7r = s2 * tau3r - s3 * tau4r - s1 * tau5r; // Apply rotations for inverse transform (real, +102.86°, -154.29°, -51.43°)
            tau7i = s2 * tau3i - s3 * tau4i - s1 * tau5i; // Apply rotations for inverse transform (imag, +102.86°, -154.29°, -51.43°)
        }

        output_buffer[target2].re = tau6r - tau7i; // Third rotated output (real)
        output_buffer[target2].im = tau6i + tau7r; // Third rotated output (imag)
        output_buffer[target5].re = tau6r + tau7i; // Sixth rotated output (real)
        output_buffer[target5].im = tau6i - tau7r; // Sixth rotated output (imag)

        tau6r = a_real + c3 * tau0r + c1 * tau1r + c2 * tau2r; // Apply 154.29° rotation (real)
        tau6i = a_imag + c3 * tau0i + c1 * tau1i + c2 * tau2i; // Apply 154.29° rotation (imag)
        if (transform_sign == 1)
        {
            tau7r = -s3 * tau3r + s1 * tau4r - s2 * tau5r; // Apply rotations for forward transform (real, -154.29°, +51.43°, -102.86°)
            tau7i = -s3 * tau3i + s1 * tau4i - s2 * tau5i; // Apply rotations for forward transform (imag, -154.29°, +51.43°, -102.86°)
        }
        else
        {
            tau7r = s3 * tau3r - s1 * tau4r + s2 * tau5r; // Apply rotations for inverse transform (real, +154.29°, -51.43°, +102.86°)
            tau7i = s3 * tau3i - s1 * tau4i + s2 * tau5i; // Apply rotations for inverse transform (imag, +154.29°, -51.43°, +102.86°)
        }

        output_buffer[target3].re = tau6r - tau7i; // Fourth rotated output (real)
        output_buffer[target3].im = tau6i + tau7r; // Fourth rotated output (imag)
        output_buffer[target4].re = tau6r + tau7i; // Fifth rotated output (real)
        output_buffer[target4].im = tau6i - tau7r; // Fifth rotated output (imag)

        for (int k = 1; k < sub_length; k++)
        {
            /**
             * @brief Combine sub-FFT results for higher indices using twiddle factors.
             *
             * For each frequency index k > 0, applies twiddle factors to rotate and combine
             * the sub-FFT results from seven recursive calls. This step leverages precomputed
             * twiddle factors to perform phase shifts, ensuring correct alignment of frequency components.
             * Mathematically, this corresponds to \(X(k) = \sum_{n=0}^{6} x(n) \cdot W_N^{kn}\),
             * where \(W_N = e^{-2\pi i / N}\) for forward transforms, adjusted by transform_sign.
             */
            index = sub_length - 1 + 6 * k;                // Index into twiddle factors for current frequency
            twiddle_real = (fft_obj->twiddle + index)->re; // Real part of first twiddle factor
            twiddle_imag = (fft_obj->twiddle + index)->im; // Imaginary part of first twiddle factor
            index++;
            twiddle2_real = (fft_obj->twiddle + index)->re; // Real part of second twiddle factor
            twiddle2_imag = (fft_obj->twiddle + index)->im; // Imaginary part of second twiddle factor
            index++;
            twiddle3_real = (fft_obj->twiddle + index)->re; // Real part of third twiddle factor
            twiddle3_imag = (fft_obj->twiddle + index)->im; // Imaginary part of third twiddle factor
            index++;
            twiddle4_real = (fft_obj->twiddle + index)->re; // Real part of fourth twiddle factor
            twiddle4_imag = (fft_obj->twiddle + index)->im; // Imaginary part of fourth twiddle factor
            index++;
            twiddle5_real = (fft_obj->twiddle + index)->re; // Real part of fifth twiddle factor
            twiddle5_imag = (fft_obj->twiddle + index)->im; // Imaginary part of fifth twiddle factor
            index++;
            twiddle6_real = (fft_obj->twiddle + index)->re; // Real part of sixth twiddle factor
            twiddle6_imag = (fft_obj->twiddle + index)->im; // Imaginary part of sixth twiddle factor

            target1 = k + sub_length;       // Target for second sub-FFT result
            target2 = target1 + sub_length; // Target for third sub-FFT result
            target3 = target2 + sub_length; // Target for fourth sub-FFT result
            target4 = target3 + sub_length; // Target for fifth sub-FFT result
            target5 = target4 + sub_length; // Target for sixth sub-FFT result
            target6 = target5 + sub_length; // Target for seventh sub-FFT result

            a_real = output_buffer[k].re;                                                                   // Save current sub-FFT real part
            a_imag = output_buffer[k].im;                                                                   // Save current sub-FFT imaginary part
            b_real = output_buffer[target1].re * twiddle_real - output_buffer[target1].im * twiddle_imag;   // Apply first twiddle (real)
            b_imag = output_buffer[target1].im * twiddle_real + output_buffer[target1].re * twiddle_imag;   // Apply first twiddle (imag)
            c_real = output_buffer[target2].re * twiddle2_real - output_buffer[target2].im * twiddle2_imag; // Apply second twiddle (real)
            c_imag = output_buffer[target2].im * twiddle2_real + output_buffer[target2].re * twiddle2_imag; // Apply second twiddle (imag)
            d_real = output_buffer[target3].re * twiddle3_real - output_buffer[target3].im * twiddle3_imag; // Apply third twiddle (real)
            d_imag = output_buffer[target3].im * twiddle3_real + output_buffer[target3].re * twiddle3_imag; // Apply third twiddle (imag)
            e_real = output_buffer[target4].re * twiddle4_real - output_buffer[target4].im * twiddle4_imag; // Apply fourth twiddle (real)
            e_imag = output_buffer[target4].im * twiddle4_real + output_buffer[target4].re * twiddle4_imag; // Apply fourth twiddle (imag)
            f_real = output_buffer[target5].re * twiddle5_real - output_buffer[target5].im * twiddle5_imag; // Apply fifth twiddle (real)
            f_imag = output_buffer[target5].im * twiddle5_real + output_buffer[target5].re * twiddle5_imag; // Apply fifth twiddle (imag)
            g_real = output_buffer[target6].re * twiddle6_real - output_buffer[target6].im * twiddle6_imag; // Apply sixth twiddle (real)
            g_imag = output_buffer[target6].im * twiddle6_real + output_buffer[target6].re * twiddle6_imag; // Apply sixth twiddle (imag)

            tau0r = b_real + g_real; // Sum of second and seventh sub-FFTs (real)
            tau3r = b_real - g_real; // Difference of second and seventh sub-FFTs (real)
            tau0i = b_imag + g_imag; // Sum of second and seventh sub-FFTs (imag)
            tau3i = b_imag - g_imag; // Difference of second and seventh sub-FFTs (imag)
            tau1r = c_real + f_real; // Sum of third and sixth sub-FFTs (real)
            tau4r = c_real - f_real; // Difference of third and sixth sub-FFTs (real)
            tau1i = c_imag + f_imag; // Sum of third and sixth sub-FFTs (imag)
            tau4i = c_imag - f_imag; // Difference of third and sixth sub-FFTs (imag)
            tau2r = d_real + e_real; // Sum of fourth and fifth sub-FFTs (real)
            tau5r = d_real - e_real; // Difference of fourth and fifth sub-FFTs (real)
            tau2i = d_imag + e_imag; // Sum of fourth and fifth sub-FFTs (imag)
            tau5i = d_imag - e_imag; // Difference of fourth and fifth sub-FFTs (imag)

            output_buffer[k].re = a_real + tau0r + tau1r + tau2r;  // Combine all sums for current output (real)
            output_buffer[k].im = a_imag + tau0i + tau1i + tau2i;  // Combine all sums for current output (imag)
            tau6r = a_real + c1 * tau0r + c2 * tau1r + c3 * tau2r; // Apply 51.43° rotation (real)
            tau6i = a_imag + c1 * tau0i + c2 * tau1i + c3 * tau2i; // Apply 51.43° rotation (imag)
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

            output_buffer[target1].re = tau6r - tau7i; // Second rotated output (real)
            output_buffer[target1].im = tau6i + tau7r; // Second rotated output (imag)
            output_buffer[target6].re = tau6r + tau7i; // Seventh rotated output (real)
            output_buffer[target6].im = tau6i - tau7r; // Seventh rotated output (imag)

            tau6r = a_real + c2 * tau0r + c3 * tau1r + c1 * tau2r; // Apply 102.86° rotation (real)
            tau6i = a_imag + c2 * tau0i + c3 * tau1i + c1 * tau2i; // Apply 102.86° rotation (imag)
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

            output_buffer[target2].re = tau6r - tau7i; // Third rotated output (real)
            output_buffer[target2].im = tau6i + tau7r; // Third rotated output (imag)
            output_buffer[target5].re = tau6r + tau7i; // Sixth rotated output (real)
            output_buffer[target5].im = tau6i - tau7r; // Sixth rotated output (imag)

            tau6r = a_real + c3 * tau0r + c1 * tau1r + c2 * tau2r; // Apply 154.29° rotation (real)
            tau6i = a_imag + c3 * tau0i + c1 * tau1i + c2 * tau2i; // Apply 154.29° rotation (imag)
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

            output_buffer[target3].re = tau6r - tau7i; // Fourth rotated output (real)
            output_buffer[target3].im = tau6i + tau7r; // Fourth rotated output (imag)
            output_buffer[target4].re = tau6r + tau7i; // Fifth rotated output (real)
            output_buffer[target4].im = tau6i - tau7r; // Fifth rotated output (imag)
        }
    }
    else if (radix == 8)
    {
        /**
         * @brief Radix-8 decomposition for eight-point sub-FFTs.
         *
         * Recursively divides the data into eight sub-FFTs of length N/8, applies the FFT to each,
         * and combines the results using twiddle factors. This extends the DIT strategy for N=8^r,
         * where \(X(k) = X_0(k) + W_N^k \cdot X_1(k) + ... + W_N^{7k} \cdot X_7(k)\),
         * with \(W_N = e^{-2\pi i / N}\) and \(X_0(k)\), ..., \(X_7(k)\) being the eight sub-FFTs,
         * adjusted by transform_sign.
         */
        int sub_length = data_length / 8, index, target1, target2, target3, target4, target5, target6, target7;
        fft_type twiddle_real, twiddle_imag, twiddle2_real, twiddle2_imag, twiddle3_real, twiddle3_imag, twiddle4_real, twiddle4_imag, twiddle5_real, twiddle5_imag, twiddle6_real, twiddle6_imag, twiddle7_real, twiddle7_imag;
        fft_type tau0r, tau0i, tau1r, tau1i, tau2r, tau2i, tau3r, tau3i, tau4r, tau4i, tau5r, tau5i, tau6r, tau6i, tau7r, tau7i, tau8r, tau8i, tau9r, tau9i;
        fft_type a_real, a_imag, b_real, b_imag, c_real, c_imag, d_real, d_imag, e_real, e_imag, f_real, f_imag, g_real, g_imag, h_real, h_imag;
        const fft_type c1 = 0.70710678118654752440084436210485, s1 = c1; // sqrt(2)/2, approximately 0.707, used for 45° rotation
        new_stride = 8 * stride;

        mixed_radix_dit_rec(output_buffer, input_buffer, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + sub_length, input_buffer + stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 2 * sub_length, input_buffer + 2 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 3 * sub_length, input_buffer + 3 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 4 * sub_length, input_buffer + 4 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 5 * sub_length, input_buffer + 5 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 6 * sub_length, input_buffer + 6 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);
        mixed_radix_dit_rec(output_buffer + 7 * sub_length, input_buffer + 7 * stride, fft_obj, transform_sign, sub_length, new_stride, factor_index + 1);

        for (int k = 0; k < sub_length; k++)
        {
            /**
             * @brief Combine sub-FFT results for higher indices using twiddle factors.
             *
             * For each frequency index k, applies twiddle factors to rotate and combine
             * the sub-FFT results from eight recursive calls. This step leverages precomputed
             * twiddle factors to perform phase shifts, ensuring correct alignment of frequency components.
             * Mathematically, this corresponds to \(X(k) = \sum_{n=0}^{7} x(n) \cdot W_N^{kn}\),
             * where \(W_N = e^{-2\pi i / N}\) for forward transforms, adjusted by transform_sign.
             */
            index = sub_length - 1 + 7 * k;                // Index into twiddle factors for current frequency
            twiddle_real = (fft_obj->twiddle + index)->re; // Real part of first twiddle factor
            twiddle_imag = (fft_obj->twiddle + index)->im; // Imaginary part of first twiddle factor
            index++;
            twiddle2_real = (fft_obj->twiddle + index)->re; // Real part of second twiddle factor
            twiddle2_imag = (fft_obj->twiddle + index)->im; // Imaginary part of second twiddle factor
            index++;
            twiddle3_real = (fft_obj->twiddle + index)->re; // Real part of third twiddle factor
            twiddle3_imag = (fft_obj->twiddle + index)->im; // Imaginary part of third twiddle factor
            index++;
            twiddle4_real = (fft_obj->twiddle + index)->re; // Real part of fourth twiddle factor
            twiddle4_imag = (fft_obj->twiddle + index)->im; // Imaginary part of fourth twiddle factor
            index++;
            twiddle5_real = (fft_obj->twiddle + index)->re; // Real part of fifth twiddle factor
            twiddle5_imag = (fft_obj->twiddle + index)->im; // Imaginary part of fifth twiddle factor
            index++;
            twiddle6_real = (fft_obj->twiddle + index)->re; // Real part of sixth twiddle factor
            twiddle6_imag = (fft_obj->twiddle + index)->im; // Imaginary part of sixth twiddle factor
            index++;
            twiddle7_real = (fft_obj->twiddle + index)->re; // Real part of seventh twiddle factor
            twiddle7_imag = (fft_obj->twiddle + index)->im; // Imaginary part of seventh twiddle factor

            target1 = k + sub_length;       // Target for second sub-FFT result
            target2 = target1 + sub_length; // Target for third sub-FFT result
            target3 = target2 + sub_length; // Target for fourth sub-FFT result
            target4 = target3 + sub_length; // Target for fifth sub-FFT result
            target5 = target4 + sub_length; // Target for sixth sub-FFT result
            target6 = target5 + sub_length; // Target for seventh sub-FFT result
            target7 = target6 + sub_length; // Target for eighth sub-FFT result

            a_real = output_buffer[k].re;                                                                   // Save current sub-FFT real part
            a_imag = output_buffer[k].im;                                                                   // Save current sub-FFT imaginary part
            b_real = output_buffer[target1].re * twiddle_real - output_buffer[target1].im * twiddle_imag;   // Apply first twiddle (real)
            b_imag = output_buffer[target1].im * twiddle_real + output_buffer[target1].re * twiddle_imag;   // Apply first twiddle (imag)
            c_real = output_buffer[target2].re * twiddle2_real - output_buffer[target2].im * twiddle2_imag; // Apply second twiddle (real)
            c_imag = output_buffer[target2].im * twiddle2_real + output_buffer[target2].re * twiddle2_imag; // Apply second twiddle (imag)
            d_real = output_buffer[target3].re * twiddle3_real - output_buffer[target3].im * twiddle3_imag; // Apply third twiddle (real)
            d_imag = output_buffer[target3].im * twiddle3_real + output_buffer[target3].re * twiddle3_imag; // Apply third twiddle (imag)
            e_real = output_buffer[target4].re * twiddle4_real - output_buffer[target4].im * twiddle4_imag; // Apply fourth twiddle (real)
            e_imag = output_buffer[target4].im * twiddle4_real + output_buffer[target4].re * twiddle4_imag; // Apply fourth twiddle (imag)
            f_real = output_buffer[target5].re * twiddle5_real - output_buffer[target5].im * twiddle5_imag; // Apply fifth twiddle (real)
            f_imag = output_buffer[target5].im * twiddle5_real + output_buffer[target5].re * twiddle5_imag; // Apply fifth twiddle (imag)
            g_real = output_buffer[target6].re * twiddle6_real - output_buffer[target6].im * twiddle6_imag; // Apply sixth twiddle (real)
            g_imag = output_buffer[target6].im * twiddle6_real + output_buffer[target6].re * twiddle6_imag; // Apply sixth twiddle (imag)
            h_real = output_buffer[target7].re * twiddle7_real - output_buffer[target7].im * twiddle7_imag; // Apply seventh twiddle (real)
            h_imag = output_buffer[target7].im * twiddle7_real + output_buffer[target7].re * twiddle7_imag; // Apply seventh twiddle (imag)

            tau0r = a_real + e_real; // Sum of current and fifth sub-FFTs (real)
            tau4r = a_real - e_real; // Difference of current and fifth sub-FFTs (real)
            tau0i = a_imag + e_imag; // Sum of current and fifth sub-FFTs (imag)
            tau4i = a_imag - e_imag; // Difference of current and fifth sub-FFTs (imag)
            tau1r = b_real + h_real; // Sum of second and eighth sub-FFTs (real)
            tau5r = b_real - h_real; // Difference of second and eighth sub-FFTs (real)
            tau1i = b_imag + h_imag; // Sum of second and eighth sub-FFTs (imag)
            tau5i = b_imag - h_imag; // Difference of second and eighth sub-FFTs (imag)
            tau2r = d_real + f_real; // Sum of fourth and sixth sub-FFTs (real)
            tau6r = d_real - f_real; // Difference of fourth and sixth sub-FFTs (real)
            tau2i = d_imag + f_imag; // Sum of fourth and sixth sub-FFTs (imag)
            tau6i = d_imag - f_imag; // Difference of fourth and sixth sub-FFTs (imag)
            tau3r = c_real + g_real; // Sum of third and seventh sub-FFTs (real)
            tau7r = c_real - g_real; // Difference of third and seventh sub-FFTs (real)
            tau3i = c_imag + g_imag; // Sum of third and seventh sub-FFTs (imag)
            tau7i = c_imag - g_imag; // Difference of third and seventh sub-FFTs (imag)

            output_buffer[k].re = tau0r + tau1r + tau2r + tau3r;       // Combine all sums for current output (real)
            output_buffer[k].im = tau0i + tau1i + tau2i + tau3i;       // Combine all sums for current output (imag)
            output_buffer[target4].re = tau0r - tau1r - tau2r + tau3r; // Combine differences for fifth output (real)
            output_buffer[target4].im = tau0i - tau1i - tau2i + tau3i; // Combine differences for fifth output (imag)

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

            output_buffer[target1].re = tau8r - tau9i; // Second rotated output (real)
            output_buffer[target1].im = tau8i + tau9r; // Second rotated output (imag)
            output_buffer[target7].re = tau8r + tau9i; // Eighth rotated output (real)
            output_buffer[target7].im = tau8i - tau9r; // Eighth rotated output (imag)

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

            output_buffer[target2].re = tau8r - tau9i; // Third rotated output (real)
            output_buffer[target2].im = tau8i + tau9r; // Third rotated output (imag)
            output_buffer[target6].re = tau8r + tau9i; // Seventh rotated output (real)
            output_buffer[target6].im = tau8i - tau9r; // Seventh rotated output (imag)

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

            output_buffer[target3].re = tau8r - tau9i; // Fourth rotated output (real)
            output_buffer[target3].im = tau8i + tau9r; // Fourth rotated output (imag)
            output_buffer[target5].re = tau8r + tau9i; // Sixth rotated output (real)
            output_buffer[target5].im = tau8i - tau9r; // Sixth rotated output (imag)
        }
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
 * @brief Performs Bluestein’s FFT algorithm for arbitrary-length signals.
 *
 * Uses Bluestein’s chirp z-transform to compute the Fast Fourier Transform (FFT) for non-power-of-2 signal lengths.
 * This algorithm transforms the DFT of length \(N\) into a convolution of the input signal with a chirp sequence,
 * padded to a power-of-2 length for efficient computation using a standard FFT. It requires precomputed exponential
 * (chirp) terms and an FFT object initialized for the padded length to handle arbitrary \(N\) efficiently.
 *
 * @param[in] input_data Input signal data (length N).
 *                      The real and imaginary components of the input signal to be transformed.
 * @param[out] output_data Output FFT results (length N).
 *                       Stores the transformed complex values after applying Bluestein’s FFT.
 * @param[in] fft_obj FFT configuration object.
 *                   Contains the signal length, transform direction, prime factors, and precomputed twiddle factors,
 *                   temporarily modified for the padded length during computation.
 * @param[in] transform_direction Direction of the transform (+1 for forward, -1 for inverse).
 *                             Determines whether to perform a forward FFT (\(e^{-2\pi i k n / N}\)) or
 *                             inverse FFT (\(e^{+2\pi i k n / N}\)).
 * @param[in] signal_length Length of the input signal (N > 0).
 *                        The size of the input and output data, which must be positive.
 * @warning If memory allocation fails or signal_length is invalid (<= 0), the function exits with an error.
 * @note Temporarily modifies the FFT object’s configuration (length, algorithm type, and twiddle factors) for the
 *       padded length, restoring the original state afterward. The algorithm uses a chirp z-transform,
 *       padding the input to a power-of-2 length (or larger if needed) and performing two FFTs and a pointwise
 *       multiplication to compute the DFT of arbitrary length \(N\).
 *       Mathematically, Bluestein’s FFT leverages the identity \(X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-2\pi i k n / N}\)
 *       by transforming it into a convolution with a chirp sequence \(h(n) = e^{\pi i n^2 / N}\), enabling efficient
 *       computation via FFT-based convolution.
 */
static void bluestein_fft(fft_data *input_data, fft_data *output_data, fft_object fft_obj, int transform_direction, int signal_length)
{
    /**
     * @brief Validate signal length.
     */
    if (signal_length <= 0)
    {
        fprintf(stderr, "Error: Signal length (%d) must be positive for Bluestein’s FFT\n", signal_length);
        exit(EXIT_FAILURE);
    }

    /**
     * @brief Calculate padded length for convolution.
     * Uses the smallest power of 2 >= 2N-1 to ensure sufficient length for circular convolution without aliasing.
     */
    int min_padded_length = 2 * signal_length - 1;
    int padded_length = (int)pow(2.0, ceil(log2((double)min_padded_length)));
    int index, loop_index;
    int original_lt = fft_obj->lt, original_N = fft_obj->N, original_sgn = fft_obj->sgn;

    /**
     * @brief Temporarily reconfigure FFT object for padded length.
     * Sets lt to 0 (mixed-radix) for the padded FFT computation.
     */
    fft_obj->N = padded_length;
    fft_obj->lt = 0;

    /**
     * @brief Allocate temporary buffers.
     */
    fft_data *yn = (fft_data *)malloc(sizeof(fft_data) * padded_length);          // Padded input
    fft_data *hk = (fft_data *)malloc(sizeof(fft_data) * padded_length);          // Chirp FFT
    fft_data *temp_output = (fft_data *)malloc(sizeof(fft_data) * padded_length); // Intermediate storage
    fft_data *yno = (fft_data *)malloc(sizeof(fft_data) * padded_length);         // Inverse FFT output
    fft_data *hlt = (fft_data *)malloc(sizeof(fft_data) * signal_length);         // Chirp sequence

    if (yn == NULL || hk == NULL || temp_output == NULL || yno == NULL || hlt == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for Bluestein’s FFT arrays\n");
        free(yn);
        free(hk);
        free(temp_output);
        free(yno);
        free(hlt);
        exit(EXIT_FAILURE);
    }

    /**
     * @brief Generate and scale chirp sequence.
     * Uses bluestein_exp with precomputed values where available, scales by 1/padded_length for normalization.
     */
    bluestein_exp(temp_output, hlt, signal_length, padded_length);
    fft_type scale = 1.0 / padded_length;
    for (loop_index = 0; loop_index < padded_length; ++loop_index)
    {
        temp_output[loop_index].im *= scale;
        temp_output[loop_index].re *= scale;
    }

    /**
     * @brief Compute FFT of the chirp sequence.
     */
    fft_exec(fft_obj, temp_output, hk);

    /**
     * @brief Multiply input by chirp sequence.
     * Applies h(n) for forward FFT or h^*(n) for inverse FFT.
     */
    if (transform_direction == 1)
    { // Forward FFT
        for (index = 0; index < signal_length; index++)
        {
            temp_output[index].re = input_data[index].re * hlt[index].re + input_data[index].im * hlt[index].im;
            temp_output[index].im = -input_data[index].re * hlt[index].im + input_data[index].im * hlt[index].re;
        }
    }
    else
    { // Inverse FFT
        for (index = 0; index < signal_length; index++)
        {
            temp_output[index].re = input_data[index].re * hlt[index].re - input_data[index].im * hlt[index].im;
            temp_output[index].im = input_data[index].re * hlt[index].im + input_data[index].im * hlt[index].re;
        }
    }

    /**
     * @brief Zero-pad the multiplied signal.
     */
    for (index = signal_length; index < padded_length; index++)
    {
        temp_output[index].re = 0.0;
        temp_output[index].im = 0.0;
    }

    /**
     * @brief Compute FFT of the padded signal.
     */
    fft_exec(fft_obj, temp_output, yn);

    /**
     * @brief Pointwise multiplication in frequency domain.
     * Multiplies with hk (forward) or hk^* (inverse).
     */
    if (transform_direction == 1)
    {
        for (index = 0; index < padded_length; index++)
        {
            fft_type temp = yn[index].re * hk[index].re - yn[index].im * hk[index].im;
            yn[index].im = yn[index].re * hk[index].im + yn[index].im * hk[index].re;
            yn[index].re = temp;
        }
    }
    else
    {
        for (index = 0; index < padded_length; index++)
        {
            fft_type temp = yn[index].re * hk[index].re + yn[index].im * hk[index].im;
            yn[index].im = -yn[index].re * hk[index].im + yn[index].im * hk[index].re;
            yn[index].re = temp;
        }
    }

    /**
     * @brief Inverse FFT to time domain.
     * Temporarily adjusts twiddle factors for inverse transform.
     */
    for (loop_index = 0; loop_index < padded_length; ++loop_index)
    {
        (fft_obj->twiddle + loop_index)->im = -(fft_obj->twiddle + loop_index)->im;
    }
    fft_obj->sgn = -1 * transform_direction;
    fft_exec(fft_obj, yn, yno);

    /**
     * @brief Apply chirp sequence again to extract final results.
     */
    if (transform_direction == 1)
    {
        for (index = 0; index < signal_length; index++)
        {
            output_data[index].re = yno[index].re * hlt[index].re + yno[index].im * hlt[index].im;
            output_data[index].im = -yno[index].re * hlt[index].im + yno[index].im * hlt[index].re;
        }
    }
    else
    {
        for (index = 0; index < signal_length; index++)
        {
            output_data[index].re = yno[index].re * hlt[index].re - yno[index].im * hlt[index].im;
            output_data[index].im = yno[index].re * hlt[index].im + yno[index].im * hlt[index].re;
        }
    }

    /**
     * @brief Restore original FFT object state.
     */
    fft_obj->sgn = original_sgn;
    fft_obj->N = original_N;
    fft_obj->lt = original_lt;
    for (loop_index = 0; loop_index < padded_length; ++loop_index)
    {
        (fft_obj->twiddle + loop_index)->im = -(fft_obj->twiddle + loop_index)->im;
    }

    /**
     * @brief Free temporary buffers.
     */
    free(yn);
    free(yno);
    free(temp_output);
    free(hk);
    free(hlt);
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
