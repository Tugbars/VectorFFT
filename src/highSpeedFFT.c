

#include "highspeedFFT.h"
#ifdef FFT_ENABLE_PREFETCH
#include "prefetch_strategy.h"
#endif
#include "time.h"
#include "simd_math.h"
#include "fft_radix2.h"
#include "fft_radix3.h"
#include "fft_radix4.h"
#include "fft_radix5.h"
#include "fft_radix7.h"
#include "fft_radix8.h"
#include "fft_radix11.h"
#include "fft_radix13.h"
#include "fft_radix16.h"
#include "fft_radix32.h"
#include <immintrin.h>
#include <pthread.h>

//==============================================================================
// PRIME FACTORIZATION AND DIVISIBILITY SYSTEM
//==============================================================================

#define LOOKUP_MAX 1024

// Core primes used for FFT (includes composite radices 4, 8 for FFT optimization)
static const int primes[] = {
    2, 3, 4, 5, 7, 8, 11, 13, 17, 23, 29, 31, 37, 41, 43, 47, 53
};
static const int num_primes = sizeof(primes) / sizeof(primes[0]);

// Actual prime-only list (for factorization)
static const int true_primes[] = {
    2, 3, 5, 7, 11, 13, 17, 23, 29, 31, 37, 41, 43, 47, 53
};
static const int num_true_primes = sizeof(true_primes) / sizeof(true_primes[0]);

// Divisibility lookup table
static unsigned char dividebyN_lookup[LOOKUP_MAX]; // 0 = not divisible, 1 = divisible

// Bluestein chirp precomputation
static const int pre_sizes[] = {1, 2, 3, 4, 5, 7, 15, 20, 31, 64};
static const int num_pre = (int)(sizeof(pre_sizes) / sizeof(pre_sizes[0]));
static fft_data *all_chirps = NULL;

// Extended prime table for large factors
static int *extended_primes = NULL;
static int num_extended = 0;
static int max_extended_prime = 0;

//==============================================================================
// UNIFIED INITIALIZATION - Runs once at program startup
//==============================================================================
__attribute__((constructor))
static void init_fft_prime_system(void)
{
    //==========================================================================
    // 1. Build divisibility lookup table (for dividebyN up to 1024)
    //==========================================================================
    memset(dividebyN_lookup, 0, LOOKUP_MAX);
    dividebyN_lookup[1] = 1; // Special case: 1 is "divisible"
    
    for (int n = 2; n < LOOKUP_MAX; n++) {
        int temp = n;
        int divisible = 1;
        
        while (temp > 1) {
            int factored = 0;
            
            // Try dividing by each prime in our supported set
            for (int j = 0; j < num_primes; j++) {
                if (temp % primes[j] == 0) {
                    temp /= primes[j];
                    factored = 1;
                    break;
                }
            }
            
            if (!factored) {
                divisible = 0;
                break;
            }
        }
        
        if (divisible) {
            dividebyN_lookup[n] = 1;
        }
    }
    
    //==========================================================================
    // 2. Build extended prime table (59 to 10000) using Sieve of Eratosthenes
    //==========================================================================
    const int limit = 10000;
    char *sieve = (char *)calloc(limit + 1, 1);
    
    if (!sieve) {
        fprintf(stderr, "Warning: Failed to allocate sieve for extended primes\n");
        return;
    }
    
    // Sieve of Eratosthenes
    for (int i = 2; i * i <= limit; i++) {
        if (!sieve[i]) {
            for (int j = i * i; j <= limit; j += i)
                sieve[j] = 1;
        }
    }
    
    // Count primes > 53 (we already have primes up to 53)
    num_extended = 0;
    for (int i = 59; i <= limit; i++)
        if (!sieve[i]) num_extended++;
    
    if (num_extended > 0) {
        extended_primes = (int *)malloc(num_extended * sizeof(int));
        
        if (extended_primes) {
            int idx = 0;
            for (int i = 59; i <= limit; i++)
                if (!sieve[i]) extended_primes[idx++] = i;
            
            max_extended_prime = extended_primes[num_extended - 1];
        } else {
            fprintf(stderr, "Warning: Failed to allocate extended prime table\n");
        }
    }
    
    free(sieve);
}

//==============================================================================
// CLEANUP - Runs at program exit
//==============================================================================
__attribute__((destructor))
static void cleanup_fft_prime_system(void)
{
    if (extended_primes) {
        free(extended_primes);
        extended_primes = NULL;
    }
}

//==============================================================================
// DIVISIBILITY CHECK - Fast lookup for N < 1024, computed for larger N
//==============================================================================
int dividebyN(int number)
{
    // Fast path: lookup table for small N
    if (number < LOOKUP_MAX) {
        return dividebyN_lookup[number];
    }
    
    // Slow path: compute for larger N
    int temp = number;
    
    // Try small primes first
    for (int i = 0; i < num_primes && temp > 1; i++) {
        while (temp % primes[i] == 0) {
            temp /= primes[i];
        }
    }
    
    return (temp == 1) ? 1 : 0;
}

//==============================================================================
// PRIME FACTORIZATION - Three-phase hybrid approach
//==============================================================================
/**
 * @brief Fast prime factorization using hybrid approach
 * 
 * Phase 1: Small primes (2-53) using existing true_primes array
 * Phase 2: Medium primes (59-10000) using precomputed extended_primes table
 * Phase 3: Large primes (>10000) using wheel factorization (6k±1)
 * 
 * @param number The number to factorize (must be > 0)
 * @param factors_array Output array for prime factors (must hold at least 32 elements)
 * @return Number of prime factors found, or 0 on error
 */
int factors(int number, int *factors_array)
{
    if (factors_array == NULL || number <= 0) {
        fprintf(stderr, "Error: Invalid inputs for factors - number: %d\n", number);
        return 0;
    }

    int index = 0;
    int n = number;
    
    //==========================================================================
    // PHASE 1: Factor using small primes (2-53)
    //==========================================================================
    for (int i = 0; i < num_true_primes && n > 1; i++) {
        int p = true_primes[i];
        
        // Early exit if p² > n (remaining n must be prime or 1)
        if (p * p > n) break;
        
        while (n % p == 0) {
            if (index >= 32) {
                fprintf(stderr, "Error: Too many prime factors (>32)\n");
                return index;
            }
            factors_array[index++] = p;
            n /= p;
        }
    }
    
    //==========================================================================
    // PHASE 2: Factor using extended prime table (59-10000)
    //==========================================================================
    if (extended_primes && n > 1) {
        for (int i = 0; i < num_extended && n > 1; i++) {
            int p = extended_primes[i];
            
            // Early exit
            if (p * p > n) break;
            
            while (n % p == 0) {
                if (index >= 32) {
                    fprintf(stderr, "Error: Too many prime factors (>32)\n");
                    return index;
                }
                factors_array[index++] = p;
                n /= p;
            }
        }
    }
    
    //==========================================================================
    // PHASE 3: Wheel factorization for very large primes (>10000)
    //==========================================================================
    if (n > 1) {
        // Check if we need wheel factorization
        // (n is either prime or has factors > max_extended_prime)
        int sqrt_n = (int)sqrt((double)n) + 1;
        
        // Only do wheel factorization if n might have large factors
        if (!extended_primes || sqrt_n > max_extended_prime) {
            // Start from first candidate after max_extended_prime
            int start_k = extended_primes ? (max_extended_prime + 6) / 6 : 1668;
            
            for (int k = start_k; 6*k - 1 <= sqrt_n && n > 1; k++) {
                // Check 6k-1
                int p1 = 6*k - 1;
                while (n % p1 == 0) {
                    if (index >= 32) {
                        fprintf(stderr, "Error: Too many prime factors (>32)\n");
                        return index;
                    }
                    factors_array[index++] = p1;
                    n /= p1;
                    sqrt_n = (int)sqrt((double)n) + 1;
                }
                
                // Check 6k+1
                int p2 = 6*k + 1;
                if (p2 <= sqrt_n) {
                    while (n % p2 == 0) {
                        if (index >= 32) {
                            fprintf(stderr, "Error: Too many prime factors (>32)\n");
                            return index;
                        }
                        factors_array[index++] = p2;
                        n /= p2;
                        sqrt_n = (int)sqrt((double)n) + 1;
                    }
                }
            }
        }
    }
    
    //==========================================================================
    // Remaining n is a prime factor (or 1)
    //==========================================================================
    if (n > 1) {
        if (index >= 32) {
            fprintf(stderr, "Error: Too many prime factors (>32)\n");
            return index;
        }
        factors_array[index++] = n;
    }
    
    return index;
}

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
// Use 0.5-ULP minimax polynomials if available
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

    const int child_scratch_base = scratch_offset + need_this_stage;

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
        fft_radix2_butterfly(output_buffer, sub_outputs, stage_tw, sub_len, transform_sign);
    }
    else if (radix == 3)
    {
        fft_radix3_butterfly(output_buffer, sub_outputs, stage_tw, sub_len, transform_sign);
    }
    else if (radix == 4)
    {
        fft_radix4_butterfly(output_buffer, sub_outputs, stage_tw, sub_len, transform_sign);
    }
    else if (radix == 5)
    {
        fft_radix5_butterfly(output_buffer, sub_outputs, stage_tw, sub_len, transform_sign);
    }
    else if (radix == 7)
    {
       fft_radix7_butterfly(output_buffer, sub_outputs, stage_tw, sub_len, transform_sign);
    }
    else if (radix == 8)
    {
       fft_radix8_butterfly(output_buffer, sub_outputs, stage_tw, sub_len, transform_sign);
    }
    else if (radix == 11)
    {
       fft_radix11_butterfly(output_buffer, sub_outputs, stage_tw, sub_len, transform_sign);
    }
    else if (radix == 13)
    {
       fft_radix13_butterfly(output_buffer, sub_outputs, stage_tw, sub_len, transform_sign);
    }
    else if (radix == 16)
    {
        fft_radix16_butterfly(output_buffer, sub_outputs, stage_tw, sub_len, transform_sign);
    }
    // Corrected Radix-32 Implementation
    else if (radix == 32)
    {
       fft_radix32_butterfly(output_buffer, sub_outputs, stage_tw, sub_len, transform_sign);
    }
    else
    {
        //==========================================================================
        // GENERAL RADIX FALLBACK - FIXED SCRATCH MANAGEMENT
        //==========================================================================

        const int r = radix;
        const int K = data_length / r; // child FFT length
        const int next_stride = r * stride;
        const int nst = r - 1;

        // How much scratch THIS stage needs (sub_outputs [+ stage_tw if not precomp])
        const int need_this = (fft_obj->twiddle_factors &&
                               factor_index < fft_obj->num_precomputed_stages)
                                  ? (r * K)
                                  : (r * K + nst * K);

        if (scratch_offset + need_this > fft_obj->max_scratch_size)
            return; // or handle error

        fft_data *sub_outputs = fft_obj->scratch + scratch_offset;
        fft_data *stage_tw = NULL;

        const int have_precomp =
            (fft_obj->twiddle_factors && factor_index < fft_obj->num_precomputed_stages);

        if (have_precomp)
        {
            stage_tw = fft_obj->twiddle_factors + fft_obj->stage_twiddle_offset[factor_index];
        }
        else
        {
            stage_tw = sub_outputs + r * K; // twiddles live after sub_outputs in scratch
        }

        // ---- Child recursion gets its OWN scratch above this stage's block ----
        const int stage_scratch = need_this;
        const int child_scratch_offset = scratch_offset + stage_scratch;

        for (int j = 0; j < r; ++j)
        {
            mixed_radix_dit_rec(
                /* child writes its final outputs here: */ sub_outputs + j * K,
                /* child reads from: */ input_buffer + j * stride,
                fft_obj, transform_sign,
                /* child length: */ K,
                /* child stride: */ next_stride,
                /* next factor idx: */ factor_index + 1,
                /* child's own scratch: */ child_scratch_offset);
        }

        // ---- Build per-stage twiddles if not precomputed ----
        if (!have_precomp)
        {
            // Current stage length (what this call is combining):
            const int N_stage = r * K; // == data_length
            // Map stage-local exponent to the GLOBAL twiddle table:
            // stride_tbl = N_global / N_stage
            const int stride_tbl = fft_obj->n_fft / data_length;

            // Fill layout: for each column k, store (r-1) twiddles [j=1..r-1]
            for (int k = 0; k < K; ++k)
            {
                const int base = nst * k;
                for (int j = 1; j < r; ++j)
                {
                    // local exponent e = (j * k) mod N_stage (DIT Cooley–Tukey)
                    const int e_local = (j * k) % N_stage;
                    // global index into twiddle table:
                    const int idxTbl = (e_local * stride_tbl) % fft_obj->n_fft;
                    stage_tw[base + (j - 1)] = fft_obj->twiddles[idxTbl];
                }
            }
            // NOTE: No extra sign flip here — fft_obj->twiddles were already
            // adjusted for forward/inverse in fft_init().
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

                    // Precompute first few powers
                    fft_data ph = {1.0, 0.0}; // ph^0

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
