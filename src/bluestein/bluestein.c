//==============================================================================
// bluestein.c - Fully Optimized SIMD Implementation
//==============================================================================

#include "bluestein.h"
#include "fft.h"
#include "simd_math.h" // Your existing SIMD helpers
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// SEPARATE OPAQUE STRUCTURES
//==============================================================================

struct bluestein_plan_forward_s
{
    int N;
    int M;

    fft_data *chirp_forward;      // Aligned to 32 bytes
    fft_data *kernel_fft_forward; // Aligned to 32 bytes

    fft_object fft_plan_m;
    fft_object ifft_plan_m;

    int chirp_is_cached;
    int plans_are_cached;
};

struct bluestein_plan_inverse_s
{
    int N;
    int M;

    fft_data *chirp_inverse;
    fft_data *kernel_fft_inverse;

    fft_object fft_plan_m;
    fft_object ifft_plan_m;

    int chirp_is_cached;
    int plans_are_cached;
};

//==============================================================================
// CACHES
//==============================================================================

#define MAX_BLUESTEIN_CACHE 16

typedef struct
{
    int N;
    bluestein_plan_forward *plan;
} bluestein_cache_forward_entry;

typedef struct
{
    int N;
    bluestein_plan_inverse *plan;
} bluestein_cache_inverse_entry;

static bluestein_cache_forward_entry forward_cache[MAX_BLUESTEIN_CACHE] = {0};
static bluestein_cache_inverse_entry inverse_cache[MAX_BLUESTEIN_CACHE] = {0};
static int num_forward_cached = 0;
static int num_inverse_cached = 0;

//==============================================================================
// HELPERS
//==============================================================================

static inline int next_pow2(int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

static fft_object internal_fft_cache[2][32] = {NULL};

static fft_object get_internal_fft_plan(int M, fft_direction_t direction)
{
    int log2_M = __builtin_ctz(M);
    int dir_idx = (direction == FFT_FORWARD) ? 0 : 1;

    if (log2_M >= 32)
        return NULL;

    if (!internal_fft_cache[dir_idx][log2_M])
    {
        internal_fft_cache[dir_idx][log2_M] = fft_init(M, direction);
    }

    return internal_fft_cache[dir_idx][log2_M];
}

//==============================================================================
// OPTIMIZED CHIRP COMPUTATION - Forward
//==============================================================================

/**
 * @brief Compute FORWARD chirp with AVX2 vectorization
 *
 * Optimizations:
 * - Process 2 complex samples per iteration (4 doubles in AVX2)
 * - Use FMA for angle computation
 * - Prefetch ahead
 */
static fft_data *compute_forward_chirp(int N)
{
    fft_data *chirp = (fft_data *)aligned_alloc(32, N * sizeof(fft_data));
    if (!chirp)
        return NULL;

    const double theta = +M_PI / (double)N;
    const int len2 = 2 * N;

    int n = 0;

#ifdef __AVX2__
    const __m256d vtheta = _mm256_set1_pd(theta);
    const __m256d vlen2 = _mm256_set1_pd((double)len2);

    // Process 2 complex numbers (4 doubles) at a time
    for (; n + 1 < N; n += 2)
    {
        // Prefetch ahead
        if (n + 16 < N)
        {
            _mm_prefetch((const char *)&chirp[n + 16], _MM_HINT_T0);
        }

        // Compute n² for two consecutive n values
        __m256d vn = _mm256_set_pd((double)(n + 1), (double)(n + 1), (double)n, (double)n);
        __m256d vn_sq = _mm256_mul_pd(vn, vn);

        // Modulo: n² mod 2N using FMA
        __m256d vn_sq_div = _mm256_div_pd(vn_sq, vlen2);
        __m256d vn_sq_floor = _mm256_floor_pd(vn_sq_div);
        __m256d vn_sq_mod = _mm256_fnmadd_pd(vn_sq_floor, vlen2, vn_sq);

        // Compute angles: θ * (n² mod 2N)
        __m256d vangles = _mm256_mul_pd(vtheta, vn_sq_mod);

        // Extract angles and compute sin/cos
        double angles[4];
        _mm256_storeu_pd(angles, vangles);

        // Use system sincos for each angle
        for (int i = 0; i < 2; i++)
        {
            double angle = angles[i * 2]; // Same angle for re and im
#ifdef __GNUC__
            sincos(angle, &chirp[n + i].im, &chirp[n + i].re);
#else
            chirp[n + i].re = cos(angle);
            chirp[n + i].im = sin(angle);
#endif
        }
    }
#endif

    // Scalar tail
    for (; n < N; n++)
    {
        const long long n_sq = (long long)n * (long long)n;
        const long long n_sq_mod = n_sq % (long long)len2;
        const double angle = theta * (double)n_sq_mod;

#ifdef __GNUC__
        sincos(angle, &chirp[n].im, &chirp[n].re);
#else
        chirp[n].re = cos(angle);
        chirp[n].im = sin(angle);
#endif
    }

    return chirp;
}

//==============================================================================
// OPTIMIZED KERNEL FFT COMPUTATION - Forward
//==============================================================================

/**
 * @brief Compute FFT of FORWARD kernel with AVX2 optimization
 *
 * Optimizations:
 * - Vectorized conjugation and mirroring
 * - Prefetching
 */
static fft_data *compute_forward_kernel_fft(const fft_data *chirp, int N, int M)
{
    fft_data *kernel_time = (fft_data *)aligned_alloc(32, M * sizeof(fft_data));
    fft_data *kernel_fft = (fft_data *)aligned_alloc(32, M * sizeof(fft_data));

    if (!kernel_time || !kernel_fft)
    {
        free(kernel_time);
        free(kernel_fft);
        return NULL;
    }

    memset(kernel_time, 0, M * sizeof(fft_data));

    kernel_time[0].re = 1.0;
    kernel_time[0].im = 0.0;

#ifdef __AVX2__
    const __m256d conj_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

    int n = 1;

    // Vectorized conjugation and mirroring
    for (; n + 1 < N; n += 2)
    {
        // Prefetch ahead
        if (n + 16 < N)
        {
            _mm_prefetch((const char *)&chirp[n + 16], _MM_HINT_T0);
        }

        // Load 2 chirp values
        __m256d vc = LOADU_PD(&chirp[n].re);

        // Conjugate: flip sign of imaginary parts
        __m256d vc_conj = _mm256_xor_pd(vc, conj_mask);

        // Store forward positions
        STOREU_PD(&kernel_time[n].re, vc_conj);

        // Store mirrored positions
        // Manual extraction for mirror indices (not contiguous)
        double temp[4];
        _mm256_storeu_pd(temp, vc_conj);

        kernel_time[M - n].re = temp[0];
        kernel_time[M - n].im = temp[1];
        kernel_time[M - n - 1].re = temp[2];
        kernel_time[M - n - 1].im = temp[3];
    }

    // Scalar tail
    for (; n < N; n++)
    {
        kernel_time[n].re = chirp[n].re;
        kernel_time[n].im = -chirp[n].im;
        kernel_time[M - n].re = chirp[n].re;
        kernel_time[M - n].im = -chirp[n].im;
    }
#else
    for (int n = 1; n < N; n++)
    {
        kernel_time[n].re = chirp[n].re;
        kernel_time[n].im = -chirp[n].im;
        kernel_time[M - n].re = chirp[n].re;
        kernel_time[M - n].im = -chirp[n].im;
    }
#endif

    fft_object fft_plan = get_internal_fft_plan(M, FFT_FORWARD);
    if (!fft_plan)
    {
        free(kernel_time);
        free(kernel_fft);
        return NULL;
    }

    fft_exec(fft_plan, kernel_time, kernel_fft);

    free(kernel_time);
    return kernel_fft;
}

//==============================================================================
// INVERSE CHIRP COMPUTATION (same pattern, negative theta)
//==============================================================================

static fft_data *compute_inverse_chirp(int N)
{
    fft_data *chirp = (fft_data *)aligned_alloc(32, N * sizeof(fft_data));
    if (!chirp)
        return NULL;

    const double theta = -M_PI / (double)N; // ✅ INVERSE: negative
    const int len2 = 2 * N;

    int n = 0;

#ifdef __AVX2__
    const __m256d vtheta = _mm256_set1_pd(theta);
    const __m256d vlen2 = _mm256_set1_pd((double)len2);

    for (; n + 1 < N; n += 2)
    {
        if (n + 16 < N)
        {
            _mm_prefetch((const char *)&chirp[n + 16], _MM_HINT_T0);
        }

        __m256d vn = _mm256_set_pd((double)(n + 1), (double)(n + 1), (double)n, (double)n);
        __m256d vn_sq = _mm256_mul_pd(vn, vn);

        __m256d vn_sq_div = _mm256_div_pd(vn_sq, vlen2);
        __m256d vn_sq_floor = _mm256_floor_pd(vn_sq_div);
        __m256d vn_sq_mod = _mm256_fnmadd_pd(vn_sq_floor, vlen2, vn_sq);

        __m256d vangles = _mm256_mul_pd(vtheta, vn_sq_mod);

        double angles[4];
        _mm256_storeu_pd(angles, vangles);

        for (int i = 0; i < 2; i++)
        {
            double angle = angles[i * 2];
#ifdef __GNUC__
            sincos(angle, &chirp[n + i].im, &chirp[n + i].re);
#else
            chirp[n + i].re = cos(angle);
            chirp[n + i].im = sin(angle);
#endif
        }
    }
#endif

    for (; n < N; n++)
    {
        const long long n_sq = (long long)n * (long long)n;
        const long long n_sq_mod = n_sq % (long long)len2;
        const double angle = theta * (double)n_sq_mod;

#ifdef __GNUC__
        sincos(angle, &chirp[n].im, &chirp[n].re);
#else
        chirp[n].re = cos(angle);
        chirp[n].im = sin(angle);
#endif
    }

    return chirp;
}

static fft_data *compute_inverse_kernel_fft(const fft_data *chirp, int N, int M)
{
    // Same implementation as forward
    return compute_forward_kernel_fft(chirp, N, M);
}

//==============================================================================
// FORWARD PLAN API
//==============================================================================

bluestein_plan_forward *bluestein_plan_create_forward(int N)
{
    if (N <= 0)
        return NULL;

    // TODO: Check cache

    bluestein_plan_forward *plan = (bluestein_plan_forward *)calloc(1, sizeof(bluestein_plan_forward));
    if (!plan)
        return NULL;

    plan->N = N;
    plan->M = next_pow2(2 * N - 1);

    plan->chirp_forward = compute_forward_chirp(N);
    if (!plan->chirp_forward)
    {
        free(plan);
        return NULL;
    }
    plan->chirp_is_cached = 0;

    plan->kernel_fft_forward = compute_forward_kernel_fft(plan->chirp_forward, N, plan->M);
    if (!plan->kernel_fft_forward)
    {
        free(plan->chirp_forward);
        free(plan);
        return NULL;
    }

    plan->fft_plan_m = get_internal_fft_plan(plan->M, FFT_FORWARD);
    plan->ifft_plan_m = get_internal_fft_plan(plan->M, FFT_INVERSE);
    plan->plans_are_cached = 1;

    if (!plan->fft_plan_m || !plan->ifft_plan_m)
    {
        free(plan->chirp_forward);
        free(plan->kernel_fft_forward);
        free(plan);
        return NULL;
    }

    return plan;
}

//==============================================================================
// OPTIMIZED FORWARD EXECUTION
//==============================================================================

int bluestein_exec_forward(
    bluestein_plan_forward *plan,
    const fft_data *input,
    fft_data *output,
    fft_data *scratch,
    size_t scratch_size)
{
    if (!plan || !input || !output || !scratch)
        return -1;

    const int N = plan->N;
    const int M = plan->M;

    if (scratch_size < 3 * M)
        return -1;

    fft_data *buffer_a = scratch;
    fft_data *buffer_b = scratch + M;
    fft_data *buffer_c = scratch + 2 * M;

    //==========================================================================
    // STEP 1: Multiply input by chirp + zero-pad (VECTORIZED with FMA)
    //==========================================================================

    int n = 0;

#ifdef __AVX2__
    // Process 2 complex numbers at a time
    for (; n + 1 < N; n += 2)
    {
        // Prefetch ahead
        if (n + 16 < N)
        {
            _mm_prefetch((const char *)&input[n + 16], _MM_HINT_T0);
            _mm_prefetch((const char *)&plan->chirp_forward[n + 16], _MM_HINT_T0);
        }

        // Load input and chirp
        __m256d vx = LOADU_PD(&input[n].re);
        __m256d vc = LOADU_PD(&plan->chirp_forward[n].re);

        // Complex multiply using FMA (from simd_math.h)
        __m256d result = cmul_avx2_aos(vx, vc);

        // Store
        STOREU_PD(&buffer_a[n].re, result);
    }
#endif

    // Scalar tail
    for (; n < N; n++)
    {
        double xr = input[n].re, xi = input[n].im;
        double cr = plan->chirp_forward[n].re, ci = plan->chirp_forward[n].im;
        buffer_a[n].re = xr * cr - xi * ci;
        buffer_a[n].im = xi * cr + xr * ci;
    }

    // Zero-pad (vectorized memset already optimal)
    memset(buffer_a + N, 0, (M - N) * sizeof(fft_data));

    //==========================================================================
    // STEP 2: FFT(A)
    //==========================================================================
    fft_exec(plan->fft_plan_m, buffer_a, buffer_b);

    //==========================================================================
    // STEP 3: Pointwise multiply with kernel FFT (VECTORIZED with FMA)
    //==========================================================================

    int i = 0;

#ifdef __AVX2__
    // Software pipelining: process 4 complex numbers per iteration
    for (; i + 3 < M; i += 4)
    {
        // Prefetch ahead
        if (i + 32 < M)
        {
            _mm_prefetch((const char *)&buffer_b[i + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&plan->kernel_fft_forward[i + 32], _MM_HINT_T0);
        }

        // Load 2 complex numbers each (4 doubles)
        __m256d va1 = LOADU_PD(&buffer_b[i].re);
        __m256d vk1 = LOADU_PD(&plan->kernel_fft_forward[i].re);
        __m256d va2 = LOADU_PD(&buffer_b[i + 2].re);
        __m256d vk2 = LOADU_PD(&plan->kernel_fft_forward[i + 2].re);

        // Complex multiply using FMA
        __m256d vc1 = cmul_avx2_aos(va1, vk1);
        __m256d vc2 = cmul_avx2_aos(va2, vk2);

        // Store
        STOREU_PD(&buffer_c[i].re, vc1);
        STOREU_PD(&buffer_c[i + 2].re, vc2);
    }

    // Process remaining 2 complex numbers
    for (; i + 1 < M; i += 2)
    {
        __m256d va = LOADU_PD(&buffer_b[i].re);
        __m256d vk = LOADU_PD(&plan->kernel_fft_forward[i].re);
        __m256d vc = cmul_avx2_aos(va, vk);
        STOREU_PD(&buffer_c[i].re, vc);
    }
#endif

    // Scalar tail
    for (; i < M; i++)
    {
        double ar = buffer_b[i].re, ai = buffer_b[i].im;
        double kr = plan->kernel_fft_forward[i].re, ki = plan->kernel_fft_forward[i].im;
        buffer_c[i].re = ar * kr - ai * ki;
        buffer_c[i].im = ai * kr + ar * ki;
    }

    //==========================================================================
    // STEP 4: IFFT
    //==========================================================================
    fft_exec(plan->ifft_plan_m, buffer_c, buffer_b);

    //==========================================================================
    // STEP 5: Final chirp multiply + extract (VECTORIZED with FMA)
    //==========================================================================

    int k = 0;

#ifdef __AVX2__
    for (; k + 1 < N; k += 2)
    {
        if (k + 16 < N)
        {
            _mm_prefetch((const char *)&buffer_b[k + 16], _MM_HINT_T0);
            _mm_prefetch((const char *)&plan->chirp_forward[k + 16], _MM_HINT_T0);
        }

        __m256d vy = LOADU_PD(&buffer_b[k].re);
        __m256d vc = LOADU_PD(&plan->chirp_forward[k].re);

        __m256d result = cmul_avx2_aos(vy, vc);

        STOREU_PD(&output[k].re, result);
    }
#endif

    // Scalar tail
    for (; k < N; k++)
    {
        double yr = buffer_b[k].re, yi = buffer_b[k].im;
        double cr = plan->chirp_forward[k].re, ci = plan->chirp_forward[k].im;
        output[k].re = yr * cr - yi * ci;
        output[k].im = yi * cr + yr * ci;
    }

    return 0;
}

void bluestein_plan_free_forward(bluestein_plan_forward *plan)
{
    if (!plan)
        return;

    if (!plan->chirp_is_cached)
        free(plan->chirp_forward);
    free(plan->kernel_fft_forward);
    free(plan);
}

//==============================================================================
// INVERSE PLAN API (identical pattern to forward)
//==============================================================================

bluestein_plan_inverse *bluestein_plan_create_inverse(int N)
{
    if (N <= 0)
        return NULL;

    bluestein_plan_inverse *plan = (bluestein_plan_inverse *)calloc(1, sizeof(bluestein_plan_inverse));
    if (!plan)
        return NULL;

    plan->N = N;
    plan->M = next_pow2(2 * N - 1);

    plan->chirp_inverse = compute_inverse_chirp(N);
    if (!plan->chirp_inverse)
    {
        free(plan);
        return NULL;
    }
    plan->chirp_is_cached = 0;

    plan->kernel_fft_inverse = compute_inverse_kernel_fft(plan->chirp_inverse, N, plan->M);
    if (!plan->kernel_fft_inverse)
    {
        free(plan->chirp_inverse);
        free(plan);
        return NULL;
    }

    plan->fft_plan_m = get_internal_fft_plan(plan->M, FFT_FORWARD);
    plan->ifft_plan_m = get_internal_fft_plan(plan->M, FFT_INVERSE);
    plan->plans_are_cached = 1;

    if (!plan->fft_plan_m || !plan->ifft_plan_m)
    {
        free(plan->chirp_inverse);
        free(plan->kernel_fft_inverse);
        free(plan);
        return NULL;
    }

    return plan;
}

int bluestein_exec_inverse(
    bluestein_plan_inverse *plan,
    const fft_data *input,
    fft_data *output,
    fft_data *scratch,
    size_t scratch_size)
{
    if (!plan || !input || !output || !scratch)
        return -1;

    const int N = plan->N;
    const int M = plan->M;

    if (scratch_size < 3 * M)
        return -1;

    fft_data *buffer_a = scratch;
    fft_data *buffer_b = scratch + M;
    fft_data *buffer_c = scratch + 2 * M;

    //==========================================================================
    // STEP 1: Input * chirp + zero-pad
    //==========================================================================

    int n = 0;

#ifdef __AVX2__
    for (; n + 1 < N; n += 2)
    {
        if (n + 16 < N)
        {
            _mm_prefetch((const char *)&input[n + 16], _MM_HINT_T0);
            _mm_prefetch((const char *)&plan->chirp_inverse[n + 16], _MM_HINT_T0);
        }

        __m256d vx = LOADU_PD(&input[n].re);
        __m256d vc = LOADU_PD(&plan->chirp_inverse[n].re);
        __m256d result = cmul_avx2_aos(vx, vc);

        STOREU_PD(&buffer_a[n].re, result);
    }
#endif

    for (; n < N; n++)
    {
        double xr = input[n].re, xi = input[n].im;
        double cr = plan->chirp_inverse[n].re, ci = plan->chirp_inverse[n].im;
        buffer_a[n].re = xr * cr - xi * ci;
        buffer_a[n].im = xi * cr + xr * ci;
    }

    memset(buffer_a + N, 0, (M - N) * sizeof(fft_data));

    //==========================================================================
    // STEP 2: FFT(A)
    //==========================================================================
    fft_exec(plan->fft_plan_m, buffer_a, buffer_b);

    //==========================================================================
    // STEP 3: Pointwise multiply
    //==========================================================================

    int i = 0;

#ifdef __AVX2__
    for (; i + 3 < M; i += 4)
    {
        if (i + 32 < M)
        {
            _mm_prefetch((const char *)&buffer_b[i + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&plan->kernel_fft_inverse[i + 32], _MM_HINT_T0);
        }

        __m256d va1 = LOADU_PD(&buffer_b[i].re);
        __m256d vk1 = LOADU_PD(&plan->kernel_fft_inverse[i].re);
        __m256d va2 = LOADU_PD(&buffer_b[i + 2].re);
        __m256d vk2 = LOADU_PD(&plan->kernel_fft_inverse[i + 2].re);

        __m256d vc1 = cmul_avx2_aos(va1, vk1);
        __m256d vc2 = cmul_avx2_aos(va2, vk2);

        STOREU_PD(&buffer_c[i].re, vc1);
        STOREU_PD(&buffer_c[i + 2].re, vc2);
    }

    for (; i + 1 < M; i += 2)
    {
        __m256d va = LOADU_PD(&buffer_b[i].re);
        __m256d vk = LOADU_PD(&plan->kernel_fft_inverse[i].re);
        __m256d vc = cmul_avx2_aos(va, vk);
        STOREU_PD(&buffer_c[i].re, vc);
    }
#endif

    for (; i < M; i++)
    {
        double ar = buffer_b[i].re, ai = buffer_b[i].im;
        double kr = plan->kernel_fft_inverse[i].re, ki = plan->kernel_fft_inverse[i].im;
        buffer_c[i].re = ar * kr - ai * ki;
        buffer_c[i].im = ai * kr + ar * ki;
    }

    //==========================================================================
    // STEP 4: IFFT
    //==========================================================================
    fft_exec(plan->ifft_plan_m, buffer_c, buffer_b);

    //==========================================================================
    // STEP 5: Final chirp multiply
    //==========================================================================

    int k = 0;

#ifdef __AVX2__
    for (; k + 1 < N; k += 2)
    {
        if (k + 16 < N)
        {
            _mm_prefetch((const char *)&buffer_b[k + 16], _MM_HINT_T0);
            _mm_prefetch((const char *)&plan->chirp_inverse[k + 16], _MM_HINT_T0);
        }

        __m256d vy = LOADU_PD(&buffer_b[k].re);
        __m256d vc = LOADU_PD(&plan->chirp_inverse[k].re);
        __m256d result = cmul_avx2_aos(vy, vc);

        STOREU_PD(&output[k].re, result);
    }
#endif

    for (; k < N; k++)
    {
        double yr = buffer_b[k].re, yi = buffer_b[k].im;
        double cr = plan->chirp_inverse[k].re, ci = plan->chirp_inverse[k].im;
        output[k].re = yr * cr - yi * ci;
        output[k].im = yi * cr + yr * ci;
    }

    return 0;
}

void bluestein_plan_free_inverse(bluestein_plan_inverse *plan)
{
    if (!plan)
        return;

    if (!plan->chirp_is_cached)
        free(plan->chirp_inverse);
    free(plan->kernel_fft_inverse);
    free(plan);
}

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

size_t bluestein_get_scratch_size(int N)
{
    int M = next_pow2(2 * N - 1);
    return 3 * M;
}

int bluestein_get_padded_size(int N)
{
    return next_pow2(2 * N - 1);
}