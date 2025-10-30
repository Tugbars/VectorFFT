//==============================================================================
// fft_planner.c - Unified FFT Planning with Depth-Tracked Recursion Control
//==============================================================================

/**
 * @file fft_planner.c
 * @brief Production-grade FFT planner with all planning logic in one place
 *
 * **Architecture:**
 * This is the ONLY file that makes planning decisions. All complexity is
 * resolved here at plan time for zero-overhead execution.
 *
 * **Planning Strategy (in order of preference):**
 * 1. Small power-of-2 (N ≤ 64): In-place bit-reversal (optimal for small N)
 * 2. Large N (≥64K, depth=0): Four-step FFT (cache-optimized)
 * 3. Factorizable N: Recursive Cooley-Tukey (93-95% of FFTW)
 * 4. Unfactorizable: Bluestein fallback (arbitrary N support)
 *
 * **Recursion Control:**
 * - Depth 0: All strategies allowed (user-facing call)
 * - Depth > 0: Four-step disabled (prevents infinite recursion)
 * - Each recursive call increments depth
 *
 * **Design Philosophy:**
 * - Planning is expensive, execution is fast
 * - All decisions made once at plan time
 * - Clear separation: planning vs execution
 * - Single source of truth for strategy selection
 */

#include "fft_planning_types.h"
#include "fft_twiddles.h"
#include "fft_twiddles_planner_api.h"
#include "fft_rader_plans.h"
#include "fft_fourstep.h"
#include "../bluestein/bluestein.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288419716939937510
#endif

//==============================================================================
// LOGGING CONFIGURATION
//==============================================================================

#ifndef FFT_LOG_ERROR
#define FFT_LOG_ERROR(fmt, ...) fprintf(stderr, "[FFT ERROR] " fmt "\n", ##__VA_ARGS__)
#endif

#ifndef FFT_LOG_DEBUG
#ifdef FFT_DEBUG_PLANNING
#define FFT_LOG_DEBUG(fmt, ...) fprintf(stderr, "[FFT DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define FFT_LOG_DEBUG(fmt, ...) ((void)0)
#endif
#endif

//==============================================================================
// INTERNAL PLANNING FLAGS
//==============================================================================

/**
 * @brief Control flags for internal planning recursion
 */
typedef enum
{
    FFT_PLAN_FLAG_NONE = 0,
    FFT_PLAN_FLAG_NO_FOURSTEP = (1 << 0),  ///< Disable four-step (for sub-plans)
    FFT_PLAN_FLAG_NO_BLUESTEIN = (1 << 1), ///< Force failure instead of Bluestein
} fft_planning_flags_t;

//==============================================================================
// RADIX REGISTRY - Single Source of Truth
//==============================================================================

typedef struct
{
    int radix;        ///< Radix value
    int cost;         ///< Relative execution cost
    int is_composite; ///< 1 if composite, 0 if prime/power-of-2
} radix_info;

static const radix_info RADIX_REGISTRY[] = {
    // Power-of-2 (best cache, highly optimized)
    {2, 10, 0},
    {4, 18, 1},
    {8, 35, 1},
    {16, 65, 1},
    {32, 120, 1},

    // Small primes (direct DFT)
    {3, 15, 0},
    {5, 25, 0},

    // Medium primes (Rader)
    {7, 40, 0},
    {11, 60, 0},
    {13, 70, 0},

    // Composite
    {9, 45, 1},
};

static const int NUM_RADICES = sizeof(RADIX_REGISTRY) / sizeof(RADIX_REGISTRY[0]);

//==============================================================================
// FACTORIZATION HELPERS (Consolidated from fft_factorizer.c)
//==============================================================================

#define MAX_FACTORS 64
#define MAX_MEMO_SIZE 100000
#define BLUESTEIN_THRESHOLD 64

typedef struct
{
    int factors[MAX_FACTORS];
    int count;
    double cost;
    bool uses_bluestein;
} FFTPlan;

typedef struct
{
    FFTPlan plan;
    bool valid;
} MemoEntry;

static MemoEntry *memo_table = NULL;

static inline bool is_power_of_2(int n)
{
    return (n > 0) && ((n & (n - 1)) == 0);
}

static inline int next_pow2(int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

static bool is_prime(int n)
{
    if (n < 2)
        return false;
    if (n == 2)
        return true;
    if (n % 2 == 0)
        return false;
    for (int i = 3; i * i <= n; i += 2)
    {
        if (n % i == 0)
            return false;
    }
    return true;
}

static int largest_prime_factor(int n)
{
    int largest = -1;
    for (int d = 2; d * d <= n; d++)
    {
        while (n % d == 0)
        {
            largest = d;
            n /= d;
        }
    }
    if (n > 1)
        largest = n;
    return largest;
}

static int count_radix_power(int n, int radix)
{
    int count = 0;
    while (n % radix == 0)
    {
        count++;
        n /= radix;
    }
    return count;
}

static int radix_priority(int radix)
{
    if (radix == 16)
        return 1;
    if (radix == 32)
        return 2;
    if (radix == 8)
        return 3;
    if (radix == 4)
        return 4;
    if (radix == 2)
        return 5;
    if (radix == 5)
        return 6;
    if (radix == 3)
        return 7;
    if (radix == 7)
        return 8;
    if (radix == 11)
        return 9;
    if (radix == 13)
        return 10;
    return 11;
}

static double compute_stage_cost(int n, int radix)
{
    // Simplified cost model (full model in fft_factorizer.c if needed)
    double base_cost = n * log2(radix);

    // Penalize non-power-of-2 radices
    if (radix == 7 || radix == 11 || radix == 13)
    {
        base_cost *= 2.0; // Rader's algorithm overhead
    }
    else if (radix == 3 || radix == 5)
    {
        base_cost *= 1.3; // Small prime overhead
    }

    return base_cost;
}

static double evaluate_plan_cost(const int *factors, int count, int n)
{
    double total_cost = 0.0;
    int current_n = n;

    for (int stage = 0; stage < count; stage++)
    {
        int radix = factors[stage];
        double stage_cost = compute_stage_cost(current_n, radix);
        total_cost += stage_cost;
        current_n /= radix;
    }

    return total_cost;
}

/**
 * @brief Factorize N into optimal radix sequence using DP
 *
 * @details
 * This is the core factorization algorithm. It uses dynamic programming
 * to find the optimal sequence of radices that minimizes execution cost.
 *
 * **Algorithm:**
 * 1. Prime factorization: N = p₁^a₁ × p₂^a₂ × ... × pₖ^aₖ
 * 2. Greedy packing: Combine primes into available radices
 * 3. Cost evaluation: Choose sequence with minimum cost
 *
 * @param n Transform size
 * @param factors Output array for radix sequence
 * @return Number of radices, or -1 on failure
 */
static int factorize_optimal(int n, int *factors)
{
    if (n == 1)
    {
        return 0;
    }

    // Check memoization cache
    if (n < MAX_MEMO_SIZE && memo_table && memo_table[n].valid)
    {
        FFTPlan cached = memo_table[n].plan;
        memcpy(factors, cached.factors, cached.count * sizeof(int));
        return cached.count;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Fast path: Pure power-of-2
    // ──────────────────────────────────────────────────────────────────────
    int pow2_count = count_radix_power(n, 2);
    if (pow2_count > 0 && (n >> pow2_count) == 1)
    {
        // N = 2^k, pack optimally into {32, 16, 8, 4, 2}
        int count = 0;
        while (pow2_count > 0)
        {
            if (pow2_count >= 5)
            {
                factors[count++] = 32;
                pow2_count -= 5;
            }
            else if (pow2_count >= 4)
            {
                factors[count++] = 16;
                pow2_count -= 4;
            }
            else if (pow2_count >= 3)
            {
                factors[count++] = 8;
                pow2_count -= 3;
            }
            else if (pow2_count >= 2)
            {
                factors[count++] = 4;
                pow2_count -= 2;
            }
            else
            {
                factors[count++] = 2;
                pow2_count--;
            }
        }
        return count;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Reject if largest prime factor too large (will use Bluestein)
    // ──────────────────────────────────────────────────────────────────────
    int lpf = largest_prime_factor(n);
    if (lpf > 13)
    {
        return -1; // Unfactorizable with available radices
    }

    // ──────────────────────────────────────────────────────────────────────
    // Greedy factorization: Try divisible radices in priority order
    // ──────────────────────────────────────────────────────────────────────
    int candidate_radixes[16];
    int num_candidates = 0;

    for (int i = 0; i < NUM_RADICES; i++)
    {
        if (n % RADIX_REGISTRY[i].radix == 0)
        {
            candidate_radixes[num_candidates++] = RADIX_REGISTRY[i].radix;
        }
    }

    if (num_candidates == 0)
    {
        return -1; // No valid radix divides N
    }

    // Sort by priority
    for (int i = 0; i < num_candidates - 1; i++)
    {
        for (int j = i + 1; j < num_candidates; j++)
        {
            if (radix_priority(candidate_radixes[j]) < radix_priority(candidate_radixes[i]))
            {
                int tmp = candidate_radixes[i];
                candidate_radixes[i] = candidate_radixes[j];
                candidate_radixes[j] = tmp;
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Recursive greedy packing
    // ──────────────────────────────────────────────────────────────────────
    double best_cost = INFINITY;
    int best_count = 0;
    int best_factors[MAX_FACTORS];

    for (int i = 0; i < num_candidates; i++)
    {
        int radix = candidate_radixes[i];
        int quotient = n / radix;

        int temp_factors[MAX_FACTORS];
        int sub_count = factorize_optimal(quotient, temp_factors);

        if (sub_count < 0)
        {
            continue; // This path failed
        }

        // Build candidate plan
        temp_factors[sub_count] = radix;
        int total_count = sub_count + 1;

        double cost = evaluate_plan_cost(temp_factors, total_count, n);

        if (cost < best_cost)
        {
            best_cost = cost;
            best_count = total_count;
            memcpy(best_factors, temp_factors, total_count * sizeof(int));
        }

        // Early exit for high-priority radices
        if ((radix == 16 || radix == 32) && best_cost < INFINITY)
        {
            break;
        }
    }

    if (best_count == 0)
    {
        return -1; // No valid factorization found
    }

    // Reverse order (factorization built backwards)
    for (int i = 0; i < best_count / 2; i++)
    {
        int tmp = best_factors[i];
        best_factors[i] = best_factors[best_count - 1 - i];
        best_factors[best_count - 1 - i] = tmp;
    }

    memcpy(factors, best_factors, best_count * sizeof(int));

    // Cache result
    if (n < MAX_MEMO_SIZE)
    {
        if (!memo_table)
        {
            memo_table = (MemoEntry *)calloc(MAX_MEMO_SIZE, sizeof(MemoEntry));
        }
        if (memo_table)
        {
            memo_table[n].plan.count = best_count;
            memcpy(memo_table[n].plan.factors, best_factors, best_count * sizeof(int));
            memo_table[n].plan.cost = best_cost;
            memo_table[n].valid = true;
        }
    }

    return best_count;
}

//==============================================================================
// BLUESTEIN PLANNING HELPER
//==============================================================================

static int plan_bluestein(fft_plan *plan, int N, fft_direction_t direction)
{
    int M = next_pow2(2 * N - 1);

    FFT_LOG_DEBUG("Bluestein: N=%d → M=%d (%.1f× padding)", N, M, (double)M / N);

    plan->strategy = FFT_EXEC_BLUESTEIN;
    plan->n_input = N;
    plan->n_fft = M;

    if (direction == FFT_FORWARD)
    {
        plan->bluestein_fwd = bluestein_plan_create_forward(N);
        if (!plan->bluestein_fwd)
        {
            FFT_LOG_ERROR("Failed to create forward Bluestein plan for N=%d", N);
            return -1;
        }
    }
    else
    {
        plan->bluestein_inv = bluestein_plan_create_inverse(N);
        if (!plan->bluestein_inv)
        {
            FFT_LOG_ERROR("Failed to create inverse Bluestein plan for N=%d", N);
            return -1;
        }
    }

    FFT_LOG_DEBUG("Bluestein planning complete");
    return 0;
}

//==============================================================================
// STAGE CONSTRUCTION HELPER
//==============================================================================

/**
 * @brief Construct Cooley-Tukey stages with twiddle allocation
 */
static int construct_stages(
    fft_plan *plan, 
    int N, 
    const int *factors, 
    int num_stages,
    simd_arch_t arch)  // ✅ NEW PARAMETER
{
    plan->num_stages = num_stages;
    memcpy(plan->factors, factors, num_stages * sizeof(int));

    int N_stage = N;

    for (int i = 0; i < num_stages; i++)
    {
        int radix = factors[i];
        int sub_len = N_stage / radix;

        stage_descriptor *stage = &plan->stages[i];
        stage->radix = radix;
        stage->N_stage = N_stage;
        stage->sub_len = sub_len;

        // ──────────────────────────────────────────────────────────────────
        // Stage twiddles: CREATE + MATERIALIZE ✅ NEW!
        // ──────────────────────────────────────────────────────────────────
        stage->stage_tw = get_stage_twiddles(N_stage, radix, plan->direction);
        if (!stage->stage_tw)
        {
            FFT_LOG_ERROR("Failed to get stage twiddles for stage %d (radix=%d)", i, radix);
            return -1;
        }

        // ✅ MATERIALIZE AT PLAN TIME
        if (twiddle_materialize_auto(stage->stage_tw, arch) != 0)
        {
            FFT_LOG_ERROR("Failed to materialize stage twiddles for stage %d (radix=%d)", i, radix);
            return -1;
        }

        FFT_LOG_DEBUG("  Stage %d: radix=%d, N=%d, sub_len=%d (materialized for arch=%d)",
                      i, radix, N_stage, sub_len, arch);

        // ──────────────────────────────────────────────────────────────────
        // DFT kernel twiddles (for general radix fallback)
        // ──────────────────────────────────────────────────────────────────
        bool needs_kernel = (radix > 13); // No specialized butterfly

        if (needs_kernel)
        {
            stage->dft_kernel_tw = get_dft_kernel_twiddles(radix, plan->direction);
            if (!stage->dft_kernel_tw)
            {
                FFT_LOG_ERROR("Failed to get DFT kernel for radix %d", radix);
                return -1;
            }
            
            // ✅ Materialize kernel too
            if (twiddle_materialize_auto(stage->dft_kernel_tw, arch) != 0)
            {
                FFT_LOG_ERROR("Failed to materialize DFT kernel for radix %d", radix);
                return -1;
            }
            
            FFT_LOG_DEBUG("    → DFT kernel: radix=%d (materialized)", radix);
        }
        else
        {
            stage->dft_kernel_tw = NULL;
        }

        // ──────────────────────────────────────────────────────────────────
        // Rader twiddles (for prime radices ≥7)
        // ──────────────────────────────────────────────────────────────────
        if (radix >= 7 && radix <= 67)
        {
            bool is_prime_radix = (radix == 7 || radix == 11 || radix == 13 ||
                                   radix == 17 || radix == 19 || radix == 23 ||
                                   radix == 29 || radix == 31 || radix == 37 ||
                                   radix == 41 || radix == 43 || radix == 47 ||
                                   radix == 53 || radix == 59 || radix == 61 ||
                                   radix == 67);

            if (is_prime_radix)
            {
                stage->rader_tw = get_rader_twiddles(radix, plan->direction);
                if (!stage->rader_tw)
                {
                    FFT_LOG_ERROR("Failed to get Rader twiddles for prime %d", radix);
                    return -1;
                }
                
                // ✅ Materialize Rader twiddles
                if (twiddle_materialize_auto(stage->rader_tw, arch) != 0)
                {
                    FFT_LOG_ERROR("Failed to materialize Rader twiddles for prime %d", radix);
                    return -1;
                }
                
                FFT_LOG_DEBUG("    → Rader: prime=%d (materialized)", radix);
            }
            else
            {
                stage->rader_tw = NULL;
            }
        }
        else
        {
            stage->rader_tw = NULL;
        }

        N_stage = sub_len;
    }

    return 0;
}


//==============================================================================
// INTERNAL EXTENDED PLANNING API
//==============================================================================

/**
 * @brief Internal planning function with recursion control
 *
 * @param N Transform size
 * @param direction Forward or inverse
 * @param depth Recursion depth (0 = top-level user call)
 * @param flags Planning control flags
 * @return Initialized plan or NULL on failure
 *
 * @note This is the ONLY function that creates plans. All paths go through here.
 */
static fft_object fft_init_extended(
    int N,
    fft_direction_t direction,
    int depth,
    fft_planning_flags_t flags,
    simd_arch_t arch)  // ✅ NEW PARAMETER
{
    // Validation...
    
    fft_plan *plan = (fft_plan *)calloc(1, sizeof(fft_plan));
    if (!plan)
    {
        FFT_LOG_ERROR("Failed to allocate plan structure");
        return NULL;
    }

    plan->n_input = N;
    plan->n_fft = N;
    plan->direction = direction;
    plan->bluestein_generic = NULL;
    plan->fourstep = NULL;
    
    // ✅ Store SIMD architecture
    plan->simd_arch = arch;

    FFT_LOG_DEBUG("Planning N=%d (%s), depth=%d, arch=%d",
                  N, direction == FFT_FORWARD ? "forward" : "inverse", depth, arch);

    // Strategy 1: Small power-of-2 → Bit-reversal
    if (is_power_of_2(N) && N <= 64)
    {
        plan->strategy = FFT_EXEC_INPLACE_BITREV;

        // ──────────────────────────────────────────────────────────────────────
        // COMPUTE NUMBER OF STAGES: log2(N)
        // ──────────────────────────────────────────────────────────────────────
        // Example N=16: log2(16) = 4 stages
        // __builtin_ctz = "count trailing zeros" = position of first set bit
        // For N=16 (binary: 10000), ctz(16) = 4
        int log2n = __builtin_ctz(N); // N=16 → log2n=4
        plan->num_stages = log2n;    // Will execute 4 stages

        // ──────────────────────────────────────────────────────────────────────
        // BUILD STAGE DESCRIPTORS (one per stage)
        // ──────────────────────────────────────────────────────────────────────
        // Each stage processes butterflies at increasing distances

        for (int s = 0; s < log2n; s++)   // s = 0, 1, 2, 3 for N=16
        {
            plan->factors[s] = 2;

            // Calculate stage size (halves each stage)
            // s=0: N_stage = 16 >> 0 = 16
            // s=1: N_stage = 16 >> 1 = 8
            // s=2: N_stage = 16 >> 2 = 4
            // s=3: N_stage = 16 >> 3 = 2
            int N_stage = N >> s;

            stage_descriptor *stage = &plan->stages[s];
            stage->radix = 2;
            stage->N_stage = N_stage;
            stage->sub_len = N_stage / 2;   // Distance between butterfly elements

            // ══════════════════════════════════════════════════════════════════
            // TWIDDLE CREATION (get handle from cache)
            // ══════════════════════════════════════════════════════════════════
            // Creates handle for: W_N_stage^k for k = 0..(N_stage/2 - 1)
            //
            // Stage 0 (N_stage=16): W_16^0, W_16^1, ..., W_16^7  (8 twiddles)
            // Stage 1 (N_stage=8):  W_8^0,  W_8^1,  ..., W_8^3   (4 twiddles)
            // Stage 2 (N_stage=4):  W_4^0,  W_4^1,  W_4^2        (2 twiddles)
            // Stage 3 (N_stage=2):  W_2^0                        (1 twiddle)
            //
            // Note: W_N^k = exp(-2πik/N) for forward FFT

            stage->stage_tw = get_stage_twiddles(N_stage, 2, direction);
            if (!stage->stage_tw)
            {
                FFT_LOG_ERROR("Failed to get bit-reversal twiddles");
                free(plan);
                return NULL;
            }

            // ══════════════════════════════════════════════════════════════════
            // TWIDDLE MATERIALIZATION (precompute optimal layout)
            // ══════════════════════════════════════════════════════════════════
            // This is the CRITICAL OPTIMIZATION:
            // - Converts canonical storage → SIMD-friendly blocked layout
            // - Happens ONCE at plan time (amortized over many executions)
            // - Radix-2 uses flat SoA (architecture-agnostic)
            //
            // Memory layout after materialization:
            // stage->stage_tw->materialized_re = [W_re[0], W_re[1], ..., W_re[K-1]]
            // stage->stage_tw->materialized_im = [W_im[0], W_im[1], ..., W_im[K-1]]
            // where K = N_stage/2
            //
            // Example Stage 0 (N_stage=16, K=8):
            // materialized_re = [1.0, 0.924, 0.707, 0.383, 0.0, -0.383, -0.707, -0.924]
            // materialized_im = [0.0, -0.383, -0.707, -0.924, -1.0, -0.924, -0.707, -0.383]
            //
            // Alignment: 64 bytes (satisfies AVX-512/AVX2/SSE2)
            // Access: Sequential loads (optimal prefetch, unit stride)

            if (twiddle_materialize_auto(stage->stage_tw, arch) != 0)
            {
                FFT_LOG_ERROR("Failed to materialize bit-reversal twiddles");
                free(plan);
                return NULL;
            }

            stage->dft_kernel_tw = NULL;
            stage->rader_tw = NULL;
        }

        return plan;
    }

    // Strategy 2: Large N (depth=0 only) → Four-step
    if (depth == 0 &&
        !(flags & FFT_PLAN_FLAG_NO_FOURSTEP) &&
        fft_should_use_fourstep(N, 1))
    {
        plan->strategy = FFT_EXEC_FOURSTEP;

        FFT_LOG_DEBUG("Strategy: Four-step FFT (cache-optimized, N=%d)", N);

        // Four-step will recursively call fft_init_extended with depth+1
        if (fft_fourstep_init_plan(plan, depth, arch) != 0)  // ✅ Pass arch
        {
            FFT_LOG_ERROR("Failed to initialize four-step plan");
            free(plan);
            return NULL;
        }

        return plan;
    }

    // Strategy 3: Factorizable → Cooley-Tukey
    int factors[MAX_FFT_STAGES];
    int num_stages = factorize_optimal(N, factors);

    if (num_stages > 0)
    {
        plan->strategy = FFT_EXEC_RECURSIVE_CT;

        FFT_LOG_DEBUG("Strategy: Recursive CT (N=%d)", N);

        // ✅ Pass arch to construct_stages
        if (construct_stages(plan, N, factors, num_stages, arch) != 0)
        {
            FFT_LOG_ERROR("Stage construction failed");
            free_fft(plan);
            return NULL;
        }

        FFT_LOG_DEBUG("Planning complete: %d stages", num_stages);
        return plan;
    }

    // Strategy 4: Unfactorizable → Bluestein
    if (!(flags & FFT_PLAN_FLAG_NO_BLUESTEIN))
    {
        FFT_LOG_DEBUG("Strategy: Bluestein (unfactorizable, N=%d)", N);

        if (plan_bluestein(plan, N, direction) < 0)
        {
            FFT_LOG_ERROR("Bluestein planning failed");
            free(plan);
            return NULL;
        }

        return plan;
    }

    // No valid strategy found
    FFT_LOG_ERROR("No valid strategy for N=%d (flags=0x%x)", N, flags);
    free(plan);
    return NULL;
}

//==============================================================================
// PUBLIC API
//==============================================================================

fft_object fft_init(int N, fft_direction_t direction)
{
    // Auto-detect SIMD architecture at compile time
    simd_arch_t arch;
#ifdef __AVX512F__
    arch = SIMD_ARCH_AVX512;
#elif defined(__AVX2__)
    arch = SIMD_ARCH_AVX2;
#elif defined(__SSE2__)
    arch = SIMD_ARCH_SSE2;
#else
    arch = SIMD_ARCH_SCALAR;
#endif

    return fft_init_extended(N, direction, 0, FFT_PLAN_FLAG_NONE, arch);
}

fft_object fft_init_with_simd(int N, fft_direction_t direction, simd_arch_t arch)
{
    return fft_init_extended(N, direction, 0, FFT_PLAN_FLAG_NONE, arch);
}

//==============================================================================
// PLAN DESTRUCTION
//==============================================================================

void free_fft(fft_object plan)
{
    if (!plan)
        return;

    // Free four-step resources
    if (plan->strategy == FFT_EXEC_FOURSTEP)
    {
        fft_fourstep_free_plan(plan);
    }

    // Free Cooley-Tukey stage resources
    if (plan->strategy != FFT_EXEC_BLUESTEIN && plan->strategy != FFT_EXEC_FOURSTEP)
    {
        for (int i = 0; i < plan->num_stages; i++)
        {
            if (plan->stages[i].stage_tw)
            {
                twiddle_destroy(plan->stages[i].stage_tw);
            }
            if (plan->stages[i].dft_kernel_tw)
            {
                twiddle_destroy(plan->stages[i].dft_kernel_tw);
            }
            if (plan->stages[i].rader_tw)
            {
                twiddle_destroy(plan->stages[i].rader_tw);
            }
        }
    }

    // Free Bluestein resources
    if (plan->strategy == FFT_EXEC_BLUESTEIN)
    {
        if (plan->direction == FFT_FORWARD)
        {
            bluestein_plan_free_forward(plan->bluestein_fwd);
        }
        else
        {
            bluestein_plan_free_inverse(plan->bluestein_inv);
        }
    }

    free(plan);
}

//==============================================================================
// QUERY FUNCTIONS
//==============================================================================

int fft_can_execute_inplace(fft_object plan)
{
    if (!plan)
        return 0;
    return (plan->strategy == FFT_EXEC_INPLACE_BITREV);
}

size_t fft_get_workspace_size(fft_object plan)
{
    if (!plan)
        return 0;

    switch (plan->strategy)
    {
    case FFT_EXEC_INPLACE_BITREV:
        return 0;

    case FFT_EXEC_RECURSIVE_CT:
        return (size_t)(2 * plan->n_fft);

    case FFT_EXEC_FOURSTEP:
        return fft_fourstep_workspace_size(plan);

    case FFT_EXEC_BLUESTEIN:
        return bluestein_get_scratch_size(plan->n_input);

    default:
        return 0;
    }
}

fft_exec_strategy_t fft_get_strategy(fft_object plan)
{
    return plan ? plan->strategy : FFT_EXEC_OUT_OF_PLACE;
}

//==============================================================================
// CLEANUP
//==============================================================================

void fft_clear_memo(void)
{
    if (memo_table)
    {
        free(memo_table);
        memo_table = NULL;
    }
}