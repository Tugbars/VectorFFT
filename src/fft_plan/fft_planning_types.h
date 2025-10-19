
/*

fft_init(N, direction)
    │
    ├─ Validate N > 0, direction valid
    │
    ├─ Allocate fft_plan struct (calloc)
    │   ├─ plan->n_input = N
    │   ├─ plan->n_fft = N
    │   ├─ plan->direction = direction
    │   └─ plan->use_bluestein = 0
    │
    ├─ factorize(N, plan->factors) → num_stages
    │     │
    │     ├─ Try radices in priority order:
    │     │   [32, 16, 13, 11, 9, 8, 7, 5, 4, 3, 2,
    │     │    17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]
    │     │
    │     ├─ While (n > 1):
    │     │   ├─ Find largest radix r that divides n
    │     │   ├─ factors[num_factors++] = r
    │     │   ├─ n /= r
    │     │   └─ IF (num_factors >= MAX_FFT_STAGES): ERROR
    │     │
    │     └─ Return: num_stages (success) or -1 (unfactorizable)
    │
    │
    ├─ ══════════════════════════════════════════════════════════
    │  BRANCH 1: BLUESTEIN PATH (unfactorizable N)
    │  ══════════════════════════════════════════════════════════
    │
    ├─ IF num_stages < 0:
    │   │
    │   └─ DISPATCH → plan_bluestein(plan, N, direction)
    │        │
    │        ├─ M = next_pow2(2*N - 1)  // Pad to power-of-2
    │        │   └─> Example: N=509 → M=1024
    │        │
    │        ├─ plan->use_bluestein = 1
    │        ├─ plan->n_input = N
    │        ├─ plan->n_fft = M
    │        │
    │        ├─ Allocate chirp twiddles: bluestein_tw[M]
    │        │   │
    │        │   ├─ sign = (direction == FFT_FORWARD) ? -1.0 : +1.0
    │        │   │
    │        │   └─ FOR k = 0..N-1:
    │        │       ├─ angle = sign * π * k² / N
    │        │       └─ bluestein_tw[k] = exp(i * angle)
    │        │           └─> Store as {cos(angle), sin(angle)}
    │        │
    │        │   └─ Zero-pad: bluestein_tw[N..M-1] = {0, 0}
    │        │
    │        ├─ ✅ Create SEPARATE forward/inverse plans for M:
    │        │   │
    │        │   ├─ IF direction == FFT_FORWARD:
    │        │   │   ├─ plan->bluestein_fwd = fft_init(M, FFT_FORWARD)  ← RECURSIVE
    │        │   │   └─ plan->bluestein_inv = fft_init(M, FFT_INVERSE)  ← RECURSIVE
    │        │   │
    │        │   └─ (Same for FFT_INVERSE direction)
    │        │       └─> ✅ Separate opaque types enforce type safety
    │        │
    │        └─ Return 0 (success) or -1 (error)
    │
    │   └─> RETURN plan (Bluestein planning complete)
    │
    │
    ├─ ══════════════════════════════════════════════════════════
    │  BRANCH 2: COOLEY-TUKEY PATH (factorizable N)
    │  ══════════════════════════════════════════════════════════
    │
    └─ ELSE (Cooley-Tukey path):
         │
         ├─ plan->num_stages = num_stages
         │
         ├─ Log factorization:
         │   └─> "Factorization: 13×11×7" (example)
         │
         │
         ├─ ════════════════════════════════════════════════════
         │  STAGE CONSTRUCTION LOOP
         │  ════════════════════════════════════════════════════
         │
         ├─ N_stage = N  // Initial transform size
         │
         └─ FOR stage i = 0 to num_stages-1:
              │
              ├─ radix = plan->factors[i]
              ├─ sub_len = N_stage / radix
              │
              ├─ stage_descriptor *stage = &plan->stages[i]
              ├─ stage->radix = radix
              ├─ stage->N_stage = N_stage
              ├─ stage->sub_len = sub_len
              │
              │
              ├─ ════════════════════════════════════════════════
              │  TWIDDLE MANAGER: Cooley-Tukey Stage Twiddles
              │  ════════════════════════════════════════════════
              │
              ├─ DISPATCH → compute_stage_twiddles(N_stage, radix, direction)
              │    │
              │    ├─ Validate: radix >= 2, N_stage >= radix
              │    │
              │    ├─ num_twiddles = (radix - 1) × sub_len
              │    │   └─> Layout: tw[k*(radix-1) + (r-1)] = W_N^(r*k)
              │    │       where W_N = exp(sign * 2πi/N_stage)
              │    │
              │    ├─ Allocate: tw = aligned_alloc(32, num_twiddles × sizeof(fft_data))
              │    │   └─> 32-byte aligned for AVX2
              │    │
              │    ├─ sign = (direction == FFT_FORWARD) ? -1.0 : +1.0
              │    ├─ base_angle = sign × 2π / N_stage
              │    │
              │    ├─ ┌─────────────────────────────────────────┐
              │    │   │ TWIDDLE COMPUTATION DISPATCH           │
              │    │   └─────────────────────────────────────────┘
              │    │
              │    ├─ IF __AVX2__ AND sub_len > 8:
              │    │   │
              │    │   └─ AVX2 Vectorized Path:
              │    │       │
              │    │       ├─ FOR r = 1 to radix-1:
              │    │       │   │
              │    │       │   ├─ offset = r - 1
              │    │       │   │
              │    │       │   └─ compute_twiddles_avx2(&tw[offset], sub_len, 
              │    │       │                             base_angle, r, radix-1)
              │    │       │        │
              │    │       │        ├─ vbase = _mm256_set1_pd(base_angle)
              │    │       │        ├─ vr = _mm256_set1_pd((double)r)
              │    │       │        │
              │    │       │        └─ FOR i = 0 to sub_len-1 (step 4):
              │    │       │            │
              │    │       │            ├─ Prefetch: &tw[(i+16) * interleave]
              │    │       │            │
              │    │       │            ├─ vi = {i, i+1, i+2, i+3}
              │    │       │            ├─ vang = vbase × vr × vi  (FMA)
              │    │       │            │
              │    │       │            ├─ Extract 4 angles
              │    │       │            │
              │    │       │            └─ FOR j = 0..3:
              │    │       │                ├─ idx = (i+j) × interleave
              │    │       │                └─ sincos_auto(angles[j], 
              │    │       │                     &tw[idx].im, &tw[idx].re)
              │    │       │                     │
              │    │       │                     ├─ IF |angle| ≤ π/4:
              │    │       │                     │   └─> sincos_minimax() 
              │    │       │                     │       (0.5 ULP, FMA polynomials)
              │    │       │                     │
              │    │       │                     └─ ELSE:
              │    │       │                         └─> sincos() / sin()+cos()
              │    │       │                             (system libc)
              │    │       │
              │    │       └─ Scalar tail for remainder
              │    │
              │    └─ ELSE (Scalar Path):
              │        │
              │        └─ FOR k = 0 to sub_len-1:
              │            └─ FOR r = 1 to radix-1:
              │                ├─ idx = k × (radix-1) + (r-1)
              │                ├─ angle = base_angle × r × k
              │                └─ sincos_auto(angle, &tw[idx].im, &tw[idx].re)
              │    │
              │    └─ Return tw (OWNED by stage)
              │
              ├─ stage->stage_tw = tw
              │
              ├─ Log: "Stage %d: radix=%d, N=%d, sub_len=%d, twiddles=%d"
              │
              │
              ├─ ════════════════════════════════════════════════
              │  RADER MANAGER: Prime Radix Convolution Twiddles
              │  ════════════════════════════════════════════════
              │
              ├─ IF is_prime(radix) AND radix >= 7:
              │   │
              │   │   // Known primes: 7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67
              │   │
              │   └─ DISPATCH → get_rader_twiddles(radix, direction)
              │        │
              │        ├─ mutex_lock()  // Thread-safe cache access
              │        │
              │        ├─ IF !g_cache_initialized:
              │        │   └─ init_rader_cache()
              │        │       ├─ memset(g_rader_cache, 0, ...)
              │        │       ├─ Pre-populate: create_rader_plan_for_prime(7)
              │        │       ├─ Pre-populate: create_rader_plan_for_prime(11)
              │        │       ├─ Pre-populate: create_rader_plan_for_prime(13)
              │        │       └─ g_cache_initialized = 1
              │        │
              │        ├─ ┌────────────────────────────────────────┐
              │        │   │ CACHE LOOKUP                          │
              │        │   └────────────────────────────────────────┘
              │        │
              │        ├─ FOR i = 0 to MAX_RADER_PRIMES-1:
              │        │   └─ IF g_rader_cache[i].prime == radix:
              │        │       ├─ entry = &g_rader_cache[i]
              │        │       │
              │        │       ├─ result = (direction == FFT_FORWARD)
              │        │       │           ? entry->conv_tw_fwd
              │        │       │           : entry->conv_tw_inv
              │        │       │
              │        │       ├─ mutex_unlock()
              │        │       └─ RETURN result  ← CACHE HIT ✅
              │        │
              │        ├─ // Cache miss - create new entry
              │        │
              │        ├─ mutex_unlock()  // Release during creation
              │        │
              │        ├─ ┌────────────────────────────────────────┐
              │        │   │ CREATE NEW RADER PLAN                 │
              │        │   └────────────────────────────────────────┘
              │        │
              │        └─ create_rader_plan_for_prime(radix)
              │             │
              │             ├─ Find primitive root g from database:
              │             │   │
              │             │   └─ g_primitive_roots[] = {
              │             │        {7,3}, {11,2}, {13,2}, {17,3}, {19,2},
              │             │        {23,5}, {29,2}, {31,3}, {37,2}, {41,6},
              │             │        {43,3}, {47,5}, {53,2}, {59,2}, {61,2}, {67,2}
              │             │      }
              │             │
              │             ├─ IF g < 0: ERROR (prime not in database)
              │             │
              │             ├─ Find free cache slot:
              │             │   └─ FOR i=0..MAX_RADER_PRIMES-1:
              │             │       IF g_rader_cache[i].prime == 0: slot=i; break
              │             │
              │             ├─ IF slot < 0: ERROR (cache full)
              │             │
              │             ├─ entry = &g_rader_cache[slot]
              │             │
              │             ├─ ┌──────────────────────────────────┐
              │             │   │ COMPUTE PERMUTATIONS            │
              │             │   └──────────────────────────────────┘
              │             │
              │             ├─ Allocate:
              │             │   ├─ entry->perm_in = malloc((radix-1) × sizeof(int))
              │             │   └─ entry->perm_out = malloc((radix-1) × sizeof(int))
              │             │
              │             ├─ compute_permutations(radix, g, perm_in, perm_out)
              │             │   │
              │             │   ├─ Input permutation (generator powers):
              │             │   │   └─ FOR i = 0 to radix-2:
              │             │   │       └─ perm_in[i] = g^i mod radix
              │             │   │           └─> [g^0, g^1, g^2, ..., g^(p-2)] mod p
              │             │   │
              │             │   └─ Output permutation (inverse mapping):
              │             │       └─ FOR i = 0 to radix-2:
              │             │           ├─ idx = perm_in[i] - 1  // Map to 0..(p-2)
              │             │           └─ perm_out[idx] = i
              │             │
              │             ├─ ┌──────────────────────────────────┐
              │             │   │ COMPUTE CONVOLUTION TWIDDLES    │
              │             │   └──────────────────────────────────┘
              │             │
              │             ├─ Allocate (32-byte aligned for AVX2):
              │             │   ├─ entry->conv_tw_fwd = aligned_alloc(32, (radix-1)×...)
              │             │   └─ entry->conv_tw_inv = aligned_alloc(32, (radix-1)×...)
              │             │
              │             ├─ FOR q = 0 to radix-2:
              │             │   │
              │             │   ├─ idx = perm_out[q]
              │             │   │
              │             │   ├─ FORWARD twiddle: exp(-2πi × idx / radix)
              │             │   │   ├─ angle_fwd = -2π × idx / radix
              │             │   │   └─ sincos_auto(angle_fwd, 
              │             │   │        &conv_tw_fwd[q].im, &conv_tw_fwd[q].re)
              │             │   │
              │             │   └─ INVERSE twiddle: exp(+2πi × idx / radix)
              │             │       ├─ angle_inv = +2π × idx / radix
              │             │       └─ sincos_auto(angle_inv, 
              │             │            &conv_tw_inv[q].im, &conv_tw_inv[q].re)
              │             │
              │             ├─ entry->prime = radix
              │             ├─ entry->primitive_root = g
              │             │
              │             ├─ Log: "Created Rader plan for prime %d (g=%d) in slot %d"
              │             │
              │             └─ Return 0 (success)
              │
              │        └─ Recursive call: get_rader_twiddles(radix, direction)
              │            └─> Now in cache, will return immediately ✅
              │
              ├─ stage->rader_tw = (returned pointer from cache)
              │   └─> ✅ NOT OWNED by stage (shared from global cache)
              │
              └─ ELSE:
                  └─ stage->rader_tw = NULL
                      └─> Non-prime or radix < 7 (e.g., 2,3,4,5,8,9,16,32)
              │
              ├─ Log: "  → Rader: prime=%d, conv_twiddles=%d" (if applicable)
              │
              └─ N_stage = sub_len  // Update for next stage
         
         │
         ├─ ════════════════════════════════════════════════════
         │  SCRATCH BUFFER ALLOCATION
         │  ════════════════════════════════════════════════════
         │
         ├─ Compute maximum scratch needed across all stages:
         │   │
         │   ├─ scratch_max = 0
         │   ├─ N_stage = N
         │   │
         │   └─ FOR i = 0 to num_stages-1:
         │       ├─ radix = plan->factors[i]
         │       ├─ sub_len = N_stage / radix
         │       ├─ stage_need = radix × sub_len
         │       ├─ IF stage_need > scratch_max:
         │       │   └─ scratch_max = stage_need
         │       └─ N_stage = sub_len
         │
         ├─ Add margin for Rader convolutions:
         │   └─ scratch_needed = scratch_max + 4×N
         │
         ├─ plan->scratch_size = scratch_needed
         ├─ plan->scratch = aligned_alloc(32, scratch_needed × sizeof(fft_data))
         │
         ├─ IF !plan->scratch: ERROR (allocation failed)
         │
         ├─ Log: "Scratch buffer: %zu elements (%.2f KB)"
         │
         └─ Log: "Planning complete!"
    
    └─ RETURN plan  ✅ PLANNING COMPLETE
```

---

## Memory Ownership Summary
```
fft_plan
├─ stages[i].stage_tw          ✅ OWNED (freed by free_stage_twiddles)
├─ stages[i].rader_tw          ❌ BORROWED (pointer to g_rader_cache[].conv_tw_*)
├─ scratch                     ✅ OWNED (freed by aligned_free)
├─ bluestein_tw               ✅ OWNED (if Bluestein, freed by aligned_free)
├─ bluestein_plan_fwd         ✅ OWNED (recursive fft_plan, freed by free_fft)
└─ bluestein_plan_inv         ✅ OWNED (recursive fft_plan, freed by free_fft)

g_rader_cache[i]  (GLOBAL, thread-safe)
├─ conv_tw_fwd                ✅ OWNED (freed by cleanup_rader_cache)
├─ conv_tw_inv                ✅ OWNED (freed by cleanup_rader_cache)
├─ perm_in                    ✅ OWNED (freed by cleanup_rader_cache)
└─ perm_out                   ✅ OWNED (freed by cleanup_rader_cache)

*/

//==============================================================================
// fft_planning_types.h (add to top-level comment)
//==============================================================================

/**
 * @file fft_planning_types.h
 * @brief Core types for FFTW-style FFT planning system
 * 
 * **NORMALIZATION CONVENTION (FFTW-compatible):**
 * 
 * Forward DFT (unnormalized):
 *   X[k] = Σ_{n=0}^{N-1} x[n] × exp(-2πikn/N)
 * 
 * Inverse DFT (unnormalized):
 *   x[n] = Σ_{k=0}^{N-1} X[k] × exp(+2πikn/N)
 * 
 * Round-trip identity:
 *   IDFT(DFT(x)) = N × x
 * 
 * Users must manually scale by 1/N if needed. This convention:
 * - Maximizes performance (no hidden multiplications)
 * - Provides flexibility (user chooses normalization point)
 * - Matches FFTW, ensuring compatibility
 * - Simplifies implementation (all butterflies unnormalized)
 */

#ifndef FFT_PLANNING_TYPES_H
#define FFT_PLANNING_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifndef FFT_LOG_ERROR
#define FFT_LOG_ERROR(fmt, ...) fprintf(stderr, "[FFT ERROR] " fmt "\n", ##__VA_ARGS__)
#endif

//==============================================================================
// CONFIGURATION
//==============================================================================

#define MAX_FFT_STAGES 32  ///< Maximum factorization depth (supports up to 2^32)

//==============================================================================
// FORWARD DECLARATIONS
//==============================================================================

/**
 * @brief Opaque type for forward Bluestein transform plans
 * 
 * Separate types for forward/inverse enforce type safety and allow
 * direction-specific optimizations (different chirp signs).
 */
typedef struct bluestein_plan_forward_s bluestein_plan_forward;

/**
 * @brief Opaque type for inverse Bluestein transform plans
 * 
 * Implementation lives in bluestein.c to hide internal complexity.
 */
typedef struct bluestein_plan_inverse_s bluestein_plan_inverse;

//==============================================================================
// BASIC TYPES
//==============================================================================

/**
 * @brief Complex number in interleaved format (AoS)
 * 
 * Layout chosen for:
 * - Easy complex arithmetic
 * - Natural alignment (16 bytes)
 * - AVX2 compatibility (2 complex = 1 YMM register)
 */
typedef struct { double re, im; } fft_data;

/**
 * @brief FFT direction: forward (analysis) or inverse (synthesis)
 * 
 * Values chosen to match sign in twiddle factor: exp(direction × 2πi/N)
 * This eliminates runtime branching in twiddle computation.
 */
typedef enum { 
    FFT_FORWARD = 1,   ///< Forward FFT (time → frequency), exp(-2πi/N)
    FFT_INVERSE = -1   ///< Inverse FFT (frequency → time), exp(+2πi/N)
} fft_direction_t;

//==============================================================================
// EXECUTION STRATEGY
//==============================================================================

/**
 * @brief Algorithm selected by planner based on transform size
 * 
 * Strategy determines memory requirements and execution path:
 * 
 * **Design rationale:**
 * - Power-of-2 sizes use bit-reversal (true in-place, zero workspace)
 * - Composite sizes use Stockham (cache-friendly, needs 1× workspace)
 * - Prime sizes use Bluestein (arbitrary N support, needs 3× workspace)
 * 
 * Strategy is fixed at plan time for optimal performance.
 */
typedef enum {
    FFT_EXEC_INPLACE_BITREV,    ///< Bit-reversal Cooley-Tukey (N = 2^k only, 0 workspace)
    FFT_EXEC_STOCKHAM,          ///< Stockham auto-sort (any composite N, 1N workspace)
    FFT_EXEC_BLUESTEIN,         ///< Bluestein chirp-z (arbitrary N, 3M workspace where M = 2^⌈log₂(2N-1)⌉)
    FFT_EXEC_OUT_OF_PLACE       ///< Generic out-of-place (reserved for future use)
} fft_exec_strategy_t;

//==============================================================================
// STAGE DESCRIPTOR
//==============================================================================

/**
 * @brief Pre-computed data for one Cooley-Tukey decomposition stage
 * 
 * **Memory ownership:**
 * - stage_tw: OWNED by stage (freed with plan)
 * - rader_tw: BORROWED from global cache (never freed by stage)
 * 
 * **Layout rationale:**
 * Stage twiddles use interleaved format tw[k*(radix-1) + (r-1)] = W^(r*k)
 * to optimize cache access during butterfly operations.
 */

typedef struct {
    int radix;         ///< Radix for this stage (2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 32, etc.)
    int N_stage;       ///< Transform size at this stage (N / product of previous radices)
    int sub_len;       ///< Butterfly stride (N_stage / radix)
    
    fft_data *stage_tw;     ///< Cooley-Tukey stage twiddles [(radix-1) × sub_len, OWNED]
    fft_data *rader_tw;     ///< Rader convolution twiddles [radix-1, BORROWED from cache]
    fft_data *dft_kernel_tw; ///< ⚡ NEW: DFT kernel twiddles [radix, OWNED]
                             ///< W_r[m] = exp(sign × 2πim/radix) for m=0..radix-1
                             ///< NULL for specialized radices (2,3,4,5,7,8,11,13)
                             ///< Populated for general radix fallback
    
} stage_descriptor;

//==============================================================================
// FFT PLAN - Everything pre-computed at planning time
//==============================================================================

/**
 * @brief Complete FFT execution plan with pre-computed twiddle factors
 * 
 * **Architecture overview:**
 * Plans follow the "compile once, execute many" pattern. All expensive
 * computations (factorization, twiddle generation, strategy selection)
 * happen once at planning time.
 * 
 * **Memory model:**
 * Plans own their twiddle factors but NOT workspace buffers. Users provide
 * workspace at execution time for maximum flexibility and thread safety.
 * 
 * **Execution paths:**
 * 1. Cooley-Tukey (power-of-2 or composite): num_stages butterflies using stages[]
 * 2. Bluestein (prime or arbitrary): delegates to bluestein_fwd/inv plans
 * 
 * **Design decisions:**
 * - No scratch buffer in plan: enables thread-safe execution with shared plans
 * - Separate Bluestein plans: type safety + direction-specific optimizations
 * - Factorization stored: enables optimal butterfly dispatch
 */
typedef struct fft_plan_struct {
    int n_input;               ///< Original input size requested by user
    int n_fft;                 ///< Actual FFT size (may differ for Bluestein: M = 2^⌈log₂(2N-1)⌉)
    fft_direction_t direction; ///< Transform direction (fixed at plan time)
    
    fft_exec_strategy_t strategy; ///< Execution algorithm selected by planner
    
    // ─────────────────────────────────────────────────────────────────────
    // Cooley-Tukey decomposition data (used if strategy != BLUESTEIN)
    // ─────────────────────────────────────────────────────────────────────
    
    int num_stages;                      ///< Number of decomposition stages (0 for Bluestein)
    int factors[MAX_FFT_STAGES];         ///< Radix sequence [r₀, r₁, ..., rₖ] where N = r₀×r₁×...×rₖ
    stage_descriptor stages[MAX_FFT_STAGES]; ///< Per-stage twiddles and metadata
    
    // ─────────────────────────────────────────────────────────────────────
    // Bluestein algorithm data (used if strategy == BLUESTEIN)
    // ─────────────────────────────────────────────────────────────────────
    
    /**
     * @brief Direction-specific Bluestein plans (type-safe union)
     * 
     * Only one field is active based on plan->direction:
     * - FFT_FORWARD: uses bluestein_fwd
     * - FFT_INVERSE: uses bluestein_inv
     * 
     * Separate types prevent accidental misuse and enable different
     * chirp computations (exp(+πin²/N) vs exp(-πin²/N)).
     */
    union {
        bluestein_plan_forward *bluestein_fwd;  ///< Forward Bluestein plan (active if direction == FFT_FORWARD)
        bluestein_plan_inverse *bluestein_inv;  ///< Inverse Bluestein plan (active if direction == FFT_INVERSE)
        void *bluestein_generic;                ///< Generic pointer for NULL checks
    };
    
} fft_plan;

/**
 * @brief Opaque plan handle for public API
 * 
 * Users interact with fft_object pointers; internal structure is hidden.
 */
typedef fft_plan* fft_object;

//==============================================================================
// RADER PLAN CACHE ENTRY
//==============================================================================

/**
 * @brief Global cache entry for prime-radix Rader algorithm data
 * 
 * **Rader's algorithm overview:**
 * For prime P, DFT reduces to:
 * 1. Separate DC term (index 0)
 * 2. Circular convolution of size P-1 using generator permutation
 * 
 * **Cache rationale:**
 * - Rader plans are expensive to compute (primitive root search, permutations)
 * - Same prime appears in many transforms (e.g., all N = 7×k)
 * - Thread-safe lazy initialization with mutex protection
 * 
 * **Memory ownership:**
 * Cache owns all fields; stages only borrow pointers. Cleanup at program exit.
 */
typedef struct {
    int prime;                 ///< Prime radix (7, 11, 13, 17, 19, 23, ..., 67)
    
    fft_data *conv_tw_fwd;     ///< Forward convolution twiddles [P-1 elements, exp(-2πi×g^q/P)]
    fft_data *conv_tw_inv;     ///< Inverse convolution twiddles [P-1 elements, exp(+2πi×g^q/P)]
    
    int *perm_in;              ///< Input permutation [P-1 elements]: [g^0, g^1, ..., g^(P-2)] mod P
    int *perm_out;             ///< Output permutation [P-1 elements]: inverse of perm_in
    
    int primitive_root;        ///< Generator g (smallest primitive root mod P)
    
} rader_plan_cache_entry;

//==============================================================================
// RADIX FUNCTION POINTER TYPES
//==============================================================================

/**
 * @brief Signature for forward radix-N butterfly implementations
 * 
 * **Calling convention:**
 * - Read from input with stride determined by algorithm (Cooley-Tukey or Stockham)
 * - Apply radix-N DFT with twiddle factors from stage_tw
 * - If prime radix (rader_tw != NULL), use Rader's convolution algorithm
 * - Write to output with stride determined by algorithm
 * 
 * **SIMD optimization:**
 * Implementations should process multiple butterflies using AVX2/AVX-512 when possible.
 */
typedef void (*radix_fv_func)(
    fft_data *restrict output,         ///< Output buffer (write with computed stride)
    fft_data *restrict input,          ///< Input buffer (read with computed stride)
    const fft_data *restrict stage_tw, ///< Stage twiddles [(radix-1)×sub_len elements]
    const fft_data *restrict rader_tw, ///< Rader twiddles [radix-1 elements, or NULL if not prime]
    int sub_len                        ///< Butterfly stride/count parameter
);

/**
 * @brief Signature for inverse radix-N butterfly implementations
 * 
 * Identical to radix_fv_func but uses inverse twiddles (conjugated).
 * Separate type enables compiler optimizations and type checking.
 */
typedef void (*radix_bv_func)(
    fft_data *restrict output,
    fft_data *restrict input,
    const fft_data *restrict stage_tw,
    const fft_data *restrict rader_tw,
    int sub_len
);

#endif // FFT_PLANNING_TYPES_H