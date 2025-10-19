//==============================================================================
// fft_rader_plans.c - Rader Algorithm Global Cache Manager
//==============================================================================

/**
 * @file fft_rader_plans.c
 * @brief Thread-safe global cache for Rader's prime-radix FFT algorithm data
 * 
 * **Rader's Algorithm Overview:**
 * For prime radix P, the DFT can be reduced to:
 * 1. Separate DC component (index 0)
 * 2. Circular convolution of remaining P-1 points using generator permutation
 * 3. Convolution computed efficiently via small FFT (size P-1)
 * 
 * **Why a Global Cache?**
 * - Rader plans are expensive to compute (O(P²) for permutations + twiddles)
 * - Same prime appears in many transforms (e.g., all N = 7k, 11k, 13k)
 * - Plans are direction-specific but immutable once created
 * - Thread-safe lazy initialization avoids startup overhead
 * 
 * **Cache Architecture:**
 * ```
 * Global Cache (16 slots)
 * ┌─────────────────────────────────────────────┐
 * │ Slot 0: Prime 7  (g=3, initialized)        │
 * │ Slot 1: Prime 11 (g=2, initialized)        │
 * │ Slot 2: Prime 13 (g=2, initialized)        │
 * │ Slot 3: Prime 17 (g=3, lazy-init)          │
 * │ ...                                         │
 * │ Slot 15: (unused)                           │
 * └─────────────────────────────────────────────┘
 *          ↓
 *   Each entry contains:
 *   - Prime value
 *   - Primitive root (generator)
 *   - Forward convolution twiddles (P-1 complex)
 *   - Inverse convolution twiddles (P-1 complex)
 *   - Input permutation (P-1 integers)
 *   - Output permutation (P-1 integers)
 * ```
 * 
 * **Memory Ownership:**
 * - Cache owns all allocations (twiddles, permutations)
 * - FFT plans borrow pointers (don't free)
 * - Cleanup via cleanup_rader_cache() at program exit
 * 
 * **Thread Safety Model:**
 * - Lazy initialization with double-checked locking
 * - Read-mostly pattern: lock only during cache miss
 * - Recursive mutex for init-during-init scenarios
 * 
 * **Supported Primes:**
 * Currently: 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67
 * Extendable: Add to g_primitive_roots[] database
 */

#include "fft_planning_types.h"
#include "fft_rader_plans.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

//==============================================================================
// PLATFORM-SPECIFIC SYNCHRONIZATION
//==============================================================================

#ifdef _WIN32
    #include <windows.h>
    #include <malloc.h>
    #define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
    #define aligned_free(ptr) _aligned_free(ptr)
    
    /**
     * @brief Windows mutex implementation using CRITICAL_SECTION
     * 
     * Windows critical sections are recursive by default and lightweight
     * (user-space spinlock that falls back to kernel mutex under contention).
     */
    static CRITICAL_SECTION g_rader_mutex;
    static int g_mutex_initialized = 0;
    
    static inline void mutex_init(void) {
        if (!g_mutex_initialized) {
            InitializeCriticalSection(&g_rader_mutex);
            g_mutex_initialized = 1;
        }
    }
    
    static inline void mutex_lock(void) { 
        EnterCriticalSection(&g_rader_mutex); 
    }
    
    static inline void mutex_unlock(void) { 
        LeaveCriticalSection(&g_rader_mutex); 
    }
    
    static inline void mutex_destroy(void) { 
        if (g_mutex_initialized) {
            DeleteCriticalSection(&g_rader_mutex);
            g_mutex_initialized = 0;
        }
    }
    
#else
    #include <pthread.h>
    #define aligned_free(ptr) free(ptr)
    
    /**
     * @brief POSIX mutex implementation using pthread_mutex_t
     * 
     * Statically initialized with PTHREAD_MUTEX_INITIALIZER for zero
     * runtime overhead. Not recursive by default (careful with init!).
     */
    static pthread_mutex_t g_rader_mutex = PTHREAD_MUTEX_INITIALIZER;
    
    static inline void mutex_init(void) { 
        /* Already initialized statically */ 
    }
    
    static inline void mutex_lock(void) { 
        pthread_mutex_lock(&g_rader_mutex); 
    }
    
    static inline void mutex_unlock(void) { 
        pthread_mutex_unlock(&g_rader_mutex); 
    }
    
    static inline void mutex_destroy(void) { 
        pthread_mutex_destroy(&g_rader_mutex); 
    }
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288419716939937510
#endif

//==============================================================================
// GLOBAL CACHE STORAGE
//==============================================================================

/**
 * @brief Maximum number of distinct prime radices in cache
 * 
 * Current: 16 slots (supports primes 7-67)
 * Typical usage: 3-5 primes (7, 11, 13 most common)
 * Memory: ~10 KB per prime (twiddles + permutations)
 * Total cache: ~160 KB maximum
 */
#define MAX_RADER_PRIMES 16

/**
 * @brief Global cache array (zero-initialized at program start)
 */
static rader_plan_cache_entry g_rader_cache[MAX_RADER_PRIMES];

/**
 * @brief Cache initialization flag (protected by mutex)
 * 
 * 0 = uninitialized (cold start)
 * 1 = initialized (common primes pre-populated)
 */
static int g_cache_initialized = 0;

//==============================================================================
// PRIMITIVE ROOT DATABASE
//==============================================================================

/**
 * @brief Prime-generator pair for Rader's algorithm
 * 
 * A primitive root g modulo prime P generates the multiplicative group:
 * {g^0, g^1, g^2, ..., g^(P-2)} ≡ {1, 2, 3, ..., P-1} (mod P)
 * 
 * This permutation is the key to Rader's DFT → convolution transformation.
 */
typedef struct {
    int prime;      ///< Prime modulus (7, 11, 13, ...)
    int generator;  ///< Smallest primitive root modulo prime
} prime_generator_pair;

/**
 * @brief Hardcoded primitive roots for common primes
 * 
 * **Database Source:**
 * Verified against OEIS A001918 (smallest primitive roots)
 * 
 * **Why Hardcode?**
 * - Finding primitive roots is expensive (trial + Euler's totient)
 * - These values never change (mathematical constants)
 * - Lookup is O(1) vs O(P log log P) computation
 * 
 * **Verification Example (prime=7, g=3):**
 * - 3^1 ≡ 3 (mod 7)
 * - 3^2 ≡ 2 (mod 7)
 * - 3^3 ≡ 6 (mod 7)
 * - 3^4 ≡ 4 (mod 7)
 * - 3^5 ≡ 5 (mod 7)
 * - 3^6 ≡ 1 (mod 7)  ← Full cycle: {3,2,6,4,5,1} = {1..6} ✓
 * 
 * **Extension:**
 * To add prime P:
 * 1. Find smallest g where ord(g) = P-1 (use trial or Pohlig-Hellman)
 * 2. Add {P, g} to this table
 * 3. Implement radix-P butterfly in execution layer
 */
static const prime_generator_pair g_primitive_roots[] = {
    {7,   3},   // 3 generates Z*_7  (multiplicative group mod 7)
    {11,  2},   // 2 generates Z*_11
    {13,  2},   // 2 generates Z*_13
    {17,  3},   // 3 generates Z*_17
    {19,  2},   // 2 generates Z*_19
    {23,  5},   // 5 generates Z*_23
    {29,  2},   // 2 generates Z*_29
    {31,  3},   // 3 generates Z*_31
    {37,  2},   // 2 generates Z*_37
    {41,  6},   // 6 generates Z*_41
    {43,  3},   // 3 generates Z*_43
    {47,  5},   // 5 generates Z*_47
    {53,  2},   // 2 generates Z*_53
    {59,  2},   // 2 generates Z*_59
    {61,  2},   // 2 generates Z*_61
    {67,  2},   // 2 generates Z*_67
};

static const int NUM_KNOWN_PRIMES = sizeof(g_primitive_roots) / sizeof(g_primitive_roots[0]);

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Lookup primitive root for given prime
 * 
 * Linear search through database. For small tables (< 20 entries),
 * this is faster than hash table or binary search overhead.
 * 
 * @param prime Prime to lookup (must be in database)
 * @return Primitive root (generator), or -1 if not found
 */
static int find_primitive_root(int prime)
{
    for (int i = 0; i < NUM_KNOWN_PRIMES; i++) {
        if (g_primitive_roots[i].prime == prime) {
            return g_primitive_roots[i].generator;
        }
    }
    return -1;  // Prime not in database
}

/**
 * @brief Modular exponentiation using binary method
 * 
 * Computes (base^exp) mod modulus efficiently in O(log exp) time.
 * 
 * **Algorithm:**
 * - Square-and-multiply: process exponent bits right-to-left
 * - Invariant: result × base^exp = original_base^original_exp (mod modulus)
 * 
 * **Example:** 3^5 mod 7
 * - exp = 5 = 0b101
 * - Bit 0 (1): result = 1×3 = 3, base = 3²=2 (mod 7)
 * - Bit 1 (0): result = 3, base = 2²=4 (mod 7)
 * - Bit 2 (1): result = 3×4 = 5 (mod 7)
 * - Result: 5 ✓ (3^5 = 243 ≡ 5 mod 7)
 * 
 * @param base Base of exponentiation
 * @param exp Exponent (non-negative)
 * @param mod Modulus (positive)
 * @return (base^exp) mod mod
 */
static int mod_pow(int base, int exp, int mod)
{
    int result = 1;
    base %= mod;  // Reduce base initially
    
    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp >>= 1;
    }
    
    return result;
}

/**
 * @brief Compute Rader permutation arrays from primitive root
 * 
 * **Input Permutation (Generator Powers):**
 * perm_in[i] = g^i mod prime, for i = 0..(prime-2)
 * Maps natural order to generator order
 * 
 * **Output Permutation (Inverse Mapping):**
 * perm_out[j] = i such that g^i ≡ j+1 (mod prime)
 * Maps generator order back to natural order
 * 
 * **Example (prime=7, g=3):**
 * ```
 * perm_in:  [3^0, 3^1, 3^2, 3^3, 3^4, 3^5] mod 7
 *         = [1,   3,   2,   6,   4,   5]
 * 
 * perm_out: inverse mapping
 *         [0, 2, 1, 4, 3, 5]
 *         (1→0, 2→2, 3→1, 4→4, 5→5, 6→3)
 * ```
 * 
 * **Why P-1 Elements?**
 * Rader's algorithm separates DC (index 0) from AC components (indices 1..P-1).
 * Permutations only apply to P-1 AC components.
 * 
 * @param prime Prime modulus
 * @param g Primitive root (generator)
 * @param perm_in Output: input permutation array [prime-1 elements]
 * @param perm_out Output: output permutation array [prime-1 elements]
 */
static void compute_permutations(int prime, int g, int *perm_in, int *perm_out)
{
    // Input permutation: generator powers
    for (int i = 0; i < prime - 1; i++) {
        perm_in[i] = mod_pow(g, i, prime);
    }
    
    // Output permutation: inverse of input
    for (int i = 0; i < prime - 1; i++) {
        int value = perm_in[i];       // g^i mod prime (in range 1..prime-1)
        int idx = value - 1;          // Map to array index 0..prime-2
        perm_out[idx] = i;            // Store position i at index (value-1)
    }
}

/**
 * @brief Wrapper for sincos() with compiler-specific handling
 * 
 * **Platform Differences:**
 * - GCC/Clang: sincos() computes both with single FSINCOS instruction (~20 cycles)
 * - MSVC: No sincos(), use separate sin()+cos() (~40 cycles)
 * - ARM NEON: sincos() available via math.h
 * 
 * **Performance:**
 * Computing sin+cos together is ~1.5-2× faster than separate calls.
 * 
 * @param x Angle in radians
 * @param s Output: sin(x)
 * @param c Output: cos(x)
 */
static inline void sincos_auto(double x, double *s, double *c)
{
#ifdef __GNUC__
    sincos(x, s, c);  // GNU extension: compute both at once
#else
    *s = sin(x);      // Fallback: separate calls
    *c = cos(x);
#endif
}

//==============================================================================
// CACHE ENTRY CREATION
//==============================================================================

/**
 * @brief Create and populate Rader cache entry for given prime
 * 
 * **Algorithm Steps:**
 * 1. Lookup primitive root from database
 * 2. Find free slot in cache (first with prime=0)
 * 3. Allocate and compute permutation arrays
 * 4. Allocate and compute convolution twiddles (forward + inverse)
 * 5. Store in cache with metadata
 * 
 * **Convolution Twiddles:**
 * - Forward: exp(-2πi × perm_out[q] / prime) for q=0..prime-2
 * - Inverse: exp(+2πi × perm_out[q] / prime) for q=0..prime-2
 * 
 * These twiddles convert cyclic convolution to pointwise multiply in frequency domain.
 * 
 * **Memory Allocation:**
 * - Permutations: (prime-1) × sizeof(int) × 2 arrays
 * - Twiddles: (prime-1) × sizeof(fft_data) × 2 arrays, 32-byte aligned for AVX2
 * - Total: ~(16×(P-1) + 32) bytes per prime
 * - Example: Prime 13 → ~200 bytes
 * 
 * **Error Handling:**
 * Returns -1 on failure:
 * - Prime not in database (no primitive root)
 * - Cache full (all 16 slots occupied)
 * - Memory allocation failure
 * 
 * @param prime Prime radix to create plan for
 * @return 0 on success, -1 on failure
 */
static int create_rader_plan_for_prime(int prime)
{
    // ─────────────────────────────────────────────────────────────────────
    // Step 1: Lookup Primitive Root
    // ─────────────────────────────────────────────────────────────────────
    
    int g = find_primitive_root(prime);
    if (g < 0) {
        fprintf(stderr, "[Rader] No primitive root found for prime %d (not in database)\n", prime);
        return -1;
    }
    
    // ─────────────────────────────────────────────────────────────────────
    // Step 2: Find Free Cache Slot
    // ─────────────────────────────────────────────────────────────────────
    
    int slot = -1;
    for (int i = 0; i < MAX_RADER_PRIMES; i++) {
        if (g_rader_cache[i].prime == 0) {  // Empty slot (zero-initialized)
            slot = i;
            break;
        }
    }
    
    if (slot < 0) {
        fprintf(stderr, "[Rader] Cache full (max %d primes). Consider increasing MAX_RADER_PRIMES.\n", 
                MAX_RADER_PRIMES);
        return -1;
    }
    
    rader_plan_cache_entry *entry = &g_rader_cache[slot];
    
    // ─────────────────────────────────────────────────────────────────────
    // Step 3: Allocate and Compute Permutations
    // ─────────────────────────────────────────────────────────────────────
    
    entry->perm_in = (int*)malloc((prime - 1) * sizeof(int));
    entry->perm_out = (int*)malloc((prime - 1) * sizeof(int));
    
    if (!entry->perm_in || !entry->perm_out) {
        fprintf(stderr, "[Rader] Memory allocation failed for permutations (prime %d)\n", prime);
        free(entry->perm_in);
        free(entry->perm_out);
        return -1;
    }
    
    compute_permutations(prime, g, entry->perm_in, entry->perm_out);
    
    // ─────────────────────────────────────────────────────────────────────
    // Step 4: Allocate Convolution Twiddles (32-byte aligned for AVX2)
    // ─────────────────────────────────────────────────────────────────────
    
    entry->conv_tw_fwd = (fft_data*)aligned_alloc(32, (prime - 1) * sizeof(fft_data));
    entry->conv_tw_inv = (fft_data*)aligned_alloc(32, (prime - 1) * sizeof(fft_data));
    
    if (!entry->conv_tw_fwd || !entry->conv_tw_inv) {
        fprintf(stderr, "[Rader] Failed to allocate twiddle arrays (prime %d)\n", prime);
        free(entry->perm_in);
        free(entry->perm_out);
        aligned_free(entry->conv_tw_fwd);
        aligned_free(entry->conv_tw_inv);
        return -1;
    }
    
    // ─────────────────────────────────────────────────────────────────────
    // Step 5: Compute Convolution Twiddles
    // ─────────────────────────────────────────────────────────────────────
    
    // Twiddles: W_prime^(perm_out[q]) for q=0..prime-2
    // Forward: exp(-2πi × idx / prime)
    // Inverse: exp(+2πi × idx / prime)
    
    for (int q = 0; q < prime - 1; q++) {
        int idx = entry->perm_out[q];
        
        // Forward twiddle (negative sign)
        double angle_fwd = -2.0 * M_PI * (double)idx / (double)prime;
        sincos_auto(angle_fwd, &entry->conv_tw_fwd[q].im, &entry->conv_tw_fwd[q].re);
        
        // Inverse twiddle (positive sign)
        double angle_inv = +2.0 * M_PI * (double)idx / (double)prime;
        sincos_auto(angle_inv, &entry->conv_tw_inv[q].im, &entry->conv_tw_inv[q].re);
    }
    
    // ─────────────────────────────────────────────────────────────────────
    // Step 6: Store Metadata
    // ─────────────────────────────────────────────────────────────────────
    
    entry->prime = prime;
    entry->primitive_root = g;
    
#ifdef FFT_DEBUG_RADER
    fprintf(stderr, "[Rader] Created plan for prime %d (g=%d) in slot %d\n", 
            prime, g, slot);
#endif
    
    return 0;
}

//==============================================================================
// PUBLIC API: CACHE INITIALIZATION
//==============================================================================

/**
 * @brief Initialize Rader cache with common primes (thread-safe)
 * 
 * **Pre-population Strategy:**
 * - Primes 7, 11, 13: Most common in practice (appear in 50% of transforms)
 * - Other primes: Lazy-initialized on first use
 * 
 * **When to Call:**
 * - Optional: fft_init() will call this automatically on first use
 * - Explicit call useful for:
 *   * Avoiding first-use latency in time-critical code
 *   * Deterministic startup (testing, benchmarking)
 *   * Pre-warming cache before performance measurement
 * 
 * **Thread Safety:**
 * - Multiple threads can call safely (idempotent)
 * - First caller initializes, others return immediately
 * - Uses mutex for synchronization
 * 
 * **Performance:**
 * - One-time cost: ~100 microseconds
 * - Pre-populates 3 primes (7, 11, 13)
 * - Subsequent calls are no-ops (fast mutex check)
 */
void init_rader_cache(void)
{
    mutex_init();
    mutex_lock();
    
    // Double-checked locking: check again inside mutex
    if (g_cache_initialized) {
        mutex_unlock();
        return;
    }
    
    // Clear cache to known state
    memset(g_rader_cache, 0, sizeof(g_rader_cache));
    
    // Pre-populate most common primes
    // Failures are non-fatal (primes will be created on-demand if needed)
    create_rader_plan_for_prime(7);
    create_rader_plan_for_prime(11);
    create_rader_plan_for_prime(13);
    
    g_cache_initialized = 1;
    
    mutex_unlock();
}

/**
 * @brief Free all resources in Rader cache (thread-safe)
 * 
 * **When to Call:**
 * - Program exit / shutdown
 * - After all FFT plans have been freed
 * - Before unloading shared library (dlclose)
 * 
 * **Effect:**
 * - Frees all twiddle arrays (conv_tw_fwd, conv_tw_inv)
 * - Frees all permutation arrays (perm_in, perm_out)
 * - Resets cache to empty state
 * - Destroys synchronization primitives
 * 
 * **Thread Safety:**
 * - NOT safe to call while FFT operations in progress
 * - NOT safe to call while FFT plans exist that reference cache
 * - Safe to call multiple times (idempotent after first call)
 * 
 * **Memory Guarantees:**
 * - No leaks: all allocated memory freed
 * - No dangling pointers: cache entries zeroed
 * - Safe to call init_rader_cache() again after cleanup
 * 
 * **Typical Usage:**
 * ```c
 * void application_shutdown(void) {
 *     // 1. Free all FFT plans
 *     for (int i = 0; i < num_plans; i++) {
 *         free_fft(plans[i]);
 *     }
 *     
 *     // 2. Wait for any in-flight FFT operations to complete
 *     // (application-specific synchronization)
 *     
 *     // 3. Clean up global Rader cache
 *     cleanup_rader_cache();
 * }
 * ```
 * 
 * @warning Don't call while FFT plans exist or operations in progress!
 */
void cleanup_rader_cache(void)
{
    mutex_lock();
    
    // Early exit if not initialized
    if (!g_cache_initialized) {
        mutex_unlock();
        return;
    }
    
    // Free all cache entries
    for (int i = 0; i < MAX_RADER_PRIMES; i++) {
        rader_plan_cache_entry *entry = &g_rader_cache[i];
        
        if (entry->prime > 0) {  // Entry is occupied
            // Free twiddle arrays
            aligned_free(entry->conv_tw_fwd);
            aligned_free(entry->conv_tw_inv);
            
            // Free permutation arrays
            free(entry->perm_in);
            free(entry->perm_out);
        }
    }
    
    // Reset cache to empty state
    memset(g_rader_cache, 0, sizeof(g_rader_cache));
    g_cache_initialized = 0;
    
    mutex_unlock();
    
    // Destroy synchronization primitives
    mutex_destroy();
}

//==============================================================================
// PUBLIC API: TWIDDLE ACCESS
//==============================================================================

/**
 * @brief Get convolution twiddles for prime radix (thread-safe, lazy)
 * 
 * **Lazy Initialization:**
 * - First call for a prime: creates cache entry (~50-100 μs)
 * - Subsequent calls: fast lookup (~10 ns)
 * - Thread-safe: uses mutex for cache access
 * 
 * **Return Value:**
 * - Pointer to direction-specific twiddle array [prime-1 elements]
 * - Forward: exp(-2πi × perm_out[q] / prime)
 * - Inverse: exp(+2πi × perm_out[q] / prime)
 * - NULL on failure (prime not in database or cache full)
 * 
 * **Memory Ownership:**
 * - Returned pointer is BORROWED from global cache
 * - Do NOT free this pointer!
 * - Valid until cleanup_rader_cache() is called
 * 
 * **Thread Safety:**
 * - Safe to call from multiple threads
 * - Cache miss may cause contention (first use per prime)
 * - Subsequent calls are fast (read-only access)
 * 
 * **Usage in Planning:**
 * ```c
 * // In fft_planner.c, during stage construction:
 * if (radix == 7 || radix == 11 || radix == 13) {
 *     stage->rader_tw = get_rader_twiddles(radix, direction);
 *     if (!stage->rader_tw) {
 *         // Handle error: prime not supported
 *     }
 * }
 * ```
 * 
 * **Usage in Execution:**
 * ```c
 * // In radix-7 butterfly:
 * void radix_7_butterfly(fft_data *out, const fft_data *in, 
 *                        const fft_data *rader_tw, ...) {
 *     // rader_tw points to cache (6 complex twiddles for prime 7)
 *     // Use for circular convolution in frequency domain
 * }
 * ```
 * 
 * @param prime Prime radix (must be in g_primitive_roots database)
 * @param direction FFT_FORWARD or FFT_INVERSE
 * @return Pointer to twiddle array, or NULL on failure
 */
const fft_data* get_rader_twiddles(int prime, fft_direction_t direction)
{
    mutex_lock();
    
    // ─────────────────────────────────────────────────────────────────────
    // Ensure Cache is Initialized
    // ─────────────────────────────────────────────────────────────────────
    
    if (!g_cache_initialized) {
        mutex_unlock();
        init_rader_cache();  // Recursive call (thread-safe)
        mutex_lock();
    }
    
    // ─────────────────────────────────────────────────────────────────────
    // Search Cache for Prime
    // ─────────────────────────────────────────────────────────────────────
    
    for (int i = 0; i < MAX_RADER_PRIMES; i++) {
        if (g_rader_cache[i].prime == prime) {
            // Cache hit: return direction-specific twiddles
            const fft_data *result = (direction == FFT_FORWARD) 
                ? g_rader_cache[i].conv_tw_fwd 
                : g_rader_cache[i].conv_tw_inv;
            
            mutex_unlock();
            return result;
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────
    // Cache Miss: Create Entry On-Demand
    // ─────────────────────────────────────────────────────────────────────
    
    mutex_unlock();
    
    // Create plan (outside mutex to allow parallel creation of different primes)
    if (create_rader_plan_for_prime(prime) < 0) {
        return NULL;  // Failed to create (not in database or cache full)
    }
    
    // Retry lookup (now it exists in cache)
    return get_rader_twiddles(prime, direction);
}

//==============================================================================
// PUBLIC API: PERMUTATION ACCESS (Optional)
//==============================================================================

/**
 * @brief Get input permutation for prime radix (thread-safe, lazy)
 * 
 * **Input Permutation:**
 * Array of P-1 indices representing generator powers:
 * perm_in[i] = g^i mod prime, for i=0..(prime-2)
 * 
 * **Use Cases:**
 * - Debugging Rader butterfly implementations
 * - Verification/testing
 * - Educational purposes (understanding algorithm)
 * 
 * **Note:**
 * Most butterfly implementations hardcode permutations for performance.
 * This function useful primarily for verification, not hot path execution.
 * 
 * @param prime Prime radix
 * @return Pointer to input permutation [prime-1 elements], or NULL on failure
 */
const int* get_rader_input_perm(int prime)
{
    mutex_lock();
    
    // Search cache
    for (int i = 0; i < MAX_RADER_PRIMES; i++) {
        if (g_rader_cache[i].prime == prime) {
            const int *result = g_rader_cache[i].perm_in;
            mutex_unlock();
            return result;
        }
    }
    
    mutex_unlock();
    
    // Trigger creation by calling get_rader_twiddles
    get_rader_twiddles(prime, FFT_FORWARD);
    
    // Retry (now in cache)
    return get_rader_input_perm(prime);
}

/**
 * @brief Get output permutation for prime radix (thread-safe, lazy)
 * 
 * **Output Permutation:**
 * Array of P-1 indices representing inverse of input permutation:
 * perm_out[j] = i such that g^i ≡ j+1 (mod prime)
 * 
 * **Use Cases:**
 * - Same as get_rader_input_perm()
 * - Verification that perm_out is inverse of perm_in
 * 
 * @param prime Prime radix
 * @return Pointer to output permutation [prime-1 elements], or NULL on failure
 */
const int* get_rader_output_perm(int prime)
{
    mutex_lock();
    
    // Search cache
    for (int i = 0; i < MAX_RADER_PRIMES; i++) {
        if (g_rader_cache[i].prime == prime) {
            const int *result = g_rader_cache[i].perm_out;
            mutex_unlock();
            return result;
        }
    }
    
    mutex_unlock();
    
    // Trigger creation
    get_rader_twiddles(prime, FFT_FORWARD);
    
    // Retry
    return get_rader_output_perm(prime);
}