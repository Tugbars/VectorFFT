//==============================================================================
// IMPLEMENTATION
//==============================================================================

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
    #include <windows.h>
    #include <malloc.h>
    #define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
    #define aligned_free(ptr) _aligned_free(ptr)
    
    // Windows mutex
    static CRITICAL_SECTION g_rader_mutex;
    static int g_mutex_initialized = 0;
    
    static inline void mutex_init(void) {
        if (!g_mutex_initialized) {
            InitializeCriticalSection(&g_rader_mutex);
            g_mutex_initialized = 1;
        }
    }
    static inline void mutex_lock(void) { EnterCriticalSection(&g_rader_mutex); }
    static inline void mutex_unlock(void) { LeaveCriticalSection(&g_rader_mutex); }
    static inline void mutex_destroy(void) { 
        if (g_mutex_initialized) {
            DeleteCriticalSection(&g_rader_mutex);
            g_mutex_initialized = 0;
        }
    }
#else
    #include <pthread.h>
    #define aligned_free(ptr) free(ptr)
    
    // POSIX mutex
    static pthread_mutex_t g_rader_mutex = PTHREAD_MUTEX_INITIALIZER;
    
    static inline void mutex_init(void) { /* Already initialized statically */ }
    static inline void mutex_lock(void) { pthread_mutex_lock(&g_rader_mutex); }
    static inline void mutex_unlock(void) { pthread_mutex_unlock(&g_rader_mutex); }
    static inline void mutex_destroy(void) { pthread_mutex_destroy(&g_rader_mutex); }
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288419716939937510
#endif

//==============================================================================
// GLOBAL CACHE
//==============================================================================

#define MAX_RADER_PRIMES 16  // Support up to 16 different prime radices

static rader_plan_cache_entry g_rader_cache[MAX_RADER_PRIMES];
static int g_cache_initialized = 0;

//==============================================================================
// PRIMITIVE ROOT DATABASE
//==============================================================================

typedef struct {
    int prime;
    int generator;  // Primitive root (smallest)
} prime_generator_pair;

// ✅ Expanded to support more primes
static const prime_generator_pair g_primitive_roots[] = {
    {7,   3},   // 3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5, 3^6=1 (mod 7)
    {11,  2},   // 2 is primitive root mod 11
    {13,  2},   // 2 is primitive root mod 13
    {17,  3},   // 3 is primitive root mod 17
    {19,  2},   // 2 is primitive root mod 19
    {23,  5},   // 5 is primitive root mod 23
    {29,  2},   // 2 is primitive root mod 29
    {31,  3},   // 3 is primitive root mod 31
    {37,  2},   // 2 is primitive root mod 37
    {41,  6},   // 6 is primitive root mod 41
    {43,  3},   // 3 is primitive root mod 43
    {47,  5},   // 5 is primitive root mod 47
    {53,  2},   // 2 is primitive root mod 53
    {59,  2},   // 2 is primitive root mod 59
    {61,  2},   // 2 is primitive root mod 61
    {67,  2},   // 2 is primitive root mod 67
};

static const int NUM_KNOWN_PRIMES = sizeof(g_primitive_roots) / sizeof(g_primitive_roots[0]);

//==============================================================================
// HELPER: Find primitive root
//==============================================================================

static int find_primitive_root(int prime)
{
    for (int i = 0; i < NUM_KNOWN_PRIMES; i++) {
        if (g_primitive_roots[i].prime == prime) {
            return g_primitive_roots[i].generator;
        }
    }
    return -1;  // Not found
}

//==============================================================================
// HELPER: Modular exponentiation (g^exp mod prime)
//==============================================================================

static int mod_pow(int base, int exp, int mod)
{
    int result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}

//==============================================================================
// HELPER: Compute permutations from primitive root
//==============================================================================

static void compute_permutations(int prime, int g, int *perm_in, int *perm_out)
{
    // Input permutation: [g^0, g^1, g^2, ..., g^(p-2)] mod p
    // These are indices 1..(p-1) in some order
    for (int i = 0; i < prime - 1; i++) {
        perm_in[i] = mod_pow(g, i, prime);
    }
    
    // Output permutation: inverse of input (where does each index go?)
    for (int i = 0; i < prime - 1; i++) {
        int idx = perm_in[i] - 1;  // Map to 0..(p-2)
        perm_out[idx] = i;
    }
}

//==============================================================================
// SINCOS WRAPPER (reuse from fft_twiddles.c logic)
//==============================================================================

static inline void sincos_auto(double x, double *s, double *c)
{
#ifdef __GNUC__
    sincos(x, s, c);
#else
    *s = sin(x);
    *c = cos(x);
#endif
}

//==============================================================================
// CREATE RADER PLAN FOR PRIME
//==============================================================================

static int create_rader_plan_for_prime(int prime)
{
    // Find primitive root
    int g = find_primitive_root(prime);
    if (g < 0) {
        fprintf(stderr, "[Rader] No primitive root found for prime %d\n", prime);
        return -1;
    }
    
    // Find free slot in cache
    int slot = -1;
    for (int i = 0; i < MAX_RADER_PRIMES; i++) {
        if (g_rader_cache[i].prime == 0) {
            slot = i;
            break;
        }
    }
    
    if (slot < 0) {
        fprintf(stderr, "[Rader] Cache full (max %d primes)\n", MAX_RADER_PRIMES);
        return -1;
    }
    
    rader_plan_cache_entry *entry = &g_rader_cache[slot];
    
    // ✅ FIXED: Allocate and compute permutations
    entry->perm_in = (int*)malloc((prime - 1) * sizeof(int));
    entry->perm_out = (int*)malloc((prime - 1) * sizeof(int));
    
    if (!entry->perm_in || !entry->perm_out) {
        fprintf(stderr, "[Rader] Memory allocation failed for prime %d\n", prime);
        free(entry->perm_in);
        free(entry->perm_out);
        return -1;
    }
    
    compute_permutations(prime, g, entry->perm_in, entry->perm_out);
    
    // ✅ Allocate convolution twiddles (aligned for SIMD)
    entry->conv_tw_fwd = (fft_data*)aligned_alloc(32, (prime - 1) * sizeof(fft_data));
    entry->conv_tw_inv = (fft_data*)aligned_alloc(32, (prime - 1) * sizeof(fft_data));
    
    if (!entry->conv_tw_fwd || !entry->conv_tw_inv) {
        fprintf(stderr, "[Rader] Failed to allocate twiddles for prime %d\n", prime);
        free(entry->perm_in);
        free(entry->perm_out);
        aligned_free(entry->conv_tw_fwd);
        aligned_free(entry->conv_tw_inv);
        return -1;
    }
    
    // ✅ Compute convolution twiddles: exp(±2πi * out_perm[q] / prime)
    for (int q = 0; q < prime - 1; q++) {
        int idx = entry->perm_out[q];
        
        // FORWARD: exp(-2πi * idx / prime)
        double angle_fwd = -2.0 * M_PI * (double)idx / (double)prime;
        sincos_auto(angle_fwd, &entry->conv_tw_fwd[q].im, &entry->conv_tw_fwd[q].re);
        
        // INVERSE: exp(+2πi * idx / prime)
        double angle_inv = +2.0 * M_PI * (double)idx / (double)prime;
        sincos_auto(angle_inv, &entry->conv_tw_inv[q].im, &entry->conv_tw_inv[q].re);
    }
    
    entry->prime = prime;
    entry->primitive_root = g;
    
#ifdef FFT_DEBUG_RADER
    fprintf(stderr, "[Rader] Created plan for prime %d (g=%d) in slot %d\n", 
            prime, g, slot);
#endif
    
    return 0;
}

//==============================================================================
// INIT CACHE (Thread-safe)
//==============================================================================

void init_rader_cache(void)
{
    mutex_init();
    mutex_lock();
    
    if (g_cache_initialized) {
        mutex_unlock();
        return;
    }
    
    // ✅ Clear cache
    memset(g_rader_cache, 0, sizeof(g_rader_cache));
    
    // ✅ Pre-populate common primes (7, 11, 13)
    create_rader_plan_for_prime(7);
    create_rader_plan_for_prime(11);
    create_rader_plan_for_prime(13);
    
    g_cache_initialized = 1;
    
    mutex_unlock();
}

//==============================================================================
// CLEANUP CACHE (Thread-safe)
//==============================================================================

void cleanup_rader_cache(void)
{
    mutex_lock();
    
    if (!g_cache_initialized) {
        mutex_unlock();
        return;
    }
    
    for (int i = 0; i < MAX_RADER_PRIMES; i++) {
        rader_plan_cache_entry *entry = &g_rader_cache[i];
        if (entry->prime > 0) {
            aligned_free(entry->conv_tw_fwd);
            aligned_free(entry->conv_tw_inv);
            free(entry->perm_in);
            free(entry->perm_out);
        }
    }
    
    memset(g_rader_cache, 0, sizeof(g_rader_cache));
    g_cache_initialized = 0;
    
    mutex_unlock();
    mutex_destroy();
}

//==============================================================================
// GET TWIDDLES (Thread-safe)
//==============================================================================

const fft_data* get_rader_twiddles(int prime, fft_direction_t direction)
{
    mutex_lock();
    
    // Ensure cache is initialized
    if (!g_cache_initialized) {
        mutex_unlock();
        init_rader_cache();
        mutex_lock();
    }
    
    // Search cache
    for (int i = 0; i < MAX_RADER_PRIMES; i++) {
        if (g_rader_cache[i].prime == prime) {
            const fft_data *result = (direction == FFT_FORWARD) 
                ? g_rader_cache[i].conv_tw_fwd 
                : g_rader_cache[i].conv_tw_inv;
            mutex_unlock();
            return result;
        }
    }
    
    // Not found - create on demand
    mutex_unlock();
    
    if (create_rader_plan_for_prime(prime) < 0) {
        return NULL;
    }
    
    // Recursive call (now it exists in cache)
    return get_rader_twiddles(prime, direction);
}

//==============================================================================
// GET PERMUTATIONS (Thread-safe)
//==============================================================================

const int* get_rader_input_perm(int prime)
{
    mutex_lock();
    
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
    return get_rader_input_perm(prime);
}

const int* get_rader_output_perm(int prime)
{
    mutex_lock();
    
    for (int i = 0; i < MAX_RADER_PRIMES; i++) {
        if (g_rader_cache[i].prime == prime) {
            const int *result = g_rader_cache[i].perm_out;
            mutex_unlock();
            return result;
        }
    }
    
    mutex_unlock();
    
    // Trigger creation by calling get_rader_twiddles
    get_rader_twiddles(prime, FFT_FORWARD);
    return get_rader_output_perm(prime);
}